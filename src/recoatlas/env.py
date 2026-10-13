"""The RecoAtlas environment: tools loaded once, many episodes on top."""

from __future__ import annotations

import functools
import json
import os
import threading
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, fields

from . import assets
from .batching import SearchBatcher
from .reports import prepare_report
from .scoring import REPORT_FORMAT, score_report, validation_errors

TOOL_NAMES = ("search_products", "get_complementary_products", "get_substitute_products")
# Tools whose returned products count as "seen" for the validity check.
DISCOVERY_TOOLS = ("search_products", "get_complementary_products")

SUBMIT_REPORT_DOC = f"""Submit the final report and end the episode.

Args:
    results: Exactly K different products returned by search_products or get_complementary_products,
        as {REPORT_FORMAT}
    report_explanation: Short explanation of the set.

Returns:
    JSON with valid, errors and the processed report. IDs are filtered and a short
    report is backfilled from discovery order, without reference items or scores.
"""


@dataclass(frozen=True)
class Settings:
    """How the tools behave. Fixed for the lifetime of an `Env`."""

    tools: str = "utility"  # "utility" (trained on purchases) or "semantic" (pretrained embeddings), as in v1
    max_tool_results: int | None = None  # optional experiment cap; paper calls are not capped
    tool_noise: float = 0.0  # share of complementary/near-duplicate results replaced by bad ones
    tool_noise_mode: str = "subcategory"  # kind of bad replacement (v1 tools/_noise.py)
    device: str = "cpu"
    # Concurrent searches share one encoder pass: several times faster under load, but a search's floats
    # then depend on its batch (scores within 1e-4, rare swaps of near-tied products). Off = exact v1 results.
    batch_search: bool = False

    @classmethod
    def from_environ(cls) -> Settings:
        """Settings from RECOATLAS_<FIELD> environment variables, e.g. RECOATLAS_DEVICE=cuda.

        Booleans are "1"/"0"; RECOATLAS_MAX_TOOL_RESULTS=none means no limit. Unset fields keep their default.
        """
        values = {}
        for field in fields(cls):
            raw = os.environ.get(f"RECOATLAS_{field.name.upper()}")
            if raw is None:
                continue
            if field.name == "max_tool_results":
                values[field.name] = None if raw.lower() == "none" else int(raw)
            elif isinstance(field.default, bool):
                values[field.name] = raw == "1"
            elif isinstance(field.default, int | float):
                values[field.name] = None if raw.lower() == "none" else type(field.default)(raw)
            else:
                values[field.name] = raw
        return cls(**values)

    def __post_init__(self):
        if self.tools not in ("utility", "semantic"):
            raise ValueError('tools must be "utility" or "semantic"')
        if self.max_tool_results is not None and not _is_positive_int(self.max_tool_results):
            raise ValueError("max_tool_results must be a positive integer or None")
        if not 0.0 <= self.tool_noise <= 1.0:
            raise ValueError("tool_noise must be between 0 and 1")


class Env:
    """One dataset's tools and requests, loaded once.

    The tool methods are stateless and thread-safe, so one `Env` can serve many
    episodes at once. Use `episode()` for one request with record-keeping.
    """

    def __init__(self, dataset: str = "hm-clothing", **settings):
        if dataset not in assets.DATASETS:
            raise ValueError(f"Unknown dataset {dataset!r}; available: {sorted(assets.DATASETS)}")
        self.dataset = dataset
        self.settings = Settings(**settings)
        self.splits: dict[str, list[dict]] = {
            s: assets.load_split(dataset, s) for s in assets.DATASETS[dataset]["splits"]
        }
        self._by_id = {split: {row["candidate_id"]: row for row in rows} for split, rows in self.splits.items()}
        self._load_tools()

    def _load_tools(self) -> None:
        try:
            from .tools.complementarity import ComplementarityTool
            from .tools.substitutability import SubstitutabilityTool
            from .tools.trained_retrieval import TrainedRetrievalTool
        except ImportError as error:
            raise ImportError(
                "Loading the tools needs the tool dependencies: pip install 'recoatlas[tools]'"
            ) from error

        files = {
            name: assets.fetch(self.dataset, name)
            for name in ("product_texts", "subcategories", "item_embeddings", "comp_embeddings", "retrieval_model")
        }
        assets.fetch_query_encoder()
        s = self.settings
        utility = s.tools == "utility"
        self._search = TrainedRetrievalTool(
            files["product_texts"],
            files["item_embeddings"],
            files["retrieval_model"],
            s.device,
            assets.QUERY_ENCODER["repo_id"],
            assets.QUERY_ENCODER["revision"],
        )
        if not utility:
            _disable_search_adapter(self._search)
        self._search_one = SearchBatcher(self._search.search_many).search if s.batch_search else self._search.search
        self._complementary = ComplementarityTool(
            product_texts_path=str(files["product_texts"]),
            complement_embeddings_path=str(files["comp_embeddings" if utility else "item_embeddings"]),
            subcategories_path=str(files["subcategories"]),
            trained=utility,
            device=s.device,
            noise=s.tool_noise,
            noise_mode=s.tool_noise_mode,
        )
        self._substitutes = SubstitutabilityTool(
            product_texts_path=str(files["product_texts"]),
            embeddings_path=str(files["item_embeddings"]),
            subcategories_path=str(files["subcategories"]),
            device=s.device,
            noise=s.tool_noise,
        )
        if not (self._search.all_items == self._complementary.all_items == self._substitutes.all_items):
            raise ValueError("Tool catalog order mismatch")
        # The search tool locks itself; these two locks keep the other tools' device work from interleaving.
        self._complementary_lock = threading.Lock()
        self._substitutes_lock = threading.Lock()
        self.catalog: dict[str, str] = self._search.product_reps  # product_id -> product text
        if len(self.catalog) != (expected := assets.DATASETS[self.dataset]["catalog_items"]):
            raise ValueError(f"{self.dataset}: expected {expected} products, found {len(self.catalog)}")
        self.item_to_subcategory: dict[str, str] = self._complementary.item_to_subcategory

    def __repr__(self) -> str:
        return f"Env({self.dataset!r}, {', '.join(f'{k}={v!r}' for k, v in asdict(self.settings).items())})"

    # Stateless tools. Each returns the same JSON string as recoatlas-v1.

    def search_products(self, query: str, top_k: int = 5) -> str:
        """Search the catalog for products matching a text query.

        Args:
            query: What to search for, e.g. "relaxed high-waisted shorts".
            top_k: Number of products to return.

        Returns:
            A JSON list of {item_id, score, description}.
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        return self._search_one(query, self._cap(top_k))

    def get_complementary_products(self, item_ids: list[str], top_k: int = 5) -> str:
        """Find products that go well with the given products, from other subcategories.

        Args:
            item_ids: Product IDs to find complements for.
            top_k: Number of products to return.

        Returns:
            A JSON list of {item_id, score, title, description}.
        """
        item_ids, top_k = _id_list(item_ids), self._cap(top_k)
        with self._complementary_lock:
            return self._complementary.get_complementary(item_ids, top_k=top_k)

    def get_substitute_products(self, item_ids: list[str], similarity_threshold: float = 0.95) -> str:
        """Remove near-duplicates from a list of products. Earlier products have priority.

        Args:
            item_ids: Candidate product IDs, highest priority first.
            similarity_threshold: Products in the same subcategory above this similarity are duplicates.

        Returns:
            A JSON object with "kept" and "removed" products.
        """
        item_ids = _id_list(item_ids)
        with self._substitutes_lock:
            return self._substitutes.get_substitutes(item_ids, similarity_threshold=float(similarity_threshold))

    def call(self, tool: str, arguments: dict) -> str:
        """Call a tool by name with keyword arguments."""
        if tool not in TOOL_NAMES:
            raise ValueError(f"Unknown tool {tool!r}; available: {TOOL_NAMES}")
        return getattr(self, tool)(**arguments)

    @staticmethod
    def returned_ids(output: str) -> list[str]:
        """Product IDs in a tool's JSON output, in order."""
        payload = json.loads(output)
        rows = payload if isinstance(payload, list) else payload.get("kept", []) + payload.get("removed", [])
        return [row["item_id"] for row in rows if isinstance(row, dict) and "item_id" in row]

    def _cap(self, top_k) -> int:
        if not _is_positive_int(top_k):
            raise ValueError("top_k must be a positive integer")
        limit = self.settings.max_tool_results
        return top_k if limit is None else min(top_k, limit)

    # Requests, episodes and scoring

    def row(self, split: str = "agent_hidden", index: int | None = None, candidate_id: str | None = None) -> dict:
        """One request of a split, chosen by position or by candidate_id."""
        if (index is None) == (candidate_id is None):
            raise ValueError("Pass exactly one of index or candidate_id")
        if index is not None:
            return self.splits[split][index]
        try:
            return self._by_id[split][candidate_id]
        except KeyError:
            raise KeyError(f"{candidate_id} not in {split}") from None

    def episode(
        self, split: str = "agent_hidden", index: int | None = None, candidate_id: str | None = None
    ) -> Episode:
        """Start an episode for one request."""
        return Episode(self, self.row(split, index, candidate_id))

    def score(self, results: list[dict] | None, references: Collection[str], seen: Sequence[str]) -> dict:
        """Process and score a report; `seen` is discovery order from search/complementarity."""
        return score_report(results, references, self.catalog, seen)


class Episode:
    """One shopping request: call the tools, `submit` a report, then `score` it."""

    def __init__(self, env: Env, row: dict):
        self.env = env
        self.split: str = row["partition"]
        self.candidate_id: str = row["candidate_id"]
        self.request: str = row["query"]
        self.k: int = len(set(row["positive_items"]))
        self.seen_ids: list[str] = []  # products returned by search or complementary calls, first seen first
        self.calls: list[dict] = []  # one {"tool", "arguments", "returned_ids"} per call
        self.submissions: list[dict] = []  # every submitted report; the last one is scored
        self._references: list[str] = list(row["positive_items"])
        self._seen: set[str] = set()

    def call(self, tool: str, **arguments) -> str:
        """Call a tool and record the products it returned."""
        output = self.env.call(tool, arguments)
        ids = Env.returned_ids(output)
        if tool in DISCOVERY_TOOLS:
            new = [i for i in dict.fromkeys(ids) if i not in self._seen]
            self.seen_ids += new
            self._seen.update(new)
        self.calls.append({"tool": tool, "arguments": arguments, "returned_ids": ids})
        return output

    def search_products(self, query: str, top_k: int = 5) -> str:
        return self.call("search_products", query=query, top_k=top_k)

    def get_complementary_products(self, item_ids: list[str], top_k: int = 5) -> str:
        return self.call("get_complementary_products", item_ids=item_ids, top_k=top_k)

    def get_substitute_products(self, item_ids: list[str], similarity_threshold: float = 0.95) -> str:
        return self.call("get_substitute_products", item_ids=item_ids, similarity_threshold=similarity_threshold)

    def submit(self, results: list[dict], report_explanation: str = "") -> dict:
        """Filter/backfill with the paper parser; return the processed report and label-free validity.

        Raw submissions are retained for auditing; the last submission is scored.
        """
        self.submissions.append({"report_explanation": report_explanation, "results": results})
        try:
            report = prepare_report(self.request, self.submissions[-1], self.seen_ids, [], self.k, self.env.catalog)
        except ValueError:
            return {"valid": False, "errors": ["INVALID_REPORT_SCHEMA"], "report": None}
        errors = validation_errors(report["results"], self.k, self.env.catalog, self._seen)
        return {"valid": not errors, "errors": errors, "report": report}

    def score(self) -> dict:
        """Score the last submission: {"sethit", "valid", "reward_direct"}."""
        results = self.submissions[-1]["results"] if self.submissions else None
        return self.env.score(results, self._references, self.seen_ids)


@functools.cache
def shared_env() -> Env:
    """One `Env` per process, for the adapters: dataset from RECOATLAS_DATASET, settings from `Settings.from_environ`."""
    return Env(os.environ.get("RECOATLAS_DATASET", "hm-clothing"), **asdict(Settings.from_environ()))


# Agent-facing tool descriptions, shared by Episode and the OpenEnv and TRL adapters.
TOOL_DOCS = {name: getattr(Env, name).__doc__ for name in TOOL_NAMES} | {"submit_report": SUBMIT_REPORT_DOC}
for _name in TOOL_NAMES:
    getattr(Episode, _name).__doc__ = TOOL_DOCS[_name]


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _id_list(item_ids) -> list[str]:
    if not isinstance(item_ids, list) or not all(isinstance(i, str) for i in item_ids):
        raise ValueError("item_ids must be a list of product ID strings")
    return item_ids


def _disable_search_adapter(search) -> None:
    """Semantic search = the pretrained encoder: switch the search LoRA off (as in v1 tools/load.py)."""
    search.model.query_encoder.base_model.disable_adapter_layers()
    layers = [m for m in search.model.query_encoder.modules() if hasattr(m, "lora_A")]
    if not layers or not all(m.disable_adapters for m in layers):
        raise ValueError("Could not disable all query adapters")
