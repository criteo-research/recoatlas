"""TRL adapter: a GRPOTrainer `environment_factory` running RecoAtlas in the training process.

    from recoatlas.trl import RecoAtlasEnvironment, trl_dataset
    trainer = GRPOTrainer(model=..., train_dataset=trl_dataset("agent_dev"),
                          environment_factory=RecoAtlasEnvironment, ...)

All rollouts share one `Env` per process, configured by RECOATLAS_* environment variables
(see `recoatlas.env.shared_env`; e.g. RECOATLAS_DEVICE=cuda). The reward is `reward_direct`:
SetHit for a valid report, -1 otherwise (including no report).
"""

from __future__ import annotations

import json

from . import assets
from .env import TOOL_DOCS, Episode, shared_env
from .scoring import REPORT_FORMAT


def trl_dataset(split: str, dataset: str = "hm-clothing"):
    """A `datasets.Dataset` with one row per request: prompt, candidate_id and split."""
    from datasets import Dataset

    return Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": row["query"]}], "candidate_id": row["candidate_id"], "split": split}
            for row in assets.load_split(dataset, split)
        ]
    )


class RecoAtlasEnvironment:
    """One TRL rollout = one RecoAtlas episode."""

    def __init__(self):
        self._env = shared_env()
        self._episode: Episode | None = None

    def reset(self, candidate_id: str, split: str = "agent_hidden", **_) -> str:
        """Start the request's episode. The returned text is appended to the user message."""
        self._episode = self._env.episode(split, candidate_id=candidate_id)
        return (
            f"\n\nUse the tools to find products, then call submit_report with exactly {self._episode.k} different "
            f"products returned by search_products or get_complementary_products, as results={REPORT_FORMAT}"
        )

    def get_reward(self) -> float:
        """reward_direct of the last submitted report; -1 if nothing was submitted."""
        return self._episode.score()["reward_direct"]

    def search_products(self, query: str, top_k: int = 5) -> str:
        return self._episode.search_products(query, top_k)

    def get_complementary_products(self, item_ids: list[str], top_k: int = 5) -> str:
        return self._episode.get_complementary_products(item_ids, top_k)

    def get_substitute_products(self, item_ids: list[str], similarity_threshold: float = 0.95) -> str:
        return self._episode.get_substitute_products(item_ids, similarity_threshold)

    def submit_report(self, results: list[dict], report_explanation: str = "") -> str:
        return json.dumps(self._episode.submit(results, report_explanation))


# TRL builds the tool schemas from the docstrings.
for _name, _doc in TOOL_DOCS.items():
    getattr(RecoAtlasEnvironment, _name).__doc__ = _doc
