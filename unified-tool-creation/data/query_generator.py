"""Simple, atomic product-search query generation for retrieval training."""

import asyncio
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI

from tools_builder.storage import atomic_json_dump


SIMPLE_QUERY_PROMPT = """You generate realistic, simple product-search queries for training a product retrieval model.

Given the product below, generate {n} diverse queries that a shopper could plausibly
have typed before choosing this product.

Product Title: {title}
Subcategory: {subcategory}
Description: {description}

The search engine is intended to handle SIMPLE product searches.
More complex user needs will be decomposed into multiple searches by another system.

Rules:
- Each query must refer to a single product type.
- Keep the intent simple and directly searchable.
- A query may include one broad qualifier such as:
  - use case
  - broad feature
  - form factor
  - skill level
- Do not combine multiple independent constraints.
- Do not describe bundles, setups, projects, or multiple product types.
- Do not use personas or broad situations that require deciding what product is needed.
- Do not use comparisons or alternatives to another product.
- Do not use exact prices or narrow numeric constraints.
- Do not use brand names or model numbers.
- Do not try to uniquely identify this exact product.
- Only mention properties supported by the product information.
- Use natural search-box language, not questions.
- 2-7 words per query.
- Make the queries meaningfully different.

Good:
"audio interface"
"portable audio interface"
"audio interface for home recording"
"beginner electric guitar"
"guitar amp for practice"
"microphone for vocals"

Bad:
"portable two input audio interface under $150"
"complete home recording setup"
"birthday gift for someone starting music"
"alternative to focusrite scarlett"
"fender player stratocaster"

Return ONLY valid JSON:
{{"queries": ["...", "...", ...]}}"""


class BaseQueryGenerator(ABC):
    """Shared LLM, validation, and resumable-cache behavior."""

    name = "base"
    max_retries = 8

    @abstractmethod
    def generate_all(
        self,
        df: pd.DataFrame,
        pos_threshold: float = 4.0,
        max_samples: Optional[int] = None,
        batch_size: int = 100,
    ) -> Dict[Tuple[str, str], str]:
        pass

    @staticmethod
    def _is_informational(query: str) -> bool:
        starts = (
            "how ", "what ", "where ", "which ", "why ", "when ",
            "can ", "should ", "could ", "would ", "is ", "are ",
            "do ", "does ",
        )
        normalized = query.lower().strip()
        return normalized.startswith(starts) or "?" in normalized

    @staticmethod
    def _is_rate_limited(error: Exception) -> bool:
        return getattr(error, "status_code", None) == 429 or "429" in str(error)

    async def _create_completion(self, **kwargs):
        """Call chat completions and return text, retrying rate limits."""
        if str(kwargs.get("model", "")).lower().startswith("gpt-5"):
            legacy_limit = kwargs.pop("max_tokens", None)
            if legacy_limit is not None:
                kwargs.setdefault("max_completion_tokens", legacy_limit)

        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as error:
                message = str(error).lower()
                if (
                    "max_tokens or model output limit was reached" in message
                    and attempt + 1 < self.max_retries
                ):
                    current_limit = int(kwargs.get("max_completion_tokens", 700))
                    kwargs["max_completion_tokens"] = min(current_limit * 2, 4000)
                    continue
                if self._is_rate_limited(error) and attempt + 1 < self.max_retries:
                    wait = min(2 ** attempt + random.uniform(0, 2), 60)
                    await asyncio.sleep(wait)
                    continue
                print(
                    f"  LLM error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{type(error).__name__}: {error}"
                )
                return None
        return None

    @staticmethod
    def _load_json_object(text: str) -> Dict:
        normalized = (text or "").strip()
        if normalized.startswith("```"):
            lines = normalized.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            normalized = "\n".join(lines).strip()
        try:
            data = json.loads(normalized)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def _load_variants_cache(cls, cache_dir: str, filename: str) -> dict:
        if not cache_dir:
            return {}
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, filename)
        if not os.path.exists(cache_path):
            return {}
        with open(cache_path) as handle:
            raw = json.load(handle)
        return raw if isinstance(raw, dict) else {}

    def _save_cache_file(self, filename: str, payload: dict):
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, filename)
        atomic_json_dump(payload, path)
        callback = getattr(self, "snapshot_callback", None)
        if callback:
            callback(path)

    def _maybe_snapshot(self, processed: int, *, force: bool = False):
        now = time.monotonic()
        if not hasattr(self, "_last_snapshot_time"):
            self._last_snapshot_time = now
            self._last_snapshot_count = 0
        item_interval = int(os.getenv("RECO_QUERY_SNAPSHOT_ITEMS", "1000"))
        second_interval = int(os.getenv("RECO_QUERY_SNAPSHOT_SECONDS", "300"))
        if (
            force
            or processed - self._last_snapshot_count >= item_interval
            or now - self._last_snapshot_time >= second_interval
        ):
            self._save_cache()
            self._last_snapshot_count = processed
            self._last_snapshot_time = now

    def unload(self):
        pass


class SimpleQueryGenerator(BaseQueryGenerator):
    """Generate short, single-product queries once per catalog item."""

    name = "simple"
    _disallowed_query_pattern = re.compile(
        r"\d|[$€£]|\b(?:bundle|setup|project|gift|birthday|"
        r"alternative\s+to|versus|vs\.?|compared?\s+to)\b",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def cache_filename_for_model(model: str) -> str:
        safe_model = "".join(
            character if character.isalnum() else "_" for character in model
        ).strip("_")
        return f"simple_v1_{safe_model}_cache.json"

    def __init__(
        self,
        item_to_subcategory: Dict[str, str],
        model: str = "gpt-5.6-luna",
        cache_dir: str = None,
        api_key: str = None,
        base_url: str = None,
        max_concurrent: int = 50,
        num_variants: int = 20,
        cache_only: bool = False,
    ):
        self.item_to_subcategory = item_to_subcategory
        self.model = model
        self.cache_dir = cache_dir
        self.max_concurrent = max_concurrent
        self.num_variants = num_variants
        self.cache_only = cache_only
        self.cache_filename = self.cache_filename_for_model(model)
        self.cache = self._load_variants_cache(cache_dir, self.cache_filename)
        if cache_only:
            self.async_client = None
        else:
            resolved_key = api_key or os.getenv("OPENAI_API_KEY")
            resolved_url = base_url or os.getenv("OPENAI_BASE_URL")
            if not resolved_key:
                raise ValueError(
                    "OPENAI_API_KEY is required unless --query-cache-only is used"
                )
            kwargs = {"api_key": resolved_key}
            if resolved_url:
                kwargs["base_url"] = resolved_url
            self.async_client = AsyncOpenAI(**kwargs)

    @staticmethod
    def _clean_text(value, limit: int, fallback: str) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return fallback
        normalized = " ".join(str(value).split())
        return normalized[:limit] if normalized else fallback

    def _normalize_variants(self, values) -> List[str]:
        if not isinstance(values, list):
            return []
        variants = []
        seen = set()
        for value in values:
            if not isinstance(value, str):
                continue
            query = " ".join(value.lower().strip().split())
            word_count = len(query.split())
            if (
                query
                and query not in seen
                and 2 <= word_count <= 7
                and not self._is_informational(query)
                and not self._disallowed_query_pattern.search(query)
            ):
                seen.add(query)
                variants.append(query)
        return variants[: self.num_variants]

    def _save_cache(self):
        self._save_cache_file(self.cache_filename, self.cache)

    def product_variants(self, item_ids=None) -> Dict[str, List[str]]:
        """Return normalized generated variants without user-frequency replication."""
        selected = set(item_ids) if item_ids is not None else set(self.cache)
        return {
            item_id: variants
            for item_id in sorted(selected)
            if (variants := self._normalize_variants(self.cache.get(item_id, [])))
        }

    async def _generate_single(
        self,
        title: str,
        subcategory: str,
        description: str,
        semaphore: asyncio.Semaphore,
    ) -> List[str]:
        async with semaphore:
            prompt = SIMPLE_QUERY_PROMPT.format(
                title=self._clean_text(title, 200, "Unknown product"),
                subcategory=self._clean_text(subcategory, 120, "Unknown"),
                description=self._clean_text(description, 600, "N/A"),
                n=self.num_variants,
            )
            text = await self._create_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1400,
            )
            if text is None:
                return []
            data = self._load_json_object(text)
            return self._normalize_variants(data.get("queries", []))

    async def _generate_batch_async(
        self, items: List[Tuple[str, str, str]]
    ) -> List[List[str]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        return await asyncio.gather(
            *[
                self._generate_single(title, subcategory, description, semaphore)
                for title, subcategory, description in items
            ]
        )

    def generate_all(
        self,
        df: pd.DataFrame,
        pos_threshold: float = 4.0,
        max_samples: Optional[int] = None,
        batch_size: int = 100,
    ) -> Dict[Tuple[str, str], str]:
        required = ["user_id", "parent_asin", "product_title", "rating"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing query-generation columns: {missing}")

        columns = required + (
            ["product_description"] if "product_description" in df.columns else []
        )
        pairs = df[df["rating"] >= pos_threshold][columns].drop_duplicates(
            subset=["user_id", "parent_asin"]
        )
        if max_samples:
            pairs = pairs.sample(n=min(max_samples, len(pairs)), random_state=42)

        item_users = defaultdict(list)
        item_metadata = {}
        for _, row in pairs.iterrows():
            item_id = row["parent_asin"]
            item_users[item_id].append(row["user_id"])
            item_metadata.setdefault(
                item_id,
                (
                    row["product_title"],
                    self.item_to_subcategory.get(item_id, "Unknown"),
                    row.get("product_description", ""),
                ),
            )

        min_cached_variants = 1 if self.cache_only else max(self.num_variants // 2, 3)
        uncached = [] if self.cache_only else [
            (item_id, *item_metadata[item_id])
            for item_id in item_users
            if len(self._normalize_variants(self.cache.get(item_id, [])))
            < min_cached_variants
        ]

        processed = 0
        try:
            for offset in tqdm(
                range(0, len(uncached), batch_size), desc="Simple product queries"
            ):
                batch = uncached[offset : offset + batch_size]
                results = asyncio.run(
                    self._generate_batch_async(
                        [
                            (title, subcategory, description)
                            for _, title, subcategory, description in batch
                        ]
                    )
                )
                for (item_id, _, _, _), variants in zip(batch, results):
                    self.cache[item_id] = variants
                processed = offset + len(batch)
                self._maybe_snapshot(processed)
        finally:
            # Keep every completed batch on Ctrl-C or unexpected client errors.
            self._maybe_snapshot(processed, force=True)

        queries = {}
        for item_id, users in item_users.items():
            variants = self._normalize_variants(self.cache.get(item_id, []))
            for user_id in users:
                if variants:
                    chooser = random.Random(f"{user_id}\0{item_id}")
                    queries[(user_id, item_id)] = chooser.choice(variants)

        return queries


# Compatibility aliases for callers that previously used the standard generator.
StandardQueryGenerator = SimpleQueryGenerator
QueryGenerator = SimpleQueryGenerator
