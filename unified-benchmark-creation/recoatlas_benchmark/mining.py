"""Paper-faithful seven-day candidate-bundle mining."""

from __future__ import annotations

import hashlib
import hmac
import itertools
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .splits import SPLIT_NAMES, validate_manifest

PREFIXES = {"shared_train": "train", "agent_dev": "dev", "agent_hidden": "hidden"}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    partition: str
    item_ids: tuple[str, ...]

    @property
    def bundle_size(self) -> int:
        return len(self.item_ids)

    def as_json(self) -> dict:
        payload = asdict(self)
        payload["item_ids"] = list(self.item_ids)
        payload["bundle_size"] = self.bundle_size
        return payload


def candidate_id(*, salt: str, user_id: str, partition: str, domain: str) -> str:
    if partition not in PREFIXES:
        raise ValueError(f"Unknown partition: {partition}")
    namespace = f"{partition}_bundle\0{user_id}"
    if domain == "hm":
        namespace = "hm\0" + namespace
    digest = hmac.new(salt.encode(), namespace.encode(), hashlib.sha256).hexdigest()
    return f"{PREFIXES[partition]}-{digest[:24]}"


def _timestamp(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def _dense_episode(events: list[tuple[str, pd.Timestamp]], days: int, cap: int) -> tuple[str, ...]:
    left = 0
    best: list[tuple[str, pd.Timestamp]] = []
    window = pd.Timedelta(days=days)
    for right, (_, right_time) in enumerate(events):
        while right_time - events[left][1] > window:
            left += 1
        if right - left + 1 > len(best):  # Strict > preserves the first maximum.
            best = events[left : right + 1]
    return tuple(item_id for item_id, _ in best[-cap:])


def mine_amazon_candidates(
    interactions: pd.DataFrame,
    *,
    manifest: Mapping[str, object],
    subcategories: Mapping[str, str],
    partitions: Iterable[str] = SPLIT_NAMES,
    positive_rating: float = 4.0,
    window_days: int = 7,
    max_items: int = 7,
) -> list[Candidate]:
    """Mine the one deterministic Amazon candidate associated with each eligible user."""
    validate_manifest(manifest)
    required = {"user_id", "parent_asin", "rating", "timestamp"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Amazon interactions are missing columns: {sorted(missing)}")
    selected_partitions = tuple(partitions)
    if any(name not in SPLIT_NAMES for name in selected_partitions):
        raise ValueError("Unknown requested partition")
    user_partition = {
        str(user): partition
        for partition in selected_partitions
        for user in manifest["splits"][partition]
    }
    frame = interactions.loc[
        pd.to_numeric(interactions["rating"], errors="coerce").ge(positive_rating)
        & interactions["user_id"].notna()
        & interactions["parent_asin"].notna()
        & interactions["timestamp"].notna(),
        ["user_id", "parent_asin", "timestamp"],
    ].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["parent_asin"] = frame["parent_asin"].astype(str)
    frame = frame[frame["user_id"].isin(user_partition) & frame["parent_asin"].isin(subcategories)]
    frame["timestamp"] = _timestamp(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.sort_values(["user_id", "timestamp", "parent_asin"])
    frame = frame.drop_duplicates(["user_id", "parent_asin"], keep="last")
    sizes = frame.groupby("user_id", sort=False).size()
    frame = frame[frame["user_id"].isin(sizes[sizes >= 3].index)]

    output: list[Candidate] = []
    for user_id, user_frame in frame.groupby("user_id", sort=False):
        events = list(user_frame[["parent_asin", "timestamp"]].itertuples(index=False, name=None))
        item_ids = _dense_episode(events, window_days, max_items)
        categories = {subcategories.get(item, "") for item in item_ids} - {""}
        if len(item_ids) < 3 or len(categories) < 2:
            continue
        partition = user_partition[user_id]
        output.append(
            Candidate(
                candidate_id=candidate_id(
                    salt=str(manifest["salt"]),
                    user_id=user_id,
                    partition=partition,
                    domain="amazon",
                ),
                partition=partition,
                item_ids=item_ids,
            )
        )
    return sorted(output, key=lambda row: row.candidate_id)


def _ordered_unique_events(
    users: np.ndarray, items: np.ndarray, days: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((days, items, users))
    users, items, days = users[order], items[order], days[order]
    keep = np.r_[(users[:-1] != users[1:]) | (items[:-1] != items[1:]), True]
    users, items, days = users[keep], items[keep], days[keep]
    order = np.lexsort((items, days, users))
    return users[order], items[order], days[order]


def mine_hm_encoded(
    users: np.ndarray,
    items: np.ndarray,
    days: np.ndarray,
    *,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    product_codes: np.ndarray,
    product_types: np.ndarray,
    merchandising_groups: np.ndarray,
    user_partitions: np.ndarray,
    salt: str,
    window_days: int = 7,
    max_items: int = 7,
) -> list[Candidate]:
    """Memory-efficient H&M miner used for the 30M-row transaction table."""
    users, items, days = _ordered_unique_events(users, items, days)
    boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    output: list[Candidate] = []
    for start, end in itertools.pairwise(boundaries):
        if end - start < 3:
            continue
        dates = days[start:end]
        left = np.searchsorted(dates, dates - window_days, side="left")
        lengths = np.arange(1, len(dates) + 1) - left
        right = int(lengths.argmax())
        count = int(lengths[right])
        if count < 3:
            continue
        chosen = items[start + right - min(count, max_items) + 1 : start + right + 1]
        size = len(chosen)
        if (
            len(set(product_types[chosen])) != size
            or len(set(product_codes[chosen])) != size
            or len(set(merchandising_groups[chosen])) != 1
        ):
            continue
        user_code = int(users[start])
        partition = str(user_partitions[user_code])
        if partition not in PREFIXES:
            continue
        output.append(
            Candidate(
                candidate_id=candidate_id(
                    salt=salt,
                    user_id=str(user_ids[user_code]),
                    partition=partition,
                    domain="hm",
                ),
                partition=partition,
                item_ids=tuple(str(value) for value in item_ids[chosen]),
            )
        )
    return sorted(output, key=lambda row: row.candidate_id)


def mine_hm_candidates(
    interactions: pd.DataFrame,
    *,
    manifest: Mapping[str, object],
    catalog: pd.DataFrame,
    partitions: Iterable[str] = SPLIT_NAMES,
    window_days: int = 7,
    max_items: int = 7,
) -> list[Candidate]:
    """Convenience H&M miner for normalized tables and small fixtures."""
    validate_manifest(manifest)
    required_interactions = {"user_id", "item_id", "timestamp"}
    required_catalog = {"item_id", "product_code", "product_type", "merchandising_group"}
    if missing := required_interactions - set(interactions):
        raise ValueError(f"H&M interactions are missing columns: {sorted(missing)}")
    if missing := required_catalog - set(catalog):
        raise ValueError(f"H&M catalog is missing columns: {sorted(missing)}")
    requested = set(partitions)
    users = [
        str(user) for name in SPLIT_NAMES if name in requested for user in manifest["splits"][name]
    ]
    user_index = pd.Index(users)
    item_frame = catalog.drop_duplicates("item_id").copy()
    item_frame["item_id"] = item_frame["item_id"].astype(str)
    item_index = pd.Index(item_frame["item_id"])
    frame = interactions.copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    user_codes = user_index.get_indexer(frame["user_id"])
    item_codes = item_index.get_indexer(frame["item_id"])
    timestamps = _timestamp(frame["timestamp"])
    valid = (user_codes >= 0) & (item_codes >= 0) & timestamps.notna().to_numpy()
    partition_lookup = {str(user): name for name in requested for user in manifest["splits"][name]}
    return mine_hm_encoded(
        user_codes[valid].astype(np.int32),
        item_codes[valid].astype(np.int32),
        timestamps[valid].to_numpy().astype("datetime64[D]").astype(np.int32),
        user_ids=user_index.to_numpy(),
        item_ids=item_frame["item_id"].to_numpy(),
        product_codes=item_frame["product_code"].to_numpy(),
        product_types=item_frame["product_type"].to_numpy(),
        merchandising_groups=item_frame["merchandising_group"].to_numpy(),
        user_partitions=np.array([partition_lookup[user] for user in user_index]),
        salt=str(manifest["salt"]),
        window_days=window_days,
        max_items=max_items,
    )


def deduplicate_reference_sets(
    candidates: Iterable[Candidate], priority: Iterable[str]
) -> tuple[list[Candidate], list[dict]]:
    """Remove identical sets without consulting queries, tools, or model outcomes."""
    rows = list(candidates)
    by_partition = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        by_partition[row.partition].append(row)
    seen: set[tuple[str, ...]] = set()
    kept: list[Candidate] = []
    removed: list[dict] = []
    for partition in priority:
        for row in sorted(by_partition.get(partition, []), key=lambda value: value.candidate_id):
            key = tuple(sorted(row.item_ids))
            if key in seen:
                removed.append({"candidate_id": row.candidate_id, "reason": "identical_set"})
                continue
            seen.add(key)
            kept.append(row)
    return sorted(kept, key=lambda row: row.candidate_id), removed
