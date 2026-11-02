"""Leakage-free temporal complementarity training and catalog evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tools_builder.benchmark_split import (
    filter_interactions_to_split,
    load_user_split_manifest,
    public_split_metadata,
)
from tools_builder.hf_upload import hf_category, hf_run, prune_verified_files, restore_files, upload_files
from tools_builder.storage import atomic_json_dump, atomic_torch_save, check_ephemeral_budget
from tools_builder.utils import compute_pmi, save_config


FORMAT_VERSION = 2


def _undirected_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _within_window(left: pd.Timestamp, right: pd.Timestamp, window_days: Optional[int]) -> bool:
    return window_days is None or abs(right - left) <= pd.Timedelta(days=window_days)


@dataclass
class ComplementaritySplit:
    train_graph: Dict[str, Dict[str, float]]
    train_counts: Dict[str, Dict[str, int]]
    validation: Dict[str, List[str]]
    metadata: Dict[str, object]


def build_temporal_validation_relevance(
    interactions: pd.DataFrame,
    catalog_items: Iterable[str],
    item_to_subcategory: Mapping[str, str],
    *,
    user_ids: Optional[set[str]] = None,
    window_days: Optional[int] = 7,
    min_rating: float = 4.0,
    cross_subcategory: bool = True,
) -> tuple[Dict[str, List[str]], Dict[str, object]]:
    """Build leave-last-out relevance for an already isolated user partition."""
    required = {"user_id", "parent_asin", "rating", "timestamp"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Interactions are missing columns: {sorted(missing)}")

    catalog = set(catalog_items)
    frame = interactions[list(required)].copy()
    frame = frame[frame["parent_asin"].isin(catalog) & (frame["rating"] >= min_rating)]
    if user_ids is not None:
        allowed = {str(user_id) for user_id in user_ids}
        frame = frame[frame["user_id"].astype(str).isin(allowed)]
    if pd.api.types.is_numeric_dtype(frame["timestamp"]):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", errors="coerce")
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["user_id", "parent_asin", "timestamp"])
    frame = frame.sort_values(["user_id", "timestamp", "parent_asin"])
    frame = frame.drop_duplicates(["user_id", "parent_asin"], keep="last")

    validation_pairs: set[tuple[str, str]] = set()
    eligible_users = users_with_targets = 0
    for _, user_frame in frame.groupby("user_id", sort=False):
        if len(user_frame) < 2:
            continue
        eligible_users += 1
        events = list(
            user_frame[["parent_asin", "timestamp"]].itertuples(index=False, name=None)
        )
        history, (target, target_time) = events[:-1], events[-1]
        found = False
        for anchor, anchor_time in history:
            different_category = (
                item_to_subcategory.get(anchor, "")
                and item_to_subcategory.get(target, "")
                and item_to_subcategory.get(anchor, "")
                != item_to_subcategory.get(target, "")
            )
            if (
                anchor != target
                and (not cross_subcategory or different_category)
                and _within_window(anchor_time, target_time, window_days)
            ):
                validation_pairs.add((anchor, target))
                found = True
        users_with_targets += int(found)

    relevance: Dict[str, List[str]] = defaultdict(list)
    for anchor, target in sorted(validation_pairs):
        relevance[anchor].append(target)
    if not relevance:
        raise ValueError("Temporal audit partition produced no validation relevance")
    fingerprint = hashlib.sha256(
        json.dumps(sorted(validation_pairs), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "split": "isolated_user_partition_temporal_leave_last_out",
        "eligible_users": eligible_users,
        "users_with_targets": users_with_targets,
        "anchors": len(relevance),
        "directed_edges": sum(map(len, relevance.values())),
        "window_days": window_days,
        "min_rating": min_rating,
        "cross_subcategory": cross_subcategory,
        "fingerprint": fingerprint,
    }
    return dict(relevance), metadata


def build_temporal_complementarity_split(
    interactions: pd.DataFrame,
    catalog_items: Iterable[str],
    item_to_subcategory: Mapping[str, str],
    *,
    window_days: Optional[int] = 7,
    min_rating: float = 4.0,
    min_train_count: int = 1,
    min_pmi: float = 0.0,
    cross_subcategory: bool = True,
    validation_user_ratio: float = 0.1,
    seed: int = 42,
    validation_user_ids: Optional[set[str]] = None,
) -> ComplementaritySplit:
    """Build a strict temporal split and remove held-out edges in both directions.

    Users are split first, preventing a shopper's history from appearing in both
    partitions. All positive interactions from training users form the graph. For
    validation users, recent earlier items point to their last distinct positive
    interaction. Any undirected validation pair is then removed from training,
    even if a training user also produced it.
    """
    required = {"user_id", "parent_asin", "rating", "timestamp"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Interactions are missing columns: {sorted(missing)}")
    if min_train_count <= 0:
        raise ValueError("min_train_count must be positive")
    if not 0 < validation_user_ratio < 1 and validation_user_ids is None:
        raise ValueError("validation_user_ratio must be between zero and one")

    catalog = set(catalog_items)
    frame = interactions[list(required)].copy()
    frame = frame[frame["parent_asin"].isin(catalog) & (frame["rating"] >= min_rating)]
    if pd.api.types.is_numeric_dtype(frame["timestamp"]):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", errors="coerce")
    else:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["user_id", "parent_asin", "timestamp"])
    frame = frame.sort_values(["user_id", "timestamp", "parent_asin"])
    # A product contributes once per user. Keep the latest signal for temporal order.
    frame = frame.drop_duplicates(["user_id", "parent_asin"], keep="last")

    def eligible_pair(left: str, right: str) -> bool:
        if left == right:
            return False
        if not cross_subcategory:
            return True
        left_cat = item_to_subcategory.get(left, "")
        right_cat = item_to_subcategory.get(right, "")
        return bool(left_cat and right_cat and left_cat != right_cat)

    user_sizes = frame.groupby("user_id", sort=False).size()
    eligible_user_ids = [str(user) for user, size in user_sizes.items() if size >= 2]
    if validation_user_ids is None:
        shuffled_users = eligible_user_ids.copy()
        random.Random(seed).shuffle(shuffled_users)
        validation_count = max(1, min(len(shuffled_users) - 1, int(len(shuffled_users) * validation_user_ratio)))
        validation_user_ids = set(shuffled_users[:validation_count])
    else:
        validation_user_ids = {str(user) for user in validation_user_ids}

    train_pair_counts: Counter[tuple[str, str]] = Counter()
    validation_pairs: set[tuple[str, str]] = set()
    train_users = validation_users = 0

    for user, user_frame in frame.groupby("user_id", sort=False):
        if len(user_frame) < 2:
            continue
        user = str(user)
        events = list(user_frame[["parent_asin", "timestamp"]].itertuples(index=False, name=None))
        if user not in validation_user_ids:
            train_users += 1
            for left_idx, (left, left_time) in enumerate(events):
                for right, right_time in events[left_idx + 1 :]:
                    if eligible_pair(left, right) and _within_window(left_time, right_time, window_days):
                        train_pair_counts[_undirected_pair(left, right)] += 1
            continue

        validation_users += 1
        history, (target, target_time) = events[:-1], events[-1]
        for anchor, anchor_time in history:
            if eligible_pair(anchor, target) and _within_window(anchor_time, target_time, window_days):
                validation_pairs.add((anchor, target))

    held_out_undirected = {_undirected_pair(anchor, target) for anchor, target in validation_pairs}
    leaked_before_removal = sum(pair in train_pair_counts for pair in held_out_undirected)
    for pair in held_out_undirected:
        train_pair_counts.pop(pair, None)
    train_pair_counts = Counter(
        {pair: count for pair, count in train_pair_counts.items() if count >= min_train_count}
    )

    directed_counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (left, right), count in train_pair_counts.items():
        directed_counts[left][right] = count
        directed_counts[right][left] = count
    pmi_graph = compute_pmi(dict(directed_counts), min_pmi=min_pmi, min_count=min_train_count)

    # PMI is unstable for rare edges. Combine it with log support, then normalize
    # per anchor so the values can be used as soft multi-positive targets.
    weighted_graph: Dict[str, Dict[str, float]] = {}
    kept_counts: Dict[str, Dict[str, int]] = {}
    for anchor, neighbors in pmi_graph.items():
        scored = {}
        counts = {}
        for complement, pmi in neighbors.items():
            count = directed_counts[anchor][complement]
            scored[complement] = max(float(pmi), 1e-3) * math.log1p(count)
            counts[complement] = count
        if scored:
            mean_weight = sum(scored.values()) / len(scored)
            weighted_graph[anchor] = {
                item: weight / max(mean_weight, 1e-8) for item, weight in scored.items()
            }
            kept_counts[anchor] = counts

    validation: Dict[str, List[str]] = defaultdict(list)
    for anchor, target in sorted(validation_pairs):
        validation[anchor].append(target)

    train_edges = {
        _undirected_pair(anchor, complement)
        for anchor, neighbors in weighted_graph.items()
        for complement in neighbors
    }
    overlap = train_edges & held_out_undirected
    if overlap:
        raise AssertionError(f"Temporal split leaked {len(overlap)} held-out undirected edges")

    fingerprint_payload = {
        "train_edges": sorted(train_edges),
        "validation": sorted((anchor, target) for anchor, values in validation.items() for target in values),
        "window_days": window_days,
        "min_train_count": min_train_count,
        "min_pmi": min_pmi,
        "cross_subcategory": cross_subcategory,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "format_version": FORMAT_VERSION,
        "split": "user_disjoint_temporal_leave_last_out_strict_undirected_edge_holdout",
        "eligible_users": len(eligible_user_ids),
        "train_users": train_users,
        "validation_users": validation_users,
        "validation_user_ratio": validation_user_ratio,
        "seed": seed,
        "train_anchors": len(weighted_graph),
        "train_undirected_edges": len(train_edges),
        "validation_anchors": len(validation),
        "validation_directed_edges": sum(map(len, validation.values())),
        "heldout_edges_seen_before_strict_removal": leaked_before_removal,
        "post_split_edge_overlap": 0,
        "window_days": window_days,
        "min_train_count": min_train_count,
        "min_pmi": min_pmi,
        "cross_subcategory": cross_subcategory,
        "fingerprint": fingerprint,
    }
    if not weighted_graph:
        raise ValueError("Complementarity training produced no training edges")
    if not validation:
        raise ValueError("Complementarity training produced no validation edges")
    return ComplementaritySplit(weighted_graph, kept_counts, dict(validation), metadata)


def _weighted_sample_without_replacement(
    values: Sequence[str], weights: Sequence[float], count: int, rng: random.Random
) -> List[str]:
    remaining = list(zip(values, weights))
    selected = []
    while remaining and len(selected) < count:
        total = sum(max(weight, 0.0) for _, weight in remaining)
        if total <= 0:
            idx = rng.randrange(len(remaining))
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            idx = len(remaining) - 1
            for candidate_idx, (_, weight) in enumerate(remaining):
                cumulative += max(weight, 0.0)
                if cumulative >= threshold:
                    idx = candidate_idx
                    break
        selected.append(remaining.pop(idx)[0])
    return selected


class BalancedComplementarityDataset(Dataset):
    """One row per anchor with deterministic epoch-varying positive/negative sets."""

    def __init__(
        self,
        train_graph: Mapping[str, Mapping[str, float]],
        item_to_idx: Mapping[str, int],
        item_to_subcategory: Mapping[str, str],
        *,
        positives_per_anchor: int = 8,
        hard_negatives_per_anchor: int = 0,
        random_negatives_per_anchor: int = 8,
        cross_subcategory: bool = True,
        seed: int = 42,
    ):
        self.graph = {anchor: dict(neighbors) for anchor, neighbors in train_graph.items() if neighbors}
        self.anchors = sorted(self.graph)
        self.item_to_idx = dict(item_to_idx)
        self.item_ids = list(item_to_idx)
        self.item_to_subcategory = item_to_subcategory
        self.positives_per_anchor = positives_per_anchor
        self.hard_negatives_per_anchor = hard_negatives_per_anchor
        self.random_negatives_per_anchor = random_negatives_per_anchor
        self.cross_subcategory = cross_subcategory
        self.seed = seed
        self.epoch = 0
        self.hard_negatives: Mapping[str, Sequence[str]] = {}

        degrees = {item: len(self.graph.get(item, {})) for item in self.item_ids}
        self.degree_buckets: Dict[int, List[str]] = defaultdict(list)
        for item, degree in degrees.items():
            self.degree_buckets[max(0, degree).bit_length()].append(item)
        self.target_bucket = {}
        for anchor, neighbors in self.graph.items():
            positive_degrees = sorted(degrees.get(item, 0) for item in neighbors)
            median_degree = positive_degrees[len(positive_degrees) // 2]
            self.target_bucket[anchor] = max(0, median_degree).bit_length()

    def __len__(self) -> int:
        return len(self.anchors)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_hard_negatives(self, values: Mapping[str, Sequence[str]]) -> None:
        self.hard_negatives = values

    def _valid_negative(self, anchor: str, candidate: str, positives: set[str]) -> bool:
        if candidate == anchor or candidate in positives:
            return False
        if self.cross_subcategory:
            anchor_cat = self.item_to_subcategory.get(anchor, "")
            candidate_cat = self.item_to_subcategory.get(candidate, "")
            if not anchor_cat or not candidate_cat or anchor_cat == candidate_cat:
                return False
        return True

    def __getitem__(self, index: int) -> Dict[str, object]:
        anchor = self.anchors[index]
        neighbors = self.graph[anchor]
        rng = random.Random(f"{self.seed}:{self.epoch}:{anchor}")
        positive_items = _weighted_sample_without_replacement(
            list(neighbors), list(neighbors.values()), self.positives_per_anchor, rng
        )
        positive_set = set(neighbors)

        negatives = []
        for candidate in self.hard_negatives.get(anchor, ()):
            if self._valid_negative(anchor, candidate, positive_set) and candidate not in negatives:
                negatives.append(candidate)
            if len(negatives) >= self.hard_negatives_per_anchor:
                break

        target_count = self.hard_negatives_per_anchor + self.random_negatives_per_anchor
        pools = [self.degree_buckets.get(self.target_bucket[anchor], ()), self.item_ids]
        for pool in pools:
            if not pool:
                continue
            attempts = 0
            max_attempts = max(100, target_count * 20)
            while len(negatives) < target_count and attempts < max_attempts:
                candidate = rng.choice(pool)
                attempts += 1
                if self._valid_negative(anchor, candidate, positive_set) and candidate not in negatives:
                    negatives.append(candidate)
            if len(negatives) >= target_count:
                break
        if not negatives:
            raise ValueError(f"No valid complementarity negatives for {anchor}")

        return {
            "anchor_idx": self.item_to_idx[anchor],
            "positive_indices": [self.item_to_idx[item] for item in positive_items],
            "positive_weights": [neighbors[item] for item in positive_items],
            "known_positive_indices": [self.item_to_idx[item] for item in neighbors],
            "negative_indices": [self.item_to_idx[item] for item in negatives],
        }


def complementarity_collate(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        "anchor_indices": torch.tensor([row["anchor_idx"] for row in batch], dtype=torch.long),
        "positive_indices": [row["positive_indices"] for row in batch],
        "positive_weights": [row["positive_weights"] for row in batch],
        "known_positive_indices": [row["known_positive_indices"] for row in batch],
        "negative_indices": [row["negative_indices"] for row in batch],
    }


class ComplementarityModel(nn.Module):
    def __init__(
        self, input_dim: int, proj_dim: int = 128, dropout: float = 0.0, temperature: float = 0.05
    ):
        super().__init__()
        hidden = proj_dim * 2
        self.anchor_head = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, proj_dim))
        self.complement_head = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, proj_dim))
        self.dropout = nn.Dropout(dropout)
        self.temperature = temperature

    def project_anchor(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.anchor_head(self.dropout(values)), dim=-1)

    def project_complement(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.complement_head(self.dropout(values)), dim=-1)


def multi_positive_complementarity_loss(
    model: ComplementarityModel,
    base_embs: torch.Tensor,
    batch: Mapping[str, object],
) -> tuple[torch.Tensor, Dict[str, float]]:
    device = base_embs.device
    anchor_indices = batch["anchor_indices"].to(device)
    candidates = sorted(
        {
            idx
            for field in ("positive_indices", "negative_indices")
            for values in batch[field]
            for idx in values
        }
    )
    if not candidates:
        raise ValueError("Complementarity batch has no candidates")
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    candidate_positions = {item_idx: position for position, item_idx in enumerate(candidates)}

    anchor_proj = model.project_anchor(base_embs[anchor_indices])
    complement_proj = model.project_complement(base_embs[candidate_tensor])
    logits = anchor_proj @ complement_proj.T / model.temperature
    valid_mask = torch.ones_like(logits, dtype=torch.bool)
    positive_mask = torch.zeros_like(logits, dtype=torch.bool)
    log_weights = torch.full_like(logits, -torch.inf)

    for row, (positives, weights, known_positives) in enumerate(
        zip(batch["positive_indices"], batch["positive_weights"], batch["known_positive_indices"])
    ):
        for item_idx in known_positives:
            position = candidate_positions.get(item_idx)
            if position is not None:
                valid_mask[row, position] = False
        weight_tensor = torch.tensor(weights, dtype=logits.dtype, device=device)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-8)
        for item_idx, log_weight in zip(positives, weight_tensor.log()):
            position = candidate_positions[item_idx]
            valid_mask[row, position] = True
            positive_mask[row, position] = True
            log_weights[row, position] = log_weight

    denominator = torch.logsumexp(logits.masked_fill(~valid_mask, -torch.inf), dim=1)
    numerator = torch.logsumexp((logits + log_weights).masked_fill(~positive_mask, -torch.inf), dim=1)
    loss = (denominator - numerator).mean()

    with torch.no_grad():
        negative_mask = valid_mask & ~positive_mask
        best_positive = logits.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
        best_negative = logits.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
        accuracy = (best_positive > best_negative).float().mean().item()
    return loss, {
        "loss": loss.item(),
        "candidate_top1_accuracy": accuracy,
        "candidate_count": float(len(candidates)),
    }


def _candidate_mask(
    anchor: str,
    item_ids: Sequence[str],
    item_to_subcategory: Mapping[str, str],
    cross_subcategory: bool,
    *,
    item_to_idx: Optional[Mapping[str, int]] = None,
    category_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = torch.ones(len(item_ids), dtype=torch.bool)
    item_to_idx = item_to_idx or {item: idx for idx, item in enumerate(item_ids)}
    anchor_idx = item_to_idx.get(anchor)
    if anchor_idx is not None:
        mask[anchor_idx] = False
    if cross_subcategory:
        if category_ids is None:
            category_names: Dict[str, int] = {}
            category_ids = torch.zeros(len(item_ids), dtype=torch.long)
            for idx, item in enumerate(item_ids):
                category = item_to_subcategory.get(item, "")
                if category:
                    category_ids[idx] = category_names.setdefault(category, len(category_names) + 1)
        anchor_category = category_ids[anchor_idx] if anchor_idx is not None else torch.tensor(0)
        mask &= (category_ids != 0) & (anchor_category != 0) & (category_ids != anchor_category)
    return mask


def _category_id_tensor(
    item_ids: Sequence[str], item_to_subcategory: Mapping[str, str]
) -> torch.Tensor:
    category_names: Dict[str, int] = {}
    result = torch.zeros(len(item_ids), dtype=torch.long)
    for idx, item in enumerate(item_ids):
        category = item_to_subcategory.get(item, "")
        if category:
            result[idx] = category_names.setdefault(category, len(category_names) + 1)
    return result


def compute_complementarity_ranking_metrics(
    anchor_embs: torch.Tensor,
    complement_embs: torch.Tensor,
    item_ids: Sequence[str],
    relevance: Mapping[str, Sequence[str]],
    item_to_subcategory: Mapping[str, str],
    *,
    cross_subcategory: bool = True,
    ks: Sequence[int] = (5, 10),
    candidate_bias: Optional[torch.Tensor] = None,
    chunk_size: int = 256,
) -> Dict[str, float]:
    """Compute macro standard Hit, Recall, Precision, MRR and binary nDCG."""
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    category_ids = _category_id_tensor(item_ids, item_to_subcategory)
    anchors = [anchor for anchor, positives in relevance.items() if anchor in item_to_idx and positives]
    totals = defaultdict(float)
    evaluated = 0
    pair_count = 0
    max_k = max(ks)
    anchor_embs = anchor_embs.float().cpu()
    complement_embs = complement_embs.float().cpu()
    if candidate_bias is not None:
        candidate_bias = candidate_bias.float().cpu()

    for start in range(0, len(anchors), chunk_size):
        chunk = anchors[start : start + chunk_size]
        anchor_idx = torch.tensor([item_to_idx[item] for item in chunk])
        scores = anchor_embs[anchor_idx] @ complement_embs.T
        if candidate_bias is not None:
            scores = candidate_bias.unsqueeze(0).expand(len(chunk), -1).clone()
        for row, anchor in enumerate(chunk):
            valid_mask = _candidate_mask(
                anchor,
                item_ids,
                item_to_subcategory,
                cross_subcategory,
                item_to_idx=item_to_idx,
                category_ids=category_ids,
            )
            positive_indices = [
                item_to_idx[item] for item in set(relevance[anchor])
                if item in item_to_idx and valid_mask[item_to_idx[item]]
            ]
            if not positive_indices:
                continue
            row_scores = scores[row].clone()
            row_scores[~valid_mask] = -torch.inf
            top_indices = row_scores.topk(min(max_k, int(valid_mask.sum()))).indices.tolist()
            positive_set = set(positive_indices)
            best_positive_score = row_scores[positive_indices].max()
            first_rank = 1 + int((row_scores > best_positive_score).sum().item())
            totals["mrr"] += 1.0 / first_rank
            pair_count += len(positive_set)
            evaluated += 1
            for k in ks:
                selected = top_indices[:k]
                hits = sum(index in positive_set for index in selected)
                totals[f"hit_at_{k}"] += float(hits > 0)
                totals[f"recall_at_{k}"] += hits / len(positive_set)
                totals[f"precision_at_{k}"] += hits / k
                dcg = sum(
                    (1.0 / math.log2(rank + 2)) if index in positive_set else 0.0
                    for rank, index in enumerate(selected)
                )
                ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(positive_set))))
                totals[f"ndcg_at_{k}"] += dcg / max(ideal, 1e-8)

    metrics = {key: value / max(evaluated, 1) for key, value in totals.items()}
    metrics.update({"anchors": evaluated, "pairs": pair_count})
    return metrics


def mine_complementarity_hard_negatives(
    model: ComplementarityModel,
    base_embs: torch.Tensor,
    train_graph: Mapping[str, Mapping[str, float]],
    item_ids: Sequence[str],
    item_to_subcategory: Mapping[str, str],
    *,
    negatives_per_anchor: int = 4,
    top_k: int = 100,
    positive_margin: float = 0.02,
    cross_subcategory: bool = True,
    chunk_size: int = 128,
) -> Dict[str, List[str]]:
    if negatives_per_anchor <= 0:
        return {}
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    category_ids = _category_id_tensor(item_ids, item_to_subcategory)
    anchors = [anchor for anchor in sorted(train_graph) if anchor in item_to_idx]
    model.eval()
    with torch.no_grad():
        anchor_proj = model.project_anchor(base_embs)
        complement_proj = model.project_complement(base_embs)
    result = {}
    for start in tqdm(range(0, len(anchors), chunk_size), desc="Mining complement hard negatives"):
        chunk = anchors[start : start + chunk_size]
        chunk_idx = torch.tensor([item_to_idx[item] for item in chunk], device=base_embs.device)
        scores = anchor_proj[chunk_idx] @ complement_proj.T
        for row, anchor in enumerate(chunk):
            known = {item_to_idx[item] for item in train_graph[anchor] if item in item_to_idx}
            valid = _candidate_mask(
                anchor,
                item_ids,
                item_to_subcategory,
                cross_subcategory,
                item_to_idx=item_to_idx,
                category_ids=category_ids,
            ).to(scores.device)
            if known:
                known_tensor = torch.tensor(sorted(known), device=scores.device)
                valid[known_tensor] = False
                best_positive = scores[row, known_tensor].max()
                valid &= scores[row] <= best_positive - positive_margin
            available = int(valid.sum().item())
            if not available:
                result[anchor] = []
                continue
            candidate_count = min(max(top_k, negatives_per_anchor), available)
            candidates = scores[row].masked_fill(~valid, -torch.inf).topk(candidate_count).indices.tolist()
            result[anchor] = [item_ids[idx] for idx in candidates[:negatives_per_anchor]]
    return result


def _degree_popularity_bias(
    train_graph: Mapping[str, Mapping[str, float]], item_ids: Sequence[str]
) -> torch.Tensor:
    values = torch.tensor([math.log1p(len(train_graph.get(item, {}))) for item in item_ids])
    return values


def train_complementarity(
    split: ComplementaritySplit,
    item_ids: Sequence[str],
    item_to_subcategory: Mapping[str, str],
    base_embs: torch.Tensor,
    *,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 0.00040792119058488896,
    proj_dim: int = 256,
    dropout: float = 0.1,
    temperature: float = 0.081621255477171,
    patience: int = 2,
    positives_per_anchor: int = 8,
    hard_negatives_per_anchor: int = 0,
    random_negatives_per_anchor: int = 8,
    hard_negative_candidates: int = 100,
    hard_negative_margin: float = 0.02,
    cross_subcategory: bool = True,
    save_dir: str = "outputs",
    backbone: str = "qwen",
    seed: int = 42,
    checkpoint_callback=None,
    resume_from_checkpoint: bool = False,
    checkpoint_every: int = 0,
) -> tuple[ComplementarityModel, List[Dict[str, object]], Dict[str, object]]:
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every cannot be negative")
    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "proj_dim": proj_dim,
        "dropout": dropout,
        "temperature": temperature,
        "positives_per_anchor": positives_per_anchor,
        "hard_negatives_per_anchor": hard_negatives_per_anchor,
        "random_negatives_per_anchor": random_negatives_per_anchor,
        "hard_negative_candidates": hard_negative_candidates,
        "hard_negative_margin": hard_negative_margin,
        "cross_subcategory": cross_subcategory,
        "seed": seed,
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    dataset = BalancedComplementarityDataset(
        split.train_graph,
        item_to_idx,
        item_to_subcategory,
        positives_per_anchor=positives_per_anchor,
        hard_negatives_per_anchor=hard_negatives_per_anchor,
        random_negatives_per_anchor=random_negatives_per_anchor,
        cross_subcategory=cross_subcategory,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=complementarity_collate)
    model = ComplementarityModel(base_embs.shape[1], proj_dim, dropout, temperature).to(base_embs.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    base_cpu = F.normalize(base_embs.float(), dim=-1).cpu()
    baseline = compute_complementarity_ranking_metrics(
        base_cpu, base_cpu, item_ids, split.validation, item_to_subcategory,
        cross_subcategory=cross_subcategory,
    )
    popularity = compute_complementarity_ranking_metrics(
        base_cpu, base_cpu, item_ids, split.validation, item_to_subcategory,
        cross_subcategory=cross_subcategory,
        candidate_bias=_degree_popularity_bias(split.train_graph, item_ids),
    )
    print(
        f"Complementarity baselines | frozen MRR={baseline['mrr']:.4f}, nDCG@10={baseline['ndcg_at_10']:.2%} | "
        f"degree-popularity MRR={popularity['mrr']:.4f}, nDCG@10={popularity['ndcg_at_10']:.2%}"
    )

    models_dir = os.path.join(save_dir, "models")
    metrics_dir = os.path.join(save_dir, "metrics")
    checkpoint_dir = os.path.join(models_dir, f"comp_checkpoints_{backbone}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    best_path = os.path.join(models_dir, f"comp_model_{backbone}.pt")
    history_path = os.path.join(metrics_dir, f"complementarity_history_{backbone}.json")

    history: List[Dict[str, object]] = []
    best_score = float("-inf")
    best_epoch = 0
    patience_counter = 0
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(latest_path):
        # RNG snapshots must remain CPU ByteTensors, including CUDA RNG states.
        # load_state_dict moves model/optimizer tensors onto their parameter devices.
        resume = torch.load(latest_path, map_location="cpu", weights_only=False)
        if resume.get("format_version") != FORMAT_VERSION:
            raise ValueError("Complementarity checkpoint has an unsupported format")
        if resume.get("split_fingerprint") != split.metadata["fingerprint"]:
            raise ValueError("Complementarity split changed; refusing an incompatible resume")
        configured_epochs = resume.get("training_epochs")
        if configured_epochs is not None and int(configured_epochs) != epochs:
            raise ValueError(
                "Complementarity training horizon changed; refusing to resume with "
                "an incompatible cosine scheduler. Start a fresh run ID instead."
            )
        saved_config = resume.get("training_config")
        if saved_config is not None and saved_config != training_config:
            raise ValueError(
                "Complementarity training configuration changed; refusing an "
                "incompatible resume. Start a fresh run ID instead."
            )
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        history = resume.get("history", [])
        best_score = float(resume.get("best_score", float("-inf")))
        best_epoch = int(resume.get("best_epoch", 0))
        patience_counter = int(resume.get("patience_counter", 0))
        start_epoch = int(resume.get("epoch", 0))
        rng_state = resume.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda"):
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        print(f"Resuming complementarity from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        hard_negatives = mine_complementarity_hard_negatives(
            model, base_embs, split.train_graph, item_ids, item_to_subcategory,
            negatives_per_anchor=hard_negatives_per_anchor,
            top_k=hard_negative_candidates,
            positive_margin=hard_negative_margin,
            cross_subcategory=cross_subcategory,
        )
        dataset.set_hard_negatives(hard_negatives)
        model.train()
        loss_total = accuracy_total = 0.0
        progress = tqdm(loader, desc=f"Complementarity epoch {epoch + 1}/{epochs}")
        for batch in progress:
            loss, train_metrics = multi_positive_complementarity_loss(model, base_embs, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            accuracy_total += train_metrics["candidate_top1_accuracy"]
            progress.set_postfix(loss=f"{loss.item():.4f}", top1=f"{train_metrics['candidate_top1_accuracy']:.2%}")
        scheduler.step()

        model.eval()
        with torch.no_grad():
            anchor_proj = model.project_anchor(base_embs).cpu()
            complement_proj = model.project_complement(base_embs).cpu()
        validation = compute_complementarity_ranking_metrics(
            anchor_proj, complement_proj, item_ids, split.validation, item_to_subcategory,
            cross_subcategory=cross_subcategory,
        )
        score = validation["ndcg_at_10"]
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": loss_total / len(loader),
            "train_candidate_top1_accuracy": accuracy_total / len(loader),
            "validation": validation,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_info)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        print(
            f"Complementarity epoch {epoch + 1}: loss={epoch_info['train_loss']:.4f}, "
            f"candidate_top1={epoch_info['train_candidate_top1_accuracy']:.2%} | "
            f"val MRR={validation['mrr']:.4f}, nDCG@10={validation['ndcg_at_10']:.2%}, "
            f"Recall@10={validation['recall_at_10']:.2%}"
        )

        checkpoint = {
            "format_version": FORMAT_VERSION,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "input_dim": base_embs.shape[1],
            "proj_dim": proj_dim,
            "temperature": temperature,
            "training_epochs": epochs,
            "training_config": training_config,
            "split_fingerprint": split.metadata["fingerprint"],
            "best_score": best_score,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "baseline": baseline,
            "popularity_baseline": popularity,
            "history": history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
        state_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        expected_bytes = max(16 * 1024 * 1024, state_bytes * 4)
        retain_epoch = bool(checkpoint_every and (epoch + 1) % checkpoint_every == 0)
        write_count = 1 + int(is_best) + int(retain_epoch)
        check_ephemeral_budget(
            expected_bytes * write_count,
            reason="complementarity epoch checkpoints",
        )
        atomic_torch_save(checkpoint, latest_path, expected_bytes=expected_bytes)
        uploaded = [latest_path]
        if is_best:
            atomic_torch_save(checkpoint, best_path, expected_bytes=expected_bytes)
            uploaded.append(best_path)
        if retain_epoch:
            epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:04d}.pt")
            atomic_torch_save(checkpoint, epoch_path, expected_bytes=expected_bytes)
            uploaded.append(epoch_path)
        atomic_json_dump(
            {
                "format_version": FORMAT_VERSION,
                "split": split.metadata,
                "baseline": baseline,
                "popularity_baseline": popularity,
                "best_epoch": best_epoch,
                "best_ndcg_at_10": best_score,
                "improves_frozen_baseline": best_score > baseline["ndcg_at_10"],
                "history": history,
            },
            history_path,
            indent=2,
        )
        if checkpoint_callback:
            checkpoint_callback(uploaded + [history_path], epoch + 1, is_best)
        if patience_counter >= patience:
            print(f"Complementarity early stopping at epoch {epoch + 1}")
            break

    if os.path.exists(best_path):
        selected = torch.load(best_path, map_location=base_embs.device, weights_only=False)
        model.load_state_dict(selected["model_state_dict"])
    result = {
        "baseline": baseline,
        "popularity_baseline": popularity,
        "best_epoch": best_epoch,
        "best_ndcg_at_10": best_score,
        "improves_frozen_baseline": best_score > baseline["ndcg_at_10"],
    }
    return model, history, result


def run_stage4(args, save_dir: str) -> None:
    data_dir = os.path.join(save_dir, "data")
    embeddings_dir = os.path.join(save_dir, "embeddings")
    metrics_dir = os.path.join(save_dir, "metrics")
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    backbone = args.backbone
    stable_inputs = [
        "data/product_texts.json",
        "data/subcategories.json",
        "data/interactions.parquet",
        f"embeddings/item_embeddings_{backbone}.pt",
    ]
    restore_files(args.hf_repo, hf_category(args), save_dir, stable_inputs)
    if getattr(args, "resume", False):
        restore_files(
            args.hf_repo,
            hf_run(args),
            save_dir,
            [
                f"models/comp_checkpoints_{backbone}/latest.pt",
                f"models/comp_model_{backbone}.pt",
                f"metrics/complementarity_history_{backbone}.json",
            ],
        )
    missing = [name for name in stable_inputs if not os.path.exists(os.path.join(save_dir, name))]
    if missing:
        raise FileNotFoundError("Missing complementarity inputs:\n- " + "\n- ".join(missing))

    with open(os.path.join(data_dir, "product_texts.json")) as handle:
        product_texts = json.load(handle)
    with open(os.path.join(data_dir, "subcategories.json")) as handle:
        item_to_subcategory = json.load(handle)["item_to_subcategory"]
    interactions = pd.read_parquet(os.path.join(data_dir, "interactions.parquet"))
    split_manifest_path = getattr(args, "benchmark_split_manifest", None)
    benchmark_manifest = None
    if split_manifest_path:
        benchmark_manifest = load_user_split_manifest(split_manifest_path)
        if benchmark_manifest.get("category") not in (None, args.category):
            raise ValueError(
                "Benchmark split category does not match this build: "
                f"{benchmark_manifest.get('category')} != {args.category}"
            )
        before = interactions["user_id"].astype(str).nunique()
        interactions = filter_interactions_to_split(
            interactions, benchmark_manifest, "shared_train"
        )
        after = interactions["user_id"].astype(str).nunique()
        if not after:
            raise ValueError("Benchmark shared_train contains no stage-1 interactions")
        print(
            "Applied protected benchmark split before complementarity graph building: "
            f"{after:,}/{before:,} interaction users retained (shared_train only)."
        )
    embedding_data = torch.load(
        os.path.join(embeddings_dir, f"item_embeddings_{backbone}.pt"),
        map_location="cpu",
        weights_only=False,
    )
    item_ids = list(product_texts)
    if embedding_data.get("item_ids") != item_ids:
        raise ValueError("Complementarity item embeddings are misaligned with product_texts")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_embs = embedding_data["item_embs"].to(device)

    window = args.copurchase_window_days if args.copurchase_window_days >= 0 else None
    print("Building leakage-free temporal complementarity split...")
    split = build_temporal_complementarity_split(
        interactions,
        item_ids,
        item_to_subcategory,
        window_days=window,
        min_train_count=args.comp_min_pair_count,
        min_pmi=args.min_pmi,
        cross_subcategory=args.comp_cross_subcategory,
        validation_user_ratio=args.comp_validation_user_ratio,
    )
    if benchmark_manifest is not None:
        split.metadata["benchmark"] = public_split_metadata(benchmark_manifest)
        split.metadata["benchmark"]["source_partition"] = "shared_train"
        split.metadata["benchmark"]["interaction_users_retained"] = after
        split.metadata["benchmark"]["selection_policy"] = (
            "user_disjoint_internal_holdout_within_shared_train"
        )
    split_path = os.path.join(metrics_dir, "complementarity_split.json")
    atomic_json_dump(split.metadata, split_path, indent=2)
    print(
        f"Complementarity split: {split.metadata['train_anchors']:,} train anchors, "
        f"{split.metadata['train_undirected_edges']:,} train edges, "
        f"{split.metadata['validation_anchors']:,} validation anchors, "
        f"{split.metadata['validation_directed_edges']:,} validation edges; overlap=0"
    )

    def publish_checkpoint(paths, epoch, is_best):
        if not args.hf_repo:
            return
        relative = [os.path.relpath(path, save_dir) for path in paths]
        label = " + best" if is_best else ""
        print(f"Publishing resumable complementarity epoch {epoch}{label}...")
        upload_files(args.hf_repo, hf_run(args), save_dir, relative)

    model, history, result = train_complementarity(
        split,
        item_ids,
        item_to_subcategory,
        base_embs,
        epochs=args.comp_epochs,
        batch_size=args.comp_batch_size,
        lr=args.comp_lr,
        proj_dim=args.comp_proj_dim,
        dropout=args.comp_dropout,
        temperature=args.comp_temperature,
        patience=args.comp_patience,
        positives_per_anchor=args.comp_positives_per_anchor,
        hard_negatives_per_anchor=args.comp_hard_negatives,
        random_negatives_per_anchor=args.comp_random_negatives,
        hard_negative_candidates=args.comp_hard_negative_candidates,
        hard_negative_margin=args.comp_hard_negative_margin,
        cross_subcategory=args.comp_cross_subcategory,
        save_dir=save_dir,
        backbone=backbone,
        checkpoint_callback=publish_checkpoint,
        resume_from_checkpoint=getattr(args, "resume", False),
        checkpoint_every=getattr(args, "comp_checkpoint_every", 0),
    )

    model.eval()
    with torch.no_grad():
        anchor_embs = model.project_anchor(base_embs).cpu().float()
        complement_embs = model.project_complement(base_embs).cpu().float()
    embeddings_path = os.path.join(embeddings_dir, f"comp_embeddings_{backbone}.pt")
    atomic_torch_save(
        {
            "format_version": FORMAT_VERSION,
            "anchor_embs": anchor_embs,
            "complement_embs": complement_embs,
            "item_ids": item_ids,
            "split_fingerprint": split.metadata["fingerprint"],
            "selection": result,
        },
        embeddings_path,
        expected_bytes=(anchor_embs.numel() + complement_embs.numel()) * 4,
    )

    final_files = [
        f"models/comp_model_{backbone}.pt",
        f"embeddings/comp_embeddings_{backbone}.pt",
        f"metrics/complementarity_history_{backbone}.json",
        "metrics/complementarity_split.json",
    ]
    if args.hf_repo:
        upload_files(args.hf_repo, hf_run(args), save_dir, final_files)
        stable_publish_allowed = not split_manifest_path or getattr(
            args, "allow_benchmark_stable_publish", False
        )
        if (
            result["improves_frozen_baseline"]
            and getattr(args, "publish_stable", True)
            and stable_publish_allowed
        ):
            upload_files(args.hf_repo, hf_category(args), save_dir, final_files)
            print("Published complementarity artifacts to the stable category path.")
        else:
            reason = (
                "did not beat frozen baseline"
                if not result["improves_frozen_baseline"]
                else "stable publishing disabled"
            )
            print(f"Kept complementarity artifacts run-scoped: {reason}.")
        prune_verified_files(
            save_dir,
            [f"models/comp_checkpoints_{backbone}/latest.pt", f"models/comp_model_{backbone}.pt"],
        )
    print(
        f"Complementarity done: best epoch={result['best_epoch']}, "
        f"best nDCG@10={result['best_ndcg_at_10']:.2%}, "
        f"frozen baseline={result['baseline']['ndcg_at_10']:.2%}"
    )
