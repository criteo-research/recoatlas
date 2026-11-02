"""Collision-aware data, loss, mining, and metrics for query-to-product retrieval."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


OBJECTIVE_NAME = "multi_positive_infonce"


def normalize_query(query: str) -> str:
    """Normalize only distinctions that a search box would not preserve."""
    return " ".join((query or "").lower().strip().split())


@dataclass(frozen=True)
class QueryGroup:
    query: str
    positive_item_ids: Tuple[str, ...]
    occurrences: int


def build_query_groups(
    queries: Mapping[Tuple[str, str], str],
    eligible_item_ids: Optional[Iterable[str]] = None,
) -> List[QueryGroup]:
    """Collapse user-weighted rows into unique queries with all known positives."""
    eligible = set(eligible_item_ids) if eligible_item_ids is not None else None
    positives = defaultdict(set)
    occurrences = defaultdict(int)
    for (_, item_id), raw_query in queries.items():
        if eligible is not None and item_id not in eligible:
            continue
        query = normalize_query(raw_query)
        if not query:
            continue
        positives[query].add(item_id)
        occurrences[query] += 1
    return [
        QueryGroup(query, tuple(sorted(items)), occurrences[query])
        for query, items in sorted(positives.items())
    ]


def merge_positive_sets(*group_collections: Sequence[QueryGroup]) -> Dict[str, set[str]]:
    """Build query-level relevance sets across product-disjoint splits."""
    merged = defaultdict(set)
    for groups in group_collections:
        for group in groups:
            merged[group.query].update(group.positive_item_ids)
    return dict(merged)


def query_group_fingerprint(groups: Sequence[QueryGroup]) -> str:
    """Stable identity for a query split used by resumable mining artifacts."""
    digest = hashlib.sha256()
    for group in groups:
        digest.update(group.query.encode("utf-8"))
        digest.update(b"\0")
        for item_id in group.positive_item_ids:
            digest.update(item_id.encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class MultiPositiveQueryDataset(Dataset):
    """One row per unique query with collision-aware positives and negatives."""

    def __init__(
        self,
        groups: Sequence[QueryGroup],
        item_to_index: Mapping[str, int],
        candidate_item_ids: Iterable[str],
        *,
        hard_negatives: Optional[Mapping[str, Sequence[str]]] = None,
        hard_negatives_per_query: int = 4,
        random_negatives_per_query: int = 4,
        seed: int = 42,
        dynamic: bool = True,
    ):
        self.groups = list(groups)
        self.item_to_index = dict(item_to_index)
        self.candidate_item_ids = tuple(
            item_id for item_id in candidate_item_ids if item_id in self.item_to_index
        )
        self.candidate_item_set = set(self.candidate_item_ids)
        if not self.groups:
            raise ValueError("Multi-positive dataset requires at least one query")
        if not self.candidate_item_ids:
            raise ValueError("Negative-sampling candidate set is empty")
        self.hard_negatives = hard_negatives or {}
        self.hard_negatives_per_query = max(0, int(hard_negatives_per_query))
        self.random_negatives_per_query = max(0, int(random_negatives_per_query))
        self.seed = seed
        self.dynamic = dynamic
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups)

    def _negative_items(self, group: QueryGroup) -> List[str]:
        forbidden = set(group.positive_item_ids)
        selected = []
        selected_set = set()
        for item_id in self.hard_negatives.get(group.query, ()):
            if (
                item_id in self.item_to_index
                and item_id in self.candidate_item_set
                and item_id not in forbidden
                and item_id not in selected_set
            ):
                selected.append(item_id)
                selected_set.add(item_id)
                if len(selected) >= self.hard_negatives_per_query:
                    break

        target = self.hard_negatives_per_query + self.random_negatives_per_query
        sampling_epoch = self.epoch if self.dynamic else 0
        rng = random.Random(_stable_seed(self.seed, sampling_epoch, group.query))
        attempts = 0
        max_attempts = max(100, target * 20)
        while len(selected) < target and attempts < max_attempts:
            attempts += 1
            item_id = self.candidate_item_ids[rng.randrange(len(self.candidate_item_ids))]
            if item_id in forbidden or item_id in selected_set:
                continue
            selected.append(item_id)
            selected_set.add(item_id)
        if not selected:
            raise ValueError(f"No valid negatives remain for query {group.query!r}")
        return selected

    def __getitem__(self, index: int) -> dict:
        group = self.groups[index]
        return {
            "query": group.query,
            "positive_indices": [
                self.item_to_index[item_id]
                for item_id in group.positive_item_ids
                if item_id in self.item_to_index
            ],
            "negative_indices": [
                self.item_to_index[item_id] for item_id in self._negative_items(group)
            ],
        }


def multi_positive_collate(batch: Sequence[dict]) -> dict:
    return {
        "queries": [row["query"] for row in batch],
        "positive_indices": [row["positive_indices"] for row in batch],
        "negative_indices": [row["negative_indices"] for row in batch],
    }


def multi_positive_info_nce(
    query_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    positive_indices: Sequence[Sequence[int]],
    negative_indices: Sequence[Sequence[int]],
    *,
    temperature: float = 0.05,
) -> Tuple[torch.Tensor, dict]:
    """Listwise multi-positive InfoNCE with collision-masked in-batch items."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(query_embeddings) != len(positive_indices) or len(query_embeddings) != len(negative_indices):
        raise ValueError("Every query needs positive and negative item indices")
    if any(not values for values in positive_indices):
        raise ValueError("Every query needs at least one positive item")

    candidate_indices = sorted({
        index
        for rows in (positive_indices, negative_indices)
        for values in rows
        for index in values
    })
    candidate_lookup = {item_index: offset for offset, item_index in enumerate(candidate_indices)}
    candidate_tensor = torch.tensor(candidate_indices, device=item_embeddings.device)
    candidates = item_embeddings.index_select(0, candidate_tensor).to(
        dtype=query_embeddings.dtype
    )
    logits = query_embeddings @ candidates.T / temperature

    positive_mask = torch.zeros_like(logits, dtype=torch.bool)
    explicit_negative_mask = torch.zeros_like(logits, dtype=torch.bool)
    for row, values in enumerate(positive_indices):
        positive_mask[row, [candidate_lookup[index] for index in values]] = True
    for row, values in enumerate(negative_indices):
        valid = [candidate_lookup[index] for index in values if index not in set(positive_indices[row])]
        if valid:
            explicit_negative_mask[row, valid] = True

    # Other queries' positives become in-batch candidates. If an item is also a
    # known positive for this query, positive_mask promotes it instead of
    # incorrectly treating it as a negative.
    log_numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
    loss = (torch.logsumexp(logits, dim=1) - log_numerator).mean()

    with torch.no_grad():
        candidate_negative_mask = ~positive_mask
        best_positive = logits.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
        best_negative = logits.masked_fill(~candidate_negative_mask, -torch.inf).max(dim=1).values
        top1_accuracy = (best_positive > best_negative).float().mean().item()
        pairwise = []
        for row in range(len(logits)):
            positives = logits[row][positive_mask[row]]
            negatives = logits[row][candidate_negative_mask[row]]
            pairwise.append((positives[:, None] > negatives[None, :]).float().mean())
        explicit_count = int(explicit_negative_mask.sum().item())

    return loss, {
        "contrastive_loss": loss.item(),
        "candidate_top1_accuracy": top1_accuracy,
        "pairwise_accuracy": torch.stack(pairwise).mean().item(),
        "candidate_count": len(candidate_indices),
        "explicit_negative_count": explicit_count,
    }


def compute_multi_positive_retrieval_metrics(
    query_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    item_ids: Sequence[str],
    relevant_item_ids: Sequence[Sequence[str]],
    *,
    ks: Tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 256,
) -> Dict[str, float]:
    """Evaluate any-hit, recall, nDCG, and best-positive rank per unique query."""
    if len(query_embeddings) != len(relevant_item_ids):
        raise ValueError("Each query must have a relevance set")
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    max_k = min(max(ks), len(item_ids))
    totals = defaultdict(float)
    best_ranks = []
    valid = 0
    for start in range(0, len(query_embeddings), chunk_size):
        query_chunk = query_embeddings[start:start + chunk_size].to(
            device=item_embeddings.device,
            dtype=item_embeddings.dtype,
        )
        scores = query_chunk @ item_embeddings.T
        top_indices = scores.topk(max_k, dim=1).indices.cpu().tolist()
        for local_row in range(len(query_chunk)):
            source_row = start + local_row
            relevant = {
                item_to_index[item_id]
                for item_id in relevant_item_ids[source_row]
                if item_id in item_to_index
            }
            if not relevant:
                continue
            valid += 1
            relevant_tensor = torch.tensor(sorted(relevant), device=scores.device)
            best_score = scores[local_row].index_select(0, relevant_tensor).max()
            best_rank = 1 + int((scores[local_row] > best_score).sum().item())
            best_ranks.append(best_rank)
            for k in ks:
                top = top_indices[local_row][: min(k, len(item_ids))]
                hits = [rank for rank, index in enumerate(top, start=1) if index in relevant]
                totals[f"hit_at_{k}"] += float(bool(hits))
                totals[f"recall_at_{k}"] += len(hits) / len(relevant)
                dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
                ideal = sum(
                    1.0 / math.log2(rank + 1)
                    for rank in range(1, min(len(relevant), k) + 1)
                )
                totals[f"ndcg_at_{k}"] += dcg / ideal if ideal else 0.0
    if not valid:
        raise ValueError("No relevant products exist in the evaluation catalog")
    result = {key: value / valid for key, value in totals.items()}
    ranks = torch.tensor(best_ranks, dtype=torch.float32)
    result.update({
        "mrr": ranks.reciprocal().mean().item(),
        "mean_rank": ranks.mean().item(),
        "median_rank": ranks.median().item(),
        "valid_queries": valid,
        "catalog_items": len(item_ids),
    })
    return result


def mine_positive_aware_hard_negatives(
    encode_queries,
    groups: Sequence[QueryGroup],
    item_embeddings: torch.Tensor,
    item_ids: Sequence[str],
    candidate_item_ids: Iterable[str],
    *,
    top_k: int = 100,
    negatives_per_query: int = 4,
    positive_ratio: float = 0.95,
    batch_size: int = 64,
    desc: str = "Mining hard negatives",
) -> Dict[str, List[str]]:
    """Mine global model-hard negatives while filtering positive-like candidates."""
    if not 0 < positive_ratio < 1:
        raise ValueError("positive_ratio must be between zero and one")
    if negatives_per_query <= 0:
        return {group.query: [] for group in groups}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    candidate_indices = [item_to_index[item_id] for item_id in candidate_item_ids if item_id in item_to_index]
    if not candidate_indices:
        raise ValueError("Hard-negative mining candidate set is empty")
    device = item_embeddings.device
    candidate_tensor = torch.tensor(candidate_indices, device=device)
    candidate_embeddings = item_embeddings.index_select(0, candidate_tensor)
    mined = {}
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for start in tqdm(range(0, len(groups), batch_size), desc=desc):
            batch = groups[start:start + batch_size]
            query_embeddings = encode_queries([group.query for group in batch])
            scores = query_embeddings @ candidate_embeddings.T
            ranked_scores, ranked_offsets = scores.topk(min(top_k, scores.shape[1]), dim=1)
            for row, group in enumerate(batch):
                positive_indices = [
                    item_to_index[item_id]
                    for item_id in group.positive_item_ids
                    if item_id in item_to_index
                ]
                positives = item_embeddings.index_select(
                    0, torch.tensor(positive_indices, device=device)
                )
                best_positive = (query_embeddings[row] @ positives.T).max().item()
                margin = max(0.02, abs(best_positive) * (1.0 - positive_ratio))
                maximum_negative_score = best_positive - margin
                forbidden = set(group.positive_item_ids)
                values = []
                for score, offset in zip(ranked_scores[row].tolist(), ranked_offsets[row].tolist()):
                    item_id = item_ids[candidate_indices[offset]]
                    if item_id in forbidden or score > maximum_negative_score:
                        continue
                    values.append(item_id)
                    if len(values) >= negatives_per_query:
                        break
                mined[group.query] = values
    return mined
