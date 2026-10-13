"""Scores for one submitted report."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from .reports import prepare_report

REPORT_FORMAT = '[{"product_id": "<ID returned by a tool>", "reasoning": "<why it fits>"}, ...]'


def sethit(predicted_ids: Sequence, references: Collection[str], catalog: Collection[str]) -> float:
    """Share of reference items among the first K submitted IDs (K = number of references).

    Same rule as recoatlas-v1 evaluation/metrics.py: positions are cut to K before
    duplicates or unknown IDs are dropped, so later positions cannot earn credit.
    """
    gold = set(references)
    if not gold:
        raise ValueError("Empty reference set")
    found = {x for x in list(predicted_ids)[: len(gold)] if isinstance(x, str) and x in catalog}
    return len(found & gold) / len(gold)


def validation_errors(results, k: int, catalog: Collection[str], seen: Collection[str]) -> list[str]:
    """Label-free checks on a report's results; an empty list means valid. Messages are meant for the agent."""
    if not isinstance(results, list):
        return [f"results must be a list: {REPORT_FORMAT}"]
    errors = []
    if len(results) != k:
        errors.append(f"results must contain exactly {k} products, got {len(results)}")
    malformed = [i for i, item in enumerate(results) if not _is_result(item)]
    if malformed:
        first = results[malformed[0]]
        got = sorted(first) if isinstance(first, dict) else type(first).__name__
        errors.append(
            f"results{malformed} are malformed (results[{malformed[0]}] has {got}); "
            f'each result needs string "product_id" and "reasoning": {REPORT_FORMAT}'
        )
    ids = [item["product_id"] for item in results if _is_result(item)]
    if len(set(ids)) != len(ids):
        errors.append("results must not repeat a product_id")
    if unknown := [i for i in ids if i not in catalog]:
        errors.append(f"unknown product_id values: {unknown}")
    if unseen := [i for i in ids if i in catalog and i not in seen]:
        errors.append(f"product_id values never returned by search_products or get_complementary_products: {unseen}")
    return errors


def score_report(results, references: Collection[str], catalog: Mapping[str, str], seen: Sequence[str]) -> dict:
    """Apply the paper parser, then return sethit, valid and reward_direct.

    `seen` is discovery order, not a set: it determines backfilling. None means
    no submission. Malformed schemas fail before backfilling and score zero.
    """
    if not isinstance(results, list):
        return {"sethit": 0.0, "valid": 0, "reward_direct": -1.0}
    if isinstance(seen, set | frozenset):
        raise ValueError("seen must preserve discovery order for paper backfilling")
    try:
        report = prepare_report("", {"results": results}, list(seen), [], len(set(references)), catalog)
    except ValueError:
        return {"sethit": 0.0, "valid": 0, "reward_direct": -1.0}
    results = report["results"]
    hit = sethit([item.get("product_id") if isinstance(item, dict) else None for item in results], references, catalog)
    valid = not validation_errors(results, len(set(references)), catalog, seen)
    return {"sethit": hit, "valid": int(valid), "reward_direct": hit if valid else -1.0}


def _is_result(item) -> bool:
    return isinstance(item, dict) and isinstance(item.get("product_id"), str) and isinstance(item.get("reasoning"), str)
