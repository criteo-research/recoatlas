"""Shared noise-injection helpers for recommender tools.

Corruption is applied AFTER a tool's clean top-k is fully computed. The clean
code path is untouched when ``noise == 0``.

Determinism: per-call seed is derived from a hash of the tool name and call
inputs, so the same call returns the same corruption across runs and machines.
Each call builds its own ``random.Random`` instance — no global RNG state — so
parallel worker threads do not interfere.

Corrupted slots preserve the replaced slot's ``score`` so the LLM sees the same
plausible descending score distribution; only ``item_id``, ``title`` and
``description`` change. This forces the LLM to rely on content reasoning rather
than score anomalies to detect injected noise.

Modes:
  - ``"subcategory"``: random item from the replaced item's subcategory.
  - ``"random"``: uniform from the full catalog.
  - ``"worst_subcategory"``: adversarial — rank the replaced item's subcategory
    by a caller-provided score (query-relevance for retrieval, mean-anchor
    complementarity for the complementarity tool), then sample uniformly from
    the bottom ``BOTTOM_FRAC`` of that ranked list (clamped to
    ``[BOTTOM_MIN, BOTTOM_MAX]``). Bottom-pool sampling (rather than absolute
    worst) prevents degenerate picks and catalog outliers while keeping the
    adversarial signal strong.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable, Iterable


# Defaults for the ``worst_subcategory`` bottom pool. Picked from the observed
# Musical_Instruments subcategory-size distribution (median 38, p90 299, max
# 1789, with ~20% of subcategories <=5 items).
#   - 25% gives a genuine "bottom" even at median sizes.
#   - min 3 avoids deterministic picks in small subcategories.
#   - max 50 keeps the bottom meaningfully bottom for large subcategories.
BOTTOM_FRAC = 0.25
BOTTOM_MIN = 3
BOTTOM_MAX = 50

_ALLOWED_MODES = ("subcategory", "random", "worst_subcategory")


def seeded_rng(tool_name: str, seed_inputs: Any) -> random.Random:
    """Build a ``random.Random`` seeded deterministically from call inputs."""
    blob = json.dumps([tool_name, seed_inputs], sort_keys=True, default=str).encode()
    seed = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")
    return random.Random(seed)


def _extract_title(text: str) -> str:
    if text.startswith("Title: "):
        sep = text.find(" | Description:")
        if sep != -1:
            return text[7:sep]
    return ""


def _bottom_pool_size(n_pool: int) -> int:
    """Size of the bottom pool to sample from, clamped to [MIN, MAX, n_pool]."""
    size = int(BOTTOM_FRAC * n_pool)
    size = max(BOTTOM_MIN, size)
    return min(BOTTOM_MAX, size, n_pool)


def _pick_worst_subcategory(
    rng: random.Random,
    *,
    replaced_item_id: str,
    exclude: set[str],
    item_to_subcategory: dict[str, str],
    subcategory_to_items: dict[str, list[str]],
    product_reps: dict[str, str],
    candidate_scores: dict[str, float],
) -> str | None:
    """Pick from the worst ``BOTTOM_FRAC`` of the replaced item's subcategory.

    Returns ``None`` if the subcategory is unknown or the filtered pool is
    empty — the caller then falls back to plain subcategory-random.
    """
    sc = item_to_subcategory.get(replaced_item_id)
    if not sc:
        return None
    pool = [
        i for i in subcategory_to_items.get(sc, [])
        if i not in exclude and i in product_reps
    ]
    if not pool:
        return None
    # Items missing from candidate_scores are treated as +inf (best), so they
    # never land in the bottom pool; this is safe for items outside the
    # scored domain.
    pool.sort(key=lambda i: candidate_scores.get(i, float("inf")))
    size = _bottom_pool_size(len(pool))
    return rng.choice(pool[:size])


def _pick_replacement(
    rng: random.Random,
    *,
    mode: str,
    replaced_item_id: str,
    exclude: set[str],
    item_to_subcategory: dict[str, str],
    subcategory_to_items: dict[str, list[str]],
    all_items: list[str],
    product_reps: dict[str, str],
    candidate_scores: dict[str, float] | None = None,
) -> str | None:
    """Pick one replacement item_id given the exclusion set, or None on failure."""
    if mode == "worst_subcategory" and candidate_scores is not None:
        pick = _pick_worst_subcategory(
            rng,
            replaced_item_id=replaced_item_id,
            exclude=exclude,
            item_to_subcategory=item_to_subcategory,
            subcategory_to_items=subcategory_to_items,
            product_reps=product_reps,
            candidate_scores=candidate_scores,
        )
        if pick is not None:
            return pick
        # Fall through to plain subcategory-random if the worst-pool lookup
        # failed (unknown subcategory or empty pool after exclusions).

    if mode in ("subcategory", "worst_subcategory"):
        sc = item_to_subcategory.get(replaced_item_id)
        pool = subcategory_to_items.get(sc, []) if sc else []
        pool = [i for i in pool if i not in exclude and i in product_reps]
        if pool:
            return rng.choice(pool)

    # Fallback for "random" mode, or when the subcategory pool is empty.
    for _ in range(16):
        cand = rng.choice(all_items)
        if cand not in exclude and cand in product_reps:
            return cand
    return None


def apply_slot_noise(
    results: list[dict],
    *,
    tool_name: str,
    noise: float,
    mode: str,
    seed_inputs: Any,
    extra_exclude: Iterable[str] = (),
    item_to_subcategory: dict[str, str],
    subcategory_to_items: dict[str, list[str]],
    all_items: list[str],
    product_reps: dict[str, str],
    candidate_scores: dict[str, float] | None = None,
) -> None:
    """Replace a fraction of ``results`` slots with plausible-but-wrong items.

    Mutates ``results`` in place. Each corrupted slot keeps its original
    ``score``; only ``item_id``, ``title`` and ``description`` are swapped.

    The number of corrupted slots is ``round(len(results) * noise)``. Slots are
    chosen uniformly at random (without replacement), so noise is distributed
    across positions rather than concentrated at the bottom.

    ``extra_exclude`` is merged with the current result item_ids to form the
    exclusion set (e.g. anchor item_ids for complementarity).

    ``candidate_scores`` is only used when ``mode="worst_subcategory"``: it maps
    ``item_id -> score`` over (a superset of) the catalog, where lower scores
    are "worse". The bottom ``BOTTOM_FRAC`` of each replaced item's subcategory
    (by this score, clamped to ``[BOTTOM_MIN, BOTTOM_MAX]``) becomes the
    sampling pool.

    Falls back to ``mode="random"`` transparently if subcategory metadata is
    missing or the subcategory pool is empty for a given slot.
    """
    if noise <= 0.0 or not results:
        return

    k = len(results)
    n_noise = round(k * max(0.0, min(1.0, noise)))
    if n_noise <= 0:
        return

    rng = seeded_rng(tool_name, seed_inputs)
    slot_indices = rng.sample(range(k), n_noise)

    current_ids = {r.get("item_id") for r in results if r.get("item_id")}
    exclude: set[str] = set(current_ids) | {i for i in extra_exclude if i}

    effective_mode = mode if (mode == "random" or item_to_subcategory) else "random"

    for slot in slot_indices:
        replaced = results[slot]
        replaced_id = replaced.get("item_id", "")
        local_exclude = exclude - {replaced_id}
        new_id = _pick_replacement(
            rng,
            mode=effective_mode,
            replaced_item_id=replaced_id,
            exclude=local_exclude,
            item_to_subcategory=item_to_subcategory,
            subcategory_to_items=subcategory_to_items,
            all_items=all_items,
            product_reps=product_reps,
            candidate_scores=candidate_scores,
        )
        if new_id is None:
            continue
        text = product_reps.get(new_id, "")
        new_entry = dict(replaced)
        new_entry["item_id"] = new_id
        new_entry["title"] = _extract_title(text)
        new_entry["description"] = text[:300]
        results[slot] = new_entry
        exclude.discard(replaced_id)
        exclude.add(new_id)


def load_subcategory_maps(
    subcategories_path: str | None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Load ``item_to_subcategory`` and ``subcategory_to_items`` from JSON.

    Returns ``({}, {})`` when ``subcategories_path`` is falsy so callers that
    never use subcategory-mode noise do not need to special-case it.
    """
    if not subcategories_path:
        return {}, {}
    with open(subcategories_path) as f:
        data = json.load(f)
    return data.get("item_to_subcategory", {}), data.get("subcategory_to_items", {})


def validate_noise_args(noise: float, noise_mode: str) -> tuple[float, str]:
    """Clamp ``noise`` to [0, 1] and validate ``noise_mode``."""
    n = max(0.0, min(1.0, float(noise)))
    if noise_mode not in _ALLOWED_MODES:
        raise ValueError(
            f"noise_mode must be one of {_ALLOWED_MODES}, got {noise_mode!r}"
        )
    return n, noise_mode
