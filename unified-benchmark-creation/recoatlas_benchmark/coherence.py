"""Coherence-judge presentations and response validation."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping

from .io import clean_text
from .mining import Candidate
from .prompts import JUDGE_INSTRUCTIONS

ALLOWED_REASONS = {
    "NO_COMMON_GOAL",
    "MOSTLY_SUBSTITUTES",
    "REDUNDANT_DUPLICATES",
    "UNSUPPORTED_COMPATIBILITY",
    "IMPLAUSIBLE_COMBINATION",
    "INSUFFICIENT_EVIDENCE",
}
ALLOWED_FITS = {"FIT", "UNCERTAIN", "CONFLICT"}


def presentation(
    candidate: Candidate,
    vote_index: int,
    product_texts: Mapping[str, str],
    subcategories: Mapping[str, str],
) -> tuple[list[str], str]:
    item_ids = list(candidate.item_ids)
    digest = hashlib.sha256(f"{candidate.candidate_id}\0vote-{vote_index}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(item_ids)
    blocks = [
        f"P{index}\nSubcategory: {subcategories.get(item_id, 'Unknown')}\n"
        f"Catalog evidence: {clean_text(product_texts.get(item_id, ''), 850)}"
        for index, item_id in enumerate(item_ids, 1)
    ]
    return item_ids, (
        JUDGE_INSTRUCTIONS
        + "\n\nCANDIDATE PRODUCTS (order is randomized):\n\n"
        + "\n\n".join(blocks)
    )


def validate_judgment(payload: dict, item_count: int) -> dict:
    verdict = str(payload.get("verdict", "")).upper()
    confidence = str(payload.get("confidence", "")).upper()
    if verdict not in {"PASS", "FAIL"} or confidence not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError("Invalid coherence verdict or confidence")
    roles = payload.get("item_roles")
    if not isinstance(roles, list):
        raise TypeError("item_roles must be a list")
    normalized_roles = []
    labels = []
    for role in roles:
        if not isinstance(role, dict):
            raise TypeError("Invalid item role")
        label = str(role.get("label", "")).upper()
        fit = str(role.get("fit", "")).upper()
        if fit not in ALLOWED_FITS:
            raise ValueError("Invalid item fit")
        labels.append(label)
        normalized_roles.append(
            {"label": label, "role": clean_text(role.get("role", ""), 240), "fit": fit}
        )
    expected = [f"P{index}" for index in range(1, item_count + 1)]
    if sorted(labels) != expected or len(set(labels)) != item_count:
        raise ValueError("item_roles must cover every product exactly once")
    reasons = payload.get("failure_reasons", [])
    if not isinstance(reasons, list):
        raise TypeError("failure_reasons must be a list")
    reasons = sorted({str(reason).upper() for reason in reasons})
    if any(reason not in ALLOWED_REASONS for reason in reasons):
        raise ValueError("Unknown coherence failure reason")
    if verdict == "PASS" and reasons:
        raise ValueError("PASS contains failure reasons")
    if verdict == "FAIL" and not reasons:
        raise ValueError("FAIL has no failure reason")
    result = {
        "verdict": verdict,
        "confidence": confidence,
        "common_goal": clean_text(payload.get("common_goal", ""), 400),
        "item_roles": normalized_roles,
        "failure_reasons": reasons,
        "rationale": clean_text(payload.get("rationale", ""), 600),
    }
    result["vote_pass"] = (
        verdict == "PASS"
        and confidence != "LOW"
        and all(role["fit"] != "CONFLICT" for role in normalized_roles)
    )
    return result
