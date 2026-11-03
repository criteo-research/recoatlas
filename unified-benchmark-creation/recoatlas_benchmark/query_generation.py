"""Frozen bundle-query generation evidence and deterministic checks."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence

from .io import clean_text
from .mining import Candidate


def stable_int(*parts: object) -> int:
    data = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def evidence_block(
    candidate: Candidate,
    product_texts: Mapping[str, str],
    subcategories: Mapping[str, str],
    attempt: int,
) -> str:
    items = list(candidate.item_ids)
    random.Random(stable_int(candidate.candidate_id, attempt, "evidence-order")).shuffle(items)
    blocks = [
        f"P{index}\nSubcategory: {subcategories.get(item_id, 'Unknown')}\n"
        f"Evidence: {clean_text(product_texts.get(item_id, ''))}"
        for index, item_id in enumerate(items, 1)
    ]
    return "\n\n".join(blocks)


def validate_query(query: object, item_ids: Sequence[str]) -> tuple[bool, str]:
    value = " ".join(str(query or "").split())
    if not 12 <= len(value.split()) <= 55:
        return False, "WORD_COUNT"
    if re.search(r"\bP[1-7]\b|\bASIN\b|candidate products|provided products", value, re.IGNORECASE):
        return False, "EVIDENCE_LABEL_LEAKAGE"
    if any(str(item_id).casefold() in value.casefold() for item_id in item_ids):
        return False, "PRODUCT_ID_LEAKAGE"
    return True, "OK"


def validate_consistency(payload: dict) -> dict:
    verdict = str(payload.get("verdict", "")).upper()
    confidence = str(payload.get("confidence", "")).upper()
    allowed_reasons = {
        "OK",
        "NOT_REALISTIC",
        "ITEM_NOT_SUPPORTED",
        "CATALOG_LEAKAGE",
        "UNSUPPORTED_CLAIM",
        "NO_SINGLE_GOAL",
    }
    reason_code = str(payload.get("reason_code", "")).upper()
    if verdict not in {"PASS", "FAIL"} or confidence not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError("Invalid consistency verdict or confidence")
    if reason_code not in allowed_reasons:
        raise ValueError("Invalid consistency reason code")
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason_code": reason_code,
        "reason": clean_text(payload.get("reason", ""), 500),
        "consistent": verdict == "PASS" and confidence != "LOW",
    }
