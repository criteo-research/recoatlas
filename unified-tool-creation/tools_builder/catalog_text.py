"""Pure catalog-text normalization helpers."""

from __future__ import annotations

import json
import re


def clean_description(values, max_length: int = 500, min_length: int = 15) -> str:
    """Normalize a sequence of description fragments and enforce its length."""
    if not len(values):
        return ""
    text = " ".join(" ".join(str(value) for value in values if value).split())
    if len(text) < min_length:
        return ""
    return text[:max_length] + "..." if len(text) > max_length else text


def first_available_year(details: str | None) -> int | None:
    """Extract a four-digit year from Amazon's Date First Available field."""
    if not details:
        return None
    try:
        payload = json.loads(details)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("Date First Available")
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return int(match.group()) if match else None
