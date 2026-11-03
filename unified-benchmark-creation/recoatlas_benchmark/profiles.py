"""Frozen domain profiles."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_profile(name_or_path: str | Path) -> dict:
    candidate = Path(name_or_path)
    if not candidate.is_file():
        candidate = ROOT / "configs" / f"{name_or_path}.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"Unknown benchmark profile: {name_or_path}")
    profile = json.loads(candidate.read_text(encoding="utf-8"))
    required = {"profile", "domain", "category", "release_prefix", "partitions"}
    missing = required - set(profile)
    if missing:
        raise ValueError(f"Profile is missing: {sorted(missing)}")
    return profile
