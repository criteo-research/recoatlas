"""Validated environment profiles for the unified tool builder."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
REQUIRED_KEYS = {
    "profile",
    "domain",
    "category",
    "launcher",
    "retrieval_architecture",
    "complementarity_architecture",
    "release_repo",
    "release_repo_type",
    "release_prefix",
}


def available_profiles() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.json"))


def load_profile(name: str) -> dict:
    if name != Path(name).name or name.endswith(".json"):
        raise ValueError("Profile must be a bare configuration name")
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(
            f"Unknown profile {name!r}; choose one of {available_profiles()}"
        )
    value = json.loads(path.read_text())
    missing = REQUIRED_KEYS - set(value)
    if missing:
        raise ValueError(f"Profile {name} is missing fields: {sorted(missing)}")
    if value["profile"] != name:
        raise ValueError(f"Profile identity mismatch in {path}")
    launcher = (ROOT / value["launcher"]).resolve()
    if ROOT not in launcher.parents or not launcher.is_file():
        raise ValueError(f"Invalid launcher in {path}: {value['launcher']}")
    if value["release_repo_type"] != "dataset":
        raise ValueError(f"Release repository must be an HF dataset: {path}")
    if not value["release_prefix"].strip("/").endswith("/tools"):
        raise ValueError(f"Release prefix must end in /tools: {path}")
    return value
