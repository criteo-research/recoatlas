"""Protected deterministic 70/15/15 user partitions."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

FORMAT_VERSION = 1
SPLIT_NAMES = ("shared_train", "agent_dev", "agent_hidden")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def _ratios(values: Sequence[float]) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("Expected three finite, positive split ratios")
    if not math.isclose(sum(result), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Split ratios must sum to one")
    return result


def assign_user(user_id: str, *, salt: str, ratios: Sequence[float] = DEFAULT_RATIOS) -> str:
    train, dev, _ = _ratios(ratios)
    digest = hmac.new(salt.encode(), str(user_id).encode(), hashlib.sha256).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train:
        return "shared_train"
    if value < train + dev:
        return "agent_dev"
    return "agent_hidden"


def _fingerprint(payload: Mapping[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def build_manifest(
    user_ids: Iterable[str], *, salt: str, category: str, ratios: Sequence[float] = DEFAULT_RATIOS
) -> dict:
    if not salt:
        raise ValueError("A non-empty private split key is required")
    resolved_ratios = _ratios(ratios)
    users = sorted({str(value) for value in user_ids if str(value)})
    if not users:
        raise ValueError("No eligible users")
    splits = {name: [] for name in SPLIT_NAMES}
    for user_id in users:
        splits[assign_user(user_id, salt=salt, ratios=resolved_ratios)].append(user_id)
    manifest = {
        "format_version": FORMAT_VERSION,
        "split_name": "deterministic_user_hash_v1",
        "assignment_unit": "user_id",
        "category": category,
        "salt": salt,
        "ratios": dict(zip(SPLIT_NAMES, resolved_ratios)),
        "eligible_user_count": len(users),
        "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "population_fingerprint": _fingerprint({"users": users}),
        "splits": splits,
    }
    manifest["fingerprint"] = _fingerprint(
        {"salt": salt, "ratios": manifest["ratios"], "splits": splits}
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported split-manifest version")
    splits = manifest.get("splits")
    ratios = manifest.get("ratios")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("The manifest must contain exactly the three RecoAtlas partitions")
    if not isinstance(ratios, Mapping):
        raise TypeError("Missing split ratios")
    resolved_ratios = _ratios([ratios[name] for name in SPLIT_NAMES])
    seen: set[str] = set()
    normalized: dict[str, list[str]] = {}
    for name in SPLIT_NAMES:
        users = [str(value) for value in splits[name]]
        if users != sorted(set(users)) or seen.intersection(users):
            raise ValueError("Split users must be sorted, unique, and disjoint")
        normalized[name] = users
        seen.update(users)
    if int(manifest.get("eligible_user_count", -1)) != len(seen):
        raise ValueError("Eligible-user count mismatch")
    if manifest.get("population_fingerprint") != _fingerprint({"users": sorted(seen)}):
        raise ValueError("Population fingerprint mismatch")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(name, -1)) != len(normalized[name]) for name in SPLIT_NAMES
    ):
        raise ValueError("Split-count mismatch")
    expected = _fingerprint(
        {
            "salt": manifest.get("salt"),
            "ratios": dict(zip(SPLIT_NAMES, resolved_ratios)),
            "splits": normalized,
        }
    )
    if manifest.get("fingerprint") != expected:
        raise ValueError("Split fingerprint mismatch")
    salt = manifest.get("salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError("Missing split key")
    for name, users in normalized.items():
        if any(assign_user(user, salt=salt, ratios=resolved_ratios) != name for user in users):
            raise ValueError("Hash assignment mismatch")


def load_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload
