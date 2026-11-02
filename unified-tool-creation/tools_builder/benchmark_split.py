"""Protected, deterministic user splits for benchmark construction.

The full manifest contains hidden user IDs and must never be published with the
public tool artifacts.  Tool builders only consume the ``shared_train`` IDs and
record the manifest fingerprint in their public metadata.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FORMAT_VERSION = 1
SPLIT_NAMES = ("shared_train", "agent_dev", "agent_hidden")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def _validated_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} split ratios")
    values = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Split ratios must be finite and positive")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Split ratios must sum to one")
    return values


def assign_user_to_split(
    user_id: str,
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    salt: str,
) -> str:
    """Assign one user with a stable hash, independent of row or input order."""
    train_ratio, dev_ratio, _ = _validated_ratios(ratios)
    digest = hmac.new(
        salt.encode("utf-8"), str(user_id).encode("utf-8"), hashlib.sha256
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_ratio:
        return "shared_train"
    if value < train_ratio + dev_ratio:
        return "agent_dev"
    return "agent_hidden"


def _fingerprint_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_user_split_manifest(
    user_ids: Iterable[str],
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    salt: str | None = None,
    category: str | None = None,
    positive_rating_threshold: float | None = 4.0,
) -> dict:
    """Build the immutable protected manifest from positive-interaction users."""
    if not salt:
        raise ValueError("A protected, non-empty split key is required")
    resolved_ratios = _validated_ratios(ratios)
    unique_users = sorted({str(user_id) for user_id in user_ids if str(user_id)})
    if not unique_users:
        raise ValueError("Cannot build a benchmark split without users")

    splits = {name: [] for name in SPLIT_NAMES}
    for user_id in unique_users:
        splits[
            assign_user_to_split(user_id, ratios=resolved_ratios, salt=salt)
        ].append(user_id)

    assignment_payload = {
        "salt": salt,
        "ratios": dict(zip(SPLIT_NAMES, resolved_ratios)),
        "splits": splits,
    }
    manifest = {
        "format_version": FORMAT_VERSION,
        "split_name": "deterministic_user_hash_v1",
        "assignment_unit": "user_id",
        "category": category,
        "salt": salt,
        "ratios": dict(zip(SPLIT_NAMES, resolved_ratios)),
        "eligible_user_count": len(unique_users),
        "counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "population_fingerprint": _fingerprint_payload({"users": unique_users}),
        "splits": splits,
    }
    if positive_rating_threshold is not None:
        manifest["positive_rating_threshold"] = float(positive_rating_threshold)
    manifest["fingerprint"] = _fingerprint_payload(assignment_payload)
    validate_user_split_manifest(manifest)
    return manifest


def validate_user_split_manifest(manifest: Mapping[str, object]) -> None:
    """Reject malformed, overlapping, or mutated protected manifests."""
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported benchmark split format version")
    split_payload = manifest.get("splits")
    if not isinstance(split_payload, Mapping) or set(split_payload) != set(SPLIT_NAMES):
        raise ValueError(f"Manifest must contain exactly these splits: {SPLIT_NAMES}")

    ratios_payload = manifest.get("ratios")
    if not isinstance(ratios_payload, Mapping):
        raise ValueError("Manifest ratios are missing")
    ratios = _validated_ratios([ratios_payload[name] for name in SPLIT_NAMES])

    resolved: dict[str, list[str]] = {}
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        users = [str(user_id) for user_id in split_payload[name]]
        if users != sorted(set(users)):
            raise ValueError(f"{name} user IDs must be unique and sorted")
        overlap = seen.intersection(users)
        if overlap:
            raise ValueError(f"User leakage across splits: {len(overlap)} IDs")
        seen.update(users)
        resolved[name] = users

    if int(manifest.get("eligible_user_count", -1)) != len(seen):
        raise ValueError("Manifest eligible_user_count does not match its user IDs")
    expected_population = _fingerprint_payload({"users": sorted(seen)})
    if manifest.get("population_fingerprint") != expected_population:
        raise ValueError("Benchmark split population fingerprint mismatch")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(name, -1)) != len(resolved[name]) for name in SPLIT_NAMES
    ):
        raise ValueError("Manifest counts do not match its user IDs")

    expected = _fingerprint_payload(
        {
            "salt": manifest.get("salt"),
            "ratios": dict(zip(SPLIT_NAMES, ratios)),
            "splits": resolved,
        }
    )
    if manifest.get("fingerprint") != expected:
        raise ValueError("Benchmark split fingerprint mismatch")
    salt = manifest.get("salt")
    if not isinstance(salt, str) or not salt:
        raise ValueError("Benchmark split key is missing")
    for split_name, users in resolved.items():
        if any(
            assign_user_to_split(user_id, ratios=ratios, salt=salt) != split_name
            for user_id in users
        ):
            raise ValueError("Benchmark split contains a hash-assignment mismatch")


def load_user_split_manifest(path: str | Path) -> dict:
    with Path(path).open() as handle:
        manifest = json.load(handle)
    validate_user_split_manifest(manifest)
    return manifest


def split_users(manifest: Mapping[str, object], split_name: str) -> set[str]:
    validate_user_split_manifest(manifest)
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown benchmark split: {split_name}")
    return set(manifest["splits"][split_name])


def filter_interactions_to_split(interactions, manifest: Mapping[str, object], split_name: str):
    """Return a copy containing only users from one protected partition."""
    if "user_id" not in interactions.columns:
        raise ValueError("Interactions are missing user_id")
    allowed_users = split_users(manifest, split_name)
    return interactions[
        interactions["user_id"].astype(str).isin(allowed_users)
    ].copy()


def public_split_metadata(manifest: Mapping[str, object]) -> dict:
    """Return metadata safe for model artifacts; deliberately omit user IDs."""
    validate_user_split_manifest(manifest)
    keys = [
            "format_version",
            "split_name",
            "assignment_unit",
            "category",
            "ratios",
            "eligible_user_count",
            "counts",
            "population_fingerprint",
            "fingerprint",
    ]
    # Amazon uses an explicit rating threshold. Purchase-only sources such as
    # H&M have no source rating and instead record their positive-event policy.
    keys.extend(
        key
        for key in ("dataset", "positive_policy", "positive_rating_threshold")
        if key in manifest
    )
    return {key: manifest[key] for key in keys}
