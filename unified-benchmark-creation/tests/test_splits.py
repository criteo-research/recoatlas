import json

import pytest

from recoatlas_benchmark.splits import build_manifest, load_manifest, validate_manifest


def test_manifest_is_deterministic_and_disjoint(tmp_path):
    users = [f"u-{index}" for index in range(200)]
    left = build_manifest(reversed(users), salt="private-test-key", category="test")
    right = build_manifest(users, salt="private-test-key", category="test")
    assert left == right
    groups = [set(left["splits"][name]) for name in left["splits"]]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    path = tmp_path / "split.json"
    path.write_text(json.dumps(left))
    assert load_manifest(path)["fingerprint"] == left["fingerprint"]


def test_manifest_mutation_is_rejected():
    manifest = build_manifest(["a", "b", "c"], salt="key", category="test")
    manifest["eligible_user_count"] += 1
    with pytest.raises(ValueError, match="count"):
        validate_manifest(manifest)
