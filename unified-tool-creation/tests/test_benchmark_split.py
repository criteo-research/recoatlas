import json

import pytest
import pandas as pd
import data.query_generator as query_generator

from tools_builder.benchmark_split import (
    SPLIT_NAMES,
    assign_user_to_split,
    build_user_split_manifest,
    filter_interactions_to_split,
    public_split_metadata,
    validate_user_split_manifest,
)


def test_assignment_is_order_independent_and_disjoint():
    users = [f"user-{index}" for index in range(1000)]
    first = build_user_split_manifest(users, category="test", salt="test-key")
    second = build_user_split_manifest(
        reversed(users), category="test", salt="test-key"
    )

    assert first["fingerprint"] == second["fingerprint"]
    sets = [set(first["splits"][name]) for name in SPLIT_NAMES]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(users)


def test_assignment_is_stable_when_population_grows():
    before = build_user_split_manifest(["a", "b", "c"], salt="test-key")
    after = build_user_split_manifest(["a", "b", "c", "d"], salt="test-key")
    for user_id in ("a", "b", "c"):
        assert assign_user_to_split(user_id, salt="test-key") == next(
            name for name in SPLIT_NAMES if user_id in before["splits"][name]
        )
        assert assign_user_to_split(user_id, salt="test-key") == next(
            name for name in SPLIT_NAMES if user_id in after["splits"][name]
        )


def test_public_metadata_never_contains_user_ids():
    manifest = build_user_split_manifest(
        ["secret-a", "secret-b"], salt="test-key"
    )
    public = public_split_metadata(manifest)
    assert "splits" not in public
    assert "salt" not in public
    assert "secret" not in json.dumps(public)


def test_secret_key_changes_assignments_and_fingerprint():
    users = [f"user-{index}" for index in range(100)]
    first = build_user_split_manifest(users, salt="protected-key-a")
    second = build_user_split_manifest(users, salt="protected-key-b")
    assert first["fingerprint"] != second["fingerprint"]
    assert first["splits"] != second["splits"]


def test_manifest_builder_refuses_a_public_default_key():
    with pytest.raises(ValueError, match="split key"):
        build_user_split_manifest(["user-a"])


def test_mutated_manifest_is_rejected():
    manifest = build_user_split_manifest(
        ["a", "b", "c", "d"], salt="test-key"
    )
    source = next(name for name in SPLIT_NAMES if manifest["splits"][name])
    manifest["splits"][source].append("injected")
    with pytest.raises(ValueError):
        validate_user_split_manifest(manifest)


def test_interaction_filter_cannot_admit_dev_or_hidden_users():
    users = [f"user-{index}" for index in range(100)]
    manifest = build_user_split_manifest(users, salt="protected-key")
    frame = pd.DataFrame({"user_id": users, "value": range(len(users))})
    filtered = filter_interactions_to_split(frame, manifest, "shared_train")
    assert set(filtered["user_id"]) == set(manifest["splits"]["shared_train"])
    assert not set(filtered["user_id"]) & set(manifest["splits"]["agent_dev"])
    assert not set(filtered["user_id"]) & set(manifest["splits"]["agent_hidden"])


def test_cache_only_query_generation_never_calls_llm(monkeypatch, tmp_path):
    generator = query_generator.SimpleQueryGenerator(
        item_to_subcategory={"item-a": "one", "item-b": "two"},
        cache_dir=str(tmp_path),
        model="test-model",
        cache_only=True,
    )
    generator.cache = {"item-a": ["basic instrument"]}

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("cache-only generation attempted an LLM call")

    monkeypatch.setattr(generator, "_generate_batch_async", fail_if_called)
    frame = pd.DataFrame(
        [
            {
                "user_id": "user-a",
                "parent_asin": "item-a",
                "product_title": "A",
                "product_description": "A description",
                "rating": 5.0,
            },
            {
                "user_id": "user-b",
                "parent_asin": "item-b",
                "product_title": "B",
                "product_description": "B description",
                "rating": 5.0,
            },
        ]
    )
    generated = generator.generate_all(frame)
    assert generated == {("user-a", "item-a"): "basic instrument"}


def test_purchase_only_manifest_does_not_claim_a_rating_threshold():
    manifest = build_user_split_manifest(
        ["customer-a", "customer-b"],
        salt="test-key",
        category="H_and_M",
        positive_rating_threshold=None,
    )
    assert "positive_rating_threshold" not in manifest
    assert "positive_rating_threshold" not in public_split_metadata(manifest)
