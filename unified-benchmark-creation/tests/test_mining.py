import pandas as pd

from recoatlas_benchmark.mining import (
    deduplicate_reference_sets,
    mine_amazon_candidates,
    mine_hm_candidates,
)
from recoatlas_benchmark.splits import build_manifest


def test_amazon_keeps_latest_repeat_and_first_densest_window():
    manifest = build_manifest(["u"], salt="key", category="Musical_Instruments")
    partition = next(name for name, users in manifest["splits"].items() if users)
    frame = pd.DataFrame(
        [
            ("u", "a", 5, "2024-01-01"),
            ("u", "b", 5, "2024-01-02"),
            ("u", "c", 5, "2024-01-03"),
            ("u", "a", 5, "2024-02-01"),  # latest a replaces the January event
            ("u", "d", 5, "2024-02-02"),
            ("u", "e", 5, "2024-02-03"),
        ],
        columns=["user_id", "parent_asin", "rating", "timestamp"],
    )
    rows = mine_amazon_candidates(
        frame,
        manifest=manifest,
        subcategories={"a": "one", "b": "two", "c": "three", "d": "four", "e": "five"},
        partitions=[partition],
    )
    assert len(rows) == 1
    assert rows[0].item_ids == ("a", "d", "e")


def test_hm_requires_distinct_product_codes_types_and_one_group():
    manifest = build_manifest(["u"], salt="key", category="H_and_M")
    partition = next(name for name, users in manifest["splits"].items() if users)
    interactions = pd.DataFrame(
        {
            "user_id": ["u", "u", "u"],
            "item_id": ["0000000001", "0000000002", "0000000003"],
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
        }
    )
    catalog = pd.DataFrame(
        {
            "item_id": ["0000000001", "0000000002", "0000000003"],
            "product_code": [1, 2, 3],
            "product_type": ["Top", "Shorts", "Belt"],
            "merchandising_group": ["Ladieswear/Divided"] * 3,
        }
    )
    rows = mine_hm_candidates(
        interactions, manifest=manifest, catalog=catalog, partitions=[partition]
    )
    assert [row.item_ids for row in rows] == [("0000000001", "0000000002", "0000000003")]
    catalog.loc[2, "product_code"] = 2
    assert not mine_hm_candidates(
        interactions, manifest=manifest, catalog=catalog, partitions=[partition]
    )


def test_reference_set_dedup_uses_declared_priority():
    from recoatlas_benchmark.mining import Candidate

    rows = [
        Candidate("train-a", "shared_train", ("1", "2", "3")),
        Candidate("hidden-a", "agent_hidden", ("3", "1", "2")),
    ]
    kept, removed = deduplicate_reference_sets(rows, ["agent_hidden", "agent_dev", "shared_train"])
    assert [row.candidate_id for row in kept] == ["hidden-a"]
    assert removed == [{"candidate_id": "train-a", "reason": "identical_set"}]
