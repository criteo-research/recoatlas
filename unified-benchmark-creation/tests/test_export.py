import json

from recoatlas_benchmark.io import read_jsonl, write_jsonl
from recoatlas_benchmark.pipeline import export_stage
from recoatlas_benchmark.publish import validate_release


def test_export_is_minimal_and_deduplicated(tmp_path):
    profile = {
        "profile": "fixture",
        "domain": "amazon",
        "category": "fixture",
        "release_repo": "owner/repo",
        "release_prefix": "fixture/data",
        "partitions": ["agent_hidden", "shared_train"],
        "deduplicate_sets_in_order": ["agent_hidden", "shared_train"],
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    rows = [
        {
            "candidate_id": "hidden-a",
            "partition": "agent_hidden",
            "item_ids": ["1", "2", "3"],
            "status": "accepted",
            "query": "A sufficiently long natural shopping request for one coherent hidden project today",
        },
        {
            "candidate_id": "train-a",
            "partition": "shared_train",
            "item_ids": ["4", "5", "6"],
            "status": "accepted",
            "query": "A sufficiently long natural shopping request for a different coherent training project today",
        },
    ]
    write_jsonl(tmp_path / "query_decisions.jsonl", rows)
    summary = export_stage(profile_name=str(profile_path), run_dir=tmp_path)
    assert summary["counts"] == {"agent_hidden": 1, "shared_train": 1}
    hidden = read_jsonl(tmp_path / "release/agent_hidden.jsonl")[0]
    assert list(hidden) == ["candidate_id", "partition", "query", "positive_items"]
    validate_release(profile, tmp_path / "release")
