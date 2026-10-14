import gzip
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from recoatlas.run_agent import select_rows
from recoatlas.run_store import RunStore


def record(cid, status="complete", score=0.5):
    return {"candidate_id": cid, "status": status, "feedback": {"sethit_at_k": score}}


def test_interruption_resume_preserves_failures_and_exports_dataset_order(tmp_path):
    output = tmp_path / "run.jsonl"
    rows = [{"candidate_id": cid} for cid in ("a", "b", "c")]
    manifest = {"revision": "pinned", "thinking": "on"}
    with pytest.raises(InterruptedError), RunStore(output, manifest, rows) as store:
        store.save(record("c", "error", 0))
        store.save(record("a"))
        assert store.progress()["mean_sethit_at_k"] is None
        raise InterruptedError
    assert not output.exists()
    with RunStore(output, manifest, rows, resume=True) as store:
        assert set(store.done) == {"a", "c"}
        store.save(record("b", score=1))
        store.export()
        assert store.progress() == {
            "evaluated_queries": 3,
            "completed": 2,
            "failed": 1,
            "pending": 0,
            "mean_sethit_at_k": 0.5,
        }
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["candidate_id"] for row in result] == ["a", "b", "c"]
    assert result[-1]["status"] == "error"


def test_resume_rejects_changed_configuration_unknown_checkpoints_and_corruption(tmp_path):
    output = tmp_path / "run.jsonl"
    rows = [{"candidate_id": "a"}]
    with RunStore(output, {"thinking": "on"}, rows) as store:
        store.save(record("a"))
        checkpoint = store.path("a")
    with pytest.raises(ValueError, match="Resume mismatch"), RunStore(output, {"thinking": "off"}, rows, resume=True):
        pass
    with (
        pytest.raises(ValueError, match="Unexpected checkpoint"),
        RunStore(output, {"thinking": "on"}, [{"candidate_id": "b"}], resume=True),
    ):
        pass
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(gzip.BadGzipFile), RunStore(output, {"thinking": "on"}, rows, resume=True):
        pass


def test_duplicate_writer_and_accidental_overwrite_are_rejected(tmp_path):
    output = tmp_path / "run.jsonl"
    rows = [{"candidate_id": "a"}]
    with RunStore(output, {}, rows):

        def other_writer():
            with pytest.raises(BlockingIOError), RunStore(output, {}, rows, resume=True):
                pass

        with ThreadPoolExecutor(1) as pool:
            pool.submit(other_writer).result()
    with pytest.raises(ValueError, match="already exists"), RunStore(output, {}, rows):
        pass


def test_shards_are_disjoint_complete_and_matched_between_thinking_modes():
    rows = [{"candidate_id": str(i), "positive_items": list(range(3 + i % 5))} for i in range(1960)]
    shards = [select_rows(rows, shard_index=i, num_shards=3) for i in range(3)]
    ids = [row["candidate_id"] for shard in shards for row in shard]
    assert len(ids) == len(set(ids)) == 1960
    assert set(ids) == {row["candidate_id"] for row in rows}
    assert select_rows(rows, smoke=True) == rows[:5]
    for index, shard in enumerate(shards):
        assert select_rows(rows, shard_index=index, num_shards=3) == shard


@pytest.mark.parametrize("index,count", [(0, 0), (-1, 2), (2, 2)])
def test_invalid_shards_are_rejected(index, count):
    with pytest.raises(ValueError):
        select_rows([], shard_index=index, num_shards=count)
