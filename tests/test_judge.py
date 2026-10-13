import json

import pytest
from conftest import fake_client

from recoatlas import judge


def test_judge_inputs_are_blind_deterministic_and_truncated():
    catalog = {"secret-id-a": "a" * 500, "secret-id-b": "B", "secret-id-c": "C"}
    ids = list(catalog)
    messages, labels = judge.make_request("query", 3, "cid", ids, catalog)
    assert (messages, labels) == judge.make_request("query", 3, "cid", ids, catalog)
    data = json.loads(messages[1]["content"].split("INPUT DATA:\n")[1])
    assert set(data) == {"request", "requested_item_count", "catalog_evidence"}
    assert set(data["catalog_evidence"]) == {"P1", "P2", "P3"}
    assert max(map(len, data["catalog_evidence"].values())) == 300
    assert "secret-id-" not in json.dumps(messages)


@pytest.mark.parametrize(
    "payload",
    [
        {"score": True, "reason": "useful", "evidence_items": []},
        {"score": 4.0, "reason": "useful", "evidence_items": []},
        {"score": 5, "reason": "useful", "evidence_items": []},
        {"score": 3, "reason": "", "evidence_items": []},
        {"score": 3, "reason": "useful", "evidence_items": ["not-supplied"]},
        {"score": 3, "reason": "useful", "evidence_items": [], "extra": 1},
    ],
)
def test_invalid_judgments_are_not_scores(payload):
    with pytest.raises(ValueError):
        judge.validate(payload, ["P1"])


def test_judge_retries_and_normalizes_first_valid_response():
    client, calls = fake_client([{}, {"score": 3, "reason": "Useful choices", "evidence_items": ["P1"]}])
    attempts = judge.judge_request(client, "my-deployment", [], ["P1"])
    assert [r["status"] for r in attempts] == ["error", "complete"]
    assert attempts[-1]["quality"] == 0.75
    assert calls[0]["model"] == "my-deployment" and calls[0]["max_completion_tokens"] == 4096


def test_failed_judge_is_unresolved_not_zero():
    client, _ = fake_client([RuntimeError("secret request") for _ in range(3)])
    attempts = judge.judge_request(client, "model", [], [])
    assert len(attempts) == 3 and all("quality" not in r for r in attempts)
    assert "secret request" not in json.dumps(attempts)


def test_dry_run_does_not_construct_api_client(tmp_path, monkeypatch, capsys):
    task = {"candidate_id": "q1", "query": "query", "positive_items": ["a", "b", "c"]}
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({i: i for i in "abc"}))
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"candidate_id": "q1", "status": "complete", "predicted_item_ids": list("abc")}))
    monkeypatch.setattr(judge.assets, "load_split", lambda *args: [task])
    monkeypatch.setattr(judge.assets, "fetch", lambda *args: catalog)
    monkeypatch.setattr(judge, "get_client", lambda **kwargs: pytest.fail("Dry run created an API client"))
    assert judge.main(["--traces", str(traces), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["api_calls"] == 0


def test_all_judge_inputs_are_validated_before_paid_calls(tmp_path, monkeypatch):
    tasks = [{"candidate_id": f"q{i}", "query": "query", "positive_items": list("abc")} for i in range(2)]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({i: i for i in "abc"}))
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        "\n".join(
            json.dumps({"candidate_id": f"q{i}", "status": "complete", "predicted_item_ids": list(ids)})
            for i, ids in enumerate(["abc", "aa"])
        )
    )
    monkeypatch.setattr(judge.assets, "load_split", lambda *args: tasks)
    monkeypatch.setattr(judge.assets, "fetch", lambda *args: catalog)
    monkeypatch.setattr(judge, "get_client", lambda **kwargs: pytest.fail("Invalid input created an API client"))
    with pytest.raises(ValueError, match="complete unique report"):
        judge.main(["--traces", str(traces), "--model", "judge", "--output", str(tmp_path / "out.jsonl")])
