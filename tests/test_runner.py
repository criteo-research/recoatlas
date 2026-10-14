import json
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from conftest import FakeEnv, fake_client

from recoatlas import run_agent
from recoatlas.run_store import RunStore


@pytest.mark.parametrize("model", ["google/gemma-4-31B-it", "zai-org/GLM-4.7-Flash"])
def test_dry_run_does_not_load_tools_create_client_or_write_files(model, monkeypatch, capsys, tmp_path):
    env = Mock(side_effect=AssertionError("Tools must not load"))
    client = Mock(side_effect=AssertionError("Client must not initialize"))
    monkeypatch.setattr(run_agent, "Env", env)
    monkeypatch.setattr(run_agent, "get_client", client)
    output = tmp_path / "unused.jsonl"
    assert run_agent.main(["--dry-run", "--thinking", "on", "--model", model, "--output", str(output)]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["model"] == model
    assert config["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert not output.exists()
    env.assert_not_called()
    client.assert_not_called()


def test_bad_profile_fails_before_loading_or_api_calls(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_agent, "get_client", Mock(side_effect=AssertionError("Client must not initialize")))
    with pytest.raises(SystemExit) as error:
        run_agent.main(["--model", "unrecognized", "--output", str(tmp_path / "never.jsonl")])
    assert error.value.code == 2
    assert "--model-profile" in capsys.readouterr().err


@pytest.mark.parametrize("profile,model", [("gemma4", "my-local-model"), ("nemotron", "my-nemotron")])
def test_cli_runs_same_agent_and_records_effective_profile(profile, model, monkeypatch, tmp_path):
    case = json.loads((Path(__file__).parent / "fixtures/paper_parity.json").read_text())["cases"][0]
    client, calls = fake_client(case["responses"])
    env = FakeEnv()
    env.splits = {
        "agent_dev": [{"candidate_id": "synthetic", "query": case["query"], "positive_items": ["i0", "i1", "i2"]}]
    }
    monkeypatch.setattr(run_agent, "Env", lambda *a, **kw: env)
    monkeypatch.setattr(run_agent.assets, "load_split", lambda *a: env.splits["agent_dev"])
    monkeypatch.setattr(run_agent, "get_client", lambda **kw: client)
    output = tmp_path / "run.jsonl"
    status = run_agent.main(
        [
            "--model",
            model,
            "--model-profile",
            profile,
            "--thinking",
            "on",
            "--json-only",
            "--split",
            "agent_dev",
            "--output",
            str(output),
        ]
    )
    assert status == 0
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["serving"]["model_profile"]["name"] == profile
    assert summary["serving"]["extra_body"]["thinking_token_budget"] == 4096
    assert summary["expected_model_revision"] is None
    assert "model_revision" not in summary["paper_config"]
    assert env.calls == case["tool_calls"]
    assert all(c["extra_body"]["chat_template_kwargs"]["enable_thinking"] for c in calls)
    original = output.read_bytes()
    monkeypatch.setattr(run_agent, "Env", Mock(side_effect=AssertionError("Already complete: no tools needed")))
    monkeypatch.setattr(run_agent, "get_client", Mock(side_effect=AssertionError("Do not repeat model calls")))
    assert (
        run_agent.main(
            [
                "--model",
                model,
                "--model-profile",
                profile,
                "--thinking",
                "on",
                "--json-only",
                "--split",
                "agent_dev",
                "--output",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    assert output.read_bytes() == original


def test_concurrent_queries_have_separate_agent_state_and_one_shared_env(tmp_path, monkeypatch):
    env = FakeEnv()
    rows = [{"candidate_id": str(i), "query": str(i), "positive_items": ["i0", "i1", "i2"]} for i in range(8)]
    barrier = threading.Barrier(4)
    agents = []

    class IsolatedAgent:
        def __init__(self, shared_env, backend, model):
            assert shared_env is env
            self.env = shared_env
            self.backend = backend
            agents.append(self)

        def run(self, query, k):
            self.query = query
            barrier.wait(timeout=5)
            assert self.query == query
            return {"results": [{"product_id": f"i{i}", "reasoning": query} for i in range(k)]}

        def protocol_stats(self):
            return {"query": self.query}

    monkeypatch.setattr(run_agent, "Agent", IsolatedAgent)
    with RunStore(tmp_path / "all.jsonl", {}, rows) as store:
        run_agent.run_queries(env, object(), "test", rows, store, concurrency=4)
        store.export()
        assert store.progress()["completed"] == 8
    assert len({id(a) for a in agents}) == 8
    for row in (json.loads(s) for s in (tmp_path / "all.jsonl").read_text().splitlines()):
        assert row["protocol"]["query"] == row["candidate_id"]
        assert {r["reasoning"] for r in row["report"]["results"]} == {row["candidate_id"]}
