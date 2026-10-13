import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from conftest import FakeEnv, fake_client

from recoatlas import run_agent


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
