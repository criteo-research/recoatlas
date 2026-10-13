import hashlib
import json
from pathlib import Path

import pytest
from conftest import FakeEnv, ScriptedBackend

from recoatlas import Agent
from recoatlas.env import Env, Settings
from recoatlas.reports import prepare_report
from recoatlas.run_agent import evaluate_query

FIXTURE = Path(__file__).parent / "fixtures/paper_parity.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"], ids=lambda c: c["name"])
def test_replay_matches_original_paper_agent(case):
    env = FakeEnv()
    backend = ScriptedBackend(case["responses"])
    agent = Agent(env, backend, model="qwen-test")
    if case["error"]:
        with pytest.raises(ValueError):
            agent.run(case["query"], case["k"])
    else:
        report = agent.run(case["query"], case["k"])
        assert digest(report) == case["report_sha256"]
    assert [digest(call) for call in backend.calls] == case["messages_sha256"]
    assert env.calls == case["tool_calls"]
    assert agent.protocol_stats() == case["protocol"]


def test_frozen_system_prompts():
    expected = json.loads(FIXTURE.read_text())["system_prompt_sha256"]
    agent = Agent(FakeEnv(), None)
    for k, sha in expected.items():
        agent.top_k = int(k)
        assert hashlib.sha256(agent._system_prompt().encode()).hexdigest() == sha


def test_near_duplicate_check_is_advisory_and_cannot_discover_products():
    env = FakeEnv()
    agent = Agent(env, None)
    observed = ["i0", "i1", "i2"]
    for ids, threshold in [(["i99"], 0.95), (["i0", "i0"], 0.95), (observed, 0.2), ([], 0.95)]:
        result, pool = agent._execute_action(
            "get_substitute_products", {"item_ids": ids, "similarity_threshold": threshold}, observed
        )
        assert "error" in result and pool == observed
    assert not env.calls
    result, pool = agent._execute_action("get_substitute_products", {"item_ids": observed}, observed)
    assert result["removed"][0]["item_id"] in pool
    assert pool == observed


def test_parser_filters_and_backfills_in_discovery_order():
    catalog = {i: i for i in "abcde"}
    raw = {"results": [{"product_id": i, "reasoning": "fit"} for i in ["e", "a", "a", "unknown"]]}
    report = prepare_report("query", raw, ["a", "c", "b"], [], 3, catalog)
    assert [r["product_id"] for r in report["results"]] == ["a", "c", "b"]
    assert report["results"][1]["reasoning"] == "Backfilled from the candidate set."
    with pytest.raises(ValueError, match="INVALID_REPORT_SCHEMA"):
        prepare_report("query", {"results": [{"item_id": "a"}]}, ["a", "b", "c"], [], 3, catalog)


def test_environment_passes_requested_count_without_twenty_result_cap():
    env = Env.__new__(Env)
    env.settings = Settings()
    env._search_one = FakeEnv().search
    assert len(json.loads(env.search_products("q", 35))) == 35
    env.settings = Settings(max_tool_results=20)
    assert len(json.loads(env.search_products("q", 35))) == 20


def test_optional_cap_from_environment(monkeypatch):
    monkeypatch.setenv("RECOATLAS_MAX_TOOL_RESULTS", "35")
    assert Settings.from_environ().max_tool_results == 35
    monkeypatch.setenv("RECOATLAS_MAX_TOOL_RESULTS", "none")
    assert Settings.from_environ().max_tool_results is None


def test_failure_is_zero_and_references_never_enter_agent_messages():
    backend = ScriptedBackend([RuntimeError("test failure")])
    agent = Agent(FakeEnv(), backend)
    result = evaluate_query(
        agent,
        {
            "query": "Need an outfit",
            "candidate_id": "q1",
            "positive_items": ["SECRET_GOLD_A", "SECRET_GOLD_B", "SECRET_GOLD_C"],
        },
    )
    assert result["status"] == "error" and result["feedback"]["sethit_at_k"] == 0
    assert "SECRET_GOLD" not in json.dumps(backend.calls)


def test_agent_state_is_reset_between_queries():
    case = json.loads(FIXTURE.read_text())["cases"][0]
    backend = ScriptedBackend(case["responses"] * 2)
    agent = Agent(FakeEnv(), backend, model="qwen-test")
    first = agent.run(case["query"], case["k"])
    stats = agent.protocol_stats()
    second = agent.run(case["query"], case["k"])
    assert first == second and agent.protocol_stats() == stats
