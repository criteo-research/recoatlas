"""Tests on the pinned MI/H&M environments. Run with: uv run pytest -m slow"""

import asyncio
import json

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", params=["hm-clothing", "amazon-musical-instruments"])
def env(request):
    import torch

    import recoatlas

    return recoatlas.load(request.param, device="cuda" if torch.cuda.is_available() else "cpu")


def test_requested_tool_count_is_preserved(env):
    assert len(json.loads(env.search_products("black jersey top", top_k=100))) == 100


def test_only_products_returned_by_tools_are_valid(env):
    ep = env.episode("agent_hidden", index=0)
    found = env.returned_ids(ep.search_products(ep.request, top_k=10))
    ep.get_substitute_products(found)  # the near-duplicate tool never makes products "seen"
    assert ep.seen_ids == found

    submitted = ep.submit([{"product_id": i, "reasoning": "r"} for i in ep._references])
    assert submitted["valid"]
    assert all(r["product_id"] in found for r in submitted["report"]["results"])
    assert ep.score()["reward_direct"] == ep.score()["sethit"]

    assert ep.submit([{"product_id": i, "reasoning": "r"} for i in found[: ep.k]])["valid"]
    assert ep.score()["reward_direct"] == ep.score()["sethit"]  # the last submission counts


def test_openenv_submit_report_ends_the_episode(env):
    from openenv.core.env_server.mcp_types import CallToolAction

    from recoatlas.openenv_env import RecoAtlasEnvironment

    oe = RecoAtlasEnvironment(env)
    meta = oe.reset(split="agent_hidden", index=0).metadata
    search = oe.step(CallToolAction(tool_name="search_products", arguments={"query": meta["request"], "top_k": 5}))
    ids = env.returned_ids(search.result.data)[: meta["k"]]

    results = [{"product_id": i, "reasoning": "r"} for i in ids]
    obs = oe.step(CallToolAction(tool_name="submit_report", arguments={"results": results}))
    assert obs.done and obs.metadata["valid"] == 1 and obs.reward == obs.metadata["sethit"]


def test_miles_service_contract(env):
    from recoatlas.miles import MilesToolService, miles_rows

    service = MilesToolService(env)
    meta = miles_rows(env, "agent_hidden")[0]["metadata"]
    search = asyncio.run(service.execute(meta["category"], "search_products", {"query": meta["query"], "top_k": 5}))
    assert search["ok"] and search["product_ids"] == env.returned_ids(search["content"])

    report = {"results": [{"product_id": i, "reasoning": "r"} for i in meta["ground_truth"]]}
    scored = asyncio.run(service.score("bundle", meta["ground_truth"], report, seen_product_ids=meta["ground_truth"]))
    assert scored["reward"] == 1.0


def test_paper_agent_with_real_tools_and_scripted_model(env):
    from recoatlas import Agent

    class Backend:
        def __init__(self):
            self.step = 0

        def complete(self, **kwargs):
            state = json.loads(kwargs["messages"][1]["content"])
            self.step += 1
            if self.step == 1:
                action, args = "search_products", {"query": "cotton summer shirt", "top_k": 35}
            elif self.step == 2:
                action, args = "search_products", {"query": "black high waist trousers", "top_k": 25}
            elif self.step == 3:
                action, args = (
                    "get_complementary_products",
                    {
                        "item_ids": [state["candidate_ledger"][0]["item_id"]],
                        "top_k": 35,
                    },
                )
            elif self.step == 4:
                action, args = (
                    "get_substitute_products",
                    {
                        "item_ids": [r["item_id"] for r in state["candidate_ledger"][:10]],
                        "similarity_threshold": 0.95,
                    },
                )
            else:
                return json.dumps(
                    {
                        "report_explanation": "Scripted integration test",
                        "results": [
                            {"product_id": r["item_id"], "reasoning": "Observed candidate"}
                            for r in state["candidate_ledger"][: state["target_count"]]
                        ],
                    }
                )
            return json.dumps(
                {"thought": "Test", "step_goal": "Gather evidence", "action": action, "action_input": args}
            )

    agent = Agent(env, Backend())
    report = agent.run("Assemble a summer outfit with distinct useful roles.", 5)
    assert len(report["results"]) == 5
    assert agent.protocol_stats()["minimum_met"]
    assert agent.protocol_stats()["diversity_check_performed"]
    assert len(report["trace"][0]["tool_result"]) == 35
