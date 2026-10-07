"""Tests on the real H&M environment (downloads ~2.8 GB once). Run with: uv run pytest -m slow"""

import asyncio
import json

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def env():
    import torch

    import recoatlas

    return recoatlas.load("hm-clothing", device="cuda" if torch.cuda.is_available() else "cpu")


def test_tool_results_are_capped(env):
    assert len(json.loads(env.search_products("black jersey top", top_k=100))) == 20


def test_only_products_returned_by_tools_are_valid(env):
    ep = env.episode("agent_hidden", index=0)
    found = env.returned_ids(ep.search_products(ep.request, top_k=10))
    ep.get_substitute_products(found)  # the near-duplicate tool never makes products "seen"
    assert ep.seen_ids == found

    ep.submit([{"product_id": i, "reasoning": "r"} for i in ep._references])
    assert ep.score() == {"sethit": 1.0, "valid": 0, "reward_direct": -1.0}  # right products, never seen

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
