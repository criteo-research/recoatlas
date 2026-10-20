import asyncio
import json

import pytest
from conftest import FakeEnv

from recoatlas.baselines import run_baseline
from recoatlas.env import Episode
from recoatlas.miles import MilesToolService
from recoatlas.trl import RecoAtlasEnvironment


def episode():
    return Episode(
        FakeEnv(),
        {"partition": "agent_dev", "candidate_id": "q1", "query": "query", "positive_items": ["i0", "i1", "i2"]},
    )


def test_episode_uses_paper_backfill_before_reward():
    ep = episode()
    ep.search_products("q", top_k=10)
    feedback = ep.submit([{"product_id": "i0", "reasoning": "fits"}])
    assert feedback["valid"]
    assert [r["product_id"] for r in feedback["report"]["results"]] == ["i0", "i1", "i2"]
    assert ep.score() == {"sethit": 1.0, "valid": 1, "reward_direct": 1.0}
    assert "sethit" not in feedback and "reward_direct" not in feedback


def test_near_duplicate_call_never_adds_seen_ids():
    ep = episode()
    ep.search_products("q", top_k=3)
    seen = list(ep.seen_ids)
    ep.get_substitute_products(["i50", "i51"])
    assert ep.seen_ids == seen


def test_trl_reward_uses_shared_parser():
    adapter = RecoAtlasEnvironment.__new__(RecoAtlasEnvironment)
    adapter._episode = episode()
    adapter.search_products("q", top_k=5)
    feedback = json.loads(adapter.submit_report([]))
    assert feedback["valid"] and adapter.get_reward() == 1.0
    adapter.submit_report([{"item_id": "i0"}])
    assert adapter.get_reward() == -1


def test_miles_reward_uses_shared_parser_and_ordered_seen_ids():
    service = MilesToolService(FakeEnv())
    result = asyncio.run(service.score("bundle", ["i0", "i1", "i2"], {"results": []}, ["i2", "i4", "i5", "i0", "i1"]))
    assert result == {"reward": 1 / 3, "metrics": {"sethit": 1 / 3, "validity": 1}}


def test_miles_preparation_is_label_free_and_backfills_before_validity():
    service = MilesToolService(FakeEnv())
    response = asyncio.run(service.prepare_report(
        {"results": [{"product_id": "i2"}, {"product_id": "i2"}, {"product_id": "unknown"}]},
        3, ["i2", "i4", "i5", "i0"],
    ))
    assert response["valid"]
    assert [item["product_id"] for item in response["report"]["results"]] == ["i2", "i4", "i5"]
    assert set(response) == {"valid", "errors", "report"}
    failed = asyncio.run(service.prepare_report({"results": [{"item_id": "i2"}]}, 3, ["i2", "i4", "i5"]))
    assert not failed["valid"] and failed["report"] is None


def test_env_can_load_only_development_or_no_queries(monkeypatch):
    from recoatlas import assets
    from recoatlas.env import Env

    downloaded = []
    monkeypatch.setattr(assets, "load_split", lambda dataset, split: downloaded.append(split) or [])
    monkeypatch.setattr(Env, "_load_tools", lambda self: None)
    assert list(Env("amazon-musical-instruments", splits=("agent_dev",)).splits) == ["agent_dev"]
    assert downloaded == ["agent_dev"]
    assert Env("amazon-musical-instruments", splits=()).splits == {}
    assert downloaded == ["agent_dev"]
    with pytest.raises(ValueError, match="Unknown splits"):
        Env("amazon-musical-instruments", splits=("typo",))


@pytest.mark.parametrize("k", [3, 5, 7])
def test_baseline_requests_paper_candidate_count(k):
    env = FakeEnv()
    ep = Episode(
        env,
        {
            "partition": "agent_dev",
            "candidate_id": "q1",
            "query": "query",
            "positive_items": [f"i{i}" for i in range(k)],
        },
    )
    run_baseline(ep)
    discovery = [args for name, args in env.calls if name != "get_substitute_products"]
    assert len(discovery) == 2
    assert all(args["top_k"] == max(20, 5 * k) for args in discovery)
