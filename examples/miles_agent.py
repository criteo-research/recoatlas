"""Run Miles' RecoAtlas agent loop on H&M requests, with recoatlas as its tool service.

This is the rollout half of Miles training (no Megatron): a Miles checkout's unchanged
`examples/recoatlas/agent.py`, an OpenAI-compatible model server, and MilesToolService.

    vllm serve Qwen/Qwen3-4B --port 8001 --served-model-name model \
        --enable-auto-tool-choice --tool-call-parser hermes
    uv run --extra tools --with openai python examples/miles_agent.py --miles PATH --n 400
"""

import argparse
import asyncio
import os
import statistics
import sys
from types import SimpleNamespace

from openai import AsyncOpenAI

import recoatlas
from recoatlas.miles import MilesToolService, miles_rows


def as_actor_handle(service: MilesToolService) -> SimpleNamespace:
    """Stand-in for a Ray actor handle: Miles calls `service.execute.remote(...)` and awaits it."""
    return SimpleNamespace(execute=SimpleNamespace(remote=service.execute), score=SimpleNamespace(remote=service.score))


async def main(args: argparse.Namespace) -> None:
    sys.path.insert(0, os.path.expanduser(args.miles))
    from examples.recoatlas.agent import Agent, AgentConfig
    from examples.recoatlas.prompt import build_system_prompt

    env = recoatlas.load("hm-clothing", device=args.device)
    service = as_actor_handle(MilesToolService(env))
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    request_kwargs = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.7}

    async def rollout(row: dict) -> dict:
        metadata = {**row["metadata"], "max_tool_rounds": 6, "report_repairs": 2}
        agent = Agent(client, service, "model", AgentConfig.from_metadata(metadata))
        return await agent.run(row["prompt"], request_kwargs, metadata)

    rows = miles_rows(env, args.split, build_system_prompt)[: args.n]
    results = await asyncio.gather(*(rollout(row) for row in rows))
    await client.close()

    # The Miles agent tracks seen products itself; recompute every reward with the package to cross-check.
    for row, result in zip(rows, results, strict=True):
        if result["report_validity"]:
            again = env.score(result["report"]["results"], row["metadata"]["ground_truth"], result["seen_product_ids"])
            assert again["reward_direct"] == result["reward"]
    valid = [r for r in results if r["report_validity"]]
    print(f"rollouts: {len(results)}, valid reports: {len(valid)}")
    print(f"mean reward: {statistics.mean(r['reward'] for r in results):.4f}")
    print(f"mean tool calls: {statistics.mean(r['agent_metrics']['tool_calls'] for r in results):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=32, help="number of requests")
    parser.add_argument("--split", default="agent_hidden")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--miles", required=True, help="path to a Miles checkout with examples/recoatlas")
    parser.add_argument("--device", default="cuda")
    asyncio.run(main(parser.parse_args()))
