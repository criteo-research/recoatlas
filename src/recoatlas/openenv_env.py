"""OpenEnv adapter: RecoAtlas as an OpenEnv MCP environment.

In-process:
    env = recoatlas.load("hm-clothing")
    oe = RecoAtlasEnvironment(env)
    oe.reset(split="agent_hidden", index=0)
    oe.step(CallToolAction(tool_name="search_products", arguments={"query": "...", "top_k": 10}))
    oe.step(CallToolAction(tool_name="submit_report", arguments={"results": [...]}))  # done=True, reward set

As one server (one copy of the tools shared by all sessions):
    uv run recoatlas-openenv --port 8000
Settings come from RECOATLAS_* environment variables (see `recoatlas.env.shared_env`), plus
RECOATLAS_MAX_SESSIONS for the server.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Action, Observation, State

from .env import TOOL_DOCS, Env, Episode, shared_env


class RecoAtlasEnvironment(MCPEnvironment):
    """One OpenEnv session = one RecoAtlas episode. All sessions share one loaded `Env`.

    Tools: search_products, get_complementary_products, get_substitute_products, submit_report.
    submit_report ends the episode, also when the report is invalid. The observation's reward is
    `reward_direct` (-1 for an invalid report), and `sethit` and `valid` are added to its metadata.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, env: Env | None = None):
        self._env = env or shared_env()
        self._episode: Episode | None = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        mcp = FastMCP("recoatlas")

        def tool(fn):
            """Register fn as an MCP tool, with the shared description for its name."""
            fn.__doc__ = TOOL_DOCS[fn.__name__]
            return mcp.tool(fn)

        # Tools run in a worker thread so one session's call does not block the server's event loop.
        @tool
        async def search_products(query: str, top_k: int = 5) -> str:
            return await asyncio.to_thread(self._current().search_products, query, top_k)

        @tool
        async def get_complementary_products(item_ids: list[str], top_k: int = 5) -> str:
            return await asyncio.to_thread(self._current().get_complementary_products, item_ids, top_k)

        @tool
        async def get_substitute_products(item_ids: list[str], similarity_threshold: float = 0.95) -> str:
            return await asyncio.to_thread(self._current().get_substitute_products, item_ids, similarity_threshold)

        @tool
        def submit_report(results: list[dict], report_explanation: str = "") -> str:
            return json.dumps(self._current().submit(results, report_explanation))

        super().__init__(mcp)

    def _current(self) -> Episode:
        if self._episode is None:
            raise RuntimeError("Call reset() before using the tools")
        return self._episode

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str = "agent_hidden",
        index: int | None = None,
        candidate_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Start an episode. Without index or candidate_id, a request is drawn at random (using seed)."""
        if index is None and candidate_id is None:
            index = random.Random(seed).randrange(len(self._env.splits[split]))
        ep = self._episode = self._env.episode(split, index=index, candidate_id=candidate_id)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return Observation(
            done=False,
            reward=0.0,
            metadata={"request": ep.request, "k": ep.k, "split": ep.split, "candidate_id": ep.candidate_id},
        )

    def step(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return self._after_step(action, super().step(action, timeout_s=timeout_s, **kwargs))

    async def step_async(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return self._after_step(action, await super().step_async(action, timeout_s=timeout_s, **kwargs))

    def _after_step(self, action: Action, obs: Observation) -> Observation:
        self._state.step_count += 1
        submitted = (
            isinstance(action, CallToolAction)
            and action.tool_name == "submit_report"
            and isinstance(obs, CallToolObservation)
            and obs.error is None
        )
        if not submitted:
            return obs
        scores = self._current().score()
        return obs.model_copy(
            update={"done": True, "reward": scores["reward_direct"], "metadata": {**obs.metadata, **scores}}
        )

    def _step_impl(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action {type(action).__name__}; use ListToolsAction or CallToolAction"},
        )

    @property
    def state(self) -> State:
        return self._state


def create_app():
    """FastAPI app with one shared Env. Serve it with a single worker so the tools are loaded once."""
    from openenv.core.env_server.http_server import create_app as openenv_create_app

    shared_env()  # load before serving, not on the first request
    return openenv_create_app(
        RecoAtlasEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="recoatlas",
        max_concurrent_envs=int(os.environ.get("RECOATLAS_MAX_SESSIONS", "256")),
    )


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve RecoAtlas as an OpenEnv environment (one worker).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
