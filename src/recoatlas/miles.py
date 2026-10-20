"""Miles adapter: RecoAtlas behind the tool-service interface of Miles' `examples/recoatlas` agent.

Miles' agent loop (`examples/recoatlas/agent.py`) talks to a service object with two methods,
`execute(category, tool_name, arguments)` and `score(task_type, ground_truth, report, seen_product_ids)`.
`MilesToolService` implements them on a shared `Env`. In a Miles job, run it as one Ray actor:

    service = ray.remote(MilesToolService).options(name="recoatlas", num_gpus=0).remote()

and build the training rows with `miles_rows(env, split, system_prompt)`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from .env import DISCOVERY_TOOLS, Env, shared_env
from .reports import prepare_report
from .scoring import validation_errors


class MilesToolService:
    """Stateless tools and scoring for Miles. Episode state (seen IDs) is kept by the Miles agent."""

    def __init__(self, env: Env | None = None):
        self.env = env or shared_env()

    async def prepare_report(self, report: Any, k: int, seen_product_ids: list[str]) -> dict:
        """Label-free paper processing before the Miles agent attempts report repair.

        No reference IDs or rewards are accepted or returned by this method.
        """
        try:
            if isinstance(report, str):
                report = json.loads(report)
            if not isinstance(report, dict):
                raise ValueError("the final answer must be a JSON object")
            if isinstance(seen_product_ids, set | frozenset):
                raise ValueError("seen IDs must preserve discovery order")
            processed = prepare_report("", report, list(seen_product_ids), [], k, self.env.catalog)
        except (TypeError, ValueError) as error:
            return {"valid": False, "errors": [str(error)], "report": None}
        errors = validation_errors(processed["results"], k, self.env.catalog, seen_product_ids)
        compact = {
            "report_explanation": processed["report_explanation"],
            "results": [{"product_id": item["product_id"], "reasoning": item["reasoning"]}
                        for item in processed["results"]],
        }
        return {"valid": not errors, "errors": errors, "report": compact}

    async def execute(self, category: str, tool_name: str, arguments: dict) -> dict:
        """Run one tool call. Returns {"ok", "content", "product_ids", "metrics"} as Miles expects."""
        try:
            if category != self.env.dataset:
                raise ValueError(f"category must be {self.env.dataset!r}")
            content = await asyncio.to_thread(self.env.call, tool_name, dict(arguments))
        except (TypeError, ValueError) as error:  # bad arguments go back to the model as a tool error
            return {"ok": False, "content": str(error), "product_ids": [], "metrics": {}}
        product_ids = self.env.returned_ids(content) if tool_name in DISCOVERY_TOOLS else []
        return {"ok": True, "content": content, "product_ids": product_ids, "metrics": {}}

    async def score(self, task_type: str, ground_truth: list[str], report: Any, seen_product_ids: list[str]) -> dict:
        """Filter/backfill, then score. Preserve discovery order in seen_product_ids.

        Returns {"reward", "metrics"}; reward = reward_direct.
        """
        if task_type != "bundle":
            raise ValueError(f"RecoAtlas requests are bundles, got task_type={task_type!r}")
        if isinstance(report, str):
            report = json.loads(report)
        results = report.get("results") if isinstance(report, dict) else None
        scores = self.env.score(results, ground_truth, seen_product_ids)
        return {"reward": scores["reward_direct"], "metrics": {"sethit": scores["sethit"], "validity": scores["valid"]}}


def miles_rows(env: Env, split: str, system_prompt: Callable[[int], str] | None = None) -> list[dict]:
    """Miles JSONL rows for one split. `system_prompt(k)` builds the system message for a request with K products."""
    rows = []
    for row in env.splits[split]:
        k = len(set(row["positive_items"]))
        messages = [{"role": "user", "content": row["query"]}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt(k)})
        rows.append(
            {
                "prompt": messages,
                "label": row["candidate_id"],
                "metadata": {
                    "category": env.dataset,
                    "task_type": "bundle",
                    "ground_truth": list(row["positive_items"]),
                    "query": row["query"],
                    "query_key": row["candidate_id"],
                    "top_k": k,  # Miles' name for the number of products in the report
                },
            }
        )
    return rows
