"""Run the frozen three-tool agent; evaluate only after each report is finished."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from . import assets
from .agent import PAPER_CONFIG, PROMPTS, Agent
from .backend import get_client, paper_backend
from .env import Env
from .scoring import sethit


def evaluate_query(agent, row):
    """References stay in this evaluator, not in Agent.run or its model messages."""
    result = {"candidate_id": row["candidate_id"], "status": "error", "predicted_item_ids": []}
    try:
        report = agent.run(row["query"], len(row["positive_items"]))
        ids = [r["product_id"] for r in report["results"]]
        result.update(status="complete", predicted_item_ids=ids, report=report)
    except Exception as error:  # noqa: BLE001 -- every failed query stays in the evaluation denominator
        result["error_type"] = type(error).__name__
    result["feedback"] = {"sethit_at_k": sethit(result["predicted_item_ids"], row["positive_items"], agent.env.catalog)}
    result["protocol"] = agent.protocol_stats()
    if hasattr(agent.backend, "stats"):
        result["api"] = agent.backend.stats()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=assets.DATASETS, default="hm-clothing")
    p.add_argument("--split", choices=["agent_dev", "agent_hidden"], default="agent_hidden")
    p.add_argument("--tools", choices=["utility", "semantic"], default="utility")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default=PAPER_CONFIG["model"])
    p.add_argument("--base-url")
    p.add_argument("--thinking", choices=["on", "off"], default="off")
    p.add_argument("--hosted", action="store_true", help="Do not send local Qwen serving parameters")
    p.add_argument("--json-only", action="store_true", help="JSON mode instead of the paper's vLLM grammar")
    p.add_argument("--limit", type=int, help="Run only the first N queries (a subset, not full-split results)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    summary_path = a.output.with_suffix(".summary.json")
    if a.output == summary_path or a.output.exists() or summary_path.exists():
        p.error("Use a new output path distinct from its .summary.json path")
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive")
    if a.hosted and a.thinking == "on":
        p.error("Thinking-on needs the local Qwen-compatible serving configuration")
    if not a.json_only:
        # Fail before loading tools or sending requests if the grammar extra is missing.
        from .output_contract import compiled_contract

        for k in range(3, 8):
            compiled_contract(k)
            compiled_contract(k, True)
    client = get_client(base_url=a.base_url)
    env = Env(a.dataset, device=a.device, tools=a.tools)
    backend = paper_backend(client, thinking=a.thinking, local=not a.hosted, grammar=not a.json_only)
    agent = Agent(env, backend, model=a.model)
    rows = env.splits[a.split][: a.limit]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    total_hit = 0.0
    with a.output.open("x") as stream:
        for row in rows:
            result = evaluate_query(agent, row)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            completed += result["status"] == "complete"
            total_hit += result["feedback"]["sethit_at_k"]
    summary = {
        "dataset": a.dataset,
        "split": a.split,
        "expected_split_queries": len(env.splits[a.split]),
        "evaluated_queries": len(rows),
        "completed": completed,
        "failed": len(rows) - completed,
        "mean_sethit_at_k": total_hit / len(rows) if rows else None,
        "model": a.model,
        "paper_config": PAPER_CONFIG,
        "thinking": a.thinking,
        "local_qwen_parameters": not a.hosted and "qwen" in a.model.lower(),
        "grammar": not a.json_only,
        "settings": asdict(env.settings),
        "artifact_sources": assets.SOURCES,
        "prompt_sha256": hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest(),
    }
    with summary_path.open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
