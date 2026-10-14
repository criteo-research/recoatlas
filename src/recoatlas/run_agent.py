"""Run the frozen three-tool agent; evaluate only after each report is finished."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import assets
from .agent import PAPER_CONFIG, PROMPTS, Agent
from .backend import get_client, paper_backend
from .env import Env
from .model_profiles import PROFILES
from .run_store import RunStore, code_fingerprint, fingerprint, write_json
from .scoring import sethit


def evaluate_query(agent, row):
    """References stay in this evaluator, not in Agent.run or its model messages."""
    result = {
        "candidate_id": row["candidate_id"],
        "requested_count": len(row["positive_items"]),
        "status": "error",
        "predicted_item_ids": [],
    }
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


def select_rows(rows, *, limit=None, smoke=False, shard_index=0, num_shards=1):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require --num-shards >= 1 and 0 <= --shard-index < --num-shards")
    if smoke:
        selected = {}
        for row in rows:
            selected.setdefault(len(row["positive_items"]), row)
        if not set(range(3, 8)).issubset(selected):
            raise ValueError("Development smoke requires at least one example for each K=3,...,7")
        return [selected[k] for k in range(3, 8)]
    return rows[:limit][shard_index::num_shards]


def run_queries(env, backend, model, rows, store, *, concurrency=1, on_progress=lambda: None):
    """Bounded concurrency: every episode has its own Agent; tools are shared, never batched."""
    pending = iter(row for row in rows if row["candidate_id"] not in store.done)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        active = set()

        def submit_one():
            row = next(pending, None)
            if row is not None:
                active.add(pool.submit(evaluate_query, Agent(env, backend, model=model), row))

        for _ in range(concurrency):
            submit_one()
        try:
            while active:
                ready, active = wait(active, return_when=FIRST_COMPLETED)
                for future in ready:
                    store.save(future.result())
                    on_progress()
                    submit_one()
        finally:
            for future in active:
                future.cancel()


def software_versions():
    result = {}
    for package in ("torch", "transformers", "peft", "openai", "huggingface_hub", "xgrammar"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
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
    p.add_argument(
        "--model-profile", choices=["auto", *PROFILES], default="auto", help="Model family for local serving aliases"
    )
    p.add_argument(
        "--hosted",
        action="store_true",
        help="Do not send local model controls; hosted reasoning stays provider-managed",
    )
    p.add_argument("--json-only", action="store_true", help="JSON mode instead of the paper's vLLM grammar")
    p.add_argument(
        "--dry-run", action="store_true", help="Print resolved serving settings; no downloads, API key or model calls"
    )
    p.add_argument("--limit", type=int, help="Run only the first N queries (a subset, not full-split results)")
    p.add_argument("--smoke", action="store_true", help="agent_dev only: first example of each size K=3,...,7")
    p.add_argument("--concurrency", type=int, default=1, help="Concurrent episodes; tools are not search-batched")
    p.add_argument("--resume", action="store_true", help="Keep recorded successes AND failures; run only missing IDs")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--model-revision", help="Expected served checkpoint pin, supplied by the launcher")
    p.add_argument("--runtime", type=Path, help="Launcher-owned server provenance JSON (no credentials)")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    try:
        backend = paper_backend(
            None, thinking=a.thinking, local=not a.hosted, grammar=not a.json_only, model_profile=a.model_profile
        )
        serving = backend.configuration(a.model)
    except ValueError as error:
        p.error(str(error))
    if a.dry_run:
        print(json.dumps(serving, indent=2))
        return 0
    if a.output is None:
        p.error("--output is required unless --dry-run is used")
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive")
    if a.concurrency < 1:
        p.error("--concurrency must be positive")
    if a.smoke and (a.split != "agent_dev" or a.limit is not None):
        p.error("--smoke requires --split agent_dev and cannot be combined with --limit")
    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards:
        p.error("Require --num-shards >= 1 and 0 <= --shard-index < --num-shards")
    if not a.json_only:
        # Fail before loading tools or sending requests if the grammar extra is missing.
        from .output_contract import compiled_contract

        for k in range(3, 8):
            compiled_contract(k)
            compiled_contract(k, True)
    # Reading small pinned queries before loading tools allows resume preflight without GPU allocation.
    all_rows = assets.load_split(a.dataset, a.split)
    rows = select_rows(all_rows, limit=a.limit, smoke=a.smoke, shard_index=a.shard_index, num_shards=a.num_shards)
    if not rows:
        p.error("The selected shard/subset is empty")
    from .env import Settings

    summary = {
        "dataset": a.dataset,
        "split": a.split,
        "expected_split_queries": len(all_rows),
        "selected_queries": len(rows),
        "selection": {"limit": a.limit, "smoke": a.smoke, "shard_index": a.shard_index, "num_shards": a.num_shards},
        "concurrency": a.concurrency,
        "model": a.model,
        "paper_config": {k: v for k, v in PAPER_CONFIG.items() if k not in {"model", "model_revision"}},
        "expected_model_revision": a.model_revision
        or (PAPER_CONFIG["model_revision"] if a.model == PAPER_CONFIG["model"] else None),
        "serving": serving,
        "thinking": serving["thinking"],
        "grammar": not a.json_only,
        "settings": asdict(Settings(device=a.device, tools=a.tools)),
        "artifact_sources": assets.SOURCES,
        "prompt_sha256": hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest(),
        "code_sha256": code_fingerprint(),
        "selected_rows_sha256": fingerprint(rows),
        "software": software_versions(),
        "runtime": json.loads(a.runtime.read_text()) if a.runtime else None,
    }
    try:
        with RunStore(a.output, summary, rows, resume=a.resume) as store:

            def progress():
                current = {**summary, **store.progress()}
                write_json(store.summary, current)
                print(json.dumps({"output": str(a.output), **store.progress()}), file=sys.stderr, flush=True)

            if len(store.done) < len(rows):
                backend.client = get_client(base_url=a.base_url)
                env = Env(a.dataset, device=a.device, tools=a.tools)
                try:
                    run_queries(env, backend, a.model, rows, store, concurrency=a.concurrency, on_progress=progress)
                finally:
                    store.export()
                    progress()
            else:
                store.export()
                progress()
            final = {**summary, **store.progress()}
    except (ValueError, BlockingIOError) as error:
        p.error(str(error))
    print(json.dumps(final, indent=2))
    return 0 if final["completed"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
