"""Blind whole-set direct-relevance judge with the paper's fixed rubric."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from importlib.resources import files
from pathlib import Path

from . import assets
from .backend import get_client

CONFIG_FILE = files("recoatlas").joinpath("prompts/direct_relevance.json")
CONFIG = json.loads(CONFIG_FILE.read_text())


def make_request(query, k, candidate_id, predictions, catalog):
    """No reference set, history, agent explanations, scores, or tool identity accepted."""
    if type(k) is not int or not 3 <= k <= 7:
        raise ValueError("Paper bundle sizes are 3 through 7")
    if (
        not isinstance(predictions, list)
        or len(predictions) != k
        or any(not isinstance(i, str) for i in predictions)
        or len(set(predictions)) != k
    ):
        raise ValueError("Judge requires a complete unique report")
    if any(i not in catalog for i in predictions):
        raise ValueError("Unknown product")
    ids = list(predictions)
    seed = hashlib.sha256(f"{CONFIG['seed']}|{candidate_id}".encode()).hexdigest()
    random.Random(int(seed, 16)).shuffle(ids)
    texts = {f"P{i + 1}": catalog[x][:297] + "..." if len(catalog[x]) > 300 else catalog[x] for i, x in enumerate(ids)}
    data = {"request": query, "requested_item_count": k, "catalog_evidence": texts}
    prompt = CONFIG["common"] + "\n" + CONFIG["rubric"] + "\n\nINPUT DATA:\n" + json.dumps(data, ensure_ascii=False)
    return [{"role": "system", "content": CONFIG["system"]}, {"role": "user", "content": prompt}], list(texts)


def validate(payload, labels):
    if not isinstance(payload, dict) or set(payload) != {"score", "reason", "evidence_items"}:
        raise ValueError("schema")
    if type(payload["score"]) is not int or not 0 <= payload["score"] <= 4:
        raise ValueError("score")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise ValueError("reason")
    if not isinstance(payload["evidence_items"], list) or any(
        not isinstance(x, str) or x not in labels for x in payload["evidence_items"]
    ):
        raise ValueError("evidence_items")


def judge_request(client, model, messages, labels):
    """Return all attempts; use the first valid judgment. Failure is unresolved, not zero."""
    attempts = []
    for attempt in range(3):
        result = {"attempt": attempt + 1, "status": "error"}
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=CONFIG["temperature"],
                max_completion_tokens=CONFIG["max_completion_tokens"],
                response_format={"type": "json_object"},
            )
            if response.choices[0].finish_reason != "stop":
                raise ValueError("Incomplete judgment")
            payload = json.loads(response.choices[0].message.content)
            validate(payload, labels)
            result.update(status="complete", payload=payload, quality=payload["score"] / 4)
        except Exception as error:  # noqa: BLE001 -- preserve failed measurements without leaking provider text
            # Provider messages can include request bodies or credentials.
            result["error_type"] = type(error).__name__
        attempts.append(result)
        if result["status"] == "complete":
            break
    return attempts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=assets.DATASETS, default="hm-clothing")
    p.add_argument("--split", default="agent_hidden", choices=["agent_dev", "agent_hidden"])
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--model", help="Accessible judge deployment; paper alias: gpt-5.6-luna")
    p.add_argument("--base-url")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs without any API calls")
    a = p.parse_args(argv)
    if not a.dry_run and (not a.output or not a.model):
        p.error("--model and --output are required unless using --dry-run")
    summary_path = a.output.with_suffix(".summary.json") if a.output else None
    if a.output and (a.output == summary_path or a.output.exists() or summary_path.exists()):
        p.error("Use a new output path distinct from its .summary.json path")
    tasks = {r["candidate_id"]: r for r in assets.load_split(a.dataset, a.split)}
    catalog = json.loads(assets.fetch(a.dataset, "product_texts").read_text())
    rows = [json.loads(s) for s in a.traces.read_text().splitlines() if s.strip()]
    if len({r["candidate_id"] for r in rows}) != len(rows):
        p.error("Duplicate prediction IDs")
    requests = []
    # All inputs are validated before constructing a client or making a paid request.
    for row in rows:
        if row["candidate_id"] not in tasks:
            p.error("Traces contain a query outside the selected split")
        if row["status"] not in ("complete", "error"):
            p.error("Unknown trace status")
        if row["status"] == "complete":
            task = tasks[row["candidate_id"]]
            messages, labels = make_request(
                task["query"], len(task["positive_items"]), task["candidate_id"], row["predicted_item_ids"], catalog
            )
            requests.append((row["candidate_id"], messages, labels))
    if a.dry_run:
        print(
            json.dumps(
                {
                    "requests": len(requests),
                    "skipped_error_reports": len(rows) - len(requests),
                    "expected_split_queries": len(tasks),
                    "api_calls": 0,
                },
                indent=2,
            )
        )
        return 0
    client = get_client(base_url=a.base_url)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    good = []
    with a.output.open("x") as stream:
        for cid, messages, labels in requests:
            attempts = judge_request(client, a.model, messages, labels)
            for result in attempts:
                stream.write(json.dumps({"candidate_id": cid, **result}) + "\n")
                stream.flush()
            if attempts[-1]["status"] == "complete":
                good.append(attempts[-1]["quality"])
    summary = {
        "dataset": a.dataset,
        "split": a.split,
        "expected_split_queries": len(tasks),
        "input_reports": len(rows),
        "skipped_error_reports": len(rows) - len(requests),
        "expected_reports": len(requests),
        "valid_judgments": len(good),
        "unresolved": len(requests) - len(good),
        "mean_direct_relevance": sum(good) / len(good) if good else None,
        "requested_model": a.model,
        "paper_deployment": CONFIG["model"],
        "same_deployment_name": a.model == CONFIG["model"],
        "traces_sha256": hashlib.sha256(a.traces.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest(),
        "artifact_sources": assets.SOURCES,
    }
    with summary_path.open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if len(good) == len(requests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
