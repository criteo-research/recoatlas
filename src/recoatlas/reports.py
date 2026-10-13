"""Original benchmark report parser."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def prepare_report(
    need: str,
    data: dict[str, Any],
    candidate_ids: list[str],
    trace: list[dict[str, Any]],
    k: int,
    catalog: Mapping[str, str],
) -> dict[str, Any]:
    # Invalid model output is an explicit failure, not a Python type crash
    # or a reason to silently fill the answer from arbitrary candidates.
    results = data.get("results", []) if isinstance(data, dict) else None
    if not isinstance(results, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("product_id"), str)
        or not isinstance(item.get("reasoning", ""), str)
        for item in results
    ):
        raise ValueError("INVALID_REPORT_SCHEMA: results must be a list of product_id/reasoning objects")
    output, seen = [], set()
    for i, item in enumerate(results, 1):
        pid = item.get("product_id", "")
        if pid not in catalog or pid in seen:
            continue
        if candidate_ids and pid not in candidate_ids:
            continue
        seen.add(pid)
        output.append(
            {
                "product_id": pid,
                "category": "",
                "rank": i,
                "score": None,
                "complementarity_score": None,
                "displayed_txt": catalog.get(pid, "")[:240],
                "input_txt": catalog.get(pid, ""),
                "reasoning": item.get("reasoning", "Selected by protocol agent."),
            }
        )

    if len(output) < k:
        for pid in candidate_ids:
            if len(output) >= k or pid in seen:
                continue
            seen.add(pid)
            output.append(
                {
                    "product_id": pid,
                    "category": "",
                    "rank": len(output) + 1,
                    "score": None,
                    "complementarity_score": None,
                    "displayed_txt": catalog.get(pid, "")[:240],
                    "input_txt": catalog.get(pid, ""),
                    "reasoning": "Backfilled from the candidate set.",
                }
            )

    explanation = data.get("report_explanation", "")
    if not explanation and trace:
        explanation = trace[-1].get("thought", "") or trace[-1].get("step_goal", "")
    normalized_trace = []
    for idx, row in enumerate(trace, 1):
        normalized_trace.append(
            {
                "round": idx,
                "reasoning": row.get("thought") or row.get("step_goal"),
                "tool_name": row.get("action"),
                "tool_args": row.get("action_input"),
                "tool_result": row.get("observation"),
            }
        )
    return {
        "need": need,
        "report_explanation": explanation,
        "trace": normalized_trace,
        "results": output[:k],
    }
