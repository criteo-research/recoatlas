"""The frozen three-tool benchmark agent. No policy inheritance or prompt variants."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any

from .env import TOOL_NAMES
from .reports import prepare_report

PROMPTS = json.loads(files("recoatlas").joinpath("prompts/agent.json").read_text())
PAPER_CONFIG = json.loads(files("recoatlas").joinpath("configs/paper_agent.json").read_text())


def candidate_ledger(trace: Sequence[Mapping[str, Any]], catalog) -> list[dict[str, Any]]:
    ledger: dict[str, dict[str, Any]] = {}
    for row in trace:
        action = row.get("action") or row.get("tool_name")
        args = row.get("action_input") or row.get("tool_args") or {}
        observation = row.get("observation") if "observation" in row else row.get("tool_result")
        if not isinstance(observation, list):
            continue
        for rank, item in enumerate(observation, 1):
            if not isinstance(item, dict) or not item.get("item_id"):
                continue
            item_id = str(item["item_id"])
            entry = ledger.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "text": catalog.get(item_id, "")[:180],
                    "best_score": None,
                    "sources": [],
                },
            )
            score = item.get("score")
            if isinstance(score, (int, float)):
                entry["best_score"] = max(float(score), entry["best_score"] or -math.inf)
            source = {"tool": action, "rank": rank}
            if action == "search_products" and isinstance(args, dict):
                source["query"] = str(args.get("query", ""))[:180]
            entry["sources"].append(source)
    return sorted(
        ledger.values(),
        key=lambda row: (row["best_score"] is not None, row["best_score"] or -1),
        reverse=True,
    )


class Agent:
    """One benchmark agent per worker; share the stateless Env, not this agent.

    run() receives only the request and K. References and post-report evaluations
    never enter a model message. The six decision rounds include guard actions;
    JSON repair requests and the finalizer are additional LLM calls, not tool rounds.
    """

    def __init__(self, env, backend, *, model: str = PAPER_CONFIG["model"]):
        settings = env.settings
        if settings.max_tool_results is not None or settings.batch_search:
            raise ValueError("The paper agent requires max_tool_results=None and batch_search=False")
        self.env = env
        self.backend = backend
        self.model = model
        self.top_k = 5
        self.max_tool_rounds = PAPER_CONFIG["max_tool_rounds"]
        self.temperature = PAPER_CONFIG["temperature"]
        self.query_timeout_seconds = PAPER_CONFIG["query_timeout_seconds"]
        self.ledger_limit = PAPER_CONFIG["ledger_limit"]
        self.min_distinct_searches = PAPER_CONFIG["min_distinct_searches"]
        self.min_comp_calls = PAPER_CONFIG["min_comp_calls"]
        self._protocol_stats = Counter()
        self.diversity_checks = []
        self._trace = []

    def _system_prompt(self):
        return (
            PROMPTS["scaffold"].format(
                top_k=self.top_k,
                tool_descriptions=PROMPTS["tool_descriptions"],
                strategy=PROMPTS["strategy"],
                min_distinct_searches=self.min_distinct_searches,
                min_comp_calls=self.min_comp_calls,
                rules=PROMPTS["rules"],
            )
            + "\n\n"
            + PROMPTS["diversity"]
        )

    def _allowed_actions(self):
        return set(TOOL_NAMES)

    def protocol_stats(self, trace=None):
        return {
            **dict(self._protocol_stats),
            **self._protocol_state(self._trace if trace is None else trace),
            "diversity_calls": len(self.diversity_checks),
            "diversity_check_performed": bool(self.diversity_checks),
        }

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            try:
                data = json.loads(raw[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _parse_tool_payload(text: str) -> Any:
        if not isinstance(text, str):
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    def _query_deadline(self) -> float | None:
        if not self.query_timeout_seconds or self.query_timeout_seconds <= 0:
            return None
        import time

        return time.perf_counter() + float(self.query_timeout_seconds)

    @staticmethod
    def _remaining_timeout(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        import time

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Query exceeded timeout before the next LLM call.")
        return remaining

    def _normalize_action_input(self, action: str, action_input: Any, decision: dict[str, Any]) -> Any:
        if isinstance(action_input, dict):
            return action_input

        if action == "search_products":
            if isinstance(action_input, str):
                return {"query": action_input}
            query = decision.get("query")
            if isinstance(query, str):
                return {"query": query}

        if action == "get_complementary_products":
            if isinstance(action_input, list):
                return {"item_ids": [item for item in action_input if isinstance(item, str)]}
            item_ids = decision.get("item_ids")
            if isinstance(item_ids, list):
                return {"item_ids": [item for item in item_ids if isinstance(item, str)]}

        if action == "get_substitute_products":
            if isinstance(action_input, list):
                return {"item_ids": [item for item in action_input if isinstance(item, str)]}
            item_ids = decision.get("item_ids")
            if isinstance(item_ids, list):
                normalized = {"item_ids": [item for item in item_ids if isinstance(item, str)]}
                if "similarity_threshold" in decision:
                    normalized["similarity_threshold"] = decision["similarity_threshold"]
                return normalized

        return action_input

    def _report_from_data(self, need, data, candidate_ids, trace):
        return prepare_report(need, data, candidate_ids, trace, self.top_k, self.env.catalog)

    @staticmethod
    def _search_queries(trace: Sequence[Mapping[str, Any]]) -> list[str]:
        queries = []
        for row in trace:
            if (row.get("action") or row.get("tool_name")) != "search_products":
                continue
            action_input = row.get("action_input") or row.get("tool_args")
            if not isinstance(action_input, dict):
                continue
            query = " ".join(str(action_input.get("query", "")).lower().split())
            if not query:
                continue
            tokens = set(query.split())
            is_distinct = True
            for prior in queries:
                prior_tokens = set(prior.split())
                union = tokens | prior_tokens
                similarity = len(tokens & prior_tokens) / len(union) if union else 1.0
                if similarity >= 0.85:
                    is_distinct = False
                    break
            if is_distinct:
                queries.append(query)
        return queries

    def _protocol_state(self, trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        searches = self._search_queries(trace)
        comp_calls = sum((row.get("action") or row.get("tool_name")) == "get_complementary_products" for row in trace)
        missing = []
        if len(searches) < self.min_distinct_searches:
            missing.append(f"distinct searches: {len(searches)}/{self.min_distinct_searches}")
        if comp_calls < self.min_comp_calls:
            missing.append(f"complementarity calls: {comp_calls}/{self.min_comp_calls}")
        return {
            "distinct_searches": len(searches),
            "complementarity_calls": comp_calls,
            "minimum_met": not missing,
            "missing": missing,
        }

    @staticmethod
    def _nested_final(decision: Mapping[str, Any]) -> dict[str, Any] | None:
        if "results" in decision:
            return dict(decision)
        nested = decision.get("action_input")
        if isinstance(nested, dict) and "results" in nested:
            return dict(nested)
        return None

    def _valid_tool_decision(self, decision: Mapping[str, Any]) -> bool:
        action = str(decision.get("action", "")).strip()
        args = self._normalize_action_input(action, decision.get("action_input", {}), dict(decision))
        return action in self._allowed_actions() and isinstance(args, dict)

    def _forced_protocol_action(
        self,
        *,
        need: str,
        candidate_ids: Sequence[str],
        trace: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        state = self._protocol_state(trace)
        if state["distinct_searches"] < self.min_distinct_searches or not candidate_ids:
            suffix = ""
            if state["distinct_searches"]:
                suffix = " Focus on a different missing functional role in the requested bundle."
            action = {
                "thought": "Protocol guard: gather evidence for another distinct bundle role before finalizing.",
                "step_goal": "Satisfy the minimum independent-search requirement.",
                "action": "search_products",
                "action_input": {"query": need + suffix, "top_k": max(10, self.top_k * 3)},
            }
        else:
            action = {
                "thought": "Protocol guard: test complementarity from an observed anchor before finalizing.",
                "step_goal": "Satisfy the minimum complementarity-evidence requirement.",
                "action": "get_complementary_products",
                "action_input": {"item_ids": [candidate_ids[0]], "top_k": max(10, self.top_k * 3)},
            }
        self._protocol_stats["forced_actions"] += 1
        return action

    def _repair_decision(
        self,
        *,
        decision: Mapping[str, Any],
        need: str,
        planning_state: Mapping[str, Any] | None,
        candidate_ids: list[str],
        last_result: Any,
        trace: list[dict[str, Any]],
        timeout: float | None,
    ) -> dict[str, Any]:
        state = self._protocol_state(trace)
        self._protocol_stats["repair_attempts"] += 1
        prompt = {
            "state": json.loads(
                self._state_block(
                    need=need,
                    planning_state=dict(planning_state or {}),
                    candidate_ids=candidate_ids,
                    last_result=last_result,
                    trace=trace,
                )
            ),
            "rejected_response": decision,
            "protocol_feedback": (
                "Do not finalize yet; required evidence is missing: " + ", ".join(state["missing"])
                if not state["minimum_met"]
                else "The response did not follow the required JSON action/final format."
            ),
        }
        raw = self.backend.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(prompt) + "\nReturn one corrected JSON response only.",
                },
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        repaired = self._parse_json_object(raw)
        nested = self._nested_final(repaired)
        if nested is not None:
            if "results" not in repaired:
                self._protocol_stats["nested_final_repairs"] += 1
            repaired = nested
        if ("results" in repaired and state["minimum_met"]) or self._valid_tool_decision(repaired):
            self._protocol_stats["repair_successes"] += 1
        return repaired

    def _decide_action(
        self,
        *,
        need: str,
        planning_state: dict[str, Any] | None,
        candidate_ids: list[str],
        last_result: Any,
        trace: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": self._state_block(
                    need=need,
                    planning_state=planning_state,
                    candidate_ids=candidate_ids,
                    last_result=last_result,
                    trace=trace,
                ),
            },
        ]
        raw = self.backend.complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        decision = self._parse_json_object(raw)
        nested = self._nested_final(decision)
        if nested is not None and "results" not in decision:
            self._protocol_stats["nested_final_repairs"] += 1
            decision = nested

        state = self._protocol_state(trace)
        premature_final = "results" in decision and not state["minimum_met"]
        invalid = "results" not in decision and not self._valid_tool_decision(decision)
        if not premature_final and not invalid:
            return decision
        if premature_final:
            self._protocol_stats["premature_finalizations"] += 1
        if invalid:
            self._protocol_stats["malformed_decisions"] += 1
        repaired = self._repair_decision(
            decision=decision,
            need=need,
            planning_state=planning_state,
            candidate_ids=candidate_ids,
            last_result=last_result,
            trace=trace,
            timeout=timeout,
        )
        repaired_state = self._protocol_state(trace)
        if "results" in repaired and not repaired_state["minimum_met"]:
            return self._forced_protocol_action(need=need, candidate_ids=candidate_ids, trace=trace)
        if "results" not in repaired and not self._valid_tool_decision(repaired):
            if not repaired_state["minimum_met"]:
                return self._forced_protocol_action(need=need, candidate_ids=candidate_ids, trace=trace)
            self._protocol_stats["unrepaired_decisions"] += 1
        return repaired

    def _state_block(self, *, need, planning_state, candidate_ids, last_result, trace) -> str:
        ledger = candidate_ledger(trace, self.env.catalog)[: self.ledger_limit]
        payload = {
            "known_task_mode": "bundle",
            "user_need": need,
            "target_count": self.top_k,
            "planning": planning_state or {},
            "tool_counts": dict(Counter(row.get("action") for row in trace if row.get("action"))),
            "candidate_ledger": ledger,
            "last_tool_result": last_result,
            "action_history": [{"action": row.get("action"), "action_input": row.get("action_input")} for row in trace],
        }
        payload["diversity_checks"] = self.diversity_checks
        return json.dumps(payload)

    def _finalize_report(self, need, candidate_ids, trace, *, timeout=None):
        ledger = candidate_ledger(trace, self.env.catalog)[: self.ledger_limit]
        strategy = (
            PROMPTS["strategy"]
            + "\n\n"
            + PROMPTS["diversity"]
            + "\nCompleted diversity checks:\n"
            + json.dumps(self.diversity_checks)
        )
        prompt = (
            f"User need: {need}\n\nStrategy:\n{strategy}\n\n"
            f"Persistent candidate ledger:\n{json.dumps(ledger)}\n\n"
            f"Select exactly {self.top_k} unique products, one strong candidate per distinct required role. "
            "Prefer direct constraint matches and do not discard a strong earlier match merely because a later "
            "tool call returned newer candidates. Return JSON with report_explanation and results; each result "
            "contains product_id and reasoning."
        )
        raw = self.backend.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Finalize a bundle from the supplied candidate ledger. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        data = self._parse_json_object(raw)
        return self._report_from_data(need, data, candidate_ids, trace)

    def _execute_action(self, action, args, candidate_ids):
        if action in {"search_products", "get_complementary_products"}:
            if action == "search_products":
                result = self.env.search_products(args.get("query", ""), top_k=int(args.get("top_k", self.top_k)))
            else:
                result = self.env.get_complementary_products(
                    [i for i in args.get("item_ids", []) if isinstance(i, str)],
                    top_k=int(args.get("top_k", self.top_k)),
                )
            parsed = self._parse_tool_payload(result)
            seen = set(candidate_ids)
            if isinstance(parsed, list):
                for row in parsed:
                    item_id = row.get("item_id") if isinstance(row, dict) else None
                    if item_id in self.env.catalog and item_id not in seen:
                        candidate_ids.append(item_id)
                        seen.add(item_id)
            return parsed, candidate_ids
        if action != "get_substitute_products":
            raise ValueError("Unknown tool")
        ids = args.get("item_ids") if isinstance(args, dict) else None
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 60
            or any(not isinstance(i, str) or i not in candidate_ids for i in ids)
            or len(set(ids)) != len(ids)
        ):
            return {"error": "Use 1–60 unique, previously observed candidate IDs."}, candidate_ids
        if args.get("similarity_threshold", 0.95) != 0.95:
            return {"error": "This evaluation fixes similarity_threshold at 0.95."}, candidate_ids
        result = json.loads(self.env.get_substitute_products(ids, similarity_threshold=0.95))
        self.diversity_checks.append(
            {
                "input_ids": ids,
                "kept_ids": [r["item_id"] for r in result["kept"]],
                "removed": [{k: r[k] for k in ("item_id", "duplicate_of", "similarity")} for r in result["removed"]],
                "similarity_threshold": 0.95,
            }
        )
        # Do not silently erase unchecked candidates or override the LLM's selection.
        return result, candidate_ids

    def run(self, msg: str, k: int) -> dict[str, Any]:
        if not isinstance(msg, str) or not msg.strip():
            raise ValueError("A nonempty shopping request is required")
        if type(k) is not int or not 3 <= k <= 7:
            raise ValueError("Paper bundle sizes are 3 through 7")
        self.top_k = k
        self._protocol_stats.clear()
        self.diversity_checks = []
        if hasattr(self.backend, "reset"):
            self.backend.reset()
        if hasattr(self.backend, "set_bundle_size"):
            self.backend.set_bundle_size(k)
        deadline = self._query_deadline()
        candidate_ids: list[str] = []
        last_result: Any = {"status": "empty"}
        trace: list[dict[str, Any]] = []
        self._trace = trace
        planning_state: dict[str, Any] = {}

        for _ in range(self.max_tool_rounds):
            decision = self._decide_action(
                need=msg,
                planning_state=planning_state,
                candidate_ids=candidate_ids,
                last_result=last_result,
                trace=trace,
                timeout=self._remaining_timeout(deadline),
            )
            if "results" in decision:
                trace.append(
                    {
                        "thought": str(decision.get("report_explanation", "")).strip(),
                        "step_goal": "Finalize recommendations.",
                        "action": None,
                        "action_input": None,
                        "observation": decision,
                    }
                )
                return self._report_from_data(msg, decision, candidate_ids, trace)
            action = str(decision.get("action", "")).strip()
            args = self._normalize_action_input(action, decision.get("action_input", {}), decision)
            if action not in self._allowed_actions() or not isinstance(args, dict):
                trace.append(
                    {
                        "thought": str(decision.get("thought", "")).strip(),
                        "step_goal": str(decision.get("step_goal", "")).strip(),
                        "action": "invalid",
                        "action_input": args if isinstance(args, dict) else None,
                        "observation": "Model returned an invalid action; stopping.",
                    }
                )
                break

            observation, candidate_ids = self._execute_action(action, args, list(candidate_ids))
            trace.append(
                {
                    "thought": str(decision.get("thought", "")).strip(),
                    "step_goal": str(decision.get("step_goal", "")).strip(),
                    "action": action,
                    "action_input": args,
                    "observation": observation,
                }
            )
            last_result = observation
            if action in {"search_products", "get_complementary_products"}:
                item_ids = [
                    row["item_id"]
                    for row in observation
                    if isinstance(row, dict) and row.get("item_id") in self.env.catalog
                ]
                last_result = {"items": item_ids, "rows": observation}

        if not candidate_ids and isinstance(last_result, dict):
            fallback_ids = last_result.get("items", [])
            candidate_ids = [item_id for item_id in fallback_ids if item_id in self.env.catalog][: self.top_k]

        return self._finalize_report(
            msg,
            candidate_ids,
            trace,
            timeout=self._remaining_timeout(deadline),
        )
