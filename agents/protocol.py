"""Protocol-style recommendation agent with a benchmark-owned JSON action loop."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Protocol  # noqa: F401

from llm_client import get_openai_user, get_sync_client, resolve_model
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tools import RecSysTool, ComplementarityTool, SubstitutabilityTool


class ChatBackend(Protocol):
    """Minimal backend contract for plain chat-completion providers."""

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Return the model's text response for the given chat messages."""


class OpenAIChatBackend:
    """Default chat backend using the existing shared OpenAI client factory."""

    def __init__(self, api_key: str = None, base_url: str = None, model: Optional[str] = None):
        self.client = get_sync_client(api_key=api_key, base_url=base_url, model=model)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        extra = {}
        if response_format is not None:
            extra["response_format"] = response_format
        if timeout is not None:
            extra["timeout"] = timeout
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            user=get_openai_user(),
            **extra,
        )
        return response.choices[0].message.content or ""


class ProtocolAgent:
    """Provider-neutral agent that drives tools through a JSON action protocol."""

    _ACTION_RULES = (
        "Return exactly one JSON object and nothing else.\n"
        "Required keys: thought, step_goal, action, action_input.\n"
        "Do not invent tool names.\n"
    )

    _HARNESS_RULES = (
        "Use add_candidates after useful retrieval steps.\n"
        "Use stop only when the current candidate set is strong enough for final selection.\n"
    )

    _NO_HARNESS_RULES = (
        "When you are ready to finish, return the final shopping report JSON instead of another action.\n"
        "The final shopping report must have keys: report_explanation and results.\n"
    )

    _PLANNING_RULES = (
        "A planning step has already been run before tool execution.\n"
        "Use the provided planning state to guide later tool calls.\n"
    )

    def __init__(
        self,
        recsys_tool: RecSysTool,
        complementarity_tool: ComplementarityTool = None,
        substitutability_tool: SubstitutabilityTool = None,
        *,
        model: str = "gpt-4.1",
        base_url: str = None,
        api_key: str = None,
        top_k: int = 5,
        max_tool_rounds: int = 5,
        temperature: float = 0.2,
        backend: ChatBackend = None,
        use_harness_actions: bool = True,
        planning: bool = False,
        prompt: str = None,
        finalize_prompt: str = None,
        prompt_path: str = None,
        finalize_prompt_path: str = None,
        query_timeout_seconds: float | None = None,
    ):
        self.recsys = recsys_tool
        self.complementarity = complementarity_tool
        self.substitutability = substitutability_tool
        self.model = resolve_model(model)
        self.top_k = top_k
        self.max_tool_rounds = max_tool_rounds
        self.temperature = temperature
        self.backend = backend or OpenAIChatBackend(api_key=api_key, base_url=base_url, model=model)
        self.use_harness_actions = use_harness_actions
        self.planning = planning
        self.prompt = prompt
        self.finalize_prompt = finalize_prompt
        self.prompt_path = prompt_path
        self.finalize_prompt_path = finalize_prompt_path
        self.query_timeout_seconds = query_timeout_seconds

    @classmethod
    def retrieval_only(
        cls,
        recsys_tool: RecSysTool,
        **kwargs,
    ) -> "ProtocolAgent":
        return cls(
            recsys_tool,
            complementarity_tool=None,
            substitutability_tool=None,
            **kwargs,
        )

    def _default_strategy_prompt(self) -> str:
        return (
            "You are a recommendation agent operating over local recommendation tools.\n"
            f"Build a strong final set of exactly {self.top_k} products.\n"
            "Prefer relevance first, then complementary coverage, then diversity.\n"
            "When the toolset supports it, use complementarity to discover related items and substitutability "
            "to prune near-duplicates.\n"
        )

    def _default_no_harness_prompt(self) -> str:
        tool_lines = ["1. search_products: Find products matching a text query"]
        if self.complementarity is not None:
            tool_lines.append(
                "2. get_complementary_products: Find products frequently bought together with given item_ids"
            )
        if self.substitutability is not None:
            tool_lines.append(
                "3. get_substitute_products: Remove near-duplicate products from a candidate set"
            )

        strategy_lines = ["- Start with search_products to find relevant items"]
        if self.complementarity is not None:
            strategy_lines.append(
                "- Call get_complementary_products with promising item_ids to discover related items"
            )
        if self.substitutability is not None:
            strategy_lines.append(
                "- Use get_substitute_products on your candidate set before final selection to prune redundancy"
            )

        final_lines = ["- From all discovered candidates, select exactly {top_k} items"]
        if self.complementarity is not None:
            final_lines.append("- Include a mix of directly relevant and complementary items")
        if self.substitutability is not None:
            final_lines.append("- Prefer the kept items from get_substitute_products over removed ones")
        else:
            final_lines.append("- Avoid recommending near-identical products; prefer variety across subcategories")

        output_lines = [
            "Output protocol:",
            "- For tool-use steps, return exactly one JSON object with keys: thought, step_goal, action, action_input.",
            "- When you are ready to finish, return exactly one JSON object with keys: report_explanation and results.",
            f'- The final results array must contain exactly {self.top_k} items in the form '
            '{{"product_id":"<id>","reasoning":"<why this item fits>"}}.',
        ]

        return (
            "You are a product recommendation assistant with these tools:\n\n"
            + "\n".join(tool_lines)
            + "\n\nStrategy:\n"
            + "\n".join(strategy_lines)
            + "\n\nFinal selection:\n"
            + "\n".join(final_lines).format(top_k=self.top_k)
            + "\n\n"
            + "\n".join(output_lines)
        )

    @staticmethod
    def _default_prompt_path() -> Path:
        return Path(__file__).resolve().parents[1] / "prompts" / "protocol_agent.txt"

    @staticmethod
    def _default_finalize_prompt_path() -> Path:
        return Path(__file__).resolve().parents[1] / "prompts" / "protocol_agent_finalize.txt"

    def _load_prompt_text(self) -> str:
        if self.prompt is not None:
            return self.prompt
        if self.prompt_path is None and not self.use_harness_actions:
            return self._default_no_harness_prompt()
        path = Path(self.prompt_path) if self.prompt_path else self._default_prompt_path()
        return path.read_text()

    def _load_finalize_prompt_text(self) -> str:
        if self.finalize_prompt is not None:
            return self.finalize_prompt
        path = Path(self.finalize_prompt_path) if self.finalize_prompt_path else self._default_finalize_prompt_path()
        return path.read_text()

    def _tool_descriptions(self) -> list[str]:
        descriptions = []
        descriptions.append(
            "- search_products\n"
            '  Action Input: {"query": "<search query>", "top_k": <int>}\n'
            "  Search the catalog for relevant products."
        )
        if self.complementarity is not None:
            descriptions.append(
                "- get_complementary_products\n"
                '  Action Input: {"item_ids": ["<item_id>", ...], "top_k": <int>}\n'
                "  Find complementary products for promising anchors."
            )
        if self.substitutability is not None:
            descriptions.append(
                "- get_substitute_products\n"
                '  Action Input: {"item_ids": ["<item_id>", ...], "similarity_threshold": <float>}\n'
                "  Remove near-duplicate items from a candidate set."
            )
        if self.use_harness_actions:
            descriptions.extend([
                "- add_candidates\n"
                '  Action Input: {"item_ids": ["<item_id>", ...]}\n'
                "  Add items into the current candidate set.",
                "- remove_candidates\n"
                '  Action Input: {"item_ids": ["<item_id>", ...]}\n'
                "  Remove weak or redundant items from the current candidate set.",
                "- stop\n"
                '  Action Input: {"reason": "<short reason>"}\n'
                "  Stop the loop and finalize recommendations from the current candidate set.",
            ])
        return descriptions

    def _rules(self) -> str:
        rules = self._ACTION_RULES
        if self.planning:
            rules += self._PLANNING_RULES
        if self.use_harness_actions:
            rules += self._HARNESS_RULES
        else:
            rules += self._NO_HARNESS_RULES
        return rules

    def _system_prompt(self) -> str:
        template = self._load_prompt_text()
        return template.format(
            top_k=self.top_k,
            tool_descriptions="\n".join(self._tool_descriptions()),
            rules=self._rules(),
        )

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

    def _state_block(
        self,
        *,
        need: str,
        planning_state: dict[str, Any] | None,
        candidate_ids: list[str],
        last_result: Any,
        trace: list[dict[str, Any]],
    ) -> str:
        candidates = [
            {
                "item_id": item_id,
                "text": self.recsys.product_reps.get(item_id, "")[:160],
            }
            for item_id in candidate_ids
            if item_id in self.recsys.product_reps
        ]
        payload = {
            "user_need": need,
            "planning": planning_state or {},
            "target_count": self.top_k,
            "current_candidates": candidates,
            "last_tool_result": last_result,
            "recent_trace": trace[-4:],
        }
        return json.dumps(payload)

    def _allowed_actions(self) -> set[str]:
        actions = {"search_products"}
        if self.complementarity is not None:
            actions.add("get_complementary_products")
        if self.substitutability is not None:
            actions.add("get_substitute_products")
        if self.use_harness_actions:
            actions.update({"add_candidates", "remove_candidates", "stop"})
        return actions

    def _normalize_action_input(self, action: str, action_input: Any, decision: dict[str, Any]) -> Any:
        if isinstance(action_input, dict):
            return action_input

        if action == "todo_writer":
            query_type = decision.get("query_type")
            plan = decision.get("plan")
            normalized = {}
            if isinstance(query_type, str):
                normalized["query_type"] = query_type
            if isinstance(plan, list):
                normalized["plan"] = [step for step in plan if isinstance(step, str)]
            if normalized:
                return normalized

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

        if action in {"add_candidates", "remove_candidates"}:
            if isinstance(action_input, list):
                return {"item_ids": [item for item in action_input if isinstance(item, str)]}
            item_ids = decision.get("item_ids")
            if isinstance(item_ids, list):
                return {"item_ids": [item for item in item_ids if isinstance(item, str)]}

        if action == "stop":
            if isinstance(action_input, str):
                return {"reason": action_input}
            reason = decision.get("reason")
            if isinstance(reason, str):
                return {"reason": reason}

        return action_input

    def _execute_action(
        self,
        action: str,
        args: dict[str, Any],
        candidate_ids: list[str],
    ) -> tuple[Any, list[str]]:
        args = args or {}
        candidate_set = set(candidate_ids)

        if action == "todo_writer":
            query_type = str(args.get("query_type", "")).strip().upper()
            plan = [step for step in args.get("plan", []) if isinstance(step, str)]
            result = {"status": "ok", "query_type": query_type, "plan": plan}
            if isinstance(args.get("query"), str) and args["query"].strip():
                result["query"] = args["query"].strip()
            return result, candidate_ids

        if action == "search_products":
            result = self.recsys.search(args.get("query", ""), top_k=int(args.get("top_k", self.top_k)))
            parsed = self._parse_tool_payload(result)
            if not self.use_harness_actions and isinstance(parsed, list):
                for row in parsed:
                    item_id = row.get("item_id") if isinstance(row, dict) else None
                    if item_id in self.recsys.product_reps and item_id not in candidate_set:
                        candidate_ids.append(item_id)
                        candidate_set.add(item_id)
            return parsed, candidate_ids

        if action == "get_complementary_products" and self.complementarity is not None:
            result = self.complementarity.get_complementary(
                [i for i in args.get("item_ids", []) if isinstance(i, str)],
                top_k=int(args.get("top_k", self.top_k)),
            )
            parsed = self._parse_tool_payload(result)
            if not self.use_harness_actions and isinstance(parsed, list):
                for row in parsed:
                    item_id = row.get("item_id") if isinstance(row, dict) else None
                    if item_id in self.recsys.product_reps and item_id not in candidate_set:
                        candidate_ids.append(item_id)
                        candidate_set.add(item_id)
            return parsed, candidate_ids

        if action == "get_substitute_products" and self.substitutability is not None:
            result = self.substitutability.get_substitutes(
                [i for i in args.get("item_ids", candidate_ids) if isinstance(i, str)],
                similarity_threshold=float(args.get("similarity_threshold", 0.95)),
            )
            parsed = self._parse_tool_payload(result)
            kept_ids = [
                row["item_id"]
                for row in parsed.get("kept", [])
                if isinstance(row, dict) and row.get("item_id") in self.recsys.product_reps
            ]
            return parsed, kept_ids or candidate_ids

        if action == "add_candidates":
            added = []
            for item_id in args.get("item_ids", []):
                if isinstance(item_id, str) and item_id in self.recsys.product_reps and item_id not in candidate_set:
                    candidate_ids.append(item_id)
                    candidate_set.add(item_id)
                    added.append(item_id)
            return {"status": "ok", "added_item_ids": added, "candidate_count": len(candidate_ids)}, candidate_ids

        if action == "remove_candidates":
            removed = {item_id for item_id in args.get("item_ids", []) if item_id in candidate_set}
            kept = [item_id for item_id in candidate_ids if item_id not in removed]
            return {"status": "ok", "removed_item_ids": sorted(removed), "candidate_count": len(kept)}, kept

        if action == "stop":
            return {"status": "stop", "reason": str(args.get("reason", "")).strip()}, candidate_ids

        return {"error": f"Unsupported action: {action}"}, candidate_ids

    def _report_from_data(
        self,
        need: str,
        data: dict[str, Any],
        candidate_ids: list[str],
        trace: list[dict[str, Any]],
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
            if pid not in self.recsys.product_reps or pid in seen:
                continue
            if candidate_ids and pid not in candidate_ids:
                continue
            seen.add(pid)
            output.append({
                "product_id": pid,
                "category": "",
                "rank": i,
                "score": None,
                "complementarity_score": None,
                "displayed_txt": self.recsys.product_reps.get(pid, "")[:240],
                "input_txt": self.recsys.product_reps.get(pid, ""),
                "reasoning": item.get("reasoning", "Selected by protocol agent."),
            })

        if len(output) < self.top_k:
            for pid in candidate_ids:
                if len(output) >= self.top_k or pid in seen:
                    continue
                seen.add(pid)
                output.append({
                    "product_id": pid,
                    "category": "",
                    "rank": len(output) + 1,
                    "score": None,
                    "complementarity_score": None,
                    "displayed_txt": self.recsys.product_reps.get(pid, "")[:240],
                    "input_txt": self.recsys.product_reps.get(pid, ""),
                    "reasoning": "Backfilled from the candidate set.",
                })

        explanation = data.get("report_explanation", "")
        if not explanation and trace:
            explanation = trace[-1].get("thought", "") or trace[-1].get("step_goal", "")
        normalized_trace = []
        for idx, row in enumerate(trace, 1):
            normalized_trace.append({
                "round": idx,
                "reasoning": row.get("thought") or row.get("step_goal"),
                "tool_name": row.get("action"),
                "tool_args": row.get("action_input"),
                "tool_result": row.get("observation"),
            })
        return {
            "need": need,
            "report_explanation": explanation,
            "trace": normalized_trace,
            "results": output[:self.top_k],
        }

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
        return self._parse_json_object(raw)

    def _run_planning_step(self, need: str, *, timeout: float | None = None) -> dict[str, Any]:
        min_plan_steps = 4
        max_plan_steps = min(8, max(min_plan_steps, self.max_tool_rounds))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are planning a product recommendation search. "
                    "Return exactly one JSON object with keys: query_type, plan, and optional query. "
                    "Use query_type values RETRIEVAL, COMPARISON, or BUNDLE.\n\n"
                    "Benchmark query types:\n"
                    "- RETRIEVAL: the user wants products matching a specific need, possibly with constraints. "
                    "Prioritize direct relevance to the user's request.\n"
                    "- COMPARISON: the user is shopping among alternatives for a product need. "
                    "Prioritize strong relevant options that give the user meaningful alternatives rather "
                    "than near-duplicates.\n"
                    "- BUNDLE: the user wants a coherent set of complementary products for a complete setup "
                    "or activity. Plan distinct roles/sub-needs, make sure the products work together, and "
                    "avoid returning only interchangeable alternatives.\n\n"
                    f"The plan must be a JSON array of {min_plan_steps} to {max_plan_steps} short strings. "
                    "Keep the plan concrete and specific to the user need. "
                    "Provide a rewritten search query when useful."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_need": need,
                        "target_count": self.top_k,
                    }
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
        data = self._parse_json_object(raw)
        query_type = str(data.get("query_type", "")).strip().upper()
        if query_type not in {"RETRIEVAL", "COMPARISON", "BUNDLE"}:
            query_type = ""
        plan = data.get("plan", [])
        if isinstance(plan, str):
            plan = [plan]
        elif not isinstance(plan, list):
            plan = []
        plan = [step.strip() for step in plan if isinstance(step, str) and step.strip()]
        result = {"status": "ok", "query_type": query_type, "plan": plan}
        query = data.get("query")
        if isinstance(query, str) and query.strip():
            result["query"] = query.strip()
        return result

    def _finalize_report(
        self,
        need: str,
        candidate_ids: list[str],
        trace: list[dict[str, Any]],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        candidate_lines = "\n".join(
            f"[{item_id}] {self.recsys.product_reps.get(item_id, '')[:300]}"
            for item_id in candidate_ids
            if item_id in self.recsys.product_reps
        )
        prompt = self._load_finalize_prompt_text().format(
            need=need,
            candidate_lines=candidate_lines,
            top_k=self.top_k,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are finalizing recommendations from a pre-built candidate set. "
                    "Only choose items already present in the candidate set."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        raw = self.backend.complete(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        data = self._parse_json_object(raw)
        return self._report_from_data(need, data, candidate_ids, trace)

    def chat_report(self, msg: str) -> dict[str, Any]:
        deadline = self._query_deadline()
        candidate_ids: list[str] = []
        last_result: Any = {"status": "empty"}
        trace: list[dict[str, Any]] = []
        planning_state: dict[str, Any] = {}

        if self.planning:
            planning_observation = self._run_planning_step(
                msg, timeout=self._remaining_timeout(deadline)
            )
            planning_state = {
                "query_type": planning_observation.get("query_type", ""),
                "plan": planning_observation.get("plan", []),
            }
            if isinstance(planning_observation.get("query"), str) and planning_observation["query"].strip():
                planning_state["query"] = planning_observation["query"].strip()
            trace.append({
                "thought": "Classify the need and prepare a concrete search plan before tool use.",
                "step_goal": "Create a planning state for the later tool loop.",
                "action": "todo_writer",
                "action_input": {"need": msg},
                "observation": planning_observation,
            })

        for _ in range(self.max_tool_rounds):
            decision = self._decide_action(
                need=msg,
                planning_state=planning_state,
                candidate_ids=candidate_ids,
                last_result=last_result,
                trace=trace,
                timeout=self._remaining_timeout(deadline),
            )
            if not self.use_harness_actions and "results" in decision:
                trace.append({
                    "thought": str(decision.get("report_explanation", "")).strip(),
                    "step_goal": "Finalize recommendations.",
                    "action": None,
                    "action_input": None,
                    "observation": decision,
                })
                return self._report_from_data(msg, decision, candidate_ids, trace)
            action = str(decision.get("action", "")).strip()
            args = self._normalize_action_input(action, decision.get("action_input", {}), decision)
            if action not in self._allowed_actions() or not isinstance(args, dict):
                trace.append({
                    "thought": str(decision.get("thought", "")).strip(),
                    "step_goal": str(decision.get("step_goal", "")).strip(),
                    "action": "invalid",
                    "action_input": args if isinstance(args, dict) else None,
                    "observation": "Model returned an invalid action; stopping.",
                })
                break

            observation, candidate_ids = self._execute_action(action, args, list(candidate_ids))
            trace.append({
                "thought": str(decision.get("thought", "")).strip(),
                "step_goal": str(decision.get("step_goal", "")).strip(),
                "action": action,
                "action_input": args,
                "observation": observation,
            })
            last_result = observation
            if action == "todo_writer" and isinstance(observation, dict):
                planning_state = {
                    "query_type": observation.get("query_type", ""),
                    "plan": observation.get("plan", []),
                }
                if isinstance(observation.get("query"), str) and observation["query"].strip():
                    planning_state["query"] = observation["query"].strip()
            if action == "stop":
                break

            if action in {"search_products", "get_complementary_products"}:
                item_ids = [
                    row["item_id"]
                    for row in observation
                    if isinstance(row, dict) and row.get("item_id") in self.recsys.product_reps
                ]
                last_result = {"items": item_ids, "rows": observation}

        if not candidate_ids and isinstance(last_result, dict):
            fallback_ids = last_result.get("items", [])
            candidate_ids = [item_id for item_id in fallback_ids if item_id in self.recsys.product_reps][: self.top_k]

        return self._finalize_report(
            msg,
            candidate_ids,
            trace,
            timeout=self._remaining_timeout(deadline),
        )

    def chat(self, msg: str) -> str:
        report = self.chat_report(msg)
        return " ".join(item["product_id"] for item in report.get("results", []))
