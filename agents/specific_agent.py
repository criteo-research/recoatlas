"""Task-mode-specific ProtocolAgent variant.

This keeps the vanilla ``ProtocolAgent`` as the control arm while adding a
separate agent whose prompt asks the model to infer the task mode from the user
need.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from .protocol import ProtocolAgent


class SpecificAgent(ProtocolAgent):
    """Protocol agent that selects a task-mode objective from the prompt."""

    _TASK_MODES = (
        "Given the user query, first identify which task mode best describes their intent. "
        "Use the selected mode's objective to guide your tool choices and final recommendations:\n\n"
        "retrieval_clear:\n"
        "The user is asking for products that directly match a fairly specific need and constraints. "
        "Favor items that satisfy the stated constraints and intent. Extra complementarity and diversity "
        "should only matter when they help the stated request.\n\n"
        "comparative_shopping:\n"
        "The user is trying to choose among options. Favor relevant products that make the comparison useful, "
        "such as different strengths, styles, brands, or use cases. The final set should help the user see good "
        "alternatives for the same need.\n\n"
        "bundle:\n"
        "The user is trying to assemble a set of products for a broader activity, setup, or goal. Favor items "
        "that play different useful roles and make sense together. The final set should feel like a coherent "
        "kit, not just a ranked list of similar products."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_context = threading.local()

    def set_query_context(self, **kwargs) -> None:
        """Receive per-query metadata without sharing it across worker threads."""
        self._thread_context.query_context = dict(kwargs)

    def _query_context(self) -> dict[str, Any]:
        return getattr(self._thread_context, "query_context", {})

    def _tool_descriptions_for_prompt(self) -> str:
        lines = [
            "1. search_products\n"
            '   Input: {"query": "<search query>", "top_k": <int>}\n'
            "   Finds products relevant to a text query."
        ]
        if self.complementarity is not None:
            lines.append(
                "2. get_complementary_products\n"
                '   Input: {"item_ids": ["<item_id>", ...], "top_k": <int>}\n'
                "   Finds products that can be purchased with the given anchor products."
            )
        if self.substitutability is not None:
            lines.append(
                "3. get_substitute_products\n"
                '   Input: {"item_ids": ["<item_id>", ...], "similarity_threshold": <float>}\n'
                "   Identifies near-duplicate or interchangeable products from a candidate set."
            )
        return "\n\n".join(lines)

    def _system_prompt(self) -> str:
        return (
            "You are an expert product recommendation agent integrated with local catalog tools.\n\n"
            "Mission:\n"
            f"Build a final recommendation set of exactly {self.top_k} unique products that best satisfies "
            "the user's specific need based on the appropriate task mode.\n\n"
            "Task modes:\n"
            f"{self._TASK_MODES}\n\n"
            "Available tools:\n"
            "You have access to the following tools to explore the catalog.\n"
            f"{self._tool_descriptions_for_prompt()}\n\n"
            "Execution rules and constraints:\n"
            f"1. Tool budget: You have a maximum of {self.max_tool_rounds} tool-use rounds before you must "
            "finalize your selection. Decide how many calls to use based on the user need, the selected "
            "task-mode objective, and the current candidate set. Each tool call should have a clear purpose.\n"
            "2. Sequential logic: Choose your tool sequence yourself. "
            "Before each action, decide what information or candidate coverage would most improve the final "
            "recommendation set for the given task mode.\n"
            f"3. Final selection: Return exactly {self.top_k} products. Do not include duplicate product_ids. "
            "Match the selected task mode.\n\n"
            "Output protocol:\n"
            "You must respond in strictly valid JSON format. Do not wrap the JSON in markdown blocks and do "
            "not include any conversational text before or after the JSON.\n\n"
            "When you need to use a tool to gather more information, output exactly one JSON object in this "
            "format:\n"
            '{"thought": "<brief decision summary and chosen task mode>", '
            '"step_goal": "<what this action should improve>", '
            '"action": "<tool name>", "action_input": {...}}\n\n'
            "When you have gathered sufficient candidates or reached your tool budget, finalize your "
            "recommendations by outputting exactly one JSON object in this format:\n"
            '{"report_explanation": "<brief summary of your strategy and how the items fulfill the task mode>", '
            '"results": [{"product_id": "<id>", '
            '"reasoning": "<concise explanation of why this specific item fits the user need>"}]}\n\n'
            f"The results array must contain exactly {self.top_k} items. Return valid JSON only."
        )

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
        prompt = (
            f"User need: {need}\n\n"
            f"Task modes:\n{self._TASK_MODES}\n\n"
            f"Candidate set:\n{candidate_lines}\n\n"
            "Return only a valid JSON object with this structure:\n"
            '{"report_explanation": "<brief strategy summary>", "results": '
            '[{"product_id": "<id>", "reasoning": "<why this item fits>"}]}\n\n'
            f"The results array must contain exactly {self.top_k} items.\n"
            "Do not include duplicate product_ids.\n"
            "Match the selected task-mode objective."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are finalizing a product recommendation report. "
                    "Use the user need, candidate set, and selected task-mode objective to produce the final answer. "
                    "Return valid JSON only."
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
