from __future__ import annotations
import threading
import time
from collections import Counter
from typing import Any

def is_retryable(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and (
        status_code in {408, 409, 425, 429} or status_code >= 500
    ):
        return True
    text = str(error).lower()
    return any(
        token in text
        for token in (
            "429",
            "500",
            "rate limit",
            "timeout",
            "timed out",
            "server error",
            "server_error",
            "internalservererror",
            "connection error",
            "502",
            "503",
            "504",
        )
    )

class ComparableCountingBackend:
    """Thread-local accounting around either hosted or local Chat Completions."""

    def __init__(
        self,
        client,
        api_user: str,
        *,
        max_output_tokens: int,
        temperature: float,
        local: bool,
        thinking: str = "off",
        thinking_token_budget: int | None = None,
        capture_model_traces: bool = False,
        strict_bundle_json: bool = False,
    ):
        self.client = client
        self.api_user = api_user
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.is_local = local
        self.thinking = thinking
        self.thinking_token_budget = thinking_token_budget
        self.capture_model_traces = capture_model_traces
        self.strict_bundle_json = strict_bundle_json
        self.state = threading.local()

    def reset(self) -> None:
        self.state.calls = 0
        self.state.usage = Counter()
        self.state.model_traces = []
        self.state.bundle_size = None

    def set_bundle_size(self, top_k):
        self.state.bundle_size = top_k

    def stats(self) -> dict[str, Any]:
        result = {
            "calls": getattr(self.state, "calls", 0),
            "usage": dict(getattr(self.state, "usage", {})),
        }
        if self.capture_model_traces:
            result["model_traces"] = list(getattr(self.state, "model_traces", []))
        return result

    def complete(
        self,
        *,
        model,
        messages,
        temperature=0.0,
        response_format=None,
        timeout=None,
    ) -> str:
        extra: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_output_tokens,
            "timeout": timeout or 120.0,
        }
        if response_format:
            extra["response_format"] = response_format
        if self.is_local and "qwen" in str(model).lower():
            extra["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.thinking == "on",
                    "preserve_thinking": False,
                }
            }
            if self.thinking == "on" and self.thinking_token_budget is not None:
                extra["extra_body"]["thinking_token_budget"] = self.thinking_token_budget
        contract = None
        if self.strict_bundle_json:
            from bundle_output_contract import VERSION, compiled_contract
            top_k = getattr(self.state, 'bundle_size', None)
            final_only = bool(messages and messages[0].get('content', '').startswith('Finalize a bundle'))
            grammar, sha = compiled_contract(top_k, final_only)
            extra.pop('response_format', None)
            extra.setdefault('extra_body', {})['structured_outputs'] = {'grammar': grammar}
            contract = {'version': VERSION, 'target_count': top_k, 'final_only': final_only, 'grammar_sha256': sha}
        last_error = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(**extra)
                self.state.calls = getattr(self.state, "calls", 0) + 1
                usage = getattr(response, "usage", None)
                if usage is not None:
                    counter = getattr(self.state, "usage", Counter())
                    for key, value in usage.model_dump().items():
                        if isinstance(value, int):
                            counter[key] += value
                    details = usage.model_dump().get("completion_tokens_details") or {}
                    if isinstance(details.get("reasoning_tokens"), int):
                        counter["reasoning_tokens"] += details["reasoning_tokens"]
                    self.state.usage = counter
                choice = response.choices[0]
                message = choice.message
                reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
                if self.capture_model_traces:
                    self.state.model_traces.append({
                        "messages": messages, "content": message.content,
                        "reasoning": reasoning, "finish_reason": choice.finish_reason,
                        "usage": usage.model_dump() if usage else None,
                        **({'output_contract': contract} if contract else {}),
                    })
                if reasoning:
                    counter = getattr(self.state, "usage", Counter())
                    counter["reasoning_characters"] += len(reasoning)
                    counter["responses_with_reasoning"] += 1
                    self.state.usage = counter
                if self.thinking == "on":
                    if choice.finish_reason == "length":
                        raise ValueError("THINKING_OUTPUT_LIMIT: response exhausted max-output-tokens")
                    if not message.content or "<think>" in message.content or "</think>" in message.content:
                        raise ValueError("THINKING_PARSER_ERROR: expected final content separated by vLLM reasoning parser")
                if contract:
                    from bundle_output_contract import validate_contract
                    if choice.finish_reason == 'length':
                        raise ValueError('BUNDLE_OUTPUT_LIMIT: response exhausted max-output-tokens')
                    validate_contract(message.content, contract['target_count'], final_only=contract['final_only'])
                return message.content or ""
            except Exception as error:
                last_error = error
                if attempt == 3 or not is_retryable(error):
                    raise
                time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(str(last_error))
