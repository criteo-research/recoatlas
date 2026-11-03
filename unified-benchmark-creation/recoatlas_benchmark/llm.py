"""Resumable JSON calls to any OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path

from openai import AsyncOpenAI

from .io import append_jsonl, read_jsonl


def parse_json_object(text: str) -> dict:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("Response is not a JSON object")
    return payload


def retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and (status in {408, 409, 425, 429} or status >= 500):
        return True
    message = str(error).casefold()
    return any(
        value in message for value in ("rate limit", "timeout", "connection", "502", "503", "504")
    )


class JsonCaller:
    def __init__(
        self,
        *,
        model: str,
        calls_path: str | Path,
        errors_path: str | Path,
        concurrency: int,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.calls_path = Path(calls_path)
        self.errors_path = Path(errors_path)
        self.client = client or AsyncOpenAI()
        self.owns_client = client is None
        self.semaphore = asyncio.Semaphore(concurrency)
        self.cache = {
            row["key"]: row["payload"]
            for row in read_jsonl(self.calls_path)
            if row.get("status") == "ok"
        }
        self._write_lock = asyncio.Lock()

    async def close(self) -> None:
        if self.owns_client:
            await self.client.close()

    async def call(
        self,
        *,
        key: str,
        messages: list[dict[str, str]],
        validator: Callable[[dict], dict] | None = None,
        max_tokens: int,
        retries: int,
    ) -> dict:
        if key in self.cache:
            return self.cache[key]
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        max_completion_tokens=max_tokens,
                        timeout=120,
                    )
                if response.choices[0].finish_reason != "stop":
                    raise ValueError(f"Incomplete response: {response.choices[0].finish_reason}")
                payload = parse_json_object(response.choices[0].message.content or "{}")
                if validator:
                    payload = validator(payload)
                row = {
                    "key": key,
                    "status": "ok",
                    "payload": payload,
                    "attempt": attempt,
                    "requested_model": self.model,
                    "response_model": getattr(response, "model", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                    "usage": response.usage.model_dump() if response.usage else None,
                }
                async with self._write_lock:
                    append_jsonl(self.calls_path, row)
                    self.cache[key] = payload
                return payload
            except Exception as error:  # noqa: BLE001 - provider and transport errors share retry handling
                last_error = error
                async with self._write_lock:
                    append_jsonl(
                        self.errors_path,
                        {
                            "key": key,
                            "attempt": attempt,
                            "error": f"{type(error).__name__}: {error}"[:1200],
                        },
                    )
                if attempt == retries or not retryable(error):
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 12))
        raise RuntimeError(f"Unresolved LLM call {key}: {last_error}") from last_error
