"""Credential-safe forced-tool Anthropic transport."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Mapping

from .contract import ModelSpec, tool_definition, validate_tool_arguments


_KEY = re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}")


def read_key(path: Path) -> str:
    matches = set(_KEY.findall(path.read_bytes()))
    if len(matches) != 1:
        raise RuntimeError("credential file must contain exactly one Anthropic key")
    return next(iter(matches)).decode("ascii")


class ForcedToolProvider:
    def __init__(self, credential_path: Path) -> None:
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=read_key(credential_path), max_retries=0, timeout=90.0)

    async def count_tokens(self, *, spec: ModelSpec, prompt: str, module: str) -> int:
        tool = tool_definition(module)
        result = await self.client.messages.count_tokens(
            model=spec.request_id,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        return int(result.input_tokens)

    async def generate(
        self, *, spec: ModelSpec, prompt: str, module: str, max_output_tokens: int
    ) -> dict[str, Any]:
        tool = tool_definition(module)
        request: dict[str, Any] = {
            "model": spec.request_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool["name"]},
        }
        if spec.temperature is not None:
            request["temperature"] = spec.temperature
        started = time.monotonic()
        response = await self.client.messages.create(**request)
        latency = time.monotonic() - started
        raw_blocks = [block.model_dump(mode="json") for block in response.content]
        calls = [block for block in raw_blocks if block.get("type") == "tool_use"]
        error = None
        parsed = None
        try:
            if len(calls) != 1 or calls[0].get("name") != tool["name"]:
                raise ValueError("exactly one recognized forced tool call is required")
            parsed = validate_tool_arguments(module, calls[0].get("input"))
        except (TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "resolved_model_id": str(response.model),
            "provider_response_id": str(response.id),
            "stop_reason": str(response.stop_reason),
            "usage": {
                "input_tokens": int(response.usage.input_tokens),
                "output_tokens": int(response.usage.output_tokens),
            },
            "latency_seconds": latency,
            "raw_content_blocks": raw_blocks,
            "tool_call_valid": error is None,
            "tool_parse_error": error,
            "parsed_tool_arguments": parsed,
        }
