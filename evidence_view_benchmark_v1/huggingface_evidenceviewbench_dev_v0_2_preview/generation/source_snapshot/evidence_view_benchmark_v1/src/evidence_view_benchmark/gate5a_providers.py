"""Provider adapters for Gate 5A, isolated from historical Gate-4 code."""

from __future__ import annotations

import time
from pathlib import Path

from .gate4_providers import Generation, read_anthropic_key


class AnthropicProviderGate5a:
    """Claude adapter with the documented Opus-4.8 temperature compatibility rule."""

    def __init__(self, credential_path: Path) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("official Anthropic SDK is unavailable") from exc
        self._client = Anthropic(api_key=read_anthropic_key(credential_path))

    def count_tokens(self, *, model: str, prompt: str) -> int:
        result = self._client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return int(result.input_tokens)

    def generate(self, *, model: str, prompt: str, max_output_tokens: int) -> Generation:
        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
        }
        if model != "claude-opus-4-8":
            request["temperature"] = 0
        started = time.monotonic()
        response = self._client.messages.create(**request)
        latency = time.monotonic() - started
        blocks = list(response.content)
        if len(blocks) != 1 or getattr(blocks[0], "type", "") != "text":
            raise RuntimeError("provider returned a non-text or multi-block response")
        return Generation(
            raw_response=str(blocks[0].text),
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
            latency_seconds=latency,
            provider_response_id=getattr(response, "id", None),
            stop_reason=getattr(response, "stop_reason", None),
            resolved_model_id=str(getattr(response, "model", model)),
        )

