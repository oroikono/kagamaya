"""Credential-safe Claude and local-Qwen provider seams."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path


_KEY_PATTERN = re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}")


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    response_id: str | None
    stop_reason: str | None


def read_anthropic_key(path: Path) -> str:
    """Return exactly one in-memory key; never print, persist, or hash it."""
    matches = set(_KEY_PATTERN.findall(path.read_bytes()))
    if len(matches) != 1:
        raise RuntimeError("credential file must contain exactly one Anthropic key")
    return next(iter(matches)).decode("ascii")


class ClaudeFileProvider:
    def __init__(self, key_file: Path) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("official anthropic SDK is unavailable") from exc
        self._client = Anthropic(api_key=read_anthropic_key(key_file))

    def count_tokens(self, *, model: str, system: str, user: str) -> int:
        response = self._client.messages.count_tokens(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return int(response.input_tokens)

    def generate(self, *, model: str, system: str, user: str, max_output_tokens: int) -> ProviderResponse:
        started = time.monotonic()
        response = self._client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_output_tokens,
        )
        latency = time.monotonic() - started
        blocks = list(response.content)
        if len(blocks) != 1 or getattr(blocks[0], "type", "") != "text":
            raise RuntimeError("provider response must contain exactly one text block")
        return ProviderResponse(
            text=str(blocks[0].text),
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
            latency_seconds=latency,
            response_id=getattr(response, "id", None),
            stop_reason=getattr(response, "stop_reason", None),
        )


class QwenTransformersProvider:
    def __init__(self, model_path: str) -> None:  # pragma: no cover - Euler GPU
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()

    def _encoded(self, system: str, user: str):  # pragma: no cover - Euler GPU
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._tokenizer(rendered, return_tensors="pt")

    def count_tokens(self, *, model: str, system: str, user: str) -> int:  # pragma: no cover
        del model
        return int(self._encoded(system, user)["input_ids"].shape[1])

    def generate(self, *, model: str, system: str, user: str, max_output_tokens: int) -> ProviderResponse:  # pragma: no cover
        del model
        encoded = self._encoded(system, user).to(self._model.device)
        started = time.monotonic()
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_output_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency = time.monotonic() - started
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        return ProviderResponse(
            text=self._tokenizer.decode(continuation, skip_special_tokens=True).strip(),
            input_tokens=int(encoded["input_ids"].shape[1]),
            output_tokens=int(continuation.shape[0]),
            latency_seconds=latency,
            response_id=None,
            stop_reason="local_generation",
        )

