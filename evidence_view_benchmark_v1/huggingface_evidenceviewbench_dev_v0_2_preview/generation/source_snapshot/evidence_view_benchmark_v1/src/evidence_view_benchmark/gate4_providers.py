"""Credential-safe provider adapters for the bounded Gate-4 smoke."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path


_ANTHROPIC_KEY = re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}")


@dataclass(frozen=True)
class Generation:
    raw_response: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    provider_response_id: str | None
    stop_reason: str | None
    resolved_model_id: str


def read_anthropic_key(path: Path) -> str:
    matches = set(_ANTHROPIC_KEY.findall(path.read_bytes()))
    if len(matches) != 1:
        raise RuntimeError("credential file must contain exactly one Anthropic key")
    return next(iter(matches)).decode("ascii")


class AnthropicProvider:
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
        started = time.monotonic()
        response = self._client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_output_tokens,
        )
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


class QwenTransformersProvider:
    def __init__(self, model_path: Path) -> None:  # pragma: no cover - Euler GPU
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()

    def _encoded(self, prompt: str):  # pragma: no cover - Euler GPU
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._tokenizer(rendered, return_tensors="pt")

    def count_tokens(self, *, model: str, prompt: str) -> int:  # pragma: no cover
        del model
        return int(self._encoded(prompt)["input_ids"].shape[1])

    def generate(self, *, model: str, prompt: str, max_output_tokens: int) -> Generation:  # pragma: no cover
        encoded = self._encoded(prompt).to(self._model.device)
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
        return Generation(
            raw_response=self._tokenizer.decode(continuation, skip_special_tokens=True).strip(),
            input_tokens=int(encoded["input_ids"].shape[1]),
            output_tokens=int(continuation.shape[0]),
            latency_seconds=latency,
            provider_response_id=None,
            stop_reason="local_generation",
            resolved_model_id=model,
        )
