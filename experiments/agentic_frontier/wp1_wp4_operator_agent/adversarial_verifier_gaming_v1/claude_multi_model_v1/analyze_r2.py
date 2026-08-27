"""Analyze completed Haiku v1 plus matched Sonnet/Opus R2 arms."""

from __future__ import annotations

from . import analyze as base
from .runner import MODEL_SPECS as V1_SPECS
from .runner_r2 import MODEL_SPECS_R2


if __name__ == "__main__":
    base.MODEL_SPECS = {
        "claude-haiku-4-5-20251001": V1_SPECS["claude-haiku-4-5-20251001"],
        **MODEL_SPECS_R2,
    }
    base.main()

