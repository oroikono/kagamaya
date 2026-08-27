"""Canonical JSON artifact helpers with optional gzip compression."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=None, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)

