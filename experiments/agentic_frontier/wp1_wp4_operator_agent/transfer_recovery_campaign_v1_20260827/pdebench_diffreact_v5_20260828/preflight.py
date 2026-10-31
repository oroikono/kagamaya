"""One-time source and upstream binding before confirmation shards run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import contract, freeze
from .confirmation import validate_upstream
from .pipeline import runtime_info, validate_source, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract.validate()
    active = freeze.verify()
    validate_upstream()
    validate_source(args.data)
    value = {
        "schema": contract.SCHEMA,
        "stage": "source_preflight",
        "contract_sha256": contract.CONTRACT_SHA256,
        "source_closure_sha256": active["source_closure_sha256"],
        "source_filename": contract.FILENAME,
        "source_bytes": contract.FILE_BYTES,
        "source_md5": contract.FILE_MD5,
        "upstream_audit_sha256": contract.UPSTREAM_AUDIT_SHA256,
        "runtime": runtime_info(),
        "confirmation_remained_sealed": True,
    }
    write_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
