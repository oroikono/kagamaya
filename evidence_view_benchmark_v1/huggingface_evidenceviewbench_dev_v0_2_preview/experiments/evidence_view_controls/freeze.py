"""Create the exclusive pre-outcome source freeze."""

from __future__ import annotations

import json

from .contract import FREEZE, digest, freeze_body, publish_exclusive


def main() -> None:
    body = freeze_body()
    payload = body | {"freeze_sha256": digest(body)}
    publish_exclusive(FREEZE, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

