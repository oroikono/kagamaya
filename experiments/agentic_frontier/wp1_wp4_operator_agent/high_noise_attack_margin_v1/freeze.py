"""Create or validate the one-time source freeze."""

from __future__ import annotations

from .contract import FREEZE, digest, freeze_body, load_freeze, publish_exclusive


def main() -> None:
    if FREEZE.exists():
        load_freeze()
        print(f"VALID {FREEZE}")
        return
    body = freeze_body()
    body["freeze_sha256"] = digest(body)
    publish_exclusive(FREEZE, body)
    load_freeze()
    print(f"CREATED {FREEZE}")


if __name__ == "__main__":
    main()
