# Reproducibility protocol

1. Clone the exact commit into a fresh directory.
2. Run `python3 scripts/verify_manifest.py` before executing analyses.
3. Run `python3 scripts/reproduce_findings.py` and require 33/33 checks.
4. Run `python3 -m unittest tests.test_reproduction`.
5. Record Python version, operating system, repository commit, manifest SHA-256, and reproduction-output SHA-256.
6. Do not invoke model providers; frozen raw calls are already included.
7. Treat the sequential result as aggregate-level unless its raw shards are separately restored and bound.

