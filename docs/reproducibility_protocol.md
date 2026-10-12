# Reproducibility protocol

This is a stored-record recount for the historical v19 archive, not solver
regeneration or a replay of the current camera-ready paper. It uses Python's
standard library and makes no network or model-provider calls.

1. Clone the exact commit into a fresh directory and work from the repository root.
2. Run `python3 scripts/verify_manifest.py` and require success before recounting.
3. Run `python3 scripts/reproduce_findings.py`, require a successful exit status
   and 33/33 checks in `output/reproduction/code_crosscheck.json`.
4. Run `python3 -m unittest discover -s tests -v`.
5. Record Python version, operating system, repository commit, manifest SHA-256, and reproduction-output SHA-256.
6. Keep the frozen inputs unchanged; do not invoke model providers or use this
   procedure to regenerate experiments.
7. Treat the sequential result as aggregate-level: its raw episode shards are
   absent. The later 2-D/coupled and Gray–Scott records remain failed-gate evidence.

Implementations and views are paired, not independent replications. See
[evaluation definitions](evaluation_protocol.md), [sample counts](data_contract.md),
and [artifact scope](artifact_links.md) for the limits of this recount and the
separate private development archives. A manifest pass verifies its stated file
inventory; it is not a scientific correctness, licensing, or privacy approval.

Each recount writes a new attempt receipt atomically, starting with `RUNNING`.
The receipt records an attempt ID, timestamps, script/manifest hashes, available
Git identity, and the Python environment. An interrupted or failed attempt is
not a previous run's PASS. Input/runtime errors record `FAIL` with execution
status `ERROR`; only a completed recount has scientific-check counters. The
recount records the manifest hash but does not verify it itself—run the manifest
command first.

Repository CI runs the same manifest-first procedure and adversarial tests on
Python 3.12 and 3.14, including a test run with assertions disabled. It makes no
provider calls and uploads no generated artifacts. Check the actual workflow
run before claiming that a revision passed CI.
