# Interface contract

## I-001 — Evidence-view action

An action token resolves to one frozen pair of temporal samples and test-function count. It cannot
edit simulator outputs, candidate terms, thresholds, verifier code, or owner labels.

## I-002 — Evaluator output

Each view returns one status and optional support plus public diagnostics. Owner truth is joined only
after the action has been fixed.

## I-003 — Agreement eligibility

For a fixed temporal-node count, modes 2, 3, 4, and 5 must all be terminal and have exactly identical
status/support signatures. Otherwise the group is ineligible and the control abstains.

## I-004 — Reproduction output

`scripts/reproduce_findings.py` must emit JSON with status `PASS`, 33 checks, and zero failures.

