# Decisions

## D-001 — Exact manuscript baseline

- Date: 2026-08-27
- Decision: bind the release to v19 of the Sim2Science spotlight manuscript.
- Witness: source hash and `MANIFEST.sha256`.
- Implication: later manuscript edits require a new decision and manifest.

## D-002 — Raw-first primary reproduction

- Date: 2026-08-27
- Decision: ship all 100 primary confirmation shards and independently recompute counts from them.
- Witness: `scripts/reproduce_findings.py`, 33/33 checks.
- Implication: prose summaries are not accepted evidence for headline numbers.

## D-003 — Private-first release

- Date: 2026-08-27
- Decision: keep the repository private until licensing, provider-output redistribution, authorship,
  and citation review are resolved.
- Witness: human release-policy decision.

