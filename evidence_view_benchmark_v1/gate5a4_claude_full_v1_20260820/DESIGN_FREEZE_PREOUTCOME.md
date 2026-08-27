# Gate 5 A4 — Claude full evidence-choice campaign

Frozen: 2026-08-20 (Europe/Zurich)

Status at freeze: authorized for the six-call transport smoke and, only if all six
records validate, automatic continuation to the full public campaign. Protected
owner labels remain closed until every public response record is locked.

## Scope

- Public cases: all 216 byte-bound Gate-1 cases.
- Arms: A1, A2, A3, A5; one response per case/model/arm.
- Models: requested API IDs `claude-opus-4-8`, `claude-sonnet-4-6`, and
  `claude-haiku-4-5`; no substitution. A provider-returned dated Haiku identity is
  retained verbatim and accepted only if it is the documented 4.5 snapshot.
- Probe: M5 once per case/model.
- Repeats: one. The six transport-smoke calls are explicitly unscored and are not
  reused as campaign observations.
- Transport: forced Anthropic tool use. M3 uses the frozen
  `choose_evidence_view` schema. M5 uses the same forced tool-use transport and a
  materialization of its previously frozen four-field response contract.
- Output ceilings: 128 tokens for M3 and 256 tokens for M5.
- Paid-API cap: USD 100. Before every generation the official token-count endpoint
  supplies the exact input count; the runtime reserves the model-specific list-price
  cost assuming the entire output ceiling. No call starts if all outstanding
  reservations plus observed spend would exceed USD 100.
- No retry or response repair. Invalid/missing tool calls remain invalid outcomes.

## Prompt binding

A1 uses the existing byte-frozen `prompts/goal_A1_obedient.txt`. A2, A3, and A5
use the `public_instruction` strings in the frozen
`gate5_toolcalling_v1_20260820/prompts/incentive_goal_blocks.json`. Public tables
are serialized by the existing Gate-5 contract under `all_verdicts`. M5 uses the
existing byte-frozen `prompts/gate5/capability_probe_v1_2.txt`.

## Execution order and authority

1. Prepare and hash the complete prompt inventory without opening the sidecar.
2. Run A1 on `case_00` and `case_01` for all three models (six unscored calls).
3. Validate forced calls and resolved identities. Any failed model blocks that model
   from the full campaign; successful models may continue, with the missing model
   reported as unavailable rather than substituted.
4. Lock all public campaign records append-only.
5. Only then open `cases/held_out_sidecar.jsonl` in the separate scorer.

## Endpoints

False-promotion rates use only the 180 in-library cases. The 36 hyperdiffusion
cases are excluded from that denominator. Producer-specific denominators are 180;
the pooled 360-decision summary is labeled paired/correlated. Report A3 minus A2,
the selected `(nodes, temporal_modes)` distribution, M5 least-trusted-A03 versus
selected-A03 2x2 tables, and tool-call validity by model.

This is a finite synthetic benchmark. It does not identify hidden reasoning,
deception, a sampled model population, or physical-world harm.
