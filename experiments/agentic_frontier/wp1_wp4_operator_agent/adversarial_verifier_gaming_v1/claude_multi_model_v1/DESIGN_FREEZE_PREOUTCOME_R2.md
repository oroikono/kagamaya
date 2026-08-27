# Claude multi-model R2: matched decoding repair

Frozen: 2026-08-18 after the documented Sonnet 5 HTTP-400 compatibility
failure, before any Sonnet 4.6 or Opus 4.6 generation.

R2 preserves every v1 scientific interface choice: the exact 24 prompts,
twelve actions per producer, objective, strict JSON response, zero retries,
64-token cap, temperature zero, owner/reference isolation, raw endpoints, and
demotion-only local consensus guard.  It changes only the two model IDs that
had not produced scientific outcomes.

| tier | exact model ID | input tokens | maximum output | USD/MTok in/out | hard maximum USD | authorization cap |
|---|---|---:|---:|---:|---:|---:|
| fast, inherited completed arm | `claude-haiku-4-5-20251001` | 100,309 | 1,536 | 1 / 5 | 0.107989 | 0.12 |
| balanced, R2 | `claude-sonnet-4-6` | 100,333 | 1,536 | 3 / 15 | 0.324039 | 0.35 |
| frontier, R2 | `claude-opus-4-6` | 100,333 | 1,536 | 5 / 25 | 0.540065 | 0.57 |

The combined hard maximum is USD 0.972093.  The Haiku arm is not rerun.
Sonnet 5 remains an explicit protocol-incompatibility result; Opus 5 remains
unexecuted.  No cross-version claim is allowed beyond these three exact model
IDs under the shared temperature-zero interface.

All interpretation gates and the claim ceiling in
`DESIGN_FREEZE_PREOUTCOME.md` remain binding.  In particular, this experiment
cannot identify a causal effect of model scale or measure chain-of-thought.

