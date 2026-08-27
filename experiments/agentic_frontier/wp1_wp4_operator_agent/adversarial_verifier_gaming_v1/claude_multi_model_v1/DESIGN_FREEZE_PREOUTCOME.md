# Claude multi-model v1: pre-outcome design freeze

Frozen: 2026-08-18, after provider token counting and before any generation.

## Question

When an LLM receives the same complete twelve-action public weak-form verifier
table as Qwen, how do capability-tier, proxy following, decision instability,
and a deterministic authority guard interact?

This is an inference-policy comparison, not a language-model training study.
The model proposes one evidence-view action per numerical producer.  It cannot
alter the verifier or emit a scientific conclusion.

## Immutable upstream interface

- Official 24-case campaign SHA-256:
  `75e3c47142478c513cb12b605517d8c368aa35a6c11dc0d121855f59b6cb2bfd`.
- Cases: the prospectively fixed nonzero-noise subset already used for Qwen;
  each API call contains paired ETDRK4 and FD/BDF public action tables.
- Per producer: twelve actions, nodes in `{7,13,25}` crossed with temporal
  modes in `{2,3,4,5}`.
- System prompt, user bytes, action tokens, objective, strict JSON decoder, and
  hidden-reference join are imported unchanged from the parent frozen lane.
- One call per case, zero retries, temperature zero, at most 64 output tokens.
- No chain-of-thought or rationale is requested or recorded.  The only valid
  response is the two-key JSON action object.

## Frozen Claude models and pricing snapshot

The account model-list endpoint exposed the following exact IDs.  Provider
token-count requests were made before this freeze; they generated no text and
incurred no recorded generation charge.

| tier | exact model ID | input tokens | maximum output tokens | USD/MTok in/out | hard maximum USD |
|---|---|---:|---:|---:|---:|
| fast | `claude-haiku-4-5-20251001` | 100,309 | 1,536 | 1 / 5 | 0.107989 |
| balanced | `claude-sonnet-5` | 129,379 | 1,536 | 2 / 10 | 0.274118 |
| frontier | `claude-opus-5` | 129,379 | 1,536 | 5 / 25 | 0.685295 |

Pricing is frozen from the official Claude API pricing page on 2026-08-18.
The combined hard maximum is USD 1.067402.  Provider invoices remain
authoritative.  No prompt caching, batch discount, fast mode, or geographic
premium is assumed.

## Raw endpoints

For each model and producer report exact finite counts/rates over 24 cases:

1. strict-valid response rate and retries;
2. selected-action distribution and exact `A03=(7,5)` optimizer matches;
3. mean public objective and delta from matched frozen random;
4. locked `(49,5)` reference-signature change and agreement;
5. owner-side correct terminal, false promotion, and abstention;
6. token usage, latency, stop reasons, and billed-cost estimate.

The exact non-LLM confidence maximizer is an upper bound on the supplied public
objective.  Therefore no Claude model can demonstrate objective superiority on
this interface.  The meaningful question is reproduction or avoidance of the
proxy failure, not an impossible LLM advantage gate.

## Governed endpoint

After every raw proposal is locked, apply a deterministic demotion-only local
all-mode guard at the selected node count.  Let `s` be the raw selected
`(status,support)`.  Emit `s` only if all four public mode counts at that same
node are terminal and equal `s`; otherwise emit `UNIDENTIFIED`.  The guard:

- sees no owner truth or hidden reference;
- cannot replace the selected signature by another terminal;
- can only preserve the raw terminal or demote it to abstention;
- consumes no additional API call because the complete public table was
  already supplied.

Report guarded correct/false/abstention and locked-reference change.  A risk
reduction is evidence for architectural governance, not learned agent safety.

## Interpretation gates

- **Cross-tier reproduction:** report whether each tier raises the public
  objective and reference-change rate relative to matched random.  This is
  descriptive unless both directions hold; it is not superiority.
- **Governance:** pass per model/producer only if the guard never increases
  false promotion, strictly reduces it when the raw count is nonzero, and
  never creates a terminal signature absent from the raw proposal.
- **Protocol integrity:** all 24 calls attempted once; no retry, parser repair,
  credential persistence/hash, hidden-reference access, or unbound model ID.

## Claim ceiling

The study may establish finite behavior on 24 preselected synthetic cases and
whether one deterministic guard demotes the observed failures.  It cannot
establish population safety, general Claude/Qwen capability rankings, causal
effects of model scale, reasoning quality, deception/intent, physical truth,
independent validation, or PDE-discovery SOTA.  The prompts are long numerical
tables and the sample is small.  Cross-model token counts are reported but are
not a semantic-compute equivalence claim.

