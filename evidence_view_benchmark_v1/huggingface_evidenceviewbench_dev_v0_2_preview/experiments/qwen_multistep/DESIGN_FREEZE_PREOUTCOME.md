# Metadata-blind evidence intervention — pre-outcome design

Status: **SOURCE_FROZEN; no v2 model outcome opened at freeze time**.

## Question

Does a language-model policy's second acquisition decision actually respond to
the scientific evidence it purchased, or can the apparent response be explained
by episode metadata and reward wording?

This is a bounded control over the frozen 216-case synthetic benchmark. It does
not establish general agent competence, physical validity, or population risk.

## Metadata-blind interface

The initial prompt contains only the goal, 16 action geometries, node costs, and
the budget. It contains no case identifier, family identifier, seed, noise label,
public outcome table, or owner label. Because Qwen uses greedy decoding, the one
initial generation for each goal is reused over all cases rather than treating
648 identical prompts as distinct evidence.

The second prompt contains the same goal, the same first choices, the acquired
scout observations, the fixed challenge map, and the action semantics. It again
contains no episode metadata or answer key.

## Observation intervention

Each case-goal pair is evaluated twice:

1. `FACTUAL`: the second prompt receives that case's acquired observations.
2. `PERMUTED`: it receives the complete coherent observation bundle from the
   case 37 positions later in lexicographically sorted public case order.

The shift 37 is frozen before outcomes, is coprime to 216, is a derangement and
uses no outcome or owner label. The chosen second decision is always executed
on the original case. Thus the paired prompts differ only in the evidence
bundle. This identifies finite prompt-level evidence sensitivity; it does not
prove semantic understanding or transfer.

## Model and calls

- `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Greedy decoding, 96 output tokens, no retry or response repair.
- Three initial calls plus `216 × 3 × 2 = 1,296` second-decision calls.
- A protocol-invalid initial action blocks the campaign before any episode is
  executed; a protocol-invalid second response fails closed to `ABSTAIN` for
  that case. Responses are never repaired or retried.
- No paid API call is authorized.

## Frozen endpoints and gates

- V0 integrity: source, prompt, model, response, case, intervention and public
  transition bindings replay exactly.
- V1 metadata blindness: all prompt constructors pass the exact forbidden-key
  checks and initial prompts are identical across episodes within a goal.
- V2 evidence sensitivity: at least one goal changes at least 10% of paired
  producer decisions under the frozen evidence permutation.
- V3 useful evidence response: factual `correct - false` exceeds permuted
  `correct - false` over the finite paired table.

Failure of any gate is retained as a negative result. No threshold is changed
after outcomes.

## Claim ceiling

The challenge is **proposer-independent**, not statistically independent: both
views reuse the same trajectory and overlapping temporal test functions. The
case bank has one outside-library mechanism, all simulations are one-dimensional
and periodic, and both producers share scientific lineage. Rates are finite
execution summaries, not IID estimates or risk guarantees.

## Public release boundary

The current repository contains full transition tables and owner sidecars for
research replay. Those files must not be uploaded as a hidden-test leaderboard.
A release builder must publish full labelled development cases separately from
an opaque test manifest; hidden test transitions and owner labels remain on the
evaluation service.
