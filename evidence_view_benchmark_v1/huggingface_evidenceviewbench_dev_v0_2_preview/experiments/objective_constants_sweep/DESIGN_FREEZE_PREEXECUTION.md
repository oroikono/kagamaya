# Retrospective public-objective constants sweep

This is a zero-call, zero-simulation sensitivity analysis over already frozen
public view tables. It does not change verifier decisions. It only recomputes
which of the twelve already-serialized views maximizes the public objective.

## Bound inputs

- `cases/cases.jsonl`: SHA-256
  `cda68d3da3b74edc0f998e651f4d23c0c5c9add2d18ff55294db38db86451f6d`
- `cases/held_out_sidecar.jsonl`: SHA-256
  `66e048f1aed71764797eea7a6306de8fcc7e800857caffc6eed692f0d337aaa9`
- frozen objective implementation: SHA-256
  `d649be5324ba751f47344da1feca232f1235a9c5695975ff62c9d5e718aac600`

## Frozen sweep

Keep the score-gap scale `0.006`, condition scale `12`, terminal bonus `10`,
node penalty `0.001*n/49`, clipping interval `[-4,4]`, action table, terminal
statuses, and lexicographic tie break unchanged. Cross the identified-residual
scale `s_I=0.05*m_I` and omitted-residual scale `s_O=0.20*m_O`, where each
multiplier is `10^(k/4)` for integer `k=-8,...,8`. This is a 17x17 grid and
includes the frozen point `(0.05,0.20)` exactly.

For every case and numerical producer, choose the lexicographically first
action among exact objective maximizers using public rows only. Only after all
choices are materialized may the sidecar be joined for scoring.

Report separately by producer:

1. false-promotion rate over 180 in-library cases;
2. correct outside-library detection over 36 hyperdiffusion cases;
3. the complete trade-off table and nondominated frontier;
4. whether any grid point attains each descriptive operating region:
   `(FP<=1%, OOG>=75%)`, `(FP<=5%, OOG>=75%)`, and
   `(FP<=10%, OOG>=90%)`.

The analysis is retrospective and finite. It is not a population guarantee,
confirmation experiment, or proof covering constants outside the grid.
