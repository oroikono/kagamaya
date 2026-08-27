# Sim2Science controls v1: pre-execution design freeze

Frozen on 2026-08-18 before any outcome from this lane was generated.

## Purpose

This lane attacks two possible alternative explanations for the reported
evidence-fidelity result.

1. **Manufactured-quadrature control.**  The large seven-node/five-test-bank
   failure may be ordinary composite-Simpson underintegration of oscillatory
   temporal weak tests rather than a new statistical interaction.  We therefore
   measure quadrature error and weak-identity defect for an analytic smooth
   trajectory at the exact node counts and temporal modes used by the paper.
2. **De-nested-bank control.**  The two-bank refinement gate compares a
   two-mode bank with a five-mode bank that contains the same first two modes.
   Agreement can therefore be mechanically correlated.  We compare that
   original nested pair with disjoint low- and high-frequency banks.

Both controls are mechanism and sensitivity tests.  Neither is an independent
physical validation, population guarantee, continuum certificate, or learned-
agent comparison.

## Control A: manufactured temporal quadrature

On the normalized time interval `s in [0,1]`, use

`u(s) = exp(0.7 s) + 0.2 sin(3 pi s)`

and the exact temporal test family used in the authority,

`eta_m(s) = sin(m pi s)`, `m = 1,...,5`.

For node counts `7, 13, 25, 49, 97`, composite Simpson quadrature is applied to
both sides of the integration-by-parts identity

`- integral u eta'_m ds = integral u' eta_m ds`.

The reference values are computed by adaptive Gauss--Kronrod quadrature with
absolute and relative tolerances `1e-13`.  Frozen endpoints are:

- relative error of the left weak target;
- relative error of the right weak term;
- absolute discrete weak-identity defect;
- observed refinement order between nested node counts;
- at each node count, the error ratio for mode 5 versus mode 2.

No monotonicity or fourth-order gate is imposed.  A measured fourth-order trend
will be described as compatibility with classical Simpson behavior, never as a
proof that this scalar control explains every PDE decision.

## Control B: disjoint-bank confirmation

The new, previously unused hidden-seed block is `62000--62019`.  It crosses:

- six frozen PDE families;
- IID and correlated observation noise;
- relative noise `0`, `0.001`, and `0.005`;
- exact nested temporal views with `7, 13, 25, 49` nodes.

This yields 120 distinct hidden family--seed trajectory realizations and 720
factorial observation conditions.  The latter, not 720 independent
trajectories, is the denominator for decision counts.

At each temporal view, three weak-test banks are evaluated with the unchanged
92-support authority and unchanged admission thresholds:

- `low`: modes `{1,2}`;
- `cumulative`: modes `{1,2,3,4,5}` (the original nested comparator);
- `high`: modes `{3,4,5}`.

The **nested gate** refines until `low` and `cumulative` emit the same terminal
status/support.  The **disjoint gate** applies the identical rule to `low` and
`high`.  If no agreement occurs at 49 nodes, the gate returns `UNIDENTIFIED`.
The gate receives only serialized bank outputs; family, seed, coefficients,
truth, future views, and owner labels are forbidden inputs.

### Frozen endpoints

For each gate and fixed fine comparator:

- owner-correct terminals;
- owner-false terminals;
- abstentions;
- mean terminal node count;
- stop-node histogram;
- pairwise transitions between the nested and disjoint gates;
- family/noise strata.

### Frozen interpretation gates

The disjoint gate is called a usable de-nesting control only if all conditions
hold on the 720 factorial observation conditions:

1. owner-false terminal rate at most 1%;
2. correct-terminal rate no more than five percentage points below the nested
   comparator;
3. mean terminal node count below 30;
4. no more than five new false terminals relative to the nested comparator;
5. every family loses at most ten percentage points of correct resolution
   relative to the nested comparator.

Failure is retained and blocks any claim that de-nesting validates the original
agreement rule.  Passing supports only a finite sensitivity result on this
reserved simulator block.

## Governance and reporting

The design file, executable source, raw compact result, validator report, and
content hashes are retained.  Counts are structured finite summaries; the 720
factorial conditions share 120 hidden trajectories and are not asserted IID.
Governance remains `audit_only/not_certified` unless an independent or formal
witness is later admitted.

