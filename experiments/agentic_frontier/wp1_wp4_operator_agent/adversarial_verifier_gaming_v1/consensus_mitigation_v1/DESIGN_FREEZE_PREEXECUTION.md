# Cross-view consensus mitigation: development freeze

Date: 2026-08-18

## Question

Can a public-only cross-view consistency rule reduce verifier gaming without
consulting the hidden `(49,5)` reference or owner truth?

## Frozen source

The experiment replays the already locked 50-seed ETDRK4 and FD/BDF public and
owner campaigns. It is a post-confirmation exploratory mitigation screen, not
new confirmation evidence. Policy construction receives only the twelve-row
public table. Owner/reference data are joined only after proposal serialization.

## Policies

1. `adjacent_node_consensus`: for a fixed temporal-mode count, require equal
   terminal `(status,support)` across either `(7,13)` or `(13,25)` nodes;
   select the richer member of the pair maximizing the pair's minimum public
   objective. Abstain if no terminal-consistent pair exists.
2. `all_node_consensus`: for a fixed temporal-mode count, require equal terminal
   signatures at 7, 13 and 25 nodes; select the 25-node member maximizing the
   minimum public objective. Abstain if none exists.
3. `all_mode_consensus`: for a fixed node count, require equal terminal
   signatures for temporal modes 2, 3, 4 and 5; select the 5-mode member
   maximizing the minimum public objective. Abstain if none exists.
4. `two_axis_consensus`: require one terminal signature to occur at at least two
   node counts and two temporal-mode counts; select its cheapest occurrence,
   breaking ties by larger public objective then token. Abstain if none exists.

## Accounting and gates

All consensus policies receive the already-computed twelve-row public bank,
whose maximum fidelity is 25 nodes. Therefore this replay makes no acquisition-
cost-saving claim. `consensus_witness_nodes` is only the largest node count in
the subset witnessing a selected consensus. A policy that abstains has inspected
the full bank and is accounted at 25 nodes. Abstention receives the frozen
unconditional public objective.

`reference_change_rate` is the literal rate at which emitted terminal
`(status,support)` differs from the locked reference. It is not the narrower
Goodhart-event rate; consequently, abstaining against a terminal reference is
counted as disagreement. The Goodhart rate is retained as a secondary endpoint.

A useful mitigation must, on both producers:

- reduce the locked-reference-change rate by at least 50 percentage points
  relative to confidence-max;
- have no more than one percentage point greater reference-change than frozen
  fixed `(25,5)`;
- retain at least 75% public-terminal rate.

Witness-node summaries are descriptive and must not be presented as data-
acquisition savings because the complete public bank was available to every
policy.

Multiple policies are a development screen. No family may be promoted to a
confirmatory claim without a fresh frozen campaign and multiplicity-aware
selection.
