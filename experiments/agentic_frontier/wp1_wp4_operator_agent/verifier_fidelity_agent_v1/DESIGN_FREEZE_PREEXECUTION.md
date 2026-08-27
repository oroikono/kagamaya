# Verification-fidelity agent v1 — prospective confirmation freeze

## Research question

Can a truth-blind controller spend substantially less temporal-resolution
budget while reproducing the terminal decision of the frozen 49-node weak-form
verifier, without increasing false scientific promotions?

This tests a distinct action in agentic science: choosing **how faithfully to
measure/verify evidence**, rather than only which hypothesis or forcing to try.
The endpoint is verifier fidelity, not raw trajectory error.

## Why this is a confirmation rather than a retuned development result

The stopping rule was selected from the already archived 20-seed spectral and
FD/BDF development corpora. This document freezes it before opening seeds
9700--9749. No threshold is fitted on the confirmation split.

## Producers and matched episodes

Two numerical producer components are evaluated separately:

1. pseudospectral ETDRK4;
2. centered nine-point finite differences with SciPy BDF time integration.

Each producer uses the same six PDE families, two noise models, three noise
levels, 92-support finite grammar, weak-form verifier, and nested temporal views
with 7, 13, 25, and 49 nodes. The two producer components are corroborating
implementations in the same repository, not independent scientific replicas.

Confirmation seeds are 9700--9749. Every seed is crossed with all family,
noise-model, and noise-level cells: 1,800 matched episodes per producer.

## Frozen controller

For five temporal test modes, inspect resolutions in the order
7 -> 13 -> 25 -> 49. Stop at the first resolution whose gated status is
`IDENTIFIED`. If no earlier resolution is identified, return the 49-node
decision. In particular, a coarse `OUT_OF_GRAMMAR` result never authorizes
early stopping.

The controller reads only public terminal status/support and node count. Owner
truth is used only after selection for evaluation.

## Baselines

- fixed 7, 13, 25, and 49 nodes;
- unsafe first-terminal stopping (`IDENTIFIED` or `OUT_OF_GRAMMAR`);
- two-resolution status/support stability;
- retrospective cheapest resolution matching the 49-node decision.

Fixed 25 nodes is the principal matched-cost comparator. Fixed 49 is the
reference verifier, not a competitor.

## Primary endpoints

For each producer separately:

- exact terminal signature agreement with 49 nodes, where signature is
  `(status, support)`;
- owner-labelled correct resolution and false promotion;
- mean terminal node count;
- terminal-node histogram;
- paired adaptive-versus-fixed-25 signature wins/losses.

## Prospective success gate

The confirmation supports the positive finite claim only if, for **both**
producer components:

1. adaptive signature agreement with 49 nodes is at least 99.5%;
2. adaptive false-promotion rate is no more than 0.2 percentage points above
   fixed 49;
3. mean terminal nodes are at most 30;
4. adaptive signature agreement exceeds fixed 25 by at least two percentage
   points;
5. adaptive mean nodes are no more than three nodes above fixed 25;
6. no hidden-truth mutation can change the selected node or terminal signature.

Failure of any gate blocks the positive claim. No seed, threshold, or stopping
rule will be changed within v1.

## Claim ceiling

Passing establishes a finite, two-implementation confirmation that a frozen
truth-blind fidelity controller can reproduce the expensive verifier endpoint
at lower temporal-node cost. It does not prove continuum correctness, physical
cost equivalence, population risk control, numerical independence, or general
optimality. The 49-node decision may itself be scientifically wrong.
