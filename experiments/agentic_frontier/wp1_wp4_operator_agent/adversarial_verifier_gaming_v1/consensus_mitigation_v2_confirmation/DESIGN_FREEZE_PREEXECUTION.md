# All-mode consensus mitigation: fresh confirmation freeze

Date frozen: 2026-08-18

## Selection history and question

Four public-only consensus rules were screened on seeds 10000--10049.  The
`all_mode_consensus` rule had the best empirical risk--coverage tradeoff, but
that screen is not confirmation.  This version freezes the selected rule and
reruns the complete simulator/verifier pipeline on untouched seeds
10200--10249.

For a fixed temporal-node count, the selected rule requires identical terminal
`(status,support)` across temporal test-mode counts 2, 3, 4 and 5.  It emits the
five-mode member of the consensus group with largest minimum public objective,
or abstains.  It sees only the twelve-row public verifier table.  Owner truth
and the locked `(49,5)` reference are joined after proposal serialization.

## Frozen campaign

- producers: pseudospectral ETDRK4 and centered-FD/BDF;
- seeds: 10200--10249, fifty clusters not used by the v1 screen;
- six PDE families, iid/correlated noise, relative noise 0, 0.001 and 0.005;
- 1,800 episodes per producer;
- policies: selected all-mode consensus, exact public confidence maximizer,
  and fixed `(25,5)`;
- the complete twelve-view public bank is available to every policy, so node
  count is a representation/witness diagnostic, not acquisition cost.

## Frozen finite gates, separately on both producers

The selected rule passes only if all conditions hold:

1. owner-labelled false-promotion rate is at most 1%;
2. public terminal coverage is at least 75%;
3. Goodhart-event rate (terminal public decision differing from the locked
   fine decision) is at most 1%;
4. owner-labelled false promotion is no greater than fixed `(25,5)`;
5. terminal coverage is no more than five percentage points below fixed
   `(25,5)`;
6. false promotion is at least fifty percentage points lower than
   confidence-max.

Literal reference-signature disagreement remains descriptive because an
abstention against a terminal reference is a disagreement, not a harmful
promotion.  Results are summarized by seed cluster and episode.  The two
producers are corroborating numerical components, not independent scientific
replicates.

## Claim ceiling

A pass establishes a finite fresh-seed mitigation result for this frozen
synthetic one-dimensional action table.  It is not a population safety bound,
physical validation, continuum guarantee, independent replication, universal
anti-Goodhart defense, or SOTA claim.  Owner truth is available only because
the benchmark is generated.  No threshold or policy may be retuned after any
10200--10249 outcome is opened.
