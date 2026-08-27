# High-noise attack-margin control (v1)

## Question

At noise levels where the locked 49-node, five-mode numerical authority is
already fallible, does maximizing its public confidence proxy still increase
false promotion relative to that same authority and to simple defenses?

This is a prospective matched control prompted by the A11 result. It may
strengthen, weaken, or falsify the attack narrative. No outcome-dependent
retuning is permitted.

## Frozen population and dependence

- Fresh seed clusters: integers 153000--153019, disjoint from A10--A12 and the
  original verifier-gaming confirmation.
- Six frozen one-dimensional PDE families.
- Noise models: iid and correlated.
- Relative noise levels: 0.01, 0.05, 0.10.
- Numerical producers: spectral ETDRK4 and centered-FD/BDF. They are paired
  corroborating components with shared task/authority lineage, not independent
  scientific replications.
- Dependence unit: the seed cluster. Producers, families, noise models, levels,
  views, and policies within a seed are matched repeated measurements.

There are 720 owner-scored episodes per producer and 1,440 total. Each public
episode contains the unchanged twelve views: nodes in {7,13,25} crossed with
temporal modes in {2,3,4,5}. The locked comparison authority is (49,5).

## Frozen policies

1. `reference_49_m5`: the locked 49-node, five-mode authority.
2. `fixed_25_m5`: public action A11, a strong non-adaptive baseline.
3. `confidence_max`: exact canonical maximizer of the frozen public objective
   over all twelve public views; this is the attack policy.
4. `all_mode_consensus`: a pre-existing public-only defense. For each node
   count, all four temporal-mode terminals must agree; otherwise it abstains.

Policy choices are serialized before the owner-side truth/reference join. The
confidence policy sees no truth, family label, or 49-node outcome. Invalid or
empty consensus proposals are scored as abstentions, never dropped.

## Endpoints

Primary, separately for each producer at 1% noise:

- correct, false-promotion, and abstention counts/rates for every policy;
- matched attack amplification
  `FP(confidence_max) - FP(reference_49_m5)`;
- paired discordances: attack-only false versus reference-only false;
- a descriptive 2,000-replicate seed-cluster bootstrap interval for the
  matched rate difference.

Secondary: the same endpoints at 5% and 10%, family/noise-model strata,
selected-action distributions, and the consensus tradeoff.

No universal-validity, population guarantee, independent-replication,
continuum, physical, or SOTA claim follows. Relative noise is specific to this
synthetic observation model and is not directly interchangeable with noise
percentages in other PDE-discovery papers.

## Interpretation fixed before outcomes

- Positive amplification: the attack remains live despite a fallible baseline.
- Near-zero amplification: the headline attack margin collapses in this regime.
- Negative amplification: the public maximizer is safer than the locked
  reference here, directly falsifying the proposed high-noise extension.
- Consensus may trade correctness for safety; abstentions count explicitly.

All outcomes are reported. Producers are never pooled as independent trials.
Node count remains a numerical-fidelity proxy, not a physical acquisition cost.
