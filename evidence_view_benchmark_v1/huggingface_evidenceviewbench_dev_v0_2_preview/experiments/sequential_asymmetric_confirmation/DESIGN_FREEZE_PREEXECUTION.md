# Sequential asymmetric controller: untouched 50-seed confirmation

Status: **prospective confirmation design**.  No confirmation outcome is open
when this document and its source closure are frozen.

## Eligibility and immutable interface

The mechanically repaired 20-seed calibration passed every frozen gate after
complete replay.  This confirmation binds that verdict by file hash and uses
the unchanged v2 controller, views, simulator calls, comparators, score
definitions, and gates.  Nothing is refit or retuned.

- Seeds: integers 10500--10549 inclusive, previously reserved and untouched.
- Episodes per producer: 50 seeds x 6 PDE families x 2 noise models x 3
  relative-noise levels = 1,800.
- Producers: spectral ETDRK4 and centered-FD/BDF, treated as paired numerical
  components rather than independent scientific replications.
- Policies: sequential all-mode controller, full-bank all-mode consensus,
  fixed `(25,5)`, fixed `(49,5)`, confidence maximum, D-optimal, and E-optimal.
- One source-bound shard per `(producer, seed)`.
- Complete serialized replay and deterministic numerical-origin regeneration.

Seed is the dependence cluster.  Every universal quantifier below ranges only
over this enumerated finite artifact, not a continuum or unseen population.

## Frozen confirmation gates (each producer separately)

1. Sequential false-promotion count is zero.
2. Sequential terminal and owner-correct rates are each at least 75%.
3. Sequential correct rate is within 5 percentage points of full-bank
   consensus.
4. Sequential false count is no higher than fixed `(25,5)` or fixed `(49,5)`.
5. Sequential false rate is at least 50 percentage points below confidence
   maximum.
6. Mean views are at most 12 and mean maximum nodes are at most 30.
7. Sequential owner-correct rate is at least 50% in every PDE family.
8. Sequential terminal coverage is at least 50% separately at relative noise
   0.001 and 0.005.
9. Every source/hash/serialized/numerical-origin replay check passes.
10. The exact 50-seed confirmation split is used with no seed outside it.

Primary estimates are finite counts/rates and seed-cluster summaries.  We also
report deterministic seed-cluster bootstrap intervals for correct rate, false
rate, and paired differences, but do not use repeated family/noise rows as IID
Bernoulli trials.  Zero observed false promotions is not a universal or
population guarantee.

## Claim ceiling

Even a pass is confirmation only for the locked one-dimensional synthetic
finite environment.  It is not physical validation, independent replication,
open-world completeness, continuum certification, or a general theorem about
scientific agents.  Two-dimensional, external-origin, and physical-observable
transfer require separate prospective contracts.
