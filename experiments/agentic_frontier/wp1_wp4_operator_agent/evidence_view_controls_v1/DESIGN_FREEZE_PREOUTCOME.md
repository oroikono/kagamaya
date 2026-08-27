# Evidence-view CPU controls v1 — pre-outcome design

Status: **DESIGN ONLY / RESERVED SEEDS CLOSED / NO OUTCOMES INSPECTED**

This isolated lane implements the three no-API controls named A10--A12 in
`REMAINING-9-DAYS.md`.  It does not modify the frozen verifier, predecessor
artifacts, or manuscript.  Any code or endpoint change after a reserved seed is
opened requires a new versioned lane.  All gates are interpretive reporting
rules, not tuning targets.

## Shared scientific ceiling and dependence unit

The 49-node decision is a locked numerical authority, not physical ground
truth.  The 97-node run is a reference-convergence control, not an independent
physical replication.  The spectral ETDRK4 and finite-difference/BDF producers
share family, parameter-seed, public-initial-condition seed, and noise-model
semantics; they are reported separately and never counted as independent
replicates.

The dependence cluster is the integer episode seed.  Every family, noise
model, noise level, resolution, and producer attached to one seed remains in
that cluster.  Descriptive row counts are reported, but uncertainty resamples
whole seed clusters.  Pseudorandom seeds instantiate the declared simulator
distribution only; no population or physical-world coverage claim follows.

All PDEs are the predecessor's nondimensional periodic controlled systems.
The candidate dictionary, five temporal weak modes, seven spatial modes,
complete 92-support enumeration, and admission thresholds are unchanged.

## Reserved, disjoint splits

The ranges were chosen and repository-scanned before outcome execution.

| Lane | Seeds | Clusters | Frozen use |
|---|---:|---:|---|
| A10 | 150000--150009 | 10 | ROC control only |
| A11 | 151000--151009 | 10 | high-noise sweep only |
| A12 | 152000--152009 | 10 | 49-to-97 convergence only |

No seed may cross lanes.  The earlier controlled campaigns used distinct seed
ranges, including 9600--10049 and several explicitly governed 70000--94199
ranges.  The exact v1 reservations above had no exact source/design occurrence
under this project before this file was created.  This is an internal
prospectivity record, not a third-party timestamp.

## A10 — residual threshold and convergence-slope ROC

Cross: six families, two noise models (`iid`, `correlated`), relative noise
`0`, `0.001`, `0.005`, two producers, and temporal nodes `7,13,25,49`, always
with five temporal modes.  Hyperdiffusion is the sole positive
outside-dictionary class; the other five families are negative.  Labels are
used only after diagnostics have been computed.

Truth-free scores, fixed before execution:

1. `residual_score = log10(max(best_residual_49, 1e-12))`;
2. `slope_score` is the ordinary-least-squares slope of
   `log10(max(best_residual_n,1e-12))` on `log10(n)` over all four nodes;
3. `composite_score = residual_score + slope_score`.

Higher scores predict outside-dictionary.  The primary endpoints are
producer-stratified AUROC for all three scores and their equal-weight producer
macro-average.  A deterministic 2,000-resample seed-cluster bootstrap, seed
20260820, supplies a descriptive 95% percentile interval for the macro-AUROC.
The frozen residual threshold `best_residual_49 >= 0.20` is also reported with
true-positive rate and specificity.  The full ROC and the empirical best balanced-
accuracy threshold are descriptive diagnostics only and may not be fed back
into the verifier.

Interpretation: residual triviality is supported only if **both** producers
have residual AUROC at least 0.99 and each admits some descriptive threshold
with true-positive rate and specificity both at least 0.95.  Failure strengthens the
paper's narrow answer to “why not just threshold residual?”  Passing falsifies
that non-triviality argument for this controlled binary task.  Slope/composite
results are reported regardless of direction.

## A11 — relative-noise sweep

Cross: six families, two noise models, relative noise `0.01`, `0.05`, `0.10`,
two producers, and the unchanged 49-node/five-mode authority.  The same
standardized latent noise innovation is scaled across levels within a matched
seed/family/model cell, so the dose-response comparison is paired.

Primary endpoints per producer and noise level are exact-truth correct-terminal
rate, false-promotion rate, abstention rate, and status counts.  Aggregate
difficulty is monotone only when correct-terminal rate is nonincreasing and
the sum of false-promotion plus abstention rates is nondecreasing across all
three levels, separately for each producer.  A level is called a
`controlled_valid_regime` only if, for each producer separately, false-
promotion rate is at most 1% and correct-terminal rate is at least 50%.  These
thresholds define a deliberately narrow simulator operating label; they do
not certify physical data or other noise laws.

Interpretation: a monotone loss of correct terminals establishes a difficulty
gradient.  If the 1% level already fails the controlled-valid rule, the paper's
existing <=0.5% regime must be stated as narrow.  Any nonmonotonicity is
reported, not repaired.

## A12 — nested 49-to-97 reference convergence

Cross: six families, two noise models, relative noise `0`, `0.001`, `0.005`,
and two producers.  Each producer creates one 97-node trajectory.  Its 49-node
view is the exact even-index nested subsample, including endpoints and shared
observed values.  Both decisions use five temporal modes and the unchanged
admission rule.  For spectral ETDRK4 the internal integrator step remains
0.00025 and the 97-node save interval is 0.00125.  For FD/BDF, spatial grid,
solver tolerances, and maximum step remain unchanged; only `t_eval` has 97
nodes.  The baseline channel is unchanged.

Primary endpoints per producer are exact `(status,support)` agreement,
correct-terminal rate at 49 and 97, false-promotion rate at 49 and 97, and
median relative weak-design and target differences.  The convergence gate
passes only if, separately for each producer:

1. exact signature agreement is at least 99%;
2. 97-node false-promotion rate is no more than 0.5 percentage points above
   the nested 49-node rate;
3. 97-node correct-terminal rate is no more than 1 percentage point below the
   nested 49-node rate.

Passing supports numerical stability of the 49-node authority within this
controlled hierarchy.  Failing falsifies the paper's use of 49 nodes as a
stable reference and must not be hidden by pooling producers.

## Fail-closed execution and artifacts

`PREOUTCOME_FREEZE.json` binds this design, implementation, validators, tests,
and all imported producer/weak-form authorities.  A run refuses missing or
mismatched hashes, wrong lane seeds, wrong noise grids, wrong node grids, or an
existing output path.  Results publish with exclusive creation and a canonical
SHA-256.  The separate validator recomputes source/freeze hashes, cardinality,
seed coverage, endpoints, summaries, and result digests.  Mutation tests must
fail on seed reuse, threshold drift, node-grid drift, label inversion, source
drift, and overwrite attempts before reserved seeds may open.

No post-hoc tuning and no manuscript edits are authorized in this lane.
