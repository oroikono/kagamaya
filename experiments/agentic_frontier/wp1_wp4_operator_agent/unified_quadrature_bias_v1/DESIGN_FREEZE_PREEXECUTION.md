# Unified quadrature-bias v1: pre-execution design freeze

Frozen on 2026-08-14 before any result from this lane was generated.

## Question

The unified-v2.1 development pilot found that the `gradient_space` acquisition,
which saved only seven temporal nodes, produced terminal but wrong scientific
statuses in 100/120 development episodes.  This experiment asks whether that
failure is explained by temporal weak-form quadrature/discretization bias rather
than by the spatial excitation itself.

This is a controlled simulator intervention, not a causal claim about physical
experiments in general.

## Frozen factors and controls

For each hidden PDE episode, generate one finest-grid `gradient_space`
trajectory with 49 saved temporal nodes.  Add one finest-grid noise realization
and obtain every coarser observation by exact nested subsampling.  Thus the PDE
coefficients, initial condition, spatial grid, excitation, latent trajectory,
noise realization at shared nodes, support grammar, action cost, and scientific
admission gate are held fixed.

- temporal nodes: **7, 13, 25, 49**;
- temporal test-function count: **2, 5**;
- quadrature rule: composite Simpson throughout;
- spatial grid: 96 periodic nodes;
- horizon: 0.12;
- action: `gradient_space`;
- intervention cost: 1.0 in every cell;
- discovery evidence: fixed baseline trajectory plus the varied channel;
- grammar: the same 92 supports of size one to three over the eight-term
  operator library used in unified v2.1;
- admission gate: residual <= 0.050, score gap >= 0.006, normalized condition
  number <= 12.0; out-of-grammar residual >= 0.20.

The fixed baseline trajectory is identical across all eight cells of an
episode.  The channel-only weak-form design and target are also compared with
the matched 49-node reference at the same temporal-mode count.

## Development population

- seeds: 4100--4119;
- families: advection--diffusion, viscous Burgers, Fisher--KPP, Allen--Cahn,
  KdV, and omitted hyperdiffusion;
- noise models: iid and correlated;
- relative noise levels: 0, 0.001, and 0.005.

This gives 720 matched hidden episodes and 5,760 factorial cells.  Seeds
9500--9999 are reserved and must not be executed by this development lane.

## Frozen endpoints

Primary endpoint, evaluated at five temporal modes:

- paired change in wrong-support promotion from 7 to 49 saved nodes.

Secondary endpoints:

- correct resolution and abstention;
- false-promotion discordance table and exact paired sign test;
- wrong-at-seven episodes that become correct, abstain, or remain wrong at 49;
- support/status changes under refinement;
- best validation residual, score gap, and normalized Gram condition;
- channel-only weak-form design and target error relative to the 49-node cell;
- interaction between temporal-node count and two versus five temporal modes;
- family-, noise-model-, and noise-level-stratified rates.

No endpoint may be silently replaced by an easier post-hoc metric.

## Interpretation gates

The contrast is called nondegenerate only if the seven-node/five-mode cell has
at least ten false promotions and at least ten false-promotion-discordant pairs
against the matched 49-node cell.  A reduction under refinement is evidence
for a simulator-specific discretization mechanism only if weak-evidence error
also decreases and the refined cell does not create a compensating false-
promotion increase.

Failure to meet these gates is retained as a negative result.  Passing the
gates does not establish population calibration, physical validity, adaptive
agent advantage, or a general causal theorem.

## Execution and claim ceiling

This is a disjoint **development** campaign.  It is neither confirmation nor
independent validation.  The producer and evaluator remain in one repository.
The governance state is audit-only unless an independent or formal witness is
later admitted.  A bounded Euler calibration is permitted only if local tests,
mutation tests, the validator, and the nondegeneracy gate pass; it is not needed
when the frozen 720-episode local campaign completes reliably.

