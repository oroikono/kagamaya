# Independent finite-difference producer v1: pre-execution freeze

Frozen on 2026-08-14 before any outcome from seeds 9600--9619 was generated
with the finite-difference producer defined in this lane.

## Question

Does the temporal-resolution failure and its progressive-fidelity repair persist
when the outcome trajectories are produced by a different numerical method from
the Fourier pseudospectral ETDRK4 producer used in the original campaign?

This is a cross-discretization robustness check inside one repository. It is not
independent scientific validation, third-party replication, or a physical-data
result.

## Frozen producer

- Spatial discretization: method of lines with periodic, centered nine-point
  finite-difference stencils. Weights are generated from polynomial moment
  equations for derivatives one through four; no Fourier differentiation or
  de-aliasing is used by the outcome producer.
- Time integration: `scipy.integrate.solve_ivp(method="BDF")`, relative
  tolerance `1e-8`, absolute tolerance `1e-10`, maximum internal step `0.001`.
- An analytic method-of-lines Jacobian is supplied for each family.
- Spatial grids, horizons, saved-node grids, coefficient ranges, initial
  conditions, and noise semantics match the original controlled design:
  baseline `(nx=64, horizon=0.24, 25 saved nodes)` and gradient-rich channel
  `(nx=96, horizon=0.12, 49 saved nodes)`.
- The 7, 13, and 25-node channel views are exact nested subsamples of the one
  49-node noisy realization.
- The weak-form evidence builder, frozen 92-support manifest, exhaustive
  enumerator, and residual/gap/conditioning admission gate are unchanged public
  consumers. They receive only observed `(x,t,u)` trajectories.

## Frozen population and cells

- seeds: **9600--9619**, unused by the earlier development and confirmation
  campaigns;
- families: advection--diffusion, viscous Burgers, Fisher--KPP, Allen--Cahn,
  KdV, and omitted hyperdiffusion;
- noise models: iid and correlated;
- relative noise levels: 0, 0.001, and 0.005;
- temporal channel nodes: 7, 13, 25, 49;
- temporal weak test modes: 2 and 5.

There are 720 hidden episodes and 5,760 factorial verifier cells. Clean solves
may be cached within a `(family, seed)` episode before noise is added; this is a
computational optimization and must not alter any cell semantics.

## Frozen endpoints

Primary endpoints at five temporal modes:

1. false terminal promotions at 7 and 49 nodes and their paired difference;
2. correct terminal outcomes at 7 and 49 nodes;
3. whether the frozen progressive ladder (7, 13, 25, 49; stop only on
   `IDENTIFIED`, otherwise refine, and return the 49-node status unchanged) has
   no more false promotions and no fewer correct outcomes than fixed 49 nodes,
   while using fewer than 49 nodes on average.

Secondary endpoints are the full resolution curves, exact paired sign test,
status/support transitions, family/noise strata, and finite-difference-versus-
analytic error for the linear advection--diffusion unit problem.

## Pre-execution gates

Before any seed 9600--9619 is submitted:

- all finite-difference derivative tests on periodic sine waves must pass their
  predeclared tolerances;
- the analytic advection--diffusion comparison must have relative Frobenius
  error at most `2e-4` on both producer grids;
- nested subsampling and shared-node noise identity tests must pass;
- a mutation that reverses the advective sign must be detected;
- a one-seed smoke campaign using seed 9590 must validate structurally and
  contain all 288 expected cells.

If a gate fails, the confirmation campaign is not launched until the
implementation—not the frozen estimand—is repaired and all gates are rerun.

## Interpretation and claim ceiling

Persistence under this producer supports only a finite cross-discretization
robustness statement for these six manufactured families, grids, noises, and
the unchanged verifier. Non-persistence is retained and localized rather than
hidden. Neither result establishes solver invariance, population calibration,
physical validity, open-world completeness, or a general theorem. Because the
producer and evaluator remain authored and executed in the same repository,
governance remains `audit_only/not_certified` unless a genuinely independent or
formal witness is later admitted.
