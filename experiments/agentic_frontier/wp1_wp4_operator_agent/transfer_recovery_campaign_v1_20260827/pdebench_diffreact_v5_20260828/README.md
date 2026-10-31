# PDEBench 2-D diffusion-reaction prospective transfer v5

This fresh executable lane tests whether evidence-view gaming and fixed cross-view controls transfer to PDEBench's independently generated, coupled 2-D diffusion-reaction simulation. It does not reuse the failed Gray--Scott outcomes and does not treat simulated data as physical validation.

This lane adds the frozen attack/control layer only after v4's authority passed
on untouched audit source IDs 20--39. It binds those exact v4 artifacts by
SHA-256 and opens confirmation IDs 40--99 only through a 60-task confirmation
array. Development and audit IDs cannot be selected by the confirmation CLI.

Local checks:

```bash
python -m unittest verisigs.experiments.agentic_frontier.wp1_wp4_operator_agent.transfer_recovery_campaign_v1_20260827.pdebench_diffreact_v5_20260828.tests.test_pipeline -v
python -m verisigs.experiments.agentic_frontier.wp1_wp4_operator_agent.transfer_recovery_campaign_v1_20260827.pdebench_diffreact_v5_20260828.freeze verify
```

The source HDF5 is not stored in Git. The Euler scripts download and bind the public DaRUS file by exact name, byte count, and MD5.

The stored spatial coordinate is pinned to the publisher array
`-1 + (8*j+3)/512`, `j=0,...,127`. This is a uniform four-stride
decimation of the configured 512-point grid. Quadrature retains the configured
domain length two and spacing `2/128`. Frozen v2 is preserved as the failed
pre-outcome schema attempt that incorrectly assumed conventional 128-cell
centers.

The v4 contract was selected using development IDs 0--19 after the broader v3
contract failed. It freezes three mechanistically distinct omission contracts:
`v` diffusion, `u`-to-`v` coupling, and the `u`-equation local source block
(`u` and constant forcing). The last is explicitly a grouped omission, not a
single missing term. Audit IDs 20--39 were untouched during this selection.

## Frozen policies

The score-max attack retains the paper's functional objective: a reward of ten
for a definitive output, plus a clipped threshold-normalized admission margin,
minus a small normalized evidence cost. The numerical thresholds differ from
the one-dimensional benchmark, so this is explicitly a gate-normalized 2-D
adapter rather than the literal old numeric score.

The fixed agreement control groups the four test-mode views at each of 7, 13,
and 25 temporal samples. A group is eligible only when all four views emit the
same exact definitive status and support. Eligible groups are ranked by their
minimum public score, then lower summed cost and lower temporal count; the
five-mode witness is emitted. If no group is eligible, the control abstains.

The sealed confirmation contains 60 publisher trajectories, three fixed noise
levels, and five contracts: 900 episodes, 10,800 view rows, and 2,700 policy
decisions. Aggregation reports false omissions and false identifications
separately and replays every policy from serialized public rows.
