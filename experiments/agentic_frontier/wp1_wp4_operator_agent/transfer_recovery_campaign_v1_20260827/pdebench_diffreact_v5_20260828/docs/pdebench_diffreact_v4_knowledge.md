# PDEBench diffusion-reaction v4 knowledge

The public simulation contains two coupled fields configured on `[-1,1]^2`
with 512 spatial points per axis and homogeneous Neumann boundaries. The
published 128-point HDF5 coordinate arrays are the exact uniform decimation
`-1+(8*j+3)/512`, `j=0,...,127`, with spacing `2/128`.

Development IDs 0--19 selected three omission contracts that passed the
predeclared 90% reference-authority gate: `v` diffusion, `u`-to-`v` coupling,
and the grouped `u` local source block (`u` and constant forcing). This third
contract must always be called a grouped mechanism omission. Audit IDs 20--39
were not used during contract selection.

The source is independently generated simulated data. It is not physical
experimental validation and does not provide an independent evaluator.
