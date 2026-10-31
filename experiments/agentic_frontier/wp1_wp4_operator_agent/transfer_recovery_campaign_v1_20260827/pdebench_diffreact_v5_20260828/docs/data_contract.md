# Data contract

Only source IDs 0--19 may be read during development and 20--39 during audit. IDs 40--99 are sealed confirmation; 100--999 are outside this experiment. Noise is one deterministic IID perturbation per field, source, and level, scaled by that field's standard deviation and shared by every view and contract.
