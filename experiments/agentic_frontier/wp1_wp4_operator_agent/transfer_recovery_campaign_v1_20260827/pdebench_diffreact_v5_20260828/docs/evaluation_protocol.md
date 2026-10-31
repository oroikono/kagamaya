# Evaluation protocol

Development feasibility and audit admission require at least 95% complete-library correctness, zero complete-library false promotions, and at least 90% correct OUTSIDE_LIBRARY decisions separately for every omitted mechanism. No pooled pass may hide a failed omission contract.

V4 passed those gates on audit IDs 20--39. V5 does not tune further. On sealed
IDs 40--99 it reports the fixed reference, score-max attack, fixed fine view,
and exact four-mode agreement control. The transfer endpoint is descriptive:
false outcome counts and rates, coverage, false-identification versus
false-omission type, per-contract outcomes, paired attack/control outcomes, and
results conditioned on a correct fine reference. No post-outcome threshold,
view-bank, score, or control revision is permitted in v5.
