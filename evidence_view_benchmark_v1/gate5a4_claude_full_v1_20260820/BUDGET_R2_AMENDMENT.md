# Budget-only R2 amendment before full execution

The validated six-call transport smoke spent USD 0.107472. The initial full-run
implementation instantiated a fresh USD-100 ledger and therefore did not formally carry
that smoke spend into the single-grant cap. No full campaign call or owner-side score had
run when this was found.

R2 adds a required nonnegative `--prior-spend-usd` input, initializes the ledger at that
amount, and reports prior plus total grant consumption. The full run is invoked with
`--prior-spend-usd 0.107472`. No prompt, schema, model, case, action, endpoint, or provider
transport changes. The validated smoke remains a transport witness under the R1 closure;
the full campaign is bound to a new R2 source closure.
