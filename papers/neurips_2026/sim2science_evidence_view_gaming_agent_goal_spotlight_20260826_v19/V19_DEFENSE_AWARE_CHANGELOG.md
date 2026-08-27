# v19 defense-aware result pass

- Added the defense-aware exhaustive group search to Results and Appendix F.8.
- Recomputed the result from all 100 raw confirmation shards: three
  agreement-confirmed false omissions and zero agreement-confirmed false
  identifications per implementation; exhaustive routing finds no additional
  agreeing-wrong condition.
- Disclosed that four individual A00 rows across paired implementations do
  yield a wrong identified support, but disagreement blocks every one.
- Scoped the exhaustive claim to the twelve frozen rows and explicitly denied
  a security guarantee.
- Renamed Table 1's ambiguous `kappa` column to `Fit cond.`; the table values
  are fitted-support condition numbers (1.528), while the caption separately
  reports full Gram conditions (7,234 and 7,730).
- Repaired the trust-report/action denominator wording.
- Preserved the five-page body without reducing font or figure size.
- Added a standard-library-only reproduction script and bound its output in
  `verify_manuscript_claims.py`.
