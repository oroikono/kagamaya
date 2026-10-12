# GitHub and Hugging Face artifacts

Archive scope reviewed on 12 October 2026. The revisions below are observed
snapshots, not interchangeable code and dataset commits. Private access is
required; these links do not authorize redistribution or imply that development
artifacts reproduce the latest manuscript.

## Paper source and historical reproduction

The paper was accepted at [Sim2Science](https://www.sim2science.com/), a NeurIPS
2026 workshop. The author-supplied decision records acceptance on 29 September
2026. The [official OpenReview venue](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/Sim2Sci)
is linked from the workshop's call for papers. A direct paper/forum URL has not
yet been verified; neither venue link is presented as a paper PDF link.

[GitHub paper repository](https://github.com/oroikono/kagamaya)
is private. Its historical default-branch release contains the v19 manuscript
and a 33-check stored-record recount. It is not the current camera-ready source.
On 8 October 2026, [PR #2](https://github.com/oroikono/kagamaya/pull/2) merged
the reproduction exit-status fix and documentation updates. It did not change
the numerical calculations or archived scientific records.

## Related development benchmark

[EvidenceViewBench-Dev](https://huggingface.co/datasets/oroikono/EvidenceViewBench-Dev)
is an existing **private dataset repository**, not a model or public demo.
Its reviewed metadata revision is `3b2a3bf003c5f6ff6ff760eeca478575afdadfd6`.
An authenticated reader can inspect the
[card at that revision](https://huggingface.co/datasets/oroikono/EvidenceViewBench-Dev/blob/3b2a3bf003c5f6ff6ff760eeca478575afdadfd6/README.md).
The card describes synthetic development data: 216 exposed records from six
seed clusters, 180 unique evidence tables, twelve initially available views and
four added fine views. All labels are exposed; there is no hidden test split.
Here, “public” in the card means selector-visible, not publicly accessible on
the Hub. Six seed clusters are the dependence unit, not 216 independent
trajectories. Those sixteen available views are not the paper's frozen
twelve-action bank. Native Hugging Face `datasets` loading was unverified in the
12 October audit; the explicit-file loader was checked. The archive is not a
byte-equivalent paper release.

## Separate extension archive

[VerifierExtension-Results](https://huggingface.co/datasets/oroikono/VerifierExtension-Results)
is private and separate from the workshop release. At reviewed metadata revision
`edfbf059a41a13e12a27fb73a3f3d603fa42ddee`, it contains separately scoped
September development and verification bundles, including completed analyses
and preserved failed runs. The corrected root card indexes ten development
bundles; the planning baseline is only one historical bundle.

Read the [pinned development inventory](https://huggingface.co/datasets/oroikono/VerifierExtension-Results/tree/edfbf059a41a13e12a27fb73a3f3d603fa42ddee/development)
and individual bundle cards for status and limits. For example,
the [18 September numerical pilot](https://huggingface.co/datasets/oroikono/VerifierExtension-Results/blob/edfbf059a41a13e12a27fb73a3f3d603fa42ddee/development/weber_uncertainty_20260918_01/README.md)
and [ranking decomposition](https://huggingface.co/datasets/oroikono/VerifierExtension-Results/blob/edfbf059a41a13e12a27fb73a3f3d603fa42ddee/development/v6_ranking_decomposition_20260918_01/README.md)
report completed, limited analyses. The 16 September uncertainty run failed its
initialization-compatibility gate; the 17 September run stopped at an engineering
import failure, with its scientific audit blocked. A completed analysis is not
automatically independent confirmation, and an engineering failure is not a
successful scientific result. Published status records were inspected; these
bundles were not scientifically regenerated in the 12 October review.

The 12 October updates changed the two root cards and the benchmark's README
manifest entry only. Remote membership, unchanged file identities, changed-file
SHA-256 hashes, and private visibility were verified. Historical scientific
receipts retain their original revisions. See the [privacy review](privacy_review_20261012.md)
for retained identifying provenance and scan limits.

## Missing current-paper bundles

Neither pinned Hugging Face snapshot contains the 5 October integration/repair
or 7 October construction bundles, nor establishes a byte-equivalent current
camera-ready release or the newer paper's positive PDEBench replay. This states
what is absent from these snapshots, not what may exist elsewhere. No
paper-specific Hugging Face Space was found among the account's listed Spaces
in the review.

The GitHub paper repository was renamed to `kagamaya` on 8 October 2026;
visibility remains private. Author, license and provider-output approvals
remain outstanding; see [release blockers](ambiguities.md). Historical receipts
are retained unchanged.
