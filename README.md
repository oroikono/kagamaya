<p align="center">
  <img src="assets/branding/kagamaya-perspective.png" width="360" alt="Kagamaya: an intact charcoal wave before a minimal glass mirror; its vermilion reflection has a missing segment." />
</p>

# Kagamaya

<p>
  <a href="https://huggingface.co/datasets/oroikono/EvidenceViewBench-Dev"><img src="assets/platforms/huggingface.svg" width="22" height="22" alt="Hugging Face" /> Development benchmark</a>
  &nbsp; · &nbsp;
  <a href="https://huggingface.co/datasets/oroikono/VerifierExtension-Results"><img src="assets/platforms/huggingface.svg" width="22" height="22" alt="Hugging Face" /> Extension archive</a>
  &nbsp; · &nbsp;
  <a href="docs/video.md"><img src="assets/platforms/youtube.svg" width="26" height="22" alt="YouTube" /> YouTube — coming soon</a>
</p>

Both are private Hugging Face datasets; see [artifact scope and pinned revisions](docs/artifact_links.md).
The YouTube entry opens the film notes until an upload is available.

Kagamaya studies how changing a trajectory's numerical representation can change
an equation-discovery verdict. This private research archive contains the
historical v19 paper source, experiment code, stored records, and an offline
script that recounts the paper's main reported results.

Paper: **“The Verifier Is Part of the Simulator: Evidence-View Gaming in Agentic Equation Discovery.”**
Accepted at [Sim2Science: ML with Imperfect Scientific Models](https://www.sim2science.com/),
a **NeurIPS 2026 workshop**. This archive is not the current camera-ready release.
The direct OpenReview paper link is pending verification; the workshop link is
not a paper PDF link.

**K-A-G-A-M-A-Y-A** · pronounced **kah-gah-MAH-yah** (four syllables).

A coined name inspired by Japanese
[*kagami* (鏡), “mirror”](https://www.kanjipedia.jp/kotoba/0001550900), and Sanskrit
[*māyā*, associated with appearance and illusion](https://www.encyclopedia.com/philosophy-and-religion/eastern-religions/hinduism/maya-magic).
Our poetic reading is **“mirror of appearances”**, not a literal translation
or an established Japanese/Sanskrit word. The [emblem](assets/branding/README.md)
uses the same metaphor: an unchanged trajectory can have a misleading numerical reflection.

[Reproduce](#quick-start) · [Artifacts](docs/artifact_links.md) ·
[Workshop film](docs/video.md) · [Emblem and assets](assets/branding/README.md)

## Quick start

From the repository root, check the SHA-256 file manifest before recounting:

```bash
python3 scripts/verify_manifest.py
python3 scripts/reproduce_findings.py
```

Both commands use Python's standard library. The recount uses stored records;
it does not rerun solvers or make network or model-provider calls.
Expected result: `33/33` checks pass, with machine-readable output at
`output/reproduction/code_crosscheck.json`. Failed or inconsistent checks return
a nonzero exit status.

Run the release smoke test with:

```bash
python3 -m unittest discover -s tests -v
```

See the [reproducibility protocol](docs/reproducibility_protocol.md) for exact
revision and environment recording.

## Results recounted

- Temporal refinement: false library rejections fall from 600 → 27 → 13 → 10
  as the sample count increases from 7 → 13 → 25 → 49 across 720 conditions.
- Exact public-score maximization: 1,500 false promotions across 1,800 conditions per implementation.
- Four-view agreement: 1,435 correct, 3 false, and 362 inconclusive outcomes
  across 1,800 conditions per implementation.
- Defense-aware enumeration: zero agreement-eligible false identifications.
- Agent controls: the four-goal, three-model action ladder and trust-report/action joins.
- Auxiliary controls: the 1% high-noise paired comparison, fine/coarse residual
  AUROCs, Qwen control, and de-nesting transitions.
- Aggregate consistency: the sequential control and later failed 2-D/Gray–Scott gates.

## Evidence and release limits

The primary 1,800-condition result is recounted from all 100 raw gzip shards.
Agent and auxiliary controls use row-level JSON/JSONL. The 50-seed sequential
result has code and a compact aggregate, but no raw episode shards here. Later
2-D/coupled and Gray–Scott records are failed-gate evidence, not positive transfer
results. Implementations and related views are paired, not independent replications;
see the [sample counts](docs/data_contract.md) and [evaluation definitions](docs/evaluation_protocol.md).

This historical v19 archive does not include the current camera-ready source or
the newer paper's positive PDEBench replay bundle. Related Hugging Face archives
contain exposed development material, completed analyses, and preserved failed
runs, not a byte-equivalent current-paper release. Their pinned revisions,
bundle-specific limits, and absent October bundles are listed in
[artifact links](docs/artifact_links.md).

The repository remains private. Public release requires provider-output
redistribution review, dataset and code licensing, authorship approval, and final
citation review; see [release blockers](docs/ambiguities.md). Do not commit
credentials or private voice data. See the [dated privacy review](docs/privacy_review_20261012.md)
for scan scope and findings; it is not an exhaustive guarantee.

## Workshop film

[![Kagamaya workshop film: Same data. Wrong equation.](assets/video/workshop-film-thumbnail.png)](docs/video.md)

Animation-led explanation of the corrected workshop paper, with optional
captions. **Narration replacement pending; not approved for upload.**
The preview opens [production notes](docs/video.md) until the reviewed video is published.

## Layout

- `papers/`: exact editable manuscript and figures.
- `experiments/`: generation, verifier, policy, control, and validation source.
- `evidence_view_benchmark_v1/`: frozen agent interfaces and scored machine-readable records.
- `scripts/`: portable reproduction and manifest validation.
- `output/audits/`: code and data audit reports.
- `docs/`: provenance, data formats, interfaces, evaluation, and reproduction instructions.

For changes, see [contributing](CONTRIBUTING.md). Current release history is in
[STATUS.md](STATUS.md).
