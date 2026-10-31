# Camera-ready reproduction additions — 31 October 2026

This release adds the original positive PDEBench v5 source and completed numerical
records, primary model-prompt dependencies/inventory, and raw 50-seed sequential
confirmation to the historical archive. `additions.json` lists selected file hashes;
the root `MANIFEST.sha256` covers the public tree.

## Offline checks

From the repository root:

```bash
python3 scripts/verify_manifest.py
python3 scripts/reproduce_findings.py
python3 -m unittest discover -s tests -v
```

The 33-check stored-record recount does not rerun solvers or providers.
To replay the original PDEBench policy code against 60 stored JSON shards:

```bash
cd experiments/agentic_frontier/wp1_wp4_operator_agent/transfer_recovery_campaign_v1_20260827
python3 -m pdebench_diffreact_v5_20260828.freeze verify
python3 -m pdebench_diffreact_v5_20260828.aggregate \
  --shard-dir results/pdebench_diffreact_v5_20260828/v5_confirmation_shards \
  --output /absolute/path/to/a-new-replay.json
```

Use a new output path outside the frozen source directory. NumPy is required;
HDF5, APIs and PDE simulation are not. Release replay covered 900 episodes,
10,800 views and 2,700 decisions and produced the exact original aggregate hash
`fae23fad88671febec97f2141c9607e0353039dc219d439a47b1ab3c0ce08f19`.
It used CPython 3.9.6/NumPy 2.0.2, distinct from the original producer preflight's
CPython 3.12.2/NumPy 2.0.1/h5py 3.11.0 and the source freeze's CPython 3.12.13.

Fresh projection/fitting requires the separately downloaded 13,243,560,688-byte
publisher HDF5. The frozen contract specifies its DOI, file ID, MD5, source commit,
source partitions and numerical gates. No raw HDF5 or trajectory arrays are copied
here. Historical Euler scripts retain original paths; configure them for a new
installation. Stored replay is not fresh numerical or physical validation.

## Model and sequential evidence

Literal primary prompt templates, schemas, public cases and the 1,080-entry prompt
inventory are restored at their original relative paths. R1 and amended R2 source
manifests are preserved separately, without overwriting historical pins. Completed
held-out truth is released for evaluation, not represented as pre-call model input.
No new inference was performed. Future closed-provider responses are not guaranteed
to match historical calls.

The original `AUTHORIZATION.json` is withheld because it contains a private
provenance reference. Thus the public primary tree is not byte-identical to every
historical control file. Fresh paid execution requires the reproducer's own
credentials, explicit authorization and budget; historical permission cannot be
reused to spend. Scientific prompts and stored-record inspection do not require
publishing that private reference.

The 100 sequential shards cover two implementations and seeds 10500–10549.
All shard semantic digests and nested owner/proposal/score digests were checked;
this is integrity verification, not PDE regeneration. Prior failed gates and
negative policy comparisons remain part of the evidence. The original full
sequential validator still imports transitive generator code outside its frozen
23-file list; the first release-check attempt stopped on a missing import before
any solve. The integrity checks above are not represented as that full replay.
See [validation results, skips and failures](validation.json).

## License and public-release scope

The adjacent MIT license applies only to newly added author-owned code, tests,
prompt templates and documentation. It does not relicense third-party data,
dependencies, model-provider outputs or previously licensed material.

PDEBench diagnostics and policy records derive from *PDEBench Datasets*, Takamoto
et al. (2022), [DOI:10.18419/DARUS-2986](https://doi.org/10.18419/darus-2986),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Processing
includes registered noise perturbations, weak-form projection, fitting and policy
evaluation. These are derived numerical records, not the original HDF5. No
endorsement is implied; dataset attribution is separate from the MIT code license.

The additions were checked for recognizable secrets, private provenance,
private drafts/maps and unexpected provider/session content. This bounded review
is not an exhaustive security/legal guarantee. No new provider outputs, coding
sessions, private authorization, unrelated experiment trees or Git history were
copied. Existing history was not rewritten or newly certified. No Docker image
is claimed in this revision.
