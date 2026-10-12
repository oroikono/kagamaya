# Credential and privacy review — 12 October 2026

This review covers the private Kagamaya repository and its two linked private
Hugging Face datasets. It is a heuristic screening result, not proof that all
secrets or identifying information are absent. No credential values are included
in this report.

## Scope and findings

| Target | Reviewed baseline | Coverage | Finding |
|---|---|---|---|
| GitHub `oroikono/kagamaya` | `0e8dc40c681cf4d6fede94cd3e36b3364e40176b` | All locally reachable blobs, commits, and tags; 19 commits at review time; the proposed current files are screened separately before publication | No credential-pattern matches. Historical author email metadata, account identifiers, local/cluster paths, and an archived SSH host address remain. |
| HF `EvidenceViewBench-Dev` | `6e9d936ce71116f04d5e14c48de1a1bede4f4d95` | All 245 current files, with file membership and hashes checked | No credential-pattern matches. A recorded model trace retains a personal cluster path. |
| HF `VerifierExtension-Results` | `1c0ca23a7a68b14a1c90669caf1f89d4a8a9bbf4` | All 151 current files, with file membership and hashes checked | No credential-pattern matches. Frozen operational provenance retains home/cluster paths and account identifiers. |

The scanner checks recognizable provider/API keys, private-key headers,
credential-bearing URLs, and identifying text. Compressed text is inspected with
bounded expansion. PDF text and metadata are inspected; other binary files receive
only partial string screening. Numerical arrays, images, encrypted content, and
unrecognized secret formats are not exhaustively interpreted. No live credential
validity test is performed.

Hugging Face history screening additionally covered all returned default-branch
commit metadata (eight benchmark commits and fifteen extension commits) and
their available root README versions. It found no credential candidates in that
limited historical scope; non-README historical payloads were not screened.

GitHub's native secret-scanning endpoint reports that secret scanning is disabled
for this repository. There is therefore no native alert coverage to report.
Remote deleted or dangling Git objects, platform account settings, discussion
bodies, and all historical Hugging Face data payloads are outside this scan.
Git author attribution and third-party maintainer/license attribution are distinct
from unintended paths, and must not be removed indiscriminately.

## What changed and what remains

New Git commits use the account's GitHub no-reply address. Landing pages and
dataset cards now state the release scope and privacy limits. This maintenance
update does not alter frozen scientific records or rewrite existing Git history.

**Private does not mean anonymous.** These archives still contain identifying
provenance. A public or anonymized release needs a separately reviewed, versioned
redaction that preserves the original hashes and records each transformation;
history rewriting and redistribution require their own approval. Existing
licensing and provider-output restrictions remain in force.

See [artifact links](artifact_links.md) for the updated immutable metadata
revisions and [release blockers](ambiguities.md) for remaining publication gates.
