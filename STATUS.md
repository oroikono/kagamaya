# Status

## 31 October 2026 — camera-ready reproduction additions

The public release adds the frozen positive PDEBench v5 source, 60 completed
result shards, literal primary prompt dependencies/inventory, the completed truth
sidecar and 100 sequential-confirmation shards. See
[release scope and commands](docs/release_20261031/README.md). The original
manuscript remains historical v19; no current camera-ready PDF is substituted.
This addition does not rerun numerical producers or model providers.

Current stage: **private historical v19 archive**. This is not the current
camera-ready release; see [artifact scope](docs/artifact_links.md).

| Stage | State | Evidence |
|---|---|---|
| Exact v19 manuscript copied | Complete | source hash + manifest |
| Primary raw evidence copied | Complete | 100 gzip shards; portable 33-check recount |
| Agent/control evidence copied | Complete | row-level JSON/JSONL recount |
| Code and data audit archived | Complete | [archived audit report](output/audits/doomsday_v19_code_crosscheck_20260827/doomsday_report.json) |
| Private GitHub push | Complete | `oroikono/kagamaya`, private, `main` |
| Public release | Blocked | A-001, A-002, A-003 |

The stored-record recount passes locally. The 50-seed sequential result remains
aggregate-level because its raw episode shards are absent. Later 2-D/coupled and
Gray–Scott results remain failed-gate evidence only. The newer positive PDEBench
bundle is not included.

Remote: <https://github.com/oroikono/kagamaya>. The initial scientific release
commit is `4408df88ab22d3642d36500e02fb2c811ee0a460`; `REMOTE_RELEASE_RECEIPT.json` records the creation
record. Public release remains blocked and the repository must stay private until its listed blockers
are resolved.

The repository was renamed to `kagamaya` on 8 October 2026 without changing
visibility or rewriting history. The original creation receipt retains its
historical repository name and URL.

On 8 October 2026, [PR #2](https://github.com/oroikono/kagamaya/pull/2) merged
`release/reproduction-and-hub-links`, including the reproduction exit-status fix
and documentation updates. It did not change the numerical calculations or
archived scientific records. Historical audit provenance is unchanged.

The 12 October maintenance update clarifies archive and Hub-card scope, adds
contribution guidance, hardens path/inventory checks, and replaces stale recount
receipts on runtime failure. The original 33 numerical checks and archived
scientific inputs remain unchanged. The [privacy review](docs/privacy_review_20261012.md)
records retained identifying provenance; this private archive is not anonymous.
