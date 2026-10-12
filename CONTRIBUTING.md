# Contributing

This is a private historical research archive. Collaborators with repository
access can use an issue or pull request to discuss a change. Public release and
redistribution remain subject to the [release blockers](docs/ambiguities.md).

Keep changes focused and preserve frozen scientific inputs, published receipts,
and historical commits. A change to a scientific claim needs new, explicitly
scoped evidence; documentation or engineering cleanup is not that evidence.
Do not commit credentials, raw coding-agent sessions, private voice material,
or unreviewed provider output.

Before proposing a change:

1. Review the working tree, exact diff, destination branch, and remote. Stage
   only the intended files.
2. Run the relevant tests and the [manifest and recount checks](docs/reproducibility_protocol.md).
   Add a regression test for a behavioral fix. Record commands actually run and
   distinguish local checks from completed CI.
3. Keep code commits and Hugging Face dataset revisions separate. Preserve
   immutable artifact links and update the reviewed inventory for changed files.
4. Follow the agreed review workflow, push only the intended branch, and verify
   the remote commit. Do not rewrite historical research receipts to tidy history.

A useful commit names the action and affected behavior. For a nontrivial change,
explain the old behavior, its consequence, the change, and observed checks.
Use a short pull-request description: purpose, change, checks, and any remaining
scope or compatibility limit. Separate engineering and media changes when they
can be reviewed independently.

For upstream contributions, follow that project's own rules.
