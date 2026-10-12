# Project plan

1. **Private closure** — exact source, raw evidence, portable replay, manifest. Gate: 33/33 and manifest pass.
2. **Remote verification** — push private repository, clone into a fresh directory, rerun replay and manifest. Gate: exact remote commit and clean-clone pass.
3. **Submission binding** — record repository URL and commit in the manuscript/rebuttal materials. Gate: paper hash matches repository source.
4. **Public-release review** — resolve A-001 through A-003. Gate: author and license approval.

At every stage, stop on failure, keep the failed check output for diagnosis,
and do not advance the release label.
