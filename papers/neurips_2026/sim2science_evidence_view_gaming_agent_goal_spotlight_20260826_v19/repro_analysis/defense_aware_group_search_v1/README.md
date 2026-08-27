# Defense-aware exhaustive group search

This retrospective analysis enumerates every node-wise four-mode group in the
frozen 12-view confirmation bank. It asks whether any agreement-eligible group
has an incorrect terminal signature, using held-out truth only for scoring.

Run from this directory:

```bash
python3 analyze.py --output analysis.json
```

The script reads the 100 immutable confirmation shards from the adjacent
experiment repository and uses only Python's standard library. It verifies the
expected 3,600 episodes, 12 actions per episode, and complete mode groups.

Result per numerical implementation: the deployed agreement rule has three
false verdicts, and exhaustive defense-aware enumeration finds exactly the same
three attackable in-library conditions. All are false omissions; no wrong
identification is agreement-eligible. Four individual coarse rows do contain a
wrong identified support, but disagreement among the remaining modes blocks
them. This finite, truth-aware oracle is neither an executable policy nor a
security guarantee beyond the frozen bank.
