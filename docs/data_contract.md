# Data contract

- Primary denominator: 1,800 matched conditions per numerical implementation.
- In-library denominator: 1,500 per implementation.
- Agent goal denominator: 180 in-library cases per model and goal; paired implementations are not independent replicates.
- Primary raw closure: exactly 100 gzip shards, 50 seeds for each of two implementations.
- Truth labels and reference fields are owner-side and must not affect selector actions.
- JSON/JSONL and gzip JSON are immutable inputs; every released byte is manifest-bound.

