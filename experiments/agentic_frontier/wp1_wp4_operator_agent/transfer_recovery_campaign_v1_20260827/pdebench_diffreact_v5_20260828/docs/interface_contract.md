# Interface contract

- I-001: source is the exact name/size/MD5-bound HDF5 file.
- I-002: each source group must contain data `[101,128,128,2]`, the exact publisher x/y arrays `-1+(8*j+3)/512`, and time `0,0.05,...,5`.
- I-003: selectable views are exactly `3 temporal counts × 4 test counts`; the reference is nonselectable `(101,5)`.
- I-004: target-specific complete libraries and three development-selected omission contracts are fixed in `contract.py`; the local-source contract is a grouped omission.
- I-005: confirmation IDs cannot be accepted by the development/audit runner.
