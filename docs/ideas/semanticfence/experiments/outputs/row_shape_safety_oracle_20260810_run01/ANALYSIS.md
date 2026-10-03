# SemanticFence row-shape safety analysis

Input SHA256: `5410e3a8853f4b839df1096e57034cb187e44d0b91b584890de210663e1775a8`

## Verdict

The frozen calibration data are strongly consistent with row/shape-local stability rather than a special-pair effect, but this is not yet a causal proof. Even a perfect row-safety oracle has a limited expert-call-count ceiling on this distribution.

## M=2 diagnostic

- Safe rows: 2768/32234 (8.5872%).
- Both-safe arrival-order pairs: 121; uniform-random-pairing expectation from the global row rate: 118.81.
- Exactly-one-safe pairs: 2526 (1290 first-only, 1236 second-only).

## Perfect-oracle call-count bound

- Batch counts: {'2': 436, '4': 242, '8': 49, '16': 2}.
- Covered rows: 2264/32234 (7.0236%).
- Saved expert calls: 1535/32234 (4.7621%).
- This is not measured latency or serving speedup.

## Required falsification

Replay frozen safe rows with multiple new partners inside the same layer/expert/M cell. A partner-dependent flip falsifies the row-local interpretation; invariance across held-out partners supports a row-safety model.
