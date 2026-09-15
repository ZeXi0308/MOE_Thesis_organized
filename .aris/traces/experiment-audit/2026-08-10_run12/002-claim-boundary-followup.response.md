Overall: **FAIL on artifact closure**, although both prior claim-label P1s are resolved and the mechanical verdict is unchanged.

Unresolved counts:

- Prior claim-label P1s: **0/2**
- New unresolved issue: **P0=0, P1=1** — stale checksums in `COMPLETE.json`

Evidence:

- Variant-equivalence is now explicitly distinguished from independent numerical ground truth in `analyzer.py:42-49,695-702,725-730`; enforced by tests at `test_joinstream_real_moe_tail.py:269-270`.
- The generated artifact repeats this honestly at `analysis.json:556-559`, `summary.md:30-37`, `COMPLETE.json:9-10,29-35`, and tracker lines `6,16,29`.
- Four-cell synthetic scope is now explicit, and `FREEZE` is bounded to the current thesis-promotion path rather than universal impossibility: `summary.md:7-10,37`, `COMPLETE.json:9-12`, and tracker `29`.
- The mechanical verdict remains `WEAKEN_UPPER_BOUND_TOO_SMALL / GATING_INSUFFICIENT / WEAKENS`: `analysis.json:560-568,580-602`, `summary.md:3-5`, `COMPLETE.json:5-7`, tracker `17`. Analyzer adjudication remains at `analyzer.py:590-653`.

New P1:

`COMPLETE.json:55-60` still records the pre-revision hashes. Recomputed current hashes differ:

- `analysis.json`: recorded `007ae…`, current `80e0d…`
- `summary.md`: recorded `499944…`, current `64fcb0…`
- tracker: recorded `f556a…`, current `6978d8…`
- analyzer: recorded `1a5bb…`, current `223022…`
- tests: recorded `ad0a4…`, current `dddf59…`

Impact: this does **not** alter the formal data or mechanical verdict, but `status: COMPLETE` is not presently hash-closed. The two requested semantic boundaries pass; the revised bundle cannot receive overall PASS until its manifest reflects the revised files. No files were edited.
