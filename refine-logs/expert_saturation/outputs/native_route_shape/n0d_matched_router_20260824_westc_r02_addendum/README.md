# N0d top-k recomputation addendum

Date: 2026-08-24  
Status: `P1_REMEDIATED / NARROW_VERDICT_UNCHANGED`

The manifest-listed scientific evidence bytes in the original `westc-r02`
campaign were not modified. Its original verdict remains SHA-256
`0775016fc43a8d4649e4b8c715cf10ad867255ebde146e0023c170ab7597e616`.
The original `CAMPAIGN_FILES.sha256` was rechecked after this addendum and every
listed entry still passed. This statement is deliberately narrower than
"the directory was untouched": two unmanifested bytecode files were created in
the local downloaded campaign after its completion sentinel.

## Why this addendum exists

The post-result integrity audit found one P1: evaluator v1 checked retained
Expert IDs for count, range, and uniqueness, but did not independently verify
that they were a valid top-k set of the retained 64 router logits. The reviewer
independently recomputed all 4,608 rows and found them consistent, so this did
not invalidate the result; it was nevertheless a fail-closed evidence gap.

Evaluator v2 closes it by requiring, for every row:

```text
min(logit[selected experts]) >= max(logit[unselected experts])
```

Pre-softmax ranking is sufficient because softmax is monotonic. The non-strict
boundary deliberately accepts either member of an exact kth-place tie, while a
selected lower-logit Expert with an omitted higher-logit Expert is invalid.
The regression suite includes both a lower-logit Expert substitution and an
exact-boundary tie case.

## Replay

- Evaluator v2 SHA-256:
  `8d761defb528c33165c3d9b9987285b7883e8d8f45aed039357891528d1b8308`
- Derived verdict:
  `n0d-verdict-v2-topk-recomputed.json`
- Derived verdict SHA-256:
  `871874e073cb20b15efee9a47152c8cf47f47af25991ac81f49f1a0a30b9d9ac`
- Runner SHA-256, unchanged:
  `7f00ad096789e20dade97c779b1c6087a902f331eafcba9e879a88d2dd9351cb`
- Result:
  `PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED`
- `selected_experts_topk_recomputed`: `true`
- Structural errors: none

The replay used temporary derived copies of the three bundles with only the
remote capture-directory string relocated to the downloaded local capture.
Scientific values and the original bundle files were not changed.

## Post-seal local-directory contamination

Targeted review found these additional local files:

```text
frozen/__pycache__/evaluate_n0d_matched_router_gate.cpython-314.pyc
frozen/__pycache__/n0d_capture_contract.cpython-314.pyc
```

Both have local mtime `2026-08-24T00:22:41+0800`, later than
`CAMPAIGN_COMPLETE.json` at `00:20:31`. They are absent from
`CAMPAIGN_FILES.sha256`; therefore that manifest proves the listed bytes but
does not prove an exact closed directory set. Their creation is consistent
with a post-download local import/replay despite the campaign launcher having
set `PYTHONDONTWRITEBYTECODE=1`; the exact creating command was not independently
logged. They are retained rather than deleted. No scientific input, process
bundle, verdict, capture, or manifest-listed file failed its hash check.

## Claim boundary

The bytecode contamination keeps local-directory provenance at `WARN` but does
not change the recomputed scientific verdict. This addendum preserves the
original ceiling:
`CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_CONFORMANCE_ONLY`. It does not authorize
a router-GEMM mechanism claim, performance or capacity gain, an action Oracle,
a Controller, native serving transfer, or multi-GPU EP evidence.
