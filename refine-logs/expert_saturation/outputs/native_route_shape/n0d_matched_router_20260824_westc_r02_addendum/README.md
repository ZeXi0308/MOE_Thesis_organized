# N0d top-k value-consistency addendum

Date: 2026-08-24  
Status: `P1_PARTIALLY_REMEDIATED / EXACT_TIE_BREAK_NOT_RECONSTRUCTED /
NARROW_VERDICT_UNCHANGED`

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
that they were a value-consistent top-k set of the retained 64 router logits.
All 4,608 rows satisfy that value-ordering condition. However, 48 rows have an
exact tie at the top-k boundary, so the retained logits alone cannot reconstruct
the GPU backend's exact tie-break identity. A CPU `torch.topk` replay chooses a
different, equally valued Expert set in 18 of those rows. This does not change
the narrow pre-top-k numerical-divergence association, but it prevents claiming
that every exact selected-ID set was independently recomputed.

Evaluator v3 fail-closes the value-consistency check by requiring, for every
row:

```text
min(logit[selected experts]) >= max(logit[unselected experts])
```

Pre-softmax ranking is sufficient because softmax is monotonic. The non-strict
boundary deliberately accepts either member of an exact kth-place tie, while a
selected lower-logit Expert with an omitted higher-logit Expert is invalid.
The regression suite includes both a lower-logit Expert substitution and an
exact-boundary tie case. The historical v2 verdict is retained append-only, but
its field name `selected_experts_topk_recomputed=true` was stronger than the
implemented invariant and is superseded by the explicit v3 scope fields below.

## Replay

- Evaluator v3 SHA-256:
  `29e82f46ff70ea69c3a63b936085503597a51be5d4298e61bd4ffc62ab494922`
- Derived verdict:
  `n0d-verdict-v3-topk-value-consistency.json`
- Derived verdict SHA-256:
  `8d7f3855f1817a0f000112be2b59befd8941ebb718412980d598dd7c77036d66`
- Runner SHA-256, unchanged:
  `7f00ad096789e20dade97c779b1c6087a902f331eafcba9e879a88d2dd9351cb`
- Result:
  `PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED`
- `selected_experts_topk_value_consistent`: `true`
- `selected_experts_exact_tie_break_recomputed`: `false`
- `exact_topk_boundary_tie_rows`: `48`
- validation scope:
  `TOPK_VALUE_CONSISTENCY_ONLY_EXACT_TIE_BREAK_NOT_RECONSTRUCTED`
- Structural errors: none

The replay used temporary derived copies of the three bundles with only
`.fresh_capture.capture_dir` relocated to the downloaded local capture via
`jq`. Scientific values and the original bundle files were not changed. The
temporary files were not retained, but this deterministic transformation
recreates their exact content hashes:

- process 0: `e9693eead3cc013c292a21cfcff477e3e6c6525d45636ede5d18cfbfb54986e2`;
- process 1: `b20cefc993d409eb8f592bd4bf656ae77b34e8f80606dde757f44c5ee5e61912`;
- process 2: `b2853624fe6cb219d8995c8a90af67212efb7cfc6fa356c0bc1a477606d6eefc`.

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

The bytecode contamination keeps local-directory provenance at `WARN`, and the
48 exact boundary ties keep exact selected-ID provenance at `WARN`. Neither
changes the v3 value-consistency result or the narrow scientific verdict. This
addendum preserves the original ceiling:
`CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_CONFORMANCE_ONLY`. It does not authorize
a router-GEMM mechanism claim, performance or capacity gain, an action Oracle,
a Controller, native serving transfer, or multi-GPU EP evidence.
