# Qwen qualification integrity review

2026-09-13 · GPT-5.6-Sol ultra, fresh read-only agent · same-family / provisional.

**WARN; P0=0, P1=1.** No confirmed loader, paging or map execution defect. The original numerical diagnostic does not authorize a performance run; the one required follow-up is a same-precall layer47 grouped-reference replay. Review stopped at this bounded scope; the upstream WiSP state implementation was not line-by-line reviewed in this pass.

| Check | Status | Evidence and boundary |
|---|---|---|
| A Reference provenance | PASS | `readback_r01/launch/wisp_v026_adapter.py:254-283`: same hidden/top-k and full weights, same fused kernel; local execution reference, not task ground truth. |
| B Denominators | PASS | Adapter `:264-280` reports raw maxabs and reference norm alongside relative L2. `readback_r01/results/qualification/pager_summary.json:20739-20742` states payload exclusions. |
| C Files/claims | PASS | Original failure retained in `analysis_attempt01.json`; original analyzer `:120-121` incorrectly equates actual KV with budget. `../analyze_qualification_v2.py:24-50` derives exact block rounding; reviewer recomputed and matched `analysis_v2.json`. `REPORT.md` retains measurement/numerical limits and no performance claim. |
| D Invoked code | WARN | `readback_r01/launch/launch.json:60-112` and `readback_r01/results/qualification/environment.json:42-53` close execution/source identity. Upstream `WispMoEState.ensure_resident` implementation was not included in the review's supplied files; required/runtime hashes were present, but that body's internals remain unaudited in this pass. |
| E Scope | PASS | One GPU/revision/episode, four requests,32 outputs and48 same-precall references. No quality, independent router, whole-resident or performance claim. |
| F Type | PASS | self_supervised_proxy for numerical conformance plus actual native GPU integration; no dataset quality ground truth. |

P1: `readback_r01/results/qualification/qualification_validation.json:6-8,15038-15042,15214-15235` retains layer47/call287 allclose=false, maxabs0.25 and relative L2 0.00221848; raw call is `pager/calls.jsonl:288`. The reviewer recomputed21 one-group references as bit-exact and27 two-group references as non-bit-exact; onlylayer47 crossed the original .01/.01 diagnostic. Expert groups are disjoint and `readback_r01/launch/wisp_expert_groups.py:203-255` sums their outputs in BF16. This identifies the likely class of difference, not semantic harmlessness or quality equivalence.

The selected follow-up retains actual partials and compares same-partition full-weight partials/final sum, FP32-cast and reversed aggregation on the same new pre-call. Two-group FP32/reversed sums need not improve. The old hidden tensors were not saved, so matching replay metadata must not be called original-hidden bit identity. Structural/accounting PASS beside allclose=false remains exactly that. The new conditional startup/performance preparation is outside this review and is not a result.
