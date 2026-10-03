# CriticalSplit-MoE 实验跟踪器

> 更新时间：2026-08-10 17:37 +0800  
> 对应冻结计划：`refine-logs/EXPERIMENT_PLAN_20260810_164938.md`  
> 实验状态：`DONE / WEAKEN_ACTION_SPACE / SIMULATION_ONLY`  
> 完整性审计：`PASS / SAME_FAMILY_PROVISIONAL / NO_P0_P1`

## Run ledger

| Run ID | Milestone | Status | Evidence | Result |
|---|---|---|---|---|
| `CS-P0-BASE` | M0 | PASS | FrontierCredit 13/13 tests；old protocol/results/summary byte-identical | 历史 `FRONTIER_SIGNAL_NOT_SUPPORTED` 保持不变 |
| `CS-P0-CONTRACT` | M1 | PASS | CriticalSplit 15/15 tests | subset conservation、ready age、token/replay、future-visibility、fail-closed 通过 |
| `CS-P0-ACTUAL` | M2 | DONE | frozen 8 cells，exact states max `75,390 < 500,000` | `split_flow == whole_flow` in 8/8；eligible cells `0` |
| `CS-P0-SHAM` | M3 | DONE | same 8 cells / same physical transition | optimal replay 中 physical/partition changed decisions 均为 `0`；identity gap `NA` |
| `CS-P0-DECIDE` | M4 | WEAKEN | frozen gate parent recompute | `WEAKEN_ACTION_SPACE`，`paper_result=false` |
| `CS-P1-ONLINE` | future | CUT | P0 C1 未通过 | 不实现 online controller，不启动 GPU |

## Mechanical gate result

| Gate | Result |
|---|---|
| actual eligible cells `>=2` | FAIL: `0` |
| eligible cells use `CRITICAL` | FAIL: no eligible cell |
| median whole capture `<0.90` | FAIL: `NA` |
| sham-applicable eligible cells `>=2` | FAIL: `0` |
| median identity gap `>=0.10` | FAIL: `NA` |
| deadline miss delta `<=0` in all cells | PASS: `0` in 8/8 |

## Result authority

- Completion authority: `artifacts/criticalsplit_pilot/20260810_173200/COMPLETE.json`
  - SHA256: `f56029fd11367de5f96477a93ab5a0f67955cd42de70f0e5f606a1af1c099c40`
- Raw results: `artifacts/criticalsplit_pilot/20260810_173200/pilot_results.json`
  - SHA256: `09e7f77ea54509b865068d9f5c4f8d565fa13893ed3d584e678a1e873b8d42b0`
- Run lock: `48e179f1b2f6965294103a018fc26ed8379a4926521b840fe89f9ec11f8eea6c`
- Preflight: `5c0897841bdacb6aa350558e1ccace82551e9f2c29c6a5599106a3362a20c62c`
- `SOURCE_POST.json`: `all_match=true`；`verify_complete()` PASS。
- Integrity audit: `artifacts/criticalsplit_pilot/audits/20260810_175144/EXPERIMENT_AUDIT.md`
  - `PASS`，无 P0/P1；仅表示 bounded `WEAKEN` 负结论可信。

## Bounded conclusion

CriticalSplit 的 join-closing critical/bulk proper-subset action 在当前冻结 8-cell full-DAG simulation 中没有产生 whole-ready exact action space 之外的 flow headroom。因此当前 action formulation 被削弱，不进入 online-policy、GPU、serving 或论文 claim。该结论不证明自然 workload 上永远不存在 critical-split headroom。
