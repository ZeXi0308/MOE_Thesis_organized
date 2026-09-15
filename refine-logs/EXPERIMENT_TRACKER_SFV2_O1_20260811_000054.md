# SemanticFence-v2 SFV2-O1 Online-observability Gate Tracker

> Formal output: `docs/ideas/semanticfence/experiments/outputs/semantic_online_observability_20260810_run01/`  
> Status: `SUCCESS_COMPLETE / PIVOT_TO_SHADOW_VERIFY / RAW_RECOMPUTE_PASS / PAPER_RESULT_FALSE`  
> Scope: one OLMoE-1B-7B BF16 / single RTX 5090 / fresh document-disjoint expert-stage forward-replay Gate; not serving or paper proof.

## Execution ledger

| Stage | Status | Evidence |
|---|---|---|
| Frozen protocol/code | `PASS` | Runner SHA-256 `e35ab714...f70a7`; test `39482a29...e4a`; source config `54022565...e5`; fresh12 input `ea29c762...ca3f3` |
| Fresh documents | `PASS` | WikiText revision `b08601e0...c3`, fingerprint `051a1127882eb518`; 12 unique documents split 6/2/4; selected hashes have zero overlap with the 1,313-entry defensive exclusion union |
| Pre-outcome closure | `PASS` | `PRE_OUTCOME_LOCK.json` SHA-256 `36a0595b...d38`; status `FROZEN_BEFORE_ANY_M1_M2_SEMANTIC_OUTCOME`; 384 scheduled edges = 192 train + 64 validation + 128 test; W=8; test outcomes at lock=0 |
| Test admission freeze | `PASS` | One validation-only threshold `0.060565232014776364`; validation admitted 4 endpoints with 0 unsafe; `TEST_ADMISSION_PLAN` states test outcomes at freeze=0 |
| Local/remote qualification | `PASS` | Primary runner 19/19 directed tests and `py_compile` pass locally and remotely; no-outcome dry-run pass; source hashes exact on the RTX 5090 host |
| Formal GPU execution | `PASS` | Exit 0; `COMPLETE.status=SUCCESS_COMPLETE`; 22 sealed artifacts hash-close; all 768 endpoint observations pass native self-replacement bitwise no-op plus M1/M2 repeat stability |
| Natural Semantic Oracle | `PASS_ACTION_FLOOR` | 77/128 safe edges; maximum matching 77 pairs; 154/255 rows = 60.3922%; 4/4 test docs; 29.2714% additive expert-stage saving |
| Witness-v1 safety | `FAIL` | 19 admitted endpoints; 5 candidate/executed pairs; 5 unsafe admitted endpoints and 4 unsafe executed pairs |
| Witness-v1 coverage | `FAIL` | 10/255 matched rows = 3.9216%, below the frozen 5% floor; 5 pairs below the required 16 |
| Witness-v1 net cost | `FAIL` | Gross saving 1.9007%; measured D2H + Python certificate/greedy overhead 19.435709 ms; net saving -183.3504% |
| Shared-helper full artifact recompute | `PASS` | `INDEPENDENT_RECOMPUTE.json` rederives lock/witness/threshold/matching/cost/verdict closure, but imports primary pure helpers; float leaves only use abs tolerance `1e-12` across Python environments |
| Independent raw-ledger recompute | `PASS` | No project-module imports; 5/5 tests; 22 hashes exact; per-edge labels, matching, admission, raw timing cost and verdict independently rederived as `PIVOT_TO_SHADOW_VERIFY` |
| Fresh-agent integrity audit | `WARN / P0=0 / P1=3` | Numerical verdict supported; warnings cover proxy semantics, same-code/full-feature independence boundary, non-observable strict mtime order after transfer and stale historical terminology; same-family provisional |
| Authority update | `PASS` | Dedicated verdict, SemanticFence README, docs/current and MANIFEST only; global tracker and IDEA_REPORT remain untouched |

## Mechanical verdict

Natural Oracle saving and row coverage both exceed 5%, so `NO_GO_NATURAL_SEMANTIC_HEADLINE` does not apply. Witness-v1 fails zero-unsafe admission, minimum 16 pairs, minimum 5% row coverage and positive net saving. The frozen rule therefore yields exactly:

`PIVOT_TO_SHADOW_VERIFY`

This retains the semantic action-space question while rejecting pre-execution witness-v1. It does not authorize threshold/feature/classifier rescue on this test split.

## Claim boundary

The highest supported claim is that a large fresh natural downstream ordered-top-k-stability proxy action space exists under this one frozen OLMoE/BF16/RTX-5090 expert-stage workload, while witness-v1 cannot exploit it safely or net-positively. This is `self_supervised_proxy`, not model-quality or task-semantic ground truth. Hash chains, exclusive-write code order and non-later mtimes support the freeze, but rsync-preserved same-second timestamps make strict external chronology independently unobservable. The projection excludes packing, queueing, full-layer/request latency, serving, EP/network and multi-GPU effects. Production execution remains `M=1` fail closed.

## Only next experiment

`SFV2-O2: Fresh No-Actuation Shadow-Verifier / Selective-Repair Gate` on a new document-disjoint split. Freeze one post-M2 verification/repair rule before outcomes; commit no unverified M2 result; measure zero unsafe commit, repaired/admitted coverage and verifier+repair net cost. Do not run another pre-execution classifier sweep.
