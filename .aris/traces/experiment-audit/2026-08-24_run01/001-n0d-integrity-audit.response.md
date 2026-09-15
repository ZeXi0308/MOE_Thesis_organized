# N0d post-result integrity audit response

Overall: `WARN` (scientific payload passes; claim protocol not yet closed).  
P0 = 0, P1 = 1. Reviewer identity: fresh same-family provisional. Repository
HEAD `b141c1d...`, worktree dirty; no files modified by the reviewer.

Core conclusion: `PRETOPK_NUMERICAL_DIVERGENCE_ASSOCIATION_REPRODUCED` may be
retained, but only as a one-model, one RTX 5090, BF16, fixed four-request/eight-
step custom-Transformers matched-prestate execution-conformance association.
It must not be written as mechanism localization, router-GEMM causality,
capacity/latency/SLO improvement, Controller evidence, native-serving evidence,
or multi-GPU evidence.

## A-F

- A. Ground/reference provenance: `PASS`. Reference tokens are independently
  reconstructed from the hash-sealed `request_ledger.jsonl`, not generated from
  the three evaluated bundles (`n0d_capture_contract.py:99-162,189-278`). The
  capture explicitly declares `scientific_ground_truth=false`
  (`capture/serial_audit.json:413-423`,
  `capture/CAPTURE_COMPLETE.json:454-464`). This is a same-model conformance
  reference, not dataset ground truth. Model ID/revision/BF16 are frozen, but
  actual loaded weight-file digests are absent; that is a provenance ceiling,
  not a refutation of the narrow result.

- B. Score normalization: `PASS`. The Gate uses raw nonzero logit differences,
  Expert-set differences, and fixed allclose tolerances; it does not divide by
  the model output's max/min/mean (`run_n0d_matched_router_gate.py:510-563`,
  `evaluate_n0d_matched_router_gate.py:121-125,585-615`).

- C. File/key/number existence: `WARN`. All three bundles, the verdict, capture,
  and both manifest levels exist and pass SHA-256. An independent invocation of
  the frozen evaluator reproduced the semantic content of `n0d-verdict.json`.
  The final status and values exist in `n0d-verdict.json:7-249,272-350`; the
  campaign sentinel binds all three bundles and the verdict
  (`CAMPAIGN_COMPLETE.json:24-39`). At audit time, however, the authority tracker
  still marked N0d `PREPARE` (`EXPERIMENT_TRACKER.md:15,29`) and its current
  entry still described N0d as the next uncertainty (`EXPERIMENT_TRACKER.md:166-
  174`). The artifact was therefore provisional until tracker/report closure.

- D. Dead code: `PASS`. Core metric and classification functions occur on the
  runner/evaluator main paths. The then-current 53 targeted tests passed. No
  phantom metric path was found.

- E. Scope versus language: `PASS`. The protocol freezes one model, four
  requests, eight steps, batch width four, and three processes
  (`N0D_MATCHED_ROUTER_GATE.md:28-37`). Its claim ceiling excludes mechanism,
  capacity, latency, Controller, native serving, and multi-GPU claims
  (`N0D_MATCHED_ROUTER_GATE.md:99-109`). All verdict unlocks remain false
  (`n0d-verdict.json:251-271`).

- F. Evaluation type: `self_supervised_proxy` plus a real-GPU execution-
  conformance measurement. It is neither `real_gt` nor simulation. The verdict
  labels its evidence `CUSTOM_TRANSFORMERS_MATCHED_PRESTATE_THREE_FRESH_PROCESS`
  (`n0d-verdict.json:251-255`).

## Targeted checks

- Same-prestate fork: confirmed. Every step clones canonical state into three
  value-equal and storage-disjoint KV branches
  (`run_n0d_matched_router_gate.py:202-267,447-463`). Bundles record
  `serial_a_only`, batch-state non-propagation, and eight fork checks
  (`process-0.json:8-15,54-126`).

- Same-decode-step restriction: confirmed. The frontier groups by
  `(request_id, decode_step)` and only permits an earlier/equal layer within the
  same step to explain an assignment change
  (`run_n0d_matched_router_gate.py:469-475,493-571`), including a prior-step
  negative test (`test_run_n0d_matched_router_gate.py:128-150`).

- Serial A/B: in every process, all 32 tokens, 512 router records, and 32,768
  logit scalars are exact. The verdict records all three exact controls
  (`n0d-verdict.json:244-249`).

- Fresh processes: PID/start-time tuples are unique and all share one boot
  (`n0d-verdict.json:272-287`); arm order is counterbalanced.

- Cross-process equality: after directly reading all three approximately 4 MB
  bundles, tokens, Expert IDs, and logits are byte-identical within each arm,
  stronger than evaluator allclose. Verdict stability is true
  (`n0d-verdict.json:231-249`).

- Capture binding: the exact seven-file seal and workload/ledger reconstruction
  are enforced by `n0d_capture_contract.py:81-96,189-278`. Local revalidation
  produced capture hash `0b3d...a1b0`, ledger hash `33d9...acfa`, and reference
  hash `024e...6fc7`, matching `n0d-verdict.json:322-347`. Full parsing found 16
  ledger requests, 128 steps, 32 batch rows, and 16,384/16,384 valid
  route-to-ledger Expert/token contributions.

- Campaign lineage: r01 completed only prepare, failed capture, declared
  `retry_performed=false`, and produced no process bundles
  (`westc_r01_aborted/campaign/CAMPAIGN_ABORTED.json:2-15`). Its capture was
  incomplete and scientifically ineligible. r02 is therefore not favorable
  trajectory selection.

- Association only: the first assignment frontier is request 000, step 1,
  layer 3; the same step's first logit difference is layer 0
  (`n0d-verdict.json:11-44`). The original bundle reports serial boundary
  `0.0`, batch boundary `0.005859375`, and assignment-layer maximum delta
  `0.015625` (`process-0.json:1038-1060`). This proves that a difference is
  visible before the changed assignment; it does not distinguish router,
  Attention/KV, padding, shape, or companion mechanisms.

## P1

The evaluator did not independently recompute
`selected_experts = softmax(router_logits).topk(8)`. It checked only Expert
count, range, and uniqueness, and then compared the stored IDs
(`evaluate_n0d_matched_router_gate.py:203-231,588-604`). The production path
does execute softmax plus top-k (`capture_continuous_decode.py:630-647`). The
reviewer independently recomputed all 4,608 router rows; all top-k Expert sets,
selected logits, and margins matched. The current result was therefore not
invalid, but the fail-closed evaluator needed that check and a tamper test.

Claim impact: the narrow association claim is supported. All mechanism,
capacity, and method claims remain unsupported. The minimum remediation is an
append-only top-k-recomputing evaluator replay and tracker closure; the next
scientific experiment remains first-frontier pre-router hidden-state capture.
