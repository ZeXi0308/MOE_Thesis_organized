# Longrun A Reality Snapshot

Date: 2026-08-13

- Repository baseline: `origin/agent/publish-current-moe-code@203a100967f89c1e9e6cdfb2238240c4120eb314` (`Add route-conditioned capacity envelope pilot`). Remote experiment branch: `agent/longrun-A-execution-conformance`.
- Dirty boundary before this task: the isolated clone was clean. This task adds only `idea-stage/longrun_A_execution_conformance/` and a new timestamped artifact bundle. It does not modify `docs/current/README.md`.
- Process authority read: the user-supplied `AGENTS.md` in this Goal request.
- Scientific authority read: `docs/current/README.md`, `docs/ideas/route_shape_slo/v2_capacity_envelope/README.md`, `LIGHTWEIGHT_STATUS.md`, and the retained v2 artifact.
- Environment: Ubuntu 22.04, one idle NVIDIA GeForce RTX 5090 (32,607 MiB), driver 580.76.05, Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.57.6, CUDA runtime 12.8. No dependency, CUDA, driver, model, or runtime upgrade is authorized.
- Canonical prior artifact: `artifacts/route_capacity_envelope/dev/20260812T170512Z/`. It is retained because it applies the conformance veto, not because its M3 diagnostic is favourable.
- Raw captures available and hash-closed on the GPU host:
  - `/tmp/bcrd-gate0-smoke-rce-steady-20260812T170512Z`
  - `/tmp/bcrd-gate0-smoke-rce-bursty-20260812T170512Z`
- Prior measured facts: exact generated-token parity coexisted with serial/batched expert-assignment mismatch. Layer match was 96.6797% steady and 94.7266% bursty; whole-step match was 81.25% and 37.50% respectively.
- Prior capacity diagnostic: retained M3-vs-M2 P95 pinball was +9.6288%, while an uncontrolled different-Python run was -24.0388%; dangerous underprediction improved in neither run. This is a stability warning, not capacity evidence.
- Available difference anchors: six hash-bound events are recorded in `EVENT_SELECTION.json`, including layer-0 steady and bursty events and later-layer events with heterogeneous logical KV lengths.
- Missing raw state: the old capture retained token/route/batch ledgers but no KV tensor snapshot or hash. Exact reconstruction therefore requires teacher-forcing the full original batched history and demanding route/token closure at every prior step plus Arm C closure at the target step.
- One causal question: from the same reconstructed pre-step state, does the first stable difference enter at batch-width execution, heterogeneous physical KV/padding, companion permutation, pre-router hidden state, router GEMM, or only top-k near-tie selection?
- Weakest causal link: target token, logical KV, position, and target cache contents must be identical across A/B/C/D with pairwise non-aliasing storage.
- One experiment: six source-selected batched states, four arms, at least three repeat calls per arm, staged target-row hidden/router/MoE/final-logit capture, followed by serial teacher-forced propagation for four events. Source hashes must close at run time; retaining the raw capture is required for self-contained provenance.
- Allowed claim ceiling: one OLMoE BF16 model on one RTX 5090 in the custom eager cached-decode runtime. No native serving, safe-capacity, SLO-goodput, controller, EP/NCCL, or production-latency claim is allowed.
- Stop: source replay or Arm C cannot reproduce the captured route/token; arm storage aliases; same-arm tensors are unstable at cross-arm scale; GPU isolation is lost.
- Continue: source closure, non-alias, and three-repeat stability pass and a stable first-divergence location is found.
- Reopen capacity action: only after execution state is trustworthy and each `running_set_budget` is rerun action-conditionally. Fixed-route replay remains forbidden.
