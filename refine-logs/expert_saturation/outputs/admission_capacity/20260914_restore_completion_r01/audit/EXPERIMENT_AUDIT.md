# Experiment Integrity Audit — restore completion r01

- Audit date: 2026-09-14
- Auditor: fresh Codex same-family reviewer, direct read-only review
- Acceptance status: **PROVISIONAL** (same-family review)
- Integrity verdict: **PASS**
- Severity counts: **P0 = 0, P1 = 0, P2 = 1**
- Repository HEAD reviewed: `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac`

The saved result is internally reproducible and supports its stated `MEASUREMENT_ONLY / NATIVE_SERVING` ceiling. The audit found no fake ground truth, metric normalization fraud, phantom result, missing terminal cell, future-trace reuse, or material denominator error. It does not establish a quality result, a full-LTR comparison, an Oracle, a strongest-baseline win, or a new method.

## Review contract

This was one bounded targeted review. I read the declared preparation, execution, readback, result, analysis, preflight, and called source files directly. I did not use a GPU or network, did not mutate source or raw artifacts, and did not create an `.aris` trace, following the explicit task restriction.

The research question audited was narrow: whether preserving the first produced output of an actually resumed preempted request changes native request completion behavior under the frozen old-campaign workload. The weakest link was whether the implementation creates and reserves only real recovery obligations using information available at the scheduling cutoff, while retaining the original LTR 200-token/10-call accounting. The allowed claim ceiling is the single-GPU, single-model, same-backend, fixed-cohort measurement recorded in [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L1-L5) and [analysis/REPORT.md](../analysis/REPORT.md#L3-L12).

## A. Provenance and quality separation — PASS

- The four terminal cells use the same pinned model revision, BF16 dtype, seed, 32-request frozen input, token budget, KV budget, APC setting, scheduler policy, and native backend. Only the restore-completion switch and its component fields change; see [campaign.json](../preparation/pkg/campaign.json#L2-L36), [run_probe.py](../preparation/pkg/run_probe.py#L55-L99), and a returned [engine_args.json](../execution/readback/results/block0-d6-restore-off/engine_args.json#L2-L19).
- Input identity is checked before execution and request prompts are captured with per-request timing/output records; see [run_probe.py](../preparation/pkg/run_probe.py#L40-L52) and [native_capture.py](../preparation/pkg/native_capture.py#L24-L52).
- There is no external or synthetic quality ground truth. The six cross-arm output differences per block are retained rather than dropped, and the report explicitly marks quality unmeasured; see [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L24-L30).
- Both favorable and unfavorable repeats remain present. The wall/throughput/mean-completion sign flips between blocks, while 27/32 requests per pair have worse max ITL under the intervention; see [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L9-L16).

## B. Denominator and nested timing — PASS

- The metric implementation uses all planned/arrived requests and computes duration from the earliest arrival to the latest completion. Request rate is completed requests divided by that duration; no proposed saving is used to reconstruct a denominator. See [metrics.py](../preparation/pkg/metrics.py#L10-L29) and [metrics.py](../preparation/pkg/metrics.py#L40-L59).
- All four cells have `n_planned = n_arrived = n_completed = 32`, zero failed and zero unfinished requests, and each completed request returned exactly 1024 output tokens.
- Direct recomputation from each `raw.json` closed the mutually exclusive wall-time identity to floating-point precision:

| Cell | Wall s | Scheduler-inclusive s | Engine excluding scheduler s | Outside-engine s | Decision-only nested s |
|---|---:|---:|---:|---:|---:|
| block0 off | 28.008990 | 1.393360 | 26.136013 | 0.479616 | 0.330790 |
| block0 on | 30.150935 | 1.650348 | 27.760508 | 0.740080 | 0.413739 |
| block1 on | 28.612380 | 1.359028 | 26.804215 | 0.449137 | 0.335236 |
| block1 off | 29.273167 | 1.528090 | 27.050714 | 0.694363 | 0.389575 |

  `decision-only` is correctly treated as nested inside scheduler/engine time and is not added again. The stored component accounting and relation checks are in [analysis.json](../analysis/analysis.json#L629-L677), [analysis.json](../analysis/analysis.json#L6457-L6468), [analysis.json](../analysis/analysis.json#L12574-L12585), and [analysis.json](../analysis/analysis.json#L18663-L18674).
- Fresh prompt positions (98,304) and fresh decode positions (32,736) are identical across arms. The measured intervention increases recompute positions from 74,982 to 96,955 and scheduled calls from 1,872 to 1,906; the report retains these full costs rather than subtracting them from the proposed arm. See [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L18-L22).

## C. Result existence, status, numbers, and attempts — PASS

- `execution.json` records a zero return code, verified preparation/readback archive hashes, and four terminal `COMPLETE` cells. Each returned `status.json` reports 32 completed requests and no error; see [execution.json](../execution/execution.json#L2-L33) and [execution.json](../execution/execution.json#L34-L87).
- The declared bundle contains exactly four returned result cells in the intended `off/on/on/off` order. The execution log contains one start and one successful finish for each cell. No additional attempt directory is declared inside this bundle, so the audit can establish completeness for the retained campaign bundle, not for activity outside it.
- Fresh calculations from the raw request rows reproduce the reported paired values:

| Pair | Throughput change | Mean completion change | Max ITL change | Completion better/worse | Max ITL better/worse | Cross-arm outputs equal/different |
|---|---:|---:|---:|---:|---:|---:|
| block0 on vs off | -7.104078% | +8.299740% | -2.281955 s | 0 / 32 | 5 / 27 | 26 / 6 |
| block1 on vs off | +2.309444% | -1.426265% | -2.745442 s | 32 / 0 | 5 / 27 | 26 / 6 |

  These agree with [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L9-L16) and the stored pair analysis at [analysis.json](../analysis/analysis.json#L24135-L24169) and [analysis.json](../analysis/analysis.json#L24467-L24501).
- All package and readback `SHA256SUMS` entries verify. The preparation and returned package inventories agree for all declared files; the only extra returned content is generated Python bytecode outside the signed inventory.

## D. Called code and dead-code risk — PASS

- I imported the current [analyze_restore_completion_probe.py](../../../../experiments/admission_capacity/analyze_restore_completion_probe.py#L133-L213) without writing bytecode and reran its `inspect` function on all four raw cells using the frozen source/metadata/reference inputs. Every recomputed cell object, excluding only the transient `_outputs` helper, exactly equals the saved `analysis/analysis.json` object.
- The called analyzer chain is live rather than decorative: the restore analyzer imports [analyze_ltr_component_probe.py](../../../../experiments/admission_capacity/analyze_ltr_component_probe.py#L15-L17) for base accounting and LTR helpers, imports [analyze_ltr_packing_probe.py](../../../../experiments/admission_capacity/analyze_ltr_packing_probe.py#L12-L16) for residency and exact planner replay, and thereby calls [analyze_prefix_cache_baseline.py](../../../../experiments/admission_capacity/analyze_prefix_cache_baseline.py#L134-L223) for raw work accounting. The restore-specific calls appear at [analyze_restore_completion_probe.py](../../../../experiments/admission_capacity/analyze_restore_completion_probe.py#L176-L208).
- The recomputation replayed 1,872 + 1,906 + 1,906 + 1,872 = **7,556** planner and ledger steps. Every recorded planned token list equals the actual scheduled token list. The saved exact-plan checks are also `true`; examples are [analysis.json](../analysis/analysis.json#L5228-L5232) and [analysis.json](../analysis/analysis.json#L11306-L11310).
- The runtime wrapper installs the component before the measured generate call, records actual scheduled tokens and victims, feeds actual selected requests back into both restore and original LTR state, and uninstalls afterward. See [ltr_recompute_native.py](../preparation/pkg/ltr_recompute_native.py#L147-L184), [ltr_recompute_native.py](../preparation/pkg/ltr_recompute_native.py#L185-L224), and [ltr_recompute_native.py](../preparation/pkg/ltr_recompute_native.py#L233-L238).
- The six targeted CPU unit tests in [test_restore_obligation.py](../../../../experiments/admission_capacity/test_restore_obligation.py#L18-L61) pass under `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v`. They cover output-cutoff release, completion/disappearance, initial prefill exclusion, independent obligations, and OFF/ON interruption behavior.

## E. Scope, strongest baseline, and cohort — PASS WITH EXPLICIT LIMIT

- The comparison is a same-cohort component ablation against the frozen old LTR behavior. It is a fair baseline for isolating this one restore rule.
- It is not the full strongest-baseline ladder. Native FCFS, least-work, most-work, a fresh independent cohort, full LTR reruns, and an Oracle are absent. The report says so directly and does not convert the component result into a method win; see [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L26-L34).
- The localization addendum correctly notes that timing had already diverged before the first scheduling-action divergence and therefore does not assign all observed timing or recompute differences to the first restore action. See [analysis/localization/REPORT.md](../analysis/localization/REPORT.md#L9-L19).

## F. Evaluation classification — PASS

- Evidence type: `NATIVE_SERVING` execution measurement on one RTX 5090 with the pinned OLMoE/vLLM configuration.
- Result class: `MEASUREMENT_ONLY` with no external quality ground truth. For the audit taxonomy this is best described as a real native-system self-measurement, not a quality evaluation.
- Acceptance remains provisional because the reviewer is from the same model family. The audit does not upgrade the result to formal independent acceptance.

## Targeted causal checks

### Past-only cutoff and identity

`begin_schedule` reads only the previous completed/output state; output or completion observed during a call releases an obligation at the next call boundary. An obligation starts only when a request was `PREEMPTED` before the call, is actually resumed, already has positive output, and still has pending work. See [restore_obligation.py](../preparation/pkg/restore_obligation.py#L27-L63) and [restore_obligation.py](../preparation/pkg/restore_obligation.py#L79-L116). The CPU first-action preflight stops at the first proposed scheduling difference and explicitly makes no future-performance claim; see [first_action.json](../preparation_checks/first_action.json) and [check_first_action.py](../preparation_checks/check_first_action.py#L20-L49).

### Actual action, victim handling, and pool reservation

- Each OFF cell created 25 actual obligations: 19 released after a new output and 6 released as interruptions. Each ON cell created 27 actual obligations: all released after new output, with zero interruption. These counts match both raw ledger replay and [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L18-L20).
- ON scheduled the `-2` restore overlay 81 times. Under that overlay, the original LTR priority remained 0 for 54 calls and -1 for 27 calls, confirming the overlay did not overwrite the original recovery-service counters.
- For every protected ON scheduling decision, the live free cache-block pool was at least the sum of the protected requests' recorded remaining-history requirement; the tightest slack was exactly zero. No protected obligation was selected as an actual victim. The implementation checks live pool state before scheduling and errors on an enabled protected preemption; see [restore_obligation.py](../preparation/pkg/restore_obligation.py#L64-L75) and [restore_obligation.py](../preparation/pkg/restore_obligation.py#L96-L105).
- The original LTR 200-token threshold and 10-call quantum remain active and are updated from actual scheduling after the restore overlay. See [recovery_service_components.py](../preparation/pkg/recovery_service_components.py#L35-L74) and [ltr_recompute_native.py](../preparation/pkg/ltr_recompute_native.py#L205-L224). OFF has eight LTR epochs per cell and ON has nine; every epoch reaches 10 selected calls, produces seven outputs, and has no quantum exhaustion without output.

### Independent policy state and preemption-bounded service

Each cell runs in its own Python process and reloads its own engine/model state; [run.sh](../preparation/pkg/run.sh#L13-L24) enforces the fixed off/on/on/off sequence and refuses to overwrite an existing result directory. No future route, victim, queue, output, or completion trace is shared between policies.

For resumed requests that were preempted again, OFF has 18 measured segments with 6/0/0 segments producing exactly 0/1/2 outputs; ON has 19 with 0/4/0. The four one-output ON segments are retained rather than counted as complete service. These values agree with [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L18-L22).

## Finding P2-1 — preparation-era status and inherited campaign prose remain visible

The original [REPORT.md](../REPORT.md#L1-L25), [EXECUTION_CONTRACT_ADDENDUM.md](../EXECUTION_CONTRACT_ADDENDUM.md#L1-L17), and qualification metadata describe the pre-execution `GPU_UNRUN / NOT_YET_RUN` state. Separately, `campaign.json` retains an inherited rank-prefix/fit-scan question and interpretation even though its cell list, comparison type, CLI, runtime config, and returned hashes encode the restore-completion experiment.

This is a documentation/provenance clarity issue, not a result-integrity failure. [RESULTS_ADDENDUM.md](../RESULTS_ADDENDUM.md#L42-L42) explicitly states that preparation metadata and inherited campaign wording are historical and identifies the actual execution contract. Canonical readers should use `RESULTS_ADDENDUM.md`, `analysis/REPORT.md`, `analysis/analysis.json`, and `execution/execution.json` for the terminal state. No raw artifact should be rewritten to remove the historical record.

## Claim impact and required next evidence

The audited evidence answers the narrow question: the restore rule was executed as specified and changes actual recovery behavior, but its complete-request performance effect is unstable across the two retained blocks. It reduces the single worst observed ITL in both blocks while worsening max ITL for 27/32 individual requests per pair, increases recompute work, and produces opposite signs for throughput and mean completion. Therefore the correct verdict remains `MEASUREMENT_ONLY`; there is no stable net method win.

No P0/P1 correction is required, and the audit stop rule is satisfied after this targeted review. Before any stronger claim, the next smallest evidence should be a fresh cohort with in-group native/least-work/most-work baselines and the same complete-request, output-difference, and full-cost accounting. Quality must be evaluated separately if output differences are used in a usefulness claim.

## Audited hashes

| Artifact | SHA-256 |
|---|---|
| `REPORT.md` | `0dbb5338bd85d58b40d4ab4e739d71f3019e08c8ba62c15f33d5fe247ee5fc8d` |
| `EXECUTION_CONTRACT_ADDENDUM.md` | `803a77d4aae62b95eeb32c2a8e77a4868e32401d041de21be2360170f63eb821` |
| `RESULTS_ADDENDUM.md` | `737ff1c4fc99aa72bdd60eb142064497668820d23f427a64d65af770f5b656f0` |
| `analysis/analysis.json` | `b8fc2b1e7142b47d3e010bdff88c4a6418d33bf5111911120ada22aec2a26810` |
| `preparation/preparation.json` | `509987428af6119f3406e1ee61bd883b4c01d746c844bafd9bda36d91b2abebc` |
| `preparation/execution.tar.gz` | `c90cef5949deb6a2263d8fffe1a22af12c3bba5a3178d5f3e0929f401fdef2fd` |
| `execution/execution.json` | `c244ce5f3d9ab97d506f5c16764af1585601f1a032785dd4c42ff6b8368e2bc0` |
| `execution/readback.tar.gz` | `bbff5317b8577abdb1f5ba7d651ef21035b9dbca77045285201ae3b110150c5d` |
| `preparation_checks/first_action.json` | `42626236e236eb786ec6c0f14db0ae14b9721c0346b577afa734d1a32f5cbadb` |
| `preparation_checks/check_first_action.py` | `db5989f4515bd0cb739a3d40351e0a3f8e9018047d16e1f555ff9f2be6e699ff` |
| `preparation_checks/smoke/component_accounting.json` | `3399238d3615fab2b3651ad15828e845662ffca7cde06823d308627d8cc0dc4a` |
| `prepare_restore_completion.py` | `adc38b2de272bcb61ef66ef74c20ba4c109057ebf07bc412b7c82fb69f40c10b` |
| `analyze_restore_completion_probe.py` | `d536a7b97d50eca3e538b7861d31a408af2cb1b8cf6f5c584df3f15266809913` |
| `analyze_ltr_packing_probe.py` | `7a88ba39a17e536f87b60788023b3042d5b9366853d32c20fd214fd5b8bbfb38` |
| `analyze_ltr_component_probe.py` | `342738deed09c6fb585137a81ed5cd3186abaa54445d0443dbbed3669ccbd1a6` |
| `analyze_prefix_cache_baseline.py` | `590ee20ee3648bb097abf82f29b23ac1e249ca588216bba4b0b13d713d18a473` |
| `restore_obligation.py` | `54f4dc62427c1c2b6c285cb6b7d22fe6f4563b4970d37a15acf407d764d42c45` |
| `test_restore_obligation.py` | `f11c2ddc07b607075a4f96b190d76507bafb2603bbf2dc97da199b64335640b9` |
| block0 off `raw.json` | `d45ae8572f10d8168a5ece8738dca0e8980307484ef24b85b1e8d8ecccca69f0` |
| block0 on `raw.json` | `e153796a1597762b2b71b172f04fcededa010572acb25e45667f1fbc89f72dfe` |
| block1 on `raw.json` | `41821c1bc3fd456a145fa2b321f40b9da6f9938f8af102bec38d3f078cb5dfae` |
| block1 off `raw.json` | `d0e46d1b75b182d0308e0dea585560deda5f9fe07c2be22a1a3329e537daa10f` |
