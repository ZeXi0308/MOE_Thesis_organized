## Overall verdict: FAIL

`reason_code: FAIL_OPEN_TELEMETRY_GATES_AND_UNVERIFIABLE_PRODUCER_SOURCE`

当前两个 canonical JSON 的数值可从现有 sealed artifacts 重算，且文档没有把 fixed-route replay、token-drift timing 或 UNRUN 分支冒充 action evidence。失败来自评估器本身：多个状态机能产生假阳性，且五个 GPU bundle 的精确 producer source 已丢失。

### A. Ground-truth/reference provenance — WARN

Evidence:

- workload 标明 WikiText-103 来源、revision、split 和选择规则：`remote_snapshot_20260823/runs/full-on-r0-20260823/workload_manifest.json:2-13`。
- runner 实际只读取 `requests[].prompt`，忽略 arrival/request identity，并重新拼接成固定长度 prompts：`experiments/run_vllm_route_shape_probe.py:202-222,330-335,431-455`。
- route、token、timing 均为模型/runtime 输出，不是 dataset GT：`run_vllm_route_shape_probe.py:480-520`。
- fixed-route、自监督属性已明确标注：`native_route_regrouping_diagnostic_v3_final2.json::$.anti_claims[2:12]`，对应文件 `:6-15`。

Details:

- 未发现 unlabeled self-derived GT 或人工 GT。
- 分类应是 `model-output observational systems diagnostic`；regrouping 是 `self_supervised fixed-trace proxy`。
- provenance 未完全闭合：所有五个 full bundle 的 `config.json:25` 记录 producer SHA `fa20398f…`, 当前唯一 runner SHA 为 `a680ab37…`，workspace 内不存在 `fa20398f…` 对应源码。旧 bundle 也缺少当前 runner 在 `:273-327,356-374` 要求的两个 `vllm_runtime_sources` hash。

### B. Score normalization — PASS

Evidence:

- route concentration、active fraction、entropy 等使用 assignment count、expert count、`log(num_experts)` 等物理分母：`run_vllm_route_shape_probe.py:77-145`。
- timing 百分比使用独立 route-OFF 值作分母，并保留 raw ON/OFF 指标：`compare_vllm_route_probe_runs.py:145-163,217-239`。
- regrouping JSON 同时保留 raw policy metrics、before/after 和 derived percentage，例如 `$.trajectory_analysis.history_vs_strongest_route_blind`，文件 `:212624-212648`。

Details:

- 未发现用模型输出自身 max/min/mean 归一化制造近 1/100%。
- `route_set_jaccard_median=1.0` 是集合精确一致性，不是自归一化分数：`native_route_pivot_analysis_v1.json:572-581`。

### C. Result existence and claim matching — FAIL

Evidence supporting existing results:

- 五个 full bundles 的 303/303 manifest-listed artifacts、五个 RUN_COMPLETE seals、288 个 NPZ 均通过 hash/shape/range/denominator 检查。
- `native_route_pivot_analysis_v1.json` 可重算为相同对象；其状态为 `COMPLETE / WORKING_SET_MEASUREMENT_ONLY / TELEMETRY_TRANSPARENCY_FAILED`：`$.status`, `$.pivot_verdict`, `$.failure_category`，文件 `:151-159,570-592`。
- regrouping canonical JSON 可 byte-exact 重算，SHA `da5280…a940`，与 addendum `:5-12` 一致；旧 v1/v2/v3 artifacts 明确 superseded。
- tracker 诚实保持 `UNRUN / NO_METHOD_GO`，且明确无 capacity/SLO-goodput/controller 结果：`EXPERIMENT_TRACKER.md:3-17,134-140`。
- decode-cap 明确 `UNRUN`：`DECODE_CAP_BRANCH_GATE.md:3-15`。

Failure:

- Seal 证明当前文件自洽，不证明其 producer/runtime 来源。旧 producer SHA 无源码，且旧 config/runtime identity 没有 vLLM source hashes：`full-on-r0-20260823/config.json:25,33-41`、`environment.json:1-10`。
- 整个 `refine-logs/expert_saturation/` 当前未跟踪；HEAD 为 `b141c1d…`。权威文件要求新正式结果同步更新 `docs/current/README.md`，否则只算来源快照：`docs/current/README.md:205-220`；当前 authority 仍称 RouteShape-SLO blocked：`docs/ideas/README.md:18-24`。

### D. Dead code and fail-open validation — FAIL

P0/P1 findings:

1. `compare_vllm_telemetry_implementations.py` 在零个可比较 route 且跨实现 token drift 时仍返回 `VALID_WINDOW_TELEMETRY_QUALIFIED`。

   - drift cells 被跳过：`:99-127`。
   - `comparable_cells`、`exact_route_cells`、`token_drift_keys` 只输出不判定：`:128-164`。
   - 实测反例：`comparable_cells=0`, `token_drift_keys=[[128,4,0,0]]` → `('VALID_WINDOW_TELEMETRY_QUALIFIED', None)`。
   - tests 未覆盖：`test_compare_vllm_telemetry_implementations.py:31-70`。

2. implementation identity 只靠可填写标签；真实 source hashes 被主动删除。

   - source hashes 被剥离：`compare_vllm_telemetry_implementations.py:66-85`。
   - 只要求 `runtime_patch_id` 字符串不同：`:212-220`。
   - 精确 original/patched hashes 仅存在于未集成的 validator：`vllm_patches/validate_valid_window_patch.py:12-22,29-67`。
   - `stock_pair` 在 `:229-235` 计算，却未传入 verdict：`:243-252`。

3. route-probe analyzer 允许事后挑选 token-clean repeats。

   - drift repeat 被剔除：`analyze_vllm_route_probe_bundles.py:222-251`。
   - 两个 clean repeats 可形成 `PAIRED_ROUTE_OFF_QUALIFIED_SUBSET`：`:449-455`。
   - 即使总体 `FAILED_TOKEN_DRIFT`，仍可能输出 `TEST_MARGINAL_PRESSURE_ACTION`：`:456-487`。
   - 反例：3 repeats 中 2 clean、1 drift，同时得到 `FAILED_TOKEN_DRIFT + QUALIFIED_SUBSET + TEST_MARGINAL_PRESSURE_ACTION`。
   - 违反冻结规则：`VALID_WINDOW_TELEMETRY_GATE.md:65-74`。

4. `compare_vllm_route_probe_runs.py` 也存在 fail-open。

   - overhead 仅做单边 `max(P95) <= 5%`：`:182-191`；大幅负漂移如 `-50%/-20%` 会被判 qualified，尽管它同样说明 ON/OFF 不可交换。
   - `verify_bundle()` 只检查主动列入 manifest 的 artifacts，不要求每行 input/route artifact 完备，也不重算 raw timing/route metrics：`:54-95`。
   - `MATCHED_CONFIG_FIELDS` 未包含 `require_exclusive_gpu`，也不要求非空 patch ID 或 runtime source hashes：`:17-39`。
   - CLI 对 `INVALID_TELEMETRY_PAIR` 仍正常退出：`:243-273`。

5. valid-window comparator 的 “route semantics” 只做 `np.load + array_equal`，没有 shape、expert range、top-k、row hash 检查：`compare_vllm_telemetry_implementations.py:99-134`。两个相同的 shape `(1,)`、值 `-999` 的数组会被计作 exact，违反 `VALID_WINDOW_TELEMETRY_GATE.md:65-70`。

6. decode-cap analyzer 同样单边接受大幅负 telemetry drift：`analyze_vllm_decode_cap_branches.py:282-304`；测试只覆盖 `+>5%`：`test_analyze_vllm_decode_cap_branches.py:103-111`。它还直接信任 sealed `summary.route_pressure` 与 timing，而不从 `routes.npz`、`requests.jsonl` 重算：`:53-100,143-169,290-296`。

7. 四臂 Gate 实际只接收每 arm 一个目录：`compare_vllm_telemetry_implementations.py:167-179,272-288`，无法实现文档要求的两个 controlled repeats：`VALID_WINDOW_TELEMETRY_GATE.md:51-65`。

全部 36 个现有单测可通过；这说明上述反例是测试盲点，而非已覆盖行为。

### E. Scope — WARN

Actual scope:

- 1 model/revision：OLMoE `6d84…`
- 1 GPU/runtime host：RTX 5090，vLLM 0.26 eager，BF16
- route-ON process runs：3；matched ON/OFF repeats：2，二者均有 token drift
- prompt regimes：P128/P512，但为 prefix-related views
- batch sizes：4/8/16，且 B4⊂B8⊂B16
- synthesized pool：96 prompt starts；1 WikiText pool
- 180 fixed-batch generate calls、1680 request executions，但不是独立 workload samples
- regrouping：90 correlated step cells → 6 trajectory summaries；这 6 个也非独立 workloads
- GPU action-conditioned runs：0
- valid-window optimized GPU runs：0
- decode-cap branch runs：0
- multi-GPU/EP runs：0

Evidence:

- correlated-cell/prefix boundary：`native_route_regrouping_diagnostic_v3_final2.json::$.anti_claims[12:14]`, `$.step_level_diagnostics.role`, `$.trajectory_analysis.independence_warning`; 文件 `:13-15,212470,212593,212607-212650`。
- addendum 同样限定：`REGROUPING_ORACLE_ADDENDUM.md:16-30`。
- four-arm single-repeat qualification缺陷见 D。

当前文档措辞总体没有越界；WARN 来自样本相关性和 evaluator 未执行冻结的 repeat scope。

### F. Evaluation classification — PASS

- Native pivot: `non-GT real-system observational measurement`，但 ON/OFF timing join 无效。
- Regrouping: `self_supervised_proxy / structural fixed-route simulation-replay`。
- Valid-window review: `static source review`; GPU/runtime/async qualification `UNRUN`。本地 patch review 与 vLLM 官方 commit 的 capturer 和 model runner 一致。
- Decode-cap: `UNRUN prospective native action-conditioned experiment`。
- Unit tests: `synthetic_only`。
- human_eval / real_gt: none.

## Claim impact

- `WORKING_SET_MEASUREMENT_ONLY`: numerical result survives as a sealed, self-consistent observational diagnostic; downgrade reproducibility to `producer-source-unverified`.
- ON/OFF timing/overhead: remains invalid. r0 drift `1/36`, wall/TPOT P95 `22.95%/29.23%`; r1 drift `6/36`, `4.96%/15.48%`: `native_route_pivot_analysis_v1.json::$.telemetry_pairs`, lines `594-658`; `$.telemetry_transparency.status=FAILED_TOKEN_DRIFT`, `:660-667`.
- Fixed-trace regrouping: survives only as structural diagnostic. `t-1` median `-0.0969%`; hindsight `7.0892%`; coalescing `25.4902%` with HHI degradation `-45.9789%`: JSON `$.trajectory_analysis`, lines `212607-212665,215010-215046`. No action/controller implication.
- `SOURCE_SAFE_WITHIN_EXISTING_CAPTURE_SUPPORT_ENVELOPE`: retain only static source-review ceiling; async/GPU safety remains unrun. Patch test named “async” only performs synchronous clone/clear: patch `:97-109`.
- Any future `VALID_WINDOW_TELEMETRY_QUALIFIED`: unsupported until the P0/P1 paths are fixed.
- Any future clean-subset `TEST_MARGINAL_PRESSURE_ACTION`: invalid post-hoc selection.
- Decode-cap/headroom/SLO/controller: no measured claim exists; stays UNRUN.
- `PROTOCOL_FROZEN_FOR_FINAL_REVIEW / MASTER-THESIS-VIABLE`: planning judgment, not method evidence; `round-2-refinement.md:246-250` itself says no method GO.

## Required actions

1. Recover and seal exact `fa20398f…` producer source, or rerun; include exact vLLM source hashes, patch-validator report, Git commit/dirty diff, and process-isolation evidence.
2. Make any token drift in any required repeat fail the whole Gate; remove qualified-subset action selection.
3. Require nonzero/full route coverage, raw route shape/range/top-k/hash checks, and recompute every decision metric from sealed NPZ/JSONL.
4. Bind stock/optimized identity to validator-approved source hashes; include `stock_pair` in verdict.
5. Require two controlled repeats and a two-sided transparency/noise guard; reject large negative as well as positive drift.
6. Make every scientific Gate failure return nonzero; add adversarial tests for zero-comparable routes, token-drift subset, label-only identity, malformed routes, missing artifacts, negative overhead and single-repeat input.
7. Keep tracker at `UNRUN / NO_METHOD_GO`; do not run N0/decode-cap until the evaluators fail closed.

## Reviewed files

All 15 requested scripts/tests/validator/patch; both result JSONs; all five requested full run bundles including 328 files/288 NPZ; `EXPERIMENT_TRACKER.md`; `REFINEMENT_REPORT.md`; `round-2-refinement.md`; `REGROUPING_ORACLE_ADDENDUM.md`; `VALID_WINDOW_REVIEW.md`; `VALID_WINDOW_TELEMETRY_GATE.md`; `DECODE_CAP_BRANCH_GATE.md`; plus authority files `docs/current/README.md` and `docs/ideas/README.md`. No file was modified.
