# RouteSlack-MoE 最终裁决

## 1. Verdict

```text
MEASUREMENT_ONLY
```

Gate 0 仍有 9 个开放 P0，formal physical strategy sample=0；当前成果只能作为 measurement/protocol characterization，不能支持 controller 或 `8xA100_CANDIDATE`。

## 2. Research audit

- `[Observed]` 旧 `--phase decode` 不是 KV-cache decode；本轮 development path 已改为 prefill 后逐 token 传入 `past_key_values`，并通过 tiny CPU cached-vs-full 测试。
- `[Observed]` `ask` 作者历史检索因 git-ai daemon lock 未取得可用会话，因此不引用作者意图；“旧路径不是 continuous-serving producer”只由历史代码和 metadata 证明。
- `[Blocked]` H1：没有双模型 natural rows×tier raw-energy surface，route effect 仍无法与 rows、batch、KV、phase、utilization 和 thermal state 分离。
- `[Blocked]` H2：fixed replica、bounded sealing、power tier 和 dispatch order 尚未进入真实 deadline/EP executor。
- `[Blocked]` H3：没有 matched-completion energy Oracle，也没有同 trace/SLO 的强 simple baseline 物理比较。
- `[Inferred]` H1/H2/H3 均未被物理证实，也未被物理证伪；缺硬件和开放 P0 不是“收益为零”。

## 3. Code review

- P0：16 个，代码级关闭 7 个，开放 9 个。
- P1：8 个；P2：2 个。
- 已修复：full forward 冒充 decode、隐式 route identity、latency-only curve 的 formal 误用、census provenance fail-open、sampler 静默失败、无 metadata 的 counter wrap、JouleQueue source-hash 崩溃、跨层 completed-token 重复计数。
- 未修复阻塞：真实 continuous-serving/stage ledger、双模型 physical service-energy surface、matched energy window、thermal/UUID/tier、合法 EP actuator、conservative energy Oracle、10 个真实 baseline、全链 provenance/split/remote-cell join 和两模型 formal-serving exactness。development cached-decode parity 已通过，但不具有正式外推效力。

详表见 `docs/current/routeslack_code_review.md`。任何开放 P0 都使 Gate 0 保持 `FAIL`。

## 4. Experiment protocol

- independent unit：surface 为 fresh input event/document；serving 为 request/input event；AB/BA block 是 thermal pairing unit；inner repeat 不是样本。
- measurement window：同 trial 保存 host monotonic wall、正确 stream CUDA events、stage/E2E timeline，以及 workload/counter/power-sample 原始边界。
- energy denominator：主指标为 `raw board J / exact matched SLO-completed output token`；idle-adjusted dynamic J/token 仅为 sensitivity。
- models：`allenai/OLMoE-1B-7B-0924@6d84c48581ece794365f2b8e9cfb043c68ade9c5` 与 `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M@1d5983076dfc67aee4a77ec06a27027f5bab6055`。
- workload：natural continuous-arrival、真实 cached one-token decode；每 model/load cell 至少 128 个独立 input event；synthetic 只做 sensitivity/dry-run。
- SLO：独立 calibration 中 unmodified default P99 × 1.10，按模型/负载 bucket 写入 config 后冻结。
- baselines：10 个冻结 online baseline 加 1 个类型隔离的 future-known Oracle。
- preregistered thresholds：同一 common natural cell effect ≥10%、95% LCB >5%，至少一个 common cell ≥15%；actionable natural energy mass ≥20%；Oracle net ≥10%；strongest-simple capture <90%；controller tax ≤ gross saving 的 20%；两模型 AND。

详见 `docs/current/routeslack_experiment_protocol.md`。

## 5. Results

- `[Observed]` 当前 protocol-critical bundle tests 124/124 PASS：BCRD 20、RouteSlack 59、route-row contracts 17、JouleQueue 28。另一次远端 broad discovery 为 141/141（route-row 额外发现 18 项）。
- `[Observed]` tiny route-v2 capture 为 8 行：1 request、2 decode steps、2 layers、top-2；metadata 为 non-formal。
- `[Observed]` synthetic identity ledger 为 16 contributions/stage × 4 stages；10 baseline 名称和 Oracle 只完成接口/管线调用，未完成真实算法或物理对比。
- `[Observed]` out-of-range surface 返回 `FALLBACK_DEFAULT`，`action_eligible=false`。
- `[Observed]` CPU decision fixture P50=0.116479 µs/call、P99=0.194960 µs/call；相对 empty-loop 的 paired mean increment=0.095192 µs，95% CI=[0.084725, 0.106695]。
- `[Observed]` CPU JSON logging fixture P50=2.896584 µs/call、P99=3.293015 µs/call；paired mean increment=2.905731 µs，95% CI=[2.855718, 2.965092]。
- `[Blocked]` physical latency N=0、physical energy N=0、physical paired difference=`N/A`、formal 95% CI=`N/A`。
- `[Observed]` canonical artifact：`artifacts/energy_slo_routeslack/20260728_115300/`，manifest SHA-256 `9c661c0bb90fbffd2cfc99b34d798feb04455cc9160e76bd9f610a57a94bde7c`。
- `[Observed]` 最终报告快照/关键源文件/GPU fail-closed 日志的 supporting bundle：`artifacts/energy_slo_routeslack/20260728_120340/`。
- `[Observed]` sealed GPU qualification：`artifacts/energy_slo_routeslack_gpu/20260728_144600/`，manifest SHA-256 `69e6303bfaadaec93bfa6f15fee0abe154325c0ce097e8d58914d9fc197a0f37`；29/29 文件远端与下载后复算一致。
- `[Observed]` 该 bundle 在 RTX 5090 上重跑 96/96 protocol-critical tests；LLM-jp 与 OLMoE 的 4-step batch-1 cached-decode logits/argmax/KV length exactness 全通过、误差为 0，development route capture 为 2,048/1,024 contributions。
- `[Observed]` NVML capability probe 的 299 点 max gap=10.925 ms、counter Δ=1,724.861 J；温度 35→58°C（Δ23°C）使 formal thermal gate 明确 FAIL。它不是 model/service strategy energy；physical model energy N 仍为 0。
- `[Observed]` latest RTX 5090 development bundle：`artifacts/energy_slo_routeslack/20260728_151500_rtx5090_dev/`，manifest SHA-256 `d23f18a0455c4ebd04f3611746bc8150eac674ba1bd97e91d24ee5e7517f05c8`。GPU/NVML 五项预检 PASS；OLMoE 和 LLM-jp 固定 revision parity 均 PASS，对照误差为 0。
- `[Observed]` 独立 parity/failed-energy follow-up：`artifacts/energy_slo_routeslack/20260728_150422_gpu_followup/`，artifact manifest SHA-256 `e588eafb78926fe35abdfa15727d414d50498e4fe0ba04568abf34621bbf654e`；19/19 声明文件无缺失、无 hash 漂移，accepted energy N=0。
- `[Observed]` 最终 supporting dry-run：`artifacts/energy_slo_routeslack/20260728_151412_final_support/`，manifest SHA-256 `b3e51f274f34148f477a1f6392dc7eb5620f733ba054faea97c690e84d501da1`；17/17 文件闭合并重新确认 123/123 tests PASS、`formal_result=false`、`Gate0=FAIL`。
- `[Blocked]` latest GPU dev bundle 中的 synthetic ABBA 两次都在 12 windows 后检出竞争 CUDA 进程并 fail closed；独立 follow-up 另为 0/12 completed windows 后拒绝。所有尝试的可接受 energy sample 合计仍为 0，不报告 A/B effect。
- `[Observed]` current-validator isolated-expert characterization：`artifacts/energy_slo_routeslack/20260728_154500_current_energy_characterization/`。LLM-jp 11/16 窗有效：rows 1/8/32/128 的 raw J/expert-row mean 分别为 0.008206、0.001004、0.000280、0.0000856，95% CI 分别为 [0.008120,0.008355]、[0.000977,0.001019]、[0.000275,0.000285]、[0.0000838,0.0000870]。
- `[Observed]` OLMoE 仅 4/16 窗有效，rows 1/8/32 的 raw J/expert-row mean 为 0.007445、0.000869、0.000278；对应 valid N 为 2/1/1，rows=128 为 0。因此 OLMoE 状态是 `CHARACTERIZATION_INCOMPLETE_INVALID_WINDOWS`，`N=1` 的退化 CI 不作为统计证据。
- `[Blocked]` 上述数值是 default-tier、prefill activation、单 expert/expert-row denominator，不是 natural route、J/SLO-completed-token 或策略 A/B。formal physical strategy latency/energy N 仍为 0，H1 两模型 AND 未达到。

## 6. Gate table

```text
Gate 0: FAIL
Gate 1: FAIL
Gate 2: FAIL
Gate 3: FAIL
Gate 4: NOT RUN
```

Gate 1–3 的 `FAIL` 表示 Gate 0 顺序阻断且没有达到 PASS 条件，不表示 H1–H3 已被物理反证。

## 7. Simple baseline versus Oracle

```text
E_default = N/A
E_strongest_simple = N/A
E_oracle = N/A
CaptureRatio = N/A
physical sample count = 0
95% CI = N/A
```

synthetic fixture 的 `cost_units` 不是 Joules，不能代入 CaptureRatio 或报告为 baseline 捕获率。

## 8. Scientific limitations

单卡 development capture 不能证明：

- route-conditioned energy variation 独立于 batch、rows、KV、phase、utilization 和 thermal state；
- 真实 expert-parallel assignment、rank slack、A2A/NCCL/RDMA 或通信计算 overlap；
- dispatch、execute、combine 与跨层依赖的端到端 TPOT/P99；
- multi-board idle、tier transition、switching/sealing/controller tax；
- 两模型 AND、Oracle net ≥10% 或 strongest-simple capture <90%。

因此本轮没有形成可外推到 8×A100 的正面系统结论。

## 9. Code changes

- `capture_native_routes.py`：cached one-token decode、step route capture、EOS/max-step 和 non-formal metadata。
- `core.py`：route-v2 显式 identity、semantic key、四阶段 conservation 和 strict loader。
- `benchmark_expert_service_curve.py`、`census_fragmentation.py`：阻止 latency-only、smoke/development provenance 升格为 formal Gate。
- `test_cached_decode.py`、`test_identity_conservation.py`、`test_bcrd_gate.py`：新增 cache/route/provenance 反例测试。
- `power_accounting.py` 及测试：传播 sampler 线程异常，显式 counter-wrap modulus，校验 raw/dynamic completed-token denominator。
- JouleQueue capture 及测试：修复 source-hash `NameError`。
- `docs/ideas/energy_slo/routeslack/experiments/`：新增 Gate-0 contracts、online/Oracle 类型隔离、fallback、paired accounting、dry-run 和 timestamped artifact bundle。
- `run_gpu_meter_preflight.py`、`run_frozen_model_gpu_exactness.py`、`finalize_gpu_gate0_bundle.py`：新增 non-formal 单卡 meter/exactness 资格验证与 fail-closed 密封；三者已复制进 sealed bundle 的 `source_snapshot/`。
- `run_model_patch_parity_probe.py`：对冻结 revision 执行 native/shared-full patch 的 prefill、forced cached decode、route、KV 一致性验证；始终 non-formal。
- `run_rtx5090_development_probe.py`：同执行窗口的 CUDA event/NVML raw energy、统一 repeat、ABBA、UUID/thermal telemetry 与竞争进程 fail-closed；只允许 development characterization。
- `run_rtx5090_energy_characterization.py`：加入实际 workload ≥10 s、完整/filtered/incomplete 状态和逐窗 thermal/gap fail-close；本轮两模型 physical run 仍硬编码 non-formal。
- artifact 保存 base commit、dirty status、git diff、命令、环境、seed、raw data 和逐文件 SHA-256；当前修改未提交、未推送，不能声称已发布。

## 10. Reproduction

```bash
cd '/Users/leandrozhao/Desktop/毕设论文资料'

# 若从干净 Python 环境开始；CUDA 主机需安装与驱动匹配的 CUDA torch wheel。
python3 -m venv .venv
./.venv/bin/python -m pip install -r experiments/shared/requirements.txt

# 1. 全部协议测试
./.venv/bin/python -B -m unittest discover -s docs/ideas/bcrd/experiments -p 'test_*.py'
./.venv/bin/python -B -m unittest discover -s docs/ideas/energy_slo/routeslack/experiments -p 'test_*.py'
./.venv/bin/python -B -m unittest \
  docs.ideas.energy_slo.route_row_fp8.experiments.test_continuous_decode_harness \
  docs.ideas.energy_slo.route_row_fp8.experiments.test_power_accounting
./.venv/bin/python -B -m unittest discover \
  -s docs/ideas/energy_slo/joulequeue/experiments -p 'test_*.py'

# 2. development-only cached decode；去掉 --offline 才会联网下载未缓存模型。
./.venv/bin/python -B docs/ideas/bcrd/experiments/capture_native_routes.py \
  --model jamesdborin/tiny-mixtral --model-key tiny-mixtral-dev \
  --dataset builtin --samples 1 --seq-len 8 --decode-steps 2 \
  --dtype float32 --phase decode --offline --allow-cpu \
  --output /private/tmp/routeslack_tiny_decode_v2.csv

# 3. 新 timestamped fail-closed dry-run；不是 formal result。
ROUTESLACK_RUN_ID="$(date +%Y%m%d_%H%M%S)"
ROUTESLACK_OUT="$PWD/artifacts/energy_slo_routeslack/$ROUTESLACK_RUN_ID"
./.venv/bin/python -B \
  docs/ideas/energy_slo/routeslack/experiments/run_routeslack_dry_run.py \
  --output-dir "$ROUTESLACK_OUT" --seed 20260728 --run-tests \
  --include-file \
  "/private/tmp/routeslack_tiny_decode_v2.csv=raw/development_tiny_cached_decode_v2.csv" \
  --include-file \
  "/private/tmp/routeslack_tiny_decode_v2.meta.json=raw/development_tiny_cached_decode_v2.meta.json"

# 4. 单 GPU development-only parity；必须使用 40-hex revision，不能外推为 serving/EP。
python3 docs/ideas/energy_slo/routeslack/experiments/run_model_patch_parity_probe.py \
  --model allenai/OLMoE-1B-7B-0924 \
  --revision 6d84c48581ece794365f2b8e9cfb043c68ade9c5 \
  --dtype bfloat16 --offline --output-dir <NEW_RUN>/parity/olmoe
python3 docs/ideas/energy_slo/routeslack/experiments/run_model_patch_parity_probe.py \
  --model llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M \
  --revision 1d5983076dfc67aee4a77ec06a27027f5bab6055 \
  --dtype bfloat16 --offline --output-dir <NEW_RUN>/parity/llmjp

# 5. synthetic development measurement-path；发现任何其他 CUDA workload 即整次拒绝。
python3 docs/ideas/energy_slo/routeslack/experiments/run_rtx5090_development_probe.py \
  --output-dir <NEW_RUN>/energy/dispatch_order_probe

# 6. 无 CUDA 主机上的预期 fail-closed probes。
./.venv/bin/python -B docs/ideas/bcrd/experiments/benchmark_expert_service_curve.py \
  --model jamesdborin/tiny-mixtral --model-key tiny-gpu-gate \
  --rows 1 --warmups 1 --trials 1 --dtype float32 --offline \
  --output /private/tmp/routeslack_gpu_gate_curve.csv
./.venv/bin/python -B docs/ideas/bcrd/experiments/capture_native_routes.py \
  --model jamesdborin/tiny-mixtral --model-key tiny-gpu-gate \
  --dataset builtin --samples 1 --seq-len 8 --decode-steps 1 \
  --dtype float32 --phase decode --offline \
  --output /private/tmp/routeslack_gpu_gate_routes.csv
```

仓库目前没有符合冻结协议的“双模型 natural continuous-serving + rows×tier raw-energy” formal runner，因此不存在诚实的一键 Experiment A–E 命令；上述 GPU probe 只验证 fail-closed 边界。

sealed GPU qualification 的复现命令与冻结 revision/参数记录在 `artifacts/energy_slo_routeslack_gpu/20260728_144600/commands.sh`；其 `config.yaml` 与 `processed/gpu_gate0_summary.json` 明确写入 `formal_result=false`、`Gate0=FAIL`、9 个 open P0 和 `physical_model_energy_samples=0`。

## 11. Next action

保留为 characterization。
