# Corrected FJRC GPU Runbook（2026-07-23）

## 0. 当前执行边界

本 runbook 完成两件事：

1. 单张 RTX 5090 上采集 native route 与 primitive LUT；
2. 在这些真实 route identity / GPU primitive timing 上运行 Level-1 logical trace replay。

它不执行、也不允许宣称 physical multi-rank timed trace。真实 receiver incast existence gate
至少需要4张独立 GPU；单卡多进程或多 stream 不满足该硬件语义。

当前状态：`PREPARED / GPU NOT OPEN / NO SCIENTIFIC RESULT`。

## 1. 冻结路径

```bash
export FJRC_REMOTE_ROOT=/root/autodl-tmp/ric_clean_v2_20260723
export FJRC_PYTHON=/root/autodl-tmp/ric_clean_v2_env/bin/python
export FJRC_CODE=$FJRC_REMOTE_ROOT/docs/archive/receiver_aware/ric_clean_v2/experiments/ric_clean_v2
export FJRC_ROUTE_ROOT=$FJRC_REMOTE_ROOT/clean_v2/routes/calibration
export FJRC_LUT=$FJRC_REMOTE_ROOT/clean_v2/fjrc_corrected/calibration/fjrc_primitive_lut_v1.json
export FJRC_PREFLIGHT_ROOT=$FJRC_REMOTE_ROOT/clean_v2/fjrc_corrected/preflight
export FJRC_DRY_ROOT=$FJRC_REMOTE_ROOT/clean_v2/fjrc_corrected/native_dry_run
export FJRC_FORMAL_ROOT=$FJRC_REMOTE_ROOT/clean_v2/fjrc_corrected/formal_cpu
```

不得把上述变量替换成已有历史输出目录。所有 producer 和 runner 均拒绝覆盖。

## 2. GPU 开机前，本地静态门

在本地 workspace 根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  docs/archive/receiver_aware/ric_clean_v2/experiments/ric_clean_v2/preflight_fjrc_corrected_gpu.py \
  --mode static \
  --repo-root "$PWD"
```

必须得到：

```text
status = READY
readiness_config_self_hash = true
reviewed_source_manifest = true
```

当前已实测通过，reviewed bundle 共24个文件。

## 3. 开机后同步 reviewed bundle

SSH 入口每次租机可能变化，只设置连接参数，不修改冻结远端路径：

```bash
export FJRC_GPU_HOST=YOUR_GPU_HOST
export FJRC_GPU_PORT=<CURRENT_PORT>
export FJRC_GPU_USER=root
export FJRC_LOCAL_ROOT="$PWD"
```

从 readiness config 生成精确同步清单；禁止 `--delete`：

```bash
python3 - <<'PY' >/tmp/fjrc_reviewed_files.txt
import json
from pathlib import Path
p = Path("docs/archive/receiver_aware/ric_clean_v2/configs/fjrc_corrected_gpu_readiness_v1.json")
x = json.loads(p.read_text())
for value in sorted(x["reviewed_sources"]):
    print(value)
print(p.as_posix())
PY

ssh -p "$FJRC_GPU_PORT" "$FJRC_GPU_USER@$FJRC_GPU_HOST" \
  'mkdir -p /root/autodl-tmp/ric_clean_v2_20260723'

rsync -az --files-from=/tmp/fjrc_reviewed_files.txt \
  -e "ssh -p $FJRC_GPU_PORT" \
  "$FJRC_LOCAL_ROOT/" \
  "$FJRC_GPU_USER@$FJRC_GPU_HOST:/root/autodl-tmp/ric_clean_v2_20260723/"
```

同步范围只包括 reviewed sources/config；不覆盖模型、data、route、state 或历史结果。

## 4. 远端 unit tests

进入远端后：

```bash
cd "$FJRC_REMOTE_ROOT"

PYTHONDONTWRITEBYTECODE=1 "$FJRC_PYTHON" -m unittest -v \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_prepare_clean_v2_data \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_capture_clean_v2_routes_gpu \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_capture_fjrc_lut_gpu \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_fjrc_corrected_level0 \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_fjrc_corrected_level1 \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_fjrc_corrected_replay \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_run_fjrc_corrected_level1 \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_preflight_fjrc_corrected_gpu \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_decide_fjrc_corrected_two_model
```

硬门：

- 0 failures；
- 0 errors；
- 本地曾 skip 的3个 torch/native parity tests 在远端不得 skip；
- 出现 foreign GPU process 必须停止，不共享 GPU 继续测。

## 5. Calibration Capture Preflight

```bash
mkdir -p "$FJRC_PREFLIGHT_ROOT"

"$FJRC_PYTHON" "$FJRC_CODE/preflight_fjrc_corrected_gpu.py" \
  --mode gpu \
  --repo-root "$FJRC_REMOTE_ROOT" \
  --remote-root "$FJRC_REMOTE_ROOT" \
  --deep-model-hash \
  --output "$FJRC_PREFLIGHT_ROOT/pre_capture_v1.json"
```

允许继续的条件：`status=READY`。缺失 route/LUT 应出现在 `planned_actions`，而不是
`blockers`。以下任一项为 blocker：

- source/config self-hash mismatch；
- Python/package/GPU identity drift；
- model tree hash mismatch；
- calibration manifest/signoff 缺失；
- route reservation ledger 存在但 route output 缺失；
- route output 存在但 ledger 缺失；
- 已有 route/LUT 无法通过 validator；
- foreign GPU process；
- 若需要 route capture，free disk 小于20 GiB；若 route 已验证且只需 LUT/replay，
  free disk 小于2 GiB；若仅验证已有完整产物，free disk 小于1 GiB。

查看待执行动作：

```bash
"$FJRC_PYTHON" - <<'PY'
import json, os
p = os.environ["FJRC_PREFLIGHT_ROOT"] + "/pre_capture_v1.json"
x = json.load(open(p))
print("status:", x["status"])
print("planned_actions:")
for value in x["planned_actions"]:
    print(" -", value)
print("blockers:", x["blockers"])
PY
```

## 6. GPU Calibration Capture

只执行 preflight 明确列出的缺失动作。

### 6.1 Primitive LUT

若存在 `capture_primitive_lut`：

```bash
"$FJRC_PYTHON" "$FJRC_CODE/capture_fjrc_lut_gpu.py" \
  --output "$FJRC_LUT"
```

该 LUT 同时包含 OLMoE 与 LLM-jp shape；只运行一次。

### 6.2 OLMoE native route

若存在 `capture_route_olmoe`：

```bash
"$FJRC_PYTHON" "$FJRC_CODE/capture_clean_v2_routes_gpu.py" \
  --model-key olmoe
```

### 6.3 LLM-jp native route

若存在 `capture_route_llmjp`：

```bash
"$FJRC_PYTHON" "$FJRC_CODE/capture_clean_v2_routes_gpu.py" \
  --model-key llmjp
```

两个 route producer 都是一锤子 reservation。进程异常退出后，不得删除 ledger 重跑；先审计
partial state。不得并发运行两个模型，因为 producer 要求一张独占 GPU。

## 7. Post-capture Preflight

使用不同输出文件，禁止覆盖 pre-capture 记录：

```bash
"$FJRC_PYTHON" "$FJRC_CODE/preflight_fjrc_corrected_gpu.py" \
  --mode gpu \
  --repo-root "$FJRC_REMOTE_ROOT" \
  --remote-root "$FJRC_REMOTE_ROOT" \
  --deep-model-hash \
  --output "$FJRC_PREFLIGHT_ROOT/post_capture_v1.json"
```

必须满足：

- `status=READY`；
- 不再包含 `capture_primitive_lut`；
- 不再包含 `capture_route_olmoe/llmjp`；
- 两模型 route validation 均有 `join_count`；
- 两模型 LUT artifact SHA 相同。

## 8. Native artifact CPU dry run

该步骤在 GPU 机器上执行，但 runner 本身不调用 CUDA。

```bash
"$FJRC_PYTHON" "$FJRC_CODE/run_fjrc_corrected_level1.py" \
  --route-root "$FJRC_ROUTE_ROOT" \
  --lut "$FJRC_LUT" \
  --model olmoe \
  --output "$FJRC_DRY_ROOT/olmoe" \
  --dry-run \
  --bootstrap-replicates 50

"$FJRC_PYTHON" "$FJRC_CODE/run_fjrc_corrected_level1.py" \
  --route-root "$FJRC_ROUTE_ROOT" \
  --lut "$FJRC_LUT" \
  --model llmjp \
  --output "$FJRC_DRY_ROOT/llmjp" \
  --dry-run \
  --bootstrap-replicates 50
```

生成 dry-run 双模型总判定：

```bash
"$FJRC_PYTHON" "$FJRC_CODE/decide_fjrc_corrected_two_model.py" \
  --olmoe-output "$FJRC_DRY_ROOT/olmoe" \
  --llmjp-output "$FJRC_DRY_ROOT/llmjp" \
  --expected-run-class CPU_DRY_RUN \
  --output "$FJRC_DRY_ROOT/two_model_decision.json"
```

dry-run 的 PASS/FAIL 不作为科学结论；这里只检查完整产物、denominator、source/LUT binding、
calibration audit 与无 pooling AND 逻辑。

## 9. Formal CPU logical replay

只有 dry-run 产物结构复核通过后执行：

```bash
"$FJRC_PYTHON" "$FJRC_CODE/run_fjrc_corrected_level1.py" \
  --route-root "$FJRC_ROUTE_ROOT" \
  --lut "$FJRC_LUT" \
  --model olmoe \
  --output "$FJRC_FORMAL_ROOT/olmoe" \
  --bootstrap-replicates 2000

"$FJRC_PYTHON" "$FJRC_CODE/run_fjrc_corrected_level1.py" \
  --route-root "$FJRC_ROUTE_ROOT" \
  --lut "$FJRC_LUT" \
  --model llmjp \
  --output "$FJRC_FORMAL_ROOT/llmjp" \
  --bootstrap-replicates 2000

"$FJRC_PYTHON" "$FJRC_CODE/decide_fjrc_corrected_two_model.py" \
  --olmoe-output "$FJRC_FORMAL_ROOT/olmoe" \
  --llmjp-output "$FJRC_FORMAL_ROOT/llmjp" \
  --expected-run-class FORMAL_CPU_TRACE_REPLAY \
  --output "$FJRC_FORMAL_ROOT/two_model_decision.json"
```

双模型 gate：

- 每模型独立 absolute miss reduction `>=5pp`；
- matched-pair bootstrap 95% CI lower `>0`；
- strict action flip `>=4/16`；
- holdout Q risk 非退化；
- OLMoE AND LLM-jp；禁止 pooling rescue。

## 10. 拉回证据

```bash
mkdir -p docs/archive/receiver_aware/ric_clean_v2/outputs/fjrc_corrected_gpu_2026-07-23

rsync -az \
  -e "ssh -p $FJRC_GPU_PORT" \
  "$FJRC_GPU_USER@$FJRC_GPU_HOST:$FJRC_REMOTE_ROOT/clean_v2/fjrc_corrected/" \
  docs/archive/receiver_aware/ric_clean_v2/outputs/fjrc_corrected_gpu_2026-07-23/
```

拉回后重算所有 SHA，先完成结果 Code Review，再解释 PASS/FAIL。

## 11. 停止条件

立即停止且不解释结果：

- unit test fail/error/skip；
- preflight BLOCKED；
- output/ledger 一次性状态不闭合；
- silently fallback、OOM 后自动缩小参数、模型 tree drift；
- 32-request denominator 或 selection/holdout isolation 失败；
- Q-map 在 matched worlds 中不一致；
- negative control 失败；
- source/LUT 不同模型不一致；
- holdout risk 返回 `INVALID_WORKLOAD_IDENTIFIABILITY`。

## 12. 后续 physical multi-rank 门

若 formal logical replay PASS，也不能直接进入单卡“网络模拟”。下一阶段必须另行满足：

- 至少4张独立 GPU；
- 同一可比较 timeline 的跨 rank timestamp；
- 真实 send/receive message conservation；
- `B={1,2,4,8}`；
- 先跑 RR-credit 自然 headroom census；
- 两模型 natural temporal incast `>=10%`；
- receiver busy-period P95 `>2×` single contribution service P95；
- 至少20% busy periods 有两个以上合法 first-credit action。

任一失败即 `NO_GO_PHYSICAL_INCAST_HEADROOM`，不靠 sleep 或人工 queue 注入救援。
