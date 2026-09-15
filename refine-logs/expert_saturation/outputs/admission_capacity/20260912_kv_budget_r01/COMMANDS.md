# 本轮执行与下一最小实验命令

2026-09-12；从仓库根目录执行。凭据不写入文件。四项KV对照已经完成，原目录不可重跑覆盖。

## 已实际执行：原封不动的KV包

```bash
python3 -u refine-logs/expert_saturation/experiments/admission_capacity/run_frozen_kv_remote.py \
  --source refine-logs/expert_saturation/outputs/admission_capacity/20260910_kv_budget_r01 \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01 \
  --host root@connect.weste.seetacloud.com --port 23478 \
  --control-path /tmp/moe-research-20260912-ssh \
  --gpu-uuid GPU-389be666-aeaa-c602-1504-85bf1dd3ac9f \
  --remote-dir /root/autodl-tmp/moe-kv-budget-20260912-r01
```

前置已登录SSH control socket、已存在vLLM/model、GPU空闲。driver不创建租赁、不安装、不下载、不自动重试。每项新引擎，退出后归档、回传、校验SHA256，再运行下一项。

首次分析直接调用 `python3 .../analyze_kv_budget.py`，因冻结`metrics.py`不在导入路径得到 `ModuleNotFoundError: No module named 'metrics'`，未生成分析结果、未重跑GPU。实际成功命令：

```bash
PYTHONPATH=refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01/frozen \
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01/analyze_kv_budget.py
```

再次复算需添加 `--output-dir /tmp/kv-budget-analysis-new`，防止覆盖。新暂停分析器亦使用全新输出：

```bash
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_pause_ledger.py \
  --raw refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01/gpu_results/repeat0-budget90/raw.json \
  --output /tmp/kv-budget-pause-new.json
python3 -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p test_pause_ledger.py -v
python3 -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity -p test_paging_cost_model.py -v
```

## 下一步准备命令：WiSP隔离环境（UNRUN）

先确认已有机器磁盘能容纳独立环境、容器内存够用；当前保留的native0.26环境不可覆盖。以下为当前端点的独立新路径，尚未创建；若路径已存在先检查其归属，不覆盖。固定源码版本，不跟随main漂移：

```bash
WISP_SRC=/root/autodl-tmp/wisp-source-20260912
WISP_ENV=/root/autodl-tmp/wisp-v0112-20260912
git clone https://github.com/nokia-applied-research/WiSP.git "$WISP_SRC"
git -C "$WISP_SRC" checkout --detach 86f69720f0de0647c51728ea2e27e4289bf3dea2
python3 -m venv "$WISP_ENV"
"$WISP_ENV/bin/python" -m pip install 'vllm==0.11.2' --extra-index-url https://download.pytorch.org/whl/cu128
"$WISP_ENV/bin/python" -m pip install --no-deps -e "$WISP_SRC"
```

版本来源：[WiSP pyproject](https://github.com/nokia-applied-research/WiSP/blob/86f69720f0de0647c51728ea2e27e4289bf3dea2/pyproject.toml)、[vLLM0.11.2 CUDA安装](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/)。使用其依赖解析，不照搬README旧cu121。5090上的这一路径尚未运行。

以下官方脚本有5个固定prompt、8输出token。OLMoE的cap8是人为受限池；`.65`仅是初始化起点，容量需由实际池验证。两臂独立进程；固定BF16、无prefetch、同步in-process。该入口不是性能实验：

```bash
OLMOE_LOCAL=/root/autodl-tmp/hf-cache/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5
WISP_OUT=/root/autodl-tmp/wisp-olmoe-identity-20260912-r01
mkdir "$WISP_OUT"
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WISP_PREFETCH=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  "$WISP_ENV/bin/python" "$WISP_SRC/scripts/check_byte_identity.py" \
  --mode vanilla --model "$OLMOE_LOCAL" --dtype bfloat16 \
  --gpu-memory-utilization 0.65 --max-model-len 256 --max-tokens 8 \
  --out "$WISP_OUT/vanilla.json"
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WISP_PREFETCH=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  "$WISP_ENV/bin/python" "$WISP_SRC/scripts/check_byte_identity.py" \
  --mode wisp --model "$OLMOE_LOCAL" --dtype bfloat16 --cap-experts 8 \
  --gpu-memory-utilization 0.65 --max-model-len 256 --max-tokens 8 \
  --out "$WISP_OUT/wisp.json"
"$WISP_ENV/bin/python" "$WISP_SRC/scripts/check_byte_identity.py" \
  --compare "$WISP_OUT/vanilla.json" "$WISP_OUT/wisp.json"
```

直接入口源码：[check_byte_identity.py](https://github.com/nokia-applied-research/WiSP/blob/86f69720f0de0647c51728ea2e27e4289bf3dea2/scripts/check_byte_identity.py)。不能原样调用默认`OFFLOAD_GB=48`的Qwen复现脚本。

**必须补的最小接线尚未实现：**在同一进程生成后读取`WispMoEState`注册层/cap、`snapshot_expert_stats()` miss及eviction，记录16层均注册与实际缺页；copy路径记录真实字节、唯一load事件和未隐藏等待，带request/step/layer身份。仅官方identity成功无法证明pager启用，脚本可能把安装异常降为warning。复制计数不是CUDA耗时；GPU profiling应独立于性能臂。

这些命令只准备下一实例；本轮未安装WiSP、未移植0.26、未跑pager、未下载Qwen。

## CPU补充统计的复算

均值和同repeat输出一致性直接从raw读取；以下打印结果，不修改原artifact：

```bash
python3 - <<'PYCODE'
import json, statistics
from pathlib import Path
b = Path('refine-logs/expert_saturation/outputs/admission_capacity/20260912_kv_budget_r01/gpu_results')
for repeat in (0, 1):
    arms = []
    for budget in (90, 95):
        raw = json.loads((b/f'repeat{repeat}-budget{budget}/raw.json').read_text())
        requests = raw['requests']
        assert raw['status'] == 'COMPLETE' and all(r['status'] == 'completed' for r in requests)
        print(repeat, budget, 'mean completion latency', statistics.mean(r['completion_s']-r['arrival_s'] for r in requests))
        arms.append({r['request_id']:r['output_token_ids'] for r in requests})
    assert arms[0].keys() == arms[1].keys()
    print('identical output sequences', sum(arms[0][k] == arms[1][k] for k in arms[0]), '/', len(arms[0]))
PYCODE
```
