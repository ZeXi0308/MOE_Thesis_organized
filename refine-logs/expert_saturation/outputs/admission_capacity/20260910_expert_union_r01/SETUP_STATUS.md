# 环境搭建与实验准备状态

2026-09-10。新 5090 机器 `connect.weste.seetacloud.com:11155`。
本文件只记录执行状态，不含任何科学结论。

## 机器

| 项目 | 值 | 与旧机对比 |
|---|---|---|
| GPU | RTX 5090, **32607 MiB** | **逐位一致**（同型号同显存） |
| 驱动 | 595.71.05 | 旧机 580.105.08（更新，CUDA 13 兼容） |
| CPU | 128 核 Xeon | 旧机 104 线程 |
| 内存 | 754 GB | 旧机 未记录 |
| 数据盘 | 50 GB（初始全空） | 旧机 50 GB / 已用 27 GB |
| Python | **3.12.3**（`/root/miniconda3`） | **逐位一致** |
| pypi | 可达（阿里云镜像，但对大 wheel 会挂） | — |
| huggingface.co | **不可达**（HTTP 000） | 旧机已有缓存，无需下载 |
| hf-mirror.com | 可达（200） | — |

显存与 Python 版本与旧机一致，是本轮选它的关键——KV 容量与捕获阶梯有望复现。

## 已完成

- venv 建于 `/root/autodl-tmp/expert-saturation/vllm-0.26`（与旧机同路径，
  部署脚本零改动复用）。
- 下载脚本 `dl.sh`：带 80 次重试、禁用 xet、`--max-workers 2`。

## 进行中

| 项目 | 状态 |
|---|---|
| `pip install vllm==0.26.0 transformers==5.15.1` | 正常推进（清华源，正在拉 torch 2.11.0 的 530 MB wheel，pip cache 455 MB） |
| 模型 `OLMoE-1B-7B-0924@6d84c485...` | 1.1 GB / 约 13 GB |

## 遇到并已定位的三个问题

1. **阿里云 pypi 镜像对 303 MB 的 vLLM wheel 挂死**（pip cache 20 秒零增长）。
   换清华源后正常。
2. **`hf` 默认走 xet，返回 401 Unauthorized**（`cas-server.xethub.hf.co`）。
   设 `HF_HUB_DISABLE_XET=1` 解决。
3. **`pkill` 杀掉 hf 进程后留下孤儿锁**，新进程在
   `.locks/models--allenai--OLMoE-1B-7B-0924/*.lock` 上死等 521 秒，
   表现为"下载停滞"。需 `rm -rf hf-cache/hub/.locks` 清理。
   **这也是 cache 体积两次回退（795M→371M、957M→553M）的原因**：
   被打断的分片被丢弃重下。

第 3 项需要在恢复后手动清理一次，否则下载不会前进。

## 恢复步骤

```bash
# 1) 清理孤儿锁（必须，否则下载卡死）
pkill -f dl.sh; pkill -f 'hf download'; sleep 3
rm -rf /root/autodl-tmp/hf-cache/hub/.locks

# 2) 重启带重试的下载
setsid nohup /root/autodl-tmp/dl.sh > /root/autodl-tmp/dl2.log 2>&1 < /dev/null &

# 3) 确认 pip 仍在跑（不要杀，它是正常的）
ps aux | grep '[p]ip install'

# 4) 两者完成后校验
V=/root/autodl-tmp/expert-saturation/vllm-0.26/bin
$V/python -c "import vllm,torch,transformers;print(vllm.__version__,torch.__version__,transformers.__version__)"
du -sh /root/autodl-tmp/hf-cache   # 应约 13G
```

期望版本（与全部封存基线一致）：`0.26.0 / 2.11.0+cu130 / 5.15.1`。

## 待上传的实验代码

已在本地写好并测试通过，尚未上传：

| 文件 | 作用 | 测试 |
|---|---|---|
| `expert_union_tracker.py` | batched 专家并集 / 驻留视野 / 负载偏斜 | **19/19 通过** |
| `run_expert_union.py` | 通过 gate forward hook 采集每层每步路由 | 需 GPU 验证 hook 挂载点 |
| `analyze_batched_idle_bounds.py` | 算术边界（已产出结果） | — |

`run_expert_union.py` 有一个**未验证假设**：MoE block 通过
`type(m).__name__.endswith("SparseMoeBlock")` 定位、router 在 `block.gate`。
上机后第一件事是确认 `model_shape.json` 写出 `n_moe_blocks=16, num_experts=64,
num_experts_per_tok=8`；不符则先修定位再采集。

## 本轮的科学产出（与环境无关，已完成）

1. **撤回了我自己的一个错误结论。**
   `20260908_kv_pressure_probe_r01` 的"闲置专家 10.50 GiB"用了 per-token 比例，
   在实测 batch 宽度 16–32 下高估 **7–60 倍**。已写
   `ADDENDUM_BATCHED_UNIT_CORRECTION.md`，原报告与数据全部保留未改。
2. **确认我的 KV 可行性接纳实验不必再跑。**
   并行工作流已实测：公式 cap29 无抢占但吞吐低 14.5%，而允许抢占的 native32
   吞吐高 **17%**、TTFT p99 从 18 s 降到 0.8 s。**"避免抢占"是负价值的**，
   我那条规则的核心卖点（可行性保证）已被间接否证。
3. **冻结了下一个实验的设计与三条可证伪预测**
   （`20260910_expert_union_r01/DECISIONS.md`），且它是仓库 `kv_budget_r01`
   点名要求、明确记为"没有启动"的前置测量。

| 字段 | 状态 |
|---|---|
| Verdict | `SETUP_IN_PROGRESS`；GPU 采集 `UNRUN` |
| 已完成 | 环境搭建 3/4、算术更正、下一实验冻结、追踪器 19/19 测试 |
| 阻断 | 模型下载需清孤儿锁；vLLM 安装未完成 |
| 未测 | 全部 route 数据；本轮无任何 GPU 性能或结构结论 |
