# MoE-EP-Eval：MoE 专家并行通信/调度优化候选的严格再评估

这个目录是一个毕业论文研究项目的完整代码库，记录了对五条 MoE (Mixture-of-Experts) 专家并行 (EP) 推理系统里通信/调度优化候选机制的严格实验评估。**核心叙事不是"我们提出了一个新优化"，而是"用严谨方法论（sealed holdout、fresh-data 去污染、跨模型统计校正、Mac 仿真到真实 GPU 硬件的交叉验证）系统性检验了五个看似合理的机制，并追踪清楚了它们各自为什么失败"**。

## 快速定位

- 全局总览表：[`../../MoE_Approach_Registry_2026-07-19.md`](../../MoE_Approach_Registry_2026-07-19.md) —— 五条路线的机制、判死证据、死因分类、跨路线元教训，一张表看全局。
- 论文整体进展审查：[`../../论文进展审查报告_2026-07-17.md`](../../论文进展审查报告_2026-07-17.md)

## 环境

```bash
# Mac / CPU-only（本项目主要开发环境，Apple M5 Pro，无 CUDA）
source ../../.venv/bin/activate
pip install -r requirements.txt

# 远程 GPU 交叉验证环境（用于 TokenRace-EP 的硬件开销测量、CreditReduce 的
# CUDA 设备迁移复现），关键包版本见下方"GPU 复现环境"一节
```

依赖版本对齐说明：`transformers==4.57.6` + `huggingface_hub==0.36.2` + `datasets==4.5.0` 这个组合在 Mac 和远程 GPU 上都验证过可用；更新的 `huggingface_hub>=1.5` 与 `datasets` 加载 `wikitext`（无 namespace 的旧式脚本数据集）之间存在已知的 URI 解析 bug（`parse_hf_uri` 在解析 `.huggingface.yaml` 元数据路径时报 `Repository id must be 'namespace/name'`），必须锁定旧版本。

## 五条候选路线一览

| 路线 | 一句话机制 | 核心代码 | 状态 | 关键结论文档 |
|---|---|---|---|---|
| CreditReduce | 用路由碰撞产生的字节信用资助部分 combine 组升级到 FP32 | `run_creditreduce_p0.py`, `creditreduce_reference.py` | KILLED | `outputs/creditreduce_p0_2026-07-17/P0-1_正式结论_2026-07-17.md` |
| RouteFidelity-EP | 为每种 backend contract 找最小路由充分统计量以保持配置排序 | `run_route_fidelity_p0.py`, `run_route_fidelity_p0b.py`, `route_fidelity_p0b_core.py` | KILLED | `../../research_report_ccfc_routefidelity_ep_2026-07-18.md` |
| WaveCredit-EP | receiver-side credit 限制同步 wave 的总在途请求 | （未落地，文献碰撞阶段淘汰） | KILLED（无代码） | `../../MoE_Approach_Registry_2026-07-19.md` |
| MassCover-EP | 用 gate/contribution 证据决定专家副本跨故障域放置 | `run_masscover_ep_p0.py` | KILLED | `outputs/masscover_ep_p0_2026-07-19/report.md` |
| **TokenRace-EP** | 打破 MoE combine 全 batch 同步屏障，逐 token 到齐即释放 | `run_tokenrace_ep_p0.py`, `run_tokenrace_ep_topk_sweep.py`, `run_tokenrace_gpu_microbench.py`, `run_tokenrace_gpu_p1_overhead.py` | **KILLED（GPU 追加验证）** | `outputs/tokenrace_gpu_p1_2026-07-19/GPU_P1_最终判决_2026-07-19.md` |

## 方法论框架（本项目最有可迁移价值的部分）

1. **Fresh/sealed 数据隔离**：每次正式确认实验都用 SHA-256 记录的 exclusion registry，确保 holdout 文档与任何历史 dev/calibration 数据零重叠，避免"用已经看过的数据确认假设"这种数据泄露。见 `outputs/creditreduce_p0_2026-07-17/frozen_historical_exclusion_registry.json`。
2. **预注册判死门槛**：每条路线在写代码之前先在对应的 `*_预注册_*.md` / 研究计划文档里锁定判死条件（比如"两模型 P99 改善必须 ≥5%"），实验只能验证或否证，不能事后调整门槛。
3. **Paired document/seed bootstrap + Holm 多重比较校正**：统计显著性判断以 article/seed 为独立单位（不是把 token 当 iid 样本），多重比较用 Holm 校正，避免假阳性。
4. **跨模型验证**：所有正式结论都要求在 OLMoE-1B-7B (E64/top-8) 和 LLM-jp (E32/top-16) 两个真实 MoE 模型上同时成立，防止方向性结论是单模型的偶然产物。
5. **Mac 仿真 → 真实 GPU 硬件交叉验证（本项目最新增加、也最重要的一条原则）**：TokenRace-EP 在 Mac 仿真上显示"OLMoE 稳健通过、LLM-jp 卡边界"，但用真实 RTX 5090 测出的 kernel launch/重批开销（~57-87μs/层，是 Mac illustrative 常数 6μs 的近 10 倍）代入后，两个模型全部反转为深度净负收益。**任何后续候选如果依赖"越来越细粒度的动态调度决策"，都必须先做一次真实硬件开销测量，不能只在 CPU 仿真上调参数敏感性。**

## GPU 复现环境

TokenRace-EP 的硬件验证和 CreditReduce 的跨硬件复现都在租用的独享 `NVIDIA RTX 5090 / 32GB`（CUDA 12.8, PyTorch 2.8.0+cu128, Python 3.12/Ubuntu 22.04）实例上完成。关键改动：`modeling.py::load_model` 增加了 `resolve_device()`（若 `torch.cuda.is_available()` 则 `.to("cuda")`，Mac 上无 CUDA 时行为完全不变），`run_creditreduce_p0.py` 里所有 tokenizer 输出都会显式 `.to(model.device)` 后再送入前向计算。GPU 上跑通实验需要的完整依赖文件清单（transitively）：

```
capture_moe.py, creditreduce_reference.py, fake_quant.py, metrics.py,
modeling.py, paths.py, policies.py, prompts.py, grouped_owner_combine.py,
run_creditreduce_p0.py, test_creditreduce_reference.py,
CreditReduce_P0_预注册_2026-07-17.md（source_manifest 的 provenance hash 需要）
```

## 复现单条实验

```bash
# CreditReduce sealed holdout（单模型，需已下载 OLMoE/LLM-jp 权重）
python run_creditreduce_p0.py --model allenai/OLMoE-1B-7B-0924 \
  --phase p0_1_holdout --offline \
  --registry-file outputs/creditreduce_p0_2026-07-17/frozen_historical_exclusion_registry.json \
  --output-dir outputs/creditreduce_p0_2026-07-17/olmoe_p0_1_holdout

# TokenRace-EP Mac 仿真 P0 + P0-B（top-K 截断消融）
python run_tokenrace_ep_p0.py --n-decode-steps 2000 --batch-sizes 32 64 128
python run_tokenrace_ep_topk_sweep.py --n-decode-steps 1500 --batch-size 128

# TokenRace-EP 真实 GPU 硬件开销测量（需 CUDA 环境）
python run_tokenrace_gpu_microbench.py --n-decode-steps 1000 --n-repeat 200
python run_tokenrace_gpu_p1_overhead.py --n-repeat 200 --n-decode-steps 1000
```

## 目录说明（部分核心文件）

- `modeling.py` / `capture_moe.py` / `grouped_owner_combine.py`：模型加载与 MoE combine 过程 hook/patch 的公共基础设施，被多条路线复用。
- `policies.py` / `fake_quant.py`：精度分配策略与量化模拟，CreditReduce/历史 QuotaEP-H 路线复用。
- `metrics.py`：KL/NLL/PPL 等质量指标累积器。
- `prompts.py`：WikiText-2 文档池加载与去重。
- `route_fidelity_p0b_core.py` / `capture_route_fidelity_p0b.py` / `freeze_route_fidelity_p0b.py` / `prepare_route_fidelity_p0b_configs.py`：RouteFidelity-EP 的 sufficient-statistic / placement-regret 核心逻辑与冻结流程。
- `outputs/`：每条路线的实验产出（`report.md`、`verdict.json`、`decision.json` 等小文件已纳入版本控制；大体量原始路由 CSV 按 `.gitignore` 排除，可用对应 `run_*.py` 加相同 seed/registry 重新生成）。

## 已知限制

- Mac 上的所有实验默认在 CPU 上运行（`modeling.py` 历史版本未显式 `.to(device)`；GPU 分支是本轮新增，向后兼容）。
- 早于 2026-07-16 的更早期路线（PLTB、R-layout、receiver-aware v1、Graceful/QTree、additive-KL MILP）已在更早的研究报告中判死，未包含在本 Approach Registry 的四条主表里，历史材料见工作区根目录的 `MoE_唯一核心创新_严格研究报告_*.md` 系列文档。
