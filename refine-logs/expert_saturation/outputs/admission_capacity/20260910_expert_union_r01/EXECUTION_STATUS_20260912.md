# 执行状态：环境、代码与上机前验证

2026-09-12。新 5090 机器 `connect.weste.seetacloud.com:23478`。
本文件只记录执行状态与已修缺陷，**不含任何科学结论**。
冻结设计见 [`DECISIONS.md`](DECISIONS.md)，未修改。

## 机器与软件栈

| 项目 | 值 | 与封存基线对比 |
|---|---|---|
| GPU | RTX 5090, **32607 MiB**, 空闲 | **逐位一致** |
| 驱动 | 595.71.05 | 旧机 580.105.08（CUDA 13 兼容） |
| Python | **3.12.3** | **逐位一致** |
| vLLM | **0.26.0** | **逐位一致** |
| PyTorch | **2.11.0+cu130** | **逐位一致** |
| Transformers | **5.15.1** | **逐位一致** |
| 数据盘 | 50 G，已用 13 G | 充足 |

搭建按 [`SETUP_STATUS.md`](SETUP_STATUS.md) 记录的三个坑规避，全部生效：
全局换清华源（阿里源对 303 MB wheel 挂死）、`HF_HUB_DISABLE_XET=1`（xet 返回 401）、
带 80 次重试的下载脚本（孤儿锁）。本次**未再出现锁死或体积回退**。

## 进度

| 项 | 状态 |
|---|---|
| venv `/root/autodl-tmp/expert-saturation/vllm-0.26` | **完成** |
| `vllm==0.26.0 transformers==5.15.1` | **完成**，`VERIFY 0.26.0 2.11.0+cu130 5.15.1` |
| 实验代码上传 `/root/autodl-tmp/expert-union/` | **完成**，4 个 `.py` 的 sha256 与本地逐位一致 |
| 远端测试 | **26/26 通过** |
| 模型 `OLMoE-1B-7B-0924@6d84c485` | 进行中，5.3 G / 约 13 G |
| **GPU 采集** | **UNRUN** |

## 上机前已验证（消除三个风险）

### 1. 引擎参数全部存在

`EngineArgs` 字段逐项探测，14/14 `OK`，含风险最高的 `enable_return_routed_experts`
与 `stream_interval`。若缺失引擎会直接起不来。

### 2. 模型结构与冻结预期一致

从下载到的 `config.json` 读出：

| 字段 | 值 | 冻结文档要求 |
|---|---|---|
| `model_type` | `olmoe` | — |
| `num_experts` | **64** | 64 ✓ |
| `num_experts_per_tok` | **8** | 8 ✓ |
| `num_hidden_layers` | **16** | `n_moe_blocks=16` 待引擎加载确认 |

`SETUP_STATUS.md` 记录的"未验证假设"中，两项已提前确认。
剩余一项（`SparseMoeBlock` 定位与 `block.gate` 是否为 router）仍需引擎加载后验证，
上机第一件事仍是检查 `model_shape.json`。

### 3. 冻结输入已定位并校验

`run_expert_union.py` 原本硬编码 `inputs_preparation/prepared/<domain>/`，
该路径在实验目录下不存在。实际输入在
`outputs/admission_capacity/20260910_kv_budget_r01/inputs_preparation/prepared/{short,long}/`，
目录结构与脚本约定一致，复用即可：

| 域 | prompt | 请求数 | 输出 |
|---|---:|---:|---:|
| short | 128 | 32 | 1024 |
| long | **3072** | 32 | 1024 |

身份校验（`workload_sha256` 与逐 prompt `prompt_token_ids_sha256`）保留且在加载时执行。
新增 `--prepared-dir` 供显式指定，默认路径行为不变。

## 本次修掉的一个真实身份错误

**问题：** hook 挂在 router 上，只按"至少有一个 decode 请求"开启采集。
一个同时携带 chunked prefill 的 step 会把**最多 1024 个 prefill token 的路由**
写入本应是 decode 宽度的并集。

**后果：** `U` 会变成 chunk 大小的函数而非 decode 批宽的函数。
研究问题本身（并集如何随 batch 宽度增长）会被摧毁——
width-16 的并集里混进 1024 个 token，必然接近饱和，
从而**伪造出"无余量"的结论**。这正是本实验要检验的假说，等于自证。

**修复：** 两道独立关卡。

1. 驱动循环只在**纯 decode step** 采集：
   `pure = decoding > 0 and not prefilling and not waiting`，否则跳过并计数。
2. hook 内的身份守卫：`logits.shape[0] != expected_tokens` 则拒绝记录并计数。
   非纯 step 时 `expected_tokens = -1`，任何行数都无法匹配。

物理上一个混合 step 确实需要其全部 token 的并集，因此被跳过的 step
**不是错误数据，而是不同口径**；跳过数与 mismatch 数均写入
`union_summary.json` 的 `collection_discipline`，不隐藏。

**回归测试：** 新增 `test_expert_union_collection.py`，7 项，含决定性的
`test_mixed_step_with_prefill_chunk_is_rejected`（1024-token chunk 必须被 width-3
并集拒绝）与 `test_wrong_expert_dimension_is_ignored_not_counted`
（同模块上的非 router 线性层不得被误认为 logits）。
本地与远端各 26/26 通过。

## 另外两处补强

- **截断口径显式化**：route 采集在 `max_route_steps` 停止，episode 因此不完整。
  `union_summary.json` 写入 `truncation.episode_truncated` 与
  "must not be used for any throughput or latency claim"，防止后续误用。
- **`--prepared-dir`**：允许复用任何封存 `prepared/`，但身份校验一律执行，
  不因换目录而跳过。

## 待办

1. 模型下载完成。两个 safetensors 分片各约 6.5 G，当前各 3.0 G，速率约 2 MB/s。
   `model.safetensors.index.json` 在分片完成后才建符号链接，**在此之前无法起引擎**。
2. 引擎加载后确认 `model_shape.json` 写出 `n_moe_blocks=16`；不符则先修定位再采集。
3. 运行 `run_session.sh`，同一次引擎启动内完成 route 采集。

### 一次体积回退的排查（虚警）

`du` 一度从 7.3 G 显示回落到 5.7 G，与 `SETUP_STATUS.md` 记录的"分片丢弃重下"
症状相同。逐项核对后确认是**虚警**：

| 检查 | 结果 |
|---|---|
| 下载重试次数 | **1**（无重试，即无中断） |
| 日志中的 error/timeout/reset/401 | **0** |
| `.incomplete` 分片大小 | 单调增长 2.36→2.87→3.06 G |
| 下载进程 | 存活 |

原因是 `du` 对稀疏文件的预分配块计数在不同时刻不同。**没有发生重下**，
不需要清锁。此处记录以免日后把同一现象再次误判为故障。

## 第二个被修掉的真实缺陷：裁决分档边界

用三组合成路由（uniform / concentrated / alternating）端到端验证裁决器时发现：

`ExpertUnionTracker.verdict()` 原本只检查 `saturation_window_99 <= 2`，
而 `horizons` 最大为 32。于是一个**每 4 步就要重搬全部专家**的层，
既不满足"立即复用"（>2），又落入 `else` 分支被判为
`CANDIDATE_RESIDENCY_HEADROOM`——把一个搬运成本每 4 步复发的情形
**误报为存在静态驻留价值**。

合成 uniform 组（width 16、独立均匀路由）实测 `sat99 = [4,4,4,4]`，
正是这个区间，原逻辑给出 `CANDIDATE`。

**修复：** `CANDIDATE` 现在要求**至少一层在全部追踪视野内从不饱和**
（`saturation_window_99 is None`），并把复用视野提为显式参数
`immediate_reuse_steps=4`，不再由 horizons 网格隐式决定。

修复后合成组分档正确：

| 合成组 | max 层中位闲置 | 从不饱和层数 | 裁决 |
|---|---:|---:|---|
| uniform w16（4 步重搬） | 0.1250 | 0 | `HEADROOM_BUT_IMMEDIATE_REUSE` |
| concentrated w16（稳定闲置子集） | 0.7500 | 4 | `CANDIDATE_RESIDENCY_HEADROOM` |

**这个修复方向是保守的**：它只会让 `CANDIDATE` 更难达成，
因此不会把负结果洗成正结果。

补 4 项边界测试（`TestVerdictBoundary`）。其中
`test_recurring_reuse_is_not_candidate` 第一次写错了 fixture——
用 4 组各 8 个专家只覆盖 32/64，于是闲置是**真实**的、`sat99=None` 是**正确**的。
断言失败后核查发现是 fixture 而非逻辑有误，遂改为 8 组平铺全部 64 个专家，
并加入 `assertEqual(len({e for g in groups for e in g}), self.E)` 防止同类错误复现。
**未通过修改断言来迁就代码。**

## 采集后分析器已就绪

`adjudicate_expert_union.py`：只读采集产物，**不拟合任何量**，机械判定 E1/E2/E3。

已用三组合成数据端到端验证，三条裁决路径均能正确触发，
且 width < 16 时自动输出 scope 警告（"E2 must be read as out of its intended scope"）。

## 会话驱动

`run_session.sh`：单次引擎启动，固定顺序——route 采集在前（冻结的前置测量、
裁决已冻结），静态扫描在后，使后者失败不会连带损失 route 数据。
含两道保护：启动前若 GPU 上有其他 compute 进程则直接 `ABORT`（共租进程会
静默改变全部计时）；采集结束立即打印 verdict 与 `model_shape.json`，
避免后续失败把结果埋掉。`HF_HUB_OFFLINE=1` 防止运行中触发网络。

本地与远端 `bash -n` 均通过。

| 字段 | 状态 |
|---|---|
| Verdict | `SETUP_COMPLETE_PENDING_MODEL`；GPU 采集 `UNRUN` |
| 已完成 | 软件栈逐位对齐、代码上传与 hash 核对、**30/30 测试**、3 个上机前风险消除、**2 个真实缺陷修复**、裁决器合成验证、会话脚本部署 |
| 阻断 | 模型下载，2 个分片各 3.0 G / 约 6.5 G |
| 未测 | 全部 route 数据；本轮无任何 GPU 结构或性能结论 |
