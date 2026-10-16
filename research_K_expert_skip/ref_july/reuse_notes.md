# 历史复用与 F1（2026-10-16）

复用仓库 `ZeXi0308/MOE_Thesis_organized`，固定本地 HEAD `5593b5ff0fb602f479b10f82ca682ba8f9adc86d`。本地为 sparse/partial clone，树中有文件但相关 blob 未下载；未 clone、未更改原仓库，缺失文件通过 GitHub 只读 connector 按该 commit 取回。以下均为历史材料，不能替代本课题同权重的新测量。

## Drop 结果的准确口径

原文件 `docs/ideas/A_rank_tail_fp8/outputs/main_experiments/olmoe_rank_mix/rank_mix_results.csv` 与生产脚本已保存为本目录同名文件。唯一含 drop 的策略是 `mix_r12bf16_r36fp8_r78drop`：OLMoE、32 样本、最大序列长 128、top-8；full PPL=15.008773，候选 PPL=15.238227，差值 +0.229454，旧 KL=4.547831，local relative MSE=0.0189851。

它是**所有 MoE 层均施加的混合输出近似**：rank1–2 BF16、rank3–6 FP8、rank7–8 drop。`run_rank_mix.py::build_lut` 对所有层/receiver group 建同一 LUT，`policies.py::_apply_layer_lut` 将 drop output 置零但不调整 gate。因此它既不是纯 tail drop，也不是 drop+renorm，不可直接填为新 T1 的独立尾部 drop 质量代价。所谓 50% byte saving 是 expert-output combine 的逻辑通信字节；脚本已先算完所有 expert 输出，完全不是权重读取或 kernel 实测节省。local MSE 是层内汇总，KL/PPL 是传播至全模型 logits 后的结果。

旧 `compute_kl` 对 `[1,T,V]` 用 `reduction="batchmean"`，所以每个样本得到的是沿 token 累加的 KL，表中再按样本平均，**不是 nats/token**。旧 PPL 是样本 PPL 的算术均值，**不是 pooled corpus PPL**。没有原始逐样本 token 数，不能从该汇总表可靠换算。本目录 `olmoe_approx_results.csv`、`llmjp_approx_results.csv`、`mixtral_approx_results.csv` 也只有 INT4/output 近似行，没有纯 drop+renorm。检查范围包括 rank_mix 系列与三模型主 approximation 表；未声称遍历所有历史私有运行。

可复用实现：原仓库 `experiments/shared/policies.py` 已支持 `rankN_drop_renorm`、`keepN_drop_renorm`，`capture_moe.py` 支持所有层替换，`metrics.py` 有 token-weighted NLL/KL 与 corpus PPL。注意旧 drop_renorm 无条件将幸存 gate 和归一到 1，而 OLMoE 默认是否归一由 `norm_topk_prob` 决定；历史 trace gate 和经常小于 1。新实现应按模型原生 renormalize 契约保留缩放，并在 T1 明示归一规则。下一步用本课题同一 checkpoint 新测纯 drop，采用 pooled NLL/token KL，同时保留 rank×gate 分桶与样本分母。

## 路由格式与真实可用数据

`capture_routes_v2.py` 输出 CSV 字段 `sample_id,layer,token_position,rank,expert_id,gate_weight`，rank 从 1 起，每文档整段 `model(**inputs)`；它记录 token 专家对，而非现成的每步 expert-count。其 metadata 含 model/dataset/split/offset/seq_len/dtype/document_sha256。把连续位置拼成 B 个 token 可用于 occupancy 回放；没有 scheduler batch_id，不能称为自然多会话 decode。当前仓库树未发现与此脚本对应的完整大规模 route CSV。

找到另一套真实 cached one-token decode CSV：`artifacts/energy_slo_routeslack/20260728_154500_rtx5090_final/evidence/prior/raw/{olmoe,llmjp}_cached_decode_routes.csv`，已连同 `.meta.json` 保存。它额外有 request_id、decode_step、phase、input_event_id 等。两模型各 2 请求×8 步×16 层；OLMoE E64/k8（2048 对），LLM-jp E32/k16（4096 对）。每请求独立执行后对齐同 decode_step 得 B=2，**非原生同时执行**。上游 metadata 明确说明 development only，原生 full 等价性尚未核验、没有 continuous batching/natural timing/能耗。OLMoE model_revision 为 `6d84c48581ece794365f2b8e9cfb043c68ade9c5`；其他权重身份遵从原 `.meta.json`，没有原始 checkpoint 二进制。

新 F1 当前右图仅展示上述历史诊断：B=2 OLMoE singleton slot fraction 0.497070，按同一 trace 层内 q 拟合的 iid 预测 0.543518；LLM-jp 分别 0.269043 和 0.378540。B=1 均为 1。只有两条短轨迹，不能用 token/层作为独立重复生成显著性区间，也不能可靠推断真实总体偏斜。**B=4–64 原生实测与 Qwen/DeepSeek 数据仍缺失**。当前结果是小样本回放模型偏差，下一步直接采集独立会话的原生 decode batch 并记录语料与权重身份。

## F1 模型与分析器

令 q_e=P(一个 token 的 top-k 集合包含 e)，而非槽归一化概率 p_e；q_e∈[0,1] 且 Σq_e=k。同一步 B 个 token iid、每个 token 专家不重复时，专家 e singleton 概率为 B q_e(1-q_e)^(B−1)，所以独占槽占比期望为

`u(B) = Σ_e q_e (1-q_e)^(B−1) / k`。

均匀 q_e=k/E 时化为 `(1-k/E)^(B−1)`。推导不要求同 token 的 k 个专家相互独立；要求跨 token iid。非同分布独立 token 可改用 `Σ_t Σ_e q_te ∏_{s≠t}(1-q_se)/(Bk)`；相关会话需要实际 batch 观测。q 按层、run 估计，不能先混层再用非线性公式。它是专家槽/逻辑唯一专家计数模型，既不是 DRAM 计数，也不自动等于延迟预测；tail/gate 可跳过子集还要由机制计数。

`code/analyze_routes.py`（仅 Python 标准库）输出 `batches.csv`、`summary.csv`、`inclusion_probabilities.csv`、`iid_predictions.csv`、`analysis.json`。三种模式明确区分：actual-decode、replay-sessions、replay-positions。固定 top-k 完整性、每 token unique expert、合法 gate 与 native forward 完成状态均检查；不够 B 的尾批不会偷偷当满批，缺失 B 明列。q 预测是同输入拟合的描述性结果，不是留出预测。

原生 `skip_worker` 可直接输入 trace JSONL、整个 snapshot JSON 或 trace 数组：

```sh
python code/analyze_routes.py --trace runs/olmoe_b8_native/snapshot.json \
  --input-format vllm-json --run-id olmoe_b8_native_r01 --model OLMoE \
  --experts 64 --top-k 8 --mode actual-decode \
  --evidence-boundary 'Native vLLM decode router capture; logging outside timed benchmark.' \
  --output-dir figures/olmoe_b8_native
```

所需原生字段：`layer`（字符串也可）、`step`、`phase=decode`、`batch_size`、`num_experts`、`top_k`、`topk_ids`、`topk_weights`、`source=vllm_actual_select_experts`、`forward_completed=true`。使用 pre-policy ids，并直接采用该矩阵的 batch 成员。当前 worker 没有持续 request_id，分析器只将行号视为单步内身份，request_count 留 null，不虚构跨步会话轨迹。每次 policy/reset 后单独保存文件与 run_id，避免 step 重用；trace_limit 应至少是欲采集 decode 步数×MoE 层数以免尾部不全。

历史回放复现（工作目录为本课题）：

```sh
python code/analyze_routes.py --trace ref_july/olmoe_cached_decode_routes.csv \
  --model OLMoE --experts 64 --top-k 8 --mode replay-sessions \
  --evidence-boundary 'Historical two-request cached decode; reconstructed batches.' \
  --output-dir figures/olmoe_historical_replay
python code/analyze_routes.py --trace ref_july/llmjp_cached_decode_routes.csv \
  --model LLM-jp --experts 32 --top-k 16 --mode replay-sessions \
  --evidence-boundary 'Historical two-request cached decode; reconstructed batches.' \
  --output-dir figures/llmjp_historical_replay
python code/plot_occupancy.py --summary figures/olmoe_historical_replay/summary.csv \
  figures/llmjp_historical_replay/summary.csv --output-dir figures
```

绘图需 matplotlib；当前复用了 `/private/tmp/format_sched_plot_venv/bin/python`，设置 MPLCONFIGDIR/XDG_CACHE_HOME 为临时可写目录。`figures/F1_occupancy.{pdf,png,svg}` 与 `F1_uniform_model.csv` 已生成。左图第三条 E256/k8 是示例细粒度形态，不冒称待测 DeepSeek-V2-Lite 拓扑。分析器已用小规模均匀/偏斜精确枚举检验公式，用手算 singleton 分母与 native JSONL 完成保护检验读入。
