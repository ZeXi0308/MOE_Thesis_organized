# 复现入口

当前进展与证据边界见 [RESULTS.md](RESULTS.md)，任务原文见 [TASK.txt](TASK.txt)。本目录不修改共享 vLLM 安装；通过独立 worker 和路由后处理接入，退出进程后解除。

远端最后可核实快照为epoch1792115499.654；之后SSH端口50328拒绝连接，实际后续状态未知。恢复连接后先核对主机boot ID、旧controller/worker身份、对应run状态及易失模型文件，再决定操作，不能依据本文重复提交。

- 已完成结果：r07 BF16七B九臂矩阵与长轨迹F1、r08b十臂free200与四任务pilot；r09虽整体FAILED，但FP8七B原生trace/调优和三B十臂质量已完成并保留。图表入口见下文及[RESULTS.md](RESULTS.md)。
- FP8 `runs/olmoe_fp8_matrix_r09b`已于epoch1792114724.516 COMPLETE，63/63格，controller63659/worker68143退出；686原始文件已逐SHA回传。两格式F2/F4和长F1均已有实测，当前仍缺独立重复与两模型完整结果。
- 最后观测时，`deepseek_bf16_matrix_r01`的controller73768为WAITING、未见worker；CPU桥70776已提交它并等待终态，原计划之后串行接续FP8。失联后的状态未知，不能据此认定它们当前实活、未运行或已退出，也不能另行提交。最后已知入口见[bridge_launch.json](deployments/qwen_partial_resume_20261016/bridge_launch.json)；旧PID64317在失联前已确认停止退出。
- Qwen首次CPU续传因真实容量不足FAILED，13个K克隆已核验后清理，C来源未改。新[DS之后续传入口](deployments/qwen_after_deepseek_20261016/commands.txt)仅在本地准备，尚未部署或执行；其容量依赖DS工作结束后另行决定是否释放K自有RAM权重，不能按假设空间直接执行，详见文末。
- 当前本地[论文PDF](paper/main.pdf)为14页，已纳入BF16与FP8完整F2/F4、双格式长轨迹F1及既有质量结果。最新14页PDF、新FP8派生图与GSM逐题配对分析尚未回传远端；远端副本不能视为本次最新产物。
- 完整test rolling、完整GSM8K与完整MC的准备命令分别在[full rolling](deployments/full_rolling_20261016/commands.txt)、[14臂GSM8K](deployments/task_quality_count_controls_20261016/commands.txt)、[MC](deployments/task_quality_full_mc_20261016/commands.txt)。新14臂入口替代原未运行的11臂GSM计划，尚须部署与GPU验证strict count core；不默认执行两批GSM。这些全量质量作业尚未提交，不在当前桥内；由主会话按剩余预算串行安排。

下文r01–r09及接续章节的命令、PID和阶段性“当前/最新”措辞均保留为对应历史记录，以本页首部的最后观测与未知状态为准；r09 controller56295/worker60387已于epoch1792109524.609退出。

## 环境与安装

已实测主机：RTX PRO 6000 Blackwell Server Edition 96GB，SM120；vLLM 0.26.0、PyTorch 2.11.0、Triton 3.6.0。OLMoE BF16完整模型已通过eager及FULL decode graph集成，并完成r07七B九臂长窗矩阵；显式Triton backend、单卡、无EPLB/LoRA。OLMoE在线FP8已完成原生调优、三B质量、七B无padding graph检查及完整63格性能矩阵；FP8 padded eager/graph输出一致性未通过。最后可核实快照未观测到DeepSeek/Qwen的GPU加载、质量或性能结果；DeepSeek请求在失联后是否获锁运行未知。用户确认没有5090，主线使用现有PRO6000的BF16/FP8。

在已有兼容环境中，可编辑安装：

```bash
python -m pip install --no-deps -e .
```

也可不安装，直接设置本目录 `code/` 到 PYTHONPATH。真机运行器 `code/run_locked.py` 自动设置 PYTHONPATH、CUDA 13 动态库路径和 sampler 配置；所有 GPU 命令均由它持既有共享锁运行。密码不进入任何脚本或文件。

## 已验证机制

```bash
python code/run_locked.py --name mechanism_unique --timeout 600 -- \
  python code/verify_mechanism.py --output runs/mechanism_unique/mechanism.json

python code/run_locked.py --name model_smoke_unique --timeout 600 -- \
  python code/smoke_model.py --model /root/autodl-tmp/moe-research-20261002/model \
  --output runs/model_smoke_unique/model_smoke.json
```

`verify_mechanism.py` 检查 B=1/2/8/32/64、global/gate/marginal 共15点：安全对齐、对独立 FP32 专家参考的数值误差、将所有未用专家置 NaN 后输出有限且不变。它是合成 BF16 权重机制验证，不是任务质量或性能测量。

`smoke_model.py` 检查真实 OLMoE 原生/关闭跳过的 full-logit KL、16层覆盖、B=1/8 的策略执行。第一轮多 prompt 捕获分组不匹配保留在 `runs/initial_mechanism_r01`；第二轮单 prompt 身份检验通过在 `runs/olmoe_model_smoke_r02`。两份原始目录已回传并按各自 SHA256 文件校验。

## 质量、性能、能耗

`code/quality_decode.py` 用真实 vLLM decode batch、固定 WikiText 后续 token 做教师强制；先 native，再各策略。full KL 按请求和位置配对，NLL/PPL 按 token 加权。prefill 产生的首 token 预测另存，排除 decode 汇总。此窗口 PPL 不称为标准全 WikiText PPL；任务准确率与自然生成采用独立评测，r08b已完成任务pilot和free200筛查，不能由教师强制指标替代。

```bash
python code/quality_decode.py --model MODEL --articles validation.jsonl \
  --output runs/QUALITY/quality.json --batch 8 --tokens 32 --prefix-tokens 128 \
  --limit-docs 32 --arms-file configs/quality_pareto.json
```

以上与以下命令实际运行都应置于 `run_locked.py -- ...` 后；示例省略外层仅为显示参数。以下为eager功能/重复测量示例；已完成的主性能结果使用下文r07的graph、2048输出协议。

```bash
python code/bench_decode.py --model MODEL --prompts validation.jsonl \
  --output-dir runs/PERFORMANCE/benchmark --arms-file configs/default_arms.json \
  --batches 1 2 4 8 16 32 64 --newtokens 256 --prompt-tokens 128 \
  --repeats 3 --warmup-seconds 3 --capture-traces
```

九臂包括原生 top-k、原生 top-(k−1/2/4)、两档 gate 和三档 marginal。原生及原生减 k 的计数另跑同长度身份重放，不把计数成本混入强基线 TPOT。计数重放必须与性能 pass 的生成 token 完全相同。策略实际执行后路由，因此候选间内部轨迹可以不同。

TPOT 是同步固定 cohort 的 host-visible token 间隔，固定 `ignore_eos=true` 控制输出长度；不能据此宣称自然生成质量。性能、能耗和路由记录分开 pass。能耗窗口在全部请求首 token 可见后开始，在其余 decode 结束时停止；主 J/token 使用 pynvml 功率梯形积分，累计能量计数器作为并列读数。小于3秒窗口标记为短窗诊断。

字节字段来自实际路由及实际权重 tensor 尺寸，是 distinct 专家逻辑权重足迹；不同策略的总逻辑足迹用自身执行轨迹计数。它不等于 DRAM 硬件读取字节，也不包括所有网络层。

`quality_pareto.json` 与性能九臂参数相同，另加dense身份臂；原生减k在质量运行中开启计数。F3用同group dense的完整top-k逻辑足迹作分母，不能用原生减k自己的before计数代替原top-k。T1独立诊断使用以下入口；每格是全模型反复应用一个rank×gate区间的策略影响，不是可相加的单事件KL。

```bash
python code/quality_decode.py --model MODEL --articles validation.jsonl \
  --output runs/T1/quality.json --batch 1 --tokens 16 --prefix-tokens 128 \
  --limit-docs 2 --arms-file configs/drop_rank_gate.json
```

## CUDA graph与强基线接线：已验证范围

GraphWorker在创建引擎前固定一个策略，每个进程只运行一臂。`smoke_graph.py`默认检查B3→4/B5→8的padded FULL replay，也可显式选择实际性能使用的七个无padding桶；dense配置额外检查同执行模式的native身份。BF16 padded检查在r06通过。FP8在r09b的B1/2/4/8/16/32/64无padding dense/marginal检查中，eager/graph输出及计数严格相同，native/dense同执行模式身份也通过；无策略、无计数的FP8 padded对照仍严格FAILED，不能把包装层DIAGNOSTIC_COMPLETE当成一致性通过。Padding路由保留，因此真实token的逻辑独占不保证padding后的物理独占；主性能矩阵逐步检查没有padding。

```bash
python code/smoke_graph.py --model MODEL --output runs/GRAPH/smoke.json \
  --spec-json '{"mode":"dense","backend":"triton"}' \
  --batches 1 2 4 8 16 32 64 --capture-sizes 1 2 4 8 16 32 64

python code/bench_graph.py --model MODEL --prompts validation.jsonl \
  --arms-file configs/default_arms.json --output-dir runs/GRAPH_MATRIX \
  --batches 1 8 32 --arms native global_j4 marginal_mid \
  --newtokens 512 --prompt-tokens 128 --repeats 3 --warmup-seconds 3
```

以上为可选子集示例；FP8另加`--quantization fp8`，实际r09b参数以其部署命令为准。`bench_graph.py --plan-only`不导入GPU运行时。正式运行每臂每重复独立创建引擎，随机臂顺序；性能进程关闭路由计数，另一个固定同策略的进程采集计数，仅在全部生成token哈希相等后关联字节。单独能耗pass包含策略及真实token标量更新的完整成本。示例的512个输出token不自动保证3秒窗口；r07/r09b正式矩阵统一2048输出，仍逐格核对纯decode窗口长度，短窗结果只作诊断。

`tune_moe.py`用当前checkpoint真实首MoE层权重及实际原生route，固定小tile网格并加入原生默认配置；仅按native时间选配置，各服务臂使用对应格式的同一个配置目录。默认BF16；`--quantization fp8`复用vLLM在线量化处理，OLMoE的两格式调优分别在r06/r09完成。OLMoE/Qwen/DeepSeek默认首MoE层为0/0/1，排除shared/dense权重。输入激活是合成值，cache conditioning不是严格cold-DRAM证明，微基准不冒充服务收益。缺失batch的原生trace会报错，不补造路由。

```bash
python code/tune_moe.py --model MODEL --traces SNAPSHOT.json \
  --output-dir runs/TUNE --batches 1 8 --plan-only
```

## 路由与图表

```bash
python code/analyze_routes.py --trace SNAPSHOT.json --input-format vllm-json \
  --run-id RUN_ID --model OLMoE --experts 64 --top-k 8 --mode actual-decode \
  --evidence-boundary 'Native decode router capture, logging outside timing.' \
  --output-dir figures/olmoe_native
```

历史图 [F1_occupancy.pdf](figures/F1_occupancy.pdf) 分开显示解析模型与有限历史回放；出处和限制见 [reuse_notes.md](ref_july/reuse_notes.md)。B=1/2 的历史回放不是当前 serving B=1–64 实测。

当前BF16长轨迹F1为[figures/f1_long_decode_r07/F1_long_decode.pdf](figures/f1_long_decode_r07/F1_long_decode.pdf)，复现命令见同目录[README.txt](figures/f1_long_decode_r07/README.txt)。它来自每B的16层×2047个真实decode步，不能与历史位置回放或短trace混作同一次测量。

论文：在 `paper/` 中运行 `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`。当前本地[PDF](paper/main.pdf)为14页阶段稿，已纳入BF16与FP8各63格完整F2/F4、双格式P128/N2048长轨迹F1、三B质量及r08b任务pilot；另外两模型完整结果与独立重复仍缺。已核验PDF SHA256为`f4710351712584f4b09ae73ebf84ee8d2233dd0ad0d9aa2f90c6da6c012f81b3`，14页均已渲染目视检查，详见[build_status.txt](paper/build_status.txt)。双格式F2/F4出版排版由[plot_format_results.py](paper/plot_format_results.py)读取两格式原CSV重建，图例在数据区外；F1源为[格式对照图](figures/f1_long_decode_fp8_r09b/F1_format_comparison.pdf)。最新PDF、新FP8图与[逐题配对分析](figures/task_quality_r08b_paired/)目前仅在本地，尚未回传远端。

## 任务与自然生成

四项任务已固定repo commit并准备在远端 `datasets/task_eval_r02`；本地保留来源清单、SHA256和位置说明。模型推理禁止自动下载，使用本地checkpoint。运行环境可用 `/root/autodl-tmp/research_C_format_sched/tools_venv/bin/python`（lm_eval0.4.13，vLLM0.26.0可见），仍必须由本课题run_locked持锁。具体配置和命令在 `configs/task_eval_protocol.json`。标准MMLU/HellaSwag/ARC-c只走prefill，作为身份对照；decode质量另由官方GSM8K生成及显式生成式多选适配测量，不能混写官方多选分数。自然生成默认200条、EOS启用，--limit仅用于明确标注的探索。

T1绘图：`python code/plot_t1.py --quality runs --output-dir figures/t1_actual_r01`，仅选择rank_gate质量结果；无实测时只生成说明清单。零动作格质量显示NA，原始数值保留；输出token与token-layer计数分别报告。

历史接续脚本：`code/run_resume.sh`用于r04未完成的计数器/性能/graph检查；`code/run_characterize.sh`准备七batch真实trace、原生调优、graph短pilot和小规模质量表。二者均须由`run_locked.py`启动，不直接绕过共享锁；当前接续应使用前部列出的r09b/串行桥入口。语料在`datasets/wikitext2_serving_r01`，bench记录是不同窗口，不保证每行来自独立文章。

r05历史目录为`runs/graph_resume_r05`，原controller24569，候锁上限3600秒、持卡上限900秒；已终止并保留21格短bench与失败的graph前置cohort检查。随后`runs/characterize_r06` / 原controller30236按`code/run_stage_r06.py`完成原生调优、graph检查、三batch质量/T1、FP8及graph pilot，原持卡上限2400秒。二者均已结束，不再按这些PID或旧命令接续；8/128 token短pilot不能当作能耗主结论。

三模型BF16调优入口 `tune_moe.py` 已支持 OLMoE/Qwen3/DeepSeek 的真实routed三投影，默认首MoE层0/0/1，排除shared/dense；CPU fixture通过，Qwen/DeepSeek GPU调优尚未运行。

自然生成：`python code/free_generate.py --model LOCAL_MODEL --prompts datasets/free_generation_validation_r01/prompts.jsonl --prompt-tokens 128 --limit 200 --batch 8 --max-tokens 256 --arms-file configs/quality_eval_arms.json --output runs/UNIQUE/free200`，须持外层GPU锁。200窗口来自28篇文章，按source_article_id联结分析。

DeepSeek权重已完整下载：终态`runs/deepseek_mirror_20261016_045110_30370/source.json`为WEIGHTS_COMPLETE，4/4分片与HF固定revision `604d5664`的SHA匹配，总31,413,626,576字节，无缺失/大小异常小文件，原下载PID30370已退出。模型位于`/dev/shm/research_K_expert_skip/deepseek_v2_lite_604d5664`，RAM易失；原始来源、迁移和small assets记录仍见`runs/deepseek_source_shm_r03/`。权重就绪不代表GPU加载通过；原持久`models/`目录仅保留小metadata，不能当成完整checkpoint。


## r06完成与r07长窗矩阵

r06八阶段全部完成；`runs/characterize_r06/SHA256.json`的1079文件均与远端一致。T1位于`figures/t1_r06`，三batch教师强制质量Pareto位于`figures/quality_r06`；均是validation探索。后续r08b已完成任务pilot，完整任务及完整test rolling仍待执行。

r07已由共享锁入口完成63/63格，原始运行命令为：

```bash
python code/run_locked.py --name olmoe_bf16_matrix_r07 --wait-seconds 3600 --timeout 7200 --   env VLLM_TUNED_CONFIG_FOLDER=/root/autodl-tmp/research_K_expert_skip/runs/characterize_r06/native_tuning/tuned_configs   python code/bench_graph.py --model /root/autodl-tmp/moe-research-20261002/model   --prompts datasets/wikitext2_serving_r01/validation_bench64.jsonl   --output-dir runs/olmoe_bf16_matrix_r07/bench --arms-file configs/default_arms.json   --batches 1 2 4 8 16 32 64 --repeats 1 --newtokens 2048 --prompt-tokens 128   --max-model-len 2304 --gpu-memory-utilization .35 --warmup-seconds 3
```

在远端实际使用`/root/miniconda3/bin/python`，外层lock设置PYTHONPATH/LD_LIBRARY_PATH。此命令为记录；接续会话必须先检查已有controller/status，不重复提交。r07共63性能格、18新模型进程，性能/能耗关闭路由计数，独立计数进程仅在输出token哈希完全相同时配对。每臂参数及九臂全集保持一致；能耗窗口实际不足3秒时仍标diagnostic。只有一个独立运行，不能以2047步当作2047次系统重复。


## 安装版0.1.3与完整rolling输入

安装 `dist/marginal_expert_skip-0.1.3-py3-none-any.whl` 后新增 `k-expert-skip-prepare-rolling` 和 `k-expert-skip-tune`；七个CLI、单源配置及真实rolling loader的干净venv安装检查见[runs/package_install_r04/verification.json](runs/package_install_r04/verification.json)。这是安装与CPU检查记录，GPU范围按各实测run另记。构造原始parquet输入可安装可选extra `rolling`（transformers、pyarrow），模型执行仍使用已验证的vLLM0.26环境。安装环境没有源码仓库的数据目录，builder须显式传 `--source-manifest` 与可访问的 `--parquet`；权重和数据不随wheel打包。

现有`datasets/wikitext2_rolling_olmoe_r01/{validation,test}`已完成CPU构造与完整token覆盖，并与远端实际OLMoE tokenizer身份匹配。r09b已通过FP8 validation首九窗1152预测及实际最后65-token短尾的十臂GPU路径检查；完整validation/test PPL尚未执行。下面保留完整test、B32十臂的直接入口，须放在外层共享锁中，目录需唯一：

```bash
/root/miniconda3/bin/python code/quality_decode.py \
  --model /root/autodl-tmp/moe-research-20261002/model \
  --tokenized-windows datasets/wikitext2_rolling_olmoe_r01/test/manifest.json \
  --output runs/rolling_test_b32_r01/quality.json --batch 32 \
  --arms-file configs/quality_pareto.json --dtype bfloat16 \
  --gpu-memory-utilization .35 --reference-max-mib 4096
```

默认是整个split；若显式使用`--limit-windows`，只能标为路径检查或部分人口，不能当完整PPL。首位置上下文128，后续窗口内增长；末尾不足整批、短窗口均计入，窗口不当作独立文章。跨模型须用各自tokenizer重建输入，Qwen/DeepSeek不能直接消费OLMoE token IDs。当前有预算的完整test接续采用[三块B8七臂入口](deployments/full_rolling_20261016/commands.txt)，覆盖全部2255窗/288602计分token，尚未GPU提交；该单B rolling质量锚点不替代B1/8/32共同32篇配对文章的F3协议。

两模型专用输入现已构造并同步至远端K同名目录：`datasets/wikitext2_rolling_deepseek_v2_lite_604d5664/{validation,test}/manifest.json` 覆盖272,943/310,997计分token；`datasets/wikitext2_rolling_qwen3_bf16_ad44e777/{validation,test}/manifest.json` 覆盖262,209/298,950计分token。按模型选择对应manifest，仍使用上述共享锁及唯一输出目录；GPU执行尚未运行。现有bench64、free200、validation/test quality32在两模型下均通过长度及前缀唯一性检查，记录为 `datasets/model_input_compatibility_r01.json`。

对应tokenizer副本位于 `ref_july/tokenizers/deepseek_v2_lite_604d5664` 与 `ref_july/tokenizers/qwen3_bf16_ad44e777`，各自 `source.json` 保存官方HF固定revision、原始文件身份和SHA256；本地transformers5.20/tokenizers0.23.3与远端5.17/0.23.2的semantic identity及输入前缀IDs一致。tokenizer目录不含权重，不能直接当作完整模型加载；Qwen的C checkpoint检查时尚缺tokenizer文件，后续仅在K自有模型目录使用已核实副本，不修改C。两模型rolling、兼容记录和tokenizer共19文件、36,531,983字节，远端传输后SHA256逐项匹配；该次输入传输没有部署code/configs。

## r07图表与r08b已完成结果

r07完整原始588文件已回传并逐SHA匹配，F2/F4和全策略CSV位于`figures/results_r07`；本次每臂只有一个独立性能运行，63个batch/策略格不等于63次独立重复。截至r07结束的历史累计GPU持锁为5122.172秒。

r08一小时候锁到期为LOCK_BUSY，没有GPU运行，原进程均退出。相同协议的`runs/task_quality_pilot_r08b`已于epoch1792108303.901 COMPLETE，controller52744/worker52785退出，持卡1827.721秒，截至该次结束历史累计6949.893秒。完整命令与环境在`deployments/task_quality_pilot_r08_20261016/r08b_launch.json`；原始5152文件已回传且逐SHA匹配。200条自由生成是来自28篇文章的完整筛查人口；四任务每leaf16样本仍仅为pilot，其中MMLU为57×16=912题。正式质量表位于`figures/task_quality_r08b`，不得再次提交同名运行。

本地新增三模型任务命令见`code/task_quality_models_commands.txt`，相应`run_task_quality.py`已接受`olmoe/qwen3_moe/deepseek_v2`的未量化checkpoint；CPU计划通过，已随r09交接部署。不能向当前候锁或运行作业使用的live code覆盖。

F5入口使用各模型/格式的`plot_results.py`输出表。以下本地CPU命令用现有OLMoE BF16数据验证，输出明确为`ONE_MODEL_PARTIAL`；未加入论文作为跨模型证据：

```bash
python code/plot_cross_model.py \
  --bench olmoe=figures/results_r07/bench_points.csv \
  --output-dir figures/cross_model_r07 --replace-derived
```

后续增加`--bench qwen3=CSV`与`--bench deepseek_v2_lite=CSV`，同模型两格式分别给表。运行、重复、格式和资源条件分面，不合并估计；`--arms`显式指定相同策略名，默认展示global_j4/gate_high/marginal_high。程序和真实单模型图面已验证，新脚本已随r09交接部署。缺失的模型/格式没有替代曲线。

## r09历史提交与部分完成

原`after_r08.py`/PID45387因r08 LOCK_BUSY停止并退出。`deployments/olmoe_fp8_matrix_r09_20261016/after_r08b.py`/PID52745在r08b成功终态并退出后部署原9项源码/配置，于epoch1792108315提交以下历史请求；`handoff_r02.json`为SUCCESSOR_SUBMITTED/controller56295。r09于epoch1792109065.236获锁，worker60387；已于epoch1792109524.609终态FAILED/内部INCOMPLETE并退出。七B原生trace/tuning与三B质量完成，padded graph失败阻止了F6、rolling和完整性能阶段；成功阶段由r09b复用，不能重交r09。

```bash
/root/miniconda3/bin/python code/run_locked.py \
  --name olmoe_fp8_matrix_r09 --wait-seconds 21600 --timeout 7200 -- \
  /root/miniconda3/bin/python code/run_olmoe_fp8_r09.py \
  --root /root/autodl-tmp/research_K_expert_skip \
  --output runs/olmoe_fp8_matrix_r09 --budget-seconds 7170
```

完整阶段参数保存在`configs/olmoe_fp8_matrix_r09.json`，各阶段实际命令/时间/状态保存在该run的`stages.json`。`--plan-only`仅stdout计划，不申请GPU或创建运行目录；本地和远端均已通过。FP8使用本次原生FP8调优目录，MoE为W8A8，外层model dtype/auto KV为BF16。预设性能为同r07的七B九臂、N2048；质量为三B十臂32篇×32decode，rolling九窗仅路径检查。原生调优/graph依赖失败不进入全矩阵；各阶段超时受总7170秒余额约束，缺失阶段不报完成。

上述历史交接器仅在r08b COMPLETE且原进程退出后修改live code，前态SHA检查已完成。任何后续部署仍须等待当前使用live code的作业退出；不覆盖运行中源码，也不自动重复失败作业。

## DeepSeek完整矩阵和质量表

`code/run_deepseek_matrix.py`及`configs/deepseek_matrix.json`已在`deployments/deepseek_matrix_20261016`暂存，本地/远端两格式CPU计划通过，尚未部署live、未提交GPU。当前由下文CPU桥70776等待r09b成功退出后串行安排，不能另行直接提交。部署后分别按config内`launch_bf16`和`launch_fp8`执行；每格式为独立输出目录，候锁21600秒、外层7200/内部7170秒，并计入原15小时总预算。计划包括完整七B九臂性能及三B十臂质量，不保证未测的DeepSeek能在该上限内全部完成。shared专家保留执行但不计入routed跳过，tail_rank4对k6仅为末两槽。

完整质量结果回传后，本地使用以下入口；部分套件也能解析，但必须保留partial状态，不能把缺失记成零：

```bash
python3 code/summarize_task_quality.py \
  --suite runs/task_quality_pilot_r08b/quality/suite.json \
  --output-dir figures/task_quality_r08b
```

入口生成CSV、LaTeX表和必要来源记录；标准MC表仅是prefill身份对照，GSM8K与自然生成才覆盖decode干预。跨套件/格式/种子不合并，论文排版时使用表内原值及其人口/状态限定。输出目录须为空以保留此前结果。

## r09部分完成后恢复

r09终态FAILED/内部INCOMPLETE；七B原生FP8 trace、专用tuning及三B十臂quality均成功并完整回传。padded graph检查失败，性能未运行，详见RESULTS。r09b已按原SHA复用这些成功产物，并完成独立native padded对照与实际无padding batch检查；这限定了通过的执行域，尚未定位padded差异根因。准确入口见下一节，不能再次提交r09同名目录。

全量OLMoE BF16 GSM8K的唯一准备入口见`deployments/task_quality_full_20261016/commands.txt`，本地/远端CPU计划通过，11臂×1319题、无limit。它未提交GPU，不能与K其他请求并发注册；实际执行需主会话在前项结束后串行安排。

## 当前唯一请求r09b

`deployments/olmoe_fp8_matrix_r09b_20261016/launch.json`记录epoch1792110034.106的实际提交；controller63659，候锁21600、外层7200/内层7170秒。已于epoch1792111130.519取得共享锁，worker68143；截至本文前部快照，完整性能为42/63格、仍RUNNING。`commands.txt`、`remote_cpu_plan.json`和固定smoke副本已部署；执行python为原r09的`/root/miniconda3/bin/python`。严格通过七B无padding验证后才运行完整63格，原r09失败记录与成功质量不改写。下一步读取该run实际状态，而非重交r09或r09b。

历史CPU交接器PID64317已主动停止并退出，原入口`deployments/deepseek_matrix_20261016/after_r09b.py`、STOPPED状态`handoff_r09b.json`及启动命令`handoff_r09b_launch.json`保留。它没有提交DS；当前唯一桥为下一节PID70776，不重启旧等待器。全量GSM目前只是准备就绪，不在该交接链内。

r09b的native padded对照严格FAILED（两个子执行PASS），七B无填充dense/marginal图检查、F6、validation九窗及实际65-token短尾均已PASS；仍需等待完整性能与外层终态，不能重交同一个run。CPU开销取`eager_thread_cpu_us_per_call`，GPU取`captured_device_us_per_call`，不要以二者或墙钟差推导另一项。该F6使用合成路由、含计数器，只衡量后处理开销，不是FP8专家GEMM或服务收益。

完整test滚动PPL入口与初跑/续跑命令见`deployments/full_rolling_20261016/commands.txt`，原数据与三个精确ID分块均已部署；`remote_cpu_plan/suite.json`仅表示计划通过。六个任务覆盖BF16/FP8各三块、全部2255窗/288602计分token/七臂；不得添加limit以伪装完整完成。实际GPU尚未提交，需在现有链结束后检查总预算与至少4GiB输出空间。成功chunk按SHA复用，失败attempt保留；最终每格式只有三块均通过才标记完整test。

## 最新串行接续（替代上文直接DS等待器）

当前CPU桥为PID70776，入口`deployments/qwen_partial_resume_20261016/after_r09b_before_deepseek.py`，实际启动见`bridge_launch.json`，状态见`bridge_r09b_qwen_deepseek.json`。原PID64317已主动停止并实际退出，`original_handoff_stop.json`和DS原STOPPED receipt保留。不要重启旧等待器或重复提交DS。

新桥等待r09b完整成功和实际退出，再执行该deployment目录的`compact_r07.py --prepare/--compact`（合计最多300秒），将189个原performance/energy JSON无损归档至`runs/olmoe_bf16_matrix_r07/archived_performance.tgz`。只有归档成员与原SHA全部验证后才移除展开副本，原`SHA256.json`不变；`storage.json`给出成员和恢复命令。远端后续分析若需要这些展开文件，应先按该记录解压并核对原SHA；本地原结果始终完整展开。归档属于存储压缩，不是数据排除。

随后调用`resume_qwen.py --execute`（CPU最多3600秒），参数以`bridge_remote_plan.json`为准；每次重读C暂停状态与容量，完整SHA验证后才标READY。Qwen private目标是`models/qwen3_bf16_ad44e777`，9/16分片在K自有`/tmp`目录、15分片在K自有`/dev/shm`目录，目标内为对应链接；这些文件易失，加载前检查READY来源与文件存在。C原始路径/进程不修改。下载失败仍会调用原DS串行handoff；Qwen GPU入口暂存于`deployments/qwen_matrix_20261016/commands.txt`，尚未在该桥内提交。整个CPU阶段不占GPU锁，DS仍沿用原共享锁和15小时预算。


## r07长轨迹F1与完整MC入口

长轨迹F1的本地CPU复现命令见`figures/f1_long_decode_r07/README.txt`；builder直接读取已完成run的native独立计数回放及对应测量身份记录，输出七B汇总和112层/B原值。可向新的完整run传入`--run --model --experts --top-k --output-dir`；不可把未完成矩阵或短trace补成完整结果。FP8同外部长度的生成轨迹比较不等于固定内容/路由的格式干预。

完整MC的准确命令与成本估计在`deployments/task_quality_full_mc_20261016/commands.txt`。远端三个CPU计划位于其`remote_cpu_plan/`，已回传并核对SHA；driver SHA256=`deb29946a39964ac0a9098ac5670ea68b2622473fe6d471f16ca200e710c4246`。该入口未提交GPU、不在当前串行桥内。每次选择一个family、显式内部时间预算；所有臂始终评完整人口。默认十臂，也可用native-first的`--arms`明确子集；产物记录exact选择并检验metric分母、task叶、原始唯一文档数和shot。标准MC仅prefill控制，不将其准确率解释为decode跳过质量。主会话在现有GPU链结束后核对预算与存储再执行，不并发注册三个请求。

本地`code/summarize_task_quality.py`的当前版还生成`gsm8k_documents.csv`、`gsm8k_paired_documents.csv`和`gsm8k_paired.csv`，按官方doc_id及三种身份hash严格配对完整人口，并分别保留两种GSM提取filter。r08b派生示例在`figures/task_quality_r08b_paired/`；全量1319题仍待运行。当前改动仅在本地，未替换运行中live或已暂存的DS副本；分析命令仍与上文相同，新输出目录必须为空。


## r09b完整结果与Qwen容量后续

r09b全量原始结果已在本地`runs/olmoe_fp8_matrix_r09b`，其686文件与终态SHA256全部一致。CPU重建F2/F4使用`python code/plot_results.py --bench runs/olmoe_fp8_matrix_r09b/bench --output-dir figures/NEW_FP8_RESULTS`；长轨迹F1命令见`figures/f1_long_decode_fp8_r09b/README.txt`。完整性能只有每臂一次运行，不能由63格或2047步构造独立运行CI。

原Qwen固定放置不得重试：首次FICLONE虽成功，但df按完整克隆大小扣可用空间，零前缀成本假设失败。失败与精确清理记录在`runs/qwen_partial_resume_20261016_094007_73728`；13个新K克隆已回收，原C完整/partial不变。先继续既有DS链；新Qwen方案必须按真实全副本容量计费，只有DS相关GPU进程退出、其后续必要任务完成后，才可由主会话单独回收可从固定源重建的K自有RAM权重并复用空间。当前没有释放DS或重启Qwen。原r07远端performance/energy已按storage.json无损归档，本地仍展开；复现读取远端原路径前需按记录恢复。

## 本地准备：DS之后的Qwen续传

新入口[deployments/qwen_after_deepseek_20261016/commands.txt](deployments/qwen_after_deepseek_20261016/commands.txt)及[capacity_plan.json](deployments/qwen_after_deepseek_20261016/capacity_plan.json)仅完成本地CPU构建检查，尚未部署、执行下载或释放DS权重，也不在已提交的串行桥中。完整C分片核验官方SHA后只读引用；缺失9/16分片放K自有rootfs，其余缺失分片放K自有`/dev/shm`，C连续前缀普通复制到独立K文件，按全部剩余私有字节计费。该入口不修改C、不调用旧reflink下载、不自行释放DS；READY仅表示全部文件身份验证通过，不代表模型加载通过。

容量表是条件计算而非当前可用空间：若DS全部必要GPU工作和补跑已结束、实际进程退出，并由主会话另行决定释放其31,413,626,576字节RAM权重，则按记录的free与现存52,428,800字节Qwen前缀，扣除新载荷及保留量后，persistent/rootfs/shm分别高于保留线8,591,059,640 / 2,801,086,216 / 5,310,190,208字节。保留线为persistent与shm各20GiB、rootfs10GiB，另留persistent实验输出2GiB、每文件系统256MiB开销及至少17,424,291,144字节C未来追加空间。执行器只用重新实测的free，不加入假设释放量；空间不足即拒绝。

SSH恢复后必须先核对旧DS controller73768、桥70776及其实际后继任务，确认boot ID、进程身份、所有必要工作和易失模型文件状态。失联不是退出或重启证据；未完成这些核对前，不释放RAM、不重复提交DS、不启动新Qwen下载。新入口的明确CPU下载上限为7200秒、最多两个下载worker，仍须由主会话选择合适I/O窗口后单独部署和执行。

## 本地准备：仅改变独占条件的对照

[singleton_ablation_20261016/commands.txt](deployments/singleton_ablation_20261016/commands.txt)提供隔离core、8臂严格消融及14臂完整任务配置；`require_singleton=False`仅关闭marginal的count资格条件，保留含上界gate/mass、rank、预算顺序及模型默认归一化，旧T1半开分桶不变。新core SHA256为`fb9ed78cf942c21b6e5c160dfece6169e8319326163a19cfa73f7ca862ab5d31`，14臂文件SHA为`7e81ca78fd1c7932a509c56f84d1733a79d91138d56e29cb332b727c9e2b92f5`。本地CPU实际21项通过、3项CUDA跳过，尚未Triton编译或GPU执行；原live code与wheel 0.1.3仍不含此开关。

不要仅设置PYTHONPATH后直接执行原`code/quality_decode.py`并假定用了新core：脚本目录可能优先导入旧实现。隔离执行需使用命令中明确的`python -m`方式并核对实际core来源；正式任务入口必须核验父进程及worker使用同一部署版本。已有DS链退出前不替换live core，不把新配置传给旧版SkipConfig。恢复后先按commands进行一次受锁的数值/graph检查，再进入完整任务评测。

完整任务入口见[task_quality_count_controls_20261016/commands.txt](deployments/task_quality_count_controls_20261016/commands.txt)，driver SHA256为`08e7d0bf30cd36af59cb66a70887b3123bf9438bdd5902c0d2f74b81dd3d3028`。它复用原harness运行官方1319题×14臂=18466次生成，5-shot/seed1701/B8/cap256，无limit；执行前强制核验live core的SHA及实际import位置，子进程固定root/code，MP=0。本地CPU plan通过，明确保留未部署与本机缺远端model config警告；完整人口校验及旧core执行拒绝已检查。当前未部署、未GPU运行。默认suite7080/job7070/outer7140秒，预算可显式指定；粗估约90分钟不是完成保证，总15 GPU小时仍由root按实耗核算。旧11臂入口保留历史而未运行，新旧不默认双跑；部分结果保留，不自动缩人口或扩预算。
