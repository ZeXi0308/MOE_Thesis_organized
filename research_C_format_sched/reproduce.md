# 当前可复现入口

本目录跟随 `RESULTS.md` 更新。真实环境见 `runs/env_20261016_r02/environment.json`，用户完整任务见 `TASK.txt`。密码不保存在脚本或文档。

## 本地数据回收

已建立 SSH ControlMaster 的会话可使用：

```bash
rsync -az -e 'ssh -S /tmp/c-format-ssh.sock -p 50328' \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/research_C_format_sched/runs/ \
  research_C_format_sched/runs/
python3 research_C_format_sched/code/verify_local_data.py
```

新会话先交互式 SSH 认证，沿用用户提供的主机；不要把密码写入命令行。下载没有 `--delete`。已有运行必须先查远端 PID 和状态，不能因为 SSH 观察中断而重启。

可在本地研究目录启动 `python3 code/sync_terminal_run.py --name <运行名> --wait-seconds 7200`。它只等待该组终态SHA清单、回传并校验，输出`runs/<运行名>.local_sync.json`；必须为`VERIFIED_LOCAL`才算回收成功。进程复用已认证socket，不保存密码、不占GPU、不删除结果。新建量化checkpoint若位于`runs/`外，仍须单独回传其完整目录并按checkpoint manifest核验。

## GPU 实验

上传本目录 code 后，在远端根目录执行。完整vLLM/质量入口需先设置 `export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib`。`run_locked.py` 非阻塞取得既有公共锁、检查计算进程，保留提交源码和失败日志，运行超时后终止本组子进程，累加20小时预算。每次使用新名称。

```bash
/root/miniconda3/bin/python code/run_locked.py --name example_new_name --timeout 300 -- \
  /root/miniconda3/bin/python code/bench_expert_formats.py --model olmoe --format fp8 \
  --tokens 8,128,2048 --routing uniform --cuda-graph \
  --out runs/example_new_name/measurements.jsonl
```

未传 `--weights` 的 BF16/FP8 是合成权重 kernel 刻画，不能作质量或端到端服务证据。4-bit 不接受无校准的权重；格式和布局以 `format_backends.py` 的 artifact 元数据约定为准。

`run_characterization.py` 是首次两格式、两形状、uniform/Zipf探索，使用CUDA graph replay、每点5次event重复及单独能耗窗口。原始点保留全部重复，不把同进程点当独立系统重复。图中的每输入token能耗仅指一个MoE层。

## 论文与质量工具

`paper/main.tex` 是可编译骨架，数据占位明确标记。标准 LaTeX/BibTeX 编译；尚无完成的质量或 serving 图表。

`code/prepare_quantization.py` 与 `code/eval_quality.py` 提供后续 pinned checkpoint、校准和 lm-eval 入口；运行前核对其 `--help`、已装依赖及空间。准备好的代码不是GPU验证证据。

质量环境为 `tools_venv/bin/python`（lm-eval 0.4.13，共用基础vLLM/CT0.17）；校准为 `calibration_venv/bin/python`（llmcompressor0.14.0/CT0.19）。不要把校准venv用于vLLM推理。版本与依赖清单在 `runs/quality_setup/`。

所有质量运行均由 `run_locked.py` 包裹，并显式传入本次 `--lock-status runs/<job>/status.json`，包括原 stock BF16 入口。以下为未执行的全量五任务命令示例；不加 `--run` 时仍是 CPU 只读计划，旧的无 `--worker-config` 调用继续可用。

```bash
env HF_HOME=/root/.cache/research_C_format_sched/hf /root/miniconda3/bin/python code/run_locked.py --name quality_olmoe_bf16_new_run --timeout 1800 -- tools_venv/bin/python code/eval_quality.py --checkpoint-manifest runs/olmoe_checkpoint/checkpoint_manifest.json --format bf16 --enforce-eager --output runs/quality_olmoe_bf16_new_run/quality --lock-status runs/quality_olmoe_bf16_new_run/status.json --run
```

使用真实 sidecar 时另传 `--worker-config <完整逐层JSON>`，入口强制 eager、设置实际 `worker_cls`/`C_FORMAT_DUAL_CONFIG`，保存 host 和全部 sidecar SHA256、独立安装报告及被评模型身份。`policy=single` 可推断四种单精度；双格式明确记作 `dual_fp8_awq` 或 `dual_fp8_nvfp4`。single FP8、single AWQ 和 dual 应使用同一个共同输入基 BF16 master 及相同的非 expert host loader，不能因 sidecar=FP8 额外开启全模型在线 FP8。`configs/quality_worker_*.example.json` 只提供层模板，必须替换真实 artifact/basis 并列齐全部 MoE 层。默认五任务仍无 `--limit`；它们的 MC prompt-loglikelihood 分数只覆盖 prefill，不是多步混合 decode 生成质量。上述接线和参数透传目前仅通过 CPU 检查，尚无 GPU 质量分数。

### F3 留出生成诊断（数据与 CPU 接口已验证，生成未运行）

`configs/quality_generation_gsm8k_128.json` 选择安装 lm-eval 0.4.13 的 `gsm8k_cot_zeroshot`，保留官方 prompt、停止串和 strict-match/flexible-extract 两种 exact-match 过滤。来源为 [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)，revision `740312add88f781978c0658806c59bc2815b9866`，main/test 共 1,319 条；seed=20261016 无放回选 128 条，ID 升序传入 harness，禁用 request cache。这里的 held-out 指未用于本项目校准或选择规则，不保证模型预训练无污染。128 题是有 gold 的小样本多步生成诊断，不能冒充完整 GSM8K 分数，也不替代原五项全量 MC。

固定 greedy、原生 chat template、`enable_thinking=false`、batch=4、每题 cap=512（显式覆盖 adapter 默认 256），自然 EOS 和官方 task stops 均生效，绝不使用强制 128-token 工作量模式。每臂最多 65,536 个生成 token，生成阶段含初始化的内部墙钟预算 900 秒；外层锁预算需额外留清理时间。首次只执行已有共同权重的固定臂及候选混合臂，不能凭未验证的任务覆盖展开全组合。样本、cap、batch 和模板在比较前固定，不按答题正确性、生成长度或格式结果筛选。

单个 pinned 数据集和 gold 人口已在 CPU 上准备完成，入口如下（再次执行须用新目录；会读取/下载该任务数据，无模型推理）：

```bash
env CUDA_VISIBLE_DEVICES= HF_HOME=/root/.cache/research_C_format_sched/hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 tools_venv/bin/python code/eval_quality.py --prepare-generation-data --generation-config configs/quality_generation_gsm8k_128.json --output runs/generation_data_new_run
```

实际产物 `runs/quality_generation_data_r01/` 的 4 文件共 277,426 字节已回传本地并逐项 SHA256 核验。`generation_summary.json` 为 `prepared_no_generation`，完整 128 题全为 `not_started`，输出 token 为 0、metrics 为空；这是数据准备，不能记作质量结果。固定版本 test 1,319 条与 train 7,473 条已进入独立任务缓存，实际评价仅选定 test 人口。

正式质量运行使用已有 `run_locked.py` 命令，传同一真实 `--checkpoint-manifest`、对应 `--worker-config`、`--generation-config configs/quality_generation_gsm8k_128.json`、`--lock-status` 与新输出目录，不能同时传 `--tasks`/`--limit`。不要为 single FP8 改成另一份全模型 FP8 host。`generation_population.json` 保存固定 ID、题目、gold、dataset fingerprint 和内容哈希；`generation_records.json` 保存完整 128 人口、实际 prompt/output token IDs、raw/scored text、官方分数及 EOS/stop/cap/error/timeout/unfinished 状态。超上下文会保留失败并停止，不截断或换题；cap 和无法提取答案的完成输出仍参与官方评分。引擎错误/超时/未完成使本臂 aggregate 无效，失败人口不从分母删除，已产生输出和失败记录全部保留；硬终止后最后快照仍可识别未完成。`generation_worker_statistics.json` 记录实际 phase/format 选择计数，只有实测证明两格式参与，才可解释为真正混合生成质量。

本地 CPU fixtures 已覆盖重排后的多请求映射、EOS/task stop/cap、超时/未完成人口及拒绝静默截断；这些不是模型质量结果。`configs/check_generation_quality_cpu.py` 在实际安装环境中绕过真实引擎构造，以明确的假输出调用官方 `generate_until`。首次因夹具缺少构造器本会设置的 `truncation_side` 失败；补全并核对该生成路径的全部构造器状态后，远端 CPU 检查退出 0，确认未构造引擎、未初始化 CUDA、未下载任务、未测质量。失败与修后结果记在 `runs/quality_setup/generation_cpu_checks.json`；无需重复检查。所有新原始人口、生成记录和分数仍须按同样回传/哈希规则保存本地。

`run_locked.py --wait-seconds 600`可有界等待共享GPU，每次仍为非阻塞取锁；取锁后核对NVML计算/图形进程，等待时间不计入GPU持锁预算。只保留一个候卡任务，连接恢复先查其status/PID。组结束后先rsync回传并执行`verify_local_data.py`，再提交下一组。

AWQ补齐曲线 `qwen3_awq_survey_r06` 已完成：`run_format_characterization.py --uniform-tokens 2048,8192 --graph-output-check bf16-roundoff` 补两个 uniform 与七个 Zipf 点，其余五个 uniform 点来自 r01；新九点持锁 118.417 秒，50 文件 954,173 字节已回传并核验。新检查要求输出有限、relative L2≤BF16 epsilon/8 且 maxabs≤BF16 epsilon×max|reference|；三个非逐位相等点的原 bitwise 标志和误差均保留，不能写成 exact。r01 的原失败与 r02–r05 四次未启动测量的失败状态也保留。

当前四格式图的本地入口是 `code/plot_fourformat_characterization.py`，默认读取 r06 补齐数据，只读取本地归档，生成 `figures/F1_qwen_four_formats.pdf`、`F2_qwen_awq_nvfp4_energy_v2.pdf` 和 `qwen_fourformat_characterization_summary.json`。schema=2 的 summary 覆盖 56/56 点、280 次进程内时延重复与 28 个 v2 能耗窗，无缺失；不插值推断精确交叉点。能耗图仅含同为 v2 的 AWQ/NVFP4 各 14 点，不混入 BF16/FP8 的旧恢复窗口。summary 保留失败行、来源哈希、数值接受条件及总/扣空闲比率，不能当作独立重复、同权重比较或 serving 收益。

`runs/awq_compiled_paths/evidence.json` 将实际 B=1 profiler 的 demangled Marlin 名称精确匹配到 SM120 SASS 中唯一选中的函数，其中含 `HMMA.16816.F32.BF16`。命令、库哈希及反汇编共五个归档文件已本地哈希核验；这是该已观察 specialization 的静态指令证据，不是动态硬件计数或整个 AWQ 扫描的逐指令覆盖。

`runs/nvfp4_compiled_paths/evidence.json` 将历史 B=1 profiler 符号精确匹配到当前 `_C_stable_libtorch.abi3.so` 的唯一 SM120 函数，含 64 处静态 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`；[NVIDIA 官方 Blackwell 指令表](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-binary-utilities/index.html#blackwell-instruction-set)将 OMMA 定义为 warp 内 FP4 矩阵乘加。五文件已本地 SHA 核验。原 trace 未保存库 SHA，故不能独立证明历史二进制身份；这不是动态执行计数、利用率或性能归因。

共同输入基校准入口新增`prepare_quantization.py calibrate --common-basis-bf16-output <目录>`：在压缩前保存完整AWQ平滑后的BF16主权重，保留norm/router/attention及原始参数名；内容身份与原HF base commit分开。`export_bf16_fp8_artifact.py`从这份master生成FP8；`export_awq_artifact.py --manifest <AWQ manifest>`及`export_ct_nvfp4_artifact.py --manifest <CT NVFP4 manifest>`接受真实校准输出。CPU检查不是GPU导出或质量证据。

## 已制备的 serving 与路由输入

`code/prepare_serving_data.py` 的 `fetch` 与 `prepare` 分开执行；已完成产物在 `datasets/serving_olmoe_r01/` 和 `datasets/serving_qwen3_r01/`，原始源文件在 `datasets/serving_sources/`。本阶段仅用 CPU/tokenizer，没有加载模型或生成输出。固定来源如下；完整仓库名、模板、版本、文件长度和哈希也在各产物的 `manifest.json`。

| 原始文件 | 仓库与固定 revision | SHA256 |
| --- | --- | --- |
| `sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json` | `anon8231489123/ShareGPT_Vicuna_unfiltered@192ab2185289094fc556ec8ce5ce1e8e587154ca` | `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| `longbench/data.zip` | `zai-org/LongBench@5e628be450b7e67fb7ae6e201bd6d8f7056f7672` | `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` |

以下为 CPU 复现命令；已有源文件会先核验复用，已有制备目录不覆盖。`fetch_status.json`、PID 与日志用于连接恢复后核实原任务，避免重复启动。远端使用 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1` 获取固定的两份文件。当前数据已下载并备份，无需再次获取。

```bash
cd /root/autodl-tmp/research_C_format_sched
env CUDA_VISIBLE_DEVICES= HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 /root/miniconda3/bin/python code/prepare_serving_data.py fetch
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /root/miniconda3/bin/python code/prepare_serving_data.py prepare --tokenizer /root/autodl-tmp/moe-research-20261002/model --output datasets/serving_olmoe_new_run
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 calibration_venv/bin/python code/prepare_serving_data.py prepare --tokenizer runs/checkpoint_probe/qwen3_bf16 --output datasets/serving_qwen3_new_run --no-enable-thinking
```

默认 seed=20261016；完整扫描 94,145 条 ShareGPT 和 400 条 qasper/multi_news，分别有 69,656 与 212 条合格，按类 reservoir sampling 各取 128 条后固定打乱顺序。所有 94,545 条记录的来源 ID、资格、排除原因和是否抽中均保留在 `population.jsonl`；选中的原始记录另存。ShareGPT 取首个人类轮，LongBench 使用 manifest 中记录的任务模板，两者再套 OLMoE 原生单 user 轮 chat template。设置 `add_generation_prompt=True`、tokenize `add_special_tokens=True`、不截断；要求 prompt 加 128-token 输出预算不超过 4,096，长输入至少 1,024 token。该 LongBench 子集用于负载，不是官方任务质量得分。

`serving_natural.jsonl` 为 256 条 greedy、自然 EOS、128-token 上限请求；`serving_fixed_workload.jsonl` 保持同一 prompt 和顺序，设置 `ignore_eos=True`、`min_tokens=max_tokens=128`。后者只固定生成长度，不是已记录 token 的回放或质量等价证明。两文件不含固定到达时间，交给 `bench_serving.py` 以相同 seed、λ 生成相同 Poisson 请求日程。`routes_natural_32.jsonl` 为两类各 16 条的固定子集；接入 `capture_model_routes.py` 时使用 `--max-prompts 32 --max-new-tokens 32 --add-special-tokens`，只作真实路由诊断，不作 vLLM 性能测量。上述消费者仍需保留自然 EOS、达到上限、错误、超时及未完成的完整人口。

Qwen3 使用自己的原生 template，并显式传入 `enable_thinking=false`；同样完整扫描 94,545 条，ShareGPT 合格 69,641、LongBench 合格 217，选中各 128 条（qasper 47、multi_news 81）。各模型的分词和资格过滤独立，因此不是跨模型逐 source ID 配对的请求集；同一模型内各 arm 仍复用完全相同的输入文件与顺序。Qwen3 prompt 范围为 ShareGPT 13–2,361、LongBench 1,081–3,962 token，遵循相同的 4,096 上下文及 128 输出预算。全部资格与排除统计在各自 manifest/population 中。

两份原始源文件共 786,770,471 字节、OLMoE 的 9 个制备 payload 文件共 36,218,172 字节，已连同 tokenizer 快照、manifest、日志和状态回传本地并核验 SHA256；Qwen3 的 9 个文件共 44,250,424 字节亦已本地核验。回执分别在两个制备目录的 `local_backup_receipt.json`。OLMoE CPU 已检查两套请求读取/顺序、32 条路由子集及保存 tokenizer 对全部 256 条长度的精确复算。两模型均尚未运行真实生成、路由捕获或 serving。

## Serving 执行入口（待 GPU 验证）

`run_serving_group.py`默认只读plan，`--run`需真实run_locked祖先；启动本地vLLM服务、执行显式trace/λ/repeat并仅清理自己创建的进程组。`dual_format_worker.DualFormatWorker`通过`C_FORMAT_DUAL_CONFIG`逐层加载sidecar，在KV预算前计入驻留；threshold/expert_split/phase均仍待GPU验证。phase使用实际computed位置与prompt边界，profile/warmup统计单独保存，正式计数随后重置。完整模型质量可用`eval_quality.py --enforce-eager`明确关闭编译/graphs；五项默认数据已缓存到`/root/.cache/research_C_format_sched/hf`。


当前校准接续配置为 `configs/olmoe_awq_common_r03.json`：原512×512样本与方法不变，首层实测成本表明1800秒不足，当前采用7200秒上限。`r02`受控停止的日志与partial输出已本地核验，不能把它当完整checkpoint。

```bash
bash configs/run_olmoe_awq_common_r01.sh --plan configs/olmoe_awq_common_r03.json
# 仅主会话取得可用GPU时用--run；当前r03正在运行，不要重复提交。
# 以下两条在本地执行，等待终态后自动回传；当前两路watch已启动。
python3 code/sync_terminal_run.py --name olmoe_awq_common_r03 --wait-seconds 10800
python3 code/sync_derived_checkpoints.py --config configs/olmoe_awq_common_r03.json --run-name olmoe_awq_common_r03 --wait-seconds 10800 --execute
```

`code/calibrate_nvfp4_from_common.py`/`configs/olmoe_nvfp4_common_r01.json`是后续独立入口，默认仅CPU计划：复用r03的AWQ-smoothed BF16母本，采用官方NVFP4激活校准且仅量化专家，不再AWQ或创建第二母本。真实calibration_venv的CPU recipe/FP4 pack接口检查通过，但尚未GPU执行；必须先有完整r03 pair及本地备份。新NVFP4输出不属于上述AWQ pair同步器范围，需另行回传并核验其全部checkpoint文件。
