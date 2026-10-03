# A 线自然结束任务与质量评估入口

状态：32 题文本与独立参考答案已备好；CPU 检查通过，**已离线重新分词并准备三组输入**，尚未执行模型或得到质量/EOS结果。本目录只读复用了相邻项目的输入与评估代码，没有读取其生成结果、修改它或使用网络/GPU。

现成入口是 `prepared/dev16_burst_kv4096`、`prepared/dev16_burst_kv768` 和 `prepared/holdout16_steady_kv768`，各含 `inputs`、`warmups/short`、`warmups/long`。前两组的测量 workload 和暖机 workload 文件逐字相同，只有配置中的资源预算不同。三组均通过原 A `load_inputs`；每次准备都核对全部32个已渲染文本与重新生成的 token hash。详见 [PREPARATION_STATUS.json](PREPARATION_STATUS.json)。

环境为 `/private/tmp/moe-c-input-env/bin/python`，已有 `tokenizers==0.23.2`，无 Transformers/Jinja2，未新增安装。`tokenizer_backend.py` 用实际 tokenizer.json重新编码；渲染器只支持与固定模板全文相等的单user分支，做明确的字面展开，**没有声称执行Jinja**。全部32个目标提示与旧固定版本逐字/逐token哈希匹配；独立暖机沿用同一受限格式。两个长暖机各2064 tokens。

`tasks_text.json` 收录 GSM8K test 32–47（dev16）与 48–63（holdout16），保留原八例 CoT 提示。`reference_answers.json` 从输入 gold 取得参考答案，并逐题附了根据题意独立推导的算术式；32 式均与 gold 精确一致。它们相对 A 的文章输入是新任务，但相邻项目已使用其中部分输入，不能称全局未见、盲测或训练数据无污染。

| 文件 | 用途 |
|---|---|
| `tasks_text.json` | 32 个完整文本问题与原提示；没有 token 数组、生成输出或目标题答案 |
| `reference_answers.json` | 外置 gold 和独立算术核对，不送入目标题提示 |
| `model_profiles.json` | Base/Instruct 权重 revision 与各自 tokenizer 三文件 hash |
| `prepare_inputs.py` | 使用已在本地的 tokenizers 与固定 metadata，产生 A 格式 inputs、独立 warmups、容量描述 |
| `tokenizer_backend.py` | 离线真实分词及固定单user模板展开；不下载或加载模型 |
| `evaluate_quality.py` | 解码 A raw；全请求准确率、自然结束/撞 cap、成对输出/答案变化与服务成本 |
| `runner_adapter.patch` | 从 `candidate_native_oldest_strong_r01/pkg` 生成的三个最小接入修改 |
| `adapter_source.json` | 补丁源文件范围；仅定位用 |
| `check_cpu.py` | 参考算术、解析边界、人工 cap/unfinished capture 会计检查 |

优先用 `allenai/OLMoE-1B-7B-0924-Instruct@7f1c97f440f06ce36705e4f2b843edb5925f4498` 做质量与自然结束探索。它使用原 Instruct chat template 和 assistant `Answer:` 前缀。Base 的 `6d84c485...` 保留为模型差异对照，使用相同八例内容的纯 completion 提示；Base 继续生成问题/答案、较少 EOS 是合理风险，不能把它强行停在正确数字就称自然结束。两个 tokenizer.json 的 hash 不同（Base `a094266a...`，Instruct `b1fb1517...`）；本入口重新分词并核对元数据，不移植 token ID。

512 tokens 是第一轮一致输出上限：这些题是数步小学应用题，并非困难长推理任务，八例提示帮助模型沿用短解答格式；但实际是否足够必须看 EOS/length 分布。256 更易截断，1024 会给循环/重复更多时间。不要逐策略调整 cap。若 512 下仍广泛撞 cap，保留该结果；之后另建整个 cohort 的 1024 敏感性格，而非将已有上限样本重新命名为正常完成。准确率低或大量格式缺失时应先分析模型/提示，不据此宣称调度损害质量。

已有 Instruct 输入的真实长度可用于容量算术：dev16 为 662–763 tokens，初始 704 页，所有请求都达 512 cap 的上界为 1216 页；holdout16 为 664–729 tokens，初始 697 页、上界 1209 页。合计32题上界 **2425 页 < 原 A 4096 页**。因此4096页格是有价值的低压成本/质量负控，不能指望出现KV抢占。候选压力配置可先选16题768页、或32题1536页；它们只是明确容量域，实际恢复动作仍须由原生记录确认。每格所有策略共享同容量、输入和上限。不要为制造恢复把实际任务输出改成固定长生成。

需要新配置时可执行如下命令；它们不下载权重/分词器，且输出目录必须不存在。上面三组已准备好，不必重复生成：

```sh
/private/tmp/moe-c-input-env/bin/python prepare_inputs.py --model instruct --split dev16 --arrival burst \
  --tokenizer-dir /path/to/pinned/instruct/metadata \
  --model-path /path/to/already/local/instruct/weights \
  --max-output-tokens 512 --usable-kv-blocks 4096 \
  --output /path/to/new/healthy-dev16-low
```

`--split holdout16`、`--split all32` 与 `--arrival steady --arrival-gap 0.2` 提供固定的输入/到达变化；本入口不替用户自动启动网格。Instruct 重新分词还要求渲染全文与旧 pinned prompt 一致、token hash一致；不一致直接报错，不悄悄截断或继续用旧ID。新包 `inputs` 和 `warmups` 应分别使用这个准备结果。

在新包应用 `runner_adapter.patch`，或者将对应小改动移入新的主原型 runner。现有已归档包保持原字节。修改点已具体落实：

1. `run_recovery_cadence.py`：把128文章/0.2秒/1024输出断言换成16/32真实任务 schema、完整 prompt 加 cap≤4096、自然 EOS 合同；从输入读取请求数、输出cap、KV页数与字节。模型继续来自 config，warmup 与测量必须同一模型。
2. `safe_static.py`：开放16/32请求、256/512/1024 cap，并保留真实物理KV类型/容量/空池检查。其 safe_cap 只是最大长度保守容量描述，不能据此声称测量实际采用该 cap；原 runner 仍用 max_num_seqs=32。
3. `request_measurement.py`：显式空 stop/stop_token_ids，并新增 `finish_reason` 与真实 `native_stop_reason`。保留旧 `stop_reason=finish_reason` 字段兼容已有分析。未加逐 token 新埋点。

本补丁没有改强基线/候选的调度动作。若用于较新的 quantum runner，保留其机制，只移植输入和测量合同；不要用本补丁覆盖后来版本。`run.sh` 的运行目录/授权环境、候选包身份及共享锁仍由主任务接入。Instruct 的权重/生成配置、实际 KV tensor、host16GiB、EOS metadata 应在新原生初始化中正常确认，不能沿用 Base 的模型载入记录。

```sh
/private/tmp/moe-c-input-env/bin/python evaluate_quality.py --inputs /path/to/healthy-dev16-low/inputs \
  --tokenizer-dir /path/to/pinned/instruct/metadata \
  --run native=/path/to/native-output \
  --run ordinary=/path/to/ordinary-output \
  --run candidate=/path/to/candidate-output \
  --output /path/to/new/quality.json
```

主质量分数是明确答案标记（`answer is/:/=`、`####`、boxed）后的**精确数值匹配**，允许等值小数/千位逗号；没有明确答案时不拿推导中的最后数字凑主分数。另报与邻项目一致的历史 last-number string-match，方便比较格式原因。未完成/缺失请求始终在分母中且记错；撞 cap 的正确答案可以计入总准确率，但不能计入 `correct_and_natural_stop`。原 A capture 没有真实 stop identifier，兼容分析只能在无额外stop合同下推断 EOS-compatible stop；新补丁保留该 identifier。真正自然结束与输出等价均必须根据实测报告，不能由 `ignore_eos=False` 设置推断。

服务代价使用实际返回token数/整段测量墙钟、arrival到完成时间、TTFT和每请求最大正输出host间隔；不把暖机计入episode，不把多token块内时间插值。所有策略的完整输出与长度都保留。16或32题只能检测明显质量问题，不证明质量等价；相同答案也不意味着相同输出轨迹或等工作量加速。

验证：`PYTHONDONTWRITEBYTECODE=1 python check_cpu.py` 通过32参考算术、9解析边界与7人工capture会计检查；三文件适配补丁在内存中编译通过。另已完成上述真实离线分词、全32源prompt匹配与9个 A 输入文件夹加载；没有安装依赖或模型运行结果。
