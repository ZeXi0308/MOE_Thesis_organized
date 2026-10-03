# LTR-style r02 → H128 同输入强基线：只读审查

状态：**审查完成；H128 LTR 性能包未构建、GPU 未运行。** 此报告只界定 root 在 G64 原生生命周期诊断合格后可构建的一个新包。已接受的 `20260915_natural_ltr_style_component_r02` 仍是 G64/T30/Q10/180 s **单格诊断包**，不能把替换输入后的运行称作 r02 执行，也不能把其 CPU fixture 称作物理 store/load 合格。

## 已核对的共同合同

隔离检出：`/private/tmp/moe-recovery-source-20260929-r01`，与主仓库所记 HEAD `76d6d888de42081c63cd440a8a67623161d8181f` 对齐。r02 `manifest.json` 的 **30/30** 文件 SHA-256 匹配。以下相对路径以此隔离检出中的 `refine-logs/expert_saturation/outputs/admission_capacity/` 为根：

| 项目 | r02 G64 | H r02 / H1 候选 |
|---|---|---|
| 测量输入文件 | `20260915_natural_ltr_style_component_r02/pkg/inputs/workload.json` SHA `7b0b86ff…`；64 请求、12.6 s 到达跨度 | `20260915_natural_cadence_holdout_r02/pkg/inputs/workload.json` SHA `dd8ac656…`；128 请求、25.4 s 到达跨度；H1 candidate 中字节相同 |
| 配套输入 config | G 文件 SHA `d7292a02…`，逻辑 `workload_sha256=1cffba78…` | H 文件 SHA `116fb4bc…`，逻辑 `workload_sha256=606f71fd…`；H1 candidate 中字节相同 |
| H 辅助输入 | r02 G manifest 无 `INPUT_STATS.json` | H `INPUT_STATS.json` SHA `8e3f594e…`；新包若携带须加入新 manifest |
| 模型/运行资源 | 同一 OLMoE revision/tokenizer、seed `20260905`、0.2 s 源序到达、自然 EOS、生成上限 1024、cap32、1024 batch tokens、4096 usable GPU KV blocks + null、16 GiB/8192 host KV blocks | H 输入 config 中这些字段相同；H1 runner 再在安装后检查实际 KV/host 分配，不能仅凭 config 断言现场相同 |
| warmup 与共用测量源码 | r02 与 H1 candidate 的 short/long warmup config/workload、`native_capture.py`、`request_measurement.py`、`memory_telemetry.py`、`metrics.py`、`rotation_native.py`、`staged_save_contract.py`、`native_store_delta.py`、`native_offload_observer.py`、`runtime_source_hashes.json` 逐字相同 | 可复用相同测量和底层原生接口；LTR 上层选择器与 H1 eager adapter 本来不同 |

r02 `pkg/run_ltr_style.py:32-42` 的 `load_inputs` 已在 CPU 上直接读取 H 原件，得到 `PASS_CPU_INPUT_ONLY`：128 个唯一请求，长度 460–3064，精确 `[i*0.2 for i in range(128)]` 到达序列，H config 的逻辑 workload SHA 与原始 prompt token ID 哈希均通过，warmup 与测量模型相同。此检查**没有**验证 GPU、模型载入、输出或运行时间。H r02 的输入原件与当前 H1 candidate 对应 3 个输入文件及 4 个 warmup 文件 SHA 逐一相同。

## 新 H128 性能包的最小差分

从已接受 r02 **复制到独立新目录**，保留 `ltr_style_native.py`、`ltr_style_selected.py`、`recovery_service_components.py` 和共同原生/测量模块的原字节。该组件仍只称 *LTR-style selected-offload*：r02 README 明言未复现完整 LTR predictor 或 CPU-SWAP。只需改变以下包边界/runner 层，不能改写旧 r02 manifest：

1. **输入与安全上界。** 将新包 `pkg/inputs/{config.json,workload.json}` 换为 H r02 的原件；可连同 `INPUT_STATS.json` 收入 manifest。两个文件必须一起换，不能用 G config 配 H workload，因为 `load_inputs` 直接核 `workload_sha256`（r02 runner `:32-42`）。用 H r02 的 `pkg/safe_static.py`，其与 r02 G 版唯一差异是 `config['requests']==128`（G 版 `:51-56`）；H 最大 prompt 3064 仍满足现有 `<=3072` 上界。warmup 文件原字节保留。
2. **冻结 H128 测量身份。** 新 `pkg/run_ltr_style.py` 把 G 特有的 `(64,334,3011)` 与 `range(64)` 检查（`:96-98`）改为 H 的 128、460–3064、同一 0.2 s 序列；建议同时核 H config 的请求数、逻辑 workload SHA 和文件 SHA。`config.update(requests=64,...)`（`:101-113`）改为 128，其余模型、EOS、cap32、固定物理 KV、offload 16 GiB 和 180 s 保持与 H1 runner `:124-140`、`:162-170` 一致。输出 `config.json` 的 `variant='ltr_style_selected'`、`ltr_config={threshold,quantum}`、`measurement_mode='performance_sparse_preemptions'` 要真实反映运行参数。
3. **性能采集入口。** r02 runner `:75-79` 当前只允许 T30/Q10 并把 `diagnostic=True` 写死，尽管 `:207-212` 已有未到达的 `measure_episode` 分支。新 H 包明确只运行 `performance` cell（或显式加 `--measurement-mode`），把 `diagnostic` 传给 `install_ltr`（旧 `:195-201` 一直传 `True`），用 `measure_episode(..., record_preemptions=True, max_seconds=180)`、与 H1 runner `:236-241` 相同的稀疏全请求采集。性能 arm 无逐 job observer；r02 `:189-194` 的 `NOT_MEASURED` 声明应保持真实。保留 `raw.json`、host/显存、drain、timing、所有失败/未完成请求与 EOS/长度原因（旧 `:215-255`）。若将来要测多个 T/Q，只扩大预先批准的 CLI 取值并将固定值写入命令、输出路径和新包身份；T30/Q10 的旧诊断本身不是已校准的性能强基线。
4. **同资源前台单格执行。** r02 `pkg/run.sh:3-10` 和 `controller.py:13-36` 固定旧 GPU UUID、Python、缓存、锁及唯一 G 结果路径。新 H 包不能复用其文件身份或在同一目录追加 cell。新单格 `run.sh` 应采用 H1 candidate `pkg/run.sh:17-48` 的**同一**显式授权 GPU UUID、固定 Python/离线缓存、共享锁、进程树 cgroup 硬限、每格 `timeout` 和剩余预算检查；输出目录 `mkdir(exist_ok=False)`（runner `:80-84`）不覆盖。两包须同一实际 GPU、host 硬限、pinned vLLM SHA、模型缓存、warmup/measurement/drain 边界及 180 s capture；H1 runner `:143-188`、`:189-215` 展示现场资源和源码核验。外层单格墙钟预算涵盖初始化和 warmup，**不等同于** 180 s capture 限时；root 还须控制整组串行总预算。
5. **新身份。** 新根目录自行生成 `manifest.json`、package/provenance 收据及逐文件 SHA，包含 H `inputs`、改后的 `safe_static.py`、runner、shell 和复用的 LTR 模块；若放入 README、检查脚本也要列入其 manifest。新 `controller` 若使用，需改掉旧 `:33-36` 的单诊断结果假设；更小的方案是 root 在共享锁下逐格调用前台 shell。任何环境入口变化也在新 SHA 下记录。原 r02 的 30 项 manifest 和 `package.sha256` 仅证明旧 G 诊断包，不能认证 H128 性能结果。

`ltr_style_native.py:15-45` 的安装前提是 pinned scheduler、cap32、单组无共享 full attention、原生 OffloadingConnector 与 selected save；`:79-89` 检查可执行 intent，`:112-136` 是阶段式 victim/target 提交，`:145-149` 排序与 native 准备恢复，`:183-209` 由实际正调度调用扣 quantum 并卸载。代码无 64 请求常量，因此 **H128 的直接障碍在输入/runner/safe_static/包入口，不需为请求数改 policy**。这仅是源码审查，不能替代现场生命周期资格。

## CPU 最小验证与执行门

新包冻结后，root 只需做与差分相关的四项 CPU 核对：

1. 逐文件 manifest SHA、原 r02 三个 LTR 模块未变、H 原件输入与 warmup 哈希一致；无 G config/路径/UUID/输出目录残留。检查新包只接受一个显式 cell，非法 T/Q、空输出路径或缺授权环境变量 fail-fast。
2. 新 runner `load_inputs` 验 H128 的请求 ID、prompt token 哈希、`workload_sha256`、到达序列和模型/warmup；`safe_static` 只把开放请求总数上界改 128，不改变 cap32 或每请求资源计算。
3. Python 编译、shell 语法；CPU fixture 对 `performance` 参数传递做定向 mock，确认 `install_ltr(..., diagnostic=False, threshold=T, quantum=Q)`，走 `measure_episode(... record_preemptions=True)`，保留失败/未完成原样，且 `config.json`/输出路径/manifest 固定相同 T/Q。旧 `check_lifecycle.py` 只覆盖 G 的 CPU 选择和 native-loop fixture，不证明 H 的性能 arm 或真实 transfer。
4. 与 H1 candidate 的固定 model/revision/seed/EOS、4096+1 物理 KV 字节、16 GiB host、cap32、1024 batch tokens、同一 H 输入、warmup、180 s capture、pinned runtime SHA 和 host cgroup/per-cell wall 预算逐项静态比较；现场再用真实收据复核。

**先决顺序：** r02 原 G64/T30/Q10 必须先在 root 获明确 GPU/host/时间授权的机器上通过实际 vLLM 类型和物理 KV/host allocation、selected store/flush、native load/ready、首个新输出、跨输出 quantum 与 EOS 资格；失败保留 `INCOMPLETE`，不把 H 性能包列入有效强基线。新机器若与旧 r02 shell 固定 GPU UUID/路径不符，连 G 诊断也须另封仅环境入口变化的包身份，原 r02 不得被称作原样运行。诊断通过后，T/Q 校准规则在查看 H128 结果前固定，H128 LTR 新包与 native full、selected/eager、H1 off/on **串行**运行；比较用同一 128 请求的 `raw.json` 和统一请求级前沿。当前所有这些 GPU 性能结论均 `UNRUN`。
