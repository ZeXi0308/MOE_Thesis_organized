# G64 同机器完整服务校准候选包

状态：`CPU_VERIFIED / GPU_UNRUN`。这是独立于已接受 G64/T30/Q10 **生命周期诊断**的新性能包身份；不会把诊断计入性能对照。包内 28 个 executable/input 文件由 `manifest.json` 固定，旧包不改。

## 固定实验

单卡、同一 64 个自然文章 prompt、逐源 0.2 s 到达、OLMoE 精确 revision、允许 EOS 且单请求至多 1024 新 token；每格 warmup、cache reset、180 s 请求 capture、drain、shutdown。固定 cap32、1024 scheduled tokens、4096 usable GPU KV blocks 加 null block、16 GiB 原生 host KV；每格从独立 engine 重新开始。现场必须核真实分配、模型、源码、锁、GPU UUID 和 cgroup 内存上限。每格完整墙钟（含初始化、warmup、drain）另由执行计划限制。H128 旧结果来自另一机器，不能用作本 5090 的定量对照。

`pkg/inputs/config.json` 内保留原始 G 输入的旧 `status` 和嵌套 `freeze_contract`，仅作来源记录；本包的性能 arm、执行模式和参数由新 `run.sh`/runner/manifest 固定，运行输出顶层 `measurement_mode` 应为 `performance_sparse_preemptions`。
继承的 H1 runner 在 native-full 输出中仍写通用字段 `selective_save='on'`；实际策略以 `store_scope='native_full'`、`native_calc_overridden=False`、没有安装 adapter 的 `selective-store.json` 及原生运行收据判定，不能单靠该通用字段归类。

`pkg/run.sh ARM ABSOLUTE_OUTPUT_DIR` 每次只执行下列一格，输出目录必须全新且在包外。固定顺序：

| arm | 策略 |
|---|---|
| `native_full_native` | vLLM 原生 full-save，无额外轮转 |
| `eager` | selected-save + eager/most-output；H1 候选 adapter 的 `commit_recheck=False` |
| `ltr_t30_q1` | r02 LTR-style selected-save，T30/Q1 |
| `ltr_t30_q10` | r02 LTR-style selected-save，T30/Q10 |
| `ltr_t200_q1` | r02 LTR-style selected-save，T200/Q1 |
| `ltr_t200_q10` | r02 LTR-style selected-save，T200/Q10 |

LTR 三个策略模块逐字来自 G64 r02；selected/eager 策略模块逐字来自 H1 候选，含其共有安全修补。两个 runner 只改性能入口/请求数/诊断开关；公共测量与原生接口、固定 G 输入、G 资源上界逐字复用。两种 selected adapter 互斥安装。`LTR-style` 只表示已移植的等待提权、优先级和正计算量子，不等同于完整官方 LTR。

同窗口 `eager` 是四点校准参照；仅全部请求完整完成、资源/身份一致、实际输出率至少为 eager 的 97%、已完成 cohort 平均 arrival-to-completion 不超过 eager 的 105% 时，按全体请求的最大生成返回间隔最小选点；平手依次用输出率、平均完成、低 T、低 Q。没有合格点就记 `NO_QUALIFYING_POINT`，不加参数。`native_full_native` 始终另列为完整系统参照。报告固定 goodput 前沿 `D_F={10,20,30,40}s × D_G={1,2,4,8,12}s`，所有失败/未完成保留；生成长度或 EOS 不同须披露。现有 `evaluate_goodput.py` 可分析逐格实际 `raw.json`，`ltr_baseline_compare.py` 的资格选择还需用真实 runtime 输出构建其严格 `resource_receipt`，不能伪造。

启动前需要外层串行会话持共同锁直到格间 GPU 空闲复核和归档读回；`run.sh` 也对同一 inode 做非阻塞锁并拒绝重复格。它要求 `GPERF_AUTHORIZED_GPU_UUID`、`GPERF_PYTHON`、`GPERF_HF_CACHE_DIR`、`GPERF_LOCK_PATH`、`GPERF_MAX_WALL_SECONDS`、`GPERF_APPROVED_HOST_BYTES`、`GPERF_CGROUP_MEMORY_MAX_FILE`、`GPERF_EXPECTED_MANIFEST_SHA256`。最后一个值先由执行者核 `manifest.json` 并固定；shell 不能自证授权。任何格失败即留原始收据、停止后续，不改输入/压力、不在已消耗身份原地重跑。G 校准完成并冻结选择后，H128 才能以另一新包身份作迁移输入；H 已被研究者看过，不能称盲测。

CPU 复核：`python3 -B check_cpu.py`。它检查 28/28 SHA、策略/输入来源字节、64 请求逻辑身份、源码编译、shell 语法与非法 CLI 拒绝。它不执行 vLLM/GPU，也不证明性能 arm 已通过真实生命周期。
