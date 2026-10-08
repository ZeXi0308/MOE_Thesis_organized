# H128 同机器完整服务迁移候选包：原生 in-flight KV guard

状态：`CPU_VERIFIED / GPU_UNRUN`。这是独立于 `candidate_h128_perf_r01` 的新包身份；尚无 H128 guard GPU 结果。H128 历史轨迹已被研究者看过，因此只能作**未参与 LTR 参数选择的迁移输入**，不能称盲测。只有 G64 的同机器 native full、selected/eager 与完整 T/Q 网格完成、依冻结规则选出一点后，才可执行本包。

`pkg/run.sh ARM ABSOLUTE_OUTPUT_DIR` 支持 `native_full_native`、`eager` 和四个预定 LTR 点 `ltr_t30_q1`、`ltr_t30_q10`、`ltr_t200_q1`、`ltr_t200_q10`。实际 H 顺序为 native full、eager、**仅 G64 选出的一个 LTR 点**。LTR arm 另要求 `HPERF_SELECTED_LTR_ARM` 与本次 arm 相同；执行计划还须保存 G64 校准报告的哈希与选择收据。若 G64 无合法 LTR 点，H 不启动 LTR 性能格。每格有不同 `launch-once-${arm}`，失败不得原地重试。各臂独立 engine 和状态推进；不能把旧 H 轨迹离线换算成新臂。

测量输入逐字采用 H1 的 128 个自然文章 prompt、逐源 0.2 s 到达；H `INPUT_STATS.json` 也在 manifest 中。模型、seed、EOS/cap1024、cap32、1024 scheduled tokens、4096 usable GPU KV blocks、16 GiB 原生 host KV、warmup、180 s capture 和 drain 与 G 包对应字段相同。实际模型、runtime、物理分配、GPU UUID、cgroup 内存、共同锁、整组/单格墙钟须现场复核。G/H 性能只有同一台 5090、同一精确环境且同一会话控制下才可定量排比；H1 的方法 on/off 若后续执行也必须在此同窗口和 selected/eager 底座下比较。

原始 `pkg/inputs/config.json` 中的旧状态/嵌套合同只是冻结输入来源；本包的性能 arm 和模式由新入口、runner 与 manifest 固定，运行输出顶层 `measurement_mode` 应为 `performance_sparse_preemptions`。
继承的 H1 runner 在 native-full 输出中仍写通用字段 `selective_save='on'`；实际策略以 `store_scope='native_full'`、`native_calc_overridden=False` 和没有安装 adapter 的运行收据判定。

本包从 `candidate_h128_perf_r01` 复制 29 个 manifest payload，仅将 `pkg/ltr_style_native.py` 逐字替换为 `candidate_g64_perf_guard_t200_r01` 的已冻结 guard adapter（SHA-256 `ccbddc6d9860a1efc2c78dcf6614883734b2388de19a3db8034632bd9b3e9fb3`）。它在直接恢复等待请求前核对原生空闲 KV blocks、in-flight 预留 blocks、目标完整历史所需 blocks 和 sequence slot；T/Q 数值仍由 H 入口选择。其余 28 个 payload 与原 H128 性能包逐字相同，包括 H 输入、runner、native full、eager 和 LTR selector。该来源身份不等于 H128 GPU 资格或收益。LTR-style 不是完整官方 LTR。

启动环境变量使用 `HPERF_` 前缀：`AUTHORIZED_GPU_UUID`、`PYTHON`、`HF_CACHE_DIR`、`LOCK_PATH`、`MAX_WALL_SECONDS`、`APPROVED_HOST_BYTES`、`CGROUP_MEMORY_MAX_FILE`、`EXPECTED_MANIFEST_SHA256`；LTR 另需 `SELECTED_LTR_ARM`。外层串行控制器应从 G 校准收据填入选中点，在同一共享锁下逐格运行、复核 GPU 无残留 worker，并归档原始输出和读回哈希。开发评价使用 `evaluate_goodput.py` 的固定 20 点前沿、真实输出率、全部到达请求的完成/失败/未完成和 EOS/输出长度差异。原生 full 是系统参照；同 selected-save 的 eager 是 LTR 动作参照。

CPU 检查：`python3 -B check_cpu.py`。它核 29/29 包内 SHA、H 输入与上界来源字节、guard/其余策略来源字节、CLI 白名单和 G 点选择护栏，不初始化 GPU；它不证明 H128 的原生生命周期或服务收益。本包 manifest SHA-256 为 `81c13cfdd5a260fcb9b8a83bc310e618f1bd1962fdb1dc59eaaecce909536a17`。
