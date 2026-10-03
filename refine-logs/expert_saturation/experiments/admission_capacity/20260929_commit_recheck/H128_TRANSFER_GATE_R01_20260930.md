# H128 同机迁移执行门（CPU 模板，GPU 未运行）

本模板只预留一次新的 H128 完整服务迁移组；第三格取 G64 四点校准后唯一选出的 LTR-style 点。H128 输入及旧机器上的结果已经被看过，因此本次只能称为**未用于本轮 LTR 参数选择的输入迁移**，不能称盲测，也不能把旧 H r02 的速度与当前 RTX 5090 混排。

## 当前冻结部分

- [H128 包](candidate_h128_perf_r01/README.md) `manifest.json` SHA-256 为 `24e0e0b927971569fb4cceb2e66cce44d31e0dfd3d99a94680bf1917a5739a0e`；本地 `python3 -B candidate_h128_perf_r01/check_cpu.py` 返回 `PASS_CPU_ONLY`，29/29 文件核验通过。H 输入为固定的 128 篇完整文章，128 个请求按 `i × 0.2 s` 到达，EOS 允许，cap 1024，单格 capture 最多 180 s；三格各自使用同一输入和独立 engine。
- [非执行模板](H128_TRANSFER_EXECUTION_TEMPLATE_R01_20260930.json)预留新 session 和三个互不重叠的 output 目录，顺序固定为 `native_full_native`、`eager`、`selected_ltr`；第三格的实际 `ltr_t{30|200}_q{1|10}` arm 当前为 `null`。新组自限 4200 s，总共三个各 900 s 的 cell；同一 GPU UUID、同一锁 inode、90 GiB cgroup、4096 usable GPU KV 块和 16 GiB host KV。外层 4200 s 还覆盖锁内模型哈希、离线解析、边界检查与原件归档。此模板包含 `null`，**不能直接作为控制器计划运行**。
- H 包 `run.sh` 对 LTR arm 另核 `HPERF_SELECTED_LTR_ARM` 与实跑 arm 完全一致，并用每个 arm 独立 `launch-once-*` 阻止原地重试。`native_full_native` 为完整原生保存系统参照；`eager` 与 LTR 共享 selected-save 后端，才是动作比较参照。三格必须各有实际 128/128 完成记录，完整失败、未完成、实际输出长度、EOS、请求级 gap、TTFT、完成时间及 peer 后果全部保留。

## 在填写第三格前必须取得

1. G64 首组的完整会话及 [只读审计 V2](A_G64_PERF_THREE_ARM_AUDIT_R01_20260930_V2.json)：审计 SHA-256 `af8685aa75d8123ea704017aea7280b80274d7c61b2ba78a97e185602bf44a96`，状态 `PILOT_THREE_ARM_COMPLETE_PARTIAL_TQ_GRID`。首组只覆盖 T30/Q10，**不能单独选点**。
2. G64 余下 T30/Q1、T200/Q1、T200/Q10 的新会话、逐文件回读哈希和独立请求/资源审计。r02 的首个 T30/Q1 已触发固定 180 s capture 上限，不能把它当作完整可选点；后续独立身份若运行，必须保留 r02 失败原件与成本。四个事前指定 LTR 点都须有可审计的完成或失败处置；只有 64/64 完整请求且来源/资源/归档门禁通过的点可进入性能选择。完整点应沿用同 G64 到达输入、同 RTX 5090/模型/vLLM/4096 GPU KV/16 GiB host KV/锁及 cgroup 条件。保留所有组的真实执行顺序及输出/EOS 差异。
3. 一份冻结的 G64 选择收据，写明首组及每个后续组的计划、会话、审计 SHA-256，四点各自的完成或失败状态，完整点相对同窗口 eager 的实际输出率与平均完成比值、全请求最大生成 gap、合格/不合格原因，以及唯一选择。合格条件是输出率比 ≥0.97 且平均完成比 ≤1.05；在合格点中先取最小最大 gap，平手依次取更高输出率、更低平均完成、低 T、低 Q。若某点无法在固定 capture 合同下完整完成，应给出该点 `INCOMPLETE` 和选择规则的适用范围，不能悄悄补时或弃点。没有合格点就记 `NO_QUALIFYING_POINT`，**不启动 H128 LTR 迁移组**，也不扩 G 网格。冻结该收据哈希后才填模板第三格和 `HPERF_SELECTED_LTR_ARM`；H 结果不能反向改变 G 点。

## 执行前仍需补齐的工程门

现有 `serial_group_g64_perf_existing_model.py` 硬编码 G64 包名及 manifest、`GPERF_` 环境变量和 G64 arm 白名单，不能用于 H 包。需要一个**新身份、单一前台的 H128 串行控制器**：在共同锁上非阻塞取得 fd 9，核锁 inode/GPU 空闲、C 来源与 A 私有模型的完整哈希和离线解析，固定 H 包 manifest 和 G 选择收据哈希，为三格传 `HPERF_` 变量，每格限时且异常停止、查 GPU worker 退出、逐文件复制归档并回读。先用 CPU fixture 和远端 `--validate-only` 核新控制器与具体计划；不得复用 G controller 或 G session/output，也不得复用旧 H 包的 `launch-once` 标记。已有 G 控制器的模型核验及隔离 runtime cache 可作实现参照，但新控制器必须冻结自己的 SHA-256。

包上传后核 29/29 payload 与 manifest 相同，确认新 session/output 不存在、GPU 没有其他 compute worker、共同锁设备/inode、A 私有 Python/model cache 和 cgroup `memory.max=96636764160` B。若锁被占、身份/资源漂移、模型校验失败或一次 cell 失败，保留错误收据，停止余格；新尝试须另订身份和预算。本模板不保留 GPU 锁，不能替代执行时的复核。

运行完成后回读整组 session 与三个 archive，每格文件哈希和请求、资源、后端来源逐项审计；使用固定 `evaluate_goodput.py` 前沿报告自然生成完整服务，不能把不同实际输出序列称等工作量加速。单次 H 顺序迁移若通过或失败，只界定此同机输入域；H 已见过、无受控重复，不能给统计显著性或独立方法贡献结论。容器缺 `memory.peak` 时保留主机峰值 `UNKNOWN`。
