# A: G64/T30/Q10 原生恢复生命周期资格（2026-09-30）

## 结论与边界

固定 OLMoE revision、vLLM 0.26.0、单 RTX 5090、4096 个可用 GPU KV 块和 16 GiB host KV 下，LTR-style recovery-only 组件实际执行了 selected native 保存、加载、恢复后新输出和后续正向调度。该结果只通过原生生命周期资格，不是 LTR 完整复现，也没有同机性能对照或方法收益结论。

控制器在共同 GPU 锁 `2304:25841682495` 下完成模型源和 A 私有系统盘模型的完整哈希检查，离线解析固定 revision 后启动唯一单格。会话 `receipt.json` 为 `CELLS_COMPLETE`、子进程退出码 0、归档 `VERIFIED`、退出时 GPU compute 列表为空。会话约 294.5 s，正式请求约 44.8 s；初始化、预热、正式请求和排空不能混算。独立本地读回核对归档清单中 27/27 个文件的 SHA-256，未修改原始文件。

修正后的只读审计为 `OBSERVED_CHAIN` 且 `issues=[]`：64/64 请求完成，无失败或未完成；53 次 READY commit 对应 53 个实际 custom rotation。全局有 53 个 selected native store job 和 53 个 native load job 的 worker 接受及完成回执，这不表示每次轮转都给该目标加载一次。42 个恢复 episode 可以逐一接到同一目标的 store、load、首次返回的新输出、之后的正向调度和 quantum release；它们覆盖 17 个不同目标请求，最多同一请求 5 次，且每条均有一个初始目标 load job。剩余 11 个 READY episode 没有目标 load job，目标通过 847–3146 token 的 recompute 后仍输出并获得后续服务；它们不是加载链失败。104 次 `WAIT_LOAD` 意图中 52 次本步实际未调度且 quantum 不减，另 52 次同一步 native 调度为正且 quantum 减一。59 个请求达到 1024 token 上限，5 个自然 EOS；恢复阶段目标的自然 EOS 没有观察到。

首版离线审计错误地把运行程序记录的 core-source 哈希清单与预检使用的 offloading-source 清单直接比较，且把可空 `lookup.matched` 与零比较，并把 `WAIT_LOAD` 意图误当作本步必然没有实际 native 调度。修正版按运行程序的真实八个 core-source 字段检查，保留两个内置固定哈希；主机快照仅缺 cgroup `memory.peak` 时接受 KV/容量字段并明示峰值未知；WAIT_LOAD 按实际调度量核对 quantum。又加入原生 engine call 与调度步的逐项对应、victim store/target load/首输出的完成顺序、marker 输出数量相等和后续每个正调度步实际返回目标输出的硬门。12 个针对性合成测试通过，首版及中间版 `INCOMPLETE` 审计作为历史诊断保留。控制器和原始归档未改动。

## 可追溯原件

- 固定控制器 SHA-256 `609d451c06b3dd34c18773091409d989397bca3640fa8f3f7eaf21307e575df8`；计划 SHA-256 `02afdce3ce43c8f0e4111c1894098b693c9725de837c912d4ccca8f6f73ac66e`。
- 远端会话读回：`moe-a-ltr-g64-session-system-copy-r02-20260930/`；归档清单 `cell-00-ltr_r02_g64/output_sha256.json` SHA-256 `640dd39793eccc0d39bf00c7219ff93f77c9ee734021df541f87a3fa3660f2e6`；本地读回记录 `A_G64_TRANSFER_READBACK_SYSTEM_COPY_R02_20260930.json`。
- 修正审计脚本 `analyze_ltr_g64_lifecycle.py` SHA-256 `06e7a85784d52ff7efa9d8b30b29ce2676495e99e3811c5e5860fc5d8b17a268`；最终审计输出 `A_G64_LTR_LIFECYCLE_AUDIT_SYSTEM_COPY_R02_20260930_V4.json` SHA-256 `4f528cc121064898e1cc071d10a6e5eb1ea568c799cc6b7911bd15903603d646`。

## 下一判据

本单格使用详细记录，不能将其吞吐或间隔直接与旧 G/H 或无诊断运行排名。下一步先在这台机器以相同自然 G64 输入、模型、资源、EOS 和预热进行串行完整服务对照：至少原生 full、selected/eager 与一个 LTR-style 点；开发域 T/Q 候选及成本预算沿已冻结 `LTR_G_CALIBRATION_DECISION.md`。保留每个完整请求的实际 token 数、EOS/上限、TTFT、最大生成间隔、完成时间、peer 恶化和最终 drain。若合法简单基线覆盖 LTR，记录边界并重审独立机制价值；在 G 开发点冻结前不把 H128 称为确认。

独立 KV tensor-value 相等未被观测，transfer 字节为聚合统计；容器没有 `memory.peak`，不能认证整个过程的 host 峰值。归档审计本身不接收外部读回清单，27/27 哈希核验由上述独立读回记录证明。
