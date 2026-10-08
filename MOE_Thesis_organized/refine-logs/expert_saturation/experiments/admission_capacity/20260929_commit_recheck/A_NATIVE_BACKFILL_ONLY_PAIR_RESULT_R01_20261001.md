# Native full-save + ordinary backfill：固定两臂已见输入结果（2026-10-01）

**有动作，固定性能判据失败。** 同一冻结 128 请求上，先运行完全原生的 full-save 服务，再运行 full-save 加普通合格等待者 backfill；两格各 **128/128** 完成。[权威分析](A_NATIVE_BACKFILL_ONLY_PAIR_RESULT_R01_20261001.json)记录 ordinary 的 **23/23** 选择→原生异步接纳→后续新输出→完成链、**0 次主动 forced rotation**。相对同组原生参照，ordinary 实际输出率仅 **94.108%**（低于冻结的 97% 下限），mean flow 为 **105.225%**（高于 105% 上限）；全体最大请求 gap 虽由 **13.101 降至 11.844 s**，九项预设判据中两项性能预算失败，因此 `all_criteria_met=false`。这不是 `NO_ACTION`，也不能用 gap 一项改善救援总判据。

| 完整请求指标 | Native full-save，第 1 格 | Native full-save + ordinary，第 2 格 |
|---|---:|---:|
| 完成 / 到达 | 128/128 | 128/128 |
| 输出 token | 126,660 | 124,956 |
| 实际输出 token/s | 1,511.666 | 1,422.598 |
| Mean flow (s) | 35.924 | 37.801 |
| TTFT 中位 / p90 (s) | 21.578 / 37.731 | 23.790 / 37.144 |
| 请求最大 gap 中位 / p95 (s) | 0.042 / 8.725 | 0.051 / 3.740 |
| 全体最大请求 gap (s) | **13.101** | **11.844** |
| 实际原生抢占 | 51 | 64 |
| 主动 forced rotation | 不适用：无适配器 | **0** |
| Stop：length / stop | 123 / 5 | 121 / 7 |

两格均用原生 full-save connector 计算方法；参照臂完全不安装 scheduler 适配器，ordinary 臂安装选择和 Q1 首输出保护钩子，但跳过全部主动 forced-rotation proposal。运行记录核对 `store_scope='native_full'`、`native_calc_overridden=false`、`allow_forced_rotations=false`、`applied_rotations=0`；第 2 格仍可发生**原生内存压力抢占**，实际有 64 次。原生参照的 forced 字段为 `NOT_APPLICABLE`，不是观测到 0 次策略旋转。普通 backfill 是已知的强简单对照，不是独立新算法；两格同后端但 scheduler 钩子不同。

## 动作与全体请求后果

Ordinary 的 23 次选择均在 `REGULAR_COMMIT` 原目标首输出释放事件所在调度步之外，均有 `ASYNC_LOAD_ADMITTED` 回执、之后实际新输出与最终完成；动作步无新增实际抢占。异步回执本身不是传输完成。门禁计数中，`KEEP_REGISTERED_TRANSFER_JOBS` 有 2,332 次，空/待处理队列 1,584 次、队首未抢占 881 次、无可容纳合格后位者 560 次；这些是调度步门禁命中，不能相加为独立请求数，也不能断定单一门禁造成性能差。此前 selected-saving ordinary 同输入有 44 次动作，本次 full-save 有 23 次，符合预先的“可能更少”预测；后端与自然轨迹均改变，不能把差额归因于某一门禁。

| Ordinary 相对原生参照，128 请求等权 | 改善 | 恶化 |
|---|---:|---:|
| Flow | 9 | 119 |
| TTFT | 18 | 110 |
| 每请求最大 gap | 69 | 59 |

固定 20 点 goodput 前沿有 5 点较高、15 点较低。两格有 **63/128** 条请求输出序列不同、**2/128** 条 stop 原因不同，总输出少 1,704 token。Stop 差异为 `memory-train-article-0027895`（1,024/`length`→196/`stop`）和 `0031269`（1,024/`length`→148/`stop`）。最大个体 flow 恶化是 `0032681` 的 50.658→55.457 s（+4.800 s，输出数同为 1,024）。全体最大 gap 的请求发生转移：原生臂的 `0032049` 由 13.101 降至 ordinary 的 1.203 s；ordinary 新的最长 gap 是 `0035191` 的 11.844 s，而它在原生臂仅为 0.089 s，个体恶化 **+11.755 s**。因此 p95 和全体最大 gap 改善不能表述为全部请求停顿改善。

[请求级定位](A_NATIVE_BACKFILL_ONLY_TARGET_PEER_R01_20261001.json)显示，23 次动作涉及 11 个不同恢复目标，其中 10 个在首个新输出后再次被实际原生抢占；被越过的队首共 14 个不同请求，4 个多次被越过。最长 gap 请求 `0035191` 的无输出区间为 62.068–73.912 s：62.070 s 实际抢占，期间一次是装不下的队首、三次是被越过的最老等待者。日志没有记录该请求在这段区间内的原生准入时刻，不能把整段停顿归为排队或某次 backfill 的因果损失。

## 已打印传输与缓存

| 正式期完整打印区间 | Native full-save | Native full-save + ordinary |
|---|---:|---:|
| Store / load (GB，十进制) | 38.174 / 15.066 | 37.482 / 17.329 |
| Store / load 打印计数 | 7,107 / 51 | 6,770 / 63 |
| Store / load CUDA 事件时间 (s) | 0.967 / 0.294 | 0.917 / 0.330 |

[传输区间结果](A_NATIVE_BACKFILL_ONLY_PRINTED_TRANSFERS_R01_20261001.json)每格正式期打印 8 条，排除首条可能混入预热的记录，仅汇总后 7 个完整重置区间；首尾未打印部分未知，不能当全程传输总量。Ordinary 的已打印 load 比原生参照多约 2.263 GB；CUDA 事件计时不含开始前的 stream 等待，也不是请求可见延迟。两格各用同一 525 文件完成缓存的独立副本；[会话缓存记录](moe-a-native-backfill-only-primary-session-r01-20261001/)按最终文件 mtime 把每格 32 个、约 15.626 MB 新增/修改文件归在初始化，预热和正式期均为 0 个。这不证明零 JIT、相同关键路径或没有后来重写而漏归类的文件。

这是一组同输入、相邻顺序、自然输出可不同的单次服务实验，不是同状态动作反事实或统计重复。它证明 full-save 后端上**实际可以**执行无主动 forced rotation 的普通 backfill，但本次其吞吐和 mean-flow 预算均失败。保留全部请求、失败预算和个体损失；不做参数救援、跨机器因果推断或新颖性主张。[原始两格会话](moe-a-native-backfill-only-primary-session-r01-20261001/)与权威 JSON 一起留存。
