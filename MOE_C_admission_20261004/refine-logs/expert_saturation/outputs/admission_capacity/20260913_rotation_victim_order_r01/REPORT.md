# 同恢复规则下的驱逐服务量排序消融

2026-09-13。**8/8 COMPLETE，256/256 次正式请求完成并回传；NATIVE_SERVING / MEASUREMENT_ONLY。** 相对原来的 least_progress，most_output 在四个配对中吞吐提高1.134%–1.560%、最大ITL减少19.638–24.030 ms，但平均完成时间增加1.004%–1.471%。排序改变了实际权衡，不能称为全面改善、统计确认或方法GO。

唯一问题：同实际KV与恢复规则下，驱逐对象排序如何改变最长暂停、完整吞吐与完成代价？A为合格running集合内计算进度比例最小者；B为同集合内原生request.num_output_tokens最多者。计数不含重算，也不是客户端消费。最长缺席恢复、30/20/30、近完成保护0.90、最多8次缺席、完整历史资金与恢复保护保持；两臂独立演进实际状态。

## 主比较

相对差均为B/A−1；四行是预定同cohort/block配对。旧20项不替代本次A/B。

| Cohort/block | A最大ITL s | B最大ITL s | 吞吐 Δ | 平均完成 Δ | 最大ITL Δ ms |
|---|---:|---:|---:|---:|---:|
| cohort0/0 | 1.010667 | 0.988988 | +1.471% | +1.147% | -21.679 |
| cohort0/1 | 1.005855 | 0.981825 | +1.134% | +1.471% | -24.030 |
| cohort1/0 | 1.008693 | 0.989056 | +1.530% | +1.123% | -19.638 |
| cohort1/1 | 1.009128 | 0.988171 | +1.560% | +1.004% | -20.957 |

全部原值、逐请求变化、四个策略配对与四个同角色重复见[analysis.json](analysis/analysis.json)及[八格表](analysis/report.md)。完整请求数相同，B的整批结束更早，但更多请求完成变晚；吞吐和平均完成不是同一个目标。

| Cohort/block | 完成变慢数 | max-ITL增大数 | 最多完成延后 s | 最大新增max-ITL s | 完整输出相同 |
|---|---:|---:|---:|---:|---:|
| cohort0/0 | 28/32 | 6/32 | 1.437390 | 0.892554 | 32/32 |
| cohort0/1 | 28/32 | 28/32 | 1.491862 | 0.908871 | 32/32 |
| cohort1/0 | 28/32 | 28/32 | 1.430288 | 0.913421 | 31/32 |
| cohort1/1 | 27/32 | 6/32 | 1.409649 | 0.889453 | 31/32 |

“变慢/增大”按原始差值严格>0，包含微小漂移，不是显著受损数。B把部分请求的暂停增加约0.63–0.91 s，最早请求完成延后约1.41–1.49 s；不能只报告全体最大值下降。各格32/32在TTFT≤5 s、mean-TPOT≤0.2 s参考条件下通过，goodput等于吞吐；该条件不约束max-ITL。没有语义质量评估；同长度、甚至输出相同，都不等于质量保证。

## 实际动作与恢复

四配对的首次驱逐选择分离在step836。此前已记录的请求进度、输出数量、running集合与块数状态对齐；没有验证KV tensor或route完全相同。A驱逐已输出742、computed3813的请求，B驱逐已输出834、computed3905的请求；恢复目标已输出709，free179块、需237块，A/B释放239/245块，均可资助恢复。后续各自演进，没有重用另一臂未来轨迹。

A每格8次强制+2次自然抢占，B为9+2；held request-step均0。最长暂停的请求已变化，不能把两臂极值差解释为同一请求恢复缩短：

| Role | 最长暂停来源 | 抢占step | 首次恢复调度 | 首个新输出 | 等待 s | 恢复跨度 s |
|---|---|---:|---:|---:|---:|---:|
| least_progress | 自然抢占 | 931 | 976 | 979 | 0.892049–0.896759 | 0.113502–0.116498 |
| most_output | 强制抢占 | 996 | 1036 | 1040 | 0.846323–0.852381 | 0.135165–0.137357 |

上表之外另含约0.30–0.40 ms的调用边界。B的最长等待为40步、恢复5调用；A为45步、恢复4调用。全局冷却20仍不是每请求等待上界，排在前面的缺席者和恢复保护均占用时间。held=0不能证明保护组件必要；路径细节见analysis.json的recovery_accounting与[独立路径提取](analysis/PATHS.md)。

## 完整成本

`wall = scheduler-inclusive + engine-non-schedule + outside-engine`；decision属于scheduler子集，不能重复相加。以下为B−A，均为host调用区间，含同步、采样和其它请求的有效decode，不称纯GPU重算成本。

| Cohort/block | 含重算调用非调度 Δ s | 纯decode非调度 Δ s | 全引擎非调度 Δ s | scheduler Δ s | 引擎外 Δ s | 完整wall Δ s |
|---|---:|---:|---:|---:|---:|---:|
| cohort0/0 | +0.215552 | -0.582945 | -0.344269 | -0.003168 | +0.015855 | -0.331582 |
| cohort0/1 | +0.222719 | -0.539951 | -0.298129 | +0.021120 | +0.022420 | -0.254589 |
| cohort1/0 | +0.215376 | -0.589526 | -0.365950 | +0.008869 | +0.014028 | -0.343054 |
| cohort1/1 | +0.207127 | -0.575210 | -0.363732 | -0.020620 | +0.033235 | -0.351117 |

每格B重算位置38,561→43,471（+4,910），含重算调用40→49，纯decode调用1119→1015，总调用1257→1162。width2纯decode调用87→8、width4为46→7，width3为3→21；更高重算量与更少收尾调用同时出现。同一最后两请求的width2尾段1158–1244→1143–1150，对应host调用区间减少约0.432–0.440 s；前四请求的完成名次则从1/2/3/4变为23/9/25/13，解释整批提前结束与平均完成变慢并存。详见[收尾路径](analysis/TAIL_PATHS.md)。实际含重算桶多251个新decode位置，纯decode桶少251，新增服务总量保持。不能把这条实际路径分解当作固定trace的反事实上界。

## 同角色重复与范围

| Cohort / role：block0→1 | 吞吐 Δ | 平均完成 Δ | 最大ITL Δ ms | 输出相同 |
|---|---:|---:|---:|---:|
| cohort0 / least_progress | +0.688% | -0.669% | -4.811 | 32/32 |
| cohort0 / most_output | +0.353% | -0.351% | -7.163 | 32/32 |
| cohort1 / least_progress | -0.375% | +0.449% | +0.435 | 32/32 |
| cohort1 / most_output | -0.345% | +0.330% | -0.884 | 32/32 |

四个同角色差值只描述本次已观察漂移，不是噪声上界，不能据此直接判显著或减去噪声。两组复用文本、每组两个顺序block，不是八个独立workload，也不把256次请求和262,144个输出token当独立重复。全部输入来自已测20项cohort0/1：这是探索性组件消融，不是新holdout、VTC/FastServe复现或完整公平基线。

## 冻结、执行与复算

- OLMoE BF16 / vLLM0.26 / RTX5090；实际KV16,089,350,144 bytes、7,671 usable blocks；cap32、3072输入/1024输出、50 ms到达、token budget1024。输入和预热逐字节复用；完整源、配置、运行环境随归档保留。
- 冻结协议[DECISIONS.md](preparation/source/DECISIONS.md)与[campaign.json](preparation/source/campaign.json)；包SHA256 `0049d6e125dadb37655c3580339a667979f43da5290c0c0afa6edfed1df8ff00`。准备文件仍保留运行前PREPARED_UNRUN，不代表当前执行状态。
- [execution.json](execution/execution.json)为COMPLETE，驱动退出0；按冻结顺序串行执行、逐格回传核验，无本轮失败或重跑。现有远端目录`/root/autodl-tmp/moe-rotation-victim-order-20260913-r01`。初始化与正式测量边界检查GPU进程，不代表持续独占或恒频证明。
- 用户另行授权清理5090后，四轮40份已核对本地raw/归档的远端重复raw副本已删除，释放11.1515 GiB；本地完整证据和远端归档保留，数据盘97%→75%。详见[清理记录](remote_cleanup/README.md)。
- 此前自动审批拒绝留存在[UPLOAD_REVIEW_RECORD.json](UPLOAD_REVIEW_RECORD.json)；普通用户消息对本包上传/八项执行的具体授权见[授权记录](UPLOAD_EXECUTION_AUTHORIZATION.json)，现已执行完毕。

```sh
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/analyze_rotation_victim_order.py --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_victim_order_r01/execution --output-dir /private/tmp/rotation-victim-order-reanalysis-NEW
```

复算输出目录须新建。原始结果不覆盖，后续修正另写addendum。完整性审阅**未完成**：fresh agent启动触及线程上限，复用审阅者在本次有界调用内也未返回产物，已中止并记ERROR/UNVERIFIED；不沿用旧审阅PASS。执行者复算及另一agent的路径诊断已完成，但不替代完整审计；见[审阅执行记录](EXPERIMENT_AUDIT.md)。

## 当前裁决

| 固定报告项 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；排序改变权衡，主问题OPEN |
| Evidence type | NATIVE_SERVING，原生进程内完整请求与host逐token记录 |
| What was measured | 8格256次完整请求，4策略配对、4同角色重复，实际动作/恢复/完整成本 |
| What was not measured | 完整独立审阅未完成；新holdout/长度/到达/KV/模型域，统计显著性/非劣、质量、消费QoE、完整公平基线 |
| Strongest baseline | 本轮同底座least_progress轮转；旧native/headroom/safe29仅为历史背景，未在本轮重跑 |
| Oracle/headroom status | 两种合法选择真实产生不同后续路径；无全动作Oracle或暂停硬界 |
| Claim ceiling | 这两组已见文本中的描述性组件作用，非方法GO或独立新颖性 |
| Failure category | 平均完成与更多请求暂停承担代价；多重算和收尾变化共同影响完整时间 |
| Resurrection condition | 未判死问题；扩大主张需独立运行域与明确主目标约束，不能靠同数据改阈值 |
| One next smallest experiment | 首个合法交换用most_output，之后恢复least_progress；检验一次选择是否足以压缩两尾请求收尾、同时减轻后续早完成请求的代价。其余条件保持、完整in-loop执行；当前UNRUN |

**直接回答：这两组文本中，原来的最少进度驱逐不是保住当前权衡的唯一排序；换成已输出最多者能提高整批吞吐并略降最大暂停，同时让多数请求更晚完成。排序贡献存在，全面净收益尚未成立。**
