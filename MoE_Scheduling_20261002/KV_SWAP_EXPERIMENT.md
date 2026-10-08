# 专家分页时原生 KV 换入的完整成本探针

待 headroom 正反序组六臂结束后执行，不与在途组并行。当前唯一问题：以额外1GiB CPU KV缓存保存/恢复，能否在Q=2048的受限专家缓存域改善完整请求，并超过无需CPU KV的简单cap10？开启已有原生swap本身不是独立贡献。本次是成本边界与强基线对照，不构建新controller。

采用三臂：recompute16、recompute10（headroom）、swap16；顺序R/H/S/S/H/R。cap10在先前独立headroom组中冻结，不用swap结果反选参照。两种不换入的策略仍全部保留，cap8的两次明显wall/flow损失亦保留于原组。若原headroom末态不完整，本组不启动。所有臂共同HND布局、禁用simple KV offload、GPU KV1GiB、expert24、engine4096、测量Q2048、共同旧16输入预热Q512/固定32输出；同source32..47、自然EOS/max512，不称独立未见测试。

S使用vLLM0.26既有native OffloadingConnector，CPU KV1GiB，offload_prompt_only=False；R/H无connector。保持原生FCFS/抢占/恢复选择，不引入旧强制rotation或victim打分。观测实际CPU/GPU KV存储、双向KV传输、所有请求持续保存的代价、warmup drain/reset、测量末drain、调度位置重复及全部engine调用。connector-only空步不丢弃；served和drained两个截止点分别报告，完整成本主参照用drained。传输跨度不与engine墙钟相加，lookup offer不当作已完成恢复。

资源区别必须明确：S额外使用1GiB pinned CPU KV缓存及其维护工作，R/H不预留该空间。这是相同GPU预算下不同CPU资源成本的系统选择，不声称相同CPU资源算法比较。现成权重母本和元数据复用，不改共享环境；每臂fresh engine，整组持共同物理GPU锁，忙碌/未知则停止，不干预其他作业。

只根据真实完整结果裁决：报告两次均值和每个配对、actual token/s、flow/TTFT/ITL/最长gap、输出长度/答案/EOS截断、专家及KV传输、抢占与重处理。输出不同不叫等工作量加速。若S未实际恢复，成本交叉假设未验证；若初始化或观测失败，保留失败并修实际兼容性，不记机制NO-GO；若S不能改善完整请求或被H覆盖，停止本域单纯开启swap的优化，不调整CPU容量/阈值挽救。若有稳定完整收益，下一最小实验用全驻留专家作负控，检验专家扫描是否对边界有可辨别影响，而非直接宣布新方法。

状态：headroom六臂完整归档后，已于45495启动controller50966，远端`/dev/shm/moe-scheduling-20261002-kv-swap-r01`，顺序R16/H10/S16/S16/H10/R16、static-cap=10，当前RUNNING。备用包SHA256 `60f9e3cdc023c42e3b9ab805bff21ac724e74f51ff6e2c933b12dc6b2d0fbf68`、35文件均核实。22937原包只暂存；现场出现C long-QA整组controller62763，本方在GPU初始化前停止，未创建launch.json/结果，无本方后台任务。不将跨服务器数据作为配对。

## r01 因果混杂与 r02 的最小隔离

r01 首个S16的实际资源确认测量前host KV为空，但12个新请求在尚未被抢占时命中624-token共同fewshot前缀；关闭GPU prefix caching不会关闭native CPU connector的跨请求hash复用。实际load 1,075,838,976B、store404,750,336B、重复调度位置11，不能全部解释为抢占恢复。整组仍完整保留为CPU prefix reuse + recovery的系统对照；纯recovery因果解释标记INVALID/confounded，不抹去实际性能。

下一唯一修正r02保持三臂顺序、数据、资源和所有参数不变，共同使用`kv_recovery_src`。仅capture为每个提交的request设置唯一`cache_salt=kv-recovery-r02/{external_id}`并把原值记录在请求raw中；所有arm、warmup均同样处理。原生hash首块包含盐，后续块继承parent hash；原生抢占把同一Request对象放回waiting，不修改盐，故仅阻断请求间复用，不阻断原请求恢复。新副本capture SHA256 `67d3f4c0fbb2673da994db8e5887d2a5de3a3913a2105c7da0f329fe5e441de1`；旧r01源码/结果不变。r02仅在r01整组终态后启动，仍需实际检查无首次调度前prefix命中以及真实恢复传输，配置盐本身不当作测量证据。

r01已经六臂COMPLETE、本地归档SHA256 `3eac86e6dd8f10e30d57d8ba5e244750a22d99b77f094ea35ff51d07679576c1`，主分析所有资源检查通过。S16相对R16平均drained+0.18%、实际rate−0.05%、flow−2.20%，无吞吐收益；首轮和反序服务时间方向不同。r02此前因C math-runway controller53727占用暂存未启；待其整组进程退出、全GPU空闲后，现于45495启动controller55074，远端`/dev/shm/moe-scheduling-20261002-kv-recovery-r02`，包SHA256 `957dc12eb46838aad21d16b871bc7481d8e2b3b317fc5b70fa837cfb19572c91`，状态RUNNING。没有改变r01解释或数据。

最终状态：r02也已六臂COMPLETE，归档SHA256 `c3877b2c18a284fffc1299a144450db38572e01f203db0987b9c98f36513ffcc`，实际隔离资格通过，全部结果见[KV_RECOVERY_RESULTS.md](KV_RECOVERY_RESULTS.md)。controller55074结束，没有本方后台GPU工作。吞吐+3.50%同时worstgap+43.35%，未作为完整延迟优化GO；先定位额外恢复等待，再决定是否需要全驻留负控或已有恢复优先基线，不机械追加矩阵。
