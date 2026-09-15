# Host cost：捕获到GC与线程CPU，残留变化仍未定位为单一根因

Verdict：`MEASUREMENT_ONLY / OBSERVATIONAL_SOURCE_LOCALIZATION`。X observer off/on/on/off四格全完成，编译覆盖与观测资格均通过。两次on的engine主机包络相差329.500ms，同线程GC重合仅相差3.801ms；不能用GC变化解释全部差额。线程CPU时间接近主机包络不等于纯Python/CPU计算瓶颈，可能包含驱动或忙等。

唯一研究问题：在无新增编译的X执行中，哪些CPU执行、GC重合或等待指标伴随到达分叉前的额外成本？本轮复用现有CPU/GC观察，增补engine、真实planner及runtime.kernel的host spans。它是原生vLLM进程内加定制pager的来源定位，不是新调度策略或F/X性能比较。

| 格 | capture s | 平均完成 s | TTFT s | 最大ITL s | 编译准备 s | 完整进程 s |
|---|---:|---:|---:|---:|---:|---:|
| 0_observer_off | 8.146565 | 5.374170 | 0.787204 | 0.189937 | 5.889346 | 62.227846 |
| 1_observer_on | 8.196133 | 5.446604 | 0.809590 | 0.205485 | 5.519757 | 59.610601 |
| 2_observer_on | 8.509791 | 5.762068 | 0.861077 | 0.207869 | 5.888007 | 61.816418 |
| 3_observer_off | 8.176339 | 5.381924 | 0.790754 | 0.194353 | 5.474158 | 59.715127 |

| 开启观测相对关闭 | 正序 | 反序 |
|---|---:|---:|
| capture | +0.608% | +4.078% |
| 平均完成 | +1.348% | +7.063% |
| 平均TTFT | +2.844% | +8.893% |
| 最大ITL | +8.186% | +6.955% |
| 完整进程 | -4.206% | +3.519% |

这些是带观测与不带观测的实际完整轨迹差，不是可扣除的纯观测开销。on会改变CPU分配/GC时机、实际到达批次和后续状态；两个配对仅10/16输出相同、route不同。off两次16/16输出及route相同，capture +0.365%；on重复11/16输出相同、route不同，capture +3.827%。各两次相关重复不建立噪声底、显著性或旧r02根因。

| 窄host span总和 | on1 wall / thread CPU ms | on2 wall / thread CPU ms | 同线程GC重合 on1 / on2 ms |
|---|---:|---:|---:|
| engine | 8156.657 / 8154.735 | 8486.156 / 8484.765 | 46.426 / 50.228 |
| planner | 243.178 / 242.152 | 256.909 / 255.848 | 24.587 / 26.643 |
| kernel | 412.713 / 412.051 | 601.275 / 600.406 | 16.639 / 18.678 |

三种span存在嵌套，表的行不能相加。GC重合按同线程实际callback区间与span裁剪并合并，不是可移除GC成本。planner wrapper不含调用方metadata参数准备；kernel wrapper不含调用方参数构造/x.contiguous，记录的是host函数至返回，不是GPU kernel执行时间。较高thread CPU同样可能计入驱动CPU工作或自旋。process CPU涵盖其它线程，不能再与thread CPU或rusage相加。

两次on分别184/183条GC事件，全部有start/stop；GC callback已恢复。原生逐step宽cpu_delta包含前后观察，不能从较窄engine span直接扣除。对应原生观察外围包络31.641/15.357ms仅覆盖该处snapshot，未包含全部逐层记录、初始化、GC时机变化或后续轨迹效应。完整准备/测量/导出/退出保留在process时间中。

`sched_schedstats`样本为0；原始schedstat仍保留非零counter delta，其有效口径未资格化，不能从这里证明无排队或某个等待根因。频率值仅作驱动样本，不代表整个步骤的实际CPU频率。off的GC/spans零条表示未观测，不是GC没有发生。

**固定输入与执行。** 同一此前已测16文档，P128/O32、0.25秒计划到达、token160/prefill32，X private21×16+stage48、总384槽/4,831,838,208专家字节、实际1GiB KV、CPU0–7/OMP8。原11runtime文件不变；每格独立空Triton cache及M1…160×双GEMM320次准备。普通单cohort warmup与arrival不变，未启用same-engine模式；新增hook仅在measurement开始，无新CUDA同步或设备读。

64请求/2048输出/10176调度位置，3600测量层调用/4048全层调用；4个原始预热请求另计8输出。on两格分别56/55engine spans、896/880planner及kernel spans，总3663条；off无新增spans。所有source、身份、span包含关系、状态恢复及compiler-domain检查通过，measurement无新增Triton compiler。12次GPU边界查询通过，共同flock覆盖整组；不主张持续物理隔离。

**证据保留。** 140归档成员，其中139个payload大小/SHA匹配，archive 7,127,208字节/SHA256 `dea1b44f5e643ec8ca5c4e977cd56eddb41f38c0af37027c1230c554686270b3`。[analysis_rebuilt.json](analysis_rebuilt.json)与远端analysis语义完全一致，issues=[]。[限定结果审计](EXPERIMENT_AUDIT.md) PASS，未发现P0/P1；复用同族reviewer，接受状态为provisional。GPU于1789326482.405978释放；当前仅CPU解释。

**前缀定位完成。** [逐调用定位](PREFIX_LOCALIZATION.md)显示，两次on的step1/call137/layer9 planner均约20.9ms，其中同线程generation2 GC分别重合14.097/14.102ms（约67%），仍有约6.8ms未被GC区间覆盖。前三步engine差32.170ms，GC重合差仅0.091ms；旧r02的layer13/14大包络未在本轮重现。首步前snapshot已出现16.575/0.099ms差异，内部未分段计时，均无已记录GC重合。两on的可观察rows/routes/plans/cache/output前缀至step10相同，step10返回1.975/2.067s跨过第9请求2s到达点，step11首次分成70/102行。这些事实只定位本轮，不追认旧r02根因。

**边界与下一步。** 强基线为同X执行关闭新观察的负控；Oracle/headroom未测，质量、SLO、稳定净收益和独立GPU计算时间均未测。失败类别是根因仍未唯一化，不是观测无效或paging family NO-GO。本次来源定位闭合并停在SOURCE_LOCATION_CEILING；停止追加同类审计或逐层数值溯源。只有出现能区分具体成本解释、改变完整请求结果的最小动作，或需要检验已有机制的独立确认设计，才开启下一GPU组；不从现有差值扣除GC/观察成本。

本轮回答：已经测到真实GC重合和几乎全程计在线程CPU上的engine host区间；GC差额不足以解释两次on的全部变化，残留不能直接归因于Python、操作系统排队或GPU。
