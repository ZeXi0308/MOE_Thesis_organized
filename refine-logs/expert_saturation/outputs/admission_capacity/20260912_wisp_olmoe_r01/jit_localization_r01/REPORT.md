# 普通U：长调用的编译来源已定位

2026-09-14。两格完整执行，冻结分析为`DESCRIPTIVE_COMPILER_LOCALIZATION`、issues为空。本次新跑出的7个超过100ms的measurement层调用，全部位于layer0；真实compiler pipeline加首次launcher/driver加载区间覆盖其host包络的97.40%–98.25%。这是实测来源定位，不是扣去编译后的假想性能，也不能逐一回溯判定旧八格的未记录事件。

同一16文档P128/O32、0.25s到达、prefill32、token budget160；普通U固定24×16槽（4,831,838,208B）与1GiB KV。两新引擎各有一次第一文档128→2 warmup，随后保留各自专家cache、独立推进请求。第一格使用新建空Triton磁盘cache，第二格原样保留；其他torch/CUDA/FlashInfer cache不宣称冷。39项输入及5份安装源码受SHA约束，原11份runtime源逐字节复用；新observer仅记录，不加GPU同步或改变模型动作。完整process保留启动/编译/预热/测量/写盘/shutdown成本。

| 格 | capture s | 平均TTFT s | 平均完成 s | 最大请求ITL s | 完整进程 s |
|---|---:|---:|---:|---:|---:|
| 0_cold_triton | 11.645552 | 1.813838 | 9.112692 | 1.011509 | 57.977016 |
| 1_retained_triton | 9.232272 | 1.183067 | 6.598253 | 0.222932 | 48.312630 |

| 阶段 | 冷cache：pipeline/磁盘命中/首次加载次数 | 冷cache主机区间并集 | 保留cache：pipeline/磁盘命中/首次加载次数 | 保留cache主机区间并集 |
|---|---|---:|---|---:|
| 初始化 | 7 / 0 / 7 | 8.484941s | 0 / 7 / 7 | 0.428006s |
| 唯一请求预热 | 2 / 0 / 2 | 0.530472s | 0 / 2 / 2 | 0.002804s |
| 请求测量 | 12 / 0 / 12 | 3.761722s | 2 / 12 / 14 | 0.642973s |

初始化中的6个事件/格没有active pager apply，保留为初始化来源，不能附会到某个请求或layer。Triton原生listener的cache_hit区分真正compiler pipeline和磁盘读取；旧jit_post_compile_hook告警在二者返回后都会触发。CompileTimer阶段采用主机时间，含编译链及中间IO；独立perf_counter区间包含lookup/加载等完整调用成本。嵌套区间只取并集，不与外层apply相加。

冷cache测量中的12次pipeline共3.755144s，12次首次加载共0.006578s；6个长layer0包络为0.540–0.818s，各含两次fused_moe_kernel编译。保留cache仍在step3、rows96遇到两个新pipeline，共0.624777s，连同加载覆盖0.639317s包络中的0.625693s（97.869%）。一次自然轨迹的磁盘预热不能覆盖所有随后可达的编译specialization。第二格这里尚未形成最大输出间隔，说明应分别看TTFT与已开始输出后的停顿。

两格共32测量请求/1024输出、5088调度位置、1600测量/1824全部层调用，另有2条预热请求/4输出。6次GPU边界检查通过；整组实际持共同flock，结束后PID22999不存在并交接APC后续组。没有已知本组外部初始化碰撞；advisory锁与边界检查仍不等于对所有不遵守锁的进程做连续监控。

两次请求输出仅9/16逐token相同，route hash不同，实际49/51个engine step；H2D为291,357,327,360/300,341,526,528B。编译延迟改变真实时钟到达的合批过程，因此capture差2.413280s不等于编译区间差3.118749s；不能逐项扣除后当作同轨迹反事实。冷→保留次序没有随机化、每条件只有一次，也不估计总体噪声底、显著性或稳定收益。

归档[attempt01.tar.gz](attempt01.tar.gz) 2,824,639B，SHA256 `c798b3c70e24cc2363610277c19c46d58803d2f07997311ff71f99463d0b4844`；90成员中的89份payload逐文件大小与SHA通过，清单本身随归档校验。私有compiled二进制留在远端，精确前后inventory保存在execution.json：0→172文件，第二格172→188文件；不改旧cache或删除失败。主要证据：[冻结分析](attempt01/analysis.json)、[冷cache事件](attempt01/results/0_cold_triton/jit_probe.json)、[保留cache事件](attempt01/results/1_retained_triton/jit_probe.json)。

| 裁决字段 | 本次结果 |
|---|---|
| Verdict | MEASUREMENT_ONLY；长调用的实际编译来源在本次探针中已确认 |
| Evidence type | 单卡native vLLM＋普通WiSP expert pager，host来源定位与完整请求 |
| What was measured | 两新引擎/32请求/1024输出，编译分类与逐层/逐step对齐，冷与保留cache完整成本 |
| What was not measured | 旧八格每次尖峰的逐事件归因、稳定性能差、全shape预热后的F/X增益、质量/生产SLO/超显存部署 |
| Strongest baseline | 普通U同一实现的编译cache状态；此处不比较新机制，F仍为后续X的强简单基线 |
| Oracle/headroom | 不从编译时间扣减构造系统Oracle；X/F可重复headroom仍未验证 |
| Claim ceiling | 本次长调用中97.40%–98.25%包络与实测编译/加载重合；短预热未覆盖全部实际specialization |
| Failure category | 先前稳定性前提不足；单次自然预热仍遗漏后续形状 |
| Resurrection condition | 在明确计费且覆盖所声明specialization的启动条件下，X/F还有可重复完整请求差异 |
| One next smallest experiment | 按已声明token预算建立固定编译specialization覆盖基线，保留全部启动成本，确认测量时无新pipeline，再比较F/X；不继续盲增warmup次数 |

直接回答：普通U的数百毫秒层调用可以由真实编译解释，本次7个长调用均有逐事件证据。保留一次自然运行的磁盘cache仍会遗漏形状，当前不能将旧或新单次比例视为稳定分页机制收益。
