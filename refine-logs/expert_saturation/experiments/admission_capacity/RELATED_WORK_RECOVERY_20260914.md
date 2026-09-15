# 恢复调度的直接近邻与待验证差异

2026-09-14；依据用户补充，由pager_literature子会话核对四篇当前可取得的arXiv正文/伪码：LTR v1、Andes v2、UniBoost v1、TokenFlow v1。本轮未执行其完整系统或逐一核验开源实现；论文机制与同runtime移植基线必须分开。此附录收紧A线研究边界，不改变已封存实验，也不把OLMoE模型使用解释成MoE特有性。

用户指出的直接碰撞成立。等待提权、固定时间片、容量/victim匹配、切换净收益、几何服务保护和传输排队成本均已有先例；“获调度尚未产生新输出”可以是待定位事件，不能单凭这个表述认定空白。

| 工作及原文 | 已覆盖的具体机制 | 当前可保留的实验问题 |
|---|---|---|
| [LTR §4.3 / Algorithm 1](https://arxiv.org/html/2408.15792v1#S4.SS3) | 饥饿计数触发提权与PriorityQuantum；max_waiting_time同时包括TTFT和最长token间隔。伪码加入batch时清等待计数、扣quantum，之后执行 | 调度轮次是否在当前重算runtime中给出了实际新增服务？先移植防饥饿模块比较；不靠“没用长度预测”区分 |
| [Andes §§4.1–4.3](https://arxiv.org/html/2404.16283v2#S4) | 按batch相关QoE收益/KV占用选请求；victims释放足够容量；上下文长度profile重算/交换，并扣除对其他请求的切换损失 | 请求长度成本与候选batch条件成本的动作排序是否有差别？不能把正常生成的batch模型说成完全忽略batch |
| [UniBoost §3.3 / Appendix A.1](https://arxiv.org/html/2606.18431v1#S3.SS3) | MemGuard以已生成decode token的几何阈值定义不可驱逐区间；伪码检查下一prefill/decode的KVNeed，并考虑优先级/交换成本 | 有效服务保护已有直接覆盖。需实测保护与后续KV增长发生冲突的状态及可执行处理，不能把“保护到有新token”作为新机制 |
| [TokenFlow §§4.2–4.3](https://arxiv.org/html/2510.02758v1#S4) | 消费缓冲与内存/I/O共同决定working set；恢复成本含驱逐排队、驱逐、加载排队、加载；用近期prefill每token成本比较重算；讨论重算与新prefill合批并拆batch | 候选batch的完整成本模型是否超过近期均值及已有batch调整？不能声称其漏计排队或未研究合批影响 |

正文未展开某个跨步保证，不等于该系统实现中不存在相应处理。当前未证明“跨步恢复资源可行性”是新颖性空白；这需要同可执行动作下的真实反例与修正收益。

A线问题可写为：固定实际KV容量下，一次恢复选择能否在后续重算和decode增长中持续可行，并尽早返回新增token，同时计入victim停顿与重复重算？强简单参照应先包含LTR-style等待提权/时间片、Andes-style长度成本净收益、UniBoost-style几何有效服务保护和TokenFlow-style近期重算成本；移植部分模块只能称style baseline，不冒充完整系统复现。

最小证据链先区分恢复获选、首次重算调用、后续重算/混合decode调用、首次新增输出返回。现有in-process账本的output_events是host收到engine结果的时间；它不是网络客户端确认，也不能分离GPU内token实际产生时刻。字段缺失标未测，不能把相邻时间戳重新命名成客户端可见性证据。

| 假说 | 首先可证伪的比较 | 若没有增量 |
|---|---|---|
| 当前容量可行仍可能无法完成有效恢复 | 同KV池，记录被选恢复到首次新增输出前的computed进度/KV/再次抢占，对照简单固定预留与几何保护 | 停止额外恢复预测器，不宣称跨步资源问题普遍存在 |
| batch状态改变完整切换代价与动作排序 | 同类上下文恢复，与不同实际decode伙伴混合；比较长度profile、近期均值和候选batch估计，包含其他请求的新输出 | 若简单成本已足够，不增加复杂模型 |
| 状态相关保护优于已有有效服务保护 | 固定时间片、几何decode阈值与一个最小状态规则，同计恢复收益/victim损失/KV锁定 | 无可重复residual则保留测量边界，不把换保护参数当贡献 |

原阶段A线曾优先定位强基线后的恢复失败事件；按当前主研究要求，贡献也可来自相同资源与预声明目标下的取舍边界改善，不要求先证明已有算法有bug。继续停止无目的调cooldown。已运行的B线有限到达U/M/F/X实验独立回答专家分页执行成本，本附录不将其收益转记到请求恢复策略。其结果仍需全请求成本与环境限制核对。

## 2026-09-15 Addendum：官方实现核验与当前采用边界

直接动作碰撞已确认：LTR已覆盖固定`PriorityQuantum`；[UniBoost MemGuard §3.3 / Algorithm 1](https://arxiv.org/html/2606.18431v1#S3.SS3)已用已获得decode服务的几何阈值限制重新驱逐，并在伪码中按当前batch依次检查下一prefill chunk/decode step的`KVNeed`与swap cost；[Andes §§4.1–4.3](https://arxiv.org/html/2404.16283v2#S4)已将首token、后续token、batch总context/KV、按上下文长度profile的重算/交换成本及所有peer的QoE损失纳入净切换决策。因此“固定/几何服务量子 + 目标与peer的增量KV可行性 + 恢复收益减切换成本”本身不是新机制。

UniBoost作者[项目页](https://yl3469.github.io/uniboost-icml26/)的Code链接指向其SGLang fork；本次固定核验提交[`e2a17c3e342cc47dfcbfd8b13990ba94fcb1a4da`](https://github.com/yl3469/sglang/tree/e2a17c3e342cc47dfcbfd8b13990ba94fcb1a4da)。该提交的[`schedule_policy.py`](https://github.com/yl3469/sglang/blob/e2a17c3e342cc47dfcbfd8b13990ba94fcb1a4da/python/sglang/srt/managers/schedule_policy.py)和[`test_uniboost_policy.py`](https://github.com/yl3469/sglang/blob/e2a17c3e342cc47dfcbfd8b13990ba94fcb1a4da/test/registered/unit/managers/test_uniboost_policy.py)可见waiting-queue Boost排序、Gamma-Ada、cache frontier与TP一致性，未见论文MemGuard的几何不可驱逐栅栏或`KVNeed`选择器。所以本地若按论文伪码接入，只能称“UniBoost MemGuard论文组件基线”，不是官方代码移植或完整UniBoost复现。Andes论文声明有vLLM参考实现，但本次核验未找到论文指向的可确认官方公开仓库；缺用户TTFT/TDS、token pacer、batch latency模型、恢复成本profile与完整QoE refiner时，也不得称Andes复现。

当前不接受、不启动几何栅栏。[原生完整保存E](../../outputs/admission_capacity/20260915_natural_native_full_gate_r01/RESULTS.md)无1–2输出后再抢占；[D/E共50个释放边界](../../outputs/admission_capacity/20260915_joint_growth_decision_r01/RESULT.md)上G2与`free_blocks==0`的动作集完全相同，E最小两步余量仍为4 blocks。`k=16`仅是按当前16-token block对齐的**待校准本地参数建议**，不是UniBoost论文已验证的本运行域参数；论文报告的默认值是`k=256`。若强基底以后出现阈值前再抢占的真实残差，最小接入点才是将[`staged_store_rotation.py`](../../outputs/admission_capacity/20260915_natural_native_full_gate_r01/pkg/staged_store_rotation.py)的“首个新输出即释放`protected`”替换为一个预声明、已校准的几何分箱跨越条件，同时保留原生完整保存和现有`recovery_guard`。当前剩余贡献只能来自强基底后有真实动作价值的决策残差，不来自再组合上述已知组件。

### Native offload backend 选择（同日源码核验）

实际安装的vLLM 0.26.0在`vllm/envs.py:2001–2004`将`VLLM_USE_SIMPLE_KV_OFFLOAD`默认设为`0`，`vllm/config/vllm.py:863–881`在`kv_offloading_backend="native"`下据此选择`OffloadingConnector`（`0`）或`SimpleCPUOffloadConnector`（`1`）；对应官方tag源码见[`envs.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/envs.py#L2001-L2004)与[`config/vllm.py`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/vllm.py#L863-L881)。因此D/E/F显式写`0`只是固定实际默认路径，不是关闭一个可直接等价替换的默认优化。更关键的是，`SimpleCPUOffloadConnector`在`enable_prefix_caching=False`时直接返回且不建立scheduler/worker（[`simple_cpu_offload_connector.py:56–87`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py#L56-L87)），而D/E/F均固定该值为`False`；只翻此开关会关闭CPU offload，不能成为当前四格中的公平一臂。

即使另开`enable_prefix_caching=True`使Simple可运行，它也不是当前“完整增量保存 + 选择性flush + host状态动作”的同义实现：默认`lazy_offload=False`的eager路径只处理已确认、可hash的整块，延后一step保存，并明确可能漏掉同step完成请求的最后整块（[`manager.py:523–531`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/simple_kv_offload/manager.py#L523-L531)）；传输在模型执行后异步提交（[`worker.py:198–217`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/simple_kv_offload/worker.py#L198-L217)），但任一preemption会同步全部在途load/store（[`worker.py:279–297`](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/simple_kv_offload/worker.py#L279-L297)）。当前`OffloadingConnector`则在`offload_prompt_only=False`时按computed/finished边界包含decode chunk，并只对受抢占或即将复用block关联的`jobs_to_flush`等待。单卡下两者都可接收同一16 GiB名义容量，但其保存覆盖、完成/flush边界、LRU管理与可观测接口不等价；源码也不证明Simple具有更好的完整请求性能。当前不主张best-native-backend或默认APC覆盖，因此无需改F或补跑Simple；只有未来扩展到该主张时，才应以单独backend-sensitivity实验先校验APC开启后的保存覆盖、恢复正确性、实际host占用与完整请求指标。

本次实际安装源码固定SHA-256（根目录`/root/autodl-tmp/expert-saturation/vllm-0.26/lib/python3.12/site-packages/vllm/`）：`envs.py`=`e078b0acb8e658faa6a8144d13a9036e2a82af40b348f419489a2607247cd936`；`config/vllm.py`=`d4c31282c664c91bee81267a2134a9609d16c33e88cc40e0f020256346510a87`；`distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py`=`866fa9cfd4451aa0b6f0a586b8d549af1bceea405e8ef08c8e0e14cfd45dc19f`；`v1/simple_kv_offload/manager.py`=`256f0b26bca3f16f958dcaf1a1759b11e6f42b02d27ca1fd522f923faeda480b`；`v1/simple_kv_offload/worker.py`=`19bc6fb2a5dbd71068266e15844c91836b7c79a513974487f6afc24cb4546104`；对照的`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`=`89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1`、`v1/kv_offload/base.py`=`6b89cf3e146bdbb995ac531e588f14326941bc804a5be103b8bfc27b57d30c31`。

### G之后的action-level贡献选择（同日更新）

时效更正：[G](../../outputs/admission_capacity/20260915_natural_recovery_cadence_r01/RESULTS.md)已用两格`native_full_native`关闭“同资源完整原生保存、无额外轮转”的参考缺口。`current20→eager0`实际只移除全局`min_steps_between_swaps`，其余selected保存、最长缺席target、victim、容量/absence/residency/progress gate和首次新输出保护均不变；动作是更频繁地强制换出一个running peer并把合格waiting request提到队首。它在action层最接近[LTR §4.3 / Algorithm 1](https://arxiv.org/html/2408.15792v1#S4.SS3)的饥饿计数触发提权与`PriorityQuantum`，不是新的调度原语。保存使该动作可恢复并带来成本；Andes已覆盖切换净收益与peer QoE损失，TokenFlow已覆盖驱逐/加载排队及重算成本。因此当前“更小上尾gap、更多请求间隔或完成代价”的结果仍是同域基线表征。

即使128篇未使用文章再次复现，决定性的最近邻比较仍是：在同一`OffloadingConnector`、selected保存、GPU/host预算、输入和完整请求分母下，将已校准并在新输入前冻结的LTR-style等待提权/时间片与`eager0`直接比较，同时保留原生系统参考并报告实际轮转、传输和peer代价。贡献的必要条件不是证明既有算法存在特定失效，也不是所有指标同时超过LTR-style和原生参考；必要的是在预先声明的目标与容许代价下，相对最近邻LTR-style得到同预算、同目标的可复现增量（包括资源—服务Pareto边界改善），或者建立一个新可复现边界并说明与最近邻的action-level差异。原生参考可以继续保有median、TTFT或completion等其他目标优势。若LTR-style已捕获同一目标的前沿，当前动作就仍只有基线表征。“load/重算启动消耗时间片却尚未交付新输出”只是一个可检验的充分候选残差；该状态若不存在，不能据此排除其他经测量成立的决策残差。仅写“保护到首次新输出”仍与UniBoost MemGuard的有效服务保护碰撞，不能据此主张新颖性。

#### G语义下的LTR组件复用边界（CPU源码判断）

现有LTR adapter不能原样与G的selected/native-save语义合法组合。[`ltr_recompute_native.install`](../../outputs/admission_capacity/20260914_ltr_component_probe_r01/preparation/pkg/ltr_recompute_native.py#L116-L173)显式要求`connector is None`、调度器实例没有既有`schedule`覆盖、没有`skipped_waiting`，并只接受`WAITING/RUNNING/PREEMPTED`；[G adapter](../../outputs/admission_capacity/20260915_natural_recovery_cadence_r01/pkg/staged_store_rotation.py#L41-L70)则必须安装在`OffloadingConnector`上，也独占`schedule`覆盖，并在异步load/store/flush中处理`WAITING_FOR_REMOTE_KVS`。因此两种安装顺序都会触发资格拒绝；这不是缺少一个开关。

可直接复用的是纯CPU的[`LTRCounters.begin_schedule/after_schedule/finish`](../../outputs/admission_capacity/20260914_ltr_component_probe_r01/preparation/pkg/recovery_service_components.py#L35-L77)接口及其只依赖截至当前调度步状态的200/10计数语义。不可直接复用的是其完整planner/action：它每步做全批priority packing、可选择多个低优先级victim并按当前history重新预留；G则做单target/单victim的两步prepare/commit、selected store、异步flush/load与首次新输出保护。已有覆盖仅包括官方提交对应的计数语义、`200/10`来源、CPU接口，以及旧d6重算后端的boost-off/on四格；没有覆盖G自然域中的阈值/量子校准、`OffloadingConnector`生命周期或同预算完整请求比较。若以后补最近邻臂，只能把计数接口接入一个与G共享保存后端且互斥安装的有界adapter，并称为“LTR-style组件对照”，不能称现有adapter直接复用或完整LTR复现；本轮不实现。


### H之后的比较合同澄清（root，2026-09-15）

[H](../../outputs/admission_capacity/20260915_natural_cadence_holdout_r02/RESULTS.md)两对在预声明3%吞吐/5%均完成成本预算内保留最大生成gap方向，但幅度降至13.60%/3.12%。这关闭一次输入迁移，不是最近邻增量或统计稳定；native full仍保留效率和多数请求停顿优势。现推进同selected-save后端的LTR-style计数/量子组件，G/H原件保持只读。旧adapter禁止connector，新的互斥接入完成前不使用旧d6结果代替。

本轮直接重读[LTR v1 §4.3 / Algorithm 1](https://arxiv.org/html/2408.15792v1#S4.SS3)：原论文的max_waiting_time同时包含TTFT与最长后续token间隔，排序针对全请求集合，量子按实际入batch迭代递减。当前recovery-only、single-target/single-victim、most_output及selected native-offload实验只移植防饥饿组件，不包含ranking predictor或原CPU-SWAP路径，也不复现其全队列目标。共同主目标仍是生成长停顿与完整服务成本，TTFT必须并列保留，不能把本文目标或实现称为完整LTR。官方固定commit链接本次web读取失败；计数实现来源沿用此前已核固定源码，不假称本次重新取回官方代码。

最小比较应给LTR-style有限且有理由的开发域校准机会，冻结后在未参与LTR参数选择的输入上比较。H虽未用于选择LTR参数，root已看过H的eager/current结果，因此不能称全程盲测或新增独立输入。不得用G现轨迹离线计数直接排名反事实服务表现；同窗口的必要参照可以新跑，已经闭合的native参考保留其原合同与时间，不为换标题重复运行。若单target串行化或沿用旧资格门人为压制计数策略，应先修正共同实现/明确action-space边界，再进行性能结论。
