# 恢复之前的服务窗口证书 v2：固定 step405

Verdict：`STRUCTURAL_PRE_RESUME_ALL_PEER_WINDOW_CONDITIONALLY_INFEASIBLE`。在405真正动作前就能发现：固定原victim释放资源后，恢复到首输出可以兑现，但若全部30个当前peer每批继续输出，再给target三个输出机会会缺6块。该结论不需要先看到409；它也不意味着fit子集或整个窗口无解。

两repeat前态相同：free=0；固定victim3571在405 before持有212块；target3640的history=3274、computed=0、held=0，完整当前历史需要205块。固定victim移出后30个resident均pending1。

输入来源纠正：v1虽然校验了receipt释放量等于before持有量，仍将从pool_after得到的212传入predict，因此其“完全pre-action输入”表述不成立。v2的212直接取before victim held，原生receipt只进入validation_only。v1脚本及JSON/MD全部保留，数值结论未变。

候选释放量的能力资格来自执行前配置和已冻结安装检查：APC关闭、唯一KV group、FullAttentionSpec/FullAttentionManager、KVCacheCoordinatorNoPrefixCache、无KV connector/延迟释放/speculative或并行cache；safe-cap-qualification与已执行源hash一致。405 before所有请求均只有一个block_counts分量，Σheld=used=6656，free+used=usable=6656。只有这个已明确的不共享后端和完整前态资源账本允许把固定victim持有的212作为候选可释放量；不能仅凭APC off或sum相等推广到其它后端。原生receipt之后独立确认释放212，不参与预测输入。

预算1024、实际chunk threshold=0。每个peer每批1token，则target每批至多994；恢复剩余3274个输入位置至少要4批，最大分配路径为994/994/994/292。3274包含3273个已经执行过的历史位置和首输出的最后一个输入，不全是重算税，不能换算为毫秒。未读取未来EOS或未来分配来推导这4批。

| 条件机会批 | target执行位置 | peer执行位置 | target恢复后累计新输出 | 累计实际新增块 | 完整当前历史保留+peer增长 | 在212块内余量 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 994 | 30 | 0 | 64 | 206 | 6 |
| 2 | 994 | 30 | 0 | 129 | 209 | 3 |
| 3 | 994 | 30 | 0 | 193 | 211 | 1 |
| 4 | 292 | 30 | 1 | 212 | 212 | 0 |
| 5 | 1 | 30 | 2 | 215 | 215 | -3 |
| 6 | 1 | 30 | 3 | 217 | 217 | -5 |
| 7 | 1 | 30 | 4 | 218 | 218 | -6 |

前4批资源可容纳；第4批首输出时205个target块加peer新增7块正好用满212块。第5/6/7批如果all-peer继续共同输出，分别再缺3/5/6块。若只看恢复时205≤212并忽略peer增长，会错误把7块余量当作恢复后仍可用；若连恢复后的增长也忽略，就会签出无法兑现的四输出承诺。

完整历史保留与物理已分配量分列：未分配的历史余量只算一次，不和完整历史重复相加。这里的保留是资源资格，不要求立即物理分配全部未来KV。

## 同一前态推导的fit子集与等待代价

用405的peer历史加条件恢复期4个新输出，可直接推导首输出时各自held块边界，随后固定FCFS逐步跳过需要新块的peer：三个额外机会批分别仍可服务28/26/25个请求（含target），新块均0。799/2820/3345从第1个额外批暂停，133/2038从第2批，1401从第3批；窗口内新增无服务等待分别≤3/2/1·t_batch。这不是旧409前态被喂回模型，计算只来自405历史和明确的每批服务假设。

原victim3571在整个条件窗口继续等待，其额外等待需记为恢复4批的总跨度再加3个后续batch跨度，不能只计peer暂停。target自身到首输出还有恢复队列/执行跨度，均未预测毫秒。每个请求在405 schedule入口的当前output age写入JSON，只有当时已经返回的新输出用于age。

符号t_batch是后续三批各自耗时的上界，不是测得的未来毫秒；没有这样的上界就保留各批跨度之和。窗口结束后谁释放资源、被暂停peer何时真正得到输出、target是否再次丢弃状态都仍需资格化，不能把三批内等待界当成完整下一输出间隔保证。

## 条件与独立验证

4批是由当前待处理位置和token预算给出的下界，并非普遍时间预测。只有固定peer集合每批各1token、target拿满剩余预算时才达到。更小chunk或不同优先级可能增加批数；若peers仍继续执行，增长只会更大。未知EOS、完成释放、改变peer集合或其它恢复动作可能改变需求，必须重算，不能把当前条件拒绝扩成全局无解。

预测函数predict只接收405 before、从before持有量得到且通过能力资格的固定victim候选释放量、当前配置及到达顺序；原生receipt和真实后续独立放在validation_only：两个repeat原动作405–408的target/peer执行量及物理free均与上述前4批条件预测匹配，实际首输出累计203；之后409原动作抢占target。这只检验原动作的已执行前缀，不是替代窗口收益或未来Oracle。

当前能回答的是：恢复前已可检测all-peer四输出窗口缺块，同时保留有明确暂停代价的fit子集动作。不能直接据此延长保护，更不能称方法GO。唯一下一GPU仍复用共享cohort3 native/most/fit/residual反序八格，先检验强简单基线是否覆盖残差；窗口动作保持条件接续。

复跑脚本接受--results-root和--output-dir；已存在结果拒绝覆盖。无GPU、无controller，未修改上一轮原件。
