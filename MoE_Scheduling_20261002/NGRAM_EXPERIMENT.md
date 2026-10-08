# 固定 prompt lookup：专家搬运摊销的直接基线

## 后续固定长度 1/2 对照（运行前冻结）

N4 的完整结果为吞吐下降 2.76%、专家搬运增加 20.62%。唯一下一实验比较 AR/N1/N2/N2/N1/AR；所有格沿用下述资源、输入、预热与完整成本合同，只把原生 `num_speculative_tokens` 设为 0/1/2。仍用 source32–47，不运行 holdout48–63，不加入动态控制器。判断依据是同组两次完整请求吞吐、flow、质量、输出量与专家搬运；不挑选最好单格，不把 chunk gap 写成 token ITL。

`ngram_short_src` 与 `launch_ngram_short.py` 已冻结，35 文件包 725,732 B，SHA256 `6b9e27a09f61e92e3c4bbca17c2b2341f7b6c03f35657d802b8d9acf9a389a74`。runner SHA `921e2afa9466ffe21c3a0a3cab8192257f06157291a92d0a463d04b0e3f1e8ea`，capture SHA `85bc204bcffc53c7f47feb002a9299ecea10c550fca25b5793ce34643763f96f`。目标 45495 `/dev/shm/moe-scheduling-20261002-ngram-short-r01`；整组锁和每格 GPU 边界检查保留。开始前两卡均有其它组，当前未启动本方测量。

分析使用 `analyze_kv_group.py --design ngram_short` 与 `analyze_ngram_activity.py --design ngram_short`，隔离资格覆盖 6 个 CPU-KV episode、允许零实际恢复，但有恢复则必须有同请求抢占前因。保留每个请求的独立策略轨迹，固定长度基线本身不作为论文创新。

## 成本与正确性模型（候选尚未实现）

完整 episode 记真实输出量为 Y、排空后的总时间为 T。互斥会计为 `T = T_pure_prefill + T_mixed + T_pure_decode + T_no_MoE + T_outside_engine_and_drain`；只有实际存在的阶段计入一次，不另加已经被阶段时间覆盖的 DMA/等待。吞吐超过 AR 的必要且充分的测量关系为 `T_candidate/T_AR < Y_candidate/Y_AR`。输出随策略改变，因此这不是相同工作量的速度比。

`analysis/ngram_cost_model.py` 已对完整 N4 数据核对阶段守恒，并写入 `results_ngram_r01/ngram_cost_model.json`。在分别固定两次 N4 已观察到的 1524 个输出这一纯代数条件下，它还需减少 0.7065/0.3908 秒（3.53%/2.00%）才能匹配同序 AR 的 token/s。这不是可实现收益上界；任何真实动作都必须重新运行自己的输出、KV、缓存和路由轨迹。

在一个已到达的 MoE 层 ℓ，令 `R_ℓ(r,j)` 为请求 r 的验证行 j 的真实 top-k 专家集合，`C_ℓ` 为进入层时实际驻留集合，`P_ℓ` 为本步必须保留的非 speculative 行所需专家。每请求选择保留输入行 0…k_r 后，`U_ℓ(k)=P_ℓ ∪ ⋃_{r,0≤j≤k_r} R_ℓ(r,j)`。在当前每个 entry-miss 专家只加载一次的 expert-major 执行器里，本层加载字节为 `B_ℓ(k)=Σ_{e∈U_ℓ(k)\C_ℓ} bytes(ℓ,e)`。因此删后缀可省的当前层字节，只来自被删行独占、且原本非驻留的专家；其他请求/保留行仍要使用的专家不能重复计为节省。共享集合使多个请求的动作耦合，单 token 分数直接相加不成立。

这项集合关系不保证 GPU 时间减少：还必须支付 row-mask/决策、保留的 attention 与 router、实际分组/kernel、后续缓存变化、KV rollback、未来补算与调度成本。已执行层的成本是沉没成本，不能扣回；尚未观察的后续层 route 不能当在线信号。当前原始 trace 未资格化 speculative row identity，故没有据它估计层内裁剪可实现收益，也未建设控制器。

greedy 正确性边界：输入行0是最后已提交 token，行j是草稿 d_j；行j的 logits 预测下一位置。若要提交 h 个匹配草稿及1个 bonus，需要行0…h共 h+1 行完整通过所有层。各层保留前缀只能单调缩短，sampler只读最终幸存行；若遇首个不匹配则提交此前匹配草稿和该处 target 纠正 token。幸存行必须保留原始因果注意力、原 router 与全部 top-k；dead row 的 dummy/KV不能影响它。已提交 bonus/纠正 token 自身 KV 尚未计算，下一轮必须计算；拒绝后缀KV必须从有效长度移除，不能误释放同块有效前缀。BF16 dispatch/归约变化仍可能改变临界 argmax，不声称位级一致；随机采样的内容依赖截断有偏反例保留在 prior_notes.md。

最小 off-by-one 反例：target在 a 后输出 A、在 aA 后输出 B，草稿是A。若仅完整保留行0，只能输出一个A；把该行预测同时当成接受草稿与bonus会错误输出AA。实现必须先通过行身份、sampler长度与KV回退资格，才能谈完整请求性能。

## 已有 trace 的无行身份松弛上界

为避免猜测 request-row 顺序，`analysis/draft_row_relaxation.py` 只计算更宽松的问题：每层允许从所有行中任意删除至多 D 行，D为该步实际scheduled draft数，甚至允许删掉真实必留行。令 S_e 是引用冷专家e的行集合，则删除集 A 的实际节省为 `Σ_e bytes_e × 1[S_e⊆A]`。仅 |S_e|≤D 的专家可能被消除；将每个这样的专家费用均分给其引用行，最大的 D 个行费用之和给出上界。当前所有专家同大小12MiB，精确有理数计算后向下取整为完整专家数。小规模枚举验证了共享引用与边界情况，不依赖启发式最优性。

两格 N4 各137步×16层的行数/step/实际entry-miss加载均匹配；松弛结果结构完全相同，保存于 `results_ngram_r01/draft_row_relaxation.json`。pure_prefill 7.877GB 的当前加载无可删草稿，上界0；mixed 69.080GB 的局部上界5.285GB，144次层调用中44次即使允许任意删D行也不能省一个冷专家；pure_decode 641.452GB的局部上界467.694GB，仍很宽。该结果不能判死候选，也不能作为有收益的证据。

每层可独立选择不同删除集、入口缓存/route保持原trace，因此把局部字节上界相加不构成真实策略的完整请求上界。报告内任意删行witness也只是松弛问题的可行解，不能当作合法后缀策略的收益下界。没有接受量损失、后续缓存/KV、未来补算和时间上界；不再扩展离线oracle，下一步仍是已冻结的固定N1/N2实际对照。

2026-10-02，代码与35文件包已冻结，四格已在45495由controller68021全部COMPLETE并归档。完整结果见`NGRAM_RESULTS.md`；fixed4无净吞吐收益，下文为保留协议。先前因D组占用暂存，现已确认其两组终态清空并现场领取。包SHA `9d9dfcc47cf6172d36fe31e84c20ed0f10137279cd488f87e62288b06f4960ef`。单块KV预留已经停止；本组是不同动作的最小运行验证，不延续其参数调整。

现有S16实际输出串上的唯一固定2/5/4检查，原生可提案1400个前缀中735个有候选（52.5%）。假定原串不变、先由target产生首token，再逐次推进匹配前缀与一个target token，请求内decode pass条件计数1400→1047。这只表示匹配机会，不是batch engine call数量、真实spec接受率或GPU加速。脚本和全部请求结果在`analysis/prompt_lookup_opportunity.*`。

直接运行已有vLLM原生ngram，`method=ngram,num_speculative_tokens=4,prompt_lookup_min=2,prompt_lookup_max=5`；不加载draft模型，不实现层内后缀撤销或专家删减。两臂AR16/N4/N4/AR16，各fresh engine、同一45495 GPU整组共享锁（22937已被其它组占用）。两臂均cap16、expert24、CPU KV1GiB、GPU KV1GiB、measured Q2048、engine4096、HND/eager。source32–47完整prompt、16同时到达、greedy自然EOS/max512；独立request cache salt保留。所有尝试、失败和正反序保留。

各engine用旧训练16输入、固定32输出、warmup Q512预热自身算法；因此相同的是warmup输入与长度，不能说两臂执行相同autoregressive warmup。测前清空CPU/GPU KV，完整记录资源、所有专家加载、全部KV保存/恢复与post-capture drain。N4可能改变抢占次数，因此隔离资格不要求N4必须实际发生CPU load，但实际命中必须仍只属于该请求抢占后的恢复。

旧行身份observer明确不支持speculative。两臂共同不安装它，none retention不依赖该observer；保留scheduler-step关联、真实所有kernel行与成本。此次**每行请求/位置身份未获资格**，不能用这些数据宣称层内裁剪已能安全执行。将来若实现那一动作，须先独立验证spec row与KV rollback映射。

分析口径：完整flow/TTFT、EOS/质量/输出差异、actual output token/s与drained wall继续有效。一次host receipt可能含多个token，其内部ITL未知；有多token chunk的请求/整格ITL统计置null，另报告正输出chunk之间的gap。不得把重复host时间产生的0当作真实零ITL。已调度decode位置包含验证候选，位置重叠也可能来自proposal rejection，不能全部解释成抢占重算。

成本模型只用完整实测：每轮实际提交token收益需要覆盖lookup、额外验证、专家新增加载/后续淘汰、KV与调度成本。先判断已有固定长度实现是否在当前运行域有净收益；若无收益，定位实际接受/额外搬运是否仍给层内停止留下残差，不直接堆自适应器。若有收益，再用新输入和强固定长度参照确认，已有ngram/spec动作本身不构成论文创新。

候选残差的有限先例核查见`prior_notes.md`：SpecMoEOff、EVICT已覆盖大量动作；AcceptMoE已覆盖逐层router与驻留状态驱动的专家删减。仅“不改幸存token原路由的层内token后缀撤销”在所核查三篇中未见直接覆盖，尚非新颖性结论。随机采样正确性未证明，本轮边界仅greedy。
