# 专家扫描摊销：直接粒度实验

准入模型六臂已失败，停止调阈值。唯一新动作是每step可调度token数512/2048/4096；保留native16准入、cap24、KV1GiB、expert-major无保护。旧训练mixed调用几乎达到入口miss-once下界，因此减少单次扫描miss的余量很小；增大prefill量可能通过减少扫描次数获益，也可能增加在服decode停顿。

可检验的最小成本分解：T_step(P,D)=T_nonMoE(P,D)+T_exposed_weight(P,D,C)+T_expert_compute(P,D)+T_host。已验证当前实现每调用每个入口缺失专家最多加载一次，故weight bytes=W*|U(P,D)\R|。当P足够大U接近64时，增加P未必继续增加这一项，但仍增加计算和等待。该式是会计和饱和假设，不把copy事件包络直接视为可相加的纯传输时间。完整目标仍为全部请求的实际wall/flow/TTFT/ITL，不由局部模型推算加速。

预声明顺序512/2048/4096/4096/2048/512，六个fresh engine，统一编译容量4096。共同warmup仅旧16条、实际token budget512；drain后设测量budget。测量用已被准入实验使用的source16..31，故是同输入探索和受控重复，不能称新的未见确认。每请求强制32输出，全部16请求/512输出，保留不同输出及实际KV抢占，不作自然完成或语义质量声明。预算改变产生各自route/KV/batch。

新增代码仅有界token预算设置，不建设新controller。六臂同物理GPU整组flock，busy/query失败即abort。若大粒度出现稳定净收益，再以最强static粒度为基线评估decode时限下的自适应粒度；不把配置调优直接当论文贡献。max4096较前一组max512的额外activation/峰值显存单列，因此仅在本组内作净性能比较。
