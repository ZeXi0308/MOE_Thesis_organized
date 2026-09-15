# 自然运行域的启动节奏：存在可检验动作差异，胜负更可能由 peer 代价决定

**建议完成一次 selected/current20→eager0 的有界对照。** 现有同域状态已经满足改变提议路径的必要条件，不是预期零动作实验；但尚未证明提前提交成功或完整服务获益。按主研究会话提供的 F 主分析，native-full 没有胜出，因此 selected/current 是当前停顿优先参照。此前模型文档把 E 称为“强底座”，这里明确降为已经资格验证的 native-full 基线候选，不能因其重算更少而推荐 full。

复用 D/E 主分析的 funding 行和原 snapshot，仅筛选原 selector 被 cooldown 阻止、同一最老 target absence≥30、无 plan、非 direct、固定 most victim 可筹措完整历史的状态。原 target/victim 排序都唯一，无需 request-ID tie break；原保护/队列/纯 decode 入口门槛已经由 `selector:swap cooldown` 状态确认。

| 必要条件 | D：selected/current | E：native-full/current |
|---|---:|---:|
| 满足上述条件的原轨迹快照 | 201 | 304 |
| 覆盖不同的原 cooldown 间隔 | 18 | 22 |
| 最早例距上次 swap / target absence | 4 / 59 steps | 4 / 60 steps |
| 最早例 free / target 需块 / victim 持块 | 130 / 191 / 154 | 136 / 193 / 156 |
| 最早例当前 prepare 共同增长 | 1 block | 2 blocks |

最早例是 D step 712、E step 745，只用于定位现有状态，不作为在线选择特征。此时原 20-step timer 拦截，而 0-step timer 会继续进入既有 target/victim/prepare 检查；双方其他可观察标量条件均满足。各组另有 **3 个**同样 funded 的候选，其准备步共同增长超过当前 free，直接反驳“funded 即 READY”。未来 prepare 的执行、物理 prefix 所有权、native 存储/flush、下一步 commit、load 和首输出仍须由运行证明。

这些是重复相关的状态快照，既不是独立试验，也不能当新增轮转数量或可节省秒数的上界。D 的 116 个、E 的 38 个 direct-funded cooldown 状态未计作动作证据：取消 timer 后它们仍应交由 native 恢复，不据此制造新 forced rotation。G2 释放边界 predictor 保持停止；此处的一步 prepare 算术仅用于说明必要条件不充分。

**预先保留的跨状态预测：若 eager 实际改变了启动节奏，决定 current/eager 完整服务排序的首要风险更可能是 peer 服务代价，而不是剩余重算。** 被恢复请求的启动前等待有缩短机会；但更多、更早的 victim 交换会把暂停传播给其他请求。E 已经有 36 次成功加载、仅 266 个重算位置，仍有 16 个有效恢复后再次抢占段，说明“恢复内部工作很少”没有消除服务重新分配。重复 H2D 加载是必要成本；D selected 还可能增加保存/flush 成本，故不能把 E 的加载阶段成本数值直接迁移到 D。

这是一条待验的排序解释，不预言必然 NO-GO，也不要求所有指标一起改善。运行时按以下同一条证据链检验：

1. 首先确认实际发生 cooldown<20 的 prepare/commit/load/新输出，且 victim、保护、token share 和资源预算未变；没有实际动作差异便停止本对照，不再解释模型收益。
2. 保留所有请求的停顿、完成和吞吐权衡；peer 由真实新抢占的 victim 身份确定，不按事后坏指标挑选。若等待缩短，但受影响 peer 的间隔/完成代价扩大、而 host-call→首输出没有相应恶化，支持 peer 代价这一预测。
3. **反证条件：** 若发生排序损失时 peer 的受影响范围与停顿/完成代价均未增加，主要变化却落在 load-submit/host-call→首输出及重复传输成本，则此预测应被否定，转向重复加载这一成本通道；若暂停和完整服务均稳定受益，就接受 eager 这个简单策略，不为保留模型另加模块。

现有资料不能给出秒级 Oracle，也不能推算自然 EOS 后的最终运行集合。这份说明只判断对照有无必要动作空间，并冻结一个可被真实运行否定的解释。无 GPU、无新 controller、无阈值扫描；下一实验身份和执行清单仍由主研究会话统一。

必要条件结果：[cadence_conditions.json](cadence_conditions.json)；复算：[cadence_conditions.py](cadence_conditions.py)。D/E 原件路径由小结果文件列出，GPU 主表未重算。
