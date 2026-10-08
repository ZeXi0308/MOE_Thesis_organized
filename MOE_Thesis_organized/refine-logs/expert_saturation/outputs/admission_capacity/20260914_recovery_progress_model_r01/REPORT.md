# A：资源进展模型与首次 victim 选择边界

Verdict: MEASUREMENT_ONLY。固定长度域内的状态进展模型通过回溯验证；首次 victim 单次替换未找到平均完成收益的模型空间。尚无在线方法 GO。

## 问题与方法

问题：动作前资源状态能否预测候选动作后的请求完成，而非事后用缺席计数解释停顿？
输入仅为截点的请求进度、KV 块数、running/waiting 顺序和截点前控制器历史。每个候选独立推进状态，不输入目标策略的未来调度、输出 token 身份或计时。模型复用冻结 AbsenceRotation 规则，模拟运行请求分配、尾部抢占、等待恢复、重算及完成释放。
范围：OLMoE、vLLM 0.26、单 RTX5090、APC off、旧 32 文档、3072 prompt/固定 1024 输出、1024 token budget。固定输出长度是该实验的已知条件；未知 EOS 不在本模型范围内。

## 已测结果

- 10 条 d6 分支/强基线轨迹，加四档压力的 16 条轨迹，共 26 条既有轨迹：逐步调度、分配后空闲块、请求完成步数匹配。10 条检查还逐步核对实际新输出事件。最初三条校准检查包含在这 26 条中，不重复计数。
- 这些是同一文档集合的回溯验证，不是 26 个独立 workload。压力适配器使用 block_count 构建占用计数，未验证物理块地址或 KV 张量；此前六格分支的有效 KV 资格是独立证据。
- 简单每步耗时模型只用压力组 block0 的 d2/d6 native 拟合：t = 5.458 ms + 0.4574 ms × scheduled_requests + 0.011651 ms × scheduled_tokens。3021 次训练调用，RMSE 3.558 ms。
- 12 条未参与拟合的既有轨迹：条件平均剩余完成时间最大绝对误差 2.10%，最后完成时间 2.60%。两个训练轨迹不算迁移证据。未建立置信区间、尾部误差界或跨 workload 泛化。
- 每步时长使用相邻调用起点间隔，末步使用 returned-start；含正常 host 间隙，末步边界采用上述近似。初次 prefill 调用不用于拟合，重算调用保留。无异常点删除。

## 首次 victim 的候选面

在已验证共同前态 step329，枚举 31 个 running victim，之后全部回到 least-progress；另保留默认 baseline。模型中没有一个候选降低平均剩余完成时间。最大预测平均损失约 0.38%；最好尾部候选 0000133 预测最大动作后沉默缩短 3.68%，但平均完成慢 0.38%。不把预测 Pareto 点当作实测收益或 Oracle。

最小接线检查：显式指定 baseline victim 与原 least 全轨迹相等；显式指定 most victim 与已验证的 first-most-then-least 全轨迹相等。全部候选仍执行资格/容量条件。模型原始版本和候选扩展版本分别保留 model_source.py、model_source_candidate.py。

单候选本地 Python 完整展开约 27–34 ms，整面约一秒；这是本地一次计时，不代表在线 GPU 主机开销，也不足以声称可每步部署。

## 科学裁决与边界

Evidence type: 原生 in-process 实测轨迹支持的 STRUCTURAL 模型，以及回溯成本预测；本轮新增 GPU 执行 0。
Strongest baseline: least-progress 为单次 victim 比较基线；native、持续 most-output 用于模型迁移核对。既有 completion-headroom 的强基线裁决保持原报告。
Oracle/headroom: 没有经过真实执行验证的全动作 Oracle；当前首次 victim 面没有预测平均收益。
Claim ceiling: 固定长度、封闭请求集合下状态到进展的模型保真性。不能主张在线控制收益、MoE 特有性、自然长度泛化、质量保证、多卡效果或统计非劣。
Failure category: 当前单事件 victim 选择缺少预测收益空间；不是整个资源调度问题失败。执行者定向检查，无新增独立审计。
Resurrection condition: 后续决策状态出现超出预测误差量级的候选完整请求收益，并经真实同前态分支验证。
One next smallest experiment: 仅 CPU 检验后续决策状态的单次动作替换，先比较冻结简单规则是否已有 material completion/pause tradeoff residual；无明显空间则不为 predictor 申请 GPU。

A 的判断：从缺席会计转向资源状态模型是合理且已取得窄域证据的一步；“因此已经能选出更好的在线动作”仍不成立。现在不值得为首次 victim 小差异追加 GPU 扫描。
