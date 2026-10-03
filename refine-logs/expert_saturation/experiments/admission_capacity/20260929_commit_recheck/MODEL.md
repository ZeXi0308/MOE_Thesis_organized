# 最小模型与可推翻预测（2026-09-29）

主问题是固定 GPU 和明确 host 预算时，恢复动作能否把保存/加载/重算转换为更多按时交付的新输出，同时保住完整服务效率。当前只检验 **H1：commit 时取消已不必要的 victim**；方法若成立也先定位为一般 serving，不给 MoE 专属性。没有使用 expert 状态；只有它在 token/KV/队列/近期延迟之外改变决策并改善请求结果时才考虑加入。

## 选择 H1 的证据与候选边界

| 候选 | 真实动作缺口、可见状态、相邻强规则 | 可区分实验与当前决定 |
|---|---|---|
| **H1 commit 即时重检（唯一实施）** | 两步 prepare/commit 间物理 free 变化；使用 commit 即时 GPU 空闲块、序列槽、目标/块所有权和 connector 状态，将原 `V={victim}` 改为 `V=∅`。native 分配器、selected/eager、兼容 LTR-style 是相邻参照；简单直接验资可能完全覆盖，无独立创新保证。 | 同底座 H1 off/on，先记录合法动作次数、目标首新输出及被保留 victim，后看完整请求 goodput/吞吐/flow。旧 D/E 保存快照还显示 6/10 个空闲槽，但只是 CPU 派生状态，不能当真实 direct 成功例。 |
| 冷恢复计算份额 | G 的两个首 resident 冷恢复状态需多次重算调用；改变单次 batch 中目标与 peer 的 token 份额，在线可见 computed/held KV/peer 余量。最强简单 `restore_first` 已在 CPU 分叉获得相同最少调用，且有 peer 输出代价。 | 同前态执行不同份额再比较完整请求；当前没有强规则后的净收益/自然频率证据，**不建第二 controller**。 |
| 保存范围 selected/full | host 实际可复用历史改变恢复成本；在线需要有效 host key/块与在途 transfer。原生 full 是强基线。 | F 四格已显示少重算不推出全服务改善，当前不重跑保存策略，只固定共同后端。 |

H1 的新增状态是**准备后、提交前的实时物理资源**，新增动作是**取消一次已经准备的抢占**。这与旧 cooldown/window/headroom/growth predictor 的提前触发或延长服务不同。台账已有一个 default-off CPU 候选，本轮的研究增量只可能来自正确槽位/屏障资格与完整请求验证；若只是修复基础 backend 计数或缺失检查，应进共同基线，不包装为算法。

G 详细诊断的 54 个真实 READY commit 与逐步资源快照对齐后全部 `free<need`（缺口 2–193 块），因此在这条轨迹 H1 没有直接恢复动作空间。其余 G/H 轻量格最多 816 次 commit 尚缺即时资源快照；旧 D/E 两次保存 CPU 候选不能给它们估发生率。运行前不改变阈值或加压来制造 direct，首次真实诊断若无合法动作就停止 H1 性能扩展。

## 状态、动作和物理不变量

`README.md` 给出字段来源、刷新时刻及未知回退。在线状态 `s_t` 仅含此刻 scheduler、KV block pool、native offload connector 和已交付输出。完整动作 `a=(r,V,m,q)`，本轮固定已接受 `r`、`m=selected-native`、`q=native`，只在 `V={old_victim}` 与 `V=∅` 之间选。原路径是 `no-op` 回退，保持旧 `commit`；过期计划、缺字段、自然 EOS、部分 load、队列忙或错误 ownership 均不得进入 direct 分支。

对于已确认无共享前缀且映射为完整独占块的后端，目标本次分配缺口为 native 实际分配所需 `need(r)`；若采用现有 16-token block 的请求视图，CPU 近似为 `max(0,ceil(num_tokens/16)-owned_valid_blocks)`，但最终必须与 native 分配器语义一致。`F>=need(r)` 只是 KV 资金条件；还必须满足空闲序列槽、target 仍可恢复、源状态可加载、先前 store/flush/load 屏障和 plan 版本。共享前缀或 staging 改变时，不能简单累加逻辑请求长度或把 `free` 当作可释放量。GPU 和 host 各按唯一物理占用计数；已登记 store 不因取消 victim 就自动消失，host 已保存块也不在新抢占后当永久丢失。旧 G/H 的 16 GiB host KV 是配置和末态有效容量，父 cgroup 大上限不是独立进程树硬预算。

可执行性判定仅返回 `{DIRECT, KEEP_OLD_COMMIT, CANCEL_STALE}` 和理由，不返回收益分数。`DIRECT` 必须在 native commit 前再次验资；实际 native 调度后反馈分配、preempted、pending transfer 和新输出，若 target 没有按预期进入可执行状态就记录失败并停止该运行，不伪造服务。两个策略从共同输入独立运行；不能从旧 trace 删除 victim gap 当作反事实。

## 最小效用模型与预测

前态只估短时域，不使用未来 EOS 或真实剩余输出。对已开始生成的请求，`g_i(t)=t-last_new_output_i`，预计停顿余量 `slack_i(s,a)=D_G-g_i(t)-T_next_i(s,a)`；未开始者改用 `D_F-(t-arrival_i)-T_first_i(s,a)`。`T_next` 由当前等待顺序、host 有效 KV、在途完成通知、剩余已知重算位置、最近 step 时长与传输排队范围估计；batch 共同服务按实际可共同执行的请求估，不默认 peers 全停。load 与 compute 若重叠，短时关键路径用两者的暴露并集/较长者，不将时间简单相加。范围过宽或字段未知时只记录区间并回退，不声称严格 SLO 保证。

H1 的事前**动作预测**：在同时满足 KV、槽位和 native 屏障的 commit，on 取消此处计划的 victim 抢占；后续原生调度仍可能自然抢占它或其他 peer。物理准入不保证目标首个新输出不晚：31 个 running 各占 1 个调度 token、1024 token 上限、目标需 994 token 的 [CPU 条件反例](H1_TOKEN_BUDGET_ADDENDUM.md)中，direct 路径只余 993 token，off 路径因驱逐一个 victim 余 994 token。原生 GPU 后果尚未验证，必须逐事件记录目标实际获分配量、首新输出、victim 与 peer 停顿。**请求级预测**是可推翻的：若这类事件自然发生且目标与 peers 的延期、保存及未来 KV 压力不抵消保留 victim 的收益，固定阈值 goodput 前沿可能改善；同时报告相对 native 的 token 吞吐和平均 flow。已登记的 `store`、batch shape 变化与未来 KV 增长均须计费，不能从一次 direct 决策推出净正效。旧 D/E victim gap 0.112/0.261 秒都非最大事件，特别不能预告最大 gap 下降。

局部核验固定共同的下 4 个正调度调用：同时报告目标首新输出、所有 peers 的新输出、已支付与仍 pending 的 save/load、GPU/host 占用、未完成恢复和下一步增长债务；时域末尾的未完成工作不计为已交付服务。完整请求独立运行才评价 `README.md` 已冻结的 `D_F×D_G` goodput 前沿、实际 tokens/s、完成/TTFT/gap 分布及全部失败。局部观察不会替代完整请求胜负，已支付的旧 prepare 工作不能作为继续驱逐 victim 的理由。

**反例与停止：** 如果没有同时满足 KV、槽位、connector 的事件，H1 在该自然域无动作空间；若 on 未改变动作，停止性能扩展；若改变动作却拖延目标、挤压 peers、增加后续转移或无完整请求增量，报告该动作/运行域负结果。若合理校准的 LTR-style 或 native 已覆盖同一前沿，吸收简单修正并停止新方法主张。所有状态仍是 `CPU_CANDIDATE / GPU_UNRUN`，直到原生生命周期和同资源完整实验真正完成。
