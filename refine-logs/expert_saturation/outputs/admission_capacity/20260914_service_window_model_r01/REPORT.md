# 恢复后的服务窗口：最小资格模型与两个反例

Verdict：`STRUCTURAL_MODEL_ONLY / REAL_ACTION_RANKING_UNVERIFIED`。当前证据支持把恢复 KV 的保护与执行 token 分配分开；不能保证首输出以后得到足够服务。最小模型可以分别拒绝“无法摊销”“KV 增长越界”“损害其它请求等待”的动作，尚未在真实运行前完成动作排序验证。

Repository HEAD：`de64dae5ea4fa3c92ec605e9847e817d8e68f3ac`。主树共享 dirty，仅只读；本工作在 `/private/tmp/moe-window-model-20260914` 独立 detached worktree。已读 `docs/current/README.md`、`docs/ideas/README.md`、`SERVING_RESOURCE_STUDY.md`、`RESULT_LEDGER.md`、首输出四格真实补记、first-dispatch 定位及 token-reservation 准备合同。当前用户定义的请求恢复问题是本轮问题；旧权威页不自动更新。

继承事实：首输出保护四格实际已完成。每个保护臂 27/27 义务由新输出释放，0 次首输出前被打断，但 4 段只产生 1 个新 token 就再抢占；重算位置从 74,982 增至 96,955。最大 ITL 下降，吞吐两对变号，27/32 请求各自最大 ITL 恶化。这里只引用已完成的生命周期测量，不重新执行四格。

唯一问题：从当前可见状态出发，一次候选恢复及其有限后续服务，是否同时存在可执行的资源窗口、足够的新输出机会和可接受的其它请求等待？最弱链路是当前历史能够装入，并不等于之后的增量 KV、执行预算和输出等待仍允许服务。

## 模型

状态保留目标及其它驻留请求的当前完整历史 `H`、有效 GPU 已计算前缀 `P`、实际持有块 `K`、最后新输出年龄 `a`；另有 block size、当前 free blocks、当前 host 使用/上限、host prefix 长度及同 request/epoch 有效性。`P` 表示可复用的内部进展；只有真实的新输出事件更新 `a`。本版不模拟新 arrival、EOS、共享前缀引用或未来 victim。`free_gpu_blocks` 必须是固定 victim 规则已确认可释放资源后的数值，不允许把持有量直接当可释放量。

只给有限候选窗口（1–32 个新输出机会）资格，不使用全轨迹 Python 调度。候选说明 co-batch 集合、恢复中其它请求的新输出事件和 action-specific batch 成本。恢复队列、传输、计算、首输出返回是一个已由后端资源顺序约束的 DAG：

本版任何将产出新 token 的 peer 必须是已经驻留且 `computed_tokens = history_tokens - 1` 的 pending=1 状态；尚有历史需要恢复的 peer 直接拒绝，不能凭输出时间 profile 跳过其恢复成本。目标也必须仍有待处理的最后输入。字段 `recompute_tokens` 只用于核验“剩余历史加首输出最后输入”的执行覆盖，其中有用的首次计算不全是重复工作，更不能把该位置数当作 marginal 毫秒税。

```text
allocate actual current-history KV
→ queue / transfer / remaining recompute dependency graph
→ first engine-returned new output
→ finite jointly batched output opportunity
→ release, re-evaluate, or observed terminal completion
```

`finish(v) = duration(v) + max(finish(parent))`，首输出延迟 `r = max finish(v)`。并行分支只有后端确实允许时才可省去依赖；共享计算/传输资源的串行边必须由 profile 写出。CPU 算术不会证明新的 overlap。首输出整个调用跨度包含其它请求有效工作，因此与摊销用的 `C_remaining` 是两个字段，缺少后者则 `amortized = null`，不能默认把整个 `r` 当重算税。

在这个**先保留全部当前历史**的特定动作中，目标第 `n` 个新输出需要的块增量为：

```text
ΔK_target(n) = max(0, ceil((H_target + n - 1) / block_tokens) - K_target)
```

`n - 1` 是因为首新输出处理的是已存在的最后一个 pending 输入；该输出自己的 KV 到下个 token-step 才形成。其它请求依其在恢复期和服务期实际获得的输出数做同样计算。恢复峰值另计临时 GPU 块；host 已用量加 staging 必须不超过固定预算。分别检查恢复阶段峰值和后续服务峰值，不因为最终可以释放便允许中途超额。真实 pager 可以采用更精细的增量分配，但本模型拒绝一个保守动作不等于拒绝该 pager 或整个工作负载。

在预先声明的诊断预算 `α = 每个新输出允许摊销的 marginal exposed 恢复毫秒` 下，恢复开始前的机会下界是：

```text
L = max(1, ceil(C_remaining / α))
U_KV = 当前历史保留、共同 batching 的全部 KV 增长仍可承受的最大 n
U_age = 所有请求从最后新输出起计算的连续等待仍满足本轮预算的最大 n
eligible(n) = valid_source ∧ resource_ok(n) ∧ age_ok(n) ∧ n >= L
```

这不是业务 SLO；下面的 `α=1 ms/token` 和 gap=10 ms 只是为了检验排序的合成输入，不能据此选择真实评价阈值。未被执行、只有部分历史恢复的请求一直累积 output age。被共同 batching 的请求允许在窗口中不断产出；窗口不是独占或永久不可抢占。

未知 EOS 意味着 `n` 是资源/调度可提供的输出机会，不是保证产生 `n` 个输出；真实终止立即释放。所有年龄判断只覆盖当前有限窗口，后续动作需重新资格化。新到达、更慢 batch、host prefix 失效、KV 峰值增大触发重估；回退仍使用固定 victim 规则和原生资源检查，记录不足额服务，不扩预算、不忙等保留、不通过内部进展重置 age。

## 支持实例：共同 batching 使下界和等待上界相容

以下毫秒均为合成 profile，没有真实性能主张。block=16；目标 `H=32, P=16, K=1, age=3 ms`，同 batch 另一请求 `H=31, P=30, K=2, age=4 ms`；free=4 块，host=100/120 bytes。恢复 DAG 为 queue 1 ms 后到首新输出 5 ms，总 `r=6 ms`；其中 prospectively charged marginal tax=4 ms，后续共同 batch=1 ms。恢复时 peer 在 2、4 ms 各有新输出。

| 候选 | 下界 | 峰值新增块 | 目标最大 age | peer 最大 age | 资格 |
|---|---:|---:|---:|---:|---|
| mixed，1 个新输出机会 | 需要 4，实际仅 1 | 1 | 9 ms | 6 ms | 可执行但不能满足该摊销预算 |
| mixed，4 个新输出机会 | 4 | 3 | 9 ms | 6 ms | 本合成模型成立 |
| solo，4 个新输出机会 | 4 | 2 | 9 ms | 13 ms | 容量足够，peer 等待越界 |

CPU 运行实际得到上述排序。它支持“窗口可以 co-batch”及“资源可行性、服务摊销、其它请求等待必须分开”，不证明 4-token 窗口在真实 vLLM 中更好。

## 反例一：首次输出可执行，摊销窗口却不存在

保持相同目标/peer 上下文，仅 free=2 块，恢复 profile 中 peer 没有新输出。第 1 个输出可执行；第 2、3 个输出共需要目标 2 个新增块，仍可执行；第 4 个时目标新增 2 块，peer 又跨 block 边界增 1 块，总需 3 > 2。因此 `U_KV=3 < L=4`。peer 等待也单独报为超额，而不是混成一个“负收益”标量。

结论仅是这个保留当前历史、共同服务且不更换 victim 的候选不能实现指定预算。减小恢复规模、暂缓目标并先服务 peer、其它合法 victim 或不同后端可能改变局面，均未由此扫描穷尽。不能写成负载无解、问题 NO-GO 或全局 Oracle 上界。

## 反例二：恢复很贵，也不应按沉没成本继续保留

当前已经恢复完成，历史上花费多少毫秒不作为函数输入。比较从**现在**到相同未来目标服务量的两动作：保留的新增 exposed cost=3 ms；切换基础新增成本=1 ms，未来再恢复 marginal cost=10 ms。若后续不再恢复概率为 1，切换成本为 1 ms，效率偏好 `SWITCH`；若一定需要恢复，成本为 11 ms，效率偏好 `KEEP`；若仅知返回概率在 `[0,1]`，结果 `UNRESOLVED`。即使效率偏好 KEEP，也仍须通过其它请求等待约束。

这里的返回概率和未来恢复成本需要从执行前可见状态估计及独立评价，不能读取未来 EOS。这一接口显式保留不确定性；没有可信估计时，保留简单规则。窗口执行到一半也只按剩余恢复/未来再恢复成本比较，不拿原总账单强制偿还。局部有效服务评价则另记真实新输出数以及之后丢弃/复用的状态，不由模型替代。

## 真实结构支持：step 406 / F148

只读现有 `first_dispatch.json`，两次均独立重算这个**同前态**证书：free=148；恢复完整当前历史还需 142 块；30 个 resident pending=1 请求只多需要 3 块；合计 145，余量 3。token budget 为 `994+30=1024`，无需新增 victim。它说明已知恢复动作可以与 resident 单 token 工作共批，因而“保护 KV 必须拿走全部执行预算”不是这一前态的资源必然结论。

真实 guard_all 后来在 408 首输出、409 解除保护、411 被抢占，与 406 的预算分配是不同事件。不能从上面单步证书推断 411 的反事实或任意窗口长度收益。本模型没有给这个前态填写合成毫秒、未来新输出或未来 EOS，也没有沿旧 trace 执行候选未来。

## 当前交付与唯一下一实验

新增 CPU 核心少于 200 行，6 个针对性测试通过；检查动作资格、增量 KV、恢复峰值、host 有效性/预算、剩余计算覆盖、peer pending=1/目标仍需输入、DAG 重叠、输出 age 以及不按沉没成本保留。examples runner 写出合成支持/反例和两个真实单步证书。Python 3.9 首次导入因 union annotation 求值失败，加入 postponed annotations 后同一测试命令通过；主会话定向检查指出 peer 可凭 profile 跳过历史恢复，已通过明确 pending=1 支持域及回归测试修复。没有 GPU 执行或性能测量。

下一项 GPU 判别已由另一共享会话封包并首次运行：同预算 `fit_scan / guard_all / guard_residual` 反序六格。只检验拆开恢复 KV 保护与 token 预算是否真实让 resident 得到新输出，并改善代价，而不同时增加窗口长度。若 fit 已覆盖收益，保留 fit；若 residual 无预算动作，实验无效；若恢复完成性还在且损害降低，修正已识别的执行分配问题；若之后仍有大量昂贵的少输出再抢占，才用**新前态/新运行**比较固定短窗口与有 KV/age 上界的机会窗口。该模型的真实 action ordering、校准成本、到达/EOS 泛化都还未验证。

Evidence type：合成 `STRUCTURAL_MODEL_ONLY` 加已有真实 `STRUCTURAL_ONE_STEP_DIAGNOSTIC`；没有新增 native-serving 数据。Strongest baseline：共享六格的 fit_scan；本模型尚无运行对比。Oracle/headroom：有限动作资格非 Oracle。Claim ceiling：模型正确性与已观察单步资源可行；不是方法 GO。Failure category：真实窗口可兑现性和性能增量仍 OPEN。Reopen/continue 条件：新的实际生命周期证明在强简单规则后仍有可执行且未被利用的有效服务机会，才值得接入模型；不以合成例子替代这种残差。

研究问题的当前回答：昂贵恢复只有在“足够新输出的机会下界”与“其它请求等待及增量 KV 的上界”相容时才值得安排该窗口；当前真实证据只证明了一步可共同执行，尚未证明保护到更多输出会改善完整请求的延迟—效率边界。
