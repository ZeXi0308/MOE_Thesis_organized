# Native KV保存/驱逐契约定位

Verdict: STRUCTURAL / CPU_INTERFACE_CHECK。已有native pager提供保存任务、驱逐时等待和恢复通知，不能直接把轮转的立即驱逐改名为swap。CPU原方法检查通过；轮转+connector真实执行仍UNRUN。

依据是此前从本机器vLLM0.26封存的native_offload_source.json；本轮只读确认方法名/行号一致，进一步远端读取时SSH被远端关闭，后续分析使用封存源码，不声称全文件最新hash复核通过。GPU未初始化、无新driver，不干扰B六格。

| 接口/源位置 | 已核实行为 | 对新动作的约束 |
|---|---|---|
| offloading/scheduler.py:926–945 | store候选只枚举本步scheduled及finished请求 | 已被驱逐且未调度的victim不会因通知自动生成保存任务 |
| scheduler.py:447–456 | 默认限制到prompt token；也受computed/num_tokens/max_offload_tokens限制 | 保存decode历史需要显式配置，不能把默认prefix保存称完整KV保存 |
| scheduler.py:1122–1163 | preempted IDs和重用block IDs将已有store job加入jobs_to_flush | 必须正确传递驱逐与重分配元数据，不能只改running列表 |
| worker.py:318–325 | 本步store延迟到下一engine step提交 | 选victim和马上回收之间存在真实执行边界 |
| worker.py:281–303 | 提交deferred/本批强制store，再wait指定jobs | 可复用已有数据保护等待机制，不另写拷贝器 |
| worker.py:328–363 | store从不返回finished_sending，完成写completed_jobs；load返回finished_recving | 等finished_sending会永远等不到store完成 |
| scheduler.py:1215–1251 | completed_jobs计数归零才complete_store并移除transfer_jobs | job的创建不等于host副本已提交完成 |
| scheduler.py:720–775 | 有transfer_jobs则lookup返回None；命中后异步load | host保存/lookup命中/恢复可服务是不同状态 |
| scheduler.py:1268–1308 | request_finished返回False，靠重用flush防止读到覆盖数据 | 不能把finished接口当通用hold-KV保存屏障 |

三个CPU case执行封存worker原方法AST：上一轮deferred job在reuse标记前submit→wait；无job的驱逐不会凭空产生store；同批强制job从store_jobs移出并submit→wait。GPU传输worker替换为fake，reuse标记只是fixture边界，未验证数据正确性/真实异步/调度器集成；不能称swap机制通过。

可执行候选必须是两阶段：在victim仍有有效KV且可被保存路径枚举时选定并登记保存；保持该KV有效直到现成worker完成/flush；之后再做带完整原生preemption元数据的释放/恢复。保存仅完整可寻址chunk，最后不足chunk的尾部可能仍需重算，需实际核对。若没有提前一个执行边界的余量，这条选择性保存假说可能缺少action space，不能靠CPU模型伪造即时copy完成。

唯一下一步：从已封存真实d6首次轮转前态构造CPU planner/connector联合fixture，核对候选victim在前一个可执行边界是否能创建store job，以及保持KV到提交期间是否阻止目标恢复。只验证这一最弱链路，不删除rotation_native.py的connector不兼容保护、不启动GPU扫描。若成立再作最小真实保存/驱逐/恢复同状态分叉；不把现成pager收益作为新贡献。

主研究问题仍OPEN；默认native prompt-only offload和观察器微优化的失败均保留。已有方法净收益未成立，未验证full-decode保存、自由生成质量、多卡或选择性保存强基线。

## 首次轮转的提前边界CPU结果

selective_boundary.json两block一致：只用step328可见的running/output选择least-progress victim 3571，事后核对命中原step329受害者。328真实31decode正常结束，free149→148，victim已有207块；等待目标的完整history需要205块，释放后静态余量150块（未扣下一步其他请求增长，非完整下一步计划）。没有从未来输出选victim，329事件仅用于验证。

封存native _build_store_jobs与_calc_num_offloadable_tokens实际方法被执行，fake host allocator允许保存、fake keys和物理block IDs仅验证接口构造。默认prompt-only生成192块/3072tokens/402653184bytes的任务，留下234已计算token；允许decode后生成206完整块/3296tokens/432013312bytes，留下10已计算token。生成一个已登记job，不代表完成复制；真实hash可用性、跨层布局、host allocation和copy正确性均未验证。还可能需要下一待生成位置，不能把10写成完整恢复调用成本。

这个结果把问题从“是否在资源上必然无提前窗口”收窄为“能否在现成connector上正确保存指定victim并通过原生复用屏障恢复”。它只支持一个旧cohort边界的条件可行性，不证明整个策略、零等待或净收益。不存在新增GPU运行。

唯一下一最小实验：实现只选这一个victim的一次性保存接口资格包，保留native block/hash/job管理，用真实KV验证保存→驱逐通知→job完成→异步恢复→首新token全链条；计入等待且保留与同底座无保存对照。先完成CPU接口适配，再协调单组GPU，不能复用本fake allocator作为结果，也不能把整个swap控制器一口气铺开。
