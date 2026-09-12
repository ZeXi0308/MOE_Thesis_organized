# 自然显存压力：实现与输入已完成，GPU UNRUN

2026-09-06；HEAD `2a37765`，分支 `agent/publish-current-moe-code`。
**Verdict：`UNRUN / BLOCKED_SSH_HANDSHAKE`。这不是系统 NO-GO。**

本轮问题：在既有 OLMoE/RTX5090 原生运行栈上，合法长上下文是否造成实际 KV
容量不足，并影响持续请求的完成？最弱环节是运行域是否存在，尚非专家信号或控制器。
上一轮核验了共享工作区新增的24个单次降档 episode 和36次原生 companion 调用，
据其负结果改变了下一研究动作，属于证据驱动的推进；本轮没有重跑或认领它们。

实际完成的工作：

- 准备32篇真实 WikiText-103 train 完整文章；同文短128/长3072输入，固定输出1024，
  相同50ms到达。Pinned模型配置确认总上下文上限4096。未重复文本填充或跨文章拼接。
- 实现8个独立进程对照：cap16/32 × 短/长上下文 × 正反序重复。共同engine32、
  max_model_len4096、token预算1024、普通显存预算0.90、固定精度/路由/全驻留权重。
- 复用此前实际执行的请求捕获与指标模块。新增真实KV块会计、内存storage去重和
  首次抢占入口保护；guard在释放KV前终止该episode，不继续被中断的scheduler。
- CPU验证了32对身份/前缀/上下文、捕获及指标复用字节、别名会计、guard不调用原方法、
  失败调度留存及wrapper还原。分析器保留全部8个缺失cells为UNRUN，没有填零伪装结果。

初始3968输入/128输出在执行前改为3072/1024，使较早请求有机会在后续长prefill接纳
期间继续活跃；这是尚无GPU数据时的域设计，原因已写入事前规则。长上下文测试不使用
旧短请求9ms门槛筛选收益，5s/0.2s仅保留为参考指标。

**远端实际状态。** 本轮对用户指定的同一主机/端口进行四次正常SSH连接检查，均关闭。
其中一次verbose诊断确认TCP连接成立，但SSH握手收到：

```text
kex_exchange_identification: banner line 0: HTTP/1.1 502 Bad Gateway
kex_exchange_identification: Connection closed by remote host
```

这只能证明当前连接路径未到达可用SSH服务，不能判断远端实例是否关闭或GPU是否空闲。
未使用替代端口、跳板或其他转发路径；本轮没有上传文件、没有启动GPU进程。
这是连接阻塞，并非本轮再次触发上传审批。用户持续研究授权已保留。

执行包为 `execution.tar.gz`：10个运行文件，838884 bytes，SHA256
`62f968b639dd1ec8951c96df20d36681039e386e3d116615226bce4048bb2175`。
其中6个代码/规则文件与4个输入配置/工作负载；不含凭据、模型权重或其他工作区材料。
`run_campaign.py` 是唯一执行入口，每个cell完成或容量保护退出后才启动下一个进程。
远端目录预定 `/root/autodl-tmp/moe-native-memory-pressure-20260906-r01`，使用既有
`/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u run_campaign.py`。
恢复连接后先核对真实GPU进程与目标目录，避免和共享会话重复运行。

若容量保护触发，保存原始未完成请求和边界事件，分类为 `CAPACITY_BOUNDARY_STOP`；
其截断完成率不与完整episode吞吐混比。普通native排队与未调度decode单独记录，
不隐去其成本。具体固定参数、暖机和解释边界见 [DECISIONS.md](DECISIONS.md)。

| 结束字段 | 当前结论 |
|---|---|
| 实际测量 / 证据类型 | 输入与实现的CPU资格检查；0个新增GPU episode |
| 支持或削弱假说 | 尚无新的GPU证据；代码通过不支持容量或方法收益 |
| 最强简单基线 | 计划固定cap16/32；后续有空间才补token/KV预算接纳 |
| Oracle / claim ceiling | 动态Oracle未运行；没有expert换入/回收、SLO方法或CCF-B结果 |
| Failure category | 连接路径返回HTTP502，未完成SSH握手 |
| Reopen / next | 同一SSH连接恢复后执行已冻结8个cells；不新增第二条实验链 |

目标仍为寻找并验证论文研究链，未标记完成。未push、未修改权威账本或封存原始结果。
