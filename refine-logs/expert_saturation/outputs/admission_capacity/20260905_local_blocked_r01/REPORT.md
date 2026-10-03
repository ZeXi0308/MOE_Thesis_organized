# 第一轮非抢占并发容量实验记录

日期：2026-09-05。仓库：MOE_Thesis_organized；分支：agent/publish-current-moe-code。
开始 HEAD：`2a37765fe522b1d74609a686f1d327ede7619a50`，开始工作区干净、领先远端一个提交。

**Verdict：`UNRUN / REMOTE_CONNECTION_BLOCKED`；接纳实现完成，科学问题尚未回答。**

已读 `docs/current/README.md`、ideas 索引、RouteShape-SLO README/STATUS、v2
LIGHTWEIGHT_STATUS、Expert-Saturation 最新 proposal 和 N0d v3 verdict。
继承 N0d 的路由差异证据，不继承容量/控制器收益。本次执行顺序来自用户的新指令，
不再把 N0e 当作容量探索的前置条件；旧权威文件与原始实验均未修改。

## 改了什么与已验证内容

新增 `experiments/admission_capacity/`：复用已有 KV stack/split，接通真正的 FCFS
非抢占接纳 cap。所有 active 请求每轮推进；降 cap 等待自然完成。旧实现会先 prefill
全部已到达请求，再截取 `active[:max_batch_size]`，因此不能直接用于该动作。
新增代码只覆盖这一接纳循环、完整请求计时、U/C 归约和可运行扫描入口，不实现新 Controller。

10 个定向检查通过，其中 4 个用真实 tiny 随机初始化 OLMoE CPU forward 验证接纳、
不同逻辑 KV 长度、完整 active 请求身份、2→1 自然排空和生效延迟、U/C/零 token、
统计 OFF/ON 输出一致性；另外 6 个验证 TTFT/TPOT、全量分母、失败/未完成与单 token。
接纳时序测试使用可控时钟，**这些是工程正确性证据，不是模型性能、容量曲线或科学结果**。

环境：本地 Python 3.9.6、Torch 2.8.0、Transformers 4.57.6；CUDA/MPS 不可用。
真实入口已运行，输出 `CUDA unavailable; pretrained GPU capacity experiment is UNRUN`。
`config.json / environment.json / commands.txt / status.json` 保存的是该次执行事实。
入口随后补充了异常时区分“已执行但指标未完成”和“从未执行”的状态计数、模型来源与
隔离检查范围说明；没有改写该次原始环境或状态文件。

两个历史已用 GPU 入口分别只尝试一次 BatchMode/8 秒超时 SSH：
`connect.westc.seetacloud.com:12106`、`connect.westd.seetacloud.com:37116` 均 exit 255，
返回 `Connection closed`。没有到达 nvidia-smi，没有启动/停止资源或修改远端环境。
因此当前 GPU、进程隔离、模型缓存和原生后端可用性均未验证。

## 本轮研究结论

| 固定字段 | 结论 |
|---|---|
| Evidence type | tiny OLMoE CPU 工程验证；真实请求级 GPU 实验 UNRUN |
| What was measured | 非抢占接纳/排空的执行行为、身份与指标会计；本地入口阻塞行为 |
| What was not measured | 预训练 OLMoE cap 容量曲线、U/C 增量、真实统计税、原生后端、质量、EP |
| Strongest baseline | 已实现实际静态 cap 扫描；尚未调优/测量，ordinary-state 动态基线未实现 |
| Oracle/headroom status | UNRUN，尚不知道最佳 cap 是否随条件变化 |
| Claim ceiling | 实现与本地工程验证；未来 GPU 探索最多 CUSTOM_CONTINUOUS_RUNTIME |
| Failure category | 运行资源不可达；没有否证 signal、action 或 method |
| Resurrection condition | 无科学判死，无需“复活”；恢复已有可达 GPU 与缓存即可继续 |
| One next smallest experiment | 同一个 OLMoE cohort，cap=2/4/8，steady/bursty，OFF/ON 与两轮反序重复 |

当前不能选择 A/B/C：**工作集与集中度是否改变最佳并发选择仍未验证。**
下一步是恢复一条已有 GPU 连接后执行已准备的真实扫描；先观察真实响应，才决定是
A 的最小压力修正、B 的无动作增量边界，还是 C 的单卡无增量。没有固定 3%/5% 判死线。
运行命令与计时/基线边界见 [入口说明](../../../experiments/admission_capacity/README.md)。
