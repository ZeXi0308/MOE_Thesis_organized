# Remote-KV 计数的观测开销

结论：**在本线固定的同步原生生命周期和 gate 观测位置，可精确地只扫描 `_inflight_prefills`，再过滤 `WAITING_FOR_REMOTE_KVS`。**

应用记录（2026-10-07）：原 `4bfde70` 观测版本六臂结束后，本次独立新 revision 仅将 `Gate.state()` 的 remote 计数应用为上述集合扫描，并用 `requests.get(request_id) is request` 排除已删除/替换对象。所有模式一致；`begin()`、采样、日志、阈值、10 秒规则和 runner 未改。复用 CPU 检查并补充计数等价的状态转换/空集/清理边界用例，8 项通过。旧归档与旧 GPU 结果不受影响；新 revision 尚无 GPU 性能结果，本次未连接或使用 GPU。

被检查文件为 `vendor/scheduler.py`，SHA256 `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`。

## 源码依据和边界

| 路径 | 源码行 | 与集合关系 |
|---|---|---|
| 集合定义 | 351–353 | 原注释明确包含 prefill chunks 和进行中的 async KV loads。 |
| 开始 async load | 978–1009 | 全文件唯一给 `WAITING_FOR_REMOTE_KVS` 赋值的位置是 982；998 将同一 request 加入集合，然后 continue，不将其计入本步 scheduled tokens。 |
| 正常 prefill | 1011–1038 | RUNNING 的未完成 prefill 也加入集合，所以不能直接用集合长度作为 remote 数。 |
| prefill 完成 | 1247–1262 | 只遍历已分配 scheduled tokens 的请求；完成 prefill 后 discard。等待 async load 的请求此前走 continue，没有 scheduled tokens。 |
| 原生抢占 | 1218–1234 | 入口断言 RUNNING，discard 后变 PREEMPTED；不从集合移除仍处 remote 状态的请求。 |
| KV 接收完成 | 2637–2645、2590–2601 | 收到完成信号先保留 remote、记录 finished ID；promotion 执行 KV 更新后转 PREEMPTED 或 WAITING。此时可能暂留集合，但过滤状态会排除它。 |
| 无效 load/重算 | 2651–2752、2765–2777、2820–2823 | 调整 computed tokens 并记录失败接收 ID，不移除集合、不改变 remote 状态；之后仍沿上述完成/promotion 路径。 |
| finish/abort，包括失败终止 | 2193–2203、2207–2213 | 先将 status 置终态，再 `_free_request()` discard。即使延迟释放 KV，该请求也不再是 remote。 |
| 延迟释放结束 | 2236–2239、2643–2649 | `_free_blocks()` 断言请求已结束；不会删除仍处 remote 的活请求而在集合中留下 remote 残影。 |

因此在 `Gate.state()` 实际调用处，下式可保持现有语义：

```python
active = len(scheduler.running) + sum(
    r.status.name == 'WAITING_FOR_REMOTE_KVS'
    for r in scheduler._inflight_prefills
)
```

这个结论有明确范围。982 赋状态至 998 入集合之间存在语句级短暂间隙，但当前 `async_scheduling=False` 且 gate 不在该分支内部回调观察，不会看见它。`add_request()` 在 2120–2140 没有禁止外部直接传入 remote 状态；若外部改状态、换 scheduler/connector、插入并发观察或改变同步配置，不能无条件沿用本结论。本线通过正常 engine 新建请求并由 scheduler 启动恢复，满足范围。

`_inflight_prefills` 是内部接口，优化应保持固定源码身份。它包含非 remote 的 prefill，因此必须保留状态过滤；也不能用它替代 `begin()` 中 PREEMPTED 等完整恢复状态的扫描。

## 小型 CPU 计数测量

在本地 macOS 26.6.2 arm64、Python 3.9.6，以 hashable request stub 和 `IntEnum.status.name` 比较完全相同的两个计数表达式。每状态 384 个 request，每种表达式 1000 次 × 3 轮，取每次调用耗时的轮次中位数。集合同时含 remote 与普通 prefill，三种状态计数均完全相等。

| 全部 requests | remote | inflight 集合大小 | 扫 requests 中位 μs | 扫 inflight 中位 μs |
|---:|---:|---:|---:|---:|
| 384 | 0 | 1 | 142.922 | 0.675 |
| 384 | 4 | 6 | 143.021 | 2.642 |
| 384 | 16 | 64 | 143.609 | 24.774 |

可复现的测量核心如下；stub 只用于测量 Python 计数，不模拟恢复生命周期：

```python
import enum, statistics, timeit
class Status(enum.IntEnum):
    WAITING=0; RUNNING=1; WAITING_FOR_REMOTE_KVS=2
class Request:
    __slots__=('status',)
    def __init__(self, status): self.status=status
for remote, local_prefill in ((0, 1), (4, 2), (16, 48)):
    requests={str(i): Request(Status.WAITING_FOR_REMOTE_KVS if i < remote
                  else Status.RUNNING if i < 128 else Status.WAITING)
              for i in range(384)}
    inflight=set(list(requests.values())[:remote+local_prefill])
    full=lambda: sum(r.status.name == 'WAITING_FOR_REMOTE_KVS'
                     for r in requests.values())
    small=lambda: sum(r.status.name == 'WAITING_FOR_REMOTE_KVS' for r in inflight)
    assert full() == small() == remote
    print(remote, len(inflight), *[
        statistics.median(timeit.repeat(fn, number=1000, repeat=3))/1000*1e6
        for fn in (full, small)])
```

这说明计数工作可从扫描全部请求缩小到扫描在途 prefill，**不是 GPU 服务收益，也不是原运行 7.473 秒 controller wall 的归因或可扣除量**。表中是应用前的计数微基准，不包含新代码额外的对象身份检查。远端 Python 3.12、CPU、实际集合大小与请求对象均不同；不能把本地比例乘到旧运行。日志、`begin()` 全请求扫描、逐队首 gate 调用及 queue 操作仍有成本。后续须在新代码版本/新输出目录中让所有比较臂一致使用，并重新测量端到端结果；不得回写旧冻结组或修饰旧计时。
