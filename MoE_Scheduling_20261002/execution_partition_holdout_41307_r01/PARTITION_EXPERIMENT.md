# 等总字节的专家/KV静态划分：最小实验

检验减少专家驻留换取 KV 容量，能否在完整自然结束服务中补偿增加的专家缺页成本。本组只识别边际取舍；静态划分不作为创新，也不预设动态交换有收益。没有价值信号就不开发活页 allocator。

`partition_src` 复制无跨层压缩的 `suffix_src`，仅修改运行入口的容量域、强制 AR/suffix off 和实际初始化字节断言；其余执行代码不变，不修旧 vendor resize。

| 臂 | 每层专家 slot | 专家 scratch（B） | GPU KV（B） | 预计 GPU KV 块 | 两池总量（B） |
|---|---:|---:|---:|---:|---:|
| expert24 | 24 | 4,831,838,208 | 1,073,741,824 | 512 | 5,905,580,032 |
| expert23 | 23 | 4,630,511,616 | 1,275,068,416 | 608 | 5,905,580,032 |
| expert20 | 20 | 4,026,531,840 | 1,879,048,192 | 896 | 5,905,580,032 |

每层每专家 12 MiB，共 16 层；少一个统一 slot 释放 192 MiB。GPU KV 块预计为 2 MiB；实际块数保留在资源记录，以实际初始化为准。入口对 scratch 的底层 CUDA storage 去重，和 `resource_snapshot.gpu_kv_unique_bytes` 相加，必须恰好为 **5.5 GiB / 5,905,580,032 B**；同时核对实际 GPU KV 等于该臂请求字节，失败则中止该格并保留 receipt。`partition_resources.json` 记录这些数值。此等量约束仅指两池，CUDA allocator rounding、其他 tensor 与峰值 allocated/reserved 均保留，不声称总进程显存逐字节相等。

全部使用 CPU KV 1 GiB、cap16、Q2048（engine4096）、原生 AR greedy、16 请求同时到达、自然 EOS/max512、相同 source32–47 输入。共同 warmup 为既有训练输入、Q512、固定32输出；每臂在自己的容量上预热，drained warmup 后重置 CPU KV/prefix 状态。模型、CPU KV 恢复、原生 kernel、expert 分组、计时边界均沿用前组。

固定顺序 `24 / 23 / 20 / 20 / 23 / 24`，六个独立 engine；不按最快结果挑选配对。`launch_partition.py` 持有与旧组相同的整组 flock，支持 `--wait-for-lock` 排队，仅在取得锁并通过 GPU 身份/占用检查后开始；不在别组格间插入。运行入口不提供推测或 suffix 动作。

```sh
python launch_partition.py --static-cap 16 --wait-for-lock --output-name results_partition_r01
```

分析直接复用 `analyze_kv_swap.analyze_kv` 或其现有双格 CLI；**不要沿用 `analyze_kv_group` 的 GPU KV 固定1GiB/块数相等判据**。按各臂初始化的实际两池总量核对资源。保留所有六格状态、全部请求与失败，先逐格再给两次均值及对应配对：23首/24首、23次/24次，20同理；结果不能丢掉慢点。

主结果为 drained 完整时间、实际返回 token/drained、flow、TTFT、host 输出间隔与质量/完整序列差异；所有 prefill/decode/mixed、重算、native load/store 与最终 drain 留在完整成本内。机制只报告实际专家 copy/分组、KV 抢占与恢复、实际重复调度位置，不以局部字节或少抢占代替净收益，不叠加 CUDA 与 engine 时间。报告 allocated/reserved 峰值差异，不能用等两池容量隐藏它们。

本组为同一开发输入两次描述性重复，硬件仍可全驻留 OLMoE、人为限制专家池。只有收益超过本组波动、完整成本与质量仍有实用价值，才考虑是否存在值得实现的在线交换动作；静态结果本身不能证明动态控制或论文贡献。
