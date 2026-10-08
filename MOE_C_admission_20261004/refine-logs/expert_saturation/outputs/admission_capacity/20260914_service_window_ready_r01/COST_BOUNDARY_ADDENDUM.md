# 原生恢复的host缓冲与请求结束边界

本轮新增问题仅为下一次完整成本对照准备：GPU现场6954/6967仍为原Qwen r04整组进程；没有追加GPU实验、修改原始运行或重复主表分析。

## 当前native实现预分配CPU KV缓冲

通过已授权SSH只读核验当前安装vLLM0.26源文件（未import vLLM或初始化GPU）：

- `v1/kv_offload/cpu/spec.py:88–105`：按`cpu_bytes_to_use // aligned_kv_bytes_per_chunk`计算CPU块数；`:135–150`以这个块数创建并缓存一个CPUOffloadingWorker。
- `v1/kv_offload/cpu/gpu_worker.py:484–512`：普通CPU路径为每个canonical KV tensor调用`torch.zeros((num_cpu_blocks, cpu_page_size_bytes), device='cpu', pin_memory=PIN_MEMORY)`，不是随实际已保存token逐块增大tensor。
- `gpu_worker.py:514–529`：store与load handler共享同一份`cpu_tensors`，不能把两个方向相加成两份host KV。
- 指针/大小数组、pending transfers、缓存索引及Python对象另有成本；KV缓冲容量不是总进程预算。是否实际pinned、RSS及峰值需要运行时读取，不能仅凭源码确认。

已完成单事件两臂的`connector.json`均实际记录`cpu_bytes_to_use=17179869184.0`、`offload_prompt_only=False`。在当前单worker、每块2MiB、chunk1且无额外alignment padding的布局下，当前源码预期创建8192块、16GiB CPU KV缓冲；这是源代码与配置推导，旧产物没有CPU tensor storage实测，CPU helper文件的旧运行哈希也未封存，不能写成旧运行host峰值已验证。

因此重复保存模型的12.50–12.97GB逻辑有效host占用，只描述缓存内容/槽位压力；不能据此声称比16GiB少用了对应的物理host资源。完整host对照应复用现有`20260914_streaming_recovery_r01/preparation/pkg/host_budget_observed.py`观测父cgroup与进程树，避免再写监控器。它已有实际使用记录，但约1Hz RSS采样不是连续峰值，RSS求和可能重复共享页，父cgroup峰值不是本实验独享峰值；也不会创建独立硬预算。现有90GiB父cgroup上限与16GiB native KV配置须分别报告。

本次核验源码SHA256：`cpu/spec.py`为`da49babe828d180362a2688b78d08dc306962238df99b8f768949dfdf3cc5862`；`cpu/gpu_worker.py`为`a3957e8f1868478a58bc2e4d54c707baef4e6b3f7383dae0ec868cf7d3a3ec98`。未修改原冻结运行包。

## 请求返回与资源清空是两个事件

原单事件`native_capture.py:198`在capture返回前取`observation_end_s`；`run_probe.py:174–177`之后才调用drain，`:193`的请求指标仍用原observation_end。已有数据如下，均为host返回口径：

| 边界 | save-off | save-on |
|---|---:|---:|
| 最后请求返回/末调用返回，秒 | 30.376865 | 34.190107 |
| observation_end，秒 | 30.379790 | 34.195823 |
| post-request drain调用数 | 0 | 0 |
| drain函数耗时，微秒 | 25.181 | 18.846 |

save-on最后load完成通知发生在末次调度开始前约7.87秒。此组没有已观测的尾部搬运；上述微秒只是pending-push检查耗时，不能称作实际尾部传输成本或资源清空时间上界。`observation_end + drain.seconds`也不是完整资源清空时刻：缺少对齐的drain绝对起止、设备整体状态和host缓存释放观测。原请求完成结果保持有效，没有因这个测量边界新增性能结论。

下一次性能对照分别记录最后新输出、请求观察结束和native剩余任务排空；保留真实重叠，不把整段恢复/传输时间强行相加。服务效率可以报告持续服务吞吐与排空成本，不能将启动/终止成本无说明地移入或移出请求makespan。新轻量入口仍须由runner负责资源边界，单函数返回不表示native缓存或全部设备资源已释放。
