# Past-Future 作者 AE 核心：原生单格完成

**128/128 完成并排空；近邻基线核心可执行，尚无同配置性能对照。** 源码事前冻结于 `c3589e2`，运行于2026-10-01 10:17:28.841514–10:19:51.718163 UTC。两次共享锁忙的尝试未启动子进程；第三次自然断点取得锁。子进程正常退出，显存归零，未重跑。

## 完整请求结果

| 指标 | 结果 |
|---|---:|
| 完成 / 未完成 | 128 / 0 |
| 20s TTFT、4s最大观察间隔联合合格 | 64 / 128 |
| 联合 goodput | 0.737599 请求/s |
| 完整到达至排空 episode | 86.767964 s |
| 平均到达至完成 | 36.866928 s |
| 最大观察生成间隔 | 0.499748 s |
| 实际输出 token | 123,036 |
| length / stop | 119 / 9 |
| 原生抢占 / 恢复 / 状态违例 | 0 / 0 / 0 |
| 实际准入 / 完成 | 128 / 128 |
| 调度调用 / 有等待队头的决策 / 拒绝调用 | 6219 / 4926 / 4798 |
| 在线决策 CPU 小计 | 1.722145 s |
| 物理 KV 峰值 | 4082 / 4096 块 |

所有128个请求均满足4s间隔目标，主点失败均与TTFT有关。10/20/30/40s TTFT各有31/64/74/98个合格，所有登记gap目标在本格不改变这些计数。决策开销已经暴露于episode中，不能再重复相加。物理占用超过token容量95%的值并不违反硬分配器界限；作者5%余量是预测的安全余量，不保证真实未来峰值不超预测阈值。`pressure_qualified=false`是沿用旧收据“实际抢占且32常驻”的标签，不能解释为没有KV压力，也不是本次基线资格条件。

## 相同真实状态上的准入判据复算

[CPU脚本](C_NATIVE_PAST_FUTURE_AE_STATE_COMPARE_V1.py)直接使用4926条已记录的resident P/O/M、队头、历史和实际score，调用未改动的已有cap包络函数与作者AE的tuple/peak函数。它不运行另一调度器、不提供另一策略轨迹或收益。

- 保持AE端点、5%余量及相同batch/slot条件，把所有预测输出改为已知cap：119次实际准入仍通过，**9次实际准入不通过**。因此历史条件采样确实改变了本次动作；不能称它等价于cap策略。4798次AE拒绝在该cap判据下也均拒绝。
- 使用旧C“居民cap包络＋新队头完整cap”的整数判据：127次实际准入通过，1次不通过；另有979次AE拒绝在该判据下容量可行，涉及35个不同队头。反复检查同一队头不能当成979个独立机会，更不能推断它们若提前准入就有服务收益。
- 具体的相反接受状态：调用3455，23个居民的包络3907块，新队头完整cap后4104块；AE全cap peak65095 token超过62259.2阈值，历史采样最坏peak62196却通过并实际准入。全部其他决策同样保留。九次预测改变的准入，其AE全cap token峰值均低于65536；不能把越过5%预测余量解释为超出全部已知cap的物理容量机会。分页端点和逐请求取整仍有差别。

这是事后、沿PF轨迹的确定状态比较；没有将已见最终EOS输入在线控制器。历史仅由本格已完成请求更新，每格重新以20个cap值初始化。没有调整seed、reserve、窗口、输入或终止规则。

## 范围与下一步

这是作者AE统计/峰值核心在固定vLLM后端上的移植：history40、参数MINIMUM_SAMPLES=200、MAXIMUM_LISTS=5、reserve0.05。实际采样列表数由作者公式决定，最多5个，不能写成每决策有200个独立完整情景。每次native调度检查一个FCFS队头、完整prefill/recompute joint budget4096；单队头上限沿用作者AE `generate_new_batch` 的 `len(can_run_list)<1`，但整个路由调用时机、执行引擎和原生抢占路径没有作为完整LightLLM系统复现。

本格是明确改变的batch4096域，旧batch1024的FIFO、退休和原生结果均不参与配对。数据为已见文章；大量cap结束及此前跨负载重复输出限制外部有效性，没有语义质量或等工作量加速主张。先完成固定native/PF/PF/native同batch4096对照，再据完整收益与代价判断该近邻基线在本域的表现。公式新颖性已被直接prior覆盖，任何性能结果都不恢复该主张。

## 原件与复算

完整目录：`C_research_artifacts/20261001/c-native-past-future-ae-pilot-v1`，远端展开原件仍保留。分析：[完整指标](native_past_future_ae_analysis_v1.json)、[逐状态对照](native_past_future_ae_state_compare_v1.json)。

本地压缩包 `c-native-past-future-ae-pilot-v1-local-copy.tar.gz`，4,761,839字节，SHA-256 `34e6a60cef3a7fe3984fa308ad33ceaac549036011c733387cae1885982fdced`。raw SHA-256 `51afa0f8a3d7f8453876d7ebb51f402d9738b81cff0f30908fe1895c726c0b23` 与launcher回执相符；log SHA也核对一致。没有上传、发布或推送结果到公共仓库。

### 后续本地补充：调用频率与输出模式

同一作者提交的 [router manager](https://github.com/WuSiYu/lightllm-ae/blob/d93ff69c07d4097b0a6a4ee67ea355e8780e3095/lightllm-server-pastfuture/lightllm/server/router/manager.py) 明确 `max_wait_tokens=2`，运行中达到该计数后尝试新队头，再通过独立prefill合并；本移植每次native调度尝试一次，并共同安排resident decode与新prefill。单队头限制本身两者相同。因而统计核心结果不能当作完整作者路由器的性能；采样调用频率和执行路径的差别须随比较保留。源码已通过官方API取回，21960字节，SHA-256 `ba809997ab6acead0f0ec5e83f6f1856d85c3d4efed96bdbf89e65e76a3a8ee6`，固定在stable `lightllm-primary-source`。当前已冻结运行保持不变。

复用相同token模式诊断函数，本pilot有37/128个请求含≥512 token短周期重复，2个为同一token连续1024次。没有语义有效性结论，也没有过滤这些请求。
