# 一次越过限制执行有效，但未保住稳定的短请求收益

2026-09-11。结论：限制越过减轻了SPT的部分长请求尾部代价，却没有在两个执行顺序中都保留短请求收益。按执行前规则停止当前越过限制的扩展，不扫描次数、年龄、seed或第三个选择器。

## 实际测量

单RTX5090、OLMoE固定revision、BF16、vLLM0.26.0，cap8、token预算1024、memutil0.70，同步in-process、graph启用、无prefix cache。16篇自然文本，mixed为128/2048交替，all_short全部128；每50ms到达，固定128输出。两个fresh engine反转正式策略顺序，分别执行6正式和6共同预热。全部12正式、12预热完成，两个进程退出0；192次正式请求执行、192次预热执行，独立文档仍为16篇。

| Block / 策略 | 短TTFT均值 ms | 长完成p95 ms | 全体完成均值 ms | wall s | 最大ITL ms |
|---|---:|---:|---:|---:|---:|
| forward / fcfs | 375.438 | 1865.729 | 1466.654 | 2.488402 | 21.400 |
| forward / short_prompt_first | 346.836 | 1947.959 | 1476.513 | 2.504024 | 22.945 |
| forward / bounded_bypass_once | 380.396 | 1913.465 | 1501.973 | 2.526995 | 23.530 |
| reverse / fcfs | 393.886 | 1877.381 | 1485.503 | 2.494602 | 21.323 |
| reverse / short_prompt_first | 339.352 | 1947.897 | 1471.645 | 2.500032 | 21.504 |
| reverse / bounded_bypass_once | 365.808 | 1893.764 | 1478.667 | 2.506668 | 27.238 |

只做同一block内比较，负延迟差表示更快：

| 一次限制的指标变化 | forward | reverse |
|---|---:|---:|
| 相对FCFS：短TTFT | +1.32% | -7.13% |
| 相对FCFS：全体完成均值 | +2.41% | -0.46% |
| 相对SPT：短TTFT | +9.68% | +7.80% |
| 相对SPT：长完成p95 | -1.77% | -2.78% |
| 相对SPT：长完成均值 | +0.67% | -0.65% |
| 相对SPT：全体完成均值 | +1.72% | +0.48% |
| 相对SPT：完成墙钟 | +0.92% | +0.27% |

全部正式请求通过原定5s TTFT / 0.2s平均TPOT参考要求。这些阈值没有区分违约风险，不据此声称安全容量提升。全短负控没有重排或越过；一次限制相对FCFS的完成均值仍变化+1.08%/−0.003%，不能把这些读数从mixed差异中扣除。

## 动作改变了谁

原到达顺序编号1–16，两个block的首次入场排列一致：

```text
FCFS：1…9, 10,11,12,13,14,15,16
SPT： 1…9, 11,13,15,10,12,14,16
限制：1…9, 11,10,13,12,15,14,16
```

SPT使第10、12、14条长请求分别被越过3、2、1次，限制后均为1次。每组总越过6→3、最大3→1；FCFS为0。记账来自实际首次调度，队列只重排而未执行不计次，同一步前缀次序也不代表额外毫秒等待。

相对SPT，第10条完成延迟减少41.070/60.915ms，第12条减少22.281/41.536ms；代价包括第13条短请求增加91.987/80.576ms，第15条增加87.289/67.783ms。第14条原来只被越过一次，变化+18.673/−0.643ms，没有稳定的额外保护。

互斥时间分解是：到达→host提交开始→首次scheduler调用开始→首token返回→完成。第11、13、15条在限制策略下相对FCFS的首token都提前，但后续生成跨度增加，forward三条最终都更晚完成。先拿到首token不等于先完成请求。名次未变的请求也有时延差异，现有记录不能分别归因kernel、host抖动、输出轨迹和采集开销。

## 完整数据与一次核对

全部raw、metrics、checks、配置、代码包、环境、命令、日志和退出状态已回传 `readback/results/{forward,reverse}/`，远端原件保留。主表为 `analysis/TABLE.md`，全量派生数据为 `analysis/summary.json`；逐请求分解在 `analysis/REQUEST_ACCOUNTING.md` 和 `analysis/request_accounting.json`。

| block | 归档字节 | 文件数 | SHA256 |
|---|---:|---:|---|
| forward | 2384801 | 57 | a3e23633e2f77f5cbc8a661ea32846887c71c40933275824c85a0166f4fa45e7 |
| reverse | 2403109 | 57 | 397257f465e0aa51be291ac8d4eb4e21a08a7090c7f27db81a6ec8942e2096eb |

fresh GPT-5.6-Sol核对为WARN、same-family/provisional：真实执行和会计通过，P1是正向性能条件未满足，不是伪造或错账。独立重算覆盖24个raw、384次请求、49152个token、7027个step/action及归档字节，见 `EXPERIMENT_AUDIT.md/json`。审计指出的过时README和可变STATUS已更新，原始数据及审计原结论保留。stderr的SM/CUDA提示原样保留；两组实际CUDA13.0、退出0、全部完成。

证据仅覆盖一个模型、单GPU、固定到达轨迹的in-process请求测量。正式臂间输出序列相等数为4–10/16，未评价任务质量、HTTP生产服务、第二模型或EP。旧主机R1 reverse原件缺口仍在，本轮没有替代或补齐那份缺失raw。

## 裁决与下一问题

| 固定报告项 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；停止当前一次越过限制扩展 |
| Evidence type | REQUEST_LEVEL / NATIVE_IN_PROCESS_GPU_INTERVENTION |
| What was measured | 12正式+12预热，真实三臂排序、完整请求时间、越过与全部失败状态 |
| What was not measured | 质量、Oracle、新文档泛化、长期过载、公平时间上界、HTTP或EP |
| Strongest baseline | SPT的短TTFT更低；FCFS避免对应长尾代价，没有全局最优主张 |
| Oracle/headroom status | 未测，三条实际策略不是精确Oracle |
| Claim ceiling | 限制实际越过重新分配等待，未在两个block同时保留所需收益 |
| Failure category | 动作有效，完整请求收益不稳定且存在成本转移 |
| Resurrection condition | 新自然请求/到达域显示不同公平问题，不以换阈值或seed重开 |
| One next smallest experiment | 全局1024不变、原生单请求prefill限512，与native1024/global512三臂对照 |

下一问题来自不同执行路径：两个FCFS mixed block各有3步，7条running仍有空位，却被已有长prefill的1018 tokens加6个decode用尽1024预算，队首128-token短请求没有执行。观察与重算脚本在 `analysis/prefill_sharing_opportunities.json` 和 `inspect_prefill_sharing.py`；这不是反事实或收益上界。新实验保持FCFS，改变单请求在同一步的预算份额，区别于已经停止的等待顺序与总预算扫描；执行前配置在 `../../independent_ideas_20260911/per_request_prefill_share_r01/DECISIONS.md`。

本轮问题已经回答：一次越过约束没有达成预先要求的稳定收益，当前规则到此停止。
