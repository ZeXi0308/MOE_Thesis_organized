# 首次入场短prompt优先：短请求TTFT改善，整体净收益尚未成立

Verdict：`MEASUREMENT_ONLY`。两个相反顺序block中，短请求TTFT均下降，长请求TTFT与
完成延迟尾部均上升；整体平均完成延迟只改善2.52%与0.22%，没有达到预先3%的继续信号。
这回答了本轮问题：waiting顺序确实可以改变请求结果，但当前简单规则主要呈现代价转移，
还不能证明稳定的完整请求净收益。

## 实际执行与基线

OLMoE-1B-7B-0924、BF16、vLLM0.26.0、单RTX5090，固定cap8、token budget1024。
两个新引擎各4共同warmup、4正式episode；8正式+8预热均COMPLETE，两个OS退出码均0。
每格16篇既有自然文本、50ms到达间隔、128输出；正式共128次请求执行，预热另128次。
mixed输入为128/2048交替；all_short为16×128且排序应为no-op。完整输入及执行源码在归档包内。

同引擎、同cohort比较FCFS与short_prompt_first。双方都用native FCFS deque和相同capture，
仅首次执行前的waiting顺序可以改变。所有KV/batch/生成输出/完成轨迹独立推进。
两引擎均4477个KV block；每格开始/结束记录GPU process isolation，未观察到其它GPU进程。
证据层级是原生in-process GPU执行的`REQUEST_LEVEL`；计时为host receipt，包含决策与采集成本。

## 核心结果

下表为SPT相对同引擎FCFS，负延迟变化表示更低；没有跨引擎平均。

| Mixed指标 | Forward | Reverse |
|---|---:|---:|
| 全体TTFT均值 | −5.12% | −1.51% |
| 短请求TTFT均值 | −14.06% | −10.38% |
| 长请求TTFT均值 | +3.09% | +6.44% |
| 全体平均TPOT | −1.49% | +0.28% |
| 全体request latency均值 | −2.52% | −0.22% |
| 短请求request latency均值 | −4.01% | −1.49% |
| 长请求request latency均值 | −1.05% | +1.02% |
| 长请求request latency p95 | +1.36% | +3.82% |
| 整批wall | −1.28% | +0.05% |

绝对量：forward FCFS/SPT的全体平均request latency为1552.637/1513.450ms，
reverse为1523.218/1519.815ms。wall分别2.576596/2.543555s和2.556812/2.558035s。
完整组别、绝对指标、ITL分位数与所有delta见 [analysis/TABLE.md](analysis/TABLE.md) 与
[analysis/summary.json](analysis/summary.json)。长请求只有每格8条，尾部分位数是描述性统计。

参考SLO仍为TTFT5s/平均TPOT0.2s，所有正式请求均通过，故没有证据支持SLO容量改善。
未按结果调整SLO，也未用全面通过覆盖连续延迟的代价。

## 动作确实执行；收益需要与运行差异区分

两个mixed SPT各3次真实队列重排，首次调度顺序均改变。原到达序号前9条保持，末7条
由`[10,11,12,13,14,15,16]`变为`[11,13,15,10,12,14,16]`。没有抽出running中的partial
prefill或decode；每格2032次既有decode推进检查，跳过、抢占、KV调整均0。

所有FCFS及all_short SPT重排数为0。全短组顺序完全相同，但SPT相对FCFS的request均值
仍为+2.00%/−0.35%，wall为+0.73%/+0.07%。这是no-op条件下的实际运行差异，不能直接
作为mixed噪声上界或扣除项；它提示不能把微小整体变化解释为稳定收益。

相同输入/seed不等于token等价：完整输出序列相等数，forward all_short为6/16、mixed
为5/16，reverse分别7/16和8/16。没有任务质量结论或token等价加速主张。

独立直接读取8个正式raw重算378个值及配对项，与分析输出差异0、最大绝对差异0；
未调用分析器/metrics函数。该一次针对性检查结束，不追加形式审计。

## 回传与复现

所有正式、预热、空输出step、逐token结果、动作日志、配置、环境、源码、命令和
stdout/stderr/退出码均已回传；无失败运行被删除。两份归档各40文件，逐字节验证且JSON可读。

| Block | 字节 | SHA256 |
|---|---:|---|
| forward | 1430785 | `c445f015a8ae99d7cd13f30667a54074023df5951d917b3eb45b76b3cd3914ce` |
| reverse | 1427149 | `d3dd9989f39985bd13a1dc954bc57069fc97d4d414250e3f148a1fe2816d566e` |

本地raw在 `readback/results/{forward,reverse}/`；执行包SHA为
`b7aa0da4e8aa97b3feffdd1af9998588c5e0fab91103ab776b55ef9d1d936fe9`。
远端原件保留于 `/root/autodl-tmp/moe-waiting-order-01a07d4b-20260908-r01`。

```bash
python3 analyze_results.py --results-dir readback/results --output-dir analysis_recomputed
# 已实际执行的GPU命令，使用记录的vLLM解释器：
python -u launch_block.py forward
python -u launch_block.py reverse
```

## 边界与唯一下一步

最强已测基线是固定cap/budget的FCFS；SPT是普通简单策略，不是新方法。Oracle/headroom
未测；未测独立新文档、任务质量、第二模型、HTTP serving、多卡EP或专家信号残余。
失败类别是`weak/unstable full-request benefit with long-request cost transfer`，不是动作
失效或整个queue/admission家族NO-GO。Claim ceiling止于上述单域请求级测量。

唯一下一实验已按冻结规则选择为 [原样受控重复一次](../waiting_order_repeat_r01/DECISIONS.md)。
重复包、输入、seed、顺序及资源都不变，原始结果永久保留；不再扫策略或阈值。
若四个block仍没有完整请求净收益，则停止当前static-SPT方法扩展。未来仅当新的自然
到达/长度域改变等待暴露或动作空间时重新考察，不能仅换seed把本结果包装成新Idea。
