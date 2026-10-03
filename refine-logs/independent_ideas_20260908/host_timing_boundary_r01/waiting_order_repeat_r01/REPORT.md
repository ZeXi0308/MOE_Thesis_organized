# 原样重复：GPU已全部完成，最后一个block回传待恢复

当前结论仍是`MEASUREMENT_ONLY`。已在本地的三个block都显示短请求TTFT降低、长请求
TTFT及完成延迟尾部增加；整体平均完成延迟的改善趋近零。第四个block已在远端确认
COMPLETE/退出0，但SSH在归档前开始于握手阶段断开，尚不能给出完整重复裁决。

## 实际完成和当前缺失

本次原样重复使用与上一轮逐字节相同的执行包、输入、seed、资源、策略和测量顺序；
没有新增参数或策略。两个新引擎各4正式+4预热，远端均COMPLETE且退出0。
forward的4正式+4预热完整回传、SHA256验证、全部JSON可读；reverse的4正式+4预热
及相应环境、日志、配置等归档仍未回传。不能把“远端完成”写成“全部本地到齐”。

最后成功的远端poll确认reverse进程28113结束、退出码0，4个正式与4个预热raw文件存在、
GPU进程列表为空；观察子集见 `LAST_SUCCESSFUL_REMOTE_OBSERVATION.json`。此文件只是
工具观察记录，不替代尚缺的raw。原始文件没有删除，恢复连接后优先补归档和回传。

连接诊断：TCP能连到原授权端口，远端在发送SSH server banner之前主动关闭连接，
报`kex_exchange_identification: Connection closed by remote host`。这发生在密码验证之前，
不能解释成凭据错误。已询问用户实例是否仍运行、SSH地址/端口是否变化；没有要求重授权限。
传输失败与误触发的空metadata验证已记录在 `TRANSFER-RETRY-001.json`，没有新GPU失败运行。

## 已在本地的三个block，全部列出

SPT相对同block FCFS，负延迟变化表示更低。R0来自[原始对照](../waiting_order_r01/REPORT.md)，
R1为本次完全相同条件的重复；不挑最好一次、不跨block平均。

| Mixed指标 | R0 forward | R0 reverse | R1 forward | R1 reverse |
|---|---:|---:|---:|---|
| 短请求TTFT均值 | −14.06% | −10.38% | −9.45% | raw未回传 |
| 长请求TTFT均值 | +3.09% | +6.44% | +5.19% | raw未回传 |
| 全体平均TPOT | −1.49% | +0.28% | +0.68% | raw未回传 |
| 全体request latency均值 | −2.52% | −0.22% | −0.03% | raw未回传 |
| 长请求request latency均值 | −1.05% | +1.02% | +1.09% | raw未回传 |
| 长请求request latency p95 | +1.36% | +3.82% | +3.57% | raw未回传 |
| 整批wall | −1.28% | +0.05% | +0.46% | raw未回传 |

R1 forward全短no-op负控：平均request latency+0.45%，wall+0.66%，没有队列重排。
该组结果不能当作mixed可直接扣除的噪声。R1 forward的mixed全体request p95+1.79%，
max+3.74%；完整输出序列相等数all_short7/16、mixed6/16，没有token或质量保持主张。

R1 forward派生结果在 `analysis_partial_before_reverse_readback/`。其摘要程序的
`PARTIAL_OR_INVALID_MEASUREMENT`在这里仅表示缺reverse block，**不是实验INVALID**；
现有8个正式/预热raw均通过检查。完整回传后用原分析器生成新`analysis/`，保留该早期输出。

## 会计、基线和证据范围

最强已测基线是相同cap8、budget1024和采集开销口径下的FCFS。SPT只重排native队列
中从未开始、已到达请求，running中的decode与partial prefill不动；生成/KV/batch独立推进。
Oracle/headroom未测；没有专家特征或新增算法。参考SLO5s/0.2s保持不变，已回传请求均通过。

证据层级：单RTX5090、OLMoE、vLLM0.26.0、BF16、16篇复用自然文本的in-process
`REQUEST_LEVEL`。未测独立新文档、质量、token conformance、HTTP生产服务、第二模型或EP。
当前失败类别是微弱且不稳定的整体收益与长请求代价转移；回传故障是独立的基础设施问题。
没有依据把整个排队/接纳问题族判死，也不能把前一次有利百分比挑作canonical。

## 回传位置与唯一下一步

R1 forward归档1422696字节、41文件，SHA256：
`91cde331056204b31443d168e7055cda048e14754cf3def40624d931c1ead615`。
raw位于 `readback/results/forward/`。远端为
`/root/autodl-tmp/moe-waiting-order-repeat-01a07d4b-20260908-r01`。

唯一下一步是恢复原授权主机连接，归档并回传已经完成的reverse block，核验后并列四组结果。
不重跑已经完成的数据，不启动第三轮重复，不根据缺失结果调整参数。只有完整证据闭合后
再选择一个新的最小科学问题；当前不为下个Controller写准备代码。

```bash
# 补齐reverse readback后，在本目录运行；输出目录必须全新：
python3 ../waiting_order_r01/analyze_results.py --results-dir readback/results --output-dir analysis
```
