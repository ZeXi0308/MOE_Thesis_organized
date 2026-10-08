# 原生固定 N4：完整四格结果（冻结）

2026-10-02。**固定 4-token ngram 未通过当前合同的净吞吐目标，但改善了最大 host 输出停顿。** N4/AR16 两重复均值：drained 17.8867→19.7996 s（+10.69%）、实际输出 79.166→76.981 token/s（−2.76%）、mean flow +5.78%；最大 host chunk gap 1.2698→0.5968 s（−53.00%）。这是完整成本与尾停顿的取舍，不是 speculative decoding family NO-GO。

合同：45495 上四个独立 engine，AR16/N4/N4/AR16 正反序；cap16、Q2048、engine 编译容量 4096、expert24、GPU KV 1 GiB、CPU KV 1 GiB。原探索输入 source32–47，16 同时到达、自然 greedy、上限 512；每臂按自身算法 full warmup，固定 32 输出/Q512。AR 无 speculation；N4 为原生 prompt lookup min2/max5、固定草稿长度4，未实现层内裁剪。

四格全部 COMPLETE、16/16 stop、无长度截断。generic metrics、隔离资格、activity 均 PASS；request salt 隔离、测前 CPU cache 为空，正 lookup 只涉及实际被抢占请求，最终无 pending 传输。两臂均未安装 row observer，request-row identity 未资格化。

下表时间为秒；capture/served/drained 都保留，rate=真实输出 token/drained 秒；均值取两格指标算术均值。测量成本包括实际执行与 drain，排除 engine 初始化及 warmup，不作减项修正。

| 格/均值 | capture / served / drained | token/s | mean flow | mean TTFT | chunk gap p95 | 最大 chunk gap | 输出 token | 正确 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00_ar16 | 17.9428 / 17.9418 / 17.9452 | 78.907 | 12.5510 | 2.6353 | 0.1584 | 1.2840 | 1416 | 8/16 |
| 01_ngram4 | 20.0181 / 20.0173 / 20.0204 | 76.122 | 13.4064 | 2.7115 | 0.2078 | 0.6006 | 1524 | 8/16 |
| 02_ngram4 | 19.5780 / 19.5778 / 19.5787 | 77.839 | 13.0764 | 2.6469 | 0.2059 | 0.5930 | 1524 | 8/16 |
| 03_ar16 | 17.8272 / 17.8269 / 17.8282 | 79.425 | 12.4854 | 2.6254 | 0.1470 | 1.2557 | 1416 | 8/16 |
| **AR16 均值** | 17.8850 / 17.8844 / 17.8867 | 79.166 | 12.5182 | 2.6304 | 0.1527 | 1.2698 | 1416 | 8/16 |
| **N4 均值** | 19.7980 / 19.7975 / 19.7996 | 76.981 | 13.2414 | 2.6792 | 0.2068 | 0.5968 | 1524 | 8/16 |

配对按第 1 次对第 1 次、第 2 次对第 2 次，均为 N4/AR16 变化；两次 drained/实际吞吐/flow 的劣化方向一致，最大 gap 的改善方向也一致。

| 配对 | drained | token/s | mean flow | 最大 host chunk gap |
|---|---:|---:|---:|---:|
| 01_ngram4/00_ar16 | +11.56% | -3.53% | +6.82% | -53.22% |
| 02_ngram4/03_ar16 | +9.82% | -2.00% | +4.73% | -52.77% |

两臂均答对同 8 题：0032、0033、0034、0035、0036、0040、0042、0045（gsm8k-test 前缀）；无题目正确性翻转。各臂两重复均 16/16 完整输出一致；跨臂仅 6/16 完整输出一致、12/16 最终答案一致。改变的答案为 0037、0039、0043、0046，均仍答错。N4 输出 1524 vs AR1416（+7.63%），故 wall 比是实际 episode 比较，不是等工作量加速，也不证明完整输出等价。

N4 所有 16 请求均有多 token chunk：整格及相应请求的 token ITL 分布、decode-token rate 均为 null；同 chunk 的重复 receipt 时间不能当零 ITL。表中只报告非空 host receipt 间 gap，排除 TTFT。AR token ITL 可解析，均值 0.1130 s、p95 0.1527 s；不能与 N4 未解析 ITL 比较。N4 每请求最大 chunk gap 的 p95 均值 0.4172 vs AR1.2588 s（−66.86%），但 pooled chunk gap mean/p95 分别 +35.34%/+35.43%；发包粒度不同，不能宣称全部延迟改善。

互斥 engine 阶段完整分账（两重复均值；GB=10^9 B）。已调度 decode 位置包含验证候选，不是已接受输出 token；完整 engine calls 均计入，本组实际无 connector-only 空步。

| 臂/阶段 | steps | prefill / decode 调度位置 | engine wall s | 专家 copy GB | groups |
|---|---:|---:|---:|---:|---:|
| AR16/pure_prefill | 1 | 2048 / 0 | 0.233779 | 7.876903 | 48 |
| AR16/mixed | 10 | 9083 / 90 | 2.115363 | 71.772930 | 454 |
| AR16/pure_decode | 168 | 0 / 1314 | 15.522762 | 515.962307 | 4699 |
| N4/pure_prefill | 1 | 2048 / 0 | 0.231167 | 7.876903 | 48 |
| N4/mixed | 9 | 9069 / 160 | 1.912174 | 69.080187 | 427 |
| N4/pure_decode | 127 | 0 / 2526 | 17.643278 | 641.451688 | 4731 |

pure-decode 步数 168→127（−24.40%），但该阶段专家 copy 515.962→641.452 GB、wall 15.5228→17.6433 s；总 engine wall 17.8719→19.7866 s。mixed 节省 0.2032 s/2.6927 GB，未抵消 pure-decode 增量。总 groups 5201→5206，专家 copy 595.612→718.409 GB（+20.62%）；少走 step 没有带来更低的完整成本。

两重复内的全部专家与原生 KV I/O 计数分别完全一致：

| 臂 | 专家 copy（B） | groups | KV load（B） | KV store（B） | 抢占数 |
|---|---:|---:|---:|---:|---:|
| AR16 | 595,612,139,520 | 5201 | 274,726,912 | 1,627,389,952 | 3 |
| N4 | 718,408,777,728 | 5206 | 92,274,688 | 1,644,167,168 | 1 |

四格实际 GPU KV 均 1,073,741,824 B/512 块、CPU pinned KV 同为 1 GiB/512 块，expert scratch 均 4,831,838,208 B。峰值 allocated 为 AR 7,044,517,376 B、N4 7,044,519,424 B（差 2 KiB），peak reserved 均 7,501,512,704 B。预留池未扩大；全部 CPU 保存/加载及末尾 drain 保留在成本中。

AR 两次均在 step49/56/72 抢占 0044/0045/0047；N4 两次仅 step35 抢占 0044。原 prompt 调度 excess 16→2；重复调度位置 20→1158，N4 数值含拒绝候选的验证位置重叠，**不能叫纯 KV 抢占重算量，也不能折算成时间节省**。

N4 每次均有 395 个 request-step proposal、1580 个实际 scheduled draft-token rows；每条草稿长度均为4，ID/count 与同一步 source request 记录匹配。非空 receipt 的 chunk-size 分布每次均为 1:938、2:72、3:47、4:19、5:45；共1121个 receipt、1524个输出 token。`sum(max(chunk_size−1,0))=403` 仅是超过每 receipt 一个 token 的输出量，**不是精确 accepted drafts，不能用 403/1580 宣称接受率**；EOS 与首步边界未提供所需接受归属，exact_acceptance_rate 保持 null。AR 每次1416个单 token receipt、无 scheduled drafts。

draft/request 账本匹配不等于模型行身份资格化。当前只支持“固定 N4 的少步数与更大专家搬运同时发生”；没有逐行身份或层内干预，不能把总 copy 增量归因到特定尾部 token/专家，更不能据此宣称层内 cutoff controller 有收益。

**下一唯一强简单参照：在同类实际完整请求合同下比较更短的固定草稿长度 1/2，先于任何 layer-cutoff controller。** 本轮不启动 holdout；固定4的净吞吐目标失败保留为结果，最大 host gap 改善保留为权衡，不扩写为整个 speculation family 失败。

证据：[完整 metrics](results_ngram_r01/metrics.json)、[活动核对](results_ngram_r01/ngram_activity.json)、[隔离资格](results_ngram_r01/isolation_qualification.json)。归档 tar 16,904,168 B，SHA256 `a34d6b22a759089928ff6ff2d60d3bca9ede1f84bb813d6bdcd12da0181c6254`。报告冻结；未改 state、raw、metrics 或代码。
