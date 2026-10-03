# 同引擎档位对照与固定重复：没有稳定超过静态策略的收益

2026-09-08。分支 `agent/publish-current-moe-code`，HEAD `2a37765fe522b1d74609a686f1d327ede7619a50`；继承未提交的准入实验实现与输出，未提交或推送本轮内容。

**研究问题已回答：在当前短文本运行域，把普通四步 ITL 反馈档位从 `[8,12,16,32]` 换成 `[8,16,24,32]`，没有可重复的完整请求收益超过同引擎最好已测静态点。** Steady 四个引擎只有一次同时胜过旧反馈和静态点，固定重复两次均负；bursty 四次均改善旧反馈，但四次均未超过静态点。本次停止这个运行域中的档位替换方法，保留测量结果，不否定准入或专家感知调度家族。

本报告合并 [原配对实验](../20260908_capture_ladder_paired_r01/EXECUTION_ADDENDUM.md) 与本目录唯一固定重复。原配对 forward 始终是预先指定的 canonical，其他引擎均为保留的比较证据；没有用最好的一次替换它。原 24-episode aligned-only 实验缺同引擎旧档位对照，不并入本表。

## 实际执行与比较口径

读取的权威入口为 `docs/current/README.md`、`docs/ideas/README.md` 和准入实验 README；本轮继承的事实是尚无已验证的专家感知方法收益，历史 padding 解释不一致。唯一最弱链路是档位成本能否通过真实接纳动作进入请求收益。执行前规则见 [配对冻结方案](../20260908_capture_ladder_paired_r01/DECISIONS.md) 与 [一次重复规则](REPEAT_DECISIONS.md)。

四个新引擎各执行 16 个测量 episode：static8/12/16/24/32、shadow32、legacy-feedback32、aligned-feedback32，各含 steady/bursty；每个 campaign 的第二引擎完整反转测量顺序。每引擎另有共同的 15 个 warmup。两种反馈各自推进 KV、生成输出、batch、queue 与 completion；cap 只限制新接纳，已有 decode 继续执行。

合计 **64 个测量 episode、2048 次完整请求执行，以及单独留存的 60 个 warmup episode、1920 次 warmup 请求执行**。测量仍重复同一 32 条 WikiText 文本，不是 2048 篇独立文档。主输入 128 tokens、固定输出 128 tokens；steady 每 50ms 到达一条，bursty 为四组各八条。主 SLO 为 TTFT ≤200ms 且请求平均 TPOT ≤9ms，平均 TPOT 不代表逐 token 间隔保证。

运行栈为 BF16 OLMoE（revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`）、单 RTX 5090、vLLM 0.26.0、PyTorch 2.11.0+cu130、CUDA 13.0、Transformers 5.15.1。共同 engine32、max_model_len256、token budget1024、显存预算0.70、FCFS、关闭 prefix cache、同步 in-process 执行与 stream_interval1。四引擎实际源码、引擎配置和框架/后端版本一致，见 [环境对比](environment_comparison.json)。

`goodput = 同时通过两项 SLO 的完整请求数 / (episode 终点 − 原点)`。同一 host 时钟分母包含排队、prefill、decode、采集、反馈及循环成本；局部时间不再重复加算。全部 32 条计划请求都进入每个 episode 的结果。最强基线是同引擎五个已测 static 的探索性 hindsight 上包络，不能称为在线选择器或动态 Oracle。

## 全部八组结果

下表 goodput 单位为 request/s；括号是联合 SLO 达标请求数，总数均为 32。百分比逐引擎计算，没有跨引擎平均来隐藏变号。

| Campaign / 顺序 | 域 | 旧反馈 goodput（达标数） | 新档位 goodput（达标数） | 最好 static goodput / cap | 新档位 vs 旧反馈 | 新档位 vs static |
|---|---|---:|---:|---:|---:|---:|
| original / forward | steady | 2.4588（9） | 1.3664（4） | 2.7779 / 16 | −44.43% | −50.81% |
| original / reverse | steady | 2.1829（8） | 4.7748（14） | 3.0764 / 24 | +118.74% | +55.21% |
| repeat / forward | steady | 2.4428（9） | 1.9655（7） | 4.9006 / 16 | −19.54% | −59.89% |
| repeat / reverse | steady | 4.4466（16） | 1.9259（7） | 3.1431 / 12 | −56.69% | −38.73% |
| original / forward | bursty | 7.7964（24） | 12.0550（32） | 12.6401 / 32 | +54.62% | −4.63% |
| original / reverse | bursty | 7.8094（24） | 12.6030（32） | 12.6169 / 24 | +61.38% | −0.11% |
| repeat / forward | bursty | 7.7179（24） | 11.1056（31） | 12.5627 / 32 | +43.89% | −11.60% |
| repeat / reverse | bursty | 7.7972（24） | 12.1463（32） | 12.5682 / 32 | +55.78% | −3.36% |

原始精度、所有静态点、TTFT/TPOT/ITL 和有限样本尾部指标见 [本次逐 cell 分析](analysis/analysis.json)、[原配对逐 cell 分析](../20260908_capture_ladder_paired_r01/analysis/analysis.json) 和 [八组汇总](analysis/combined.json)。Steady 原配对两次新档位时长近似相同（2.927/2.932s），达标数却为 4/14，说明不能只看吞吐百分比而忽略 SLO 门槛和排队/生成时间的权衡。Bursty 的旧规则四次均只有 24 条联合达标，新档位改善为 31–32 条，但最好静态点已覆盖这项收益。

## 已有轨迹定位：调度先分叉，随后才降档

固定重复完成后，只读比较四条 aligned steady 轨迹。以原 forward 为 reference，按同一个已记录 scheduler step 序号逐项比较 `(request_id, computed_before, prefill_tokens, decode_tokens)`；三组首次差异都涉及执行内容，不只是行顺序。

| 比较轨迹 | 首个不同 step | reference / 比较轨迹的 scheduler 开始时间 | 首次应用 cap 动作 step（reference / 比较） |
|---|---:|---:|---:|
| original reverse | 32 | 295.331 / 303.880ms | 89 / 71 |
| repeat forward | 2 | 52.881 / 47.830ms | 89 / 68 |
| repeat reverse | 2 | 52.881 / 31.052ms | 89 / 99 |

Step32 的一侧尚未跨过 300ms 到达点，只 decode 六条；另一侧已加入新请求的 128-token prefill。Step2 的 reference 已跨过 50ms 到达点，另外两条尚未到达。所有这些分叉处 target cap 都为 32，早于双方第一次反馈动作。因此，**记录中的 batch/prefill 差异在降档之前已经出现**；不能把不同重复的最终结果全部归因于后续档位动作。这不是 kernel、padding 或专家机制的来源定位，也不证明后续 cap 动作没有作用。

四条轨迹第一次降档分别发生于 0.827/0.676/0.598/0.833s，当时活跃请求为 17/14/12/17、等待均为 0，目标均由 32 降至 24。首次观察到 cap 限制新接纳的机会则在 0.906/0.806/0.857/0.903s，当时 cap 已变为 8/16/16/16，实际活跃为 18/16/17/18。这区分了“写入目标”和“限制接纳”，没有把 active 高于目标误判为暂停已有请求。四个首次动作的历史窗口都含一个 128-token prefill；此事实只用于描述输入窗口，不能单独解释阈值触发或方法效果。

完整窗口、请求/计算位置、首次 binding 与全部三组比较保存在 [first_branch.json](analysis/first_branch.json)。没有证明两条轨迹具有相同 KV/hidden/pre-action state，也没有离线改写策略输出。这个定位已关闭最小可观测差异问题，不再沿相同短文本配置追加第三 campaign。

## 数据留存与检查

64/64 测量 episode 的原始请求指标重算、输入身份、配置和完成性检查通过；两个 campaign 合计检查 260096 次既有 decode 推进，反馈 future cutoff 与所有 step/receipt 对齐通过。所有测量和 warmup raw 均可读取。这里的 targeted checks 只支持实验语义与会计，不替代科学收益。

每个独立引擎结束后先完整回传并验证，再开始下一引擎；四个引擎均以 0 退出。两目录的 `gpu_results/` 保存请求/step 原始数据、warmup raw、配置、环境、日志和退出记录；[本次 TRANSFER](TRANSFER.json) 与 [原配对 TRANSFER](../20260908_capture_ladder_paired_r01/TRANSFER.json) 记录远端/本地一致的归档 SHA256。远端原件保留。GPU 进程检查覆盖 cell 边界，不代表全程连续隔离；最后一次读取时 GPU 已空闲。

执行源码的权威是 [原执行包](../20260908_capture_ladder_paired_r01/execution.tar.gz)，SHA256 `8fb235b3502c69d884df13c8ca5127481c7cfba4b516726286ef1a0043defdf6`。全部 GPU 运行结束后，另做了一处未来使用的修复：`run_native_capacity.py` 对所有模式保存完整 warmup raw，不再仅限配对模式；差异保存在 [future_warmup_retention.patch](future_warmup_retention.patch)，语法编译通过。该修复没有用于本表运行，不冒充已执行源码，也不能补回更早 aligned-only campaign 缺失的 warmup。

从仓库根目录可用 `analyze_paired.py --results-dir <gpu_results> --plans-dir <原配对/plans> --output-dir <新分析目录>` 重算每个 campaign；`analyze_diagnostics.py --results-dir <gpu_results> --analysis-dir <新分析目录>` 重算语义与暴露检查。两脚本位于原配对目录；本目录 [analyze_combined.py](analyze_combined.py) 复算八组汇总和首次调度分叉，拒绝覆盖不同结果。原 raw、原准备报告与原冻结规则未改写。

## 结论边界与唯一下一实验

| 结束字段 | 结论 |
|---|---|
| Verdict | `MEASUREMENT_ONLY`；当前短文本档位替换为有条件停止，稳定方法收益未成立 |
| Evidence type | `NATIVE_SERVING` 中同步 in-process 的请求测量；单模型、单 GPU、有限重复 cohort |
| What was measured | 两条普通反馈、五个静态 cap、shadow、实际接纳/排队、完整 TTFT/平均 TPOT/goodput、一次固定重复与已有轨迹分叉 |
| What was not measured | 专家信号增量、专家/HBM瓶颈、任务质量、自然 EOS、第二模型、EP、生产服务或同 pre-action state 的因果分叉 |
| Strongest baseline | 同引擎 static8/12/16/24/32 最好点；未声称复现完整 nearest prior-art 算法 |
| Oracle/headroom | 没有动态 Oracle；新档位相对静态的稳定 residual 未观察到，不能据此说所有动作 headroom 为零 |
| Claim ceiling | 当前配置下 ordinary-feedback 工程差异及请求级稳定性边界，没有 MoE 专属方法贡献 |
| Failure category | steady 效果不稳定，bursty 改善被简单静态基线覆盖；不属于未运行或无效实验 |
| Resurrection condition | 新自然运行域产生实际容量约束，或新的可执行动作改变暴露成本；换 seed/阈值/文本挑收益不构成复活 |
| One next smallest experiment | 执行既有自然长上下文 cap16/32 对照，先回答真实 KV 容量约束是否进入请求结果 |

唯一下一项复用 [自然长上下文冻结方案](../20260906_native_memory_pressure_r01/DECISIONS.md)：32 篇真实文章的同文 128/3072-token 输入、1024-token 固定输出，cap16/32，短/长与正反顺序共八个独立进程 episode。它增加持续 KV 占用和 chunked-prefill 竞争，主问题是运行域资格，不能继承短文本 9ms 阈值筛选收益。开跑前只需核实既有包的完整 warmup 留存及逐 cell 回传，所有边界退出都保留。若全部容纳且 cap 不构成容量约束，记录运行域边界后停止；若首次抢占保护触发，记录 `CAPACITY_BOUNDARY_STOP`，不将截断完成率当成完整吞吐，再据真实约束判断最简单 KV 预算准入是否需要比较。

本轮直接结论：**当前档位替换不能作为稳定方法收益；下一步转向尚未测过的自然容量运行域，不再抢救这组短文本反馈参数。** 长上下文 GPU 状态仍为 UNRUN，研究长期目标继续。
