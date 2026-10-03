# Q10 保护范围的原生定位（2026-10-01）

**结论：存在实际触发、超出目标增长保留所需的队列排除。** 保持原 Q10 策略，仅记录等待循环的实际 break、peer hold 和调度计数；同已见 H128 的 128/128 请求全部完成，单格控制器 168.058 秒，exit0，归档 VERIFIED，退出 GPU EMPTY。稀疏记录未启用旧的全量调度或分配快照。这是源头定位，不与旧性能格排名。

- 195 个实际延长episode、1,755 个延长保护调度步全部实际命中非目标 waiting-loop break，涉及 97 个队首请求；所有被挡队首都为 PREEMPTED。
- 其中 174 个步、24 个延长episode、24 个队首请求、22 个受保护请求满足数值余量条件：原生在途预留 R=0、队首现有 GPU 块为0、剩余 token budget>0、空闲块足以容纳队首完整历史并留出当前目标未来增长。
- 这174步的队首 transfer job、连接器全部注册 job、skipped queue 均为0，pending push=False；数值余量为5–354块。
- 另外123个延长步发生至少一次 peer hold，共1,029个请求步计数。重复请求步不能当独立请求或相加为暴露延迟。

首个数值可容纳反例：队首 `0008645`，历史2,958 token，需要185块，当前held0；空闲235块，受保护目标未来增长0块，R=0，剩余token预算999。原有保护分支仍执行break。它证明这次排除确实由保护分支执行，并非所有排除都可由瞬时块不足解释；它不证明移除分支后一定完成原生准入，也不量化单动作收益。

原始记录：`moe-a-protection-scope-diag-session-r01-20261001/cell-00-protection_scope_q10_diagnostic/archive/` 的 `selective-store.json`（`protection_scope_steps`）与 `raw.json`。冻结计划 `PROTECTION_SCOPE_DIAG_PLAN_R01_20261001.json`；执行包 `candidate_protection_scope_diag_r01`。

补充复算：所有1,755步都有其它请求获调度，合计40,729个peer调度token；它不同于主机实际输出计数。174个有余量break之后都观察到队首新输出，中位等待0.118s，但输出前均无adapter接纳事件，普通native准入的精确时刻未记录，不能补造。完整结果见 `A_PROTECTION_SCOPE_DIAG_RESULT_R01_20261001.json`，复算脚本 `analyze_protection_scope_diag_r01.py`。

## 唯一下一项

在同一实际等待循环边界，若队首满足上述已知状态、所有权、完整历史资金和sequence slot条件，提前结束延长保护，让原生调度尝试该队首；未知或不足则保留原Q10行为。提前结束后不承诺10个输出。必须记录队首实际接纳/首新输出和原目标直到最终完成的代价，零接纳不宣称效用。

拟运行一组 Q1 / 带条件释放的Q10 / 普通Q10，三格同包、独立同源缓存副本。Q1为较强简单参照，普通Q10为消融；沿用rate≥97%、meanflow≤105%、maxgap下降的探索判据（相对Q1），保留全部请求/工作量差异。这是有自然状态依据的简单动作延伸，未建立独立新意或稳定收益；不扫描q、不改seed、不把开发数据当确认。
