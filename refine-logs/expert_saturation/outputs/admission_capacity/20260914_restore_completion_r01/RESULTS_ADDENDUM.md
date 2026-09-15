# 首输出恢复义务：首次四格结果

Verdict：`MEASUREMENT_ONLY`，首输出义务已真实兑现，但没有稳定完整请求净收益。主问题`OPEN`；不能把当前保护实现的代价扩写为资源调度问题NO-GO。

Evidence type：`NATIVE_SERVING`，单RTX5090/OLMoE-1B-7B BF16/vLLM0.26.0同步in-process，旧d6 32文档、3072输入/1024输出、50ms steady、调度token预算1024、APC off、实际KV13960740864字节/6656可用块。两个反序相关repeat，不是新holdout。

What was measured：相同rank-prefix/FCFS/LTR200/10后端，off仅记录恢复义务；on把已实际开始且已有输出历史的PREEMPTED恢复保留到首新输出可见，期间priority=-2，随后恢复原规则。没有额外显存或延长服务量子。各臂独立执行全部未来状态，128请求/131072输出全部完成，保留完整成本与逐请求损失。

| Cell | Wall s | Requests/s | Mean completion s | Max request ITL s | Calls | Preemptions | Recompute positions |
|---|---:|---:|---:|---:|---:|---:|---:|
| block0-d6-restore-off | 28.008989628 | 1.142490337 | 22.066759620 | 6.669039249 | 1872 | 25 | 74982 |
| block0-d6-restore-on | 30.150935479 | 1.061326937 | 23.898243201 | 4.387084458 | 1906 | 27 | 96955 |
| block1-d6-restore-on | 28.612379804 | 1.118397009 | 22.611804016 | 4.141310960 | 1906 | 27 | 96955 |
| block1-d6-restore-off | 29.273166627 | 1.093151295 | 22.938974577 | 6.886752767 | 1872 | 25 | 74982 |

On/off吞吐−7.104078% / +2.309444%，平均完成+8.299740% / −1.426265%；最长ITL减少2.281955 / 2.745442s。第一对全部32请求完成更慢、第二对全部32更快。两对各有5/32最大ITL改善、27/32恶化；最大恶化发生在请求3259，分别+0.415146 / +0.246207s。不能只报告最大停顿改善而省略其它请求损失。

义务逐步核验：每个off有25个已执行恢复义务，19个以new_output释放、6个被再次抢占打断；每个on有27个，全以new_output释放、无打断，81次实际-2调度。on的全部义务聚合剩余历史均不超过真实free blocks，最大187块；没有抢占尚未兑现的义务。

实际恢复段严格截止到同请求下一次抢占。off 18段再抢占，0/1/2新输出段6/0/0；on 19段，0/1/2段0/4/0。on四段只产1个新token即又抢占：3640首次405→411、3571 520→525、3475 648→654、3345 784→792，各执行3273/3382/3498/3614个重算位置。首输出兑现不等于重算得到足够后续服务，也不能从0段减少直接推出重算减少。on实际上每次多21973重算位置、34个调度call，held resident累积916→1663 request-calls。fresh prefill98304/fresh decode32736均相同。

成本全部保留：wall=scheduler inclusive+engine excluding scheduler+outside engine。四格scheduler秒1.393360/1.650348/1.359028/1.528090；decision秒0.330790/0.413739/0.335236/0.389575嵌套其中，不重复相加。每个on 9个原LTR有效量子、off 8个，均没有量子耗尽仍无新输出；不据此扫描200/10。

两臂内部实际调度/输出路径相同，32/32输出一致，但off repeat wall相差+1.264177s，on−1.538556s。相关重复不是噪声界。跨臂两对26/32完整输出相同、6条轨迹不同，质量未测；不能将输出相同率写成准确率。原件phase marker明确区分应用预热和measurement；没有逐事件JIT归因，不用旧warning_once解释时间波动。

What was not measured：业务SLO-goodput、质量、第二cohort/模型/arrival域、与fit_scan或most/least的同组净收益比较、完整LTR、全动作Oracle、生产P99。

Strongest baseline：本组使用相同rank-prefix后端去掉首输出优先级；既有fit_scan/least/most及native是后续系统主张必须面对的强基线。当前较弱rank-prefix的同底座因果消融不能宣布胜过这些策略。首输出保护也已存在于旧轮转实现及近邻机制，不能包装成新颖性。

Oracle/headroom status：全请求Oracle未运行。已测零输出中断可由本动作消除，但需要计入新调度/重算和其它请求代价；没有局部saving相加的反事实上界。

Claim ceiling：旧固定运行域的首输出完成性与请求级代价。新runner记录真实resolved chunk threshold=0，四格合计7556步通过冻结planner完整重放，实际计划/执行、过去可见output计数、所有义务事件、真实KV守恒也逐步匹配。完整原件未修改，状态为测量结论，不是method GO。

Failure category：完成性成功，服务摊销与损害转移失败；完整wall/平均完成符号翻转，不能声称稳定吞吐改善。造成新增重算的后续事件需要定位，不能把全部21973位置归因于首个改变动作。

Resurrection condition：只有新动作消除已测执行优先级/重算税并通过同组强基线，或新运行域改变这项代价，才进入确认；不靠调长量子或删除不利repeat救原结果。

One next smallest experiment：先从同一个已记录首分叉前态，检查能否保留恢复KV而继续服务原本可执行的resident decode，将KV保留和执行优先级分开；只检查第一不同动作的合法性，不用旧未来trace预测收益。只有该成本来源和可行动作都成立后，才冻结新的真实同底座对照；当前没有追加GPU组。

研究问题的直接回答：保留到首新输出能缩短最坏生成停顿，但这版优先级实现增加了重复工作和多数请求停顿，尚不能稳定保住完整服务量。

证据：[analysis.json](analysis/analysis.json)、[原件](execution/readback)、[执行合同](EXECUTION_CONTRACT_ADDENDUM.md)。输入包SHA256 `c90cef5949deb6a2263d8fffe1a22af12c3bba5a3178d5f3e0929f401fdef2fd`；全量回读SHA256 `bbff5317b8577abdb1f5ba7d651ef21035b9dbca77045285201ae3b110150c5d`。准备包metadata及原REPORT的UNRUN/暂停描述是历史记录，本addendum记录实际执行；原campaign中继承的prefix文案不改变本次CLI开关和执行合同。
