# 恢复期间为就绪工作预留 token：六格实测补记

Verdict：`MEASUREMENT_ONLY / OPEN`。修正减少了当前首输出保护实现的计算独占损失，但相对同组 fit-scan 仍是暂停—服务量权衡，没有方法胜出。原准备合同与包内 UNRUN 是历史状态，本补记及六格终态记录当前执行结果。

Evidence type：`NATIVE_SERVING`，单5090 / OLMoE BF16 / vLLM0.26.0；旧d6自然文档，32×3072输入/1024输出，50ms steady，cap32、token budget1024、APC off、KV13,960,740,864 bytes / 6656可用16-token块。三臂各独立状态演进、反序两次。

What was measured：六格192请求全部完成、196608输出，失败0；同组 fit_scan（原200/10）、guard_all（rank-prefix首输出保护）、guard_residual（同保护，预留resident pending1 token）。全部实际plan、义务、资源、ID/时间/成本检查通过。

| cell | wall s | requests/s | mean completion s | max request ITL s | calls | repeated positions | preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|
| block0-fit_scan | 27.902426 | 1.146854 | 21.781419 | 4.919893 | 1862 | 75742 | 22 |
| block0-guard_all | 29.163272 | 1.097271 | 23.030224 | 4.292699 | 1906 | 96955 | 27 |
| block0-guard_residual | 27.947377 | 1.145009 | 21.945337 | 4.338545 | 1858 | 89448 | 25 |
| block1-guard_residual | 28.354072 | 1.128586 | 22.122407 | 4.355581 | 1858 | 89448 | 25 |
| block1-guard_all | 28.900181 | 1.107259 | 22.930755 | 4.157151 | 1906 | 96955 | 27 |
| block1-fit_scan | 27.601698 | 1.159349 | 21.558972 | 4.823253 | 1862 | 75742 | 22 |

| comparison | throughput change | mean completion change | max ITL delta s | per-request max ITL better/worse | equal output sequences |
|---|---:|---:|---:|---|---|
| block0_residual_vs_guard_all | +4.351% | -4.711% | +0.045846 | 28/4 | 27/32 |
| block0_residual_vs_fit_scan | -0.161% | +0.753% | -0.581348 | 26/6 | 28/32 |
| block1_residual_vs_guard_all | +1.926% | -3.525% | +0.198430 | 28/4 | 27/32 |
| block1_residual_vs_fit_scan | -2.653% | +2.613% | -0.467672 | 1/31 | 28/32 |

guard_residual 相对 guard_all 两次全部32请求完成更早，但整组maxITL增加0.046/0.198s；相对fit两次全部32请求完成更晚。第一对26/32的maxITL改善，第二对仅1/32改善，全部损害保留。TTFT和mean-TPOT逐请求及分位数在analysis.json，不将动作发生前的时差归因于后续动作。

每个residual实际54步改变预算；79个受保护selected request-call中，1999个后续就绪候选有1998个被选择。该数是各自实际轨迹上的机会/调用数，不能与guard_all相减冒充固定请求净增量。两个residual各25义务均到首新输出解除、零中断；79 selected call发生于78 step，不能混为一个分母。每次仍有5段只产1个新token就再抢占，恢复完成并未保证成本摊销。fit的0/1/2新输出后重抢占段为2/8/3；guard_all为0/4/0，residual为0/5/0。

相对guard_all，residual重算96955→89448、调用1906→1858、held resident-call1663→186；相对fit仍多13706重算位置，但少4调用。首输出保护和同step就绪服务可以并存，现实现的资源保留不必要求独占1024计算token。

完整wall包括全部调度、引擎、观察及调用间成本。六格scheduler inclusive为1.416871, 1.437074, 1.382453, 1.427551, 1.433305, 1.382187秒；decision嵌套在其中，不能另加。各臂repeat调度/输出路径一致，fit wall变化−0.301s、guard_all−0.263s、residual+0.407s；这不是噪声上界或统计非劣证明。跨臂输出不同的请求已逐一保留，输出相同不等于质量通过。

固定成本模型检查：沿实际step299之后执行序列应用旧native三项系数，不重拟合。对guard_all的条件最后完成变化估计−0.350s，实际−1.128/−0.281s；对fit估计+0.143s，实际+0.079/+0.628s。当前四个比较的方向吻合，但模型误差可大于动作差；其输入含实际未来schedule，不能当在线动作预测或Oracle，也不能把拟合常数当不可消除物理下界。详见analysis/cost_transfer/analysis.json。

实际定位：[FINDINGS](analysis/localization/FINDINGS.md)。guard/residual首次分叉406，同11,295返回token前缀和可见元数据，未声称KV张量相同；994恢复+30就绪工作全部执行并各返回新token。对fit先在406改变执行顺序，直到521才改变token/victim集合，不能把521称为共同前态。最长gap是3475的435→436，854–1047共194步未服务，97.42%在首次恢复前；1048–1051恢复连续完成。54次预算扣留事件1383候选发生数中1382实际服务，唯一未服务者后来被选为victim；全部事件1024预算用满，未发现预留导致空转。

What was not measured：新文档/到达过程、同组native/least/most、完整LTR、自由EOS与质量、业务SLO/生产P99、多模型/多GPU均未在本组六格验证。该组只修正一种集成后的执行成本，不构成新颖性。

Strongest baseline：同组fit_scan；guard_all是有无预算修正的直接消融。共享旧d6 most_output结果提供下一比较动机，但未与本组六格配对，不能据此直接排名或宣布胜出。

Oracle/headroom status：只有实际调度与同前态flag-off单步对照；没有经过执行验证的全动作性能上界。

Claim ceiling：旧cohort上的恢复完成性及预算分配代价。可以确认修正减少多数请求遭遇的恢复独占损失；不能确认暂停和完整服务量同时超过强简单策略。

Failure category：当前实现仍有恢复前等待和短服务再抢占；无效实验/未执行不在此处冒充NO-GO。只关闭“首输出保护+就绪token预留即可稳定胜过fit”的当前主张，主问题保持OPEN。

Resurrection condition：在独立文档、同资源同组强基线下出现可重复的暂停—完整服务量增量，并能定位到真实动作；不通过增大保护量子或事后选SLO挽救。

One next smallest experiment：已准备但GPU未运行的cohort3（排除前128文档）上，以原参数执行native/most_output/fit_scan/guard_residual及反序八格，判断普通强基线是否已覆盖当前权衡；不实现新Controller。

研究问题的本轮答案：为就绪工作保留token能降低恢复保护的执行代价，仍不足以同时保住fit的服务量并消除最长暂停。

执行与复算：[execution.json](execution/execution.json)、[analysis.json](analysis/analysis.json)、[分析器](../../../experiments/admission_capacity/analyze_restore_token_reservation.py)。包SHA f2e20f73711fd5e54906309b90fa7cea845264f17229d1db0a6138db0cac7450；metadata ed37ef0b7fbdc6fdc3fc15024ff4555adf4e176c28667b2431c7362929d8fcdc；回读SHA ea8b18f275d547ee97ee4014df8304d505246dc86edfbd5470eb9ea7ea72d491。共同锁覆盖整组，终态1789329697.549690，本组进程退出后下一会话接续。原始文件未覆盖。

复算命令：
```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_restore_token_reservation.py --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260914_restore_token_reservation_r01/execution --output-dir /private/tmp/token-reservation-recheck-new --expected-metadata-sha256 ed37ef0b7fbdc6fdc3fc15024ff4555adf4e176c28667b2431c7362929d8fcdc
```
输出目录必须此前不存在。
