# A：共同增长能否提供新的victim容量排序？

本线长期职责：资源演进与恢复动作排序。此次是对原主方自然保存资格数据新增的条件候选分析，不是新GPU运行、主表复算或B/C生命周期/执行份额分析。

**结论：当前25个prepare前态不支持为容量排序引入复杂短期预测器。** 在一个16-token KV块周期的条件模型里，最大可释放块策略的所有同分候选均达到最佳容量余量，25/25无容量regret。实际most_output与最佳余量不同19/25，但这不是19次错误决策：victim自身后续等待、保存价值、搬运时间和整体完成代价均未排名。

状态使用当时free、computed、prompt、已输出量、allocated、tracker资格；不使用真实未来EOS、完成次序、route或load完成事件。25次prepare全部纳入，实际victim全部出现在合法候选中。资格复用原progress=(computed−prompt)/max_output及absence/residency条件。第一次草稿误用了total-length分母，已纠正；错误输出以INVALID_progress_denominator_r01.json保留，不能引用其科学计数。原数据和执行包未改。

条件模型只问：假设立即释放一个victim的独占块，让target恢复当前历史并再产16个输出，同时每个保留peer再计算16位置，容量是否足够？

```text
growth_i(h) = max(0, ceil((computed_i+h)/16) − held_i)
need_target(k) = max(0, ceil((prompt+output+k−1)/16) − held_target)
margin(v) = free + held_v − need_target(k) − Σ[i≠v] growth_i(h)
```

最终输出的KV尚未计算，因此target需求使用k−1。这不是预留策略，也不要求让peer同步实际执行。假设没有新准入、没有释放/共享引用/驱逐、没有prepare期增长或异步load等待；它比较的是假想立即swap后的空间包络，**不保证native动作可执行或持续服务**。不把max_output当真实EOS；这里完全不给未来释放信用。实际独占块条件之外不能用held代替可释放量。

固定h=k=16因为它是一个KV块周期，不按事件结果挑阈值。代数上最大化margin等价于最大化held_v+growth_v(h)；在紧凑分配的纯decode状态、h=16时growth_v=1，排序退化为最大held。实际25前态验证了同样的排序结论。它不是新算法新颖性主张。

| 实际most_output余量为负的prepare | 当前margin / blocks | 最佳合法margin / blocks |
|---|---:|---:|
| 818 | -10 | 107 |
| 898 | -21 | 117 |
| 1157 | -23 | -23 |
| 1342 | -6 | 59 |

两例已经由原方定位的短恢复提供相反的后续研究线索：step898可在条件空间上通过别的victim获得余量；step1157所有合法候选都不足，单换victim无法满足本包络。**负margin不等于实际必短恢复，正margin也不保证不中断**：执行、准入和通知时序仍可能改变资源状态。不得由这个表推断受损概率或完整请求收益。

另复用同一组的step903/1161当前状态做一调用需求检查：free=0，joint decode需求分别6/5块；目标自己的下一位置均无需新块。它与原方已定位结果一致，不增加独立证据数；新贡献是前态候选比较及其适用边界。

验证：CPU逐位置枚举覆盖computed=0..63、额外已分配0..2块、增长0/1/16/17位置；全部与块端点公式一致。另检查一个“一步下更多释放只形成同分”的候选反例，以及两目标自身0增长。没有GPU或性能测试。

**改变的研究决定：** 不实现复杂共同增长victim predictor；保留最大可释放量为该容量子目标的强简单基线。仅空间余量无法覆盖step1157类状态，后续必须将可执行服务分配/切换代价与容量条件分开，不默认延长保护或等待会更好。

**唯一下一项建议：** 原主方完成同预算native完整保存资格后，复用其自然运行前态检查“最大释放也无法资助短期共同进展”的状态是否仍出现；若仍出现再比较最小动作，若消失则停止当前selected-save下该缺口的机制扩展。当前可以自主推进模型，GPU执行仍归原方。

证据级别STRUCTURAL_CONDITIONAL_CURRENT_STATE_CANDIDATE_SCREEN；最强容量对照maximum released blocks；无秒级Oracle/质量/方法GO。科学问题未判死，只否定本批状态中复杂容量排序相对简单规则的必要性。

复算（仓库根目录，输出新路径）：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01/joint_growth_model/check_joint_growth.py --selective refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_saved_recovery_gate_r01/execution_weste_26862/readback/results/diagnostic-current/selective-store.json --output /private/tmp/joint-growth-recompute.json
```
