# commit重检能改变哪部分恢复成本

本方继续负责保存语义、恢复生命周期和成本；A负责动作与adapter。本轮复用A在各25个commit中找到的全部两次`RESUME_WITHOUT_VICTIM`，只补充原轨迹中后续成本与保存状态，不重新筛victim、重放策略或运行GPU。

**研究判断：该动作是有依据的基线完善候选，但现有两例不能预告主要长停顿改善。** 它可能避免victim下一次恢复及强制flush暴露；prepare保存状态不会被当前候选撤销。两个被改变的victim事件都不是原轨迹的最大gap事件，净收益必须由独立演进验证，不能从原轨迹直接减掉一次加载或整段gap。

| 全部改变动作的事实上下文 | selected D859 | native-full E1399 |
|---|---:|---:|
| 当时free / target需要 | 193 / 95块 | 277 / 106块 |
| 保留victim及其持块 | 0748，233块 | 3790，184块 |
| victim后续已完成load job | 124 | 2011 |
| 实际加载前缀 / 按原生路径重建字节 | 3712位置 / 464MiB | 2928位置 / 366MiB |
| 该次历史重算 | 8位置 | 5位置 |
| 该次L→F输出间隔 | 0.112243s | 0.260893s |
| 后续输出 / 该服务段结束 | 79 / 再抢占 | 120 / 完成 |
| 原轨迹最大恢复gap | 0748的step943，3.531065s | 0748的step1021，3.971221s |

载入字节从同版本、no-APC、完整连续前缀路径重建，连接同请求的唯一accepted/completed load；不是逐job传输数组直接读值。原selected的0748较晚确实出现长停顿，但不能将step859的一次避免驱逐等同于消除step943之后的事件。两例target都是0271，事实L→F分别1.638750s、3.577566s；commit重检只改变当前victim是否驱逐，没有把当前commit时间前移，所以不能把target已经发生的等待记成可追溯消除。

## READY不是保存已经完成

两个commit同为`READY`，保存阶段却不同：

- **D：已登记、尚未提交。** prepare858创建job122，232块/464MiB；commit snapshot为4863396.77288654，实际dispatch到4863396.774762406才发生，完成上报为4863396.798338556。`store_delta`本来也写明`transfer_completion=UNKNOWN`。所以“prepare已发生”不能改写成“commit前D2H已完成”。
- **E：此前增量保存已完成。** prepare1398本步新增store为0；3790此前60个job合计183块/366MiB，逻辑索引0..182。最后job2000的完成上报4864780.99016655早于commit snapshot4864781.069946102。累计已完成保存量仍不等于额外分配或任意时刻可释放host容量。

原adapter在`commit_reason(..., save_enabled=False)`路径检查计划、状态和资源资格；它没有在此承诺host已就绪。随后native flush/异步load通知继续承担真实传输依赖。模型应至少区别`registered/unsubmitted`、`submitted/pending`和`completion-reported`，不能以一个READY标签替代三者。

A的[候选adapter](../commit_recheck/staged_store_rotation_candidate.py)的direct分支只保留victim、建立target保护/队列次序并记录独立计数；没有删除connector/worker的store任务。D的job122已经位于原生延迟提交路径，去掉preempt及相应强制flush本身不会取消复制；E此前完成的保存当然也不能撤销。因此两例都不应扣减这些D2H或预分配16GiB host。D的未提交传输尚不是沉没传输，但仍是**该候选保留的后续工作**；已完成的E传输才是沉没成本。跳过强制flush可能改变暴露时序，不等于复制字节被节省。

## 最小成本关系及验证目标

当前候选的成本变化应保留以下分量，而非先合成未经验证的毫秒收益：

```
可能少一次victim load及其尾部重算
+ 保留victim导致的后续额外/转移恢复与batch代价
+ 原生flush/提交的暴露时序变化
+ 目标和全部其他请求等待变化
```

prepare已创建的store不由该动作撤销，不能重复扣款。464/366MiB和0.112/0.261s都只是原轨迹上下文，不是净saving或Oracle上界。保留victim若导致另一请求随后被驱逐，净加载量可能增加；当前两条原轨迹不能给这个反事实赋值，JSON明确保留`net_load_bytes_saved=null`和`counterfactual_output_gap_s=null`。

**唯一下一建议：** 将commit重检作为固定selected/current底座上的最小基线完善检验，保持cooldown/窗口不变；资格应确认target实际恢复、victim没有被该commit驱逐、已登记store继续通过原生路径完成，并保留后来所有请求的加载/输出/再次抢占。只测E的“旧保存都已完成”情形不能覆盖D的未提交store分支。若真实轨迹无该动作，就保留no-action，不提高压力制造机会；若仅少一次局部load而完整服务无增量，保留简单实现的工程边界，不包装主线方法。

没有新增执行身份或GPU组，主方统一决定接受版本；本方不修改A的adapter。当前成本边界可以直接用于解释其下一实际结果。

## 复算

```bash
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/commit_recheck_cost/analyze.py \
  --outputs refine-logs/expert_saturation/outputs/admission_capacity \
  --output /tmp/commit-recheck-cost-context.json
```

脚本已实际运行，覆盖A公布的全部改变动作，检查唯一victim段、target身份、accepted/completed load、实际前缀和prepare/commit时序。源码在独立worktree开发，本目录[analysis.json](analysis.json)为唯一共享结果；D/E原始诊断、A候选和轻量四格均未改变。状态为`FACTUAL_COST_CONTEXT_NOT_ACTION_VALUE`，证据是同版本原生诊断派生上下文；没有新样本、性能反事实、客户端时间或方法GO。
