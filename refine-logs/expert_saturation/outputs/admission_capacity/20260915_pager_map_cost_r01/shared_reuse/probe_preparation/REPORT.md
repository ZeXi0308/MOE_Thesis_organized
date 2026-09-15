# 条件调用入口：恢复前态后才能测映射发布成本

2026-09-15。**本轮排除了一个会污染下一实验的执行方式，并把单层探针推进到CPU准备完成。** 未运行CUDA，也未新增独立工作负载或请求级结果。

## 新证据与影响

原先选定的两个调用只保存route、plan和末态，尚不足以直接构造入口。本轮从同一原始X轨迹的前一次同层调用恢复完整slot、map、LRU ticks/clock，再调用封存的`plan_shared_pool`，两者逐项重现原plan。

| 条件状态 | 目标/前同层call | active专家 | H2D/D2D专家 | 首次符号执行读错 | 不复位重复旧plan读错 |
|---|---|---:|---:|---:|---:|
| 有加载 | 112 / 96 | 56 | 38 / 18 | 0 | 18 |
| 全命中 | 983 / 967 | 8 | 0 / 0 | 0 | 0 |

有加载调用的18个命中专家先从private搬到shared，然后原private位置被H2D覆盖；第二次复用旧plan会把这些位置的新专家搬到旧专家的目的槽。因此不能用不复位的固定plan循环测同一动作。全命中负控恰好不会揭露此问题。完整前态、逐项plan相等检查和错误专家身份见[prepared.json](prepared.json)。这只否定该探针实现方式，没有声称原X runtime存在此错误。

[probe_shared_map_publication.py](../../../../../experiments/admission_capacity/probe_shared_map_publication.py)现在每次重新建立相同CPU元数据、private权重内容及local map，再经原`ordered_dispatch`调用原planner/oneshot。复位耗时单列；主计时从该前态进入调用开始，到本调用GPU工作完成结束。故它回答的是条件调用成本，不能把复位剔除后的结果直接当连续服务收益。

## 公平底座与计费

三模式仍为原阻塞、只改第二张映射、两张一起异步；顺序为三模式及反序。**性能底座直接调用未修改的封存oneshot**，三者持有相同预分配bank，候选额外Python封装算入其成本。并非给baseline也加包装后再比较。每模式独立warmup，保留全部sample和失败；默认每块3次warmup、12次测量仅为待接受设置，不是已冻结GPU合同或独立样本量。

完整调用墙钟包含route回读、实际plan、权重搬运、两张map、原统计/控制、kernel及返回后的GPU drain。host return和drain互斥相加得到完整调用；不加上嵌套host_apply或CUDA spans。诊断profiler另跑，不参与该主计时。其CPU/CUDA活动与trace导出接口依据[PyTorch 2.11文档](https://docs.pytorch.org/docs/2.11/profiler.html)，用于检查实际设备执行及复制，不能把kernel前event标记冒称真实kernel启动。

## 数值、资源和未验证边界

沿用封存[共享池执行源码](../../../20260912_wisp_olmoe_r01/logical_alignment_validation_r01/source/run_shared_pool_pager.py)的384槽几何和full64数值参照；[fused_experts入口](../../../20260912_wisp_olmoe_r01/fresh_cohort_r01/attempt01/instrumentation/installed_source/fused_moe.py)明确使用SILU、无量化、router weight不乘于输入。driver显式传`FUSED_MOE_UNQUANTIZED_CONFIG`，不凭构造器默认值猜配置。

固定seed生成FP32正态权重再转pinned BF16，以及合成hidden；64-way logits强制实现记录中的top8顺序，softmax64后gather，不再归一化top8。它们与真实route/plan共用，但**不包含历史模型权重、hidden或router概率**。数值资格检查active与最终private专家字节、各模式对full64输出finite及bit-equal；资格和参考分配不进性能样本。

共享pool 4.5GiB，host pinned master 0.75GiB，两类bank各28,672B。数值资格还需额外0.75GiB GPU full64参照，测试前释放；输入、workspace、CPU临时FP32和allocator保留由实际peak/HWM记录，不能将静态账当最终峰值。这里的单层host预算不是完整16层服务的host预算。

已执行CPU准备并得到上表、源码解析通过；真实CUDA导入、数值资格、profiler产物及完整调用时间均UNRUN。入口使用共同flock、初始化前与块边界检查全部GPU计算进程，不结束他人进程；当前未获本支线执行身份，因此没有封包、上传或运行。main负责接受唯一版本/窗口后才能实际调用GPU入口。

## 复算与唯一下一步

从共享仓库根目录，使用新输出路径：

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/probe_shared_map_publication.py prepare --repo-root . --case-file refine-logs/expert_saturation/outputs/admission_capacity/20260915_pager_map_cost_r01/shared_reuse/probe_cases.json --source-dir refine-logs/expert_saturation/outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_validation_r01/source --output-dir /tmp/map-publication-preparation-recheck
```

GPU入口是同脚本的`gpu`动作，接受`--prepared`、同hash`--source-dir`、独占新`--output-dir`、实际共同`--lock-file`和device；这些必须来自主方接受的整组合同，不在此发明另一把锁。若数值资格失败，不作性能排名；完整调用无收益空间就停止map-staging实现；有空间再提出强X下完整请求对照。两种状态的rows/miss不同，不能用它们的绝对时间差孤立归因搬运，只比较每个固定前态内的模式差异。
