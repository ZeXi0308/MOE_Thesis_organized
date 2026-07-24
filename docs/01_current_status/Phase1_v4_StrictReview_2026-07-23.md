# Phase 1 v4 严格复核（2026-07-23）

## 直接结论

上一轮排序**不能直接进入实现**。主线 I1 与现有 FJRC 科学对象重复；备选 I4/I5
重复已经过 GPU 有效性审计并判为系统 NO-GO 的 expert prefetch；I2 在更早 Phase 1
已因 replica/reroute prior-art 拥挤和单卡不可形成关键证据而淘汰。上一轮评分遗漏了
“历史对象去重”这一硬门，因此总分排名无效。

## 阻塞性问题

1. **I1 是 FJRC 换名。** FJRC 已研究 multi-sender joint first admission、receiver
   credit 与 fork-join deadline risk，并明确把唯一信息增量定义为 keyed sibling
   deficit/correlation。新名称没有构成新 scientific object。
2. **FJRC 旧协议本身被 supersede。** 旧 `R0` 只比 `B` 多 keyed queue map，实际测量
   普通 q-map，不是 join phase；arrival/background/pairing/simple-baseline generator
   也未唯一冻结。
3. **I4/I5 忽略已有系统 NO-GO。** 正确 `(layer,expert)` cache key 与 full top-k 后，
   OLMoE/LLM-jp working set 饱和；safe-budget oracle 接近 0，不能靠 admission 或
   abstention predictor 复活同一 residency/prefetch 对象。
4. **I2 已被前序筛选淘汰。** receiver shadow-price replica/reroute 与已有 load-aware
   replication/placement/min-cost routing 高度邻近，且 5090 只能给解析上界。

## 高风险问题

- I6 风险预算虽可做统计验证，但当前 dynamic FP8/QuantizeOnce 没有低于 BF16 的真实
  decode operating region；没有系统杠杆时，allocation 正结果不是 Energy-SLO 结果。
- I7 与已有 Verify/TriageAudit scientific object 重叠；canonical paired state 和 shadow
  overhead 尚未解决。
- I9 的 batching-energy primitive 有正证据，但 deadline-aware coalescing 已与 AMoE/
  Festina 邻近；缺少独立增量。
- I10 与现有 Quality Isolation/quality debt 对象重叠，且宿主执行机制未成立。
- I8 具有真实物理上限，但当前 novelty 仅为“需要 fusion”；在写 kernel 前必须审计
  Triton/Transformer Engine/DeepEP/FlashInfer 等 producer-consumer fusion 现状。

## 一般问题

- “负结果价值”评分过度乐观，多个候选已有相同负结果，重复失败不再产生新知识。
- 单卡可跑不等于单卡能验证核心因果对象；I2 与 receiver physical incast尤其如此。
- 把实现难度反向计分后，总分会机械偏向低 novelty 的分析任务。

## 实验偏差风险

- 用 token-position 对齐 wave 冒充自然 temporal overlap；
- 用 synthetic queue depth放大 receiver headroom；
- baseline 禁止读 request identity，从而人为削弱 request-FCFS；
- 把 q-map 差异与 join-phase差异同时改变，无法归因；
- 用 candidate 自己定义的 risk/credit 指标自证；
- 在没有可执行低精度快区时，把 KL allocation 写成 energy optimization。

## 建议修复

1. I1 降级为“现有 FJRC 的 corrected necessity test”，不再计新 Idea。
2. 第一实验只验证 `R=Q+J` 相对 `Q` 的 join-phase信息增量；不写 credit controller。
3. matched worlds 保持 exact q-map、task universe、deadline、service、route完全相同，
   只交换由其他 sender 已完成 siblings产生的可达 join phase。
4. request-FCFS、EDF、least-laxity、join-remaining-work、aggregate credit均作为强基线；
   baseline可读 request/join identity，但不能读已完成 sibling bitmap。
5. I8 保留为独立高风险后备，但先做 prior-art/ABI 插入点审计。

## 是否允许进入 GPU 实验

**否。** 允许进入 corrected FJRC 实验设计；不允许运行正式 GPU。只有设计完成并经
逻辑审查确认信息集隔离、matched-world可达性和 baseline公平后，才允许实现 Level 0
oracle fixture。GPU LUT 不是当前第一步。

