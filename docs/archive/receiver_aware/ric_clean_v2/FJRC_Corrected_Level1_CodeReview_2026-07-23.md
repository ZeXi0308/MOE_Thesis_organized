# Corrected FJRC Level 1 Code Review（2026-07-23）

## 结论

`BLOCKED_LEVEL1_FORMAL_RUNNER / GPU RUN NOT APPROVED`

native pair producer 和 LUT validator路径可以继续复用，但当前实现尚不是完整 Level 1
oracle runner。不得用mock route结果或旧LUT直接启动GPU/科学回放。

Source SHA-256：

- `fjrc_corrected_level1.py`：
  `4cebae9141efe3e472cc23fa534c5b3f7f0dc9930665429008dde018532b1413`
- `test_fjrc_corrected_level1.py`：
  `44d4a4270a299c22695b9170c377b9138ed4b3a93aa7b6a08b70b0c10694d447`
- reused LUT producer：
  `c6133b0b2d4c6a58a936a7d50d1c5a59a3cbe3c09849ba33d711f501f9e80273`

## 1. 代码执行链路与实验逻辑

1. `load_verified_joins()`负责clean-v2 route/data/placement/signoff闭包。
2. `extract_service_lut()`重算FJRC LUT summary并提取pack、analytic cut、depth-1 unpack、
   combine median。
3. `request_split()`按receiver做32 selection / 32 holdout request-disjoint split。
4. `_choose_roles()`只依赖route structure选择同sender的可交换prior siblings，以及来自
   不同sender的两个current candidates。
5. `select_holdout_scenarios()`每receiver固定4个holdout requests、组成2对，总计16对/
   32 requests；输入顺序不影响选择。
6. `materialize_pair()`生成Level-0可消费的task/join/world对象。

## 2. 已确认正确的关键实现

- request split为32/32、零重叠，每receiver严格8 requests；
- holdout固定16 scenarios、32个互异requests；
- pair选择不读取deadline miss、policy outcome或oracle gain；
- route输入顺序反转时scenario选择完全相同；
- prior pair具有同sender/resource signature，两个candidate来自不同sender；
- common-sender support不足时输出`BLOCKED_INSUFFICIENT_MATCHED_SUPPORT`；
- Q observations跨world相同，J observations不同；
- model/LUT不匹配、LUT self-hash/summary/environment异常均fail closed。

## 3. 潜在bug、偏差和混杂因素

### 阻塞性

1. **四段会计尚未进入corrected simulator。** 当前把pack+cut+unpack相加为一个
   `service_us`；虽然数值来源分开，event ledger没有分别执行/校验三资源。
2. **q-map仍由scenario直接给0。** 没有从background/prior receiver event history重算，
   无法证明exact queue state与join phase真正正交。
3. **deadline退化。** pair内两个request使用相同deadline factor；EDF/least-laxity会
   退化为identity tie-break，不能作为强baseline。
4. **arrival generator缺失。** selection split没有实际选择arrival/rho/deadline cell，
   因此当前mock可辨识度可能由固定future release人为制造。
5. **完整baseline与统计runner缺失。** 尚无request-FCFS、EDF、least-laxity、SRPT、
   projected-finish、32-request CVaR90、bootstrap、两模型AND gate。
6. **旧LUT绑定superseded protocol。** primitive本身可迁移，但正式consumer必须同时绑定
   corrected design/addendum和旧artifact语义，不能只信旧protocol SHA。
7. **真实artifact不可用。** 本地没有route trace/FJRC LUT；远端5090 SSH当前直接关闭，
   未完成native small-sample dry run。

### 非阻塞但需报告

- 单receiver、B=1；B=2/4尚未实现；
- analytic 200Gb/s cut，不是NCCL/RDMA；
- service使用median，未做paired realization/sensitivity。

## 4. 必须修改项与建议修改项

### 必须修改

1. corrected core拆分pack/cut/unpack事件和资源守恒；
2. 从合法background ledger重算q-map并验证两world逐receiver exact相同；
3. selection-only arrival/deadline generator，冻结后holdout不得重选；
4. 全部强baseline、request-level fold、CVaR90与paired bootstrap；
5. corrected protocol迁移binding及完整artifact runner；
6. 恢复真实route/LUT入口后完成native 2-pair CPU dry run。

### 建议修改

- 将service median替换为相同trial-index paired realizations；
- 先报告natural support census，再决定是否值得完整16-pair materialization；
- B=2/4只在主B=1冻结后作为敏感性实现。

## 5. CPU / 小样本 dry run

- corrected Level 0 tests：12/12 PASS；
- Level 1 pair producer tests：5/5 PASS；
- FJRC LUT validator tests：5/5 PASS；
- 合计22/22 PASS；`py_compile`与`git diff --check`通过。

Mock producer验证的是结构与fail-closed路径，**不构成native route实验结果**。

## 6. 是否允许进入GPU实验

**GPU Run Approved：否。**

当前允许保留Level 1 pair producer，等待补齐上述P0并恢复真实artifact入口。若远端资源
恢复，也只能先取回/重捕LUT和做2-pair CPU dry run；不得直接跑正式campaign。

