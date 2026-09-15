# Corrected FJRC Level 0 Code Review（2026-07-23）

## 结论

`CPU LEVEL-0 APPROVED / GPU RUN NOT APPROVED`

Corrected information-lattice reference 已足以验证代码是否能隔离 `Q` 与 `J`、识别
strict first-action flip，并对 equal-phase、shuffled-key、fanout-1 返回零增量。它尚未
消费 native route/LUT，也没有 Level 1 baseline/统计 runner，因此不得进入 GPU实验。

Reviewed source SHA-256：

- `fjrc_corrected_level0.py`：
  `4f88d97ca6110a8e66e09f9e3cdd265fca0e9570190c987d620e4c97dba79611`
- `test_fjrc_corrected_level0.py`：
  `1c432a65d2382ce4896e798b43b5206462f2a503f4f5ca4d094961ec8a97bdac`
- experiment design：
  `499531611403dff9cc5e7ff7c18b0dd7d5f2dc15d71137430f18e23bac2f19ad`

## 1. 代码执行链路与实验逻辑

1. `validate_scenario()` 验证完整 task/join universe、候选来自至少两个 sender和两个
   joins、单 receiver locus、receiver q-map、prior/future exactly-once partition，以及
   两 world future work/resource多重集相同。
2. `observation()` 生成四个隔离视图：`B0`、`Q`、`J`、`R=Q+J`。
3. `simulate()` 对指定 first-credit task执行 B=1 receiver service；其余任务按冻结
   ready/id顺序 full-drain，全部 siblings完成后执行 once-per-join combine。
4. `optimize_arm()` exact枚举两 world的 first action。相同 observation label必须共享
   action；不同 label才允许 world-specific action。
5. `optimize_information_lattice()` 强制 B0/Q不可区分 worlds、J/R必须区分，并只报告
   `Q-to-R`增量。
6. controls 使用同一 simulator/optimizer：equal phase、shuffled/uninformative key、
   fanout-1。

## 2. 已确认正确的关键实现

- **信息隔离：** Q observation不含 completed siblings；J不含 queue map；R才同时包含。
- **nonanticipativity：** Q 两 world label相同时，枚举器只允许同一个first action。
- **可达 partition：** 每个 auxiliary task在每 world恰属于 prior或fixed-after，不能
  重复或丢失；future work/resource/release多重集跨 world相同。
- **任务守恒：** dry run每 world为4个task = 1 prior + 3 post-t0；unique completion=4；
  2个 joins各 combine一次。
- **单 receiver fail-closed：** 输入多个 receiver resource直接拒绝，避免被全局`now`
  错误串行化。
- **strict flip敏感性：** positive fixture中Q最优集合非 singleton且不能分 world；R的
  唯一policy为 `(a,b)`，两 world首动作不同。
- **负对照：** equal-phase、shuffled-key、fanout-1均返回零增量/无strict flip。
- **额外属性检验：** 固定seed随机生成200个合法service/deadline场景，Q始终不可区分、
  R objective从不劣于Q、shuffled-key始终零增量；14个场景触发strict flip，证明实现
  同时能识别正/负路径。

## 3. 潜在 bug、偏差与混杂因素

### 当前已修复

1. 初版设计要求交换prior identity但future identity逐项相同，违反exactly-once；已改为
   future work/resource多重集相同、identity互补。
2. 初版测试用同一修改同时破坏两个不变量，无法定位失败原因；已拆成单因素负例。
3. 初版允许多 receiver却用global `now`串行；已限制为exactly one receiver。
4. 初版positive fixture每join有两个对称candidate，导致task-level unique flip结构上
   不可出现；已改为两sender/两join各一个current candidate，并新增true-flip断言。

### 仍存在但不阻塞 Level 0

- q-map是scenario输入，不是从完整background ledger重算；Level 1必须补。
- prior history只证明service可容纳与时间因果，没有显式pack/cut/unpack资源事件；
  Level 1必须复用四段ledger。
- post-first order是冻结FCFS，不代表可部署controller，也未实现B=2/4。
- 当前只有normalized service toy fixture，不含native top-k 8/16 route。
- 没有EDF、least-laxity、request-FCFS、remaining-work等强baseline。
- 没有32-request aggregation、CVaR90、bootstrap、selection/holdout producer。
- 没有credit codec、staleness、control tax或CUDA timing。

## 4. 必须修改项与建议修改项

### 进入 Level 1 代码审查前必须完成

1. native route consumer：校验route/data/placement/signoff SHA并构造至少16个disjoint
   pairs；不足即BLOCKED。
2. 从合法 prior/background event ledger逐receiver重算 q-map，禁止调用方自报。
3. pack/cut/unpack/combine完整会计，每task一次、每join combine一次。
4. 实现并隔离全部强baseline；request-FCFS可读identity但不可读bitmap。
5. 32-request denominator、world folding、CVaR90、paired bootstrap和两模型AND gate。
6. config/raw/summary/environment/source manifest原子保存；输出存在时拒绝覆盖。

### 建议修改

- 为B=2/4实现组合action enumerator，但不得在主B=1结果后调阈值；
- 加入property-based testing库或等价固定生成器落盘；
- 将Metrics序列化schema与正式runner分离，避免dataclass repr进入artifact。

## 5. 最小 CPU / 小样本 dry run

执行：

```text
python3 -m unittest -v \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_fjrc_oracle_core \
  docs.ideas.receiver_aware.experiments.ric_clean_v2.test_fjrc_corrected_level0
```

结果：旧core 11项 + corrected 12项，共23项全部PASS。

Corrected positive fixture：

```text
Q selected=('a','a'), objective=(0.75,2.875,9.5), unique=false
R selected=('a','b'), objective=(0.50,2.750,9.5), unique=true
strict_flip=true
equal-phase=true, shuffled-key=true, fanout-1=true
```

这些是刻意构造的unit fixture，只说明reference能检测信息差，**不是 scientific result**。

## 6. 是否允许进入 GPU 实验

**GPU Run Approved：否。**

允许进入下一步：编写 Level 1 native-route/LUT consumer与CPU oracle runner，然后重新
Code Review。当前不得上传5090执行，也不得把toy fixture中的25pp miss差写成headroom。

