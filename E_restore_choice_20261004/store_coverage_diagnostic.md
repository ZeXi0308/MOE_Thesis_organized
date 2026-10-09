# STORE 覆盖诊断（自动生成）

仅 CPU 读取已有证据；无 GPU、无新服务动作。完成回报不证明持续驻留，CPU 回放不证明时延收益。

| 输入 | 请求/恢复 | 分类 |
|---|---:|---|
| oct08_e246_store_replay53005_r01/00_recompute | 320/52 | `{"NOT_CREATED": 1, "COVERED_AT_NEXT_LOOKUP": 2, "INSUFFICIENT_EVIDENCE": 25, "INSUFFICIENT_UNLINKED_COMMITS": 0, "NO_FULL_BLOCK_GAP": 24}` |
| oct08_e246_store_replay53005_r01/01_recompute | 320/52 | `{"COVERED_AT_NEXT_LOOKUP": 3, "INSUFFICIENT_EVIDENCE": 25, "INSUFFICIENT_UNLINKED_COMMITS": 0, "NO_FULL_BLOCK_GAP": 24}` |
| oct08_two_burst160_20s53005_reverse_r01/00_recompute | 320/38 | `{"INSUFFICIENT_UNLINKED_COMMITS": 38, "NO_FULL_BLOCK_GAP": 0}` |

## /Users/zhaozhenyu/Desktop/毕业设计/E_restore_choice_20261004/execution/oct08_e246_store_replay53005_r01/00_same_engine/00_recompute/raw.json

观察缺口：`[]`。
全程 STORE/LOAD 报告字节：135499087872 / 11221860352。

| 请求/事件 | 判断 | 游标前→后 | 窗口新建字节 | 后续 Host hit |
|---|---|---:|---:|---:|
| measured/E246/0 | NOT_CREATED | 197→197 | 0 | 1168 |
| measured/E246/1 | COVERED_AT_NEXT_LOOKUP | 197→73 | 12582912 | 3168 |
| measured/E246/10 | COVERED_AT_NEXT_LOOKUP | 198→198 | 2097152 | 3184 |
| measured/E214/502 | INSUFFICIENT_EVIDENCE | 220→172 | 60817408 | None |
| measured/E215/505 | INSUFFICIENT_EVIDENCE | 212→129 | 211812352 | None |
| measured/E217/507 | INSUFFICIENT_EVIDENCE | 223→223 | 41943040 | None |
| measured/E218/508 | INSUFFICIENT_EVIDENCE | 226→226 | 44040192 | None |
| measured/E219/509 | INSUFFICIENT_EVIDENCE | 225→225 | 46137344 | None |
| measured/E221/510 | INSUFFICIENT_EVIDENCE | 202→202 | 50331648 | None |
| measured/E222/511 | INSUFFICIENT_EVIDENCE | 220→220 | 54525952 | None |
| measured/E223/512 | INSUFFICIENT_EVIDENCE | 195→195 | 56623104 | None |
| measured/E225/513 | INSUFFICIENT_EVIDENCE | 194→194 | 58720256 | None |
| measured/E226/514 | INSUFFICIENT_EVIDENCE | 218→218 | 62914560 | None |
| measured/E227/515 | INSUFFICIENT_EVIDENCE | 200→200 | 65011712 | None |
| measured/E229/516 | INSUFFICIENT_EVIDENCE | 203→203 | 67108864 | None |
| measured/E230/517 | INSUFFICIENT_EVIDENCE | 204→204 | 69206016 | None |
| measured/E231/518 | INSUFFICIENT_EVIDENCE | 195→195 | 71303168 | None |
| measured/E233/519 | INSUFFICIENT_EVIDENCE | 194→194 | 75497472 | None |
| measured/E234/520 | INSUFFICIENT_EVIDENCE | 204→204 | 77594624 | None |
| measured/E235/521 | INSUFFICIENT_EVIDENCE | 212→212 | 81788928 | None |
| measured/E237/522 | INSUFFICIENT_EVIDENCE | 187→187 | 81788928 | None |
| measured/E238/523 | INSUFFICIENT_EVIDENCE | 202→202 | 83886080 | None |
| measured/E239/524 | INSUFFICIENT_EVIDENCE | 194→194 | 90177536 | None |
| measured/E241/525 | INSUFFICIENT_EVIDENCE | 207→207 | 92274688 | None |
| measured/E242/526 | INSUFFICIENT_EVIDENCE | 188→188 | 94371840 | None |
| measured/E243/527 | INSUFFICIENT_EVIDENCE | 200→200 | 96468992 | None |
| measured/E245/528 | INSUFFICIENT_EVIDENCE | 179→179 | 98566144 | None |
| measured/E246/529 | INSUFFICIENT_EVIDENCE | 201→201 | 111149056 | None |

## /Users/zhaozhenyu/Desktop/毕业设计/E_restore_choice_20261004/execution/oct08_e246_store_replay53005_r01/00_same_engine/01_recompute/raw.json

观察缺口：`[]`。
全程 STORE/LOAD 报告字节：135499087872 / 11484004352。

| 请求/事件 | 判断 | 游标前→后 | 窗口新建字节 | 后续 Host hit |
|---|---|---:|---:|---:|
| measured/E246/0 | COVERED_AT_NEXT_LOOKUP | 197→73 | 10485760 | 3152 |
| measured/E246/1 | COVERED_AT_NEXT_LOOKUP | 197→197 | 2097152 | 3168 |
| measured/E246/34 | COVERED_AT_NEXT_LOOKUP | 199→199 | 2097152 | 3200 |
| measured/E214/502 | INSUFFICIENT_EVIDENCE | 220→172 | 60817408 | None |
| measured/E215/505 | INSUFFICIENT_EVIDENCE | 212→129 | 211812352 | None |
| measured/E217/507 | INSUFFICIENT_EVIDENCE | 223→223 | 41943040 | None |
| measured/E218/508 | INSUFFICIENT_EVIDENCE | 226→226 | 44040192 | None |
| measured/E219/509 | INSUFFICIENT_EVIDENCE | 225→225 | 46137344 | None |
| measured/E221/510 | INSUFFICIENT_EVIDENCE | 202→202 | 50331648 | None |
| measured/E222/511 | INSUFFICIENT_EVIDENCE | 220→220 | 54525952 | None |
| measured/E223/512 | INSUFFICIENT_EVIDENCE | 195→195 | 56623104 | None |
| measured/E225/513 | INSUFFICIENT_EVIDENCE | 194→194 | 58720256 | None |
| measured/E226/514 | INSUFFICIENT_EVIDENCE | 218→218 | 62914560 | None |
| measured/E227/515 | INSUFFICIENT_EVIDENCE | 200→200 | 65011712 | None |
| measured/E229/516 | INSUFFICIENT_EVIDENCE | 203→203 | 67108864 | None |
| measured/E230/517 | INSUFFICIENT_EVIDENCE | 204→204 | 69206016 | None |
| measured/E231/518 | INSUFFICIENT_EVIDENCE | 195→195 | 71303168 | None |
| measured/E233/519 | INSUFFICIENT_EVIDENCE | 194→194 | 75497472 | None |
| measured/E234/520 | INSUFFICIENT_EVIDENCE | 204→204 | 77594624 | None |
| measured/E235/521 | INSUFFICIENT_EVIDENCE | 212→212 | 81788928 | None |
| measured/E237/522 | INSUFFICIENT_EVIDENCE | 187→187 | 81788928 | None |
| measured/E238/523 | INSUFFICIENT_EVIDENCE | 202→202 | 83886080 | None |
| measured/E239/524 | INSUFFICIENT_EVIDENCE | 194→194 | 90177536 | None |
| measured/E241/525 | INSUFFICIENT_EVIDENCE | 207→207 | 92274688 | None |
| measured/E242/526 | INSUFFICIENT_EVIDENCE | 188→188 | 94371840 | None |
| measured/E243/527 | INSUFFICIENT_EVIDENCE | 200→200 | 96468992 | None |
| measured/E245/528 | INSUFFICIENT_EVIDENCE | 179→179 | 98566144 | None |
| measured/E246/529 | INSUFFICIENT_EVIDENCE | 201→201 | 111149056 | None |

## /Users/zhaozhenyu/Desktop/毕业设计/E_restore_choice_20261004/execution/oct08_two_burst160_20s53005_reverse_r01/00_same_engine/00_recompute/raw.json

观察缺口：`["host_history_observer_missing_or_disabled"]`。
全程 STORE/LOAD 报告字节：135289372672 / 0。

| 请求/事件 | 判断 | 游标前→后 | 窗口新建字节 | 后续 Host hit |
|---|---|---:|---:|---:|

## 既有干预对照（第二份减第一份）

实际游标干预 1 次。全部请求重复位置差 -1974；同伴重复位置差 0。
全程 STORE 字节差 0；LOAD 字节差 262144000。
已有实测完成均值差 0.03334885719232261 s；排空完成时刻差 0.14841239899396896 s。CPU 分析未生成新的时延收益。
GPU 轨迹未记录可归因的同伴驱逐；该项及由驱逐造成的后续重算为 null，不是零。
补写窗口 measured/E246：新建任务字节 0→10485760；下一 Host hit [1168, 3152]。

## 既有 CPU 驱逐反例

新增 1 chunk；字节未知。移除驻留 key `['B0']`；前缀 {'A': 2, 'B': 8}→{'A': 8, 'B': 0}。
没有后续请求执行，重复计算与时延为 null；没有原始 STORE 完成事件，不能升级为完整任务生命周期证据。

## 判定边界

- NOT_CREATED：仅已覆盖窗口内该块无任务；不是全局从未写入或上游 bug。
- NOT_YET_COMPLETED：原生任务尚未确认完成，不等于 CUDA 数据未传完。
- COMPLETED_THEN_EVICTED：需要同一 STORE 生命周期的显式驱逐证据；历史 GPU 输入不具备。
- INSUFFICIENT_EVIDENCE：缺观测、顺序或映射；不得把缺记录写成未创建。
- COVERED_AT_NEXT_LOOKUP：只证明该次前缀命中覆盖该块，不证明持续驻留。

事实与解释：任务、游标、调度区间和后续前缀为事件证据；游标抑制枚举为源码支持的局部解释。重复工作差不能自动归因于同伴驱逐，更不能直接换算墙钟收益。
