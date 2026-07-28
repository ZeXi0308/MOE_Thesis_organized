# Energy-SLO

Energy-SLO 目录已按两条 formulation 拆分，协议、代码与结果分别就近保存。

| 分支 | 当前边界 | 目录 |
|---|---|---|
| Route-row FP8 / 单卡能耗 characterization | 有真实单卡微基准；没有 arrival、KV decode、P99 或系统 controller 证明 | [`route_row_fp8/`](route_row_fp8/) |
| JouleQueue | v1 Phase 4 blocked；无正式科学结果 | [`joulequeue/`](joulequeue/) |

两个分支都不能把单卡功耗或预量化 GEMM 数字外推为 serving Energy-SLO 结论。
