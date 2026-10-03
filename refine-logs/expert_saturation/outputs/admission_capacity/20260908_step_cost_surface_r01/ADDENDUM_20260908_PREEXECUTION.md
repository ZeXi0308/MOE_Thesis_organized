# 执行前纠正：成本重建不等于策略因果证据

2026-09-08。保留 REPORT、分析文件和全部 Sep6 raw。本补充没有新增 GPU 结果。

1. REPORT 第5项将 width17 写成 bucket16 有误：按已记录捕获集合，它属于 bucket24。
   forward steady feedback 的413个 pure steps 中，width17有40步，width10有125步，
   width5有105步；原 mean waste `0.3014023406` 可重算，但归桶解释需更正。
2. `0.301` 是一个 forward steady cell。reverse steady 为 `0.15703125`，
   两个 bursty 分别 `0.0013586957` / `0.0386904762`，不能将0.301视为旧规则稳定常量。
3. 该 forward 的早期降档发生时 actual active 为17；写入target不立即改变执行宽度。
   “沿错误坐标轴控制导致失败”目前只是待检验解释，尚未由真实对照确认为原因。
4. `0.0169–0.1534ms` 是这批 host 时间重建与请求 TPOT 的残差；它不界定运行漂移、
   因果归因和全部统计推断误差。“统计不可区分”也不能仅从中位数比值得出。
5. 旧下一实验缺少同次旧规则，且改caps同时改warmup；旧F1把width条件平台和整个episode
   中位数混比。`--reverse-conditions` 在旧runner中也不会反转全部static档位。
   新执行方案见 [同次配对规则](../20260908_capture_ladder_paired_r01/DECISIONS.md)：
   同引擎两条ladder、共同warmup、完整反序重复。历史毫秒区间仅作数值复现检查。

当前研究问题仍开放：档位替换能否带来完整请求净收益；现有数据只支持成本结构的观察。
