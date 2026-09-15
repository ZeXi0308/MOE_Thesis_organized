# JouleQueue v1

- 冻结协议：[`JouleQueue_Phase2_冻结实验协议_2026-07-22.md`](JouleQueue_Phase2_冻结实验协议_2026-07-22.md)
- Code Review：[`JouleQueue_CodeReview_Phase4_2026-07-22.md`](JouleQueue_CodeReview_Phase4_2026-07-22.md)
- 代码、配置与测试：[`experiments/`](experiments/)

JouleQueue 的 native route/expert capture 复用已归档的 [`CJC capture producer`](../../../archive/receiver_aware/cjc/experiments/)；这是显式代码依赖，不代表 CJC 重新成为活动方向。

状态：Phase 4 blocked。缺少真实 activation/surface producer、独立 input-event 能耗样本和闭合 serving oracle，因此不能给科学 Go/No-Go。
