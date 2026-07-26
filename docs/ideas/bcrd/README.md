# BCRD：Batch-Conscious Replica Dispatch

状态：`DESIGNED_AND_IMPLEMENTED / NOT_FORMALLY_RUN`。

BCRD 研究 fixed replica set 内 contribution-to-replica assignment 与 bounded seal time，目标是判断 least-load 是否因打碎同一 expert 的 rows 而浪费足够多的批效率和 SLO 容量。

- [研究设计与三门验证协议](研究设计与三门验证协议.md)
- [实验代码、测试与运行边界](experiments/README.md)
- [当前总裁决](../../current/README.md)

现有 smoke 只检查会计与决策分支，不能作为正式科学结果。
