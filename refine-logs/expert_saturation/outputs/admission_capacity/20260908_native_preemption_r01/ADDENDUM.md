# 本地监控修复

首项 `repeat0-native32` 通过冻结执行包启动后，本地driver的查询字符串在添加
异常重试时被错误增加缩进，远端状态查询得到IndentationError。远端模型进程独立运行，
本错误不在采集器或GPU执行包内。

直接检查同一launcher PID22900及其child22901的权威终态：EXITED、returncode0、
32/32 COMPLETE、实际抢占2次，GPU已空闲。随后仅停止本地driver PID29822的错误查询
循环，保存原driver与状态快照，修复查询字符串并增加显式`--resume`接管同一PID。
首项不重新启动、不替换数据；先回传原结果，再按冻结顺序运行剩余三项。

保留 `execute_and_readback_initial.py`、`execution-state-before-monitor-repair.json`。
修复后用AST提取并编译了全部3段内嵌远端Python，验证所发现的缩进问题。
本轮所有GPU实验仍使用同一未修改执行包，输入、采集器、公式及四cell顺序不变。
