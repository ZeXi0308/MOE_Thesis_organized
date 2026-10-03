# 自然长度 prefill 尾块：已准备，GPU未运行

当前问题：不把输入统一截成128/2048后，原生FCFS是否仍出现多块prefill最后一块不超过32 tokens的现象。

已实际完成离线输入、执行脚本、分析脚本与两块prepare-only检查。16篇完整文章共29103 tokens，长度733–3011，与上一轮16个document IDs不重合。原文/token hash和上下文约束检查通过。两次fresh-engine执行各1组预热、1组正式测量，计划保持native1024/FCFS/cap8/threshold0及固定128输出。

**没有GPU结果。** 冻结包上传两次均在创建进程前被自动审批超时拦截；随后两次只读SSH检查均返回Connection closed，退出255。没有启动实验，不能推断远端当前GPU占用。

后续复核发现同一入口在SSH认证前返回HTTP502，当前路由经本机utun4隧道；具体故障点未确认。需恢复可用SSH路径，不能据此认定GPU主机关机。新证据见[连接路径补充](ADDENDUM_NETWORK_PATH.md)。VPN与路由均未修改。

- [冻结问题与执行规则](DECISIONS.md)
- [实际输入及离线准备](INPUTS.md)
- [运行包](execution.tar.gz)与[包记录](EXECUTION_PACKAGE.json)：168533字节，16文件，SHA256 `92f78162aa1160fcd6c376c3beb463f33135ad610a0695fc51f70f924560ac41`
- [当前状态](STATUS.json)、[本轮记录](REPORT.md)、[审批超时记录](UPLOAD-approval-timeouts.json)、[SSH观察](REMOTE-observation.json)

下一步仍是执行已冻结的原生基线。连接和执行权限恢复后，先确认本轮目录和结果是否存在，再上传/校验/prepare，逐块执行并回传；不改输入、阈值或扩大策略矩阵。

当前长期目标状态为blocked：第三个连续回合的SSH复核仍失败。实验本身保持UNRUN，配置和运行包未改变，恢复连接后继续。
