
实际启动补记：1789356674现场前A四格全终态/释放、GPU空、90GiB与数据盘7.14GB余量通过，monitor77578、worker77591首次启动。1789357019现场第1/16片下载1,840,250,880B，数值/性能仍未开始；当前RUNNING_LOADING。入口及科学配置保持封存；观察SSH为只读，断连不自动重跑。


## 观察连接中断，远端终态未验证（2026-09-14）

最后确认现场1789357367.965：worker77591存活，首分片3,999,417,504B下载及SHA通过，数值资格/性能均未开始。后续四次SSH检查均在认证前关闭；调试确认TCP连通，但远端未完成SSH标识/密钥交换。临时ControlPath均已不存在。当前不能确认原任务仍在运行或已退出，也不能确认释放GPU；没有重启、重传输入、删除分片或改动冻结条件。原任务终态须待实例状态/连接恢复后读取launch、worker、flock及GPU现场。


连续三轮goal接续均在同一SSH握手阻塞，本轮仍exit255；目标标记BLOCKED_CONNECTIVITY，科学任务未完成、远端状态UNKNOWN。当前无本地后台观察器。恢复连接后先读原launch/receipt及PID77578/77591身份与GPU/flock现场，再归档原结果；不能仅因观察失效启动新attempt。


## 恢复访问后的实际核对（2026-09-14 22:32）

当前连接已是GPU-bd5e9bb9-f98b-db5b-cc1c-5857c39f0bdc，容器PID1于22:28:39启动；原GPU-70fa…不同。当前77578/77579/77591均不存在、无匹配Qwen进程，GPU空且共同锁可取。原r03仅2完整校验/消费分片、2325源张量；第三片残1,934,622,720B。最后硬件记录1789358407.614766，anon8,765,890,560B，OOM计数0；这不是之后退出原因的证明。原RUNNING/INITIALIZING与仅MONITOR_STARTED/CHILD_STARTED回执保持，退出码/确切原因UNKNOWN。0数值资格、0性能，归类INTERRUPTED_DURING_LOADING_NO_TERMINAL_RECEIPT。

63成员归档4,368,384B/SHA844fe211ca304eb63f6739def80a3ffc66625ca3deadb22931bd6468da08e91b，远端原件前后稳定，回读61载荷及38冻结输入逐项校验通过，见readback_20260914_r01和readback_verification_20260914_r01.json。旧残片保留。下一使用新attempt并记录新硬件；先重验环境/空间，等待已登记短组，不复用原目录。
