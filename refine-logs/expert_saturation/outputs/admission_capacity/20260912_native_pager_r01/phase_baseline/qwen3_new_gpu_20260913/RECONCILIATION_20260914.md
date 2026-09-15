# westc r02：旧加载中断已核对

2026-09-14只读回查当前westc:53036，GPU UUID与原新主机相同。原monitor5090、parent5092、worker5127均不存在，未发现对应Qwen运行进程。

结论是 **INTERRUPTED_DURING_LOADING_NO_TERMINAL_RECEIPT**。原launch仍RUNNING、comparison仍INITIALIZING，原件不修改。日志确认4个完整分片经SHA验证并消费，共4845源张量；最后事件为第5片DOWNLOAD，残片2,515,533,824B保留远端。layer47张量/诊断/归因gate、数值raw和性能格均未产生。

monitor只留下MONITOR_STARTED/CHILD_STARTED，最后进度约6193.85s；未取得EXITED、退出码或结束原因。进程现在不存在不证明此前被谁停止、OOM、超时或机器关机。旧采样不能用作成功加载耗时，也不补写科学NO-GO。

[现场核对](attempt02/readback_20260914_r01/inspection.json)；[原件manifest](attempt02/readback_20260914_r01/manifest.json)。55成员（53份原文件及核对/manifest），4,824,076B，SHA256 `46df6a2f009d0b3f9610429a69e10c6ef215cf81a5555c7b5ad8e3e00de976bb`。归档前后原件size/SHA稳定，本地逐项核对相同。模型残片和Python缓存未回传、未删除；不重新运行该目录。

本尝试是westc的qwen3-new-gpu目录，区别于returned weste的qwen3-localized-static目录；后者终态不可据此追认。

唯一接续：新attempt03复用原模型、17份runtime、数值条件及通过后的静态32/16/16/32，只补完整生命周期共享锁与脱离观察连接的终态记录。当前CPU准备，上传/GPU为0；启动前检查当前内存、磁盘、传输与协作队列。原数值门槛不放宽，不新增调度器或参数扫描。
