# Qwen 第二次资格化：下载网络不可达

`BLOCKED_NETWORK_BEFORE_WEIGHT_CONSUMPTION`。r02已通过修正后的原生空串配置检查并调用串行加载器。首分片请求在约120秒后报 `URLError(OSError(101, 'Network is unreachable'))`；0字节、0个已消费tensor、0条请求。不是OOM或模型执行失败，48层数值和完整内存峰值仍未测。

子进程85130已退出1，父进程142.03秒含跑前等待；原始零字节文件、空内容SHA与失败原因保留在 [loader receipt](readback_r01/results/loader_receipt.jsonl)，完整终态见 [launch](readback_r01/launch/launch.json)。回读43文件、4,027,547 bytes，SHA `a2930e44c58e919345b0ea1fcfc6a32156e64ba49712c9ce986bdc530f900e7a`。

实际同机CPU探测表明，匿名HF镜像与ModelScope可读取该首分片的前1024字节，前缀SHA相同，声明总大小符合冻结清单；ModelScope config完整SHA亦相同。这只是可达性证据，不能代替完整分片SHA。接续只修改下载传输来源到同一HF冻结revision的镜像路径，保留完整源/目标覆盖与16片全SHA验证，其余输入、算法、资源和参考流程不变；无需再次消耗GPU时间验证不可达的源站。
