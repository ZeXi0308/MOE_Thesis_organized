# 首次实际加载尝试

状态：`IMPLEMENTATION_FAILURE_BEFORE_WEIGHT_DOWNLOAD`。子进程84781退出1，0条请求，0个下载分片；不是OOM，也不是调度或模型家族NO-GO。

父进程先等待共享GPU任务完成，连续3次空闲检查后启动。真实vLLM已解析Qwen架构、构造模型并进入自定义loader。`download_model()` 将 `model_config.model_weights` 的非空默认值误判为覆盖配置，在调用串行iterator之前抛出 `ValueError: model_weights override unsupported`。错误在本轮loader参数检查，尚未检验真实权重加载、主机54GiB master峰值、48层参考或请求执行。

已保留 [完整日志](readback_r01/launch/run.log)、[进程终态](readback_r01/launch/launch.json)、[GPU与主机采样](readback_r01/launch/hardware.jsonl) 和全部运行源码。33个采样仅见本轮PID84781，NVML总used峰值约4.06GiB、cgroup anon峰值约2.00GiB；这些不是完整模型峰值。父进程209.60s包括等待；子进程约16.60s。回读42文件、4,017,504 bytes，SHA `1e4b9c5170a4a371f05585760d4902d38fa4dbe12893c2c9846d8ae7623b7acf`。

唯一修正：允许 `model_weights=None` 或与已核验配置的本地模型目录完全相同，仍拒绝其它权重来源。使用实际安装ModelConfig源码验证默认值，再于新的attempt02目录执行同输入/预算；原始包、source、CPU夹具及本轮证据保留不改。

实际源码定位补充：安装的 `config/model.py` SHA `7d52e060db54067a21591882bd6e3c2dd2486ac9041dde1117f97023e71bf89f` 第116行默认是空字符串，本地模型在895–896行直接返回；904行赋模型URI仅适用于对象存储分支。因此上文“非空默认值”解释不准确，实际是 `"" is not None` 触发错误。修正须接受 `None`、原生空串默认及同一本地目录，并拒绝其它来源；仅接受同目录仍会重复失败。
