# 资源恢复后的固定运行入口

状态：本地准备完成；r02尚未上传或执行，0.95/0.90性能仍UNRUN。
使用调用者已配置的SSH认证，任何凭据均不进入本目录。

```sh
python3 prepare_remote_20260908.py
python3 -u execute_and_readback_20260908.py
python3 analyze_kv_budget.py
```

先从本目录执行stage脚本。它核对GPU空闲、新建独立远端r02目录、上传冻结包并比对
SHA256。只有上传成功才运行driver；driver固定95→90→90→95，每项完整回传后才继续。
全部结果回传后再运行分析。不要执行r01 driver来复用失败cell。

远端目录 `/root/autodl-tmp/moe-kv-budget-20260908-r02`，Python为
`/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python`；目标仍是授权端点
`connect.westd.seetacloud.com:37116`。若端点变更，先核实用户提供的新目标与环境，
不得静默换服务器并沿用旧GPU资格。

冻结执行包846345bytes，SHA256
`e9649d8314d593d9c0b5a879483e471d8ddda7bd57482aa3acf9ead6d9c2880b`。
stage/driver/分析脚本在本地使用，不修改包内实验源码。一次查询失败不是重启依据；
已启动的任务必须先查原PID和终态。stage脚本遇到已有目录会停止，避免覆盖唯一数据。
