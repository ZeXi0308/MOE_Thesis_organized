# 新端点上的固定运行入口

状态：冻结包已上传并通过 SHA256 校验；后台执行流程已启动，当前等待软件和模型完整。
本地 runner PID 为 21199；实际状态见 `await-readiness-state-20260910.json`。
四项 GPU 测量尚未开始。当前已有 runner，不要另行启动 driver 或第二个 runner。
使用调用者已配置的 SSH 认证，凭据不进入本目录。

当前入口为 `await_readiness_then_run_20260910.py`。它每45秒检查安装成功、固定包版本、
原模型下载进程及完整分片；就绪后校验权重SHA256和GPU身份/空闲，再依次运行原driver
和分析。每项仍先完整回传再继续。失败后保留状态并停止，不自动重启失败实验。
`await-readiness-runner.*.log` 为入口日志，`await-driver-20260910.*.log` 为后续driver日志。
当前临时SSH认证helper供该后台流程使用，完成或明确停止后由执行者清理。

先由执行者核实新端点的 GPU 身份、空闲计算进程、运行环境与 pinned 模型缓存。
新端点是 `root@connect.weste.seetacloud.com:11155`，独立远端目录为
`/root/autodl-tmp/moe-kv-budget-20260910-r01`。Python 路径暂用
`/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python`，需先核实存在且环境合格；
两份本地驱动必须使用同一已核实路径。冻结包要求 GPU 0 和
`HF_HUB_CACHE=/root/autodl-tmp/hf-cache/hub`。详见 ADDENDUM.md。

下面保留原始分步复现命令；当前后台流程正在管理后两步，不应并行手动执行：

```sh
python3 prepare_remote_20260910.py
python3 -u execute_and_readback_20260910.py
python3 analyze_kv_budget.py
```

stage 脚本再次核对 GPU 空闲，新建远端目录，上传冻结包并比对 SHA256；
遇已有目录停止，避免覆盖原始数据。只有上传成功才运行 driver。
driver 固定 95→90→90→95，每项退出后完整回传、核对归档 SHA256、读取核心 raw
与暖机数据，然后才启动下一项。失败 cell 完整回传后停止；查询失败只观察原 PID，
不能当作重启依据。不得用旧日期 driver 的 resume 代替新 attempt。
全部结果回传后再运行分析；分析默认读取本目录 gpu_results，并只写新的 analysis。
分析复用同级 20260908_native_preemption_r01 及其既有本地 helper 依赖，未改科学逻辑。

冻结执行包逐字继承 r02：846345 bytes，SHA256
`e9649d8314d593d9c0b5a879483e471d8ddda7bd57482aa3acf9ead6d9c2880b`。
包内 DECISIONS.md / ADDENDUM.md 是 20260908 冻结协议与历史来源；
本次端点、运行资格与状态以本目录两份同名文档为准。旧执行记录不作为本次证据。
