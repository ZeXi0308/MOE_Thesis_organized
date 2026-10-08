# G 执行计划组合与 KV 容量

重审判决为**因关键对照未运行而暂存联合方法；问题总体尚未确定**，详见 [研究卡及当前判决](DECISION.md)。已执行容量包络与compact真实启动：可用36827 blocks覆盖high全请求最大需求34430，因此该端点在冻结负载上的KV瓶颈被排除。dense因共享锁忙未运行，完整服务比较未运行。原停止判决保留在`archive/`，不冒充实验证伪。

最新资源状态（2026-10-08 21:50）：连续三轮公共锁不可用，当前其他Python进程占用88288 MiB，目标已因资源阻塞挂起推进；dense仍未运行。未初始化CUDA、未追加GPU实验，也未改变科学判决。资源释放后唯一待补项仍为冻结dense启动。

- [假设与竞争解释](hypotheses.md)、[主结果与关键副作用](service_comparison.md)
- [候选与成本表](candidate_costs.md)
- [冻结实验协议](research_contract.md)、[近邻工作与半页贡献假说](related_work.md)
- [当前源码审计](backend_audit.md)、[既有环境与轨迹审计](local_environment_audit.md)
- `native_sources/` 是当前远端源码只读快照；`evidence/source_sha256.json` 固定其版本。

## 本地 CPU 复现

从毕业设计目录执行；不加载模型，不导入 CUDA 运行时。两项历史复算依赖原有相邻 `D_prefill_budget_20261004` 目录。

```bash
python3 -B G_execution_plans/extract_historical_feasibility.py
python3 -B G_execution_plans/extract_historical_service.py
python3 -B G_execution_plans/capacity_envelope.py --startup G_execution_plans/evidence/review_20261008/compact.json --startup G_execution_plans/evidence/review_20261008/dense.json
python3 -B G_execution_plans/analyze_startup.py
python3 -B G_execution_plans/validate_cpu.py
python3 G_execution_plans/moe_probe.py --dry-run
python3 G_execution_plans/startup_probe.py --plan native --dry-run --output /tmp/g-native-dry-run.json
```

startup 输出路径必须尚不存在。`capacity_envelope.py`只按冻结输入及已核对语义计算逻辑容量界，不模拟时间或预测吞吐。`analyze_startup.py`复算已归档的真实启动。早期CPU检查只验证语法/锁边界；真实运行与失败状态另见`evidence/review_20261008/`。该目录保留实际执行的两份脚本，配置和SHA见`evidence/review_endpoint_config.json`，容量源码的SHA见`capacity_source_sha256.json`。

## 远端最小命令

使用原服务器、模型和E已有RECORD-only Torch视图，不改共享安装。下面复现本次compact；当前唯一值得优先补的单元是同配置`--plan dense`。本次已实际运行compact；不要自动扫描或等锁重试。

```bash
ssh -p 53005 root@connect.westd.seetacloud.com 'mkdir -p /tmp/G_execution_plans'
scp -P 53005 G_execution_plans/startup_probe.py G_execution_plans/g_worker.py G_execution_plans/moe_probe.py G_execution_plans/runtime_bootstrap.py root@connect.westd.seetacloud.com:/tmp/G_execution_plans/
ssh -p 53005 root@connect.westd.seetacloud.com
cd /tmp/G_execution_plans
PYTHONPATH=/root/autodl-tmp/moe-e-restore-choice-20261004/package_view:/tmp/G_execution_plans LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/torch/lib timeout --kill-after=10s 420s /root/miniconda3/bin/python startup_probe.py --plan compact --output compact-new.json
```

两探针均先非阻塞取得公共锁，忙时退出75、零CUDA初始化；获锁后检查无其他计算进程。不要加等待循环。每集合新进程，阶段预算原定15分钟GPU，本轮compact进程45.47秒，dense尝试0.115秒锁忙退出。`native`不是本阶段需要增加的单元。

`moe_probe.py` 比较 native/tuned Triton 与两个明确 tile，固定同一输入/权重/路由，先用独立 FP32 参考以 BF16 容差 rtol 0.03、atol 0.01 检验，再用相同 graph replay 方法计时。synthetic 仅算子自检，不能证明真实路由竞争力；真实数据使用 `--bundle layer.pt`，字典须包含 BF16 `x [M,2048]`、`w1 [64,2048,2048]`、`w2 [64,2048,1024]`、整数 `topk_ids [M,8]`、FP32 `topk_weights [M,8]`，M 为 1/32/128，不重算或归一化路由。此 bundle 尚未采集。functional 算子的 scratch 分配不同于服务 modular workspace，不能拿它估 B/C 的内存收益。

`startup_probe.py` 每次只加载一个全集合，compact/native/dense为10/51/67个请求桶，最大512；桶数不是graph数，compact实际19个graph，dense尚未测。固定BF16、TP1、0.9总显存策略，KV自动分配；当前实测TritonExperts，CUTLASS仍未验证。结果记录实际backend、容量证明的runtime条件、workspace/allocator、估计、capture差额、清cache前后、KV storage/blocks和耗时。

这是观测原型，仍按现有运行时“估计→KV→capture”执行，不是精确组合优化器。不会在线删图、扩KV、增加conditional graph或逐层同步。没有强基线残余损失与新状态决策价值之前不开发选择器；当前证书是机会空间诊断，不宣称论文贡献或服务性能上界。
