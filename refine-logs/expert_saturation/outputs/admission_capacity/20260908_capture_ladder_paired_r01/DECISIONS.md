# 同次引擎中的接纳档位对照（执行前冻结）

2026-09-08。继承 `agent/publish-current-moe-code@2a37765` 和已有未提交准入实现。
本记录替代 Sep8 `step_cost_surface_r01/next_experiment/DECISIONS.md` 的执行方案，
原方案与原始数据均保留。冻结时本实验 GPU 执行数为 0。

研究问题：在相同原生引擎和完整请求成本下，把普通 ITL 反馈档位从
`[8,12,16,32]` 改为 `[8,16,24,32]`，是否改善请求结果，并超过同次最强已测静态点？
最弱环节是观测的捕获桶成本能否通过真实接纳动作进入请求收益。

## 继承与纠正

已读 `docs/current/README.md`、`docs/ideas/README.md`、准入实验 README、
Sep6 native knee/policy 原始配置与数据，以及 Sep8 成本重建报告和冻结方案。
没有已验证的专家感知方法收益。Sep8 是历史数据重建，不能把相关性提升为因果解释。
自然长上下文容量实验仍保留 UNRUN，本轮只推进这一条档位对照链。

旧方案 F2/F3 只比较跨日历史旧规则，且 `--caps` 同时改变预热路径；旧 F1 又将
按实际 width 筛出的历史平台与整个 static24 episode 中位数相比。
这些是具体设计风险，故在任何新 GPU 数据产生前改为同次对照。

## 唯一实验与固定项

- 两个顺序执行的新引擎。每个引擎使用共同预热集合 `[8,12,16,24,32]`。
- 每个引擎 8 个 arm：static8/12/16/24/32、shadow32、old-feedback32、aligned-feedback32；
  各做 steady/bursty，共 16 个测量 episode；两引擎共 32 个。
- shadow 使用旧档位并保持实际 cap32，用于观察普通反馈采集的成本。
- 第二个引擎完整反转第一引擎的 16 项计划。共同预热顺序不变。
- 复用 Sep6 native knee 的 32 条文本、128 prompt/128 output tokens、seed 20260905、
  steady 50ms / bursty 每组8条的冻结到达序列。它们是重复文本，不是新独立样本。
- OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`、BF16、vLLM 0.26.0，
  engine32、max_model_len256、token预算1024、显存预算0.70、FCFS、无前缀缓存、
  同步调度、stream_interval1。主 SLO TTFT 200ms / 平均 TPOT 9ms；参考 5s/0.2s。
- 复用原反馈阈值、四步窗口和非抢占实现。每个 episode 排空后重新推进自己的 KV、
  输出、batch、队列和完成时间。没有新 predictor 或专家信号。
- 远端先读 GPU 型号、空闲进程、软件/模型缓存。若非同型号 RTX5090 或软件版本改变，
  先记录为运行域变更；不悄悄套用旧毫秒区间或安装另一套环境。

## 分母、判读与停止

总服务时间 = 同一 host 时钟的 episode 终点减原点；其中包含排队、prefill、decode、
采集、反馈决策和循环成本，不能再加一次局部税。goodput = 同时通过 TTFT 和
平均 TPOT 的已完成请求数 / 总服务时间。失败、未完成和所有计划请求均保留。
平均 TPOT 不承诺逐 token ITL；P99 仅为这个有限样本的描述。

主对比是每引擎/每regime aligned vs old 的 goodput、达标数、TTFT/TPOT及ITL。
再比较同次五个 static 的最高 goodput；它是探索性 hindsight 静态上包络，非动态 Oracle。
报告所有四组配对，不聚合掩盖变号。forward 固定为 canonical，reverse 为受控重复。

- 若两规则实际接纳/宽度轨迹未分开，判当前运行域缺动作暴露，不归因于档位收益。
- 若对旧规则改善但未胜同次静态上包络，只形成普通反馈的工程改进线索。
- 只有同一regime两重复均相对 old 及 static 有 >=3% goodput 净增益且连续时延不显示
  阈值噪声主导，才保留为探索信号；小于3%仍报告原数，不写方法成立。
- 若变号或接近噪声，唯一下一步是一次相同配置受控重复；不先解释专家机制。
- 若稳定无收益，停止本运行域的这次档位替换，不调阈值/文本抢救。
- 历史 F1 `[9.30,9.51]ms` 仅检验绝对数值复现；不通过不能单独判死阶梯。
  结构诊断须按实际 width 与上下文区间在同引擎比较；桶未被实际日志确认则不报 padding
  的硬件解释。仅靠本次两条 ladder 仍不能分离档位对齐和动作大小/排空时延的贡献。

证据上限为单模型、单卡、原生 vLLM 同步 in-process 请求测量；未测 U/C 增量、
任务质量、自然 EOS、第二模型、EP 或生产服务。没有 full-request Oracle。
负结果复活条件仅为新自然运行域产生可执行且可重复的档位收益空间。

## 执行与回传合同

本地 `execution.tar.gz` 包含必要代码、冻结输入、计划和必要 patch，不含凭据或权重。
预定远端目录 `/root/autodl-tmp/moe-capture-ladder-paired-20260908-r01`，必须新建。
GPU解释器预期 `/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python`，待只读核实。
在解包根目录分别运行：

```bash
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u run_block.py forward
# forward 数据回传本地、确认完整可读后，才执行：
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u run_block.py reverse
```

一次调用只启动一个新引擎，不自动进入下一引擎。stdout/stderr 分开保留，退出状态落盘。
全部实际采集的 warmup raw、测量 raw、失败 raw 与逐cell配置/检查均保留。
运行中按已完成cell增量拉回；每引擎结束立即完整回传到本目录 `gpu_results/<block>/`，
以及该 block 的日志、环境和退出信息。用一份归档校验值确认传输，再读取核心 JSON。
未确认本地完整前不删除远端数据。全部运行保留，不替换 canonical、不覆盖旧 raw。

当前远端凭据使用被自动审批拒绝；本文件只冻结可审阅的执行范围，没有启动 GPU。
