# 进入 GPU 前准备记录

2026-09-05；`agent/publish-current-moe-code@2a37765fe522b1d74609a686f1d327ede7619a50`。
开始工作区只有上一轮新增的 admission_capacity 实现与阻塞记录。

**Verdict：`PREPARED_CPU_CHECKED_GPU_UNRUN`。首轮 custom runtime 的输入、入口、预检、
原始记录和结果分析已接通；科学实验仍为 `UNRUN`。**

本轮沿用已读 current/ideas 权威入口、RouteShape-SLO 状态、Expert-Saturation proposal
及 N0d v3 证据边界。唯一问题是“GPU 可用后能否直接开始真实容量实验”，最弱环节是
输入是否可固定复用、失败与未完成结果是否仍能正确分析。本轮没有重新连接远程 GPU。

## 已完成

- 离线读取固定 OLMoE revision 的真实 tokenizer；选定 16 条 WikiText 文本，各 128 tokens。
  **16/16 文本 SHA 与 token-ID SHA 均匹配已有 manifest**。精确 token IDs、request/document
  身份与两条具体到达序列保存在 [prepared/workload.json](prepared/workload.json)。
- [prepared/config.json](prepared/config.json) 固定 cap=2/4/8、16 输出 tokens、steady/bursty、
  OFF/ON、两轮反序重复，共 24 个 cell。两种到达方式保持相同首尾到达时间，均为 0–1.5 秒。
  TTFT=5 秒、mean TPOT=0.2 秒只是待实测校准的探索起点，不是业务 SLO 或研究结果。
- 离线预检通过：Python 3.9.6、Torch 2.8.0、Transformers 4.57.6；OLMoE 16 层、64 专家、
  top-8、4096 上下文，DynamicCache 接口存在。三个权重 shard 可读，合计 13,838,721,960 bytes；
  **仅检查存在与大小，没有加载或重新哈希权重**。详情见 [prepared/preflight.json](prepared/preflight.json)。
- `--prepared-dir` 可直接复用输入，不允许同时覆盖实验参数。文本/token/arrival 漂移会在创建
  运行目录与加载模型前被拒绝。单 cell 600 秒、整次执行 1200 秒预算在调用边界检查。
- 离线分析从 raw 重新计算 TTFT/TPOT/ITL、goodput 和达标比例；保留缺失、失败、未完成；
  比较相邻 OFF cap 的边际响应，单独报告 ON 的 U/C 和 OFF/ON 轨迹差异。没有 cell 时返回 UNRUN。
- 修复提前 OOM 的会计路径：全部计划请求留在 raw，未到达请求另列；不延长真实观察时间，
  也不因未来请求存在而丢失当前失败/未完成指标。另修复到达边界处可能出现负 sleep 的小竞态。

## 实际验证

**15 项定向测试通过**，范围仍限接纳/身份/计时/会计；其中 tiny 随机 OLMoE forward 是 CPU
工程验证，分析器数据是明确标注的 synthetic fixture，均不作为性能证据。

另外实际执行了：离线输入准备、准备目录复用、token 与 arrival 两种漂移拒绝、缺 CUDA 入口、
无 cell 分析。复用前后 workload 完全一致；缺 CUDA 没有生成 curves，缺 cell 没有生成最佳 cap。
验证原始输出见 [verification.log](verification.log)。没有扩大历史审计、创建 Controller 或改写旧 artifact。

| 固定字段 | 当前结论 |
|---|---|
| Evidence type | CPU 工程验证 + 实际离线 tokenizer/cache 检查 |
| What was measured | 输入身份、接纳状态推进、请求会计、分析路径及失败边界 |
| What was not measured | 预训练 OLMoE GPU 响应、统计成本、U/C 动作增量、原生后端、EP |
| Strongest baseline | 静态 cap 扫描已准备；ordinary-state 动态基线尚待首轮数据后实现 |
| Oracle/headroom status | UNRUN；观察到的最佳静态 cap 不会被标成 action Oracle |
| Claim ceiling | 进入 GPU 前的工程准备完成；没有容量或方法 GO |
| Failure category | 当前本地无 CUDA；GPU/BF16/显存/进程隔离需在目标机器实际检查 |
| Resurrection condition | 无科学判死；可用 GPU 与相同模型缓存即可进入实跑 |
| One next smallest experiment | 执行这份已准备的 24-cell 小扫描；出现预算/OOM/退化 SLO 时保留全部结果再解释调整 |

直接回答：**首轮 custom runtime 在进入 GPU 前可完成的准备已完成。U/C 是否改变最佳并发
选择仍未验证；A/B/C 路线要由真实 GPU 数据决定。** 执行与分析命令见
[入口说明](../../../experiments/admission_capacity/README.md)。
