# 唯一中间静态预算512：mixed请求域的最小对照

2026-09-08，执行前冻结。所有GPU测量与预热均为UNRUN。

本轮承接 `../next_prefill_budget/REPORT.md` 已声明的唯一下一实验：上一轮256预算
减少部分尾部token间隔，但增加混合步骤暴露、TTFT与整批完成代价。这里只检查
预先选定的中间静态预算512能否缓和该权衡，不建立新Idea，不实现Controller，
不扫其它预算、阈值、seed或输入。原始256/1024数据与已执行源码保持不变。

唯一问题：在相同自然mixed请求域，512相对于同引擎1024基线，是否同时改善主要
完整请求指标，还是仅重新分配ITL、TTFT、平均TPOT与完成时间的代价？
最弱环节是简单静态中间点的完整请求成本，而非专家信号或预测器。

## 冻结运行

从上一轮实际执行包复制mixed输入、runner、capture、metrics与launcher。
`source_origin.json`绑定源执行包及回传包SHA256、复制文件原hash；`midpoint.patch`
保留最小修改。自然mixed的config/workload、runtime三文件及launcher保持逐字节相同。
只改变预算候选256→512，并收缩为mixed两个条件；不复制all_short再运行。

- 模型OLMoE-1B-7B-0924，revision/tokenizer revision均为
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`，BF16，vLLM0.26.0。
- 相同16篇真实文章及实际token IDs，输入长度 `[128,2048]×8`；固定128输出token，
  seed20260905，50ms到达序列，ignore_eos=True；不重选或拼接文本。
- 编译配置保持cap8、max_model_len4096、max_num_batched_tokens1024、chunked prefill、
  gpu_memory_utilization0.70、同步FCFS、禁前缀缓存。排空episode后才改变scheduler内部
  `max_num_scheduled_tokens`为512或1024，启动与每个episode前核验字段和容量契约。
- 每个新引擎先按mixed512、mixed1024执行两个共同warmup并全部保留；forward测量
  mixed512、mixed1024；reverse仅将测量顺序反转。合计4测量+4预热，64次正式请求执行，
  输入仍只有16篇文章，不能称64个独立样本。
- 完整保存所有正式、预热及失败raw、空输出step的receipt、环境/软件/源码hash、
  stdout/stderr、OS退出码和命令；非抢占、KV调整及每step既有decode推进保护全部保留。

## 判读边界与停止条件

主观察为全体及短/长请求的连续TTFT、平均TPOT、ITL分布、完成数与整批墙钟。
参考SLO保持TTFT5s/TPOT0.2s，不根据结果换SLO，不能用参考SLO全面达标掩盖连续指标恶化。
主比较始终是每个新引擎内的512与1024；不把旧256数据当本轮同次对照。

先核验1024实际暴露超过512的step，以及512实际限制token预算；动作未暴露则报告
未能检验该预算差异。出现抢占、KV调整或已有decode未推进时保存失败并停止，
不把暂停decode造成的表面ITL改善算作收益。所有成本继续进入请求分母。

512如果没有在两个引擎内同时改善主要完整请求指标，只记录观察到的权衡，不制造
method GO。不挑某个百分位替代其它恶化的指标；符号不稳定或接近已有运行波动时
保留不确定性。本冻结包不授权第三预算或追加重复；任何下一步必须依据完整保留结果
重新收敛为一个问题。即便512稳定占优，也仅说明此域的简单静态配置更合适。

最强基线为同引擎1024静态点；不是全局最优、动态Oracle或专家感知方法。
本轮不测任务质量、输出一致性、GPU kernel专属时间、第二模型、HTTP生产服务或多卡EP。
各episode独立推进KV、batch、自由生成输出和完成轨迹；相同文本与seed不保证输出相同。
Claim ceiling为单模型单卡原生in-process下的REQUEST_LEVEL测量/动作响应。

## 执行与保留

使用全新远端目录，先检查GPU独占和软件/模型缓存。每次只运行一个block；forward
完整回传并校验后才启动reverse。launcher保存OS退出码，runner保存阶段raw及日志。

```bash
python run_prefill_budget.py --block forward --output-dir plans/forward --prepare-only
python run_prefill_budget.py --block reverse --output-dir plans/reverse --prepare-only
# 明确执行时使用已核验vLLM解释器：
python -u launch_block.py forward
# forward数据完整回传后：
python -u launch_block.py reverse
```

准备检查不能替代GPU结果；当前4个测量、4个预热全部UNRUN。
