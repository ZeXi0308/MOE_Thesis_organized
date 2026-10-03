# 固定接纳上限下的prefill预算：独立的最小真实动作对照

2026-09-08。GPU UNRUN；此时尚无本实验输出。一次只推进本问题。
本线程认领本目录，来源和与其他进程的边界见上级 `DECISIONS.md`。

问题：在普通全驻留OLMoE运行域，固定最多8条active请求时，收紧每步token预算能否
减少长prefill对已在decode请求的ITL影响，代价是否转移到了新请求TTFT及总完成时间？
它检验一个已知简单动作的实际收益空间，不是新型专家感知算法。

依据：上级32个既有episode的精确时间分解已排除大量步间host时间；aligned steady
两次平均TPOT差异主要落在含prefill步骤的时间区间，但这只是位置关联，因果仍未验证。
因此本次实际改变prefill预算，不以减去该时间桶来制造收益。

## 与已有工作的边界

- 其他进程继续接纳ladder/捕获宽度/同配置复测；本实验接纳上限始终8，不做ladder或hold。
- 自然KV压力已有独立8-cell方案；本实验cap8、输出128，不缩小KV池或expert cache造压力。
- 分块prefill是既有动作。[vLLM 0.26官方说明](https://docs.vllm.ai/en/v0.26.0/configuration/optimization/#chunked-prefill)
  已讨论token预算对ITL、TTFT的影响。本实验不能以再次观察权衡宣称独立论文贡献。
- 直接改变`max_num_batched_tokens`可能改变编译hash和缓冲区；
  [官方配置源码](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/scheduler.py)
  明确将它纳入hash。因此共同引擎保持编译上限1024，仅在episode完全排空后改变
  [scheduler的实际token上限](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/sched/scheduler.py)
  `max_num_scheduled_tokens`。这是固定版本内部接口，实际启动需检查存在和预算生效。

## 冻结输入与运行

复用已准备自然长文章的前16篇，文档/全文hash及实际token前缀逐条核对。
mixed按原顺序取`[128,2048]×8`；all_short使用相同16篇、全部128tokens。
没有跨文章拼接、重复填充或根据性能选文本。每条固定128输出token、seed20260905；
steady每50ms到达，0~0.75s后排空。输入明文身份和实际IDs均在prepared内。

一个新引擎做4个测量episode：all_short256、all_short1024、mixed256、mixed1024。
第二个新引擎完整反序，合计8个测量、128次请求执行；不是128个独立输入。
两个引擎都先按同一上述顺序做4个完整warmup，全部raw保留。
模型revision、BF16、vLLM0.26、同步调度、FCFS、禁前缀cache、engine cap8、
max_model_len4096、max_num_batched_tokens1024、普通显存预算0.70、chunked prefill启用。
不同时修改精度、权重驻留、请求排序或输出停止条件。

1024是沿用已有引擎token范围的简单基线；256是一次较小预算干预。
两档同属公平可执行动作范围，但最高已测值不称全局最优或Oracle。

## 实际观测与判读

主结果为每请求TTFT、平均TPOT、ITL分布、完整episode墙钟和完成数；分别报告短/长输入。
5s/0.2s只保留为参考SLO，旧短请求9ms不作为本长短域收益筛选器。
若需SLO资格化，必须先依据此次连续时间校准并独立重跑，不能原地改标签宣布收益。

- 确认256确实改变了每步prefill token量，以及decode经历的mixed step数量/间隔；
  若1024实际每步都未超过256，这个cell不能作为有效预算干预证据。
- all_short只作为负控候选。若其预算也经常绑定，明示负控不干净，不能强称零干预。
- 每次schedule保留原active decode身份，若被跳过、抢占或KV调整则保存失败raw并停止；
  不通过暂停请求降低表面ITL。若此处失败，先定位动作语义，不判所有prefill调度失败。
- 成本分母保留全部排队、prefill、decode、采集、host开销和失败请求。长prefill产生
  没有输出的step时也记录receipt；不能复用“每步都有output”的旧分析假设。
- 256若稳定改善ITL却恶化TTFT/完成时间，只记录权衡；稳定全面受1024支配则停止当前小预算。
- 接近噪声或符号翻转时只做一次同配置受控重复；禁止逐个改seed/阈值挑正结果。
- 两档静态点足够解释结果时，停在测量/简单配置层；不因此增加U/C predictor。

Claim ceiling：单模型、单卡、原生in-process自然文本混合域的真实动作响应。
未测GPU kernel专属时间、专家信号、动态Oracle、第二模型、生产服务或方法新颖性。
当前8个测量及8个warmup均UNRUN，不能用输入/代码检查替代性能。

## 执行与归档

runner使用本目录runtime内复制的已验证捕获代码，仅增加无输出step的receipt和
预算动作的必要检查，不编辑其他线程的共享runtime。源版本与最小patch随包保留。
新远端目录应为 `/root/autodl-tmp/moe-independent-prefill-budget-01a07d4b-20260908-r01`。
先核对同一GPU空闲和软件/模型缓存，再运行forward；该引擎数据完整回传本地后才运行reverse。
全部正式/预热/失败raw、命令、环境、stdout/stderr、退出状态保留，不删除远端唯一副本。
当前SSH密码被服务器拒绝；等待有效连接信息，不绕过认证，也不与其他进程争抢GPU。

解包后使用已核实的vLLM解释器分别启动单引擎，launcher负责保存OS退出码，runner负责
全部阶段raw与stdout/stderr；不自动连续占用第二引擎：

```bash
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u launch_block.py forward
# forward 全量回传本地并确认JSON/归档完整后：
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u launch_block.py reverse
```
