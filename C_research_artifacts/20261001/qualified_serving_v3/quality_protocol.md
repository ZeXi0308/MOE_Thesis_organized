# 健康任务资格与 APC 保留下的最小准入机会协议

预声明时间：2026-10-01 13:26 UTC。只读已冻结输入、评分源码、V1/V2计划与既有C结果后新建；未连接远端、未运行GPU、未读取任何新 Instruct 生成输出。本目录及C目录层未发现额外AGENTS.md。新机器45495的GPU、锁、版本和模型文件资格由主执行方现场核实，本文件不沿用旧UUID/inode或旧机器性能作配对。下述新增数值门是两模型后续研究的开发筛查，**不改原OLMoE计划的完成/资格判据，不把原计划未设阈值的结果事后改判为无效**。

## 本次只关闭一个不确定性

能否在输出有用、自然结束、APC开启的真实任务中，找到原生调度及简单并发限制仍未解释的完整请求代价？当前不是证明一个已经成立的新方法。

继承事实：旧base OLMoE的BBH27题仅6题正确且0 EOS；长共享前缀BBH对照中，APC已把活跃唯一块峰4096降到1022、抢占4降到0，两臂仅1/32正确。单donor22次真实转移没有主点收益。Past-Future已有对应峰值公式，旧独立包络公式创新主张撤回。这些结果要求保留APC并先验证任务；不能关缓存、只选撞cap题或换seed制造新动机。

本协议分Q0原16题资格、Q1第二类16题资格、P0约32请求原生/强简单机会小块。后一级只有前一级通过才运行。协议编制过程没有下载模型、修改原数据或实现准入控制器。

主执行方新增机器工作分工：旧机原会话的OLMoE-Instruct V2下载继续，禁止重复启动或接管；新机独立准备 `Qwen/Qwen2.5-1.5B-Instruct` 作为第二架构。Qwen复用完全相同的16题`chat_user_content`（八条示例及问题）、最后Answer: cue、gold和采样合同，但必须使用Qwen自身的官方chat template/tokenizer重新渲染并冻结新token IDs及模型revision。不能复用OLMoE的prompt IDs或把其EOS id50279强加给Qwen。Qwen revision、native KV layout和现场hash由新cell冻结；本协议不杜撰这些未核的身份。

每个模型各自应用相同的新开发筛查。两个模型或两台卡只能分别说明任务可用性，**不能跨卡配对速度、吞吐、策略差值，也不能把另一卡的一臂当本卡baseline**。若后续P0在Qwen执行，每个P0策略必须在同Qwen revision、同卡、同物理KV预算内配对。P0的2048可用块是显式约束；实际字节按该模型layout另记，OLMoE的4GiB不移植给Qwen。任何在新输出出现后改输入/template/cap的分支都另立版本并保留原结果。

## Q0：原16 GSM8K输入可用于资格，不能证明质量等价

完整复用 `../20261001_c_instruct_gsm8k_inputs_v1`。模型 `allenai/OLMoE-1B-7B-0924-Instruct`，revision/tokenizer revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`。来源为官方GSM8K test固定commit `b0bb162abedc65e1fdd8e93ed090fd7598ee68bc` 的物理行0–15；保留全部16题，八条原顺序CoT示例、单user chat及末尾Answer: cue均不变。

| 冻结文件 | SHA256 |
|---|---|
| config.json | 0e915e375a02518ead58aeb5ca4087b7336b78d01e3b8ca70c9bb1df29549b1f |
| workload.json | 6e51744ac762adab1fa3bf710efc413283f2a13cd3bb1a0560c841781d4dc11a |
| SOURCE_RECEIPT.json | 7fbdeea8f973c3ca6283e063941ad833f9b523bea6dc1f28e2975fec17914179 |
| ../C_INSTRUCT_GSM8K_ANALYZE_V1.py | 97013bcc91e31dfda37d95521ff8b5b0dba662cba482d9e080cc373fa192021d |

仍用greedy、seed20260905、max_tokens1024/min0、ignore_eos=false、无text stop；16题全在t=0到达，完整drain。APC开启，warmup后清空prefix cache，只测冷起点这一明确状态。其余沿用原固定BF16/vLLM0.26/context4096/32seq/token budget1024。若新机器无法兑现这些参数，先写机器绑定增补再运行，不将不同配置误记为原计划重复。

已有输入几何：prompt660–740、prompt+cap最大1764；独立full-cap和1727块，小于4096；公共前缀626 tokens/39整块，理想共享full-cap上界1142块。故Q0即使成功也只是输出资格，**不可能据此证明原8GiB KV预算下存在声明上界准入压力**。不为了压力改这些题的cap。

评分沿用现有历史last-number解析：删除数字分组逗号，取固定regex最后数值子串，和冻结gold作字符串匹配。保留原始文本、token IDs和每题错误；不另选更有利解析器。中间token检查点只作描述，不据答案提前结束或选择请求。

### 看新结果前冻结的继续门

以下是分配一次后续开发小块的弱筛查门，不是论文质量标准：

- 16/16请求完成，身份、tokenizer、EOS与drain检查均通过；失败/未完成不能从分母删除。
- 0条空白文本输出；自然EOS至少12/16。OLMoE旧分析按其已核的null stop_reason合同；新通用cell同时保存实际native sampling。无text stop且finish_reason=stop时，stop_reason为空或属于模型hf/generation_config声明的合法EOS ID，才归自然EOS；Qwen合法额外EOS不能被误记为非EOS。未知stop另记未资格，不能直接算自然结束。
- 固定评分正确至少8/16，其中正确且自然EOS至少8/16。
- 现有“最后256 token、最短周期1–16”的重复后缀标记至多1/16。

选择8/16正确是“多数或一半任务已可产生可检验有用结果”的探索性下限，不援引原作者成绩，也不保证模型足够好用于投稿。原作者200随机样本/不同cap与本16题不是同一评测。16个源顺序题不独立代表完整任务分布；即使两个策略都8/16，也可能各自错不同题，不能称质量等价。

未过质量门：保留所有题和输出，停止P0，不以调seed、换题、缩长文或看到答案后选cap救结果。只有可直接定位的tokenizer/template/解析实现错误可以另建修复版本，旧结果保留并准确区分INVALID与模型/任务质量负结果。下载未完成、模型哈希失败或CUDA前退出没有生成质量结论。

## Q1：用现有文章补齐第二类任务，不重复下载

第二类固定为**文档摘要**，不是另一套数学题。来源直接复用 `../20261001_c_retirement_fresh_inputs_v1/workload.json` 的 `source_requests[:16]`，eligible ranks832–847，取各行原`prompt`全文。它们是已见开发文章，不能再叫未见确认；摘要任务本身在新结果前冻结。不得因文长、模型错误、生成长度或运行收益改选文章。

使用Instruct自身同一chat template，一个user消息；固定英文指令：

```text
Summarize the article below in three concise sentences. State the main subject and its most important facts using only information supported by the article. Do not add outside facts.

Article:
{original_article_text}
```

greedy、同seed、EOS自然、min0、max512、无text stop。以Instruct tokenizer重新编码原全文加指令；不沿用base模型token IDs作为新chat IDs。若任一完整prompt+512不满足4096，Q1输入资格失败，先报告；不得截断或跳过该文章。Q1为16题全部t=0的native/APC资格单格，仍用8GiB可用KV；无策略比较。

在任何摘要输出出现前，为每篇源文保存**三条核心事实及原文支持句**。事实按固定顺序选取：文章开头首个非标题段的主题/身份事实、随后两个不同的核心事件或属性；只据源文，不能据模型答案反选参考。三条须去重，长复杂句拆为可核验命题；若开头信息不足，按源文顺序继续。参考标注及其哈希应在Q1前冻结。尚未生成的标注文件和摘要输入生成器标为待实现，不能虚构已存在路径。

每条输出按盲化request ID核查：三条核心事实覆盖数0–3；输出内每条可核验事实是否有原文支持；是否出现实质错误的人物/组织/时间/数量/因果归属。摘要“有用”定义为覆盖至少2/3核心事实且无实质不支持事实。自动重合指标如后续ROUGE只能辅助，不能替代事实判断；由单一模型做评分时必须注明judge局限，保留源文、参考、输出和逐项判定供复核。不要使用同一生成答案作为参考。

Q1继续门：16/16完整、0空白、自然EOS≥14/16、重复后缀≤1/16、有用摘要≥12/16。该门在输出前冻结；少于三句或多于三句单独记录，但不以纯格式差异伪装事实错误。Q1未通过则不拼成两类健康任务，停止P0并报告哪个资格失败。

## P0：保留APC的32请求原生/强简单准入机会探针

Q0与Q1都过门后，复用16 GSM8K与16摘要组成32个**唯一请求**；不复制同一题32遍。每个任务保留其原cap1024/512，真实未来EOS和正确答案不输入调度。按`gsm0,summary0,gsm1,summary1,…`固定顺序，一次Poisson到达：NumPy `default_rng(20261001)`，第一项0，后31个间隔为`exponential(scale=0.2,size=31)`累加；保存完整arrivals和NumPy版本/hash，禁止重抽。固定随机发生器实现与结果数组后才能启动。输入不按输出长度排序。

**唯一压力点**：2048 usable block、1 null，沿OLMoE几何即4GiB可用KV；模型权重/BF16/token budget1024/context4096相同。4GiB是主动声明的受限KV运行域，不代表使用全部显存的最佳部署，也不与旧8GiB绝对吞吐拼表。新卡若不能在相同native布局兑现2MiB/块，先核布局并改写冻结资源表，不能直接套旧字节数。只测这一点；没有压力即记录零机会，不继续减KV/增cap/调到达救假说。

只比较两个已经能表达的native策略，均APC on、FCFS、相同prefix cache冷起点、相同必要事件记录：

- N32：native max_num_seqs=32，其余默认固定。
- S16：native max_num_seqs=16，固定并发限制这一强简单参照；16在看新数据前选定，不做网格挑赢家。它不是所有可能简单策略的最优点，也不是完整Past-Future复现。

执行顺序固定N32/S16/S16/N32，逐格独立演化、完整drain，32请求×4=128次执行。两反序对只是开发描述，不是独立episode总体确认。第一个N32完整结果若既无抢占，又无到达后因KV阻塞的真实准入等待，且物理块压力不足以造成动作差异，则**停止剩余三格**：该预声明无机会分支不筛除任何已运行请求、不称S16有效，也不改压力。存在压力才完成剩余三格；不得边看性能边调整并发数。

### 必须记录而不扩成大诊断工程

复用已有APC allocation观测思路，在一致native边界记录：真实running/waiting及request ID；首次准入/首次输出/完成；phase和每次scheduled token；唯一活跃物理块、共享活跃块/refcount、可回收hashed free块、分配失败/原生抢占原因、full-input/in-flight reservations；每请求输出、EOS/length、host返回时刻与任务质量。缓存hit计数本身不等于物理共享，必须至少有两个request同物理块的witness。预分配KV tensor大小不能当动态消费量。

shadow机会账本只读当前状态：

```text
shared-aware resident hard-cap bound
  = current unique active physical blocks
    + sum_i max(0, conservative full-cap blocks_i
                    - native currently owned logical blocks_i)
```

式中共享当前块只计一次，未来尚未拥有的各请求增长先按私有保守计数；真实EOS不参与。候选请求只有在native当下可证明获取相同前缀且该前缀在检查到allocation间仍有合法ownership时才可扣共享前缀，否则按完整新请求上界。这是解释保守机会的shadow账本，不是已经证明的APC-aware安全准入控制器，也不把逻辑可装状态当成已执行新准入。token、槽和native保留检查分开列，不能把memory-fit当progress-fit。

**旧M0/full-bound/Past-Future适配器基于无APC前提，禁止直接装进本块。** 若后续选候选，先补共享live-prefix状态与分配/释放语义，再以APC不变的N32、S16以及直接近邻比较。此协议没有授权靠关APC获得方法优势。

### P0指标和分支

全部32请求留在分母。先报任务质量、完成率、自然EOS/截断、TTFT、flow、每请求最大host-return gap、输出tokens/s、episode含drain时间；并列各任务、长短输入和最差请求。可观察一次返回多个token时不称真实逐token ITL。固定20/4沿用为历史可比描述点，另完整报告10/20/30/40×1/2/4/8/12前沿；不从本32请求挑新SLO。增加“正确数学/有用摘要且满足截止且完成 / 全episode秒数”的有用完成率，保留原始不含质量指标便于核对。

- **零机会**：原生APC已消除压力，停止这个小域，不再降低KV或改到达；转回任务/运行域的独立问题真实性，不实现M0/M1/M2。
- **简单规则覆盖**：S16降低抢占/尾部且没有明显代价时，它成为后续强简单参照；不把简单并发门换名当新算法。只在其后仍有真实可执行且可计价的残余时保留新机制。
- **残余存在**：要求能逐事件指出是哪类请求在何资源/执行条件下受到代价、另一合法动作具体改变什么；只许再选一个最小动作。相关step数、block差或oracle逻辑差不等于goodput收益。
- **质量/安全不合格**：数学或摘要任何一类低于Q0/Q1固定门则不能声称健康域成立；逐题质量变化完整保留。一个并发配置相对另一配置少≥2个正确数学答案或≥2个有用摘要，视为该pilot质量差异警报，暂停性能归因；即使少于2也不称质量等价。tokenizer/账本/失败记录缺失另记INVALID，不和有效负结果混为一类。

不对这32题做非劣统计声称。确认阶段须另外固定未用于选择的文档/request和独立episodes，冻结质量容忍度、阈值与样本量；不得把32个耦合请求、tokens或20个阈值充当独立重复。第二架构仍需另立模型资格；base和Instruct同架构权重不能充当跨架构稳健性。

## 预算、现有入口与尚未实现部分

Q0/Q1各1格，P0最多4格，总计最多6格；按单格900s上限为1.5 GPUh的**推理单元上限建议**。下载/环境准备另列实测，旧1560s launcher含暂存/清理的合同不与900s服务上限混用。本协议不重复下载已有输入/metadata；可校验复用同revision已完整验证的模型权重。模型传输失败不反复消耗GPU实验名额，也不算科学负结果。

已有Q0分析真实命令：

```sh
python3 C_research_artifacts/20261001/C_INSTRUCT_GSM8K_ANALYZE_V1.py \
  --inputs-dir C_research_artifacts/20261001/20261001_c_instruct_gsm8k_inputs_v1 \
  --run-dir '<新的实际运行根>/native' \
  --metadata-dir C_research_artifacts/20261001/20261001_c_instruct_model_metadata_v1 \
  --output '<不存在的新分析路径>.json'
```

尖括号处是必须由真实回执填写的占位，不是已有结果路径。现有cell V1/V2只接受固定16 GSM8K，不能冒称可以直接传32混合任务。Q1摘要输入/参考标注、通用32请求cell、APC共享shadow账本、N32/S16组级launcher均**待实现**；仅复用真实可用模块，不伪造完整命令。

新 `health_native.py` 的16题输出可由本目录独立 `health_analyze.py` 评分；它不加载模型、不初始化CUDA、不会覆盖原始结果或已存在分析：

```sh
python3 -B C_research_artifacts/20261001/qualified_serving_v3/health_analyze.py \
  --run-dir '<新health_native实际输出根>' \
  --inputs C_research_artifacts/20261001/20261001_c_instruct_gsm8k_inputs_v1 \
  --output '<不存在的health分析路径>.json'
```

分析器严格沿全部16题，核原始输入SHA、实际渲染prompt与输出身份、模型EOS及native sampling、APC reset、完整drain；输入渲染不同不是输出质量差异。结果分`COMPLETE/INVALID_OR_INCOMPLETE`与`PASS/FAIL_DEVELOPMENT_SCREEN`两个维度，科学负结果不被叫作无效。已用纯CPU合成夹具核额外模型EOS、缺失请求仍保留16分母、非EOS stop拒绝、EOS数量不达标、历史数值解析及周期反例；这些检查没有产生模型质量数据。

完成标准：先回答Instruct是否越过预声明有用输出门；再回答APC开启下是否仍有值得动手的容量准入残余。通过资格不是论文成立，强基线解释完残余时应停止该方法域，保留有效边界结果。
