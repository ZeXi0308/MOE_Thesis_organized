# 当前容量可行之后，服务窗口是否兑现

Verdict: **MEASUREMENT_ONLY / FUNDING_IS_NOT_SUFFICIENT_FOR_USEFUL_SERVICE**。原资金过滤六格全部完成；本次只读复用其原件，新增恢复生命周期与全部生成间隔阈值的服务量分析。过滤消除了局部资源拒绝，却在两轮中都出现新的零输出和两输出再抢占段。相对least的完整收益符号反转，相对most是延迟与吞吐交换；没有服务窗口方法GO，也没有整个问题NO-GO。

## 证据与比较范围

原执行方为`/private/tmp/moe-a-recovery-components-20260914`，原bundle为`20260914_funding_filter_comparison_r01`，统一分析`analysis_r02.json`。原包SHA `27a392a8e46e590426416d8aee305a3cf270a0bcf9275e756365404e18f858bc`；完整回读SHA `7ad278ed5df66db7729338f534cd522c60f76958800ec60fc9ec52b5cf3a8329`，133554053 bytes。controller1917整组COMPLETE/exit0，结束1789396822.5909417，192/192请求完成。没有重启或丢弃任何一格。

六格共同使用新GPU `GPU-bd5e9bb9-f98b-db5b-cc1c-5857c39f0bdc`，OLMoE/vLLM0.26、6656可用块、无APC/host KV offload、交替2560/3072输入及1024强制输出，原50ms到达序列。共享父host上限90GiB，不是独立进程树硬隔离，本组没有host峰值测量。只在新GPU六格内部比较，不与旧context GPU耗时配对。旧文档复用、两次反序不是独立workload；EOS与质量未验证。

计时是同步引擎step返回后的host记录，非客户端收到。所有完整请求、等待及观察成本进入原完整测量窗口；初始化、预热、序列化及网络传输在窗口之外。

## 当前资源资格修复后的完整结果

| 动作相对基线 | block0吞吐 | block1吞吐 | block0最大ITL变化 | block1最大ITL变化 | 两轮平均完成变化 |
|---|---:|---:|---:|---:|---:|
| filtered / least | -3.011% | +1.272% | +108.586ms | -18.059ms | +3.134% / -1.125% |
| filtered / most | -5.706% | -5.096% | -72.886ms | -162.330ms | -1.250% / -1.949% |

filtered每轮未资助成功的proposal由18降至0，但重算83510→89761，强制交换20→21，总调用1445→1452。第0轮32请求全部比least晚完成；第1轮全部更早完成，两轮仍分别29/28请求的自身最大ITL更差。重复漂移保留，不把所有请求当独立样本或把正的一轮挑成canonical。

## 有效服务与内部工作分账

下表每轮结构相同；所有恢复段均闭合，无删失。

| 规则 | 恢复段 | 无新输出再丢弃 | 1–2输出再丢弃 | 有输出后再丢弃 | 恢复后持续至完成 | 下一恢复实际重执行的原恢复位置 |
|---|---:|---:|---:|---:|---:|---:|
| most | 26 | 0 | 0 | 8 | 18 | 28151 |
| least | 24 | 0 | 0 | 18 | 6 | 62272 |
| filtered | 26 | 1 | 1 | 19 | 6 | 67791 |

“1–2输出再丢弃”包含在“有输出后再丢弃”内，不能相加。most/least闭合再丢弃段分别最少11/5新输出。再执行位置与重算总数重叠，也不能加成额外总成本；已有输出支持过的KV不等于全无价值。

filtered两轮的具体事件均是request `memory-train-article-0020902`：

- 831被抢占，891开始恢复，894返回首新输出，896又被抢占。3732历史位置已重算，实际仅2新输出；失效前缀3734位置。下一段实际重新执行了这3732个原恢复位置。
- 1015被抢占，1026–1028连续重算2991位置，1029在首新输出前再被抢占。部分前缀在三次恢复调用之间确实复用，随后2991位置才实际失效并在下一段重新执行。

两段恢复调用跨度包含混合batch的其它请求工作、scheduler及host时间，不作为纯恢复税。没有因为被调度或内部前缀增长而重置输出等待。

定位显示两个不同边界：895已观察到首新输出，`recovery_completed=0020902`且`recovery_target=None`；896没有forced动作，peer0019436在computed3408时申请1个位置，free=0分配失败，原生FCFS从队尾抢占0020902。因此这段保护按其首输出合同已经正常结束，后续两输出窗口却不能继续维持。

1026则是native自动resume，adapter proposal处于swap cooldown，`recovery_target=None`；1027–1029返回`native recovery already underway`。目标最终要执行781位置，已有187块，完整历史需236块，还缺49块；此前peer先增长2块，剩48块，原生抢占目标自身。轮转适配器只在显式交换时设置保护，自然恢复没有自动继承；30-step驻留限制也只筛选adapter victim，不能阻止原生容量抢占。不能把零输出段说成一个已启用保护合同被违反，亦不能归给近邻论文。详见`event_localization.json`。

## 不选有利阈值的权衡

沿全部经验断点计算`Q(g)=自身最大engine-return gap≤g的已完成请求数/完整episode wall`，与同轮most/least两条分别完整执行策略的较优包络比较。filtered高于包络的非零区间为4/33及20/33；第0轮只有`[1.523572,1.657915)`秒，第1轮区间见`tradeoff.json`。第0轮增量仅0.006395 requests/s，不能把窄区间事后选成业务SLO。

因此不能宣称filtered被强基线在全部延迟目标下支配；也不能由局部包络改善推出稳定方法收益。这里没有联合TTFT/TPOT约束、未来Oracle或请求内切换。主张上限是这两个实测完整轨迹组的描述性权衡。

## 对最小窗口模型的影响与唯一接续

模型仍用`L=ceil(C_remaining/alpha)`表达恢复前的有效服务机会下界；恢复峰值、共同batch的增量KV、其它请求输出年龄给出上界，并须检查native ready状态。当前事件支持“资助完整历史只解决恢复入口”这一必要区别，没有提供可填入`C_remaining`的边际毫秒，更没有证明长保护优于most。

合成支持例仍在`../../20260914_service_window_model_r01/REPORT.md`：L=4时共同batch可提供4次机会，独占却使peer等待越界；free减少到2块时U_KV=3<L。合成毫秒不移植到本次真实事件，未知EOS不当保证输出。

本次失败类别是**局部可执行动作不保证后续有效服务，完整修复收益未稳定**。保留most强基线，不把filtered新暴露的短段误报为most残差，也不立即添加固定/几何/模型窗口。要恢复窗口机制，须在强基线后找到可执行且具有完整服务收益的增量；有限候选不是全局上界。

原定下一GPU判别复用A原已登记的native两阶段保存off/on单事件资格，固定16GiB缓存能力，检查保存完成、原生load-ready、真正输出和完整成本。本报告整理时，该两臂也已执行完成、各32请求，原方正在统一回读分析；这里不提前宣称净收益，也不重复启动。它用于检验恢复底座，不声称已有逐请求后端切换接口。资格失败则修适配器；正确但无净收益则保留重算底座；只有成本与有效服务闭环成立后才考虑窗口决策。第二模型Qwen旧attempt只完成部分加载，不当性能负结果。

## 复现

从主工作区执行现有`analyze_effective_recovery_service.py`，参数`--workspace /private/tmp/moe-a-recovery-components-20260914 --campaigns 20260914_funding_filter_comparison_r01 --expected-cells 6 --output-dir NEW_DIRECTORY`。`analysis.json`保留逐段原始路径/SHA及守恒、执行回执、时钟嵌套、前缀连续性和输出边界检查。

`tradeoff.py --qualified ORIGINAL_BUNDLE/analysis_r02.json --library MAIN_WORKSPACE/refine-logs/expert_saturation/experiments/admission_capacity --output NEW_JSON`复用已有阈值分析，不生成反事实轨迹。没有新增GPU或测试矩阵。
