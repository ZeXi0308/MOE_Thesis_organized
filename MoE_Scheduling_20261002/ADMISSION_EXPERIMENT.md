# 系统调度第一轮：服务成本驱动的prefill准入

目标：直接检验限制新增prefill的时机能否减少完整请求代价，胜过原生和固定并发强基线。仅一条新优化主线；组间流水当前负结果保留，不再调group参数。

实测依据（旧GSM8K0..15、cap24）：mixed步骤中位202.67ms、pure decode中位89.21ms；每步平均expert搬运7.788GB与2.860GB。相同decode width仍有差异，但未匹配cache/请求状态，不能解释为因果prefill税。prefill激活专家中位63/64，并非必然全64。

成本模型先采用最小可部署形式：旧训练episode按decode宽度给出T_decode和T_mixed初值，运行时只用已完成step进行EWMA更新。每次native schedule前比较 Q*T_decode（一decode步暂缓造成的排队代价）与 D*max(T_mixed-T_decode,0)*ceil(new_prompt_tokens/token_budget)（新增prefill工作带来的在服请求代价估计），选择开放一个新请求或暂缓。两者单位是request-seconds，但horizon不同，是待实测的局部启发式，非全局最优或可靠反事实。仍有running prefill时它继续执行，记录该假设边界，决策只管后续waiting准入。

共享age guard由旧训练数据的单请求服务估计得到1.35174s，用于防止无限等待；age-only基线使用同一guard而不使用服务成本。空running必放行；preempted/blocked/streaming恢复时开放原生fallback，避免门禁阻塞恢复。门禁不主动驱逐running，但原生KV抢占仍可能发生，完整记录。

**直接六臂，预声明顺序：native16、static2、static4、static8、age_gate、model。** 同一vLLM eager expert-major cap24、KV1GiB、token budget512、16请求同时到达、每请求固定32token。模型只使用旧源序0..15校准和共同warmup；测量用源序16..31，prompt661–701token，与旧组ID和prompt hash均不重叠，未按旧质量结果筛选。所有arm fresh engine，warmup都是旧16条、native16；drain后才应用各自策略。每臂独立生成未来route/KV/output，不离线借用别臂轨迹。

指标为完整512输出token的episode wall、flow/TTFT/ITL、全部请求完成/输出差异、实际搬运字节、组数、preemption、模型决策和预测更新。观察与决策开销均留在请求/episode路径。若model不能胜过最优静态cap或age-only，不称模型贡献成立。固定token是第一轮等工作量净成本探针，正信号后再验证自然EOS、答案质量、bursty与新模型/实际显存压力；本轮不够支持论文主张。

当前成本模型只是MoE运行域的服务曲线模型，尚无显式cache状态预测，更不能称已确立MoE独有新颖性。先实测动作与strong baseline，正信号才引入有明确残差的缓存状态项，不叠加复杂预测器。
