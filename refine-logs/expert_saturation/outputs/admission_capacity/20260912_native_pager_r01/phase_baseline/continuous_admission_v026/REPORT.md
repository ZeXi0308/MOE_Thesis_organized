# 时钟到达下的准入与阶段 prefill

2026-09-13；**MEASUREMENT_ONLY / STOP_PHASE32_IN_TESTED_CLOCK_EPISODE**。六格完成，完整身份、调度与成本会计PASS。二元phase32规则没有显示超出两种普通静态策略的完整请求优势，本运行域停止推进该规则。

评价使用当前paging校准0–11之后的连续source12–19，暖机0/1/2另取；这些是项目既有公开文章的短前缀，不称全项目未使用数据或完整自然文章。输入长度[64,128,32,96,128,32,64,128]，输出上限[32,16,48,24,16,64,40,24]，声明到达秒[0,0,0.4,0.8,2,2.4,4,4.4]。相同OLMoE BF16/vLLM0.26、expert cap16、KV512MiB、token64、编译maxseq3，统一已验证nested观测，无prefetch/retention/动态缓存。每格独立引擎，共同static32和phase32暖机后重置pager；264个输出及全部提交、等待和执行都计费。

A为cap3/static32，B为cap2/static32，C为cap2/phase32；顺序A,B,C,C,B,A。C只根据当前是否有ready decode选择threshold32或0（仍受原生总budget64限制），每步已有decode必须推进。

| Cell | 完整wall s | mean TTFT s | mean完成延迟 s | 全请求最大ITL ms | payload GiB / groups / calls |
|---|---:|---:|---:|---:|---:|
| cell00_cap3_static32 | 7.861657 | 1.005124 | 3.216340 | 192.673 | 288.257812 / 3460 / 109 |
| cell01_cap2_static32 | 8.437254 | 1.626214 | 3.232236 | 172.438 | 305.625000 / 3177 / 146 |
| cell02_cap2_phase32 | 8.304350 | 1.638298 | 3.220134 | 177.373 | 297.691406 / 3196 / 145 |
| cell03_cap2_phase32 | 8.348866 | 1.655103 | 3.246419 | 177.759 | 297.691406 / 3196 / 145 |
| cell04_cap2_static32 | 8.420699 | 1.616995 | 3.220501 | 172.917 | 305.625000 / 3177 / 146 |
| cell05_cap3_static32 | 7.817079 | 0.992839 | 3.194067 | 183.452 | 288.257812 / 3460 / 109 |

C两格都只有step0选择threshold0，并真实给单个请求64个prefill token；其后全部选择32。它在已有decode持续存在的队列期间没有第二次放宽机会，不能把启动阶段收益写成持续在线调度能力。

C−B完整wall为−132.904/−71.832ms（−1.575%/−0.853%），mean TTFT却+12.084/+38.108ms，mean完成延迟−12.102/+25.918ms翻号，最大ITL+4.935/+4.842ms。C相对A的wall慢5.631%/6.803%，mean TTFT高62.995%/66.704%。A在本轮完整wall、mean TTFT和payload最强；B保留最大ITL更低这一端的权衡，不宣称A逐请求支配。

TTFT按三段闭合：`声明到达→实际提交 + 提交→首次真实schedule + 首schedule→首token`。B−A的mean TTFT增加621/624ms，主要是native队列增加668/689ms；首schedule到首token反而减少48/47ms。C−B的后段减少31.6/28.6ms，却被提交lag增加17.1/18.9ms和native等待增加26.6/47.9ms抵销。不能只报局部prefill加速，或将等待再加到已包含它的wall上。

每格672个prefill位置+256个decode位置=928个真实scheduled token rows，16层逐行join闭合；48请求/1584输出全部完成，无抢占或跳过ready decode。B比A少283个expert group（−8.18%），但engine调用109→146、payload反而增加6.02%；分组更少不足以判断完整搬运量。C较B少一次engine调用、少7.93359GiB payload，不能替代完整请求等待账本。

同臂两次完整route/group/load轨迹及8/8输出一致；跨臂A−B有2/8、B−C有3/8、A−C有4/8最终输出不同。各策略独立自然执行，未证明质量等价。所有逐请求正负差、同臂重复差与phase会计保留在[summary.json](summary.json)。两个顺序block不是统计确认，观测重复差不是噪声界；未设置事后SLO或百分比GO门槛。

证据层级为带实验pager的NATIVE_SERVING有限时钟episode；未测持续稳态容量、生产尾延迟、质量、真实超显存模型或exact Oracle。被否定的仅是本域这条“无ready decode才放宽”的持续调度用途；尚未覆盖真实多次排空的到达过程或其他合法prefill动作。复活须由独立真实到达证据显示反复动作机会及相对强静态的完整成本余量，不能靠为规则插空档、换seed或调本episode阈值。

唯一下一最小实验转到原任务要求的真实超显存模型资格化：先完成Qwen3-30B-A3B BF16串行分片加载器的CPU生命周期检查、同输入重新tokenize及完整资源预算，再做最小模型加载/请求smoke。它是后续调度研究的运行域检查，不复活phase32，不把OLMoE人工小缓存结果外推为必需offload。当前仅CPU准备，未下载新模型或声称GPU结果。

六个worker均已退出；父周期298.923s（含每格空闲等待），子进程总计237.286s。30个GPU边界通过，469个硬件样本只见各cell自己的worker；这些记录不保证连续独占或CPU隔离。12次暖机36请求/72输出完成；每个引擎关闭后结束其cap/threshold生命周期。源码与回读SHA在[完成回执](readback_completion.json)中，原始数据不改。
