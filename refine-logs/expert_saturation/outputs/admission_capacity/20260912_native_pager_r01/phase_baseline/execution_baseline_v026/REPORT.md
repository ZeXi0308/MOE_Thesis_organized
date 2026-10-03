# 同预算执行基线：4/4 COMPLETE

Verdict：**EXPERT_EXECUTION_BASELINE_ESTABLISHED / MEASUREMENT_ONLY**。已有专家分组实现，在本次相同预算、相同输入与共同旧请求前缀下，同时降低完整请求时间、已有请求最大生成间隔和新请求 TTFT。研究问题保持 OPEN；本轮没有新增 scheduler，也不把已有底座收益归为调度贡献。

证据为 OLMoE BF16 / native vLLM0.26 eager / 单 RTX5090 的小规模真实请求执行。专家槽16、scratch3GiB、实际KV512MiB、token budget64、maxseq3、CPU0–7/OMP8，两臂都采用同一个已验证的批量 expert-map 更新。输入source[4,5,10]长32/32/128，输出16/16/8；旧请求各返回4个token后加入新请求，prefill固定16。ABBA为token/expert/expert/token，四格同engine，每格相同 token16→expert16→token16 预热，在入场回调才切换测量执行方式，retention与完成后解除prefill限制均关闭。

| 指标 | token两次 | expert两次 | expert相对token，正/反序 |
|---|---:|---:|---:|
| 完整capture wall | 3.30749 / 3.30851 s | 2.44242 / 2.44984 s | −26.155% / −25.953% |
| 旧请求最大ITL | 306.163 / 305.752 ms | 152.755 / 154.054 ms | −50.107% / −49.615% |
| 新请求TTFT | 2.07861 / 2.08279 s | 1.13751 / 1.15167 s | −45.275% / −44.705% |
| 最后一个旧请求完成 | 3.25240 / 3.25448 s | 2.35010 / 2.35738 s | 两次改善 |
| 完整专家tensor-copy payload | 155.84765625 GiB，两次相同 | 101.58984375 GiB，两次相同 | −34.815% |
| 分组数 | 1626，两次相同 | 1032，两次相同 | −36.531% |

首mixed调用从282.451/282.062ms降至134.565/134.909ms；实际分组130→48、加载1157→462、payload13.55859375→5.4140625GiB。只能把它称为执行组织的实测效果；load CUDA spans含提交间隙，payload不是硬件PCIe wire计数。 [首mixed诊断](diagnostic_first_mixed.json)显示跨臂16层的active union有12层相同，第4/6/10/14层不同；两臂各自repeat均相同。因此加载差是各执行方式真实独立演进后的结果，不是把同一未来route固定后仅删除重载得到的反事实。

12次测量请求执行、160个输出全部完成，额外40次预热请求/488输出完整保留。共5024层调用，其中1216测量调用；身份、step行数、cache变迁、分组/字节、map计数检查通过。四格全部输出token相同，共同前缀记录、逻辑前状态、初始cache和分配地址一致。物理KV block IDs跨臂不同，未比较KV tensor或中间logits字节。分析器重算SHA时恢复expert_to_slot的整数key，以匹配runner在JSON序列化前的排序；原始数据未改。

实际测量allocator峰值均为4,735,429,120 bytes，reserved峰值4,800,380,928 bytes。两臂expert scratch和KV相同；expert分组仍有额外部分输出及masked map，首mixed每层输出暂存73,728 bytes，派生kernel activation884,736 bytes（不含排序/kernel内部）。共同前缀可能决定整段峰值，不能由相同峰值推出瞬时workspace完全相同。主机专家权重pinned12GiB，公共map staging4KiB。

完整post-init cycle52.048s，父进程79.196s，包括相同预热、reset、观测、raw写出、trace flush和关闭。两者是混合四格的总成本，不分摊为某臂加速。capture包含在线采集和动作开销；map validation在capture之后。157个GPU采样只见worker60757，初始化/episode边界均通过；这些不是连续独占锁。

本次token同臂wall漂移+1.016ms，expert+7.417ms。它们是描述性观测，不是总体噪声上界、置信区间或非劣证据。每臂仅两次、一个三文档组，未测独立到达过程、任务质量、SLO-goodput、真实超显存模型或跨模型推广。既有相关工作/实现的执行收益不构成新颖性。没有exact Oracle，不能把当前剩余延迟当成可全部移除的headroom。

模型修正：成本函数必须包含实际execution组织，`T(chunk,state,executor)`；旧token分组的曲线和文档×chunk响应不自动适用于expert分组。固定one8规则仍停止。唯一下一实验是在本expert底座上重新测既有static8/16/32动作曲面，保留同资源与共同前态，回答强执行基线后还剩多少prefill权衡；新曲面未运行，当前不加selector。

证据：[summary.json](summary.json)、[原始结果](readback_r01/results/)、[实际源码](source/)、[运行日志](readback_r01/launch/run.log)、[预先固定协议](protocol.json)。协议中的PREPARED_UNRUN是运行前快照；当前完成状态见readback_r01/results/status.json与readback_completion.json。自动审批首次拒绝上传；核验本会话针对同主机研究代码上传/执行的明确批准后，相同命令获准执行。拒绝及授权证据保留。

重算命令（仓库根）：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/execution_baseline_v026/analyze_execution_comparison.py --input-dir refine-logs/expert_saturation/outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/execution_baseline_v026/readback_r01/results --out /private/tmp/execution-comparison-recomputed.json
```
