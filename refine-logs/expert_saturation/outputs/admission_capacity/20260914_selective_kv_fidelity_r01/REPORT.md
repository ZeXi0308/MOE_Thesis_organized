# 单次选择性保存：首次恢复前缀保真

Verdict: FIRST_RESTORED_PREFIX_MATCH。原生进程内 vLLM0.26、OLMoE BF16、单RTX5090、受控6656 usable KV块、native16GiB host offload；单save-on诊断格32请求完成、32768输出。不是净收益或质量结论。

3296个token、206个完整逻辑块、16层逐层SHA完全一致，每层27000832bytes，总432013312bytes。恢复后物理块映射不同，token前缀身份相同。目标为原文档3571，排除本次首次恢复的已保存前缀搬运损坏。

真实事件：328登记store114，329原生抢占/flush114，worker确认store完成；330登记load115，worker确认load完成。快照before在328返回、computed3306/output235；after在331返回、computed3296/output235；raw中332才从3296执行11个位置，333继续正常decode。因此本次after快照实际位于首轮恢复计算之前，此事实由轨迹确定，不是hook一般时序保证。

指纹读取使用每层真实4D张量视图和当前block映射，沿逻辑token顺序串接字节，不按物理地址比较。独立分析器核对层集合、3296tokens/206块/完整字节数、请求身份、第一load完成ID和快照顺序，结果MATCH，0差异层。

额外同步/读取/CPU哈希before0.419092s、after0.372862s；本格所有墙钟结果均受诊断影响，不与无指纹两臂作性能比较。实际一次store432013312bytes、两次load共864026624bytes；这里只核对首次load，第二次未采样。

未验证：其他请求KV、未保存的10个已计算位置、第二次加载保真、自由生成质量、跨模型/自然负载、独立host预算和稳定完整请求净收益。7条其他请求此前文本分叉的原因仍未定位，本格不能把它们全部归为正常数值差异。

原save-off/on两臂+2.37%完整平均延迟事实保留，但动作前已有+0.927s漂移，见前实验TIME_PARTITION_ADDENDUM；不能据该两次运行判定保存机制净正或净负。

下一最小实验：同原非指纹包、同16GiB底座与旧d6输入，增加完整交错重复来分辨单次保存的请求级净效果；保存前缀正确性此次通过，不把单事件重复升级成完整在线轮转方法。固定完整平均完成为主观察，同时保留wall、所有请求max-ITL、恢复工作和动作前时间差，不作post-hoc漂移扣除。

运行group PID70641/exit0/finished1789352443.5779352；新SSH确认PID退出和GPU空，已释放。归档SHA5198a6e7ad745a9763eafa6acb895272e78db83aeaff942cf1b36f5177ad43de。执行者定向核验完成；fresh same-family审阅进行中，非独立接受。

定向复核完成：GPT-5.6-Sol ultra fresh同模型族审阅PASS/provisional，无P0/P1；独立复算analysis一致并核对328–332真实时序。支持边界未扩大，见EXPERIMENT_AUDIT.md。
