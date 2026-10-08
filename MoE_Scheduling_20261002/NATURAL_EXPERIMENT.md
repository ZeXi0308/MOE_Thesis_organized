# 静态粒度优化的自然完成确认

前一固定32token探索中512/2048/4096各自正反序；新确认保留全部三预算及正反序，不只取最优者。顺序512/2048/4096/4096/2048/512，六个fresh engine，同一编译容量4096，expert cap24、KV1GiB、native16准入、同旧train16和512预算/固定32输出warmup。测量使用原GSM8K source32..47、完整提示、同时到达、greedy temperature0。与训练0..15及探索16..31的ID和prompt hash无交集，未读取选择组的旧输出/质量。

测量契约：ignore_eos=False/min_tokens=0/max_tokens=512，不额外字符串stop。到EOS自然结束；达到512的length终止必须单列，不冒充自然答案完成。最长episode600s，失败仍保留全部raw。生成内容/长度因调度可能不同，所以报告实际输出总数、吞吐、全部请求wall/flow/TTFT/ITL与准确率，不称等工作量加速。准确率固定沿用历史Open-Instruct/GSM8K的逗号移除、最后数字抽取，保存全部解码答案并可人工核查；16样本只检明显质量损失，不证明质量等价。

本轮只是已有token-budget动作在MoE offload域的实测优化确认。Sarathi-Serve已含按TBT选择预算，不能据一次配置收益写成新算法。进一步的方法贡献仍需超出最佳static与近邻已有策略的增量、真正显存压力及更广工作负载。自然完成结果将决定是否值得继续。
