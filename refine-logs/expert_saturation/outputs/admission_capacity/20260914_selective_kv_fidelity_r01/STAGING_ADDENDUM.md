原包已上传 /root/selective-kv-fidelity-20260914-r01.tar.gz，远端 ls 确认857527bytes。首次上传连接关闭，第二次scp exit0。旧控制连接无响应；新连接确认解包目录不存在；再建连接在提交校验命令时被远端关闭。远端SHA/18文件校验与解包未确认，GPU_UNRUN，无driver。B第二cohort已释放，仍排context-victim及streaming之后。下一只读核验远端目录及SHA，不直接重传或启动。

后续核验完成：新非交互SSH读回原包SHA匹配；使用vLLM虚拟环境Python绝对路径解包，18项文件SHA全部一致，results不存在。远端 /root/selective-kv-fidelity-20260914-r01 已暂存，GPU仍UNRUN。共享socket失效，不复用；未启动driver。

结果分析入口已实现：experiments/admission_capacity/analyze_selective_kv_fidelity.py <cell>。独立比较16层×3296token、206唯一逻辑块、每层27000832bytes/总432013312bytes、请求/首load身份及快照顺序；允许物理块重映射。CPU替身3项通过（重映射一致、指定层变化、缺层拒绝），不构成GPU证据。快照位于engine.step返回，明确不保证在恢复计算之前。context六格已有完整释放登记，下一仍先streaming、后本格。
