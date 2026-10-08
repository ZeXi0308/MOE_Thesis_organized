# 动作分支准备：前缀可见状态通过，KV内容尚待原生核对

本轮问题：既有四份d6轮转运行能否提供共同动作前态的重建入口？

新增只读检查`E/check_rotation_branch_prefix.py`：按engine调用结束边界取step329之前的输出，逐请求核对累计新token与当时内存快照输出计数；比较四份raw的329步实际调度/计算进度、32请求前态、KV块计数和已生成token序列。

结果：四份均8909个前缀输出token，实际执行序列逐项相同，step329前state逐项相同。空闲148块、已用6508块，已计算总103908位置。完整调度遥测并非完全相同：早期waiting数量首次差异出现在第9/2/7步；单列保留，不删掉差异后声称整个历史一样。第一版完整telemetry比较见branch_prefix_check.json；第二版增加executed维度，见branch_prefix_check_r02.json。

这些只是共同前态的必要可见条件；raw没有KV张量内容、分配器完整顺序及随机状态。不能用token相同替代KV相同，未执行任何候选分支或Oracle。

已读取实际westc vLLM0.26的worker/utils.py bind_kv_cache与attention/backends/flash_attn.py：本机FLASH_ATTN使用逻辑(B,H,N,2D)，forward_context[layer].kv_cache是对应Tensor。不要套用旧(2,B,N,H,D)布局。

`E/recovery_kv_fingerprint.py`实现该明确布局下按逻辑block顺序、仅computed有效token的SHA256；分块拷贝，避免全缓存同时复制。既有remote venv CPU测试通过：物理迁移不改变指纹、未计算尾部及自由块不影响、有效byte变化能识别、空KV、缺块拒绝、字节数、跨32块分片BF16与一次性参考逐字节一致。cuda_initialized=false；见kv_fingerprint_cpu_check.json。原生集成UNRUN，不能称已核对真实GPU KV。

唯一下一执行：在冻结d6 runner中接入step329前一次诊断钩子，记录当前逻辑状态和每层每请求有效KV指纹；两次新引擎重建同前缀，先验证真实KV内容是否相同。哈希涉及GPU到CPU复制与同步，必须计为诊断开销，不拿带校验时间的端到端数冒充机制性能。若KV一致，再从校验前态执行least/most/defer并各自推进；若不一致，先定位差异，不构造伪同态标签。本轮未上传新GPU包、未排他占卡、未启动GPU。

与旧first_most_then_least的区别：旧d2实验已回答一次换victim不足以复现持续轮转；这里首先检验d6动作条件前态是否可重建，不把旧实验当作未跑，也不重复以一次策略切换宣称新方法。

APC兼容轮转d2四格已由其他会话完成并主分析，本会话只读其analysis，不重复运行/归因；d6仍明确APC-off边界。后续GPU必须重新读协调队列及现场检查，不能沿用本轮空闲印象。

E指refine-logs/expert_saturation/experiments/admission_capacity。证据层级CPU_OBSERVED_PREFIX_ALIGNMENT/CPU_TESTED_HELPER；在线模型OPEN，原生状态指纹与动作分支UNRUN。
