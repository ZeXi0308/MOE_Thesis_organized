# 共同 KV 增长的决策区分力：固定 CPU 检验

在对全部边界批量计分前固定如下问题和规则：对 D（natural saved recovery）和 E（native-full strong baseline）的全部 `target_new_output` 首输出保护释放边界，现有资源状态能否区分紧接着的 native 抢占风险？D 的两个短段已经在前序定位中见过，因此本检验是覆盖全部边界的探索性解释检查，不是盲测或独立 holdout。不筛选请求或事后 step；两组只是诊断实例，不比较性能。

输入来自该步 scheduler 运行前的 eligibility snapshot。适用条件只使用当时状态：没有在途 rotation plan、保护已经释放、全部 running 请求处于 decode-ready 状态（RUNNING、已有输出、computed=prompt+output−1）、running 数量不超过已配置 token/sequence 预算。所有不适用边界及原因保留。

固定 block_size=16。令 `c_i` 为已计算位置、`b_i` 为持块数量、`F` 为当前 free blocks：

`G(h) = sum_i max(0, ceil((c_i+h)/16) - b_i)`。

主检验 `h=1`：`G(1)>F` 预测紧接着一次 scheduler 调用会发生 native 抢占。基线仅看 `F==0`。标签从原始抢占事件提取；同 step / victim 的 accepted staged commit 属于 forced，不计 native。

同一公式附带一个固定的两步可持续性检查 `h=2`，标签为该步及下一步是否发生 native 抢占。它假定当前 running 集合各继续执行两个 token，不预测 EOS、完成释放或未来 admission；因此只是结构包络，误报可以揭示这一假设的局限。它不是另一个拟合 predictor，也不做 horizon / threshold 扫描。

报告覆盖率、混淆计数、报警基数、与 free-only 基线的区分、全部反例。请求 ID 只用于身份链接和标签说明，不进入公式；不使用未来 EOS、实际最终长度、轻量性能指标、离线最佳 step 或反事实重放收益。
