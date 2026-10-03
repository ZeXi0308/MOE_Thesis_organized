# 2026-09-12：预测输入与跨策略边界修正

原 `REPORT.md` 和 `analysis/kv_deficit.json` 保留。本次重新分析相同四项原始数据，结果在 [causal_cutoff_addendum.json](analysis/causal_cutoff_addendum.json)；没有新的 GPU 执行。

**结论：空闲块耗尽在原推进路径上可被早期状态解释；原生抢占的受害者数与等待时长没有被证明是跨策略下界。**

## 修正内容与重算结果

- block size 改为同 cell 在测量前记录的 `safe-cap-qualification.json`，不再从后续受害者推断或从另一 cell 携带常量。
- 全部已声明请求首次入场后立即关闭 admission 前缀。预测宽度只来自原有 `[119,519]` 窗口，要求相同请求集、纯 decode、成功调用、无完成/抢占和池变化。
- 预测在第 519 步调用返回后才可用，锚点为第 520 步及此前已知的剩余块。两个低预算 cell 均为 573 空闲块、32 请求、16 token/块，预测耗尽步仍为 **807**；实测首次抢占均在 **806**。
- 两个高预算 cell 的条件耗尽预测为 1184/1208；原轨迹在 1025 已开始完成释放。预测不再适用于发生释放之后的增长路径。
- 第 520 步至预测耗尽有 287 步；不是从第 94 步起就拥有 713 步的预测提前量。普通新准入在第 94 步后已没有 waiting 新请求可控制，但这不排除对已入场工作施加合法动作。
- 使用实际未来完成时间得到的等待误差约 0.90–0.95 ms，字段改为 `reconstructed_wait_s` / `wait_reconstruction_abs_error_s`。它是等待账本重建，不能称为在线等待预测验证。
- `predicted_preemption` 改为 `terminal_co_residency_exceeds_pool`；`min_victims` 改为 `fixed_path_victim_estimate`。估计的 2 个受害者依赖原路径增长率、未来释放时刻和给定 yield，不是优化上界或普适最少数量。
- 条件完成步修正为 `first_service_step + prefill_steps + max_output_tokens - 2`：零起始编号，最后一个 prefill step 已产生首个输出。只适用于不中断的逐步服务，不能用声明长度为任意排队过程提供完成时间上界。

## 两个精确资源状态反例

均为 CPU 结构检查：块大小 1、两个请求、各 prompt 1 token / 输出上限 3。最后返回的 token 不需要继续写入 KV；每请求实际完成前最多 3 块。时间编号只是事件顺序，不是 GPU 延迟。

1. **终态最大值不能直接相加为必达峰值。** 池 5 块，初态 A 已有 2 块/2 输出，B 有 1 块/1 输出。下一步二者都推进，峰值 `(3,2)=5`，A 完成释放；再下一步 B 完成。没有暂缓或抢占，而同时终态需求 `3+3>5`。
2. **改变增长率即可打破原路径受害者估计。** 池 4 块，共同初态 `(1,1)`。同步推进到 `(2,2)` 后，下步 A 无法增长而需回收。保留 B 的 KV，让 A 两步推进到 `(3,1)` 并完成释放，再推进 B，即可在相同首次释放步 2 下零抢占完成，峰值始终 ≤4。`BridgeRequirement(1,2,2,2)` 给出的 1 个受害者不适用于该动作。

反例推翻的是下界表述；它们不证明选择性暂停能改善最长 ITL、吞吐、公平性或原生 vLLM 性能，也没有实现新的 runtime controller。

## 验证与接续

`python3 -m unittest test_kv_deficit_model -q`：**40 项通过**，含上述反例、未来 width 不影响当前窗口，以及窗口内重算会被拒绝的针对性检查。四项 CPU 重分析完成；原输出未覆盖。

仍先执行已准备的原生 full/chunk 恢复准入同池对照。该 GPU 实验因上传/执行授权被自动审批拦截而 `UNRUN`。若出现恢复期间反复抢占，再针对后续 KV 增长保障实现最小动作。本次不构成问题级 NO-GO 或同预算方法收益。

基线 HEAD：`7059fc98e0d98a127ff4edda86d44f7f634d26f4`，代码为未提交修改。
生成时 SHA256：`kv_deficit_model.py=b1ee3b360a82053ac1b7948a82e534e6a7ea742ac66fde78be16192099dbccf0`；`analyze_kv_deficit.py=26cddba1b645c325d1e4c2a9c7ef6f44d45f67072062679ea64d9d55e14a9566`。
