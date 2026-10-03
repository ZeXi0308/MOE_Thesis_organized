# H1 direct 准入的在途预留条件反例

状态：只读 CPU 分析；没有 H1 GPU 现场故障证据，也不改变已冻结 H1 包或本次 H128 LTR 迁移。

H1 候选的 [`_direct_resume_reason`](candidate_h1/pkg/staged_store_rotation.py) 在 `free >= target_need` 时可以返回 `DIRECT_READY`，但没有显式核对 native 在途 prefill 预留。固定版 [scheduler](liveness_pinned_sources_20260930/scheduler.py) 在异步 load 时把 `_inflight_prefill_reserved_blocks()` 传给 KV 分配器。条件性 CPU 反例为 `free=8, target_need=8, reserved=1`：H1 helper 接受，native 的有效可分配块只有 7，分配返回 `None`。native 会保留目标在 waiting；H1 随后保护目标并要求本次正调度 token，因零进展抛错，不能把此路径解释成 H1 会安全等待。

**当前模式的可达性限制：** H1 runner 使用 `population_mode='open'`；其 commit 检查要求 `skipped_waiting` 为空、所有 running 都是 pure decode。固定版 scheduler 的正预留来自 skipped 中的异步 load 或 running 中未完的 prefill，并在退出这些状态时清除。因此正常当前 open-mode commit 应有 `reserved=0`。上述反例证明 helper 的局部不变量不完整，不能证明当前 H128 自然输入会触发。后续 H1 新候选可在 direct 门显式要求预留为零，未知值回退旧 commit，并用 CPU 反例验证；这属于共同正确性硬化，不计 H1 方法收益。
