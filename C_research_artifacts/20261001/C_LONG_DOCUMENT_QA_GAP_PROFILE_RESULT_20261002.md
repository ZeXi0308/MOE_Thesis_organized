# LongBench150：density 长步的单次 Python profile

这是对既有 density 顺序的一次**诊断运行**，仅在固定调度步 379／380／381 对同步 `engine.step` 加 `cProfile`。原始 [selected-step profile](</Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001/density-gap-profile-dev-v1/native/selected-step-python-profile.json>)、[调度轨迹](</Users/zhaozhenyu/Desktop/毕业设计/C_research_artifacts/20261001/density-gap-profile-dev-v1/native/measured-steps.json>) 和[轻量提取 JSON](long_document_qa_gap_profile_v1.json)保留可核对数字。此格不是未插桩的速度对照。

| 调度步 | 调度 token / 请求 | `engine.step` host 墙钟 | 编译调用观察 |
|---|---:|---:|---|
| 379 | 1024 / 11 | 0.026792 s | 未见下述 `make_cubin`／LLVM 调用 |
| **380** | **911 / 11** | **0.733188 s** | Triton `make_cubin` 2 次 |
| 381 | 11 / 11 | 0.015349 s | 未见下述 `make_cubin`／LLVM 调用 |

步 380 的 `make_cubin` 两次累计 **0.220962 s**；其中下层 `posix.waitpid` 两次 self **0.217870 s**。同一步还记录 LLVM `optimize_module` 两次 self **0.082756 s**、`translate_to_asm` 两次 self **0.060391 s**，以及 Triton code-generator `visit` self **0.063767 s**。这些是**嵌套和递归调用**，不能相加为编译总时间或墙钟分摊。`visit` 的记录出现 cumulative=0 而 self>0；步 380 全部 self 合计仅 0.591332 s，小于 0.733188 s 墙钟。因此也不能把差额称作 GPU 执行、内核耗时或未解释的某一确定部件。

本次完整 150/150 请求、434 次 schedule、3951 输出 token、126 自然 EOS／24 封顶、0 抢占；逐请求输出 ID、文本、prompt ID SHA、结束原因与同机两次 primary density 格**全部相同**，提交顺序以及434次调度的 token 总数和请求数也相同。故长步中显著的 Triton JIT 编译活动是直接观察，不应把此前约0.7秒的停顿直接归为文档组连续排序的固有执行成本。但尚未用精确 hook 证实哪个完整形状触发 specialization，也不能据单次插桩剔除全部停顿、声称某个编译百分比或重新计算未插桩策略收益。[先前形状计数](long_document_qa_long_step_shape_v1.json)只给出 911-token 调度总量的首次形状线索，调度 token 总量也未被证明等于 fused-MoE 的实际编译形状。

原始完整归档 `density-gap-profile-dev-v1-complete.tar.gz` 为 3,394,118 字节，SHA-256 `60f74ee849ca1c40889475186d9b0fc6de3ce0a5af6ff584bd658f6cc507b636`；profile JSON SHA-256 `ff5742061052ff85ba158046952d8ca189caf83724bc60adf150c0312b5b44d5`。运行已完成、子进程回收、私有 stage 清除；这些记录仅支持上述局部诊断。
