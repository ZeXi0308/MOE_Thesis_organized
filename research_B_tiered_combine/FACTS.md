# 实现前事实快照（2026-10-16）

已获取并阅读七月设计说明、历史主报告及汇总、QuotaEP-H 结论、codec 勘误和关键代码。ref_july 是连接器读取 main 的选定文件快照（逐文件 Git blob SHA 见 SOURCE_FILES.json），不是完整 clone；两次 CLI clone 失败。历史报告的旧建议不覆盖当前任务。

1. 七月 GPU 48.11×/117.35× 是等字节 head/tail INT4 的 **KL 损伤比**，不能在 F1 标作 contribution share 比。旧 contribution 比约 5.43×/9.39×。原始质量结论不重证，新四模型刻画独立报告实际定义。
2. FP8-first 62.5% 点的 INT4 质量证据保留；它不是 NVFP4 的质量保证。旧 tiny-Mixtral 的近 one-hot 特例不作代表性主证据。固定 rank 不是既有质量最优选择；保留其低运行成本动机。
3. QuotaEP-H 在 EP=8、两模型两 placement 支持 output-aware 优于 gate/rank；不能直接推出 EP=2 gate 最优。按本任务先实现 gate mass 主臂，partial norm 变体；同 placement grouped BF16 是模式 II 的质量 reference，以隔离加法结合次序误差。
4. 旧 fake_quant 的 NVFP4 全局 scale 是逐行 FP32；当前 codec 按整个输入张量一个 FP32。旧 FP4 bucketize 在 ties 偏向较小幅值，新实现用 nearest-even，边界数值不可宣称逐位复现旧代理。旧格式脚本 FP8 字节模型还使用 block-128 scale，与 rowwise FP32 不一致，不能照搬账目。
5. 旧 58 µs 包括 pinned H2D，不能充当 GPU-GPU codec 开销。本课题所有量化/反量化保持 GPU 驻留；动态全局 scale 的归约成本必须计入。单次 quant launch 可以接收预先给定的 GPU scale；动态算 scale 需要额外归约，不隐瞒。
6. 实机 Torch 2.11.0+cu130、Triton 3.6.0、vLLM 0.26.0，单 PRO 6000。AgRsAll2AllManager 是 all_gatherv/reduce_scatterv。不能泛称所有非 DeepEP 后端都一样；FP8/NVFP4 模型实际 kernel 尚未运行确认。
7. fused_moe.py:1765–1793：第二次 expert GEMM 写 intermediate_cache3，按 apply_router_weight_on_input 控制权重已乘入；moe_sum 前输出不能再次乘 gate。必须从 topk_weights 显式排序，不能默认 topk 返回顺序。
8. 当前 GPU 0 MiB/100% 利用率且无容器内计算进程，不能把它当空闲做性能测量；torch 的 CUDA 可用检查成功。这不是 codec 失败证据。先推进 CPU reference/实现，待资源状态澄清后运行微基准。
