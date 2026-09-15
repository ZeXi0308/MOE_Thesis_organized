# Qwen 固定 prefill 32/16 对照

**PREPARED / UNRUN。** CPU runner/capture 检查和输入准备完成；必须先完成 Qwen 原生加载、48层数值诊断和四请求资格化，才执行本目录性能对照。没有性能结果。

唯一问题：真实超显存 Qwen BF16 模型中，缩小固定 prefill 是否仍表现为生成停顿、新请求等待与完整搬运摊薄之间的权衡？这是普通静态基线比较，不含新 controller。

固定 expert cap48、KV512MiB、token64、maxseq4、BF16/expert/LRU/nested；单次加载后运行32/16/16/32。每格共同做16→32两次暖机，随后排空并重置pager；参考复制与计算保持关闭。正式四文档来自已用公开源索引4–7，用冻结Qwen tokenizer重新分词；输入长度64/64/128/32，输出32/32/16/24，绝对到达0/0/2/4秒。输入SHA `714931e38ed05732c04a476e18e43124ab230ef0955ffdbd6d8a4c6cdf920780`；时钟选择不使用资格化参考计算时间。

每格正式388个计算位置/104输出，四格16请求/416输出；预热另32请求/64输出。静态阈值在原生schedule之前生效，已有ready decode每步推进。CPU测试不代表真实GPU执行，详见 [cpu_checks.json](cpu_checks.json) 与 [protocol.json](protocol.json)。

`source/run_native_pager.py --qwen-static-prefill` 是入口，使用同样四个loader路径参数。遗留CLI `--prompt-tokens 64 --output-tokens 8 --arrival-interval 0.05` 是固定兼容占位，实际输入、逐请求输出长度和到达时钟从冻结prepared文件读取；不能按这些占位值解释本实验工作量。

回读后的分析命令：

```bash
python3 analyze_qwen_static_prefill.py \
  --input-dir readback_r01/results/comparison \
  --launch-dir readback_r01/launch \
  --out summary.json
```

每条TTFT分为声明到达到提交、native queue、首schedule到首token；完整wall含等待与提交滞后，不再重复相加。纯decode最多4×8=32个专家，在cap48内只有一个group，但仍可缺页并需H2D。记录真实payload、calls、groups、逐请求生成间隔及完成；不得从单组或少groups直接推吞吐。所有臂自然生成，后续状态独立，输出差异保留。

两次每臂、单engine ABBA仅支持该有限episode的描述性权衡，不给出噪声上界、显著性、质量等价、持续容量、生产尾延迟或method GO。16/32两点也不是全局最强静态或exact Oracle。无论结果是改善、损害还是权衡，都用于收束本次真实超显存测量；不自动再开机制或阈值搜索。
