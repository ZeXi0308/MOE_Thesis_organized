# Logical alignment 数值资格

**Verdict：QUALIFIED_NUMERICAL_ONLY / MEASUREMENT_ONLY。** 在此次实际输入及16层上，logical64分桶后映射到384物理槽的输出与原physical384路径逐位相同。性能、任务质量、长请求恢复、其他模型尚未测量；该结果不升级旧X的净收益裁决。

| 项目 | 实测 |
|---|---:|
| 请求 / 生成token | 5 / 40 |
| 原pager全部 / measurement层调用 | 288 / 176 |
| measurement真实行数 | 160行×64次；5行×112次 |
| actual同输入比较 | 176/176 finite、allclose、bit_equal；maxabs=0 |
| 各层首次大输入的前缀比较 | 16层×1/16/32/64/128/160，96/96逐位相同 |
| 实际helper调用 | 513：actual两路352 + 额外前缀160 + 负控1 |
| 固定首call错map负控 | finite=true，allclose=false；maxabs=0.23370361328125，相对L2=1.0077784286 |
| 实际专家唯一存储 / KV | 4,831,838,208 B（384槽）/ 1,073,741,824 B |
| 完整子进程费用 | 38.828256588 s，仅资格成本，不用于性能对比 |

数值门槛在执行前固定为finite且allclose(atol=0.01, rtol=0.01)；bit_equal只报告，未事后更改门槛。160前缀复用同一actual比较，未重复计为额外kernel。错map在第一measurement调用112上，将全部active物理槽循环错配；其输出未返回模型。其余每次均返回logical路径输出推进后续模型状态，参考路径仅作当前同状态比较。

原11份运行时代码、物理权重存储、private21×16+shared48、LRU计划和拷贝路径不变。真实helper记录显示原路global384/map384/True，新路global64/map64/False，BF16 GEMM配置相同。两项旧qualification flag关闭；warmup走原路，measurement才启用本资格包装。所有额外参考/前缀/负控时间均保留。

本检查复用fresh-cohort的第二组前5文档，P128/O8、同时到达、token160/prefill32；不是新性能holdout。CPU分析将原始request/output_events、pager call IDs、上下文、资源和helper记录关联，结果issues=[]。张量本身未保存，CPU没有重新执行allclose；数值证据是运行时实际torch比较及其源码路径。

## 原件与边界

- [协议](protocol.json)、[CPU复算](analysis.json)、[原件](attempt01/results/qualification/logical_alignment.json)、[回读核对](readback_verification.json)。[独立限定复核](EXPERIMENT_AUDIT.md) PASS，P0/P1=0；fresh同族审计，接受状态provisional。
- 输入包102403 B，SHA `daac675524e2bee19ea9f5db127823a886e2fdb47290d55267529eb7bd664edd`，22项输入逐个核对。
- 回读包716378 B，SHA `2aaabe8803bf3bfd10b7fce7d7ce74e087c6c96c70573f425a8f2425b6570d07`，49成员/48载荷，全部hash/size一致。
- `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac`共享dirty；运行身份以封包文件hash为准。
- remote `/root/autodl-tmp/logical-alignment-qualification-20260914-root-r01`，controller48237，finished1789330305.644020。driver末次GPU空检查通过，随后下一会话接手；本组没有保留窗口。

证据层级为原生eager pager集成的同状态数值资格，不是生产服务/语义质量证明。最强数值参考是已资格验证的physical X；强性能基线F尚未在新变体对照中运行。Oracle/full-request headroom未测。本轮关闭的是索引与分桶合法性的当前执行例，不把零数值差异外推至所有输入、精度和硬件。没有发现本路径的数值失败；遇到新shape/model/dtype需重新限定资格。

唯一下一实验：同384槽/1GiB KV、独立文档、F/X/logical-X及反序六个新引擎，先覆盖实际编译域，保留启动/准备/映射/拷贝全部成本，判断新路径是否优于F和X且维持请求生成进度。logical分桶使用已有vLLM原语，不作为独立机制新颖性主张。
