# 候选计划与集合成本

重审后compact已完成一次真实启动，dense因锁忙未运行；其他算子比较尚未执行。NA不是零。源码兼容不等于竞争力与正确性已验证。机器、版本、源码及原始启动结果位于`evidence/`。

| 候选 | 真实执行差异 | 内存机制 | 时间 / 稳态集合成本 / 最终 KV |
|---|---|---|---|
| native Triton | 当前已安装 tuned/default 配置，按 M 选择 | 与其他 Triton tile 共享；相同配置按 alias 去重 | 均未测 |
| Triton T16 / T64 | M tile 16 / 64，改变实际 kernel specialization 和专家填充 | 主 workspace 形状相同并共享；不能假设两倍驻留 | 均未测 |
| FlashInfer CUTLASS BF16 | 独立 CUTLASS kernel，SM120 被显式允许 | 共享 workspace；权重布局/持久转换须计入 | 未测，强基线候选 |
| compact | 同最大512，实际10 PIECEWISE+9 FULL | workspace128 MiB，graph池reserved28 MiB；估计132 MiB/capture差76 MiB不相加 | 稳态整设备非KV14.037 GiB（含模型/context等）；KV36828总/36827可用；启动35.932s |
| native / dense | 同最大512，较密batch桶集合 | 非可加共享成本，不能从compact按图数外推 | native未运行；dense锁忙退出75，未初始化CUDA |
| 既有 D 的 43 PIECEWISE + 27 FULL 图 | 历史默认 graph 组合 | 原生日志全集 capture 差约 0.17 GiB、估计 0.14 GiB、capture 6 s | KV 36765 blocks，含 1 保留块；仅历史背景 |

TRTLLM BF16 限制为 SM100 family，不能因设备名含 Blackwell 就纳入 SM120；B12x 和 CuteDSL 的 NVFP4 路径也不能作为 BF16 候选。依据和行号见 `backend_audit.md`。

模型参数导出的 KV 成本为 128 KiB/token，16-token block 为 2 MiB。Triton 模块化路径的两个主 buffer 合计 64 KiB/token，global workspace manager 按最大使用量复用，不乘以 16 层或保留 tile 数。profile 先暖最大 token 数，因此删除小图未必释放 workspace。这些是结构推导，不是实测集合内存。

历史 0.17 GiB 是舍入的 capture 前后设备 free-memory 差，采样早于最终清 allocator cache；它不等于精确 graph 独立驻留，也不保证可退还 KV。即使将这个量全部假想返还，也仅约 87 个 KV blocks，远小于历史 high 的 4733-block 余量。这个对比削弱容量敏感性的预期，不证明当前 GPU 的门槛 C 为假。

原型记录整个集合，不相加单计划峰值。native每图最小1 MiB估算、临时池和正式capture须分开；估计改变blocks不自动证明空间节省。本次清cache释放106 MiB而KV总块数仍36828，确认释放cache不自动扩容KV。一个已完成端点不足以识别组合间内存效应。
