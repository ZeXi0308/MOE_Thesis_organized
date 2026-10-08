# 更强 instruction checkpoint 的独立资格

前一分支已经结束：Qwen2.5-1.5B 的固定摘要 N32 为严格有用 5/16，N1 为 4/16，两者都未达到原门；不再搜索并发，也不启动原混合 P0。本次只改变模型 checkpoint，继续关闭“能否获得两类有用、自然结束的任务输出”这个前提。它不是调度策略实验。

固定模型为 `Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。四片权重合计 15,231,271,888 bytes；官方固定 revision 的七个配置/tokenizer/index 文件已经按响应 commit 与 Git blob ETag 核实，清单在 `qwen7b_model_manifest.json`。权重需逐片下载并核完整 SHA，不能把 metadata 或静态显存估计算作模型已可运行。

复用不变的 `health_native.py` 与 `health_native_summary.py`。Q0 为同一 16 道 GSM8K、八示例、原 Answer: cue、cap1024；Q1 为同一 16 篇全文、同一摘要指令与生成前冻结的三条事实参考、cap512。均使用新 checkpoint 自身的 tokenizer/template，greedy、自然 EOS、min0、无文本 stop、seed20260905。输入不能按旧输出筛选，也不截断。Q0 完成且通过原门后才运行 Q1；1.5B 的通过结果不能授予 7B 资格。

判据沿用 `quality_protocol.md`：Q0 全部完成、0空白、至少12自然EOS、至少8正确且EOS、周期后缀至多1；Q1 全部完成、0空白、至少14自然EOS、至少12篇严格有用、周期后缀至多1。摘要盲化配置/性能标签后按原文评分，并保留 full/partial 上界和模型评审局限。任何有效失败都保留；不改题、seed、cap、质量门救结果。质量通过仅允许设计下一开发探针，不构成质量等价或论文结果。

新机器45495，GPU `3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e`，既有共享锁 `2304:25841682495`。BF16、vLLM0.26、APC开启、FCFS、同步、32seq、1024 batch tokens、4096 context、8GiB KV不变。每格两次短预热、清空APC后全部16请求同时到达、完整drain；单child上限1200秒，原有生成循环上限900秒。输出分别为 `qwen7b-q0-v1`、`qwen7b-q1-v1`，不存在才可启动。

权重持久保留于本任务独占 `/dev/shm/c-qwen7b-qualified-20261001-v1`，另计真实容器RAM占用。现有其它会话约13.84GB模型缓存和本任务1.5B约3.10GB缓存不得删除或修改。每次仅准备一片，网络deadline1200秒、父child1260秒，核对metadata/marker、shm容量、cgroup内存余量及GPU空闲，使用既有nonblocking共享锁；结束后释放，不自动接着下载或启动GPU。失败partial保留但不充当验证权重。推理前重新核对全部文件、内存与GPU状态。只允许回收本launcher创建的超时进程组，不能终止其他任务。

配置静态计算每token KV为 `2×28×4×128×2 = 57,344 B`。32条各4096 token总共仅7GiB，低于8GiB；因此本资格域在完整上界下就可容纳，不运行一个已知没有容量动作机会的32请求混合压力探针。两类资格若都通过，另行以实际tokenizer和native合理默认并发检查固定真实请求的几何，再决定一个原生容量存在性实验。当前没有128请求实验冻结或新控制器，不能降低KV或增加输出cap来挽救旧P0。

代码、输入、模型清单、原参考与本计划统一冻结到 `qwen7b_freeze_v1.json` 后才允许远端准备。运行命令为既有venv Python执行 `qwen7b_launch_v1.py --prepare-shard model-00001-of-00004.safetensors`（其余片分别单次执行）、`--run-q0`，通过后 `--run-q1`。准备耗时与GPU实验耗时分别记账。
