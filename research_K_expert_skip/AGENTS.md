# Marginal-cost-aware expert skipping

遵循 [工作区科研原则](../AGENTS.md)，当前明确任务见 [TASK.txt](TASK.txt)，当前进展见 RESULTS.md。用户要求机制、图表与论文持续建设，负结果写为“当前效果 X，下一步 Y”；不增加审计包、预注册或 reviewer 流程。

- 仅修改本目录，不覆盖其他研究线的代码、成果或运行进程。
- 主会话统一负责远端 GPU，沿用 `/root/autodl-tmp/moe-research-gpu.lock`，先核对进程、非阻塞取锁；总预算约 15 GPU 小时。子任务仅 CPU/文献/本地实现，不自行提交 GPU。
- 真实质量、性能与格式比较使用对应同一权重，保留失败和未完成状态；模型估算、路由计数的逻辑权重字节与硬件 DRAM 计数分开报告。
- 凭据不写入文件，不回显；远端结果及时回传本地，保留复现配置与源码。
- 修改范围分工：主会话负责集成、运行、RESULTS.md/configs/reproduce.md；子任务可分别负责 core policy、历史数据分析、paper/，不互相覆盖。
