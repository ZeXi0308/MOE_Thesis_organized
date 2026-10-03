# Qwen3 westc r03：原数值定位与条件静态对照接续

**CPU_PREPARED / GPU_UNRUN。** 只接续旧合同，不新增机制或阈值。r02旧RUNNING和失败分片保留；最新归档确认4片/4845源张量消费、无数值或性能结果，退出原因未知。

原30 payload文件及17 runtime逐字节不变；原detached_launch.py不变。一次加载Qwen3-30B-A3B BF16，48层/cap48/token64/maxseq4/KV预算512MiB，实际KV应按原block公式为511.5MiB。先原4请求layer47同precall分组归因与48层finite gate，通过才继续原static32/16/16/32四格；未通过保留失败且性能0格。局部资格不等于任务质量或resident全引擎等价。

父进程新增共同锁`/root/autodl-tmp/moe-research-gpu.lock`，加载前非阻塞获取，覆盖GPU预检、加载、资格、四格、子进程结束和终态写出。锁FD由本次worker继承。GPU忙/查询失败立即ABORT；其他进程不终止。TERM/HUP/INT进入仅自身worker PGID的TERM→20秒→KILL回收；NVML初始化、shutdown异常保留。普通SSH断开不控制detached monitor或父进程。SIGKILL、主机丢失、磁盘写失败无法保证终态，此时保留UNKNOWN并独立回读；不自动重试。

父cgroup **90GiB=96636764160B**、匿名守卫 **82GiB=88046829568B**、开始前anon<8GiB保持原样，运行中检查current/stat/OOM和limit。不是独占RAM保证。设备UUID仍`GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9`，由最新Qwen只读记录核对，启动仍必须现场确认UUID/显存/占用。

完整下载 **61,066,575,648B**；最大分片 **3,999,975,472B**。同hf-mirror源64MiB短样本13.75626秒、4.8784MB/s，仅线性估计下载 **12517.7秒（3.477小时）**，不含SHA/克隆/加载/JIT/数值资格/四格，也不保证持续带宽。原总上限21600秒、单episode600秒不变。串行每次仅一片，完整SHA和消费成功后只删除自己的该片；失败partial保留，不预存61GB、不改下载算法。数据盘最新余9,464,983,552B，启动仍要求>6GiB；旧partial、其他数据和原加载成本不清除。

CPU检查：源码/参数/科学协议一致；busy不launch；worker继承锁、退出0/7真实保留；监控故障及父SIGTERM只回收自身组、无关组仍活且终态落盘。NVML/memory/preflight使用明确CPU替身，真实OS subprocess/flock；不是GPU资格。初次本地权限失败与修复边界在revision.json保留。

新stage `/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r03`；results `/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r03`。本目录只封包不启动；父线程按现有context/streaming/fidelity协调顺序接续，共同锁不替代队列。最终分析继续用原analyze_qwen_localized_static.py，原静态两点不是全局Oracle，也不完成整个研究问题。
