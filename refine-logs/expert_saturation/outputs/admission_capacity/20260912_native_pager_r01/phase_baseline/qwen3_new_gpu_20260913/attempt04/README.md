# Qwen3 r04：新实例与根盘分片接续

**CPU_PREPARED / GPU_UNRUN。** Q3冻结input_manifest38项逐字节核验通过，原Q3不改。沿用全部30 payload文件（17 runtime）、原detached launcher、模型revision、请求、数值gate及通过后的static32/16/16/32。没有新机制或阈值。

新GPU `GPU-bd5e9bb9-f98b-db5b-cc1c-5857c39f0bdc`，5090/32607MiB/driver595.71.05；CPU `Intel(R) Xeon(R) Platinum 8470Q`，cpu.max `2500000 100000`。环境原件`../environment_20260914_2240.json`记录boot_id `f2bb6547-7857-4d80-9f95-1014f63c5590`、PID1 start `1789396119.2`，11项安装源码匹配，框架版本不变。它不是启动时GPU空闲证明；本包启动前重查UUID/显存/进程，并保存`launch.host_identity_before/after`（boot_id、原PID1 stat/start ticks、CLK_TCK、kernel btime与推导start unix）及`gpu_identity`。

仅存储行政路径改变：stage `/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r04`；results `/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r04`；loader workspace **`/root/qwen-streaming-shards-20260914-r04`**。三者独占新建，已存在拒绝。父进程分别要求根盘分片余量≥6GiB、数据盘结果余量≥1GiB，记录各路径、device、available bytes和是否不同文件系统。原serial loader仍在workspace下独占生成serial子目录，按原顺序逐片完整SHA/消费/移除自己的成功片；失败partial保留，不预载61GB到盘、不改变下载算法。

最新父线程只读记录root free 12,309,196,800B、data free 3,657,932,800B；这些随前组运行变化，启动必须重查。本包不删除pip缓存、模型、旧片或其他实验。完整61,066,575,648B下载成本仍保留；新实例没有本包测得的持续带宽预测，旧3.48小时短样本外推不作新机保证。原总21600秒、每episode600秒、90GiB父界限/82GiB匿名守卫/起始anon<8GiB保持不变。

共同flock覆盖加载、资格、四格和退出；worker继承锁FD。busy/query/foreign失败不启动或只回收本次PGID。TERM/HUP/INT有父级清理；SIGKILL/主机丢失/磁盘写失败仍可能缺终态，保留UNKNOWN并独立回读。资格先检查48层finite和layer47同precall分组逐位归因；通过后继续原四格，失败保留且性能0格。局部资格不是质量、resident全引擎等价或方法GO。

三类CPU检查通过：原payload/科学参数相同及双盘阈值单独拒绝；busy拒绝/worker锁继承/真实exit0与7；监控失败和父SIGTERM只清理自身组、保留无关组及终态。真实Linux host identity读取本地未执行，测试有明确替身；不是GPU资格。结果在cpu_checks.json，复现`python3 cpu_check.py`。本包只准备，父线程按funding-filter→staged-store等已登记前序完整终态后现场接续，不自动候卡或retry。
