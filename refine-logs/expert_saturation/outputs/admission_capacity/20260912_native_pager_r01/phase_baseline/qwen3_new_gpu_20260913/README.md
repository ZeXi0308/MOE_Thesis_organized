# 新 RTX5090：数值定位与条件静态对照

**r02 RUNNING_LOADING；r01 ADMINISTRATIVELY_INTERRUPTED_DURING_LOADING。** 用户提供新主机后，恢复同一研究任务。新GPU UUID `GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9`，32607MiB，driver595.71.05，Xeon8470Q，主机cgroup90GiB。旧主机的中断运行仍为REMOTE_STATE_UNVERIFIED，保留原证据；新机读数不能与旧机合并成同环境重复。

复用已在新机安装的vLLM0.26.0/Torch2.11.0+cu130/Transformers5.15.1，未修改该环境。11个vLLM关键源码、Torch memory源码及WiSP源码SHA均匹配，完整依赖记录在environment_check02.json。首次检查误用WiSP导入路径的失败保留，随后按实际runner路径通过。

独立WiSP目录为`/root/autodl-tmp/qwen3-research-wisp-86f69720/source`。包SHA `af3a807edeb668cc0b229bea8c33437dc8e9bdef2e05676c08132a4185ba74cc`；17个运行源码和输入与原localized-static包逐字节相同，仅父启动器绑定新UUID/目录/90GiB预算，并把匿名内存保护阈值收紧到82GiB。该阈值不等于物理空闲内存；全部cgroup current/stat/OOM事件仍保留。

原r01的monitor3313、父3314、worker3326已退出，`attempt01_completion.json`记录最终状态。已完整校验/消费2片共2325源张量，第3片752877568B部分文件保留远端；0数值资格化/0性能cell。父wall1428.133秒，2820资源采样无foreign GPU/OOM增量。这是加载时限修订引起的行政中断，不是科学失败。回读SHA `327d646c8b1ca092214443b12725095eb80fd1df9c1af6ba2d51565643097a1a`，54文件、4331573B；不含模型部分分片。

流程沿用原资格化和layer47同分组逐位归因条件，通过后才做static32/16/16/32。完整加载/资格化/切换/四格/关闭成本保留。当前尚无新数值或性能结果；CPU迁移检查不等于GPU成功，也不完成研究目标。

加载时限检查：第1片约774秒，第2片从DOWNLOAD到下一片DOWNLOAD约408秒，速度明显变化。64MiB备用源单连接样本4.25MB/s、有效四分段样本6.84MB/s，后者按各段HTTP206/Content-Range及合并前缀SHA通过；不证明全片完整性或持续带宽。失败的首个分段样本与一次本地DNS失败保留。未替换运行中的下载路径。第二片约9.8MB/s的观测使原7200秒时限仍有完成机会，因此继续r01；线性预计不保证后续时限。`transport_timing_decision01.json`记录早期判断，`watch_attempt01.jsonl`保留后续实际状态。

第3片连续采样回落到3–4MB/s后，以上继续判断被新数据修正。逐项核验PID/父PID/命令/输出路径/PGID/SID及没有性能cell后，仅SIGTERM本worker3326进程组；父监控自然回收，完整原记录保留。`administrative_stop_attempt01.json`记录原因和实际动作。

`attempt02/`已使用21600秒总启动时限与新输出路径，包SHA `06cd9efc13f87bd01172758580f077846ac6118831e387c7593e8d00facfff94`。17个runtime源码、输入、预算、数值归因条件及600秒单episode时限不变。r02通过原GPU空闲检查，monitor5090、父5092、worker5127在独立只读观察中确认存活。stage为`/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r02`，results为`/root/autodl-tmp/qwen3-new-gpu-localized-static-v026-r02`，观察记录为`attempt02/watch_attempt01.jsonl`。启动SSH已结束，观察连接不会控制实验进程。两次尝试的加载成本分别保留，不合并成一次成功启动成本。

最新状态：[2026-09-14只读回查](RECONCILIATION_20260914.md)确认r02旧进程已不在，只完成4分片/4845源张量；0数值/0性能。原RUNNING保留，退出码/原因未知。新r03仅CPU准备，不与returned weste尝试混合。
