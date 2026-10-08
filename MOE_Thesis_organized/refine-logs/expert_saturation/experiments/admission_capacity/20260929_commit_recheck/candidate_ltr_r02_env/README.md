# LTR-style r02 G64 环境入口候选（CPU 已核，GPU 未运行）

这个新目录仅准备已接受 `20260915_natural_ltr_style_component_r02` 的**单格** G64/T30/Q10 原生生命周期诊断在新机器上所需的环境入口。来源包的 30/30 manifest 文件在同一仓库 HEAD `76d6d888de42081c63cd440a8a67623161d8181f` 的隔离检出中逐文件 SHA-256 匹配。候选复制其 25 个 `pkg/` 文件，仅替换 `pkg/run.sh`；runner、LTR 策略、工作负载、warmup、测量、safe cap 和固定 vLLM 源码哈希这 24 个文件逐字不变。旧接受包及其 archive 身份不变；此处的 26 文件 `manifest.json` 和 `provenance.json` 是**新候选身份**，原 archive 字节 SHA 未重验。

`pkg/run.sh` 只接受一个绝对且位于候选包外的输出目录，仍固定 `--ltr-threshold 30 --ltr-quantum 10` 并执行同一个 `run_ltr_style.py`。启动前必须由唯一 GPU 执行者设置 `R02_AUTHORIZED_GPU_UUID`、`R02_PYTHON`、`R02_HF_CACHE_DIR`、`R02_LOCK_PATH`、`R02_MAX_WALL_SECONDS`、`R02_APPROVED_HOST_BYTES`、`R02_CGROUP_MEMORY_MAX_FILE`、`R02_EXPECTED_MANIFEST_SHA256`；最后一个值由 root 在现场读取前固定为 `f58340cd5228ea7339f0d4081176aab265583cb824a26026d19b8f7ae93cc8a7`。shell 核进程自身的 cgroup v2 `memory.max` 属于所给文件、有限且不超过批准 host 字节，要求共同锁文件预先存在，打开时不截断；若父执行者已持有 fd 9，先核 fd 与同一锁文件 inode 相同，再复用，使外层锁覆盖换格与归档。它在锁内核物理 GPU UUID、外部固定的 manifest SHA、精确 26 文件集合和已安装 pinned vLLM 源码（强制 `PYTHONOPTIMIZE=0`，使原 `preflight.py` 的断言生效；`PYTHONDONTWRITEBYTECODE=1` 避免本地 bytecode 改变包集合）。哈希预检、源码预检、runner 均按整格剩余墙钟受限；`launch-once` 标记阻止同一 staging 改输出路径重跑，预检失败也消耗身份。root 另核整个串行实验总预算、实际 GPU UUID/进程、host 占用、固定 revision 的离线模型和输出归档；脚本参数本身不代表用户授权。

本包只回答旧 r02 的原生 store/load/flush、量子与 EOS 生命周期是否真实可运行，**不提供性能对照或完整 LTR 结论**。资格失败保留 `INCOMPLETE` 并停止后续性能格；成功后也须另封同 H128 输入的性能包，不能把本包改称 H128 强基线。与本目录的 fair LTR successor overlay 是不同对象：后者改策略，未通过原生资格，本候选未引入它。

CPU 检查：来源 30/30 SHA、候选 26/26 SHA、24 个未改 `pkg/` 文件字节匹配，`bash -n pkg/run.sh`，以及只读 `check_lifecycle.py` fixture。新 shell 的 cgroup/GPU/锁执行路径必须在用户批准的 Linux 机器上做现场验证；本地 CPU 检查不能替代它。当前新 SSH 地址本地 DNS 未解析，GPU 身份、host 与总时间/费用范围尚待用户明确；没有远端连接、上传、GPU 初始化或后台作业。
