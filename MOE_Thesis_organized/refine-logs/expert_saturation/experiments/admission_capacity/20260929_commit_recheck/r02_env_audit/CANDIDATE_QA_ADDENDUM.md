# `candidate_ltr_r02_env` 稳定版本 QA 补记

只读检查版本：`pkg/run.sh` SHA-256 `facd7679755a784e3804552f4eee252cc22a938b9400d9939e764c6147846d57`，`manifest.json` SHA-256 `72e6184933e870aee1a890a4fff976e9709c90b893b75e81f4dd14102b0db1e7`。来源 r02 和候选各有 25 个 `pkg/` 文件，唯一差分为 `pkg/run.sh`；其余 **24/24** 原字节不变。候选清单覆盖当前 25 个 `pkg/` 文件及 `verify_package.py`，**26/26** SHA-256 匹配，provenance 中的 shell SHA 相符；`bash -n` 和 verifier CPU 执行通过。该静态结果不能证明 Linux cgroup、锁、GPU 或 native 生命周期。

## 上机前仍须关闭的门

1. **同一真实锁与 GPU 授权：未由脚本自证。** `R02_LOCK_PATH` 只检验为绝对路径；若指向新文件，`exec 9>` 会创建它，甚至截断已有文件，两个 controller 可各持不同的锁而并行。继承 fd 9 的 inode 检查只证明它与**传入路径**相同，不证明与 H1 和其他进程的共同锁相同。root 必须在单一执行记录中固定并现场验证共同锁的实际路径/inode，且不在占用期间删除重建锁文件。`CUDA_VISIBLE_DEVICES` 直接取自自由文本 `R02_AUTHORIZED_GPU_UUID`；runner 仅核可见卡数量为 1、查询全机 compute 进程，没有在 GPU 初始化前将选中物理 UUID 与用户授权值独立交叉核对。root 的现场 UUID、显存/进程和前组终态检查仍是启动条件；查询失败即 ABORT。
2. **整格墙钟硬限尚未完整覆盖。** `started_s` 在 verifier/preflight 之前记录，二者完成后会扣减已过时间；但 `timeout` 仅包住 `run_ltr_style.py`。若哈希检查或源码 preflight 卡住，`R02_MAX_WALL_SECONDS` 不会主动终止它们。获批时长若要求硬上限，须在外层对整次 shell/进程树设置同一上限，或把这两段也放入受限时钟；同时保留当前 runner 内的 TERM/KILL 与 `INCOMPLETE` 原件。180 s 只是测量 capture，三次 warmup、初始化、drain 和关闭仍另计。
3. **一次性诊断身份尚靠 root 外部约束。** 新候选不含原 r02 `controller.py` 的 `launch-once`；它允许任意绝对输出目录。`run_ltr_style.py` 的 `mkdir(exist_ok=False)` 防同路径覆盖，却允许同一候选换目录多次运行。执行前须冻结唯一输出目录和单次领取记录；失败保留且不得挑选有利重跑。若需重试，应另立明确的新包/运行身份和原因，不能称原单格结果。
4. **包身份与模型缓存须现场锚定。** 当前 verifier 只检验 26 个 manifest 条目各自的哈希和数量，不强制“manifest 名称集合等于实际文件集合”；manifest 本身及 README/provenance 未由 verifier 的外部固定 SHA 保护。当前树经独立审查完整，但远端启动前 root 应锚定上面的 manifest SHA、逐项验证上传后的集合与 26/26 字节，再记录执行收据。`R02_HF_CACHE_DIR` 仅要求非空；脚本不在 CUDA 初始化前确认该目录及固定 OLMoE/tokenizer revision 已离线存在。`HF_HUB_OFFLINE=1` 会禁止在线获取，但缺缓存会消耗初始化预算后失败；现场先只读确认缓存，缺项 ABORT。

候选已正确把 `PYTHONOPTIMIZE=0`、cgroup v2 当前进程 `memory.max` 的有限值/上限检查、继承 fd 9 的同 inode 检查和相同 T30/Q10 runner 放进新入口；这些是有效的**前置检查**。目前应保持 `CPU_CANDIDATE_GPU_UNRUN`，直到 root 获得明确资源/预算授权并把上述现场门写入唯一执行收据。原 r02 与本候选均未在本次 QA 中运行 GPU。
