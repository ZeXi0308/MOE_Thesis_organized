# G64 r02 环境候选最终静态 QA（2026-09-29）

审查固定字节：`pkg/run.sh` SHA-256 `ffcd7abdfb44f5f31d440311fd75e0c3981319ee2a85f9468f7aa5d247333009`；`manifest.json` SHA-256 `8c63588870e641f27daf1f2108bef4f5269bef11ba2e2eb2b44c2f8d409e4388`。重新比较原 r02 与候选各 25 个 `pkg/` 文件：仅 `pkg/run.sh` 改动，其他 **24/24** 字节不变。当前 manifest 恰覆盖 25 个 `pkg/` 文件和 verifier，**26/26** 哈希匹配；provenance 的 shell/manifest SHA 相符，`bash -n` 与 verifier CPU 执行通过。前一份 `CANDIDATE_QA_ADDENDUM.md` 针对旧 `facd…` 版本列出的代码缺口，以此处固定新版本的结论为准。

## 四处缺口的当前状态

1. **锁与物理卡：代码门已加强，授权仍须现场证明。** shell 要求锁文件预先存在且为普通非 symlink 文件，`exec 9<>` 不再截断；继承 fd 9 时核其 inode 与路径相同，随后 `flock -n`。在锁内有受限时钟的 `nvidia-smi` UUID 查询，确保传入的 GPU UUID 在本机恰出现一次；原 runner 另核可见 GPU 数和 compute 进程。`R02_LOCK_PATH` 和 `R02_AUTHORIZED_GPU_UUID` 仍是 root 提供的值，脚本无法判断它们是否等于用户授权或 H1 实际使用的锁。root 须现场固定同一锁 inode、用户批准的物理卡、显存/进程和前组终态；共享锁文件在整组期间不得删除或替换。存在性检查与 `<>` 打开之间仍有常规文件替换竞态，不能把脚本当作对不合作进程的强制隔离。
2. **时限：先前昂贵阶段无硬限的问题已关闭。** `nvidia-smi`、26 文件 verifier、vLLM 源码 preflight 和完整 runner 均经 `run_bounded`，每段使用同一个 `started_s` 的剩余秒数并留 5 秒 TERM/KILL 收尾。较早的 shell 参数/cgroup/manifest 小检查不在 `timeout` 中；它们在 GPU 初始化前，root 仍须把整组墙钟/费用另行计入获批总预算。Linux 上 `timeout`、cgroup 和锁继承的实际行为尚未现场验证。
3. **单次身份：已按 staging 原子限制。** 锁与包身份检查后 `mkdir ../launch-once`，同一 staging 再次运行会在 GPU 前失败；`run_ltr_style.py` 仍拒绝覆盖已有输出目录。复制出全新 staging 当然能再跑，是否允许另一次领取只能由 root 的唯一执行记录决定。旧 `controller.py` 不在新候选中；root 必须单独保存 shell stdout/stderr、退出码、开始/结束时间及所有预检 ABORT，即使 runner 尚未创建输出目录。
4. **包与离线模型：包完整性门已关闭，缓存内容门仍外置。** shell 要求 root 事先固定的 64 位 manifest SHA，并在 verifier 前核它；verifier 要求 manifest 名称与实际 `pkg/` 文件及自身**精确相等**，再核 26/26 内容。README 现在列出 `R02_EXPECTED_MANIFEST_SHA256` 及本版值。`R02_HF_CACHE_DIR` 只核目录存在，尚不核固定 OLMoE/model/tokenizer revision 文件是否完整；`HF_HUB_OFFLINE=1` 阻止在线补取，但缺文件可能到初始化时才失败。root 应在 GPU 初始化前只读确认固定 revision 的离线缓存，否则 ABORT。

**结论：** 这版没有发现阻止 CPU 封包接受的新增策略/输入漂移或明显脚本语法错误。它仍是 `CPU_CANDIDATE_GPU_UNRUN`，不能从本地 26/26 和上述代码门推出可上机、生命周期成功或性能结果。真正启动前必须有用户明确的机器/GPU/host 硬限/时长费用授权、同一锁和进程现场证据、固定模型缓存及失败日志收据；本次审查未连接 SSH、执行 GPU 或 controller。
