# 整组串行执行封装（CPU 已检，GPU 未运行）

`serial_group.py` 是 root 的**执行封装**，不做恢复决策，也不改变两个候选包的 manifest。它目前只认识已冻结的 `candidate_ltr_r02_env` G64 资格单格和 `candidate_h1` 的五种 H128 格；尚未建立 H128 LTR 强基线性能包，所以当前不得把两种输入作为同输入性能组排比。第一份实际会话计划只能在用户明确机器、GPU、host 与总时间/费用范围且现场身份通过后冻结。计划不得包含凭证。

执行者在授权的 Linux 主机上先只读核 GPU UUID/进程与显存、共同锁文件及 inode、当前进程 cgroup v2 硬限、离线模型的**固定 revision**、pinned Python/vLLM 源码和包字节；共享锁路径须是其它执行者实际使用的同一文件。计划固定 `device:inode`；封装取得 fd 9 后及每格前均核它与路径和计划三者一致。把批准的总时间上限与费用上限核成保守的会话墙钟上限。将完整绝对路径及实际批准值写入**新** JSON 计划，先算其 SHA-256 并离线审阅。示例结构如下，`<...>` 必须逐项替换，不能直接运行：

```json
{
  "schema_version": 1,
  "authorization_reference": "<非机密的用户授权记录>",
  "authorized_gpu_uuid": "GPU-<已批准且现场核对的物理 UUID>",
  "approved_host_bytes": 1,
  "approved_total_wall_seconds": 1,
  "lock_path": "/<已存在的共同锁文件>",
  "expected_lock_device_inode": "<现场固定的 device:inode>",
  "session_dir": "/<全新会话收据目录>",
  "python": "/<固定 Python 可执行文件>",
  "hf_cache_dir": "/<已核 revision 的离线模型缓存>",
  "cgroup_memory_max_file": "/sys/fs/cgroup/<当前进程 cgroup>/memory.max",
  "cells": [
    {
      "kind": "ltr_r02_g64",
      "package_dir": "/<精确候选 staging>/candidate_ltr_r02_env",
      "output_dir": "/<全新单格输出目录>",
      "max_wall_seconds": 1
    }
  ]
}
```

上面的数字 `1` 仅是不可执行占位。真实总上限须至少覆盖**所有格的完整单格上限之和 + 每格 45 秒 + 20 秒**；每格上限须大于 6 秒。启动、异常清理、归档和核 GPU 空闲共用这些余量；若剩余总时间已不足某格的完整预定上限，就在该格启动前停止，不缩短后再测。`h1` 格另外填 `variant`、`mode`、`gate`，仅允许当前 H1 入口的五种组合；科学执行顺序仍按 [RUN_PLAN.md](RUN_PLAN.md) 固定。实际批准的成本费率/上限由 root 在冻结计划前换算成保守的总墙钟，封装只执行墙钟上限，不能自证授权或价格。

```sh
sha256sum /absolute/path/to/frozen-plan.json
python3 serial_group.py /absolute/path/to/frozen-plan.json \
  --expected-plan-sha256 <上一步的 64 位 SHA> --validate-only
python3 serial_group.py /absolute/path/to/frozen-plan.json \
  --expected-plan-sha256 <同一 SHA>
```

`--validate-only` 不取得锁、不调用 `nvidia-smi`、不启动包。正式执行时，封装要求现有普通锁文件并持有同一 fd 9 直至每格结束、输出复制及双侧 SHA readback；每格入口复核 fd 9 inode。整组只在前台串行启动一个包，限制剩余会话时间，并在进入下一格前查 `nvidia-smi` 的 compute process 清单；故障、超时、归档失败或状态不明立即停止后续格，留下 `receipt.json`、`launch.log`、归档及散列（若生成）。单格原入口仍核 manifest、固定源码、cgroup `memory.max` 和物理 GPU UUID；所有 launch-once 标记保持其原语义，失败不得原地重试。

`nvidia-smi` 的 compute process 清单不能证明所有显存占用或逃离进程组的 worker 已消失。任何失败后 root 必须现场复核 GPU、进程树、共同锁及 host 占用；有存活 worker 或 UNKNOWN 时，不启动下一会话。会话封装不替代机器资源许可，也不自动清理外部进程。当前 DNS 尚未解析，授权范围与机器现场未核，本封装及两候选包均 `GPU_UNRUN`。
