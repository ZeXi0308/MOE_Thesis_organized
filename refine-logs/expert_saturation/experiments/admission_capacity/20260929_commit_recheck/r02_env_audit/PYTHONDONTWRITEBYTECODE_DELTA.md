# r02 环境候选：字节码开关增量确认

固定版本：`pkg/run.sh` SHA-256 `789872801de7eeb7e31bf4a4f53352f7a080153451b7c9ce193b2b6b07bfae58`；manifest SHA-256 `8a109e0296b21c8961f355ef0376256865d7961932e27df484ec0189ba735704`。把新 shell 中唯一的 ` PYTHONDONTWRITEBYTECODE=1` 删除后，字节 SHA 恰恢复上一审查版本 `ffcd7abd…333009`，故这次 shell 差分只有该导出项。它位于 verifier、preflight 和 runner 启动之前，子 Python 进程的常规导入不会在包内生成 `.pyc`/`__pycache__`，使后续精确文件集合校验可重复；已有额外文件仍会被 verifier 拒绝。

重新核对原 r02 的 25 个 `pkg/` 文件，仅 `run.sh` 改动，其余 **24/24** 原字节不变。候选 26 项 manifest 精确覆盖当前 payload 且 **26/26** 哈希匹配，provenance 中的 shell/manifest SHA 相符；`bash -n` 与本地 verifier 均通过。此前 `FINAL_CANDIDATE_QA.md` 的现场授权、共同锁、固定 revision 离线缓存、失败日志及 Linux 运行时资格边界仍适用。本次未连接 SSH 或运行 GPU。
