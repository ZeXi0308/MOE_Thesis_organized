# r02 环境候选：输出路径门禁增量确认

固定版本 `pkg/run.sh` SHA-256 `5d6deff0c07969faa47513a2dbc2af04059d3fe30f66315c06dcdc72fd177583`；manifest SHA-256 `eb46b043af4c70b5f3d7cf6f06d7e679ab5561c3c01958cb7806ce1ffc1638a3`。从新 shell 删除 `package_root`/`resolved_output` 判断的五行后，SHA 精确恢复上一版 `78987280…fae58`；差分仅为此路径门禁。它在取得锁、创建 `launch-once` 和启动 Python 前，将绝对输出路径经 `realpath -m` 规范化，并拒绝候选包根目录及其子目录，防止本次结果写入包内影响后续精确文件集合校验。

原 r02 的 24 个非入口 `pkg/` 文件仍逐字不变；新 manifest 精确覆盖 26 个 payload 且 **26/26** 哈希匹配，provenance 的 shell/manifest SHA 相符，`bash -n` 与本地 verifier 通过。`FINAL_CANDIDATE_QA.md` 所列现场授权、共同锁、固定 revision 离线缓存、失败日志和 Linux 运行时资格边界仍适用。未连接 SSH 或运行 GPU。
