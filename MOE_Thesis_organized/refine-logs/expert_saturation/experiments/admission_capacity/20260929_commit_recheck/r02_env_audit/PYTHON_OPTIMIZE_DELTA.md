# r02 环境候选：解释器优化级别门禁增量确认

固定版 `pkg/run.sh` SHA-256 `0e0c44e29060385b9e2b49ae8478bc60ade15932d04670e56989a627fa239a32`，manifest SHA-256 `f58340cd5228ea7339f0d4081176aab265583cb824a26026d19b8f7ae93cc8a7`。删除新增的三行 `sys.flags.optimize == 0` 检查后，shell SHA 精确恢复上一版 `5d6deff0…177583`；本次仅加了这一段。它用指定 `R02_PYTHON` 在原 `preflight.py` 前、同一剩余墙钟内检查实际解释器优化级别，防止 `-O` 令原 preflight 的 `assert` 失效；不改变 LTR 策略、输入或 runner。

原 r02 的 24 个非入口 `pkg/` 文件仍逐字不变；候选 manifest 精确覆盖 **26/26** 文件且哈希相符，provenance 的 shell/manifest SHA 相符，`bash -n` 和本地 verifier 通过。仍需现场核用户授权资源、共同锁、固定 revision 离线缓存及 Linux 实际生命周期；本次未连接 SSH 或运行 GPU。
