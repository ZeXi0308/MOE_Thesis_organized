# 5090远端重复副本清理

2026-09-13，用户授权清理现有5090无用文件后执行。只删除四轮已完成且本地有完整raw及回传归档的远端重复raw.json：headroom-fast 4格、fourarm 8格、holdout 20格、victim-order 8格，共40个文件。

删除前已核验每个本地归档SHA256与原执行记录一致、本地raw与远端raw完整SHA256/字节数一致，并确认没有相关runner在运行；删除操作前再次核对远端哈希。没有按模糊文件名递归删除目录。

结果：删除11,973,860,770 bytes（11.1515 GiB）；数据盘由97%降到75%，可用约1.6GB→13GB。保留本地所有raw与归档、远端压缩归档及其余文件、模型缓存与运行环境。旧报告记录的远端results/raw路径现为已清理副本，完整科学证据在各bundle的本地execution目录；旧报告未改写。

具体40路径、本地备份与指纹：[candidate_manifest.json](candidate_manifest.json)；远端比对：[remote_verification.json](remote_verification.json)；逐文件删除回执：[deletion_receipt.json](deletion_receipt.json)。远端也保存REMOTE_DUPLICATE_CLEANUP.json。
