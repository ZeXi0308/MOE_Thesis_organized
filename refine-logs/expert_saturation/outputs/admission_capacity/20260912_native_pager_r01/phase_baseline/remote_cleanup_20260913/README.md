# 5090清理对账

2026-09-13。以下依据已有回执核对，本次未访问远端。清理已结束，没有待执行的迁移任务。

| 项目 | 结果 | 回执 |
|---|---|---|
| 共享实验的远端重复raw | 已删除40个文件，共11,973,860,770 bytes；本地raw、对应tar及远端tar全部保留。包含本目录核验过的36个raw，另有headroom-fast的4个。 | [删除回执](../../../20260913_rotation_victim_order_r01/remote_cleanup/deletion_receipt.json)、[保留检查](../../../20260913_rotation_victim_order_r01/remote_cleanup/postcheck.json) |
| pip HTTP下载缓存 | `/root/.cache/pip/http-v2`已清理并确认不存在，释放系统盘4,227,780,608分配字节；模型、venv和实验数据未动。 | [缓存清理回执](pip_cleanup_receipt.json) |
| 旧raw迁移尝试 | 36个源raw此前已由共享清理删除，源文件检查失败，在任何数据搬移或删除前退出；`verified=[]`、`migrated=[]`、`deleted=[]`。不重试、不再待执行。 | [迁移回执](migration_receipt.json)、[检查与搬移顺序](remote_migration_source.py) |

本目录[本地备份清单](local_backup_manifest.json)与[早期审批拒绝](approval_rejection.json)保留为历史记录；其待核验/未执行状态不代表当前仍有清理任务。36个候选路径及SHA均已与共享删除回执对应。本次仅补此对账说明，原回执、raw、报告和结论台账不改。
