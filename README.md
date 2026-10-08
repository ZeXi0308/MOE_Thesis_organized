# 毕业设计：完整工作区快照

本分支保存 2026-10-08 发布任务读取的整个工作区：主研究仓库、C 准入工作树、B/D/E/F/G 研究方向、历史调度工作、证据归档、提示词、共享科研规范及文档。

这是各目录当前文件的快照，不合并或改写原研究分支历史。各目录的最新结果入口决定其科研结论；此发布操作没有重新运行 GPU 实验。

## 文件与原始数据

不超过 2 MiB 的普通文件保留原目录结构。更大的文件无损 gzip 压缩后存放在 `.workspace-data/`，按原内容 SHA-256 去重；单个压缩对象超过 48 MiB 时分片。所有原路径、大小、校验值、压缩对象和符号链接均记录在 `WORKSPACE_MANIFEST.json`。压缩存储不改变原始数据内容。

克隆后还原全部数据（约需 87 GiB 原始文件空间，另加 Git 与压缩对象）：

```bash
python3 restore_workspace.py
```

只还原某个方向，或只校验而不还原：

```bash
python3 restore_workspace.py --prefix E_restore_choice_20261004/
python3 restore_workspace.py --verify-only
```

为保护本地修改，还原器拒绝覆盖内容不同的已有文件。文件快照会记录复制时检测到的源文件变动；各文件不保证来自同一原子时刻。

排除嵌套 Git 元数据及可再生成的依赖/缓存目录：`.git`、`__pycache__`、`.venv`、`venv`、`node_modules`、`.cache`、`.pytest_cache`。各研究目录原有 `.gitignore` 也随快照保存，但本次完整快照包含除此处明确排除项以外的原始实验文件，不受那些忽略规则限制。
