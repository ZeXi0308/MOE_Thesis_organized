# CPU 执行记录

- 首次执行在旧 `cohort0-block0-least_progress` 读取 `effective_victim_order` 时遇到 KeyError；尚未创建分析输出，没有新的科学结果。
- 只读核对旧冻结 `config.json`：`rotation_victim_order=least_progress`。分析器仅对缺少逐步字段的旧固定排序格式使用该配置；不修改旧决策。未知排序仍拒绝。
- 同三份原始输入重新执行完成：3581 个前态块守恒、3179 次选择器调用、26 个交换前态、506 个候选、26 次原生实际释放；所有检查通过。完整原始输入路径与 SHA256 保存在 `analysis.json`。
- CPU 负控通过：排名第一不可资助但另一个可资助时检测到漏动作；可资助第一名不误报；缺失块数、第一名不在候选集合中均拒绝。
- Python AST/JSON 解析通过，限定路径 `git diff --check` 无输出。脚本248行；没有 GPU、下载、原始文件写入或共享主文档修改。
