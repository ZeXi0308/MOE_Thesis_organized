# 毕设论文资料 · 文档入口

> 2026-07-20 整理。根目录不再堆散落 markdown；从这里开始读。

## 先读什么

| 优先级 | 文档 | 说明 |
|---|---|---|
| 1 | [Approach Registry](01_current_status/MoE_Approach_Registry_2026-07-19.md) | 各路线当前状态总表 |
| 2 | [16 候选复核修订版](01_current_status/16候选_复核与重跑_进展报告_2026-07-19.md) | 判死/未过门/脚本路径 |
| 3 | [选题完整时间线](01_current_status/MoE_全部候选选题完整时间线_2026-07-19.md) | 16 条历史 |
| 4 | [GPU 四轮审计](02_gpu_audits/) | 2026-07-20 有效性实验 |
| 5 | [批判性研究设计](01_current_status/三条MoE研究方向_批判性研究设计与投稿路线_2026-07-20.md) | 投稿向总设计 |

## 目录结构

```
docs/
  01_current_status/     当前状态、registry、时间线、深化设计
  02_gpu_audits/         GPU 有效性审计报告（首轮→第四轮）
  03_candidate_reports/  单候选协议/审查（TokenRace、RouteFidelity…）
  04_research_plans/     research_plan / research_report / 严格研究报告
  05_idea_a_legacy/      早期 Idea A 主实验与 TaskList
  06_meetings_notes/     会议、讲稿、早期方案
  07_literature/         文献笔记与相关 PDF
  08_run_conclusions/    从 outputs 抽出的结论文档（按路线）
  99_archive/            被新版取代的旧稿

literature/              PDF 原文（AdapMoE、HOBBIT…）
experiments/idea_a_mac/  代码 + 原始实验产物
  outputs/README.md      实验 run 目录索引（数据仍留此处）
archive/                 根目录 smoke outputs、临时残留
```

## 实验数据在哪

- **原始 run 目录**（CSV/JSON/日志）仍在  
  `experiments/idea_a_mac/outputs/<run_name>/`  
  未搬家，避免打断脚本与复现路径。
- **可读结论文档**已复制/移动到 `docs/08_run_conclusions/<topic>/`。
- `outputs/` 顶层散落的旧报告已移走，原位置留有指向新路径的 stub。

## 代码

- 主实验代码：`experiments/idea_a_mac/`
- 系统侧（若有）：`experiments/idea_a_system/`
