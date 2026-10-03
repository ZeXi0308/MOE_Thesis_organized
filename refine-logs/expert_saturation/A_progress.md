# A progress — 2026-10-02T10:02:01.322235+00:00

本轮目标完成于“形成足以决定换方向的具体证据”这一允许的终点。决定：停止当前 h/d 自适应量子规则的开发，保留已实现的原生资源/执行机制，下一研究问题转向 victim/peer 完成代价。不是整个恢复量子家族 NO-GO，也未宣称达到 CCF-C 投稿标准。

## 主要实测结论
- 强对照普通回填/原生/原有fund均完成128请求。native/ordinary/fund P95输出间隔6.965/3.885/1.575秒；fund比ordinary实际速率低1.52%、平均flow高1.85%，86/128请求完成更晚。原有funding/priority/protection并非新增创新。
- 新原型跨全量子物理KV、运行槽和执行token预留，并选择性接纳peer。26包内与6纯模型CPU检查已通过；六格真实lease动作均通过核对，受保护目标无实际抢占，peer/victim最终均输出并完成。条件进展不等于无条件q输出或墙钟SLO。
- 两轮q1/fixed4/adaptive比较按相反顺序执行、各128请求。adaptive相对fixed4的P95变化+14.44%/-10.80%，实际速率-1.00%/+0.74%，平均flow+0.35%/+1.62%。已知成本desiredq始终3（103/103、75/75次）。排序不稳定，不继续靠阈值搜索包装独立优势。
- 健康dev16低压三臂均16自然EOS、7/16正确；512页五臂均16自然EOS、8/16正确，块内序列一致。独立16题稳态512页五臂均自然EOS，但native/ordinary/q1/fixed4/adaptive正确数10/8/8/9/8，非native每臂有10/16序列不同。全部健康域均无恢复量子动作；不能主张质量等价或将差异归因于自适应动作。
- 两个健康512页组有完整原生传输完成计数、分开capture/drain。copy-work可重叠，不是暴露请求延迟。文章组完整controller CPU、传输开销及活跃自然任务质量仍未测；上游完整相关系统和结构消融未完成。

## 交付与资源
统一索引：`experiments/admission_capacity/20260929_commit_recheck/recovery_quantum_20261002/README.md`。其中含模型、冻结候选、可复用入口、全部raw和失败、结果JSON及图。决定记录`output/RESEARCH_DECISION_20261002.json`；最终工作稿`paper/draft.tex`与`output/pdf/draft.pdf`，6页、3图、1表，双次编译无警告，最新全部页面已视觉检查。

22个原生实验单元共1360次请求执行（不是1360独立任务）完成。最终controller35189在1790934663.1462655结束、GPU边界为空并释放共同锁；现场后查确认controller消失、无A作业或排队实验。A自有SSH控制连接已关闭。全部权限/上传授权已解决，无待确认事项；无push或公开发布。

唯一未来研究问题：什么当前可观察的peer/victim代价与剩余服务能解释完整请求损失？重新打开本规则需新的异质成本/实际服务headroom证据和有恢复动作的自然任务对照，不以改名或阈值调整复活。历史记录保留如下，旧RUNNING/UNRUN/BLOCKED字段不覆盖此状态。

---
## Historical checkpoint retained below
# A progress — 2026-10-02T01:25:08.952855+00:00

Independent A line; goal BLOCKED on SSH access, NOT READY. No A GPU job; all controllers/monitors/export/download complete.

## Latest: repeated oldest recovery R02 positive exploratory block
All128perarm complete; all20frozencriteriaPASS. Native/queue/fund p95gap7.938750/9.581802/1.552190s, globalmax12.066355/12.883049/2.187850s; rate1507.069617/1507.384391/1497.005644; flow37.055118/37.590503/37.487052s. Fundvsnative gap~80.45%lower,rate-.668%,flow+1.166%. Fund58episodes/57actualevictions/1directresume/21sources/0cancel; fullnative/Q1/retire/rawjoinsallclose. Preempts46/48/103, so in this block morepreemptions accompaniedshorterstalls. Fullrequestmaxgap57better71worse,median.041560→.055986;goodput14higher6lower versusboth. Naturaloutputs127401/128154/127286,stops4/3/4;noequalworkspeedup.
Canonical A_NATIVE_OLDEST_REPEAT_TRIPLET_RESULT_R02_20261002.json SHAb1cc9b0487e3ed372f277606fd763c750d9c3aecc1948aa06dbd58da4482f5da. Candidatebfe6858307624b09dcac0d4e57a47ee965c9279e643b4d98c002ae3fde0a4946; analyzer7e954006f8a7028f8a92461d14bdb7360a4e9b04ffe0c77952015d4b799a2d33. Runtime436.408824s,allGPUEMPTY; raw130files/188411114B,81hashesverified; tar0cbe2ff369943c43bada213deece807006874ba9b289d4b3aabb3f38c230b47c. Remote/localfullraw andtar retained. R01zero-cell25.071s diskabort included. R02dedup1326immutablepayloadpaths across63completedownpackages,allpaths retained; root2145824768Bbeforecontroller. Neverrepeatpreps/controllers/exports. Warmseedimmutable.

## One next direct experiment: frozen, SSH unavailable
Strong ordinary / unchanged repeated fund / native is FROZEN and UNRUN. Candidate manifest7c5221fb80402843e10c459471045baf7c2ea50541d0eb4f461c26ce46046c01; analyzer90418d39bdb9843f86cb96ccef6d657ac5666a43f6942d10d1a4a1844b5c9d22; plan81840eb19c2378958b31f624c92e8c0ed09e7a50b5dcc63c796eb409c46077a4; staged tar39de3699d2ed17f8cd657a6ce20340361743bc636c52d2b4ad7b1d6afa49249c (5150387B). Primary22937 and backup45495 both return SSH connection refused. Upload failed before launch; prepare/controller NEVEREXECUTED. Current endpoint requested; existing broad resource authorization remains. Three consecutive goalturns confirm bothports refuse; blocked status recorded, no automaticretry pending. Resume frozen upload + prepared commonlock script, then all3cells/export/readback/frozenanalysis. No remote job currently submitted by A.
Primary criteria: fund p95gap below both,rate>=97%,meanflow<=105%; all128/all20frontiers; ordinary must actually admit freefit requests. CacheOPT/UniBoost overlappriority/funding/protection, so no novelty/freshconfirmation/READY claim.
Local residual analysis: all58 anchor→nextoutput63–79.5ms (median67.7ms); worst2.18785s gap includes2.11708s beforeanchor and70.77ms after. Aggregate pendingtransfergate546 rejects cannot attribute specific delay. Of71worsened requestmaxima, medianincrease4.35ms but7increase>1s. Full gap segments and pairs retained in A_NATIVE_OLDEST_REPEAT_GAP_SEGMENTS_R02_20261002.json.

## Latest direct experiment: actual oldest recovery funding R03
Three arms native / queue_only / queue_fund complete, each128 requests, no failure. One actual victim preemption, native target restoration and Q1 output protection executed; all frozen one-shot criteria pass. Target source0042224 trigger-to-output9.155/8.930/.0673s, preceding output gap10.508/10.256/1.713s. Different physical states (6/7/71 prior outputs), so not an identical-state causal effect. Displaced victim pause7.238s absolute; its native/queue counterpart pauses7.553/7.611s, not evidence of a7.24s incremental penalty. Target later full-trajectory maxgap4.480s remains in metrics.
Output rate1513.008/1517.654/1509.396, meanflow37.471/37.308/36.135s, globalmaxgap11.827/12.756/11.643s. Outputs128148/128154/126304, naturalstops3/3/5. Full128 effects/20goodputfrontier and all sequence differences retained; no equal-work acceleration claim.
Canonical A_NATIVE_OLDEST_ADMISSION_TRIPLET_RESULT_R03_20261002.json SHAf76147d85e97cd2885554bee50767987696d6cee6ba6711e9af48c8e35ad9940. CandidateR02 manifestf29d62e904355cc1c2261ffe7352a791ec69a2ec772fb37f8afdfe0de8348928. Corrected commit gate explicitly permits only private known disjoint running STORE jobs, preserving native victim flush. R01 cancelled with0actualforced; R02 abortedbeforeanycell on diskreserve. All retained.

## Completed failure boundaries
Size-only min/max victim, completion deferral, full-running BidKV, density continuation and both fresh current-guard blocks fail their service criteria; no unchanged expansions. Native shorter host-prefix restore does not reduce full-history GPU need for the first new output. Earlier artifacts and exact claims remain in RESULT_LEDGER.md.

## Resources / retention
All completed raw results remain verified locally; repeated R02 remote compressed and uncompressed raw last observed intact. Strong prepare has NOT compacted any repeatedR02raw or changed remote paths. One-shotR03 remote uncompressed raw was compacted earlier; verified remote tar and local full raw retained. Never rerun completed prep/controllers/exports; warmseed immutable. Primary commonlock2304:29005388732, GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36. No open A PTYs. Paper30pages includes repeated main result/action followup; full128 request-gap ECDF added, buildPASS; Figure10 atp25. Full followup scripts/results retained locally.
