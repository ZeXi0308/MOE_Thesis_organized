# 一页事实快照：HORIZON 实现入口

| 复用项 | 已核对的入口与约束 |
|---|---|
| 容量账本 | B `recovery_start_gate/progress_capacity.py:estimate`：T/原生 reserved/G 去重；`observe_progress.py:_snapshot` 在同一次 waiting allocate 入口给 KNOWN/UNKNOWN。完整快照扫描物理块，适合作为算术对照，不宜每次候选决策全量扫描。 |
| 租约 | B `pkg/recovery_lease.py` 是 dataclass 纯决策模式，保护目标输出配额且计同行年龄；不能原样当作同行增长视野租约。 |
| 排序与 runner | B `recovery_service_age/service_age.py` 复用最后 client receipt，未知回退 arrival；旧代码有总预算8，HORIZON 必须去掉该限制并仅重排连续 PREEMPTED 前缀。旧 cell 固定输出，必须替换为 Instruct+自然EOS。 |
| cap 对照 | B `recovery_start_gate/simple_cap/run_group.py` 在共同预热后改运行 cap，保留编译 maxseq256；旧 2.5 GiB 门槛升级为用户要求 10 GiB。 |
| 观测 | B `pkg/native_offload_observer.py`、`request_measurement.py`、`metrics.py` 复用 host 时间与全人口；maxgap/joint 需沿 analyzer，不能把旧 metrics 的平均 TPOT 达标偷换成 maxgap 达标。 |
| MC | C `research/C_admission_20261004/async_mc.py` 复用异步 depth2/延迟释放校验与 `mc_budget.py:peak_envelope`；是已有 MC 原则适配，非新算法。full-budget 与 MC 分开。 |
| STORE | E `STORE_DIAGNOSTIC.md` 与 `native_store_replay.py` 的旧正例是指定事件/仅一次，不能直接作为通用修复；新公共修复只回退游标，原生 builder/prepare_store 负责过滤与传输。 |

实际 pinned `scheduler.py` SHA `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`：running 分配 564、尾部 victim 601、waiting 分配 935、成功 LOAD/恢复提交 979–1038、native remaining/reserved 2522–2540。waiting 门控须在原分配之前并以 break 保持 FCFS；租约只在实际成功后登记。非尾 victim 可能已在本轮 scheduled map 中，必须复用原 PRIORITY 分支的回滚语义并修正 req_index，不能只替换 pop。

`kv_cache_manager.py` SHA `3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf`。offload scheduler SHA `89ac26a80fbc29b9bcaa5a0daba88fb6309247f9bb053e48d2d61efa7d9d66f1`：`update_state_after_alloc` 777，external=0 直接返回，BLOCK_LEVEL cursor 849；原生 STORE builder 926。以上三项已在新远端重新 hash 一致。

历史数据保留于 B RESULTS 顶部与文末：62 个新增缺页同行不在初始 G12；stall8 开发对照 P99 改善而完整 flow/TPS 未确立；cap224 消除抢占但增加 TTFT；Q1 只达首输出不足以支持持续服务。A victim 表给出 max-release 改善 gap 却伤害 flow 的取舍。C v25 是 MC 适配收益，不能归到 HORIZON。新实验不复现 once/8 探针。

远端：`/root/autodl-tmp/moe-b-recovery-order-20261004`，Python `/root/miniconda3/bin/python`，torch overlay `runtime-overlay`；LD_LIBRARY_PATH 从 `plan-r03.json` 读取。GPU UUID `GPU-12433ef7-c83f-e383-fb23-85550b63b957` 与旧实验不同；缓存 Instruct revision `7f1c97f440f06ce36705e4f2b843edb5925f4498`。登录凭据不写入文件。
