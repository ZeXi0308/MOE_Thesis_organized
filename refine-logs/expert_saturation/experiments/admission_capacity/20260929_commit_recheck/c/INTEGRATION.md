# C 交付：commit 时重检后端接口

## 来源与适用基底

- 2026-09-15 共享台账第 608、618、673、683 行已经记录两次 CPU 容量事件、一个 default-off `recheck_commit_funding` adapter、CPU 分支检查和后续成本分析。本交付是**缺失工作树原件时的最小重建**，不是新发现或第一次实现。
- 旧原件在主 checkout 为 sparse `S` 项；随后 root 提供的只读隔离检出 `/private/tmp/moe-recovery-source-20260929-r01` 已物化旧 `commit_recheck/` 文件。已逐行读取旧 patch、候选、报告和测试，并复跑其 `test_commit_adapter.py`：PASS。旧补丁针对 E 版 adapter，`CPU_CHECK.json` 声明基底 SHA256 `9b98fab8…646197e0`；旧组件只检查 `free>=target.remaining_blocks`，没有 commit 时序列槽、target transfer/partial KV 或全体物理块唯一归属门禁。旧 CPU fixture 未覆盖满槽。旧组件不能直接用作 G/H 的安全 H1 接入。
- `commit_recheck.patch` 是对共同源码的独立重建，基底 SHA256 `d1002357cee992a54ed1762bb24967fdd21ec307cfad28d2389e471ecfe63697`；root 后续已把它集成到 shared 文件。该共同源码原为早期 `save: bool` 版，不是 G/H 的完整 `store_scope=selected|native_full` adapter。新 `accepted_gh_commit_recheck.patch` 针对精确 G/H/F 包源码 SHA256 `24629c0bbd8aa2c121310a053826fdb3bafd7159bf6f2c9444f514ec9a56e31c`；它只在独立候选 staging 中应用，已接受包仍只读。

## 动作与门禁

`install(..., commit_recheck=False)` 默认保留原 prepare、目标、victim 排序、selected store、保护和原生 flush/load。只在既有 `commit_reason == READY` 的同一步，`True` 才读取 native 当前状态，满足下列全部条件时跳过**本次强制 victim 抢占**，仍将原 target 提到 waiting 队首，由原生 scheduler 分配、加载或重算：

1. target 仍为 `PREEMPTED` 且在 waiting；无 foreign pending queue；原生槽位字段已知且 `len(running)+num_waiting_for_streaming_input < max_num_running_reqs`。**空闲序列槽是硬门禁。**
2. target 尚无部分 GPU KV；connector 的 target transfer 状态已知且无在途 job；实际 `pool.get_num_free_blocks()` 足以容纳 target 当前完整 KV 块数。没有用 output cap 预测剩余长度或未来 EOS。
3. 每个 running 请求的块列表与 `KVCacheManager.get_blocks` 一致；块为非 null、池中同一对象、`ref_cnt == 1`，且全体运行请求块 ID 不重复；非运行请求没有残留物理块。检查未知或失败则保留旧 swap。

结果事件区分 `commit_recheck` 的 `DIRECT_READY`/各 `KEEP_*`、真正 `direct_commit`、旧 `applied_rotations`。直接恢复不调用 `_preempt_request`，不伪造 store/flush，也不取消 prepare 已登记的 selected store；worker 的实际回收/覆盖屏障仍由 native connector 管理。发生资源不足、无槽或归属不明时保留原 swap。commit 身份/step 过期、目标终止、foreign pending queue 或 victim 在途 load 时取消本次自定义 intent，让原生 scheduler 自行继续。新旧臂都接收这些共同安全修正；性能比较不得把它们算成 H1 增量。

旧 D859/E1399 的 `selected.json/full.json` 确有 commit `running_count=26/22`，而该运行域配置 cap32；若同一资格配置成立，可静态推断有 6/10 个序列槽。旧报告没有在 adapter 的 direct 决策点实际调用槽位门禁或执行 direct，也没有证明 connector/load、目标首输出、peer 损益。因此两例仍不能判为**已执行** direct，更不能推断完整请求收益。

## 与三类强基线的关系

- selected/eager：H1 只改同一 selected 保存底座的 commit victim 动作；eager 的启动节奏不变。先把本补丁的共同安全修正带入 off/on 两臂，锁定同一输入和物理预算，才可比较。
- native full/no-extra-rotation：这是没有本次自定义 prepare/commit 的系统参照，不能把 `commit_recheck=True` 当其一个模式。`store_scope=native_full` 是**另一个维度**，指原生增量保存范围；勿与 no-extra-rotation 混淆。
- 已接受 LTR-style r02：隔离检出中的 `pkg/ltr_style_native.py` 已核读。其 `T30/Q10` 选择器在**最初** `PRIORITIZE_WAITING` 时已有 `len(running)<max_num_running_reqs` 的无 victim 路径；但选中 `PREPARE_SELECTED` 后，其下一次 commit 仍无条件 preempt 原 victim。它没有覆盖本次 commit 后状态变化动作。该包只有 CPU 接入，原生 G64/T30/Q10 生命周期诊断仍 UNRUN；不在本轮同时改它的量子、pending-load 或优先级规则。

## 应用与 CPU 验证

共同源码版的 CPU 复核在仓库根目录执行。测试从 Git `HEAD` 取清洁基底并在临时目录应用 patch，因此 shared 文件被 root 集成后仍可重复运行：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/c/test_commit_recheck.py
```

CPU 检查实际将 patch 应用到临时共同源码副本，编译并运行其 `install` 内层 `begin/schedule` 函数体；覆盖 direct/off、空槽/缺槽、streaming session 占满槽及字段未知、物理块不足/共享、partial target、target pending job、victim pending load、pending store 保留、foreign queue、stale step/对象、commit 前 EOS 和保护期终止。它使用模拟 allocator、connector job 和 schedule 结果；没有运行完整 pinned native `install`、真实 load/store 完成、GPU 或完整服务。两个基底的 patch 应用与各 4 项 unittest 均 PASS；GPU 为 `UNRUN`。

## 对已接受 F/G/H 的独立候选

Git 索引确认 F、G、H 三份 `pkg/staged_store_rotation.py` 为同一 blob `c9ca3896…`。只读获取到独立 staging 的字节 SHA256 为 `24629c0b…a56e31c`。E 原 pkg blob `f5572c71…`、旧 commit 候选 blob `696059e1…`、shared 原版 blob `6af71437…` 都不同。G/H 接受的 `open_population`、`diagnostic`、`store_scope`、`global_cooldown_steps` 和 EOS 路径必须保留；旧 E patch 与 shared patch 均不能直接充当 G/H 运行包补丁。

精确 F/G/H adapter 安装时对 open population 固定 cap32；commit 分支仍无条件 `running.remove(victim); _preempt_request(...)`。它只有 prepare 时 selector 的 `free>=need` no-op，**没有** prepare 后资源变化的 commit 直恢动作。因此可构造独立 H1 候选包，但尚未证明 G/H 自然事件会触发 direct。`accepted_gh_commit_recheck.patch` 对该精确源码增 default-off `commit_recheck`，保留 selected/eager 的原 selector、保存与 cooldown；on 仅在既有 READY 且实时 KV、槽位、target transfer 和物理归属均通过时省略这次 victim 抢占。off/on 均接受相同的请求身份和 victim pending-load 安全取消；这些共同修正不计作 H1 收益。native full/no-extra-rotation 仍独立系统参照。

验证命令（仅 staging/CPU，**不应用到 F/G/H 原目录**）：

```sh
# 需有可读的隔离检出；此命令只写 /private/tmp
git -C /private/tmp/moe-recovery-source-20260929-r01 show HEAD:refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r02/pkg/staged_store_rotation.py > /private/tmp/accepted_gh_staged_store_rotation_20260929.py
H1_TEST_MODE=gh H1_ACCEPTED_GH_BASE=/private/tmp/accepted_gh_staged_store_rotation_20260929.py H1_ACCEPTED_GH_PACKAGE_ROOT=/private/tmp/moe-recovery-source-20260929-r01/refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r02/pkg PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/c/test_commit_recheck.py
```

上述 GH 专用 4 项 CPU 测试通过；它在临时 `pkg/` 副本应用精确补丁，测试实际 `begin/schedule` 闭包，不启动完整 install、pinned native scheduler、真实 GPU/host 传输或模型。root 构造候选包时需复制一份已接受 G/H payload 到**新目录**，核 manifest/source hash、单独接 runner 默认关闭开关并把开关写入 config/raw、off/on 同 selected/eager 底座；不能复用旧 E 的 native-full diagnostic runner patch。先串行完成 LTR 原生生命周期诊断与公平校准，再做 H1 真实资格和完整服务比较。真实资格必须观察 direct 发生时的 `running/max_num_running_reqs`、target load/首新输出、victim 未被 forced preempt、原 prepare store 仍合法、native flush/ownership 与后续 peer 抢占。若无 direct 事件则记 `NO_ACTION` 并停，不按旧 D/E 次数预告效果。GPU 状态仍为 `UNRUN`。

## pinned connector 生命周期与原生准入复核（只读，2026-09-29）

**`_req_status` 不会因正常 preempt 消失。** G/H 包的 `runtime_source_hashes.json` 固定 `v1/core/sched/scheduler.py=2ed2a550…c1d3941`、`offloading_connector.py=59e544b8…667bd539`；从 `20260914_load_ready_contract_r01/native_source.json` 提取的这两份源码 SHA256 与固定值逐字节相符。原生 `Scheduler.add_request` 先登记请求再调用 `connector.on_new_request(request)`；固定的 `OffloadingConnector` 直接转发给 `connector_scheduler.on_new_request`。官方 [vLLM v0.26.0 OffloadingConnectorScheduler 源码](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) 中，`on_new_request` 建立 `_req_status[id]`（约 652–662 行），`_update_req_states` 对 preempt 只清 group block IDs（约 813–819 行）；`_maybe_cleanup_finished_req` 和 job 完成路径仅删除 **finished** 请求的状态（约 422–427、1129–1149 行）。因此已经进入本 adapter 的活跃 `PREEMPTED` target 正常应有 `_req_status`；`KEEP_UNKNOWN_TARGET_TRANSFER` 是异常状态的保守回退，不应是永久默认结果。pending store/load 仍会使 `transfer_jobs` 非空，此时 `KEEP_PENDING_TARGET_TRANSFER` 合理。该 offloading scheduler 的包内固定 SHA256 是 `89ac26a8…7d9d66f1`；隔离检出没有其完整源码，本地下载官方 tag 在网络超时，因此此处对该文件的行级判断依据是官方 **tag 源码**，尚未逐字节核对包内哈希。首个原生诊断应记录活跃 target 的 status 是否存在及 job ID 数，而非预设它为空或不存在。

**`free >= need` 是 commit 前容量筛选，不是同一步准入证明。** 对 G/H 已限定的单 full-attention group、16-token block、APC 关闭、零水位、无 lookahead、target 零 GPU KV，`ceil((prompt+已产生 output)/16)` 与其**当前完整历史**的冷启动块量一致；它没有使用未来 output cap。固定原生 scheduler 的 waiting 分支先要求未暂停、`preempted_reqs` 没有非自定义抢占、仍有 token budget，并用 `len(running)+num_waiting_for_streaming_input < max_num_running_reqs` 判断序列槽；C 的两个补丁现已改为同一槽位公式，任一字段未知即 KEEP。`num_waiting_for_streaming_input` 初始为零，但 resumable streaming session 可增加它；本轮单次文本请求预计为零，仍应在资格记录中确认。原生 waiting 路径随后调用 connector lookup；[官方 connector 源码](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) 明确允许 lookup 返回 `None` 延期，或命中后异步 load。固定原生 scheduler 再调用 `allocate_slots(..., full_sequence_must_fit=True, reserved_blocks=...)`；[vLLM v0.26.0 KVCacheManager 源码](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/kv_cache_manager.py) 在实际分配时重读 free blocks，并在 async load 时扣留 in-flight 预留块。固定 `schedule` 的 running 循环发生在 waiting 循环之前，而 H1 gate 位于两者之前；若 `free==need` 且任一 running 请求本步增长一个 KV 块，target 到原生分配时可能已经不满足 full-history check。上述固定 scheduler 源码可由 `20260914_load_ready_contract_r01/native_source.json` 重取并复核 SHA256；KV manager 的包内哈希为 `3f4af8d2…f77ccf`，本地未取得其完整固定文件，引用的行级行为来自官方 tag。

**交给 root 的集成边界。** C 已修正两个独立 patch 的槽位公式，并加入“`len(running)<cap` 但 streaming 占满槽”及字段未知的 CPU 反例；root 的候选副本尚待同步。对同一步 direct，应记录原生 `scheduled_resumed_reqs`、`WAITING_FOR_REMOTE_KVS`、lookup 延期和首新输出；旧候选的 `direct_commit` 未区分计算与异步 load，更不能直接记为目标已产生新输出。当前候选在 `DIRECT_READY` 后仍设置 `protected`，若原生因 running 增长、lookup 延期或预算不足未调度 target，其 `Ready target made no progress` 断言会使实验 **fail-fast/INCOMPLETE**。root 已决定本轮保持该严格停机语义，不将失败计作 direct 成功，也不扩控制逻辑；需要从原生结果判明原因后再决定是否追加容量余量。已接受 F/G/H 包和 root 候选均未在 C 审查中修改。

## H1 候选的异步 load 收据补丁（CPU，2026-09-29）

进一步只读核对 root 候选 `pkg/staged_store_rotation.py` SHA256 `8aa05c3ce0d5fa8c0971a43a24256409984248be6eb014c21624c79610da850d`：`phase=='direct_resume'` 时，其保护检查允许 `target.status==WAITING_FOR_REMOTE_KVS` 且 `num_scheduled_tokens[target]==0`，随后无条件增加 `direct_commits`。在 [vLLM v0.26.0 原生 scheduler](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/core/sched/scheduler.py) 的 waiting 路径，async hit 会先通过 `allocate_slots`、`connector.update_state_after_alloc`，再把目标置为 `WAITING_FOR_REMOTE_KVS`；[OffloadingConnectorScheduler](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) 的 `update_state_after_alloc` 为目标生成 load job，随后 `build_connector_meta` 发出该 job。**这代表原生已为恢复分配 KV 并启动加载，但尚无计算 token 或首新输出。** 因此 async load 可计作“已准入”的 direct 动作，前提是实际 metadata、connector job 和目标物理块一致；仅凭状态字符串不能满足收据。

`candidate_h1_async_receipt.patch` 只改候选的 direct 后验与事件，不改 controller 动作。若本步目标有正的 scheduled tokens，`direct_commit.native_admission=SCHEDULED_TOKENS`；若 tokens 为 0 且目标为 `WAITING_FOR_REMOTE_KVS`，须同时有 target `meta.load_jobs`、匹配的 `cs._jobs` 非 store 条目和已分配 target 物理块，才记 `ASYNC_LOAD_ADMITTED`。事件保留 `scheduled_tokens`、`load_job_ids`；`direct_commits` 在上述两种真实准入路径才递增，**不代表目标已经产生新输出**。缺任一 async 收据立即抛错，原 runner 将该资格运行记 `INCOMPLETE`，不计作成功 direct。补丁对上述候选 SHA 生成，应用后 adapter SHA256 为 `5c6825cdc55aae17c91c49ca2e3ae03792feaac95c907133bce8ad6c8d129241`；root 须在独立候选中顺序应用并更新 manifest。C 未修改候选或已接受包。

定向测试从 G/H 原字节应用 C 的 `accepted_gh_commit_recheck.patch`，验证所得 SHA 恰为上述候选基底，再应用本补丁并运行真实 `install` 内层闭包。四例覆盖同步 token 准入、异步 load 真实收据、状态有但 load metadata 缺失、状态与 job 有但目标物理块缺失；均 PASS。模拟器没有实际 vLLM allocator、worker transfer 或 GPU，真实首次诊断仍须核 load 完成与首新输出。复算命令：

```sh
H1_ACCEPTED_GH_BASE=/private/tmp/accepted_gh_staged_store_rotation_20260929.py H1_ACCEPTED_GH_PACKAGE_ROOT=/private/tmp/moe-recovery-source-20260929-r01/refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r02/pkg PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/c/test_candidate_async_receipt.py
```
