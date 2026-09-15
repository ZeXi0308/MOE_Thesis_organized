# Corrected FJRC Pre-GPU Readiness（2026-07-23）

## 结论

`LOCAL PRE-GPU READY / REMOTE CALIBRATION CAPTURE CONDITIONALLY READY / FORMAL PHYSICAL RUN NOT APPROVED`

GPU 未开期间能够完成的准备已经闭合：代码、配置、路径、执行顺序、一次性状态检查、产物集合、
两模型 AND gate 和停止条件均已冻结。开机后不再设计实验，只执行 remote-only checks 和 runbook。

## 本轮新增资产

1. `configs/fjrc_corrected_gpu_readiness_v1.json`
   - 冻结远端路径、模型 tree hash、package version、artifact 路径与 action-aware disk gate；
   - 绑定24个 reviewed source/protocol/test 文件；
   - config self-hash：
     `178fbf5f1168755658dfb896cfc75a27189ce4de970c1ee5714d3e984d9a087b`。
2. `preflight_fjrc_corrected_gpu.py`
   - `static`：无 GPU 检查 config self-hash 和 reviewed source hash；
   - `gpu`：检查 Python/package/CUDA/5090、foreign process、磁盘、模型 tree、signoff、
     route/LUT validator、one-shot ledger 和 dry-run artifact state；
   - 只输出 blockers/actions，不执行 inference 或实验。
3. `decide_fjrc_corrected_two_model.py`
   - 验证两个7文件 bundle；
   - 验证 run class、32-request denominator、selection-only calibration、source set、共同 LUT；
   - 执行 OLMoE AND LLM-jp，不允许 pooling；
   - 原子、不可覆盖地输出 self-hashed decision。
4. `FJRC_Corrected_GPU_Runbook_2026-07-23.md`
   - reviewed bundle 精确同步；
   - remote tests；
   - pre/post capture preflight；
   - LUT/route capture；
   - 两模型 native CPU dry run；
   - formal CPU replay、two-model decision、证据拉回；
   - 全部停止条件。

## 最终验证

- 联合回归：70 tests passed；
- 3 tests skipped：仅因本机无 torch，均已列为远端硬门；
- static preflight：`READY`；
- config self-hash：PASS；
- 24-file reviewed source hash：PASS；
- one-shot empty-state planner：精确输出5个动作；
- orphan ledger without route output：硬阻塞；
- source drift：硬阻塞；
- gate surface drift：硬阻塞；
- one-model FAIL：two-model AND 返回 FAIL；
- overwrite protection：PASS；
- `py_compile`：PASS；
- `git diff --check`：PASS。

## 本轮审查中修复的问题

1. venv Python 采用 `resolve()` 比较会误判合法虚拟环境；改为比较实际 invocation path，
   resolved path 只记录不判错。
2. route/LUT/dry-run symlink 可能绕过真实路径语义；现全部 fail closed。
3. two-model decision 原先没有冻结 gate 字段全集；现字段缺失、增加或非 bool 都阻塞。
4. selection calibration 仅检查 `r_outcomes_read=false` 不足；现同时要求
   `FROZEN_FROM_SELECTION_Q_ONLY`。
5. one-shot reservation 异常状态缺少单测；现覆盖 ledger 存在但 output 缺失的不可恢复状态。

## 开机后的唯一剩余工作

1. 同步24个 reviewed files + readiness config；
2. 远端重跑70项测试，本地 skip 的3项必须实际 PASS；
3. 运行 `gpu --deep-model-hash` preflight；
4. 仅执行 preflight `planned_actions` 中缺失的 route/LUT capture；
5. post-capture preflight；
6. 两模型 native CPU dry run；
7. review 产物后再运行 formal CPU logical replay。

## Approval 分层

### 单卡 Calibration Capture

当前是 `CONDITIONALLY READY`。远端3个 torch tests 与 GPU preflight 全通过后，可给出：

`GPU CALIBRATION CAPTURE APPROVED`

这只批准 route/LUT calibration input capture，不批准科学性能实验。

### Level-1 logical replay

native dry run bundle 通过复核后，可运行 formal CPU replay。即使结果 PASS，也只证明 native route
identity + synthetic timing workload 下的条件性信息价值。

### Physical multi-rank experiment

当前 `NOT APPROVED`。至少需要4张独立 GPU 和新的 timed send/receive implementation；单张5090
只能做结果判定前的 calibration，不能产生 receiver incast / NCCL / RDMA / TPOT/P99 claim。

## Artifact SHA-256

- readiness config file：
  `23fc8575fb0f685cc190367c9b77e5c424967a769e0a7f6a7ccd8187ff6aa7b8`
- runbook：
  `0f5fd7deb9192e692702637c48b4da46c46bec06244e75cbfa256d40446077ca`
- preflight：
  `2af9af8f7b697042c033f09016b4146c99a40353b42e82ed067469859b446567`
- two-model decision：
  `eaf29fb23881528561b1a91e7720288647ee81dd8f5bd262e8a5bc2bb68e442d`
- preflight tests：
  `1cda94f1e7916d0236044e35905739e2081a28f1e3b527c8784ce895a5705cd6`
- two-model tests：
  `8fd325cff473a66bc3077f9c7b59cc772d6dcfa32aab1fc4e0185f822d538849`
