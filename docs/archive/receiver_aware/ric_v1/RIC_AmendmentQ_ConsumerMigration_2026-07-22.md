# RIC Amendment Q — outcome-blind consumer migration

> **SUPERSEDED / ABANDONED 2026-07-23.** 本修正案及 v6 consumer-migration
> 路线不再用于生成正式科学结果。迁移已接触 sealed 资产所需的
> provenance 链持续膨胀，并暴露 root/census、direct-API 与 one-shot
> ledger 等 P0。远端从未生成 sealed scenario 或正式 decision；因此
> 改用全新 data namespace 与干净 bundle，从 manifest 生产前重新冻结。
> 本文只保留为审计负资产，禁止复活或引用其未来跑数。

状态：`FROZEN_CONSUMER_ONLY_AMENDMENT_NO_SCIENTIFIC_RESULT`

本修正案不改写 `RIC_Phase2_冻结实验协议_2026-07-22.md`，也不改变其
protocol SHA、模型、数据窗口、arm、指标、阈值或停止规则。它只修复正式
scenario consumer 的一个 P0：sealed replay 必须复用 calibration service LUT，
旧实现却把 sealed data manifest 误用为 LUT 的 producer namespace。

## Q1. 发现时点与证据边界

P0 在 formal sealed data manifest 和两模型 native route 已生成之后、任何
sealed scenario、policy replay、MILP outcome、gate 或 decision 生成之前发现。
这些既有 artifact 只能作为 `INPUT_ONLY` / `CAPTURE_ONLY` 不可变输入；既有
v5 scenario-consumer signoff 对 sealed 路径记为 `SUPERSEDED / NOT_TESTED`，
不得被描述为结果。

## Q2. 唯一允许的迁移

新的、重新 Phase-4 review 的 consumer 可以复用既有不可变 data、route 和
calibration LUT，但必须同时：

1. 以历史 reviewed-source snapshot 重放并验证每个 upstream producer signoff
   的完整 source manifest、reviewed scope、review report、test report 与 git-head
   evidence；不得把历史 producer 冒充为 current producer；
2. snapshot 文件 SHA-256 固定为
   `15db8b79ea590fa4c4354835c8ba472928433a685c4df82f8ff7c9d2e155a9b8`；
3. 对 data manifest 重新执行 current strict field validation；
4. 显式提供 calibration manifest，并逐项等于 sealed manifest 冻结的
   `calibration_manifest_self_hash`、`calibration_manifest_file_sha256` 和
   `calibration_selected_list_sha256`；
5. sealed route 仍绑定 sealed manifest；service LUT 仍绑定 calibration
   manifest。两个 namespace 必须分别进入 scenario tree 与 current consumer
   signoff；
6. 在唯一、显式传入的 authoritative formal bundle root 上生成一次性
   pre-outcome registry；registry 必须拒绝 symlink 与坏 JSON，以 `O_EXCL`
   写入，并登记完整 path census 的 path/size/SHA 及所有实际
   data/route/LUT/capability 输入；
7. historical verifier 必须要求 producer signoff 文件 SHA 与该 registry
   完全相等。仅重算 self-hash、沿用旧 review 引用的 post-review signoff 不得
   被接受；
8. current consumer signoff 绑定本修正案 SHA、历史 snapshot SHA、权威 root、
   registry SHA、所有 input manifest/trace/LUT/capability/signoff SHA；本修正案
   本身必须进入 current consumer 的 exact reviewed scope；
9. 修复后从 calibration scenario、calibration lock 与 oracle 重新运行，再进行
   sealed one-shot replay。旧 calibration/oracle 只能作为已完成 upper-bound 线索，
   不能授权修复后的 sealed replay。

## Q3. 禁止事项

禁止测量 sealed LUT；禁止重选 sealed data；禁止删除或重置 one-shot ledger；
禁止修改 route/LUT/data metadata 后重签；禁止把 LUT 的 expected data binding
设为 `None`；禁止修改阈值或 arm；禁止在生成/查看 sealed scenario 或 outcome
之后补写迁移证明；禁止用 decoy directory 代替 authoritative bundle root。

## Q4. Claim 边界

本修正案只授权 outcome-blind 的 consumer 修复与重新验证。它本身不是实验
证据，也不支持 receiver-awareness、tail latency 或 RDMA 的任何科学结论。
