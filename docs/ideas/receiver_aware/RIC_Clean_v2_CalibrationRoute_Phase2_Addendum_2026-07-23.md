# RIC-Clean-v2 Calibration Route/LUT Phase 2 冻结补充协议

状态：**FROZEN FOR IMPLEMENTATION / NO SCIENTIFIC RESULT**  
冻结日期：2026-07-23

本补充协议只定义 N1 前置的 calibration route 与 service LUT。它不修改已冻结的
calibration data manifest，且不开放 sealed。

## Route capture

- 输入固定为 clean-v2 calibration manifest
  `e07d5b4d42b1f9e59d21ccab1b40fa31fb9052e19d4e8f83ec31a19b3b21d545` 所在路径；每模型输出固定到
  clean bundle 的 `clean_v2/routes/calibration/<model>`，禁止替代输入/输出 CLI。
- 模型目录固定为 `/root/autodl-tmp/models/{olmoe,llmjp}`；校验 config 中完整 model-tree
  SHA、Python/datasets/transformers/tokenizers identity 与 CUDA device。
- 正式设备为单张 RTX 5090；开始前有其他 GPU compute process 则 BLOCKED。
- 原生模型不 monkey-patch。独立 hook census 必须覆盖全部 MoE layers；捕获真实
  `aten.topk` ordered tuple，并与 raw gate logits 独立重算 tuple、native MoE output 独立
  reconstruction 同时一致。
- 全部 calibration requests、128 tokens、全部 MoE layers 都写 route rows；后续 replay
  layer 仅按冻结 outcome-blind layer assignment 标记，不删除背景 rows。
- route row identity、placement、route/parity/metadata SHA、data manifest self/file SHA、
  data Phase-4 signoff、base protocol/config、route addendum 与 reviewed source closure 全绑定。
- 任一 tokenizer length、layer census、top-k ordered tuple、weight dtype/value、output parity、
  model tree、环境、identity 或会计不符即 BLOCKED；不得补 request 或改 layer。

## Service LUT

- 只消费同一 clean calibration manifest 与已签字的 clean route artifact；不得从 logits
  另算 route 代替 native tuple。
- 对每个冻结 replay layer 的每个 expert，必须由其自己实际 routed activation 提供 row-1
  输入；缺 `(layer,expert)` 直接 `BLOCKED_ROUTE_SPECIFIC_SERVICE_COVERAGE`。
- 主 expert service surface 覆盖所有 `(layer,expert)`，保存 raw trials 与 warmups；4-expert
  conservative max 只能作 sensitivity，不能替代主面。
- sender pack / receiver unpack 若使用 proxy，必须在 LUT 实现协议逐字定义并单列 source
  tag；H2D 仅可标 `H2D_NOT_RDMA`，analytic network 仅为 derived L2 proxy。
- LUT metadata 绑定 data、route trace/parity/placement、模型树/runtime、raw/summary、
  producer signoff 与完整 reviewed source closure。

Route 与 LUT 均为 calibration input artifact，`scientific_result=false`；只有 N1 runner 经
独立 Phase-4 review 后才可给 necessity/headroom verdict。
