# RouteContract Kill Pilot G0 Verdict

- 时间：2026-07-29
- 协议：`ROUTECONTRACT_KILL_PILOT_PROTOCOL_20260729.md`
- 决定：`G0 FAIL / ABANDON_PAPER_FRAMING`
- 执行状态：`PREFLIGHT_ONLY / NO_GPU_PILOT_RUN`

## 结论

冻结协议要求 OLMoE、Qwen-MoE、Mixtral 各一个 model-version-bound 真实 route capsule，并明确拒绝用 random tiny config 伪装真实 capsule。当前 RTX 5090 主机只有真实预训练 OLMoE。为预检准备的 Qwen-MoE 和 Mixtral 是 Hugging Face `hf-internal-testing/tiny-random-*`，只能验证架构/API，不是 natural/pretrained route evidence。

因此 G0 在任何 GPU 结果产生前失败。未运行 clean relation、mutant、baseline 或 capability-sealing 实验；未生成任何 detection-rate 数据。不得把本结果写成“RouteContract 方法已被实验证伪”；被否定的是当前 CCF-B framing 在冻结资产/资源约束下的可执行性。

## 可核对的 preflight 证据

| 对象 | 类型 | 关键配置 | 判定 |
|---|---|---|---|
| `/root/autodl-tmp/models/olmoe/config.json` | 真实预训练 OLMoE 本地资产 | hidden 2048, 64 experts, top-8; config SHA-256 `3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335` | 满足 OLMoE 资产前提 |
| `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` | 随机 tiny 测试 checkpoint | hidden 64, 4 experts, top-2; config SHA-256 `077d8624fd32610d94c91c97023f24989f0dcdbe711894e282a15096607a3ea9` | 不是真实 Qwen-MoE route capsule |
| `hf-internal-testing/tiny-random-MixtralForCausalLM` | 随机 tiny 测试 checkpoint | hidden 64, 4 experts, top-2; config SHA-256 `6d56663fd6d3af901e60cc56215e25114f25d821655ee4735cb8c8ada08146ff` | 不是真实 Mixtral route capsule |
| vLLM source | pinned upstream source | tag `v0.26.0`, commit `568afb3a13806beb53bb2e6bd518269357b237c0` | 仅 preflight，未运行 kernel |

## 远程环境处置

- vLLM 隔离安装在完成前主动停止，避免在 G0 已失败时继续消耗磁盘/网络。
- `screen` 无残留 session；`nvidia-smi` 无 compute process。
- 未删除任何论文资产、原始结果或封存证据。本轮下载的 tiny 预检文件和未完成隔离环境可重建，暂保留以便核对。

## 研究边界

RouteContract 对应的 novelty review 为 `6.0/10, CAUTION`；最强近邻是 vLLM 现有 MoE reference/permutation/weighted-unpermute 测试与 TENSURE 的稀疏计算 metamorphic fuzzing 组合。即使未来补齐真实 capsule，仍需证明至少两个既有强 baseline 独有漏掉的 bug class，并有跨实现的 maintainer-confirmed 新缺陷，才能重新评估。当前不进入该工程。

