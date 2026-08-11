# RouteShield-MoE

> 当前状态：`PLAUSIBLE_SECURITY_HYPOTHESIS / PROTOCOL_ONLY / NOT_CURRENT_MAINLINE`
> 证据截止：2026-07-29

RouteShield 只保留为一个可证伪的安全候选：在不改变 router、top-k、expert identity、权重和输出语义的前提下，验证按 tenant 产生的真实 expert/physical-rank footprint 做隔离，是否能比 token/request 级公平更好地保护共租 victim。

它目前不是已成立的方向，也不替代 [`docs/current/README.md`](../../current/README.md) 中的当前执行线。

## 当前边界

- RepetitionCurse 已证明“路由集中可被黑盒文本武器化”这个外部问题，但它不证明本仓库的两个模型、目标 placement 和 optimized EP backend 上仍有可隔离的 victim harm。
- RTX 5090 可以测 route concentration、时间持续性和因果前缀可观测性；它不能测 NCCL/RDMA/A2A、真实 rank queue、collective barrier 或 victim request P99。
- 单卡回放的 P99 只能标记为 `REPLAYED_TTFT_P99`；即使通过，最高也只能输出 `QUALIFIED_FOR_8XA100_EXISTENCE_GATE`。
- 当前已有开发态 raw paired-block 重算器，但没有 tenant-qualified native continuous-prefill ledger、已解决的物理 placement/executed-dispatch 证据、双租户 full-path denominator、full request-DAG 或可独立审核的 exact Oracle certificate。Gate-0 因此仍是 `BLOCKED_PROTOCOL_NOT_AUTHORIZED`；即使授权位打开，也仍会是 `BLOCKED_MISSING_FORMAL_EVIDENCE`。

## 文件

- [`RouteShield_Gate0_冻结协议_2026-07-29.md`](RouteShield_Gate0_冻结协议_2026-07-29.md)：主张边界、threat model、反事实、强 baseline、统计门和 kill rules。
- [`experiments/configs/gate0_v1.json`](experiments/configs/gate0_v1.json)：机器可读预注册；未解决的 formal 输入故意保持 `UNRESOLVED_*`。
- [`experiments/schema.py`](experiments/schema.py)：tenant-qualified route ledger 合同与身份守恒检查。
- [`experiments/census.py`](experiments/census.py)：expert/rank concentration 与早期前缀持续性 census。
- [`experiments/protocol.py`](experiments/protocol.py)：就绪性检查、统一指标公式和汇总合同诊断；当前实现结构性禁止从手填 aggregate JSON 产生科学 GO/NO-GO。
- [`experiments/raw_recompute.py`](experiments/raw_recompute.py)：开发态 hash-bound request/block loader、配对闭合、P99 与 joint paired-block bootstrap 重算；不具备 formal 授权。
- [`Raw_Capsule_Contract_v1.md`](Raw_Capsule_Contract_v1.md)：raw capsule 文件、统计和信任边界。

## 本地验证

```bash
python3 -m unittest discover -v \
  -s docs/ideas/routeshield/experiments -p 'test_*.py'

python3 docs/ideas/routeshield/experiments/run_gate0.py \
  --config docs/ideas/routeshield/experiments/configs/gate0_v1.json \
  --output /tmp/routeshield-gate0-readiness.json
```

第二个命令当前必须输出 blocked 状态；这是协议正常工作，不是 Gate 实验失败。

`--smoke` 只运行确定性合成夹具；`--metrics-json` 只做不可信 aggregate 合同诊断；`--raw-bundle` 才会从 request/block 行重算，但当前仍只能输出 `RAW_RECOMPUTE_DIAGNOSTIC_ONLY` 或 `RAW_RECOMPUTE_SMOKE_ONLY`。
