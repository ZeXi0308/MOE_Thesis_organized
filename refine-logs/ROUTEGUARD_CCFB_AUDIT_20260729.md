# RouteGuard-KV CCF-B Audit

> Verdict：`KILL_CCF_B / CONTINUE_R0A_KILL_PROBE_ONLY`  
> Review：fresh same-family, provisional  
> Evidence截止：2026-07-29  
> Scientific result：none

## Bottom line

RouteGuard-KV 可继续作为冻结的低成本机制存在性/负结果 probe，但不再作为 CCF-B 主候选。RTX 5090 smoke 与 calibration 只证明 runner、三臂 state isolation、route lock、identity controls 和 artifact integrity；它们不进入 R0-A scientific decision。

## Collision matrix

| Proposed claim | Closest collision | Remaining scope | Verdict |
|---|---|---|---|
| quantization perturbation causes expert-set/gate shift and extra quality loss | [EAC-MoE](https://aclanthology.org/2025.acl-long.633/), [EAQuant](https://arxiv.org/abs/2506.13329), [VSRAQ](https://arxiv.org/abs/2606.05688), ExpertQuant | perturbation source narrowed to decode KV cache under BF16 weights | narrow characterization |
| `free / set_locked / fully_locked` router-mediated contrast | EAC-MoE already cross-injects original/quantized routing scores; routing-consistency works cover top-k boundary | separate route-set from gate-weight contribution | estimator refinement |
| use route-mediated signal for matched-byte per-layer K/V protection | [KVTuner](https://arxiv.org/abs/2502.04420), [MoE-nD](https://arxiv.org/abs/2604.17695), [TriRoute](https://arxiv.org/abs/2607.06601), [vLLM skip layers](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/) | possible residual signal after attention/output-aware baselines | unverified score replacement, not a new action space |
| sealed table yields serving benefit | KVTuner, vLLM and [LMDeploy](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html) already provide deployable KV quantization paths | only a new heterogeneous kernel/layout could add mechanism depth | execution validation, not current novelty |

## Fatal objection

Even all-positive experiments would not yet distinguish the contribution from applying EAC-MoE-style route-shift diagnosis to a KV-cache perturbation source, then using that score inside KVTuner/vLLM's existing layer-protection action space.

## Evidence ladder

- 5090 smoke v2：`50/50`, integrity `PASS`; engineering only.
- 5090 calibration 20260729：`200/200`, integrity `PASS`, no control failure; calibration only.
- R0-A formal：not run; sealed documents remain unopened by analysis.
- R0-B/R1/R2：not run.
- native INT4/HBM/throughput/serving/multi-GPU：not established by BF16 QDQ proxy.

Fresh 7D mean score：`4.0/10`; originality `3/10`, scientific evidence `1/10`, system mechanism depth `2/10`.

## Allowed continuation

- No allocator, heterogeneous kernel or R2 work for CCF-B rescue.
- The exact frozen R0-A formal may be run only as a low-opportunity-cost graduation-thesis characterization/negative-result probe after its separate approval workflow.
- A positive R0-A does not authorize R1.
- Reopening a paper route would require two-model residual value over attention-error, top-N, skip-layer and output-aware routing baselines under a joint non-additive configuration Oracle; this is a new review gate, not an automatic next phase.
