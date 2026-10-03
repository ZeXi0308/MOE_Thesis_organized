# Fixed-width native MoE companion probe, frozen before GPU execution

2026-09-06. The latest single-action admission campaign already completed24
episodes in the shared workspace and stopped its triggered-down formulation.
Do not duplicate it. Execute only the retained conformance conditional backup.

Question: when a target's real layer input, global width and row position are
fixed, can changing companions change its native fused-MoE output?
No native conformance method, performance gain or paper GO is presumed.

Use pinned OLMoE BF16, native vLLM0.26 on the authorized single RTX5090.
Capture32 real prompt-last-token inputs to zero-indexed layer3 mlp with a hook
registered after native engine initialization. Use32 frozen prompts128 from
native_knee prepared input. Native eager prefill, all requests submitted before
one4096-token-budget step, no prefix cache. Verify real input_batch request IDs
and query offsets; copy actual layer inputs, not router-logit reconstructions.
No historical N0d state or arrival-domain replication is claimed.

Targets are source indices0/16: gate0-060 / dataset row99 and gate0-088 / row142.
They were selected by source order before any probe output, not by route or gain.
Fixture collection runs once. Preserve its exact tensors and use the same bytes
in both fresh process repetitions. Each process loads the same model weights.

Probe width16, target fixed at row7 in every arm. Original companions are the
other15 members of the target's half of the32 sources, in source order.
Permutation reverses those15 companions while retaining target at7.
Replacement uses the first15 sources from the opposite half, retaining target7.
No adversarial/skew optimization or parameter search. Retain exact source indices.

Use the loaded native gate, native router.select_experts and native routed
expert backend. Do not replace routing with CPU topk or rebuild weights in HF.
Record actual GPU-selected IDs/weights, router scores, per-expert token counts,
target output, backend class and available actual execution metadata.
Compare native full mlp output with split native gate/router/experts as a
local extraction sanity check. Both calls use same fixed input; no timing claim.

All3 arms,2 targets,3 within-process repetitions,2 fresh processes:36 measured
arm calls. Orders A/B/C, C/B/A, B/A/C; warm all arms before measurement.
Inputs cloned when native operators may mutate storage. Retain raw tensors,
same-arm repeat checks, bitwise/numeric output deltas, and actual route changes.
Do not treat layers, elements or repeated calls as independent text samples.

If same-arm outputs vary, classify instability before explaining companions.
If target output is identical despite changed internal loads/positions, stop
this fixed-width transfer formulation in the tested operator/domain. If there
is stable target divergence, first locate gate/route versus expert output;
only then consider the existing native batch-invariance implementation as the
next minimal competing intervention. Do not start a verifier or full KV replay.

Zero-token U/C is undefined; here actual16-token calls are nonempty. U counts
activated experts/all64; C uses max count divided by mean over all64 experts.
These are route statistics, not measured HBM or congestion.

Evidence ceiling: native isolated layer-operator numerical measurement on real
prefill-end activations, one model/layer/width/GPU. Not full-request semantics,
generated-token quality, TPOT/SLO, native CUDA-graph serving or EP generality.
No new controller, broad audit, .aris or repeated source snapshot. Preserve
failed attempts separately and the single actually uploaded execution package.
