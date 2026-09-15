# Logical-alignment qualification integrity audit

**Audit time (UTC):** 2026-09-13T20:26:50Z  
**Reviewer:** fresh Codex reviewer, same-family, read-only review route  
**Acceptance status:** provisional  
**Repository HEAD observed:** `de64dae5ea4fa3c92ec605e9847e817d8e68f3ac` with a shared dirty worktree  
**Audited scope:** only `logical_alignment_qualification_r01`, all 48 retained `attempt01` payloads, and the two hash-matched installed Python source references named by the protocol  

## Verdict

**PASS (provisional); P0: 0, P1: 0.**

The retained evidence supports this narrow answer: in this one OLMoE BF16, eager Triton, single-RTX-5090 run, every one of the 176 real measurement-layer calls produced the same recorded output for the existing `physical384 / ignore_invalid=True` assignment path and the tested `logical64 / ignore_invalid=False` then physical-map path. The model advanced with the separately computed logical output. This is a same-call numerical/path qualification only.

The result does not establish latency, throughput, memory improvement, task quality, native multi-GPU/EP behavior, wheel/CUDA binary provenance, or a general result beyond this frozen runtime and cohort.

## Integrity findings

### A. Differential-reference provenance: PASS

The reference is generated from the same call's `hidden_states`, `topk_weights`, `topk_ids`, physical weight tensors, activation, and quantization configuration. The second invocation changes only `global_num_experts` from 384 to 64 and slices the same map to the 64 logical IDs (`instrumentation/run_logical_alignment.py:97-103`). The protocol labels the physical result as a numerical differential reference rather than task ground truth (`protocol.json`, `positive.reference`). This is an explicitly scoped model-output proxy, not fake task GT.

Evaluation type: `synthetic_proxy`, subtype `same_input_kernel_differential`.

### B. Score normalization: PASS

Acceptance is `finite && torch.allclose(atol=0.01, rtol=0.01)` with bit equality and raw `maxabs` also recorded (`instrumentation/run_logical_alignment.py:48-56`). `relative_l2` uses the reference norm, but it is diagnostic and appears beside the raw values; it is not converted into a self-normalized score or used to move the frozen threshold.

### C. Result existence and immutable readback: PASS

- All 22 execution inputs rehash to `input_hashes.json`; its SHA-256 is `9e2107a1be6a12c0182c9caed23f331f1e937b3832dcea51013d983d7cd09ea9`, matching `execution_freeze.json:9` and `results/execution.json.checked.manifest_sha256`.
- All 48 entries in `attempt01/readback_manifest.json` match their current byte counts and SHA-256 values. There are no extra or missing files in `attempt01` apart from the manifest itself.
- The input archive seal is recorded as `daac675524e2bee19ea9f5db127823a886e2fdb47290d55267529eb7bd664edd` (`execution_freeze.json:8`) and the launch record carries the same value. The readback archive record is `2aaabe8803bf3bfd10b7fce7d7ce74e087c6c96c70573f425a8f2425b6570d07` (`readback_verification.json:4`). The archives themselves are not retained here, so this audit independently rehashed the extracted payloads rather than claiming to reopen either archive.
- Strict parsing found no duplicate mapping keys or non-finite JSON constants across 28 JSON files and 291 retained JSONL records (319 objects total).

### D. Executed helper and returned path: PASS

The wrapper intercepts the installed `_prepare_expert_assignment`, requires the original caller to pass `ignore_invalid_experts=True`, invokes the original helper exactly once, and restores it in `finally` (`instrumentation/run_logical_alignment.py:59-88`). The installed source confirms that `fused_experts_impl` calls this helper (`installed_source/fused_moe.py:1714-1725`), while `moe_align_block_size` implements the tested `False` branch by aligning logical IDs first and then applying `expert_map[expert_ids]` (`installed_source/moe_align_block_size.py:90-103`).

For all 176 actual calls, the retained invocation records contain exactly one physical helper call `(E=384, map=[384], True)` followed by exactly one logical helper call `(E=64, map=[64], False)`, with the same real kernel config inside each pair. The 288-call pager trace has consecutive call IDs; its 176 measurement records join one-for-one and in order with `actual_calls` (IDs 112 through 287), with no identity mismatch in call ID, layer, context, or row count.

The model cannot silently advance with the reference through this Python path: `compare` returns both independently computed tensors, and the outer hook returns `actual` after setting `returned_path="logical64_false"` (`instrumentation/run_logical_alignment.py:97-107,154-155`). The installed implementation allocates a separate output and intermediate buffers before dispatch and writes `moe_sum` into that output (`installed_source/fused_moe.py:1675-1704,1727-1795`). No retained Python operation writes into `hidden_states`, `topk_weights`, `topk_ids`, `w1`, or `w2`.

### E. Actual-call, prefix, and negative-control scope: PASS

- Actual calls: 176 = 16 layers x 11 scheduler steps. There are four 160-row prefill steps (64 calls) and seven 5-row decode steps (112 calls). Every call is marked measurement, non-validation, `shared_pool_oneshot`, complete, finite, allclose, and bit-equal; recorded `maxabs=0` and `relative_l2=0` for all 176.
- Prefix cases: 96 = 16 layers x widths `{1,16,32,64,128,160}`. Each layer uses its first real >=160-row call (call IDs 112-127). The 160-row case reuses that actual comparison exactly once; the other 80 cases slice real prefix rows and independently execute both helper paths. No duplicated or fabricated rows are introduced by the wrapper (`instrumentation/run_logical_alignment.py:140-153`). All 96 have recorded finite/allclose/bit-equal results and `maxabs=0`.
- Wrong-map control: the first measurement call has all 64 experts active. A cyclic permutation of the 64 mapped physical slots is passed through the same logical helper, is never returned to the model, remains finite, and fails allclose as frozen: `bit_equal=false`, `maxabs=0.23370361328125`, `relative_l2=1.0077784286170393` (`logical_alignment.json:171634-172627`; construction at `instrumentation/run_logical_alignment.py:129-139`). This rejects the obvious failure modes where the map is ignored or the negative arm returns the reference.

### F. Cohort, flags, and resources: PASS within the declared scope

Independent reconstruction from `prepared/workload.json`, runtime `workload.json`, and `raw.json` found five distinct retained requests, identical selected source rows and 128-token prompt IDs, eight cumulative one-token output events per request, and 40 generated tokens total. Every request completed by fixed length; no preemption, feedback action, or event injection occurred. `status.json:49-50` independently reports 5 completed requests and 40 tokens.

The executed configuration records BF16, eager mode, Triton MoE, synchronous scheduling, chunked prefill, prefix caching disabled, `max_num_batched_tokens=160`, `long_prefill_token_threshold=32`, 1 GiB KV, and no legacy `--verify-kernel` or `--pool-qualification` flag (`config.json:11-32`, `engine_args.json:4-20`, `shared_pool.json:2-18`). The actual model shape is 16 layers, 64 logical experts, top-k 8. The physical pool is 384 slots with shapes `[384,2048,2048]` and `[384,2048,1024]`; retained allocation is 336 private plus 48 shared slots, backed by two distinct CUDA storages totaling 4,831,838,208 bytes (`measurement_resources.json:2-61`).

The parent runner takes a nonblocking file lock before its boundary check and retains it across the subprocess (`run_qualification.py:69-111`). Parent before/after samples show the expected GPU UUID with no compute process. The subprocess samples before initialization and before model load show no compute process; its after-run sample contains only its own PID and no foreign PID. This establishes the declared cooperative lock and sampled boundary checks. It is not continuous proof against an uncooperative process appearing between samples.

## Numerical evidence boundary

The raw output tensors were not retained. Therefore this audit did **not** recompute `finite`, `allclose`, `bit_equal`, `maxabs`, or `relative_l2` on CPU. It verified their original GPU-side computation code, the complete scalar records, the helper call trace, the pager join, the negative control, and all retained hashes. Treat the exact numerical booleans as original GPU comparison evidence.

The installed Python references rehash to the values enforced at execution: `fused_moe.py = a8015d...04683` and `moe_align_block_size.py = c3f7fc...a31b`. This checks the retained installed Python semantics, not the provenance of the installed wheel, custom-op binary, or CUDA compilation.

## Claim impact

- **Supported provisionally:** for this frozen OLMoE/vLLM 0.26 BF16 eager Triton run, logical-64 alignment followed by the recorded physical-slot mapping was numerically identical to the physical-384 reference on every observed actual call and every frozen prefix case.
- **Unsupported:** performance or resource benefit; task-level quality; other models, seeds, GPUs, runtimes, quantization modes, EP/multi-GPU paths, unseen shapes, or general backend equivalence.
- **No method/Oracle headroom claim:** this experiment is a correctness/path gate and does not measure an optimization baseline or performance headroom.

## Failure category and next step

Failure category: none within the frozen qualification. No additional run is needed to answer its bounded question. If a separate performance experiment is pursued, it must be a new pre-frozen artifact set and must use this qualification only as a correctness gate; none of this 38.8-second qualification wall time is performance evidence.

Direct answer: **yes, the specified logical64/False then physical-map path is qualified for the 176 observed actual calls and the 96 frozen real-prefix cases, provisionally under this same-family audit.**
