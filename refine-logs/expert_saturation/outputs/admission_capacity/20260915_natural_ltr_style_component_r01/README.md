# Recovery-only LTR-style selected-offload component

**CPU_READY / GPU_UNRUN.** One diagnostic cell, `diagnostic-ltr-t30-q10`; 64 measured requests, 180 s episode limit. This package implements a compatible nearest-policy component. It does not reproduce full LTR, its predictor, CPU-SWAP, or a performance result. Root accepts the exact package before any GPU launch. No CURRENT, G/H evidence, or shared historical source is changed.

The question is whether LTR waiting counters and positive-compute-call service quantum can execute through the existing native selected-save recovery backend, including pending loads and growth after first output. `threshold=30`, `quantum=10` are fixed for this diagnostic. No action is a valid domain boundary; retain the original result and stop, without lowering the threshold or adding cells.

| Fixed input / resource | Contract |
|---|---|
| Workload | G's unchanged 64 full articles and token IDs; arrivals `i × 0.2 s`, span 12.6 s; no EOS-based selection |
| Generation | Same OLMoE model/revision/tokenizer and seed; natural EOS allowed, `ignore_eos=False`, `min_tokens=0`, cap 1024; cap is not a natural-EOS model |
| GPU / host | GPU0 `GPU-4015b79d-bed6-3a4b-9d2b-0c17be96d0e5`; 4096 usable + one null block, 8,592,031,744 physical KV bytes; 16 GiB / 8192 native host blocks |
| Runtime | Pinned vLLM 0.26 source; synchronous, APC off, native offload, `VLLM_USE_SIMPLE_KV_OFFLOAD=0`, full-history reservation; cap32 / 1024 scheduled tokens |
| Initialization | G's engine initialization, three application warmups, native drain/cache reset; retain all intervals, host allocations/peaks and final drain |

`LTRCounters` is copied unchanged from `20260914_ltr_component_probe_r01/preparation/pkg/recovery_service_components.py`, whose source attribution is hao-ai-lab/vllm-ltr commit `13bbf6ff3dab661791d41362551b089e5f77c91c`. The private selector derives from that group's `cpu_ltr_style_selected_r01/ltr_style_selected.py`. Counters count all live requests and retain the verified threshold/rearm rule. Quantum spends one for each actual positive `num_scheduled_tokens` call, regardless of token count; pending load and backend-censored calls do not spend it. First output does not end priority.

New candidate selection scans boosted preempted requests by arrival and original queue tie order; it latches only an accepted executable intent. With both KV and a sequence slot available it restores without eviction; otherwise one legal lower-priority running victim can fund KV and/or free a slot. Victim ranking remains most returned outputs / original running tie order. No G cooldown, absence, residency, progress-fraction, max-absence or first-output-release gate remains. PREEMPTED restricts forced/free recovery targets; all-live counters also reorder RUNNING priorities and can affect requests already in prefill, so this does not claim initial TTFT is unaffected. This is a whole recovery-policy component, not a trigger-only ablation. All output loss and unfinished requests remain counted.

The reused physical contract requires unshared owned blocks, pure decode and nonterminal-by-declared-cap preparation, current full-history funding, next-call commit validation, no victim pending load, registered selected store sources and native flush. It never uses future EOS. Prepare events retain victim current outputs/cap and target remaining quantum. Foreign pending queues censor new intents before latching; if encountered during an active epoch the queue gate/latch is released without resetting counters or changing ownership. Boosted running requests keep priority ordering. If a next-block reservation cannot be honored, only custom peer holds/reservation are withdrawn: native allocation may preempt ordinary peers. If native preempts the target, it releases the latch while retaining counters/remaining quantum. Custom victim and natural preemption events are separate. No unpreemptible-quantum or finite wall-clock guarantee is claimed.

Run entry after root acceptance, fresh staging, actual two-GPU idle check, and installed-source/package checks:

```sh
/root/miniconda3/bin/python /root/autodl-tmp/natural-ltr-style-component-20260915-r01/controller.py
```

`controller.py` verifies payload hashes, creates a one-launch marker and runs `pkg/run.sh` in the foreground. The shell holds `/root/autodl-tmp/moe-research-gpu.lock` for the whole cell; busy/unknown is ABORT, with no interference or background waiting. Do not run this local-preparation package before acceptance. Result entry: `results/diagnostic-ltr-t30-q10/{raw.json,selective-store.json,offload-events.json,status.json}` plus timing/memory/host/drain files. Failure retains original evidence and leaves the scientific result INCOMPLETE; there is no automatic retry.

Diagnostic qualification must distinguish actual selected store/load completion, ready scheduling, first new output, and quantum expiry after output. Existing native capture/worker observer and new intent/allocation events provide those observations; CPU fixtures cannot substitute for them. Complete-request gap distribution/max, TTFT, throughput, mean completion, outputs/EOS, host and drain are descriptive costs here, not a timing contrast or SLO verdict. The only next step after CPU acceptance is this single native diagnostic; future calibration is outside this package.
