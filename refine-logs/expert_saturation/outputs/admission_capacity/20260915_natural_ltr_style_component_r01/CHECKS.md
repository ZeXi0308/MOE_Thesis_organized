# Necessary CPU qualification

`cpu_checks.json`: **PASS_CPU_ONLY**. Execute from repository root:

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_ltr_style_component_r01/check_lifecycle.py
```

The private selector checks first-executable scanning/accept-only latch, pending-load zero quantum, one decrement per positive call rather than per token, quantum across first output, free KV+slot recovery without eviction, slot-full use of a legal victim, and counter retention after release. Actual adapter closure fixtures check registered selected incremental stores → next-call commit → pending load without Q spend; target change cancels, omitted native flush rejects. Free recovery with a foreign pending queue is censored without latching; an active foreign queue releases the gate and retains Q.

The one-block growth check reuses `experiments/admission_capacity/verify_native_recovery_execution.py` and the pinned `20260914_load_ready_contract_r01/native_source.json`: actual native `schedule`, preemption, cached payload and post-schedule methods run with the adapter's actual hooks. With target requiring one block and free=0, the native allocation loop preempts the ordinary peer and assigns target one token, Q10→9. If the CPU allocator refuses target even after release, native preempts the target, Q stays10 and latch clears. **Allocator/connector transfers/returned model outputs remain fixtures; this is not GPU allocation or a load-completion qualification.** The existing unrelated blocked-queue promotion qualification is reused, not rerun.

Changed selector/adapter/runner/controller compile, one-cell CLI help and `bash -n pkg/run.sh` pass. G input/measurement/safety/native store-contract/capture/observer source files are copied unchanged. Counters are copied unchanged. No old G/H diagnostic, EOS suite, resource sweep or GPU initialization ran. Counting insert/replacement lines against source copies: adapter120, selector25, runner19, run.sh1, controller1 and CPU check150 = **316 lines** before this documentation (small resource/queue fixes are included).

Remaining GPU gates: installed type/source/resource checks; actual asynchronous load ownership/completion/promotion, EOS lifecycle, positive-call Q evolution beyond first output and complete cost capture. No CPU result establishes policy performance or novelty. Final package manifest pins the delivered files; G/H remain read-only.
