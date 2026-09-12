# Profiler diagnostic status

**GPU UNRUN — upload blocked by automatic approval review.**

Local implementation and interval-accounting checks are complete. Eight
execution files are frozen in `execution.tar.gz` (28,078 bytes), SHA256
`275e2fb7b9fd12b5abd5e5532871f74bbe91c306c08ca6e531f4606e84ca6eaf`.
The local-only `analyze_profile.py` is not part of this upload.

Execution files: `DECISIONS.md`, `run_native_capacity.py`, `native_capture.py`,
`metrics.py`, `profile_probe.py`, `run_campaign.py`, `original/config.json`,
`original/workload.json`. Configuration, workload, capture and metrics are
byte-identical to the files already on the same remote host; this was verified
using remote SHA256 values. New files/edits add diagnostic profiling and run
order, not new input documents, credentials or model weights.

Automatic approval rejected the upload twice, including after this inventory
and identity evidence were provided. Its reason was that the prior explicit
file-transfer authorization did not cover this particular new payload;
continued research authorization was considered insufficient for that transfer.
The user was asked to approve these eight files and four episodes specifically.
No alternate transfer path or indirect execution was attempted.

No profiler result, GPU attribution, action headroom or method claim follows
from preparation. The planned off/on/on/off episodes remain unrun until the
transfer is authorized. Earlier arrival-swap evidence remains unchanged.

Read-only backup check: single-layer replay through vLLM's loaded
`RoutedExperts.forward_modular(x, topk_weights, topk_ids, ...)` is feasible,
but this is a runtime transfer of the SpectatorRoute question, not a new idea.
The executed stack uses Triton unquantized modular MoE with batch invariance
disabled. The supported batch-invariant implementation must be a control in
any future conformance probe. Python hooks do not automatically execute during
an already-captured CUDA graph replay. No such probe was implemented or run.

Historical distinction: the prior Native C8 replay used natural grouping in
Transformers 4.57.6 / Torch 2.8, not this vLLM backend. Its same-rank policy net
reward changed from proxy +25 to native 0, with 0/33 positive native cells;
this should not be paraphrased as 25 positive actions transferring at 0/25.
It provides no reason to repeat the old repair policy. The only next GPU
experiment remains the frozen four-episode profiling diagnostic.
