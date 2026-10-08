# Transport amendment before any Instruct GPU generation

The frozen V1 unit acquired the shared lock at 2026-10-01 13:10:30 UTC. Its sequential download transferred 356,515,840 bytes in about 138 seconds, approximately 2.6 MB/s. At that rate the 13.84 GB checkpoint would take about 90 minutes, exceeding the frozen 1200-second download budget. C sent SIGTERM only to its identified launcher at 13:13:35 UTC, before CUDA/model initialization. The launcher reaped child29543, verified zero GPU memory and no compute processes, removed its owned RAM stage, and released the lock. All original logs and the cancellation decision remain retained. There are no measured request outputs or model-quality results from V1.

A bounded eight-way 8 MiB range probe then returned seven exact HTTP206 ranges and one read timeout: 58,720,256 completed bytes in 62.272 seconds including the failing tail. This establishes range support but not sustained full-checkpoint throughput. A subsequent direct-CDN probe was deferred because the shared lock was busy; it did not run.

V2 changes only transport and private versioned destinations:

- Resolve one HTTPS redirect per shard and reuse its HTTPS destination for range requests.
- At most eight workers, disjoint 32 MiB ranges, direct writes to one partial file. Require HTTP206, the exact Content-Range and exact byte count for each range; verify whole-file SHA-256 and size before loading.
- At most three attempts for a transiently failed range, still within the original 1200-second global download deadline. Reject wrong range/status semantics or hash disagreement. No unlimited retry or queued job.
- The same fixed16 prompts, model/revision, tokenizer, runtime, sampler, warmups, scoring and output contracts remain. The sole cell code change imports downloaderV2. The launcher changes downloader import and the versioned cell/freeze/output/stage names; its 1560-second limit and owned-child cleanup remain.
- New remote output `c-instruct-gsm8k-qualification-dev-v2`; new temporary stage `/dev/shm/c-olmoe-instruct-20261001-pilot-v2`. V1 is never overwritten or relaunched.

The transport change responds to measured network behavior. It is not an input, threshold, model or sampling adjustment following inference results. V2 remains unrun at the time this amendment is written; no result is assumed.
