# Separate bounded model preparation from the unchanged GPU qualification

V2 parallel transfer measured roughly 6.6 MB/s after its first 245 seconds. The whole 13.84 GB checkpoint would still exceed the original 1200-second download limit. Repeated whole-model attempts would discard useful verified bytes. At 13:35:32 UTC, after shard1 had passed its exact byte count and official full-file SHA-256, C retained it by hardlink and interrupted only its own V2 launcher before CUDA initialization. The original stage was removed, child reaped and GPU released. V2 contains no generated request outputs.

## Persistent preparation cache and actual resource cost

Cache: `/dev/shm/c-instruct-verified-cache-20261001`, mode0700, owned by C. `cache-identity.json` binds the official model ID, revision, original manifest SHA, ownerC and persistent-cache purpose. It initially contains the verified 4,997,744,872-byte first shard. A hardlink added no physical weight pages. After all three shards, the cache will occupy **13,838,721,960 bytes** of actual tmpfs/container memory. This remains allocated between preparation units and is explicitly included in cgroup headroom; it is not reported as released host memory when a temporary stage is removed.

This deliberately supersedes the initial plan to discard every weight at the end of a whole-model attempt. The measured transfer rate and lack of persistent disk capacity justify retaining completed preparation. The newly authorized spare GPU is a distinct original RTX5090 with an existing independent lock and compatible environment, but its data disk also has only about0.81GB free. It has not run a C model job in this turn.

## Bounded units

Prepare only one missing shard per invocation, under the existing primary shared nonblocking lock. Reuse V2's eight32MiB range workers, exact transport checks, at most three transient attempts per range and1200-second deadline. Preserve already verified shards. Each successful unit returns the GPU/resource lock before the next natural checkpoint; there is no queued retry or multi-shard background job.

Once all weights are verified, the original fixed16-request GSM8K qualification runs as one short GPU unit using a new V3 destination. Its model, revision, inputs, token IDs, seed, sampling, warmups, native APC/runtime and scoring remain unchanged. The only cell change imports a cached-model preparation function. That function rehashes each official shard and hardlinks it into the fresh launcher-owned stage, copying the small frozen metadata. It performs no network access.

The V3 headroom gate reserves26GiB plus the small metadata allocation, because the full weight cache is already charged to cgroup memory and hardlinks do not duplicate it. Temporary stage removal leaves the explicitly recorded source cache. These are model preparation/storage changes, not scheduling methods or model-quality results. No new prompt, seed, cap, cohort or scoring adjustment is made.
