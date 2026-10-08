# CPU hard-cap packing challenge — frozen before execution

Comparison domain: synchronous single-token pure decode, FCFS, 16-token blocks, 4,096 usable KV blocks, 32 sequence slots, known hard output caps, no prefix cache/speculation/async/offload. The current source is the retirement adapter at worktree HEAD f5b27506fc6aa1612e81b20abca2bddc5497b47b. No original implementation or raw artifact is edited.

Independent challenger: enumerate every integer future round through the largest remaining hard cap; count each resident through its cap-reaching boundary with the same conservative extra-token convention; reserve each new request at its full hard-cap block bound; take the longest fitting FIFO prefix. Compare its peak and selected prefix against the actual imported M0 code. This is a hard-cap global packing component challenge, NOT a CacheOPT/WAIT reproduction or a full native runtime test.

Generated domain: exhaustive ordered states with 0–2 resident requests, P in {1,14,15,16,17,31}, O in {1,2,15}, R in {1,2,16}; 1,000 deterministic near-capacity states with 0–31 residents and heterogeneous caps. Waiting requests obey P+M<=4096. Also compare each recoverable state in the existing fresh two M0 cells. Same-domain failure means any peak or prefix mismatch. Instrumentation alignment failures are reported separately and never silently omitted.

Additional diagnostics: contrast conservative M0 with a source-derived exact last-computed-token occupancy model, retaining allocations until sampling/stop/free; this is not the same comparison convention. Count mixed-prefill and HoL observed-state opportunities only where logs recover resident states and queue alignment. These are correlated structural observations, not policy counterfactuals, natural performance, or future EOS inputs.

Allowed ceiling: STRUCTURAL / CPU action equality in the stated domain. No latency, throughput, GPU qualification, or service-safety claim follows. Stop after this one bounded challenge; no GPU run or controller implementation.
