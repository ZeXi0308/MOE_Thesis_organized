# First-output obligation: delivery, cost, and transferred delays

MEASUREMENT_ONLY_DELIVERY_COST_TRANSFER

Real independent trajectories and actual ledger/receipt alignment only. No fixed-future counterfactual. Returned prefixes and visible prestate equality are not full-engine or KV-byte checkpoint equality. First-output to later preemption is bounded by the containing scheduler interval; nested costs are not additive. Later request states differ across arms; only the initial common obligation has a matched action-prefix comparison.

| pair | first action divergence | common outputs off/on | elapsed delta before divergence s | max ITL off→on s |
|---|---:|---|---:|---|
| block0-d6-restore-on | 406 | 11295/11295 | +0.414703 | 6.669039→4.387084 |

First shared restore memory-train-article-0003640 starts at step405; off releases as interrupted at 484, on releases as new_output at 409 and is re-preempted at 411 after 1 new output. On maximum gap belongs to memory-train-article-0003640 at steps829–1032.
| block1-d6-restore-on | 406 | 11295/11295 | -0.389886 | 6.886753→4.141311 |

First shared restore memory-train-article-0003640 starts at step405; off releases as interrupted at 484, on releases as new_output at 409 and is re-preempted at 411 after 1 new output. On maximum gap belongs to memory-train-article-0003475 at steps862–1065.

Delivery: both on cells fulfill27 first-output obligations, with81 actually selected -2 calls and zero interrupted obligations. Off has six interruptions. All six zero-output re-preemption residencies disappear; four one-output residencies remain on, no two-output residencies. This is successful first-output delivery, not continuing-service protection.
Cost: repeated executed positions increase74982→96955 (+21973), scheduling calls1872→1906, actual preemptions25→27, held-resident calls916→1663 in both repeats. Fresh prompt/decode position counts are unchanged. Full completion/wall benefit changes sign across repeats, and timing already differs before the first action divergence; do not assign all timing changes to the intervention.
Transfer: five requests improve maximum ITL and27 worsen in each pair. The largest stable ITL harm is request3259; its actual gap, the other top affected gaps, per-request recompute changes, and all output differences remain in analysis.json.
Verdict: the obligation delivers its promised first new output and reduces the global longest generation pause in these two repeats. The remaining failure is incomplete cost amortization and redistributed pauses after release; net efficiency is not established. Do not tune200/10 or blindly lengthen protection. The next bounded causal question is whether an already qualified competing recovery should start when its first-output work will soon be discarded; it must account for the victim and every held request, not assume a longer lease is beneficial.
