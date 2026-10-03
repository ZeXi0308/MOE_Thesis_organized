# Native residency victim pilot r01
Parent: candidate_native_backfill_lazy_perf_r01, unchanged workload/model/save/backend/backfill/Q1.
Three rules share instrumentation and native execution; A_NATIVE_VICTIM_RULE selects tail, arrival, service_density.
Only allocation failure changes victim in unprocessed running suffix. No forced swap or additional hold/waiting restriction.
A residence starts at native running admission (after async load if present), before first execution. Density is new outputs in current residence / currently held unshared physical KV blocks; it is an online resource-normalized service heuristic, not a measured transfer-time model.
Mixed/unknown suffix falls back to native tail. Ties choose later suffix index. Existing native release/load ownership logic is unchanged.
This exploratory comparison is not a novelty or untouched-confirmation claim.
