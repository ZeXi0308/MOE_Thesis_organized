# Repeated-save native interface boundary

Source: sealed native offloading/scheduler.py in 20260914_kv_roundtrip_feasibility_r01/native_offload_source.json.

_build_store_jobs starts from group.next_stored_chunk_idx and filters keys through manager.prepare_store. A repeated preparation may emit a suffix, a sparse subset of missing keys, or no new job. A complete-prefix job cannot be required on every save. Absence of a job does not prove old keys are resident: allocator failure, existing keys and no new full blocks have distinct meanings.

get_num_new_matched_tokens clears connector group.block_ids, updates keys, hit counters and request state. It is not a read-only cache query and must not be called speculatively on a running victim. Native recovery remains responsible for real lookup, transfer and recompute fallback.

Implemented native_store_delta.py validates actual new-job request registration, source block membership/order/uniqueness and corresponding offload keys without calling lookup or mutating connector state. It explicitly returns UNKNOWN for full-prefix residency and completion. Eight CPU fixture cases cover complete, suffix, sparse and empty payload cases plus four corruptions. These are synthetic interface fixtures, not native builder/GPU validation.

Repeated adapter not yet integrated. It must allow a correctly registered incremental store and let native flush handle all pending jobs; it must not fabricate a full-prefix StoreEvidence from a partial job. The current repeated CPU model's no-eviction/successful-allocation assumption remains unvalidated. No GPU window claimed; B Qwen retains its group.
