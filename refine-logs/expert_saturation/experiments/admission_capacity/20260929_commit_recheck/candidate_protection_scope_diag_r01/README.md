# Unchanged-Q10 protection scope observation

One native H128 diagnostic follows the failed Q1/Q10 pilot. The sparse performance runner remains enabled so full scheduler/allocator/offload snapshots stay disabled. The new isolated adapter records only actual extended-protection waiting-loop breaks, peer holds and per-step scheduled counts. Policy decisions and Q10 remain unchanged.

This run locates reached exclusions and necessary numerical headroom. It is not a performance comparison or proof that a blocked request would actually be admitted. Missing transfer state remains unknown. Reuse the common nonblocking lock; preserve every result and do not restart an existing session.

Controller: `run_protection_scope_diag_r01.py`; plan: `PROTECTION_SCOPE_DIAG_PLAN_R01_20261001.json`. The exploratory question and positive/negative decision are frozen there before launch.
