# Repeated adapter implemented, execution unverified

staged_store_rotation.py adds repeated prepare/commit atop the pinned native schedule. It inherits current most-output selection, absence/residency/progress guards and cohort activation. Changed target/victim/funding cancels that staged action. Save only opens on its preparation step; native_store_delta checks new incremental jobs without a speculative lookup. Commit checks actual pending store IDs against native flush metadata. Target protection permits an asynchronous load to hold blocks and releases after first new output; native misses still recompute.

No complete-prefix residency is assumed from earlier jobs. Unsupported concurrent blocked requests during protection stop the diagnostic instead of silently weakening guarantees. No GPU run, complete scheduler fixture or repeated-policy correctness result yet. Only import, existing incremental-job fixtures and native waiting-prefix fixtures passed. Old single-event packaged code and raw outputs are unchanged.

Next: exercise actual repeated prepare/commit closures with current-state fixtures, then align any cancellation/concurrency differences with the CPU model before preparing the GPU pilot. B Qwen6954 owns the registered long group; no upload/driver or new window reserved here.
