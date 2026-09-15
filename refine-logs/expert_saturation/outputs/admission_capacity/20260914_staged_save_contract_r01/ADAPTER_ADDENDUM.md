# Single-event native adapter implementation

`staged_store_once.py` is implemented but GPU_UNRUN. It reuses the frozen rotation schedule AST hooks in a separate adapter; `rotation_native.py` and its connector rejection remain unchanged. Installation requires the pinned scheduler SHA, explicit 16-token blocks, synchronous APC-off single-group full attention and native decode-enabled OffloadingConnector.

At step 329 it selects the current most-output request, saves only its already computed complete prefix, and requires an actual native store job whose source block list equals the plan and whose registration belongs to that request. At 330 it rechecks state and capacity, invokes native preemption dynamically, checks the store IDs appear in native flush metadata, and promotes the original absent request. Other running requests may grow only outside its reserved recovery space. A pending remote load is allowed to hold blocks without executing. Protection ends on its first new output.

This is a fail-fast qualification adapter: cancellation terminates the probe, not a deployed fallback policy. The save-off arm uses the same preparation and queue action. This one event is not the full repeated most-output method and cannot claim its performance. Native metadata is checked before returning schedule output; physical flush and kernel behavior still require actual worker execution. Source pinning is inherited for Scheduler; installed connector/worker hashes must also be checked when packaging the GPU run.

Validation performed: Python import and compilation; all four AST hook sites match the sealed native scheduler with SHA `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`. No mock-engine or GPU execution yet. Next: exercise the adapter against the extracted native schedule with CPU fixtures, then package one save-off/save-on qualification when the original GPU queue can be observed and released.

The authorized SSH connection again closed before authentication. Remote monitor77578 terminal state remains unknown; no experiment was started or restarted.
