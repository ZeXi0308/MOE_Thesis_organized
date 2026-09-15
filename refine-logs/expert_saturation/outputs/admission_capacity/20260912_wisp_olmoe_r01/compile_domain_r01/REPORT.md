# Compile-domain r01: implementation failure before measurement

Verdict: `INVALID_EXPERIMENT` for compiler-domain qualification; F/X comparison remains `UNRUN`. Evidence type: real native engine initialization and retained failure trace.

First F process exited 1 after 42.951954032 s. The new snapshot used `vars(state)` on WiSP `WispMoEState`, whose pinned source declares `__slots__` without `__dict__`. Setup failed after 0.000753922 s, with 0/320 compile-domain invocations and no request warmup or measurement. The other three cells were never launched. This says nothing about F/X performance, capacity, or compiler-domain coverage.

The CPU helper check used a `SimpleNamespace` state and therefore missed this actual layout mismatch. Original runtime code and the declared workload were unchanged. The narrow fix is explicit reads of all six pinned stats fields; a new r02 check must use the real slots layout. r01 input and raw files remain immutable.

Measured: initialized engine and full failed process cost. Not measured: request latency, completion, full 320-call preparation, qualified F/X comparison. Strongest baseline remains ordinary full-stage F; no new Oracle/headroom was measured. Claim ceiling: implementation failure localization. Failure category: snapshot interface mismatch. Reopen condition: fix that interface in a new attempt and pass the same domain qualification. One next experiment: identical F/X/X/F in compile_domain_r02 after the queued LTR group, retaining all setup costs and r01 failure.

At 1789323736.5421655 UTC epoch, controller 28829 was absent, the GPU query was empty and the shared flock was available; the group was explicitly released to LTR. These are boundary checks, not continuous-isolation proof.

Readback: [attempt01](attempt01) contains 61 size/SHA-verified payload files plus the retrieval manifest. Archive 586750 bytes, SHA256 `b822c24a80f9f97b74a1bcc67f2b4221062d5c88781bc238f568bdeb6584b79b`. Failure and private compiler cache remain on the host. No favorable cell selection or runtime retry occurred.
