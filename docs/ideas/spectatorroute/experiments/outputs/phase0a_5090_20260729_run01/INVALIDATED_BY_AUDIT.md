# Phase-0A run01 audit invalidation

> Authority: `INVALIDATED_PRE_AUDIT_DIAGNOSTIC_ONLY`  
> Phase-0B authorization: `REVOKED`  
> Raw artifacts: preserved; do not delete, overwrite, or reinterpret as formal evidence.

This run reported `64/64` positive victims, `8192/8192` joint-positive cells,
and `0` unstable cells on the real RTX 5090. Those observations are retained only
as a diagnostic signal because the executed frozen runner predates the mandatory
integrity audit fixes:

- no GPU UUID / foreign compute-process contamination guard;
- BF16 changed-element counting used numeric equality instead of raw 16-bit equality;
- no parent-held watchdog covering trace parsing and aggregation;
- no mandatory pinned-model real-GPU acceptance artifact;
- no final `COMPLETE.json` sentinel written after every success artifact.

The treatment, denominator, M grid, repeats, and gate must remain unchanged in the
audited rerun. This file does not modify any raw run artifact or manufacture a new
result.
