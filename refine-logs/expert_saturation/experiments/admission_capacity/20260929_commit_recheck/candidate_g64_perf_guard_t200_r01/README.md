# G64 corrected guard T200 paired performance candidate

Status: **CONDITIONAL_UNRUN**. This is a new one-shot package identity with no GPU execution or performance result. Do not upload or execute it until the complete earlier T30 three-cell group has been archived and audited.

The 28 manifest-listed payload files, including `pkg/run.sh`, inputs, and `verify_package.py`, are byte-identical to `candidate_g64_perf_inflight_guard_r02`. The manifest SHA-256 remains `597bedb4573531d49c4e173fc8dc3b8c6c937a05fdc9e8272c0562b5fbc43c78`. Only this identity's README and provenance differ. No earlier launch-once marker is copied or removed.

**Launch gate:** the T30 group (`ltr_t30_q1`, `eager`, `ltr_t30_q10`) must have complete readback, accepted package/model/lock identity, full-service completion, and valid paired performance measurements in all three cells. A partial or failed group cannot authorize this T200 block. The qualification of the underlying inflight guard remains a prerequisite as recorded for the source package.

After that gate, the separate SHA-pinned plan `G64_PERF_GUARD_T200_PLAN_R01_20260930.json` may run `ltr_t200_q10`, `eager`, `ltr_t200_q1` serially under the same nonblocking GPU lock, with 900 seconds per cell and 4200 seconds total. Each output path is unique. The eager measurement in this group is the reference for these T200 arms; do not substitute the earlier T30 eager result. There is no automatic retry.
