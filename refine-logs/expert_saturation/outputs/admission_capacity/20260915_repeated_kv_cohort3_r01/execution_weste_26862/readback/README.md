# Repeated saving on preselected disjoint cohort3

PREPARED_LOCAL / GPU_UNRUN. One question: does the fixed staged-most save on/off comparison retain its complete-request tradeoff on disjoint documents?

Reuse the service-r01 runtime/policy/measurement unchanged. Only the already-selected cohort3 long input replaces the original long input; warmup uses the same runner and counts. Keep GPU0 UUID4015…d0e5, 6656 usable GPU blocks, actual configured 16GiB host KV, BF16 OLMoE, 3072/1024 forced lengths,32 requests,50ms arrivals. No threshold/selector changes. Same-host serial block0-off/on then block1-on/off, four lightweight performance cells; no additional detailed diagnostic cells. Reuse service-r01 analysis entry, only performance comparisons are claimed; missing diagnostic cells remain UNRUN, never filled from another workload.

Cohort3 was selected and run previously for a different scheduler comparison. It is disjoint from the current original cohort by document ID/content/prompt-token hashes, but is the same corpus and fixed-length arrival distribution, not an untouched population holdout. No selection on saved-KV outcomes. Retain every cell including failure, output differences and unfavorable requests. Compare mean completion, wall/throughput, max output gap and TTFT together, without a new SLO or post-hoc winning-metric switch.

Stop at the four predefined cells, even if effects flip or action counts change. A sign flip means no robust transfer claim; localize against fixed input/resource/phase costs before proposing any repeat. Positive results support this document transfer only; unknown EOS, length heterogeneity, quality and novelty remain open. Strong baseline is identical staged-most without saves at the same allocated GPU/host cap.

Run only after previously registered dual-instance and recovery-execution groups are terminal and explicitly released, with live process/lock checks. This package has not been uploaded or launched. The A model session owns this new four-cell package; no other driver should launch it. No background waiter.
