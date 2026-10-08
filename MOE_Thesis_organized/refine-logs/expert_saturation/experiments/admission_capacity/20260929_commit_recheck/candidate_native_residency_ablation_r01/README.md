# Residence epoch / KV normalization ablation r01
Parent candidate_native_residency_victim_r01. Same native hooks, native full saving, ordinary backfill, inputs, actions and unknown fallback.
Three pressure-point victim scores (max over identical unprocessed qualified suffix):
- service_density: (output - output_at_current_native_running_admission) / held_physical_pages
- residence_outputs: output - output_at_current_native_running_admission (drop memory normalization)
- lifetime_density: output / held_physical_pages (drop epoch reset)
All scores are observed counters. No new threshold, forced preemption, waiting freeze, fixed future length or data selection.
Hypothesis: the observed lower worst gap needs epoch reset and memory normalization beyond these two simpler factors; the full all-request tradeoff is retained.
This is a seen-input mechanism ablation, not untouched confirmation.
