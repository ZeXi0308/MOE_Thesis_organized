BidKV default-score adaptation; GPU UNRUN.

Official source commit5ee80256d263d58b1e512d9d436d47e9bac564ba, selector SHA3ad8af893f3693a5ffbe3b9367153cca14fed50fb1b195cdbab924b1e5849a99, source https://github.com/vLLM-HUST/vllm-ascend-hust-bidkv/blob/5ee80256d263d58b1e512d9d436d47e9bac564ba/vllm_ascend_bidkv/selector.py . Derived scoring helper retains Apache-2.0 attribution.

Same native-full ordinary-backfill platform and already-seen128 documents as the continuation probe. Adds A_NATIVE_VICTIM_RULE=bidkv_score with exact default utility and tie order. Every arm records the same score state. Restricts selection to this experiment's qualified unprocessed running suffix, retains conservative mixed/unknown fallback, and uses the experiment's existing continuation switch. This is NOT a full BidKV system reproduction.

241 scoring/tie comparisons against the five reviewed methods extracted from pinned official source pass. No outcome-selected parameter tuning. No controller or GPU plan selected yet; current live continuation package is unchanged.
