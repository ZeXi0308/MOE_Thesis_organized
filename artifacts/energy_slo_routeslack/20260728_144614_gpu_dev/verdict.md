# RTX 5090 development probe verdict

`BATCH1_CACHED_ROUTE_CAPTURE_DEVELOPMENT_PROBE_PASS`

- LLM-jp: 512 contributions = 2 decode steps x 16 layers x top-16.
- OLMoE: 256 contributions = 2 decode steps x 16 layers x top-8.
- NVML total-energy counter readable; 5 ms request produced 12.024 ms maximum observed gap.
- Both capture metadata files are fail-closed (`formal_eligible=false`).
- Gate 0 remains FAIL: no natural continuous serving, EP, matched completion, same-window service-energy runner, thermal pairing, or native-vs-patched two-model parity in this artifact.
