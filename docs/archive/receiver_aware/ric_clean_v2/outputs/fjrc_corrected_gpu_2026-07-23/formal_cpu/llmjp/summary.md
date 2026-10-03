# Corrected FJRC Level-1 Replay

- Model: `llmjp`
- Run class: `FORMAL_CPU_TRACE_REPLAY`
- Status: `FAIL`
- Timing source: `DETERMINISTIC_SYNTHETIC_WORKLOAD_OVER_NATIVE_ROUTE_IDENTITIES`
- Evidence boundary: logical replay only; not a GPU, NCCL, RDMA, TPOT, or serving result.
- Requests: `32`
- Q miss rate: `0.34375000`
- R miss rate: `0.34375000`
- Q-R absolute miss reduction: `0.00000000`
- Q-R relative CVaR90 reduction: `0.04517255`
- Strict action flips: `16/16`
- Paired bootstrap 95% CI: `(0.0, 0.0)`
