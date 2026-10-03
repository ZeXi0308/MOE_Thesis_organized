# Corrected FJRC Level-1 Replay

- Model: `olmoe`
- Run class: `CPU_DRY_RUN`
- Status: `FAIL`
- Timing source: `DETERMINISTIC_SYNTHETIC_WORKLOAD_OVER_NATIVE_ROUTE_IDENTITIES`
- Evidence boundary: logical replay only; not a GPU, NCCL, RDMA, TPOT, or serving result.
- Requests: `32`
- Q miss rate: `0.51562500`
- R miss rate: `0.50000000`
- Q-R absolute miss reduction: `0.01562500`
- Q-R relative CVaR90 reduction: `0.24896081`
- Strict action flips: `16/16`
- Paired bootstrap 95% CI: `(0.0, 0.046875)`
