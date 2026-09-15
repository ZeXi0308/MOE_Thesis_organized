# RouteShield Gate-0 contracts

This package validates a preregistered RouteShield route census and recomputes
development diagnostics from hash-bound raw paired blocks. It does not
implement a production scheduler, validate a full request-DAG/Oracle
certificate, or convert single-GPU replay into EP or serving-P99 evidence.

## Commands

```bash
python3 -m unittest discover -v \
  -s docs/ideas/routeshield/experiments -p 'test_*.py'

python3 docs/ideas/routeshield/experiments/run_gate0.py \
  --config docs/ideas/routeshield/experiments/configs/gate0_v1.json \
  --output /tmp/routeshield-readiness.json

python3 docs/ideas/routeshield/experiments/census.py \
  --config docs/ideas/routeshield/experiments/configs/gate0_v1.json \
  --routes /path/to/tenant_routes.csv \
  --output /tmp/routeshield-census.json

python3 docs/ideas/routeshield/experiments/run_gate0.py \
  --config docs/ideas/routeshield/experiments/configs/gate0_v1.json \
  --raw-bundle /path/to/capsule/manifest.json \
  --output /tmp/routeshield-raw-diagnostic.json
```

The frozen config deliberately contains unresolved formal artifacts and
`formal_execution_authorized=false`; the readiness command must therefore fail
closed. Do not replace those fields with development hashes or old BCRD /
RouteSlack artifacts.

## Evidence boundary

- `schema.py` validates tenant-qualified contribution identities and route
  closure. Formal rank rows additionally require `EXECUTED_DISPATCH`, a unique
  dispatch ID, exact expected token/chunk/layer closure, a frozen tokenizer,
  and membership of `(rank, replica_instance, device_uuid)` in a hash-verified
  placement snapshot. Snapshot membership alone is not described as physical
  execution proof.
- `census.py` reports count-based route footprints in causal observation-time
  order. It deliberately leaves service-weighted remaining work unresolved. A
  formal rank footprint also requires the route ledger bytes to match the
  preregistered hash. Development mode suppresses every rank-derived field.
- `protocol.py` validates formulas and the shape of a future replay-result
  summary. Aggregate JSON is untrusted and cannot produce a canonical verdict.
- `raw_recompute.py` rejects malformed/duplicate/non-finite JSON, unsafe or
  symlinked paths, unlisted files, hash/size/row-count mismatches, missing paired
  arms, censored requests, request-world drift, queue growth, and non-exact
  Oracle statuses. It recomputes request TTFT, empirical P99, provenance hashes,
  wall-clock goodput, and paired-block intervals from rows.
- The full-DAG replay and exact legal Oracle remain mandatory blockers. A
  single-layer or summed-rank simulator must report `INVALID_REQUEST_DAG`.
- Census summaries report request count and unique document/prompt-cluster
  counts separately; repeated requests do not become independent samples.

## Output boundary

- `--smoke` is synthetic and always non-formal.
- `--metrics-json` is aggregate-shape diagnosis only.
- a `DEVELOPMENT` raw bundle can emit only a diagnostic threshold branch;
  `formal_result` remains `false`.
- a `FORMAL` bundle is integrity-checked and then stopped at
  `BLOCKED_FORMAL_RAW_EVALUATOR_NOT_APPROVED` while
  `FORMAL_RAW_EVALUATOR_IMPLEMENTED=false`.

The complete capsule contract is in
[`../Raw_Capsule_Contract_v1.md`](../Raw_Capsule_Contract_v1.md).
