# Outputs

One diagnostic-only RTX 5090 development bundle is retained under:

```text
artifacts/route_capacity_envelope/dev/20260812T170512Z/
```

It contains exactly `config.json`, `windows.csv`, `metrics.json`, `report.md`,
`commands.sh`, and `run.log`. Its verdict is
`PIVOT_TO_EXECUTION_CONFORMANCE`; `capacity_claim_authorized=false` and
`action_oracle_authorized=false`. Generated bundles are ignored by Git and
remain local/external evidence.

The canonical report is immutable run output. Interpret its timing/M0--M4
table only together with `LIGHTWEIGHT_STATUS.md`, which discloses that execution
isolation was not established.

Remote raw traces were written to
`/tmp/bcrd-gate0-smoke-rce-{steady,bursty}-20260812T170512Z` and are not
duplicated into the repository artifact; their continued remote liveness must
be checked before reuse. A superseded same-configuration pilot
was not proven to be an isolated repeat and emitted a capacity verdict without
the required conformance veto. It was rejected from the canonical tree and
copied at audit time to ephemeral local quarantine at
`/private/tmp/rce-superseded-170328Z/`. It is an uncontrolled stability
warning, not a second canonical bundle. The retained bundle was selected for
its correct veto, not for its favorable diagnostic metric.
