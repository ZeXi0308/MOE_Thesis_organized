# Residence-density fresh confirmation: both blocks failed

Same128 source-ordered new-at-freeze requests, opposite three-arm orders. No replacement or pooling. All128 requests complete in every arm. Actual natural output sequences/counts differ.

| Block / rule | actual output tokens/s | mean flow s | maximum request gap s | output tokens |
|---|---:|---:|---:|---:|
| first / service_density | 1483.151 | 37.933 | 10.071 | 128296 |
| first / arrival | 1509.758 | 37.640 | 11.573 | 128300 |
| first / tail | 1503.579 | 36.999 | 8.391 | 127291 |
| second / tail | 1503.279 | 37.744 | 8.357 | 128154 |
| second / arrival | 1496.275 | 36.521 | 13.492 | 126388 |
| second / service_density | 1485.910 | 38.049 | 9.034 | 128296 |

Both original12-part criteria fail only density_vs_tail_lower_max_gap. The first development win and factor ablation do not survive the native-tail comparison on these inputs.

Density-vs-tail goodput: block1 higher2/lower18; block2 higher9/lower11 of20 fixed points. Density-vs-arrival:8/12 and3/17. Full request/frontend/raw comparisons are retained in the JSON, rather than a best-block selection.

Next experiment changes a concrete scheduler action: legal successor continuation after current-request self-preemption. That does not restore the rejected density-only claim. These same128 inputs are now development data.
