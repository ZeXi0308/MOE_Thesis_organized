# Capture-alignment prediction test (sealed data, no execution)

## Static caps under the main arrival load

| regime | cap | on capture point | bucket | median width | mean padding waste | median pure step ms | median TPOT ms | median TTFT ms | goodput per episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bursty | 4 | True | 4 | 4 | 0.000 | 5.200 | 5.246 | 1657.2 | [0.7321, 0.7297] |
| bursty | 8 | True | 8 | 8 | 0.000 | 6.746 | 6.765 | 625.7 | [2.2393, 2.2518, 2.2438, 2.2543] |
| bursty | 12 | False | 16 | 12 | 0.165 | 8.395 | 8.329 | 318.0 | [5.0408, 4.9982] |
| bursty | 16 | True | 16 | 16 | 0.001 | 8.196 | 8.282 | 52.7 | [12.1598, 12.2906, 12.1929, 12.1339] |
| bursty | 32 | True | 32 | 16 | 0.002 | 8.151 | 8.257 | 25.9 | [12.4789, 12.6564, 12.5518, 12.4565] |
| steady | 4 | True | 4 | 4 | 0.004 | 5.175 | 5.421 | 1801.8 | [0.6996, 0.6972] |
| steady | 8 | True | 8 | 8 | 0.021 | 6.726 | 7.171 | 820.7 | [2.0076, 2.0393, 2.032, 1.9882] |
| steady | 12 | False | 16 | 12 | 0.194 | 8.343 | 8.940 | 545.6 | [1.9922, 1.9903] |
| steady | 16 | True | 16 | 15 | 0.094 | 8.300 | 8.984 | 159.0 | [1.3759, 2.0873, 2.0626, 2.0569] |
| steady | 32 | True | 32 | 14 | 0.161 | 8.650 | 9.807 | 18.1 | [1.5295, 2.7024, 2.7003, 1.5386] |

## P1: two caps inside the same bucket

### steady

- same bucket: True (bucket 16)
- pure step cost: cap12 8.343 ms vs cap16 8.300 ms, ratio 0.9949
- request TPOT: cap12 8.940 ms vs cap16 8.984 ms, ratio 1.0049
- padding waste: cap12 0.194 vs cap16 0.094
- goodput: cap12 1.991 vs cap16 2.060 (+3.44%)
- **step cost equal within 5%: True**
- **cap16 goodput higher: True**

### bursty

- same bucket: True (bucket 16)
- pure step cost: cap12 8.395 ms vs cap16 8.196 ms, ratio 0.9764
- request TPOT: cap12 8.329 ms vs cap16 8.282 ms, ratio 0.9943
- padding waste: cap12 0.165 vs cap16 0.001
- goodput: cap12 5.019 vs cap16 12.176 (+142.58%)
- **step cost equal within 5%: True**
- **cap16 goodput higher: True**

## P2/P3: padding waste versus goodput ranking

### steady

- caps by ascending padding waste: [(4, 0.004), (8, 0.021), (16, 0.094), (32, 0.161), (12, 0.194)]
- caps by descending goodput: [(32, 2.119), (16, 2.06), (8, 2.02), (12, 1.991), (4, 0.698)]
- worst-waste cap 12 (on capture point: False); best-goodput cap 32

### bursty

- caps by ascending padding waste: [(4, 0.0), (8, 0.0), (16, 0.001), (32, 0.002), (12, 0.165)]
- caps by descending goodput: [(32, 12.515), (16, 12.176), (12, 5.019), (8, 2.248), (4, 0.731)]
- worst-waste cap 12 (on capture point: False); best-goodput cap 32

