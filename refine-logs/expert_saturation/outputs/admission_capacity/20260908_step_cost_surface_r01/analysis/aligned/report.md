# Capture-alignment prediction test (sealed data, no execution)

## Static caps under the main arrival load

| regime | cap | on capture point | bucket | median width | mean padding waste | median pure step ms | median TPOT ms | median TTFT ms | goodput per episode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bursty | 8 | True | 8 | 8 | 0.000 | 6.729 | 6.760 | 619.5 | [2.2543, 2.2441] |
| bursty | 16 | True | 16 | 16 | 0.001 | 8.157 | 8.201 | 40.6 | [12.1721, 12.2677] |
| bursty | 24 | True | 24 | 16 | 0.002 | 8.191 | 8.339 | 27.6 | [12.4473, 12.5988] |
| bursty | 32 | True | 32 | 16 | 0.002 | 8.187 | 8.373 | 27.7 | [12.4782, 12.5416] |
| steady | 8 | True | 8 | 8 | 0.023 | 6.690 | 7.198 | 804.8 | [2.03, 2.025] |
| steady | 16 | True | 16 | 16 | 0.093 | 8.235 | 8.886 | 187.7 | [3.8415, 2.4258] |
| steady | 24 | True | 24 | 14 | 0.139 | 8.678 | 9.714 | 19.1 | [1.8909, 2.2955] |
| steady | 32 | True | 32 | 14 | 0.166 | 8.715 | 9.923 | 18.1 | [1.5309, 1.5296] |

## P1: two caps inside the same bucket

## P2/P3: padding waste versus goodput ranking

### steady

- caps by ascending padding waste: [(8, 0.023), (16, 0.093), (24, 0.139), (32, 0.166)]
- caps by descending goodput: [(16, 3.134), (24, 2.093), (8, 2.028), (32, 1.53)]
- worst-waste cap 32 (on capture point: True); best-goodput cap 16

### bursty

- caps by ascending padding waste: [(8, 0.0), (16, 0.001), (24, 0.002), (32, 0.002)]
- caps by descending goodput: [(24, 12.523), (32, 12.51), (16, 12.22), (8, 2.249)]
- worst-waste cap 32 (on capture point: True); best-goodput cap 24

