# Frozen predictions F1-F4, adjudicated

aligned cells 24; sealed comparison cells 24

## F1 as stated: static24 pure-step median on the 24-bucket plateau

| regime | episodes | static24 median pure step ms | realised median width | frozen band ms | F1-as-stated |
|---|---:|---:|---:|---|---:|
| steady | 2 | 8.678 | 14 | [9.3, 9.51] | **False** |
| bursty | 2 | 8.191 | 16 | [9.3, 9.51] | **False** |

## F1 mechanism: steps whose realised width is 17-24

- aligned run: 617 steps, per-width medians {17: 9.284, 18: 9.222, 19: 9.293, 20: 9.244, 21: 9.432, 22: 9.379, 23: 9.403, 24: 9.344}
- sealed run: 421 steps, per-width medians {17: 9.36, 18: 9.304, 19: 9.364, 20: 9.369, 21: 9.477, 22: 9.45, 23: 9.508, 24: 9.463}
- aligned plateau inside frozen band +/-0.1 ms: **True**
- max abs per-width difference between the two independent runs: **0.125 ms**

## F2/F3/F4: aligned ladder versus sealed ladder

### steady

- feedback goodput: aligned [3.9721, 3.6275] vs sealed [1.8987, 2.0184]
- feedback padding waste: aligned 0.229 vs sealed 0.229
- feedback median decode width: aligned 9 vs sealed 10
- feedback median TPOT: aligned 8.661 ms vs sealed 8.888 ms
- applied actions: aligned [5, 5] vs sealed [8, 4]
- best measured static: aligned 3.134 vs sealed 2.119
- feedback vs best measured static in its own engine: +21.26%
- **F2 waste reduced: True**
- **F3 TPOT not worse: True**
- **F4 beats best static: True**

### bursty

- feedback goodput: aligned [11.969, 12.0718] vs sealed [12.2323, 9.4838]
- feedback padding waste: aligned 0.003 vs sealed 0.020
- feedback median decode width: aligned 16 vs sealed 12
- feedback median TPOT: aligned 8.330 ms vs sealed 8.227 ms
- applied actions: aligned [4, 5] vs sealed [2, 3]
- best measured static: aligned 12.523 vs sealed 12.504
- feedback vs best measured static in its own engine: -4.01%
- **F2 waste reduced: True**
- **F3 TPOT not worse: False**
- **F4 beats best static: False**

## Repeat spread (noise floor)

| arm | goodput repeats | relative spread % |
|---|---|---:|
| steady_static8 | [2.03, 2.025] | 0.24 |
| steady_static16 | [3.8415, 2.4258] | 58.36 |
| steady_static24 | [1.8909, 2.2955] | 21.39 |
| steady_static32 | [1.5309, 1.5296] | 0.08 |
| steady_shadow32 | [1.1443, 1.5355] | 34.19 |
| steady_feedback32 | [3.9721, 3.6275] | 9.50 |
| bursty_static8 | [2.2543, 2.2441] | 0.45 |
| bursty_static16 | [12.1721, 12.2677] | 0.79 |
| bursty_static24 | [12.4473, 12.5988] | 1.22 |
| bursty_static32 | [12.4782, 12.5416] | 0.51 |
| bursty_shadow32 | [12.399, 12.5509] | 1.22 |
| bursty_feedback32 | [11.969, 12.0718] | 0.86 |
