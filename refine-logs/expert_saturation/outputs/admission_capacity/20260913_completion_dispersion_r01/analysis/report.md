# Width-cost accounting across admission policies

Reference arm: `native32`. Evidence ceiling: descriptive accounting of executed traces.

| arm | wall s | max ITL s | completion span s | tail waste s | preempt | SLO |
|---|---:|---:|---:|---:|---:|---:|
| native32 | 22.804 | 4.444 | 2.459 | 1.133 | 2 | 32/32 |
| rotate_r0 | 22.930 | 1.006 | 2.146 | 0.797 | 10 | 32/32 |
| rotate_r1 | 22.902 | 1.006 | 2.149 | 0.801 | 10 | 32/32 |
| headroom_r0 | 23.488 | 1.382 | 6.074 | 1.701 | 0 | 32/32 |
| safe29 | 26.832 | 0.308 | 7.953 | 4.747 | 0 | 29/32 |

## Deltas against `native32`

| arm | d wall s | d tail waste s | explained | d max ITL s | d span s |
|---|---:|---:|---:|---:|---:|
| rotate_r0 | +0.126 | -0.336 | -267.3% | -3.438 | -0.313 |
| rotate_r1 | +0.098 | -0.331 | -337.8% | -3.438 | -0.309 |
| headroom_r0 | +0.684 | +0.569 | 83.1% | -3.062 | +3.616 |
| safe29 | +4.028 | +3.615 | 89.7% | -4.136 | +5.494 |

Sign agreement with wall delta: tail waste 2/4, max ITL 0/4.

## Measured per-width decode cost

- **native32** — w1: 4.45ms (4.451 ms/tok, n=128)  w2: 5.51ms (2.757 ms/tok, n=104)  w30: 18.71ms (0.624 ms/tok, n=98)  w31: 18.81ms (0.607 ms/tok, n=122)  w32: 19.44ms (0.608 ms/tok, n=708)
- **rotate_r0** — w2: 5.56ms (2.778 ms/tok, n=87)  w4: 6.80ms (1.699 ms/tok, n=46)  w30: 18.84ms (0.628 ms/tok, n=79)  w31: 18.99ms (0.613 ms/tok, n=109)  w32: 19.52ms (0.610 ms/tok, n=708)
- **rotate_r1** — w2: 5.55ms (2.776 ms/tok, n=87)  w4: 6.80ms (1.701 ms/tok, n=46)  w30: 18.81ms (0.627 ms/tok, n=79)  w31: 19.02ms (0.614 ms/tok, n=109)  w32: 19.50ms (0.609 ms/tok, n=708)
- **headroom_r0** — w1: 5.05ms (5.048 ms/tok, n=290)  w28: 18.54ms (0.662 ms/tok, n=33)  w30: 18.84ms (0.628 ms/tok, n=79)  w31: 18.95ms (0.611 ms/tok, n=129)  w32: 19.47ms (0.609 ms/tok, n=701)
- **safe29** — w3: 6.61ms (2.205 ms/tok, n=941)  w29: 17.77ms (0.613 ms/tok, n=937)

## Threshold sensitivity of `low width`

| frac | native32 | rotate_r0 | rotate_r1 | headroom_r0 | safe29 |
|---|---|---|---|---|---|
| 0.3 | 1.038 | 0.704 | 0.708 | 1.597 | 4.651 |
| 0.4 | 1.087 | 0.751 | 0.755 | 1.693 | 4.700 |
| 0.5 | 1.133 | 0.797 | 0.801 | 1.701 | 4.747 |
| 0.6 | 1.161 | 0.826 | 0.830 | 1.760 | 4.781 |
| 0.7 | 1.191 | 0.856 | 0.861 | 1.828 | 4.811 |
