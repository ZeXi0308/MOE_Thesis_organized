# TokenRace-EP GPU P0：真实硬件重批开销 + 门槛重算

GPU: NVIDIA GeForce RTX 5090, torch 2.8.0+cu128, cuda 12.8

**GATE (real hardware, best-case rebatch overhead): FAIL**

- llmjp B=32: moderate P99 improvement -10.60% < gate 5% (WITH real-hardware best-case rebatch overhead included)
- llmjp B=64: moderate P99 improvement -10.91% < gate 5% (WITH real-hardware best-case rebatch overhead included)
- llmjp B=128: moderate P99 improvement -11.23% < gate 5% (WITH real-hardware best-case rebatch overhead included)

## (A) 真实FFN scaling拟合

- OLMoE (hidden=2048,inter=1024): base_us=57.528, per_token_us=0.0362, R2=0.5546
- LLM-jp (hidden=512,inter=1024): base_us=56.867, per_token_us=0.0013, R2=0.0412
- (对比 Mac仿真illustrative值: BASE_LAUNCH_US=6.0, PER_TOKEN_US=0.35)

## (B) 真实gather/重批开销

- OLMoE dims gather mean_us=14.532
- LLM-jp dims gather mean_us=14.528
- bare kernel launch floor us=12.576
- **best-case rebatch_overhead_us(每层每次release) = 27.106**

## (C) 门槛重算结果（moderate场景，含best-case重批开销）

| model | B | P50改善 | P99改善 | mean改善 |
|---|---|---|---|---|
| olmoe | 32 | 4.46% | 6.44% | 4.23% |
| olmoe | 64 | 6.06% | 6.60% | 5.53% |
| olmoe | 128 | 6.32% | 6.98% | 5.86% |
| llmjp | 32 | -12.67% | -10.60% | -12.75% |
| llmjp | 64 | -12.90% | -10.91% | -13.01% |
| llmjp | 128 | -12.70% | -11.23% | -13.05% |