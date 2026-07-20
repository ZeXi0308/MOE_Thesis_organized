# TokenRace-EP GPU P0：真实硬件重批开销 + 门槛重算

GPU: NVIDIA GeForce RTX 5090, torch 2.8.0+cu128, cuda 12.8

**GATE (real hardware, best-case rebatch overhead): FAIL**

- llmjp B=32: moderate P99 improvement -11.33% < gate 5% (WITH real-hardware best-case rebatch overhead included)
- llmjp B=64: moderate P99 improvement -10.94% < gate 5% (WITH real-hardware best-case rebatch overhead included)
- llmjp B=128: moderate P99 improvement -10.64% < gate 5% (WITH real-hardware best-case rebatch overhead included)

## (A) 真实FFN scaling拟合

- OLMoE (hidden=2048,inter=1024): base_us=57.056, per_token_us=0.0345, R2=0.4860
- LLM-jp (hidden=512,inter=1024): base_us=55.965, per_token_us=0.0063, R2=0.5724
- (对比 Mac仿真illustrative值: BASE_LAUNCH_US=6.0, PER_TOKEN_US=0.35)

## (B) 真实gather/重批开销

- OLMoE dims gather mean_us=14.304
- LLM-jp dims gather mean_us=14.408
- bare kernel launch floor us=12.288
- **best-case rebatch_overhead_us(每层每次release) = 26.644**

## (C) 门槛重算结果（moderate场景，含best-case重批开销）

| model | B | P50改善 | P99改善 | mean改善 |
|---|---|---|---|---|
| olmoe | 32 | 4.96% | 5.64% | 4.30% |
| olmoe | 64 | 6.06% | 6.98% | 5.78% |
| olmoe | 128 | 6.07% | 7.77% | 5.88% |
| llmjp | 32 | -12.77% | -11.33% | -12.93% |
| llmjp | 64 | -12.49% | -10.94% | -12.96% |
| llmjp | 128 | -12.69% | -10.64% | -13.06% |