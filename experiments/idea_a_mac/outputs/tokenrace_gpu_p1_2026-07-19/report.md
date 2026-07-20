# TokenRace-EP GPU P1：CUDA Graph代价 + 多组释放代价

GPU: NVIDIA GeForce RTX 5090

平均CUDA graph replay相对eager的优势(=token_race无法享用的部分): 39.57us/次调用

## 多组重批开销 (us, batch=128)

- olmoe: 2组=47.90us, 3组=68.03us, 4组=87.36us, 6组=127.20us
- llmjp: 2组=48.00us, 3组=68.10us, 4组=87.65us, 6组=126.61us

## 场景(ii): 2组重批 + CUDA graph代价 (overhead=87.52us/层)

| model | B | P50改善 | P99改善 |
|---|---|---|---|
| olmoe | 32 | -38.51% | -32.40% |
| olmoe | 64 | -35.80% | -30.78% |
| olmoe | 128 | -35.58% | -30.59% |
| llmjp | 32 | -61.73% | -53.37% |
| llmjp | 64 | -61.74% | -52.79% |
| llmjp | 128 | -62.18% | -53.31% |

## 场景(iii): 4组重批，无CUDA graph代价 (overhead=87.50us/层)

| model | B | P50改善 | P99改善 |
|---|---|---|---|
| olmoe | 32 | -38.49% | -32.39% |
| olmoe | 64 | -35.79% | -30.77% |
| olmoe | 128 | -35.57% | -30.58% |
| llmjp | 32 | -61.72% | -53.35% |
| llmjp | 64 | -61.73% | -52.78% |
| llmjp | 128 | -62.17% | -53.30% |