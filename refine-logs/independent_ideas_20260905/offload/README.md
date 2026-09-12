# Offload / Prefetch：先核算运行域，再测自然 miss

当前为 `STRUCTURAL_CAPACITY_CALCULATION / GPU_UNRUN`。没有新 prefetcher。
旧 resident/full-top-k formulation 的 NO-GO 保留；旧 runner 的离线带宽与计算时间
拼接不能变成新 offload runtime 的收益证据。

`inspect_residency.py` 不加载权重，直接读取本地 safetensors header，按
`(layer, expert)` 统计全部三投影。它使用 checkpoint 内的 KV heads、层数和 head dim，
核算 BF16 dense KV：

```text
KV bytes = active requests × total cached tokens × 2(K,V) × layers × KV heads × head dim × 2 bytes
required bytes >= stored BF16 weights + dense KV
```

总缓存长度包括生成；当前 OLMoE 的 checkpoint 上限为 4096。大于上限直接拒绝，
不把未经验证的 RoPE 扩展当作自然压力。`--hbm-gib` 与可选 reserve 都是明确的
规划假设；小于 HBM 只得到 `FIT_NOT_PROVEN`，因为 workspace、activation、allocator、
CUDA graph 等尚未计入。下界超过 HBM 也只证明给定无共享 dense-KV 模型无法全驻留，
不证明该并发有 SLO 价值，更不证明预取可改善请求。

```bash
.venv/bin/python refine-logs/independent_ideas_20260905/offload/inspect_residency.py \
  --snapshot "$OFFLOAD_MODEL_SNAPSHOT" --hbm-gib 32 \
  --caps 2,4,8,16,32 --contexts 144,4096 --output-dir /tmp/offload-budget-r01

.venv/bin/python -m unittest discover \
  -s refine-logs/independent_ideas_20260905/offload -p 'test_*.py' -v
```

本机已实际读取缓存 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5` 的三个
shard header；结果见 [prepared_r01/capacity.json](prepared_r01/capacity.json)。
权重 tensor 数据为 12.88794 GiB，其中专家 12 GiB，共 16×64 个专家对象。
假设 32 GiB GPU，8×144 tokens 的 weights+KV 下界为 13.02857 GiB；8×4096 为
16.88794 GiB；32×4096 为 28.88794 GiB。这些都不是实际显存峰值。

## 下一最小实验

先选业务有意义、context 合法的一个 16–32 requests 自然 episode；根据真实 peak
nonexpert/KV/workspace 需求确定可用专家容量，不用任意 `cache_capacity=8`。
先执行 demand-LRU，再按独立 calibration 的频率做 static pinning + 同一 demand
fallback。每个 arm 独立实际执行自己的 KV、route、cache eviction 和未来状态。
低压力全驻留 cell 是负控；如果 backend 支持，CPU 执行专家也是与搬运竞争的简单动作。

producer 需要逐 `(request, step, layer, expert, fetch_id)` 记录 resident/hit/miss、
fetch start/end、consumer 的其它依赖全部 ready 时刻、实际 consumer start、CPU/GPU
执行选择和完整 request completion。共享 fetch 不可按消费者数重复收费。先报告
自然 miss 及真实 consumer waiting 区间，不把各 fetch duration 相加成 request saving。
只有带资源与依赖的完整时间线或实际干预复跑，才能判断等待是否进入请求关键路径。

均全驻留：关闭该 cell 的 prefetch action-space 问题；有 miss 但无暴露等待：停止该
cell 的预取机制；有等待：先比较简单缓存、降低非抢占 cap 和 CPU/GPU 执行选择。
只有上述基线后仍有 residual，才值得实现 future-next-use 的收费上界和最小预取。
本目录尚未接通这样的 native producer，也没有实际执行 LRU/pinning 或 transfer Oracle。

已有 [MoE-Infinity](https://arxiv.org/abs/2401.14361) 研究专家缓存/预取，
[Fiddler](https://arxiv.org/abs/2402.07033) 研究 CPU/GPU 编排；因此“预测专家再预取”
本身不构成新贡献。可能的 residual 是 KV 与专家权重竞争时的请求级容量边界，仍待测。
