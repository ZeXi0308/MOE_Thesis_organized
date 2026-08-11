# SemanticFence-MoE 实验跟踪器

> 更新时间：2026-08-09 23:26 +0800  
> 对应冻结计划：`refine-logs/EXPERIMENT_PLAN_20260809_202112.md`  
> 总状态：`LOCAL_IMPLEMENTATION_VERIFIED / GPU_ACCEPTANCE_BLOCKED_CONNECTION_CLOSED`。没有新 GPU 结果，没有创建 acceptance、lock、contract 或 scientific `COMPLETE.json`。

## 唯一实验

`SF-P0`：OLMoE / RTX 5090 calibration-to-fresh-eval expert rebatching Pilot。

| ID | 阶段 | 当前状态 | 已闭合证据 / 下一门 |
|---|---|---|---|
| `SF-P0-DATA` | fresh data | `DONE_LOCAL` | fresh-32 恰好 32 条、全文 hash 唯一、四源排除 union=1137、overlap=0；manifest SHA-256 `2608ef5d...695d7` |
| `SF-P0-UNIT` | implementation/preflight | `DONE_LOCAL` | 4 个实现文件、4 个测试文件、冻结 config 已落盘；41/41 CPU tests 通过，`py_compile` 通过，两轮独立只读复审 PASS |
| `SF-P0-ACCEPT` | real GPU acceptance | `BLOCKED_CONNECTION_CLOSED` | `connect.westd.seetacloud.com:47792` 于 2026-08-09 最终预检返回 exit 255 / `Connection closed`；未观测 GPU UUID/driver/stack |
| `SF-P0-SEAL` | pre-science lock | `NOT_CREATED_BY_DESIGN` | 必须先有完整 `ACCEPTANCE.json` + `ACCEPTANCE_COMPLETE.json`；当前禁止伪造 |
| `SF-P0-CAL` | calibration/contract | `NOT_RUN` | 等待同一真实 RTX 5090 acceptance 与新 lock |
| `SF-P0-EVAL` | fresh A/B/C/D | `NOT_RUN` | 等待 calibration-only contract seal；fresh workload 只能在 seal 后首次执行 |
| `SF-P0-DECIDE` | parent recompute | `NOT_RUN` | 只有 40 份 raw BF16、trace、contract 与 worker/parent closure 完整后才可写 `COMPLETE.json` |

## 本地实现闭合

1. **Calibration-only 边界**：calibration worker 只读冻结 calibration manifest；fresh-32 capture 被移到 `CONTRACT_SEAL.json` 之后。
2. **唯一 treatment**：A/B/C/D 共享 `(layer, expert, row-id)` traversal；只改变组内 packing。10 个正式 repeat 使用冻结轮转顺序并保留 pair ID。
3. **Fail closed contract**：contract 按 `(layer, expert, M, signature)` 建立；任一 repeat mismatch、support 不足、多 signature 歧义、未知 stack/M/signature 都回退 `M=1` 或判 evidence incomplete。
4. **Pre-call authority**：完整 descriptor 在执行前独占写入并 seal，绑定 config、model weights、source、stack/math state、call/row identity、padding、三投影 shape 与 expected signature。
5. **Raw evidence**：四臂 × 10 repeats 保存 40 份 canonical raw-BF16 文件；parent 按 `<u2` 独立重算逐 row hash/mismatch、victim、coverage、latency 与全部 decision fields。
6. **Trace/TOCTOU**：trace 新进程在反序列化前注册动态类型；contract 在 fresh evaluation 前 seal，并在 evaluation 后、trace 后再次核验同一 file/content hash。
7. **Completion-last**：缺少任一 raw/hash/trace/environment/contract closure 时没有 scientific authority；`COMPLETE.json` 只能最后写且不可覆盖。

## Fresh-32 数据账本

| Artifact | SHA-256 |
|---|---|
| `eval_manifest.jsonl` | `2608ef5d93a9da36b816eddfb8e3bf495631f0f19934a703bc813757443695d7` |
| `provenance.json` | `d9f0e1918a9754cc041030e313e95181d6a1480871304fe632ee5f3707e55f9a` |
| `exclusion_report.json` | `32584b7506edec204d7379aab841badc22bd543fdfd4dc402c89a2468191ef85` |
| `artifact_hashes.json` | `ec962592c61f04262248b4e34242d9c76e4d2aaa56b687ebd0ad09da82b21060` |

独立复核：32/32 capped tokenizer length=2081；历史 registry、旧 calibration、旧 sealed、旧 smoke 四源 canonical full-text union=1137；fresh overlap=0。

## 已验证命令

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s docs/ideas/semanticfence/experiments -p 'test_*.py'
```

结果：`Ran 41 tests ... OK`。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  docs/ideas/semanticfence/experiments/*.py
```

结果：exit 0。

## GPU 恢复后的原子 launch

以下命令只能在可连接、空闲且确认为 `NVIDIA GeForce RTX 5090` 的同一主机执行；每个输出路径必须是新的空路径。

```bash
python docs/ideas/semanticfence/experiments/run_pilot_5090.py acceptance \
  --config docs/ideas/semanticfence/experiments/configs/pilot_5090_v1.json \
  --repo-root "$PWD" \
  --model-path /root/autodl-tmp/models/olmoe \
  --output-dir <new-acceptance-dir>

python docs/ideas/semanticfence/experiments/run_pilot_5090.py seal \
  --config docs/ideas/semanticfence/experiments/configs/pilot_5090_v1.json \
  --repo-root "$PWD" \
  --acceptance-artifact <new-acceptance-dir>/ACCEPTANCE.json \
  --output <new-lock.json>

python docs/ideas/semanticfence/experiments/run_pilot_5090.py run \
  --config docs/ideas/semanticfence/experiments/configs/pilot_5090_v1.json \
  --repo-root "$PWD" \
  --model-path /root/autodl-tmp/models/olmoe \
  --acceptance-artifact <new-acceptance-dir>/ACCEPTANCE.json \
  --frozen-lock <new-lock.json> \
  --output-dir <new-run-dir>
```

## 冻结判据

- **SUPPORT**：A 10/10 raw-bit stable；B 至少 8/64 victims 有 10/10 bitwise-stable mismatch；D mismatch=0、覆盖至少 8 victims、至少两个自然 `M>1`、padding=0；D/A paired median latency reduction≥10%；C 不同时覆盖 D。
- **WEAKEN**：B 问题实例不足、D 任一 mismatch、D 退化为 M1/单一 M/padding、收益<10%，或 C 同时不差于 D。
- **UNABLE**：A 不稳定，route/row/weight/dtype/order 不守恒，split 泄漏，signature 漂移，raw/trace/hash/environment/completion 不完整，GPU 污染或超时。

当前实验已经足以推进下一轮探索，停止继续扩展审计项。
