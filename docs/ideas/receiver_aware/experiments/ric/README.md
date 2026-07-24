# RIC-v1 实验流水线

状态：**IMPLEMENTATION / NO SCIENTIFIC RESULT**

本目录严格服务于 [`RIC_Phase2_冻结实验协议_2026-07-22.md`](../../RIC_Phase2_冻结实验协议_2026-07-22.md)，现已包含从冻结输入到 oracle/replay gate 的完整实现骨架：

- WikiText冻结窗口、双 tokenizer长度与历史排除 manifests；
- 未 patch native forward 的完整 route capture；
- forward内只读捕获 native `aten.topk`，并以另一 call-site exact replay；
- `output_router_logits` 与独立 gate hook在 flatten 前逐层 raw-tensor identity；
- ordered expert + BF16 effective-weight tuple hash与 native MoE output重建；
- 真实 expert-output 的 ready-result pack/service reorder；
- 单一 persistent sender-local CUDA stream上的 streaming release、full-layer-barrier因果对照与 canonical reduction；
- streaming主效应和 streaming-minus-barrier interaction的四个 paired LCB，以及逐 trial事件前驱复算；
- candidate streaming/full-barrier两种 release mode各自独立的 CUDA profiler/NVTX诊断；
- 5090 service/control LUT、exact-load time-normalized virtual-EP scenarios；
- information-constrained replay、matched-world exact oracle、会计与 G1/G2/G3 runner；
- strict-JSON、stage-specific signoff、raw trials、环境/provenance和 self-hashed artifacts。

这些代码存在不等于科学结论。开发输出必须保持 `NOT_TESTED`；只有全局 Phase 4 `SIGNED-OFF` 后才能 formal calibration，且只有 G1通过后才能创建 sealed输入。

## 文件

```text
prepare_data.py           WikiText/data manifest + historical exclusion registry
capture_routes_gpu.py     native top-k/route tuple/output parity + placement
measure_capability_gpu.py persistent-stream 5090 capability probe
measure_service_lut_gpu.py real expert/pack/H2D/host-control raw service LUT
build_scenarios.py        normalized workload census + L2 virtual-EP graph
replay.py                 S/B/R0/Rcmp/Rwire sender-local policy replay
oracle.py / run_oracle.py matched-world information-constrained exact oracle
accounting.py             complete-trace metrics/bootstrap/conservation
run_experiment.py         calibration lock + G1/G2/G3 runner
formal_provenance.py      strict JSON/source closure/signoff verification
build_phase4_signoff.py   stage-specific reviewed-source attestation builder
test_*.py                 fail-closed/adversarial tests
```

## 本地测试

从仓库根运行：

```bash
python3 -m unittest discover \
  -s docs/ideas/receiver_aware/experiments/ric \
  -p 'test_*.py'
python3 -m py_compile docs/ideas/receiver_aware/experiments/ric/*.py
```

无 CUDA 的本地环境只验证纯函数和 fail-closed gates；不能生成 5090 route/capability/LUT artifact。

## Calibration dev smoke

dev 只允许 calibration。以下输出必须保持 `NOT_TESTED`：

Amendments L–O 前生成的 route/capability/LUT/scenario dev artifacts均已 `SUPERSEDED`：它们只用于发现 native top-k tie、arm-specific CUDA stream、JSON key normalization、finite-horizon workload与 full-barrier因果 estimand问题，不能作为 G1/G2/G3证据。旧 5us/10% barrier跨-policy timing等价门已由 Amendment O 明确取代；barrier timing只作诊断，正式 G1要求四个 paired LCB严格大于零并通过全 trial事件前驱门。修复后必须从 data manifest开始重跑，不允许混用旧 stage hash。

```bash
python3 docs/ideas/receiver_aware/experiments/ric/prepare_data.py \
  --role calibration \
  --mode dev \
  --output-dir /tmp/ric-calibration-data \
  --cache-dir /root/autodl-tmp/hf-cache

python3 docs/ideas/receiver_aware/experiments/ric/capture_routes_gpu.py \
  --model-key olmoe \
  --data-manifest /tmp/ric-calibration-data/data_manifest_calibration.json \
  --mode dev \
  --output-dir /tmp/ric-route-olmoe \
  --model-path /root/autodl-tmp/models/olmoe

python3 docs/ideas/receiver_aware/experiments/ric/measure_capability_gpu.py \
  --model-key olmoe \
  --data-manifest /tmp/ric-calibration-data/data_manifest_calibration.json \
  --mode dev \
  --output-dir /tmp/ric-capability-olmoe \
  --model-path /root/autodl-tmp/models/olmoe

python3 docs/ideas/receiver_aware/experiments/ric/measure_service_lut_gpu.py \
  --model-key olmoe \
  --data-manifest /tmp/ric-calibration-data/data_manifest_calibration.json \
  --mode dev \
  --output-dir /tmp/ric-service-lut-olmoe \
  --model-path /root/autodl-tmp/models/olmoe

python3 docs/ideas/receiver_aware/experiments/ric/build_scenarios.py \
  --role calibration --mode dev --model-key olmoe --link-gbps 200 \
  --data-manifest /tmp/ric-calibration-data/data_manifest_calibration.json \
  --route-dir /tmp/ric-route-olmoe \
  --service-lut-dir /tmp/ric-service-lut-olmoe \
  --output-dir /tmp/ric-scenario-olmoe-link200

python3 docs/ideas/receiver_aware/experiments/ric/build_scenarios.py \
  --role calibration --mode dev --model-key olmoe --link-gbps 100 \
  --primary-scenario-dir /tmp/ric-scenario-olmoe-link200 \
  --data-manifest /tmp/ric-calibration-data/data_manifest_calibration.json \
  --route-dir /tmp/ric-route-olmoe \
  --service-lut-dir /tmp/ric-service-lut-olmoe \
  --output-dir /tmp/ric-scenario-olmoe-link100
```

100/400 Gbps sensitivity禁止自行调用 RNG 或重算 normalization，必须通过 `--primary-scenario-dir`消费 200 Gbps self-hashed scenario；缺失即 hard-fail。

LLM-jp 使用 `--model-key llmjp --model-path /root/autodl-tmp/models/llmjp`。显式 `--model-path` 直接加载完整离线目录，不触发 repo-id/revision/cache lookup；producer仍在 artifact中绑定冻结 repo/revision和本地 tree manifest。只有未使用本地目录、缓存不完整且明确允许联网时才加 `--allow-download`。

## Formal discipline

formal 运行必须额外传 `--signoff`，且 attestation需逐 producer绑定：

```text
status = SIGNED-OFF
open_p0 = 0
protocol_sha256
config_sha256
producer source sha256
data role / manifest sha256 / model key（按入口）
model_tree_manifest_sha256
```

formal calibration runner的签字还必须按 model 同时绑定
`capability_probe_sha256` 与 `capability_producer_signoff_sha256`；只绑定可重新 self-hash 的 capability JSON 不足以进入 G1。

规则：

- `--mode dev --role sealed` 必须 hard-fail；
- capability producer永远只接受 calibration manifest；
- formal GPU producer必须传显式 `--model-path`；它们对目录下每个文件（包括所有 safetensors shards）逐字节 SHA-256，独立重算的 `model_tree_manifest_sha256` 必须与 Phase-4 signoff一致，禁止只用 revision label或大文件 size。
- formal输出目录必须不存在，producer临时目录成功后原子 rename；
- sealed manifest首次创建即写 `sealed_consumption_record.json`；
- selected text与任一历史 manifest碰撞时 hard-fail，禁止自动跳到下一条文本或下一窗口；
- route capture不能调用旧 `patch_mixtral_moe`；
- capability使用的是本机真实 expert output和非默认本地 CUDA stream，仍然是 `NOT_NCCL / NOT_RDMA`；producer与consumer都会从两份 profiler trace逐 GPU activity复算唯一 stream ordinal，context外 checksum/hash回落 default stream会 hard-fail。
- service LUT逐 repeat保存 expert execution、pack、canonical reduction、`H2D_NOT_RDMA`与 host raw tax；host path直接调用唯一 canonical `wire.py`，包括 state/record build、identity hash、encode/decode、collision-checked apply、epoch/sequence apply和 sender policy-cache lookup，禁止第二套 codec。
- payload bytes由真实 weighted expert-output tensor的 dtype、末维 elements与 element size自描述，连同 layout hash写入 raw/summary/metadata；禁止 runner猜模型常数。
- contract transfer仅由 canonical message bytes × 200 Gbps的 `analytic_network`记账，`H2D_NOT_RDMA`不得代替；`0/5/20/50 us`只作为单列 `synthetic_delay`。raw host与同 record-count harness都先保存，不得静默将减 harness后的负数裁为零，也不得把 summary median冒充 raw独立样本。

在 Phase 4 新报告明确 `SIGNED-OFF` 前，只允许上述 calibration/dev smoke；不得创建或消费 sealed数据，也不得把 artifact解释为 G0/G1通过。
