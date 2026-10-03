**2026-09-13 — SOURCE-QUALIFIED / GPU UNRUN。** 本说明只确认接线与理论预算；没有下载权重、修改 runner 或启动 GPU。目标是 Qwen3-30B-A3B BF16 四请求静态资源 smoke，不是调度收益实验。

1. **模型与可复用入口。** 已冻结的 [config](qwen3.config.json) / [manifest](qwen3.manifest.json) 对应 revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`。48 层、128 routed experts、top8、H=2048、expert intermediate=768、4 KV heads、head_dim=128；`intermediate_size=6144` 不是本模型 routed expert 的宽度。此配置没有 shared expert。vLLM 官方 v0.26 Qwen 类使用原生 internal gate，随后在 modular 路径选择 top-k，再调用现有 adapter 所拦截的 `forward_modular -> UnquantizedFusedMoEMethod.apply`。保留原生归一化与权重，不能自己重算路由。[官方 config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)，[v0.26 Qwen 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/models/qwen3_moe.py)

   本地实际 v0.26 镜像 [moe_runner.py](/private/tmp/moe_vllm026_installed/model_executor/layers/fused_moe/runner/moe_runner.py:350) 对无 shared expert、无 input transform 返回 `shared_experts_input=None`；其 549/798 行维持先 gate / top-k、后 expert execution。现有 [adapter](../admission_width_baseline_v026/source/wisp_v026_adapter.py:94) 的 shared-expert 拒绝条件因此与本配置相容。仍限 BF16、bias-free、TRITON modular、eager、TP/PP/DP=1、无 EP/EPLB/DBO/LoRA/async/prefix-cache；不能外推到带 shared expert 的其它模型。Qwen 类本体来自官方 v0.26 tag，尚未核对远端安装文件 SHA，须在 smoke 前补匹配。

2. **新 runner 最少改动（尚未实现）。** 从 [既有 runner](../admission_width_baseline_v026/source/run_native_pager.py) 的普通非 event 路径复用引擎/完整请求 capture：允许 cap48/56；model/tokenizer 指向同一冻结的本地小型 metadata；shape 检查改为 48/128/top8，并读 `moe_intermediate_size`；显式编译 maxseq4，提交四条实际请求，固定静态 prefill/token/KV 参数后再 smoke。现有 batched map 按 `state.num_experts` 分配，expert-major 分组按实际 expert IDs 工作，不需为了 128 experts 重写 map、cache 或 router。

   新 loader 必须在 `LLMEngine.from_engine_args` 之前注册，pager 的 `install()` 仍须在模型构造之前调用。Qwen 架构已经在 [v0.26 模型注册表](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/models/registry.py) 注册，无需新造模型类。四请求 smoke 不应调用要求八请求的 continuous runner 参数入口，也不引入 phase/admission 动态动作。CPU diagnostics、实际 cap、完整 receipts 与 paging bytes 保留；不能把普通 capture 的空 existing-decode 集合当成验证通过。

3. **loader contract。** `register_model_loader(name)` 要求 `BaseModelLoader` 子类；工厂以 `Loader(load_config)` 构造。最小实现给出 `download_model(model_config)` 和 `load_weights(model, model_config)->None`，继承 `BaseModelLoader.load_model()`，保留原生 `initialize_model -> load_weights -> process_weights_after_loading -> eval`。Qwen 模型构造器是关键字参数 `vllm_config, prefix`；模型自身 `load_weights(iterator)->set[str]` 返回映射后的 vLLM 目标参数名，不是 HF 源 key。只调用一次模型 load_weights，让原生 AutoWeightsLoader 处理 q/k/v 堆叠和逐 expert 映射。[注册](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/model_loader/__init__.py)，[Base loader](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/model_loader/base_loader.py)

   可复用 `DefaultModelLoader.track_weights_loading(self, model, loaded_set)` 的目标覆盖检查，但它会为具有 postprocess quant method 的模块补入参数名，不能证明逐 expert 输入完整。另需 serial iterator 的 index 源 key 精确覆盖、无重复、每片 SHA 与自然 EOF。Default loader 构造器只允许三个标准 extra-config keys；直接继承它再塞自定义 manifest 配置会失败。Base 子类显式解析自己的配置更直接。[Default loader](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/model_loader/default_loader.py)，[AutoWeightsLoader](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/models/utils.py)

4. **内存公式（GiB 为 2^30 bytes，非实测峰值）。** 单专家 BF16 权重 `3*H*I*2 = 9,437,184 B = 9 MiB`。48 层完整 CPU master 为 **54 GiB**，每层 **1.125 GiB**；index 总 tensor bytes `61,064,245,248`，非 expert 权重差额 **2.870510 GiB**。[已冻结 index](qwen3.index.json)

   | cap | GPU expert scratch `48*cap*9MiB` | +非expert+示例KV512MiB | 再加当前整层 postload transient |
   |---|---:|---:|---:|
   | 48 | 20.250000 GiB | 23.620510 GiB | 24.745510 GiB |
   | 56 | 23.625000 GiB | 26.995510 GiB | 28.120510 GiB |

   最后一列是保守会计项叠加，KV 未必与后处理峰值同时存在。实际还要加入 CUDA context、原生 attention/kernel workspace、激活、allocator reserved/fragmentation、验证临时权重与观测状态。Root 已选下一资格化优先 **cap48 / KV512MiB / token64 / 四请求 maxseq4**；不是 GPU 池最大值或同预算最优，也尚未证明可加载。BF16 KV 每 token 为 `2*48*4*128*2=98,304 B`；512MiB 仅是约 5461 token slots 的未考虑 block 取整上界。四请求不改变固定 scratch 总量。

   当前 WiSP [create/postload 源码](/private/tmp/moe-pager-review-wisp-20260912/src/wisp/integrations/vllm/fused_moe.py:673) 实际先分配普通 CPU master，再在 postload pin；不能按其旧 docstring 宣称构造即 pinned。vLLM [device_loading_context](/private/tmp/moe_vllm026_installed/model_executor/model_loader/utils.py:155) 会把当前层 CPU 参数暂移 GPU，之后 WiSP 创建 scratch、改为空 placeholder；必须计入该层 1.125GiB 临时 GPU 权重。CPU 加载阶段需 `54GiB + 当前shard驻留页 + 尚被消费者持有的clone + runtime/OS`；最大 shard 是 **3.725267 GiB**。postload pin 阶段另有当前层普通/pinned 副本短时并存，结构项约 `54+1.125GiB`，并非全模型双份 pinned。实际 RSS/锁页可用量仍待加载验证，serial 删除磁盘 shard 不等于所有 mmap 引用已释放。

5. **离线与显式下载。** 原 runner 在导入 torch/vLLM 前设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。保留本地 config/tokenizer 加载；serial shard fetch 若用标准 HTTP 客户端访问冻结 revision/filename，不受 HF SDK 离线变量拦截，也不应走默认 loader 的全 snapshot 下载。若 fetch 使用 `hf_hub_download`，必须放到显式在线下载子进程并在导入 SDK 前设置环境；在已导入的主进程临时改环境不能可靠解除离线状态。[HF 环境变量定义](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)

6. **输入身份。** 原 prepared 中保留 `source_requests[].prompt` 与文档/文本 hash，可对同一公开自然文本用冻结 Qwen tokenizer 重新编码；不得复用 OLMoE token IDs。记录 tokenizer revision、相同 prompt hash、新 token IDs hash/实际长度；若固定前缀长度不足，回到相同原公开文档取更长自然文本，不 padding 或合成补词。本轮没有读取/打印正文、重新分词或选择正式四条输入。

本地源码锚点 SHA：runner `29d16549131480a5bd439f653ed002213b57a316ef32b70ac526497e1b140d7c`；adapter `b6c25e2656678636a596403fba7d9318465a4bee5b0eaf9c7b457dadcbafdd93`；实际 loader/utils `01b9e0bfc7bd9a0c94251cfb1c01ee733c4f57ce3d8e1df8ee9d2aa4df70e543`；实际 moe_runner `3c000e17f726c9995a9a950db952415a4b2198d0778e3cedda96355421edfdae`。

**唯一接续：** 合并 serial iterator 的 CPU 资格化后，在形成可运行命令之前再次核对实际 host/cgroup 内存与锁页限制、磁盘可用量及单片 staging 空间、GPU 占用、已安装 vLLM/Qwen/loader 源码与冻结 metadata；重新 tokenize 同一真实 public prompt，冻结四请求输入与完整预算，再做一次原生加载+静态完整请求 smoke。该次只回答真实超显存 BF16 模型是否能合法运行与正确记账；当前仍不下载权重。
