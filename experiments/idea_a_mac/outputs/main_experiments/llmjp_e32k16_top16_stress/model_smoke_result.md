# Model Smoke Result

model: `llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M`

- model_type: `mixtral`
- layers: `16`
- hidden_size: `512`
- intermediate_size: `1024`
- num_experts: `32`
- num_experts_per_tok: `16`
- load_seconds: `473.63`
- forward_seconds: `0.17`
- logits_shape: `(1, 12, 99584)`
- torch_device: `cpu`
- dtype: `bfloat16`
