import json, sys, tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
sys.path.insert(0, sys.argv[1])
from calibration_common_basis import model_basis, save_bf16_master
class Tokenizer:
    def save_pretrained(self, path):
        (Path(path)/"tokenizer_config.json").write_text("{}")
model=torch.nn.Module()
model.model=torch.nn.Module()
layer=torch.nn.Module()
model.model.layers=torch.nn.ModuleList([layer])
layer.post_attention_layernorm=torch.nn.Module()
layer.post_attention_layernorm.weight=torch.nn.Parameter(torch.ones(128,dtype=torch.bfloat16))
layer.post_attention_layernorm.variance_epsilon=1e-5
layer.mlp=torch.nn.Module()
layer.mlp.gate=torch.nn.Linear(128,2,bias=False,dtype=torch.bfloat16)
layer.mlp.experts=torch.nn.ModuleList()
for e in range(2):
    expert=torch.nn.Module()
    for name in ("gate_proj","up_proj","down_proj"):
        setattr(expert,name,torch.nn.Linear(128,128,bias=False,dtype=torch.bfloat16))
    layer.mlp.experts.append(expert)
config={"model_type":"olmoe","hidden_size":128,"intermediate_size":128,"num_experts":2,"num_experts_per_tok":1,"num_hidden_layers":1}
with tempfile.TemporaryDirectory() as temp:
    source=Path(temp)/"source"; source.mkdir()
    dest=Path(temp)/"master"; dest.mkdir()
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    filename="model.safetensors"
    save_file(original,str(source/filename))
    (source/"config.json").write_text(json.dumps(config))
    manifest={"identity":{"weight_files":[{"path":filename}]}}
    with torch.no_grad():
        layer.post_attention_layernorm.weight.div_(2)
        layer.mlp.gate.weight.mul_(2)
        for expert in layer.mlp.experts:
            expert.gate_proj.weight.mul_(2)
            expert.up_proj.weight.mul_(2)
            expert.gate_proj.register_parameter("weight_scale",torch.nn.Parameter(torch.ones(128,1),requires_grad=False))
            expert.gate_proj.quantization_status="frozen"
    basis=model_basis(model,config)
    result=save_bf16_master(model,Tokenizer(),source,dest,manifest,basis)
    saved=load_file(str(dest/filename))
    assert set(saved)==set(original)
    for name,value in saved.items():
        parent,attr=name.rsplit(".",1)
        assert torch.equal(value,getattr(model.get_submodule(parent),attr))
    assert result["quantization_buffers_saved"] is False
    assert not (json.loads((dest/"config.json").read_text()).get("quantization_config"))
    print(json.dumps({"cpu_toy_master_roundtrip":"passed","tensor_count":len(saved),"gpu_used":False,"not_real_calibration":True}))
