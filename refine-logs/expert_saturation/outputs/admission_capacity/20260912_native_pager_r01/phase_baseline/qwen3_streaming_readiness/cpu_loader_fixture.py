"""Execute installed vLLM loader AST with CPU model/platform/transport fixtures."""
import argparse
import ast
from abc import ABC, abstractmethod
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import torch


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    sources = root / "installed_sources/model_executor/model_loader"
    events, source_hashes, control = [], {}, {"missing": False}
    logger = NS(debug=lambda *a: None, info=lambda *a: None, warning=lambda *a: None)
    env = dict(ABC=ABC, abstractmethod=abstractmethod, torch=torch, nn=torch.nn,
               instrument=lambda **kw: lambda f: f, logger=logger,
               set_default_torch_dtype=lambda dtype: nullcontext(),
               current_platform=NS(is_cuda_alike=lambda: False, is_xpu=lambda: False),
               log_model_inspection=lambda model: None, _LOAD_FORMAT_TO_MODEL_LOADER={})
    def extract(filename, symbol, method=None):
        path = sources / filename
        source_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == symbol)
        if method:
            node.body = [n for n in node.body if getattr(n, "name", None) == method]
            node.bases, node.decorator_list = [], []
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
        exec(compile(tree, str(path), "exec"), env)
    extract("base_loader.py", "BaseModelLoader")
    extract("base_loader.py", "_has_online_quant")
    extract("__init__.py", "register_model_loader")
    extract("default_loader.py", "DefaultModelLoader", "track_weights_loading")
    base = env["BaseModelLoader"]
    modules = {
        "vllm": dict(__version__="0.26.0"), "vllm.model_executor": {},
        "vllm.model_executor.model_loader": dict(register_model_loader=env["register_model_loader"]),
        "vllm.model_executor.model_loader.base_loader": dict(BaseModelLoader=base),
        "vllm.model_executor.model_loader.default_loader": dict(DefaultModelLoader=env["DefaultModelLoader"]),
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        module.__path__ = []
        sys.modules[name] = module
    spec = importlib.util.spec_from_file_location("qwen_serial_loader_fixture", root / "qwen_serial_loader.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    assert env["_LOAD_FORMAT_TO_MODEL_LOADER"][adapter.LOAD_FORMAT] is adapter.QwenSerialLoader
    assert adapter.QwenSerialLoader.load_model is base.load_model
    model_dir, workspace = output / "model_metadata", output / "workspace"
    model_dir.mkdir()
    workspace.mkdir()
    (model_dir / "config.json").write_bytes((root / "qwen3.config.json").read_bytes())
    extra = {"manifest": str(root / "qwen3.manifest.json"), "index": str(root / "qwen3.index.json"),
             "workspace": str(workspace), "receipt": str(output / "not_downloaded.jsonl")}
    config = NS(device=None, model_loader_extra_config=extra)
    revision = json.loads((root / "qwen3.manifest.json").read_text())["revision"]
    model_config = NS(model=str(model_dir), revision=revision, dtype=torch.bfloat16, quantization=None)
    vllm_config = NS(device_config=NS(device="cpu"), load_config=config)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_experts = torch.nn.Module()
            self.routed_experts.w13_weight = torch.nn.Parameter(torch.empty(2, dtype=torch.bfloat16), requires_grad=False)
            self.routed_experts.w2_weight = torch.nn.Parameter(torch.empty(2, dtype=torch.bfloat16), requires_grad=False)
            self.routed_experts.quant_method = NS(process_weights_after_loading=lambda layer: None)
        def load_weights(self, weights):
            events.append("model.load_weights")
            loaded = set()
            for name, tensor in weights:
                getattr(self.routed_experts, name).copy_(tensor)
                loaded.add(name)
            return {"w13_weight"} if control["missing"] else loaded
        def eval(self):
            events.append("eval")
            return super().eval()
    def initialize_model(**kwargs):
        assert kwargs["prefix"] == "fixture" and kwargs["model_config"] is model_config
        events.append("initialize")
        return Tiny()
    env["initialize_model"] = initialize_model
    env["process_weights_after_loading"] = lambda *args: events.append("postprocess")
    def transport_spy(model, manifest, index, work, receipt, *, validate_loaded, fetch):
        assert [manifest, index, work, receipt] == [Path(extra[k]).resolve() for k in ("manifest", "index", "workspace", "receipt")]
        assert fetch.keywords == {"timeout": 120}
        events.append("serial")
        loaded = model.load_weights((name, torch.ones(2, dtype=torch.bfloat16)) for name in ("w13_weight", "w2_weight"))
        validate_loaded(model, loaded)
        events.append("target_coverage")
        return loaded
    adapter.load_serial = transport_spy  # transport has its own real safetensors CPU fixture
    loader = adapter.QwenSerialLoader(config)
    loader.download_model(model_config)
    assert not events and not list(workspace.iterdir()) and not Path(extra["receipt"]).exists()
    result = loader.load_model(vllm_config, model_config, prefix="fixture")
    expected = ["initialize", "serial", "model.load_weights", "target_coverage", "postprocess", "eval"]
    assert events == expected and not result.training and torch.equal(result.routed_experts.w13_weight, result.routed_experts.w2_weight)
    successful_order = list(events)
    events.clear()
    control["missing"] = True
    extra["receipt"] = str(output / "missing_target.jsonl")
    loader = adapter.QwenSerialLoader(config)
    try:
        loader.load_model(vllm_config, model_config, prefix="fixture")
    except ValueError as error:
        assert "not initialized" in str(error)
    else:
        raise AssertionError("target coverage failure accepted")
    assert events == ["initialize", "serial", "model.load_weights"]
    bad_index = output / "bad_index.json"
    bad_index.write_text("{}")
    try:
        adapter.QwenSerialLoader(NS(model_loader_extra_config=dict(extra, index=str(bad_index))))
    except ValueError as error:
        assert "index SHA256" in str(error)
    else:
        raise AssertionError("unfrozen index accepted")
    report = dict(status="PASS_CPU_ADAPTER_CONTRACT_ONLY", torch=torch.__version__,
                  load_format=adapter.LOAD_FORMAT, source_sha256=source_hashes, order=successful_order,
                  checks=["installed registration", "inherited Base.load_model", "metadata validation without fetch",
                          "one serial call", "one model.load_weights", "MoERunner target aliases",
                          "postprocess auto-fill cannot hide missing expert target", "HTTP timeout=120",
                          "native postprocess then eval", "index mismatch rejected"],
                  fixture_dependencies=["vllm import modules", "CPU model/platform", "serial transport spy"],
                  real_vllm_gpu_load=False)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
