"""vLLM 0.26 adapter for frozen Qwen BF16 shard loading; no custom load_model."""
import hashlib
import json
from functools import partial
from pathlib import Path
import re
from urllib.request import urlopen
import torch
import vllm
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from serial_shards import load_serial, require, unique

LOAD_FORMAT = "qwen_bf16_serial_v026"


@register_model_loader(LOAD_FORMAT)
class QwenSerialLoader(BaseModelLoader):
    def __init__(self, load_config):
        super().__init__(load_config)
        require(vllm.__version__ == "0.26.0", "loader requires vLLM 0.26.0")
        extra = load_config.model_loader_extra_config
        require(isinstance(extra, dict) and set(extra) == {"manifest", "index", "workspace", "receipt"},
                "loader extra_config requires exactly manifest/index/workspace/receipt")
        require(all(isinstance(value, str) and value for value in extra.values()), "loader paths must be strings")
        self.paths = {key: Path(value).expanduser().resolve() for key, value in extra.items()}
        self.targets_path = self.paths["receipt"].with_name(self.paths["receipt"].name + ".targets.json")
        self._metadata()

    def _metadata(self):
        manifest = json.loads(self.paths["manifest"].read_bytes(), object_pairs_hook=unique)
        require(manifest["repository"] == "Qwen/Qwen3-30B-A3B", "unexpected model repository")
        require(re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]), "freeze commit revision")
        require(re.fullmatch(r"[0-9a-f]{64}", manifest["config_sha256"]), "config SHA256")
        index_bytes = self.paths["index"].read_bytes()
        require(hashlib.sha256(index_bytes).hexdigest() == manifest["index_sha256"], "index SHA256")
        weights = json.loads(index_bytes, object_pairs_hook=unique)["weight_map"]
        require(weights and all(isinstance(key, str) and key for key in weights), "source tensor names")
        require(set(weights.values()) == set(manifest["shards"]), "index/shard coverage")
        for name, spec in manifest["shards"].items():
            require(re.fullmatch(r"[\w.-]+\.safetensors", name), "shard basename")
            require(type(spec["bytes"]) is int and spec["bytes"] > 0, "shard bytes")
            require(re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]), "shard SHA256")
        require(self.paths["workspace"].is_dir(), "workspace must already exist")
        require(self.paths["receipt"].parent.is_dir() and not self.paths["receipt"].exists(), "receipt must be new")
        require(not self.targets_path.exists(), "target coverage receipt must be new")
        return manifest

    def download_model(self, model_config):
        """Validate only local frozen metadata; intentionally fetch no weights."""
        manifest = self._metadata()
        model_dir = Path(model_config.model)
        require(model_dir.is_dir(), "use a local frozen config/tokenizer directory")
        require(hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest()
                == manifest["config_sha256"], "model config differs from frozen revision")
        require(model_config.revision in (None, manifest["revision"]), "model revision mismatch")
        require(model_config.dtype == torch.bfloat16 and model_config.quantization is None, "BF16 unquantized only")
        model_weights = getattr(model_config, "model_weights", None)
        if model_weights == "":  # vLLM 0.26 local-model default: no override
            model_weights = model_config.model
        require(model_weights is None or (
            isinstance(model_weights, (str, Path)) and bool(str(model_weights))
            and Path(model_weights).expanduser().resolve() == model_dir.resolve()),
            "model_weights must be None or the validated local model directory")

    def load_weights(self, model, model_config):
        self.download_model(model_config)
        load_serial(model, self.paths["manifest"], self.paths["index"],
                    self.paths["workspace"], self.paths["receipt"],
                    validate_loaded=self._validate_targets, fetch=partial(urlopen, timeout=120))

    def _validate_targets(self, model, loaded):
        # Native MoERunner reports w13/w2 without its registered routed_experts prefix.
        aliases = {}
        for prefix, module in model.named_modules():
            experts = getattr(module, "routed_experts", None)
            if experts is not None:
                stem = prefix + "." if prefix else ""
                for leaf, _ in experts.named_parameters(recurse=False):
                    if leaf in {"w13_weight", "w2_weight"}:
                        aliases[stem + leaf] = stem + "routed_experts." + leaf
        raw, expected = set(loaded), {name for name, _ in model.named_parameters()}
        normalized = {aliases.get(name, name) for name in raw}
        missing = sorted(expected - normalized)
        if not missing:
            DefaultModelLoader.track_weights_loading(self, model, set(raw))
        proof = dict(status="FAIL" if missing else "PASS", raw_returned_names=sorted(raw),
                     raw_names_sha256=hashlib.sha256(json.dumps(sorted(raw), separators=(",", ":")).encode()).hexdigest(),
                     expected_named_parameters=sorted(expected), aliases=aliases,
                     normalized_returned_names=sorted(normalized), missing=missing)
        with self.targets_path.open("x", encoding="utf-8") as output:
            json.dump(proof, output, indent=2)
            output.write("\n")
        require(not missing, "BF16 target parameters not initialized: " + ", ".join(missing))
