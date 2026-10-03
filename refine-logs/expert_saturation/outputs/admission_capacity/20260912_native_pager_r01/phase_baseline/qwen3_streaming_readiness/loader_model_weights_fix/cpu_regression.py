"""ModelConfig's installed empty default and local branch, without vLLM/GPU init."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    fix = Path(__file__).resolve().parent
    ready = fix.parent
    sys.path.insert(0, str(ready))
    from serial_shards import require, unique
    actual = ready / "installed_sources/config/model.py"
    actual_sha = hashlib.sha256(actual.read_bytes()).hexdigest()
    require(actual_sha == "7d52e060db54067a21591882bd6e3c2dd2486ac9041dde1117f97023e71bf89f", "actual config source SHA")
    def extract(path, symbol, members, env):
        node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == symbol)
        node.body = [n for n in node.body if getattr(n, "name", None) in members
                     or isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) in members]
        node.bases, node.decorator_list = [], []
        tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), env)
        return env[symbol]
    model_dir = output / "model_metadata"
    model_dir.mkdir()
    (model_dir / "config.json").write_bytes((ready / "qwen3.config.json").read_bytes())
    local_checks = []
    def is_runai_obj_uri(value):
        assert Path(value).resolve() == model_dir.resolve()
        local_checks.append(value)
        return False  # Explicit fixture: both inputs are the existing local directory.
    Config = extract(actual, "ModelConfig", {"model_weights", "maybe_pull_model_tokenizer_for_runai"},
                     dict(is_runai_obj_uri=is_runai_obj_uri))
    config = Config()
    config.model, config.tokenizer = str(model_dir), str(model_dir)
    assert config.model_weights == ""
    config.maybe_pull_model_tokenizer_for_runai(config.model, config.tokenizer)
    assert config.model_weights == "" and len(local_checks) == 2
    config.dtype, config.quantization, config.revision = torch.bfloat16, None, None
    paths = dict(manifest=ready / "qwen3.manifest.json", index=ready / "qwen3.index.json",
                 workspace=output, receipt=output / "no_download.jsonl")
    def loader(path):
        cls = extract(path, "QwenSerialLoader", {"_metadata", "download_model"},
                      dict(Path=Path, hashlib=hashlib, json=json, re=__import__("re"),
                           torch=torch, require=require, unique=unique))
        instance = cls()
        instance.paths, instance.targets_path = paths, output / "no_targets.json"
        return instance
    old_path = ready / "qwen_serial_loader.py"
    old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
    try:
        loader(old_path).download_model(config)
    except ValueError as error:
        assert str(error) == "model_weights override unsupported"
    else:
        raise AssertionError("r01 default rejection was not reproduced")
    fixed = loader(fix / "qwen_serial_loader.py")
    passed = []
    for label, value in (("None", None), ("native_empty_default", ""), ("same_local", config.model)):
        config.model_weights = value
        fixed.download_model(config)
        passed.append(label)
    other = output / "other_model"
    other.mkdir()
    (other / "config.json").write_bytes((model_dir / "config.json").read_bytes())
    rejected = []
    for label, value in (("different_local_same_config", str(other)), ("external_uri", "s3://different/model")):
        config.model_weights = value
        try:
            fixed.download_model(config)
        except ValueError as error:
            assert "validated local model directory" in str(error)
            rejected.append(label)
        else:
            raise AssertionError("external model_weights accepted")
    assert not paths["receipt"].exists() and not fixed.targets_path.exists()
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == old_sha
    report = dict(status="PASS_CPU_DEFAULT_REGRESSION_ONLY", actual_config_sha256=actual_sha,
                  native_default="", native_local_branch_leaves_default=True,
                  reproduced_r01_rejection=True, passed=passed, rejected=rejected,
                  old_loader_sha256=old_sha, fixed_loader_sha256=hashlib.sha256((fix / "qwen_serial_loader.py").read_bytes()).hexdigest(),
                  transport_invoked=False, gpu_access=False,
                  scope="installed field/method AST; local URI predicate fixture; actual metadata validation")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
