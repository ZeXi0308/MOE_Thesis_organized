"""Tiny real BF16 safetensors: retained views, failure retention, frozen identity."""
import argparse, hashlib, io, json, weakref
from contextlib import contextmanager
from pathlib import Path
import torch
from safetensors.torch import save_file
import serial_shards as serial

def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source, scratch = output / "source", output / "scratch"
    source.mkdir()
    scratch.mkdir()
    arrays = {"a": torch.arange(6).reshape(2, 3).to(torch.bfloat16), "b": torch.ones(3, dtype=torch.bfloat16)}
    index, specs = {"weight_map": {}}, {}
    for key, tensor in arrays.items():
        name = key + ".safetensors"
        save_file({key: tensor}, str(source / name))
        data = (source / name).read_bytes()
        specs[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        index["weight_map"][key] = name
    index_path, manifest_path = output / "index.json", output / "manifest.json"
    index_path.write_text(json.dumps(index))
    manifest = dict(repository="fixture/model", revision="a" * 40, shards=specs,
                    index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    mapped_refs, opened = [], []
    original = serial.safe_open
    @contextmanager
    def tracked_open(*args, **kwargs):
        with original(*args, **kwargs) as handle:
            opened.append(args[0])
            class Tracked:
                keys = handle.keys
                def get_tensor(self, key):
                    tensor = handle.get_tensor(key)
                    mapped_refs.append(weakref.ref(tensor))
                    return tensor
            try:
                yield Tracked()
            finally:
                opened.pop()
    serial.safe_open = tracked_open
    def fetch(url):
        assert not opened and all(ref() is None for ref in mapped_refs)
        assert len(list(scratch.rglob("*.safetensors"))) == 1  # new empty owned file only
        return io.BytesIO((source / url.rsplit("/", 1)[-1]).read_bytes())
    class Model:
        calls = 0
        def load_weights(self, weights):
            self.calls += 1
            self.values, self.retained_views = {}, []
            for key, tensor in weights:
                self.retained_views.append(tensor.view(-1))
                self.values[key] = torch.empty_like(tensor).copy_(tensor)
            return set(self.values)
    def validate(model, loaded):
        serial.require(loaded == set(arrays), "target coverage")
    model = Model()
    args = (model, manifest_path, index_path, scratch, output / "success.jsonl")
    serial.load_serial(*args, validate_loaded=validate, fetch=fetch)
    assert model.calls == 1 and not opened and not list(scratch.iterdir())
    assert all(torch.equal(model.values[k], v) for k, v in arrays.items())
    assert all(ref() is None for ref in mapped_refs) and len(model.retained_views) == 2
    class Early(Model):
        def load_weights(self, weights):
            self.retained = next(weights)
            return {self.retained[0]}
    try:
        serial.load_serial(Early(), manifest_path, index_path, scratch, output / "early.jsonl",
                           validate_loaded=validate, fetch=fetch)
    except ValueError as error:
        assert "did not exhaust" in str(error)
    else:
        raise AssertionError("early return accepted")
    assert not opened and all(ref() is None for ref in mapped_refs)
    assert len(list(scratch.rglob("*.safetensors"))) == 1
    scratch = output / "corrupt_scratch"
    scratch.mkdir()
    corrupt = Model()
    def corrupt_fetch(url):
        data = fetch(url).read()
        return io.BytesIO(data[:-1] + bytes([data[-1] ^ 1]))
    try:
        serial.load_serial(corrupt, manifest_path, index_path, scratch, output / "corrupt.jsonl",
                           validate_loaded=validate, fetch=corrupt_fetch)
    except ValueError as error:
        assert "hash/bytes" in str(error) and not corrupt.values and not opened
    else:
        raise AssertionError("corrupt shard accepted")
    assert all(hashlib.sha256((source / n).read_bytes()).hexdigest() == s["sha256"] for n, s in specs.items())
    (output / "result.json").write_text(json.dumps(dict(status="PASS_CPU_ONLY", torch=torch.__version__,
        checks=["two real BF16 shards", "one load_weights call", "copy results", "retained clone views",
                "mapped tensors released", "close before next shard", "early return rejected and retained",
                "corrupt shard rejected before yield", "source files unchanged"]), indent=2) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
