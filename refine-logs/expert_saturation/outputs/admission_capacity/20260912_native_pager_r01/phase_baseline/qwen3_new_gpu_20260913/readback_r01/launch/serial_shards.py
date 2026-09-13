"""CPU shard stream; caller installs WiSP before constructing the vLLM model.

Call load_serial once from BaseModelLoader.load_weights, before postprocessing.
Only this invocation's private downloaded copies are removed; failures retain
the incomplete shard. Each yielded tensor owns a clone, never mmap storage.
"""
import hashlib, json, re, tempfile, time
from pathlib import Path
from urllib.request import urlopen
from safetensors import safe_open

def require(ok, message):
    if not ok:
        raise ValueError(message)

def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result

def load_serial(model, manifest_path, index_path, workspace, receipt_path, *, validate_loaded, fetch=urlopen):
    """validate_loaded must reject missing target parameters; returns loaded set."""
    manifest_bytes, index_bytes = Path(manifest_path).read_bytes(), Path(index_path).read_bytes()
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique)
    require(re.fullmatch(r"[\w.-]+/[\w.-]+", manifest["repository"]), "repository")
    require(re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]), "freeze commit revision")
    require(hashlib.sha256(index_bytes).hexdigest() == manifest["index_sha256"], "index SHA256")
    weight_map = json.loads(index_bytes, object_pairs_hook=unique)["weight_map"]
    shards = manifest["shards"]
    require(weight_map and set(weight_map.values()) == set(shards), "index/shard coverage")
    for name, spec in shards.items():
        require(re.fullmatch(r"[\w.-]+\.safetensors", name), "shard basename")
        require(type(spec["bytes"]) is int and spec["bytes"] > 0, "shard size")
        require(re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]), "shard SHA256")
    with Path(receipt_path).open("x", encoding="utf-8") as log:
        owned = Path(tempfile.mkdtemp(prefix="serial-shards-", dir=workspace))
        def record(event, **data):
            log.write(json.dumps(dict(event=event, unix_s=time.time(), **data)) + "\n")
            log.flush()
        record("START", owned_directory=str(owned), revision=manifest["revision"],
               manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
               index_sha256=manifest["index_sha256"])
        seen, completed = set(), False
        def stream():
            nonlocal completed
            for filename in sorted(shards):
                spec, path = shards[filename], owned / filename
                digest, count, consumed = hashlib.sha256(), 0, False
                url = "https://huggingface.co/{}/resolve/{}/{}".format(
                    manifest["repository"], manifest["revision"], filename)
                record("DOWNLOAD", shard=filename, expected_bytes=spec["bytes"])
                try:
                    with path.open("xb") as sink, fetch(url) as source:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            count += len(chunk)
                            require(count <= spec["bytes"], "download exceeds frozen size")
                            sink.write(chunk)
                            digest.update(chunk)
                    actual_hash = digest.hexdigest()
                    record("DOWNLOADED", shard=filename, bytes=count, sha256=actual_hash)
                    require(count == spec["bytes"] and actual_hash == spec["sha256"], "shard hash/bytes")
                    keys = {key for key, shard in weight_map.items() if shard == filename}
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        require(set(handle.keys()) == keys, "shard keys differ from index")
                        for key in sorted(keys):
                            require(key not in seen, "duplicate source tensor: " + key)
                            # No mapped tensor escapes, even if downstream retains a view.
                            yield key, handle.get_tensor(key).clone()
                            seen.add(key)
                    consumed = True
                finally:
                    # Closing the original generator first exits safe_open above.
                    if consumed:
                        path.unlink()
                        record("CONSUMED_REMOVED", shard=filename, tensors=len(keys),
                               bytes=count, sha256=digest.hexdigest())
                    elif path.exists():
                        record("INCOMPLETE_RETAINED", shard=filename, bytes=path.stat().st_size,
                               received_sha256=digest.hexdigest())
            completed = True
        generator = stream()
        try:
            try:
                loaded = model.load_weights(generator)
                require(completed and seen == set(weight_map), "consumer did not exhaust every source tensor")
                require(isinstance(loaded, set), "load_weights must report loaded target names")
                validate_loaded(model, loaded)
                owned.rmdir()
            finally:
                generator.close()
        except BaseException as error:
            record("FAILED", error=repr(error), consumed_tensors=len(seen))
            raise
        record("COMPLETE", shards=len(shards), source_tensors=len(seen), target_names=len(loaded))
        return loaded
