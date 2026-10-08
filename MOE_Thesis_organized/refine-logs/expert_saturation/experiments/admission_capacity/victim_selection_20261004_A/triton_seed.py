"""Copy one frozen union of completed Triton caches into private arm caches.

Only __grp__*.json child_paths are normalized/relocated. Kernel metadata,
including extern_libs, and all compiled file bytes remain unchanged.
"""
import hashlib
import json
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _root(path):
    path = Path(path)
    _require(not path.is_symlink() and path.is_dir(), f'Not a regular cache directory: {path}')
    return path.resolve()


def _scan(path):
    root = _root(path)
    files, normalized, groups = {}, {}, {}
    keys = sorted(root.iterdir())
    _require(keys, f'Empty seed/source cache: {root}')
    for key in keys:
        _require(not key.is_symlink() and key.is_dir(), f'Invalid cache key directory: {key}')
        children = sorted(key.iterdir())
        _require(children, f'Empty cache key: {key}')
        for file in children:
            _require(not file.is_symlink() and file.is_file(), f'Invalid cache file: {file}')
            rel = str(file.relative_to(root))
            payload = file.read_bytes()
            files[rel] = normalized[rel] = payload
            if not (file.name.startswith('__grp__') and file.suffix == '.json'):
                continue
            group = json.loads(payload)
            _require(isinstance(group, dict) and set(group) == {'child_paths'} and
                     isinstance(group['child_paths'], dict) and group['child_paths'],
                     f'Unexpected Triton group metadata: {file}')
            relative_children = {}
            for name, target in group['child_paths'].items():
                _require(isinstance(name, str) and name not in ('', '.', '..') and
                         Path(name).name == name and isinstance(target, str),
                         f'Invalid child path in {file}')
                relative = str(Path(key.name) / name)
                expected = root / relative
                _require(target in (relative, str(expected)), f'Child escapes its own cache key: {target}')
                _require(not expected.is_symlink() and expected.is_file(), f'Missing group child: {expected}')
                relative_children[name] = relative
            groups[rel] = {'child_paths': relative_children}
            normalized[rel] = _json_bytes(groups[rel])
    return files, normalized, groups, len(keys)


def _summary(files, normalized, groups, key_dirs):
    inventory = {rel: hashlib.sha256(payload).hexdigest() for rel, payload in normalized.items()}
    return dict(files=len(files), bytes=sum(map(len, files.values())),
                normalized_bytes=sum(map(len, normalized.values())),
                normalized_sha256=hashlib.sha256(_json_bytes(inventory)).hexdigest(),
                key_dirs=key_dirs, groups=len(groups),
                child_paths=sum(len(group['child_paths']) for group in groups.values()))


def inspect_seed(seed):
    """Validate files/children and hash the normalized inventory, without writes.

Also accepts a relocated private cache: its normalized digest is identical.
The digest hashes compact sorted JSON {relative_path: SHA256(normalized_file)}.
"""
    return _summary(*_scan(seed))


def _write_files(files, dest):
    for rel, payload in sorted(files.items()):
        file = dest / rel
        file.parent.mkdir(exist_ok=True)
        # New regular files, never symlinks/hardlinks or replacement writes.
        with file.open('xb') as stream:
            stream.write(payload)


def build(source_cache_dirs, dest):
    """Create a new relative-path seed; first source wins each entire key.

The caller supplies the frozen source order. No source files are modified.
"""
    dest = Path(dest)
    _require(not dest.exists() and not dest.is_symlink(), f'Seed destination already exists: {dest}')
    destination = dest.resolve()
    chosen, payloads, source_key_counts = set(), {}, {}
    for source in source_cache_dirs:
        source = _root(source)
        _require(not destination.is_relative_to(source), 'Seed destination must be outside source caches')
        _, normalized, _, _ = _scan(source)
        new_keys = {Path(rel).parts[0] for rel in normalized} - chosen
        for rel, payload in normalized.items():
            if Path(rel).parts[0] in new_keys:
                payloads[rel] = payload
        chosen.update(new_keys)
        source_key_counts[str(source)] = source_key_counts.get(str(source), 0) + len(new_keys)
    _require(chosen, 'No source keys supplied')
    destination.mkdir(exist_ok=False)
    _write_files(payloads, destination)
    result = inspect_seed(destination)
    result['source_key_counts'] = source_key_counts
    return result


def install(seed, dest, expected_sha256):
    """Install independent files in an existing empty private cache directory."""
    seed, dest = _root(seed), _root(dest)
    _require(not any(dest.iterdir()), f'Private Triton cache is not empty: {dest}')
    files, normalized, groups, key_dirs = _scan(seed)
    source = _summary(files, normalized, groups, key_dirs)
    _require(source['normalized_sha256'] == expected_sha256, 'Frozen Triton seed digest differs')
    # A built seed must itself use the canonical relative representation.
    _require(all(files[rel] == normalized[rel] for rel in groups), 'Seed group metadata is not canonical relative JSON')
    payloads = dict(files)
    for rel, group in groups.items():
        payloads[rel] = _json_bytes({'child_paths': {
            name: str(dest / child) for name, child in group['child_paths'].items()}})
    _write_files(payloads, dest)
    result = inspect_seed(dest)
    _require(result['normalized_sha256'] == expected_sha256, 'Installed Triton cache differs from seed')
    result.update(relocated_groups=len(groups), source_key_counts={str(seed): key_dirs})
    return result
