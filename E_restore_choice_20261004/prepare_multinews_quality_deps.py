"""Fetch only pinned small pure-Python wheels into E's isolated /tmp, without pip/install."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def keep(path, data, expected):
    if sha(data) != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite different file: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=Path('/private/tmp/E-multinews-quality-20261007'))
    args = parser.parse_args()
    target = args.target.resolve()
    if not target.is_relative_to(Path('/tmp').resolve()) or not target.name.startswith('E-multinews-quality-'):
        raise ValueError('Use an E-multinews-quality-* directory under /tmp')
    receipt = json.loads((Path(__file__).resolve().parent/'quality_tools/rouge_dependency_receipt.json').read_text())
    for package in receipt['packages']:
        path = target/package['wheel_filename']
        blob = path.read_bytes() if path.is_file() else urllib.request.urlopen(package['wheel_url'], timeout=30).read()
        keep(path, blob, package['wheel_sha256'])
        with zipfile.ZipFile(io.BytesIO(blob)) as wheel:
            names = [n for n in wheel.namelist() if not n.endswith('/')]
            if set(names) != set(package['files']):
                raise ValueError('Wheel member set differs from frozen receipt')
            for name in names:
                posix = PurePosixPath(name)
                if posix.is_absolute() or '..' in posix.parts:
                    raise ValueError('Unsafe wheel member')
                keep(target/'site'/name, wheel.read(name), package['files'][name])
    print(json.dumps(dict(status='ISOLATED_DEPENDENCIES_READY', site=str(target/'site'),
        packages={p['name']:p['version'] for p in receipt['packages']}, no_environment_install=True)))


if __name__ == '__main__':
    main()
