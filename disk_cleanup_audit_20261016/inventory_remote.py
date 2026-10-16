import os, glob, json, stat, hashlib

roots = glob.glob('/root/moe-*') + glob.glob('/tmp/moe-*') + glob.glob('/root/autodl-tmp/moe-*') + ['/root/F_output_isolation']
skip = {'hf', 'model', 'wheels', 'uv-cache', 'cache', 'compile_cache', 'vendor', '.git', '__pycache__', 'runtime-overlay', 'pkg'}
rows = []
for base in roots:
    if not os.path.isdir(base) or os.path.islink(base):
        continue
    for top, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip and not os.path.islink(os.path.join(top, d))]
        for name in files:
            if not name.endswith(('.json', '.jsonl', '.log', '.tar.gz', '.tgz', '.csv', '.npz', '.npy')):
                continue
            p = os.path.join(top, name)
            s = os.lstat(p)
            if not stat.S_ISREG(s.st_mode) or s.st_size < 1048576:
                continue
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(4*1024*1024), b''):
                    h.update(b)
            after = os.stat(p)
            if (s.st_size, s.st_mtime_ns, s.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                continue
            rows.append({'path': p, 'name': name, 'size': s.st_size, 'allocated': s.st_blocks*512, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino, 'device': s.st_dev, 'nlink': s.st_nlink, 'sha256': h.hexdigest()})
print(json.dumps(rows))
