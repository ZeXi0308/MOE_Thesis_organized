import os, sys, json, hashlib, glob, stat

def emit(event, **kw):
    print(json.dumps(dict(event=event, **kw), ensure_ascii=False), flush=True)

def disks():
    result = {}
    for p in ['/', '/root/autodl-tmp']:
        s = os.statvfs(p)
        result[p] = {'available_bytes':s.f_bavail*s.f_frsize, 'total_bytes':s.f_blocks*s.f_frsize}
    return result

emit('before', disks=disks())
opened = set()
for proc in glob.glob('/proc/[0-9]*'):
    for fd in glob.glob(proc+'/fd/*')+[proc+'/cwd']:
        try:
            opened.add(os.readlink(fd))
        except OSError:
            pass
    try:
        with open(proc+'/maps') as f:
            for line in f:
                parts=line.rstrip().split(None,5)
                if len(parts)==6 and parts[5].startswith('/'):
                    opened.add(parts[5])
    except OSError:
        pass

removed = 0
allocated = 0
for row in VERIFIED_MATCHES:
    p = row['path']
    try:
        allowed = (p.startswith('/root/moe-') or p.startswith('/tmp/moe-') or p.startswith('/root/autodl-tmp/moe-') or p.startswith('/root/F_output_isolation/'))
        excluded = set(p.split('/')) & {'hf','model','wheels','uv-cache','cache','compile_cache','vendor','.git','__pycache__','runtime-overlay','pkg'}
        if not allowed or excluded or os.path.realpath(p)!=p or not row['local_backups']:
            emit('skip', path=p, reason='scope guard')
            continue
        if p in opened:
            emit('skip', path=p, reason='open by process')
            continue
        before = os.lstat(p)
        if not stat.S_ISREG(before.st_mode) or (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)!=(row['size'],row['mtime_ns'],row['inode'],row['device']):
            emit('skip', path=p, reason='changed since audit')
            continue
        h=hashlib.sha256()
        with open(p,'rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):
                h.update(b)
        after=os.lstat(p)
        if h.hexdigest()!=row['sha256'] or (after.st_size,after.st_mtime_ns,after.st_ino)!=(before.st_size,before.st_mtime_ns,before.st_ino):
            emit('skip', path=p, reason='hash or stat changed')
            continue
        os.unlink(p)
        removed+=1
        allocated+=before.st_blocks*512
        emit('deleted', path=p, sha256=row['sha256'], local_backups=row['local_backups'], allocated=before.st_blocks*512)
    except Exception as e:
        emit('error', path=p, error=str(e))
emit('after', disks=disks(), deleted_files=removed, deleted_allocated_bytes=allocated)
