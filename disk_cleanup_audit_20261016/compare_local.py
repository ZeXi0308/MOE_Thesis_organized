import os, json, hashlib, collections
from pathlib import Path
base = Path(__file__).resolve().parent
remote = json.loads((base/'remote_inventory.json').read_text())
local = json.loads((base/'local_inventory.json').read_text())
sizes = {r['size'] for r in remote}
index = collections.defaultdict(list)
count = 0
for row in local:
    if row['size'] not in sizes:
        continue
    p = row['path']
    before = os.stat(p)
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    after = os.stat(p)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        continue
    index[(before.st_size, h.hexdigest())].append(p)
    count += 1
matches = []
unmatched = []
for row in remote:
    backups = index.get((row['size'], row['sha256']))
    if backups:
        matches.append(dict(row, local_backups=backups))
    else:
        unmatched.append(row)
(base/'verified_matches.json').write_text(json.dumps(matches, ensure_ascii=False, indent=2))
(base/'unmatched_preserved.json').write_text(json.dumps(unmatched, ensure_ascii=False, indent=2))
groups = collections.Counter()
for row in matches:
    p = row['path']
    group = '/'.join(p.split('/')[:4]) if p.startswith('/root/autodl-tmp/') else '/'.join(p.split('/')[:3])
    groups[group] += row['allocated']
print(json.dumps({'local_hashed':count,'remote_candidates':len(remote),'matched':len(matches),'unmatched_preserved':len(unmatched),'matched_GiB':sum(r['allocated'] for r in matches)/2**30,'groups_GiB':{k:round(v/2**30,3) for k,v in groups.most_common()}}, ensure_ascii=False, indent=2))
