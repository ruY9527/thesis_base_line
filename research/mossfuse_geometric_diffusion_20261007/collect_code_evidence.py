"""Cache immutable public author code for analysis, without changing active models."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = 'd6c7eb4a487046aecae5265cabf81b0c23506465'
REPO = 'dusongcheng/MossFuse'
destination = ROOT / 'author_code'
destination.mkdir(exist_ok=True)
entries = []
for name in ['MossFuseNet.py', 'net.py', 'train.py', 'utils.py', 'README.md', 'degradation_net.py', 'dataset.py']:
    old = Path(os.environ['TEMP']) / 'd2_code_plan_20261006' / 'MossFuse' / (name + '.txt')
    url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{name}'
    data = old.read_bytes() if old.exists() else urllib.request.urlopen(url, timeout=45).read()
    path = destination / (name + '.txt')
    path.write_bytes(data)
    entries.append({'name': name, 'url': url, 'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'retrieval': 'existing immutable cache' if old.exists() else 'public author repository'})
active = {}
for name in ['model.py', 'train_cave_M.py', 'train_dataloader.py', 'utils.py']:
    path = ROOT.parents[1] / name
    active[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT / 'source-manifest.json').write_text(json.dumps({'repository': REPO, 'commit': COMMIT, 'author_code': entries, 'active_baseline_hashes_before': active}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'cached_files': len(entries), 'active_baseline_files_recorded': len(active)}))
