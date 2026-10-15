"""Read-only retrieval of public author sources for the GEWDiff analysis."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'author_code'
DEST.mkdir(exist_ok=True)

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Codex-paper-analysis'})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()

repo = 'zhu-xlab/GEWDiff'
info = json.loads(get(f'https://api.github.com/repos/{repo}/commits/main'))
commit = info['sha']
files = ['model/RWT.py', 'model/edm.py', 'model/unet3d.py', 'train.py',
         'data/dataset.py', 'data/process_mask_edge.ipynb', 'test.py',
         'test_wdc.py', 'test_enmap.py', 'run.sh', 'requirements.txt', 'README.md']

def fetch(name):
    url = f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
    try:
        raw = get(url)
        path = DEST / (name.replace('/', '__') + '.txt')
        path.write_bytes(raw)
        return {'file': name, 'commit': commit, 'url': url,
                'browser_url': f'https://github.com/{repo}/blob/{commit}/{name}',
                'local_path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'status': 'ok'}
    except Exception as error:
        return {'file': name, 'status': 'failed', 'error': str(error)}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(fetch, files))
(ROOT / 'source-manifest.json').write_text(
    json.dumps({'repo': repo, 'commit': commit, 'retrieved_date': '2026-10-07',
                'execution': 'source reading only, author code not executed',
                'files': records}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'commit': commit, 'files': [{k: r.get(k) for k in
      ['file', 'status', 'bytes', 'error']} for r in records]}, indent=2))
