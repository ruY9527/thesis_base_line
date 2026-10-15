"""Download public author source snapshots for static comparison only."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent

def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'Codex-research-source-reader'})
    with urllib.request.urlopen(request,timeout=35) as response:
        return response.read()

def collect(repo):
    try:
        info=json.loads(get(f'https://api.github.com/repos/{repo}'))
        commit=json.loads(get(f'https://api.github.com/repos/{repo}/commits/{info["default_branch"]}'))['sha']
        tree=json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))['tree']
        dest=ROOT/'author_code'/repo.split('/')[-1]
        dest.mkdir(parents=True,exist_ok=True)
        (dest/'tree.json').write_text(json.dumps(tree,ensure_ascii=False,indent=2),encoding='utf-8')
        paths=[item['path'] for item in tree if item['type']=='blob' and
               (item['path'].endswith('.py') or item['path'].lower().endswith('readme.md'))]
        records=[]
        for name in paths:
            url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
            try:
                raw=get(url)
                path=dest/(name.replace('/','__')+'.txt')
                path.write_bytes(raw)
                records.append(dict(file=name,url=url,local_path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                                    bytes=len(raw),status='ok'))
            except Exception as error:
                records.append(dict(file=name,status='failed',error=str(error)))
        result=dict(repo=repo,commit=commit,execution='static reading only; author code not executed',
                    files=records,paper_paths=[t['path'] for t in tree if t['path'].lower().endswith('.pdf')])
        (dest/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'repo':repo,'commit':commit,'files':[dict(file=r['file'],status=r['status'],bytes=r.get('bytes')) for r in records],
                          'paper_paths':result['paper_paths']},ensure_ascii=False),flush=True)
    except Exception as error:
        print(json.dumps({'repo':repo,'error':str(error)}),flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    list(executor.map(collect,['LRuiRui517/SDP-Net','wangting1907/CLoRF-Fusion']))
