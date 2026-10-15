import concurrent.futures
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
PYTHON=Path('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
IDS={
 'sdp':'10.1016/j.knosys.2026.116227',
 'ggat':'10.1016/j.patcog.2026.113592',
 'clorf':'arxiv:2405.17818',
 'hysure':'arxiv:1411.4005',
 'stereo':'arxiv:1804.05307',
 'local_unmix':'10.1109/TGRS.2017.2718728',
 'dtv':'arxiv:1710.05705',
 'sa_rfm':'https://raw.githubusercontent.com/mlresearch/v306/main/assets/zhang26df/zhang26df.pdf'
}

def fetch(item):
    name,identifier=item
    try:
        result=subprocess.run([str(PYTHON),str(ROOT/'fetch_sources.py'),identifier],capture_output=True,
                              encoding='utf-8',errors='replace',timeout=360)
        (ROOT/f'fetch-{name}.json').write_text(result.stdout,encoding='utf-8')
        (ROOT/f'fetch-{name}.stderr.txt').write_text(result.stderr,encoding='utf-8')
        data=json.loads(result.stdout)
        record={k:data.get(k) for k in ['status','parsed_path','parsed_lines','parsed_chars','sentinel','source','read_plan','tried']}
        print(json.dumps({'name':name,**record},ensure_ascii=False),flush=True)
        return {'name':name,'identifier':identifier,**record}
    except Exception as error:
        print(json.dumps({'name':name,'error':str(error)}),flush=True)
        return {'name':name,'identifier':identifier,'status':'failed','error':str(error)}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    records=list(executor.map(fetch,IDS.items()))
(ROOT/'retrieval-manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
