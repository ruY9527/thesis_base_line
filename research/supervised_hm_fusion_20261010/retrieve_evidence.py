import argparse
import concurrent.futures
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UA = 'Mozilla/5.0 (academic research; public evidence retrieval)'

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read(), r.headers.get('Content-Type', ''), r.url

def retrieve(task):
    key, url, kind = task
    target = ROOT / 'sources' / key
    target.mkdir(parents=True, exist_ok=True)
    try:
        raw, ct, final = fetch(url)
        suffix = '.pdf' if raw.startswith(b'%PDF') else '.json' if 'json' in ct else '.html'
        path = target / ('source' + suffix)
        path.write_bytes(raw)
        result = {'id': key, 'requested_url': url, 'final_url': final, 'content_type': ct, 'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'status': 'ok', 'retrieved_on': '2026-10-10'}
        if kind == 'pdf' and not raw.startswith(b'%PDF'):
            result['status'] = 'unavailable_pdf'
            result['error'] = 'Publisher endpoint did not return a PDF; no full-text access confirmed.'
        if kind == 'crossref':
            records = json.loads(raw)['message']['items']
            selected = [{'title': x.get('title'), 'DOI': x.get('DOI'), 'container-title': x.get('container-title'), 'published': x.get('published'), 'published-online': x.get('published-online'), 'published-print': x.get('published-print'), 'URL': x.get('URL'), 'author': x.get('author'), 'abstract': x.get('abstract'), 'link': x.get('link')} for x in records]
            (target / 'metadata.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
            result['matches'] = [{k: x[k] for k in ('title', 'DOI', 'container-title', 'published-online')} for x in selected[:3]]
        if raw.startswith(b'%PDF'):
            try:
                import pymupdf
                doc = pymupdf.open(path)
                txt = '\n'.join(f'\n## PDF PAGE {i+1}\n' + p.get_text() for i, p in enumerate(doc))
                (target / 'fulltext.txt').write_text(txt, encoding='utf-8')
                result['pages'] = len(doc)
            except Exception as e:
                result['extraction_error'] = str(e)
        if key in ('eccv2026', 'cvpr2026', 'cvpr2025', 'iccv2025'):
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(raw, 'html.parser')
                hits = []
                for a in soup.find_all('a'):
                    title = a.get_text(' ', strip=True)
                    if re.search('hyperspectral|spectral image|spectral reconstruction', title, re.I):
                        hits.append({'title': title, 'url': urllib.parse.urljoin(final, a.get('href', ''))})
                (target / 'spectral_titles.json').write_text(json.dumps(hits, ensure_ascii=False, indent=2), encoding='utf-8')
                result['spectral_titles'] = hits
            except Exception as e:
                result['index_error'] = str(e)
        return result
    except Exception as e:
        return {'id': key, 'url': url, 'status': 'failed', 'error': str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True)
    args = ap.parse_args()
    tasks = json.loads(Path(args.batch).read_text(encoding='utf-8'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(retrieve, tasks))
    out = ROOT / (Path(args.batch).stem + '-results.json')
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    for r in results:
        print(json.dumps(r, ensure_ascii=False))

if __name__ == '__main__':
    main()
