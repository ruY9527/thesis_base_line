import csv, hashlib, json, re
from pathlib import Path
import pymupdf

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
EXPECTED={
 'model.py':'8F866B52264B94E3DFCD00137C0E451A338404398F78B35B1C6414A3DDD49B66',
 'train_cave_M.py':'C9D377055D65F706F80E92CDFD585C2AE9BE6B895970557F2FC0119401CC6EFC',
 'train_dataloader.py':'F7C3562A2F32D758FD9FB81574F172AC6ADC7BC07200A75EE39FB58767CE8578',
 'utils.py':'74198FDB7468CF441175C84AB8D6DA41D80AA56265EF697D5F16CEA23744D944',
 'calculate_metrics.py':'3831057F6EE92212D22DBCA275870FEF23F131A35A137153473DEDE0AEB96D94'
}
integrity=[]
for name,before in EXPECTED.items():
    after=hashlib.sha256((REPO/name).read_bytes()).hexdigest().upper()
    integrity.append(dict(file=str(REPO/name),before_sha256=before,after_sha256=after,unchanged=before==after))
(ROOT/'baseline-integrity.json').write_text(json.dumps({'as_of':'2026-10-10','initial_model_py_was_modified':True,'files':integrity,'all_unchanged':all(x['unchanged'] for x in integrity)},indent=2),encoding='utf8')
papers=json.loads((ROOT/'literature-matrix.json').read_text(encoding='utf8'))['papers']
rows=list(csv.DictReader((ROOT/'literature-matrix.csv').open(encoding='utf8')))
missing=[]
for path in ROOT.glob('*.md'):
    for url in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf8')):
        if re.match(r'^(https?://|#|C:/|D:/)',url): continue
        local=ROOT/url
        if not local.exists(): missing.append({'document':path.name,'target':url})
retrievals=[x for p in sorted(ROOT.glob('batch*-results.json')) for x in json.loads(p.read_text(encoding='utf8'))]
pdfs=[]
for path in sorted((ROOT/'sources').glob('*/source.pdf')):
    doc=pymupdf.open(path)
    pdfs.append({'id':path.parent.name,'path':str(path),'pages':len(doc),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(ROOT/'evidence-ledger.json').write_text(json.dumps({'as_of':'2026-10-10','requests':retrievals,'local_pdfs':pdfs,'reused':json.loads((ROOT/'reused-sources.json').read_text(encoding='utf8'))},ensure_ascii=False,indent=2),encoding='utf8')
checks={
 'matrix_records':len(papers),
 'matrix_unique_ids':len({x['id'] for x in papers})==len(papers),
 'csv_json_match':len(rows)==len(papers) and all(a['title']==b['title'] for a,b in zip(rows,papers)),
 'ris_record_count':(ROOT/'references.ris').read_text(encoding='utf8').count('ER  -'),
 'missing_local_links':missing,
 'baseline_all_unchanged':all(x['unchanged'] for x in integrity),
 'local_pdf_count':len(pdfs),
 'scope':'Artifact structure, source acceptance and baseline integrity only; not DeepPaperNote lint or model testing.'
}
checks['passed']=checks['matrix_unique_ids'] and checks['csv_json_match'] and checks['ris_record_count']==len(papers) and not missing and checks['baseline_all_unchanged']
(ROOT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(checks,ensure_ascii=False))

