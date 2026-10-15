import json, hashlib, shutil
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parent
old=ROOT.parent/'cross_direction_review_20261007'/'fulltext'
reused={'srlf':'url_d592a59bf26e9693','uafl':'url_2bb5993df40328fd','pino':'url_f2374da1c30123d5'}
provenance=[]
for key,folder in reused.items():
    src=old/folder/'raw.pdf'; dest=ROOT/'sources'/key/'source.pdf'
    dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
    provenance.append({'id':key,'original_local_path':str(src),'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'retrieval':'Existing local official PDF reused; venue/identity rechecked 2026-10-10.'})
pages={'srlf':[3,6],'uafl':[2,6],'emrdiff':[2,6],'co2if':[2,5],'onehsi_pdf':[4],'lgdnet_pdf':[2]}
out=ROOT/'page_evidence';out.mkdir(exist_ok=True)
for key,chosen in pages.items():
    doc=pymupdf.open(ROOT/'sources'/key/'source.pdf')
    (ROOT/'sources'/key/'fulltext.txt').write_text('\n'.join(f'\n## PDF PAGE {i+1}\n'+p.get_text() for i,p in enumerate(doc)),encoding='utf8')
    for p in chosen: doc[p].get_pixmap(matrix=pymupdf.Matrix(1.15,1.15)).save(out/f'{key}_p{p+1}.png')
for path in ROOT.glob('batch*-results.json'):
    records=json.loads(path.read_text(encoding='utf8'))
    for r in records:
        if r.get('status')=='ok' and (r.get('bytes')==0 or ('publisher' in r.get('id','') and not r.get('path','').endswith('.pdf'))):
            r['status']='unavailable_pdf';r['error']='Empty/non-PDF publisher response. Full text unavailable.'
    path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'reused-sources.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf8')
print('Prepared full texts and visual page evidence; corrected PDF access statuses.')
