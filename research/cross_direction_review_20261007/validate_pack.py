"""Verify artifact integrity; does not claim scientific novelty or model performance."""
from pathlib import Path
from collections import Counter
import json
import re
import hashlib
import sys

ROOT = Path(__file__).resolve().parent
errors = []
all_json = list(ROOT.rglob('*.json'))
for p in all_json:
    try:
        json.loads(p.read_text(encoding='utf-8'))
    except Exception as exc:
        errors.append(f'JSON invalid: {p}: {exc}')
sources = json.loads((ROOT/'sources.json').read_text(encoding='utf-8'))
chosen = sources['core_fulltext'] + sources['opponents_fulltext']
for p in chosen:
    key = p['id']
    f = json.loads((ROOT/f'fetch-{key}.json').read_text(encoding='utf-8'))
    content = Path(f['parsed_path']).read_text(encoding='utf-8')
    if f['status'] != 'ok' or f"END-OF-PAPER:{f['sentinel']}" not in content:
        errors.append(f'invalid full-text evidence: {key}')
    # The helper counts newline-delimited lines; str.splitlines also splits
    # PDF-extracted Unicode line separators, present in PINO and UniPAN.
    if content.count('\n') != f['parsed_lines']:
        errors.append(f'line-count changed: {key}')
    if len(content) < 5000 or 'Verifying your browser' in content:
        errors.append(f'body not credible: {key}')
snapshot = json.loads((ROOT/'baseline-snapshot.json').read_text(encoding='utf-8'))
unchanged = {}
for key,f in snapshot['files'].items():
    unchanged[key] = hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
    if not unchanged[key]:
        errors.append(f'baseline changed: {key}')
linked = []
for p in ROOT.rglob('*.md'):
    if 'fulltext' in p.parts:
        continue
    content = p.read_text(encoding='utf-8')
    for value in re.findall(r'\]\((D:/[^)\n]+)\)',content):
        value = re.sub(r':\d+$','',value)
        linked.append(value)
        if not Path(value).exists():
            errors.append(f'local link missing in {p.name}: {value}')
    if any(ord(c)<32 and c not in '\n\r\t' for c in content):
        errors.append(f'control character in {p.name}')
scores = json.loads((ROOT/'ideation/idea-scores.json').read_text(encoding='utf-8'))['ideas']
required = {'idea_id','title','summary','hypothesis','contribution_type','source_basis','assumptions','novelty_questions','minimum_validation','disconfirming_evidence','kill_criteria','closest_work','differentiation','scores','risk','estimated_effort','status','rationale','blocking_questions'}
for idea in scores:
    if not required <= idea.keys():
        errors.append(f'idea fields missing: {idea["idea_id"]}')
    if any(not isinstance(v,int) or not 1<=v<=5 for v in idea['scores'].values()):
        errors.append(f'idea score invalid: {idea["idea_id"]}')
if len(scores)!=15 or len({x['idea_id'] for x in scores})!=15:
    errors.append('idea count or identity inconsistent')
query_rows=[x for x in (ROOT/'novelty-review/search-log.md').read_text(encoding='utf-8').splitlines() if re.match(r'\|\d+\|',x)]
check=json.loads((ROOT/'analytic-checks.json').read_text(encoding='utf-8'))
if not(check['rank_R']==3 and check['dim_null_R']==28 and check['max_MSI_difference']<1e-12 and check['max_LR_difference']<1e-12 and check['HR_RMSE_difference']>0):
    errors.append('analytic counterexample inconsistent')
summary = {
    'status':'passed' if not errors else 'failed',
    'json_files_parsed':len(all_json),
    'core_fulltext':len(sources['core_fulltext']),
    'recent_requested_main_venues_fulltext':sum('main' in p['venue'] for p in sources['core_fulltext']),
    'additional_fulltext_opponents':len(sources['opponents_fulltext']),
    'extended_abstract_or_metadata_catalog':len(sources['extended_catalog']),
    'exact_queries_including_repeats':len(query_rows),
    'ideas_by_status':dict(Counter(p['status'] for p in scores)),
    'local_artifact_links_checked':len(linked),
    'baseline_file_hashes_unchanged':unchanged,
    'errors':errors,
    'limits':'integrity and structural checks only; no literature completeness, novelty proof, GPU training or empirical improvement',
}
(ROOT/'validation-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
sys.exit(bool(errors))
