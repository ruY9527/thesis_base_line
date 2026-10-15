"""Task-scoped analytic checks, baseline fingerprints, and PDF page renders.

No network training or baseline edits. The null-space example is linear algebra,
not empirical evidence of reconstruction gains.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent

def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

# Extract only the literal matrix, without importing utils.py (CUDA side effect).
tree = ast.parse((REPO / 'utils.py').read_text(encoding='utf-8-sig'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_F')
R = np.array(ast.literal_eval(function.body[0].value.args[0]), dtype=float)
R /= R.sum(axis=1, keepdims=True)
_, _, vr = np.linalg.svd(R, full_matrices=True)
v = vr[-1]

# An illustrative, explicitly different spatial operator: block average x8,
# not the repository's Gaussian blur. Non-uniqueness holds for either operator.
side, scale = 16, 8
D = np.zeros((4, side * side))
for row in range(2):
    for col in range(2):
        for i in range(row * scale, (row + 1) * scale):
            for j in range(col * scale, (col + 1) * scale):
                D[2 * row + col, side * i + j] = 1 / (scale * scale)
_, _, vd = np.linalg.svd(D, full_matrices=True)
q = vd[-1]
delta = np.outer(v, q)
delta *= .15 / np.max(np.abs(delta))
base = np.full_like(delta, .5)
left, right = base - delta, base + delta
results = {
    'kind': 'analytic_synthetic_counterexample_not_training',
    'spectral_operator': 'exact create_F literal, row normalized',
    'spatial_operator': 'illustrative 16x16 to 2x2 block average x8; NOT baseline Gaussian',
    'rank_R': int(np.linalg.matrix_rank(R)), 'dim_null_R': 31 - int(np.linalg.matrix_rank(R)),
    'max_R_delta': float(np.max(np.abs(R @ delta))),
    'max_delta_Dt': float(np.max(np.abs(delta @ D.T))),
    'max_MSI_difference': float(np.max(np.abs(R @ left - R @ right))),
    'max_LR_difference': float(np.max(np.abs(left @ D.T - right @ D.T))),
    'HR_RMSE_difference': float(np.sqrt(np.mean((left - right) ** 2))),
    'range_left': [float(left.min()), float(left.max())],
    'range_right': [float(right.min()), float(right.max())],
    'conclusion': 'identical dual observations can correspond to distinct bounded HR cubes; residual alone cannot certify truth',
    'limits': 'synthetic spectra are not asserted to be real materials; no new theorem, no trained performance or calibrated coverage',
}
assert results['max_MSI_difference'] < 1e-12 and results['max_LR_difference'] < 1e-12
write('analytic-checks.json', results)
fingerprints = {}
for name in ['model.py', 'train_cave_M.py', 'train_dataloader.py', 'utils.py', 'calculate_metrics.py']:
    p = REPO / name
    fingerprints[name] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
write('baseline-snapshot.json', {
    'files': fingerprints,
    'train_files': sorted(p.name for p in (REPO / 'cave/cave_train').glob('*.mat')),
    'test_files': sorted(p.name for p in (REPO / 'cave/cave_test').glob('*.mat')),
    'user_preferences': {'training': 'supervised', 'focus': 'method innovation', 'gpu': '4090', 'time_budget': 'not specified'},
})

out = ROOT / 'evidence-pages'
out.mkdir(exist_ok=True)
rendered = []
for key, pages in {'diff_spectra': [6, 7], 'pino': [4], 'umnet': [3], 'uafl': [3], 'hipandas': [5], 'srlf': [3], 'unipan': [4], 'carl': [3]}.items():
    record = json.loads((ROOT / f'fetch-{key}.json').read_text(encoding='utf-8'))
    raw = Path(record['parsed_path']).with_name('raw.pdf')
    doc = pdfium.PdfDocument(str(raw))
    for page in pages:
        target = out / f'{key}-p{page}.png'
        doc[page - 1].render(scale=1.7).to_pil().save(target)
        rendered.append({'paper': key, 'page': page, 'path': str(target)})
    doc.close()
write('rendered-pages.json', rendered)
print(json.dumps(results, ensure_ascii=False, indent=2))
print('Rendered pages:', len(rendered))
