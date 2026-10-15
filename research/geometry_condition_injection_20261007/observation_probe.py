"""Exploratory operator examples, not a trained model or natural-scene benchmark."""
import ast
import json
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
tree = ast.parse((ROOT / 'utils.py').read_text(encoding='utf-8'))
scope = dict(np.__dict__)
scope.update(np=np, signal=signal)
selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name in {'create_F', 'fspecial', 'Gaussian_downsample'}],
                      type_ignores=[])
exec(compile(selected, str(ROOT / 'utils.py'), 'exec'), scope)
R = scope['create_F']()
D = scope['Gaussian_downsample']
psf = scope['fspecial']('gaussian', 8, 3)
rng = np.random.default_rng(20261007)

# In this project's ideal, common spatial operator, the two degradations commute.
x = rng.uniform(.25, .75, (31, 64, 64))
m = np.einsum('mc,chw->mhw', R, x)
y = D(x, psf, 8)
commutation = np.max(np.abs(D(m, psf, 8) - np.einsum('mc,chw->mhw', R, y)))

# Same local dimension and full MSI rank can have very different sensitivity.
_, rs, rv = np.linalg.svd(R, full_matrices=True)
spectra = []
for mixing in [1., .1, .01]:
    U = np.column_stack([rv[0], mixing * rv[1] + np.sqrt(1 - mixing**2) * rv[3]])
    sv = np.linalg.svd(R @ U, compute_uv=False)
    spectra.append({'rowspace_fraction': mixing, 'local_rank': 2,
                    'msi_rank': int(np.linalg.matrix_rank(R @ U)),
                    'singular_values': sv.tolist(), 'condition_number': float(sv[0]/sv[-1])})

# A perturbation in both the spatial and spectral null spaces is invisible to both.
# Build the spatial null vector using the exact project's 2D operator, without
# approximating its even-sized kernel alignment or symmetric boundary convention.
n = 32
columns = []
for j in range(n):
    image = np.zeros((1, n, n))
    image[0, :, j] = 1
    columns.append(D(image, psf, 8)[0, 0])
A = np.stack(columns, axis=1)
_, _, av = np.linalg.svd(A, full_matrices=True)
seed = np.sin(np.arange(n) * 2.1)
v = seed - av[:A.shape[0]].T @ (av[:A.shape[0]] @ seed)
v /= np.max(np.abs(v))
z = rv[3] / np.max(np.abs(rv[3]))
delta = .1 * z[:, None, None] * np.tile(v, (n, 1))[None]
x0 = np.full_like(delta, .5)
x1 = x0 + delta
results = {
    'evidence_class': 'synthetic algebraic examples; no training; no natural scene performance evidence',
    'seed': 20261007,
    'spectral_response_shape': list(R.shape), 'spectral_response_rank': int(np.linalg.matrix_rank(R)),
    'spectral_response_singular_values': rs.tolist(),
    'common_operator_commutation_max_abs': float(commutation),
    'same_rank_different_sensitivity': spectra,
    'joint_null_example': {
        'dimensions': list(x1.shape), 'value_range': [float(x1.min()), float(x1.max())],
        'hr_max_difference': float(np.max(np.abs(delta))),
        'hr_rmse_difference': float(np.sqrt(np.mean(delta**2))),
        'msi_max_difference': float(np.max(np.abs(np.einsum('mc,chw->mhw', R, delta)))),
        'lr_hsi_max_difference': float(np.max(np.abs(D(delta, psf, 8)))),
        'spatial_mean_abs_gradient_difference': float(np.mean(np.abs(np.diff(delta, axis=2)))),
        'interpretation': 'Same ideal observed pair does not uniquely specify all HR geometry. '
                          'Signed perturbation is synthetic, not a measured material spectrum.'
    }
}
(OUT / 'observation-probe.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=False, indent=2))
