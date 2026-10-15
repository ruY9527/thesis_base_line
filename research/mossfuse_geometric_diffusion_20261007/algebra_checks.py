"""Untrained operator checks; these results cannot establish fusion quality."""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.manual_seed(20261007)
root = Path(__file__).resolve().parent
bands, side, scale = 31, 32, 8
R = torch.rand(3, bands)
R = R / R.sum(1, keepdim=True)
P = torch.linalg.pinv(R) @ R
N = torch.eye(bands) - P
edge = torch.linspace(0, 1, side).repeat(side, 1)
a = 1 - 0.5 * edge

def project(x, p):
    return torch.einsum('ij,jhw->ihw', p, x)

def A(x):
    return a * project(x, P) + project(x, N)

def Ainv(x):
    return project(x, P) / a + project(x, N)

coord = torch.arange(scale - 1) - (scale // 2 - 1)
g = torch.exp(-coord.square() / (2 * 2.0 ** 2))
kernel = torch.outer(g, g)
kernel /= kernel.sum()

def D(x):
    c = x.shape[0]
    return F.conv2d(x[None], kernel[None, None].repeat(c, 1, 1, 1), stride=scale, groups=c)[0]

# A small spatial operator with MossFuse's valid convolution, odd kernel and stride.
out_side = D(torch.zeros(1, side, side)).shape[-1]
rows = []
for i in range(out_side):
    for j in range(out_side):
        row = torch.zeros(side, side)
        row[i * scale:i * scale + scale - 1, j * scale:j * scale + scale - 1] = kernel
        rows.append(row.flatten())
Dm = torch.stack(rows)
w = torch.randn(side * side)
w = w - Dm.T @ torch.linalg.solve(Dm @ Dm.T, Dm @ w)
w /= w.norm()
v = N @ torch.randn(bands)
v /= v.norm()
delta = v[:, None, None] * w.reshape(side, side)
x = torch.randn(bands, side, side)
u = torch.randn(bands, side, side, requires_grad=True)
probe = torch.randn_like(u)
gradient = torch.autograd.grad((A(u) * probe).sum(), u)[0]
internal_conv_errors = {}
for target_side in [16, 4]:
    try:
        F.conv2d(torch.zeros(1, 1, target_side, target_side), torch.ones(1, 1, 31, 31), stride=32)
    except RuntimeError as error:
        internal_conv_errors[str(target_side)] = str(error)
assert set(internal_conv_errors) == {'16', '4'}
result = {
    'status': 'untrained_float64_operator_checks_only',
    'R_rank': int(torch.linalg.matrix_rank(R)),
    'P_idempotence_max_abs': float((P @ P - P).abs().max()),
    'R_null_max_abs': float((R @ N).abs().max()),
    'A_inverse_roundtrip_max_abs': float((Ainv(A(x)) - x).abs().max()),
    'geometry_scale_min_max': [float(a.min()), float(a.max())],
    'visible_noise_variance_ratio_at_strong_edge': 0.25,
    'null_noise_variance_ratio_at_strong_edge': 1.0,
    'whitening_gradient_chain_max_abs': float((gradient - A(probe)).abs().max()),
    'common_nullspace_delta_norm': float(delta.norm()),
    'common_nullspace_R_delta_max_abs': float(project(delta, R).abs().max()),
    'common_nullspace_D_delta_max_abs': float(D(delta).abs().max()),
    'constant_geometry_commutes_with_D': True,
    'variable_geometry_D_product_vs_center_scale_max_abs': float((D(a[None] * x) - a[3::8, 3::8][None] * D(x)).abs().max()),
    'internal_scale_examples': [
        {'native_scale': 8, 'hr_size': 512, 'lr_hsi_size': 64, 'internal_target_size': 64, 'internal_input_size': 8, 'internal_kernel': 7, 'valid_convolution_possible': True},
        {'native_scale': 32, 'hr_size': 512, 'lr_hsi_size': 16, 'internal_target_size': 16, 'internal_input_size': None, 'internal_kernel': 31, 'valid_convolution_possible': False},
        {'native_scale': 32, 'hr_patch': 128, 'lr_hsi_patch': 4, 'internal_target_size': 4, 'internal_input_size': None, 'internal_kernel': 31, 'valid_convolution_possible': False}
    ],
    'actual_torch_x32_internal_conv_errors': internal_conv_errors,
    'scope': 'Synthetic random row-normalized SRF and known Gaussian PSF; not measured dataset, trained network, or exact posterior sampler.',
}
assert result['A_inverse_roundtrip_max_abs'] < 1e-10
assert result['whitening_gradient_chain_max_abs'] < 1e-10
assert result['common_nullspace_R_delta_max_abs'] < 1e-10
assert result['common_nullspace_D_delta_max_abs'] < 1e-10
(root / 'algebra-results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
manifest_path = root / 'source-manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['active_baseline_hashes_after'] = {
    name: hashlib.sha256((root.parents[1] / name).read_bytes()).hexdigest()
    for name in manifest['active_baseline_hashes_before']
}
manifest['active_baseline_unchanged'] = manifest['active_baseline_hashes_before'] == manifest['active_baseline_hashes_after']
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
