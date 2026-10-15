"""Read-only probes on three training scenes; not a trained fusion experiment."""
import ast
import json
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy import signal
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import Net

def load_functions():
    tree = ast.parse((ROOT / 'utils.py').read_text(encoding='utf-8'))
    wanted = {'create_F', 'fspecial', 'Gaussian_downsample'}
    clean = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
    scope = dict(np.__dict__)
    scope.update({'np': np, 'signal': signal})
    exec(compile(clean, str(ROOT / 'utils.py'), 'exec'), scope)
    return scope

def metrics(x, y):
    mse = np.mean((x - y) ** 2, axis=0)
    cos = np.sum(x * y, axis=1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12)
    valid = np.linalg.norm(x, axis=1) > 1e-5
    return {'psnr_mean_band_db': float(np.mean(-10 * np.log10(np.maximum(mse, 1e-30)))),
            'psnr_global_db': float(-10 * np.log10(max(float(np.mean(mse)), 1e-30))),
            'sam_deg_nonzero_pixels': float(np.mean(np.arccos(np.clip(cos[valid], -1, 1))) * 180 / np.pi),
            'rmse_unit_range': float(np.sqrt(np.mean(mse)))}

def pca_basis(x):
    mu = np.mean(x, axis=0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt.T

def float_lifting(x):
    lo = (x[:, 0:30:2] + x[:, 1:30:2]) / 2
    return np.column_stack([lo, x[:, -1]]), x[:, 0:30:2] - x[:, 1:30:2]

def inverse_float_lifting(lo, hi):
    out = np.empty((len(lo), 31))
    out[:, 0:30:2] = lo[:, :15] + hi / 2
    out[:, 1:30:2] = lo[:, :15] - hi / 2
    out[:, -1] = lo[:, -1]
    return out

def main():
    funcs = load_functions()
    response = funcs['create_F']()
    psf = funcs['fspecial']('gaussian', 8, 3)
    scenes = sorted((ROOT / 'cave' / 'cave_train').glob('*.mat'), key=lambda p: p.name)[:3]
    records = []
    for path in scenes:
        cube = loadmat(path)['b'].astype(np.float64)
        cube /= cube.max()
        assert cube.shape[-1] == 31
        lr = funcs['Gaussian_downsample'](cube.transpose(2, 0, 1), psf, 8).transpose(1, 2, 0).reshape(-1, 31)
        msi = np.tensordot(response, cube.transpose(2, 0, 1), axes=([1], [0]))
        degraded_msi = funcs['Gaussian_downsample'](msi, psf, 8).transpose(1, 2, 0).reshape(-1, 3)
        projected_lr = lr @ response.T
        # All HR projection coefficients below are oracles, not observable predictions.
        target = cube[::4, ::4, :].reshape(-1, 31)
        mu, basis = pca_basis(lr)
        probes = []
        for k in [8, 12, 16, 20, 24, 31]:
            recon = (target - mu) @ basis[:, :k] @ basis[:, :k].T + mu
            probes.append({'k': k, **metrics(target, recon)})
        low_lr, high_lr = float_lifting(lr)
        low_hr, high_hr = float_lifting(target)
        design_lr = np.column_stack([np.ones(len(lr)), low_lr])
        weight = np.linalg.lstsq(design_lr, high_lr, rcond=None)[0]
        predicted_high = np.column_stack([np.ones(len(target)), low_hr]) @ weight
        no_residual = inverse_float_lifting(low_hr, predicted_high)
        lo_mu, lo_basis = pca_basis(low_lr)
        rwa_probes = []
        for k in [8, 12, 16]:
            low_recon = (low_hr - lo_mu) @ lo_basis[:, :k] @ lo_basis[:, :k].T + lo_mu
            high_recon = np.column_stack([np.ones(len(target)), low_recon]) @ weight
            rwa_probes.append({'k': k, **metrics(target, inverse_float_lifting(low_recon, high_recon))})
        records.append({'scene': path.name, 'hr_shape': list(cube.shape), 'sampled_hr_pixels': len(target),
                        'clean_D_msi_minus_R_lr_max_abs': float(np.max(np.abs(degraded_msi - projected_lr))),
                        'basis_fitted_only_on_lr_hsi': True, 'pca_oracle_projection': probes,
                        'continuous_rwa_zero_regression_residual_oracle_coarse': metrics(target, no_residual),
                        'continuous_rwa_pca_oracle_projection': rwa_probes,
                        'float_lifting_exact_inverse_max_error': float(np.max(np.abs(inverse_float_lifting(low_hr, high_hr) - target)))})
    # Build the actual baseline degradation as a small matrix to demonstrate ambiguity.
    side = 16
    operator = np.empty((4, side * side))
    for j in range(side * side):
        impulse = np.zeros((1, side, side))
        impulse.reshape(-1)[j] = 1
        operator[:, j] = funcs['Gaussian_downsample'](impulse, psf, 8).reshape(-1)
    _, _, vr = np.linalg.svd(response, full_matrices=True)
    _, _, vd = np.linalg.svd(operator, full_matrices=True)
    spectral = vr[-1]
    spatial = vd[-1]
    perturbation = np.outer(spectral, spatial)
    network = Net(31, 3)
    lowrank_network = Net(20, 3)
    torch.set_num_threads(2)
    from thop import profile
    macs31, _ = profile(network, inputs=(torch.zeros(1, 31, 8, 8), torch.zeros(1, 3, 64, 64)), verbose=False)
    macs20, _ = profile(lowrank_network, inputs=(torch.zeros(1, 20, 8, 8), torch.zeros(1, 3, 64, 64)), verbose=False)
    out = {'date': '2026-10-07', 'status': 'untrained_read_only_training_scene_diagnostics',
           'scope': 'Three alphabetically first training scenes only, no test-set selection; not representative performance.',
           'metric_protocol': 'Unit-peak mean per-band PSNR, no clipping; HR pixels sampled every 4 pixels, basis LR-only. Oracle coefficients use HR ground truth solely to bound compression error; these are not achievable fusion predictions.',
           'rwa_protocol': 'Continuous average/difference lifting with an unpaired last band; deliberately not the author integer floor/round implementation. LR-fitted regression. Zero high-frequency regression residual.',
           'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'cpu',
           'gpu_total_memory_gib': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None,
           'baseline_params': sum(p.numel() for p in network.parameters()),
           '20_band_network_params_same_width_depth': sum(p.numel() for p in lowrank_network.parameters()),
           'baseline_macs_batch1_64_patch': macs31,
           '20_band_network_macs_same_width_depth': macs20,
           'nullspace': {'R_rank': int(np.linalg.matrix_rank(response)), 'D_rank': int(np.linalg.matrix_rank(operator)),
                        'R_delta_max_abs': float(np.abs(response @ perturbation).max()),
                        'delta_D_transpose_max_abs': float(np.abs(perturbation @ operator.T).max()),
                        'delta_norm': float(np.linalg.norm(perturbation)),
                        'interpretation': 'Nonzero perturbation invisible to both linear observations. Can scale and add to an interior-positive image while preserving boundedness.'},
           'scenes': records}
    output = Path(__file__).with_name('feasibility-results.json')
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
