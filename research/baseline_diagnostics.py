"""Read-only CAVE diagnostics; writes results to a separate output directory.

Run with the project's thesis_baseline Python environment. No training is done.
Metrics reproduce calculate_metrics.py on clipped reconstructions. Literature
scores must not be merged into this evaluator's measured comparison table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calculate_metrics import Loss_PSNR, Loss_RMSE, Loss_SAM
from model import Net


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def gaussian(size=8, sigma=3):
    grid = np.arange(size) - (size - 1) / 2
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def downsample(hsi, kernel, scale):
    return np.stack([signal.convolve2d(band, kernel, boundary='symm', mode='same')
                     [::scale, ::scale] for band in hsi])


def spectral_response():
    # Same response as utils.create_F; isolated to avoid utils' CUDA side effects.
    response = np.array([
        [2.,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,2,6,11,17,21,22,21,20,20,19,19,18,18,17,17],
        [1.,1,1,1,1,1,2,4,6,8,11,16,19,21,20,18,16,14,11,7,5,3,2,2,1,1,2,2,2,2,2],
        [7.,10,15,19,25,29,30,29,27,22,16,9,2,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]])
    return response / response.sum(axis=1, keepdims=True)


def tiled_reconstruction(lr, size, mode=None, model=None, msi=None,
                         scale=8, patch=64, stride=32):
    height, width = size
    if min(height, width) < patch or patch % scale or stride % scale:
        raise ValueError('Patch/stride must align with scale; image must fit patch.')
    # Preserve the existing evaluator's duplicated terminal tiles for comparability.
    ys = list(range(0, height - patch + 1, stride)) + [height - patch]
    xs = list(range(0, width - patch + 1, stride)) + [width - patch]
    output = torch.zeros((1, lr.shape[1], height, width), device=lr.device)
    counts = torch.zeros((1, 1, height, width), device=lr.device)
    for y in ys:
        for x in xs:
            tile = lr[:, :, y // scale:(y + patch) // scale,
                      x // scale:(x + patch) // scale]
            if model is None:
                pred = F.interpolate(tile, size=(patch, patch), mode=mode,
                                     align_corners=False)
            else:
                pred = model(tile, msi[:, :, y:y + patch, x:x + patch])
            output[:, :, y:y + patch, x:x + patch] += pred.clamp(0, 1)
            counts[:, :, y:y + patch, x:x + patch] += 1
    if not torch.all(counts > 0):
        raise ValueError('Uncovered pixels in reconstruction.')
    return output / counts


def phase_aligned_interpolation(lr, size, mode, scale=8):
    """Match scipy even-kernel 'same' samples at HR coordinate 8*j - 0.5.

    F.interpolate(align_corners=False) places its LR centers at 8*j + 3.5.
    This diagnostic adjusts only the interpolation grid, not the degradation.
    Boundary treatment is border replication; output is clipped like the baseline.
    """
    height, width = size
    yy, xx = torch.meshgrid(torch.arange(height, dtype=lr.dtype),
                            torch.arange(width, dtype=lr.dtype), indexing='ij')
    source_y = (yy + 0.5) / scale
    source_x = (xx + 0.5) / scale
    grid = torch.stack([2 * (source_x + 0.5) / lr.shape[3] - 1,
                        2 * (source_y + 0.5) / lr.shape[2] - 1], dim=-1).unsqueeze(0)
    return F.grid_sample(lr, grid.to(lr.device), mode=mode, padding_mode='border',
                         align_corners=False).clamp(0,1)


def metrics(reference, prediction, scale=8):
    ref = reference.squeeze(0).permute(1, 2, 0).cpu()
    pred = prediction.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1)
    diff = pred.double() - ref.double()
    band_mse = diff.square().mean(dim=(0, 1))
    means = ref.double().mean(dim=(0, 1))
    if torch.any(means <= 0):
        ergas = None  # Undefined for a zero-mean reference band.
    else:
        ergas = float(100 / scale * torch.sqrt((band_mse / means.square()).mean()))
    return {
        'psnr': float(Loss_PSNR()(ref, pred)),
        'rmse_255': float(Loss_RMSE()(ref, pred)),
        'sam_deg': float(Loss_SAM()(ref.numpy(), pred.numpy())),
        'ergas_scale8': ergas,
        'mae_reconstructed': float(diff.abs().mean()),
    }


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'research/baseline_comparison_20261006')
    parser.add_argument('--checkpoint', type=Path,
                        help='Optional trusted weights for the current Net; not inferred from CSV.')
    parser.add_argument('--checkpoint-label', default='provided_checkpoint')
    parser.add_argument('--append-phase-check', action='store_true',
                        help='Reuse completed interpolation rows and add phase diagnostics.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous_manifest = None
    if args.append_phase_check:
        with (args.output/'dataset_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
            previous_manifest = list(csv.DictReader(stream))
    torch.set_num_threads(4)
    with args.record.open(encoding='utf-8-sig', newline='') as stream:
        records = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if not records or any(not math.isfinite(v) for row in records for v in row.values()):
        raise ValueError('Empty or nonfinite training record.')
    best = max(records, key=lambda row: row['val_psnr'])
    late = [row for row in records if row['epoch'] >= 1000]
    train_files = sorted((ROOT / 'cave/cave_train').glob('*.mat'))
    test_files = sorted((ROOT / 'cave/cave_test').glob('*.mat'))
    if not test_files:
        raise ValueError('No CAVE test scenes found.')
    manifest = []
    for split, files in [('train', train_files), ('test', test_files)]:
        for path in files:
            manifest.append({'split': split, 'scene': path.name, 'path': str(path),
                             'sha256': sha256(path)})
    train_hashes = {item['sha256'] for item in manifest if item['split'] == 'train'}
    duplicate_content = [item['scene'] for item in manifest
                         if item['split'] == 'test' and item['sha256'] in train_hashes]
    if previous_manifest is not None and previous_manifest != manifest:
        raise ValueError('Dataset changed; cannot append diagnostics from different scene files.')
    write_csv(args.output / 'dataset_manifest.csv', manifest)
    network = Net().eval()
    model_info = {'parameters': sum(p.numel() for p in network.parameters()),
                  'architecture_binding_to_uploaded_run': 'unverified',
                  'cuda_available': torch.cuda.is_available()}
    try:
        from thop import profile
        macs, _ = profile(network, inputs=(torch.zeros(1,31,8,8),
                                          torch.zeros(1,3,64,64)), verbose=False)
        model_info['thop_macs_per_64_patch_batch1'] = macs
        model_info['thop_note'] = 'MACs convention; not interchangeable with paper FLOPs.'
    except ImportError:
        model_info['thop_note'] = 'thop unavailable; no compute estimate.'
    if args.checkpoint:
        # Profile hooks/buffers must not affect strict checkpoint loading.
        network = Net().eval()
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        network.load_state_dict(state, strict=True)
        model_info['checkpoint'] = str(args.checkpoint.resolve())
        model_info['checkpoint_sha256'] = sha256(args.checkpoint)
        # CSV provenance still requires external matching even if shapes agree.
    rows = []
    if args.append_phase_check:
        previous = json.loads((args.output/'diagnostics.json').read_text(encoding='utf-8'))
        if previous['record_sha256'] != sha256(args.record):
            raise ValueError('CSV differs from the existing diagnostics.')
        if args.checkpoint:
            raise ValueError('Evaluate checkpoints in a fresh output directory, not phase-append mode.')
        with (args.output/'interpolation_per_scene.csv').open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                if row['method'].endswith('_phase_aligned'):
                    continue
                rows.append({k:v if k in ('scene','method') else float(v) if v else None
                             for k,v in row.items()})
    normalizations = []
    with torch.inference_mode():
        for path in test_files:
            raw = np.asarray(hdf5storage.loadmat(str(path))['b'])
            if raw.ndim != 3 or raw.shape[2] != 31 or not np.isfinite(raw).all():
                raise ValueError(f'Invalid HSI {path.name}: {raw.shape}')
            max_value = float(raw.max())
            if max_value <= 0:
                raise ValueError(f'Zero scene: {path.name}')
            hsi = np.transpose(raw / max_value, (2,0,1))
            height, width = hsi.shape[1:]
            height, width = height // 8 * 8, width // 8 * 8
            hsi = hsi[:, :height, :width]
            lr = torch.from_numpy(downsample(hsi, gaussian(), 8)).float().unsqueeze(0)
            ref = torch.from_numpy(hsi).float().unsqueeze(0)
            normalizations.append({'scene': path.name, 'raw_min': float(raw.min()),
                                   'raw_max': max_value, 'height': height, 'width': width})
            for mode in ['bilinear', 'bicubic']:
                for reconstruction in ([] if args.append_phase_check else ['full', 'tiled']):
                    start = time.perf_counter()
                    if reconstruction == 'full':
                        prediction = F.interpolate(lr, size=(height,width), mode=mode,
                                                   align_corners=False).clamp(0,1)
                    else:
                        prediction = tiled_reconstruction(lr, (height,width), mode=mode)
                    elapsed = time.perf_counter() - start
                    rows.append({'scene': path.name, 'method': f'{mode}_{reconstruction}',
                                 **metrics(ref,prediction), 'inference_seconds_cpu': elapsed})
                start = time.perf_counter()
                prediction = phase_aligned_interpolation(lr,(height,width),mode)
                elapsed = time.perf_counter()-start
                rows.append({'scene':path.name,'method':f'{mode}_phase_aligned',
                             **metrics(ref,prediction),'inference_seconds_cpu':elapsed})
            if args.checkpoint:
                msi = torch.from_numpy(np.tensordot(spectral_response(),hsi,axes=([1],[0]))).float().unsqueeze(0)
                start = time.perf_counter()
                prediction = tiled_reconstruction(lr,(height,width),model=network,msi=msi)
                elapsed = time.perf_counter()-start
                rows.append({'scene':path.name,'method':args.checkpoint_label,
                             **metrics(ref,prediction),'inference_seconds_cpu':elapsed})
            print(f'Evaluated {path.name}', flush=True)
    write_csv(args.output / 'interpolation_per_scene.csv', rows)
    write_csv(args.output / 'scene_normalization.csv', normalizations)
    summary = []
    for method in sorted({row['method'] for row in rows}):
        subset = [row for row in rows if row['method']==method]
        item = {'method': method, 'scenes': len(subset)}
        for key in ['psnr','rmse_255','sam_deg','ergas_scale8','mae_reconstructed']:
            values = [row[key] for row in subset if row[key] is not None]
            item[key+'_mean'] = float(np.mean(values)) if values else None
            item[key+'_scene_std'] = float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(item)
    write_csv(args.output / 'interpolation_summary.csv',summary)
    diagnostics = {
        'record_path':str(args.record.resolve()), 'record_sha256':sha256(args.record),
        'record_points':len(records), 'best_psnr_row':best, 'last_row':records[-1],
        'best_rmse_row':min(records,key=lambda row:row['val_rmse']),
        'best_sam_row':min(records,key=lambda row:row['val_sam']),
        'late_psnr_mean':float(np.mean([row['val_psnr'] for row in late])) if late else None,
        'late_psnr_range':float(np.ptp([row['val_psnr'] for row in late])) if late else None,
        'train_scenes':len(train_files),'test_scenes':len(test_files),
        'cross_split_identical_file_hashes':duplicate_content,
        'duplicate_check_limit':'Byte-identical files only; no semantic duplicate proof.',
        'protocol':{'scale':8,'kernel_size':8,'kernel_sigma':3,'boundary':'symm',
                    'sampling':'0::8','normalization':'scene / scene.max()',
                    'patch':64,'stride':32,'terminal_tile_duplicate_preserved':True},
        'model':model_info,'measured_methods':summary,
        'timing_note':'CPU single pass, no warmup; diagnostic timing only, not a speed benchmark.',
        'csv_to_local_dataset_binding':'unverified; no scene manifest accompanied the uploaded record',
        'legacy_loss_note':'val_loss averages raw tile MAE; final image metrics use clipped averaged tiles.',
        'phase_note':'Phase-aligned grids assume current even 8x8 symmetric kernel and 0::8 sampling; do not use unchanged for different degradation operators.',
    }
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'best_psnr':best,'model':model_info,'measured':summary},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
