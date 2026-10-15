"""Check the algebra of a synthetic diagnostic; this does not evaluate Net."""
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
from scipy import signal


def main():
    root = Path(__file__).resolve().parents[2]
    source = (root / "utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"create_F", "fspecial", "Gaussian_downsample"}
    selected = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
        type_ignores=[],
    )
    env = {"np": np, "signal": signal, "ogrid": np.ogrid,
           "exp": np.exp, "finfo": np.finfo}
    exec(compile(selected, "selected_original_utils_functions", "exec"), env)
    response = env["create_F"]()
    psf = env["fspecial"]("gaussian", 8, 3)
    downsample = env["Gaussian_downsample"]
    side, scale = 16, 8
    pixels = side * side
    spatial = np.empty(((side // scale) ** 2, pixels))
    for k in range(pixels):
        basis = np.zeros((1, side, side))
        basis.reshape(-1)[k] = 1
        spatial[:, k] = downsample(basis, psf, scale).reshape(-1)

    rng = np.random.default_rng(20261006)
    row = np.linalg.pinv(response) @ response
    direction = rng.normal(size=31)
    visible = row @ direction
    private = (np.eye(31) - row) @ direction
    visible *= 0.04 / np.max(np.abs(visible))
    private *= 0.04 / np.max(np.abs(private))
    initial_pattern = rng.normal(size=pixels)
    pattern = initial_pattern - np.linalg.pinv(spatial) @ (spatial @ initial_pattern)
    pattern *= 0.15 / np.max(np.abs(pattern))
    cubes = {(a, b): 0.4 + (visible + a * private)[:, None] *
             (0.45 + b * pattern)[None, :]
             for a in (0, 1) for b in (0, 1)}
    hsi = {key: cube @ spatial.T for key, cube in cubes.items()}
    msi = {key: response @ cube for key, cube in cubes.items()}
    mixed = cubes[1, 1] - cubes[1, 0] - cubes[0, 1] + cubes[0, 0]
    expected = private[:, None] * pattern[None, :]
    errors = {
        "spectral_null_max_abs": float(np.max(np.abs(response @ private))),
        "spatial_null_max_abs": float(np.max(np.abs(spatial @ pattern))),
        "hsi_unchanged_under_spatial_factor_max_abs": max(
            float(np.max(np.abs(hsi[a, 1] - hsi[a, 0]))) for a in (0, 1)),
        "msi_unchanged_under_spectral_factor_max_abs": max(
            float(np.max(np.abs(msi[1, b] - msi[0, b]))) for b in (0, 1)),
        "mixed_response_identity_max_abs": float(np.max(np.abs(mixed - expected))),
        "mixed_response_spectral_measurement_max_abs": float(np.max(np.abs(response @ mixed))),
        "mixed_response_spatial_measurement_max_abs": float(np.max(np.abs(mixed @ spatial.T))),
    }
    assert max(errors.values()) < 1e-12, errors
    assert min(float(c.min()) for c in cubes.values()) > 0
    assert max(float(c.max()) for c in cubes.values()) < 1
    result = {
        "status": "algebraic_checks_passed",
        "evidence_class": "synthetic_numeric_algebra_check_only",
        "uses_real_hsi_data": False,
        "network_training_or_evaluation": False,
        "seed": 20261006,
        "source_utils_sha256": hashlib.sha256((root / "utils.py").read_bytes()).hexdigest(),
        "operator": {"R_shape": list(response.shape), "R_rank": int(np.linalg.matrix_rank(response)),
                     "D_shape": list(spatial.shape), "D_rank": int(np.linalg.matrix_rank(spatial)),
                     "patch_side": side, "scale": scale, "psf_size": 8, "sigma": 3,
                     "boundary": "symm", "mode": "same", "sampling": "[0::8,0::8]"},
        "errors": errors,
        "nonzero_responses": {
            "hsi_spectral_factor_norm": float(np.linalg.norm(hsi[1, 0] - hsi[0, 0])),
            "msi_spatial_factor_norm": float(np.linalg.norm(msi[0, 1] - msi[0, 0])),
            "mixed_output_norm": float(np.linalg.norm(mixed)),
            "mixed_output_max_abs": float(np.max(np.abs(mixed))),
            "cube_min": min(float(c.min()) for c in cubes.values()),
            "cube_max": max(float(c.max()) for c in cubes.values())},
        "limitations": ["Procedural spectra and patterns, not natural-image validation.",
                        "Joint-null interaction is specified by the generator's mixing prior, not uniquely identified by R and D.",
                        "Does not establish loss-induced damage, novelty, PSNR, RMSE, or SAM improvement."]}
    destination = Path(__file__).with_name("quartet_algebra_check.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
