"""Cheap algebra preflight. No network training, natural data, or metric claim."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
from scipy import signal


def main():
    root = Path(__file__).resolve().parents[2]
    source_path = root / "utils.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                               and n.name in {"create_F", "fspecial", "Gaussian_downsample"}],
                          type_ignores=[])
    env = {"np": np, "signal": signal, "ogrid": np.ogrid,
           "exp": np.exp, "finfo": np.finfo}
    exec(compile(selected, "original_operator_functions", "exec"), env)
    R = env["create_F"]()
    side, scale = 16, 8
    psf = env["fspecial"]("gaussian", 8, 3)
    n = side ** 2
    D = np.empty((4, n))
    for k in range(n):
        e = np.zeros((1, side, side))
        e.reshape(-1)[k] = 1
        D[:, k] = env["Gaussian_downsample"](e, psf, scale).reshape(-1)
    P = np.linalg.pinv(R) @ R
    Q = np.eye(R.shape[1]) - P
    rng = np.random.default_rng(20261006)
    w = P @ rng.normal(size=31)
    w *= .04 / np.max(np.abs(w))
    mu, t0 = .4, .45
    pair_errors, responses, oracle_errors, additive_errors = [], [], [], []
    for _ in range(5):
        v = Q @ rng.normal(size=31)
        v *= .04 / np.max(np.abs(v))
        q = rng.normal(size=n)
        q -= np.linalg.pinv(D) @ (D @ q)
        q *= .15 / np.max(np.abs(q))
        cubes = {(a, b): mu + (w + a * v)[:, None] * (t0 + b * q)[None, :]
                 for a in (0, 1) for b in (0, 1)}
        mixed = cubes[1, 1] - cubes[1, 0] - cubes[0, 1] + cubes[0, 0]
        expected = np.outer(v, q)
        assert np.max(np.abs(mixed - expected)) < 1e-12
        assert min(x.min() for x in cubes.values()) > 0
        assert max(x.max() for x in cubes.values()) < 1
        # Two distinct valid truths with the same observations: no universal inverse.
        x0 = cubes[1, 1]
        x1 = x0 + expected
        assert 0 < x1.min() < x1.max() < 1
        pair_errors.append(max(float(np.max(np.abs(R @ (x1-x0)))),
                               float(np.max(np.abs((x1-x0) @ D.T)))))
        responses.append(float(np.linalg.norm(expected)))
        for (a, b), x in cubes.items():
            h, m = x @ D.T, R @ x
            # Oracle is told the family constants mu,w,t0. This is an explicit prior.
            av_hat = (h.mean(axis=1)-mu) / t0 - w
            rw = R @ w
            bq_hat = rw @ (m - (R @ np.full(31, mu))[:, None]) / (rw @ rw) - t0
            rec = mu + np.outer(w + av_hat, t0 + bq_hat)
            oracle_errors.append(float(np.max(np.abs(rec-x))))
        # Best additive representation of this balanced 2x2 target retains only
        # main effects; its missing mixed component contributes ||mixed||^2/16
        # averaged squared Frobenius error per output. This is not Net's error.
        tensor = np.stack([cubes[0,0], cubes[0,1], cubes[1,0], cubes[1,1]]).reshape(2,2,31,n)
        additive = tensor.mean(axis=1,keepdims=True)+tensor.mean(axis=0,keepdims=True)-tensor.mean(axis=(0,1),keepdims=True)
        err = float(np.mean(np.sum((additive-tensor)**2,axis=(2,3))))
        assert abs(err - np.linalg.norm(mixed)**2/16) < 1e-12
        additive_errors.append(err)
    assert max(pair_errors) < 1e-12
    assert max(oracle_errors) < 1e-12
    assert min(responses) > .01
    assert min(additive_errors) > 0
    # Plausible comparator-win case: truly additive target has zero mixed term.
    additive_cube = {(a,b): mu + a*np.outer(v,np.ones(n)) + b*np.outer(w,q)
                     for a in (0,1) for b in (0,1)}
    zero_mix = additive_cube[1,1]-additive_cube[1,0]-additive_cube[0,1]+additive_cube[0,0]
    assert np.max(np.abs(zero_mix)) < 1e-12
    # Orthogonal features can still form a product downstream.
    ph, pm = np.array([2.,0.]), np.array([0.,3.])
    # Constant anti-aligned heads can satisfy geometric separation without roles.
    const_h = np.tile([1.,0.],(4,1))
    const_m = -const_h
    # Difference loss follows from the same full labels; common output drift
    # cancels in the difference and must still be penalized by full reconstruction.
    errors = {k: np.full_like(x, .01) for k, x in cubes.items()}
    error_mix = errors[1,1]-errors[1,0]-errors[0,1]+errors[0,0]
    assert np.max(np.abs(error_mix)) == 0
    result = {
        "status": "numeric_preflight_passed",
        "evidence_class": "exploratory_algebra_only",
        "new_network_training": False, "uses_natural_data": False,
        "seed": 20261006, "quartet_families_checked": 5,
        "operator": {"R_shape":list(R.shape),"D_shape":list(D.shape),
                     "side":side,"scale":scale,"kernel":8,"sigma":3,
                     "boundary":"symm","sampling":"0::8"},
        "same_observations_distinct_truths_max_error":max(pair_errors),
        "nonzero_joint_target_norm_range":[min(responses),max(responses)],
        "restricted_family_oracle_max_error":max(oracle_errors),
        "oracle_extra_information":["fixed known w","fixed known mu","fixed known t0",
                                    "assumed multiplicative generator; no added joint-null component"],
        "best_additive_balanced_quartet_mean_squared_frobenius_error_range":
            [min(additive_errors),max(additive_errors)],
        "additive_target_mixed_response_max_abs":float(np.max(np.abs(zero_mix))),
        "orthogonal_features_dot":float(ph @ pm),
        "orthogonal_features_downstream_product":float(ph[0]*pm[1]),
        "constant_anti_aligned_head_cosine":float(const_h[0] @ const_m[0]),
        "constant_head_sample_variance":float(const_h.var(axis=0).sum()),
        "common_output_drift_full_mae":.01,
        "common_output_drift_mixed_error":float(np.max(np.abs(error_mix))),
        "mix_loss_information_relation":"Mixed error is e11-e10-e01+e00; its L1 is bounded by the sum of full L1 errors. It adds emphasis, not independent target information.",
        "source_utils_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "interpretation":[
            "An unrestricted observation inverse cannot distinguish the null-space collision.",
            "A stated restricted family prior makes a joint target recoverable in this toy family.",
            "An additive comparator can win on a truly additive target.",
            "Orthogonality does not imply absence of functional interaction.",
            "Geometric head separation alone does not establish noncollapsed semantic roles.",
            "No architecture effect, natural PSNR/RMSE/SAM gain or novelty is established."
        ]
    }
    Path(__file__).with_name("preflight-results.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
