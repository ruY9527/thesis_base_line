"""Restricted observation examples, not model performance or natural-scene evidence."""
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
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in {'create_F', 'fspecial', 'Gaussian_downsample'}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/'utils.py'), 'exec'), scope)
R = scope['create_F']()
D = scope['Gaussian_downsample']
kernel = scope['fspecial']('gaussian', 8, 3)
_, _, V = np.linalg.svd(R, full_matrices=True)
n = 32
cols = []
for j in range(n):
    image = np.zeros((1,n,n))
    image[0,:,j] = 1
    cols.append(D(image,kernel,8)[0,0])
spatial_operator = np.stack(cols,axis=1)
_, _, Q = np.linalg.svd(spatial_operator, full_matrices=True)
v = np.sin(np.arange(n)*2.1)
v = v - Q[:4].T @ (Q[:4] @ v)
fine = np.tile(v,(n,1)); fine /= np.linalg.norm(fine)
coarse = np.ones((n,n)); coarse /= np.linalg.norm(coarse)
lam = .001
cases = []
for spatial_name,phi in [('coarse',coarse),('spatial_null',fine)]:
    for spectral_name,z in [('msi_visible',V[0]),('msi_null',V[3])]:
        e_h = float(np.linalg.norm(D(phi[None],kernel,8))**2)
        e_m = float(np.linalg.norm(R@z)**2)
        h = e_h + e_m
        cases.append(dict(spatial_mode=spatial_name,spectral_mode=spectral_name,
                          lr_energy=e_h,msi_energy=e_m,joint_energy=h,
                          msi_only_gate=e_m/(e_m+lam),joint_gate=h/(h+lam)))
# Rotation invariance and positive-semidefinite normal matrix for fixed basis/templates.
U = V[[0,3]].T
Psi = np.column_stack([coarse.ravel(),fine.ravel()])
DPsi = np.column_stack([D(p.reshape(1,n,n),kernel,8).ravel() for p in Psi.T])
AH = np.kron(DPsi,U); AM = np.kron(Psi,R@U)
H = AH.T@AH+AM.T@AM
G = np.linalg.solve(H+lam*np.eye(4),H)
rotation = np.array([[.6,-.8],[.8,.6]])
AHr = np.kron(DPsi,U@rotation); AMr = np.kron(Psi,R@U@rotation)
Hr = AHr.T@AHr+AMr.T@AMr
Gr = np.linalg.solve(Hr+lam*np.eye(4),Hr)
T = np.kron(np.eye(2),rotation)
results = dict(evidence_class='exploratory synthetic algebraic examples',
               source_operator='utils.py: create_F, fspecial, Gaussian_downsample',
               conditions='known fixed orthonormal basis and spatial templates; no model training',
               ridge_lambda=lam,cases=cases,
               restricted_joint_eigenvalues=np.linalg.eigvalsh(H).tolist(),
               rotation_invariance_max_error=float(np.max(np.abs(G-T@Gr@T.T))),
               conclusion='MSI-only support can suppress a coarse spectral mode that LR-HSI observes. '
                          'Joint-null perturbations remain unobservable. Neither gate is a proof of reconstruction accuracy.')
(OUT/'joint-support-probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False,indent=2))
