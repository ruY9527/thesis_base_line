# Novelty Report: geometry-enhanced residual diffusion around MossFuse

Review mode: standalone self-review in the same context. Not materially independent, not a validated systematic literature review. Date 2026-10-07.

## Contribution Under Review

Candidate: preserve MossFuse's shared/spatial-complementary/spectral-complementary feature roles, condition a residual diffusion model on input-derived compatible geometry, and optionally precondition physical MSI-visible spectral directions while maintaining observational fidelity.

## Claim Decomposition

- Framing: selective geometric detail transfer rather than indiscriminate MSI edge sharpening.
- Method: role-specific condition injection, geometry compatibility, residual correction; optional fixed invertible spectral-direction preconditioner.
- Evaluation: wrong-edge/misregistration probes, visible/null error reporting, same-condition deterministic comparator, internal versus native-scale separation.
- Artifact/dataset: none claimed new.
- Finding: none established; no training executed.

## Strongest Overlaps

See `prior-art-matrix.md` for explicit threat ratings. Direct threats include DDIF, SSDG, ARGS-Diff, SDP-Net and DDSR; several newest texts remain only abstract-level evidence.

## Novelty-Killing Objections

1. Residual diffusion + wavelet/MSI conditions is established by DDIF/KANDiff; dual-observation diffusion by HSR-Diff.
2. Space/spectrum disentangled time-dependent modulation is directly threatened by SSDG; two-stage modulation by DifferentialDiff.
3. Observation-only spatial/spectral diffusion already exists in ARGS-Diff (CVPR 2025).
4. Selective spatial guidance and preventing spectral shifts are directly threatened by SDP-Net, while PG-FDN supplies physical priors for residual generation.
5. Spectral range/null decomposition is established by DDSR/DDNM. Fixed A_G whitening is a coordinate preconditioner, not new inverse-problem information or a novel SDE by itself.
6. MossFuse already has spatial and spectral aggregation, blind operators and robustness evaluation. Claiming those missing would misstate the baseline.
7. An identical-condition deterministic residual network could match the proposed benefit more cheaply; edge gains alone cannot establish diffusion necessity.
8. The observation-only residual target needs an internal-scale task. Original ×32 images/patches may be too small for a second same-factor degradation; a successful supervised or ×8 version does not validate original ×32 unsupervised fusion.

## Decision

- Broad “MossFuse + geometry + diffusion” positioning: **2/5**, major reframe required.
- Narrow surviving positioning, `novelty_decision_rating`: **3/5**, narrow proceed only.
- `decision_confidence_rating`: **2/5**; material full-text uncertainties remain.
- Strongest overlap threat: **5/5** for generic residual/disentangled conditioning and range/null claims.

- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 2
- Narrowest defensible positioning: Test whether observation-compatible geometry, constrained by physical spectral observability and MossFuse feature roles, reduces erroneous structural transfer, and determine whether iterative residual generation improves over a same-condition deterministic correction at acceptable cost.
- What would change the decision: Downgrade or abandon new-method positioning if a recent full text implements equivalent control or same-condition deterministic correction explains all benefit; keep usefulness separate from novelty.
- Literature assurance mode: unlinked
- Literature assurance verdict: focused-search-only
- Unresolved high-priority novelty-critical questions: N1, N2, N3, N4, N5, N6

Impact positioning 3/5 means a plausible useful research question, not demonstrated impact.

Narrowest defensible positioning: **Test whether observation-compatible geometry, constrained by physical spectral observability and MossFuse feature roles, reduces erroneous structural transfer, and determine whether iterative residual generation improves over a same-condition deterministic correction at acceptable cost.**

This is an empirical mechanism question. It is not a first-use claim, a proof of spectral/topological invariance, or certification of model effectiveness. Rating governs positioning only, not empirical validity or engineering readiness.

## Change conditions / unresolved questions

- N1: Complete SSDG/SDP-Net/PG-FDN/DifferentialDiff full-text comparison; downgrade if the claimed control paths are already equivalent.
- N2: Determine whether a deterministic refiner or geometry-only gate replaces diffusion without loss.
- N3: Determine whether geometry compatibility actually reacts to misregistration/noise, particularly when clean D(M)=R(Y) makes compatibility constant.
- N4: Determine whether the observation-only internal-scale prior transfers to the native scale; original ×32 is unresolved.
- N5: Compare fixed A_G to identical conditioning and equivalent noise/loss weighting; remove theoretical-newness language if only reweighting changes.
- N6: Distinguish learned 48-channel roles from 31-band physical observability; measure HSI-only spectral detail rather than assuming it is protected.

## Claims to qualify

Do not claim first geometry-aware fusion diffusion, first role-separated diffusion, first R†R decomposition, guaranteed spectral protection, exact posterior sampling, state-of-the-art accuracy, original MossFuse speed retained, or original-scale unsupervised trainability already demonstrated.

## Evidence and history

Full MossFuse v4 and pinned author code read. Core GEWDiff/HSR-Diff/DDIF/ARGS-Diff full texts were read earlier in the conversation. Several newest overlaps only have publisher abstract/method-paragraph evidence. Existing project notes already contain observability/compatibility concepts; they must not be relabeled as newly invented here.

Candidate was refined after source/closest-work inspection and before any model training. Synthetic operator diagnostics check math only. No trained performance outcomes selected the candidate. Final experiment protocol should be fixed separately before confirmatory evaluation.

## Paper-Review Handoff

No submitted manuscript is under review. Any future draft should qualify first-use and guarantee language and include the closest precedents plus deterministic, wrong-geometry and native-scale controls. No fabricated paper quotation or reviewer verdict is supplied.
