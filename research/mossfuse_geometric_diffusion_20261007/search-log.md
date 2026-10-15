# Search Log and reading record

Date: 2026-10-07. Mode: focused primary-source searches supplementing the earlier GEWDiff transfer corpus; no systematic-recall claim. Filters: the table records query strings exactly; primary-source-only interpretation was applied after retrieval, without fabricating a domains filter. Zero direct hits for an exact combination do not establish novelty.

| Query | Source | Filters | Purpose | Closest hits | Notes |
|---|---|---|---|---|---|
| site:openaccess.thecvf.com Self-Learning Hyperspectral Adaptive Residual Guided Zhu 2025 | Web search | Query site qualifier | Falsify a new observation-only diffusion claim | ARGS-Diff official CVPR abstract and metadata | Confirms formal CVPR 2025 publication; prior complete arXiv reading reused |
| site:arxiv.org 2412.04802 Unsupervised Hyperspectral Self-Supervised Modality | Web search | Query site qualifier | Verify source identity | MossFuse v4 | Canonical skill fetcher retrieved HTML; full text read including references and sentinel |
| "MossFuse" "Geometric" diffusion | Web search | None | Exact combination overlap | No decisive exact implementation | Absence of a hit is weak evidence |
| hyperspectral multispectral fusion geometry guided diffusion wavelet residual | Web search | None | Falsify generic combination novelty | PG-FDN, DifferentialDiff, SDP-Net, DDIF, KANDiff, WCGE-Net | Publisher sources retained; blogs/encyclopedia ignored |
| "hyperspectral" "diffusion" "geometry" fusion spectral observable | Web search | None | Search spectral/geometry overlap | Broader geometry and heterogeneous fusion hits | No direct novelty conclusion drawn |
| "diffusion" "spectral response" "null space" hyperspectral fusion | Web search | None | Falsify new spectral projection claim | DDSR | Publisher method paragraph explicitly says Range–Null Space correction; projection is established |
| "Geometric Enhanced Wavelet" "AAAI" 2026 | Web search | None | Verify final venue | AAAI DOI and formal PDF | Reuses prior fully read formal PDF |
| "Unsupervised Hyperspectral Image Super-Resolution via Self-Supervised Modality Decoupling" IJCV 2026 | Web search | None | Verify final venue | Springer DOI page, DBLP | Formal March 2026 publication; arXiv metadata year 2024 is not venue year |
| site:mdpi.com/2072-4292/16/15/2692 DDSR null space projection guidance diffusion spectral reconstruction | Web search | Query site qualifier | Verify physical observability precedent | DDSR §3.3 | Exact domain/page open returned 429; indexed publisher paragraph usable, full paper not claimed read |
| site:openaccess.thecvf.com HSR-Diff Hyperspectral Image Super-Resolution Diffusion Model 2023 | Web search | Query site qualifier | Confirm fusion-diffusion precedent | HSR-Diff official ICCV page | Prior full text reading reused |
| site:github.com/dusongcheng/MossFuse Decoder_HR MossFuse | Web search | Query site qualifier | Verify author code provenance | Author repository | Code read from immutable commit |

## Retrieval/reading details

- MossFuse selected source: https://arxiv.org/html/2412.04802v4 . Skill's canonical fetch-and-parse helper produced `fulltext/arxiv_2412.04802/{raw.html,parsed.md,meta.json}`. Full 2234-line paper read in chunks; a clipped figure caption was reread. Completion marker quoted: `<!-- END-OF-PAPER:f5ca9f7ca223 -->`.
- Code provenance: public author GitHub, commit `d6c7eb4a487046aecae5265cabf81b0c23506465`. Seven files copied from pinned cache or public raw endpoint; SHA-256 recorded. Active baseline hashes recorded before/after and matched.
- Code findings anchored in source: Decoder_HR's existing fuse1/fuse2; author training entry's actual loss1/loss2; BlurDown valid convolution and SRF_Down; dataset center phase and original ×32 patch geometry. No unexecuted source issue was called an observed training failure.
- Direct opens of CVF formal HTML and DBLP returned 403/cache miss in one request; search indexed official CVF and Springer pages returned publication records. No access controls bypassed.
- Earlier full texts, GEWDiff code details, and searches reside in `../gewdiff_transfer_20261007/`; existing MossFuse studies in adjacent research directories were used as context, not independent evidence of newness.
- Recent-neighbor full texts remain a limitation: especially SSDG, SDP-Net, PG-FDN and DifferentialDiff. No claim of exhaustive search, complete citation chasing, or unique first use.
- Algebra checks use synthetic SRF/PSF only. They test projector/whitening math, common-null ambiguity and sampling dimensions, not training results.

## Selection history

The user specified the GEWDiff–MossFuse combination before this analysis. Role-specific residual conditioning was refined after reading code, and the optional visible/null preconditioner after identifying spectral-transfer risks. Existing project ideas already mention spectral observability and observational compatibility. No trained outcome was inspected to support this candidate; synthetic algebra probes were used to check feasibility. Proposal is exploratory, not preregistered or confirmatory.
