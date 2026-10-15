Separating Intrinsic Ambiguity from Estimation Uncertainty in Deep Generative Models for Linear Inverse Problems

    ![](/static/base/1.0.1/images/icons/smileybones-small.svg)
    arXiv is now an independent nonprofit!
    Learn more
    ×

    License: arXiv.org perpetual non-exclusive license

arXiv:2605.15050v1 [cs.LG] 14 May 2026

# Separating Intrinsic Ambiguity from Estimation Uncertainty in Deep Generative Models for Linear Inverse Problems

Yuxin Guo
††thanks: Corresponding author: yuxinguo@andrew.cmu.edu
Affiliation: Neuroscience Institute, Carnegie Mellon University

Affiliation: Machine Learning Department, Carnegie Mellon University

  
Dongrui Deng

Affiliation: Neuroscience Institute, Carnegie Mellon University

Affiliation: Machine Learning Department, Carnegie Mellon University

  
Pulkit Grover

Affiliation: Neuroscience Institute, Carnegie Mellon University

Affiliation: Department of Electrical and Computer Engineering, Carnegie Mellon University

###### Abstract

Recently, deep generative models have been used for posterior inference in inverse problems, including high-stakes applications in medical imaging and scientific discovery, where the uncertainty of a prediction can matter as much as the prediction itself. However, posterior uncertainty is difficult to interpret because it can mix ambiguity inherent to the forward operator with uncertainty propagated through inference. We introduce a structural decomposition of posterior uncertainty that isolates intrinsic ambiguity. A cascade formulation makes this ambiguity accessible for calibration analysis, enabling qualitative diagnostics and simulation-based calibration tests that reveal failure modes that remain hidden when models are selected by reconstruction quality alone. We first validate the approach on a Gaussian example with analytical posterior structure, then illustrate the decomposition on accelerated magnetic resonance imaging (MRI), and finally apply the calibration diagnostics to electroencephalography (EEG) source imaging.

## 1 Introduction

Inverse problems are common in fields such as natural sciences, medical imaging, and astronomy. These problems are often ill-posed, in the sense that the measurement is many-to-one, where distinct underlying states can produce the same measurement. Consequently, reconstruction uncertainty is not only induced by measurement noise, but also arises from the geometry of the forward operator itself. In such settings, the observations identify only a subset of the underlying system, while the remaining degrees of freedom are fundamentally unresolvable from the data.

Classical optimization methods address ill-posed inverse problems by introducing explicit constraints, such as regularization that impose sparsity, and are typically evaluated by reconstruction metrics such as mean squared error (MSE). More recently, generative models aim to sample plausible reconstructions from a learned data prior, and are additionally evaluated by distributional metrics, such as Fréchet Inception Distance (FID) [1]
and Contrastive Language–Image Pretraining (CLIP) score [2]. However, they are limited because they assess either the reconstruction fidelity or the distributional similarity between generated samples and real data — without examining whether the model faithfully captures the posterior uncertainty over all the possible reconstructions consistent with a given measurement.
This raises a fundamental question: how should the uncertainty represented by these posterior samples be interpreted? In particular, existing uncertainty estimates typically conflate two qualitatively distinct sources of variability — ambiguity that is intrinsic to the ill-posed problem, and uncertainty that arises from measurement noise. Answering this question is especially consequential in high-stakes applications, where a model must not only produce accurate reconstructions but also faithfully reflect how much of its uncertainty is irreducible.

In this paper, we propose a decomposition of posterior uncertainty in inverse problems that separates intrinsic ambiguity from uncertainty propagated by noisy measurements. Building on this decomposition, we utilize a cascade generative framework that makes these components directly inspectable. This, in turn, provides diagnostic metrics that can complement reconstruction loss in model selection and parameter tuning. We validate the framework on a Gaussian example with known analytical posterior structure, accelerated MRI for qualitative visualization, and EEG source imaging as a realistic large-scale ill-posed inverse problem. Rather than performing an exhaustive benchmark, we use representative generative models to illustrate how uncertainty can be distributed differently between intrinsic ambiguity and noise-propagated variability, and how these differences lead to distinct qualitative and quantitative failure modes. In practice, this distinction helps determine whether uncertainty is dominated by irreducible non-identifiability or by factors that could be mitigated through better measurements, modeling, or inference. The factorized formulation also yields a modular perspective on posterior modeling: identifiable inference and ambiguity modeling. In this paper, we focus on the latter as the central object for uncertainty calibration.

## 2 Background

Inverse Problems
Inverse problems arise across domains, including medical imaging, astronomy, and image restoration [3, 4, 5]. Classical methods address ill-posed inverse problems by imposing regularization or handcrafted constraints to select plausible reconstructions from an underdetermined solution set. More recently, generative methods have been used to learn distributions over possible reconstructions, including plug-and-play priors and diffusion posterior samplers [6, 7]. We refer to models that directly sample full reconstructions x^∼pθ​(x∣y)\hat{x}\sim p_{\theta}(x\mid y) as monolithic posterior models. Several diffusion inverse solvers further exploit the geometry of the forward operator. For example, DDRM [8] uses spectral structure to construct restoration samplers, while DDNM [9] uses a range–null decomposition to enforce measurement consistency and refine null-space content during reverse diffusion. Related null-space generative methods also learn conditional models for the unmeasured component in MRI [10]. These works use the geometry of the forward operator to construct posterior samples. We use the same geometric view for a different goal. The question is not only whether a model can generate plausible null-space components, but whether the induced conditional distribution over those components is calibrated. We therefore use the range–null view as a calibration tool rather than only as a restoration mechanism.

Posterior uncertainty and calibration. Recent work [11] systematically evaluates posterior uncertainty in plug-and-play diffusion solvers and shows that solvers with comparable reconstruction quality can produce qualitatively different posterior variance. This raises a calibration question beyond whether posterior variability is large or small, but which component of uncertainty should be calibrated, and what distribution should it be calibrated against? A scalar posterior variance alone does not answer this question. Simulation-based calibration (SBC) [12] instead tests whether posterior samples are statistically consistent with draws from the data-generating distribution. Across repeated simulations x∗∼p⁡(x)x^{*}\sim p(x) and y∼p⁡(y∣x∗)y\sim p(y\mid x^{*}), a calibrated sampler should assign the true draw x∗x^{*} a uniform rank among samples from the approximate posterior.

Our contribution
We first ask what posterior uncertainty is composed of in a linear inverse problem. Our decomposition identifies intrinsic ambiguity as the principal target of calibration, reflecting uncertainty that is irreducible under the forward operator. We then use a cascade formulation to make this ambiguity directly accessible for both qualitative and quantitative calibration analysis.

## 3 Decomposing Posterior Uncertainty

We consider the standard linear inverse problem

𝐲=𝐀𝐱+𝜺,\mathbf{y}=\mathbf{Ax}+\boldsymbol{\varepsilon},

(1)

where 𝐀∈ℝn×p\mathbf{A}\in\mathbb{R}^{n\times p} denotes the forward operator, 𝐱∈ℝp\mathbf{x}\in\mathbb{R}^{p} is the unknown signal, and 𝐲∈ℝn\mathbf{y}\in\mathbb{R}^{n} is the measurement. Throughout the inverse problems considered here, the measurement dimension is typically much smaller than the signal dimension, i.e., n≪pn\ll p, which makes solving for 𝐱\mathbf{x} ill-posed. The term ε\varepsilon denotes measurement noise, which is often modeled as Gaussian in practice.

In the linear setting, this decomposition is naturally induced by the geometry of the forward operator. Let
𝐀=𝐔​𝚺​𝐕⊤\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top} be the singular value decomposition (SVD) of 𝐀\mathbf{A}, with r=rank⁡(𝐀)r=\mathrm{rank}(\mathbf{A}), and partition the right singular vectors as 𝐕=[𝐕r,𝐕n]\mathbf{V}=[\mathbf{V}_{r},\mathbf{V}_{n}], where 𝐕r∈ℝp×r\mathbf{V}_{r}\in\mathbb{R}^{p\times r} and 𝐕n∈ℝp×(p−r)\mathbf{V}_{n}\in\mathbb{R}^{p\times(p-r)}. The columns of 𝐕r\mathbf{V}_{r} span the identifiable subspace associated with nonzero singular values of 𝐀\mathbf{A}, and the columns of 𝐕n\mathbf{V}_{n} span Null⁡(𝐀)\mathrm{Null}(\mathbf{A}). Then any signal 𝒙\boldsymbol{x} can be written as

𝐱=𝐕r​𝜶+𝐕n​𝜷,\mathbf{x}=\mathbf{V}_{r}\boldsymbol{\alpha}+\mathbf{V}_{n}\boldsymbol{\beta},

(2)

where 𝜶∈ℝr\boldsymbol{\alpha}\in\mathbb{R}^{r} and 𝜷∈ℝp−r\boldsymbol{\beta}\in\mathbb{R}^{p-r} are coefficient vectors with respect to the bases 𝐕r\mathbf{V}_{r} and 𝐕n\mathbf{V}_{n}, respectively. This decomposition is deterministic for any fixed 𝐱\mathbf{x}.
In the posterior inference problem, however, 𝐱\mathbf{x} is treated as a random variable with distribution p⁡(𝐱∣𝐲)p(\mathbf{x}\mid\mathbf{y}), and therefore
the induced coefficients 𝜶=𝐕r⊤​𝐱\boldsymbol{\alpha}=\mathbf{V}_{r}^{\top}\mathbf{x}
and 𝜷=𝐕n⊤​𝐱\boldsymbol{\beta}=\mathbf{V}_{n}^{\top}\mathbf{x} are random variables as well.
Substituting the decomposition into the forward model gives

𝐲=𝐀⁡(𝐕r​𝜶+𝐕n​𝜷)+𝜺=𝐀𝐕r​𝜶+𝜺.\mathbf{y}=\mathbf{A}(\mathbf{V}_{r}\boldsymbol{\alpha}+\mathbf{V}_{n}\boldsymbol{\beta})+\boldsymbol{\varepsilon}=\mathbf{A}\mathbf{V}_{r}\boldsymbol{\alpha}+\boldsymbol{\varepsilon}.

(3)

Thus, the measurement depends only on 𝜶\boldsymbol{\alpha}, while 𝜷\boldsymbol{\beta} is invisible to the forward operator. In this sense, 𝜶\boldsymbol{\alpha} captures the measurement-identifiable component of 𝐱\mathbf{x}, whereas 𝜷\boldsymbol{\beta} is intrinsically ambiguous.

The posterior inference problem can be equivalently expressed in terms of the coefficient vectors (𝜶,𝜷)(\boldsymbol{\alpha},\boldsymbol{\beta}). Moreover, Eq. (3) shows that the likelihood satisfies
p⁡(𝐲∣𝜶,𝜷)=p⁡(𝐲∣𝜶)p(\mathbf{y}\mid\boldsymbol{\alpha},\boldsymbol{\beta})=p(\mathbf{y}\mid\boldsymbol{\alpha}), assuming measurement noise independent of the signal. Therefore, the decomposed model implies the conditional independence relation 𝜷⟂𝐲|𝜶\boldsymbol{\beta}\perp\mathbf{y}\mid\boldsymbol{\alpha}, under which the posterior factorizes as

p⁡(𝐱∣𝐲)=p⁡(𝜶,𝜷∣𝐲)=p⁡(𝜶∣𝐲)​p​(𝜷∣𝜶,𝐲)=p⁡(𝜶∣𝐲)​p​(𝜷∣𝜶).p(\mathbf{x}\mid\mathbf{y})=p(\boldsymbol{\alpha},\boldsymbol{\beta}\mid\mathbf{y})=p(\boldsymbol{\alpha}\mid\mathbf{y})\,p(\boldsymbol{\beta}\mid\boldsymbol{\alpha},\mathbf{y})=p(\boldsymbol{\alpha}\mid\mathbf{y})\,p(\boldsymbol{\beta}\mid\boldsymbol{\alpha}).

(4)

This factorization separates two qualitatively distinct forms of uncertainty. Uncertainty in p⁡(𝜶∣𝐲)p(\boldsymbol{\alpha}\mid\mathbf{y}) reflects noise and inference error, which can in principle be reduced by better measurements or stronger inference. By contrast, 𝜷\boldsymbol{\beta} is fundamentally unidentifiable from 𝐲\mathbf{y}, and its uncertainty at fixed
𝜶\boldsymbol{\alpha} is the object of our calibration analysis. We define the intrinsic ambiguity at the oracle-identifiable state 𝜶∗\boldsymbol{\alpha}^{*} as:

𝒜⁡(𝜶∗):=𝕍⁡(𝜷∣𝜶∗),\mathcal{A}(\boldsymbol{\alpha}^{*}):=\mathbb{V}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}),

where 𝕍\mathbb{V} denotes the covariance matrix. This is the irreducible uncertainty induced by information loss in the forward operator. In practice, however, standard monolithic posterior samplers only expose 𝕍⁡(𝜷∣𝐲)\mathbb{V}(\boldsymbol{\beta}\mid\mathbf{y}). By the law of total variance and the conditional independence
𝜷⟂𝐲|𝜶\boldsymbol{\beta}\perp\mathbf{y}\mid\boldsymbol{\alpha} from Eq. (4), this quantity satisfies

𝕍⁡(𝜷∣𝐲)\displaystyle\mathbb{V}(\boldsymbol{\beta}\mid\mathbf{y})
=𝔼p⁡(𝜶∣𝐲)[𝕍(𝜷∣𝜶,𝐲)]+𝕍p⁡(𝜶∣𝐲)(𝔼[𝜷∣𝜶,𝐲])\displaystyle=\mathbb{E}_{p(\boldsymbol{\alpha}\mid\mathbf{y})}\!\left[\mathbb{V}(\boldsymbol{\beta}\mid\boldsymbol{\alpha},\mathbf{y})\right]+\mathbb{V}_{p(\boldsymbol{\alpha}\mid\mathbf{y})}\!\left(\mathbb{E}[\boldsymbol{\beta}\mid\boldsymbol{\alpha},\mathbf{y}]\right)

=𝔼p⁡(𝜶∣𝐲)​[𝕍​(𝜷∣𝜶)]⏟posterior-averaged intrinsic ambiguity+𝕍p⁡(𝜶∣𝐲)​(𝔼​[𝜷∣𝜶])⏟propagated uncertainty from inferring ​𝜶\displaystyle=\underbrace{\mathbb{E}_{p(\boldsymbol{\alpha}\mid\mathbf{y})}\!\left[\mathbb{V}(\boldsymbol{\beta}\mid\boldsymbol{\alpha})\right]}_{\text{posterior-averaged intrinsic ambiguity}}+\underbrace{\mathbb{V}_{p(\boldsymbol{\alpha}\mid\mathbf{y})}\!\left(\mathbb{E}[\boldsymbol{\beta}\mid\boldsymbol{\alpha}]\right)}_{\text{propagated uncertainty from inferring }\boldsymbol{\alpha}}

(5)

In the noiseless limit, p⁡(𝜶∣𝐲)p(\boldsymbol{\alpha}\mid\mathbf{y}) collapses to a point mass at 𝜶∗\boldsymbol{\alpha}^{*}, the second term vanishes, and 𝕍⁡(𝜷∣𝐲)=𝒜⁡(𝜶∗)\mathbb{V}(\boldsymbol{\beta}\mid\mathbf{y})=\mathcal{A}(\boldsymbol{\alpha}^{*}). This indicates that in the idealized case with a noise-free measurement system that perfectly recovers the identifiable component, the remaining uncertainty is exactly the intrinsic ambiguity. This remaining uncertainty is not an inference error to be eliminated, but reflects the underdetermined nature of the inverse problem, and should therefore be calibrated rather than driven to zero. Hence, we identify the relevant calibration target as the conditional ambiguity distribution p⁡(𝜷∣𝜶∗)p(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}), which monolithic posterior models do not directly expose. In the next section, we show how a cascade formulation makes this explicitly accessible.

## 4 Methods

### 4.1 Cascade Model

Motivated by the factorization in Eq. (4), we model the posterior in two stages: a range model for the measurement-identifiable component and a null model for the intrinsically ambiguous component. The resulting cascade architecture is shown in Figure 1, and the corresponding graphical model interpretation is shown in Appendix A.

Range model
The range model approximates the conditional object p⁡(𝜶∣𝐲)p(\boldsymbol{\alpha}\mid\mathbf{y}) and is responsible for inference of the measurement-identifiable component. In the supervised training setting considered here, oracle 𝜶∗\boldsymbol{\alpha}^{*} is available because the ground-truth 𝐱\mathbf{x} is known and can be projected onto the identifiable subspace. During training, 𝜶∗\boldsymbol{\alpha}^{*} serves as the supervision target for the range model. At inference time, only 𝐲\mathbf{y} is observed, and the model produces an estimate or distribution over 𝜶\boldsymbol{\alpha}. This stage may be implemented either probabilistically or deterministically, with the deterministic case viewed as a point-mass approximation at 𝜶^​(𝐲)\hat{\boldsymbol{\alpha}}(\mathbf{y}).

Null model
The null model approximates p⁡(𝜷∣𝜶)p(\boldsymbol{\beta}\mid\boldsymbol{\alpha}) and is responsible for the conditional generation of the intrinsically ambiguous component. During training, oracle 𝜶∗\boldsymbol{\alpha}^{*} is used as the conditioning input and the corresponding 𝜷∗\boldsymbol{\beta}^{*} as the target output. This gives direct access to the model conditional pϕnull​(𝜷∣𝜶∗)p^{\mathrm{null}}_{\boldsymbol{\phi}}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}), which is the key object for calibration analysis of intrinsic ambiguity. Expressive generative modeling is most critical in this stage, since the intrinsic ambiguity that remains after fixing 𝜶∗\boldsymbol{\alpha}^{*} must be represented here.

Inference and Evaluation
At inference time, posterior samples are generated in cascade form by first obtaining α^\hat{\alpha} from the range model and then sampling 𝜷^∼pϕnull​(𝜷∣𝜶^)\hat{\boldsymbol{\beta}}\sim p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\hat{\boldsymbol{\alpha}}). The final prediction 𝐱^\hat{\mathbf{x}} is reconstructed according to Eq. (2). During evaluation, oracle 𝜶∗\boldsymbol{\alpha}^{*} from the test set is used only for calibration diagnostics, whereas reconstruction quality is assessed on end-to-end cascade predictions formed from inferred 𝜶^\hat{\boldsymbol{\alpha}} and sampled 𝜷^\hat{\boldsymbol{\beta}}.

𝐲\mathbf{y}Range Modelq𝜽range​(𝜶∣𝒚)q_{\boldsymbol{\theta}}^{\mathrm{range}}(\boldsymbol{\alpha}\mid\boldsymbol{y})𝜶^\hat{\boldsymbol{\alpha}}Null Modelpϕnull​(𝜷∣𝜶)p_{\phi}^{\mathrm{null}}(\boldsymbol{\beta}\mid\boldsymbol{\alpha})𝜷^\hat{\boldsymbol{\beta}}Reconstruction𝐱^\hat{\mathbf{x}}𝜶∗\boldsymbol{\alpha}^{*}Posterior Calibration Diagnostics

*Figure 1: Cascade architecture. The null model sits at the intersection of two operational paths. Inference (blue, horizontal): given 𝐲\mathbf{y}, the range model produces 𝜶^\hat{\boldsymbol{\alpha}} and the null model produces 𝜷^|𝜶^\hat{\boldsymbol{\beta}}\mid\hat{\boldsymbol{\alpha}}; reconstruction combines both to yield 𝐱^\hat{\mathbf{x}}. Diagnostic (orange dashed, vertical): the null model is queried at oracle 𝜶∗\boldsymbol{\alpha}^{*} to expose pϕnull​(𝜷∣𝜶∗)p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}) for posterior calibration. At training time, both modules use oracle 𝜶∗\boldsymbol{\alpha}^{*}: the range model on (𝐲,𝜶∗)(\mathbf{y},\boldsymbol{\alpha}^{*}) pairs, the null model on (𝜶∗,𝜷∗)(\boldsymbol{\alpha}^{*},\boldsymbol{\beta}^{*}) pairs.*

### 4.2 Posterior Calibration Metric

Simulation-based calibration.
The decomposition in Section 3 identifies pϕnull​(𝜷∣𝜶∗)p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}) as the relevant object for assessing whether a model is well calibrated in terms of intrinsic ambiguity. Here oracle 𝜶∗\boldsymbol{\alpha}^{*} is obtained from the test set and is used only for calibration diagnostics. Ideally, one would like to directly compare the full model distribution pϕnull​(𝜷∣𝜶∗)p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}) against the corresponding data-distribution p⁡(𝜷∣𝜶∗)p(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}). In practice, however, direct calibration in this high-dimensional space is intractable. We therefore use simulation-based calibration (SBC), a diagnostic for posterior calibration based on repeated simulation from the generative model [12], recently adopted for amortized diffusion posteriors in simulation-based inference [13]. Under calibration, a true draw 𝜷∗∼p⁡(𝜷∣𝜶∗)\boldsymbol{\beta}^{*}\sim p(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}) should be statistically indistinguishable from samples 𝜷~(1),…,𝜷~(L)∼pϕnull​(𝜷∣𝜶∗)\tilde{\boldsymbol{\beta}}^{(1)},\dots,\tilde{\boldsymbol{\beta}}^{(L)}\sim p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}). Accordingly, in SBC, the rank of a chosen scalar summary T⁡(𝜷∗)T(\boldsymbol{\beta}^{*}) among {T⁡(𝜷~(ℓ))}ℓ=1L\{T(\tilde{\boldsymbol{\beta}}^{(\ell)})\}_{\ell=1}^{L} should be uniformly distributed. Deviations from uniformity indicate miscalibration. This provides a necessary but not sufficient condition for correct posterior calibration.

Since SBC requires scalar test statistics, we evaluate calibration through interpretable functionals of the ambiguous component 𝜷\boldsymbol{\beta}, rather than attempting to diagnose calibration separately in every coordinate. We introduce the Euclidean norm ‖𝜷‖2\|\boldsymbol{\beta}\|_{2} and the peak-to-total ratio ‖𝜷‖∞/‖𝜷‖2\|\boldsymbol{\beta}\|_{\infty}/\|\boldsymbol{\beta}\|_{2} as our test statistics (see Appendix F for a discussion on the challenges of per-dimensional SBC in high dimensions). In particular, the statistic ‖𝜷‖2\|\boldsymbol{\beta}\|_{2} captures the overall magnitude of the measurement-invisible component, while ‖𝜷‖∞/‖𝜷‖2\|\boldsymbol{\beta}\|_{\infty}/\|\boldsymbol{\beta}\|_{2} characterizes how that magnitude is distributed across dimensions. We use ‖𝜷‖2\|\boldsymbol{\beta}\|_{2} rather than ‖𝜷‖22\|\boldsymbol{\beta}\|_{2}^{2} simply for consistency with the second statistic. For SBC, both choices are equivalent since they yield identical ranks. The squared norm nevertheless provides a useful interpretation:
𝔼⁡[‖𝜷‖22∣𝜶∗]=‖𝔼⁡[𝜷∣α∗]‖22+tr⁡(𝕍⁡(𝜷∣𝜶∗))\mathbb{E}\!\left[\|\boldsymbol{\beta}\|_{2}^{2}\mid\boldsymbol{\alpha}^{*}\right]=\left\|\mathbb{E}\!\left[\boldsymbol{\beta}\mid\boldsymbol{}\alpha^{*}\right]\right\|_{2}^{2}+\operatorname{tr}\!\left(\mathbb{V}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*})\right). Thus, when 𝔼⁡[𝜷∣𝜶∗]=0\mathbb{E}[\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}]=0, ‖𝜷‖22\|\boldsymbol{\beta}\|_{2}^{2} reduces in expectation to the total conditional variance across dimensions. Together, these two statistics probe complementary aspects of whether pϕnull​(𝜷∣𝜶∗)p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}) is calibrated to the corresponding data-generating distribution p⁡(𝜷∣𝜶∗)p(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}).

Qualitative diagnostic.
SBC provides a quantitative test of calibration but requires repeated simulation across many values of 𝜶∗\boldsymbol{\alpha}^{*}. As a complementary qualitative diagnostic, the cascade also exposes an intrinsic ambiguity map in the original signal space. Given an oracle identifiable component 𝜶∗\boldsymbol{\alpha}^{*}, we draw KK samples 𝜷(k)∼pϕnull​(𝜷∣𝜶∗)\boldsymbol{\beta}^{(k)}\sim p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}), reconstruct them in the original coordinates as 𝐱(k)=𝐕𝐫​𝜶∗+𝐕𝐧​𝜷(k)\mathbf{x}^{(k)}=\mathbf{V_{r}}\boldsymbol{\alpha}^{*}+\mathbf{V_{n}}\boldsymbol{\beta}^{(k)}, and compute the per-coordinate empirical variance

𝕍^k​[𝐱(k)]→K→∞diag⁡(𝐕𝐧​𝕍​(𝜷∣𝜶∗)​𝐕𝐧⊤).\widehat{\mathbb{V}}_{k}[\mathbf{x}^{(k)}]\;\xrightarrow{K\to\infty}\;\mathrm{diag}\!\left(\mathbf{V_{n}}\,\mathbb{V}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*})\,\mathbf{V_{n}}^{\top}\right).

(6)

We refer to this quantity as the intrinsic ambiguity map. In settings where the original-space coordinate system is shared across test cases (e.g., a fixed cortical mesh in EEG), these maps may also be averaged across 𝜶∗∼ptest\boldsymbol{\alpha}^{*}\sim p_{\mathrm{test}} to obtain a population-level view. When coordinates differ across examples (e.g., MRI slices or anatomies), the map is inspected on a per-example basis.

A calibrated null model should place intrinsic ambiguity in geometrically plausible regions of the signal space, providing a useful visual sanity check for domain experts. For example, in accelerated MRI one expects higher ambiguity in structures poorly constrained by the retained Fourier coefficients, while in EEG one expects larger ambiguity in deep or midline brain sources. This diagnostic is complementary to SBC: the spatial map reflects *marginal* per-coordinate variance, whereas SBC probes scalar summary statistics of the *joint* conditional distribution pϕnull​(𝜷∣𝜶∗)p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}), such as total energy or concentration.

## 5 Experiments

### 5.1 Gaussian Analytical Example

We first study a controlled Gaussian example in which the relevant conditional distributions are analytically tractable. This provides a sanity check for both the proposed decomposition and the calibration diagnostics. Unlike in realistic inverse problems, the oracle ambiguity distribution p⁡(𝜷∣𝜶∗)p(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}) is available in closed form in this setting, allowing direct comparison between the learned null model and the true conditional target.

*Figure 2: Left: Per-dimension 𝜷\boldsymbol{\beta} variance calibration. Each dot corresponds to one of the q=64q=64 𝜷\boldsymbol{\beta} dimensions and compares the empirical variance of DDPM samples from pϕnull​(𝜷∣𝜶∗)p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}) with the corresponding analytical variance from diag⁡(𝚺𝜼)\mathrm{diag}(\boldsymbol{\Sigma}_{\boldsymbol{\eta}}). The dashed line indicates equality. The mean empirical-to-analytical variance ratio across dimensions is 0.9850.985, indicating reasonable calibration. Right: SBC diagnostics. Rank histograms for the test statistics ‖𝜷‖2\|\boldsymbol{\beta}\|_{2} and ‖𝜷‖∞/‖𝜷‖2\|\boldsymbol{\beta}\|_{\infty}/\|\boldsymbol{\beta}\|_{2} under the learned DDPM null model. Ranks are normalized to [0,1][0,1] and plotted as densities, so the ideal calibrated density is 11. The gray band shows the per-bin 99%99\% binomial confidence interval under uniform ranks. Histograms are computed with 200200 posterior samples for each of the 500500 test cases.*

We generate data as a simple linear Gaussian instantiation of the factorization in Eq. (4), where 𝒚\boldsymbol{y} depends only on 𝜶\boldsymbol{\alpha} by construction, so that 𝜷⟂𝒚|𝜶\boldsymbol{\beta}\perp\boldsymbol{y}\mid\boldsymbol{\alpha} holds exactly (see Appendix C for parameter construction and
training details):

𝜶\displaystyle\boldsymbol{\alpha}
∼𝒩⁡(0,𝐈r),\displaystyle\sim\mathcal{N}(0,\mathbf{I}_{r}),

(7)

𝜷|𝜶\displaystyle\boldsymbol{\beta}\mid\boldsymbol{\alpha}
∼𝒩⁡(𝐂​𝜶,𝚺𝜼),\displaystyle\sim\mathcal{N}(\mathbf{C}\boldsymbol{\alpha},\boldsymbol{\Sigma}_{\boldsymbol{\eta}}),

(8)

𝒚|𝜶\displaystyle\boldsymbol{y}\mid\boldsymbol{\alpha}
∼𝒩⁡(𝐀​𝜶,σy2​𝐈n),\displaystyle\sim\mathcal{N}(\mathbf{A}\boldsymbol{\alpha},\sigma_{y}^{2}\mathbf{I}_{n}),

(9)

where 𝜶∈ℝr\boldsymbol{\alpha}\in\mathbb{R}^{r} is the identifiable component, 𝜷∈ℝq\boldsymbol{\beta}\in\mathbb{R}^{q} is the ambiguous component, and 𝐲∈ℝn\mathbf{y}\in\mathbb{R}^{n} is the measurement. Here 𝐀∈ℝn×r\mathbf{A}\in\mathbb{R}^{n\times r} is the forward operator on the identifiable component, 𝐂∈ℝq×r\mathbf{C}\in\mathbb{R}^{q\times r} controls the dependence of the ambiguous component on 𝜶\boldsymbol{\alpha}, and 𝚺𝜼∈ℝq×q\boldsymbol{\Sigma}_{\boldsymbol{\eta}}\in\mathbb{R}^{q\times q} specifies the intrinsic ambiguity covariance. By construction, the measurement depends only on 𝜶\boldsymbol{\alpha}, while 𝜷\boldsymbol{\beta} remains unobserved. In our experiments, we use r=32r=32, q=64q=64, and n=32n=32. The oracle ambiguity distribution is therefore 𝜷|𝜶∗∼𝒩⁡(𝐂​𝜶∗,𝚺𝜼)\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}\sim\mathcal{N}(\mathbf{C}\boldsymbol{\alpha}^{*},\boldsymbol{\Sigma}_{\boldsymbol{\eta}}), with conditional covariance 𝕍⁡(𝜷∣𝜶∗)=𝚺𝜼\mathbb{V}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*})=\boldsymbol{\Sigma}_{\boldsymbol{\eta}}.

To ensure a nontrivial calibration target, 𝚺𝜼\boldsymbol{\Sigma}_{\boldsymbol{\eta}}
is chosen with full off-diagonal structure, requiring the null model to match not only marginal variances but also cross-dimensional correlations. We then train a conditional DDPM [14] null model as a representative expressive sampler for pϕnull​(𝜷∣𝜶)p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}) and evaluate it at oracle 𝜶∗\boldsymbol{\alpha}^{*}. Figure 2 shows close agreement between the empirical per-dimension variance from DDPM samples and the analytical conditional variance, while the SBC histograms remain close to uniform for the test statistics ‖𝜷‖2\|\boldsymbol{\beta}\|_{2} and ‖𝜷‖∞/‖𝜷‖2\|\boldsymbol{\beta}\|_{\infty}/\|\boldsymbol{\beta}\|_{2}. This controlled example therefore serves as a sanity check: in a setting where the conditional ambiguity distribution is known analytically, a DDPM null model is reasonably calibrated against the exact target, and the proposed diagnostics correctly reflect this behavior.

### 5.2 Accelerated MRI

We use accelerated MRI as a visually intuitive setting to illustrate the posterior variance decomposition introduced in Section 3. In this setting, a standard monolithic posterior model would expose only the total posterior uncertainty, which conflates intrinsic ambiguity with uncertainty propagated from noisy inference. Because the identifiable component is available in closed form under Fourier undersampling, accelerated MRI allows these two contributions to be separated explicitly and visualized in image space.

The forward operator is 𝐀=𝐌𝐅\mathbf{A}=\mathbf{M}\mathbf{F}, where 𝐅∈ℂNv×Nv\mathbf{F}\in\mathbb{C}^{N_{v}\times N_{v}} is the two-dimensional discrete Fourier transform and 𝐌∈ℝNv×Nv\mathbf{M}\in\mathbb{R}^{N_{v}\times N_{v}} is a diagonal binary mask retaining a subset of Fourier coefficients. Measurements are 𝐲σ=𝐌𝐅𝐱+𝜺σ\mathbf{y}_{\sigma}=\mathbf{M}\mathbf{F}\mathbf{x}+\boldsymbol{\varepsilon}_{\sigma}, where 𝜺σ∼𝒩⁡(0,σ2​𝐈)\boldsymbol{\varepsilon}_{\sigma}\sim\mathcal{N}(0,\sigma^{2}\mathbf{I}) denotes measurement noise in k-space. For a clean image 𝐱\mathbf{x}, we define the oracle identifiable component as 𝐱α∗=𝐅−1​𝐌𝐅𝐱\mathbf{x}_{\alpha}^{*}=\mathbf{F}^{-1}\mathbf{M}\mathbf{F}\mathbf{x} and the null component as 𝐱β=𝐱−𝐱α∗\mathbf{x}_{\beta}=\mathbf{x}-\mathbf{x}_{\alpha}^{*}. This is the identifiable/null decomposition from Section 3, expressed directly in image space rather than in singular-vector coordinates. By construction, 𝐀𝐱𝜷=𝐌𝐅⁡(𝐱−𝐅−1​𝐌𝐅𝐱)=𝟎\mathbf{A}\mathbf{x}_{\boldsymbol{\beta}}=\mathbf{M}\mathbf{F}(\mathbf{x}-\mathbf{F}^{-1}\mathbf{M}\mathbf{F}\mathbf{x})=\mathbf{0}, so 𝐱𝜷\mathbf{x}_{\boldsymbol{\beta}} lies in the null space of 𝐀\mathbf{A}. In the noise-free case, 𝐱α∗=𝐅−1​𝐲\mathbf{x}_{\alpha}^{*}=\mathbf{F}^{-1}\mathbf{y}. Under noisy measurements, however, the available closed-form estimate is 𝐱^α=𝐅−1​𝐲σ=𝐱α∗+𝐅−1​𝜺σ\hat{\mathbf{x}}_{\alpha}=\mathbf{F}^{-1}\mathbf{y}_{\sigma}=\mathbf{x}_{\mathbf{\alpha}^{*}}+\mathbf{F}^{-1}\boldsymbol{\varepsilon}_{\sigma}.

![Refer to caption](2605.15050v1/mri_results.png)

*Figure 3: Posterior uncertainty decomposition for accelerated MRI. Panel (a) shows a noise-level sweep over 50 held-out slices, with each point estimated by Monte Carlo sampling. Intrinsic ambiguity remains constant, while total null uncertainty increases with k-space noise. The shaded gap indicates noise-propagated uncertainty. Panel (b) shows a representative ground-truth slice. Panel (c) shows intrinsic ambiguity, estimated as 𝕍pϕ​(𝐱β∣𝐱α∗)\mathbb{V}_{p_{\phi}}(\mathbf{x}_{\beta}\mid\mathbf{x}_{\alpha}^{*}) by fixing the oracle identifiable component. Panel (d) shows total null uncertainty at σ=0.2\sigma=0.2, estimated by marginalizing over noisy identifiable estimates x^α=F−1​yσ\hat{x}_{\alpha}=F^{-1}y_{\sigma}. Both maps share the same color scale and report per-pixel variance in image intensity.*

Because 𝐱𝜶∗\mathbf{x}_{\boldsymbol{\alpha}}^{*} is available in closed form for clean training images, we do not train a separate range model and only train a conditional diffusion model pϕnull​(𝐱𝜷∣𝐱𝜶)p_{\phi}^{\mathrm{null}}(\mathbf{x}_{\boldsymbol{\beta}}\mid\mathbf{x}_{\boldsymbol{\alpha}}) using oracle training pairs (𝐱𝜶∗,𝐱𝜷∗)(\mathbf{x}_{\boldsymbol{\alpha}}^{*},\mathbf{x}_{\boldsymbol{\beta}}^{*}). This gives a clean setting in which propagated uncertainty is controlled directly by measurement noise, rather than by details of range-model training. For a unitary Fourier transform 𝐅\mathbf{F}, and under a Lipschitz assumption on the conditional null mean (i.e. m⁡(𝐱𝜶)=𝔼pϕ​[𝐱𝜷∣𝐱𝜶]m(\mathbf{x}_{\boldsymbol{\alpha}})=\mathbb{E}_{p_{\phi}}[\mathbf{x}_{\boldsymbol{\beta}}\mid\mathbf{x}_{\boldsymbol{\alpha}}]), the propagated uncertainty term in Eq. (5) is bounded by O⁡(σ2)O(\sigma^{2}); see Appendix E.

We use single-coil knee images from the NYU fastMRI dataset [15, 16] with an undersampling mask retaining 25% of Fourier coefficients. Figure 3 visualizes the model-predicted uncertainty decomposition. The noise-level sweep in Fig. 3(a) summarizes this decomposition across held-out slices by averaging the per-pixel variance over all pixels and slices. Intrinsic ambiguity remains approximately flat as the k-space noise level increases. This is expected because the intrinsic ambiguity map defined in Section 4.2 conditions on the oracle identifiable component 𝐱𝜶∗\mathbf{x}_{\boldsymbol{\alpha}}^{*}, and is therefore independent of measurement noise. It reflects ambiguity induced by the forward operator and the data distribution. In contrast, total null uncertainty increases as measurement noise corrupts the identifiable estimate 𝐱^α=𝐅−1​𝐲\hat{\mathbf{x}}_{\alpha}=\mathbf{F}^{-1}\mathbf{y} and propagates into the conditional null distribution. The shaded region visualizes this noise-propagated uncertainty.

Figures 3(b)–(d) show this decomposition spatially for a representative slice. Panel (b) shows the ground-truth image. Panel (c) shows the model-predicted intrinsic ambiguity map described in Section 4.2, corresponding to 𝕍pϕ​(𝐱𝜷∣𝐱𝜶∗)\mathbb{V}_{p_{\phi}}(\mathbf{x}_{\boldsymbol{\beta}}\mid\mathbf{x}_{\boldsymbol{\alpha}}^{*}). Panel (d) shows the model-predicted total null uncertainty, corresponding to 𝕍pϕ​(𝐱𝜷∣𝐲)\mathbb{V}_{p_{\phi}}(\mathbf{x}_{\boldsymbol{\beta}}\mid\mathbf{y}) after marginalizing over noisy identifiable estimates 𝐱^𝜶\hat{\mathbf{x}}_{\boldsymbol{\alpha}}. If only panel (d) were available, as in a monolithic posterior sampler, the source of uncertainty would be unclear. The decomposition gives this uncertainty a physical interpretation. Panel (c) reflects uncertainty due to information loss from the forward operator and is therefore tied to the k-space undersampling protocol. The additional uncertainty in panel (d) reflects noise propagated through 𝐱^𝜶\hat{\mathbf{x}}_{\boldsymbol{\alpha}}, which can in principle be reduced by denoising, repeated acquisition, or better measurement.

### 5.3 EEG Source Imaging

We next evaluate our calibration analysis and cascade framework on EEG source imaging, a severely ill-posed inverse problem that provides a stress test for posterior uncertainty. The task is to recover high-resolution voxel-wise brain activity from sparse scalp measurements [17]. Since many distinct source configurations can explain the same EEG measurement, classical methods rely heavily on regularization, thereby imposing strong inductive biases on the reconstructions. We consider the standard linear forward model, 𝐲=𝐀𝐱+𝜺\mathbf{y}=\mathbf{Ax}+\boldsymbol{\varepsilon}, where 𝐱∈ℝNv\mathbf{x}\in\mathbb{R}^{N_{v}} denotes voxel-wise source activity, 𝐲∈ℝNs\mathbf{y}\in\mathbb{R}^{N_{s}} denotes scalp measurements, and 𝐀∈ℝNs×Nv\mathbf{A}\in\mathbb{R}^{N_{s}\times N_{v}} is the leadfield matrix. Because Ns≪NvN_{s}\ll N_{v} in practice, the forward operator has a large null space, and the decomposition from Section 3 is especially relevant in EEG source imaging: only a low-dimensional component of source activity is measurement-identifiable, while a large component remains intrinsically ambiguous.

We construct a single-subject head model from the MNE sample dataset [18], based on a realistic MRI-derived anatomy. The leadfield matrix is computed via a three-layer boundary element method (BEM) on a cortical source space with fixed-orientation dipoles along the cortical normal, yielding 𝐀∈ℝ60×4699\mathbf{A}\in\mathbb{R}^{60\times 4699} (i.e. 60 EEG sensors and 4699 cortical sources). The linear forward model is exact in the quasi-static regime of Maxwell’s equations [19]. Source activity is simulated as spatially localized Gaussian patches with random central vertices, widths σ∼Uniform⁡(5,20)\sigma\sim\mathrm{Uniform}(5,20) mm, amplitudes ∼Uniform⁡(0.5,2.0)\sim\mathrm{Uniform}(0.5,2.0), and signs flipped with probability 0.50.5. The number of patches is sampled uniformly from {1,2,3}\{1,2,3\} for each example. Measurements are generated with added Gaussian noise calibrated to signal-to-noise ratio (SNR) =5=5.

Following the cascade strategy, we independently train multiple range and null models, then evaluate their combinations, as described in Section 4.1. The range model is instantiated as a DDPM, variational autoencoder (VAE), or multilayer perceptron (MLP), while the null model is instantiated as either a DDPM or VAE. We evaluate all range–null combinations on 450450 held-out test cases. The main text focuses on two cascades that share the same DDPM range model but differ in the null model: *cascade A* pairs the DDPM range model with a DDPM null model, whereas *cascade B* pairs the DDPM range model with a VAE null model at Kullback–Leibler (KL) weight λKL=1.0\lambda_{\mathrm{KL}}=1.0 [20]. We use the posterior sample mean over 200200 samples as the point reconstruction, which is the standard Bayes estimator under squared-error loss and reflects the average behavior of the trained model. Reconstruction quality is measured by Pearson correlation between the posterior mean reconstruction and the ground-truth source signal, a measurement of spatial pattern similarity commonly used in EEG source-imaging evaluations [21]. The two cascades achieve reasonable reconstructions even under the challenging three-patch source configuration, as shown in Fig. 4, and have nearly identical Pearson correlation with the ground truth (0.6980.698 vs. 0.6930.693). We therefore ask whether models that are nearly indistinguishable under standard reconstruction metrics differ in their calibration of intrinsic ambiguity. Remaining range–null combinations and architecture details are deferred to Appendix D.

Qualitative diagnostic. We plot the averaged intrinsic ambiguity map (calculated according to Section 4.2) on the cortical surface as a qualitative calibration check, contrasting the DDPM and VAE null models. As shown in Figure 5, the DDPM null produces larger intrinsic ambiguity at deeper brain voxels and along the longitudinal fissure, qualitatively consistent with the well-known reduced sensitivity of EEG to deep and medial sources [22, 23]. The VAE null, by contrast, produces a uniformly low-variance profile across all brain regions, indicating that the model assigns negligible intrinsic ambiguity regardless of source location. This visual under-dispersion already signals miscalibration of pϕnull​(𝜷∣𝜶∗)p^{\mathrm{null}}_{\phi}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}), motivating a quantitative test.

![Refer to caption](2605.15050v1/EEG_recon.png)

*Figure 4: Example source reconstruction for a held-out three-patch test case. Both cascades produce visually similar posterior mean reconstructions and recover the dominant spatial pattern of the ground truth, motivating our comparison of their intrinsic-ambiguity calibration beyond standard reconstruction quality. Color indicates signed source amplitude.*

![Refer to caption](2605.15050v1/EEG_intrinsic_map.png)

*Figure 5: Averaged intrinsic ambiguity map on the cortical surface. Per-voxel intrinsic ambiguity map averaged across 450450 test cases (M=200M=200 posterior samples per case). Left: DDPM null model produces variance concentrated along the cortical midline and at deep medial regions. Right: VAE null model produces variance that is uniformly low across the cortex. Both models share a linear color scale capped at the 95th percentile of the DDPM map.*

*Figure 6: Similar reconstruction does not imply similar calibration of intrinsic ambiguity. With the DDPM range model fixed, the DDPM null and VAE null models achieve similar posterior-mean reconstruction quality, yet their null-space SBC histograms differ substantially. Left: posterior quantile of the true null-space energy ‖β∗‖2\|\beta^{*}\|_{2}. Right: posterior quantile of the true peak-to-total ratio ‖β∗‖∞/‖β∗‖2\|\beta^{*}\|_{\infty}/\|\beta^{*}\|_{2}. The DDPM null has higher null-space energy, while the VAE null is under-energetic. Both models are more isotropic than the true data distribution.*

SBC Quantitative diagnostic. Figure 6 reports SBC histograms at oracle 𝜶∗\boldsymbol{\alpha}^{*} for the DDPM and VAE null models, using the geometric test statistics ‖𝜷‖2\|\boldsymbol{\beta}\|_{2} and ‖𝜷‖∞/‖𝜷‖2\|\boldsymbol{\beta}\|_{\infty}/\|\boldsymbol{\beta}\|_{2} introduced in Section 4.2. Neither null model yields uniform histograms; the deviations exceed the 99%99\% confidence interval expected under perfect calibration, revealing two qualitatively distinct failure modes. On total energy ‖𝜷‖2\|\boldsymbol{\beta}\|_{2}, the two models fail in opposite directions: the DDPM null concentrates rank mass near zero, indicating that its samples systematically carry more energy than the true data-generating distribution, while the VAE null concentrates rank mass near one, indicating collapse toward values smaller than the truth. On peak-to-total ‖𝜷‖∞/‖𝜷‖2\|\boldsymbol{\beta}\|_{\infty}/\|\boldsymbol{\beta}\|_{2}, both null models concentrate rank mass near one, indicating samples that are more diffuse than the true patch-structured 𝜷\boldsymbol{\beta}. The VAE failure on ‖𝜷‖2\|\boldsymbol{\beta}\|_{2} matches the dark cortex map in Figure 5: a model that systematically samples small 𝜷\boldsymbol{\beta} values produces both small ‖𝜷‖2\|\boldsymbol{\beta}\|_{2} relative to truth and small per-vertex source-space variance. The DDPM failure, by contrast, is invisible in the cortex map — the spatial pattern is plausible — but clearly visible in the SBC histograms, demonstrating that SBC catches calibration failures that qualitative diagnostic alone cannot. Crucially, the two null models achieve nearly identical reconstruction quality when paired with the same range model. This instantiates the cancellation pathology anticipated by the decomposition of Section 3: miscalibration of the intrinsic ambiguity distribution can persist even when end-to-end reconstruction is comparable, and the cascade structure is precisely what makes this miscalibration visible.

## 6 Discussion

Our experiments demonstrate that evaluating models merely by reconstruction quality fails to recognize important uncertainty miscalibration. In the EEG experiment, cascades with nearly identical reconstruction quality exhibited different uncertainty calibration. The cascade formulation addresses this by turning intrinsic ambiguity into an evaluable target. Its modular structure further allows each stage to be assessed independently: the range model through measurement consistency, and the null model through uncertainty calibration. The range–null combination results in Appendix D support this separation, showing that reconstruction quality varies primarily with the range model, whereas null-model choice primarily changes calibration behavior. Although the cascade imposes a particular factorization for evaluation, it does not constrain the model class used in either stage. Architectural advances in posterior sampling, diffusion models, or domain-specific networks can be incorporated as range or null models.

We recognize the following limitations in our work. First, direct calibration of the intrinsic ambiguity distribution requires paired (𝐱,𝐲)(\mathbf{x},\mathbf{y}) data, so the diagnostic applies most directly to supervised settings rather than inference-only plug-and-play settings. Second, SBC provides a necessary but not sufficient check of calibration. Its sensitivity depends on the scalar statistics, which should be designed for the downstream application. Third, our current decomposition is most direct for known linear forward operators, where the geometry of the forward operator 𝐀\mathbf{A} gives an explicit range–null split. Extending the diagnostic to nonlinear or operator-uncertain inverse problems would require replacing this closed-form split with an approximate or learned notion of identifiable and ambiguous components. Beyond these limitations, an interesting direction is to investigate whether the cascade structure can be extended to amortized or foundation-model settings, where paired supervision may be partially available. Better-calibrated uncertainty estimates also have potential societal relevance in high-stakes inverse problems such as medical imaging. Distinguishing irreducible ambiguity from reducible inference uncertainty could support more reliable uncertainty reporting and guide data acquisition, although clinical or scientific deployment would require task-specific validation and domain oversight.

## References

- [1]
M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter (2017)

Gans trained by a two time-scale update rule converge to a local nash equilibrium.

Advances in neural information processing systems 30.

Cited by: §1.

- [2]
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021)

Learning transferable visual models from natural language supervision.

In International conference on machine learning,

pp. 8748–8763.

Cited by: §1.

- [3]
M. Lustig, D. Donoho, and J. M. Pauly (2007)

Sparse mri: the application of compressed sensing for rapid mr imaging.

Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 58 (6), pp. 1182–1195.

Cited by: §2.

- [4]
J. Starck, E. Pantin, and F. Murtagh (2002)

Deconvolution in astronomy: a review.

Publications of the Astronomical Society of the Pacific 114 (800), pp. 1051–1069.

Cited by: §2.

- [5]
C. Dong, C. C. Loy, K. He, and X. Tang (2015)

Image super-resolution using deep convolutional networks.

IEEE transactions on pattern analysis and machine intelligence 38 (2), pp. 295–307.

Cited by: §2.

- [6]
H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye (2022)

Diffusion posterior sampling for general noisy inverse problems.

arXiv preprint arXiv:2209.14687.

Cited by: §2.

- [7]
G. Daras, H. Chung, C. Lai, Y. Mitsufuji, J. C. Ye, P. Milanfar, A. G. Dimakis, and M. Delbracio (2024)

A survey on diffusion models for inverse problems.

arXiv preprint arXiv:2410.00083.

Cited by: §2.

- [8]
B. Kawar, M. Elad, S. Ermon, and J. Song (2022)

Denoising diffusion restoration models.

Advances in neural information processing systems 35, pp. 23593–23606.

Cited by: §2.

- [9]
Y. Wang, J. Yu, and J. Zhang (2022)

Zero-shot image restoration using denoising diffusion null-space model.

arXiv preprint arXiv:2212.00490.

Cited by: §2.

- [10]
J. Wen, R. Ahmad, and P. Schniter (2023)

A conditional normalizing flow for accelerated multi-coil mr imaging.

Proceedings of machine learning research 202, pp. 36926.

Cited by: §2.

- [11]
X. Qiu, T. Yang, Z. Liu, G. Wang, and L. Shen (2026)

Benchmarking uncertainty quantification of plug-and-play diffusion priors for inverse problems solving.

arXiv preprint arXiv:2602.04189.

Cited by: §2.

- [12]
S. Talts, M. Betancourt, D. Simpson, A. Vehtari, and A. Gelman (2018)

Validating bayesian inference algorithms with simulation-based calibration.

arXiv preprint arXiv:1804.06788.

Cited by: Appendix F,
§2,
§4.2.

- [13]
M. Nautiyal, A. Hellander, and P. Singh (2025)

Condisim: conditional diffusion models for simulation based inference.

arXiv preprint arXiv:2505.08403.

Cited by: §4.2.

- [14]
J. Ho, A. Jain, and P. Abbeel (2020)

Denoising diffusion probabilistic models.

Advances in neural information processing systems 33, pp. 6840–6851.

Cited by: §5.1.

- [15]
F. Knoll, J. Zbontar, A. Sriram, M. J. Muckley, M. Bruno, A. Defazio, M. Parente, K. J. Geras, J. Katsnelson, H. Chandarana, et al. (2020)

FastMRI: a publicly available raw k-space and dicom dataset of knee images for accelerated mr image reconstruction using machine learning.

Radiology: Artificial Intelligence 2 (1), pp. e190007.

Cited by: §5.2.

- [16]
J. Zbontar, F. Knoll, A. Sriram, T. Murrell, Z. Huang, M. J. Muckley, A. Defazio, R. Stern, P. Johnson, M. Bruno, et al. (2018)

FastMRI: an open dataset and benchmarks for accelerated mri.

arXiv preprint arXiv:1811.08839.

Cited by: §5.2.

- [17]
C. M. Michel and D. Brunet (2019)

EEG source imaging: a practical review of the analysis steps.

Frontiers in neurology 10, pp. 325.

Cited by: §5.3.

- [18]
A. Gramfort, M. Luessi, E. Larson, D. A. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, et al. (2013)

MEG and eeg data analysis with mne-python.

Frontiers in Neuroinformatics 7, pp. 267.

Cited by: §5.3.

- [19]
M. Hämäläinen, R. Hari, R. J. Ilmoniemi, J. Knuutila, and O. V. Lounasmaa (1993)

Magnetoencephalography—theory, instrumentation, and applications to noninvasive studies of the working human brain.

Reviews of modern Physics 65 (2), pp. 413.

Cited by: §5.3.

- [20]
I. Higgins, L. Matthey, A. Pal, C. P. Burgess, X. Glorot, M. M. Botvinick, S. Mohamed, and A. Lerchner (2016)

Beta-vae: learning basic visual concepts with a constrained variational framework.

In International Conference on Learning Representations,

Cited by: §5.3.

- [21]
R. Sun, W. Zhang, A. Bagić, and B. He (2023)

Deep learning based source imaging provides strong sublobar localization of epileptogenic zone from meg interictal spikes.

NeuroImage 281, pp. 120366.

Cited by: §5.3.

- [22]
O. Hauk, M. Stenroos, and M. S. Treder (2022)

Towards an objective evaluation of eeg/meg source estimation methods–the linear approach.

Neuroimage 255, pp. 119177.

Cited by: §5.3.

- [23]
A. Chamanzar, M. Behrmann, and P. Grover (2021)

Neural silences can be localized rapidly using noninvasive scalp eeg.

Communications biology 4 (1), pp. 429.

Cited by: §5.3.

## Appendix A Graphical Model Interpretation

We introduce the graphical model to understand how the projection geometry naturally induces the decomposition of denoising inference and null space learning tasks, leading to the cascade model.

Figure 7 illustrates this factorization as a probabilistic graphical model.
We assume the data 𝐱\mathbf{x} is generated under data distribution pdata​(𝐱)p_{\text{data}}(\mathbf{x}),
and suppose we are under the realizability assumption; we can parameterize it as pϕ∗​(𝐱)p_{\phi^{*}}(\mathbf{x}). The data distribution pϕ∗​(𝐱)p_{\phi^{*}}(\mathbf{x}) jointly generates the row-space and null-space components via marginals pϕ∗​(𝜶)p_{\phi^{*}}(\boldsymbol{\alpha}) and pϕ∗​(𝜷)p_{\phi^{*}}(\boldsymbol{\beta}). The cascade model we propose, Figure 1, factorizes the posterior over 𝐱\mathbf{x} into two sequential stages. First, the row-space component 𝜶\boldsymbol{\alpha} is inferred from the observation 𝐲\mathbf{y} via the posterior qθ​(𝜶∣𝐲)q_{\theta}(\boldsymbol{\alpha}\mid\mathbf{y}), which approximates the true posterior p⁡(𝜶∣𝐲)p(\boldsymbol{\alpha}\mid\mathbf{y}). Second, given the fixed row-space component 𝜶∗\boldsymbol{\alpha}^{*} — obtained directly from the range-space of 𝐱\mathbf{x} projected by 𝐀\mathbf{A} — the null-space component 𝜷\boldsymbol{\beta} is drawn from the conditional pϕ∗​(𝜷∣𝜶∗)p_{\phi^{*}}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}). This conditional is learned via a generative model, which captures the statistical dependence between the null-space and the fixed row-space component. Since 𝜷\boldsymbol{\beta} lies entirely in the null-space of 𝐀\mathbf{A}, it receives no direct signal from 𝐲\mathbf{y}; its uncertainty is instead governed purely by the
learned prior pϕ∗​(𝜷∣𝜶∗)p_{\phi^{*}}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}). This cascade structure reflects a fundamental asymmetry: 𝜶\boldsymbol{\alpha} lives in the row-space of 𝐀\mathbf{A} and is therefore constrained by the measurement, while 𝜷\boldsymbol{\beta} lives in the null-space of 𝐀\mathbf{A} and remains unidentifiable from 𝐲\mathbf{y} alone. The full reconstruction is then given
by 𝐱=𝐕r​𝜶+𝐕n​𝜷\mathbf{x}=\mathbf{V}_{r}\boldsymbol{\alpha}+\mathbf{V}_{n}\boldsymbol{\beta}.

*Figure 7: Probabilistic graphical model for the structured inverse problem under the realizability assumption pϕ∗​(𝐱)=pdata​(𝐱)p_{\phi^{*}}(\mathbf{x})=p_{\text{data}}(\mathbf{x}). The latent variable 𝐱\mathbf{x} decomposes into a row-space component 𝜶=𝐕r⊤​𝐱\boldsymbol{\alpha}=\mathbf{V}_{r}^{\top}\mathbf{x} and a null-space component 𝜷=𝐕n⊤​𝐱\boldsymbol{\beta}=\mathbf{V}_{n}^{\top}\mathbf{x}, reconstructed as 𝐱=𝐕r​𝜶+𝐕n​𝜷\mathbf{x}=\mathbf{V}_{r}\boldsymbol{\alpha}+\mathbf{V}_{n}\boldsymbol{\beta}. The observation 𝐲\mathbf{y} informs 𝜶\boldsymbol{\alpha} via the posterior qθ​(𝜶∣𝐲)q_{\theta}(\boldsymbol{\alpha}\mid\mathbf{y}), approximating the true posterior p⁡(𝜶∣𝐲)p(\boldsymbol{\alpha}\mid\mathbf{y}), while 𝜷\boldsymbol{\beta} is drawn from the conditional pϕ∗​(𝜷∣𝜶)p_{\phi^{*}}(\boldsymbol{\beta}\mid\boldsymbol{\alpha}).*

## Appendix B Training and Architecture

We instantiate the cascade with a deterministic range model and a conditional DDPM null model, as an example. Other choices (probabilistic range models or VAE null models) are compatible with the same framework; Section D reports results for several such combinations. Algorithm 1 summarizes training and inference for this instantiation, and Figure 8 illustrates the conditional DDPM denoiser architecture for the EEG experiment.

*Algorithm 1 Training and inference for the deterministic-range, DDPM-null cascade*

            0:

           Training set {(𝐱i,𝐲i)}i=1N\{(\mathbf{x}_{i},\mathbf{y}_{i})\}_{i=1}^{N}, forward operator 𝐀\mathbf{A}

            1:

           Setup: Compute SVD 𝐀=𝐔​𝚺​𝐕⊤\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}; partition 𝐕=[𝐕r,𝐕n]\mathbf{V}=[\mathbf{V}_{r},\mathbf{V}_{n}]

            2:

           Project: 𝜶i∗←𝐕r⊤​𝐱i\boldsymbol{\alpha}_{i}^{*}\leftarrow\mathbf{V}_{r}^{\top}\mathbf{x}_{i}, 𝜷i∗←𝐕n⊤​𝐱i\boldsymbol{\beta}_{i}^{*}\leftarrow\mathbf{V}_{n}^{\top}\mathbf{x}_{i} for all ii

            3:

           

            4:

           // Train deterministic range model f𝛉:𝐲↦𝛂^f_{\boldsymbol{\theta}}:\mathbf{y}\mapsto\hat{\boldsymbol{\alpha}}

            5:

           for epoch =1,…,Erange=1,\ldots,E_{\mathrm{range}} do

            6:

            for minibatch {(𝐲i,𝜶i∗)}\{(\mathbf{y}_{i},\boldsymbol{\alpha}_{i}^{*})\} do

            7:

             Update 𝜽\boldsymbol{\theta} to minimize ‖𝜶i∗−f𝜽​(𝐲i)‖22\|\boldsymbol{\alpha}_{i}^{*}-f_{\boldsymbol{\theta}}(\mathbf{y}_{i})\|_{2}^{2}

            8:

            end for

            9:

           end for

            10:

           

            11:

           // Train null model with DDPM denoising loss

            12:

           for epoch =1,…,Enull=1,\ldots,E_{\mathrm{null}} do

            13:

            for minibatch {(𝜶i∗,𝜷i∗)}\{(\boldsymbol{\alpha}_{i}^{*},\boldsymbol{\beta}_{i}^{*})\} do

            14:

             Sample t∼Uniform​{1,…,T}t\sim\mathrm{Uniform}\{1,\ldots,T\}, ϵ∼𝒩⁡(𝟎,𝐈)\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})

            15:

             𝜷t←α¯t​𝜷i∗+1−α¯t​ϵ\boldsymbol{\beta}_{t}\leftarrow\sqrt{\bar{\alpha}_{t}}\,\boldsymbol{\beta}_{i}^{*}+\sqrt{1-\bar{\alpha}_{t}}\,\boldsymbol{\epsilon}

            16:

             Update ϕ\boldsymbol{\phi} to minimize ‖ϵ−ϵϕ​(𝜷t,𝜶i∗,t)‖22\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\boldsymbol{\beta}_{t},\boldsymbol{\alpha}_{i}^{*},t)\|_{2}^{2}

            17:

            end for

            18:

           end for

            19:

           

            20:

           // Inference: sample 𝐱^∼p⁡(𝐱∣𝐲)\hat{\mathbf{x}}\sim p(\mathbf{x}\mid\mathbf{y})

            21:

           Compute 𝜶^=f𝜽​(𝐲)\hat{\boldsymbol{\alpha}}=f_{\boldsymbol{\theta}}(\mathbf{y})

            22:

           Sample 𝜷^∼pϕnull​(𝜷∣𝜶^)\hat{\boldsymbol{\beta}}\sim p_{\boldsymbol{\phi}}^{\mathrm{null}}(\boldsymbol{\beta}\mid\hat{\boldsymbol{\alpha}}) via reverse DDPM diffusion

            23:

           return 𝐱^=𝐕r​𝜶^+𝐕n​𝜷^\hat{\mathbf{x}}=\mathbf{V}_{r}\hat{\boldsymbol{\alpha}}+\mathbf{V}_{n}\hat{\boldsymbol{\beta}}

![Refer to caption](2605.15050v1/fig/denoiser.jpg)

*Figure 8: Denoiser architecture used in the conditional diffusion posterior. The network takes the noisy source representation 𝐱t\mathbf{x}_{t}, EEG measurement 𝐲\mathbf{y}, and diffusion timestep tt as input, and predicts the diffusion noise ϵ^ϕ​(𝐱t,𝐲,t)\hat{\boldsymbol{\epsilon}}_{\phi}(\mathbf{x}_{t},\mathbf{y},t). The EEG measurement is also encoded into a conditioning vector used for feature-wise linear modulation (FiLM) in each residual block. A linear skip connection from 𝐱t\mathbf{x}_{t} is added to the final output.*

## Appendix C Gaussian Analytical Example: Data Generation and Model Details

#### Generative model.

Data are generated as a Gaussian instantiation of the factorization in
Eq. (4), where 𝒚\boldsymbol{y} depends only on
𝜶\boldsymbol{\alpha} by construction, so that
𝜷⟂𝒚|𝜶\boldsymbol{\beta}\perp\boldsymbol{y}\mid\boldsymbol{\alpha} holds exactly:

𝜶\displaystyle\boldsymbol{\alpha}
∼𝒩⁡(𝟎,𝐈r),\displaystyle\sim\mathcal{N}(\mathbf{0},\mathbf{I}_{r}),

(10)

𝜷|𝜶\displaystyle\boldsymbol{\beta}\mid\boldsymbol{\alpha}
∼𝒩⁡(𝐂​𝜶,𝚺𝜼),\displaystyle\sim\mathcal{N}(\mathbf{C}\boldsymbol{\alpha},\,\boldsymbol{\Sigma}_{\boldsymbol{\eta}}),

(11)

𝒚|𝜶\displaystyle\boldsymbol{y}\mid\boldsymbol{\alpha}
∼𝒩⁡(𝐀​𝜶,σy2​𝐈n).\displaystyle\sim\mathcal{N}(\mathbf{A}\boldsymbol{\alpha},\,\sigma_{y}^{2}\mathbf{I}_{n}).

(12)

We use dimensions r=32r=32, q=64q=64, n=32n=32, and measurement noise
σy=0.3\sigma_{y}=0.3.

#### Parameter construction.

The forward operator 𝐀∈ℝn×r\mathbf{A}\in\mathbb{R}^{n\times r} is drawn as a
random Gaussian matrix and orthogonalized via QR decomposition, so that
𝐀⊤​𝐀=𝐈r\mathbf{A}^{\top}\mathbf{A}=\mathbf{I}_{r} and the identifiable subspace is
isotropically covered.
The cross-covariance matrix 𝐂∈ℝq×r\mathbf{C}\in\mathbb{R}^{q\times r} is drawn
i.i.d. from 𝒩⁡(0,r−1)\mathcal{N}(0,r^{-1}), so that
‖𝐂​𝜶‖2≈‖𝜶‖2\|\mathbf{C}\boldsymbol{\alpha}\|^{2}\approx\|\boldsymbol{\alpha}\|^{2} in
expectation.

The intrinsic ambiguity covariance 𝚺𝜼∈ℝq×q\boldsymbol{\Sigma}_{\boldsymbol{\eta}}\in\mathbb{R}^{q\times q} is constructed as 𝚺𝜼=𝐐​𝚲​𝐐⊤\boldsymbol{\Sigma}_{\boldsymbol{\eta}}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}, where 𝐐∈O⁡(q)\mathbf{Q}\in O(q)
is a random orthogonal matrix obtained by QR decomposition of a Gaussian matrix,
and 𝚲=diag⁡(λ1,…,λq)\boldsymbol{\Lambda}=\mathrm{diag}(\lambda_{1},\ldots,\lambda_{q}) has
eigenvalues spaced geometrically from λmax=8.0\lambda_{\max}=8.0 to
λmin=0.1\lambda_{\min}=0.1 (condition number ≈80\approx 80).
This construction yields a fully correlated covariance with off-diagonal
structure, ensuring that calibration requires matching not only marginal scales
but also the geometry of the ambiguous subspace.
Samples 𝜼∼𝒩⁡(𝟎,𝚺𝜼)\boldsymbol{\eta}\sim\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_{\boldsymbol{\eta}}) are drawn via the Cholesky factor
𝐋\mathbf{L} as 𝜼=𝐋𝐳\boldsymbol{\eta}=\mathbf{L}\mathbf{z},
𝐳∼𝒩⁡(𝟎,𝐈q)\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_{q}).

#### Analytical posteriors.

The Gaussian structure admits closed-form posteriors used as ground truth for
calibration:

𝚺𝜶|y\displaystyle\boldsymbol{\Sigma}_{\boldsymbol{\alpha}|y}
=(𝐈r+σy−2​𝐀⊤​𝐀)−1,𝝁𝜶|y=σy−2​𝚺𝜶|y​𝐀⊤​𝐲,\displaystyle=\bigl(\mathbf{I}_{r}+\sigma_{y}^{-2}\mathbf{A}^{\top}\mathbf{A}\bigr)^{-1},\quad\boldsymbol{\mu}_{\boldsymbol{\alpha}|y}=\sigma_{y}^{-2}\boldsymbol{\Sigma}_{\boldsymbol{\alpha}|y}\mathbf{A}^{\top}\mathbf{y},

(13)

Cov⁡(𝜷∣𝒚)\displaystyle\mathrm{Cov}(\boldsymbol{\beta}\mid\boldsymbol{y})
=𝐂​𝚺𝜶|y​𝐂⊤+𝚺𝜼,\displaystyle=\mathbf{C}\,\boldsymbol{\Sigma}_{\boldsymbol{\alpha}|y}\,\mathbf{C}^{\top}+\boldsymbol{\Sigma}_{\boldsymbol{\eta}},

(14)

𝜷|𝜶∗\displaystyle\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*}
∼𝒩⁡(𝐂​𝜶∗,𝚺𝜼).\displaystyle\sim\mathcal{N}\!\left(\mathbf{C}\boldsymbol{\alpha}^{*},\,\boldsymbol{\Sigma}_{\boldsymbol{\eta}}\right).

(15)

The oracle ambiguity distribution p⁡(𝜷∣𝜶∗)p(\boldsymbol{\beta}\mid\boldsymbol{\alpha}^{*})
is thus available in closed form, enabling direct comparison against the learned
null model.

#### Null model training and evaluation.

A conditional DDPM is trained on N=100,000N=100{,}000 i.i.d. samples of
(𝜶∗,𝜷∗)(\boldsymbol{\alpha}^{*},\boldsymbol{\beta}^{*}) pairs using a linear noise
schedule with T=1000T=1000 diffusion steps (β1=10−4\beta_{1}=10^{-4},
βT=0.02\beta_{T}=0.02), a two-block MLP denoiser with hidden dimension 256, the
Adam optimizer at learning rate 3×10−43\times 10^{-4}, and batch size 256, for
10510^{5} gradient steps.
SBC diagnostics are computed using Ms​b​c=200M_{sbc}=200 posterior samples per test point over Nsbc=500N_{\mathrm{sbc}}=500 held-out values of 𝜶∗\boldsymbol{\alpha}^{*}, using the scalar test statistics
T1=‖𝜷‖2T_{1}=\|\boldsymbol{\beta}\|_{2} and T2=‖𝜷‖∞/‖𝜷‖2T_{2}=\|\boldsymbol{\beta}\|_{\infty}/\|\boldsymbol{\beta}\|_{2}.

## Appendix D EEG cascade results

Figure 9 reports posterior mean reconstruction quality for all trained range–null combinations. The dominant pattern is that Pearson correlation varies primarily across range models, while differences across null models are comparatively small within each row. This is expected because the posterior mean reconstruction is strongly influenced by the measurement-identifiable component, which is determined by the range model. In contrast, the null model primarily controls the distribution of intrinsically ambiguous components.

This grid supports the model-selection motivation of the main text. A practitioner selecting a cascade using reconstruction correlation alone would find several near-equivalent combinations. However, as shown in Section 5.3, models with similar reconstruction quality can differ substantially in their calibration of intrinsic ambiguity. Thus, the range–null grid illustrates why reconstruction metrics alone are insufficient for selecting posterior models in ill-posed inverse problems.

![Refer to caption](2605.15050v1/result_grid.png)

*Figure 9: Pearson correlation between the posterior mean reconstruction and the ground-truth source signal across range–null model assemblies, averaged over 450 held-out test cases. Cells report mean ±\pm s.e.m. across test cases. The right column reports the measurement residual, ‖𝐲−𝐀​𝐱^‖2/‖𝐲‖2\|\mathbf{y}-\mathbf{A}\hat{\mathbf{x}}\|_{2}/\|\mathbf{y}\|_{2}, which depends only on the range-stage reconstruction.*

## Appendix E Noise propagation bound for accelerated MRI

We provide a short argument for why the propagated uncertainty term in Eq. 5 is controlled by measurement noise in the accelerated MRI setting.

The propagated uncertainty term in Eq. 5 is 𝕍𝜶|𝐲​(𝔼​[𝜷∣𝜶])\mathbb{V}_{\boldsymbol{\alpha}\mid\mathbf{y}}\!\left(\mathbb{E}[\boldsymbol{\beta}\mid\boldsymbol{\alpha}]\right).
In the MRI notation, this corresponds to 𝕍𝐱^𝜶​(m⁡(𝐱^𝜶))\mathbb{V}_{\hat{\mathbf{x}}_{\boldsymbol{\alpha}}}\!\left(m(\hat{\mathbf{x}}_{\boldsymbol{\alpha}})\right), where m⁡(𝐱𝜶)=𝔼pϕnull​[𝐱𝜷∣𝐱𝜶]m(\mathbf{x}_{\boldsymbol{\alpha}})=\mathbb{E}_{p_{\phi}^{\mathrm{null}}}[\mathbf{x}_{\boldsymbol{\beta}}\mid\mathbf{x}_{\boldsymbol{\alpha}}] is the conditional null mean.
The forward operator is 𝐀=𝐌𝐅\mathbf{A=MF}, where 𝐅\mathbf{F} is the two-dimensional Fourier transform and 𝐌\mathbf{M} is a binary k-space mask. For a clean image 𝐱\mathbf{x}, the oracle identifiable component is 𝐱𝜶∗=𝐅−1​𝐌𝐅𝐱\mathbf{x}_{\boldsymbol{\alpha}}^{*}=\mathbf{F}^{-1}\mathbf{MFx}. Under noisy measurements 𝐲σ=𝐌𝐅𝐱+𝜺σ\mathbf{y}_{\sigma}=\mathbf{MFx}+\boldsymbol{\varepsilon}_{\sigma}, the closed-form identifiable estimate is

𝐱^𝜶=𝐅−1​𝐲σ=𝐱𝜶∗+𝐅−1​𝜺σ.\hat{\mathbf{x}}_{\boldsymbol{\alpha}}=\mathbf{F}^{-1}\mathbf{y}_{\sigma}=\mathbf{x}_{\boldsymbol{\alpha}}^{*}+\mathbf{F}^{-1}\boldsymbol{\varepsilon}_{\sigma}.

Thus 𝐱^𝜶−𝐱α∗=𝐅−1​𝜺σ\hat{\mathbf{x}}_{\boldsymbol{\alpha}}-\mathbf{x}_{\alpha}^{*}=\mathbf{F}^{-1}\boldsymbol{\varepsilon}_{\sigma}.
For a unitary normalization of the Fourier transform, ‖𝐅−1​𝜺σ‖22=‖𝜺σ‖22\|\mathbf{F}^{-1}\boldsymbol{\varepsilon}_{\sigma}\|_{2}^{2}=\|\boldsymbol{\varepsilon}_{\sigma}\|_{2}^{2}, and therefore 𝔼​‖𝐱^α−𝐱α∗‖22=O⁡(σ2)\mathbb{E}\|\hat{\mathbf{x}}_{\alpha}-\mathbf{x}_{\alpha}^{*}\|_{2}^{2}=O(\sigma^{2}).

Under the assumption that mm is LL-Lipschitz, then

tr​(𝕍𝐱^α)​(m⁡(𝐱^α))\displaystyle\text{tr}(\mathbb{V}_{\hat{\mathbf{x}}_{\alpha}})\!\left(m(\hat{\mathbf{x}}_{\alpha})\right)
≤𝔼​‖m⁡(𝐱^α)−m⁡(𝐱α∗)‖22\displaystyle\leq\mathbb{E}\left\|m(\hat{\mathbf{x}}_{\alpha})-m(\mathbf{x}_{\alpha}^{*})\right\|_{2}^{2}

≤L2​𝔼​‖𝐱^α−𝐱α∗‖22=O⁡(σ2).\displaystyle\leq L^{2}\mathbb{E}\|\hat{\mathbf{x}}_{\alpha}-\mathbf{x}_{\alpha}^{*}\|_{2}^{2}=O(\sigma^{2}).

(16)

Hence, the propagated uncertainty term vanishes as σ→0\sigma\to 0, supporting the interpretation of Fig. 3(a): intrinsic ambiguity is fixed when conditioning on 𝐱𝜶∗\mathbf{x}_{\boldsymbol{\alpha}}^{*}, while additional null uncertainty under noisy conditioning is controlled by perturbations in 𝐱^𝜶\hat{\mathbf{x}}_{\boldsymbol{\alpha}}.

## Appendix F Simulation-Based Calibration for High-Dimensional Parameters

Simulation-Based Calibration (SBC) [12] provides a principled diagnostic
for posterior inference by exploiting a self-consistency property of Bayes’ rule. If
θsim∼p⁡(θ)\theta^{\mathrm{sim}}\sim p(\theta) and ysim∼p⁡(y∣θsim)y^{\mathrm{sim}}\sim p(y\mid\theta^{\mathrm{sim}}),
then θsim\theta^{\mathrm{sim}} is itself a draw from the posterior p⁡(θ∣ysim)p(\theta\mid y^{\mathrm{sim}}),
and its rank among approximate posterior draws {θ(m)}m=1M\{\theta^{(m)}\}_{m=1}^{M} must be uniformly
distributed on {0,…,M}\{0,\ldots,M\}. Concretely, the rank statistic is

rk=∑m=1M[θk(m)<θksim],r_{k}=\sum_{m=1}^{M}\mathbf{1}\!\left[\theta_{k}^{(m)}<\theta_{k}^{\mathrm{sim}}\right],

(17)

and any systematic deviation from uniformity in {rk}\{r_{k}\} signals that the approximate posterior
is miscalibrated. This diagnostic is sensitive to three distinct failure modes: model
misspecification, sampler pathology, and implementation errors.

#### Challenges in High Dimensions.

For 𝜽∈ℝD\boldsymbol{\theta}\in\mathbb{R}^{D}, the standard procedure yields DD separate rank sequences, one
per coordinate. This raises two difficulties. First, simultaneous uniformity tests across DD
parameters require multiple-testing correction, which reduces statistical power. Second, and more
fundamentally, passing DD marginal tests does not imply joint calibration: a posterior can have
correct marginals while being badly wrong about inter-coordinate correlations.

#### Scalar Aggregates for Exchangeable Parameters.

When the coordinates of 𝜽\boldsymbol{\theta} are semantically homogeneous and exchangeable under the prior
— for instance, when each 𝜽k\boldsymbol{\theta}_{k} represents the posterior mean of the kk-th data dimension
— scalar aggregates of 𝜽\boldsymbol{\theta} provide computationally efficient and interpretable SBC
diagnostics. We propose two complementary statistics.

The first is the ℓ2\ell_{2} norm ‖𝜽‖2\|\boldsymbol{\theta}\|_{2}, which aggregates total energy across dimensions.
Its rank probes whether the posterior captures the correct overall scale: systematic shrinkage or
overdispersion manifests as a rank distribution skewed toward 00 or MM, respectively.

The second is the peakedness ratio ‖𝜽‖∞/‖𝜽‖2\|\boldsymbol{\theta}\|_{\infty}/\|\boldsymbol{\theta}\|_{2}, which measures the degree
of concentration of 𝜽\boldsymbol{\theta} across its coordinates. This ratio lies in [D−1/2,1][D^{-1/2},1], with
values near D−1/2D^{-1/2} indicating diffuse, roughly uniform energy and values near 11 indicating
near-sparse concentration in a single coordinate. A posterior that over-smooths a sparse signal
will produce a ratio systematically smaller than the truth, and SBC will detect this as a
rank distribution biased toward 00.

#### Complementarity.

These two statistics probe orthogonal aspects of the posterior geometry, as summarized below:

Statistic
Sensitive to
Blind to

‖𝜽‖2\|\boldsymbol{\theta}\|_{2}
Overall scale / shrinkage
Sparsity structure

‖𝜽‖∞/‖𝜽‖2\|\boldsymbol{\theta}\|_{\infty}/\|\boldsymbol{\theta}\|_{2}
Sparsity / concentration
Overall energy level

Running SBC on both simultaneously provides a two-dimensional diagnostic well-suited to
structured, high-dimensional posteriors where sparsity is a meaningful property of the signal.

<!-- END-OF-PAPER:cdd680e21009 -->
