Rethinking Coupled Tensor Analysis for Hyperspectral Super-Resolution: Recoverable Modeling Under Endmember Variability

    ![](/static/base/1.0.1/images/icons/smileybones-small.svg)
    arXiv is now an independent nonprofit!
    Learn more
    ×

    License: arXiv.org perpetual non-exclusive license

arXiv:2512.19489v1 [eess.IV] 22 Dec 2025

# Rethinking Coupled Tensor Analysis for Hyperspectral Super-Resolution: Recoverable Modeling Under Endmember Variability Thanks: Submitted to the editors on Sep 9, 2025; revision submitted on Nov 2, 2025; accepted on Dec. 16.

Meng Ding
††thanks: School of Mathematics, Southwest Jiaotong University, Chengdu, China
().
Email: dingmeng56@swjtu.edu.cn

  
Xiao Fu
††thanks: Corresponding author. School of Electrical Engineering and Computer Science, Oregon State University, Corvallis, OR 97331, United States
().
Email: xiao.fu@oregonstate.edu

###### Abstract

This work revisits the hyperspectral super-resolution (HSR) problem, i.e., fusing a pair of spatially co-registered hyperspectral (HSI) and multispectral (MSI) images to recover a super-resolution image (SRI) that enhances the spatial resolution of the HSI. Coupled tensor decomposition (CTD)-based methods have gained traction in this domain, offering recoverability guarantees under various assumptions. Existing models such as canonical polyadic decomposition (CPD) and Tucker decomposition provide strong expressive power but lack physical interpretability. The block-term decomposition model with rank-(Lr,Lr,1)(L_{r},L_{r},1) terms (the LL1 model) yields interpretable factors under the linear mixture model (LMM) of spectral images, but LMM assumptions are often violated in practice—primarily due to nonlinear effects such as endmember variability (EV).
To address this issue, we propose representing spectral images using a more flexible block-term tensor model with rank-(Lr,Mr,Nr)(L_{r},M_{r},N_{r}) terms (the LMN model). This modeling choice retains interpretability, subsumes CPD, Tucker, and LL1 as special cases, and robustly accounts for non-ideal effects such as EV, offering a balanced tradeoff between expressiveness and interpretability for HSR. Importantly, under the LMN model for HSI and MSI, recoverability of the SRI can still be established under proper conditions—providing strong theoretical support. Extensive experiments on synthetic and real datasets demonstrate the effectiveness and robustness of the proposed method.

###### keywords

hyperspectral super-resolution,
endmember variability,
block-term tensor decomposition,
recoverability.

###### Funding.

The work of M. Ding was supported in part by the National Natural Science Foundation of China (12201522) and Natural Science Foundation of Sichuan Province of China (2024NSFSC1389). The work of X. Fu was supported in part by the National Science Foundation (NSF) under Project ECCS 2024058.

††runningheads: Rethinking CTD for Hyperspectral Super-Resolution / M. Ding and X. Fu

## 1 Introduction

The tradeoff between spatial and spectral resolution in spectral imaging is well known, primarily due to limitations in the image acquisition process [67]. Hyperspectral images (HSIs) typically offer high spectral resolution but suffer from low spatial resolution, while multispectral images (MSIs) exhibit the opposite characteristics. To recover a super-resolution image (SRI) with high resolution in both spatial and spectral domains, hyperspectral super-resolution (HSR) has been widely studied. The goal of HSR is to integrate a pair of spatially co-registered HSI and MSI so that the resulting SRI combines the spectral richness of the HSI with the spatial detail of the MSI; see [67, 43].

Early approaches to HSR relied on heuristic methods such as component substitution [2] and multiresolution analysis [1]. Later on, more advanced algebraic and optimization techniques—such as low-rank and nonnegative matrix factorization [33, 30, 52]—were introduced to better address the fusion challenge; see, e.g., [68, 59, 58]. These methods typically leverage the linear mixture model (LMM) for spectral images [17, 37, 16], along with carefully designed degradation models that reflect the data acquisition process, to cast the HSR task as various coupled matrix factorization (or low-rank matrix recovery) problems—forming a class of structured inverse problems.
The LMM describes the spectral image pixels as nonnegative combinations of endmembers (spectral signature of materials within the pixels), and is widely used in spectral image analysis (see the survey [37]).
Interestingly, as noted in [31, 34], the recoverability of the SRI using low-rank matrix factorization can be theoretically guaranteed under certain conditions—for example, when each spectral pixel consists of only a small number of constituent materials.

More recently, a series of promising tensor factorization-based approaches have been proposed for HSR [25, 42, 12]. In contrast to matrix factorization methods, these techniques model spectral images as third-order tensors, thereby preserving the inherent multi-dimensional structure of the SRI. The coupled tensor decomposition (CTD) framework assumes that the spectral images follow a tensor factorization model, such as the canonical polyadic (CP) decomposition model [21], the Tucker model [53], or the block-term decomposition with multilinear rank-(Lr,Lr,1)(L_{r},L_{r},1) terms (the LL1 model) [10].
By leveraging the algebraic properties of the CP decomposition, the work [25] established theoretical guarantees for recoverability of the SRI under mild conditions. Similar recoverability results have also been demonstrated under the Tucker and LL1-based models, based on their respective decomposition structures; see [42, 12].
Other structural tensor models, such as tensor ring decomposition [70], tensor singular value decomposition [27, 71], and related methods [20, 60, 32], have also been explored for HSR, but these efforts primarily focus on algorithmic design and empirical performance, without providing formal recoverability analysis, to our best knowledge.

### 1.1 Challenges in Recoverability-Guaranteed HSR via CTD

While CPD-, Tucker-, and LL1-based CTD formulations all offer recoverability guarantees, they involve important tradeoffs. The LL1 model, grounded in the LMM, assigns clear physical meaning to its latent factors—enabling effective incorporation of prior information for structural regularization, which is especially beneficial under noise. However, the LMM assumption has its own limitations, as it fails to capture effects like endmember variability (EV) [69, 26, 4].
When EV is present, the endmembers vary across pixels and this violates the LMM grossly, making models relying on the LMM suffer from substantial performance degradation.
In contrast, CPD and Tucker models [25, 42] serve as universal tensor approximators [49] that can flexibly model scenarios containing EV and other nonlinear effects (via adjusting the tensor rank). Yet, their latent factors lack physical interpretability, making it difficult to apply meaningful regularization in practice. This oftentimes leads to performance deterioration under noisy and complex scenarios [12].

In recent years, efforts have been made to address EV issues in spectral image modeling and CTD-based HSR, with particular emphasis on the recoverability of the SRI. In [3], a CTD approach based on the Tucker decomposition was proposed to account for EV, with exact SRI recovery established under certain conditions. Subsequently, [43] modeled the HSR problem with EV using a coupled LL1 formulation and likewise provided recoverability guarantees. However, both works [43, 3] adopted an inter-image EV model, assuming that variability only occurs across HSI and MSI, whereas EV across spectral pixels [50] was not considered.

### 1.2 Contributions

This work revisits CTD-based HSR methods that offer recoverability guarantees, and proposes a new CTD model that strikes a balance among robustness to across-pixel EV, physical interpretability, and SRI recoverability. We should mention that while EV was repeatedly considered in various prior works for HSR (see, e.g., [5, 66, 7]), these approaches do not rely on tensor models and lack recoverability guarantees; hence, they are beyond the scope of this study. Our detailed contributions are as follows:

- •

Using LMN Model to Incorporate EV.
Our key idea is to adopt the tensor block-term decomposition (BTD) in rank-(Lr,Mr,Nr)(L_{r},M_{r},N_{r}) terms—referred to as the “LMN” model [10]—to represent spectral images under EV. The LMN model extends beyond the classical LMM, while retaining strong physical interpretability in the presence of EV, thereby facilitating the incorporation of prior information to improve HSR performance.

- •

Recoverability Supports.
Building on this model, we provide a theoretical analysis showing that the CTD formulation with LMN-structured spectral images guarantees recoverability of the SRI under reasonable conditions. Our proof begins with a new essential uniqueness condition for the LMN decomposition, which could be of broader interest beyond HSR. Leveraging this result, we generalize existing analyses of CTD approaches based on CPD, Tucker, and LL1 models [25, 42, 12] to establish recoverability under LMN.
Unlike [43, 3] which focuses on inter-image EV across HSI and MSI, our result demonstrates recoverability in scenarios where unknown EV is present across spectral pixels.

- •

Regularization and Algorithm Design.
Moreover, we leverage the physical interpretation of the LMN latent factors to design model-based constraints and regularization. We propose a first-order inexact block coordinate descent algorithm to solve the resulting constrained CTD problem. Extensive simulations and real-data experiments validate the effectiveness of the proposed model and algorithm.

Notation. The scalars, vectors, matrices, and tensors are represented by the lowercase (xx) or uppercase (XX), boldface lowercase (𝒙\bm{x}), boldface uppercase (𝑿\bm{X}), and underlined boldface uppercase (𝑿¯\underline{\bm{X}}), respectively.
The symbols [𝒙]i[\bm{x}]_{i}, [𝑿]i,j[\bm{X}]_{i,j}, and [𝑿¯]i,j,k[\underline{\bm{X}}]_{i,j,k} respectively denote the ii-th, (i,j)(i,j)-th, and (i,j,k)(i,j,k)-th element of 𝒙∈ℝI\bm{x}\in\mathbb{R}^{I}, 𝑿∈ℝI×J\bm{X}\in\mathbb{R}^{I\times J}, and 𝑿¯∈ℝI×J×K\underline{\bm{X}}\in\mathbb{R}^{I\times J\times K}.
‖𝑿‖F=∑i,j[𝑿]i,j2\|\bm{X}\|_{F}=\sqrt{\sum_{i,j}[\bm{X}]_{i,j}^{2}} and ‖𝑿¯‖F=∑i,j,k[𝑿¯]i,j,k2\|\underline{\bm{X}}\|_{F}=\sqrt{\sum_{i,j,k}[\underline{\bm{X}}]_{i,j,k}^{2}} respectively denote the Frobenius norms of the matrix 𝑿∈ℝI×J\bm{X}\in\mathbb{R}^{I\times J} and the tensor 𝑿¯∈ℝI×J×K\underline{\bm{X}}\in\mathbb{R}^{I\times J\times K}.
The mode-nn product of a tensor 𝑿¯∈ℝI1×I2×⋯×IN\underline{\bm{X}}\in\mathbb{R}^{I_{1}\times I_{2}\times\cdots\times I_{N}} and a matrix 𝒀∈ℝIn×J\bm{Y}\in\mathbb{R}^{I_{n}\times J} yields a tensor 𝑿¯×n𝒀∈ℝI1×⋯In−1×J×In+1⋯×IN\underline{\bm{X}}\times_{n}\bm{Y}\in\mathbb{R}^{I_{1}\times\cdots I_{n-1}\times J\times I_{n+1}\cdots\times I_{N}}, which each element is given by
[𝑿¯×n​𝒀]i1​…​in−1​j​in+1​…​iN=∑in=1In[𝑿¯]i1,…,iN​[𝑼]j,in[\underline{\bm{X}}_{\times n}\bm{Y}]_{i_{1}\ldots i_{n-1}ji_{n+1}\ldots i_{N}}=\sum_{i_{n}=1}^{I_{n}}[\underline{\bm{X}}]_{i_{1},\ldots,i_{N}}[\bm{U}]_{j,i_{n}}.

## 2 Background

### 2.1 Problem Setting of HSR

We consider the classical setting [25, 12, 59, 68, 57, 3, 4, 42, 43] where a pair of spatially co-registered HSI and MSI are available.
The term “co-registration” means that the HSI and MSI cover the exactly identical spatial region with a shared underlying continuous spatial coordinate system.
The HSI and MSI images are denoted as

𝒀¯H∈ℝIH×JH×KH&𝒀¯M∈ℝIM×JM×KM,\underline{\bm{Y}}_{\rm H}\in\mathbb{R}^{I_{\rm H}\times J_{\rm H}\times K_{\rm H}}\quad\&\quad\underline{\bm{Y}}_{\rm M}\in\mathbb{R}^{I_{\rm M}\times J_{\rm M}\times K_{\rm M}},

respectively,
where IHI_{\rm H} and JHJ_{\rm H} are the vertical and horizontal spatial dimensions, respectively, KHK_{\rm H} denotes the number of wavelengths, and IM,JMI_{\rm M},J_{\rm M} and KMK_{\rm M} are defined in the same way for the MSI.
Note that under the HSR setting, the following inequalities hold [25, 12, 42]:

KH≫KM,IM​JM≫IH​JH;K_{\rm H}\gg K_{\rm M},\quad I_{\rm M}J_{\rm M}\gg I_{\rm H}J_{\rm H};

i.e., the HSI has a higher spectral resolution yet the MSI’s spatial resolution is much finer.
The goal of HSR is to fuse 𝒀¯H\underline{\bm{Y}}_{{\rm H}} and 𝒀¯M\underline{\bm{Y}}_{{\rm M}} to produce an SRI, denoted as 𝒀¯S∈ℝIM×JM×KH\underline{\bm{Y}}_{\rm S}\in\mathbb{R}^{I_{\rm M}\times J_{\rm M}\times K_{\rm H}}—which takes the spatial resolution of the MSI and the spectral resolution of the HSI.

![Refer to caption](2512.19489v1/SRI_degradation2.png)

*Figure 1: Illustration of spatial and spectral degradations from SRI (𝒀¯S\underline{\bm{Y}}_{\rm S}) to HSI (𝒀¯H\underline{\bm{Y}}_{\rm H}) and MSI (𝒀¯M\underline{\bm{Y}}_{\rm M}), respectively; figure adapted from [25, 12].*

In the literature, the HSI and MSI are assumed to be downsampled and degraded from 𝒀¯S\underline{\bm{Y}}_{\rm S}; see, e.g., [25, 12, 43, 3]. To be specific, the spatial degradation model to obtain the HSI is as follows [25, 12, 43, 3]:

𝒀¯H(:,:,k)=𝑷1𝒀¯S(:,:,k)𝑷2⊤,k=1,…,KH,\displaystyle\underline{\bm{Y}}_{\rm H}(:,:,k)=\bm{P}_{1}\underline{\bm{Y}}_{\rm S}(:,:,k)\bm{P}_{2}^{\top},~k=1,\ldots,K_{\rm H},

(1)

where 𝒀¯S(:,:,k)\underline{\bm{Y}}_{\rm S}(:,:,k) denotes the kkth band of 𝒀¯S\underline{\bm{Y}}_{\rm S}, 𝑷1∈ℝIH×IM\bm{P}_{1}\in\mathbb{R}^{I_{\rm H}\times I_{\rm M}} and 𝑷2∈ℝJH×JM\bm{P}_{2}\in\mathbb{R}^{J_{\rm H}\times J_{\rm M}} denote two spatial degradation matrices applied to the vertical and horizontal dimensions.
These operators are assumed to consists of blurring kernels and downsampling operations [25, 12, 42].
The operations in (1) can be expressed using the tensor mode-product:

𝒀¯H=𝒀¯S×1𝑷1×2𝑷2,\displaystyle\underline{\bm{Y}}_{\rm H}=\underline{\bm{Y}}_{\rm S}\times_{1}\bm{P}_{1}\times_{2}\bm{P}_{2},

(2)

where ×n\times_{n} denotes the mode-nn product.
Similarly, the MSI is obtained via spectral degradation from the SRI:

𝒀¯H(i,j,:)=𝑷M𝒀¯S(i,j,:),∀(i,j)∈[IM]×[JM],\displaystyle\underline{\bm{Y}}_{\rm H}(i,j,:)=\bm{P}_{\rm M}\underline{\bm{Y}}_{\rm S}(i,j,:),~\forall(i,j)\in[I_{\rm M}]\times[J_{\rm M}],

where 𝒀¯S(i,j,:)\underline{\bm{Y}}_{\rm S}(i,j,:) denotes the (i,j)(i,j)th spectral pixel of 𝒀¯S\underline{\bm{Y}}_{\rm S}, and 𝑷M∈ℝKM×KH\bm{P}_{\rm M}\in\mathbb{R}^{K_{\rm M}\times K_{\rm H}} is a spectral degradation matrix [25, 12, 43, 3], which is equivalently expressed as follows:

𝒀¯M=𝒀¯S×3𝑷M.\displaystyle\underline{\bm{Y}}_{\rm M}=\underline{\bm{Y}}_{\rm S}\times_{3}\bm{P}_{\rm M}.

(3)

The spectral degradation is often modeled as relatively simple operations such as weighting and aggregation [25, 12, 42].
The degradation model in Eqs (2)-(3) is illustrated in Fig. 1.
Note that all the matrices 𝑷1\bm{P}_{1}, 𝑷2\bm{P}_{2}, and 𝑷M\bm{P}_{\rm M} are “compressing” matrices (i.e., fat matrices). Therefore, recovering 𝒀¯S\underline{\bm{Y}}_{\rm S} from the compressed measurements 𝒀¯H\underline{\bm{Y}}_{\rm H} and 𝒀¯M\underline{\bm{Y}}_{\rm M} is an ill-posed inverse problem. That is, even if 𝑷1\bm{P}_{1}, 𝑷2\bm{P}_{2}, and 𝑷M\bm{P}_{\rm M} are known, there might be an infinite number of solutions of 𝒀¯S\underline{\bm{Y}}_{\rm S} that satisfy Eqs (2)-(3), making establishing recoverability of SRI nontrivial.

### 2.2 CTD-based HSR via Various Tensor Models

Starting from [25],
a series of CTD based HSR approaches, e.g., [25, 12, 42, 43, 3], showed promising capability in establishing recoverability of 𝒀¯S\underline{\bm{Y}}_{\rm S} under reasonable assumptions. The CTD formulations can be summarized as follows:

min𝜽\displaystyle\min_{\bm{\theta}}~
‖𝒀¯H−𝒀¯S(𝜽)×1𝑷1×2𝑷2‖F2+‖𝒀¯M−𝒀¯S(𝜽)×3𝑷M‖F2,\displaystyle\left\|\underline{\bm{Y}}_{\rm H}-\underline{\bm{Y}}_{\rm S}(\bm{\theta})\times_{1}\bm{P}_{1}\times_{2}\bm{P}_{2}\right\|^{2}_{F}+\left\|\underline{\bm{Y}}_{\rm M}-\underline{\bm{Y}}_{\rm S}(\bm{\theta})\times_{3}\bm{P}_{\rm M}\right\|^{2}_{F},

where 𝒀¯S​(𝜽)\underline{\bm{Y}}_{\rm S}(\bm{\theta}) represents a tensor model parameterized by 𝜽\bm{\theta}.
In the above, the two tensors 𝒀¯H\underline{\bm{Y}}_{\rm H} and 𝒀¯M\underline{\bm{Y}}_{\rm M} are decomposed simultaneously with their latent factors shared across 𝜽\bm{\theta}—making the two tensors “coupled” through the latent factors and leading to the terminology CTD.
As shown in [25, 12, 42, 43, 3], 𝒀¯S​(𝜽)\underline{\bm{Y}}_{\rm S}(\bm{\theta}), by picking a proper tensor model 𝒀¯S​(𝜽)\underline{\bm{Y}}_{\rm S}(\bm{\theta}), it is possible to show that any optimal solution of (2.2) recovers the ground truth SRI, i.e., 𝒀¯S​(𝜽⋆)=𝒀¯S\underline{\bm{Y}}_{\rm S}(\bm{\theta}^{\star})=\underline{\bm{Y}}_{\rm S}.
The work in [25] also first established that, under certain conditions, such recoverability holds even without knowing 𝑷1\bm{P}_{1} and 𝑷2\bm{P}_{2} if the CPD model is used to represent 𝒀¯S​(𝜽)\underline{\bm{Y}}_{\rm S}(\bm{\theta}). This property was then shown to hold for other tensor models, e.g., the LL1 model [12].
A number of representative models for 𝒀¯S​(𝜽)\underline{\bm{Y}}_{\rm S}(\bm{\theta}) that admit provable recoverability are discussed in the Sec. 2.2.1-Sec. 2.2.3.

#### 2.2.1 CPD-based CTD for HSR

In [25], the idea is to model 𝒀¯S\underline{\bm{Y}}_{\rm S} using a CPD model, i.e.,

𝒀¯S(𝜽)=∑f=1F𝑨(:,f)∘𝑩(:,f)∘𝑪(:,f),\displaystyle\underline{\bm{Y}}_{\rm S}(\bm{\theta})=\sum_{f=1}^{F}\bm{A}(:,f)\circ\bm{B}(:,f)\circ\bm{C}(:,f),

(4)

where 𝜽=(𝑨,𝑩,𝑪)\bm{\theta}=(\bm{A},\bm{B},\bm{C}),
𝑨∈ℝIM×F\bm{A}\in\mathbb{R}^{I_{\rm M}\times F}, 𝑩∈ℝJM×F\bm{B}\in\mathbb{R}^{J_{\rm M}\times F}, and 𝑪∈ℝKH×F\bm{C}\in\mathbb{R}^{K_{\rm H}\times F} are the latent factors,
and FF is the tensor rank. Note that any tensor can be expressed as a CPD model, with a sufficiently large FF [49].
More interestingly, the work [25] empirically showed that real spectral images can be approximated by (4) with a relatively low rank FF, which is a result of the fact that spectral images exhibit correlations among all three modes.
The work [25] showed that the optimal solution of (2.2) ensures recovering the SRI.
To be specific, assume that 𝒀¯S=∑f=1F𝑨♮(:,f)∘𝑩♮(:,f)∘𝑪♮(:,f)\underline{\bm{Y}}_{\rm S}=\sum_{f=1}^{F}\bm{A}^{\natural}(:,f)\circ\bm{B}^{\natural}(:,f)\circ\bm{C}^{\natural}(:,f) is the ground truth SRI.
Denote the optimal solution of (2.2) as 𝑨⋆\bm{A}^{\star}, 𝑩⋆\bm{B}^{\star}, and 𝑪⋆\bm{C}^{\star} under the parameterization in (4). Then, it was shown in [25] that the following holds under reasonable conditions (e.g., FF is sufficiently low in (4)):

𝒀¯S=∑f=1F𝑨⋆(:,f)∘𝑩⋆(:,f)∘𝑪⋆(:,f)=∑f=1F𝑨♮(:,f)∘𝑩♮(:,f)∘𝑪♮(:,f);\displaystyle\underline{\bm{Y}}_{\rm S}=\sum_{f=1}^{F}\bm{A}^{\star}(:,f)\circ\bm{B}^{\star}(:,f)\circ\bm{C}^{\star}(:,f)=\sum_{f=1}^{F}\bm{A}^{\natural}(:,f)\circ\bm{B}^{\natural}(:,f)\circ\bm{C}^{\natural}(:,f);

(5)

that is, any optimal solution of the CTD formulation under the CPD model provably recovers the SRI.
The CTD perspective provided in [25] established recoverability of the SRI for the first time.

#### 2.2.2 Tucker-based CTD for HSR

Later on, the work [42] replaced the CPD model in (4) with a Tucker model, i.e.,

𝒀¯S(𝜽)=𝑫¯×1𝑨×2𝑩×3𝑪,\displaystyle\underline{\bm{Y}}_{\rm S}(\bm{\theta})={\underline{\bm{D}}}\times_{1}{\bm{A}}\times_{2}{\bm{B}}\times_{3}{\bm{C}},

(6)

where 𝜽=(𝑫¯,𝑨,𝑩,𝑪)\bm{\theta}=(\underline{\bm{D}},\bm{A},\bm{B},\bm{C}),
𝑨∈ℝIM×L{\bm{A}}\in\mathbb{R}^{I_{\rm M}\times L}, 𝑩∈ℝJM×M{\bm{B}}\in\mathbb{R}^{J_{\rm M}\times M}, and 𝑪∈ℝKH×N{\bm{C}}\in\mathbb{R}^{K_{\rm H}\times N} are factor matrices, and 𝑫¯∈ℝL×M×N\underline{\bm{D}}\in\mathbb{R}^{L\times M\times N} is the core tensor, and the tuple (L,M,N)(L,M,N) is often referred to the multilinear Tucker rank.
Note that L≤IML\leq I_{\rm M}, M≤JMM\leq J_{\rm M}, and N≤KHN\leq K_{\rm H} and that every tensor admits a Tucker representation with sufficiently large LL, MM, and NN.
Using this coupled Tucker model, the work [42] showed that recoverability of the SRI can be retained if (L,M,N)(L,M,N) are sufficiently small.
They also proposed a fast HSR algorithm leveraging the HOSVD algorithm for the Tucker decomposition.

![Refer to caption](2512.19489v1/tensor_comparison.png)

*Figure 2: Illustration of three tensor decompositions. Top to bottom: CPD, Tucker, and LL1, respectively; illustration adapted from [12].*

#### 2.2.3 LL1-based CTD for HSR

Following the idea in [45], the work [12] proposed to use the LL1 model for the SRI, i.e.,

𝒀¯S=∑r=1R(𝑨r𝑩r⊤)∘𝑪(:,r),\displaystyle\underline{\bm{Y}}_{\rm S}=\sum_{r=1}^{R}\left(\bm{A}_{r}\bm{B}_{r}^{\top}\right)\circ\bm{C}(:,r),

(7)

where 𝑨r∈ℝIM×Lr\bm{A}_{r}\in\mathbb{R}^{I_{\rm M}\times L_{r}}, 𝑩r∈ℝJM×Lr\bm{B}_{r}\in\mathbb{R}^{J_{\rm M}\times L_{r}}, and 𝑪∈ℝKH×R\bm{C}\in\mathbb{R}^{K_{\rm H}\times R} are factor matrices; see Fig. 2.
There is an interesting link between the LL1 model and the LMM of spectral images.
Denote 𝒚S(i,j)=𝒀¯S(i,j,:)\bm{y}_{\rm S}^{(i,j)}=\underline{\bm{Y}}_{\rm S}(i,j,:)
as the pixel located at (i,j)(i,j).
Assume that there are RR materials contained in the SRI, whose spectral signatures are 𝒄1,…,𝒄R\bm{c}_{1},\ldots,\bm{c}_{R}. Then, under the LMM, the pixel can be expressed as
𝒚S(i,j)=∑r=1R𝑺r​(i,j)​𝒄r\bm{y}_{\rm S}^{(i,j)}=\sum_{r=1}^{R}\bm{S}_{r}(i,j)\bm{c}_{r};
i.e., the pixels 𝒚S(i,j)\bm{y}_{\rm S}^{(i,j)} is a weighted sum of RR spectral signatures (or endmembers) represented by 𝒄1,…,𝒄R\bm{c}_{1},\ldots,\bm{c}_{R}. The term 𝑺r​(i,j)\bm{S}_{r}(i,j) represents the “abundance” of material rr at the location (i,j)(i,j), and 𝑺r∈ℝIM×JM\bm{S}_{r}\in\mathbb{R}^{I_{\rm M}\times J_{\rm M}} is the abundance map indicating the amount of material rr over the region.
Note that if one puts all pixels 𝒚S(i,j)\bm{y}_{\rm S}^{(i,j)} together, we have

𝒀¯S=∑r=1R𝑺r∘𝑪(:,r),\displaystyle\underline{\bm{Y}}_{\rm S}=\sum_{r=1}^{R}\bm{S}_{r}\circ{\bm{C}}(:,r),

(8)

where 𝑪=[𝒄1,…,𝒄R]\bm{C}=[\bm{c}_{1},\ldots,\bm{c}_{R}]. If rank⁡(𝑺r)≤Lr{\rm rank}(\bm{S}_{r})\leq L_{r}, one can re-express it as
𝑺r=𝑨r​𝑩r⊤\bm{S}_{r}=\bm{A}_{r}\bm{B}_{r}^{\!\top\!}, which recovers the LL1 model in (7).
It was noticed that assuming the abundance maps to be low rank matrices is reasonable, as the spread of a material in the spatial domain often changes gracefully [13, 45, 12].

### 2.3 The Expressiveness and Interpretability Tradeoff

Fig. 2 illustrates the tensor models that ensure recoverability of 𝒀¯S\underline{\bm{Y}}_{\rm S} under the CTD formulation in (2.2). Several key observations are outlined below:

First, a salient feature of the CTD framework in (2.2) is its ability to characterize the recoverability of the SRI in relation to the complexity of the underlying tensor model—often determined by notions of tensor rank; see [42, 12, 25]. Recoverability guarantees are crucial in inverse problems, where infinitely many spurious solutions may exist. They help ensure that the recovered solution is meaningful.

![Refer to caption](2512.19489v1/Urban_Variability.png)

*Figure 3: Illustration of endmember variability of the Urban dataset, i.e., the spectrum of a single material (e.g., asphalt, tree, grass, and roof) can vary from pixel to pixel. (The source image is a subimage of the Urban data, which is collected by the HYDICE sensor over an urban area at Copperas Cove, TX, U.S.)*

Second, the CPD and Tucker models are known for their expressive power: with sufficiently high rank (CPD) or multilinear rank (Tucker), they can approximate any spectral image. However, a limitation of these models is that their latent factors generally lack physical interpretability in the context of HSR. This makes it difficult to impose physical meaning-based regularization or constraints on the latent factors—which are often essential for improving robustness in challenging or noisy conditions.

Third, the LL1 model enjoys a meaningful connection to the classical LMM (cf. Sec. 2.2.3). This connection enables the application of structured constraints on the latent factors (e.g., nonnegativity or spatial smoothness on 𝑺r\bm{S}_{r}), providing interpretability and control. Nonetheless, the LMM—and by extension, LL1—struggles to capture more complex, realistic phenomena such as EV. While CPD and Tucker can, in principle, model such complexities by increasing model rank, doing so often results in higher computational cost and numerical instability. Moreover, without structural regularization, increased rank may not yield actual performance gains.

Beyond the three aforementioned tensor models, various other tensor decomposition approaches have also been explored for representing 𝒀¯S\underline{\bm{Y}}_{\rm S} within the CTD framework—such as tensor ring [20] and tensor singular value decomposition [60].
However, these models are not the focus of this work, as recoverability guarantees were not primary considerations under these formulations. Moreover, they face similar challenges as previously discussed: namely, the trade-off between model expressiveness and physical interpretability, particularly in complex scenarios.

## 3 Rethinking CTD: Balancing Interpretability and Expressiveness

### 3.1 Modeling Spectral Images with EV

The LL1 model-based HSR in [12] showed state-of-the-art performance when the spectral images closely follow the LMM (in their experiments, the images were generated following the LMM with small noise perturbations). This demonstrates the importance of using prior information and physical meaning in HSR. However, as we mentioned, the LMM is not always accurate.

![Refer to caption](2512.19489v1/Tree_material.png)

*Figure 4: Illustration of the mode-3 fibers of 𝑻¯r\underline{\bm{T}}_{r}, which correspond to the spectral signatures of the same materials across the IM×JMI_{M}\times J_{M} space. The key postulate is that, as all of the spectra are variations of the same endmember, all the fibers are similar to each other. Therefore, each 𝑻¯r\underline{\bm{T}}_{r} has a low multi-linear rank.*

Fig. 3 shows a case where the LMM is clearly violated. Under the LMM, the endmembers 𝒄1,…,𝒄R\bm{c}_{1},\ldots,\bm{c}_{R} do not change across all pixels. However, one can see that the spectral signatures of asphalt, tree, grass, and roof all change, sometimes drastically—showing visible EV effects.
In the presence of EV, relying on the LMM may not always obtain satisfactory results for spectral imaging tasks; see discussions in [3, 4, 26, 43, 69, 50].

![Refer to caption](2512.19489v1/LMN_BTD.png)

*Figure 5: Illustration of the LMN model defined in [9, 10]. The LMN model subsumes the CPD, Tucker, and LL1 models in Fig. 2 as its special cases.*

*Table 1: The fitting errors of using CPD/Tucker/LL1/LMN models to approximate the Urban data of size 200×200×162200\times 200\times 162 (The numbers in the lower left subscript indicate the corresponding rank settings).*

num. of paras.
∼\sim30K
∼\sim45K
∼\sim60K
∼\sim75K
∼\sim90K
∼\sim105K
∼\sim120K

CPD(F)\textrm{CPD}_{(F)}
0.1394(52)0.1394_{(52)}
0.0992(80)0.0992_{(80)}
0.0765(106)0.0765_{(106)}
0.0605(133)0.0605_{(133)}
0.0495(160)0.0495_{(160)}
0.0415(186)0.0415_{(186)}
0.0355(212)0.0355_{(212)}

Tucker(L,M,N)\textrm{Tucker}_{(L,M,N)}
0.1336(51,54,3)0.1336_{(51,54,3)}
0.1073(66,66,4)0.1073_{(66,66,4)}
0.0869(90,88,3)0.0869_{(90,88,3)}
0.0753(96,96,4)0.0753_{(96,96,4)}
0.0670(108,104,4)0.0670_{(108,104,4)}
0.0591(120,118,4)0591_{(120,118,4)}
0.0557(120,118,5)0.0557_{(120,118,5)}

LL1(L)\textrm{LL1}_{(L)}
0.1470(18)0.1470_{(18)}
0.1122(27)0.1122_{(27)}
0.0892(37)0.0892_{(37)}
0.0772(46)0.0772_{(46)}
0.0725(50)0.0725_{(50)}
0.0724(50)0.0724_{(50)}
0.0724(50)0.0724_{(50)}

LMN(L,M,N)\textrm{LMN}_{(L,M,N)}
0.1273(14,14,5)\textbf{0.1273}_{(14,14,5)}
0.0884(22,22,3)\textbf{0.0884}_{(22,22,3)}
0.0671(28,28,4)\textbf{0.0671}_{(28,28,4)}
0.0563(32,32,5)\textbf{0.0563}_{(32,32,5)}
0.0456(39,39,4)\textbf{0.0456}_{(39,39,4)}
0.0392(44,44,4)\textbf{0.0392}_{(44,44,4)}
0.0341(49,49,4)\textbf{0.0341}_{(49,49,4)}

*Table 2: The fitting errors of using CPD/Tucker/LL1/LMN models to approximate the Pavia University data of size 200×200×203200\times 200\times 203 (The numbers in the lower left subscript indicate the corresponding rank settings).*

num. of paras.
∼\sim20K
∼\sim30K
∼\sim40K
∼\sim50K
∼\sim60K
∼\sim70K
∼\sim80K

CPD(F)\textrm{CPD}_{(F)}
0.1707(39)0.1707_{(39)}
0.1320(59)0.1320_{(59)}
0.1059(79)0.1059_{(79)}
0.0877(99)0.0877_{(99)}
0.0746(119)0.0746_{(119)}
0.0646(139)0.0646_{(139)}
0.0582(155)0.0582_{(155)}

Tucker(L,M,N)\textrm{Tucker}_{(L,M,N)}
0.1640(39,37,3)0.1640_{(39,37,3)}
0.1523(42,42,7)0.1523_{(42,42,7)}
0.1165(62,60,4)0.1165_{(62,60,4)}
0.0964(78,78,3)0.0964_{(78,78,3)}
0.0875(80,98,3)0.0875_{(80,98,3)}
0.0799(91,91,4)0799_{(91,91,4)}
0.0728(98,101,4)0.0728_{(98,101,4)}

LL1(L)\textrm{LL1}_{(L)}
0.1909(12)0.1909_{(12)}
0.1546(18)0.1546_{(18)}
0.1291(24)0.1291_{(24)}
0.1107(30)0.1107_{(30)}
0.0949(37)0.0949_{(37)}
0.0853(43)0.0853_{(43)}
0.0777(49)0.0777_{(49)}

LMN(L,M,N)\textrm{LMN}_{(L,M,N)}
0.1636(10,4,4)\textbf{0.1636}_{(10,4,4)}
0.1199(16,3,4)\textbf{0.1199}_{(16,3,4)}
0.0999(20,3,4)\textbf{0.0999}_{(20,3,4)}
0.0819(25,3,4)\textbf{0.0819}_{(25,3,4)}
0.0710(29,3,4)\textbf{0.0710}_{(29,3,4)}
0.0609(34,3,4)\textbf{0.0609}_{(34,3,4)}
0.0548(38,3,4)\textbf{0.0548}_{(38,3,4)}

#### 3.1.1 Using LMN Model to Incorporate EV

Under the LMM, the spectral pixel of the SRI at location (i,j)(i,j) is expressed as follows

𝒚S(i,j)=∑r=1R𝑺r​(i,j)​𝒄r,∀(i,j).\displaystyle\bm{y}_{\rm S}^{(i,j)}=\sum_{r=1}^{R}\bm{S}_{r}(i,j)\bm{c}_{r},~\forall(i,j).

(9)

However, when EV exists, the spectral signature 𝒄r\bm{c}_{r} changes across (i,j)(i,j). Denote 𝒄r(i,j)\bm{c}_{r}^{(i,j)} as the spectral signature variation of material rr at the pixel location (i,j)(i,j). Then, the following pixel model is more plausible:

𝒚S(i,j)=∑r=1R𝑺r​(i,j)​𝒄r(i,j),∀(i,j).\displaystyle\bm{y}_{\rm S}^{(i,j)}=\sum_{r=1}^{R}\bm{S}_{r}(i,j)\bm{c}_{r}^{(i,j)},~\forall(i,j).

(10)

Following (10), the SRI can be expressed as

𝒀¯S=∑r=1R𝑻¯r,𝑻¯r(i,j,:)=𝑺r(i,j)𝒄r(i,j).\displaystyle\underline{\bm{Y}}_{\rm S}=\sum_{r=1}^{R}\underline{\bm{T}}_{r},\quad\underline{\bm{T}}_{r}(i,j,:)=\bm{S}_{r}(i,j)\bm{c}_{r}^{(i,j)}.

(11)

Note that 𝑻¯r(i,j,:)\underline{\bm{T}}_{r}(i,j,:) is often called a mode-3 fiber of a third-order tensor 𝑻¯r\underline{\bm{T}}_{r}, which represents the spectrum of material rr in pixel (i,j)(i,j) in our context.

In (11), the EV effect exists across all pixels, i.e., the 𝒄r(i,j)\bm{c}_{r}^{(i,j)}’s for all (i,j)(i,j) are different. However, as endmember variation is graceful over space, the 𝒄r(i,j)\bm{c}_{r}^{(i,j)}’s in the entire image are still non-trivially correlated. Therefore, it is reasonable to model 𝑻¯r\underline{\bm{T}}_{r} using a low multi-linear rank Tucker model—this is the key of our approach (also see Fig. 4 for illustration of 𝑻¯r\underline{\bm{T}}_{r}).
Hence, each 𝑻¯r\underline{\bm{T}}_{r} in (11) can be written as

𝑻¯r=𝑫¯r×1𝑨r×2𝑩r×3𝑪r,∀r,\displaystyle\underline{\bm{T}}_{r}=\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{C}_{r},~\forall r,

(12)

where 𝑫¯r∈ℝLr×Mr×Nr\underline{\bm{D}}_{r}\in\mathbb{R}^{L_{r}\times M_{r}\times N_{r}}, 𝑨r∈ℝIM×Lr\bm{A}_{r}\in\mathbb{R}^{I_{\rm M}\times L_{r}}, 𝑩r∈ℝJM×Mr\bm{B}_{r}\in\mathbb{R}^{J_{\rm M}\times M_{r}}, and 𝑪r∈ℝKH×Nr\bm{C}_{r}\in\mathbb{R}^{K_{\rm H}\times N_{r}}—with reasonable Lr,MrL_{r},M_{r} and NrN_{r}.
Under (12), the SRI can be expressed as

𝒀¯S=∑r=1R𝑫¯r×1𝑨r×2𝑩r×3𝑪r,∀r.\displaystyle\underline{\bm{Y}}_{\rm S}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{C}_{r},~\forall r.

(13)

This way, the SRI is modeled as a block-term decomposition with multi-linear rank-(Lr,Mr,Nr)(L_{r},M_{r},N_{r}) terms (LMN) [10]; see the illustration in Fig. 5. To further support this hypothesis, we extract 𝑻¯r\underline{\bm{T}}_{r} from the Jasper Ridge and Washington DC datasets and plot the singular values mode-1, 2, and 3 unfoldings (see Figs. 16-17 in Appendix A). One can see that all matrix unfoldings of 𝑻¯r\underline{\bm{T}}_{r} are low rank matrices. This attest to the effectiveness of modeling 𝑻¯r\underline{\bm{T}}_{r} using a low multilinear rank Tucker tensor. Hence, by employing the LMN model to represent the spectral images naturally accounts for endmember variability.

Note that unlike CPD or Tucker whose latent factors do not have physical interpretations, the terms in (11) still exhibits strong physical connections. That is, each 𝑻¯r\underline{\bm{T}}_{r} is contributed by a specific material.
More over, the tensor rank FF in CPD does not have physical meaning, yet the RR in (10) still corresponds to the number of materials contained in the image.
Hence, the spatial and spectral characteristics of a single material can still be used to serve as structural regularization/constraints—we will leverage this for algorithm design.

#### 3.1.2 Numerical Validation for LMN-based Modeling

To validate our postulate, we fit real-world hyperspectral images using various tensor models.
We use the normalized reconstruction error (NRE):

NRE=‖𝒀¯​(𝜽)−𝒀¯‖F/‖𝒀¯‖F\texttt{NRE}=\|\underline{\bm{Y}}(\bm{\theta})-\underline{\bm{Y}}\|_{F}/\|\underline{\bm{Y}}\|_{F}

(where 𝒀¯​(𝜽)\underline{\bm{Y}}(\bm{\theta}) is the approximated spectral image by the used tensor model, and 𝒀¯\underline{\bm{Y}} is the spectral image being fitted to) under different number of tensor model parameters.
We use real hyperspectral images, namely, the Urban data [41] and the Pavia University data [29], as 𝒀¯\underline{\bm{Y}}. Note that these images are well-known to exhibit endmember variability [72, 4].

Tables 1-2 show the results when 𝒀¯​(𝜽)\underline{\bm{Y}}(\bm{\theta}) is modeled using CPD, Tucker, LL1, and LMN, respectively, where all models use approximately the same amount of parameters.
For two datasets, we set the number of materials R=4R=4 and choose ranks of four tensor models so that the corresponding number of parameters is approximately equal to a given value.
One can see that when using similar amounts of parameters, the LMN model outputs the smallest fitting errors relative to the other models. In particular, the LMN model often obtains an NRE that is 50% lower than that of the LL1 model. This attests to the effectiveness of relaxing the model of 𝑻¯r\underline{\bm{T}}_{r} from an LL1 structure (i.e., a matrix outer-product with a vector) to an LMN Tucker structure in the context of HSI approximation.
In addition, note that both the CPD and Tucker models can also attain competitive NREs. As mentioned, both CPD and Tucker are universal tensor approximators, with sufficiently high rank. Nonetheless, the lose interpretability of their latent components.

### 3.2 CTD Formulation under LMN Model

Under the LMN model in (11) and (10), and
according to the degradation model in (3) and (2), we have

𝒀¯H=∑r=1R𝑫¯r×1(𝑷1𝑨r)×2(𝑷2𝑩r)×3𝑪r,\displaystyle\underline{\bm{Y}}_{{\rm H}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}(\bm{P}_{1}\bm{A}_{r})\times_{2}(\bm{P}_{2}\bm{B}_{r})\times_{3}\bm{C}_{r},

(14)

and

𝒀¯M=∑r=1R𝑫¯r×1𝑨r×2𝑩r×3(𝑷M𝑪r).\displaystyle\underline{\bm{Y}}_{{\rm M}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r}).

(15)

Using the observation models in (14) and (15), we rewrite the CTD formulation in (2.2) as follows:

min𝜽12‖𝒀¯H−∑r=1R𝑫¯r×1(𝑷1𝑨r)×2(𝑷2𝑩r)×3𝑪r‖F2\displaystyle\min_{\bm{\theta}}~\frac{1}{2}\left\|\underline{\bm{Y}}_{\rm H}-\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}(\bm{P}_{1}\bm{A}_{r})\times_{2}(\bm{P}_{2}\bm{B}_{r})\times_{3}\bm{C}_{r}\right\|^{2}_{F}

+12‖𝒀¯M−∑r=1R𝑫¯r×1𝑨r×2𝑩r×3(𝑷M𝑪r)‖F2,\displaystyle+\frac{1}{2}\left\|\underline{\bm{Y}}_{\rm M}-\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r})\right\|^{2}_{F},

(16)

where 𝜽=({𝑫¯r,𝑨r,𝑩r,𝑪r}r=1R)\bm{\theta}=(\{\underline{\bm{D}}_{r},\bm{A}_{r},\bm{B}_{r},\bm{C}_{r}\}_{r=1}^{R}).

The works in [25, 42, 12] also considered cases where the spatial degradation operator 𝑷1\bm{P}_{1} and 𝑷2\bm{P}_{2} are unknown. This setup is meaningful, as the spatial operators are nontrivial to estimate (yet the spectral operator is relatively easy to acquire, e.g., by inspecting sensor specifications [25, 12]). Our formulation can also be amended to cover this case:

min𝜽′12‖𝒀¯H−∑r=1R𝑫¯r×1𝑨~r×2𝑩~r×3𝑪r‖F2\displaystyle\min_{\bm{\theta}^{\prime}}~\frac{1}{2}\left\|\underline{\bm{Y}}_{\rm H}-\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\widetilde{\bm{A}}_{r}\times_{2}\widetilde{\bm{B}}_{r}\times_{3}\bm{C}_{r}\right\|^{2}_{F}

+12‖𝒀¯M−∑r=1R𝑫¯r×1𝑨r×2𝑩r×3(𝑷M𝑪r)‖F2,\displaystyle+\frac{1}{2}\left\|\underline{\bm{Y}}_{\rm M}-\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r})\right\|^{2}_{F},

(17)

where 𝜽′=({𝑫¯r,𝑨~r,𝑩~r,𝑨r,𝑩r,𝑪r}r=1R)\bm{\theta}^{\prime}=(\{\underline{\bm{D}}_{r},\widetilde{\bm{A}}_{r},\widetilde{\bm{B}}_{r},\bm{A}_{r},\bm{B}_{r},\bm{C}_{r}\}_{r=1}^{R}), and 𝑨~r∈ℝIH×Lr\widetilde{\bm{A}}_{r}\in\mathbb{R}^{I_{H}\times L_{r}} and 𝑩~r∈ℝJH×Mr\widetilde{\bm{B}}_{r}\in\mathbb{R}^{J_{H}\times M_{r}} are used to model 𝑷1​𝑨r\bm{P}_{1}\bm{A}_{r} and 𝑷2​𝑩r\bm{P}_{2}\bm{B}_{r}, respectively.

### 3.3 Recoverability Guarantees

One of the most appealing aspects of the CTD framework proposed by [25] is that it provided a systematic way of establishing recoverability of 𝒀¯S\underline{\bm{Y}}_{\rm S}. This allows a suite of CTD framework to have recoverability guarantees of 𝒀¯S\underline{\bm{Y}}_{{\rm S}}; see the CPD-based [25], the LL1-based [12], and the Tucker-based [42] versions, respectively.

In this work, we show that the LMN model is not an exception—i.e., using the LMN model under the CTD model still retains similar recoverability guarantees.
The proof framework resembles those in [25, 12].
Nonetheless, as we will leverage the essential uniqueness of the LMN decomposition to approach the recoverability problem, careful custom analyses are provided to carry out the proof.

###### Definition 1 (Essential Uniqueness).

The tensor 𝐘¯=∑r=1R𝐃¯r×1𝐀r×2𝐁r×3𝐂r\underline{\bm{Y}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{C}_{r} has an essnetially unique LMN decomposition if the following is met: for any representation 𝐘¯=∑r=1R𝐃¯¯r×1𝐀¯r×2𝐁¯r×3𝐂¯r\underline{\bm{Y}}=\sum_{r=1}^{R}\underline{\bar{\bm{D}}}_{r}\times_{1}\bar{\bm{A}}_{r}\times_{2}\bar{\bm{B}}_{r}\times_{3}\bar{\bm{C}}_{r}, we have

𝑨¯π⁡(r)=𝑨r​𝚯a,r,𝑩¯π⁡(r)=𝑩r​𝚯b,r,𝑪¯π⁡(r)=𝑪r​𝚯c,r,\displaystyle\bar{\bm{A}}_{\pi(r)}=\bm{A}_{r}\bm{\Theta}_{a,r},~\bar{\bm{B}}_{\pi(r)}=\bm{B}_{r}\bm{\Theta}_{b,r},~\bar{\bm{C}}_{\pi(r)}=\bm{C}_{r}\bm{\Theta}_{c,r},

(18)

𝑫¯¯π⁡(r)×1𝚯a,r×2𝚯b,r×3𝚯c,r=𝑫¯r,∀r,\displaystyle\underline{\bar{\bm{D}}}_{\pi(r)}\times_{1}\bm{\Theta}_{a,r}\times_{2}\bm{\Theta}_{b,r}\times_{3}\bm{\Theta}_{c,r}=\underline{\bm{D}}_{r},~\forall r,

(19)

where π\pi is a permutation of {1,…,R}\{1,\ldots,R\}, and 𝚯a,r\bm{\Theta}_{a,r}, 𝚯b,r\bm{\Theta}_{b,r}, and 𝚯c,r\bm{\Theta}_{c,r} are nonsingular matrices.

We note that existing LMN decomposition conditions cover a range of cases in terms of the relationship between L,M,NL,M,N and I,J,KI,J,K. However, the these conditions are not realistic in the context of HSR. For example, the seminal paper [10] proved that, under reasonable conditions, LMN uniqueness holds if

L=M,I≥LR,J≥MR,N≥3,𝑪ris full column rank,1≤r≤R;\displaystyle L=M,I\geq LR,J\geq MR,N\geq 3,\bm{C}_{r}~\textrm{is full column rank},1\leq r\leq R;

(20)

or 
N>L+M−2,min⁡(⌊IL⌋,R)+min⁡(⌊JM⌋,R)+min⁡(⌊KN⌋,R)≥2​R+2;\displaystyle N>L+M-2,\min\left(\left\lfloor\frac{I}{L}\right\rfloor,R\right)+\min\left(\left\lfloor\frac{J}{M}\right\rfloor,R\right)+\min\left(\left\lfloor\frac{K}{N}\right\rfloor,R\right)\geq 2R+2;

(21)

see [10, Theorem 5.1, Theorem 5.5], respectively.
We note that the L=ML=M part in Eq. (20) is too special to meet by spectral image tensors.
In addition, the first inequality in (21) essentially means that the spectral mode of the spectral images have less correlations relative to the spatial dimensions, which is also hard to justify in the context of HSR.

To overcome these unrealistic conditions for spectral images, we first extend the results in [10] and
show a new identifiability result of LMN decomposition:

###### Theorem 2 (Identifiability of LMN).

Let ({𝐀r∈ℝI×Lr,𝐁r∈ℝJ×Mr,𝐂r∈ℝK×Nr,𝐃¯r∈ℝLr×Mr×Nr}r=1R)(\{\bm{A}_{r}\in\mathbb{R}^{I\times L_{r}},\bm{B}_{r}\in\mathbb{R}^{J\times M_{r}},\bm{C}_{r}\in\mathbb{R}^{K\times N_{r}},\underline{\bm{D}}_{r}\in\mathbb{R}^{L_{r}\times M_{r}\times N_{r}}\}_{r=1}^{R}) be the latent factors of the LMN tensor 𝐘¯=∑r=1R𝐃¯r×1𝐀r×2𝐁r×3𝐂r\underline{\bm{Y}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{C}_{r} where Lr=LL_{r}=L, Mr=MM_{r}=M, and Nr=NN_{r}=N. Assume that 𝐀r\bm{A}_{r}, 𝐁r\bm{B}_{r}, 𝐂r\bm{C}_{r}, and 𝐃¯r\underline{\bm{D}}_{r} are drawn from any absolutely continuous distributions. Then, the LMN decomposition of 𝐘¯\underline{\bm{Y}} is essentially unique almost surely, if

I≥L​R,J≥M​R,L​M≥N≥max⁡{⌈LM⌉+⌈ML⌉,3},1≤r≤R.\displaystyle I\geq LR,~~J\geq MR,~~LM\geq N\geq\max\left\{\left\lceil\frac{L}{M}\right\rceil+\left\lceil\frac{M}{L}\right\rceil,3\right\},~1\leq r\leq R.

###### Proof.

The proof is a combination of the new development in [14] and the proof pipeline in [10, Theorem 5.1]. The details are in Appendix B.

Note that under the above new theorem, the hard-to-justify conditions in the context of spectral imaging, i.e., L=ML=M and N>L+M−2N>L+M-2, no longer exist. The conditions in Theorem 2 only requires that RR and L,M,NL,M,N are sufficiently small. The physical meaning is clear: when the number of endmembers in the spectral image is not too large and the correlations of the underlying component associated with each endmember (i.e.,
𝑻¯r\underline{\bm{T}}_{r}) across three modes are sufficiently high, the LMN decomposition is essentially unique. Based on this uniqueness theorem, we show the following:

###### Theorem 3.

Assume that 𝐘¯S\underline{\bm{Y}}_{{\rm S}} follows the LMN model in (10)-(11) with Lr=L,Mr=M,Nr=NL_{r}=L,M_{r}=M,N_{r}=N for r=1,…,Rr=1,\ldots,R, and that 𝐘¯H\underline{\bm{Y}}_{{\rm H}} and 𝐘¯M\underline{\bm{Y}}_{{\rm M}} follow the degradation models in (14) and (15), respectively. Also assume that each of 𝐃¯r∈ℝL×M×N\underline{\bm{D}}_{r}\in\mathbb{R}^{L\times M\times N}, 𝐀r∈ℝIM×L\bm{A}_{r}\in\mathbb{R}^{I_{\rm M}\times L}, 𝐁r∈ℝJM×M\bm{B}_{r}\in\mathbb{R}^{J_{\rm M}\times M}, and 𝐂r∈ℝKH×N\bm{C}_{r}\in\mathbb{R}^{K_{\rm H}\times N} is drawn from any absolutely continuous distribution, and that 𝐏1\bm{P}_{1}, 𝐏2\bm{P}_{2}, and 𝐏M\bm{P}_{M} have full row rank. Let {𝐃¯r⋆,𝐀r⋆,𝐁r⋆,𝐂r⋆}r=1R\{\underline{\bm{D}}_{r}^{\star},\bm{A}_{r}^{\star},\bm{B}_{r}^{\star},\bm{C}_{r}^{\star}\}_{r=1}^{R} represent any feasible solution of Problem (16). Then, the ground-truth 𝐘¯S\underline{\bm{Y}}_{{\rm S}} is uniquely recovered almost surely by

𝒀¯S=∑r=1R𝑫¯r⋆×1𝑨r⋆×2𝑩r⋆×3𝑪r⋆,\underline{\bm{Y}}_{{\rm S}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}^{\star}\times_{1}\bm{A}_{r}^{\star}\times_{2}\bm{B}_{r}^{\star}\times_{3}\bm{C}_{r}^{\star},

if the following conditions are met:

IH​JH≥L​M​R,\displaystyle I_{\rm H}J_{\rm H}\geq LMR,

(22a)

IM≥L​R,JM≥M​R,\displaystyle I_{\rm M}\geq LR,~J_{\rm M}\geq MR,

(22b)

L​M≥N≥max⁡{⌈L/M⌉+⌈M/L⌉,3}\displaystyle LM\geq N\geq\max\{\lceil L/M\rceil+\lceil M/L\rceil,3\}

(22c)

for all r∈[R]r\in[R].

The theorem asserts that, as long as the spatial region of interest does not have an excessive amount of endmembers and each endmember’s variations exhibit sufficient correlations across the region, recoverability of the SRI is guaranteed. Note that the result is naturally robust to the cases where EV is present—as discussed, even in the presence of EV, 𝑻¯r(i,j,:)\underline{\bm{T}}_{r}(i,j,:) for different (i,j)(i,j)’s (i.e., endmember rr appearing at different locations) are still correlated as shown in Fig. 4. The result also does not rely on the somewhat stringent conditions in existing EV-aware CTD frameworks (e.g., that EV only occurs across HSI and MSI [43, 3]).

Similarly, we show recoverability of the semi-blind recovery formulation (17) where the spatial degradation operators are unknown:

###### Theorem 4.

Under the same assumptions as in Theorem 3, assume that
{𝐀~r⋆,𝐁~r⋆,𝐃¯r⋆,\{\widetilde{\bm{A}}_{r}^{\star},\widetilde{\bm{B}}_{r}^{\star},\underline{\bm{D}}_{r}^{\star}, 𝐀r⋆,\bm{A}_{r}^{\star}, 𝐁r⋆,𝐂r⋆}r=1R\bm{B}_{r}^{\star},\bm{C}_{r}^{\star}\}_{r=1}^{R} is any feasible solution of Problem (17). Then,
if KM≥2​NK_{M}\geq 2N, IH≥L​RI_{\rm H}\geq LR, JH≥M​RJ_{\rm H}\geq MR, and L​M≥N≥max⁡{⌈L/M⌉+⌈M/L⌉,3}LM\geq N\geq\max\{\lceil L/M\rceil+\lceil M/L\rceil,3\}, 1≤r≤R1\leq r\leq R,
the ground-truth 𝐘¯S\underline{\bm{Y}}_{S} is uniquely recovered by
𝐘¯S=∑r=1R𝐃¯r⋆×1𝐀r⋆×2𝐁r⋆×3𝐂r⋆\underline{\bm{Y}}_{{\rm S}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}^{\star}\times_{1}\bm{A}_{r}^{\star}\times_{2}\bm{B}_{r}^{\star}\times_{3}\bm{C}_{r}^{\star} with probability one.

###### Remark 5.

Theorems 3–4 confirm that adopting the LMN model in the CTD framework retains similar recoverability guarantees previously established for other tensor models. To some extent, this outcome is expected, since the LMN model subsumes CPD, LL1, and Tucker as special cases. Nevertheless, its applicability under EV and the resulting practical implications for HSR are significant. Moreover, while our proof strategy shares similarities with those in [25, 12] for CPD and LL1, the technical underpinning is far from straightforward. Two aspects deserve particular emphasis: (i) unlike [25, 12], which directly invoke existing uniqueness conditions, we derive new conditions for LMN uniqueness tailored to the HSR setting (see Theorem 2); and (ii) because the definition of essential uniqueness for LMN differs from those of CPD and LL1, the proof requires careful handling of the associated mathematical nuances, making the overall recoverability analysis nontrivial to complete.

## 4 Regularization Design and Algorithm

In this section, we design practical algorithms to tackle (16) and (17). In particular, we will design regularization terms that incorporate physical meaning of 𝑻¯r\underline{\bm{T}}_{r} for performance enhancement.

### 4.1 Regularization Design for Problems (16) and (17)

One of the characteristics of the endmembers is that they spread smoothly over space [36, 22, 38].
In the spectral domain, the endmember’s variations are also smooth.
This means that the mode-1, mode-2 and mode-3 “fibers” of 𝑻¯r\underline{\bm{T}}_{r} (i.e., 𝑻¯r(:,j,k)\underline{\bm{T}}_{r}(:,j,k), 𝑻¯r(i,:,k)\underline{\bm{T}}_{r}(i,:,k), and 𝑻¯r(i,j,:)\underline{\bm{T}}_{r}(i,j,:)) all exhibit a certain degree of smoothness; see [28] for illustration of fibers.
Note that the mode-1, mode-2 and mode-3 fibers of 𝑻¯r\underline{\bm{T}}_{r} reside in range⁡(𝑨r){\rm range}(\bm{A}_{r}), range⁡(𝑩r){\rm range}(\bm{B}_{r}) and range⁡(𝑪r){\rm range}(\bm{C}_{r}), respectively.
Therefore, to regularize the smoothness of these fibers, we propose to regularize the bases of these range spaces:

ϕr​(𝑨r,𝑩r,𝑪r)=ϕp,ε​(𝑯1​𝑨r)+ϕp,ε​(𝑯2​𝑩r)+ϕTik​(𝑯3​𝑪r),\displaystyle\phi_{r}(\bm{A}_{r},\bm{B}_{r},\bm{C}_{r})=\phi_{p,\varepsilon}(\bm{H}_{1}\bm{A}_{r})+\phi_{p,\varepsilon}(\bm{H}_{2}\bm{B}_{r})+\phi_{\rm Tik}(\bm{H}_{3}\bm{C}_{r}),

(23)

where ϕp,ε​(𝑿)=∑i∑j(xi,j2+ε)p2\phi_{p,\varepsilon}(\bm{X})=\sum_{i}\sum_{j}(x_{i,j}^{2}+\varepsilon)^{\frac{p}{2}} with ε>0\varepsilon>0 [12, 15].
We employ this nonconvex total variation regularization ϕp,ε​(𝑯1​𝑨r)+ϕp,ε​(𝑯2​𝑩r)\phi_{p,\varepsilon}(\bm{H}_{1}\bm{A}_{r})+\phi_{p,\varepsilon}(\bm{H}_{2}\bm{B}_{r}) to promote spatial smoothness. Here, 𝑯1∈ℝ(IM−1)×IM\bm{H}_{1}\in\mathbb{R}^{(I_{{\rm M}}-1)\times I_{{\rm M}}} with 𝑯1(i,i)=1,𝑯1(i,i+1)=−1,i=1,…,IM−1\bm{H}_{1}(i,i)=1,\bm{H}_{1}(i,i+1)=-1,i=1,\ldots,I_{{\rm M}}-1, and 𝑯2∈ℝ(JM−1)×JM\bm{H}_{2}\in\mathbb{R}^{(J_{{\rm M}}-1)\times J_{{\rm M}}} is defined in the same way. When one takes 0<p≤10<p\leq 1, the regularization promotes the small total variation; see [12].
We use the Tikhonov regularization, i.e. ϕTik​(𝑯3​𝑪r)=‖𝑯3​𝑪r‖F2\phi_{\rm Tik}(\bm{H}_{3}\bm{C}_{r})=\|\bm{H}_{3}\bm{C}_{r}\|_{F}^{2} in (23), to represent the spectral smoothness, where the Tikhonov matrix is expressed as follows [6, 51]:

𝑯3=[1−21⋯00001−2⋯000⋮⋮⋮⋮⋮⋮⋮000⋯−210000⋯1−21]∈ℝ(KH−2)×KH.\bm{H}_{3}=\left[\begin{array}[]{ccccccc}1&-2&1&\cdots&0&0&0\\
0&1&-2&\cdots&0&0&0\\
\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots\\
0&0&0&\cdots&-2&1&0\\
0&0&0&\cdots&1&-2&1\\
\end{array}\right]\in\mathbb{R}^{(K_{H}-2)\times K_{\rm H}}.

This regularization promotes second order smoothness—reflecting the characteristics of the spectral signatures (see Fig. 3).
For the tensor 𝑫¯r\underline{\bm{D}}_{r}, we apply the regularization φr​(𝑫¯r)=12​‖𝑫¯r‖F2\varphi_{r}(\underline{\bm{D}}_{r})=\frac{1}{2}\|\underline{\bm{D}}_{r}\|^{2}_{F} to combat the scaling/counter-scaling effect in factorization models [15].

![Refer to caption](2512.19489v1/Ridge_ori_sparsity.png)
![Refer to caption](2512.19489v1/Ridge_T1_sparsity.png)
![Refer to caption](2512.19489v1/Ridge_T2_sparsity.png)
![Refer to caption](2512.19489v1/Ridge_T3_sparsity.png)
![Refer to caption](2512.19489v1/Ridge_T4_sparsity.png)

![Refer to caption](2512.19489v1/Ridge_ori_mode_1_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T1_mode_1_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T2_mode_1_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T3_mode_1_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T4_mode_1_smooth.png)

![Refer to caption](2512.19489v1/Ridge_ori_mode_2_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T1_mode_2_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T2_mode_2_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T3_mode_2_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T4_mode_2_smooth.png)

![Refer to caption](2512.19489v1/Ridge_ori_mode_3_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T1_mode_3_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T2_mode_3_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T3_mode_3_smooth.png)
![Refer to caption](2512.19489v1/Ridge_T4_mode_3_smooth.png)

(a)  𝒀¯S\underline{\bm{Y}}_{\rm S}
(b)  𝑻¯1\underline{\bm{T}}_{1}
(c)  𝑻¯2\underline{\bm{T}}_{2}
(d)  𝑻¯3\underline{\bm{T}}_{3}
(e)  𝑻¯4\underline{\bm{T}}_{4}

*Figure 6: Comparison of sparsity (first row) and spatial smoothness (second row to fourth row: mode-1, mode-2, and mode-3, respectively) exhibited by original 𝒀¯S\underline{\bm{Y}}_{\rm S} and each 𝑻¯r\underline{\bm{T}}_{r} for Jasper Ridge dataset with four materials.*

Using the above, our optimization objective when spatial degradation is known amounts to the following:

minimize𝜽⁡ℒ+λ​∑r=1Rϕr​(𝐀r,𝐁r,𝐂r)+η​∑r=1Rφr​(𝐃¯r),\displaystyle\minimize_{\bm{\theta}}~{\cal L}+\lambda\sum_{r=1}^{R}\phi_{r}(\bm{A}_{r},\bm{B}_{r},\bm{C}_{r})+\eta\sum_{r=1}^{R}\varphi_{r}(\underline{\bm{D}}_{r}),

(24)

where 𝜽\bm{\theta} collects all the optimization variables, ℒ{\cal L} represents the loss in (16), and λ\lambda and η\eta are two non-negative regularization parameters. We should mention that the formulation is continuously differentiable, which allows us to design efficient first-order algorithms for handling the problem.

Similarly, (17) is recast as

minimize𝜽′⁡ℒ~+λ​∑r=1Rϕr​(𝐀r,𝐁r,𝐂r)+η​∑r=1Rφr​(𝐃¯r),\displaystyle\minimize_{\bm{\theta}^{\prime}}~\widetilde{{\cal L}}+\lambda\sum_{r=1}^{R}\phi_{r}(\bm{A}_{r},\bm{B}_{r},\bm{C}_{r})+\eta\sum_{r=1}^{R}\varphi_{r}(\underline{\bm{D}}_{r}),

(25)

where ℒ~\widetilde{\cal L} represents the loss in (17).

![Refer to caption](2512.19489v1/WDC_ori_sparsity.png)
![Refer to caption](2512.19489v1/WDC_T1_sparsity.png)
![Refer to caption](2512.19489v1/WDC_T2_sparsity.png)
![Refer to caption](2512.19489v1/WDC_T3_sparsity.png)
![Refer to caption](2512.19489v1/WDC_T4_sparsity.png)

![Refer to caption](2512.19489v1/WDC_ori_mode_1_smooth.png)
![Refer to caption](2512.19489v1/WDC_T1_mode_1_smooth.png)
![Refer to caption](2512.19489v1/WDC_T2_mode_1_smooth.png)
![Refer to caption](2512.19489v1/WDC_T3_mode_1_smooth.png)
![Refer to caption](2512.19489v1/WDC_T4_mode_1_smooth.png)

![Refer to caption](2512.19489v1/WDC_ori_mode_2_smooth.png)
![Refer to caption](2512.19489v1/WDC_T1_mode_2_smooth.png)
![Refer to caption](2512.19489v1/WDC_T2_mode_2_smooth.png)
![Refer to caption](2512.19489v1/WDC_T3_mode_2_smooth.png)
![Refer to caption](2512.19489v1/WDC_T4_mode_2_smooth.png)

![Refer to caption](2512.19489v1/WDC_ori_mode_3_smooth.png)
![Refer to caption](2512.19489v1/WDC_T1_mode_3_smooth.png)
![Refer to caption](2512.19489v1/WDC_T2_mode_3_smooth.png)
![Refer to caption](2512.19489v1/WDC_T3_mode_3_smooth.png)
![Refer to caption](2512.19489v1/WDC_T4_mode_3_smooth.png)

(a)  𝒀¯S\underline{\bm{Y}}_{\rm S}
(b)  𝑻¯1\underline{\bm{T}}_{1}
(c)  𝑻¯2\underline{\bm{T}}_{2}
(d)  𝑻¯3\underline{\bm{T}}_{3}
(e)  𝑻¯4\underline{\bm{T}}_{4}

*Figure 7: Comparison of sparsity (first row) and spatial smoothness (second row to fourth row: mode-1, mode-2, and mode-3, respectively) exhibited by original 𝒀¯S\underline{\bm{Y}}_{\rm S} and each 𝑻¯r\underline{\bm{T}}_{r} for Washington DC dataset with four materials.*

###### Remark 6.

Besides the above-mentioned smoothness regularization, one could incorporate a variety of regularization terms or constraints that reflect the prior information of 𝐓¯r\underline{\bm{T}}_{r}. Some examples include:

- •

Sparsity: It is reasonable to assume 𝑻¯r\underline{\bm{T}}_{r} is sparse in space as a single material does not appear in all pixels [44].

- •

Nonlocal self-similarity: One can impose the nonlocal self-similarity regularization on each material 𝑻¯r\underline{\bm{T}}_{r}, which is motivated by the fact that a spectral image often contains many repetitive local patterns, thus a local patch often has many similar patches across the spatial domain [24, 75, 62, 19].

- •

Nonnegativity: Per the physical meaning of 𝑻¯r(i,j,:)\underline{\bm{T}}_{r}(i,j,:), namely, the spectral signature of material rr at pixel (i,j)(i,j), adding nonnegativity on 𝑻¯r\underline{\bm{T}}_{r} makes intuitive sense.

Nonetheless, in this work, our focus is not exhausting potential regularization/constraints. Instead, we use the spatial/spectral smoothness regularization in (23) to showcase the effectiveness of imposing structural constraints on the latent factors of the LMN model for CTD-based HSR.

###### Remark 7.

One may argue that the CPD and Tucker models could also incorporate similar ideas by imposing constraints or regularization on their latent components. However, there are subtle yet important distinctions. In CPD and Tucker, constraints on latent factors can only capture aggregated characteristics arising from multiple endmembers at the spectral-pixel level. By contrast, in the LMN model, constraints are imposed directly on each individual endmember, whose physical meaning and characteristics are arguably more interpretable.
To illustrate this phenomenon, we examine the spatial sparsity, spatial smoothness, and spectral smoothness of 𝐘¯S\underline{\bm{Y}}_{\rm S} and 𝐓¯r\underline{\bm{T}}_{r} using the Jasper Ridge and Washington DC datasets (Figs. 6 and 7). The 𝐓¯r\underline{\bm{T}}_{r}’s are obtained using the nonnegative BTD implementation in Tensorlab [54]. Spatial sparsity is quantified by the number of zero entries. For a tensor 𝐗¯\underline{\bm{X}}, mode-1 and mode-2 spatial smoothness are visualized through the absolute values of the elements of 𝐗¯×1𝐇1\underline{\bm{X}}\times_{1}{\bm{H}_{1}} and 𝐗¯×2𝐇2\underline{\bm{X}}\times_{2}{\bm{H}_{2}}, respectively, while spectral smoothness is shown via 𝐗¯×3𝐇3\underline{\bm{X}}\times_{3}{\bm{H}_{3}}.
The results (with all the values normalized by the largest) show that both sparsity and smoothness are more pronounced in the 𝐓¯r\underline{\bm{T}}_{r} components than in the aggregated 𝐘¯S\underline{\bm{Y}}_{\rm S}. This is likely because the spatial structure of 𝐘¯S\underline{\bm{Y}}_{\rm S} is influenced by mixtures of multiple materials (e.g., the Urban image in Fig. 3 does not exhibit strong spatial sparsity or small total variation).

### 4.2 Accelerated Alternating Gradient Descent

Both the formulations in (24) and (25) admit continuously differentiable objective functions. Hence, we employ the alternating accelerated gradient approach from [61, 12, 13, 64] to tackle the problem. The same design principle was used for the LL1-based HSR approach in [12].

For (24),
the algorithm updates the block variables {𝑨r}r=1R\{\bm{A}_{r}\}_{r=1}^{R}, {𝑩r}r=1R\{\bm{B}_{r}\}_{r=1}^{R}, {𝑪r}r=1R\{\bm{C}_{r}\}_{r=1}^{R}, and {𝑫¯r}r=1R\{\underline{\bm{D}}_{r}\}_{r=1}^{R} in an alternating manner.
For each variable, we take a one-step gradient with an extrapolation based acceleration, e.g.,

𝑨r(t+1)\displaystyle\bm{A}_{r}^{(t+1)}
←𝑨ˇr(t)−α(t)​𝑮𝑨ˇr(t),\displaystyle\leftarrow\check{\bm{A}}_{r}^{(t)}-\alpha^{(t)}\bm{G}_{\check{\bm{A}}_{r}}^{(t)},

(26)

where α(t)\alpha^{(t)} is the step size and 𝑮𝑨ˇr(t)\bm{G}_{\check{\bm{A}}_{r}}^{(t)} is the gradient of the objective taken at 𝑨ˇr(t)\check{\bm{A}}_{r}^{(t)}, in which

𝑨ˇr(t)←𝑨r(t)+μ(t)​(𝑨r(t)−𝑨r(t−1)),\displaystyle\check{\bm{A}}_{r}^{(t)}\leftarrow\bm{A}_{r}^{(t)}+\mu^{(t)}(\bm{A}_{r}^{(t)}-\bm{A}_{r}^{(t-1)}),

(27)

using a pre-defined combination parameter μ(t)\mu^{(t)}.
After updating 𝑨r\bm{A}_{r} for r=1,…,Rr=1,\ldots,R in parallel, the algorithm moves to the next block (i.e., 𝑩r\bm{B}_{r} for r=1,…,Rr=1,\ldots,R) and updates in the same manner.

The same algorithm is applied to handling (25), with the blocks iteratively changed to {𝑨~r}r=1R\{\widetilde{\bm{A}}_{r}\}_{r=1}^{R}, {𝑩~r}r=1R,{𝑨r}r=1R\{\widetilde{\bm{B}}_{r}\}_{r=1}^{R},\{\bm{A}_{r}\}_{r=1}^{R}, {𝑩r}r=1R\{\bm{B}_{r}\}_{r=1}^{R}, {𝑪r}r=1R\{\bm{C}_{r}\}_{r=1}^{R}, and {𝑫¯r}r=1R\{\underline{\bm{D}}_{r}\}_{r=1}^{R}.

Convergence of the algorithm has also been well supported in the literature [61, 48, 13, 64]. The early work [61, 64] showed asymptotic convergence of such algorithms, with careful selection of α\alpha and μ\mu. The work of [48] established finite-step convergence (or, iteration complexity) of this algorithm, showing that O⁡(1/T)O(1/T)-stationary point (see definitions in [48, 13]) is attained after TT iterations. The work of [13] extended the results of [48] by considering nonconvex constraints. Invoking these results, we have the following convergence result:

###### Proposition 8.

Using the alternating accelerated gradient algorithm to tackle (24) and (25) produces solution sequences that converge to their respective stationary points under proper choices of α(t)\alpha^{(t)} and μ(t)\mu^{(t)} for each block. In particular, there exist α(t)\alpha^{(t)} and μ(t)\mu^{(t)} for each block such that the iteration complexity is O⁡(1/T)O(1/T); i.e., the solution reaches an O⁡(1/T)O(1/T)-stationary points of (24) and (25).

The proof is by invoking the results in [48, 13] and thus is omitted.
Note that although the proof is conceptually the same as previous works, underpinning the exact choice of the parameters (e.g., step size), involves (somewhat tedious) problem-specific derivations.
To facilitate reproducible research,
the detailed algorithm description, the ways of obtaining the step size α(t)\alpha^{(t)} and the extrapolation parameter μ(t)\mu^{(t)}, and a per-iteration complexity analysis, are presented in the supplementary materials. We refer to the proposed approach as CTD with LMN for Image fusion and Multispectral-hyperspectral Boosting (CLIMB).

## 5 Experiments

In this section, we conduct various experiments to demonstrate the effectiveness of the proposed algorithm.

### 5.1 Experiment Settings

#### 5.1.1 Baselines

We compare our CLIMB algorithm with a number of baselines, including CNMF [68],
FUSE [58],
SCOTT [42], STEREO [25], SCLL1 [12], CBSTAR[3], BTDvar [43], NPTSR [63], and NLSTF [56].
Note that the first two methods are coupled low-rank matrix factorization-based HSR approaches; SCOTT, STEREO, and SCLL1 are CTD methods based on CP, Tucker, and LL1 models, respectively; CBSTAR is Tucker-based method accounting for endmember variability; BTDvar is LL1-based tensor approach accounting for endmember variability;
NPTSR and NLSTF consider the nonlocal self-similarity prior and combine the tensor sparse representation.
All experiments are performed using MATLAB 2023b on a desktop with 3.4 GHz i7 CPU and 64GB RAM.

![Refer to caption](2512.19489v1/Pavia_Variability.png)
  
![Refer to caption](2512.19489v1/WDC_Variability.png)
  
![Refer to caption](2512.19489v1/Ridge_Variability.png)

(a) Pavia University
(b) Washington DC
(c) Jasper Ridge

*Figure 8: Illustration of endmember variability of the subimages of Pavia University, Washington DC, and Jasper Ridge datasets. (The spectral signatures of materials of each dataset are manually selected pure pixels.)*

#### 5.1.2 Degradation Model

In the semi-real data experiments, we adopt the simulation strategy from [25, 12]. Specifically, a clean hyperspectral image serves as the “ground-truth” SRI, from which the simulated HSI and MSI are generated following Wald’s protocol [55]. The availability of ground-truth allows for quantitative evaluation of the recovered SRI. To generate the HSI, the SRI is first blurred band-by-band using a 9×99\times 9 Gaussian kernel and then downsampled by a factor of 8 along each spatial dimension. To generate the MSI, the spectral degradation matrix 𝑷M\bm{P}_{M} performs band selection and aggregation based on the characteristics of the multispectral sensor, as detailed in [25]. In this work, we consider two sensors: LANDSAT11
                1

              https://landsat.gsfc.nasa.gov/ and QuickBird22
                2

              https://www.satimagingcorp.com/satellite-sensors/quickbird/. Unless stated otherwise, zero-mean white Gaussian noise is added to both HSI and MSI with an SNR of 35dB. All results are averaged over 10 random trials with varying noise realizations.

Note that as we use real hyperspectral images as SRIs, due to variable illumination and atmospheric conditions, EV naturally occurs in the obtained spectral images (see Figs. 3 and 8).

#### 5.1.3 Quantitative Metrics

For the semi-real data experiments, we again follow the established convention [25, 12] to evaluate the quality of the recovered SRIs using six standard metrics [35, 67]. Specifically, we employ reconstruction signal-to-noise ratio (RSNR) (↑∞\uparrow\hskip-1.42262pt\infty), structural similarity index (SSIM) (↑1\uparrow\hskip-1.42262pt1), cross-correlation (CC) (↑1\uparrow\hskip-1.42262pt1), relative global dimensional error (ERGAS) (↓0\downarrow\hskip-1.42262pt0), root mean square error (RMSE) (↓0\downarrow\hskip-1.42262pt0), and spectral angle mapper (SAM) (↓0\downarrow\hskip-1.42262pt0). Definitions of these metrics can be found in [35, 57]. In general, higher values of SSIM, RSNR, and CC, and lower values of ERGAS, RMSE, and SAM indicate better reconstruction performance.

*Table 3: Performance for Urban data with the degradation known. (The highest and second-highest values are highlighted in bold and underlined, respectively.)*

Methods
RSNR
SSIM
CC
ERGAS
RMSE
SAM

CNMF
22.02
0.9616
0.9864
0.4085
0.0207
0.0738

FUSE
25.13
0.9778
0.9930
0.2813
0.0144
0.0504

SCOTT
23.74
0.9612
0.9900
0.3356
0.0169
0.0696

STEREO
24.52
0.9604
0.9915
0.3038
0.0154
0.0663

SCLL1
26.66
0.9837
0.9948
0.2521
0.0121
0.0504

CBSTAR
25.19
0.9798
0.9929
0.2862
0.0143
0.0612

BTDvar
23.62
0.9519
0.9893
0.3532
0.0171
0.0723

NPTSR
27.98
0.9815
0.9962
0.2058
0.0104
0.0436

NLSTF
25.93
0.9778
0.9944
0.2444
0.0131
0.0529

CLIMB
28.28
0.9809
0.9965
0.1970
0.0100
0.0434

#### 5.1.4 Parameter Settings

In our algorithm, the ranks are set as Lr=Mr=LL_{r}=M_{r}=L, and Nr=NN_{r}=N for all r=1,…,Rr=1,\ldots,R. The parameters λ\lambda, η\eta, LL, and NN are selected following the heuristic strategy proposed in [12], which involves tuning based on metrics such as the SAM computed between the observed HSI and the reconstructed HSI (after applying spatial degradation to the estimated SRI). This approach has been found effective in practice. The influence of these parameters on performance is further discussed in Sec. 5.5.2.
For the nonconvex total variation regularization, we set p=0.5p=0.5 and ε=0.01\varepsilon=0.01.
The initializations of 𝑫¯r(0),𝑨r(0),𝑩r(0),𝑪r(0){\underline{\bm{D}}_{r}^{(0)},\bm{A}_{r}^{(0)},\bm{B}_{r}^{(0)},\bm{C}_{r}^{(0)}} are as follows: 𝑫¯r(0)\underline{\bm{D}}_{r}^{(0)} is generated following the method in [32]; 𝑨r(0)\bm{A}_{r}^{(0)} and 𝑩r(0)\bm{B}_{r}^{(0)} are obtained from the right singular matrices of the skinny SVD of the mode-11 and mode-22 unfoldings 𝒀M(1)\bm{Y}_{\rm M}^{(1)} and 𝒀M(2)\bm{Y}_{\rm M}^{(2)} of the observed MSI 𝒀¯M\underline{\bm{Y}}_{\rm M}, where each unfolding 𝒀M(n)\bm{Y}_{\rm M}^{(n)} contains the mode-nn fibers as columns; 𝑪r(0)\bm{C}_{r}^{(0)} is initialized by applying the vertex component analysis [39] to the mode-33 unfolding 𝒀H(3)\bm{Y}_{\rm H}^{(3)} of the observed HSI 𝒀¯H\underline{\bm{Y}}_{\rm H}.
The algorithm terminates when the relative change in the objective function falls below 10−810^{-8} or the number of iterations exceeds 1000. For baselines, parameters are carefully tuned following the recommendations in corresponding papers.

### 5.2 Semi-real Experiments with Known Degradation Operators

In this subsection, we test the performance of our method using semi-real data under the
assumption that all 𝑷1\bm{P}_{1}, 𝑷2\bm{P}_{2}, and 𝑷M\bm{P}_{M} are known.

![Refer to caption](2512.19489v1/Urban_CNMF_band_43.png)
![Refer to caption](2512.19489v1/Urban_FUSE_band_43.png)
![Refer to caption](2512.19489v1/Urban_SCOTT_band_43.png)
![Refer to caption](2512.19489v1/Urban_STEREO_band_43.png)
![Refer to caption](2512.19489v1/Urban_SCLL1_band_43.png)
![Refer to caption](2512.19489v1/Urban_CBSTAR_band_43.png)
![Refer to caption](2512.19489v1/Urban_BTDvar_band_43.png)
![Refer to caption](2512.19489v1/Urban_NPTSR_band_43.png)
![Refer to caption](2512.19489v1/Urban_NLSTF_band_43.png)
![Refer to caption](2512.19489v1/Urban_CBTD_band_43.png)
![Refer to caption](2512.19489v1/Urban_SRI_band_43.png)

![Refer to caption](2512.19489v1/Urban_CNMF_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_FUSE_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_SCOTT_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_STEREO_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_SCLL1_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_CBSTAR_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_BTDvar_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_NPTSR_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_NLSTF_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_CBTD_band_43_res.png)
![Refer to caption](2512.19489v1/Urban_SRI_band_43_res.png)

![Refer to caption](2512.19489v1/Urban_CNMF_sam_map.png)
![Refer to caption](2512.19489v1/Urban_FUSE_sam_map.png)
![Refer to caption](2512.19489v1/Urban_SCOTT_sam_map.png)
![Refer to caption](2512.19489v1/Urban_STEREO_sam_map.png)
![Refer to caption](2512.19489v1/Urban_SCLL1_sam_map.png)
![Refer to caption](2512.19489v1/Urban_CBSTAR_sam_map.png)
![Refer to caption](2512.19489v1/Urban_BTDvar_sam_map.png)
![Refer to caption](2512.19489v1/Urban_NPTSR_sam_map.png)
![Refer to caption](2512.19489v1/Urban_NLSTF_sam_map.png)
![Refer to caption](2512.19489v1/Urban_CBTD_sam_map.png)
![Refer to caption](2512.19489v1/Urban_SRI_sam_map.png)

(a) CNMF
(b) FUSE
(c) SCOTT
(d) STEREO
(e) SCLL1
(f) CBSTAR
(g) BTDvar
(h) NPTSR
(i) NLSTF
(j) CLIMB
(k) SRI

*Figure 9: The recovered results of Urban with the degradation known. First row: the recovered SRIs of the 43rd band; Second row: the corresponding residual images of the 43rd band; Third row: the SAM maps.*

(a) RSNR
(b) SSIM
(c) CC

(d) ERGAS
(e) RMSE
(f) SAM

*Figure 10: Reconstruction metrics for Jasper Ridge under different noises.*

#### 5.2.1 Urban Dataset

In the first experiment, we use the Urban dataset with 307×307307\times 307 pixels and 210 wavelengths, captured by the HYDICE sensor over Copperas Cove, TX, U.S. After removing bands severely affected by dense water vapor and atmospheric distortions, the used subimage has a size of 200×200×162200\times 200\times 162. By applying the aforementioned spatial degradation and LANDSAT-based spectral degradation, we generate the HSI 𝒀¯H∈ℝ50×50×162\underline{\bm{Y}}_{\rm H}\in\mathbb{R}^{50\times 50\times 162} and the MSI 𝒀¯M∈ℝ200×200×6\underline{\bm{Y}}_{\rm M}\in\mathbb{R}^{200\times 200\times 6}. The tensor rank is set to R=4R=4 following [23].

Table 3 summarizes the reconstruction performance under SNR=35​dB\mathrm{SNR}=35\,\mathrm{dB}. As shown, the proposed method achieves the best performance across most evaluation metrics.

Fig. 9 shows the 43rd spectral bands of the recovered SRIs obtained by different methods, the corresponding absolute residual images (i.e., abs(𝒀¯S(:,:,k)−𝒀¯^S(:,:,k))\textrm{abs}(\underline{\bm{Y}}_{\rm S}(:,:,k)-\underline{\widehat{\bm{Y}}}_{\rm S}(:,:,k)) for k=43k=43), and the SAM maps. From Fig. 9, we observe the following: (1) the proposed CLIMB method achieves the best visual quality compared to the baselines, as illustrated in the enlarged region; (2) our method yields smaller residual values across most pixels than those of the baselines; and (3) the SAM map produced by our method is more consistent with the ground truth shown in the rightmost column.

#### 5.2.2 Jasper Ridge Dataset

For the second experiment, we use the Jasper Ridge dataset, which has the full size of 512×614×224512\times 614\times 224. This dataset was captured by the AVIRIS sensor [18] over Jasper Ridge, California [47].
After removing the water absorption bands, we use a subscene of size 100×100×198100\times 100\times 198.
The HSI is generated with a spatial resolution of 25×2525\times 25 and 198 spectral bands, while the MSI is of size 100×100×6100\times 100\times 6, obtained via LANDSAT-based spectral degradation. Following the setting in [74], the tensor rank is set to R=4R=4.

To evaluate the performance of all methods under varying noise levels, Fig. 10 presents the reconstruction metrics across different SNR values ranging from 20 dB to 50 dB. As shown, the proposed CLIMB consistently achieves the best performance across most metrics. A more detailed numerical comparison at SNR=35\mathrm{SNR}=35 dB is provided in Table 4.

*Table 4: Performance for Jasper Ridge data with the degradation known. (The highest and second-highest values are highlighted in bold and underlined, respectively.)*

Methods
RSNR
SSIM
CC
ERGAS
RMSE
SAM

CNMF
25.66
0.9649
0.9904
0.3538
0.0152
0.0720

FUSE
25.29
0.9652
0.9915
0.3455
0.0158
0.0756

SCOTT
28.67
0.9753
0.9932
0.2912
0.0107
0.0624

STEREO
27.28
0.9624
0.9893
0.3950
0.0126
0.0752

SCLL1
27.36
0.9769
0.9924
0.3276
0.0124
0.0630

CBSTAR
27.50
0.9751
0.9924
0.3129
0.0122
0.0675

BTDvar
28.35
0.9738
0.9927
0.2983
0.0111
0.0644

NPTSR
29.65
0.9757
0.9948
0.2676
0.0096
0.0642

NLSTF
27.34
0.9657
0.9905
0.4599
0.0125
0.0811

CLIMB
29.95
0.9791
0.9948
0.2645
0.0092
0.0577

*Table 5: Performance for Washington DC data with spatial degradation known. (The highest and second-highest values are highlighted in bold and underlined, respectively.)*

Methods
RSNR
SSIM
CC
ERGAS
RMSE
SAM

CNMF
22.54
0.9760
0.9581
0.9821
0.0152
0.0533

FUSE
18.55
0.9402
0.9453
0.8340
0.0240
0.1098

SCOTT
25.08
0.9827
0.9511
0.8691
0.0113
0.0477

STEREO
26.61
0.9802
0.9222
1.7985
0.0095
0.0504

SCLL1
27.51
0.9892
0.9759
0.6336
0.0086
0.0376

CBSTAR
25.46
0.9805
0.9555
0.9540
0.0108
0.0490

BTDvar
25.14
0.9797
0.9649
0.7284
0.0113
0.0554

NPTSR
29.80
0.9918
0.9704
1.0179
0.0066
0.0326

NLSTF
25.21
0.9795
0.9646
2.4879
0.0112
0.0555

CLIMB
30.37
0.9921
0.9680
0.7011
0.0062
0.0326

#### 5.2.3 Washington DC Dataset

For this experiment, we use a subimage from the Washington DC dataset collected by the HYDICE sensor33
                3

              https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html,
containing 210 bands in the spectral range of 400-2500 nm, with the spatial size of 1208×3071208\times 307.
After removing bands corrupted by atmospheric effects, the resulting SRI 𝒀¯S\underline{\bm{Y}}_{\rm S} has a size of 120×120×191120\times 120\times 191. The HSI 𝒀¯H∈ℝ30×30×191\underline{\bm{Y}}_{\rm H}\in\mathbb{R}^{30\times 30\times 191} is generated via spatial degradation, and the MSI 𝒀¯M∈ℝ120×120×6\underline{\bm{Y}}_{\rm M}\in\mathbb{R}^{120\times 120\times 6} is obtained using LANDSAT spectral degradation. We set the tensor rank to R=4R=4.

Table 5 summarizes the performance of all methods. Fig. 11 displays the recovered SRIs, absolute residual images, and SAM maps on the Indian Pines scene. The proposed method better preserves spatial smoothness and sharp edges compared to the baselines. Both quantitative and qualitative results highlight its superior performance.

![Refer to caption](2512.19489v1/WDC_CNMF_band_28.png)
![Refer to caption](2512.19489v1/WDC_FUSE_band_28.png)
![Refer to caption](2512.19489v1/WDC_SCOTT_band_28.png)
![Refer to caption](2512.19489v1/WDC_STEREO_band_28.png)
![Refer to caption](2512.19489v1/WDC_SCLL1_band_28.png)
![Refer to caption](2512.19489v1/WDC_CBSTAR_band_28.png)
![Refer to caption](2512.19489v1/WDC_BTDvar_band_28.png)
![Refer to caption](2512.19489v1/WDC_NPTSR_band_28.png)
![Refer to caption](2512.19489v1/WDC_NLSTF_band_28.png)
![Refer to caption](2512.19489v1/WDC_CBTD_band_28.png)
![Refer to caption](2512.19489v1/WDC_SRI_band_28.png)

![Refer to caption](2512.19489v1/WDC_CNMF_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_FUSE_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_SCOTT_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_STEREO_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_SCLL1_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_CBSTAR_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_BTDvar_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_NPTSR_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_NLSTF_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_CBTD_band_28_res.png)
![Refer to caption](2512.19489v1/WDC_SRI_band_28_res.png)

![Refer to caption](2512.19489v1/WDC_CNMF_sam_map.png)
![Refer to caption](2512.19489v1/WDC_FUSE_sam_map.png)
![Refer to caption](2512.19489v1/WDC_SCOTT_sam_map.png)
![Refer to caption](2512.19489v1/WDC_STEREO_sam_map.png)
![Refer to caption](2512.19489v1/WDC_SCLL1_sam_map.png)
![Refer to caption](2512.19489v1/WDC_CBSTAR_sam_map.png)
![Refer to caption](2512.19489v1/WDC_BTDvar_sam_map.png)
![Refer to caption](2512.19489v1/WDC_NPTSR_sam_map.png)
![Refer to caption](2512.19489v1/WDC_NLSTF_sam_map.png)
![Refer to caption](2512.19489v1/WDC_CBTD_sam_map.png)
![Refer to caption](2512.19489v1/WDC_SRI_sam_map.png)

(a) CNMF
(b) FUSE
(c) SCOTT
(d) STEREO
(e) SCLL1
(f) CBSTAR
(g) BTDvar
(h) NPTSR
(i) NLSTF
(j) CLIMB
(k) SRI

*Figure 11: The recovered results of Washington DC with the degradation known. First row: the recovered SRIs of the 28th band; Second row: the corresponding residual images of the 28th band; Third row: the SAM maps.*

#### 5.2.4 Pavia University Dataset

Here, we use a subimage from the Pavia University dataset, captured by the ROSIS sensor [29].
The original dataset contains 610×340610\times 340 pixels and 115 bands.
After removing bands affected by water absorption, the used SRI and HSI have sizes of 200×200×103200\times 200\times 103 and 50×50×10350\times 50\times 103, respectively. The MSI 𝒀¯M∈ℝ200×200×4\underline{\bm{Y}}_{\rm M}\in\mathbb{R}^{200\times 200\times 4} is generated using QuickBird spectral degradation. We set R=4R=4.

*Table 6: Performance for Pavia University data with spatial degradation known. (The highest and second-highest values are highlighted in bold and underlined, respectively.)*

Methods
RSNR
SSIM
CC
ERGAS
RMSE
SAM

CNMF
20.40
0.9489
0.9799
0.5438
0.0216
0.0785

FUSE
20.32
0.9389
0.9777
0.5321
0.0217
0.0811

SCOTT
24.60
0.9575
0.9906
0.3160
0.0132
0.0612

STEREO
25.40
0.9586
0.9919
0.2919
0.0121
0.0623

SCLL1
26.97
0.9776
0.9942
0.2456
0.0101
0.0502

CBSTAR
18.37
0.8499
0.9639
0.6096
0.0272
0.0941

BTDvar
25.20
0.9623
0.9917
0.3046
0.0124
0.0611

NPTSR
28.10
0.9788
0.9956
0.2171
0.0089
0.0448

NLSTF
26.85
0.9766
0.9942
0.2500
0.0102
0.0491

CLIMB
28.19
0.9773
0.9957
0.2152
0.0088
0.0448

(a) RSNR
(b) CC
(c) ERGAS
(d) RMSE

*Figure 12: RSNR, CC, ERGAS, and RMSE values of each band of Pavia University image.*

Table 6 reports the reconstruction performance under SNR=35​dB\mathrm{SNR}=35\,\mathrm{dB}, where the proposed CLIMB again outperforms baselines across most metrics. Fig. 12 plots RSNR, CC, ERGAS, and RMSE across spectral bands.

We note that the nonlocal prior-based method NPTSR and our method achieve comparable performance under the setting where spatial degradation operators are known. Nonetheless, our performance margin becomes more articulated in more challenging scenarios, as will be shown shortly in the next subsection.

![Refer to caption](2512.19489v1/Ridge_CNMF_band_34.png)
![Refer to caption](2512.19489v1/Ridge_FUSE_band_34.png)
![Refer to caption](2512.19489v1/Ridge_SCOTT_band_34.png)
![Refer to caption](2512.19489v1/Ridge_STEREO_band_34.png)
![Refer to caption](2512.19489v1/Ridge_SCLL1_band_34.png)
![Refer to caption](2512.19489v1/Ridge_CBSTAR_band_34.png)
![Refer to caption](2512.19489v1/Ridge_BTDvar_band_34.png)
![Refer to caption](2512.19489v1/Ridge_NPTSR_band_34.png)
![Refer to caption](2512.19489v1/Ridge_NLSTF_band_34.png)
![Refer to caption](2512.19489v1/Ridge_CBTD_band_34.png)
![Refer to caption](2512.19489v1/Ridge_SRI_band_34.png)

![Refer to caption](2512.19489v1/Ridge_CNMF_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_FUSE_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_SCOTT_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_STEREO_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_SCLL1_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_CBSTAR_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_BTDvar_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_NPTSR_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_NLSTF_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_CBTD_band_34_res.png)
![Refer to caption](2512.19489v1/Ridge_SRI_band_34_res.png)

![Refer to caption](2512.19489v1/Ridge_CNMF_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_FUSE_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_SCOTT_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_STEREO_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_SCLL1_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_CBSTAR_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_BTDvar_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_NPTSR_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_NLSTF_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_CBTD_sam_map.png)
![Refer to caption](2512.19489v1/Ridge_SRI_sam_map.png)

(a) CNMF
(b) FUSE
(c) BSCOTT
(d) BSTEREO
(e) BSCLL1
(f) CBSTAR
(g) BTDvar
(h) NPTSR
(i) NLSTF
(j) BCLIMB
(k) HSI

*Figure 13: The recovered results of Jasper Ridge with the spatial degradation unknown. First row: the recovered SRIs of the 34th band; Second row: the corresponding residual images of the 34th band; Third row: the SAM maps.*

*Table 7: Performance for Jasper Ridge data with the spatial degradation unknown. (The highest and second-highest values are highlighted in bold and underlined, respectively.)*

Methods
RSNR
SSIM
CC
ERGAS
RMSE
SAM

CNMF
25.52
0.9639
0.9903
0.3589
0.0155
0.0738

FUSE
16.91
0.8203
0.9520
0.8230
0.0414
0.1312

BSCOTT
27.17
0.9724
0.9926
0.3160
0.0127
0.0682

BSTEREO
24.33
0.9459
0.9855
0.4754
0.0176
0.0883

BSCLL1
27.62
0.9785
0.9928
0.3168
0.0121
0.0638

CBSTAR
19.68
0.9393
0.9794
0.5524
0.0301
0.0899

BTDvar
24.95
0.9602
0.9890
0.3812
0.0164
0.0800

NPTSR
27.02
0.9744
0.9917
0.5748
0.0129
0.0692

NLSTF
26.92
0.9647
0.9890
0.5587
0.0131
0.0809

BCLIMB
28.28
0.9716
0.9929
0.2973
0.0112
0.0668

### 5.3 Semi-real Experiments with Unknown Spatial Degradation Operators

In this subsection, we evaluate the proposed semi-blind CLIMB model (i.e., (25), referred to as BCLIMB) under the setting where the spatial degradation operator is unknown. The experiments use the Jasper Ridge and Pavia University datasets with the same configurations as before.

Since CP-, Tucker-, and LL1-based CTD methods can also operate in the semi-blind setting, we include their corresponding variants—BSTEREO [25], BSCOTT [42], and BSCLL1 [12]—as baselines. Additionally, for CBSTAR and BTDvar, we estimate the spatial degradation matrices 𝑷1\bm{P}_{1} and 𝑷2\bm{P}_{2} using the approach in [11].

Fig. 13 presents the recovered SRIs, residual images, and SAM maps for the Jasper Ridge dataset. Compared to the baselines, BCLIMB better preserves edges and textures.

Tables 7 and 8 report the reconstruction metrics for the Jasper Ridge and Pavia University datasets, respectively. In both cases, BCLIMB outperforms the baselines across most evaluation metrics.

*Table 8: Performance for Pavia University data with the spatial degradation unknown. (The highest and second-highest values are highlighted in bold and underlined, respectively.)*

Methods
RSNR
SSIM
CC
ERGAS
RMSE
SAM

CNMF
20.50
0.9512
0.9814
0.5387
0.0213
0.0771

FUSE
18.04
0.9109
0.9621
0.6586
0.0282
0.0897

BSCOTT
24.29
0.9588
0.9903
0.3286
0.0137
0.0640

BSTEREO
23.33
0.9470
0.9886
0.3690
0.0153
0.0680

BSCLL1
26.06
0.9755
0.9936
0.2776
0.0116
0.0565

CBSTAR
18.23
0.9118
0.9645
0.6568
0.0276
0.0865

BTDvar
19.39
0.8995
0.9726
0.6066
0.0242
0.0932

NPTSR
25.98
0.9770
0.9948
0.2626
0.0113
0.0520

NLSTF
26.13
0.9763
0.9936
0.2732
0.0111
0.0508

BCLIMB
27.50
0.9754
0.9949
0.2333
0.0095
0.0474

![Refer to caption](2512.19489v1/Real_CNMF_20.png)
![Refer to caption](2512.19489v1/Real_FUSE_20.png)
![Refer to caption](2512.19489v1/Real_SCOTT_20.png)
![Refer to caption](2512.19489v1/Real_STEREO_20.png)
![Refer to caption](2512.19489v1/Real_SCLL1_20.png)
![Refer to caption](2512.19489v1/Real_CBSTAR_20.png)

(a) CNMF
(b) FUSE
(c) BSCOTT
(d) BSTEREO
(e) BSCLL1
(f) CBSTAR

![Refer to caption](2512.19489v1/Real_BTDvar_20.png)
![Refer to caption](2512.19489v1/Real_NPTSR_20.png)
![Refer to caption](2512.19489v1/Real_NLSTF_20.png)
![Refer to caption](2512.19489v1/Real_CBTD_20.png)
![Refer to caption](2512.19489v1/Real_HSI_20.png)

(g) BTDvar
(h) NPTSR
(i) NLSTF
(j) BCLIMB
(k) HSI

*Figure 14: The recovered SRIs of the real-data experiment.*

### 5.4 Real Data Experiment

To further demonstrate the effectiveness of the proposed method, we evaluate all algorithms on a real HSI-MSI fusion task [65]. The MSI data is collected by the Sentinel-2A satellite, which contains 13 spectral bands. Following the setup in [59], we select 4 bands, corresponding to central wavelengths of 490 nm, 560 nm, 665 nm, and 842 nm, with 360×360360\times 360 spatial resolution.
The HSI data is acquired by the Hyperion sensor, originally with 220 bands. After removing severely corrupted bands, a subscene of size 120×120×89120\times 120\times 89 is used in this experiment.

Note that in the real scenario, both spatial and spectral degradation operators are unknown. For CBSTAR, BTDvar, NPTSR, and NLSTF, we estimate the spatial degradation matrices 𝑷1\bm{P}_{1}, 𝑷2\bm{P}_{2} and the spectral degradation matrix 𝑷M\bm{P}_{\rm M} from the observed HSI and MSI using the method in [11].
For the matrix-based baselines CNMF and FUSE, we apply their built-in procedures to estimate the degradations. For the tensor-based baselines SCOTT, STEREO, and SCLL1, we adopt their blind versions, i.e., BSCOTT, BSTEREO, and BSCLL1. The rank parameter is set to R=4R=4 based on visual inspection.

*Table 9: Performance for Washington DC and Pavia University datasets under different regularization terms with SNR=35dB.*

Datasets
Metrics
no priors

only spectral

smoothness

only spatial

smoothness

CLIMB

Washington DC
RSNR
29.01
29.57
29.81
30.37

SSIM
0.9904
0.9914
0.9917
0.9921

CC
0.9714
0.9727
0.9678
0.9680

ERGAS
0.8377
0.7996
1.0120
0.7011

RMSE
0.0072
0.0068
0.0066
0.0062

SAM
0.0384
0.0358
0.0349
0.0326

Pavia University
RSNR
24.81
25.21
27.78
28.19

SSIM
0.9492
0.9537
0.9756
0.9773

CC
0.9909
0.9916
0.9952
0.9957

ERGAS
0.3113
0.3000
0.2262
0.2152

RMSE
0.0129
0.0124
0.0092
0.0088

SAM
0.0690
0.0625
0.0465
0.0448

![Refer to caption](2512.19489v1/para_mu.png)

(a) λ\lambda
(b) η\eta
(c) L
(d) N

*Figure 15: RSNR (dB) and SSIM values under different parameters on Jasper Ridge dataset. When changing one parameter, the others are fixed to the “optimal values” as reported in the figures.*

Fig. 14 shows the reconstructed SRI images produced by all methods. All approaches enhance spatial resolution and yield improved visual quality compared to the input HSI (see subfigure (k)).
However, FUSE, CBSTAR, BTDvar, and NLSTF introduce noticeable artifacts such as glitches and striping. While CNMF preserves sharp edges, it appears to miss certain objects and fine details, particularly in the lower-right region. The baselines BSCOTT, BSTEREO, BSCLL1, and NPTSR perform reasonably well in recovering the SRI.
In comparison, the proposed BCLIMB better preserves both texture details and spatial smoothness, as evident in the highlighted regions.

### 5.5 Discussions

#### 5.5.1 Effect of Regularization Terms

We evaluate the contribution of each regularization term in the proposed model. Table 9 compares the reconstruction performance using spectral smoothness only, spatial smoothness only, and both (i.e., the full CLIMB) on the semi-real Washington DC and Pavia University datasets under SNR=35\mathrm{SNR}=35 dB.

For the Washington DC dataset, even without regularization, the proposed method outperforms the baselines, highlighting its strength in modeling endmember variability. Adding either spectral or spatial smoothness further improves performance, confirming the value of each prior. The best results are achieved when both priors are applied, supporting the design choice of integrating them into the LMN-based HSR framework.

#### 5.5.2 Effect of Parameter Selection

Fig. 15 shows the sensitivity of the proposed algorithm to key parameters, including the regularization weights λ\lambda and η\eta, and the rank parameters LL (or MM) and NN. The figure reports RSNR and SSIM values on the Jasper Ridge dataset with SNR=35\mathrm{SNR}=35 dB under different parameter settings. The results demonstrate that CLIMB maintains strong performance over a wide range of values, indicating its robustness to parameter selection.

## 6 Conclusion and Future Works

In this work, we revisited CTD-based hyperspectral super-resolution with a particular emphasis on modeling endmember variability. We introduced a tensor formulation based on the LMN decomposition, which generalizes the CPD, Tucker, and LL1 models as special cases. This formulation accommodates EV while preserving physical interpretability.
We established that, similar to existing CTD approaches, the super-resolution image under the LMN model can be exactly recovered under reasonable conditions. Moreover, we proposed a CTD algorithm based on our model by incorporating prior knowledge of the LMN latent factors—specifically, spatial and spectral smoothness—via nonconvex total variation and Tikhonov regularizations.
Extensive experiments on semi-real and real datasets, both of which exhibit nontrivial EV, validated the proposed method and consistently demonstrated superior reconstruction quality compared to state-of-the-art baselines. These results confirm the LMN framework as a flexible and interpretable tool for advancing hyperspectral super-resolution under endmember variability.

Some directions are worth exploring as future works.
First, the LMN model requires specification of more parameters compared to other models such as CPD and LL1 (e.g., the multi-linear rank of each 𝑻¯r\underline{\bm{T}}_{r}). Although the model is not very sensitive to these parameters, having a mechanism to automatically determine these parameters would be more ideal. This may be done with Bayesian tensor modeling techniques; see, e.g., [8].
Second, the basic assumption of our method (and most of the CTD methods) is that the HSI and MSI are spatially co-registered—i.e., the images cover exactly the same spatial region with the same underlying coordinate system.
In practice, this assumption is nontrivial to satisfy and may take a lot of visual/manual tuning, especially when the acquired raw HSI and MSI are spatially misaligned [73, 46].
Therefore, it would be meaningful to handle the unregistered HSR problem with recoverability and interpretability.

## Acknowledgement

The authors would like to thank Dr. Yang Xu and Dr. Wei Wan for providing their source code.

## References

- [1]
B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli (2002)

Context-driven fusion of high spatial and spectral resolution images based on oversampled multiresolution analysis.

IEEE Trans. Geosci. Remote Sens. 40 (10), pp. 2300–2312.

Cited by: §1.

- [2]
B. Aiazzi, S. Baronti, and M. Selva (2007)

Improving component substitution pansharpening through multivariate regression of MS ++ Pan data.

IEEE Trans. Geosci. Remote Sens. 45 (10), pp. 3230–3239.

Cited by: §1.

- [3]
R. A. Borsoi, C. Prévost, K. Usevich, D. Brie, J. C. M. Bermudez, and C. Richard (2021)

Coupled tensor decomposition for hyperspectral and multispectral image fusion with inter-image variability.

IEEE J. Sel. Top. Signal Process. 15 (3), pp. 702–717.

Cited by: 2nd item,
§1.1,
§2.1,
§2.1,
§2.1,
§2.2,
§2.2,
§3.1,
§3.3,
§5.1.1.

- [4]
R. A. Borsoi, T. Imbiriba, J. C. M. Bermudez, C. Richard, J. Chanussot, L. Drumetz, J. Tourneret, A. Zare, and C. Jutten (2021)

Spectral variability in hyperspectral data unmixing: a comprehensive review.

IEEE Geosci. Remote Sens. Mag. 9 (4), pp. 223–270.

Cited by: §1.1,
§2.1,
§3.1.2,
§3.1.

- [5]
R. A. Borsoi, T. Imbiriba, and J. C. M. Bermudez (2020)

Super-resolution for hyperspectral and multispectral image fusion accounting for seasonal spectral variability.

IEEE Trans. Image Process. 29 (), pp. 116–127.

Cited by: §1.2.

- [6]
S. Boyd and L. Vandenberghe (2004)

Convex optimization.

 Cambridge University Press.

Cited by: §4.1.

- [7]
A. Camacho, E. Vargas, and H. Arguello (2022)

Hyperspectral and multispectral image fusion addressing spectral variability by an augmented linear mixing model.

Int. J. Remote Sens. 43 (5), pp. 1577–1608.

Cited by: §1.2.

- [8]
L. Cheng, Z. Chen, and Y. Wu (2023)

Bayesian tensor decomposition for signal processing and machine learning.

 Springer Nature.

Cited by: §6.

- [9]
L. De Lathauwer (2008)

Decompositions of a higher-order tensor in block terms—Part I: lemmas for partitioned matrices.

SIAM J. Matrix Anal. Appl. 30 (3), pp. 1022–1032.

Cited by: Figure 5,
Lemma 11.

- [10]
L. De Lathauwer (2008)

Decompositions of a higher-order tensor in block terms—Part II: definitions and uniqueness.

SIAM J. Matrix Anal. Appl. 30 (3), pp. 1033–1066.

Cited by: Appendix B,
1st item,
§1,
Figure 5,
§3.1.1,
§3.3,
§3.3,
§3.3,
§3.3.

- [11]
R. Dian, A. Guo, and S. Li (2023)

Zero-shot hyperspectral sharpening.

IEEE Trans. Pattern Anal. Mach. Intell. 45 (10), pp. 12650–12666.

Cited by: §5.3,
§5.4.

- [12]
M. Ding, X. Fu, T-Z. Huang, J. Wang, and X-L. Zhao (2021)

Hyperspectral super-resolution via interpretable block-term tensor modeling.

IEEE J. Sel. Top. Signal Process. 15 (3), pp. 641–656.

Cited by: Appendix C,
Appendix E,
Appendix E,
Appendix E,
2nd item,
§1.1,
§1,
Figure 1,
Figure 2,
§2.1,
§2.1,
§2.1,
§2.1,
§2.1,
§2.1,
§2.2.3,
§2.2.3,
§2.2,
§2.2,
§2.3,
§3.1,
§3.2,
§3.3,
§3.3,
§4.1,
§4.2,
§5.1.1,
§5.1.2,
§5.1.3,
§5.1.4,
§5.3,
Remark 5.

- [13]
M. Ding, X. Fu, and X. Zhao (2023)

Fast and structured block-term tensor decomposition for hyperspectral unmixing.

IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens. 16 (), pp. 1691–1709.

Cited by: Appendix E,
Appendix E,
Appendix E,
§2.2.3,
§4.2,
§4.2,
§4.2.

- [14]
I. Domanov, N. Vervliet, E. Evert, and L. De Lathauwer (2024)

Decomposition of a tensor into multilinear rank-(M_r,N_r,⋅){(M\_{{r}},N\_{{r}},\cdot)} terms.

SIAM J. Matrix Anal. Appl. 45 (3), pp. 1310–1334.

Cited by: §3.3,
Lemma 9.

- [15]
X. Fu, K. Huang, W.-K. Ma, N. D. Sidiropoulos, and R. Bro (2015)

Joint tensor factorization and outlying slab suppression with applications.

IEEE Trans. Signal Process. 63 (23), pp. 6315–6328.

Cited by: Appendix E,
§4.1,
§4.1.

- [16]
N. Gillis and W. Ma (2015)

Enhancing pure-pixel identification performance via preconditioning.

SIAM J. Imaging Sci. 8 (2), pp. 1161–1186.

Cited by: §1.

- [17]
N. Gillis (2014)

Successive nonnegative projection algorithm for robust nonnegative blind source separation.

SIAM J. Imaging Sci. 7 (2), pp. 1420–1450.

Cited by: §1.

- [18]
R. O. Green and M. L. Eastwood (1998)

Imaging spectroscopy and the airborne visible/infrared imaging spectrometer (AVIRIS).

Remote Sens. Environ. 65 (3), pp. 227–248.

Cited by: §5.2.2.

- [19]
J. Guo, Y. Guo, Q. Jin, M. K. Ng, and S. Wang (2022)

Gaussian patch mixture model guided low-rank covariance matrix minimization for image denoising.

SIAM J. Imaging Sci. 15 (4), pp. 1601–1622.

Cited by: 2nd item.

- [20]
W. He, Y. Chen, N. Yokoya, C. Li, and Q. Zhao (2022)

Hyperspectral super-resolution via coupled tensor ring factorization.

Pattern Recognit. 122, pp. 108280.

Cited by: §1,
§2.3.

- [21]
F. L. Hitchcock (1927)

The expression of a tensor or a polyadic as a sum of products.

J. Math. Phys. 6 (1-4), pp. 164–189.

Cited by: §1.

- [22]
M. Iordache, J. M. Bioucas-Dias, and A. Plaza (2012)

Total variation spatial regularization for sparse hyperspectral unmixing.

IEEE Trans. Geosci. Remote Sens. 50 (11), pp. 4484–4502.

Cited by: §4.1.

- [23]
S. Jia and Y. Qian (2007)

Spectral and spatial complexity-based hyperspectral unmixing.

IEEE Trans. Geosci. Remote Sens. 45 (12), pp. 3867–3879.

Cited by: §5.2.1.

- [24]
Q. Jin, I. Grama, C. Kervrann, and Q. Liu (2017)

Nonlocal means and optimal weights for noise removal.

SIAM J. Imaging Sci. 10 (4), pp. 1878–1920.

Cited by: 2nd item.

- [25]
C. I. Kanatsoulis, X. Fu, N. D. Sidiropoulos, and W.-K. Ma (2018)

Hyperspectral super-resolution: a coupled tensor factorization approach.

IEEE Trans. Signal Process. 66 (24), pp. 6503–6517.

Cited by: Appendix C,
2nd item,
§1.1,
§1,
Figure 1,
§2.1,
§2.1,
§2.1,
§2.1,
§2.1,
§2.1,
§2.2.1,
§2.2.1,
§2.2.1,
§2.2,
§2.2,
§2.3,
§3.2,
§3.3,
§3.3,
§5.1.1,
§5.1.2,
§5.1.3,
§5.3,
Lemma 10,
Remark 5.

- [26]
C. Kervazo, N. Gillis, and N. Dobigeon (2021)

Provably robust blind source separation of linear-quadratic near-separable mixtures.

SIAM J. Imaging Sci. 14 (4), pp. 1848–1889.

Cited by: §1.1,
§3.1.

- [27]
M. Kilmer and C. Martin (2011)

Factorization strategies for third-order tensors.

Linear Algebra Appl. 435 (3), pp. 641–658.

Cited by: §1.

- [28]
T. G. Kolda and B. W. Bader (2009)

Tensor decompositions and applications.

SIAM Review 51 (3), pp. 455–500.

Cited by: §4.1.

- [29]
B. Kunkel, F. Blechinger, R. Lutz, R.Doerffer, H. van der Piepen, and M. Schroder (1988)

ROSIS (reflective optics system imaging spectrometer)-a candidate instrument for polar platform missions.

In Proc. SPIE,

Vol. 868, pp. 134–142.

Cited by: §3.1.2,
§5.2.4.

- [30]
D. Lee and H. Seung (1999)

Learning the parts of objects by non-negative matrix factorization.

Nature 401 (), pp. 788–791.

Cited by: §1.

- [31]
Q. Li, W.-K. Ma, and Q. Wu (2018)

Hyperspectral super-resolution: exact recovery in polynomial time.

In IEEE Workshop Stat. Signal Process.,

Vol. , pp. 378–382.

Cited by: §1.

- [32]
S. Li, R. Dian, L. Fang, and J. M. Bioucas-Dias (2018)

Fusing hyperspectral and multispectral images via coupled sparse tensor factorization.

IEEE Trans. Image Process. 27 (8), pp. 4118–4130.

Cited by: §1,
§5.1.4.

- [33]
C. Lin, R. Wu, W. Ma, C. Chi, and Y. Wang (2018)

Maximum volume inscribed ellipsoid: a new simplex-structured matrix factorization framework via facet enumeration and convex optimization.

SIAM J. Imaging Sci. 11 (2), pp. 1651–1679.

Cited by: §1.

- [34]
H. Liu, R. Wu, and W. Ma (2019)

Is there any recovery guarantee with coupled structured matrix factorization for hyperspectral super-resolution?.

In Proc. IEEE Int. Workshop Comput. Adv. Mult.-Sens. Adapt. Process.,

pp. .

Cited by: §1.

- [35]
L. Loncan, L. B. de Almeida, J. M. Bioucas-Dias, X. Briottet, J. Chanussot, N. Dobigeon, S. Fabre, W. Liao, G. A. Licciardi, M. Simões, J. Tourneret, M. A. Veganzones, G. Vivone, Q. Wei, and N. Yokoya (2015)

Hyperspectral pansharpening: a review.

IEEE Geosc. Remote Sens. Mag. 3 (3), pp. 27–46.

Cited by: §5.1.3.

- [36]
Y. Luo, X. Zhao, K. Ye, and D. Meng (2025)

NeurTV: total variation on the neural domain.

SIAM J. Imaging Sci. 18 (2), pp. 1101–1140.

Cited by: §4.1.

- [37]
W.-K. Ma, J. M. Bioucas-Dias, T. Chan, N. Gillis, P. Gader, A. J. Plaza, A. Ambikapathi, and C. Chi (2014)

A signal processing perspective on hyperspectral unmixing: insights from remote sensing.

IEEE Signal Process. Mag. 31 (1), pp. 67–81.

Cited by: §1.

- [38]
Z. Ma and M. K. Ng (2025)

Multispectral image restoration by generalized opponent transformation total variation.

SIAM J. Imaging Sci. 18 (1), pp. 246–279.

Cited by: §4.1.

- [39]
J. M. P. Nascimento and J. M. B. Dias (2005)

Vertex component analysis: a fast algorithm to unmix hyperspectral data.

IEEE Trans. Geosci. Remote Sens. 43 (4), pp. 898–910.

Cited by: §5.1.4.

- [40]
Y.E. Nesterov (1983)

A method for solving the convex programming problem with convergence rate O⁡(1/k2){O}(1/k^{2}).

Dokl. Akad. Nauk SSSR 269 (), pp. 543–547.

Cited by: Appendix E.

- [41]
J. Pan and M. K. Ng (2023)

Separable quaternion matrix factorization for polarization images.

SIAM J. Imaging Sci. 16 (3), pp. 1281–1307.

Cited by: §3.1.2.

- [42]
C. Prévost, K. Usevich, P. Comon, and D. Brie (2020)

Hyperspectral super-resolution with coupled Tucker approximation: recoverability and SVD-based algorithms.

IEEE Trans. Signal Process. 68 (), pp. 931–946.

Cited by: 2nd item,
§1.1,
§1,
§2.1,
§2.1,
§2.1,
§2.1,
§2.2.2,
§2.2.2,
§2.2,
§2.2,
§2.3,
§3.2,
§3.3,
§5.1.1,
§5.3.

- [43]
C. Prévost, R. A. Borsoi, K. Usevich, D. Brie, J. C. M. Bermudez, and C. Richard (2022)

Hyperspectral super-resolution accounting for spectral variability: coupled tensor LL1-based recovery and blind unmixing of the unknown super-resolution image.

SIAM J. Imaging Sci. 15 (1), pp. 110–138.

Cited by: 2nd item,
§1.1,
§1,
§2.1,
§2.1,
§2.1,
§2.2,
§2.2,
§3.1,
§3.3,
§5.1.1.

- [44]
Y. Qian, S. Jia, J. Zhou, and A. Robles-Kelly (2011)

Hyperspectral unmixing via L1/2L_{1/2} sparsity-constrained nonnegative matrix factorization.

IEEE Trans. Geosci. Remote Sens. 49 (11), pp. 4282–4297.

Cited by: 1st item.

- [45]
Y. Qian, F. Xiong, S. Zeng, J. Zhou, and Y. Y. Tang (2017)

Matrix-vector nonnegative tensor factorization for blind unmixing of hyperspectral imagery.

IEEE Trans. Geosci. Remote Sens. 55 (3), pp. 1776–1792.

Cited by: §2.2.3,
§2.2.3.

- [46]
Y. Qu, H. Qi, C. Kwan, N. Yokoya, and J. Chanussot (2022)

Unsupervised and unregistered hyperspectral image super-resolution with mutual dirichlet-net.

IEEE Trans. Geosci. Remote Sens. 60 (), pp. 1–18.

Cited by: §6.

- [47]
D. E. Sabol, D. A. Roberts, J. B. Adams, and M. O. Smith (1993)

Mapping and monitoring changes in vegetation communities of Jasper Ridge, CA, using spectral fractions derived from AVIRIS images.

In Proc. JPL Airborne Geosci. Workshop,

Vol. , pp. 157–160.

Cited by: §5.2.2.

- [48]
M. Shao, Q. Li, W. Ma, and A. M. So (2019)

A framework for one-bit and constant-envelope precoding over multiuser massive miso channels.

IEEE Trans. Signal Process. 67 (20), pp. 5309–5324.

Cited by: §4.2,
§4.2.

- [49]
N. D. Sidiropoulos, L. De Lathauwer, X. Fu, K. Huang, E. E. Papalexakis, and C. Faloutsos (2017)

Tensor decomposition for signal processing and machine learning.

IEEE Trans. Signal Process. 65 (13), pp. 3551–3582.

Cited by: §1.1,
§2.2.1.

- [50]
B. Somers, G. P. Asner, L. Tits, and P. Coppin (2011)

Endmember variability in spectral mixture analysis: a review.

Remote Sens. Environ. 115 (7), pp. 1603–1616.

Cited by: §1.1,
§3.1.

- [51]
Y. Song, E. Djermoune, J. Chen, C. Richard, and D. Brie (2019)

Online deconvolution for industrial hyperspectral imaging systems.

SIAM J. Imaging Sci. 12 (1), pp. 54–86.

Cited by: §4.1.

- [52]
M. Teboulle and Y. Vaisbourd (2020)

Novel proximal gradient methods for nonnegative matrix factorization with sparsity constraints.

SIAM J. Imaging Sci. 13 (1), pp. 381–421.

Cited by: §1.

- [53]
L. R. Tucker (1966)

Some mathematical notes on three-mode factor analysis.

Psychometrika 31 (3), pp. 279–311.

Cited by: §1.

- [54]
N. Vervliet, O. Debals, L. Sorber, M. V. Barel, and L. D. Lathauwer (2016)

Tensorlab 3.0.

Cited by: Appendix A,
Remark 7.

- [55]
L. Wald, T. Ranchin, and M. Mangolini (1997)

Fusion of satellite images of different spatial resolutions: assessing the quality of resulting images.

Photogramm. Eng. Remote Sens. 63 (6), pp. 691–699.

Cited by: §5.1.2.

- [56]
W. Wan, W. Guo, H. Huang, and J. Liu (2020)

Nonnegative and nonlocal sparse tensor factorization-based hyperspectral image super-resolution.

IEEE Trans. Geosci. Remote Sens. 58 (12), pp. 8384–8394.

Cited by: §5.1.1.

- [57]
Q. Wei, J. M. Bioucas-Dias, N. Dobigeon, J. Tourneret, M. Chen, and S. Godsill (2016)

Multiband image fusion based on spectral unmixing.

IEEE Trans. Geosci. Remote Sens. 54 (12), pp. 7236–7249.

Cited by: §2.1,
§5.1.3.

- [58]
Q. Wei, N. Dobigeon, and J. Tourneret (2015)

Fast fusion of multi-band images based on solving a Sylvester equation.

IEEE Trans. Image Process. 24 (11), pp. 4109–4121.

Cited by: §1,
§5.1.1.

- [59]
R. Wu, W.-K. Ma, X. Fu, and Q. Li (2020)

Hyperspectral super-resolution via global-local low-rank matrix estimation.

IEEE Trans. Geosci. Remote Sens. 58 (10), pp. 7125–7140.

Cited by: §1,
§2.1,
§5.4.

- [60]
T. Xu, T. Huang, L. Deng, and N. Yokoya (2022)

An iterative regularization method based on tensor subspace representation for hyperspectral image super-resolution.

IEEE Trans. Geosci. Remote Sens. 60 (), pp. 1–16.

Cited by: §1,
§2.3.

- [61]
Y. Xu and W. Yin (2013)

A block coordinate descent method for regularized multiconvex optimization with applications to nonnegative tensor factorization and completion.

SIAM J. Imaging Sci. 6 (3), pp. 1758–1789.

Cited by: §4.2,
§4.2.

- [62]
Y. Xu, Z. Wu, J. Chanussot, P. Comon, and Z. Wei (2020)

Nonlocal coupled tensor CP decomposition for hyperspectral and multispectral image fusion.

IEEE Trans. Geosci. Remote Sens. 58 (1), pp. 348–362.

Cited by: 2nd item.

- [63]
Y. Xu, Z. Wu, J. Chanussot, and Z. Wei (2019)

Nonlocal patch tensor sparse representation for hyperspectral image super-resolution.

IEEE Trans. Image Process. 28 (6), pp. 3034–3047.

Cited by: §5.1.1.

- [64]
Y. Xu and W. Yin (2017)

A globally convergent algorithm for nonconvex optimization based on block coordinate update.

J. Sci. Comput. 72 (2), pp. 700–734.

Cited by: Appendix E,
§4.2,
§4.2.

- [65]
J. Yang, Y.-Q. Zhao, and J. C.-W. Chan (2018)

Hyperspectral and multispectral image fusion via deep two-branches convolutional neural network.

Remote Sens. 10 (5), pp. 1–23.

Cited by: §5.4.

- [66]
F. Ye, Z. Wu, Y. Xu, H. Liu, and Z. Wei (2022)

Bayesian hyperspectral image super-resolution in the presence of spectral variability.

IEEE Trans. Geosci. Remote Sens. 60 (), pp. 1–13.

Cited by: §1.2.

- [67]
N. Yokoya, C. Grohnfeldt, and J. Chanussot (2017)

Hyperspectral and multispectral data fusion: a comparative review of the recent literature.

IEEE Geosc. Remote Sens. Mag. 5 (2), pp. 29–56.

Cited by: §1,
§5.1.3.

- [68]
N. Yokoya, T. Yairi, and A. Iwasaki (2012)

Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion.

IEEE Trans. Geosci. Remote Sens. 50 (2), pp. 528–537.

Cited by: §1,
§2.1,
§5.1.1.

- [69]
A. Zare and K. Ho (2013)

Endmember variability in hyperspectral analysis: addressing spectral variability during spectral unmixing.

IEEE Signal Process. Mag. 31 (1), pp. 95–104.

Cited by: §1.1,
§3.1.

- [70]
Q. Zhao, G. Zhou, S. Xie, L. Zhang, and A. Cichocki (2016)

Tensor ring decomposition.

Arxiv: arXiv:1606.05535.

Cited by: §1.

- [71]
H. Zheng, Y. Lou, G. Tian, and C. Wang (2024)

A scale-invariant relaxation in low-rank tensor recovery with an application to tensor completion.

SIAM J. Imaging Sci. 17 (1), pp. 756–783.

Cited by: §1.

- [72]
Y. Zhou, A. Rangarajan, and P. D. Gader (2016)

A spatial compositional model for linear unmixing and endmember uncertainty estimation.

IEEE Trans. Image Process. 25 (12), pp. 5987–6002.

Cited by: §3.1.2.

- [73]
Y. Zhou, A. Rangarajan, and P. D. Gader (2020)

An integrated approach to registration and fusion of hyperspectral and multispectral images.

IEEE Trans. Geosci. Remote Sens. 58 (5), pp. 3020–3033.

Cited by: §6.

- [74]
F. Zhu, Y. Wang, B. Fan, S. Xiang, G. Meng, and C. Pan (2014)

Spectral unmixing via data-guided sparsity.

IEEE Trans. Image Process. 23 (12), pp. 5412–5427.

Cited by: §5.2.2.

- [75]
L. Zhuang, X. Fu, M. K. Ng, and J. M. Bioucas-Dias (2021)

Hyperspectral image denoising based on global and nonlocal low-rank factorizations.

IEEE Trans. Geosci. Remote Sens. 59 (12), pp. 10438–10454.

Cited by: 2nd item.

Supplemental Materials for

*“Rethinking Coupled Tensor Analysis for Hyperspectral Super-Resolution: Recoverable Modeling Under Endmember Variability”*

*Meng Ding and Xiao Fu

*

## Appendix A Low Multilinear Rank of Materials of Spectral Images

To verify the low multilinear rank of 𝑻¯r\underline{\bm{T}}_{r}, in Figs. 16 and 17, we show the singular values of the mode-nn (where n=1,2,3n=1,2,3) unfoldings of 𝑻¯r\underline{\bm{T}}_{r} extracted from Jasper Ridge and Washington DC.
The terms 𝑻¯r\underline{\bm{T}}_{r}’s are obtained using the nonnegative BTD implementation in Tensorlab [54].
Note that if 𝑻¯r\underline{\bm{T}}_{r} admits low multi-linear rank, then its mode-nn unfoldings all should exhibit low matrix rank.

(a)  𝑻¯1\underline{\bm{T}}_{1}
(b)  𝑻¯2\underline{\bm{T}}_{2}
(c)  𝑻¯3\underline{\bm{T}}_{3}
(d)  𝑻¯4\underline{\bm{T}}_{4}

*Figure 16: The singular values of unfolding matrices (top to bottom: mode-1, mode-2, and mode-3, respectively) of each 𝑻¯r\underline{\bm{T}}_{r} for Jasper Ridge dataset.*

From Figs. 16 and 17, one can see that for Jasper Ridge, the first 30 principal components (the first 30 left, right singular vectors and singular values) of the mode-1 unfolding matrices of 𝑻¯r\underline{\bm{T}}_{r}s contain more than 90%90\% of its energy.
For mode-2 and 3 unfolding matrices of 𝑻¯r\underline{\bm{T}}_{r}s, the first 30 and 5 principal components contain more than 90%90\% and 98%98\% of the corresponding energy.
For Washington DC, the first 40 (40, 4) principal components of the mode-1 (2, 3) unfolding matrices of 𝑻¯r\underline{\bm{T}}_{r}s contain about 90%90\% (90%90\%, 98%98\%) of its energy.

(a)  𝑻¯1\underline{\bm{T}}_{1}
(b)  𝑻¯2\underline{\bm{T}}_{2}
(c)  𝑻¯3\underline{\bm{T}}_{3}
(d)  𝑻¯4\underline{\bm{T}}_{4}

*Figure 17: The singular values of unfolding matrices (top to bottom: mode-1, mode-2, and mode-3, respectively) of each 𝑻¯r\underline{\bm{T}}_{r} for Washington DC dataset.*

## Appendix B Proof of Theorem 2

To prove the theorem, we first invoke the following lemma:

###### Lemma 9 (Corollary 2.6 [14]).

Let 𝐘¯∈ℝI×J×K\underline{\bm{Y}}\in\mathbb{R}^{I\times J\times K} be a sum of RR rank-(L,M,⋅)(L,M,\cdot) terms, i.e.,
𝐘¯=∑r=1R𝐃¯r×1𝐀r×2𝐁r\underline{\bm{Y}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r} with
𝐀r∈ℝI×L,𝐁r∈ℝJ×M,𝐃¯r∈ℝL×M×K,r=1,…,R\bm{A}_{r}\in\mathbb{R}^{I\times L},\bm{B}_{r}\in\mathbb{R}^{J\times M},\underline{\bm{D}}_{r}\in\mathbb{R}^{L\times M\times K},~r=1,\ldots,R. Assume that 𝐀r\bm{A}_{r}, 𝐁r\bm{B}_{r}, and 𝐃¯r\underline{\bm{D}}_{r} are drawn from any absolutely continuous distributions. Then, the rank-(L,M,⋅)(L,M,\cdot) decomposition of 𝐘¯\underline{\bm{Y}} is essentially unique almost surely, if

I≥L​R,J≥M​R,K≥max⁡{⌈L/M⌉+⌈M/L⌉,3},1≤r≤R.\displaystyle I\geq LR,~J\geq MR,~K\geq\max\{\lceil L/M\rceil+\lceil M/L\rceil,3\},~1\leq r\leq R.

Note that any LMN representation has a corresponding (L,M,⋅)(L,M,\cdot) representation.
Indeed, assume that 𝒀¯=∑r=1R𝑫¯r×1𝑨r×2𝑩r×3𝑪r\underline{\bm{Y}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{C}_{r}.
One can let

𝑫¯r′=𝑫¯r×3𝑪r,{\underline{\bm{D}}}_{r}^{\prime}=\underline{\bm{D}}_{r}\times_{3}\bm{C}_{r},

which leads to

𝒀¯=∑r=1R𝑫¯r×1𝑨r×2𝑩r×3𝑪r=∑r=1R𝑫¯r′×1𝑨r×2𝑩r.\displaystyle\underline{\bm{Y}}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{C}_{r}=\sum_{r=1}^{R}{\underline{\bm{D}}}_{r}^{\prime}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}.

(28)

Next, let us consider that the LMN tensor on the LHS of (28) has dimensions satisfying (9).
Then, the corresponding (L,M,⋅)(L,M,\cdot) representation on the RHS in (28) is essentially unique. Next, we follow the idea in [10, Theorem 5.1] to show that the LMN decomposition on the LHS is also essentially unique.

By the essential uniqueness in Lemma 9, we have that, for any 𝑨¯r∈ℝI×L\bar{\bm{A}}_{r}\in\mathbb{R}^{I\times L} and 𝑩¯r∈ℝJ×M\bar{\bm{B}}_{r}\in\mathbb{R}^{J\times M} that satisfy 𝒀¯=∑r=1R𝑫¯¯r′×1𝑨¯r×2𝑩¯r\underline{\bm{Y}}=\sum_{r=1}^{R}\bar{\underline{\bm{D}}}_{r}^{\prime}\times_{1}\bar{\bm{A}}_{r}\times_{2}\bar{\bm{B}}_{r}, the following holds:

𝑨¯π⁡(r)=𝑨r​𝚯a,r,𝑩¯π⁡(r)=𝑩r​𝚯b,r,\displaystyle\bar{\bm{A}}_{\pi(r)}=\bm{A}_{r}\bm{\Theta}_{a,r},~\bar{\bm{B}}_{\pi(r)}=\bm{B}_{r}\bm{\Theta}_{b,r},

(29)

where 𝚯a,r\bm{\Theta}_{a,r} and 𝚯b,r\bm{\Theta}_{b,r} are nonsingular matrices. Let 𝑫¯¯r′=𝑫¯¯r×3𝑪¯r\bar{\underline{\bm{D}}}_{r}^{\prime}=\bar{\underline{\bm{D}}}_{r}\times_{3}\bar{\bm{C}}_{r} and 𝑫¯r′=𝑫¯r×3𝑪r{\underline{\bm{D}}}_{r}^{\prime}=\underline{\bm{D}}_{r}\times_{3}\bm{C}_{r}, we have

𝑫¯¯′π⁡(r)×1𝚯a,r×2𝚯b,r=𝑫¯r′,\displaystyle\underline{\bar{\bm{D}}}^{\prime}_{\pi(r)}\times_{1}\bm{\Theta}_{a,r}\times_{2}\bm{\Theta}_{b,r}=\underline{\bm{D}}_{r}^{\prime},

(30)

i.e.,

𝑫¯¯π⁡(r)×1𝚯a,r×2𝚯b,r×3𝑪¯π⁡(r)=𝑫¯r×3𝑪r.\displaystyle\underline{\bar{\bm{D}}}_{\pi(r)}\times_{1}\bm{\Theta}_{a,r}\times_{2}\bm{\Theta}_{b,r}\times_{3}\bar{\bm{C}}_{\pi(r)}=\underline{\bm{D}}_{r}\times_{3}\bm{C}_{r}.

(31)

Considering the mode-3 unfolding 𝑫r(3)∈ℝN×L​M\bm{D}_{r}^{(3)}\in\mathbb{R}^{N\times LM} and 𝑫¯π⁡(r)(3)∈ℝN×L​M\bar{\bm{D}}_{\pi(r)}^{(3)}\in\mathbb{R}^{N\times LM} of tensors 𝑫¯¯r\underline{\bar{\bm{D}}}_{r} and 𝑫¯¯π⁡(r)\underline{\bar{\bm{D}}}_{\pi(r)}, we have

𝑪¯π⁡(r)​𝑫¯π⁡(r)(3)=𝑪r​𝑫r(3)​(𝚯b,r−1⊗𝚯a,r−1)⊤,\displaystyle\bar{\bm{C}}_{\pi(r)}\bar{\bm{D}}_{\pi(r)}^{(3)}=\bm{C}_{r}\bm{D}_{r}^{(3)}(\bm{\Theta}_{b,r}^{-1}\otimes\bm{\Theta}_{a,r}^{-1})^{\top},

where ⊗\otimes denotes the Kronecker product.
Since 𝑫¯r\underline{\bm{D}}_{r} and 𝑪r\bm{C}_{r} are drawn from any absolutely continuous distributions, and N≤L​MN\leq LM, then rank​(𝑫r(3))=N\textrm{rank}(\bm{D}_{r}^{(3)})=N, and rank​(𝑪r)=N\textrm{rank}(\bm{C}_{r})=N is full column rank almost surely.
We also get that

N\displaystyle N
=rank​(𝑪r​𝑫r(3)​(𝚯b,r−1⊗𝚯a,r−1)⊤)≤rank​(𝑪¯π⁡(r))≤N,\displaystyle=\textrm{rank}(\bm{C}_{r}\bm{D}_{r}^{(3)}(\bm{\Theta}_{b,r}^{-1}\otimes\bm{\Theta}_{a,r}^{-1})^{\top})\leq\textrm{rank}(\bar{\bm{C}}_{\pi(r)})\leq N,

N\displaystyle N
=rank​(𝑪r​𝑫r(3)​(𝚯b,r−1⊗𝚯a,r−1)⊤)≤rank​(𝑫¯π⁡(r)(3))≤N,\displaystyle=\textrm{rank}(\bm{C}_{r}\bm{D}_{r}^{(3)}(\bm{\Theta}_{b,r}^{-1}\otimes\bm{\Theta}_{a,r}^{-1})^{\top})\leq\textrm{rank}(\bar{\bm{D}}_{\pi(r)}^{(3)})\leq N,

i.e., rank​(𝑪¯π⁡(r))=N\textrm{rank}(\bar{\bm{C}}_{\pi(r)})=N and rank​(𝑫¯π⁡(r)(3))=N\textrm{rank}(\bar{\bm{D}}_{\pi(r)}^{(3)})=N almost surely.

Let 𝚯c,r=𝑫r(3)​(𝚯b,r−1⊗𝚯a,r−1)⊤​(𝑫¯π⁡(r)(3))†∈ℝN×N\bm{\Theta}_{c,r}=\bm{D}_{r}^{(3)}(\bm{\Theta}_{b,r}^{-1}\otimes\bm{\Theta}_{a,r}^{-1})^{\top}(\bar{\bm{D}}_{\pi(r)}^{(3)})^{\dagger}\in\mathbb{R}^{N\times N}, where (𝑫¯π⁡(r)(3))†(\bar{\bm{D}}_{\pi(r)}^{(3)})^{\dagger} is the pseudo-inverse of 𝑫¯π⁡(r)(3)\bar{\bm{D}}_{\pi(r)}^{(3)}, then 𝑪¯π⁡(r)=𝑪r​𝚯c,r\bar{\bm{C}}_{\pi(r)}=\bm{C}_{r}\bm{\Theta}_{c,r}.
Due to N=rank​(𝑪¯π⁡(r))≤rank​(𝚯c,r)≤NN=\textrm{rank}(\bar{\bm{C}}_{\pi(r)})\leq\textrm{rank}(\bm{\Theta}_{c,r})\leq N, we have rank​(𝚯c,r)=N\textrm{rank}(\bm{\Theta}_{c,r})=N, i.e., 𝚯c,r\bm{\Theta}_{c,r} is a nonsingular matrix.
We rewrite (31) as

𝑫¯¯π⁡(r)×1𝚯a,r×2𝚯b,r×3(𝑪r𝚯c,r)=𝑫¯r×3𝑪r.\displaystyle\underline{\bar{\bm{D}}}_{\pi(r)}\times_{1}\bm{\Theta}_{a,r}\times_{2}\bm{\Theta}_{b,r}\times_{3}(\bm{C}_{r}\bm{\Theta}_{c,r})=\underline{\bm{D}}_{r}\times_{3}\bm{C}_{r}.

(32)

By applying mode-3 multiplication by (𝑪r⊤​𝑪r)−1​𝑪r⊤(\bm{C}_{r}^{\top}\bm{C}_{r})^{-1}\bm{C}_{r}^{\top} to both sides of (32), we have

𝑫¯¯π⁡(r)×1𝚯a,r×2𝚯b,r×3𝚯c,r=𝑫¯r.\displaystyle\underline{\bar{\bm{D}}}_{\pi(r)}\times_{1}\bm{\Theta}_{a,r}\times_{2}\bm{\Theta}_{b,r}\times_{3}\bm{\Theta}_{c,r}=\underline{\bm{D}}_{r}.

(33)

Therefore, the LMN decomposition of 𝒀¯\underline{\bm{Y}} is essentially unique almost surely. This completes the proof.

## Appendix C Proof of Theorem 3

The proof steps resemble those in [25, 12]. Nonetheless, as the LMN structure and its essential uniqueness definition are different from those of the CPD and LL1 models in [25, 12], careful modifications are made to accommodate the changes in tensor models.

To proceed, we present two important lemmas.

###### Lemma 10.

[25]
Let 𝐀~=𝐏​𝐀\widetilde{\bm{A}}=\bm{P}\bm{A}, where the elements of 𝐀\bm{A} are drawn from any absolutely continuous joint distribution in ℝI×L\mathbb{R}^{I\times L}, and 𝐏∈ℝI′×I\bm{P}\in\mathbb{R}^{I^{\prime}\times I} is full row rank. Then, the elements of 𝐀~\widetilde{\bm{A}} follow an absolutely continuous joint distribution in ℝI′×L\mathbb{R}^{I^{\prime}\times L}.

###### Lemma 11.

[9]
Assume {𝐀r∈ℝI×Lr}r=1R\{\bm{A}_{r}\in\mathbb{R}^{I\times L_{r}}\}_{r=1}^{R} and {𝐁r∈ℝJ×Mr}r=1R\{\bm{B}_{r}\in\mathbb{R}^{J\times M_{r}}\}_{r=1}^{R} are drawn from any absolutely continuous joint distributions, then
rank​([𝐁1⊗𝐀1,…,𝐁R⊗𝐀R])=min⁡(I​J,∑r=1RLr​Mr)\textrm{rank}([\bm{B}_{1}\otimes\bm{A}_{1},\ldots,\bm{B}_{R}\otimes\bm{A}_{R}])=\min(IJ,\sum_{r=1}^{R}L_{r}M_{r}).

Assume
{𝑫¯r⋆,𝑨r⋆,𝑩r⋆,𝑪r⋆}r=1R\{\underline{\bm{D}}_{r}^{\star},\bm{A}_{r}^{\star},\bm{B}_{r}^{\star},\bm{C}_{r}^{\star}\}_{r=1}^{R} to be an optimal solution to (16),
and let {𝑫¯r,𝑨r,𝑩r,𝑪r}r=1R\{\underline{\bm{D}}_{r},\bm{A}_{r},\bm{B}_{r},\bm{C}_{r}\}_{r=1}^{R} represent the ground-truth factors of the SRI. Then, we have the following expressions:

𝒀¯M\displaystyle\underline{\bm{Y}}_{\rm M}
=∑r=1R𝑫¯r⋆×1𝑨r⋆×2𝑩r⋆×3𝑷M𝑪r⋆,\displaystyle=\sum_{r=1}^{R}\underline{\bm{D}}_{r}^{\star}\times_{1}\bm{A}_{r}^{\star}\times_{2}\bm{B}_{r}^{\star}\times_{3}\bm{P}_{\rm M}\bm{C}_{r}^{\star},

=∑r=1R𝑫¯r×1𝑨r×2𝑩r×3𝑷M𝑪r,\displaystyle=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{P}_{\rm M}\bm{C}_{r},

(34)

𝒀¯H\displaystyle\underline{\bm{Y}}_{\rm H}
=∑r=1R𝑫¯r⋆×1(𝑷1𝑨r⋆)×2(𝑷2𝑩r⋆)×3𝑪r⋆,\displaystyle=\sum_{r=1}^{R}\underline{\bm{D}}_{r}^{\star}\times_{1}(\bm{P}_{1}\bm{A}_{r}^{\star})\times_{2}(\bm{P}_{2}\bm{B}_{r}^{\star})\times_{3}\bm{C}_{r}^{\star},

=∑r=1R𝑫¯r×1(𝑷1𝑨r)×2(𝑷2𝑩r)×3𝑪r.\displaystyle=\sum_{r=1}^{R}\underline{\bm{D}}_{r}\times_{1}(\bm{P}_{1}\bm{A}_{r})\times_{2}(\bm{P}_{2}\bm{B}_{r})\times_{3}\bm{C}_{r}.

(35)

Since that 𝑷1\bm{P}_{1}, 𝑷2\bm{P}_{2}, and 𝑷M\bm{P}_{\rm M} are full row rank,
using Lemma 10, 𝑷1​𝑨r\bm{P}_{1}\bm{A}_{r}, 𝑷2​𝑩r\bm{P}_{2}\bm{B}_{r}, and 𝑷M​𝑪r\bm{P}_{\rm M}\bm{C}_{r} are drawn from certain absolutely continuous distributions.
Under the conditions stated in Theorem 3, the LMN representation of MSI 𝒀¯M\underline{\bm{Y}}_{\rm M} is essentially unique, which means that from (34), the following expressions hold:

𝑨π⁡(r)⋆=𝑨r​𝚯a,r,𝑩π⁡(r)⋆=𝑩r​𝚯b,r,𝑷M​𝑪π⁡(r)⋆=𝑷M​𝑪r​𝚯c,r,\displaystyle\bm{A}^{\star}_{\pi(r)}=\bm{A}_{r}\bm{\Theta}_{a,r},~\bm{B}^{\star}_{\pi(r)}=\bm{B}_{r}\bm{\Theta}_{b,r},~\bm{P}_{\rm M}\bm{C}^{\star}_{\pi(r)}=\bm{P}_{\rm M}\bm{C}_{r}\bm{\Theta}_{c,r},

(36)

where π\pi is permutation of: {1,…,R}\{1,\ldots,R\}, 𝚯a,r\bm{\Theta}_{a,r}, 𝚯b,r\bm{\Theta}_{b,r}, and 𝚯c,r\bm{\Theta}_{c,r} are nonsingular matrices.

To proceed, we consider the mode-3 unfolding of 𝒀¯H\underline{\bm{Y}}_{\rm H}, i.e., rearranging the pixels of 𝒀¯H\underline{\bm{Y}}_{\rm H} as following:

𝒀H(3)=[𝒀¯H(1,1,:),𝒀¯H(2,1,:),…,𝒀¯H(IH,JH,:)].\displaystyle\bm{Y}_{H}^{(3)}=\left[\underline{\bm{Y}}_{H}(1,1,:),\underline{\bm{Y}}_{H}(2,1,:),\ldots,\underline{\bm{Y}}_{H}(I_{\rm H},J_{\rm H},:)\right].

From (35) and (36), we have

𝒀H(3)\displaystyle\bm{Y}_{H}^{(3)}

(37)

=\displaystyle=
[𝑪1,…,𝑪R]​blockdiag​(𝑫1(3),…,𝑫R(3))​[𝑷2​𝑩1⊗𝑷1​𝑨1,…,𝑷2​𝑩R⊗𝑷1​𝑨R]⊤\displaystyle[\bm{C}_{1},\ldots,\bm{C}_{R}]\textrm{blockdiag}(\bm{D}_{1}^{(3)},\ldots,\bm{D}_{R}^{(3)})[\bm{P}_{2}\bm{B}_{1}\otimes\bm{P}_{1}\bm{A}_{1},\ldots,\bm{P}_{2}\bm{B}_{R}\otimes\bm{P}_{1}\bm{A}_{R}]^{\top}

=\displaystyle=
[𝑪π⁡(1)⋆,…,𝑪π⁡(R)⋆]​blockdiag​(𝑫π⁡(1)⋆(3),…,𝑫π⁡(R)⋆(3))​[𝑷2​𝑩π⁡(1)⋆⊗𝑷1​𝑨π⁡(1)⋆,…,𝑷2​𝑩π⁡(R)⋆⊗𝑷1​𝑨π⁡(R)⋆]⊤\displaystyle[\bm{C}_{\pi(1)}^{\star},\ldots,\bm{C}_{\pi(R)}^{\star}]\textrm{blockdiag}(\bm{D}_{\pi(1)}^{\star(3)},\ldots,\bm{D}_{\pi(R)}^{\star(3)})[\bm{P}_{2}\bm{B}_{\pi(1)}^{\star}\otimes\bm{P}_{1}\bm{A}_{\pi(1)}^{\star},\ldots,\bm{P}_{2}\bm{B}_{\pi(R)}^{\star}\otimes\bm{P}_{1}\bm{A}_{\pi(R)}^{\star}]^{\top}

=\displaystyle=
[𝑪π⁡(1)⋆,…,𝑪π⁡(R)⋆]​blockdiag​(𝑫π⁡(1)⋆(3)​(𝚯b,1⊗𝚯a,1),…,𝑫π⁡(R)⋆(3)​(𝚯b,R⊗𝚯a,R))\displaystyle[\bm{C}_{\pi(1)}^{\star},\ldots,\bm{C}_{\pi(R)}^{\star}]\textrm{blockdiag}\left(\bm{D}_{\pi(1)}^{\star(3)}(\bm{\Theta}_{b,1}\otimes\bm{\Theta}_{a,1}),\ldots,\bm{D}_{\pi(R)}^{\star(3)}(\bm{\Theta}_{b,R}\otimes\bm{\Theta}_{a,R})\right)

[𝑷2​𝑩1⊗𝑷1​𝑨1,…,𝑷2​𝑩R⊗𝑷1​𝑨R]⊤,\displaystyle\phantom{=\;\;}\hskip 227.62204pt[\bm{P}_{2}\bm{B}_{1}\otimes\bm{P}_{1}\bm{A}_{1},\ldots,\bm{P}_{2}\bm{B}_{R}\otimes\bm{P}_{1}\bm{A}_{R}]^{\top},

where ⊗\otimes denotes the Kronecker product.
Note that 𝑷1​𝑨r∈ℝIH×L\bm{P}_{1}\bm{A}_{r}\in\mathbb{R}^{I_{H}\times L} and 𝑷2​𝑩r∈ℝJH×M\bm{P}_{2}\bm{B}_{r}\in\mathbb{R}^{J_{H}\times M} are drawn from any absolutely continuous distributions, then rank​([𝑷2​𝑩1⊗𝑷1​𝑨1,…,𝑷2​𝑩R⊗𝑷1​𝑨R])=min⁡(IH​JH,L​M​R)\textrm{rank}([\bm{P}_{2}\bm{B}_{1}\otimes\bm{P}_{1}\bm{A}_{1},\ldots,\bm{P}_{2}\bm{B}_{R}\otimes\bm{P}_{1}\bm{A}_{R}])=\min(I_{H}J_{H},LMR) using Lemma 11. Due to IH​JH≥L​M​RI_{H}J_{H}\geq LMR, [𝑷2​𝑩1⊗𝑷1​𝑨1,…,𝑷2​𝑩R⊗𝑷1​𝑨R][\bm{P}_{2}\bm{B}_{1}\otimes\bm{P}_{1}\bm{A}_{1},\ldots,\bm{P}_{2}\bm{B}_{R}\otimes\bm{P}_{1}\bm{A}_{R}] has full column rank.
Multiplying ([𝑷2​𝑩1⊗𝑷1​𝑨1,…,𝑷2​𝑩R⊗𝑷1​𝑨R]⊤)†\left([\bm{P}_{2}\bm{B}_{1}\otimes\bm{P}_{1}\bm{A}_{1},\ldots,\bm{P}_{2}\bm{B}_{R}\otimes\bm{P}_{1}\bm{A}_{R}]^{\top}\right)^{\dagger} to the right of (37), we obtain that

𝑪π⁡(r)⋆​𝑫π⁡(r)⋆(3)​(𝚯b,r⊗𝚯a,r)=𝑪r​𝑫r(3),1≤r≤R,\displaystyle\bm{C}_{\pi(r)}^{\star}\bm{D}_{\pi(r)}^{\star(3)}(\bm{\Theta}_{b,r}\otimes\bm{\Theta}_{a,r})=\bm{C}_{r}\bm{D}_{r}^{(3)},~1\leq r\leq R,

i.e.,

𝑫¯π⁡(r)⋆×1𝚯a,r×2𝚯b,r×3𝑪π⁡(r)⋆=𝑫¯r×3𝑪r.\displaystyle\underline{\bm{D}}_{\pi(r)}^{\star}\times_{1}\bm{\Theta}_{a,r}\times_{2}\bm{\Theta}_{b,r}\times_{3}\bm{C}_{\pi(r)}^{\star}=\underline{\bm{D}}_{r}\times_{3}\bm{C}_{r}.

(38)

Combining (36) and (38), one can get

𝑫¯π⁡(r)⋆×1𝑨π⁡(r)⋆×2𝑩π⁡(r)⋆×3𝑪π⁡(r)⋆=𝑫¯r×1𝑨r×2𝑩r×3𝑪r.\displaystyle\underline{\bm{D}}_{\pi(r)}^{\star}\times_{1}\bm{A}_{\pi(r)}^{\star}\times_{2}\bm{B}_{\pi(r)}^{\star}\times_{3}\bm{C}_{\pi(r)}^{\star}=\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{C}_{r}.

Hence, the ground-truth can be recovered by
𝒀¯S=∑r=1R𝑫¯r⋆×1𝑨r⋆×2𝑩r⋆×3𝑪r⋆\underline{\bm{Y}}_{\rm S}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}^{\star}\times_{1}\bm{A}_{r}^{\star}\times_{2}\bm{B}_{r}^{\star}\times_{3}\bm{C}_{r}^{\star}.
This completes the proof.

## Appendix D Proof of Theorem 4

Assume that ({𝑨~r⋆,𝑩~r⋆}r=1R,{𝑫¯r⋆,𝑨r⋆,𝑩r⋆,𝑪r⋆}r=1R)(\{\widetilde{\bm{A}}_{r}^{\star},\widetilde{\bm{B}}_{r}^{\star}\}_{r=1}^{R},\{\underline{\bm{D}}_{r}^{\star},\bm{A}_{r}^{\star},\bm{B}_{r}^{\star},\bm{C}_{r}^{\star}\}_{r=1}^{R}) is an optimal solution of (17), then

𝒀¯H=∑r=1R𝑫¯r⋆×1𝑨~r⋆×2𝑩~r⋆×3𝑪r⋆,\displaystyle\underline{\bm{Y}}_{\rm H}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}^{\star}\times_{1}\widetilde{\bm{A}}_{r}^{\star}\times_{2}\widetilde{\bm{B}}_{r}^{\star}\times_{3}\bm{C}_{r}^{\star},

(39)

𝒀¯M=∑r=1R𝑫¯r⋆×1𝑨r⋆×2𝑩r⋆×3(𝑷M𝑪r⋆).\displaystyle\underline{\bm{Y}}_{\rm M}=\sum_{r=1}^{R}\underline{\bm{D}}_{r}^{\star}\times_{1}\bm{A}_{r}^{\star}\times_{2}\bm{B}_{r}^{\star}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r}^{\star}).

(40)

Based on the assumptions and Theorem 2, the MSI decomposition is essentially unique almost surely. Thus,
we have the following with probability one:

𝑨π1​(r)⋆=𝑨r​𝚯a,r,𝑩π1​(r)⋆=𝑩r​𝚯b,r,𝑷M​𝑪π1​(r)⋆=𝑷M​𝑪r​𝚯c,r,\displaystyle\bm{A}^{\star}_{\pi_{1}(r)}=\bm{A}_{r}\bm{\Theta}_{a,r},~\bm{B}^{\star}_{\pi_{1}(r)}=\bm{B}_{r}\bm{\Theta}_{b,r},~\bm{P}_{\rm M}\bm{C}^{\star}_{\pi_{1}(r)}=\bm{P}_{\rm M}\bm{C}_{r}\bm{\Theta}_{c,r},

𝑫¯⋆π1​(r)×1𝚯a,r×2𝚯b,r×3𝚯c,r=𝑫¯r,\displaystyle\underline{\bm{D}}^{\star}_{\pi_{1}(r)}\times_{1}\bm{\Theta}_{a,r}\times_{2}\bm{\Theta}_{b,r}\times_{3}\bm{\Theta}_{c,r}=\underline{\bm{D}}_{r},

(41)

where π1\pi_{1} is permutation of {1,…,R}\{1,\ldots,R\}, and 𝚯a,r\bm{\Theta}_{a,r}, 𝚯b,r\bm{\Theta}_{b,r}, and 𝚯c,r\bm{\Theta}_{c,r} are nonsingular matrices.
The π1\pi_{1}, 𝚯a,r\bm{\Theta}_{a,r}, 𝚯b,r\bm{\Theta}_{b,r}, and 𝚯c,r\bm{\Theta}_{c,r} are associated with the MSI decomposition, i.e., the decomposition model in (40).

Moreover, applying Theorem 2 to the HSI and the equality constraint in (39), the following holds almost surely:

𝑨~π2​(r)⋆=𝑨r​𝚯~a,r,𝑩~π2​(r)⋆=𝑩r​𝚯~b,r,𝑪π2​(r)⋆=𝑪r​𝚯~c,r,\displaystyle\widetilde{\bm{A}}^{\star}_{\pi_{2}(r)}=\bm{A}_{r}\widetilde{\bm{\Theta}}_{a,r},~\widetilde{\bm{B}}^{\star}_{\pi_{2}(r)}=\bm{B}_{r}\widetilde{\bm{\Theta}}_{b,r},~\bm{C}^{\star}_{\pi_{2}(r)}=\bm{C}_{r}\widetilde{\bm{\Theta}}_{c,r},

𝑫¯⋆π2​(r)×1𝚯~a,r×2𝚯~b,r×3𝚯~c,r=𝑫¯r,\displaystyle\underline{\bm{D}}^{\star}_{\pi_{2}(r)}\times_{1}\widetilde{\bm{\Theta}}_{a,r}\times_{2}\widetilde{\bm{\Theta}}_{b,r}\times_{3}\widetilde{\bm{\Theta}}_{c,r}=\underline{\bm{D}}_{r},

(42)

where π2\pi_{2} is permutation of {1,…,R}\{1,\ldots,R\}, and 𝚯~a,r\widetilde{\bm{\Theta}}_{a,r}, 𝚯~b,r\widetilde{\bm{\Theta}}_{b,r}, and 𝚯~c,r\widetilde{\bm{\Theta}}_{c,r} are nonsingular matrices associated with the decomposition model in (39).

Next, we hope to show that π1=π2\pi_{1}=\pi_{2} and 𝚯c,r=𝚯~c,r\bm{\Theta}_{c,r}=\widetilde{\bm{\Theta}}_{c,r}.
From (41) and (42), it is seen that

𝑷M​𝑪r⋆=𝑷M​𝑪π1​(r)​𝚯c,π1​(r),\displaystyle\bm{P}_{\rm M}\bm{C}^{\star}_{r}=\bm{P}_{\rm M}\bm{C}_{\pi_{1}(r)}\bm{\Theta}_{c,\pi_{1}(r)},

(43)

𝑪r⋆=𝑪π2​(r)​𝚯~c,π2​(r).\displaystyle\bm{C}^{\star}_{r}=\bm{C}_{\pi_{2}({r})}\widetilde{\bm{\Theta}}_{c,\pi_{2}({r})}.

(44)

Combining the above, we have

𝑷M​[𝑪1,…,𝑪R]​𝚷1​𝚲c=𝑷M​[𝑪1,…,𝑪R]​𝚷2​𝚲~c,\displaystyle\bm{P}_{\rm M}[\bm{C}_{1},\ldots,\bm{C}_{R}]\bm{\Pi}_{1}{\bm{\Lambda}}_{c}=\bm{P}_{\rm M}[\bm{C}_{1},\ldots,\bm{C}_{R}]\bm{\Pi}_{2}\widetilde{\bm{\Lambda}}_{c},

(45)

where 𝚷1∈ℝN​R×N​R\bm{\Pi}_{1}\in\mathbb{R}^{NR\times NR} and 𝚷2∈ℝN​R×N​R\bm{\Pi}_{2}\in\mathbb{R}^{NR\times NR} are block permutation matrices that permutes the blocks 𝑪1,…,𝑪R\bm{C}_{1},\ldots,\bm{C}_{R}.
These block permutation matrices are entirely determined by π1\pi_{1} and π2\pi_{2}.
For example, if R=2R=2 and π1={2,1}\pi_{1}=\{2,1\}, we have

𝚷1=[𝟎N×N,𝑰N×N,𝑰N×N,𝟎N×N.].\displaystyle\bm{\Pi}_{1}=\begin{bmatrix}\bm{0}_{N\times N},~\bm{I}_{N\times N},\\
\bm{I}_{N\times N},~\bm{0}_{N\times N}.\end{bmatrix}.

(46)

Hence, our task now boils down to proving 𝚷1=𝚷2\bm{\Pi}_{1}=\bm{\Pi}_{2}.
In addition, we have

𝚲c\displaystyle{\bm{\Lambda}}_{c}
=blockdiag⁡(𝚯𝒄,π1​(1),…,𝚯c,π1​(R)),\displaystyle={\rm blockdiag}(\bm{\Theta}_{\bm{c},\pi_{1}(1)},\ldots,\bm{\Theta}_{c,\pi_{1}(R)}),

(47)

𝚲~c\displaystyle\widetilde{\bm{\Lambda}}_{c}
=blockdiag⁡(𝚯~𝒄,π2​(1),…,𝚯~c,π2​(R)).\displaystyle={\rm blockdiag}(\widetilde{\bm{\Theta}}_{\bm{c},\pi_{2}(1)},\ldots,\widetilde{\bm{\Theta}}_{c,\pi_{2}(R)}).

(48)

Next, we look at the qq-th columns on both sides of (45):

𝑷M𝑪𝒁1(:,q)=𝑷M𝑪𝒁2(:,q),q=1,…,NR,\displaystyle\bm{P}_{\rm M}\bm{C}\bm{Z}_{1}(:,q)=\bm{P}_{\rm M}\bm{C}\bm{Z}_{2}(:,q),~q=1,\ldots,NR,

(49)

where 𝒁1=𝚷1​𝚲c\bm{Z}_{1}=\bm{\Pi}_{1}\bm{\Lambda}_{c} and 𝒁2=𝚷2​𝚲~c\bm{Z}_{2}=\bm{\Pi}_{2}\widetilde{\bm{\Lambda}}_{c}.

By Lemma 10, 𝑷M​𝑪\bm{P}_{M}\bm{C} is drawn from an absolutely continuous joint distribution.
By assuming that KM≥2​NK_{M}\geq 2N, any 2​N2N columns of 𝑷M​𝑪\bm{P}_{M}\bm{C} are linearly independent.
Re-writing (49) as follows:

𝑷M𝑪(𝒁1(:,q)−𝒁2(:,q))=𝟎.\displaystyle\bm{P}_{M}\bm{C}(\bm{Z}_{1}(:,q)-\bm{Z}_{2}(:,q))=\bm{0}.

(50)

Note that the matrix 𝒁1(:,q)−𝒁2(:,q)\bm{Z}_{1}(:,q)-\bm{Z}_{2}(:,q) has at most 2​N2N nonzero elements.
However, any 2​N2N columns of 𝑷M​𝑪\bm{P}_{M}\bm{C} are linearly independent.
This means that (50) holds if and only if 𝒁1(:,q)=𝒁2(:,q)\bm{Z}_{1}(:,q)=\bm{Z}_{2}(:,q) for q=1,…,N​Rq=1,\ldots,NR, which leads to 𝚷1=𝚷2\bm{\Pi}_{1}=\bm{\Pi}_{2} and 𝚲c=𝚲~c\bm{\Lambda}_{c}=\widetilde{\bm{\Lambda}}_{c}.
Then 𝑪π1​(r)⋆=𝑪r​𝚯c,r\bm{C}^{\star}_{\pi_{1}(r)}=\bm{C}_{r}\bm{\Theta}_{c,r}.
Combining (41), we have

𝑫¯π1​(r)⋆×1𝑨π1​(r)⋆×2𝑩π1​(r)⋆×3𝑪π1​(r)⋆=𝑫¯r×1𝑨r×2𝑩r×3𝑪r.\displaystyle\underline{\bm{D}}_{\pi_{1}(r)}^{\star}\times_{1}\bm{A}_{\pi_{1}(r)}^{\star}\times_{2}\bm{B}_{\pi_{1}(r)}^{\star}\times_{3}\bm{C}_{\pi_{1}(r)}^{\star}=\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}\bm{C}_{r}.

Hence, we can recover 𝒀¯S=∑r=1R𝑫¯π1​(r)⋆×1𝑨π1​(r)⋆×2𝑩π1​(r)⋆×3𝑪π1​(r)⋆.\underline{\bm{Y}}_{\rm S}=\sum_{r=1}^{R}\underline{\bm{D}}_{\pi_{1}(r)}^{\star}\times_{1}\bm{A}_{\pi_{1}(r)}^{\star}\times_{2}\bm{B}_{\pi_{1}(r)}^{\star}\times_{3}\bm{C}_{\pi_{1}(r)}^{\star}.
This completes the proof.

## Appendix E Algorithm Details for Model (24)

In this section, we present the details for solving Problem (24), including the algorithm design, parameter setting, and computation analysis.
The algorithm follows the paradigm of alternating accelerated proximal gradient framework in [64] and the design of HSR/HU algorithms in [12, 13].
The proposed CLIMB for handling (24) is summarized in Algorithm 1.

*Algorithm 1 CLIMB for solving (24).*

Input:
HSI 𝒀¯H\underline{\bm{Y}}_{{\rm H}}, MSI 𝒀¯M\underline{\bm{Y}}_{{\rm M}};
pre-defined sequences of μr,i(t)\mu_{r,i}^{(t)} and αr,i(t)\alpha_{r,i}^{(t)}, r=1,…,R,i=1,…,4r=1,\ldots,R,~i=1,\ldots,4;
starting points {𝑫¯r(0),𝑨r(0),𝑩r(0),𝑪r(0)}r=1R\left\{\underline{\bm{D}}_{r}^{(0)},\bm{A}_{r}^{(0)},\bm{B}_{r}^{(0)},\bm{C}_{r}^{(0)}\right\}_{r=1}^{R}.

Parameters: λ\lambda, η\eta, LL, and NN.

𝑨ˇr(0)=𝑨r(0)\check{\bm{A}}_{r}^{(0)}=\bm{A}_{r}^{(0)}, 𝑩ˇr(0)=𝑩r(0)\check{\bm{B}}_{r}^{(0)}=\bm{B}_{r}^{(0)}, 𝑪ˇr(0)=𝑪r(0)\check{\bm{C}}_{r}^{(0)}=\bm{C}_{r}^{(0)}, 𝑫¯ˇr(0)=𝑫¯r(0)\check{\underline{\bm{D}}}_{r}^{(0)}=\underline{\bm{D}}_{r}^{(0)}, t=0t=0.

repeat

for r=1:Rr=1:R

%%\%\% update 𝑨r\bm{A}_{r} %%\%\%

𝑨r(t+1)←𝑨ˇr(t)−αr,1(t)​𝑮𝑨ˇr(t)\bm{A}_{r}^{(t+1)}\leftarrow\check{\bm{A}}_{r}^{(t)}-\alpha_{r,1}^{(t)}\bm{G}_{\check{\bm{A}}_{r}}^{(t)};
  𝑨ˇr(t+1)←𝑨r(t+1)+μr,1(t)​(𝑨r(t+1)−𝑨r(t))\check{\bm{A}}_{r}^{(t+1)}\leftarrow\bm{A}_{r}^{(t+1)}+\mu_{r,1}^{(t)}(\bm{A}_{r}^{(t+1)}-\bm{A}_{r}^{(t)});

%%\%\% update 𝑩r\bm{B}_{r} %%\%\%

𝑩r(t+1)←𝑩ˇr(t)−αr,2(t)​𝑮𝑩ˇr(t)\bm{B}_{r}^{(t+1)}\leftarrow\check{\bm{B}}_{r}^{(t)}-\alpha_{r,2}^{(t)}\bm{G}_{\check{\bm{B}}_{r}}^{(t)};
  𝑩ˇr(t+1)←𝑩r(t+1)+μr,2(t)​(𝑩r(t+1)−𝑩r(t))\check{\bm{B}}_{r}^{(t+1)}\leftarrow\bm{B}_{r}^{(t+1)}+\mu_{r,2}^{(t)}(\bm{B}_{r}^{(t+1)}-\bm{B}_{r}^{(t)});

%%\%\% update 𝑪r\bm{C}_{r} %%\%\%

𝑪r(t+1)←𝑪ˇr(t)−αr,3(t)​𝑮𝑪ˇr(t)\bm{C}_{r}^{(t+1)}\leftarrow\check{\bm{C}}_{r}^{(t)}-\alpha_{r,3}^{(t)}\bm{G}_{\check{\bm{C}}_{r}}^{(t)};
  𝑪ˇr(t+1)←𝑪r(t+1)+μr,3(t)​(𝑪r(t+1)−𝑪r(t))\check{\bm{C}}_{r}^{(t+1)}\leftarrow\bm{C}_{r}^{(t+1)}+\mu_{r,3}^{(t)}(\bm{C}_{r}^{(t+1)}-\bm{C}_{r}^{(t)});

%%\%\% update 𝑫¯r\underline{\bm{D}}_{r} %%\%\%

𝑫¯r(t+1)←𝑫¯ˇr(t)−αr,4(t)​𝑮𝑫¯ˇr(t)\underline{\bm{D}}_{r}^{(t+1)}\leftarrow\check{\underline{\bm{D}}}_{r}^{(t)}-\alpha_{r,4}^{(t)}\bm{G}_{\check{\underline{\bm{D}}}_{r}}^{(t)};
  𝑫¯ˇr(t+1)←𝑫¯r(t+1)+μr,4(t)​(𝑫¯r(t+1)−𝑫¯r(t))\check{\underline{\bm{D}}}_{r}^{(t+1)}\leftarrow\underline{\bm{D}}_{r}^{(t+1)}+\mu_{r,4}^{(t)}(\underline{\bm{D}}_{r}^{(t+1)}-\underline{\bm{D}}_{r}^{(t)});

end

t=t+1t=t+1;

until satisfying the stopping criterion.

Output: 𝑨^r=𝑨r(t)\widehat{\bm{A}}_{r}=\bm{A}_{r}^{(t)}, 𝑩^r=𝑩r(t)\widehat{\bm{B}}_{r}=\bm{B}_{r}^{(t)}, 𝑪^r=𝑪r(t)\widehat{\bm{C}}_{r}=\bm{C}_{r}^{(t)}, and 𝑫¯^r=𝑫¯r(t)\widehat{\underline{\bm{D}}}_{r}=\underline{\bm{D}}_{r}^{(t)}. Recover 𝒀¯^S=∑r=1R𝑫¯^r×1𝑨^r×2𝑩^r×3𝑪^r\underline{\widehat{\bm{Y}}}_{\rm S}=\sum_{r=1}^{R}\underline{\widehat{\bm{D}}}_{r}\times_{1}\widehat{\bm{A}}_{r}\times_{2}\widehat{\bm{B}}_{r}\times_{3}\widehat{\bm{C}}_{r}.

To ensure convergence, it is crucial to select {μr,i(t)}\{\mu_{r,i}^{(t)}\}.
As in previous works [12, 13], we find that selecting the sequences {μr,i(t)}\{\mu_{r,i}^{(t)}\} following the Nesterov’s extrapolation strategy [40] as a heuristic works fairly well.
Therefore, we simply set

μr,i(t)=γr,i(t)−1γr,i(t+1),γr,i(t+1)=1+1+4​(γr,i(t))22,\displaystyle\mu_{r,i}^{(t)}=\frac{\gamma_{r,i}^{(t)}-1}{\gamma_{r,i}^{(t+1)}},~~\gamma_{r,i}^{(t+1)}=\frac{1+\sqrt{1+4\left(\gamma_{r,i}^{(t)}\right)^{2}}}{2},

with γr,i(0)=1\gamma_{r,i}^{(0)}=1 for r=1,…,R,i=1,…,4r=1,\ldots,R,~i=1,\ldots,4. In addition, we choose the step sizes αr,i(t)\alpha_{r,i}^{(t)} as αr,1(t)=1/L𝑨r(t),αr,2(t)=1/L𝑩r(t),αr,3(t)=1/L𝑪r(t),αr,4(t)=1/L𝑫¯r(t)\alpha_{r,1}^{(t)}=1/L_{\bm{A}_{r}}^{(t)},~\alpha_{r,2}^{(t)}=1/L_{\bm{B}_{r}}^{(t)},~\alpha_{r,3}^{(t)}=1/L_{\bm{C}_{r}}^{(t)},~\alpha_{r,4}^{(t)}=1/L_{\underline{\bm{D}}_{r}}^{(t)}, in which L𝑨r(t)L_{\bm{A}_{r}}^{(t)}, L𝑩r(t)L_{\bm{B}_{r}}^{(t)}, L𝑪r(t)L_{\bm{C}_{r}}^{(t)}, and L𝑫¯r(t)L_{\underline{\bm{D}}_{r}}^{(t)} are given in follows.

Next, we give the expressions of gradients 𝑮𝑨r\bm{G}_{\bm{A}_{r}}, 𝑮𝑩r\bm{G}_{\bm{B}_{r}}, 𝑮𝑪r\bm{G}_{\bm{C}_{r}}, and 𝑮𝑫¯r\bm{G}_{\underline{\bm{D}}_{r}}, and the parameters L𝑨r(t)L_{\bm{A}_{r}}^{(t)}, L𝑩r(t)L_{\bm{B}_{r}}^{(t)}, L𝑪r(t)L_{\bm{C}_{r}}^{(t)}, and L𝑫¯r(t)L_{\underline{\bm{D}}_{r}}^{(t)}.
When we update the rr-th block 𝑨r\bm{A}_{r}, 𝑩r\bm{B}_{r}, 𝑪r\bm{C}_{r}, and 𝑫¯r\underline{\bm{D}}_{r} by fixing other variables, the optimization problem is

min𝑫¯r,𝑨r,𝑩r,𝑪r\displaystyle\min_{\underline{\bm{D}}_{r},\bm{A}_{r},\bm{B}_{r},\bm{C}_{r}}
12‖𝒀¯~H−𝑫¯r×1(𝑷1𝑨r)×2(𝑷2𝑩r)×3𝑪r‖F2\displaystyle\frac{1}{2}\left\|\underline{\widetilde{\bm{Y}}}_{H}-\underline{\bm{D}}_{r}\times_{1}(\bm{P}_{1}\bm{A}_{r})\times_{2}(\bm{P}_{2}\bm{B}_{r})\times_{3}\bm{C}_{r}\right\|_{F}^{2}

+\displaystyle+
12‖𝒀¯~M−𝑫¯r×1𝑨r×2𝑩r×3(𝑷M𝑪r)‖F2+λϕr(𝑨r,𝑩r,𝑪r)+η2∥𝑫¯r∥F2,\displaystyle\frac{1}{2}\left\|\underline{\widetilde{\bm{Y}}}_{M}-\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r})\right\|_{F}^{2}+\lambda\phi_{r}(\bm{A}_{r},\bm{B}_{r},\bm{C}_{r})+\frac{\eta}{2}\|\underline{\bm{D}}_{r}\|_{F}^{2},

where 𝒀¯~H=𝒀¯H−∑r~≠r𝑫¯r~×1(𝑷1𝑨r~)×2(𝑷2𝑩r~)×3𝑪r~\underline{\widetilde{\bm{Y}}}_{H}=\underline{\bm{Y}}_{H}-\sum_{\tilde{r}\neq r}\underline{\bm{D}}_{\tilde{r}}\times_{1}(\bm{P}_{1}\bm{A}_{\tilde{r}})\times_{2}(\bm{P}_{2}\bm{B}_{\tilde{r}})\times_{3}\bm{C}_{\tilde{r}} and 𝒀¯~M=𝒀¯M−∑r~≠r𝑫¯r~×1𝑨r~×2𝑩r~×3(𝑷M𝑪r~)\underline{\widetilde{\bm{Y}}}_{M}=\underline{\bm{Y}}_{M}-\sum_{\tilde{r}\neq r}\underline{\bm{D}}_{\tilde{r}}\times_{1}\bm{A}_{\tilde{r}}\times_{2}\bm{B}_{\tilde{r}}\times_{3}(\bm{P}_{M}\bm{C}_{\tilde{r}}).

For simplicity, we rewrite the 𝑨r\bm{A}_{r}-subproblem as

𝒥⁡(𝑨r)=\displaystyle\mathcal{J}(\bm{A}_{r})=
12‖𝒀¯~H−𝑫¯r(t)×1(𝑷1𝑨r)×2(𝑷2𝑩r(t))×3𝑪r(t)‖F2\displaystyle{\frac{1}{2}\left\|\underline{\widetilde{\bm{Y}}}_{H}-\underline{\bm{D}}_{r}^{(t)}\times_{1}(\bm{P}_{1}\bm{A}_{r})\times_{2}(\bm{P}_{2}\bm{B}_{r}^{(t)})\times_{3}\bm{C}_{r}^{(t)}\right\|_{F}^{2}}

+\displaystyle+
12‖𝒀¯~M−𝑫¯r(t)×1𝑨r×2𝑩r(t)×3(𝑷M𝑪r(t))‖F2+λϕp,ε(𝑯1𝑨r).\displaystyle{\frac{1}{2}\left\|\underline{\widetilde{\bm{Y}}}_{M}-\underline{\bm{D}}_{r}^{(t)}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}^{(t)}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r}^{(t)})\right\|_{F}^{2}}+\lambda\phi_{p,\varepsilon}(\bm{H}_{1}\bm{A}_{r}).

Since the design of ϕp,ε​(𝑯1​𝑨r)\phi_{p,\varepsilon}(\bm{H}_{1}\bm{A}_{r}), the gradient 𝑮𝑨r\bm{G}_{\bm{A}_{r}} of 𝒥⁡(𝑨r)\mathcal{J}(\bm{A}_{r}) exists but is hard to compute. Therefore, we first construct a tight quadratic upper bounded function ℒ⁡(𝑨r,𝑩r(t),𝑪r(t),𝑫¯r(t),𝑨r(t)){\cal L}(\bm{A}_{r},\bm{B}_{r}^{(t)},\bm{C}_{r}^{(t)},\underline{\bm{D}}_{r}^{(t)};\bm{A}_{r}^{(t)}) such that
ℒ⁡(𝑨r(t),𝑩r(t),𝑪r(t),𝑫¯r(t),𝑨r(t))≥𝒥⁡(𝑨r(t)){\cal L}(\bm{A}_{r}^{(t)},\bm{B}_{r}^{(t)},\bm{C}_{r}^{(t)},\underline{\bm{D}}_{r}^{(t)};\bm{A}_{r}^{(t)})\geq{\cal J}(\bm{A}_{r}^{(t)})
and
∇𝑨rℒ​(𝑨r(t),𝑩r(t),𝑪r(t),𝑫¯r(t),𝑨r(t))=∇𝑨r𝒥​(𝑨r(t))\nabla_{\bm{A}_{r}}{\cal L}(\bm{A}_{r}^{(t)},\bm{B}_{r}^{(t)},\bm{C}_{r}^{(t)},\underline{\bm{D}}_{r}^{(t)};\bm{A}_{r}^{(t)})=\nabla_{\bm{A}_{r}}{\cal J}(\bm{A}_{r}^{(t)}).

Then, we compute 𝑮𝑨r(t)=∇𝑨rℒ​(𝑨r(t),𝑩r(t),𝑪r(t),𝑫¯r(t),𝑨r(t))\bm{G}_{\bm{A}_{r}}^{(t)}=\nabla_{\bm{A}_{r}}{\cal L}(\bm{A}_{r}^{(t)},\bm{B}_{r}^{(t)},\bm{C}_{r}^{(t)},\underline{\bm{D}}_{r}^{(t)};\bm{A}_{r}^{(t)}).
Note that according to the work in [15], ϕp,ε​(𝒙)\phi_{p,\varepsilon}(\bm{x}) (0<p≤10<p\leq 1) admits a majorizer ϕ~​(𝒙,𝒙(t))\widetilde{\phi}(\bm{x},\bm{x}^{(t)}) as

ϕ~​(𝒙,𝒙(t))\displaystyle\widetilde{\phi}(\bm{x},\bm{x}^{(t)})
=∑i[𝒘(t)]i​[𝒙]i2+2−p2​(2p​[𝒘(t)]i)pp−2+ε​[𝒘(t)]i\displaystyle=\sum_{i}[\bm{w}^{(t)}]_{i}[\bm{x}]_{i}^{2}+\frac{2-p}{2}\Big(\frac{2}{p}[\bm{w}^{(t)}]_{i}\Big)^{\frac{p}{p-2}}+\varepsilon[\bm{w}^{(t)}]_{i}

=p2​𝒙⊤​𝑾(t)​𝒙+const,\displaystyle=\frac{p}{2}\bm{x}^{\top}\bm{W}^{(t)}\bm{x}+{\rm const},

(51)

where [𝒘(t)]i=p2​(([𝒙(t)]i)2+ε)p−22[\bm{w}^{(t)}]_{i}=\frac{p}{2}\big(([\bm{x}^{(t)}]_{i})^{2}+\varepsilon\big)^{\frac{p-2}{2}}, 𝑾(t)\bm{W}^{(t)} is a diagonal matrix with [𝑾(t)]i,i=[𝒘(t)]i[\bm{W}^{(t)}]_{i,i}=[\bm{w}^{(t)}]_{i} and const{\rm const} is a constant. Hence, we get the quadratic majorizer as

ℒ⁡(𝑨r,𝑩r(t),𝑪r(t),𝑫¯r(t),𝑨r(t))\displaystyle{\cal L}(\bm{A}_{r},\bm{B}_{r}^{(t)},\bm{C}_{r}^{(t)},\underline{\bm{D}}_{r}^{(t)};\bm{A}_{r}^{(t)})

(52)

=\displaystyle=
12‖𝒀¯~H−𝑫¯r(t)×1(𝑷1𝑨r)×2(𝑷2𝑩r(t))×3𝑪r(t)‖F2\displaystyle{\frac{1}{2}\left\|\underline{\widetilde{\bm{Y}}}_{H}-\underline{\bm{D}}_{r}^{(t)}\times_{1}(\bm{P}_{1}\bm{A}_{r})\times_{2}(\bm{P}_{2}\bm{B}_{r}^{(t)})\times_{3}\bm{C}_{r}^{(t)}\right\|_{F}^{2}}

+\displaystyle+
12‖𝒀¯~M−𝑫¯r(t)×1𝑨r×2𝑩r(t)×3(𝑷M𝑪r(t))‖F2+λϕ~(𝑯~1𝒂r,𝑯~1𝒂r(t)),\displaystyle{\frac{1}{2}\left\|\underline{\widetilde{\bm{Y}}}_{M}-\underline{\bm{D}}_{r}^{(t)}\times_{1}\bm{A}_{r}\times_{2}\bm{B}_{r}^{(t)}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r}^{(t)})\right\|_{F}^{2}}+\lambda\widetilde{\phi}(\tilde{\bm{H}}_{1}{\bm{a}}_{r},\tilde{\bm{H}}_{1}{\bm{a}}_{r}^{(t)}),

where 𝑯~1=𝑯1⊗𝑰L\tilde{\bm{H}}_{1}=\bm{H}_{1}\otimes\bm{I}_{L} (𝑰L\bm{I}_{L} is the identify matrix with size L×LL\times L), and 𝒂r(t)\bm{a}_{r}^{(t)} is the vectorization of 𝑨r(t)\bm{A}_{r}^{(t)}.

The gradient 𝑮𝑨r(t)\bm{G}_{\bm{A}_{r}}^{(t)} can be expressed as follows:

𝑮𝑨r(t)=\displaystyle\bm{G}_{\bm{A}_{r}}^{(t)}=
𝑷1⊤​(𝑷1​𝑨r(t)​𝑽r​1(t)−𝒀~H(1))​(𝑽r​1(t))⊤+(𝑨r​𝑼r​1(t)−𝒀~M(1))​(𝑼r​1(t))⊤+λ​p​𝑯~1⊤​𝑾r​1(t)​𝑯~1​𝒂r(t),\displaystyle\bm{P}_{1}^{\top}(\bm{P}_{1}\bm{A}_{r}^{(t)}\bm{V}_{r1}^{(t)}-\widetilde{\bm{Y}}_{H}^{(1)})(\bm{V}_{r1}^{(t)})^{\top}+(\bm{A}_{r}\bm{U}_{r1}^{(t)}-\widetilde{\bm{Y}}_{M}^{(1)})(\bm{U}_{r1}^{(t)})^{\top}+\lambda p\tilde{\bm{H}}_{1}^{\top}\bm{W}_{r1}^{(t)}\tilde{\bm{H}}_{1}\bm{a}_{r}^{(t)},

where 𝒀~H(1)=(𝒀¯~H)(1)\widetilde{\bm{Y}}_{H}^{(1)}=(\underline{\widetilde{\bm{Y}}}_{H})_{(1)}, 𝒀~M(1)=(𝒀¯~M)(1)\widetilde{\bm{Y}}_{M}^{(1)}=(\underline{\widetilde{\bm{Y}}}_{M})_{(1)}, 𝑽r​1(t)=(𝑫¯r(t)×2(𝑷2𝑩r(t))×3𝑪r(t))(1)\bm{V}_{r1}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{2}(\bm{P}_{2}\bm{B}_{r}^{(t)})\times_{3}\bm{C}_{r}^{(t)})_{(1)}, 𝑼r​1(t)=(𝑫¯r(t)×2𝑩r(t)×3(𝑷M𝑪r(t)))(1)\bm{U}_{r1}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{2}\bm{B}_{r}^{(t)}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r}^{(t)}))_{(1)}, and
𝑾r​1(t)\bm{W}_{r1}^{(t)} is diagonal with
[𝑾r​1(t)]i,i=([𝑯~1​𝒂r(t)]i2+ε)p−22[\bm{W}_{r1}^{(t)}]_{i,i}=([\tilde{\bm{H}}_{1}{\bm{a}}_{r}^{(t)}]_{i}^{2}+\varepsilon)^{\frac{p-2}{2}}.

The parameter L𝑨r(t)L_{\bm{A}_{r}}^{(t)} is determined by the the Lipschitz constant of the 𝑨r\bm{A}_{r}-subproblem (52). One can see that (52) is Lipschitz continuous with the following Lipschitz constant:

L𝑨r(t)\displaystyle L_{\bm{A}_{r}}^{(t)}
=σmax​(𝑷1⊤​𝑷1)​σmax​(𝑽r​1(t)​(𝑽r​1(t))⊤)+σmax​(𝑼r​1(t)​(𝑼r​1(t))⊤)+λ​p​σmax​(𝑯~1⊤​𝑾r​1(t)​𝑯~1).\displaystyle=\sigma_{\rm max}\left(\bm{P}_{1}^{\top}\bm{P}_{1}\right)\sigma_{\rm max}\left(\bm{V}_{r1}^{(t)}(\bm{V}_{r1}^{(t)})^{\top}\right)+\sigma_{\rm max}\left(\bm{U}_{r1}^{(t)}(\bm{U}_{r1}^{(t)})^{\top}\right)+\lambda p\sigma_{\textrm{max}}\left(\tilde{\bm{H}}_{1}^{\top}\bm{W}_{r1}^{(t)}\tilde{\bm{H}}_{1}\right).

Similarly, we can obtain the gradients 𝑮𝑩r(t)\bm{G}_{\bm{B}_{r}}^{(t)}, 𝑮𝑪r(t)\bm{G}_{\bm{C}_{r}}^{(t)}, and parameters L𝑩r(t)L_{\bm{B}_{r}}^{(t)}, L𝑪r(t)L_{\bm{C}_{r}}^{(t)} as follows.

𝑮𝑩r(t)\displaystyle\bm{G}_{\bm{B}_{r}}^{(t)}
=𝑷2⊤​(𝑷2​𝑩r(t)​𝑽r​2(t)−𝒀~H(2))​(𝑽r​2(t))⊤+(𝑩r​𝑼r​2(t)−𝒀~M(2))​(𝑼r​2(t))⊤+λ​p​𝑯~2⊤​𝑾r​2(t)​𝑯~2​𝒃r(t),\displaystyle=\bm{P}_{2}^{\top}(\bm{P}_{2}\bm{B}_{r}^{(t)}\bm{V}_{r2}^{(t)}-\widetilde{\bm{Y}}_{H}^{(2)})(\bm{V}_{r2}^{(t)})^{\top}+(\bm{B}_{r}\bm{U}_{r2}^{(t)}-\widetilde{\bm{Y}}_{M}^{(2)})(\bm{U}_{r2}^{(t)})^{\top}+\lambda p\tilde{\bm{H}}_{2}^{\top}\bm{W}_{r2}^{(t)}\tilde{\bm{H}}_{2}\bm{b}_{r}^{(t)},

L𝑩r(t)\displaystyle L_{\bm{B}_{r}}^{(t)}
=σmax​(𝑷2⊤​𝑷2)​σmax​(𝑽r​2(t)​(𝑽r​2(t))⊤)+σmax​(𝑼r​2(t)​(𝑼r​2(t))⊤)+λ​p​σmax​(𝑯~2⊤​𝑾r​2(t)​𝑯~2),\displaystyle=\sigma_{\rm max}\left(\bm{P}_{2}^{\top}\bm{P}_{2}\right)\sigma_{\rm max}\left(\bm{V}_{r2}^{(t)}(\bm{V}_{r2}^{(t)})^{\top}\right)+\sigma_{\rm max}\left(\bm{U}_{r2}^{(t)}(\bm{U}_{r2}^{(t)})^{\top}\right)+\lambda p\sigma_{\textrm{max}}\left(\tilde{\bm{H}}_{2}^{\top}\bm{W}_{r2}^{(t)}\tilde{\bm{H}}_{2}\right),

where 𝑽r​2(t)=(𝑫¯r(t)×1(𝑷1𝑨r(t+1))×3𝑪r(t))(2)\bm{V}_{r2}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{1}(\bm{P}_{1}\bm{A}_{r}^{(t+1)})\times_{3}\bm{C}_{r}^{(t)})_{(2)}, 𝑼r​2(t)=(𝑫¯r(t)×1𝑨r(t+1)×3(𝑷M𝑪r(t)))(2)\bm{U}_{r2}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{1}\bm{A}_{r}^{(t+1)}\times_{3}(\bm{P}_{M}\bm{C}_{r}^{(t)}))_{(2)}, 𝑯~2=𝑯2⊗𝑰M\tilde{\bm{H}}_{2}=\bm{H}_{2}\otimes\bm{I}_{M} (𝑰M\bm{I}_{\rm M} is the identify matrix with size M×MM\times M), and
𝑾r​2(t)\bm{W}_{r2}^{(t)} is a diagonal matrix with
[𝑾r​2(t)]i,i=([𝑯~2​𝒃r(t)]i2+ε)p−22[\bm{W}_{r2}^{(t)}]_{i,i}=([\tilde{\bm{H}}_{2}{\bm{b}}_{r}^{(t)}]_{i}^{2}+\varepsilon)^{\frac{p-2}{2}}, where 𝒃r(t)\bm{b}_{r}^{(t)} is the vectorization of 𝑩r(t)\bm{B}_{r}^{(t)}.

*Table 10: Complexity of each term of CLIMB.*

Terms
Complexity

𝑷1⊤​(𝑷1​𝑨r(t)​𝑽r​1(t)−𝒀~H(1))​(𝑽r​1(t))⊤\bm{P}_{1}^{\top}(\bm{P}_{1}\bm{A}_{r}^{(t)}\bm{V}_{r1}^{(t)}-\widetilde{\bm{Y}}_{H}^{(1)})(\bm{V}_{r1}^{(t)})^{\top}
𝒪⁡(JH​JM​Mr+IH​Nr​(Lr​Mr+Mr​JH+JH​KH)+JH​Lr​(Nr​Mr+Nr​KH+IH​KH)+IM​JH​KH​(IH+Lr))\mathcal{O}(J_{\rm H}J_{\rm M}M_{r}+I_{\rm H}N_{r}(L_{r}M_{r}+M_{r}J_{\rm H}+J_{\rm H}K_{\rm H})+J_{\rm H}L_{r}(N_{r}M_{r}+N_{r}K_{\rm H}+I_{\rm H}K_{\rm H})+I_{\rm M}J_{\rm H}K_{\rm H}(I_{\rm H}+L_{r}))

(𝑨r​𝑼r​1(t)−𝒀~M(1))​(𝑼r​1(t))⊤(\bm{A}_{r}\bm{U}_{r1}^{(t)}-\widetilde{\bm{Y}}_{M}^{(1)})(\bm{U}_{r1}^{(t)})^{\top}
𝒪⁡(KH​KM​Nr+Lr​Nr​(JM​Mr+JM​KM+IM​Lr)+IM​JM​(Nr​Mr+Nr​KM+KM​Lr))\mathcal{O}(K_{\rm H}K_{\rm M}N_{r}+L_{r}N_{r}(J_{\rm M}M_{r}+J_{\rm M}K_{\rm M}+I_{\rm M}L_{r})+I_{\rm M}J_{\rm M}(N_{r}M_{r}+N_{r}K_{\rm M}+K_{\rm M}L_{r}))

𝑷2⊤​(𝑷2​𝑩r(t)​𝑽r​2(t)−𝒀~H(2))​(𝑽r​2(t))⊤\bm{P}_{2}^{\top}(\bm{P}_{2}\bm{B}_{r}^{(t)}\bm{V}_{r2}^{(t)}-\widetilde{\bm{Y}}_{H}^{(2)})(\bm{V}_{r2}^{(t)})^{\top}
𝒪⁡(IH​IM​Lr+IH​Nr​(Lr​Mr+Mr​JH+JH​KH)+IH​KH​(IM​Mr+Mr​JH+IM​KH)+IH​Mr​Nr​(KH+Lr))\mathcal{O}(I_{\rm H}I_{\rm M}L_{r}+I_{\rm H}N_{r}(L_{r}M_{r}+M_{r}J_{\rm H}+J_{\rm H}K_{\rm H})+I_{\rm H}K_{\rm H}(I_{\rm M}M_{r}+M_{r}J_{\rm H}+I_{\rm M}K_{\rm H})+I_{\rm H}M_{r}N_{r}(K_{\rm H}+L_{r}))

(𝑩r​𝑼r​2(t)−𝒀~M(2))​(𝑼r​2(t))⊤(\bm{B}_{r}\bm{U}_{r2}^{(t)}-\widetilde{\bm{Y}}_{M}^{(2)})(\bm{U}_{r2}^{(t)})^{\top}
𝒪⁡(KH​KM​Nr+Lr​Nr​(IM​Lr+IM​KM+IM​Mr)+IM​JM​(Nr​Mr+Nr​KM+KM​Mr))\mathcal{O}(K_{\rm H}K_{\rm M}N_{r}+L_{r}N_{r}(I_{\rm M}L_{r}+I_{\rm M}K_{\rm M}+I_{\rm M}M_{r})+I_{\rm M}J_{\rm M}(N_{r}M_{r}+N_{r}K_{\rm M}+K_{\rm M}M_{r}))

𝑷M⊤​(𝑷M​𝑪r(t)​𝑽r​3(t)−𝒀~M(3))​(𝑽r​3(t))⊤\bm{P}_{M}^{\top}(\bm{P}_{M}\bm{C}_{r}^{(t)}\bm{V}_{r3}^{(t)}-\widetilde{\bm{Y}}_{M}^{(3)})(\bm{V}_{r3}^{(t)})^{\top}
𝒪⁡(IM​Nr​(Mr​Lr+JM​KM+JM​Mr)+IM​JM​(KH​Nr+KM​KH+KM​Mr)+IM​Mr​Nr​(JM+Lr))\mathcal{O}(I_{\rm M}N_{r}(M_{r}L_{r}+J_{\rm M}K_{\rm M}+J_{\rm M}M_{r})+I_{\rm M}J_{\rm M}(K_{\rm H}N_{r}+K_{\rm M}K_{\rm H}+K_{\rm M}M_{r})+I_{\rm M}M_{r}N_{r}(J_{\rm M}+L_{r}))

(𝑪r​𝑼r​3(t)−𝒀~H(3))​(𝑼r​3(t))⊤(\bm{C}_{r}\bm{U}_{r3}^{(t)}-\widetilde{\bm{Y}}_{H}^{(3)})(\bm{U}_{r3}^{(t)})^{\top}
𝒪⁡(IH​IM​Lr+JH​JM​Mr+IH​Nr​(Mr​JH+Mr​Lr+JH​KH))\mathcal{O}(I_{\rm H}I_{\rm M}L_{r}+J_{\rm H}J_{\rm M}M_{r}+I_{\rm H}N_{r}(M_{r}J_{\rm H}+M_{r}L_{r}+J_{\rm H}K_{\rm H}))

(𝑼𝑫r(t))⊤​(𝑼𝑫r(t)​𝒅r(t)−𝒚m)(\bm{U}^{(t)}_{\bm{D}_{r}})^{\top}(\bm{U}^{(t)}_{\bm{D}_{r}}\bm{d}_{r}^{(t)}-\bm{y}_{m})
𝒪⁡(KM​Lr​(IM​JM+JM​Mr+Mr​Nr))\mathcal{O}(K_{\rm M}L_{r}(I_{\rm M}J_{\rm M}+J_{\rm M}M_{r}+M_{r}N_{r}))

(𝑽𝑫r(t))⊤​(𝑽𝑫r(t)​𝒅r(t)−𝒚h)(\bm{V}^{(t)}_{\bm{D}_{r}})^{\top}(\bm{V}^{(t)}_{\bm{D}_{r}}\bm{d}_{r}^{(t)}-\bm{y}_{h})
𝒪⁡(KH​Lr​(IH​JH+JH​Mr+Mr​Nr))\mathcal{O}(K_{\rm H}L_{r}(I_{\rm H}J_{\rm H}+J_{\rm H}M_{r}+M_{r}N_{r}))

𝑯~1⊤​𝑾r​1(t)​𝑯~1​𝒂r(t)\tilde{\bm{H}}_{1}^{\top}\bm{W}_{r1}^{(t)}\tilde{\bm{H}}_{1}\bm{a}_{r}^{(t)}
𝒪⁡(IM​Lr)\mathcal{O}(I_{\rm M}L_{r})

𝑯~2⊤​𝑾r​2(t)​𝑯~2​𝒃r(t)\tilde{\bm{H}}_{2}^{\top}\bm{W}_{r2}^{(t)}\tilde{\bm{H}}_{2}\bm{b}_{r}^{(t)}
𝒪⁡(JM​Mr)\mathcal{O}(J_{\rm M}M_{r})

𝑯3⊤​𝑯3​𝑪r(t)\bm{H}_{3}^{\top}\bm{H}_{3}\bm{C}_{r}^{(t)}
𝒪⁡(KH2​Nr)\mathcal{O}(K_{\rm H}^{2}N_{r})

αr,i\alpha_{r,i}
𝒪⁡(Lr3+Mr3+Nr3)\mathcal{O}(L_{r}^{3}+M_{r}^{3}+N_{r}^{3})

𝑮𝑪r(t)\displaystyle\bm{G}_{\bm{C}_{r}}^{(t)}
=𝑷M⊤​(𝑷M​𝑪r(t)​𝑽r​3(t)−𝒀~M(3))​(𝑽r​3(t))⊤+(𝑪r​𝑼r​3(t)−𝒀~H(3))​(𝑼r​3(t))⊤+2​λ​𝑯3⊤​𝑯3​𝑪r(t),\displaystyle=\bm{P}_{M}^{\top}(\bm{P}_{M}\bm{C}_{r}^{(t)}\bm{V}_{r3}^{(t)}-\widetilde{\bm{Y}}_{M}^{(3)})(\bm{V}_{r3}^{(t)})^{\top}+(\bm{C}_{r}\bm{U}_{r3}^{(t)}-\widetilde{\bm{Y}}_{H}^{(3)})(\bm{U}_{r3}^{(t)})^{\top}+2\lambda\bm{H}_{3}^{\top}\bm{H}_{3}\bm{C}_{r}^{(t)},

L𝑪r(t)\displaystyle L_{\bm{C}_{r}}^{(t)}
=σmax​(𝑷M⊤​𝑷M)​σmax​(𝑽r​3(t)​(𝑽r​3(t))⊤)+σmax​(𝑼r​3(t)​(𝑼r​3(t))⊤)+2​λ​σmax​(𝑯3⊤​𝑯3),\displaystyle=\sigma_{\rm max}\left(\bm{P}_{M}^{\top}\bm{P}_{M}\right)\sigma_{\rm max}\left(\bm{V}_{r3}^{(t)}(\bm{V}_{r3}^{(t)})^{\top}\right)+\sigma_{\rm max}\left(\bm{U}_{r3}^{(t)}(\bm{U}_{r3}^{(t)})^{\top}\right)+2\lambda\sigma_{\textrm{max}}\left(\bm{H}_{3}^{\top}\bm{H}_{3}\right),

where 𝑽r​3(t)=(𝑫¯r(t)×1𝑨r(t+1)×2𝑩r(t+1))(3)\bm{V}_{r3}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{1}\bm{A}_{r}^{(t+1)}\times_{2}\bm{B}_{r}^{(t+1)})_{(3)} and 𝑼r​3(t)=(𝑫¯r(t)×1(𝑷1𝑨r(t+1))×2(𝑷2𝑩r(t+1))(3)\bm{U}_{r3}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{1}(\bm{P}_{1}\bm{A}_{r}^{(t+1)})\times_{2}(\bm{P}_{2}\bm{B}_{r}^{(t+1)})_{(3)}.

The 𝑫¯r\underline{\bm{D}}_{r}-subproblem can be rewritten as

𝒥⁡(𝑫¯r)=\displaystyle\mathcal{J}(\underline{\bm{D}}_{r})=
12‖𝒀¯~H−𝑫¯r×1(𝑷1𝑨r(t+1))×2(𝑷2𝑩r(t+1))×3𝑪r(t+1)‖F2\displaystyle{\frac{1}{2}\left\|\underline{\widetilde{\bm{Y}}}_{H}-\underline{\bm{D}}_{r}\times_{1}(\bm{P}_{1}\bm{A}_{r}^{(t+1)})\times_{2}(\bm{P}_{2}\bm{B}_{r}^{(t+1)})\times_{3}\bm{C}_{r}^{(t+1)}\right\|_{F}^{2}}

+\displaystyle+
12‖𝒀¯~M−𝑫¯r×1𝑨r(t+1)×2𝑩r(t+1)×3(𝑷M𝑪r(t+1))‖F2+η2∥𝑫¯r∥F2.\displaystyle{\frac{1}{2}\left\|\underline{\widetilde{\bm{Y}}}_{M}-\underline{\bm{D}}_{r}\times_{1}\bm{A}_{r}^{(t+1)}\times_{2}\bm{B}_{r}^{(t+1)}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r}^{(t+1)})\right\|_{F}^{2}}+\frac{\eta}{2}\|\underline{\bm{D}}_{r}\|_{F}^{2}.

Then one can easily obtain the gradient 𝑮𝑫¯r(t)\bm{G}_{\underline{\bm{D}}_{r}}^{(t)} and parameter L𝑫¯r(t)L_{\underline{\bm{D}}_{r}}^{(t)}.

𝑮𝑫¯r(t)\displaystyle\bm{G}_{\underline{\bm{D}}_{r}}^{(t)}
=(𝑼𝑫¯r(t))⊤​(𝑼𝑫¯r(t)​𝒅r(t)−𝒚m)+(𝑽𝑫¯r(t))⊤​(𝑽𝑫¯r(t)​𝒅r(t)−𝒚h)+η​𝒅r(t),\displaystyle=(\bm{U}^{(t)}_{\underline{\bm{D}}_{r}})^{\top}(\bm{U}^{(t)}_{\underline{\bm{D}}_{r}}\bm{d}_{r}^{(t)}-\bm{y}_{m})+(\bm{V}^{(t)}_{\underline{\bm{D}}_{r}})^{\top}(\bm{V}^{(t)}_{\underline{\bm{D}}_{r}}\bm{d}_{r}^{(t)}-\bm{y}_{h})+\eta\bm{d}_{r}^{(t)},

L𝑫¯r(t)\displaystyle L_{\underline{\bm{D}}_{r}}^{(t)}
=σmax2​(𝑷M​𝑪(t+1))​σmax2​(𝑨r(t+1))​σmax2​(𝑩r(t+1))\displaystyle=\sigma^{2}_{\rm max}\left(\bm{P}_{M}\bm{C}^{(t+1)}\right)\sigma^{2}_{\rm max}\left(\bm{A}_{r}^{(t+1)}\right)\sigma^{2}_{\rm max}\left(\bm{B}_{r}^{(t+1)}\right)

+σmax2​(𝑪r(t+1))​σmax2​(𝑷1​𝑨r(t+1))​σmax2​(𝑷2​𝑩r(t+1))+η,\displaystyle+\sigma^{2}_{\rm max}\left(\bm{C}_{r}^{(t+1)}\right)\sigma^{2}_{\rm max}\left(\bm{P}_{1}\bm{A}_{r}^{(t+1)}\right)\sigma^{2}_{\rm max}\left(\bm{P}_{2}\bm{B}_{r}^{(t+1)}\right)+\eta,

where 𝑼𝑫¯r(t)=(𝑷M​𝑪r(t+1))⊗𝑩r(t+1)⊗𝑨r(t+1)\bm{U}^{(t)}_{\underline{\bm{D}}_{r}}=(\bm{P}_{M}\bm{C}_{r}^{(t+1)})\otimes\bm{B}_{r}^{(t+1)}\otimes\bm{A}_{r}^{(t+1)}, 𝑽𝑫¯r(t)=𝑪r(t+1)⊗(𝑷1​𝑨r(t+1))⊗(𝑷2​𝑩r(t+1))\bm{V}^{(t)}_{\underline{\bm{D}}_{r}}=\bm{C}_{r}^{(t+1)}\otimes(\bm{P}_{1}\bm{A}_{r}^{(t+1)})\otimes(\bm{P}_{2}\bm{B}_{r}^{(t+1)}), 𝒅r(t)\bm{d}_{r}^{(t)}, 𝒚m\bm{y}_{m}, and 𝒚h\bm{y}_{h} denote the vectorization of 𝑫¯r(t)\underline{\bm{D}}_{r}^{(t)}, 𝒀¯M\underline{\bm{Y}}_{\rm M}, and 𝒀¯H\underline{\bm{Y}}_{\rm H}, respectively.

The detailed complexity cost of the proposed algorithm is listed in Table 10. Assuming IM≈JM≈KH,IH≈JH≥KM,Lr≈Mr≥NrI_{\rm M}\approx J_{\rm M}\approx K_{\rm H},~I_{\rm H}\approx J_{\rm H}\geq K_{\rm M},~L_{r}\approx M_{r}\geq N_{r}, the computation of the gradients 𝑮𝑨r(t)\bm{G}^{(t)}_{\bm{A}_{r}}, 𝑮𝑩r(t)\bm{G}^{(t)}_{\bm{B}_{r}}, 𝑮𝑪r(t)\bm{G}^{(t)}_{\bm{C}_{r}}, and 𝑮𝑫¯r(t)\bm{G}^{(t)}_{\underline{\bm{D}}_{r}} takes
𝒪⁡(IM2​(IH​(IH+Lr)+Lr​(KM+Nr)))\mathcal{O}(I_{\rm M}^{2}(I_{\rm H}(I_{\rm H}+L_{r})+L_{r}(K_{\rm M}+N_{r}))),
𝒪⁡(IM2​(IH​(IH+Lr)+Lr​(KM+Nr)))\mathcal{O}(I_{\rm M}^{2}(I_{\rm H}(I_{\rm H}+L_{r})+L_{r}(K_{\rm M}+N_{r}))),
𝒪⁡(IM2​KH​(KM+Nr))\mathcal{O}(I_{\rm M}^{2}K_{\rm H}(K_{\rm M}+N_{r})), and
𝒪⁡(IM2​Lr​KM+KH​Lr​(IH2+IH​Lr+Lr​Nr))\mathcal{O}(I_{\rm M}^{2}L_{r}K_{\rm M}+K_{\rm H}L_{r}(I_{\rm H}^{2}+I_{\rm H}L_{r}+L_{r}N_{r})), respectively.

Another part costs many computation flops is to compute the step sizes αr,i(t)\alpha_{r,i}^{(t)}. To reduce the computation, instead of computing the exact values, we compute their upper bounds as in [12, 13]. For example, we have

σmax​(𝑯~1⊤​𝑾r​1(t)​𝑯~1)≤σmax​(𝑯~1⊤)​σmax​(𝑾r​1(t))​σmax​(𝑯~1),\sigma_{\textrm{max}}(\tilde{\bm{H}}_{1}^{\top}\bm{W}_{r1}^{(t)}\tilde{\bm{H}}_{1})\leq\sigma_{\textrm{max}}(\tilde{\bm{H}}_{1}^{\top})\sigma_{\max}(\bm{W}_{r1}^{(t)})\sigma_{\rm max}(\tilde{\bm{H}}_{1}),

where the symbol σmax​(𝑿)\sigma_{\max}(\bm{X}) denotes the largest singular value of the matrix 𝑿\bm{X}. Here, σmax​(𝑾r​1(t))\sigma_{\max}(\bm{W}_{r1}^{(t)}) is simply chosen to be the largest value of the diagonal matrix 𝑾r​1(t)\bm{W}_{r1}^{(t)} and the other two terms are pre-computed. Therefore, computing αr,i(t)\alpha_{r,i}^{(t)} takes 𝒪⁡(Lr3+Mr3+Nr3)\mathcal{O}(L_{r}^{3}+M_{r}^{3}+N_{r}^{3}) flops. In summary, the per-iteration complexity is
𝒪⁡(IM3​(IH+Lr))\mathcal{O}(I_{\rm M}^{3}(I_{\rm H}+L_{r})).

*Algorithm 2 BCLIMB for solving (25).*

Input:
HSI 𝒀¯H\underline{\bm{Y}}_{{\rm H}}, MSI 𝒀¯M\underline{\bm{Y}}_{{\rm M}};
pre-defined sequences of μr,i(t)\mu_{r,i}^{(t)}, αr,i(t)\alpha_{r,i}^{(t)}, μ~r,j(t)\widetilde{\mu}_{r,j}^{(t)}, and α~r,j(t)\widetilde{\alpha}_{r,j}^{(t)} (r=1,…,R,i=1,…,4,j=1,2r=1,\ldots,R,~i=1,\ldots,4,~j=1,2);
starting points {𝑫¯r(0),𝑨~r(0),𝑩~r(0),𝑨r(0),𝑩r(0),𝑪r(0)}r=1R\left\{\underline{\bm{D}}_{r}^{(0)},\widetilde{\bm{A}}_{r}^{(0)},\widetilde{\bm{B}}_{r}^{(0)},\bm{A}_{r}^{(0)},\bm{B}_{r}^{(0)},\bm{C}_{r}^{(0)}\right\}_{r=1}^{R}.

Parameters: λ\lambda, η\eta, pp, ε\varepsilon, LL, and NN.

𝑫¯ˇr(0)=𝑫¯r(0)\check{\underline{\bm{D}}}_{r}^{(0)}=\underline{\bm{D}}_{r}^{(0)}, 𝑨¯r(0)=𝑨~r(0)\bar{\bm{A}}_{r}^{(0)}=\widetilde{\bm{A}}_{r}^{(0)}, 𝑩¯r(0)=𝑩~r(0)\bar{\bm{B}}_{r}^{(0)}=\widetilde{\bm{B}}_{r}^{(0)}, 𝑨ˇr(0)=𝑨r(0)\check{\bm{A}}_{r}^{(0)}=\bm{A}_{r}^{(0)}, 𝑩ˇr(0)=𝑩r(0)\check{\bm{B}}_{r}^{(0)}=\bm{B}_{r}^{(0)}, 𝑪ˇr(0)=𝑪r(0)\check{\bm{C}}_{r}^{(0)}=\bm{C}_{r}^{(0)}, t=0t=0.

repeat

for r=1:Rr=1:R

%%\%\% update 𝑨~r\widetilde{\bm{A}}_{r} %%\%\%

𝑨~r(t+1)←𝑨¯r(t)−α~r,1(t)​𝑮𝑨¯r(t)\widetilde{\bm{A}}_{r}^{(t+1)}\leftarrow\bar{\bm{A}}_{r}^{(t)}-\widetilde{\alpha}_{r,1}^{(t)}\bm{G}_{\bar{\bm{A}}_{r}}^{(t)};
  𝑨¯r(t+1)←𝑨~r(t+1)+μ~r,1(t)​(𝑨~r(t+1)−𝑨~r(t))\bar{\bm{A}}_{r}^{(t+1)}\leftarrow\widetilde{\bm{A}}_{r}^{(t+1)}+\widetilde{\mu}_{r,1}^{(t)}(\widetilde{\bm{A}}_{r}^{(t+1)}-\widetilde{\bm{A}}_{r}^{(t)});

%%\%\% update 𝑨r\bm{A}_{r} %%\%\%

𝑨r(t+1)←𝑨ˇr(t)−αr,1(t)​𝑮𝑨ˇr(t)\bm{A}_{r}^{(t+1)}\leftarrow\check{\bm{A}}_{r}^{(t)}-\alpha_{r,1}^{(t)}\bm{G}_{\check{\bm{A}}_{r}}^{(t)};
  𝑨ˇr(t+1)←𝑨r(t+1)+μr,1(t)​(𝑨r(t+1)−𝑨r(t))\check{\bm{A}}_{r}^{(t+1)}\leftarrow\bm{A}_{r}^{(t+1)}+\mu_{r,1}^{(t)}(\bm{A}_{r}^{(t+1)}-\bm{A}_{r}^{(t)});

%%\%\% update 𝑩~r\widetilde{\bm{B}}_{r} %%\%\%

𝑩~r(t+1)←𝑩¯r(t)−α~r,2(t)​𝑮𝑩¯r(t)\widetilde{\bm{B}}_{r}^{(t+1)}\leftarrow\bar{\bm{B}}_{r}^{(t)}-\widetilde{\alpha}_{r,2}^{(t)}\bm{G}_{\bar{\bm{B}}_{r}}^{(t)};
  𝑩¯r(t+1)←𝑩~r(t+1)+μ~r,2(t)​(𝑩~r(t+1)−𝑩~r(t))\bar{\bm{B}}_{r}^{(t+1)}\leftarrow\widetilde{\bm{B}}_{r}^{(t+1)}+\widetilde{\mu}_{r,2}^{(t)}(\widetilde{\bm{B}}_{r}^{(t+1)}-\widetilde{\bm{B}}_{r}^{(t)});

%%\%\% update 𝑩r\bm{B}_{r} %%\%\%

𝑩r(t+1)←𝑩ˇr(t)−αr,2(t)​𝑮𝑩ˇr(t)\bm{B}_{r}^{(t+1)}\leftarrow\check{\bm{B}}_{r}^{(t)}-\alpha_{r,2}^{(t)}\bm{G}_{\check{\bm{B}}_{r}}^{(t)};
  𝑩ˇr(t+1)←𝑩r(t+1)+μr,2(t)​(𝑩r(t+1)−𝑩r(t))\check{\bm{B}}_{r}^{(t+1)}\leftarrow\bm{B}_{r}^{(t+1)}+\mu_{r,2}^{(t)}(\bm{B}_{r}^{(t+1)}-\bm{B}_{r}^{(t)});

%%\%\% update 𝑪r\bm{C}_{r} %%\%\%

𝑪r(t+1)←𝑪ˇr(t)−αr,3(t)​𝑮𝑪ˇr(t)\bm{C}_{r}^{(t+1)}\leftarrow\check{\bm{C}}_{r}^{(t)}-\alpha_{r,3}^{(t)}\bm{G}_{\check{\bm{C}}_{r}}^{(t)};
  𝑪ˇr(t+1)←𝑪r(t+1)+μr,3(t)​(𝑪r(t+1)−𝑪r(t))\check{\bm{C}}_{r}^{(t+1)}\leftarrow\bm{C}_{r}^{(t+1)}+\mu_{r,3}^{(t)}(\bm{C}_{r}^{(t+1)}-\bm{C}_{r}^{(t)});

%%\%\% update 𝑫¯r\underline{\bm{D}}_{r} %%\%\%

𝑫¯r(t+1)←𝑫¯ˇr(t)−αr,4(t)​𝑮𝑫¯ˇr(t)\underline{\bm{D}}_{r}^{(t+1)}\leftarrow\check{\underline{\bm{D}}}_{r}^{(t)}-\alpha_{r,4}^{(t)}\bm{G}_{\check{\underline{\bm{D}}}_{r}}^{(t)};
  𝑫¯ˇr(t+1)←𝑫¯r(t+1)+μr,4(t)​(𝑫¯r(t+1)−𝑫¯r(t))\check{\underline{\bm{D}}}_{r}^{(t+1)}\leftarrow\underline{\bm{D}}_{r}^{(t+1)}+\mu_{r,4}^{(t)}(\underline{\bm{D}}_{r}^{(t+1)}-\underline{\bm{D}}_{r}^{(t)});

end

t=t+1t=t+1;

until satisfying the stopping criterion.

Output: 𝑨^r=𝑨r(t)\widehat{\bm{A}}_{r}=\bm{A}_{r}^{(t)}, 𝑩^r=𝑩r(t)\widehat{\bm{B}}_{r}=\bm{B}_{r}^{(t)}, 𝑪^r=𝑪r(t)\widehat{\bm{C}}_{r}=\bm{C}_{r}^{(t)}, and 𝑫¯^r=𝑫¯r(t)\widehat{\underline{\bm{D}}}_{r}=\underline{\bm{D}}_{r}^{(t)}. Recover 𝒀¯^S=∑r=1R𝑫¯^r×1𝑨^r×2𝑩^r×3𝑪^r\underline{\widehat{\bm{Y}}}_{\rm S}=\sum_{r=1}^{R}\underline{\widehat{\bm{D}}}_{r}\times_{1}\widehat{\bm{A}}_{r}\times_{2}\widehat{\bm{B}}_{r}\times_{3}\widehat{\bm{C}}_{r}.

## Appendix F Algorithm Details for Model (25)

In this section, we present the details for our semi-blind CLIMB (BCLIMB) for handling (25), which is
a simple extension of CLIMB and
is summarized in Algorithm 2.
When the spatial degradation operators are unknown, one can apply the similar calculations of gradients and parameters as in Appendix E.
Then we can obtain the following expressions.

𝑮𝑨~r(t)\displaystyle\bm{G}_{\widetilde{\bm{A}}_{r}}^{(t)}
=(𝑨~r(t)​𝑽¯r​1(t)−𝒀H(1))​(𝑽¯r​1(t))⊤,\displaystyle=(\widetilde{\bm{A}}_{r}^{(t)}\bar{\bm{V}}_{r1}^{(t)}-\bm{Y}_{H}^{(1)})(\bar{\bm{V}}_{r1}^{(t)})^{\top},

L𝑨~r(t)\displaystyle L_{\widetilde{\bm{A}}_{r}}^{(t)}
=σmax​(𝑽¯r​1(t)​(𝑽¯r​1(t))⊤),\displaystyle=\sigma_{\rm max}\left(\bar{\bm{V}}_{r1}^{(t)}(\bar{\bm{V}}_{r1}^{(t)})^{\top}\right),

where 𝑽¯r​1(t)=(𝑫¯(t)×2𝑩~r(t)×3𝑪r(t))(1)\bar{\bm{V}}_{r1}^{(t)}=(\underline{\bm{D}}^{(t)}\times_{2}\widetilde{\bm{B}}_{r}^{(t)}\times_{3}\bm{C}_{r}^{(t)})_{(1)} and 𝒀H(1)=(𝒀¯H−∑r¯≠r𝑫¯r¯(t)×1𝑨~r¯(t)×2𝑩~r¯(t)×3𝑪r¯(t))(1)\bm{Y}_{H}^{(1)}=(\underline{\bm{Y}}_{H}-\sum_{\bar{r}\neq r}\underline{\bm{D}}^{(t)}_{\bar{r}}\times_{1}\widetilde{\bm{A}}^{(t)}_{\bar{r}}\times_{2}\widetilde{\bm{B}}^{(t)}_{\bar{r}}\times_{3}\bm{C}^{(t)}_{\bar{r}})_{(1)}.

*Table 11: Complexity of each term of BCLIMB.*

Terms
Complexity

(𝑨~r(t)​𝑽¯r​1(t)−𝒀H(1))​(𝑽¯r​1(t))⊤(\widetilde{\bm{A}}_{r}^{(t)}\bar{\bm{V}}_{r1}^{(t)}-\bm{Y}_{H}^{(1)})(\bar{\bm{V}}_{r1}^{(t)})^{\top}
𝒪⁡(IH​Nr​(Lr​Mr+Mr​JH+JH​KH)+JH​Lr​(Nr​Mr+Nr​KH+IH​KH))\mathcal{O}(I_{\rm H}N_{r}(L_{r}M_{r}+M_{r}J_{\rm H}+J_{\rm H}K_{\rm H})+J_{\rm H}L_{r}(N_{r}M_{r}+N_{r}K_{\rm H}+I_{\rm H}K_{\rm H}))

(𝑨r​𝑼¯r​1(t)−𝒀M(1))​(𝑼¯r​1(t))⊤(\bm{A}_{r}\bar{\bm{U}}_{r1}^{(t)}-\bm{Y}_{M}^{(1)})(\bar{\bm{U}}_{r1}^{(t)})^{\top}
𝒪⁡(KH​KM​Nr+Lr​Nr​(JM​Mr+JM​KM+IM​Lr)+IM​JM​(Nr​Mr+Nr​KM+KM​Lr))\mathcal{O}(K_{\rm H}K_{\rm M}N_{r}+L_{r}N_{r}(J_{\rm M}M_{r}+J_{\rm M}K_{\rm M}+I_{\rm M}L_{r})+I_{\rm M}J_{\rm M}(N_{r}M_{r}+N_{r}K_{\rm M}+K_{\rm M}L_{r}))

(𝑩~r(t)​𝑽¯r​2(t)−𝒀H(2))​(𝑽¯r​2(t))⊤(\widetilde{\bm{B}}_{r}^{(t)}\bar{\bm{V}}_{r2}^{(t)}-\bm{Y}_{H}^{(2)})(\bar{\bm{V}}_{r2}^{(t)})^{\top}
𝒪⁡(IH​Nr​(Lr​Mr+Mr​JH+JH​KH)+IH​KH​(IM​Mr+Mr​JH+IM​KH))\mathcal{O}(I_{\rm H}N_{r}(L_{r}M_{r}+M_{r}J_{\rm H}+J_{\rm H}K_{\rm H})+I_{\rm H}K_{\rm H}(I_{\rm M}M_{r}+M_{r}J_{\rm H}+I_{\rm M}K_{\rm H}))

(𝑩r​𝑼¯r​2(t)−𝒀M(2))​(𝑼¯r​2(t))⊤(\bm{B}_{r}\bar{\bm{U}}_{r2}^{(t)}-\bm{Y}_{M}^{(2)})(\bar{\bm{U}}_{r2}^{(t)})^{\top}
𝒪⁡(KH​KM​Nr+Lr​Nr​(IM​Lr+IM​KM+IM​Mr)+IM​JM​(Nr​Mr+Nr​KM+KM​Mr))\mathcal{O}(K_{\rm H}K_{\rm M}N_{r}+L_{r}N_{r}(I_{\rm M}L_{r}+I_{\rm M}K_{\rm M}+I_{\rm M}M_{r})+I_{\rm M}J_{\rm M}(N_{r}M_{r}+N_{r}K_{\rm M}+K_{\rm M}M_{r}))

𝑷M⊤​(𝑷M​𝑪r(t)​𝑽¯r​3(t)−𝒀M(3))​(𝑽¯r​3(t))⊤\bm{P}_{M}^{\top}(\bm{P}_{M}\bm{C}_{r}^{(t)}\bar{\bm{V}}_{r3}^{(t)}-\bm{Y}_{M}^{(3)})(\bar{\bm{V}}_{r3}^{(t)})^{\top}
𝒪⁡(IM​Nr​(Mr​Lr+JM​KM+JM​Mr)+IM​JM​(KH​Nr+KM​KH+KM​Mr)+IM​Mr​Nr​(JM+Lr))\mathcal{O}(I_{\rm M}N_{r}(M_{r}L_{r}+J_{\rm M}K_{\rm M}+J_{\rm M}M_{r})+I_{\rm M}J_{\rm M}(K_{\rm H}N_{r}+K_{\rm M}K_{\rm H}+K_{\rm M}M_{r})+I_{\rm M}M_{r}N_{r}(J_{\rm M}+L_{r}))

(𝑪r​𝑼¯r​3(t)−𝒀H(3))​(𝑼¯r​3(t))⊤(\bm{C}_{r}\bar{\bm{U}}_{r3}^{(t)}-\bm{Y}_{H}^{(3)})(\bar{\bm{U}}_{r3}^{(t)})^{\top}
𝒪⁡(IH​IM​Lr+JH​JM​Mr+IH​Nr​(Mr​JH+Mr​Lr+JH​KH))\mathcal{O}(I_{\rm H}I_{\rm M}L_{r}+J_{\rm H}J_{\rm M}M_{r}+I_{\rm H}N_{r}(M_{r}J_{\rm H}+M_{r}L_{r}+J_{\rm H}K_{\rm H}))

(𝑼¯𝑫¯r(t))⊤​(𝑼¯𝑫¯r(t)​𝒅r(t)−𝒚m)(\bar{\bm{U}}^{(t)}_{\underline{\bm{D}}_{r}})^{\top}(\bar{\bm{U}}^{(t)}_{\underline{\bm{D}}_{r}}\bm{d}_{r}^{(t)}-\bm{y}_{m})
𝒪⁡(KM​Lr​(IM​JM+JM​Mr+Mr​Nr))\mathcal{O}(K_{\rm M}L_{r}(I_{\rm M}J_{\rm M}+J_{\rm M}M_{r}+M_{r}N_{r}))

(𝑽¯𝑫¯r(t))⊤​(𝑽¯𝑫¯r(t)​𝒅r(t)−𝒚h)(\bar{\bm{V}}^{(t)}_{\underline{\bm{D}}_{r}})^{\top}(\bar{\bm{V}}^{(t)}_{\underline{\bm{D}}_{r}}\bm{d}_{r}^{(t)}-\bm{y}_{h})
𝒪⁡(KH​Lr​(IH​JH+JH​Mr+Mr​Nr))\mathcal{O}(K_{\rm H}L_{r}(I_{\rm H}J_{\rm H}+J_{\rm H}M_{r}+M_{r}N_{r}))

𝑯~1⊤​𝑾r​1(t)​𝑯~1​𝒂r(t)\tilde{\bm{H}}_{1}^{\top}\bm{W}_{r1}^{(t)}\tilde{\bm{H}}_{1}\bm{a}_{r}^{(t)}
𝒪⁡(IM​Lr)\mathcal{O}(I_{\rm M}L_{r})

𝑯~2⊤​𝑾r​2(t)​𝑯~2​𝒃r(t)\tilde{\bm{H}}_{2}^{\top}\bm{W}_{r2}^{(t)}\tilde{\bm{H}}_{2}\bm{b}_{r}^{(t)}
𝒪⁡(JM​Mr)\mathcal{O}(J_{\rm M}M_{r})

𝑯3⊤​𝑯3​𝑪r(t)\bm{H}_{3}^{\top}\bm{H}_{3}\bm{C}_{r}^{(t)}
𝒪⁡(KH2​Nr)\mathcal{O}(K_{\rm H}^{2}N_{r})

αr,i\alpha_{r,i}, α~r,j\widetilde{\alpha}_{r,j}
𝒪⁡(Lr3+Mr3+Nr3)\mathcal{O}(L_{r}^{3}+M_{r}^{3}+N_{r}^{3})

𝑮𝑨r(t)\displaystyle\bm{G}_{\bm{A}_{r}}^{(t)}
=(𝑨r​𝑼¯r​1(t)−𝒀M(1))​(𝑼¯r​1(t))⊤+λ​p​𝑯~1⊤​𝑾r​1(t)​𝑯~1​𝒂r(t),\displaystyle=(\bm{A}_{r}\bar{\bm{U}}_{r1}^{(t)}-\bm{Y}_{M}^{(1)})(\bar{\bm{U}}_{r1}^{(t)})^{\top}+\lambda p\tilde{\bm{H}}_{1}^{\top}\bm{W}_{r1}^{(t)}\tilde{\bm{H}}_{1}\bm{a}_{r}^{(t)},

L𝑨r(t)\displaystyle L_{\bm{A}_{r}}^{(t)}
=σmax​(𝑼¯r​1(t)​(𝑼¯r​1(t))⊤)+λ​p​σmax​(𝑯~1⊤​𝑾r​1(t)​𝑯~1),\displaystyle=\sigma_{\rm max}\left(\bar{\bm{U}}_{r1}^{(t)}(\bar{\bm{U}}_{r1}^{(t)})^{\top}\right)+\lambda p\sigma_{\textrm{max}}\left(\tilde{\bm{H}}_{1}^{\top}\bm{W}_{r1}^{(t)}\tilde{\bm{H}}_{1}\right),

where 𝑼¯r​1(t)=(𝑫¯r(t)×2𝑩r(t)×3(𝑷M𝑪r(t)))(1)\bar{\bm{U}}_{r1}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{2}\bm{B}_{r}^{(t)}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r}^{(t)}))_{(1)} and 𝒀M(1)=(𝒀¯M−∑r¯≠r𝑫¯r¯(t)×1𝑨r¯(t)×2𝑩r¯(t)×3(𝑷M𝑪r¯(t)))(1)\bm{Y}_{M}^{(1)}=(\underline{\bm{Y}}_{M}-\sum_{\bar{r}\neq r}\underline{\bm{D}}^{(t)}_{\bar{r}}\times_{1}\bm{A}^{(t)}_{\bar{r}}\times_{2}\bm{B}^{(t)}_{\bar{r}}\times_{3}(\bm{P}_{M}\bm{C}^{(t)}_{\bar{r}}))_{(1)}.

𝑮𝑩~r(t)\displaystyle\bm{G}_{\widetilde{\bm{B}}_{r}}^{(t)}
=(𝑩~r(t)​𝑽¯r​2(t)−𝒀H(2))​(𝑽¯r​2(t))⊤,\displaystyle=(\widetilde{\bm{B}}_{r}^{(t)}\bar{\bm{V}}_{r2}^{(t)}-\bm{Y}_{H}^{(2)})(\bar{\bm{V}}_{r2}^{(t)})^{\top},

L𝑩~r(t)\displaystyle L_{\widetilde{\bm{B}}_{r}}^{(t)}
=σmax​(𝑽¯r​2(t)​(𝑽¯r​2(t))⊤),\displaystyle=\sigma_{\rm max}\left(\bar{\bm{V}}_{r2}^{(t)}(\bar{\bm{V}}_{r2}^{(t)})^{\top}\right),

where 𝑽¯r​2(t)=(𝑫¯r(t)×1𝑨~r(t+1)×3𝑪r(t))(2)\bar{\bm{V}}_{r2}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{1}\widetilde{\bm{A}}_{r}^{(t+1)}\times_{3}\bm{C}_{r}^{(t)})_{(2)} and 𝒀H(2)=(𝒀¯H−∑r¯≠r𝑫¯r¯(t)×1𝑨~r¯(t)×2𝑩~r¯(t)×3𝑪r¯(t))(2)\bm{Y}_{H}^{(2)}=(\underline{\bm{Y}}_{H}-\sum_{\bar{r}\neq r}\underline{\bm{D}}^{(t)}_{\bar{r}}\times_{1}\widetilde{\bm{A}}^{(t)}_{\bar{r}}\times_{2}\widetilde{\bm{B}}^{(t)}_{\bar{r}}\times_{3}\bm{C}^{(t)}_{\bar{r}})_{(2)}.

𝑮𝑩r(t)\displaystyle\bm{G}_{\bm{B}_{r}}^{(t)}
=(𝑩r​𝑼¯r​2(t)−𝒀M(2))​(𝑼¯r​2(t))⊤+λ​p​𝑯~2⊤​𝑾r​2(t)​𝑯~2​𝒃r(t),\displaystyle=(\bm{B}_{r}\bar{\bm{U}}_{r2}^{(t)}-\bm{Y}_{M}^{(2)})(\bar{\bm{U}}_{r2}^{(t)})^{\top}+\lambda p\tilde{\bm{H}}_{2}^{\top}\bm{W}_{r2}^{(t)}\tilde{\bm{H}}_{2}\bm{b}_{r}^{(t)},

L𝑩r(t)\displaystyle L_{\bm{B}_{r}}^{(t)}
=σmax​(𝑼¯r​2(t)​(𝑼¯r​2(t))⊤)+λ​p​σmax​(𝑯~2⊤​𝑾r​2(t)​𝑯~2),\displaystyle=\sigma_{\rm max}\left(\bar{\bm{U}}_{r2}^{(t)}(\bar{\bm{U}}_{r2}^{(t)})^{\top}\right)+\lambda p\sigma_{\textrm{max}}\left(\tilde{\bm{H}}_{2}^{\top}\bm{W}_{r2}^{(t)}\tilde{\bm{H}}_{2}\right),

where 𝑼¯r​2(t)=(𝑫¯r(t)×2𝑩r(t)×3(𝑷M𝑪r(t)))(2)\bar{\bm{U}}_{r2}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{2}\bm{B}_{r}^{(t)}\times_{3}(\bm{P}_{\rm M}\bm{C}_{r}^{(t)}))_{(2)} and 𝒀M(2)=(𝒀¯M−∑r¯≠r𝑫¯r¯(t)×1𝑨r¯(t)×2𝑩r¯(t)×3(𝑷M𝑪r¯(t)))(2)\bm{Y}_{M}^{(2)}=(\underline{\bm{Y}}_{M}-\sum_{\bar{r}\neq r}\underline{\bm{D}}^{(t)}_{\bar{r}}\times_{1}\bm{A}^{(t)}_{\bar{r}}\times_{2}\bm{B}^{(t)}_{\bar{r}}\times_{3}(\bm{P}_{M}\bm{C}^{(t)}_{\bar{r}}))_{(2)}.

𝑮𝑪r(t)\displaystyle\bm{G}_{\bm{C}_{r}}^{(t)}
=𝑷M⊤​(𝑷M​𝑪r(t)​𝑽¯r​3(t)−𝒀M(3))​(𝑽¯r​3(t))⊤+(𝑪r​𝑼¯r​3(t)−𝒀H(3))​(𝑼¯r​3(t))⊤+2​λ​𝑯3⊤​𝑯3​𝑪r(t),\displaystyle=\bm{P}_{M}^{\top}(\bm{P}_{M}\bm{C}_{r}^{(t)}\bar{\bm{V}}_{r3}^{(t)}-\bm{Y}_{M}^{(3)})(\bar{\bm{V}}_{r3}^{(t)})^{\top}+(\bm{C}_{r}\bar{\bm{U}}_{r3}^{(t)}-\bm{Y}_{H}^{(3)})(\bar{\bm{U}}_{r3}^{(t)})^{\top}+2\lambda\bm{H}_{3}^{\top}\bm{H}_{3}\bm{C}_{r}^{(t)},

L𝑪r(t)\displaystyle L_{\bm{C}_{r}}^{(t)}
=σmax​(𝑷M⊤​𝑷M)​σmax​(𝑽¯r​3(t)​(𝑽¯r​3(t))⊤)+σmax​(𝑼¯r​3(t)​(𝑼¯r​3(t))⊤)+2​λ​σmax​(𝑯3⊤​𝑯3),\displaystyle=\sigma_{\rm max}\left(\bm{P}_{M}^{\top}\bm{P}_{M}\right)\sigma_{\rm max}\left(\bar{\bm{V}}_{r3}^{(t)}(\bar{\bm{V}}_{r3}^{(t)})^{\top}\right)+\sigma_{\rm max}\left(\bar{\bm{U}}_{r3}^{(t)}(\bar{\bm{U}}_{r3}^{(t)})^{\top}\right)+2\lambda\sigma_{\textrm{max}}\left(\bm{H}_{3}^{\top}\bm{H}_{3}\right),

where
𝑽¯r​3(t)=(𝑫¯r(t)×1𝑨r(t+1)×2𝑩r(t+1))(3)\bar{\bm{V}}_{r3}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{1}\bm{A}_{r}^{(t+1)}\times_{2}\bm{B}_{r}^{(t+1)})_{(3)}, 𝑼¯r​3(t)=(𝑫¯r(t)×1𝑨~r(t+1)×2𝑩~r(t+1))(3)\bar{\bm{U}}_{r3}^{(t)}=(\underline{\bm{D}}_{r}^{(t)}\times_{1}\widetilde{\bm{A}}_{r}^{(t+1)}\times_{2}\widetilde{\bm{B}}_{r}^{(t+1)})_{(3)},
𝒀M(3)=(𝒀¯M−∑r¯≠r𝑫¯r¯(t)×1𝑨r¯(t)×2𝑩r¯(t)×3(𝑷M𝑪r¯(t)))(3)\bm{Y}_{M}^{(3)}=(\underline{\bm{Y}}_{M}-\sum_{\bar{r}\neq r}\underline{\bm{D}}^{(t)}_{\bar{r}}\times_{1}\bm{A}^{(t)}_{\bar{r}}\times_{2}\bm{B}^{(t)}_{\bar{r}}\times_{3}(\bm{P}_{M}\bm{C}^{(t)}_{\bar{r}}))_{(3)}, 𝒀H(3)=(𝒀¯H−∑r¯≠r𝑫¯r¯(t)×1𝑨~r¯(t)×2𝑩~r¯(t)×3𝑪r¯(t))(3)\bm{Y}_{H}^{(3)}=(\underline{\bm{Y}}_{H}-\sum_{\bar{r}\neq r}\underline{\bm{D}}^{(t)}_{\bar{r}}\times_{1}\widetilde{\bm{A}}^{(t)}_{\bar{r}}\times_{2}\widetilde{\bm{B}}^{(t)}_{\bar{r}}\times_{3}\bm{C}^{(t)}_{\bar{r}})_{(3)}.

𝑮𝑫¯r(t)\displaystyle\bm{G}_{\underline{\bm{D}}_{r}}^{(t)}
=(𝑼¯𝑫¯r(t))⊤​(𝑼¯𝑫¯r(t)​𝒅r(t)−𝒚m)+(𝑽¯𝑫¯r(t))⊤​(𝑽¯𝑫¯r(t)​𝒅r(t)−𝒚h)+η​𝒅r(t),\displaystyle=(\bar{\bm{U}}^{(t)}_{\underline{\bm{D}}_{r}})^{\top}(\bar{\bm{U}}^{(t)}_{\underline{\bm{D}}_{r}}\bm{d}_{r}^{(t)}-\bm{y}_{m})+(\bar{\bm{V}}^{(t)}_{\underline{\bm{D}}_{r}})^{\top}(\bar{\bm{V}}^{(t)}_{\underline{\bm{D}}_{r}}\bm{d}_{r}^{(t)}-\bm{y}_{h})+\eta\bm{d}_{r}^{(t)},

L𝑫¯r(t)\displaystyle L_{\underline{\bm{D}}_{r}}^{(t)}
=σmax2​(𝑷M​𝑪(t+1))​σmax2​(𝑨r(t+1))​σmax2​(𝑩r(t+1))\displaystyle=\sigma^{2}_{\rm max}\left(\bm{P}_{M}\bm{C}^{(t+1)}\right)\sigma^{2}_{\rm max}\left(\bm{A}_{r}^{(t+1)}\right)\sigma^{2}_{\rm max}\left(\bm{B}_{r}^{(t+1)}\right)

+σmax2​(𝑪r(t+1))​σmax2​(𝑷1​𝑨r(t+1))​σmax2​(𝑷2​𝑩r(t+1))+η,\displaystyle+\sigma^{2}_{\rm max}\left(\bm{C}_{r}^{(t+1)}\right)\sigma^{2}_{\rm max}\left(\bm{P}_{1}\bm{A}_{r}^{(t+1)}\right)\sigma^{2}_{\rm max}\left(\bm{P}_{2}\bm{B}_{r}^{(t+1)}\right)+\eta,

where 𝑼¯𝑫¯r(t)=(𝑷M​𝑪r(t+1))⊗𝑩r(t+1)⊗𝑨r(t+1)\bar{\bm{U}}^{(t)}_{\underline{\bm{D}}_{r}}=(\bm{P}_{M}\bm{C}_{r}^{(t+1)})\otimes\bm{B}_{r}^{(t+1)}\otimes\bm{A}_{r}^{(t+1)}, 𝑽¯𝑫¯r(t)=𝑪r(t+1)⊗𝑨~r(t+1)⊗𝑩~r(t+1)\bar{\bm{V}}^{(t)}_{\underline{\bm{D}}_{r}}=\bm{C}_{r}^{(t+1)}\otimes\widetilde{\bm{A}}_{r}^{(t+1)}\otimes\widetilde{\bm{B}}_{r}^{(t+1)}.

We list the detailed complexity cost of the proposed BCLIMB in Table 11.
Assuming IM≈JM≈KH,IH≈JH≥KM,Lr≈Mr≥NrI_{\rm M}\approx J_{\rm M}\approx K_{\rm H},~I_{\rm H}\approx J_{\rm H}\geq K_{\rm M},~L_{r}\approx M_{r}\geq N_{r}, the computation of the gradients 𝑮𝑨~r(t)\bm{G}^{(t)}_{\widetilde{\bm{A}}_{r}}, 𝑮𝑨r(t)\bm{G}^{(t)}_{\bm{A}_{r}}, 𝑮𝑩~r(t)\bm{G}^{(t)}_{\widetilde{\bm{B}}_{r}}, 𝑮𝑩r(t)\bm{G}^{(t)}_{\bm{B}_{r}}, 𝑮𝑪r(t)\bm{G}^{(t)}_{\bm{C}_{r}}, and 𝑮𝑫¯r(t)\bm{G}^{(t)}_{\underline{\bm{D}}_{r}} takes
𝒪⁡(IM2​IH​(IH+Lr))\mathcal{O}(I_{\rm M}^{2}I_{\rm H}(I_{\rm H}+L_{r})),
𝒪⁡(IM2​Lr​(KM+Nr))\mathcal{O}(I_{\rm M}^{2}L_{r}(K_{\rm M}+N_{r})),
𝒪⁡(IM2​IH​(IH+Lr))\mathcal{O}(I_{\rm M}^{2}I_{\rm H}(I_{\rm H}+L_{r})),
𝒪⁡(IM2​Lr​(KM+Nr))\mathcal{O}(I_{\rm M}^{2}L_{r}(K_{\rm M}+N_{r})),
𝒪⁡(IM2​KH​(KM+Nr)+IH​Nr​(IM+Lr2+IH​Lr))\mathcal{O}(I_{\rm M}^{2}K_{\rm H}(K_{\rm M}+N_{r})+I_{\rm H}N_{r}(I_{\rm M}+L_{r}^{2}+I_{\rm H}L_{r})), and
𝒪⁡(IM2​Lr​KM+KH​Lr​(IH2+IH​Lr+Lr​Nr))\mathcal{O}(I_{\rm M}^{2}L_{r}K_{\rm M}+K_{\rm H}L_{r}(I_{\rm H}^{2}+I_{\rm H}L_{r}+L_{r}N_{r})), respectively.
Computing αr,i(t),α~r,i(t)\alpha_{r,i}^{(t)},\widetilde{\alpha}_{r,i}^{(t)} takes 𝒪⁡(Lr3+Mr3+Nr3)\mathcal{O}(L_{r}^{3}+M_{r}^{3}+N_{r}^{3}) flops. In summary, the per-iteration complexity is
𝒪⁡(IM3​(IH+Lr))\mathcal{O}(I_{\rm M}^{3}(I_{\rm H}+L_{r})).

<!-- END-OF-PAPER:ee1cc3423c45 -->
