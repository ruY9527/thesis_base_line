Unsupervised Hyperspectral Image Super-Resolution via Self-Supervised Modality Decoupling

    ![](/static/base/1.0.1/images/icons/smileybones-small.svg)
    arXiv is now an independent nonprofit!
    Learn more
    ×

    License: arXiv.org perpetual non-exclusive license

arXiv:2412.04802v4 [eess.IV] 24 Mar 2026

# Unsupervised Hyperspectral Image Super-Resolution via Self-Supervised Modality Decoupling

Songcheng Du

Email: dusongcheng@mail.nwpu.edu.cn

Affiliation: School of Computer Science, Northwestern Polytechnical University, Xi’an, 100190, China

  
Yang Zou

Email: archerv2@mail.nwpu.edu.cn

Affiliation: School of Computer Science, Northwestern Polytechnical University, Xi’an, 100190, China

  
Zixu Wang

Email: wangzixu0827@mail.nwpu.edu.cn

Affiliation: School of Computer Science, Northwestern Polytechnical University, Xi’an, 100190, China

  
Xingyuan Li

Email: xingyuan_lxy@163.com

Affiliation: School of Software Technology, Dalian University of Technology, Dalian, 116086, China

  
Ying Li

Email: lybyp@nwpu.edu.cn

Affiliation: School of Computer Science, Northwestern Polytechnical University, Xi’an, 100190, China

  
Changjing Shang

Email: cns@aber.ac.uk

Affiliation: Institute of Mathematics, Physics and Computer Science, Aberystwyth University, Aberystwyth SY23 3DB, U.K

  
Qiang Shen

Email: qqs@aber.ac.uk

Affiliation: Institute of Mathematics, Physics and Computer Science, Aberystwyth University, Aberystwyth SY23 3DB, U.K

###### Abstract

Fusion-based hyperspectral image super-resolution aims to fuse low-resolution hyperspectral images (LR-HSIs) and high-resolution multispectral images (HR-MSIs) to reconstruct high spatial and high spectral resolution images. Current methods typically apply direct fusion from the two modalities without effective supervision, leading to an incomplete perception of deep modality-complementary information and a limited understanding of inter-modality correlations. To address these issues, we propose a simple yet effective solution for unsupervised HMIF, revealing that modality decoupling is key to improving fusion performance. Specifically, we propose an end-to-end self-supervised Modality-Decoupled Spatial-Spectral Fusion (MossFuse) framework that decouples shared and complementary information across modalities and aggregates a concise representation of both LR-HSIs and HR-MSIs to reduce modality redundancy. Also, we introduce the subspace clustering loss as a clear guide to decouple modality-shared features from modality-complementary ones. Systematic experiments over multiple datasets demonstrate that our simple and effective approach consistently outperforms the existing HMIF methods while requiring considerably fewer parameters with reduced inference time.
The source code is in MossFuse.

###### keywords

Hyperspectral image super-resolution, Hyperspectral and multispectral image fusion, Modality decoupling, Self-supervised learning

††equal-contributors: These authors contributed equally to this work.††equal-contributors: These authors contributed equally to this work.

![Refer to caption](2412.04802v4/first.png)

*Figure 1: Algorithmic core difference with existing HMIF methods (I, II, and III) and efficiency comparison of our proposed MossFuse. We reveal that the modality decoupling is essential for unsupervised HMIF. Our simple solution significantly outperforms the existing HMIF methods while requiring considerably fewer parameters and reducing inference time.*

## 1 Introduction

Hyperspectral imaging achieves continuous spectral coverage with narrow sampling intervals (e.g., 10 nm), providing detailed reflectance signatures for precise material discrimination Yu et al. (2024); Liang et al. (2025). This unique capability makes it particularly valuable in domains such as environmental monitoring and earth observation Wang et al. (2023); Lin et al. (2023); Li et al. (2025d); Zhang et al. (2023). In the real world, however, optical systems typically offer either high spatial resolution with coarse spectral information (i.e., HR-MSI) or a greater number of spectral bands at lower spatial resolution (i.e., LR-HSI) due to the physical limitations of sensors Li et al. (2022b); Li et al. (2022a); Li et al. (2025b). To improve the usability of hyperspectral images (HSIs), the fusion of observed LR-HSI and HR-MSI has emerged as an efficient and promising method, drawing considerable attention Li et al. (2022d); Li et al. (2022c); Wu et al. (2023); Yang et al. (2025).

Without losing generality, HR-MSI and LR-HSI can be considered as the spatial and spectral degradation of a high-resolution hyperspectral image (HR-HSI), respectively Wang et al. (2020b); Hu et al. (2024). The related observation models can be mathematically expressed as Y=X​R\text{Y}=\text{X}R and x=C​Xx=C\text{X}, where X∈ℝH×W×B\text{X}\in\mathbb{R}^{H\times W\times B}, Y∈ℝH×W×b\text{Y}\in\mathbb{R}^{H\times W\times b} and x∈ℝh×w×Bx\in\mathbb{R}^{h\times w\times B} represent the target HR-HSI, the HR-MSI and the LR-HSI, respectively.
Here, H⁡(h)H(h), W⁡(w)W(w), and B⁡(b)B(b) denote the height, width, and number of bands of the images, respectively, where h≪Hh\ll H, w≪Ww\ll W, b≪Bb\ll B.
The matrix R∈ℝB×bR\in\mathbb{R}^{B\times b} represents the spectral response function (SRF), and C∈ℝh​w×H​WC\in\mathbb{R}^{hw\times HW} represents the point spread function (PSF), which is often modeled as a convolution with a blur kernel KK followed by a spatial downsampling operation.

Recently, numerous methods have been proposed to address the hyperspectral and multispectral image fusion (HMIF) problem Li et al. (2023c); Gao et al. (2023), which can be mainly categorized into three types, as illustrated in Fig. 1. In (I), the matrix factorization-based techniques facilitate information fusion through linear transformations Kawakami et al. (2011); Kanatsoulis et al. (2018). In (II), the coupled branch architectures address the information deficiency in each modality by employing a multi-step cross-fusion process, which independently enhances both modalities Yao et al. (2020). In (III), the repetitive integration approach combines observed LR-HSI and HR-MSI with a main branch to prevent information loss Dong et al. (2021); Xie et al. (2019); Li et al. (2023d); Xu et al. (2025). These architectures generally require the independent extraction of spatial and spectral feature representations from HR-MSI and LR-HSI prior to fusion. However, two major challenges persist: 1) Limited exploitation of the deep modality-complementary information, where the direct fusion of two modalities without effective supervision fails to adequately capture the deep modality-complementary information across modalities. This leads to redundant features and the deterioration of valuable modality-specific characteristics, ultimately yielding suboptimal fusion results. 2) Reduced efficiency, as the repeated extraction of redundant features increases computational costs. To overcome these inherent limitations, we develop an alternative approach that focuses on learning only the essential modality-complementary information through modality decoupling rather than relying on direct fusion.

Specifically, as shown in Fig. 1 (Decomposition), the LR-MSI can be obtained from both spatial degradation on the given HR-MSI Y and spectral degradation on the LR-HSI xx. Motivated by this degradation consistency observation, we note a key insight: HR-MSIs and LR-HSIs inherently contain both shared representations (LR-MSIs) and mutually exclusive complementary features in the latent space. In particular, we identify two distinct complementary components: 1) spatial high-frequency details lost during HR-MSI’s spatial degradation process and 2) fine-grained spectral signatures discarded in LR-HSI’s spectral dimensionality reduction. Building on this, we propose an end-to-end Modality-Decoupled Spatial-Spectral Fusion (MossFuse) framework that reconstructs high-quality HR-HSIs with significantly fewer parameters and reduced inference time.

We first introduce a coarse-to-fine multi-modality decoupling process to decompose each given LR-HSI and HR-MSI into modality-complementary and modality-shared components within a latent space. To guide this decomposition, we devise a novel subspace clustering loss that effectively disentangles modality-shared and complementary information. Furthermore, we establish a self-supervised constraint mechanism that reinforces the decoupling process via a reversible transformation framework, enhancing the completeness and fidelity of the decoupled representations. Building on the inherent degradation properties of LR-HSIs and HR-MSIs, we introduce a dedicated degradation estimation module that accurately infers degradation parameters, significantly boosting the framework’s robustness in unsupervised learning scenarios. Ultimately, we implement an adaptive modality aggregation strategy that effectively fuses the refined subspace representations, enabling accurate reconstruction of HR-HSIs.

In contrast to existing complex architectures, our work shows that effective fusion can be achieved through the integration of thoroughly decoupled modality-shared LR-MSIs and modality-complementary spatial and spectral information. This is achieved without relying on excessively deep networks or large parameter settings, yielding high-fidelity HR-HSIs with significantly improved efficiency in both computational complexity and inference speed. To this end, we propose MossFuse for unsupervised HMIF tasks, revealing the significance of modality decomposition.

Our contributions are summarized as follows:

- •

We present a simple and effective solution for HMIF, demonstrating that effective fusion can be achieved through the integration of precisely decoupled modality components: the shared LR-MSI representation and complementary spatial and spectral information. To the best of our knowledge, this is the first work to address HMIF from a modality decomposition perspective.

- •

We introduce a specialized subspace clustering loss to enforce the decomposition process, effectively aggregating the shared LR-MSI components while discerning them from the modality-complementary spatial and spectral components.

- •

We develop an end-to-end self-supervised modality-decoupled spatial-spectral fusion framework that decouples the shared and complementary information across modalities while aggregating the concise representation of the HSIs and MSIs to reduce the modality redundancy.

- •

We present systematic experimental results validating our discovery and approach. Our simple and effective method outperforms the existing HMIF techniques across five popularly used datasets while requiring considerably fewer parameters with reduced inference time.

## 2 Related Work

### 2.1 Linear Decomposition based Fusion

Fusion methods using linear decomposition typically decompose high-resolution multispectral images (HR-MSIs) and low-resolution hyperspectral images (LR-HSIs) to extract spatial and spectral features Han et al. (2020); Fu et al. (2021); Li et al. (2023e); Ren et al. (2021); Zou et al. (2024a); Li et al. (2025c). These methods can be broadly classified into matrix and tensor factorization paradigms Li et al. (2018); Xu et al. (2020); Ye et al. (2023). The matrix factorization method assumes that each pixel in an HSI is a linear combination of a finite set of reflectances. Kawakami et al. Kawakami et al. (2011) initially applied an unmixing technique to estimate basis reflectance spectra from LR-HSI and subsequently used the coefficients obtained from HR-MSI to achieve the desired outcomes based on a sparse prior.
Yokoya et al. Yokoya et al. (2011) introduced coupled non-negative matrix factorization (CNMF) to alternately estimate the spectral signatures and their corresponding coefficients.
Unlike matrix factorization, tensor factorization directly decomposes an HR-HSI scene into a core tensor and dictionaries corresponding to the spatial and spectral dimensions. For instance, Dian et al. Dian et al. (2017) and Li et al. Li et al. (2018) employed Tucker decomposition to capture non-local and coupled structural information, respectively. Kanatsoulis et al. Kanatsoulis et al. (2018) applied Canonical Polyadic (CP) decomposition to decompose the HR-HSI, estimating each factor matrix via least squares.
Dian et al. Dian et al. (2023b) further integrate CP decomposition with deep neural networks by proposing an adaptive low-rank prior learning module, which leverages tensor factorization to enforce structured low-rank constraints while enabling efficient feature learning in a 1D spectral space.
Wu et al. Wu et al. (2024b) proposed a cross-domain tensor singular value decomposition (t-SVD) framework to more effectively exploit the underlying low-rank structure of hyperspectral data.
Zhang et al. Zhang et al. (2018) proposed a cluster-sparse field-based HSI reconstruction framework that explicitly models intrinsic priors in both spectral and spatial domains.
Fu et al. Fu et al. (2017) leveraged spectral correlations and spatial non-local self-similarity in degraded HSIs to learn an adaptive spatial-spectral dictionary.
Dian et al. Dian et al. (2024a) introduce a generalized tensor nuclear norm (GTNN)-based fusion method that leverages clustering to enhance structural modeling of 3D HSI tensors.
Moreover, Akhtar et al. Akhtar et al. (2015) proposed a Bayesian sparse coding-based approach, where the HR-HSI is reconstructed by performing a linear combination of two key elements: a spectral dictionary, learned from the LR-HSI, and sparse coefficients, estimated from the HR-MSI.
In general, the effectiveness of such methods heavily relies on the accuracy of prior knowledge about the observed scene. However, in real-world scenarios, multiple light interactions with surfaces complicate linear representations, causing the solution to diverge from the true one.

### 2.2 Deep Learning based Fusion

Deep learning techniques, known for their ability to automatically capture the underlying structure of data Li et al. (2024b); Li et al. (2023a), have inspired neural network-based fusion methods Liu et al. (2024); Li et al. (2025e); Li et al. (2024c); Wang et al. (2024a); Zou et al. (2024b). Like linear decomposition approaches, many recent deep learning-based networks use a dual-branch autoencoder architecture to separately extract spatial and spectral information Yao et al. (2020).
Qu et al. Qu et al. (2018) pioneered a deep learning-based unsupervised fusion network that incorporates a Dirichlet distribution-induced layer within a multi-stage alternating optimization process.
Yao et al. Yao et al. (2020) presented a network with improved interpretability by incorporating spectral unmixing theory Keshava and Mustard (2002) into a coupled-autoencoder framework, using cross-fusion layers to exchange spatial and spectral information.
Liu et al. Liu et al. (2025b) proposed a low-rank Transformer network for spatial-spectral fusion, which effectively exploits both the spatial details and spectral characteristics embedded in MSI and LR-HSI.
In addition, Dian et al. Dian et al. (2023b) propose an imaging model-guided optimization framework, which integrates physical acquisition priors with learned representations to regularize the reconstruction process by exploiting the intrinsic spatial and spectral priors of HSI.
Further, to address the zero-shot hyperspectral fusion problem, Dian et al. Dian et al. (2023a) propose a novel and accurate method for quantitatively estimating the spectral and spatial response functions of imaging sensors degradation priors that enable faithful reconstruction without requiring ground-truth data.
To reinforce adaptability to different samples, more recent methods incorporate Test-Time Adaptation (TTA) Wang et al. (2020a), fine-tuning the network on test data. Zhang et al. Zhang et al. (2024a) proposed an unsupervised adaptation learning (UAL) framework that first learned a general image using deep networks and then adapted it to a specific HSI. Guo et al. Guo et al. (2023) proposed a coordination optimization framework to estimate the desired HSIs and their degradation Du et al. (2023), along with a partial fine-tuning strategy to cut computational costs, despite being time-consuming and having relatively poor generalization. To investigate the correlation between spatial and spectral information, Wu et al. Wu et al. (2024a) proposed a dual-branch super-resolution (SR) framework for HR-HSI reconstruction, where the interaction between the two branches serves as mutual constraints to enhance reconstruction quality. Additionally, Liu et al. proposed the SSMSFuse net Liu et al. (2025a) that coupled information at multiple scales with a two-branch network (Spa-Net and Spe-Net) for spectral and spatial coupling.
Moreover, with the popularity of the Mamba structure in the field of computer vision, Zhu et al. Zhu et al. (2025) and Xiao et al. Xiao et al. (2025) propose MCIFNet and SINet, respectively, which both model HSI/MSI fusion using continuous implicit networks and show robust long-range modeling.
More recently, with a focus on practical deployment, Li et al. Li et al. (2025a); Li et al. (2024a) introduced a zero-shot fusion strategy, which optimizes the network solely based on the observed HR-MSI and LR-HSI pair, without relying on additional training datasets.

## 3 The Proposed Method

![Refer to caption](2412.04802v4/figure/network.png)

*Figure 2: Architecture of MossFuse. (a) Main pipeline, consisting of four key components: I. Multi-Modality Decoupling, II. Self-supervised Constraint, III. Degradation Estimation, and IV. Modality Aggregation; with losses associated with each component being: ℒSC\mathcal{L}_{\text{SC}} (subspace clustering loss), ℒSCT\mathcal{L}_{\text{SCT}} (self-supervised constraint loss), ℒDE\mathcal{L}_{\text{DE}} (degradation estimation loss), and ℒMA\mathcal{L}_{\text{MA}} (modality aggregation loss). (b) and (c) Detailed architecture of Spatial- and Spectral-Aware Aggregation Blocks, respectively.*

### 3.1 Motivation

The fusion of a high-resolution multispectral image (HR-MSI) Y∈ℝH×W×b\text{Y}\in\mathbb{R}^{H\times W\times b} and a low-resolution hyperspectral image (LR-HSI) x∈ℝh×w×Bx\in\mathbb{R}^{h\times w\times B} aims to reconstruct the underlying high-resolution hyperspectral image (HR-HSI) X∈ℝH×W×B\text{X}\in\mathbb{R}^{H\times W\times B}. The observed data are generated through a coupled distinct degradation processes:

Y=Dλ​(X),x=Ds​(X),\text{Y}=D_{\lambda}(\text{X}),\quad x=D_{s}(\text{X}),

(1)

where Dλ:ℝB→ℝbD_{\lambda}:\mathbb{R}^{B}\to\mathbb{R}^{b} denotes spectral degradation (e.g., band integration via sensor spectral response functions), and Ds:ℝH×W→ℝh×wD_{s}:\mathbb{R}^{H\times W}\to\mathbb{R}^{h\times w} denotes spatial degradation (e.g., blur followed by downsampling).

A key observation arises from applying consistent degradations across modalities. Specifically, when we apply spatial degradation DsD_{s} to Y and spectral degradation DλD_{\lambda} to xx, both yield the same low-resolution multispectral image (LR-MSI):

y:=Ds​(Y)=Ds​(Dλ​(X))=Dλ​(Ds​(X))=Dλ​(x),y:=D_{s}(\text{Y})=D_{s}(D_{\lambda}(\text{X}))=D_{\lambda}(D_{s}(\text{X}))=D_{\lambda}(x),

(2)

considering the commutativity of DsD_{s} and DλD_{\lambda} Dian et al. (2023a). This consistency implies that Y and xx share a common latent subspace corresponding precisely to yy, which we define as the modality-shared representation, denoted FSF^{S}.

Beyond this shared component, each modality also contains unique, complementary information: The HR-MSI Y provides high-frequency spatial details absent in xx; The LR-HSI xx preserves fine-grained spectral signatures lost in Y.

We formalize these as modality-complementary representations, denoted FYCF_{\text{Y}}^{C} and FxCF_{x}^{C}, respectively. Formally:

Y↦FS⊕FYC,x↦FS⊕FxC,\text{Y}\mapsto F^{S}\oplus F_{\text{Y}}^{C},x\mapsto F^{S}\oplus F_{x}^{C},

(3)

where the shared component FSF^{S} acts as a structural bridge between modalities, while the complementary components encode discriminative attributes essential for faithful reconstruction.

However, current HMIF methods often rely on simplistic fusion strategies that directly merge different modalities, failing to capture deep cross-modality complementary relationships. This leads to two key issues: (1) feature redundancy from overlapping representations and (2) loss of modality-specific discriminative attributes, such as spectral fidelity or spatial sharpness. These shortcomings reduce fusion efficacy and output quality. To address this, we propose a modality-decoupling paradigm that selectively learns essential complementary features while eliminating redundant information.

In the following, we will elaborate on our methodology and implementation details.

### 3.2 Method Overview

As illustrated in Fig. 2(a), the main pipeline of our proposed modality-decoupled spatial-spectral fusion framework primarily integrates four key components: multi-modality decoupling, self-supervised constraint, degradation estimation, and modality aggregation, all of which work collaboratively to effectively generate high-quality spatial-spectral outputs.

Given paired inputs Y and xx, we first employ a multi-modality decoupling process to decouple them into modality-shared and modality-complementary components in the latent subspace, guided by the subspace clustering loss ℒSC\mathcal{L}_{\text{SC}}. Then, to reinforce the completeness and fidelity of the decoupled representations, we implement a self-supervised learning strategy that incorporates spatial- and spectral-aware aggregation operations during the self-supervised constraint process. Furthermore, we estimate spatial and spectral degradation parameters based on the degradation models of HR-MSI and LR-HSI in the degradation estimation process. Finally, once the modality-shared and modality-complementary components are well separated and clustered, we reconstruct the desired HR-HSI X^\hat{\text{X}} by aggregating information from two input modalities through the modality aggregation process.

### 3.3 Multi-Modality Decoupling

We design a multi‑modality decoupling process to separate the LR‑HSI (xx) and HR‑MSI (YY) into modality‑shared and modality‑complementary components within a latent space. First, both inputs are aligned spatially by upsampling xx to match the size of Y. They are then passed through base encoders (comprising four spectral‑wise Transformer (SWT) blocks Zamir et al. (2022)) to obtain shallow features FYBF^{B}_{\text{Y}} and FxBF^{B}_{x}.

Further, we design two couples of modality‑shared and modality‑complementary encoders to decouple the base features FYBF^{B}_{\text{Y}} and FxBF^{B}_{x} into shared and complementary components. The shared encoders produce FYSF^{S}_{\text{Y}} and FxSF^{S}_{x}, capturing information shared to both modalities, while the complementary encoders yield FYCF^{C}_{\text{Y}} (spatially complementary details) and FxCF^{C}_{x} (spectrally complementary signatures). Formally, this decomposition can be expressed as:

FYB=FYS⊕FYC,FxB=FxS⊕FxC.F^{B}_{\text{Y}}=F^{S}_{\text{Y}}\oplus F^{C}_{\text{Y}},F^{B}_{x}=F^{S}_{x}\oplus F^{C}_{x}.

(4)

To extract spatially modality-complementary information, which captures degraded spatial details with up to ×32 resolution differences between LR-MSI and HR-MSI, we design the spatial modality-complementary encoder with three Large Kernel CNN (LK-CNN) blocks Zhang et al. (2024b). These blocks leverage innovative kernel configurations to balance feature extraction efficiency and extended receptive field capture.
For spectrally modality-complementary information, which contains high-resolution spectral cues absent in HR-MSI, we use an SWT block to extract residual spectral details from LR-HSI (xx).
To capture the modality-shared information, which reflects the common latent structure across both inputs—spatially degraded HR-MSI and spectrally degraded LR-HSI, we adopt modality-shared encoders that integrate both LK-CNN and SWT blocks. This joint structure is designed to extract spatial and spectral attributes simultaneously, enabling accurate alignment in the shared latent LR-MSI space.

### 3.4 Self-supervised Constraint

Considering simply decoupling and clustering of the subspace features decoupled from different modalities may compromise the completeness and fidelity of the resulting representations. To address this, we introduce a self-supervised learning strategy that leverages feature aggregation operators to preserve both the integrity and discriminative quality of the decoupled subspace features.

Specifically, to constrain the two modality-shared representations FYSF^{S}_{\text{Y}} and FxSF^{S}_{x}, we concatenate them channel-wise and apply a 1×11\times 1 convolution, yielding the representation FyF_{y} of the LR-MSI.

Also, we employ the spatial-aware aggregation block ΦθYs​p​a​(⋅,⋅)\Phi^{spa}_{\theta_{\text{Y}}}(\cdot,\cdot) and the spectral-aware aggregation block Φθxs​p​e​(⋅,⋅)\Phi^{spe}_{\theta_{x}}(\cdot,\cdot) to recover the original HR-MSI Y and LR-HSI xx from the decoupled representations [FYCF^{C}_{\text{Y}}, FYSF^{S}_{\text{Y}}], and [FxCF^{C}_{x}, FxSF^{S}_{x}], respectively. Formally:

FY=ΦθYs​p​a​(FYC,FYS),Fx=Φθxs​p​e​(FxC,FxS).F_{\text{Y}}=\Phi^{spa}_{\theta_{\text{Y}}}(F^{C}_{\text{Y}},F^{S}_{\text{Y}}),\quad F_{x}=\Phi^{spe}_{\theta_{x}}(F^{C}_{x},F^{S}_{x}).

(5)

Finally, the three aggregated representations are decoded by their respective decoders 𝒟θi​(⋅)\mathcal{D}_{\theta_{i}}(\cdot), i∈{Y,x,y}i\in\{\text{Y},x,y\}, and the reconstructed HR-MSI(Y^\hat{\text{Y}}), LR-HSI(x^\hat{x}), and LR-MSI(y^\hat{y}) can be formulated as follows:

Y^=𝒟θY​(FY),x^=𝒟θx​(Fx),y^=𝒟θy​(Fy).\hat{\text{Y}}=\mathcal{D}_{\theta_{\text{Y}}}(F_{\text{\text{Y}}}),\hat{x}=\mathcal{D}_{\theta_{x}}(F_{x}),\hat{y}=\mathcal{D}_{\theta_{y}}(F_{y}).

(6)

Thus, the decoupled process can be well constrained with the two inputs as references. More details are provided in Section 3.7.

Note that both the spatial- and spectral-aware aggregation blocks are designed with two parallel branches: a convolutional branch that injects the modality-complementary representations and a spectral-wise self-attention (SWSA) branch that injects the modality-shared representations. These branches are integrated via a cross-external gated attention mechanism, which adaptively modulates the fusion of two types of features by dynamically adjusting feature importance across spatial and channel dimensions, respectively, enabling efficient and comprehensive feature fusion. The details of these two blocks are as follows:

Spatial-Aware Aggregation Block.
As illustrated in Fig. 2(b), our dual-branch block operates on two inputs: the modality-shared feature FSF^{S} and the complementary spatial feature FYCF^{C}_{\text{Y}}. The SWSA branch generates query-key-value triplets (QSQ^{S},KSK^{S},VSV^{S}) through linear projections of FSF^{S}, while the convolutional branch processes FYCF^{C}_{\text{Y}} with depth-wise convolution (DW-Conv) to produce base feature FbCF^{C}_{b}. To enable feature interaction, we employ a spatial attention (SA) mechanism Hu et al. (2018); Chen et al. (2023) to extract an attention map Fs​aC∈ℝH×W×1F^{C}_{sa}\in\mathbb{R}^{H\times W\times 1} from FbCF^{C}_{b}. This spatial guidance is then fused with the SWSA branch through element-wise multiplication:

QS​C=S​A​(FbC)⊙QS,Q^{SC}=SA(F_{b}^{C})\odot Q^{S},

(7)

where QS​CQ^{SC} represents the spatially refined query item, S​A​(⋅)SA(\cdot) denotes the spatial attention operation and ⊙\odot denotes the element-wise multiplication operation. The final output of the SWSA module is computed via scaled element-wise multiplication attention with learnable temperature dd. Formally:

FS′=softmax​(QS​C​(KS)Td)​VS.F^{S^{\prime}}=\text{softmax}\left(\frac{Q^{SC}(K^{S})^{T}}{d}\right)V^{S}.

(8)

For comprehensive feature integration, the transformer branch generates channel guidance through a channel attention (CA) mechanism Hu et al. (2018). Particularly, we derive a channel attention map Fc​aS∈ℝ1×1×CF^{S}_{ca}\in\mathbb{R}^{1\times 1\times C} from the enhanced feature FS′F^{S^{\prime}}, where CC denotes the channel dimension. This attention map is then leveraged to recalibrate the convolutional branch via channel-wise multiplication dynamically: FC′=C​A​(FS′)⊙FbCF^{C^{\prime}}=CA(F^{S^{\prime}})\odot F^{C}_{b}, where C​A​(⋅)CA(\cdot) denotes the channel attention operation. The final fused representation FfF^{f} integrates cross-dimension information through residual summation followed by linear projection: Ff=L​i​n​e​a​r​(FS′+FC′)F^{f}=Linear(F^{S^{\prime}}+F^{C^{\prime}}).

Spectral-Aware Aggregation Block. As depicted in Fig. 2(c), our architecture maintains dual-branch design, consistent with the Spatial-Aware Aggregation Block: The transformer branch converts modality-shared feature FSF^{S} into QSQ^{S}, KSK^{S}, and VSV^{S} via linear projections, while the convolutional branch processes modality-complementary spectral feature FYCF^{C}_{\text{Y}} with DW-Conv to obtain spectral base feature FbCF^{C}_{b}.

To establish bidirectional spectral-spatial modulation, we first leverage channel attention (CA) Hu et al. (2018) to extract spectral guidance from FbCF^{C}_{b} and interact with the SWSA branch:

QS​C=C​A​(FbC)⊙QS,Q^{SC}=CA(F_{b}^{C})\odot Q^{S},

(9)

where QS​CQ^{SC} denotes the spectrally refined query. To reinforce the convolutional branch, spatial attention (SA) is applied to the SWSA branch output FS′F^{S^{\prime}}, followed by interaction with the convolutional features FbCF_{b}^{C}. Formally:

FC′=S​A​(FS′)⊙FbC,F^{C^{\prime}}=SA(F^{S^{\prime}})\odot F_{b}^{C},

(10)

where FS′=softmax​(QS​C​(KS)Td)​VS.F^{S^{\prime}}=\text{softmax}\left(\frac{Q^{SC}(K^{S})^{T}}{d}\right)V^{S}.

The final fused feature combines both branches through residual summation and linear projection:

Ff=Linear​(FS′+FC′).F^{f}=\text{Linear}(F^{S^{\prime}}+F^{C^{\prime}}).

(11)

### 3.5 Degradation Estimation

To alleviate the ill-posed nature of unsupervised HMIF task, we establish physics-grounded degradation models based on the degradation representations of HR-MSIs and LR-HSIs.

Spatial Degradation Estimation.
Consensus from recent blind super-resolution studies Wang et al. (2024b); Gu et al. (2019) indicates that real-world spatial degradation can be effectively modeled by anisotropic Gaussian kernels (AGK). Based on this, we parameterize the point spread function (PSF) CC through three parameters: Eigenvalues λ1\lambda_{1} and λ2\lambda_{2} controlling kernel elongation, and Rotation angle θK\theta_{K} determining orientation. The covariance matrix Σ∈ℝ2×2\Sigma\in\mathbb{R}^{2\times 2} is constructed via eigenvalue decomposition:

Σ=R⁡(θK)⋅diag​(λ1,λ2)⋅R​(θK)T,\Sigma=R(\theta_{K})\cdot\text{diag}(\lambda_{1},\lambda_{2})\cdot R(\theta_{K})^{T},

(12)

where R⁡(θK)=[cos⁡θK−sin⁡θKsin⁡θKcos⁡θK]R(\theta_{K})=\begin{bmatrix}\cos\theta_{K}&-\sin\theta_{K}\\
\sin\theta_{K}&\cos\theta_{K}\end{bmatrix} is the rotation matrix. The final blur kernel KK is then generated through the standard 2D Gaussian formulation K∝exp⁡(−12​𝐱T​Σ−1​𝐱)K\propto\exp\left(-\frac{1}{2}\mathbf{x}^{T}\Sigma^{-1}\mathbf{x}\right).

Spectral Degradation Estimation.
Considering spectral degradation, the spectral response function (SRF) characterizes the sensitivity of a sensor at specific wavelengths, effectively representing an integration operation across the spectral channel dimension. Due to the inherent complexity of material properties, accurately modeling SRFs using simple mathematical formulations remains a significant challenge, as highlighted in recent studies Liu et al. (2022); Lian et al. (2024); Li et al. (2024a); Li et al. (2025a). Given that SRFs can be discretized and expressed as a weighted summation of spectral curves, we propose to model the SRF using a 1×11\times 1 convolutional operation, which inherently incorporates various physical constraints.

Specifically, to ensure model robustness, we employ a clamped activation function during each iteration to maintain the non-negativity of parameters. Furthermore, a normalization operator is applied to enforce the sum-to-one constraint across the spectral dimension, ensuring the SRF adheres to the required physical properties.

### 3.6 Modality Aggregation

With the modality-shared and modality-complementary components effectively and accurately decoupled from the HR-MSI and LR-HSI, we proceed to the modality aggregation process to reconstruct the desired HR-HSI (X^\hat{\text{X}}) by integrating these components. Starting with the modality-shared information, we progressively incorporate complementary spatial and spectral information. This process refines cross-attention in the spatial- and spectral-aware aggregation blocks using gate-like mechanisms and residual connections.

Recall that FYSF^{S}_{\text{Y}} and FxSF^{S}_{x} represent the same modality information, we first combine these shared components and aggregate the spatial complementary information from FYCF^{C}_{\text{Y}} through the spatial-aware aggregation block, denoted as ΦθXs​p​a​(⋅,⋅)\Phi^{spa}_{\theta_{\text{X}}}(\cdot,\cdot). Following this, the spectral complementary information FxCF^{C}_{x} is integrated using the spectral-modality aggregation block, denoted as ΦθXs​p​e​(⋅,⋅)\Phi^{spe}_{\theta_{\text{X}}}(\cdot,\cdot). Thus, the latent feature of the HR-HSI can be formulated as:

FX=ΦθXs​p​e​(ΦθXs​p​a​(FYS+FxS,FYC),FxC).F_{\text{X}}=\Phi^{spe}_{\theta_{\text{X}}}(\Phi^{spa}_{\theta_{\text{X}}}(F^{S}_{\text{Y}}+F^{S}_{x},F^{C}_{\text{Y}}),F^{C}_{x}).

(13)

Ultimately, the desired HR-HSI (X^\hat{\text{X}}) is reconstructed via the decoder 𝒟θX​(⋅)\mathcal{D}_{\theta_{\text{X}}}(\cdot), which incorporates four SWT blocks. Formally, this can be expressed as: X^=𝒟θX​(FX)\hat{\text{X}}=\mathcal{D}_{\theta_{\text{X}}}(F_{\text{X}}).

Datasets
CAVE Yasuma et al. (2010)
Harvard Chakrabarti and Zickler (2011)
NTIRE2018 Arad et al. (2018)

Methods
PSNR ↑\uparrow
SSIM ↑\uparrow
SAM ↓\downarrow
ERGAS ↓\downarrow
PSNR ↑\uparrow
SSIM ↑\uparrow
SAM ↓\downarrow
ERGAS ↓\downarrow
PSNR ↑\uparrow
SSIM ↑\uparrow
SAM ↓\downarrow
ERGAS ↓\downarrow

CSU Lanaras et al. (2015)
29.33
0.852
16.23
1.37
36.72
0.946
11.82
0.84
27.75
0.864
13.20
2.26

NSSR Dong et al. (2016)
28.36
0.910
11.88
1.82
31.23
0.890
12.52
1.54
33.03
0.843
11.43
2.76

HySure Simoes et al. (2014)
27.25
0.858
16.72
4.21
35.36
0.876
8.23
0.86
33.75
0.857
14.84
1.86

MHFNet Xie et al. (2019)
38.45
0.970
6.88
1.87
38.86
0.933
6.72
0.74
43.44
0.989
1.78
0.23

EDIP-NetLi et al. (2025a)
38.77
0.977
6.81
1.01
40.12
0.965
6.79
0.79
40.89
0.979
2.07
0.46

CUCaNetYao et al. (2020)
35.46
0.929
10.43
0.97
37.44
0.940
7.46
0.68
39.66
0.986
4.69
0.49

UAL Zhang et al. (2024a)
37.78
0.976
9.24
0.68
40.92
0.936
7.35
0.64
44.83
0.993
1.32
0.10

DAEM Guo et al. (2023)
39.83
0.978
6.79
0.54
42.87
0.975
4.06
0.46
46.73
0.996
0.96
0.05

Ours
42.15
0.983
6.48
0.42
44.62
0.991
3.76
0.39
48.64
0.999
0.77
0.04

*Table 1: Quantitative comparison of different methods on three synthetic datasets. The best results are in bold, while the second-best results are in underlined.*

### 3.7 Network Optimization

Subspace Clustering Loss. To guide the multi-modality decoupling process, we propose a subspace clustering loss ℒSC\mathcal{L}_{\text{SC}} that encourages the network to align modality‑shared components while simultaneously repelling them from the modality‑complementary spatial and spectral features.

As defined earlier, the multi-modality decoupling process decomposes the HR-MSI (Y) and LR-HSI (xx) into four components: a modality-complementary spatial component FYCF^{C}_{\text{Y}}, a modality-complementary spectral component FxCF^{C}_{x} and two modality-shared components FYSF^{S}_{\text{Y}} and FxSF^{S}_{x}. The contrastive subspace clustering loss ℒSC\mathcal{L}_{\text{SC}} is computed by:

ℒSC=−log⁡f⁡(FYS,FxS)∑i∈{Y,x}f⁡(FiS,FiC)+∑m∈{S,C}f⁡(FYm,Fxm),\displaystyle\scalebox{0.98}{$\mathcal{L}_{\text{SC}}=-\log\frac{f(F^{S}_{\text{Y}},F^{S}_{x})}{\sum_{i\in\{\text{Y},x\}}f(F^{S}_{i},F^{C}_{i})+\sum_{m\in\{\text{S,C}\}}f(F^{m}_{\text{Y}},F^{m}_{x})}$},

(14)

where f⁡(a,b)f(a,b) = expc​o​s​(a,b)\exp^{cos(a,b)} is the latent space similarity metric, measuring the cosine similarity loss between two features aa and bb.

This design is motivated by subspace clustering theory Elhamifar and Vidal (2013) and contrastive representation learning Wang and Isola (2020), both of which effectively promote latent structure separation and alignment. By maximizing the affinity of FYSF^{S}_{\text{Y}} and FxSF^{S}_{x} and minimizing their affinity with modality‑complementary spaces, ℒSC\mathcal{L}_{\text{SC}} encourages a clear decomposition of features into distinct latent subspaces. Through gradient descent optimization of ℒSC\mathcal{L}_{\text{SC}}, redundancy across modalities is significantly reduced, improving the fusion quality.

Self-supervised Constraint Loss.
We propose the self-supervised constraint loss ℒSCT\mathcal{L}_{\text{SCT}} to preserve the integrity and fidelity of the decoupled modality information. We first incorporate two inputs HR-MSI (Y) and LR-HSI (xx) as supervisory items to form the ℒSCT1\mathcal{L}_{\text{SCT1}}, ensuring the reversibility of the separated modalities. Formally:

ℒSCT1=‖Y−Y^‖1+‖x−x^‖1.\mathcal{L}_{\text{SCT1}}=\|\text{Y}-\hat{\text{Y}}\|_{1}+\|x-\hat{x}\|_{1}.

(15)

Moreover, we apply ℒSCT2\mathcal{L}_{\text{SCT2}} on the decoded LR-MSI y^\hat{y} to guide the modality decoupling. Recall that the LR-MSI can be obtained from both spatial degradation on the HR-MSI Y and spectral degradation on the LR-HSI xx, denoted as yYy_{\text{Y}} and yxy_{x}, respectively. Therefore, we can construct the ℒSCT2\mathcal{L}_{\text{SCT2}}, such that:

ℒSCT2=‖yY−y^‖1+‖yx−y^‖1=‖C​Y−y^‖1+‖x​R−y^‖1.\begin{split}\mathcal{L}_{\text{SCT2}}&=\|y_{\text{Y}}-\hat{y}\|_{1}+\|y_{x}-\hat{y}\|_{1}\\
&=\|C\text{Y}-\hat{y}\|_{1}+\|xR-\hat{y}\|_{1}.\end{split}

(16)

Thus, ℒSCT\mathcal{L}_{\text{SCT}} can be obtained by the combination of the two terms:

ℒSCT=ℒSCT1+ℒSCT2.\mathcal{L}_{\text{SCT}}=\mathcal{L}_{\text{SCT1}}+\mathcal{L}_{\text{SCT2}}.

(17)

Degradation Estimation Loss.
Based on the degradation consistency theory, the LR-MSI can be equivalently derived from either HR-MSI through spatial degradation or LR-HSI through spectral degradation, respectively: yY=C​Y,yx=x​Ry_{\text{Y}}=C\text{Y},y_{x}=xR.

To ensure reliable estimation of spatial and spectral degradation parameters, we introduce the degradation estimation loss ℒDE\mathcal{L}_{\text{DE}}:

ℒDE=‖yY−yx‖1=‖C​Y−x​R‖1.\mathcal{L}_{\text{DE}}=\|y_{\text{Y}}-y_{x}\|_{1}=\|C\text{Y}-xR\|_{1}.

(18)

Modality Aggregation Loss. To aggregate the decoupled modality features and reconstruct the desired HR-HSI X^\hat{\text{X}} in an unsupervised manner, we leverage the HSI degradation model along with the estimated spatial and spectral degradation parameters SRF (RR) and PSF (CC). The modality aggregation loss is defined as:

ℒMA=‖Y−X^​R‖1+‖x−C​X^‖1,\mathcal{L}_{\text{MA}}=\|\text{Y}-\hat{\text{X}}R\|_{1}+\|x-C\hat{\text{X}}\|_{1},

(19)

where ∥⋅∥\|\cdot\| represents the L1 loss, which effectively preserves both spatial structure and spectral fidelity during reconstruction.

Overall, the total training loss of the MossFuse, comprising the aforementioned loss terms, can be formulated as:

ℒtotal=ℒMA+α1​ℒSC+α2​ℒSCT+α3​ℒDE.\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{MA}}+\alpha_{1}\mathcal{L}_{\text{SC}}+\alpha_{2}\mathcal{L}_{\text{SCT}}+\alpha_{3}\mathcal{L}_{\text{DE}}.

(20)

## 4 Experiment

![Refer to caption](2412.04802v4/figure/CAVE_error_.png)

*Figure 3: Visual reconstruction results and error images for 11th band of HR-HSI images from CAVE dataset. Zoomed-in regions are provided to highlight details.*

![Refer to caption](2412.04802v4/figure/Harvard_error_.png)

*Figure 4: Visual reconstruction results and error images for the 11th band of the HR-HSI images from Harvard dataset. Zoomed-in regions are provided to highlight details.*

![Refer to caption](2412.04802v4/figure/NTIRE_error_.png)

*Figure 5: Visual reconstruction results and error images for 11th band of HR-HSI images from NTIRE2018 dataset. Please zoom in to see the details better.*

![Refer to caption](2412.04802v4/figure/CAVE_spectral.png)

*(a) CAVE dataset*

![Refer to caption](2412.04802v4/figure/Harvard_spectral.png)

*(b) Harvard dataset*

![Refer to caption](2412.04802v4/figure/NTIRE_spectral.png)

*(c) NTIRE2018 dataset*

*Figure 6: Comparison of reconstructed spectra curves in (a) CAVE, (b) Harvard, and (c) NTIRE2018 datasets. Zoomed-in regions are provided to highlight details.*

![Refer to caption](2412.04802v4/figure/NCALM_L.png)

*Figure 7: Visual reconstruction results from NCALM (upper part) and WV-2 (bottom part) datasets. We show images with 25-16-7 and 5-3-2 as R-G-B, respectively.*

### 4.1 Experiment Setup

Dataset.
In this study, we conducted comprehensive evaluations using a diverse collection of five commonly used benchmark datasets in this field, comprising three synthetic HSI datasets including CAVE Yasuma et al. (2010), Harvard Chakrabarti and Zickler (2011), and NTIRE2018 Arad et al. (2018), as well as two real remote sensing HSI datasets, including National Center for Airborne Laser Mapping (NCALM) Xu et al. (2019) and Worldview2 (WV-2) Li et al. (2023b), for evaluation.

Specifically, the CAVE dataset includes 32 indoor hyperspectral images, each containing 31 spectral bands covering the visible spectrum from 400 to 700 nm with a spatial resolution of 512 × 512 pixels. The Harvard dataset collects 50 outdoor HSI samples recorded under daylight illumination, each with a high spatial resolution of 1392 × 1040 pixels and comprising 31 spectral bands ranging from 420 nm to 720 nm at 10 nm intervals. The NTIRE 2018 dataset, from the first spectral image super-resolution challenge, includes 54 natural HSIs for training and 5 for testing, each with a spatial resolution of 1392 × 1300 pixels and 31 spectral bands from 400 nm to 700 nm at 10 nm intervals. The NCALM dataset, which was published by the 2018 IEEE GRSS Data Fusion Contest, features a 48-band HSI covering an extensive spectral range from 380 to 1050 nm with a spatial resolution of 1202 × 4172 pixels, accompanied by a 3-band MSI at 24040 × 83440 pixels. The WV-2 dataset, captured by a commercial satellite, includes an 8-band HSI with a spatial resolution of 418 × 685 pixels and a corresponding 3-band MSI at 1677 × 2633 pixels.

For the synthetic datasets (CAVE, Harvard, and NTIRE 2018), we established the following training-testing splits: 20, 30, and 54 pairs of images for training, respectively, with the remainder used for testing. To generate corresponding HR-MSI and LR-HSI pairs, we employed a simulation process using the spectral response function (SRF) and point spread function (PSF) from Jiang et al. (2013). Specifically, the HR-MSI was simulated through spectral downsampling, while the LR-HSI was obtained by applying the selected PSF, followed by spatial downsampling with a scale factor of 32 to the HR-HSI.

For the real-world remote sensing datasets NCALM and WV-2, we adopted the standard evaluation protocol established in Li et al. (2023b), utilizing the top half of both HSI and MSI images for training and the bottom half for testing.

Compared Methods.
To demonstrate the superiority and generality of the proposed approach, we perform extensive comparisons with both traditional optimization-based and recent deep learning-based unsupervised fusion methods.
Specifically, three representative tradition-based algorithms are included: CSU Lanaras et al. (2015), NSSR Dong et al. (2016) and HySure Simoes et al. (2014). In addition, five state-of-the-art (SOTA) unsupervised deep fusion networks are employed for comprehensive comparison, including: MHF Xie et al. (2019), CUCaNet Yao et al. (2020), DEIP-Net Li et al. (2025a), UAL Zhang et al. (2024a), and DAEM Guo et al. (2023).

Evaluation Measurements.
To comprehensively assess the effectiveness and reliability of the proposed method, for the three simulated dataset, four quantitative metrics are adopted Li et al. (2023b): the peak signal-to-noise ratio (PSNR), the structural similarity index (SSIM) Wang et al. (2004), the spectral angle mapper (SAM) Yuhas et al. (1992), and the relative dimensionless global error in synthesis (ERGAS) Wald (2000). For the two real datasets, three no-reference metrics DλD_{\lambda}, DsD_{s}, and HQNR, are employed.

### 4.2 Implementation Details

Our algorithm was implemented using the PyTorch framework, with all experiments conducted on an NVIDIA 4090 GPU. For optimizing the parameters, we utilized the Adam optimizer Kingma and Ba (2014) with settings β1=0.9\beta_{1}=0.9, β2=0.999\beta_{2}=0.999 and ϵ=10−8\epsilon=10^{-8}. To adjust the learning rate, we implemented a cosine annealing decay strategy, reducing it from 10−310^{-3} to 10−610^{-6} with the number of epochs set to 300. Moreover, to accommodate hardware constraints, we preprocessed the training samples by cropping them into 128 ×\times 128. Given the limited scale of available datasets, we applied comprehensive data augmentation techniques, including random rotation and flipping operations on all input pairs. The training process utilized a batch size of 2 across all three datasets to maintain consistency. As for the multi-term loss functions defined in Eq.20, we carefully balanced the contribution of each component by setting the weighting coefficients α1\alpha_{1} to α3\alpha_{3} to 0.1,1, and 1, respectively. This configuration ensures that all loss terms maintain comparable orders of magnitude during optimization, thereby facilitating stable training convergence.
Regarding feature dimensions throughout the network: the input HR-MSI Y∈ℝH×W×b\text{Y}\in\mathbb{R}^{H\times W\times b} and upsampled LR-HSI x∈ℝh×w×Bx\in\mathbb{R}^{h\times w\times B} are processed by two separate base encoders to generate initial features FYB,FxB∈ℝH×W×CF^{B}_{\text{Y}},F^{B}_{x}\in\mathbb{R}^{H\times W\times C}, where C=48C=48 is the hidden channel dimension. Moreover, all subsequent feature maps, including the decoupled modality-shared (FYSF^{S}_{\text{Y}} and FxSF^{S}_{x}) modality-complementary (FYCF^{C}_{\text{Y}} and FxCF^{C}_{x}) representations preserve the spatial resolution and dimensionality H×W×CH\times W\times C. Finally, the aggregated latent feature FXC∈ℝH×W×CF^{C}_{\text{X}}\in\mathbb{R}^{H\times W\times C} is mapped to the target HR-HSI output X^∈ℝH×W×B\hat{\text{X}}\in\mathbb{R}^{H\times W\times B} in the decoder via channel reshaping.

### 4.3 Performance Comparison

Synthetic Dataset.
Our comprehensive evaluation demonstrates the superior performance of MossFuse through both quantitative metrics and qualitative visual analysis across the three synthetic datasets.

The quantitative comparison, presented in Table 1, reveals that MossFuse consistently outperforms all competing methods across three benchmark datasets, CAVE, Harvard, and NTIRE2018.
On the CAVE dataset, our method establishes new performance records with a 2.32 dB PSNR improvement and a 0.31 SAM reduction compared to the strongest baseline, reflecting both enhanced reconstruction fidelity and exceptional spectral preservation. For the Harvard dataset, MossFuse achieves a PSNR of 44.62 (+1.75 dB over the second-best method) and an ERGAS score of 0.39, indicating statistically significant improvements in both spectral consistency and spatial accuracy. Notably, the NTIRE 2018 dataset evaluation reveals MossFuse’s performance with a PSNR of 48.64 and near-perfect SSIM of 0.999—metrics that surpass competing approaches by substantial margins. These cross-dataset results systematically validate our method’s robustness in maintaining critical hyperspectral features while minimizing reconstruction errors across diverse imaging scenarios.
Traditional methods, including CSU Lanaras et al. (2015) and HySure Simoes et al. (2014), exhibit inherent limitations in modeling intricate spectral-spatial interdependencies, primarily stemming from their restricted representational capacity for high-dimensional feature learning.

For qualitative assessment, we provide detailed visual comparisons across all three synthetic datasets in Fig. 3, Fig. 4, and Fig. 5. Each figure presents a dual-view analysis: the top row displays the reconstructed spectral band images, while the bottom row shows the corresponding reconstruction error heatmaps. Two strategically selected regions of interest are magnified for detailed comparison. The visual results clearly demonstrate that MossFuse generates reconstructions that are visually indistinguishable from the ground truth, consistently achieving the lowest reconstruction error across all test cases.

To further assess the spectral fidelity of the reconstructed results, we conducted a detailed spectral signature analysis, as illustrated in Fig. 6. Specifically, we extract normalized spectral intensity curves from representative sample points and compare them across different methods. The results show that MossFuse achieves the closest alignment with the ground truth spectral signatures, exhibiting minimal distortion across wavelengths. This quantitative spectral analysis, combined with the visual results, provides compelling evidence of our method’s effectiveness in preserving both spatial details and spectral accuracy, setting a new benchmark in fusion-based hyperspectral image super-resolution performance.

Datasets
NCALM
WV-2

Methods
DλD_{\lambda} ↓\downarrow
DsD_{s} ↓\downarrow
HQRN ↑\uparrow
DλD_{\lambda} ↓\downarrow
DsD_{s} ↓\downarrow
HQRN ↑\uparrow

MHFNet Xie et al. (2019)
0.056
0.042
0.904
0.036
0.048
0.918

EDIP-NetLi et al. (2025a)
0.047
0.028
0.926
0.029
0.038
0.934

UAL Zhang et al. (2024a)
0.045
0.021
0.935
0.023
0.033
0.945

DAEM Guo et al. (2023)
0.042
0.019
0.940
0.021
0.031
0.949

Ours
0.038
0.016
0.946
0.017
0.028
0.955

*Table 2: Quantitative comparison of different methods on two real datasets. The best results are in bold, while the second-best results are in underlined.*

Methods

FLOPs

(G)

Param

(M)

Training

Time

(min)

Test

Time

(s)

Test

Iterations

MHFNet Xie et al. (2019)
21226
129.4
1340
21.36
1

EDIP-Net Li et al. (2025a)
394.53
3.51
/
1594.60
7000

CUCaNetYao et al. (2020)
7226
2.28
/
1806.18
10000

UAL Zhang et al. (2024a)
3599
7.13
325
28.36
1500

DAEM Guo et al. (2023)
1.67
0.02
119
19.56
250

Ours
279.42
0.55
215
0.15
1

*Table 3: Efficiency comparison across methods, with best and second-best results in bold and underline, respectively.*

![Refer to caption](2412.04802v4/figure/degradation.png)

*Figure 8: Comparison of degradation parameters PSF and SRF estimated by different methods.*

Real Dataset.
For real-world remote sensing datasets such as NCALM and WorldView-2, Table 2 presents a comprehensive comparison between MossFuse and several state-of-the-art methods. Our approach achieves the best performance across all no‑reference metrics, clearly demonstrating its ability to deliver consistent, high‑quality results across diverse real‑world remote sensing scenarios.

To further assess its effectiveness, we conducted qualitative visual evaluations using synthetic pseudo‑RGB images generated from the reconstructed HSIs. As shown in Fig. 7, the results confirm that MossFuse excels in preserving spatial details and maintaining spectral consistency compared to competing methods. In particular, our method delivers sharper edges in architectural elements, better preserves fine textures in complex urban structures, and produces more natural color rendering while effectively suppressing spectral artifacts.

Collectively, these quantitative and qualitative results highlight MossFuse’s superior performance in real-world HSI-MSI fusion. By achieving an optimal balance of spatial detail enhancement and spectral fidelity without requiring ground-truth data, MossFuse proves to be a robust and reliable solution for practical remote sensing applications.

![Refer to caption](2412.04802v4/figure/violin_3.png)

*Figure 9: The distribution of two measurement metrics (PSNR, SAM) for the different methods under various degradations. The black lines and the white tangles denote medium and mean values.*

### 4.4 Algorithmic Analysis

Computational Cost Analysis.
We present a comprehensive comparison of computational performance metrics in Table 3, evaluated on the CAVE dataset. Notably, methods like DAEM Guo et al. (2023) and UAL Zhang et al. (2024a) require sample-specific fine-tuning during inference, involving multiple optimization steps (see Table 3, last column), which substantially increases their actual runtime beyond what single-inference FLOPs suggest and limits their adaptability to dynamic scenarios.

In contrast, MossFuse achieves efficient inference by preserving only the essential components: multi-modality decoupling and modality aggregation processes, allowing high-quality HR-HSI reconstruction in a single-shot inference. This simple but effective architecture significantly enhances computational efficiency compared to methods requiring iterative optimization or fine-tuning during inference, making it suitable for real-world applications where processing time and resource constraints are critical factors.

Degradation Estimation Analysis.
We conduct a comprehensive analysis of degradation parameter estimation by comparing our method with several fully blind approaches, including HySure Simoes et al. (2014) and DAEM Guo et al. (2023), as illustrated in Fig. 8. Our algorithm demonstrates exceptional accuracy in estimating degradation parameters across both spatial and spectral dimensions, achieving near-optimal results.

Moreover, given the diversity of degradation models in real-world scenarios, we further assess the framework’s robustness by evaluating the performance distribution of different algorithms under six groups of degradation conditions Guo et al. (2023) on the CAVE dataset. Fig. 9 presents violin plots of two evaluation metrics, PSNR and SAM, illustrating the reconstruction performance across various degradation settings. These results demonstrate that our algorithm excels not only in accuracy but also in robustness compared to others.

In unsupervised hyperspectral and multispectral image fusion (HMIF) tasks, the absence of ground truth implies that the accuracy of degradation model estimation is crucial for high-fidelity HR-HSI reconstruction. Given the remarkable results achieved, a critical question naturally arises: Is the observed state-of-the-art performance primarily attributed to the degradation estimation module, rather than the core modality decoupling architecture?

To investigate this aspect, we assume knowledge of the degradation model and adapt several state-of-the-art (SOTA) fully supervised fusion methods to operate under the same unsupervised setting for fair comparison.

In this setup, the loss function is constructed with both the HR-MSI and the LR-HSI as dual reconstruction constraints, rather than their original supervised losses. Formally, the loss is defined as:

ℒ=‖Y−X^​R‖1+‖x−C​X^‖1,\mathcal{L}=\|\text{Y}-\hat{\text{X}}R\|_{1}+\|x-C\hat{\text{X}}\|_{1},

(21)

where ∥⋅∥\|\cdot\| represents the L1 norm, X^\hat{\text{X}} is the reconstructed HR-HSI, and the degradation operators RR (spectral downsampling) and CC (spatial downsampling) are fixed and given. Such a self-supervised formulation enables effective model training without requiring access to ground-truth HR-HSI, making it suitable for the unsupervised fusion task.

The adapted methods span a diverse range of recent architectural paradigms: (i) Transformer-based models: 3DT-Net Ma et al. (2023) and DCTransformer Ma et al. (2024) employ transformer blocks to capture long-range spatial–spectral dependencies via self-attention, thus overcoming CNNs’ limited receptive field and enhancing feature representation for hyperspectral reconstruction; (ii) Mamba-based models: SRLF-Net Liu et al. (2025c) adopts a selective re-learning mechanism to refine degraded features, while SINet Xiao et al. (2025) fuses Mamba’s long-range modeling with CNN’s local extraction to adaptively enhance global–local complementarity; and (iii) Frequency-domain-based models: FSDFF Tan et al. (2024) learns frequency correlations to improve spectral fidelity, and FeINFN Liang et al. (2024) jointly models spatial and Fourier domains to enhance high-frequency details and expand the receptive field.

For fair comparison, our MossFuse is trained under the same SRF and PSF degradations. As shown in Table 4, MossFuse surpasses all supervised baselines on the CAVE dataset (PSNR, SSIM, SAM, ERGAS), verifying its superior reconstruction accuracy in an unsupervised setting.

Methods
Deg. Model
PSNR ↑\uparrow
SSIM ↑\uparrow
SAM ↓\downarrow
ERGAS ↓\downarrow

MHFNet Xie et al. (2019)
unknown
38.45
0.970
6.88
1.87

known
39.96
0.978
6.32
1.06

EDIP-Net Li et al. (2025a)
unknown
38.77
0.977
6.81
1.01

known
40.16
0.980
6.60
0.86

CUCaNetYao et al. (2020)
unknown
35.46
0.929
10.43
0.97

known
40.06
0.980
6.83
0.53

UAL Zhang et al. (2024a)
unknown
37.78
0.976
9.24
0.68

known
39.86
0.980
7.24
0.56

DAEM Guo et al. (2023)
unknown
39.83
0.978
6.79
0.54

known
40.98
0.983
6.60
0.48

3DT-Net Ma et al. (2023)
known
40.52
0.980
6.67
0.52

DCTransformer Ma et al. (2024)
known
42.34
0.984
6.52
0.42

SRLF-Net Liu et al. (2025c)
known
42.56
0.983
6.51
0.44

SINet Xiao et al. (2025)
known
41.92
0.979
6.57
0.45

FSDFF Tan et al. (2024)
known
41.86
0.981
6.57
0.48

FeINFN Liang et al. (2024)
known
41.68
0.982
6.60
0.49

Ours
unknown
42.15
0.983
6.48
0.42

known
42.86
0.986
6.43
0.40

*Table 4: Quantitative comparison of various methods on CAVE dataset with known and unknown degradation models with best and second-best results in bold and underlined, respectively.*

![Refer to caption](2412.04802v4/beta.png)

*Figure 10: Performance under mismatched degradation conditions. (a) Degradation with Gaussian noise. The bottom row visualizes the corresponding corrupted degradation functions. For PSF, the left side shows the corrupted version(C′C^{\prime}), while the right side shows the ideal (clean) PSF(CC). For SRF, the solid line represents the corrupted SRF(R′R^{\prime}), whereas the dashed line indicates the ideal SRF(RR). (b) Degradation with Poisson noise. The bottom row visualizes input example corrupted by Poisson noise.*

With access to precise degradation parameters, all unsupervised methods exhibit notable performance improvements, with CUCaNet showing the most significant gains. The two supervised methods yield even better spectral accuracy, likely due to their larger parameter capacity. Nevertheless, our approach consistently outperforms all baselines across four evaluation metrics, affirming the effectiveness and generalizability of the proposed modality decomposition paradigm in unsupervised HMIF tasks.

To further evaluate the robustness and generalization capability of our method under real-world conditions, we investigate how performance is affected when the estimated degradation parameters deviate from the ideal or ground-truth settings. Such deviations can arise from two sources: (1) the inherent limitations of the network in accurately estimating degradation parameters, and (2) real-world deviations from ideal assumptions due to sensor noise, atmospheric interference, system calibration errors, and other factors.

Based on the CAVE dataset, we first simulate these effects by injecting controlled Gaussian noise into both the SRF(RR) and PSF(CC) during the inference phase. This procedure generates HR-MSI and LR-HSI inputs that differ from the training distribution, thereby introducing a domain shift and simulating practical scenarios involving inaccurate or mismatched degradation estimation.

Specifically, given the degradation parameters RR and CC used during training, we introduce two perturbation control factors, γ1\gamma_{1} and γ2\gamma_{2}, to regulate the intensity of additive Gaussian noise applied to SRF and PSF, respectively:

R′=R+𝒩⁡(0,γ12),C′=C+𝒩⁡(0,γ22).R^{\prime}=R+\mathcal{N}(0,\gamma_{1}^{2}),\quad C^{\prime}=C+\mathcal{N}(0,\gamma_{2}^{2}).

(22)

We set γ1∈{0,3∗10−3,6∗10−3,9∗10−3,12∗10−3,15∗10−3}\gamma_{1}\in\{0,3*10^{-3},6*10^{-3},9*10^{-3},12*10^{-3},15*10^{-3}\}, and γ2=γ1/3}\gamma_{2}=\gamma_{1}/3\}, reflecting the fact that SRF and PSF exhibit different sensitivities to noise.

These perturbed degradation parameters R′R^{\prime} and C′C^{\prime} are then used to synthesize test samples, ensuring a domain shift between training and testing. The results are illustrated in Fig. 10 (a). We present the PSNR values of our method (Ours) and a representative baseline (DAEM) under varying noise intensities. The bottom part of the figure displays the corrupted PSF and SRF corresponding to each noise level.

Beyond Gaussian noise, Poisson noise is incorporated to simulate degradation estimation errors arising from inherent sensor noise. Since Poisson noise is signal-dependent, we simulate it by first scaling the normalized input image Y,x∈[0,1]\text{Y},x\in[0,1] by a gain factor δ\delta, then sampling from a Poisson distribution with mean δ⋅Y,δ⋅x\delta\cdot\text{Y},\delta\cdot x, yielding Ynoisy∼𝒫⁡(δ⋅Y)\text{Y}_{\text{noisy}}\sim\mathcal{P}(\delta\cdot\text{Y}) and xnoisy∼𝒫⁡(δ⋅x)x_{\text{noisy}}\sim\mathcal{P}(\delta\cdot x) respectively, and finally rescaling the result back to [0,1][0,1]. We set γ3=1/δ\gamma_{3}=1/\delta, with γ3∈{0,2−13,2−12,2−11,2−10,2−9}\gamma_{3}\in\{0,2^{-13},2^{-12},2^{-11},2^{-10},2^{-9}\}, corresponding to varying noise levels. Smaller values of γ3\gamma_{3} amplify the relative noise intensity, simulating low-light or high-noise acquisition conditions.

As shown in Fig. 10, our method achieves consistently high performance under both Gaussian and Poisson noise conditions. Even in the presence of severe degradation, MossFuse significantly outperforms DAEM, demonstrating strong robustness across diverse and realistic noise environments.

These experiments highlight the robustness of our framework under imperfect degradation estimation and confirm its suitability for real-world deployment.

![Refer to caption](2412.04802v4/pca.png)

*Figure 11: Four decoupled features visualization and distribution after PCA, showing the separation between modality-shared and modality-complementary representations.*

![Refer to caption](2412.04802v4/figure/alpha1.png)

*(a) Performance with different α1\alpha_{1}*

![Refer to caption](2412.04802v4/figure/alpha2.png)

*(b) Performance with different α2\alpha_{2}*

![Refer to caption](2412.04802v4/figure/alpha3.png)

*(c) Performance with different α3\alpha_{3}*

*Figure 12: Comparison of algorithm’s performance with different settings of the training loss weight parameters α1\alpha_{1}, α2\alpha_{2}, and α3\alpha_{3}.*

Design Paradigm Analysis.
Our framework explicitly separates shared and complementary modalities, which adopts a design choice that fundamentally differs from both classical and recent approaches. Traditional linear decomposition methods (e.g., CNMF Yokoya et al. (2011), Tucker Kanatsoulis et al. (2018)) approximate hyperspectral–multispectral fusion under linear mixing or low-rank priors, which, while interpretable, cannot fully capture nonlinear modality interactions.

Model-driven tensor decomposition methods such as deep low-rank tensor Dian et al. (2024b), Generalized Tensor Nuclear Norm (GTNN) Dian et al. (2024a), and tensor-ring/subspace mechanism Xu et al. (2021) leverage structured priors (e.g., low-rankness, factorization) to capture high-order correlations. However, they typically rely on strong assumptions (e.g., fixed ranks, tensor formats), require careful hyperparameter tuning, and are sensitive to model specification, limiting their flexibility and generalization.

Recent learning-based frameworks such as DAEM Guo et al. (2023) and UAL Zhang et al. (2024a) take different routes: DAEM emphasizes iterative degradation modeling and multi-stage decomposition, while UAL relies on unsupervised adaptation and test-time optimization, updating parameters per sample.

However, all these approaches lack an explicit mechanism for structural disentanglement of shared and complementary information. In contrast, MossFuse enforces modality-specific subspaces through subspace clustering and self-supervised reconstruction, enabling efficient, single-shot fusion without per-sample optimization.

Modality Decoupling Capability Analysis.
Our methodology fundamentally decouples the HR-MSI and LR-HSI into two distinct components: modality-shared components FYSF^{S}_{\text{Y}} and FxSF^{S}_{x}, represented by the latent LR-MSI, and modality-complementary components FYCF^{C}_{\text{Y}} and FxCF^{C}_{x}, consisting of spatial and spectral complementary information, respectively. To demonstrate the effectiveness of this decoupling process, we first visualize the above four critical decoupled features and further project them into a low-dimensional space using Principal Component Analysis (PCA), as shown in Fig. 11.

The visualization and PCA results reveal two key observations: (1) The two modality-shared features FYSF^{S}_{\text{Y}} and FxSF^{S}_{x} have nearly identical visual patterns and show highly overlapping distributions in the PCA space, indicating a strong consistency in the decoupling process. Since LR-MSI contains limited redundant information, the extracted shared features tend to form compact and concentrated clusters, reflecting their aligned representation across modalities; (2) The modality-complementary features demonstrate distinct characteristics. The spatial complementary features (FYCF^{C}_{\text{Y}}) contain rich spatial textures with relatively low spectral variation, while the spectral complementary features (FxCF^{C}_{x}) emphasize spectral diversity with less spatial structure. These two types of complementary feature are separated from each other as well as from the shared features within the PCA space, highlighting their individual roles: each of which encodes complex residual information specific to their modality. This naturally results in more dispersed and modality-specific feature distributions.

These findings validate MossFuse’s effectiveness in disentangling and preserving both shared and modality-specific representations.

Hyper-parameter Analysis.

Recall that our training loss is formulated as:

ℒtotal=ℒMA+α1​ℒSC+α2​ℒSCT+α3​ℒDE,\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{MA}}+\alpha_{1}\mathcal{L}_{\text{SC}}+\alpha_{2}\mathcal{L}_{\text{SCT}}+\alpha_{3}\mathcal{L}_{\text{DE}},

(23)

where ℒMA\mathcal{L}_{\text{MA}} denotes the modality aggregation loss (supervises the final reconstruction results), ℒSC\mathcal{L}_{\text{SC}} is the subspace clustering loss, ℒSCT\mathcal{L}_{\text{SCT}} is the self-supervised constraint loss, and ℒDE\mathcal{L}_{\text{DE}} corresponds to the degradation estimation loss. Each loss term is designed to address a specific aspect of the fusion task, and their contributions are balanced by the corresponding weights α1\alpha_{1}, α2\alpha_{2}, and α3\alpha_{3}.

To comprehensively evaluate the sensitivity of the proposed framework to these hyperparameters and to further understand the contribution of each loss component, we perform a series of experiments by individually varying one hyperparameter while keeping the others fixed. The experimental results, summarized in Fig. 12, illustrate how different configurations of loss weighting influence the final fusion performance (PSNR, blue curves) and the corresponding loss values (red curves).

1. Effect of varying α𝟏\mathbf{\alpha_{1}}.
Fig. 12 (a) shows the relationship between the PSNR metric and the subspace clustering loss ℒSC\mathcal{L}_{\text{SC}} under varying values of α1∈{10−5,10−4,10−3,10−2,10−1,100,101}\alpha_{1}\in\{10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},10^{0},10^{1}\}. As α1\alpha_{1} increases, ℒSC\mathcal{L}_{\text{SC}} consistently decreases and eventually stabilizes around 0.5. The PSNR, however, exhibits a bell-shaped curve: it first increases and then decreases, reaching a peak near α1=0.1\alpha_{1}=0.1. This observation can be interpreted as follows: when α1\alpha_{1} is small, the ℒSC\mathcal{L}_{\text{SC}} is insufficiently weighted, rendering it ineffective in guiding the accurate decomposition of modality-shared features (FYSF^{S}_{\text{Y}} and FxSF^{S}_{x}), and modality-complementary features (FYCF^{C}_{\text{Y}}, and FxCF^{C}_{x}). Conversely, when α1=0.1\alpha_{1}=0.1 becomes too large, the overemphasis on ℒSC\mathcal{L}_{\text{SC}} disrupts the optimization balance among different modules, which in turn degrades the overall fusion performance and leads to a drop in PSNR.

2. Effect of varying α𝟐\mathbf{\alpha_{2}}.
The self-supervised constraint loss ℒSCT\mathcal{L}_{\text{SCT}} is designed to preserve the integrity and fidelity of the decoupled modality features. While its effectiveness has been preliminarily verified through ablation (by removing it entirely and partially), we further investigate its influence by varying α2={10−4,0.2,0.5,0.8,1,1.2,1.5}\alpha_{2}=\{10^{-4},0.2,0.5,0.8,1,1.2,1.5\}, as shown in Fig. 12 (b). When α2\alpha_{2} is too small, ℒSCT\mathcal{L}_{\text{SCT}} contributes minimally, preventing the model from fully leveraging the self-supervised constraints, which results in weak feature regularization, suboptimal convergence, and degraded PSNR performance. As α2\alpha_{2} increases, both the loss and the final PSNR gradually stabilize, indicating that moderate values (e.g., around 1.0) provide a good balance between constraint enforcement and overall optimization.

3. Effect of varying α𝟑\mathbf{\alpha_{3}}.
The degradation estimation loss ℒDE\mathcal{L}_{\text{DE}} is leveraged to learn the spatial and spectral degradation parameters and further mitigate the ill-posed nature of unsupervised HMIF. To quantify its impact, we varied α3∈{10−4,0.2,0.5,0.8,1,1.2,1.5}\alpha_{3}\in\{10^{-4},0.2,0.5,0.8,1,1.2,1.5\}, and recorded the corresponding changes in PSNR and ℒDE\mathcal{L}_{\text{DE}}, as shown in Fig. 12 (c).
The results reveal that when α3\alpha_{3} is too small, the model struggles to effectively learn the degradation patterns, resulting in poor convergence and fusion quality. Once α3\alpha_{3} exceeds a certain threshold (e.g., 0.2), ℒDE\mathcal{L}_{\text{DE}} converges, and further weight increases have a negligible impact on performance, indicating a moderate weighting of ℒDE\mathcal{L}_{\text{DE}} is sufficient to achieve optimal results.

Downstream Experiments Analysis. Beyond reconstruction fidelity, the value of HR-HSI lies in its utility for downstream vision tasks. We evaluate MossFuse on representative applications including classification and material identification to validate its practical effectiveness as below.

1. Classification.: To evaluate the benefit of fusion for semantic interpretation, we conduct a classification experiment on the Houston dataset (349×1905 spatial, 144 spectral bands), using SpectralFormer Hong et al. (2021) as the classifier under standard settings. We simulate LR-HSI using applying an 8×8 Gaussian blur (σ\sigma=2) followed by ×4 spatial downsampling, and generate HR-MSI by averaging every 36 adjacent spectral bands. The fused HR-HSI is then input to the classifier and compared against the upsampled LR-HSI (without fusion).

As shown in Table 5, the fused HR-HSI achieves higher accuracies across most categories. The visual comparison provided Fig. 13 further highlights regions where MossFuse enables more accurate classification, particularly in spectrally ambiguous or fine-structured areas. This demonstrates that our method preserves discriminative spectral information beneficial for semantic understanding.

2. Material identification. To assess the fidelity of reconstructed spectra for physical interpretation, we examine material distinguish ability using real and fake peppers from the CAVE dataset (Fig. 14). We extract and compare their spectral signatures from the fused HR-HSI. As shown, the two materials exhibit readily distinguishable reflectance curves, especially beyond 500nm, despite similar visual appearance. This confirms that MossFuse recovers physically meaningful spectral variations, enabling reliable material differentiation.

![Refer to caption](2412.04802v4/figure/classification.png)

*Figure 13: Classification results of LR-HSI (top), predicted HR-HSI (middle), and reference (bottom).*

Category
Accuracy(%)

LR-HSI
Predicted HR-HSI

Healthy grass
85.29
83.29

Stressed grass
97.05
99.06

Synthetic grass
74.75
83.56

Trees
93.01
95.17

Soil
96.05
99.24

Water
91.38
93.01

Residential
84.46
86.29

Commercial
70.58
71.98

Road
81.07
78.47

Highway
58.03
66.51

Railway
66.72
74.67

Parking lot 1
66.50
82.52

Parking lot 2
67.80
67.72

Tennis court
89.25
89.88

Running track
82.12
90.49

Average accuracy
80.27
84.12

*Table 5: Classification results of LR-HSI and predicted HSI with best results in bold.*

Configurations
Parameters(M)↓\downarrow
Test Time(s)↓\downarrow
CAVE Yasuma et al. (2010)
Harvard Chakrabarti and Zickler (2011)

PSNR ↑\uparrow
SSIM ↑\uparrow
SAM ↓\downarrow
ERGAS ↓\downarrow
PSNR ↑\uparrow
SSIM ↑\uparrow
SAM ↓\downarrow
ERGAS ↓\downarrow

1
w/o ℒSC\mathcal{L}_{\text{SC}}
0.549
0.154
37.83
0.967
9.86
0.70
38.95
0.932
6.25
0.63

w/o aggre
0.549
0.154
39.86
0.975
7.88
0.56
40.45
0.942
5.63
0.55

w/o repel
0.549
0.154
38.74
0.972
8.46
0.61
40.03
0.938
5.82
0.60

2
res_block
0.548
0.155
39.36
0.974
8.03
0.59
40.36
0.941
5.66
0.57

dilated conv
0.547
0.153
40.06
0.976
7.54
0.56
40.56
0.945
5.59
0.58

SWT
0.552
0.156
38.76
0.970
8.76
0.62
39.88
0.933
5.68
0.67

3
concat &\& conv
0.551
0.156
38.92
0.970
8.93
0.61
40.02
0.936
5.62
0.59

concat &\& SWT
0.554
0.159
39.02
0.973
8.76
0.60
39.86
0.940
5.61
0.62

cross
0.552
0.156
40.29
0.976
7.42
0.52
41.88
0.950
4.94
0.54

4
weighted fusion
0.553
0.154
41.83
0.978
6.72
0.46
43.57
0.988
3.84
0.42

cross2
0.552
0.156
41.29
0.971
7.06
0.51
43.39
0.985
3.83
0.42

5
w/o ℒSCT\mathcal{L}_{\text{SCT}}
0.549
0.154
39.72
0.973
7.75
0.53
41.87
0.956
4.84
0.52

w/o ℒSCT2\mathcal{L}_{\text{SCT2}}
0.549
0.154
40.33
0.976
7.63
0.45
41.96
0.952
4.88
0.53

6
w/o decoupling
0.395
0.122
38.63
0.969
8.83
0.66
39.32
0.936
5.97
0.60

7
conv DE
0.549
0.154
39.34
0.974
7.93
0.56
40.38
0.940
5.75
0.58

Ours
0.549
0.154
42.15
0.983
6.48
0.42
44.62
0.991
3.76
0.39

*Table 6: Ablation experiment results on CAVE and Harvard datasets. The best results are in bold, while the second-best results are in underlined.*

![Refer to caption](2412.04802v4/figure/material.png)

*Figure 14: Application for material discrimination.*

![Refer to caption](2412.04802v4/figure/ablation_3Dnew.png)

*Figure 15: Visualization of cumulative reconstruction error across all spectral bands for HR-HSIs from CAVE (top) and Harvard (bottom) datasets in ablation study. Higher error values are indicated by brighter intensity, highlighting regions where fused result deviates more from ground truth.*

Ablation Study. To rigorously validate our architectural design, we conduct comprehensive ablation studies on the CAVE and Harvard datasets, systematically evaluating four core innovation components of MossFuse: 1.Subspace Clustering Loss, 2.LK-CNN Block, 3.Spatial- and Spectral-Aware Blocks, 4.Fusion Strategy of Shared Representations, 5.Self-supervised Constraint Process, 6.Decoupling Mechanism, and 7.Degradation Estimation Process. The quantitative results, presented in Table 6, demonstrate the contribution of each component to the overall reconstruction performance. In addition, we provide visual comparisons of the reconstruction errors in Fig. 15, exhibiting the cumulative error across all spectral bands. These visualizations illustrate that removing any of the core components leads to increased artifacts and degraded fidelity, particularly in texture details and spectral consistency.

1. Subspace Clustering Loss. To quantify the contribution of the Subspace Clustering Loss (ℒSC\mathcal{L}_{\text{SC}}), we conduct three ablation variants: We first fully remove the ℒSC\mathcal{L}_{\text{SC}} to assess its contribution, denoted as ’w/o ℒSC\mathcal{L}_{\text{SC}}’; In addition, we modify the ℒSC\mathcal{L}_{\text{SC}} by separately removing the aggregate term and the repulsion term from the loss, labeled as ’w/o aggre’ and ’w/o repel,’ respectively. Specifically, we reformulate ℒSC\mathcal{L}_{\text{SC}} as the following two terms:

ℒSC′=log⁡(f⁡(FYC,FxC)+∑i∈{Y,x}f⁡(FiS,FiC)CLOSE,\mathcal{L}_{\text{SC}}^{\prime}=\log({f(F^{C}_{\text{Y}},F^{C}_{x})+\sum_{\scriptscriptstyle i\in\{\text{Y},x\}}f(F^{S}_{i},F^{C}_{i})},

(24)

ℒSC′′=−log⁡(f⁡(FYS,FxS)),\mathcal{L}_{\text{SC}}^{\prime\prime}=-\log(f(F^{S}_{\text{Y}},F^{S}_{x})),

(25)

thereby selectively disabling modality-shared features alignment or modality-complementary features separation. As shown in Table 6 and Fig. 15, all variants exhibit significant performance degradation, validating the necessity of both mechanisms. The decline in ’w/o aggre’ underscores the importance of enforcing consistency between shared features, while the drop in ’w/o repel’ reveals the risk of modality decoupling without explicit modality discrimination. These results confirm that ℒSC\mathcal{L}_{\text{SC}} optimally balances ’aggregation’ and ’repulsion’ to enhance representation disentanglement.

2. LK-CNN Block.
To address the large spatial resolution gap (up to ×32\times 32) and effectively extract high-frequency spatial details, we employ the lightweight Large-Kernel CNN (LK-CNN) Zhang et al. (2024b). It consists of three depthwise-separable 3×33\times 3 convolutions with batch normalization and ReLU activations, achieving a 7×77\times 7 receptive field while maintaining efficiency. To validate its effectiveness, we replaced it with three alternatives: a standard Residual Block, a stack of three dilated convolution layers (’dilated conv’), and an SWT block, denoting these variants as ’res_block’, ’dilated conv’, and ’SWT’, respectively. As shown in Table 6 labeled ’2’, all three alternatives result in noticeable performance degradation. In particular, the Residual Block is limited in both feature extraction capacity and efficiency, while dilated convolutions, although more efficient, fail to match the overall effectiveness of the LK‑CNN design. These results justify the use of LK‑CNN blocks as an effective and balanced spatial feature extractor.

3. Spatial- and Spectral-Aware Aggregation Blocks. To evaluate the necessity of our spatial-and spectral-aware aggregation blocks, we replace them with three baseline fusion strategies: concatenation followed by a 3×\times3 convolution, concatenation followed by an SWT block and cross-attention, denoted as ’concat&\&conv’, ’concat&\&SWT’, and ’cross’, respectively. As shown in Table 6, both substitutions lead to significant performance degradation (e.g., -3.23 dB PSNR for ’concat&\&conv’, 3.13dB for ’concat&\&SWT’, and -1.86 dB for ’cross’), revealing the critical limitations of naive fusion approaches. The performance gap further demonstrates that effective multimodal fusion requires interaction from both spatial and spectral dimensions, rather than treating modalities as homogeneous inputs.

4. Fusion Strategy of Shared Representations. To validate the efficiency of the current fusion scheme for the two shared representations FYSF^{S}_{\text{Y}} and FxSF^{S}_{x}, we compare with two alternative fusion strategies, including: (1) weighted fusion, which leverage a learnable scalar weight ω\omega to compute ω⊙FYS+(1−ω)⊙FxS\omega\odot F^{S}_{\text{Y}}+(1-\omega)\odot F^{S}_{x}; (2) cross-attention-based fusion (marked as cross2). As shown in Table 6, both alternatives introduce additional parameters or computational overhead while yielding consistently lower fusion accuracy. In addition, the error visualization in Fig. 15 exhibits larger reconstruction errors as compared to ours. These results demonstrate that our simple additive fusion not only reduces model complexity but also enhances performance by enabling more stable and effective aggregation of modality-shared information.

5. Self-supervised Constraint Process. To validate the efficacy of the self-supervised constraint mechanism, we conduct two targeted ablations: We first completely remove this process labeled as ’w/o ℒMCT\mathcal{L}_{\text{MCT}}’; In addition, we selectively remove the shared-representation supervision branch while preserving the primary constraint term of HR-MSI and LR-HSI (marked as ’w/o ℒMCT2\mathcal{L}_{\text{MCT2}}’). As evidenced by the significant performance drops recorded in Table 6, this analysis quantitatively demonstrates that the completeness and fidelity of the decoupled subspace features act as a necessary condition for model optimization. Moreover, the observed performance improvement achieved through constraining the two modality-shared features FYSF^{S}_{\text{Y}} and FxSF^{S}_{x} to the latent representation of LR-MSI further validates our core architectural design principle that the LR-HSI and HR-MSI can be decoupled into modality-complementary and modality-shared components that collectively characterize the latent representations of LR-MSI embedding.

6. Decoupling Mechanism. To comprehensively validate the effectiveness of our proposed decoupling mechanism, we carry out ablation studies beyond the investigation into the loss functions (e.g., ℒS​C\mathcal{L}_{SC}, ℒS​C​T\mathcal{L}_{SCT}), to also consider the impact of decoupling from an architectural standpoint. The case is therefore marked as ’w/o decoupling’, by removing the dedicated modality-shared encoder and modality-complementary encoders, while adopting a single encoder for each modality (namely, one for HR-MSI and one for the upsampled LR-HSI). These unified encoders directly generate modality-specific features without enforcing any explicit decomposition into shared and complementary components. These features are then concatenated and fed into a shallow 1×11\times 1 convolutional fusion block, followed by the decoder to reconstruct the final HR-HSI. This baseline closely resembles typical encoder-fusion-decoder architectures used in existing fusion methods, thus providing a fair comparison. As evidenced by the significant performance drops recorded in Table 6 and the increased reconstruction errors in Fig. 15, both quantitative and visual analyses confirm the effectiveness of the decoupling mechanism.

7. Degradation Estimation Process. The physics-driven degradation estimation process serves as a foundational component for solving the ill-posed hyperspectral inverse problem by explicitly modeling the SRF and PSF. This systematic parameter estimation enables physically meaningful constraints for both the final reconstructed hyperspectral image and the intermediate LR-MSI generation. To assess the effectiveness of our approach, we conduct ablation studies by replacing this principled estimation framework with unconstrained conventional convolutional layers (denoted as ’conv DE’). As demonstrated quantitatively in Table 6, and qualitatively through error distribution analysis in Fig. 15,
this substitution leads to significant performance degradation, with error heatmaps revealing amplified spectral distortions in high-frequency regions. These comparative results provide quantitative and visual confirmation of our degradation estimation module’s critical role in maintaining spectral-spatial consistency throughout the reconstruction pipeline.

## 5 Conclusion

In this paper, we have presented a fundamental insight into the fusion-based hyperspectral image super-resolution task by revealing the critical importance of modality decoupling. Specifically, we have introduced an end-to-end self-supervised modality-decoupled spatial-spectral fusion (MossFuse) framework that decouples shared and complementary information across modalities and aggregates a concise representation of the LR-HSIs and HR-MSIs to reduce modality redundancy. To reinforce this process, we have developed a novel subspace clustering loss that simultaneously guides the network in aggregating shared LR-MSI components while discerning them from the modality-complementary spatial and spectral components. Through extensive experimentation, we have validated that the integration of precisely decoupled components: the modality-shared LR-MSIs and modality-complementary spatial, spectral information, substantially enhances fusion performance. This improvement is achieved without resorting to complex architectural designs or extensive parameterization. As such, this work results in superior-quality HR-HSI reconstruction with significantly reduced computational overhead in terms of both model parameters and inference time. For future work, we will explore the integration of large-scale pretraining and semantic supervision to further enhance the generalizability of our framework across diverse scenes and sensor domains. Further work will also include experimental investigations into more forms of noise distribution, which would help reveal the limitations of the present work in more effectively dealing with unseen signal degradation types.

## 6 Acknowledgements

This research is supported by National Natural Science Foundation of China (62271400).

## 7 Data Availability Statement

All code and data supporting the findings and experiments of this paper are publicly available in the GitHub repository at https://github.com/dusongcheng/MossFuse.

## References

- Akhtar et al. (2015)
N. Akhtar, F. Shafait, and A. Mian

Bayesian sparse representation for hyperspectral image super resolution.

In Proceedings of the IEEE conference on computer vision and pattern recognition,

pp. 3631–3640.

Cited by: §2.1.

- Arad et al. (2018)
B. Arad, O. Ben-Shahar, and R. Timofte

Ntire 2018 challenge on spectral reconstruction from rgb images.

In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops,

pp. 929–938.

Cited by: Table 1,
§4.1.

- Chakrabarti and Zickler (2011)
A. Chakrabarti and T. Zickler

Statistics of real-world hyperspectral images.

In CVPR 2011,

pp. 193–200.

Cited by: Table 1,
§4.1,
Table 6.

- Chen et al. (2023)
Z. Chen, Y. Zhang, J. Gu, L. Kong, X. Yang, and F. Yu

Dual aggregation transformer for image super-resolution.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 12312–12321.

Cited by: §3.4.

- Dian et al. (2017)
R. Dian, L. Fang, and S. Li

Hyperspectral image super-resolution via non-local sparse tensor factorization.

In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,

pp. 5344–5353.

Cited by: §2.1.

- Dian et al. (2023a)
R. Dian, A. Guo, and S. Li

Zero-shot hyperspectral sharpening.

IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (10), pp. 12650–12666.

Cited by: §2.2,
§3.1.

- Dian et al. (2024a)
R. Dian, Y. Liu, and S. Li

Hyperspectral image fusion via a novel generalized tensor nuclear norm regularization.

IEEE Transactions on Neural Networks and Learning Systems 36 (4), pp. 7437–7448.

Cited by: §2.1,
§4.4.

- Dian et al. (2024b)
R. Dian, Y. Liu, and S. Li

Spectral super-resolution via deep low-rank tensor representation.

IEEE Transactions on Neural Networks and Learning Systems 36 (3), pp. 5140–5150.

Cited by: §4.4.

- Dian et al. (2023b)
R. Dian, T. Shan, W. He, and H. Liu

Spectral super-resolution via model-guided cross-fusion network.

IEEE Transactions on Neural Networks and Learning Systems 35 (7), pp. 10059–10070.

Cited by: §2.1,
§2.2.

- Dong et al. (2016)
W. Dong, F. Fu, G. Shi, X. Cao, J. Wu, G. Li, and X. Li

Hyperspectral image super-resolution via non-negative structured sparse representation.

IEEE Transactions on Image Processing 25 (5), pp. 2337–2352.

Cited by: Table 1,
§4.1.

- Dong et al. (2021)
W. Dong, C. Zhou, F. Wu, J. Wu, G. Shi, and X. Li

Model-guided deep hyperspectral image super-resolution.

IEEE Transactions on Image Processing 30, pp. 5754–5768.

Cited by: §1.

- Du et al. (2023)
S. Du, Y. Leng, X. Liang, J. Li, W. Liu, and Q. Du

Degradation aware unfolding network for spectral super-resolution.

IEEE Geoscience and Remote Sensing Letters 21, pp. 1–5.

Cited by: §2.2.

- Elhamifar and Vidal (2013)
E. Elhamifar and R. Vidal

Sparse subspace clustering: algorithm, theory, and applications.

IEEE transactions on pattern analysis and machine intelligence 35 (11), pp. 2765–2781.

Cited by: §3.7.

- Fu et al. (2021)
X. Fu, S. Jia, M. Xu, J. Zhou, and Q. Li

Fusion of hyperspectral and multispectral images accounting for localized inter-image changes.

IEEE Transactions on Geoscience and Remote Sensing 60, pp. 1–18.

Cited by: §2.1.

- Fu et al. (2017)
Y. Fu, A. Lam, I. Sato, and Y. Sato

Adaptive spatial-spectral dictionary learning for hyperspectral image restoration.

International Journal of Computer Vision 122, pp. 228–245.

Cited by: §2.1.

- Gao et al. (2023)
L. Gao, J. Li, K. Zheng, and X. Jia

Enhanced autoencoders with attention-embedded degradation learning for unsupervised hyperspectral image super-resolution.

IEEE Transactions on Geoscience and Remote Sensing 61, pp. 1–17.

Cited by: §1.

- Gu et al. (2019)
J. Gu, H. Lu, W. Zuo, and C. Dong

Blind super-resolution with iterative kernel correction.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 1604–1613.

Cited by: §3.5.

- Guo et al. (2023)
W. Guo, W. Xie, K. Jiang, Y. Li, J. Lei, and L. Fang

Toward stable, interpretable, and lightweight hyperspectral super-resolution.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 22272–22281.

Cited by: §2.2,
Table 1,
§4.1,
§4.4,
§4.4,
§4.4,
§4.4,
Table 2,
Table 3,
Table 4.

- Han et al. (2020)
X. Han, J. Yu, J. Xue, and W. Sun

Hyperspectral and multispectral image fusion using optimized twin dictionaries.

IEEE Transactions on Image Processing 29, pp. 4709–4720.

Cited by: §2.1.

- Hong et al. (2021)
D. Hong, Z. Han, J. Yao, L. Gao, B. Zhang, A. Plaza, and J. Chanussot

SpectralFormer: rethinking hyperspectral image classification with transformers.

IEEE Transactions on Geoscience and Remote Sensing 60, pp. 1–15.

Cited by: §4.4.

- Hu et al. (2018)
J. Hu, L. Shen, and G. Sun

Squeeze-and-excitation networks.

In Proceedings of the IEEE conference on computer vision and pattern recognition,

pp. 7132–7141.

Cited by: §3.4,
§3.4,
§3.4.

- Hu et al. (2024)
Q. Hu, X. Wang, J. Jiang, X. Zhang, and J. Ma

Exploring the spectral prior for hyperspectral image super-resolution.

IEEE Transactions on Image Processing.

Cited by: §1.

- Jiang et al. (2013)
J. Jiang, D. Liu, J. Gu, and S. Süsstrunk

What is the space of spectral sensitivity functions for digital color cameras?.

In 2013 IEEE Workshop on Applications of Computer Vision (WACV),

pp. 168–179.

Cited by: §4.1.

- Kanatsoulis et al. (2018)
C. I. Kanatsoulis, X. Fu, N. D. Sidiropoulos, and W. Ma

Hyperspectral super-resolution: a coupled tensor factorization approach.

IEEE Transactions on Signal Processing 66 (24), pp. 6503–6517.

Cited by: §1,
§2.1,
§4.4.

- Kawakami et al. (2011)
R. Kawakami, Y. Matsushita, J. Wright, M. Ben-Ezra, Y. Tai, and K. Ikeuchi

High-resolution hyperspectral imaging via matrix factorization.

In CVPR 2011,

pp. 2329–2336.

Cited by: §1,
§2.1.

- Keshava and Mustard (2002)
N. Keshava and J. F. Mustard

Spectral unmixing.

IEEE signal processing magazine 19 (1), pp. 44–57.

Cited by: §2.2.

- Kingma and Ba (2014)
D. P. Kingma and J. Ba

Adam: a method for stochastic optimization.

arXiv preprint arXiv:1412.6980.

Cited by: §4.2.

- Lanaras et al. (2015)
C. Lanaras, E. Baltsavias, and K. Schindler

Hyperspectral super-resolution by coupled spectral unmixing.

In Proceedings of the IEEE international conference on computer vision,

pp. 3586–3594.

Cited by: Table 1,
§4.1,
§4.3.

- Li et al. (2023a)
J. Li, S. Du, R. Song, Y. Li, and Q. Du

Progressive spatial information-guided deep aggregation convolutional network for hyperspectral spectral super-resolution.

IEEE Transactions on Neural Networks and Learning Systems.

Cited by: §2.2.

- Li et al. (2023b)
J. Li, S. Du, R. Song, Y. Li, and Q. Du

Progressive spatial information-guided deep aggregation convolutional network for hyperspectral spectral super-resolution.

IEEE Transactions on Neural Networks and Learning Systems.

Cited by: §4.1,
§4.1,
§4.1.

- Li et al. (2022a)
J. Li, S. Du, R. Song, C. Wu, Y. Li, and Q. Du

HASIC-net: hybrid attentional convolutional neural network with structure information consistency for spectral super-resolution of rgb images.

IEEE Transactions on Geoscience and Remote Sensing 60, pp. 1–15.

Cited by: §1.

- Li et al. (2022b)
J. Li, S. Du, C. Wu, Y. Leng, R. Song, and Y. Li

Drcr net: dense residual channel re-calibration network with non-local purification for spectral super resolution.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 1259–1268.

Cited by: §1.

- Li et al. (2022c)
J. Li, D. Hong, L. Gao, J. Yao, K. Zheng, B. Zhang, and J. Chanussot

Deep learning in multimodal remote sensing data fusion: a comprehensive review.

International Journal of Applied Earth Observation and Geoinformation 112, pp. 102926.

Cited by: §1.

- Li et al. (2025a)
J. Li, K. Zheng, L. Gao, Z. Han, Z. Li, and J. Chanussot

Enhanced deep image prior for unsupervised hyperspectral image super-resolution.

IEEE Transactions on Geoscience and Remote Sensing.

Cited by: §2.2,
§3.5,
Table 1,
§4.1,
Table 2,
Table 3,
Table 4.

- Li et al. (2025b)
J. Li, K. Zheng, L. Gao, Z. Han, Z. Li, and J. Chanussot

Enhanced deep image prior for unsupervised hyperspectral image super-resolution.

IEEE Transactions on Geoscience and Remote Sensing.

Cited by: §1.

- Li et al. (2024a)
J. Li, K. Zheng, L. Gao, L. Ni, M. Huang, and J. Chanussot

Model-informed multi-stage unsupervised network for hyperspectral image super-resolution.

IEEE Transactions on Geoscience and Remote Sensing.

Cited by: §2.2,
§3.5.

- Li et al. (2023c)
J. Li, K. Zheng, Z. Li, L. Gao, and X. Jia

X-shaped interactive autoencoders with cross-modality mutual learning for unsupervised hyperspectral image super-resolution.

IEEE transactions on geoscience and remote sensing 61, pp. 1–17.

Cited by: §1.

- Li et al. (2023d)
J. Li, K. Zheng, W. Liu, Z. Li, H. Yu, and L. Ni

Model-guided coarse-to-fine fusion network for unsupervised hyperspectral image super-resolution.

IEEE Geoscience and Remote Sensing Letters 20, pp. 1–5.

Cited by: §1.

- Li et al. (2022d)
J. Li, K. Zheng, J. Yao, L. Gao, and D. Hong

Deep unsupervised blind hyperspectral and multispectral data fusion.

IEEE Geoscience and Remote Sensing Letters 19, pp. 1–5.

Cited by: §1.

- Li et al. (2018)
S. Li, R. Dian, L. Fang, and J. M. Bioucas-Dias

Fusing hyperspectral and multispectral images via coupled sparse tensor factorization.

IEEE Transactions on Image Processing 27 (8), pp. 4118–4130.

Cited by: §2.1.

- Li et al. (2023e)
S. Li, R. Dian, and H. Liu

Learning the external and internal priors for multispectral and hyperspectral image fusion.

Science China Information Sciences 66 (4), pp. 140303.

Cited by: §2.1.

- Li et al. (2024b)
X. Li, J. Liu, Z. Chen, Y. Zou, L. Ma, X. Fan, and R. Liu

Contourlet residual for prompt learning enhanced infrared image super-resolution.

In European Conference on Computer Vision,

pp. 270–288.

Cited by: §2.2.

- Li et al. (2025c)
X. Li, Z. Wang, Y. Zou, Z. Chen, J. Ma, Z. Jiang, L. Ma, and J. Liu

Difiisr: a diffusion model with gradient guidance for infrared image super-resolution.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 7534–7544.

Cited by: §2.1.

- Li et al. (2025d)
Y. Li, Z. Li, D. Liu, and L. Li

Frequency domain intra pattern copy for jpeg xs screen content coding.

IEEE Transactions on Circuits and Systems for Video Technology.

Cited by: §1.

- Li et al. (2025e)
Z. Li, J. Liao, C. Tang, H. Zhang, Y. Li, Y. Bian, X. Sheng, X. Feng, Y. Li, C. Gao, et al.

Ustc-td: a test dataset and benchmark for image and video coding in 2020s.

IEEE Transactions on Multimedia.

Cited by: §2.2.

- Li et al. (2024c)
Z. Li, Z. Yuan, L. Li, D. Liu, X. Tang, and F. Wu

Object segmentation-assisted inter prediction for versatile video coding.

IEEE Transactions on Broadcasting.

Cited by: §2.2.

- Lian et al. (2024)
J. Lian, L. Wang, L. Zhu, R. Dian, Z. Xiong, and H. Huang

Physics-inspired degradation models for hyperspectral image fusion.

arXiv preprint arXiv:2402.02411.

Cited by: §3.5.

- Liang et al. (2025)
L. Liang, P. Xie, Y. Zhang, J. Li, Z. Zhang, J. Li, and A. Plaza

DBMLLA: double-branch mamba-like linear attention network for hyperspectral image classification.

IEEE Transactions on Geoscience and Remote Sensing.

Cited by: §1.

- Liang et al. (2024)
Y. Liang, Z. Cao, S. Deng, H. Dou, and L. Deng

Fourier-enhanced implicit neural fusion network for multispectral and hyperspectral image fusion.

Advances in Neural Information Processing Systems 37, pp. 63441–63465.

Cited by: §4.4,
Table 4.

- Lin et al. (2023)
C. Lin, S. Huang, T. Lin, and P. C. Wu

Metasurface-empowered snapshot hyperspectral imaging with convex/deep (code) small-data learning theory.

Nature communications 14 (1), pp. 6979.

Cited by: §1.

- Liu et al. (2022)
J. Liu, Z. Wu, L. Xiao, and X. Wu

Model inspired autoencoder for unsupervised hyperspectral image super-resolution.

IEEE Transactions on Geoscience and Remote Sensing 60, pp. 1–12.

Cited by: §3.5.

- Liu et al. (2024)
J. Liu, X. Li, Z. Wang, Z. Jiang, W. Zhong, W. Fan, and B. Xu

PromptFusion: harmonized semantic prompt learning for infrared and visible image fusion.

IEEE/CAA Journal of Automatica Sinica 12, pp. 1–14.

Cited by: §2.2.

- Liu et al. (2025a)
S. Liu, Y. Fan, Q. Hu, B. Li, Y. Zhang, and S. Liu

SSMSFuse: a spectral and spatial multi-scale coupling fusion model for hyperspectral and multispectral image.

IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

Cited by: §2.2.

- Liu et al. (2025b)
Y. Liu, R. Dian, and S. Li

Low-rank transformer for high-resolution hyperspectral computational imaging.

International Journal of Computer Vision 133 (2), pp. 809–824.

Cited by: §2.2.

- Liu et al. (2025c)
Y. Liu, J. Liu, R. Dian, and S. Li

A selective re-learning mechanism for hyperspectral fusion imaging.

In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 7437–7446.

Cited by: §4.4,
Table 4.

- Ma et al. (2023)
Q. Ma, J. Jiang, X. Liu, and J. Ma

Learning a 3d-cnn and transformer prior for hyperspectral image super-resolution.

Information Fusion 100, pp. 101907.

Cited by: §4.4,
Table 4.

- Ma et al. (2024)
Q. Ma, J. Jiang, X. Liu, and J. Ma

Reciprocal transformer for hyperspectral and multispectral image fusion.

Information Fusion 104, pp. 102148.

Cited by: §4.4,
Table 4.

- Qu et al. (2018)
Y. Qu, H. Qi, and C. Kwan

Unsupervised sparse dirichlet-net for hyperspectral image super-resolution.

In Proceedings of the IEEE conference on computer vision and pattern recognition,

pp. 2511–2520.

Cited by: §2.2.

- Ren et al. (2021)
K. Ren, W. Sun, X. Meng, G. Yang, J. Peng, and J. Huang

A locally optimized model for hyperspectral and multispectral images fusion.

IEEE Transactions on Geoscience and Remote Sensing 60, pp. 1–15.

Cited by: §2.1.

- Simoes et al. (2014)
M. Simoes, J. Bioucas-Dias, L. B. Almeida, and J. Chanussot

A convex formulation for hyperspectral image superresolution via subspace-based regularization.

IEEE Transactions on Geoscience and Remote Sensing 53 (6), pp. 3373–3388.

Cited by: Table 1,
§4.1,
§4.3,
§4.4.

- Tan et al. (2024)
L. Tan, R. Dian, S. Li, and J. Liu

Frequency-spatial domain feature fusion for spectral super-resolution.

IEEE Transactions on Computational Imaging 10, pp. 589–599.

Cited by: §4.4,
Table 4.

- Wald (2000)
L. Wald

Quality of high resolution synthesised images: is there a simple criterion?.

In Third conference” Fusion of Earth data: merging point measurements, raster maps and remotely sensed images”,

pp. 99–103.

Cited by: §4.1.

- Wang et al. (2020a)
D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell

Tent: fully test-time adaptation by entropy minimization.

arXiv preprint arXiv:2006.10726.

Cited by: §2.2.

- Wang and Isola (2020)
T. Wang and P. Isola

Understanding contrastive representation learning through alignment and uniformity on the hypersphere.

In International conference on machine learning,

pp. 9929–9939.

Cited by: §3.7.

- Wang et al. (2024a)
W. Wang, L. Deng, R. Ran, and G. Vivone

A general paradigm with detail-preserving conditional invertible network for image fusion.

International Journal of Computer Vision 132 (4), pp. 1029–1054.

Cited by: §2.2.

- Wang et al. (2023)
X. Wang, Q. Hu, Y. Cheng, and J. Ma

Hyperspectral image super-resolution meets deep learning: a survey and perspective.

IEEE/CAA Journal of Automatica Sinica 10 (8), pp. 1668–1691.

Cited by: §1.

- Wang et al. (2024b)
Y. Wang, W. Li, Y. Gui, Q. Du, and J. E. Fowler

A generalized tensor formulation for hyperspectral image super-resolution under general spatial blurring.

arXiv preprint arXiv:2409.18731.

Cited by: §3.5.

- Wang et al. (2020b)
Z. Wang, B. Chen, R. Lu, H. Zhang, H. Liu, and P. K. Varshney

FusionNet: an unsupervised convolutional variational network for hyperspectral and multispectral image fusion.

IEEE Transactions on Image Processing 29, pp. 7565–7577.

Cited by: §1.

- Wang et al. (2004)
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli

Image quality assessment: from error visibility to structural similarity.

IEEE Transactions on Image Processing 13 (4), pp. 600–612.

Cited by: §4.1.

- Wu et al. (2023)
C. Wu, D. Wang, Y. Bai, H. Mao, Y. Li, and Q. Shen

HSR-diff: hyperspectral image super-resolution via conditional diffusion models.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 7083–7093.

Cited by: §1.

- Wu et al. (2024a)
S. Wu, K. Zhang, X. Liu, S. Shi, C. Bian, et al.

Unsupervised blind spectral–spatial cross-super-resolution network for hsi and msi fusion.

IEEE Transactions on Geoscience and Remote Sensing 62, pp. 1–14.

Cited by: §2.2.

- Wu et al. (2024b)
Z. Wu, Y. Li, T. Huang, L. Deng, and G. Vivone

CroDoSR: tensor cross-domain rank for hyperspectral image super-resolution.

IEEE Transactions on Geoscience and Remote Sensing.

Cited by: §2.1.

- Xiao et al. (2025)
L. Xiao, S. Guo, F. Mo, Q. Song, Y. Yang, Y. Liu, X. Wei, T. Yang, and R. Dian

Spatial invertible network with mamba-convolution for hyperspectral image fusion.

IEEE Journal of Selected Topics in Signal Processing.

Cited by: §2.2,
§4.4,
Table 4.

- Xie et al. (2019)
Q. Xie, M. Zhou, Q. Zhao, D. Meng, W. Zuo, and Z. Xu

Multispectral and hyperspectral image fusion by ms/hs fusion net.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 1585–1594.

Cited by: §1,
Table 1,
§4.1,
Table 2,
Table 3,
Table 4.

- Xu et al. (2021)
H. Xu, M. Qin, S. Chen, Y. Zheng, and J. Zheng

Hyperspectral-multispectral image fusion via tensor ring and subspace decompositions.

IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 14, pp. 8823–8837.

Cited by: §4.4.

- Xu et al. (2025)
J. Xu, Y. Su, X. Sun, J. Li, H. Li, J. Gao, X. Feng, and R. Liu

FusGAT: graph attention-based fusion network for unsupervised hyperspectral image super-resolution.

IEEE Geoscience and Remote Sensing Letters.

Cited by: §1.

- Xu et al. (2020)
Y. Xu, Z. Wu, J. Chanussot, and Z. Wei

Hyperspectral images super-resolution via learning high-order coupled tensor ring representation.

IEEE transactions on neural networks and learning systems 31 (11), pp. 4747–4760.

Cited by: §2.1.

- Xu et al. (2019)
Y. Xu, B. Du, L. Zhang, D. Cerra, M. Pato, E. Carmona, S. Prasad, N. Yokoya, R. Hänsch, and B. Le Saux

Advanced multi-sensor optical remote sensing for urban land use and land cover classification: outcome of the 2018 ieee grss data fusion contest.

IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12 (6), pp. 1709–1724.

Cited by: §4.1.

- Yang et al. (2025)
P. Yang, Y. Ma, X. Mei, Q. Chen, M. Wu, and J. Ma

Deep blind super-resolution for hyperspectral images.

Pattern Recognition 157, pp. 110916.

Cited by: §1.

- Yao et al. (2020)
J. Yao, D. Hong, J. Chanussot, D. Meng, X. Zhu, and Z. Xu

Cross-attention in coupled unmixing nets for unsupervised hyperspectral super-resolution.

In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIX 16,

pp. 208–224.

Cited by: §1,
§2.2,
Table 1,
§4.1,
Table 3,
Table 4.

- Yasuma et al. (2010)
F. Yasuma, T. Mitsunaga, D. Iso, and S. K. Nayar

Generalized assorted pixel camera: postcapture control of resolution, dynamic range, and spectrum.

IEEE Transactions on Image Processing 19 (9), pp. 2241–2253.

Cited by: Table 1,
§4.1,
Table 6.

- Ye et al. (2023)
F. Ye, Z. Wu, X. Jia, J. Chanussot, Y. Xu, and Z. Wei

Bayesian nonlocal patch tensor factorization for hyperspectral image super-resolution.

IEEE Transactions on Image Processing.

Cited by: §2.1.

- Yokoya et al. (2011)
N. Yokoya, T. Yairi, and A. Iwasaki

Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion.

IEEE Transactions on Geoscience and Remote Sensing 50 (2), pp. 528–537.

Cited by: §2.1,
§4.4.

- Yu et al. (2024)
H. Yu, Z. Ling, K. Zheng, L. Gao, J. Li, and J. Chanussot

Unsupervised hyperspectral and multispectral image fusion with deep spectral-spatial collaborative constraint.

IEEE Transactions on Geoscience and Remote Sensing.

Cited by: §1.

- Yuhas et al. (1992)
R. H. Yuhas, A. F. Goetz, and J. W. Boardman

Discrimination among semi-arid landscape endmembers using the spectral angle mapper (sam) algorithm.

In JPL, Summaries of the Third Annual JPL Airborne Geoscience Workshop. Volume 1: AVIRIS Workshop,

Cited by: §4.1.

- Zamir et al. (2022)
S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M. Yang

Restormer: efficient transformer for high-resolution image restoration.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,

pp. 5728–5739.

Cited by: §3.3.

- Zhang et al. (2024a)
L. Zhang, J. Nie, W. Wei, and Y. Zhang

Unsupervised test-time adaptation learning for effective hyperspectral image super-resolution with unknown degeneration.

IEEE Transactions on Pattern Analysis and Machine Intelligence.

Cited by: §2.2,
Table 1,
§4.1,
§4.4,
§4.4,
Table 2,
Table 3,
Table 4.

- Zhang et al. (2018)
L. Zhang, W. Wei, Y. Zhang, C. Shen, A. Van Den Hengel, and Q. Shi

Cluster sparsity field: an internal hyperspectral imagery prior for reconstruction.

International Journal of Computer Vision 126, pp. 797–821.

Cited by: §2.1.

- Zhang et al. (2023)
M. Zhang, C. Zhang, Q. Zhang, J. Guo, X. Gao, and J. Zhang

ESSAformer: efficient transformer for hyperspectral image super-resolution.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 23073–23084.

Cited by: §1.

- Zhang et al. (2024b)
Y. Zhang, X. Ding, and X. Yue

Scaling up your kernels: large kernel design in convnets towards universal representations.

arXiv preprint arXiv:2410.08049.

Cited by: §3.3,
§4.4.

- Zhu et al. (2025)
C. Zhu, S. Deng, X. Song, Y. Li, and Q. Wang

Mamba collaborative implicit neural representation for hyperspectral and multispectral remote sensing image fusion.

IEEE Transactions on Geoscience and Remote Sensing.

Cited by: §2.2.

- Zou et al. (2024a)
Y. Zou, Z. Chen, Z. Zhang, X. Li, L. Ma, J. Liu, P. Wang, and Y. Zhang

Contourlet refinement gate framework for thermal spectrum distribution regularized infrared image super-resolution.

arXiv preprint arXiv:2411.12530.

Cited by: §2.1.

- Zou et al. (2024b)
Y. Zou, X. Li, Z. Jiang, and J. Liu

Enhancing neural radiance fields with adaptive multi-exposure fusion: a bilevel optimization approach for novel view synthesis.

In Proceedings of the AAAI Conference on Artificial Intelligence,

Vol. 38, pp. 7882–7890.

Cited by: §2.2.

<!-- END-OF-PAPER:f5ca9f7ca223 -->
