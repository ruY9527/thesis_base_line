Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model

    ![](/static/base/1.0.1/images/icons/smileybones-small.svg)
    arXiv is now an independent nonprofit!
    Learn more
    ×

    License: arXiv.org perpetual non-exclusive license

arXiv:2505.11800v1 [cs.CV] 17 May 2025

# Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model

Jian Zhu

  
He Wang

  
Yang Xu11
          1

        Corresponding authors.

  
Zebin Wu11
          1

        Corresponding authors.

  
Zhihui Wei

Affiliation: School of Computer Science and Engineering,

Affiliation: Nanjing University of Science and Technology, Nanjing, China

Email: {123106010719,he_wang,xuyangth90,wuzb,gswei}@njust.edu.cn

###### Abstract

Hyperspectral and multispectral image (HSI-MSI) fusion involves combining a low-resolution hyperspectral image (LR-HSI) with a high-resolution multispectral image (HR-MSI) to generate a high-resolution hyperspectral image (HR-HSI). Most deep learning-based methods for HSI-MSI fusion rely on large amounts of hyperspectral data for supervised training, which is often scarce in practical applications. In this paper, we propose a self-learning Adaptive Residual Guided Subspace Diffusion Model (ARGS-Diff), which only utilizes the observed images without any extra training data. Specifically, as the LR-HSI contains spectral information and the HR-MSI contains spatial information, we design two lightweight spectral and spatial diffusion models to separately learn the spectral and spatial distributions from them. Then, we use these two models to reconstruct HR-HSI from two low-dimensional components, i.e, the spectral basis and the reduced coefficient, during the reverse diffusion process. Furthermore, we introduce an Adaptive Residual Guided Module (ARGM), which refines the two components through a residual guided function at each sampling step, thereby stabilizing the sampling process. Extensive experimental results demonstrate that ARGS-Diff outperforms existing state-of-the-art methods in terms of both performance and computational efficiency in the field of HSI-MSI fusion. Code is available at https://github.com/Zhu1116/ARGS-Diff.

## 1 Introduction

Hyperspectral images (HSI) are specialized images that contain extensive continuous spectral band information. Compared to natural images, HSIs provide more detailed spectral data, reflecting the intrinsic properties of various objects. This distinctive characteristic of HSIs makes them widely applicable across various domains, such as agricultural monitoring [4, 10], environmental protection [12], urban planning [9], and mineral exploration [14, 27]. However, due to hardware limitations, existing hyperspectral sensors often fail to deliver sufficient spatial resolution. To achieve higher spatial resolution, a common approach is HSI-MSI fusion, which involves fusing low-resolution hyperspectral images (LR-HSI) with high-resolution multispectral images (HR-MSI) to generate high-resolution hyperspectral images (HR-HSI) [42]. In the past decades, HSI-MSI fusion has garnered significant attention, and numerous related methods have emerged.

Existing HSI-MSI fusion methods can be roughly categorized into two types, i.e, model-based methods [26, 16, 40, 28] and deep learning (DL)-based methods [25, 36, 39, 32]. Model-based approaches enhance image resolution by integrating spatial and spectral information from low-resolution hyperspectral images and high-resolution multispectral images. However, they rely on manually designed feature extractors and often assume independence between spatial and spectral information, which poses limitations in handling high-dimensional data and impedes the effective capture of complex features. In particular, when faced with the high dimensionality and nonlinearity of hyperspectral images, these methods struggle to deliver optimal performance and frequently require precise prior knowledge, thereby restricting their applicability.

In recent years, deep learning (DL) has made significant advancements in HSI-MSI fusion. These methods often rely on deep convolutional neural networks (DCNNs) [32, 41, 38] or Transformer networks [31, 6, 33, 15] to extract intricate spatial and spectral features from the images. DCNNs are particularly effective at capturing local patterns and hierarchical features, making them well-suited for spatial information extraction. On the other hand, Transformer networks excel at modeling long-range dependencies, allowing them to effectively capture global spectral information. While these methods outperform traditional model-based approaches in terms of feature extraction, they often rely heavily on large volumes of paired HSI-MSI data, which can be difficult to acquire in practical applications.
Recently, diffusion models (DMs) [8, 30, 18, 5], a novel generative models, have shown tremendous potential in image generation. With a more stable training process and superior generation quality, diffusion models have been widely applied in various image tasks, including image super-resolution [24, 34, 37]. In the realm of hyperspectral image fusion [23, 35, 7], PLRDiff [23] introduced a low-rank diffusion model tailored for hyperspectral image pan-sharpening, which leverages a pretrained deep diffusion model to provide image priors through Bayesian principles. In [21], a spatial-spectral-bilateral cycle diffusion framework (S2CycleDiff) was proposed to progressively generate high-resolution hyperspectral images by learning a customized conditional cycle-diffusion network. However, despite these advancements, achieving high-quality results generally requires thousands of iterations, resulting in slow inference speeds and substantial computational costs. Some approaches [29, 19, 17] have developed accelerated sampling strategies to reduce the number of iterations to a few dozen. Nevertheless, each inference step still demands substantial memory and floating-point computations, particularly in HSI-MSI fusion. Moreover, most edge devices (e.g., drones) are limited by storage and computational resources, making it challenging to deploy these demanding methods in practical remote sensing applications. Consequently, it is crucial to design a model that achieves low resource consumption and fast inference speeds while maintaining generation quality.

In this work, we propose a novel self-learning approach, the Adaptive Residual Guided Subspace Diffusion Model (ARGS-Diff). Based on the fact that HSI can be effectively reconstructed from the product of two low-dimensional components: the spectral basis and the reduced coefficient, we design and train two lightweight spectral and spatial networks and then use them to independently reconstruct the spectral basis and the reduced coefficient, respectively. The training process relies solely on the observed LR-HSI and HR-MSI to fully exploit the rich spectral and spatial information inherent in each of them. Furthermore, we introduce the Adaptive Residual Guided Module (ARGM) to ensure more stable convergence during the sampling process. In our method, both components need to be updated simultaneously at each step, and improper alignment between them may lead to instability or collapse. The ARGM alleviates this issue by applying a residual-guided function at each sampling step. It calculates the residual between the reconstructed two components and the target, then adjusts them to reduce the discrepancy, effectively guiding them toward better alignment with the true values. By incorporating this module, our approach demonstrates enhanced robustness to noise and achieves more stable and reliable reconstruction performance.

In summary, our contributions are as follows:

- •

We propose a self-learning Adaptive Residual Guided Subspace Diffusion Model (ARGS-Diff), which effectively learns spectral and spatial distributions from the observed LR-HSI and HR-MSI using specially designed lightweight spectral and spatial networks, and reconstructs the HR-HSI from the estimated spectral basis and reduced coefficient.

- •

We introduce the Adaptive Residual Guided Module (ARGM), which refines the spectral and spatial components through a residual-guided function, effectively enhancing the stability and convergence of the sampling process.

- •

Extensive experiments conducted on four datasets demonstrate the superior performance of the proposed method over existing state-of-the-art approaches.

## 2 Preliminaries

### 2.1 Diffusion Models

The diffusion model reconstructs data by progressively adding noise during a forward process and learning the reverse denoising pathway to recover the original data [8, 29]. Typically, a diffusion process is composed of a forward process with TT steps of noise addition, followed by a reverse process over the same TT steps. Given a noise schedule {α¯t}t=1T\{\bar{\alpha}_{t}\}_{t=1}^{T} and clean input data 𝐱0\mathbf{x}_{0}, the noise addition in each step is defined by:

𝐱t=α¯t​𝐱0+(1−α¯t)​ϵ,ϵ∼𝒩⁡(0,I).\mathbf{x}_{t}=\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}+(\sqrt{1-\bar{\alpha}_{t}})\epsilon,\quad\epsilon\sim\mathcal{N}(0,I).

(1)

The model is trained to predict the noise ϵ\epsilon by minimizing the error between the estimated and true noise:

ϵθ​(𝐱t,t)≈ϵ=𝐱t−α¯t​𝐱01−α¯t.\epsilon_{\theta}(\mathbf{x}_{t},t)\approx\epsilon=\frac{\mathbf{x}_{t}-\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}}{\sqrt{1-\bar{\alpha}_{t}}}.

(2)

In the reverse process, starting from random noise, the model iteratively refines the estimate over TT steps. At each step tt, the model estimates the clean input 𝐱^0\hat{\mathbf{x}}_{0} as:

𝐱^0=𝐱t−(1−α¯t)​ϵθ​(𝐱t,t)α¯t.\hat{\mathbf{x}}_{0}=\frac{\mathbf{x}_{t}-(1-\bar{\alpha}_{t})\epsilon_{\theta}(\mathbf{x}_{t},t)}{\sqrt{\bar{\alpha}_{t}}}.

(3)

Subsequently, 𝐱t−1\mathbf{x}_{t-1} is computed by:

𝐱t−1=α¯t−1​𝐱^0+(1−α¯t−1)​ϵθ.\mathbf{x}_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\hat{\mathbf{x}}_{0}+(\sqrt{1-\bar{\alpha}_{t-1}})\epsilon_{\theta}.

(4)

The final reconstructed output 𝐱0\mathbf{x}_{0} is obtained after all reverse steps are completed.

![Refer to caption](2505.11800v1/dps3.png)

*Figure 1: The overall framework of the proposed ARGS-Diff.*

### 2.2 HSI-MSI Fusion

In HSI-MSI fusion problem, the input data consists of a LR-HSI 𝒳∈ℝh×w×C\mathcal{X}\in\mathbb{R}^{h\times w\times C} and a HR-MSI 𝒴∈ℝH×W×c\mathcal{Y}\in\mathbb{R}^{H\times W\times c}, which are fused to produce a HR-HSI 𝒵∈ℝH×W×C\mathcal{Z}\in\mathbb{R}^{H\times W\times C}. Typically, the relationship between the LR-HSI, HR-MSI, and HR-HSI is represented through degradation models as follows:

𝒳=ℋ⁡(𝒵)+𝒩𝒳,\mathcal{X}=\mathcal{H}(\mathcal{Z})+\mathcal{N}_{\mathcal{X}},

(5)

𝒴=𝒵×3𝐑+𝒩𝒴,\mathcal{Y}=\mathcal{Z}\times_{3}\mathbf{R}+\mathcal{N}_{\mathcal{Y}},

(6)

where ℋ⁡(⋅)\mathcal{H}(\cdot) represents downsampling in the spatial domain, 𝐑\mathbf{R} is the spectral response function (SRF) matrix, ×3\times_{3} denotes mode-3 multiplication, 𝒩𝒳\mathcal{N}_{\mathcal{X}} and 𝒩𝒴\mathcal{N}_{\mathcal{Y}} are the measurements noise, which is assumed to be the Gaussian noise.

A common approach for HSI-MSI fusion is to decompose the HSI into two low-dimensional components, process each separately, and then combine them to synthesize a clean HR-HSI. The decomposition is expressed as:

𝒵=𝒜×3𝐄,\mathcal{Z}=\mathcal{A}\times_{3}\mathbf{E},

(7)

where 𝒜∈ℝH×W×d\mathcal{A}\in\mathbb{R}^{H\times W\times d} represents the reduced coefficient, 𝐄∈ℝd×C\mathbf{E}\in\mathbb{R}^{d\times C} represents the spectral basis, dd is the number of subspace dimensions, and d≪Cd\ll C.

Compared to the high dimensional of HSI, these two low-dimensional components have reduced data complexity, making them easier to process. In this paper, we follow this approach by training a spectral and spatial network using the LR-HSI and HR-MSI. During the sampling process, these networks separately reconstruct the corresponding spectral basis and reduced coefficient. The final fused HR-HSI is obtained from the product of these two components.

## 3 Proposed Method

### 3.1 Overview

As described in Eq. (7), the HSI can be decomposed into two low-dimensional components, i.e, the spectral basis and the reduced coefficient. We train two separate networks to reconstruct these components, which are then combined to form the final HR-HSI.
Specifically, We design a spectral network and a spatial network, and train them using the LR-HSI and HR-MSI based on the forward process of the diffusion model to separately extract the spectral and spatial distribution information. The detailed network architecture and training procedure are described in Section 3.2.

During the sampling process, we leverage the posterior sampling theory [2] to iteratively reconstruct the spectral basis and reduced coefficient using the well-trained networks, conditioning on the observed LR-HSI and HR-MSI. The detailed sampling process is outlined in Section 3.3. Additionally, since we need to update two components simultaneously at each sampling step, which may lead to instability if they are not properly aligned, we introduce the Adaptive Residual Guided Module (ARGM) to iteratively refine the spectral and spatial components at each step, ensuring the stability of the sampling process, as described in Section 3.4. After TT sampling steps, the HR-HSI is obtained by taking the product of the reconstructed spectral basis and reduced coefficient.

### 3.2 Self-learning Subspace Networks

Recent advancements in diffusion models have introduced many techniques, such as attention mechanisms, to enhance performance. However, these innovations often come with a significant increase in network size, resulting in higher resource consumption and longer image generation times. To address this challenge, we design two lightweight networks specifically tailored for hyperspectral image fusion: a spectral network and a spatial network. This design significantly reduces sampling time without compromising performance in our method.

The spectral network is a fully connected network consisting of five layers, where the input and output feature dimensions match the number of spectral bands, CC, in the LR-HSI. The spatial network adopts a ”UNet-like” architecture [22], comprising nine convolutional layers: four downsampling layers, one intermediate layer, and four upsampling layers. Each convolutional layer includes two residual blocks, and the input and output channels are set to the number of subspace dimensions, dd.

To optimize network training, we restructure the training samples. For the spectral network, LR-HSI, which contains abundant spectral information, is leveraged to represent the spectral distribution of the HR-HSI. Specifically, we randomly select dd pixel spectra from the LR-HSI and concatenate them into a sample of size (d,C)(d,C) for the spectral network’s input. For the spatial network, HR-MSI, which retains rich spatial information, is used for training. We extract a patch from the HR-MSI, randomly select one band from cc spectral bands within the patch, and repeat this process dd times to form a sample of size (Hpatch,Wpatch,d)(H_{\text{patch}},W_{\text{patch}},d). This approach not only facilitates effective training but also introduces diversity into the input samples, enabling the network to better capture the spatial and spectral features of the images.

### 3.3 Subspace Reverse Diffusion Process

Recently, several methods [1, 2] have been proposed to generate images using various conditional inputs. In general, given a condition 𝐲\mathbf{y}, these methods model the reverse sampling process as pθ​(𝐱t−1|𝐱t,𝐲)p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{y}) incorporating a gradient term into the noise estimation function ϵθ\epsilon_{\theta}. This is formulated as:

ϵ^θ​(𝐱t,t)=ϵθ​(𝐱t,t)−ρ​∇𝐱tℒ​(𝐱^0,y),\hat{\epsilon}_{\theta}(\mathbf{x}_{t},t)=\epsilon_{\theta}(\mathbf{x}_{t},t)-\rho\nabla_{\mathbf{x}_{t}}\mathcal{L}(\hat{\mathbf{x}}_{0},y),

(8)

where ρ\rho denotes the step size, and ℒ\mathcal{L} is a guidance function that measures the distance between the predicted image 𝐱^0\hat{\mathbf{x}}_{0} and the target image. In our case, we use the LR-HSI 𝒳\mathcal{X} and HR-MSI 𝒴\mathcal{Y} as conditions, employing spatial and spectral networks to estimate the spectral basis and the reduced coefficient. Specifically, Eq. (8) can be rewritten as:

\displaystyle
s^θ​(𝒜t,t)=sθ​(𝒜t,t)−ρ1​∇𝒜tℒ​(𝒜^0,𝐄^0,𝒳,𝒴),\displaystyle\hat{s}_{\theta}(\mathcal{A}_{t},t)=s_{\theta}(\mathcal{A}_{t},t)-\rho_{1}\nabla_{\mathcal{A}_{t}}\mathcal{L}(\hat{\mathcal{A}}_{0},\hat{\mathbf{E}}_{0},\mathcal{X},\mathcal{Y}),

(9)

c^ζ​(𝐄t,t)=cζ​(𝐄t,t)−ρ2​∇𝐄tℒ​(𝒜^0,𝐄^0,𝒳,𝒴),\displaystyle\hat{c}_{\zeta}(\mathbf{E}_{t},t)=c_{\zeta}(\mathbf{E}_{t},t)-\rho_{2}\nabla_{\mathbf{E}_{t}}\mathcal{L}(\hat{\mathcal{A}}_{0},\hat{\mathbf{E}}_{0},\mathcal{X},\mathcal{Y}),

where sθ​(⋅)s_{\theta}(\cdot) and cζ​(⋅)c_{\zeta}(\cdot) represent the spatial and spectral networks, respectively, ρ1\rho_{1} and ρ2\rho_{2} are the step size for updating sθ​(𝒜t,t)s_{\theta}(\mathcal{A}_{t},t) and cζ​(𝐄t,t)c_{\zeta}(\mathbf{E}_{t},t). The guidance function is defined as:

ℒ⁡(𝒜^0,𝐄^0,𝒳,𝒴)\displaystyle\mathcal{L}(\hat{\mathcal{A}}_{0},\hat{\mathbf{E}}_{0},\mathcal{X},\mathcal{Y})
=‖ℋ⁡(𝒜^0×3𝐄^0)−𝒳‖22\displaystyle=\ \|\mathcal{H}(\hat{\mathcal{A}}_{0}\times_{3}\hat{\mathbf{E}}_{0})-\mathcal{X}\|^{2}_{2}

(10)

+λ1∥𝒜^0×3𝐄^0×3𝐑−𝒴∥22,\displaystyle+\lambda_{1}\|\hat{\mathcal{A}}_{0}\times_{3}\hat{\mathbf{E}}_{0}\times_{3}\mathbf{R}-\mathcal{Y}\|^{2}_{2},

where λ1\lambda_{1} is weight to balance the two observations. 𝒜^0\hat{\mathcal{A}}_{0} and 𝐄^0\hat{\mathbf{E}}_{0} can be estimated from Eq. (3) as:

𝒜^0=𝒜t−(1−α¯t)​sθ​(𝒜t,t)α¯t,\displaystyle\hat{\mathcal{A}}_{0}=\frac{\mathcal{A}_{t}-(\sqrt{1-\bar{\alpha}_{t}})s_{\theta}(\mathcal{A}_{t},t)}{\sqrt{\bar{\alpha}_{t}}},

(11)

𝐄^0=𝐄t−(1−α¯t)​cζ​(𝐄t,t)α¯t.\displaystyle\hat{\mathbf{E}}_{0}=\frac{\mathbf{E}_{t}-(\sqrt{1-\bar{\alpha}_{t}})c_{\zeta}(\mathbf{E}_{t},t)}{\sqrt{\bar{\alpha}_{t}}}.

Next, 𝒜t−1\mathcal{A}_{t-1} and 𝐄t−1\mathbf{E}_{t-1} are sampled from Eq. (1) as:

𝒜t−1=α¯t−1​𝒜^0+(1−α¯t−1)​s^θ​(𝒜t,t),\displaystyle\mathcal{A}_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\hat{\mathcal{A}}_{0}+(\sqrt{1-\bar{\alpha}_{t-1}})\hat{s}_{\theta}(\mathcal{A}_{t},t),

(12)

𝐄t−1=α¯t−1​𝐄^0+(1−α¯t−1)​c^ζ​(𝐄t,t).\displaystyle\mathbf{E}_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\hat{\mathbf{E}}_{0}+(\sqrt{1-\bar{\alpha}_{t-1}})\hat{c}_{\zeta}(\mathbf{E}_{t},t).

After 𝒜t−1\mathcal{A}_{t-1} and 𝐄t−1\mathbf{E}_{t-1} are sampled from Eq. (12) at current step, they are further refined using the proposed ARGM, as detailed in Section 3.4. Upon completion of the sampling process, the reconstructed HR-HSI 𝒵0\mathcal{Z}_{0} is derived from 𝒜0×3𝐄0\mathcal{A}_{0}\times_{3}\mathbf{E}_{0}. The proposed method is summarized in Algorithm 1.

Remark: To accelerate convergence, we integrate the Adam optimizer [11] into Eq. (9) with the following updates:

𝐦t−1(𝒜)=β1​𝐦t(𝒜)+(1−β1)​∇𝒜tℒ​(𝒜^0,𝐄^0,𝒳,𝒴),\displaystyle\mathbf{m}_{t-1}^{(\mathcal{A})}=\beta_{1}\mathbf{m}_{t}^{(\mathcal{A})}+(1-\beta_{1})\nabla_{\mathcal{A}_{t}}\mathcal{L}(\hat{\mathcal{A}}_{0},\hat{\mathbf{E}}_{0},\mathcal{X},\mathcal{Y}),

(13)

𝐯t−1(𝒜)=β2​𝐯t(𝒜)+(1−β2)​∇𝒜tℒ​(𝒜^0,𝐄^0,𝒳,𝒴)2,\displaystyle\mathbf{v}_{t-1}^{(\mathcal{A})}=\beta_{2}\mathbf{v}_{t}^{(\mathcal{A})}+(1-\beta_{2})\nabla_{\mathcal{A}_{t}}\mathcal{L}(\hat{\mathcal{A}}_{0},\hat{\mathbf{E}}_{0},\mathcal{X},\mathcal{Y})^{2},

𝐦t−1(𝐄)=β1​𝐦t(𝐄)+(1−β1)​∇𝐄tℒ​(𝒜^0,𝐄^0,𝒳,𝒴),\displaystyle\mathbf{m}_{t-1}^{(\mathbf{E})}=\beta_{1}\mathbf{m}_{t}^{(\mathbf{E})}+(1-\beta_{1})\nabla_{\mathbf{E}_{t}}\mathcal{L}(\hat{\mathcal{A}}_{0},\hat{\mathbf{E}}_{0},\mathcal{X},\mathcal{Y}),

𝐯t−1(𝐄)=β2​𝐯t(𝐄)+(1−β2)​∇𝐄tℒ​(𝒜^0,𝐄^0,𝒳,𝒴)2,\displaystyle\mathbf{v}_{t-1}^{(\mathbf{E})}=\beta_{2}\mathbf{v}_{t}^{(\mathbf{E})}+(1-\beta_{2})\nabla_{\mathbf{E}_{t}}\mathcal{L}(\hat{\mathcal{A}}_{0},\hat{\mathbf{E}}_{0},\mathcal{X},\mathcal{Y})^{2},

where 𝐦t−1\mathbf{m}_{t-1} and 𝐯t−1\mathbf{v}_{t-1} represent the first and second moment estimates in the Adam optimizer, with superscripts (𝒜)(\mathcal{A}) and (𝐄)(\mathbf{E}) denoting the corresponding spatial and spectral matrices. The parameters β1\beta_{1} and β2\beta_{2} are the decay rates for these moment estimates.

We then rectify the first and second moment estimates as:

𝐦^t−1(𝒜)=𝐦t−1(𝒜)1−β1T−t,𝐯^t−1(𝒜)=𝐯t−1(𝒜)1−β2T−t,\displaystyle\hat{\mathbf{m}}_{t-1}^{(\mathcal{A})}=\frac{\mathbf{m}_{t-1}^{(\mathcal{A})}}{1-\beta_{1}^{T-t}},\hat{\mathbf{v}}_{t-1}^{(\mathcal{A})}=\frac{\mathbf{v}_{t-1}^{(\mathcal{A})}}{1-\beta_{2}^{T-t}},

(14)

𝐦^t−1(𝐄)=𝐦t−1(𝐄)1−β1T−t,𝐯^t−1(𝐄)=𝐯t−1(𝐄)1−β2T−t.\displaystyle\hat{\mathbf{m}}_{t-1}^{(\mathbf{E})}=\frac{\mathbf{m}_{t-1}^{(\mathbf{E})}}{1-\beta_{1}^{T-t}},\hat{\mathbf{v}}_{t-1}^{(\mathbf{E})}=\frac{\mathbf{v}_{t-1}^{(\mathbf{E})}}{1-\beta_{2}^{T-t}}.

Finally, we rewrite Eq. (9) as:

s^θ​(𝒜t,t)=sθ​(𝒜t,t)−ρ1​𝐦^t−1(𝒜)𝐯^t−1(𝒜)+ϵ,\displaystyle\hat{s}_{\theta}(\mathcal{A}_{t},t)=s_{\theta}(\mathcal{A}_{t},t)-\rho_{1}\frac{\hat{\mathbf{m}}_{t-1}^{(\mathcal{A})}}{\sqrt{\hat{\mathbf{v}}_{t-1}^{(\mathcal{A})}}+\epsilon},

(15)

c^ζ​(𝐄t,t)=cζ​(𝐄t,t)−ρ2​𝐦^t−1(𝐄)𝐯^t−1(𝐄)+ϵ,\displaystyle\hat{c}_{\zeta}(\mathbf{E}_{t},t)=c_{\zeta}(\mathbf{E}_{t},t)-\rho_{2}\frac{\hat{\mathbf{m}}_{t-1}^{(\mathbf{E})}}{\sqrt{\hat{\mathbf{v}}_{t-1}^{(\mathbf{E})}}+\epsilon},

where ϵ\epsilon is a small constant introduced to avoid division by zero.

### 3.4 ARGM

Unlike existing diffusion posterior sampling methods [2, 23], which update only a single intermediate sample at each sampling step, our approach requires simultaneous updates of both the spectral basis and the reduced coefficient. This introduces additional instability into the sampling process, as both components must be adjusted to align with each other in each iteration. To alleviate this issue, we introduce the Adaptive Residual Guided Module (ARGM), which facilitates better alignment between the spectral and spatial components, resulting in more stable convergence.

Specifically, after obtaining 𝒜t−1\mathcal{A}_{t-1} and 𝐄t−1\mathbf{E}_{t-1} at the tt-th sampling step, we compute a residual loss that quantifies the discrepancy between the reconstructed two components and the target to refine them. The residual loss can be calculated by:

ℒ⁡(𝒜t−1,𝐄t−1,𝒳,𝒴)\displaystyle\mathcal{L}(\mathcal{A}_{t-1},\mathbf{E}_{t-1},\mathcal{X},\mathcal{Y})
=‖ℋ⁡(𝒜t−1×3𝐄t−1)−𝒳‖22\displaystyle=\ \|\mathcal{H}(\mathcal{A}_{t-1}\times_{3}\mathbf{E}_{t-1})-\mathcal{X}\|^{2}_{2}

(16)

+λ2∥𝒜t−1×3𝐄t−1×3𝐑−𝒴∥22,\displaystyle+\lambda_{2}\|\mathcal{A}_{t-1}\times_{3}\mathbf{E}_{t-1}\times_{3}\mathbf{R}-\mathcal{Y}\|^{2}_{2},

where λ2\lambda_{2} is a weight balancing the two objectives. The first term ensures that the predicted LR-HSI ℋ⁡(𝒜t−1×3𝐄t−1)\mathcal{H}(\mathcal{A}_{t-1}\times_{3}\mathbf{E}_{t-1}) is consistent with the observed image 𝒳\mathcal{X}, while the second term penalizes the discrepancy between the predicted HR-MSI (𝒜t−1×3𝐄t−1×3𝐑\mathcal{A}_{t-1}\times_{3}\mathbf{E}_{t-1}\times_{3}\mathbf{R}) and the target 𝒴\mathcal{Y}. This ensures that the spectral and spatial components are aligned both individually and in combination.

Then, the spatial and spectral components are updated by:

𝒜t−1:=𝒜t−1−ρ1r​∇𝒜t−1ℒ​(𝒜t−1,𝐄t−1,𝒳,𝒴),\displaystyle\mathcal{A}_{t-1}:=\mathcal{A}_{t-1}-\frac{\rho_{1}}{r}\nabla_{\mathcal{A}_{t-1}}\mathcal{L}(\mathcal{A}_{t-1},\mathbf{E}_{t-1},\mathcal{X},\mathcal{Y}),

(17)

𝐄t−1:=𝐄t−1−ρ2r​∇𝐄t−1ℒ​(𝒜t−1,𝐄t−1,𝒳,𝒴),\displaystyle\mathbf{E}_{t-1}:=\mathbf{E}_{t-1}-\frac{\rho_{2}}{r}\nabla_{\mathbf{E}_{t-1}}\mathcal{L}(\mathcal{A}_{t-1},\mathbf{E}_{t-1},\mathcal{X},\mathcal{Y}),

where rr is a ratio that controls the relative step sizes for updating the spatial component and the spectral component, respectively.

In this way, the ARGM computes the residual between the reconstructed two components and the target at each step, and then uses the residual-guided function to adjust both components. On the one hand, this brings the components closer to the target direction, reducing the instability that arises from updating both components simultaneously. On the other hand, it compensates for the prediction errors introduced by the model at each sampling step, thus making the model more robust and less sensitive to noise and discrepancies.

By iteratively refining the spatial and spectral components using residual-guided function, the ARGM helps stabilize the convergence of the diffusion sampling process, leading to improved performance and robustness in the HSI-MSI fusion task.

*Algorithm 1 Reverse Diffusion Process of ARGS-Diff*

Input: LR-HSI 𝒳\mathcal{X}, HR-MSI 𝒴\mathcal{Y}, 𝒜T\mathcal{A}_{T} and 𝐄T\mathbf{E}_{T} sampled from 𝒩⁡(0,I)\mathcal{N}(0,I), spatial network sθs_{\theta}, spectral network cζc_{\zeta}, balance weight λ1\lambda_{1}, λ2\lambda_{2}, step size ρ1\rho_{1}, ρ2\rho_{2}, ratio rr, Adam params 𝐦T(𝒜)\mathbf{m}_{T}^{(\mathcal{A})}, 𝐯T(𝒜)\mathbf{v}_{T}^{(\mathcal{A})}, 𝐦T(𝐄)\mathbf{m}_{T}^{(\mathbf{E})}, 𝐯T(𝐄)\mathbf{v}_{T}^{(\mathbf{E})}, β1\beta_{1}, β2\beta_{2}, ϵ\epsilon

Output: HR-HSI 𝒵0\mathcal{Z}_{0}

Initialize 𝐦T(𝒜)=𝐯T(𝒜)=𝐦T(𝐄)=𝐯T(𝐄)=𝟎\mathbf{m}_{T}^{(\mathcal{A})}=\mathbf{v}_{T}^{(\mathcal{A})}=\mathbf{m}_{T}^{(\mathbf{E})}=\mathbf{v}_{T}^{(\mathbf{E})}=\mathbf{0}

for *t=T,T−1,⋯1t=T,T-1,\cdots 1* do

   step 1: estimate 𝒜^0\hat{\mathcal{A}}_{0}, 𝐄^0\hat{\mathbf{E}}_{0} by Eq. (11)

   step 2: calculate ℒ⁡(𝒜^0,𝐄^0,𝒳,𝒴)\mathcal{L}(\hat{\mathcal{A}}_{0},\hat{\mathbf{E}}_{0},\mathcal{X},\mathcal{Y}) by Eq.(10)

   step 3: calculate 𝐦^t−1(𝒜)\hat{\mathbf{m}}_{t-1}^{(\mathcal{A})}, 𝐯^t−1(𝒜)\hat{\mathbf{v}}_{t-1}^{(\mathcal{A})}, 𝐦^t−1(𝐄)\hat{\mathbf{m}}_{t-1}^{(\mathbf{E})}, 𝐯^t−1(𝐄)\hat{\mathbf{v}}_{t-1}^{(\mathbf{E})} by Eq. (13), Eq. (14)

   step 4: estimate s^θ​(𝒜t,t)\hat{s}_{\theta}(\mathcal{A}_{t},t), c^ζ​(𝐄t,t)\hat{c}_{\zeta}(\mathbf{E}_{t},t) by Eq. (15)

   step 5: sample 𝒜t−1\mathcal{A}_{t-1}, 𝐄t−1\mathbf{E}_{t-1} by Eq. (12)

   step 6: calculate ℒ⁡(𝒜t−1,𝐄t−1,𝒳,𝒴)\mathcal{L}(\mathcal{A}_{t-1},\mathbf{E}_{t-1},\mathcal{X},\mathcal{Y}) by Eq. (16)

   step 7: update 𝒜t−1\mathcal{A}_{t-1}, 𝐄t−1\mathbf{E}_{t-1} by Eq. (17)

end for

return 𝒵0=𝒜0×3𝐄0\mathcal{Z}_{0}=\mathcal{A}_{0}\times_{3}\mathbf{E}_{0}

## 4 Experiment

### 4.1 Datasets and Evaluation Metrics

We conducted a series of simulated experiments using three public datasets to assess the performance of the proposed method for HSI fusion: Pavia University, Chikusei, and KSC. The original dimensions of these datasets are 610×340×103610\times 340\times 103, 2517×2335×1282517\times 2335\times 128, and 512×614×176512\times 614\times 176. We cropped the central regions of each to obtain HSIs with sizes 256×256×103256\times 256\times 103, 256×256×128256\times 256\times 128, and 256×256×176256\times 256\times 176. The LR-HSI was generated by downsampling the HR-HSI using bicubic interpolation with a scale factor of 4, and the HR-MSI was obtained by performing mode-3 tensor multiplication of the HR-HSI with a simulated spectral response function. Additionally, to evaluate the method’s real-world applicability, we conducted experiments on the DFC2018 Houston dataset, which provides real hyperspectral data. The following four commonly used evaluation metrics were employed to assess model performance: peak signal-to-noise ratio (PSNR), spectral angle mapper (SAM), relative dimensionless global error in synthesis (ERGAS) , and structural similarity (SSIM).

### 4.2 Implementation Details

All experiments were carried out on an NVIDIA GeForce RTX 4090 GPU. To test the robustness of the model to noise, we added noise to all LR-HSI and HR-MSI inputs with a Signal-to-Noise Ratio (SNR) of 35. The diffusion sampling step TT was set to 500, using an exponential noise schedule [20]. The balance weights λ1\lambda_{1} and λ2\lambda_{2} were both set to 1, while the step sizes ρ1\rho_{1} and ρ2\rho_{2} were set to 0.05, and the ratio rr was set to 10. The Adam optimizer was used with β1=0.9\beta_{1}=0.9 and β2=0.999\beta_{2}=0.999, and ϵ=1×10−8\epsilon=1\times 10^{-8}. The number of subspace dimensions dd was set to 8.

### 4.3 Experimental Results

#### 4.3.1 Results on Simulated Data

We compare the proposed method with six well-established approaches, including traditional methods such as CNMF [40] and HySure [28], deep learning-based methods like CuCaNet [39] and MIAE [13], and diffusion model-based approaches such as PLRDiff [23] and S2CycleDiff [21]. The quantitative results are summarized in Tables 1, 2, and 3. From these results, it is clear that the proposed method consistently outperforms all other competing approaches across all evaluation metrics on three different datasets. In terms of PSNR, the proposed method achieves the highest values, surpassing the second-best method, MIAE, by 1.31 dB, 1.37 dB, and 1.27 dB on the respective datasets. This demonstrates the superior ability of ARGS-Diff to reconstruct high-fidelity hyperspectral images, preserving both spectral and spatial details with greater accuracy. When examining the computational efficiency, traditional methods such as CNMF and HySure offer faster inference speeds, but they result in poorer fusion performance. Deep learning methods like CuCaNet and MIAE show improved accuracy, but at the cost of longer processing times. Diffusion model-based methods like PLRDiff and S2CycleDiff also demand significant inference time, while the proposed method achieves high-quality fusion results with a reduced runtime.

Representative visual results are illustrated in Figure 2, showcasing the superior image quality of proposed method. To further highlight the performance, we include error maps that demonstrate the proposed method’s ability to minimize bias across all three datasets, resulting in high-fidelity HR-HSI reconstructions. Overall, the proposed approach excels in both quantitative and qualitative evaluations, establishing its superiority in HSI-MSI fusion.

*Table 1: The quantitative results on the Pavia dataset. The best and second-best values are highlighted.*

Methods
PSNR↑\uparrow
SAM↓\downarrow
EGARS↓\downarrow
SSIM↑\uparrow
Time (s)

CNMF
38.82
3.13
1.95
0.974
7

HySure
37.85
3.19
2.32
0.970
5

CuCaNet
38.86
2.95
1.98
0.970
401

MIAE
41.02
2.73
1.64
0.975
20

PLRDiff
40.11
3.40
1.94
0.970
79

S2CycleDiff
38.77
3.21
2.13
0.970
297

ARGS-Diff
42.33
2.64
1.49
0.977
12

*Table 2: The quantitative results on the Chikusei dataset. The best and second-best values are highlighted.*

Methods
PSNR↑\uparrow
SAM↓\downarrow
EGARS↓\downarrow
SSIM↑\uparrow
Time (s)

CNMF
38.14
2.17
2.59
0.967
11

HySure
38.71
2.25
2.65
0.967
5

CuCaNet
39.01
2.09
2.55
0.968
433

MIAE
40.53
1.95
2.51
0.969
24

PLRDiff
38.44
2.40
2.79
0.963
79

S2CycleDiff
39.73
2.08
2.53
0.968
318

ARGS-Diff
41.90
1.77
2.10
0.970
12

*Table 3: The quantitative results on the KSC dataset. The best and second-best values are highlighted.*

Methods
PSNR↑\uparrow
SAM↓\downarrow
EGARS↓\downarrow
SSIM↑\uparrow
Time (s)

CNMF
39.88
3.47
1.86
0.977
15

HySure
40.11
3.17
1.80
0.978
6

CuCaNet
40.39
3.05
1.72
0.977
474

MIAE
42.36
2.58
1.42
0.979
32

PLRDiff
40.07
3.63
1.83
0.977
79

S2CycleDiff
40.19
3.15
1.79
0.978
349

ARGS-Diff
43.63
2.54
1.25
0.980
13

![Refer to caption](2505.11800v1/imgs/pavia/LRHSI.png)

![Refer to caption](2505.11800v1/imgs/pavia/CNMF.png)

![Refer to caption](2505.11800v1/imgs/pavia/HySure.png)

![Refer to caption](2505.11800v1/imgs/pavia/CuCaNet.png)

![Refer to caption](2505.11800v1/imgs/pavia/MIAE.png)

![Refer to caption](2505.11800v1/imgs/pavia/PLRDiff.png)

![Refer to caption](2505.11800v1/imgs/pavia/S2CycleDiff.png)

![Refer to caption](2505.11800v1/imgs/pavia/Ours.png)

![Refer to caption](2505.11800v1/imgs/pavia/REF.png)

![Refer to caption](2505.11800v1/imgs/pavia/LRHSI_err.png)

![Refer to caption](2505.11800v1/imgs/pavia/CNMF_err.png)

![Refer to caption](2505.11800v1/imgs/pavia/HySure_err.png)

![Refer to caption](2505.11800v1/imgs/pavia/CuCaNet_err.png)

![Refer to caption](2505.11800v1/imgs/pavia/MIAE_err.png)

![Refer to caption](2505.11800v1/imgs/pavia/PLRDiff_err.png)

![Refer to caption](2505.11800v1/imgs/pavia/S2CycleDiff_err.png)

![Refer to caption](2505.11800v1/imgs/pavia/Ours_err.png)

![Refer to caption](2505.11800v1/imgs/pavia/REF_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/LRHSI.png)

![Refer to caption](2505.11800v1/imgs/chikusei/CNMF.png)

![Refer to caption](2505.11800v1/imgs/chikusei/HySure.png)

![Refer to caption](2505.11800v1/imgs/chikusei/CuCaNet.png)

![Refer to caption](2505.11800v1/imgs/chikusei/MIAE.png)

![Refer to caption](2505.11800v1/imgs/chikusei/PLRDiff.png)

![Refer to caption](2505.11800v1/imgs/chikusei/S2CycleDiff.png)

![Refer to caption](2505.11800v1/imgs/chikusei/Ours.png)

![Refer to caption](2505.11800v1/imgs/chikusei/REF.png)

![Refer to caption](2505.11800v1/imgs/chikusei/LRHSI_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/CNMF_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/HySure_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/CuCaNet_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/MIAE_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/PLRDiff_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/S2CycleDiff_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/Ours_err.png)

![Refer to caption](2505.11800v1/imgs/chikusei/REF_err.png)

![Refer to caption](2505.11800v1/imgs/ksc/LRHSI.png)

![Refer to caption](2505.11800v1/imgs/ksc/CNMF.png)

![Refer to caption](2505.11800v1/imgs/ksc/HySure.png)

![Refer to caption](2505.11800v1/imgs/ksc/CuCaNet.png)

![Refer to caption](2505.11800v1/imgs/ksc/MIAE.png)

![Refer to caption](2505.11800v1/imgs/ksc/PLRDiff.png)

![Refer to caption](2505.11800v1/imgs/ksc/S2CycleDiff.png)

![Refer to caption](2505.11800v1/imgs/ksc/Ours.png)

![Refer to caption](2505.11800v1/imgs/ksc/REF.png)

![Refer to caption](2505.11800v1/imgs/ksc/LRHSI_err.png)

*LR-HSI*

![Refer to caption](2505.11800v1/imgs/ksc/CNMF_err.png)

*CNMF*

![Refer to caption](2505.11800v1/imgs/ksc/HySure_err.png)

*HySure*

![Refer to caption](2505.11800v1/imgs/ksc/CuCaNet_err.png)

*CuCaNet*

![Refer to caption](2505.11800v1/imgs/ksc/MIAE_err.png)

*MIAE*

![Refer to caption](2505.11800v1/imgs/ksc/PLRDiff_err.png)

*PLRDiff*

![Refer to caption](2505.11800v1/imgs/ksc/S2CycleDiff_err.png)

*S2CycleDiff*

![Refer to caption](2505.11800v1/imgs/ksc/Ours_err.png)

*ARGS-Diff*

![Refer to caption](2505.11800v1/imgs/ksc/REF_err.png)

*Reference*

*Figure 2: Visual results and reconstruction error maps obtained by different methods. The first, third, and fifth rows are the pseudo-color images of Pavia, Chikusei, and KSC datasets, and the second, fourth, and sixth rows correspond to their error maps.*

![Refer to caption](2505.11800v1/imgs/houston/LRHSI.png)

*LR-HSI*

![Refer to caption](2505.11800v1/imgs/houston/HRMSI.png)

*HR-MSI*

![Refer to caption](2505.11800v1/imgs/houston/CNMF.png)

*CNMF*

![Refer to caption](2505.11800v1/imgs/houston/HySure.png)

*HySure*

![Refer to caption](2505.11800v1/imgs/houston/CuCaNet.png)

*CuCaNet*

![Refer to caption](2505.11800v1/imgs/houston/MIAE.png)

*MIAE*

![Refer to caption](2505.11800v1/imgs/houston/PLRDiff.png)

*PLRDiff*

![Refer to caption](2505.11800v1/imgs/houston/Ours.png)

*ARGS-Diff*

*Figure 3: Visual results obtained by different methods on the Houston dataset.*

#### 4.3.2 Results on Real-Data

Figure 3 presents the reconstructed pseudo-color images from the DFC2018 Houston dataset, generated using various methods. The reconstructed HSIs from the comparison methods exhibit varying degrees of deficiencies in color consistency, naturalness, and saturation control. In contrast, ARGS-Diff effectively preserves the authenticity and vibrancy of colors while avoiding pseudo-color artifacts and color distortions. The resulting spectral images exhibit high quality, closely resembling real-world scenes. This demonstrates ARGS-Diff’s superior ability to preserve fine details and maintain overall color balance, ensuring the consistency of both spatial and spectral information in the reconstructed hyperspectral images. These results highlight the effectiveness of the proposed method, particularly in real-world applications, where accurate color reproduction and detail preservation are critical.

### 4.4 Ablation Study

In this section, we assess the impact of the Adaptive Residual Guided Module (ARGM) on the performance of the proposed method. The results, summarized in Table 4, clearly demonstrate that incorporating ARGM leads to a significant improvement in the quality of the reconstructed images. Specifically, the inclusion of ARGM increased the PSNR by 0.47 dB, 0.57 dB, and 0.61 dB across the three datasets, while the SAM improved by 0.11, 0.12, and 0.16, underscoring its positive contribution to the fusion performance. Importantly, the computational overhead introduced by ARGM is minimal, adding only 1-2 seconds to the overall runtime, which has a negligible impact on the total processing time.

Additionally, Figure 4 presents the reconstruction results of the reduced coefficient 𝒜\mathcal{A} at different sampling steps t={500,400,300,200,100,0}t=\{500,400,300,200,100,0\} on the Pavia dataset. It is evident that, with the incorporation of ARGM, the reconstructed reduced coefficient becomes more informative and detailed. This visual improvement further confirms the significant enhancement brought by ARGM, which facilitates more accurate and robust fusion, contributing to better alignment of the spectral and spatial components.

![Refer to caption](2505.11800v1/imgs/ARGM_wo/x_0999.png)

![Refer to caption](2505.11800v1/imgs/ARGM_wo/x_0799.png)

![Refer to caption](2505.11800v1/imgs/ARGM_wo/x_0599.png)

![Refer to caption](2505.11800v1/imgs/ARGM_wo/x_0479.png)

![Refer to caption](2505.11800v1/imgs/ARGM_wo/x_0199.png)

![Refer to caption](2505.11800v1/imgs/ARGM_wo/x_0001.png)

![Refer to caption](2505.11800v1/imgs/ARGM_w/x_0999.png)

*t=500t=500*

![Refer to caption](2505.11800v1/imgs/ARGM_w/x_0799.png)

*t=400t=400*

![Refer to caption](2505.11800v1/imgs/ARGM_w/x_0599.png)

*t=300t=300*

![Refer to caption](2505.11800v1/imgs/ARGM_w/x_0399.png)

*t=200t=200*

![Refer to caption](2505.11800v1/imgs/ARGM_w/x_0199.png)

*t=100t=100*

![Refer to caption](2505.11800v1/imgs/ARGM_w/x_0001.png)

*t=0t=0*

*Figure 4: Intermediate results of the reduced coefficient 𝒜\mathcal{A}. The first and second rows represent ’w/o ARGM’ and ’w/ ARGM’.*

*Table 4: The quantitative ablation results of ARGM.*

Datasets
Methods
PSNR
SAM
Time (s)

Pavia
w/o ARGM
41.86
2.75
11

w/ ARGM
42.33
2.64
12

Chikusei
w/o ARGM
41.33
1.89
11

w/ ARGM
41.90
1.77
12

KSC
w/o ARGM
43.02
2.70
11

w/ ARGM
43.63
2.54
13

### 4.5 Analysis of Hyperparameters

dd and TT. In this part, we examine the parameter sensitivity related to the number of subspace dimensions and the number of sampling steps. Using the Pavia dataset as a case study, the results are shown in Figure 5. As dd increases from 1 to 8, performance improves significantly; however, beyond this point, performance gradually declines. This drop can be attributed to the increasing number of parameters that need to be updated when d becomes too large, which complicates the convergence process. Consequently, we set dd to 8.
For the number of sampling steps TT, we observe substantial improvement as TT increases from 100 to 500. However, beyond 500 steps, the performance reaches a plateau and may even slightly decrease, suggesting that the model has reached saturation and further increases may lead to overfitting. Therefore, we set TT to 500.

![Refer to caption](2505.11800v1/imgs/RT/d.png)

![Refer to caption](2505.11800v1/imgs/RT/T2.png)

*Figure 5: Sensitivity analysis of the parameters dd and TT.*

ρ1\rho_{1} and ρ2\rho_{2}. In Eq. (9), we introduce two hyperparameters, ρ1\rho_{1} and ρ2\rho_{2}, which control the update step sizes for the predicted components. To determine an appropriate combination of values, we conduct a hyperparameter search over ρ1,ρ2∈{0.01,0.02,0.03,0.04,0.05,0.06,0.07}\rho_{1},\rho_{2}\in\{0.01,0.02,0.03,0.04,0.05,0.06,0.07\} using the Pavia dataset. The PSNR results are shown in Table 5, from which we observe that the best performance is achieved when ρ1=ρ2=0.05\rho_{1}=\rho_{2}=0.05. Therefore, for the Pavia dataset, we set ρ1\rho_{1} and ρ2\rho_{2} to 0.05.

*Table 5: Sensitivity analysis of the parameters ρ1\rho_{1} and ρ2\rho_{2} on Pavia dataset. The best result is highlighted.*

                                ρ
                                1

                                ρ
                                2

0.01
0.02
0.03
0.04
0.05
0.06
0.07

0.01
33.70
38.96
34.09
36.88
34.67
35.07
31.84

0.02
40.75
40.69
40.82
41.15
41.41
41.50
40.73

0.03
41.35
41.55
41.71
41.57
41.59
41.91
41.80

0.04
41.75
41.89
41.77
41.93
41.90
42.10
41.69

0.05
41.72
42.00
42.13
42.25
42.33
42.27
41.67

0.06
41.66
41.97
42.13
42.15
42.23
42.11
41.95

0.07
41.67
41.80
41.97
42.19
42.14
41.78
41.94

### 4.6 Inference Time and Model Scale Anaiysis

In this section, we compare the proposed method with other diffusion model approaches in terms of parameter count, memory consumption, and runtime. Table 6 summarizes the results for the PLRDiff, S2CycleDiff, and the proposed ARGS-Diff, using the Pavia dataset as an example. The proposed method has a parameter count of just 21.85M, which is approximately 1/20th and 1/30th of the parameter counts for PLRDiff and S2CycleDiff, respectively. It also consumes only 2.11GB of memory, significantly less than the other two methods. Furthermore, the proposed method completes sampling in just 12 seconds, significantly outperforming PLRDiff (79 seconds) and S2CycleDiff (297 seconds) in terms of processing time. Overall, the proposed method not only excels in fusion accuracy but also offers substantial advantages in terms of model size, memory usage, and runtime, making it highly suitable for real-world applications.

*Table 6: The model size and computational consumption of the proposed method and other diffusion model-based methdods.*

Methods
PSNR
Params (M)
Memory (GB)
Time (s)

PLRDiff
40.11
391.05
6.21
79

S2CycleDiff
38.77
606.44
23.34
297

ARGS-Diff
42.33
21.85
2.11
12

## 5 Conclusion

In this paper, we propose a self-learning Adaptive Residual Guided Subspace Diffusion Model (ARGS-Diff) for HSI-MSI fusion. We carefully design two lightweight spectral and spatial networks, training them using the observed LR-HSI and HR-MSI to extract the rich spectral and spatial information inherent in each. These well-trained networks are then employed to reconstruct two low-dimensional components: the spectral basis and the reduced coefficient. The HR-HSI is obtained by taking the product of these two components. Additionally, we introduce the Adaptive Residual Guided Module (ARGM), which further enhances the sampling process by stabilizing the updates of the spectral and spatial components. Extensive experimental evaluations demonstrate the superiority of the proposed method, achieving remarkable performance in the field of HSI-MSI fusion.

## References

- [1]

Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein.

Universal guidance for diffusion models.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 843–852, 2023.

- [2]

Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye.

Diffusion posterior sampling for general noisy inverse problems.

In *The Eleventh International Conference on Learning Representations*, 2023.

- [3]

Adele Cutler and Leo Breiman.

Archetypal analysis.

*Technometrics*, 36(4):338–347, 1994.

- [4]

Laura M Dale, André Thewis, Christelle Boudry, Ioan Rotar, Pierre Dardenne, Vincent Baeten, and Juan A Fernández Pierna.

Hyperspectral imaging applications in agriculture and agro-food product quality and safety control: A review.

*Applied Spectroscopy Reviews*, 48(2):142–159, 2013.

- [5]

Prafulla Dhariwal and Alexander Nichol.

Diffusion models beat gans on image synthesis.

*Advances in neural information processing systems*, 34:8780–8794, 2021.

- [6]

Alexey Dosovitskiy.

An image is worth 16x16 words: Transformers for image recognition at scale.

*arXiv preprint arXiv:2010.11929*, 2020.

- [7]

Wen-jin Guo, Weiying Xie, Kai Jiang, Yunsong Li, Jie Lei, and Leyuan Fang.

Toward stable, interpretable, and lightweight hyperspectral super-resolution.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 22272–22281, 2023.

- [8]

Jonathan Ho, Ajay Jain, and Pieter Abbeel.

Denoising diffusion probabilistic models.

*Advances in neural information processing systems*, 33:6840–6851, 2020.

- [9]

Xin Huang, Dawei Wen, Jiayi Li, and Rongjun Qin.

Multi-level monitoring of subtle urban changes for the megacities of china using high-resolution multi-view satellite imagery.

*Remote Sensing of Environment*, 196:56–75, 2017.

- [10]

Sami Khanal, Kushal Kc, John P Fulton, Scott Shearer, and Erdal Ozkan.

Remote sensing in agriculture—accomplishments, limitations, and opportunities.

*Remote Sensing*, 12(22):3783, 2020.

- [11]

Diederik P Kingma.

Adam: A method for stochastic optimization.

*arXiv preprint arXiv:1412.6980*, 2014.

- [12]

Jun Li, Yanqiu Pei, Shaohua Zhao, Rulin Xiao, Xiao Sang, and Chengye Zhang.

A review of remote sensing for environmental monitoring in china.

*Remote Sensing*, 12(7):1130, 2020.

- [13]

Jianjun Liu, Zebin Wu, Liang Xiao, and Xiao-Jun Wu.

Model inspired autoencoder for unsupervised hyperspectral image super-resolution.

*IEEE Transactions on Geoscience and Remote Sensing*, 60:1–12, 2022.

- [14]

Lei Liu, Jun Zhou, Dong Jiang, Dafang Zhuang, Lamin R Mansaray, and Bing Zhang.

Targeting mineral resources with remote sensing and field data in the xiemisitai area, west junggar, xinjiang, china.

*Remote sensing*, 5(7):3156–3171, 2013.

- [15]

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo.

Swin transformer: Hierarchical vision transformer using shifted windows.

In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 10012–10022, 2021.

- [16]

Laetitia Loncan, Luis B De Almeida, José M Bioucas-Dias, Xavier Briottet, Jocelyn Chanussot, Nicolas Dobigeon, Sophie Fabre, Wenzhi Liao, Giorgio A Licciardi, Miguel Simoes, et al.

Hyperspectral pansharpening: A review.

*IEEE Geoscience and Remote Sensing Magazine*, 3(3):27–46, 2015.

- [17]

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan LI, and Jun Zhu.

Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps.

In *Advances in Neural Information Processing Systems*, pages 5775–5787, 2022.

- [18]

Yuchun Miao, Lefei Zhang, Liangpei Zhang, and Dacheng Tao.

Dds2m: Self-supervised denoising diffusion spatio-spectral model for hyperspectral image restoration.

In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 12086–12096, 2023.

- [19]

Alexander Quinn Nichol and Prafulla Dhariwal.

Improved denoising diffusion probabilistic models.

In *International conference on machine learning*, pages 8162–8171. PMLR, 2021.

- [20]

Li Pang, Xiangyu Rui, Long Cui, Hongzhong Wang, Deyu Meng, and Xiangyong Cao.

Hir-diff: Unsupervised hyperspectral image restoration via improved diffusion models.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3005–3014, 2024.

- [21]

Jiahui Qu, Jie He, Wenqian Dong, and Jingyu Zhao.

S2cyclediff: Spatial-spectral-bilateral cycle-diffusion framework for hyperspectral image super-resolution.

In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 4623–4631, 2024.

- [22]

Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

U-net: Convolutional networks for biomedical image segmentation.

In *Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18*, pages 234–241. Springer, 2015.

- [23]

Xiangyu Rui, Xiangyong Cao, Li Pang, Zeyu Zhu, Zongsheng Yue, and Deyu Meng.

Unsupervised hyperspectral pansharpening via low-rank diffusion model.

*Information Fusion*, 107:102325, 2024.

- [24]

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi.

Image super-resolution via iterative refinement.

*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(4):4713–4726, 2022.

- [25]

Giuseppe Scarpa, Sergio Vitale, and Davide Cozzolino.

Target-adaptive cnn-based pansharpening.

*IEEE Transactions on Geoscience and Remote Sensing*, 56(9):5443–5457, 2018.

- [26]

Vijay P Shah, Nicolas H Younan, and Roger L King.

An efficient pan-sharpening method via a combined adaptive pca approach and contourlets.

*IEEE Transactions on Geoscience and Remote Sensing*, 46(5):1323–1335, 2008.

- [27]

Hojat Shirmard, Ehsan Farahbakhsh, R Dietmar Müller, and Rohitash Chandra.

A review of machine learning in processing remote sensing data for mineral exploration.

*Remote Sensing of Environment*, 268:112750, 2022.

- [28]

Miguel Simoes, José Bioucas-Dias, Luis B Almeida, and Jocelyn Chanussot.

A convex formulation for hyperspectral image superresolution via subspace-based regularization.

*IEEE Transactions on Geoscience and Remote Sensing*, 53(6):3373–3388, 2014.

- [29]

Jiaming Song, Chenlin Meng, and Stefano Ermon.

Denoising diffusion implicit models.

In *International Conference on Learning Representations*, 2021.

- [30]

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.

Score-based generative modeling through stochastic differential equations.

*arXiv preprint arXiv:2011.13456*, 2020.

- [31]

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

In *Advances in Neural Information Processing Systems*, 2017.

- [32]

Wu Wang, Weihong Zeng, Yue Huang, Xinghao Ding, and John Paisley.

Deep blind hyperspectral image fusion.

In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4150–4159, 2019.

- [33]

Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao.

Pyramid vision transformer: A versatile backbone for dense prediction without convolutions.

In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 568–578, 2021.

- [34]

Yinhuai Wang, Jiwen Yu, and Jian Zhang.

Zero-shot image restoration using denoising diffusion null-space model.

In *The Eleventh International Conference on Learning Representations*, 2023.

- [35]

Chanyue Wu, Dong Wang, Yunpeng Bai, Hanyu Mao, Ying Li, and Qiang Shen.

Hsr-diff: Hyperspectral image super-resolution via conditional diffusion models.

In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 7083–7093, 2023.

- [36]

Qi Xie, Minghao Zhou, Qian Zhao, Zongben Xu, and Deyu Meng.

Mhf-net: An interpretable deep network for multispectral and hyperspectral image fusion.

*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(3):1457–1473, 2020.

- [37]

Qiao Xue, Qingqing Ye, Haibo Hu, Youwen Zhu, and Jian Wang.

Ddrm: A continual frequency estimation mechanism with local differential privacy.

*IEEE Transactions on Knowledge and Data Engineering*, 35(7):6784–6797, 2022.

- [38]

Jingxiang Yang, Yong-Qiang Zhao, and Jonathan Cheung-Wai Chan.

Hyperspectral and multispectral image fusion via deep two-branches convolutional neural network.

*Remote Sensing*, 10(5):800, 2018.

- [39]

Jing Yao, Danfeng Hong, Jocelyn Chanussot, Deyu Meng, Xiaoxiang Zhu, and Zongben Xu.

Cross-attention in coupled unmixing nets for unsupervised hyperspectral super-resolution.

In *Computer Vision – ECCV 2020*, pages 208–224, 2020.

- [40]

Naoto Yokoya, Takehisa Yairi, and Akira Iwasaki.

Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion.

*IEEE Transactions on Geoscience and Remote Sensing*, 50(2):528–537, 2011.

- [41]

Xueting Zhang, Wei Huang, Qi Wang, and Xuelong Li.

Ssr-net: Spatial–spectral reconstruction network for hyperspectral and multispectral image fusion.

*IEEE Transactions on Geoscience and Remote Sensing*, 59(7):5953–5965, 2020.

- [42]

Man Zhou, Jie Huang, Keyu Yan, Danfeng Hong, Xiuping Jia, Jocelyn Chanussot, and Chongyi Li.

A general spatial-frequency learning framework for multimodal image fusion.

*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024.

Supplementary Material

## 6 Details of Network Architecture

![Refer to caption](2505.11800v1/spa.png)

*(a) Spatial Network*

![Refer to caption](2505.11800v1/spe.png)

*(b) Spectral Network*

*Figure 6: The detailed network structures using the Pavia dataset as an example.*

The detailed network structure is illustrated in Figure  6. To explore the most cost-effective spatial-spectral network, we conducted ablation experiments on these two networks. Specifically, for the spatial network, we performed an ablation study on the number of channels within the network, controlled by the channel_mult parameter. For the spectral network, we examined the effect of varying the dimensions of hidden layers, determined by the hidden_dims parameter. The experimental results are summarized in Table 7. Considering both accuracy and model size, we selected the spatial network with channel_mult = {1,2,3,4} and the spectral network with hidden_dims = {256,512,256}. (Note: Based on our existing experimental results, the ablation study of the spatial network is conducted with the optimal spectral network configuration by default, and vice versa for the spectral network ablation study.)

*Table 7: Ablation analysis of spectral and spatial networks.*

Networks
channel_mult/hidden_dims
PSNR
Params (M)

Spa. Net
1,1,2,4
41.55
15.60

1,2,3,4
42.33
21.46

1,2,4,8
42.40
57.03

Spe. Net
256,256
41.64
0.19

256,512,256
42.33
0.39

256,512,512,256
42.36
0.65

## 7 Impact of Two Components

Since our method simultaneously updates both the spectral and spatial components, we conduct an ablation study to investigate their individual contributions by evaluating the cases where only the spectral component or only the spatial component is updated. Specifically, when updating only the spectral component, we fix the spatial component by deriving 𝒜\mathcal{A} from 𝒴\mathcal{Y} (HR-MSI) using the Archetypal Analysis unmixing method [3]. Conversely, when updating only the spatial component, we fix the spectral component by estimating 𝐄\mathbf{E} from 𝒳\mathcal{X} (LR-HSI) following the approach of PLRDiff [23]. The final results are presented in Table 8, showing that when only the spectral component is updated, the fusion accuracy significantly decreases. Additionally, when only the spatial component is updated, the PSNR of the fusion results remains approximately 2 dB lower than when both components are updated simultaneously.

*Table 8: Ablation study on two components.*

Update
PSNR↑\uparrow
SAM↓\downarrow
EGARS↓\downarrow
SSIM↑\uparrow

Only 𝐄\mathbf{E}
21.77
11.82
13.59
0.296

Only 𝒜\mathcal{A}
40.53
3.28
1.77
0.969

𝐄\mathbf{E} & 𝒜\mathcal{A}
42.33
2.64
1.49
0.977

## 8 Sensitivity Analysis of Hyperparameters

We present the parameter analysis of the balance weights λ1\lambda_{1} and λ2\lambda_{2} in Table 9. Since the performance is optimal and relatively stable around (1,1)(1,1), we set both values to 1.

*Table 9: Sensitivity analysis of the parameters λ1\lambda_{1} and λ2\lambda_{2}.*

                              λ
                              2

                              λ
                              1

0.1
0.5
1
2
5

0.1
42.08
42.22
42.33
41.91
41.44

0.5
42.15
42.31
42.25
41.90
41.63

1
42.26
42.30
42.33
41.99
41.59

2
42.04
42.06
42.13
42.33
41.40

5
41.61
41.80
41.41
41.19
40.00

<!-- END-OF-PAPER:88c4203a3fa0 -->
