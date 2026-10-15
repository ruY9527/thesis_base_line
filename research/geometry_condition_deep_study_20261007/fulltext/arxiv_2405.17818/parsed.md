Hyperspectral and multispectral image fusion with arbitrary resolution through self-supervised representations

    ![](/static/base/1.0.1/images/icons/smileybones-small.svg)
    arXiv is now an independent nonprofit!
    Learn more
    ×

    License: CC BY 4.0

arXiv:2405.17818v2 [cs.CV] 25 Nov 2024

# Hyperspectral and multispectral image fusion with arbitrary resolution through self-supervised representations

Ting Wang

Affiliation: Department of Statistics and Data Science, Southern University of Science and Technology, Shenzhen, P.R. China

  
Zipei Yan

Affiliation: Department of Electronic Engineering, The Chinese University of Hong Kong, Hong Kong SAR, P.R. China

  
Jizhou Li

Affiliation: Department of Electronic Engineering, The Chinese University of Hong Kong, Hong Kong SAR, P.R. China

  
Xile Zhao

Affiliation: School of Mathematical Science, University of Electronic Science and Technology of China, Chengdu, P.R. China

  
Chao Wang

Email: wangc6@sustech.edu.cn

Affiliation: Department of Statistics and Data Science, Southern University of Science and Technology, Shenzhen, P.R. China

  
Michael Ng

Affiliation: Department of Mathematics, Hong Kong Baptist University, Hong Kong SAR, P.R. China

###### Abstract

The fusion of a low-resolution hyperspectral image (LR-HSI) with a high-resolution multispectral image (HR-MSI) has emerged as an effective technique for achieving HSI super-resolution (SR). Previous studies have mainly concentrated on estimating the posterior distribution of the latent high-resolution hyperspectral image (HR-HSI), leveraging an appropriate image prior and likelihood computed from the discrepancy between the latent HSI and observed images. Low rankness stands out for preserving latent HSI characteristics through matrix factorization among the various priors.
However, the primary limitation in previous studies lies in the generalization of a fusion model with fixed resolution scales, which necessitates retraining whenever output resolutions are changed. To overcome this limitation, we propose a novel continuous low-rank factorization (CLoRF) by integrating two neural representations into the matrix factorization, capturing spatial and spectral information, respectively. This approach enables us to harness both the low rankness from the matrix factorization and the continuity from neural representation in a self-supervised manner. Theoretically, we prove the low-rank property and Lipschitz continuity in the proposed continuous low-rank factorization. Experimentally, our method significantly surpasses existing techniques and achieves user-desired resolutions without the need for neural network retraining. Code is available at https://github.com/wangting1907/CLoRF-Fusion.

###### keywords

Low-rank factorization, arbitrary resolution, image fusion, continuous representation.

††equal-contributors: Equal contributions.††equal-contributors: Equal contributions.

### 1 Introduction

![Refer to caption](2405.17818v2/Fig1.png)

*Figure 1: The pipeline of CLoRF for arbitrary resolution. (A) Train the CLoRF. (B) Use the trained CLoRF to infer arbitrary resolutions of HSIs with given spatial and spectral coordinates. (C) An example of CLoRF for super-resolution on the Pavia University (336×336×93336\times 336\times 93). The CLoRF is trained given its LR-MSI (168×168×4168\times 168\times 4) and HR-HSI (42×42×5042\times 42\times 50), then infers the original resolution. Bicubic interpolation directly upsamples from LR. In the spectral domain, each band has distinct brightness values, and bicubic interpolation estimates the missing bands by referencing adjacent ones. As a result, this can cause discrepancies in the color representation of the interpolated image, such as in band 60, when compared to the GT image. (D) Visualize the spectrum of a random pixel from the results on (C).*

Hyperspectral images (HSIs) have widespread applications across various fields due to their rich spectral information. The abundant spectral details offered by HSIs facilitate accurate scene interpretation and enhance the efficacy of numerous applications, including object classification Gao et al. ((2014)) and anomaly detection Guo et al. ((2014)). However, the inherent trade-off between spectral and spatial resolution in HSI systems, constrained by hardware limitations, often results in HSIs with lower spatial resolution than RGB, panchromatic (PAN), and multispectral images (MSI). To enhance the spatial resolution of HSIs, a natural approach is to fuse LR-HSI and HR-MSI, known as hyperspectral and multispectral image fusion (HSI-MSI fusion). HSI-MSI fusion resembles MSI pansharpening, where low spatial resolution MSI is merged with high-resolution PAN imagery. However, directly applying these pansharpening methods to fuse HSI and MSI images suffers from challenges, as PAN images have limited spectral information, leading to spectral distortion Loncan et al. ((2015)). Consequently, numerous approaches tailored for HSI-MSI fusion are introduced, which can be generally categorized into model-based methods and deep learning-based models.

![Refer to caption](2405.17818v2/Fig2.png)

*Figure 2: Illustration of the proposed CLoRF for MSI-HSI fusion. The spatial coordinates and spectral coordinates of HR-MSI and LR-HSI are fed into the Spatial-INR Φθ​(⋅)\Phi_{\theta}(\cdot) and Spectral-INR Ψα​(⋅)\Psi_{\alpha}(\cdot) to generate coefficients and the bases, respectively. Thereafter, the generated coefficients and the bases are multiplied to recover the HR-HSI.*

Model-based methods leverage the low-rank structure of HSIs by characterizing their low-rankness through matrix factorization, decomposing the HSI matrix into the basis and coefficients or endmembers and abundances. Therefore, the principles of matrix factorization-based methods rely on appropriate prior information and the likelihood determined by the relationships between the latent HSI and the observed LR-HSI and HR-MSI. Given the known degradation model, most existing methods focus on modeling prior information for HSIs, including explicit and implicit methods. The explicit methods employ handcrafted explicit prior information, such as low-rankness L. Zhang et al. ((2018)); K. Wang et al. ((2020)), smoothness Simoes et al. ((2014)), sparsity Dong et al. ((2016)), and non-local similarity Dian et al. ((2017)), to depict the prior distribution of the latent HSI for super-resolution (SR). Implicit smooth regularizations introduce basis functions to parameterize the prior information, extending the modeling to functional representations. For example, Yokota et al.; Debals et al. utilize non-negative matrix factorization parameterized by basis functions to reveal implicit smoothness.

Compared to matrix-based methods, tensor-based methods can directly process HSI data and have garnered significant attention Dian et al. ((2019)); Chen et al. ((2022)). For example, a nonlocal sparse tensor factorization method based on Tucker decomposition was introduced in N. Liu et al. ((2021)), which factorizes an HSI into a sparse core tensor multiplied by dictionaries along both the spatial and spectral dimensions. In J. Zhang et al. ((2024)), a novel fusion model is proposed within the tensor ring decomposition framework, rather than using subspace decomposition-based fidelity terms. Although these tensor decomposition-based methods are effective in preserving the spectral-spatial correlation of HSI, they face the challenge of dimensionality catastrophe.
Furthermore, with the success of deep learning, Dian et al.; Z. Wang et al. implicitly introduce a denoising operator into the optimization framework to learn the deep image prior, which is shared across all HSIs using deep neural networks with different structures. Although these methods have achieved significant success in HSI-MSI fusion, manually designed explicit prior information may make the regularization terms problematic, resulting in significant computational complexity. Meanwhile, implicitly designed prior information sometimes becomes inappropriate for capturing the complex and detailed structures of HSIs. Moreover, these methods are restricted to fixed resolution and cannot be instantly applied to recovering HSIs with arbitrary resolution.

Most deep learning-based methods typically involve supervised training on large datasets consisting of observed and ground truth data to learn a complex function mapping. Some representative deep learning methods are based on convolution neural network (CNN) for HSI-MSI fusion, such as the two-stream fusion network designed for HR-MSI and LR-HSI X. Wang, Wang et al. ((2023)); Khader et al. ((2023)); Jia et al. ((2023)); W. Wang et al. ((2024)). With the remarkable performance of implicit neural representations (INRs) in continuous multi-dimensional data representation, some prior research X. Wang, Cheng et al. ((2023)); He et al. ((2024)) combined INR and CNN for HSI-MSI fusion. Although these methods have demonstrated promising results in fusion, they heavily rely on high-quality training data for supervised learning. Collecting high-quality ground-truth data is extremely time-consuming and costly. In addition, unsupervised fusion methods Zheng et al. ((2020)); Nguyen et al. ((2022)); Wu et al. ((2024)) are introduced, eliminating the need for expensive training data. However, these methods typically rely on complex network structures. Besides, they are also confined to fixed resolution, and inferring results at the user-desired resolutions requires retraining. Arbitrary-resolution HSI-MSI fusion is an emerging area within the field of image fusion. Unlike traditional fusion tasks, which typically merge images at fixed resolutions, arbitrary-resolution fusion leverages both HR-MSI and LR-HSI to generate fused outputs at any desired spatial and spectral resolutions W. Wang et al. ((2024)); He et al. ((2021)). This advanced fusion technique provides greater flexibility in adjusting both the spatial and spectral resolutions of the resulting HR-HSI, offering significant potential for a wide range of applications He et al. ((2021)), such as object detection Qu et al. ((2017)) and land use/cover classification Gao et al. ((2014)). However, arbitrary-resolution hyperspectral image fusion presents greater challenges compared to standard fusion methods.

To address the aforementioned issues, we propose an unsupervised HSI-MSI fusion framework for continuous low-rank factorization (CLoRF). Specifically, CLoRF integrates two implicit neural representations into the low-rank factorization, capturing continuous spatial and spectral information of HR-MSI and LR-HSI, respectively. Unlike classic discrete matrix factorization, we define the continuous matrix function factorization following tensor function Luo et al. ((2024)), which characterizes low-rankness in continuous representation. Each continuous function is a realization of INR parameterized by multi-layer perceptrons (MLPs). As MLPs are Lipschitz smooth, further characterizing smoothness in continuous representation. Compared to previous fusion methods, as shown in Fig. 1, CLoRF can achieve arbitrary resolution in both spatial and spectral domains without additional image information and retraining.

Our contributions are summarized as follows:

- 1.

We propose a novel unsupervised HSI-MSI fusion method: CLoRF, which represents HSIs in a continuous representation using low-rank function factorization. CLoRF can infer with arbitrary resolution without the need for retraining.

- 2.

We theoretically prove that the implicit regularization terms of low-rankness and smoothness are unified in continuous representation, which justifies the potential effectiveness for HSIs.

- 3.

We experimentally demonstrate that CLoRF significantly surpasses existing techniques, confirming its wide applicability and superiority and further expanding its application to HSI and PAN image fusion (HSI-PAN fusion).

The structure of the remainder of this paper is outlined as follows. Sec. 2 provides a review of related work. Sec. 3 introduces the proposed CLoRF in detail. Sec. 4 presents the experimental results and subsequent discussion, and Sec. 5 concludes this work.

### 2 Related Work

#### 2.1 Implicit Neural Representation

INR offers a novel approach to representing implicitly defined, continuous, differentiable signals parameterized by neural networks Sitzmann et al. ((2020)). INR has demonstrated remarkable performance in representing complex data structures, such as 3D reconstruction Mildenhall et al. ((2021)); Sitzmann et al. ((2020)); Takikawa et al. ((2021)) and 2D image super-resolution Dupont et al. ((2021)); Anokhin et al. ((2021)) and generation Xu & Jiao ((2023)); Chen et al. ((2021)), etc. Recently, INR-based approaches have been explored for HSIs, such as HSI SR K. Zhang et al. ((2022)), unmixing T. Wang et al. ((2024)), and fusion X. Wang, Cheng et al. ((2023)); Deng et al. ((2023)). Despite these commendable efforts, INR still encounters challenges in HSIs. For instance, HSIs consist of numerous spectral bands obtained through continuous imaging within a specific spectral range. However, INR itself may not possess sufficient stability to directly learn a valid continuous representation from the spectral domain of HSIs. In the fusion task, learning in both spatial and spectral domains is necessary; therefore, utilizing a single INR for learning representation may confront limited representation capabilities.

#### 2.2 HSI-MSI Fusion via Continuous Representation

Several recent studies explore INRs for HSI-MSI fusion. For instance, X. Wang, Cheng et al. proposes spatial-INR and spectral-INR for spatial and spectral resolution reconstruction, respectively. Besides, Deng et al. proposes an innovative fusion method that integrates CNN and INR. Based on them, He et al. develops two spectral-spatial INRs for arbitrary-resolution hyperspectral pansharpening. Although these fusion methods utilize spatial-spectral-based INRs, they significantly diverge from CLoRF. First, these methods rely on local implicit image functions Chen et al. ((2021)) for HSI SR. They employ a complex CNN network to encode spatial and spectral features, then feed these features and the three-dimensional coordinates of HSIs into an MLP to recover HSIs. Second, these HSI-MSI fusion methods are supervised learning, which heavily relies on pairs of images for training. Third, they do not perform low-rank matrix decomposition on HR-HSI, nor do they leverage the low-rank and smooth physical information inherent in HSIs. As a result, the computational cost of these INRs is high, primarily due to the large size of HSIs, which results in large three-dimensional coordinates. Conversely, our method is model-based and unsupervised, finely encoding the low-rankness and smoothness into the continuous spatial-spectral factorization function. Consequently, this improves the stability of the continuous representation in HSIs and significantly reduces the complexity of the network structure and the computational cost. Therefore, our method is more feasible and general in various HSIs.

### 3 Proposed Method

In this section, we first present the problem formulation for HSI-MSI fusion, then introduce the details of our proposed framework in the following section.

#### 3.1 Problem Formulation

Given the HR-MSI and LR-HSI data, we aim to approximate their corresponding HR-HSI data. Specifically, the HR-HSI, LR-HSI, and HR-MSI data are transformed into the matrix format along the spectral dimension. The HR-HSI is denoted as 𝐙∈ℝL×N\mathbf{Z}\in\mathbb{R}^{L\times N}, where LL is the number of spectral bands, and N=H∗WN=H*W is the total number of pixels, in which HH and WW indicate spatial resolution. The LR-HSI is denoted as 𝐗∈ℝL×n\mathbf{X}\in\mathbb{R}^{L\times n}, where nn represents the number of LR spatial pixels, i.e., n≪Nn\ll N. Finally, the HR-MSI is denoted as 𝐘∈ℝl×N\mathbf{Y}\in\mathbb{R}^{l\times N}, where l≪Ll\ll L signifies that 𝐘\mathbf{Y} has fewer spectral bands than 𝐗\mathbf{X}.

The LR-HSI 𝐗\mathbf{X} can be interpreted as a diminished-quality representation of HR-HSI 𝐙\mathbf{Z} in the spatial dimension, which is formulated as follows:

𝐗=𝐙𝐁𝐒+𝐍h,\mathbf{X}=\mathbf{Z}\mathbf{B}\mathbf{S}+\mathbf{N}_{h},

(1)

where 𝐍h∼𝒩⁡(𝟎,σh​𝐈)\mathbf{N}_{h}\sim\mathcal{N}(\boldsymbol{0},\sigma_{h}\mathbf{I}) represents the additive Gaussian noise. Besides, 𝐁∈ℝN×N\mathbf{B}\in\mathbb{R}^{N\times N} is a spatial blurring operator of 𝐙\mathbf{Z}, representing the point spread function (PSF) of the hyperspectral sensor. Additionally, 𝐒∈ℝN×n\mathbf{S}\in\mathbb{R}^{N\times n} is the spatial downsampling matrix.

Similarly, the HR-MSI 𝐘\mathbf{Y} can be considered as a downsampled realization of HR-HSI 𝐙\mathbf{Z} in the spectral dimension, which is formulated as:

𝐘=𝐇𝐙+𝐍m,\mathbf{Y}=\mathbf{H}\mathbf{Z}+\mathbf{N}_{m},

(2)

where 𝐇∈ℝl×L\mathbf{H}\in\mathbb{R}^{l\times L} is the spectral response function (SRF) and 𝐍m∼𝒩⁡(𝟎,σm​𝐈)\mathbf{N}_{m}\sim\mathcal{N}(\boldsymbol{0},\sigma_{m}\mathbf{I}) denotes the additive Gaussian noise.

As HSIs generally have a low-rank structure, thus they lie in a low-dimensional subspace Simoes et al. ((2014)); Zhuang & Bioucas-Dias ((2018)). The low-rank factorization aims to approximate a target matrix 𝐙\mathbf{Z} as a product of two matrices:

𝐙≈𝐄𝐀,\mathbf{Z}\approx\mathbf{E}\mathbf{A},

(3)

where 𝐄∈ℝL×K\mathbf{E}\in\mathbb{R}^{L\times K} is a spectral dictionary and 𝐀∈ℝK×N\mathbf{A}\in\mathbb{R}^{K\times N} is a coefficient matrix, respectively. And
K≪LK\ll L represents a hyperparameter controlling the number of spectral bases. The low-rank factorization representation offers three main advantages. First, it maximizes the utilization of strong correlations among the spectral bands. Second, by keeping KK small (where K≪LK\ll L), the size of the spectral mode is reduced, thereby enhancing computational efficiency. Third, each column of matrix 𝐙\mathbf{Z} can be linearly represented by the columns of matrix 𝐄\mathbf{E} using the coefficients in matrix 𝐀\mathbf{A}.
The rows of matrix 𝐀\mathbf{A} maintain the spatial structures of matrix 𝐙\mathbf{Z}. Note that (3) is not a unique factorization for 𝐙\mathbf{Z}. One could obtain another pairs 𝐄^=𝐄𝐁\hat{\mathbf{E}}=\mathbf{E}\mathbf{B} and 𝐀^=𝐁−1​𝐀\hat{\mathbf{A}}=\mathbf{B}^{-1}\mathbf{A}, with any inverse matrix 𝐁∈ℝK×K\mathbf{B}\in\mathbb{R}^{K\times K}.

By integrating Eq.(3) into Eq.(1) and Eq.(2), 𝐗\mathbf{X} and 𝐘\mathbf{Y} are formulated as:

𝐗=𝐄𝐀𝐁𝐒+𝐍h,𝐘=𝐇𝐄𝐀+𝐍m.\mathbf{X}=\mathbf{E}\mathbf{A}\mathbf{B}\mathbf{S}+\mathbf{N}_{h},\quad\mathbf{Y}=\mathbf{H}\mathbf{E}\mathbf{A}+\mathbf{N}_{m}.

(4)

Thereafter, the fusion problem is transformed into the task of estimating the spectral dictionary 𝐄\mathbf{E} and its corresponding coefficient matrix 𝐀\mathbf{A} from matrices 𝐗\mathbf{X} and 𝐘\mathbf{Y}, which follows the following optimization problem:

min𝐄,𝐀⁡‖𝐗−𝐄𝐀𝐁𝐒‖F2+λ​‖𝐘−𝐇𝐄𝐀‖F2,\min_{\mathbf{E},\mathbf{A}}\|\mathbf{X}-\mathbf{E}\mathbf{A}\mathbf{B}\mathbf{S}\|_{\text{F}}^{2}+\lambda\|\mathbf{Y}-\mathbf{H}\mathbf{E}\mathbf{A}\|_{\text{F}}^{2},

(5)

where ∥⋅∥F\|\cdot\|_{\text{F}} denotes the Frobenius norm, and λ\lambda denotes the balancing factor. As there is a lack of specific prior information, the problem Eq.(5) is undetermined; therefore, existing works focus on exploring the appropriate prior information. Nonetheless, these methods process HR-HSI within the dimensions of two modalities and fail to fuse the arbitrary resolutions of HSIs effectively.

#### 3.2 Deep Continuous Low-rank Factorization Model

INR is widely adopted for learning continuous data representation, such as HSIs X. Wang, Cheng et al. ((2023)); Deng et al. ((2023)). However, simply utilizing a single INR to represent the HSI volume results in low efficiency and expensive computation, as it neglects the distinct low-rank structure of HSIs. Conversely, we propose a continuous low-rank factorization (CLoRF) model for learning HSI representation continuously and effectively. Our method fully explores the low-rank structure of HSIs by simultaneously learning its low-rank continuous representation by spatial and spectral INR. As a result, our approach effectively captures the low-rankness and smoothness of HSIs while overcoming the computational burden associated with existing INR-based methods X. Wang, Cheng et al. ((2023)); Deng et al. ((2023)) in HSIs.

As illustrated in Fig. 2, we present an overview of the proposed CLoRF. Specifically, CLoRF consists of two steps: low-rank decomposition and learning. The low-rank decomposition breaks down the HSI data space into two smaller subspaces: spatial basis 𝐀\mathbf{A} and spectral transformation 𝐄\mathbf{E}. Additionally, the spatial and spectral components are parameterized by two neural networks, using two INRs to learn the low-rank continuous representation of the HR-HSI.
Inspired by the advance of Sinusoidal Representation Networks (SIRENs) Sitzmann et al. ((2020)), we employ two SIRENs to estimate 𝐄\mathbf{E} and 𝐀\mathbf{A} in Eq.(3), respectively. Specifically, we denote one SIREN Ψθ​(⋅)\Psi_{\theta}(\cdot) parameterize by θ\theta
for approximating 𝐄\mathbf{E}, and another SIREN Φα​(⋅)\Phi_{\alpha}(\cdot) parameterized by α\alpha for approximating 𝐀\mathbf{A}, which is defined as follows:

𝐄^​(𝐛,θ)=[Ψθ​(b1),Ψθ​(b2),…,Ψθ​(bL)]T,𝐀^​(𝐎,α)=[Φα​(𝐨11),Φα​(𝐨12),…,Φα​(𝐨H​W)],\begin{split}\mathbf{\hat{E}}(\mathbf{b};\theta)&=[\Psi_{\theta}(b_{1}),\Psi_{\theta}(b_{2}),\dots,\Psi_{\theta}(b_{L})]^{T},\\
\mathbf{\hat{A}}(\mathbf{O};\alpha)&=[\Phi_{\alpha}(\mathbf{o}_{11}),\Phi_{\alpha}(\mathbf{o}_{12}),\dots,\Phi_{\alpha}(\mathbf{o}_{HW})],\end{split}

where
Ψθ​(bi):ℝ→ℝK\Psi_{\theta}(b_{i}):\mathbb{R}\rightarrow\mathbb{R}^{K} is a spectral basis, with bi∈ℝb_{i}\in\mathbb{R} being the 1D coordinate for the ii-th band index of the LR-HSI. Besides, Φα​(𝐨i​j):ℝ2→ℝK\Phi_{\alpha}(\mathbf{o}_{ij}):\mathbb{R}^{2}\rightarrow\mathbb{R}^{K} is a spatial basis with the 2D coordinate 𝐨i​j∈ℝ2\mathbf{o}_{ij}\in\mathbb{R}^{2} of the HR-MSI. And we denote the spectral bases as 𝐛=[b1,b2,…,bL]T\mathbf{b}=[b_{1},b_{2},\dots,b_{L}]^{T} and the spatial bases as 𝐎=[𝐨11;𝐨12;…;𝐨H​W;].\mathbf{O}=[\mathbf{o}_{11};\mathbf{o}_{12};\dots;\mathbf{o}_{HW};]. Both networks aim to learn how to map from a fixed coordinate to the target representation. Here, we formalize these networks as follows:

Ψθ(bi)=𝐖1d1(⋯(σ(𝐖11bi+𝐜11))⋯)+𝐜1d1,\displaystyle\Psi_{\theta}(b_{i})=\mathbf{W}^{1}_{d_{1}}(\cdots(\sigma(\mathbf{W}^{1}_{1}b_{i}+\mathbf{c}^{1}_{1}))\cdots)+\mathbf{c}^{1}_{d_{1}},

Φα(𝐨i​j)=𝐖2d2(⋯(σ(𝐖21𝐨i​jT+𝐜21)⋯)+𝐜2d2,\displaystyle\Phi_{\alpha}(\mathbf{o}_{ij})=\mathbf{W}^{2}_{d_{2}}(\cdots(\sigma(\mathbf{W}^{2}_{1}\mathbf{o}_{ij}^{T}+\mathbf{c}^{2}_{1})\cdots)+\mathbf{c}^{2}_{d_{2}},

where σ\sigma denotes the activation function, θ=({𝐖i1}i=1d1,{𝐜i1}i=1d1)\theta=(\{\mathbf{W}^{1}_{i}\}_{i=1}^{d_{1}},\{\mathbf{c}^{1}_{i}\}_{i=1}^{d_{1}}) and
α=({𝐖i2}i=1d2,{𝐜i2}i=1d2)\alpha=(\{\mathbf{W}^{2}_{i}\}_{i=1}^{d_{2}},\{\mathbf{c}^{2}_{i}\}_{i=1}^{d_{2}}) contains weight matrices and bias vectors for spectral- and spatial-INR, respectively.
Our method is a natural progression of low-rank factorization from discrete mesh grids to the continuous domain. And the target HR-HSI is approximated as 𝐙^=𝐄^​(𝐛,θ)​𝐀^​(𝐎,α).\mathbf{\hat{Z}}=\mathbf{\hat{E}}(\mathbf{b};\theta)\mathbf{\hat{A}}(\mathbf{O};\alpha).

As matrix 𝐀^​(𝐎,α)\mathbf{\hat{A}}(\mathbf{O};\alpha) preserves the spatial structures of HSIs. Without loss of generality, we consider the spatial smoothness of HSIs. Moreover, a total variation (TV) loss on the predicted coefficient matrix 𝐀^​(𝐎,α)\mathbf{\hat{A}}(\mathbf{O};\alpha) is further incorporated for noise-disruption scenarios. Mathematically, the TV regularization of 𝐀^​(𝐎,α)\mathbf{\hat{A}}(\mathbf{O};\alpha) is formulated as:

∑k=1KTV​(𝐚^k)=∑k=1K(‖𝐃h​𝐚^k‖1+‖𝐃w​𝐚^k‖1),\sum_{k=1}^{K}\text{TV}(\mathbf{\hat{a}}_{k})=\sum_{k=1}^{K}(\|\mathbf{D}_{h}{\mathbf{\hat{a}}}_{k}\|_{1}+\|\mathbf{D}_{w}{\mathbf{\hat{a}}}_{k}\|_{1}),

(6)

where 𝐚^k\hat{\mathbf{a}}_{k} is the kk-th row of 𝐀^​(𝐎,α)\mathbf{\hat{A}}(\mathbf{O};\alpha).
𝐃h\mathbf{D}_{h} and 𝐃w\mathbf{D}_{w} denote the differential operation along the height and width direction in the matrix form of 𝐚^k\hat{\mathbf{a}}_{k}, respectively. Here, ∥⋅∥1\|\cdot\|_{1} indicates the ℓ1\ell_{1} norm. By incorporating the TV loss, we promote spatial smoothness and improve
the overall quality of the fusion.

Therefore, the optimization problem with TV prior can be summarized
as follows:

minθ,α⁡ℒMSI+λ​ℒHSI+η​∑k=1KTV​(𝐚^k),\min_{\theta,\alpha}\mathcal{L}_{\text{MSI}}+\lambda\mathcal{L}_{\text{HSI}}+\eta\sum\limits_{k=1}^{K}\text{TV}(\mathbf{\hat{a}}_{k}),

(7)

where ℒMSI=‖𝐗−𝐄^​(𝐛,θ)​𝐀^​(𝐎,α)​𝐁𝐒‖F2,\mathcal{L}_{\text{MSI}}=\|\mathbf{X}-\mathbf{\hat{E}}(\mathbf{b};\theta)\mathbf{\hat{A}}(\mathbf{O};\alpha)\mathbf{BS}\|_{\text{F}}^{2}, ℒHSI=‖𝐘−𝐇​𝐄^​(𝐛,θ)​𝐀^​(𝐎,α)‖F2\mathcal{L}_{\text{HSI}}=\|\mathbf{Y}-\mathbf{H}\mathbf{\hat{E}}(\mathbf{b};\theta)\mathbf{\hat{A}}(\mathbf{O};\alpha)\|_{\text{F}}^{2}, and η\eta is the regularization parameter.

Approximating 𝐄^​(𝐛,θ∗)\mathbf{\hat{E}}(\mathbf{b};\theta^{\ast}) and 𝐀^​(𝐎,α∗)\mathbf{\hat{A}}(\mathbf{O};\alpha^{\ast}) to maintain the low-rank representation of 𝐙^\mathbf{\hat{Z}} can be achieved after training networks, with θ∗,α∗\theta^{\ast},\alpha^{\ast} corresponding to the parameters of well-trained networks. Recall the networks take the coordinates as the input, and the optimization in Eq.(7) does not involve the ground-truth HSIs as the supervision label; therefore, our method is self-supervised. We employ the Adam optimizer for optimization, which is a stochastic gradient descent algorithm. Moreover, we can infer an arbitrary-resolution HSI by inputting any scale coordinates {𝐛~,𝐎~}\{\tilde{\mathbf{b}},\tilde{\mathbf{O}}\} into the well-trained network, i.e., 𝐄^​(𝐛~,θ∗)​𝐀^​(𝐎~,α∗)\mathbf{\hat{E}}(\tilde{\mathbf{b}};\theta^{\ast})\mathbf{\hat{A}}(\tilde{\mathbf{O}};\alpha^{\ast}).

Compared to existing INRs-based fusion methods, our method stands out for several advantages. First, it fully exploits the low-rank and smooth prior of HSIs through low-rank continuous learning representations. Second, its computational complexity is significantly reduced through continuous low-rank factorization. Third, it can achieve the user-desired resolution at arbitrary locations in HSIs by inputting any scale spatial and spectral coordinates.

#### 3.3 Theoretical Analysis

In this section, we theoretically demonstrate that the low-rank and smooth regularizations are implicitly unified in continuous low-rank matrix factorization. Our analysis is inspired by the concept of tensor function factorization in Luo et al. ((2024)). Here, we start with rank factorization in the matrix computation field.

###### Theorem 1 (rank factorization Piziak & Odell ((1999))).

Let 𝐗∈ℝn1×n2\mathbf{X}\in\mathbb{R}^{n_{1}\times n_{2}}, where rank⁡(𝐗)=K\mathrm{rank}(\mathbf{X})=K, then there exists two matrices 𝐔∈ℝn1×K\mathbf{U}\in\mathbb{R}^{n_{1}\times K}, 𝐕∈ℝn2×K\mathbf{V}\in\mathbb{R}^{n_{2}\times K} such that 𝐗=𝐔𝐕T\mathbf{X}=\mathbf{U}\mathbf{V}^{T}.

Subsequently, we provide a detailed introduction to the proposed continuous representation of HSIs. Let f⁡(⋅):𝒜f×𝒵f→ℝf(\cdot):\mathcal{A}_{f}\times\mathcal{Z}_{f}\rightarrow\mathbb{R} be a bounded real function, where 𝒜f⊂ℝ2\mathcal{A}_{f}\subset\mathbb{R}^{2}, 𝒵f⊂ℝ\mathcal{Z}_{f}\subset\mathbb{R} are definition domains in spatial and spectral domains, respectively. The function ff gives the value of data at any coordinate in 𝒟f:=𝒜f×𝒵f\mathcal{D}_{f}:=\mathcal{A}_{f}\times\mathcal{Z}_{f}. We interpret ff as a matrix function since it maps a spatial and spectral coordinate to the corresponding value, implicitly representing matrix data.

###### Definition 1 (sampled matrix set).

For a matrix function f⁡(⋅):𝒟f→ℝf(\cdot):\mathcal{D}_{f}\to\mathbb{R}, we define the sampled matrix set 𝒮⁡[f]\mathcal{S}[f] as

𝒮[f]:={𝐌|𝐌(i,j)=f(𝐬i,bj),𝐬i∈𝒜f,bj∈𝒵f,𝐌∈ℝn1×n2,n1,n2∈ℕ+},\begin{split}\mathcal{S}[f]:=\{&\mathbf{M}|\mathbf{M}_{(i,j)}=f(\mathbf{s}_{i},b_{j}),\mathbf{s}_{i}\in\mathcal{A}_{f},\\
&{b}_{j}\in\mathcal{Z}_{f},\mathbf{M}\in\mathbb{R}^{n_{1}\times n_{2}},n_{1},n_{2}\in\mathbb{N}_{+}\},\end{split}

where 𝐬i,bj\mathbf{s}_{i},b_{j} denote the spatial and spectral coordinates, respectively.

Regarding the definition of rank, we expect any matrix sampled on 𝒮⁡[f]\mathcal{S}[f] to be low-rank. Naturally, we can then define the rank of the matrix function as follows.

###### Definition 2 (matrix function rank).

Given a matrix function f:𝒟f=𝒜f×𝒵f→ℝf:\mathcal{D}_{f}=\mathcal{A}_{f}\times\mathcal{Z}_{f}\to\mathbb{R}, we define a measure of its complexity, denoted by MF​-​rank​[f]\mathrm{MF}\text{-}\mathrm{rank}[f] (function rank of f⁡(⋅)f(\cdot)), as the supremum of the matrix rank in the sampled matrix set 𝒮⁡[f]:\mathcal{S}[f]{:}

MF​-​rank​[f]:=sup𝐌∈𝒮⁡[f]rank⁡(𝐌).\mathrm{MF}\text{-}\mathrm{rank}[f]:=\sup_{\mathbf{M}\in\mathcal{S}[f]}\mathrm{rank}(\mathbf{M}).

We call a matrix function f⁡(⋅)f(\cdot) as a low-rank matrix function if K≪min⁡{n1,n2}K\ll\min\{n_{1},n_{2}\}. When f⁡(⋅)f(\cdot) is defined on a given matrix, we will show that the MF-rank degenerates into the discrete case, i.e., the classical matrix rank.

###### Proposition 2.

Consider 𝐗∈ℝn1×n2\mathbf{X}\in\mathbb{R}^{n_{1}\times n_{2}} as an arbitrary matrix.
Let 𝒜f=𝒩(l1)×𝒩(l2)​(l1​l2=n1)\mathcal{A}_{f}=\mathcal{N}^{(l_{1})}\times\mathcal{N}^{(l_{2})}\ (l_{1}l_{2}=n_{1}) represent a two-dimensional discrete set, and 𝒵f=𝒩(n2)\mathcal{Z}_{f}=\mathcal{N}^{(n_{2})} is an one-dimensional discrete set, where 𝒩(k)\mathcal{N}^{(k)} is the set {1,2,…,k}\{1,2,\dots,k\}. We denote 𝒟f=𝒜f×𝒵f\mathcal{D}_{f}=\mathcal{A}_{f}\times\mathcal{Z}_{f} and define
the matrix function f⁡(⋅):𝒟f→ℝf(\cdot):\mathcal{D}_{f}\rightarrow\mathbb{R} as f⁡(𝐬,b)=𝐗(𝐬,b)f(\mathbf{s},b)=\mathbf{X}_{(\mathbf{s},b)} for any (𝐬,b)∈𝒟f(\mathbf{s},b)\in\mathcal{D}_{f}. Consequently, MF\mathrm{MF}-rank⁡[f]=rank⁡(𝐗)\mathrm{rank}[f]=\mathrm{rank}(\mathbf{X}).

The proof of Proposition 2 is shown in Appendix 6. Proposition 2 expands the concept of rank from discrete matrices to matrix functions for continuous representations. Analogous to classical matrix representations, it is pertinent to consider whether a low-rank matrix function ff can employ certain matrix factorization strategies to encode its low-rank. We provide an affirmative response, as stated in the theorem below.

###### Theorem 3 (continuous low-rank factorization).

Let f⁡(⋅):𝒟f=𝒜f×𝒵f→ℝf(\cdot):\mathcal{D}_{f}=\mathcal{A}_{f}\times\mathcal{Z}_{f}\to\mathbb{R}
be a bounded matrix function, where 𝒜f⊂ℝ2\mathcal{A}_{f}\subset\mathbb{R}^{2}, 𝒵f⊂ℝ\mathcal{Z}_{f}\subset\mathbb{R}.
If MF\mathrm{MF}-rank⁡[f]=K\mathrm{rank}[f]=K, then there exist two functions fspatial​(⋅):𝒜f→ℝKf_{\mathrm{spatial}}(\cdot):\mathcal{A}_{f}\rightarrow\mathbb{R}^{K}, fspectral​(⋅):𝒵f→ℝKf_{\mathrm{spectral}}(\cdot):\mathcal{Z}_{f}\rightarrow\mathbb{R}^{K} such that f⁡(𝐬,b)=fspatial​(𝐬)⋅fspectralT​(b)f(\mathbf{s},b)=f_{\mathrm{spatial}}(\mathbf{s})\cdot f^{T}_{\mathrm{spectral}}(b) for any pair of inputs (𝐬,b)∈𝒟f(\mathbf{s},b)\in\mathcal{D}_{f}.

The proof of Theorem 3 is illustrated in Appendix 7. Theorem 3 is a natural extension of rank factorization (Theorem 1) from discrete meshgrid to the continuous domain. Specifically, we employ two MLPs Φα​(⋅)\Phi_{\alpha}(\cdot) and Ψθ​(⋅)\Psi_{\theta}(\cdot) with parameters θ\theta and α\alpha to parameterize the factor functions fspatial​(⋅)f_{\mathrm{spatial}}(\cdot) and fspectral​(⋅)f_{\mathrm{spectral}}(\cdot).

###### Remark 1.

1) In Theorem 1, the low-rank matrix decomposition definitely exists but is non-unique. This is because the representation of eigenvectors in SVD is not unique, which leads to different 𝐔\mathbf{U} and 𝐔′\mathbf{U}^{\prime}, as well as 𝐕\mathbf{V} and 𝐕′\mathbf{V}^{\prime} in the decomposition, such as 𝐔′=c​𝐔\mathbf{U}^{\prime}=c\mathbf{U}, 𝐕′=𝐕/c\mathbf{V}^{\prime}=\mathbf{V}/c, where cc is a nonzero scalar.

2) CLoRF comprises two subnetworks with identical architectures: Spatial-INR and Spectral-INR. This network framework, equipped with low-rank and smooth priors, learns to generate a spectral dictionary and spatial coefficient matrix, enabling a low-rank representation of HSIs through network training. The solution spaces of spatial and spectral basis are constrained by the parameters of the INRs. Furthermore, by incorporating TV prior into the loss function, the solution space of the spatial coefficient matrix is further restricted, effectively reducing the ambiguity in the low-rank factorization.

Smoothness is another prevalent attribute in HSIs, such as the spatial and spectral smoothness of HSIs Sun et al. ((2021)). Here, we theoretically validate that our method incorporates implicit smooth regularization derived from the specific structures of MLPs.

###### Theorem 4 (Lipschitz continuity).

Let 𝐗∈ℝn1×n2\mathbf{X}\in\mathbb{R}^{n_{1}\times n_{2}}, and Φα​(⋅):𝒜f→ℝK\Phi_{\alpha}(\cdot):\mathcal{A}_{f}\rightarrow\mathbb{R}^{K}, Ψθ​(⋅):𝒟f→ℝK\Psi_{\theta}(\cdot):\mathcal{D}_{f}\rightarrow\mathbb{R}^{K} be two MLPs structured with parameters α,θ\alpha,\theta where 𝒜f⊂ℝ2\mathcal{A}_{f}\subset\mathbb{R}^{2}, 𝒵f⊂ℝ\mathcal{Z}_{f}\subset\mathbb{R} are bounded, i.e., ‖𝐬‖1≤ζ,\|\mathbf{s}\|_{1}\leq\zeta, b≤ζb\leq\zeta for any 𝐬∈𝒜f\mathbf{s}\in\mathcal{A}_{f}, b∈𝒵fb\in\mathcal{Z}_{f}. Suppose the MLPs share the same activation function σ⁡(⋅)\sigma(\cdot) and depth dd with 𝐜i1=𝐜i2=𝟎,∀i\mathbf{c}^{1}_{i}=\mathbf{c}^{2}_{i}=\mathbf{0},\forall\ i. Besides, we assume that

- •

σ\sigma is Lipschitz continuous with the Lipschitz constant κ\kappa, and σ⁡(0)=0\sigma(0)=0;

- •

‖𝐖i1‖1,‖𝐖i2‖1\|\mathbf{W}_{i}^{1}\|_{1},\|\mathbf{W}_{i}^{2}\|_{1} are bounded by a positive constant η\eta for all ii.

Define a matrix function f⁡(⋅):𝒟f=𝒜f×𝒵f→ℝf(\cdot):\mathcal{D}_{f}=\mathcal{A}_{f}\times\mathcal{Z}_{f}\rightarrow\mathbb{R} as f⁡(𝐬,b)=Φα​(𝐬)⋅Ψθ​(b)Tf(\mathbf{s},b)=\Phi_{\alpha}(\mathbf{s})\cdot\Psi_{\theta}(b)^{T}. Then, the following inequalities hold for any (𝐬1,b1),(𝐬2,b2)∈𝒟f(\mathbf{s}_{1},b_{1}),(\mathbf{s}_{2},b_{2})\in\mathcal{D}_{f}:

|f⁡(𝐬1,b1)−f⁡(𝐬2,b2)|\displaystyle|f(\mathbf{s}_{1},b_{1})-f(\mathbf{s}_{2},b_{2})|
≤δ​‖𝐬1−𝐬2‖1+δ​|b1−b2|,\displaystyle\leq\delta\|\mathbf{s}_{1}-\mathbf{s}_{2}\|_{1}+\delta|b_{1}-b_{2}|,

where δ=η2​d+1​κ2​d−2​ζ\delta=\eta^{2d+1}\kappa^{2d-2}\zeta,
and ζ=max⁡{‖𝐬1‖1,|b1|}\zeta=\max\{\|\mathbf{s}_{1}\|_{1},|b_{1}|\}.

The proof is shown in Appendix 8. This smoothness is implicitly encoded with mild assumptions regarding nonlinear activation functions and weight matrices, which are readily attainable in a real-world implementation. For instance, we utilized the sine activation function, while also ensuring that the weights in the MLP network Sitzmann et al. ((2020)) remain bounded.

###### Remark 2.

In Theorem 4, we observe that the degree of smoothness, denoted as δ\delta, is associated with the Lipschitz constant κ\kappa and the upper bound η\eta of the weight matrices. Therefore, we can manipulate two variables in practice to achieve a balance in implicit smoothness:

1) We utilize the Sine function σ(⋅)=sin(ω0⋅)\sigma(\cdot)=\sin(\omega_{0}\cdot) as the nonlinear activation function in the MLPs. Since the Sine function is Lipschitz continuous, we can effectively adjust its Lipschitz constant κ\kappa by varying the value of ω0\omega_{0}. Specifically, a smaller ω0\omega_{0} results in a lower Lipschitz constant κ\kappa, leading to smoother outcomes.

2) To manage the upper bound η\eta of the MLP weight matrices, we can adjust the trade-off parameter in the energy regularization of the MLP weights, commonly referred to as weight decay in contemporary deep learning optimizers. This approach allows us to control the strength of η\eta.

###### Remark 3.

Assume the assumptions in Theorem 4 are satisfied. We define f⁡(⋅):=[Φα,Ψθ]​(⋅)f(\cdot):=[\Phi_{\alpha},\Psi_{\theta}](\cdot). Then, for any matrix 𝐌∈𝒮⁡[f]\mathbf{M}\in\mathcal{S}[f] that is sampled using coordinates vectors 𝐬∈𝒜f\mathbf{s}\in\mathcal{A}_{f}, t∈𝒵ft\in\mathcal{Z}_{f}, where 𝒮⁡[f]\mathcal{S}[f] represents the set of sampled matrices from the matrix function f⁡(⋅)f(\cdot) as defined in Definition 1, the following inequalities hold for (i,j),i=1,2,⋯,n1,j=1,2,⋯n2:(i,j),i=1,2,\cdots,n_{1},j=1,2,\cdots n_{2}:

|𝐌(𝐬i,tj)−𝐌(𝐬i−1,tj−1)|≤δ​‖𝐬i−𝐬i−1‖1+δ​|tj−tj−1|,\small|\mathbf{M}_{(\mathbf{s}_{i},t_{j})}-\mathbf{M}_{(\mathbf{s}_{i-1},t_{j-1})}|\leq\delta\|\mathbf{s}_{i}-\mathbf{s}_{i-1}\|_{1}+\delta|t_{j}-t_{j-1}|,

(8)

where δ=η2​d+1​κ2​d−2​ζ\delta=\eta^{2d+1}\kappa^{2d-2}\zeta, and ζ=max⁡{‖𝐬1‖1,|b1|}\zeta=\max\{\|\mathbf{s}_{1}\|_{1},|b_{1}|\}.

Remark 3 states that for any sampled matrix 𝐌∈S⁡[f]\mathbf{M}\in S[f],
the difference between adjacent elements is constrained by the distance between the corresponding coordinates, with the inclusion of a constant factor.

### 4 Experiments and Analysis

In this section, we evaluate the performance of our method on five datasets separately. Additionally, we compare several SOTA methods and evaluate the fusion results qualitatively and quantitatively. Finally, we expand the application of CLoRF to HSI-PAN fusion and verify its efficacy on a dataset.

#### 4.1 Experimental Details

1) Datasets:
We evaluate the performance of fusion using both synthetic and real datasets.
Seven simulated datasets, including Pavia University Dell’Acqua et al. ((2004)), Pavia Center Dell’Acqua et al. ((2004)), Indian Pines Baumgardner et al. ((2015)), Washington DC Zhuang & Ng ((2021)); Zhuang & Bioucas-Dias ((2018)), University of Houston Le Saux et al. ((2018)), Peppers Yasuma et al. ((2010)), and Superballs Yasuma et al. ((2010)) were used for simulations in our experiments.
The seven synthetic datasets, each with a simulated PSF and SRF, are used to generate two observed images: LR-HSI and HR-MSI. More specifically, we utilize a Gaussian blur of 5×55\times 5 pixels with a 0 mean and 1 standard deviation to simulate the PSF for generating the LR-HSI. The downsampling ratio was set to 4 for all datasets. The spectral response of the IKONOS satellite (a Nikon D700 camera) was used to simulate the SRF for generating the HR-MSI. The i.i.d Gaussian noise was added to HR-MSI and LR-HSI with signal-to-noise ratios (SNRs) of 30 dB, respectively. The details for the eight datasets are summarized as follows.

- •

Pavia University: The image measures 610×340×115610\times 340\times 115 pixels with a spatial resolution of 1.3 meters and spectral coverage spanning from 0.43 μ​m\mu m to 0.86 μ​m\mu m. Due to the effects of noise and water vapor absorption, 12 bands were removed. An area covering 336×336336\times 336 pixels in the lower-left corner of the image and containing 103 bands was selected for the experiment.

- •

Pavia Center: The size of the Pavia Center is 1096×1096×1151096\times 1096\times 115 and spectral ranging from 0.38 to 1.05 μ​m\mu m. After removing bands caused by water vapor absorption and low SNRs, the subregion consisting of 336×336×93336\times 336\times 93 pixels were chosen from the full dataset as a reference HR-HSI.

- •

Indian Pines: The Indian Pines has 224 spectral bands with a size of 145×145145\times 145 pixels. the spectral wavelength range is from 0.4 to 2.5 μ​m\mu m.
After removing bands caused by water vapor absorption and low SNRs, the subregion consisting of 144×144×191144\times 144\times 191 pixels was chosen from the full dataset as a reference HR-HSI.

- •

Washington DC Mall: The Washington DC Mall has a region of 1280×3071280\times 307 pixels, and the image consists of 210 bands with spectral wavelength ranging from 0.4 to 2.5 μ​m\mu m. After removing the low SNRs and atmospheric absorption bands, 191 bands were kept. The subregion consisting of 304×304×191304\times 304\times 191 pixels was clipped from the full dataset as a reference HR-HSI.

- •

University of Houston: The University of Houston contains 601×2384601\times 2384 pixels and 48 bands ranging from 0.38 to 1.05 μ​m\mu m. The sub-image consisting of 320×320×46320\times 320\times 46 pixels was chosen from the whole dataset as a reference HR-HSI for our experiments.

- •

Peppers and Superballs: We utilize the widely-used CAVE dataset Yasuma et al. ((2010)), which is a ground-based HSI dataset commonly employed in HSI-MSI fusion. The dataset comprises 32 high-resolution HSIs, each with a size of 512×512×31512\times 512\times 31.

- •

Real Data: The LR-HSI is collected by the Hyperion sensor Yang et al. ((2018)), which is of the size 120×120×89120\times 120\times 89. The HR-MSI with 13 bands is taken by the Sentinel-2A satellite. Four bands are employed for the test, and the spatial downsampling factor is 3, i.e., the size of
HR-MSI is 360×360×4360\times 360\times 4. These four bands are extracted from bands 2, 3, 4, and 8, with the central wavelengths being 490, 560, 665, and 842 nm, respectively.

2) Comparison methods: Since our method is an unsupervised algorithm, we mainly compare it to unsupervised fusion methods; such a choice also aligns with the practical demands of real-world scenarios. Specifically, we compare the proposed method with several SOTA unsupervised fusion methods, including
CNMF Yokoya et al. ((2012)), HySure Simoes et al. ((2014)), HyCoNet Zheng et al. ((2020)),
CNN-FUS Dian et al. ((2021)), MSE-SURE Nguyen et al. ((2022)), and E2E-FUS Z. Wang et al. ((2023)). For baselines: CNMF, HySure, CNN-FUS, and E2E-FUS methods are implemented using MATLAB R2023a on Windows 11 with 32GB RAM, while HyCoNet, MSE-SURE, CLoRF, and supervised methods are evaluated on an RTX 4090 GPU with 32GB RAM. Besides, parameters in baselines are manually fine-tuned to achieve optimal results in all experiments.

3) Evaluation metrics: Four distinct quantitative metrics are employed to assess the efficacy of fusion results, where ground truth (GT) is given. These metrics include the mean structured similarity (MSSIM), an extension of SSIM to assess HSI quality by averaging across all spectral bands; mean peak signal-to-noise ratio (MPSNR), which is calculated as the average PSNR across all bands extended for HSI; the spectral angle mapper (SAM) index; and the relative global dimension error (ERGAS) index.

4) Hyperparameters of CLoRF:
For all tasks, both the spatial-INR and spectral-INR adopt the SIREN network with an initial network parameter ω0=30\omega_{0}=30. The spatial-INR has 5 hidden layers with a size of 512, while the spectral-INR has 2 hidden layers with a size of 128 (In Indian Pines, the spatial-MLP has 3 hidden layers with a size of 512, while the spectral-MLP also has 3 hidden layers with a size of 256). Spatial-INR and spectral-INR are jointly optimized based on the loss function 7, using the Adam optimizer for optimization.
The learning rate is set to 3e-5, and the training epochs are fixed to 30000. An early stopping strategy is implemented to prevent overfitting for all datasets. Furthermore, the hyperparameters are summarized in Table 1.

*Table 1: Hyperparameters used for training the proposed model.*

Hyperparameter
Pavia University
Pavia Center
Indian Pines
Washington DC
Houston
Peppers
Superballs

KK
9
9
22
11
10
10
10

λ\lambda
1.25
1.25
0.55
1.25
1
1
1

η\eta
0.0025
0.0050
0.0060
0.0025
0.0025
0.0025
0.0025

*Table 2: Quantitative performance comparison with different algorithms on the different datasets. The best results are bold-faced, and runner-ups are underlined. (MPSNR ↑, MSSIM ↑, SAM ↓, ERGAS ↓).*

Metric
CNMF
HySure
HyCoNet
CNN-FUS
MSE-SURE
E2E-FUS
CLoRF

Pavia University
MPSNR
32.70
32.86
40.10
41.57
41.31
40.92
42.23

MSSIM
0.92
0.93
0.97
0.98
0.98
0.98
0.99

SAM
3.55
5.81
3.10
2.38
2.20
2.31
2.05

ERGAS
3.63
3.71
1.41
1.38
1.44
1.50
1.31

Pavia Center
MPSNR
38.97
33.96
41.88
42.16
42.87
43.12
43.47

MSSIM
0.97
0.93
0.98
0.98
0.99
0.99
0.99

SAM
7.70
8.42
3.77
4.10
3.51
3.67
3.64

ERGAS
2.35
3.56
1.45
1.47
1.37
1.33
1.26

Indian Pines
MPSNR
27.48
26.54
29.44
31.07
30.85
31.68
31.96

MSSIM
0.94
0.93
0.93
0.94
0.95
0.95
0.95

SAM
3.11
3.97
2.87
2.70
2.54
2.49
2.56

ERGAS
2.52
2.84
2.10
1.78
1.93
1.73
1.70

Washington DC
MPSNR
28.47
26.37
33.35
32.24
33.52
36.15
36.75

MSSIM
0.94
0.94
0.98
0.96
0.98
0.98
0.99

SAM
4.94
7.67
3.51
4.07
2.56
2.56
2.64

ERGAS
3.22
4.79
2.67
2.37
1.98
1.53
1.44

University of Houston
MPSNR
31.71
29.41
37.02
37.38
39.75
38.84
40.55

MSSIM
0.95
0.92
0.97
0.97
0.99
0.98
0.99

SAM
2.77
4.77
2.50
2.46
1.61
1.76
1.47

ERGAS
2.10
3.13
1.19
1.31
0.90
1.06
0.83

Peppers
MPSNR
40.00
36.21
41.15
44.86
43.25
43.23
47.65

MSSIM
0.96
0.94
0.96
0.99
0.99
0.98
0.99

SAM
11.62
10.59
4.77
7.36
6.34
7.02
4.01

ERGAS
4.92
9.04
6.29
2.65
2.87
2.94
1.94

Superballs
MPSNR
43.88
39.14
43.05
45.26
44.74
45.80
46.14

MSSIM
0.97
0.96
0.98
0.98
0.98
0.98
0.98

SAM
10.84
10.40
7.98
7.47
6.84
7.42
6.40

ERGAS
3.58
5.82
5.98
2.92
3.10
2.72
2.71

![Refer to caption](2405.17818v2/Fig3.png)

![Refer to caption](2405.17818v2/Fig4.png)

*Figure 3: The first row shows the Pavia University image (1st band) of the estimated HR-HSI, and the second row shows the error map between the estimated image and GT.*

![Refer to caption](2405.17818v2/Fig5.png)

![Refer to caption](2405.17818v2/Fig6.png)

*Figure 4: The first row shows the Pavia Center image (8th band) of the estimated HR-HSI, and the second row shows the error map between the estimated image and GT.*

![Refer to caption](2405.17818v2/Fig7.png)

![Refer to caption](2405.17818v2/Fig8.png)

*Figure 5: The first row shows the Indian Pines image (160th band) of the estimated HR-HSI, and the second row shows the error map between the estimated image and GT.*

![Refer to caption](2405.17818v2/Fig9.png)

![Refer to caption](2405.17818v2/Fig10.png)

*Figure 6: The first row shows the Washington DC image (5th band) of the estimated HR-HSI, and the second row shows the error map between the estimated image and GT.*

![Refer to caption](2405.17818v2/Fig11.png)

![Refer to caption](2405.17818v2/Fig12.png)

*Figure 7: The first row shows the University of Houston image (46th band) of the estimated HR-HSI, and the second row shows the error map between the estimated image and GT.*

![Refer to caption](2405.17818v2/Fig36.png)

![Refer to caption](2405.17818v2/Fig37.png)

*Figure 8: The first row shows the Peppers image (22th band) of the estimated HR-HSI, and the second row shows the error map between the estimated image and GT.*

![Refer to caption](2405.17818v2/Fig38.png)

![Refer to caption](2405.17818v2/Fig39.png)

*Figure 9: The first row shows the Superballs image (30th band) of the estimated HR-HSI, and the second row shows the error map between the estimated image and GT.*

*Figure 10: Reconstructed spectral signatures of two randomly selected locations at different datasets. From left to right, they are Pavia University, Indian Pines, Washington DC, and University of Houston. To provide a clearer comparison, we have compared CLoRF alongside E2E-FUS and GT.*

#### 4.2 Evaluation on Synthetic Data

Here, we consider synthetic data and evaluate proposed CLoRF under various scenarios.

1) HSI-MSI fusion:
The fusion performance of all methods on seven datasets, evaluated across MPSNR, MSSIM, SAM, and ERGAS metrics, is shown in Table 2. CLoRF consistently outperforms others in terms of quantitative results across most scenarios, affirming the efficacy of the HSI-MSI fusion task. Moreover, CLoRF exhibits commendable performance in spatial structures. our method outperforms E2E-FUS by 2.19% in terms of MPSNR. Nonetheless, in certain datasets, particularly those with a high number of bands, CLoRF slightly lags behind MSE-SURE in SAM index performance.
It is notable that MSE-SURE, throughout its training phase, is equipped with knowledge of noise levels in LR-HSI and HR-MSI, and leverages back-projection techniques to enhance spectral detail capture, while our method does not require this information.

To assess the effectiveness of various methods in preserving spatial structures, Fig. 3-9 illustrates estimated HR-HSI on seven simulated datasets. The first rows exhibit fusion results virtually, showcasing the 1st band of Pavia University, the 8th band of Pavia Center, the 160th band of Indian Pines, the 5th band of Washington DC Mall, the 46th band of the University of Houston, the 22nd band of the Peppers, and the 30th band of the Superballs. Conversely, the second rows depict corresponding error maps, illustrating the mean squared error (MSE) between GT and the estimated HR-HSI. While the results for the images in the first rows appear similar across most methods, with differences almost imperceptible to the naked eye, notable disparities emerge in the error maps depicted in the second column. It is evident that CLoRF exhibits minimal noise and preserves superior spatial structures. As shown in Fig. 10, we plot the two randomly selected spectral vectors reconstructed by CLoRF and E2E-FUS with four datasets. It is apparent that CLoRF effectively preserves the high-frequency information compared to E2E-FUS in some pixels. However, in some specific pixels, the performance of the E2E-FUS method is superior to CLoRF.

2) Runtime and complexity analysis: We have conducted a runtime and complexity analysis of various methods on the Houston dataset, with the experimental results reported in Table 3. Please note that the runtime for supervised learning methods includes both training and inference time. Plug-and-play methods require pre-trained networks. In comparison to self-supervised learning methods, CLoRF demonstrates a significant advantage in terms of runtime efficiency and GFLOPs. Additionally, CLoRF excels in running time and model parameters when compared to supervised learning baselines.

*Table 3: Runtime, parameters, and GFLOPs of all methods for Houston dataset. The best results are bold-faced.*

Methods
Times (s)
Paras (M)
GFLOPs

Model-based
CNMF
5.70 (CPU)
/
/

HySure
57.60 (CPU)
/
/

Plug-and-Play
CNN-FUS
114.06 (CPU)
/
/

E2E-FUS
107.00 (CPU)
/
/

Deep-learning
HyCoNet
1332.38 (GPU)
0.58
152.63

MSE-SURE
478.23 (GPU)
1.10
149.92

(Self-supervised)
CLoRF
446.86 (GPU)
1.36
121.75

Deep-learning
ConSSFCNN
473.51 (GPU)
4.79
78.37

ResTFNet
525.07 (GPU)
23.24
84.73

(Supervised)
SSR-Net
504.51 (GPU)
2.87
46.69

MCT-Net
1128.71 (GPU)
36.09
117.86

3) Performance on different downsampling ratios:
We evaluate the model’s performance when there is a significant difference in resolution between the input LR-HSI and the desired HR-HSI. Specifically, we choose the downsampling ratio in {4,8,16}\{4,8,16\}. The experimental results are shown in Table 4, where we observe that our method still demonstrates its advantage when there is a large resolution gap between the LR-HSI and HR-HSI.

*Table 4: Performance on different downsampling ratios with 30 dB noise for Pavia University. The best results are bold-faced.*

Ratios
Methods
MPSNR
MSSIM
SAM
ERGAS

4
CNMF
32.70
0.92
3.55
3.63

HySure
32.86
0.93
5.81
3.71

HyCoNet
40.10
0.97
3.10
1.41

CNN-FUS
41.57
0.98
2.38
1.38

MSE-SURE
41.31
0.98
2.20
1.44

E2E-FUS
40.92
0.98
2.31
1.50

CLoRF
42.23
0.99
2.05
1.31

8
CNMF
24.23
0.74
7.67
4.82

HySure
23.87
0.74
12.55
4.88

HyCoNet
39.67
0.97
3.18
0.73

CNN-FUS
40.09
0.97
2.76
0.85

MSE-SURE
40.71
0.98
2.40
0.87

E2E-FUS
40.29
0.98
2.61
0.83

CLoRF
41.56
0.98
2.29
0.69

16
CNMF
21.27
0.62
12.32
3.50

HySure
17.54
0.58
21.06
4.67

HyCoNet
39.47
0.97
3.25
0.37

CNN-FUS
40.08
0.98
2.70
0.42

MSE-SURE
39.63
0.98
2.92
0.44

E2E-FUS
40.21
0.98
2.63
0.42

CLoRF
41.11
0.98
2.33
0.37

4) Comparison with supervised methods: To provide a broader context for the performance of the proposed approach, we compare it with recent four supervised methods, such as ConSSFCNN Han et al. ((2018)), ResTFNet X. Liu et al. ((2020)), SSR-Net X. Zhang et al. ((2020)), MCT-Net X. Wang, Wang et al. ((2023)). For fairness, since our approach is self-supervised, we evaluate it on the same test sets of the datasets used by the supervised methods, as shown in Table 5. Our method demonstrates comparable performance to MCT-Net and even outperforms the supervised methods on the Pavia Center dataset.

*Table 5: Performance of different supervised methods on two datasets, and the downsampling ratio is 4 with 30 dB noise. The best results are bold-faced.*

5) Performance under degeneration with estimated PSF and SRF: To test the impact of degradation operators in the spatial and spectral domains on HR-HSI fusion results, we assume that the degradation operators are unknown in the simulated dataset, and we estimate these operators using the method suggested in Simoes et al. ((2014)). In Table 6, we present the results of both semi-blind and fully-blind experiments. Note that the CNMF and HySure methods are fully blind in the experiment. They are not included in the semi-blind experiments for comparison. As shown in Table 6, blind fusion is more challenging than semi-blind fusion, all methods perform similarly underestimated degradations in the spatial domain. However, in blind fusion, the performance of all methods decreases to some extent. Nevertheless, our method still demonstrates greater robustness in comparison to other approaches.

*Table 6: Performance under generation produced by semi-blind (estimate PSF) and blind (estimate both PSF and SRF) with 30 dB noise and the downsampling ratio is 4 for Pavia University. The best results are bold-faced.*

Methods
MPSNR
MSSIM
SAM
ERGAS

Semi-blind
HyCoNet
40.10
0.97
3.10
1.41

CNN-FUS
40.04
0.97
2.55
1.64

MSE-SURE
41.11
0.98
2.20
1.49

E2E-FUS
40.77
0.98
2.34
1.52

CLoRF
41.40
0.98
2.32
1.40

Blind
CNMF
32.70
0.92
3.55
3.63

HySure
32.86
0.93
5.81
3.71

HyCoNet
38.06
0.95
3.97
1.83

CNN-FUS
38.32
0.96
3.02
2.12

MSE-SURE
36.14
0.96
4.36
2.90

E2E-FUS
31.33
0.84
7.53
5.34

CLoRF
38.98
0.97
3.00
1.88

#### 4.3 Arbitrary Resolution

Here, we evaluate the performance of CLoRF to fuse HSIs at arbitrary spatial and spectral resolutions. As shown in Fig. 1, our model is well-trained based in an unsupervised manner, and we can input any scale spatial and spectral position coordinates during the inference stage, obtaining arbitrary resolutions in both spatial and spectral domains.
We use PSF and downsampling ratio of the same size as in Sec. 4.1, but different sizes of SRF (sampled from SRF in Sec. 4.1) to synthesize the reduced-resolution LR-HSI (42×42×5042\times 42\times 50) and HR-MSI (168×168×4168\times 168\times 4) with Pavia University for training spatial-INR and spectral-INR. Then, we predict HR-HSIs of arbitrary resolutions using different scales of spatial and spectral coordinates. For a fair comparison, our method is directly compared with bicubic interpolation from the original GT. As shown in Table 7, we can fix spectral resolution to obtain arbitrary spatial resolution, fix spatial resolution to obtain arbitrary spectral resolution, or simultaneously achieve resolutions in both spatial and spectral domains. CLoRF performs nearly as well as bicubic interpolation in spatial resolution when spectral resolution is fixed, yet it notably surpasses bicubic when spatial resolution is fixed while achieving spectral resolution. Moreover, when simultaneously enhancing resolutions in both spatial and spectral domains, CLoRF consistently outperforms bicubic interpolation.
From Fig. 11-12, it can be seen that when simultaneously upsampling in both spatial and spectral domains, CLoRF obtains finer details compared to the bicubic interpolation.
Furthermore, we train a smaller size of data (LR-HSI (42×42×5042\times 42\times 50) and HR-MSI (168×168×4168\times 168\times 4) with Pavia University for training spatial-INR and spectral-INR) and infer the desired HR-HSI with spatial super-resolution factor in {2,4,8,16}\{2,4,8,16\} and a fixed spectral upsampling resolution 93.
Figure 13 demonstrates that the trained CLoRF model successfully super-resolves the original HR-HSI (168×168×50)(168\times 168\times 50) to arbitrary resolutions with a large range of flexibility. However, at a super-resolution factor of 16, the image exhibits slight fluctuations within some regions, which shows the inherent limitations.

![Refer to caption](2405.17818v2/Fig15.png)

*Figure 11: An example of CLoRF for super-resolution on the Pavia Unversity with a spatial resolution of 336×336336\times 336 (60th band) and its corresponding error map while simultaneously achieving arbitrary resolutions in both spatial and spectral domains.*

*Figure 12: Visualize the spectral signatures of different pixels in different spectral resolutions while simultaneously obtaining arbitrary resolutions in both spatial and spectral domains.*

![Refer to caption](2405.17818v2/Fig40.png)

*Figure 13: The original resolution of HR-HSI is (168,168,50)(168,168,50), the spatial super-resolution factor in {2,4,8,16}\{2,4,8,16\}. Visualize the 40th band at different resolutions for Pavia University.*

*Table 7: Experimental results in spatial (fixed spectral), spectral (fixed spatial), and (spatial, spectral) with arbitrary resolution, metric: MPSNR/MSSIM. The original dimension of HR-HSI is (168, 168, 50). The best results are bold-faced.*

Arbitrary resolution in the spectral domain

Dimension
(168,168,61)
(168,168,72)
(168,168, 83)
(168,168,93)

Bicubic
37.64 /0.97
36.60/0.95
38.76/0.96
37.23/0.95

CLoRF
42.42/0.99
42.85/0.98
42.64/0.98
42.56/0.98

Arbitrary resolution in the spatial domain

Dimension
(210,210,50)
(252,252,50)
(294,294,50)
(336,336,50)

Bicubic
30.07/0.90
29.68/0.89
30.24/0.89
29.29/0.88

CLoRF
29.86/0.88
29.52/0.87
30.04/0.86
29.22/0.86

Arbitrary resolution in the spectral and spatial domain

Dimension
(210,210,61)
(252,252,72)
(294,294,83)
(336,336,93)

Bicubic
28.87/0.87
28.00/0.85
28.56/0.85
27.48/0.83

CLoRF
29.53/0.88
29.21/0.87
29.60/0.86
28.61/0.84

#### 4.4 Ablation Study

We conduct a series of ablation studies to verify the effectiveness of CLoRF.

1) Loss function: As shown in Table 8, incorporating TV loss into the loss function can lead to better recovery quality. On each dataset, the TV loss is capable of enhancing PSNR by 1-2 dB.

*Table 8: Ablation study of the TV loss. The best results are bold-faced.*

Metric
w/o
w/

Pavia University
MPSNR
41.48
42.23

MSSIM
0.98
0.99

University of Houston
MPSNR
39.04
40.55

MSSIM
0.98
0.99

*Table 9: Ablation study of the different activation functions. (MPSNR / MSSIM). The best results are bold-faced.*

Act.Func
Pavia University
Houston

ReLU
25.92/0.65
24.58/0.54

ReLU+PE
36.26/0.93
37.56/0.98

Gauss
35.96/0.90
36.10/0.96

Spder
41.63/0.98
39.16/0.98

Sine
42.23/0.99
40.55/0.99

2) Different activation functions:
Table 9 displays the results obtained using different activation functions, such as ReLU, ReLU+Position Encoding (PE), Gauss Ramasinghe & Lucey ((2022)), Spder Shah & Sitawarin ((2024)), and Sine Sitzmann et al. ((2020)). Notably, the Sine activation function demonstrates outstanding performance across all metrics. Therefore, we have made it the default activation function for our model.

3) Impact of hyper-parameters: We examine six hyperparameters: KK, λ\lambda, η\eta, the hidden depth of two MLPs, and the learning rate (LR). Each hyperparameter is explored within a specified range while the others are held constant. Specifically, for Pavia University, we explore KK within the range of 5 to 17. For λ\lambda, we consider values from the set {0.5,0.75,1,1.25,1.5,1.75}\{0.5,0.75,1,1.25,1.5,1.75\}, For η\eta, the search space consists of {10−3,2.5×10−3,5×10−3,7.5×10−3,10−2}\{10^{-3},2.5\times 10^{-3},5\times 10^{-3},7.5\times 10^{-3},10^{-2}\}. Regarding the hidden depth of the two MLPs, we initially set the hidden depth of the spectral MLP at 2 and explored the depth of the spatial MLP from 1 to 7. Similarly, we fix the hidden depth of the spatial MLP at 5 and then search for the hidden depth of the spectral MLP from 1 to 6. For LR, we consider values from the set {103,5×103,104,5×104,105,3×105,5×105}\{10^{3},5\times 10^{3},10^{4},5\times 10^{4},10^{5},3\times 10^{5},5\times 10^{5}\}. The results are displayed in Fig. 14.

*Figure 14: Impact of hyperparameters on MPSNR and MSSIM. The red dot represents the best result obtained by traversing all values.*

#### 4.5 Evaluation on Real Dataset

To further illustrate the effectiveness of the proposed method, we conduct a real-world fusion experiment.
It is important to note that the ground-truth spatial and spectral degradation operators are not available for real data. Consequently, we estimate these operators using the approach suggested in Simoes et al. ((2014)). For parameter selection in real data, we first apply the classical noise estimation algorithm proposed in Bioucas-Dias & Nascimento ((2008)) to obtain a preliminary estimate of the noise intensity of the LR-HSI, which is SNR=32.75 dB. Since the estimated noise intensity is close to SNR = 30 dB, In our method, we set KK and η\eta to be consistent with Pavia University, and λ=1.8\lambda=1.8. The experimental parameters for the other methods remain consistent with those of the Pavia University dataset.

Since there is no ground-truth for the HR-HSI, we visualize the estimated HR-HSIs of all methods in Fig. 15 together with the image quality score measured by a non-reference image quality metric Yang et al. ((2017)). We find that the proposed CLoRF recovers more details and obtains the best image quality.

![Refer to caption](2405.17818v2/Fig31.png)

*Figure 15: Visual the real data (14th band) of the estimated HR-HSI, evaluated with a no-reference hyperspectral imaging quality score for various methods applied to the real HypSen dataset in blind fusion tasks.*

#### 4.6 PAN-HSI Fusion

In this section, we extend our proposed fusion method to the PAN-HSI fusion. PAN image, compared to MSI, has fewer bands, making fusion more challenging. Below are the results and details of some experiments we conducted.

1) Dataset: The dataset acquired for the PRISMA contest1 Vivone et al. ((2023)), namely RR1. The image is of size 900×900×59900\times 900\times 59. The synthetic datasets, plus a simulated PSF and SRF, were used to generate two observed images: LR-HSI and PAN. PSF is set to be the same as the 4.1. The wavelength Vivone et al. ((2023)) was used to simulate the SRF to generate the PAN image. The i.i.d Gaussian noise is added to the LR-HSI (30 dB) and PAN (30 dB) image.

2) Compared Methods: We primarily compare CLoRF with two categories of baselines: model-based methods include GS Laben & Brower ((2000)), GSA Aiazzi et al. ((2007)), AWLP Vivone et al. ((2014)), MTF-GLP Aiazzi et al. ((2006)), and HySure Simoes et al. ((2014)), while the unsupervised deep learning method is R-PNN Guarino et al. ((2024)). All training parameters for CLoRF remain the same for Pavia University.

As shown in Table 10, CLoRF outperforms the other methods in terms of MPSNR and ERGAS. However, due to spectral distortion in the PAN image, CLoRF fails to learn a continuous representation in the spectral domain effectively.
For clarity, we depict error maps for two scenarios: noise-free and noisy, as shown in Fig. 16. The results indicate that CLoRF outperforms other methods in spatial domains.

*Table 10: Quantitative performance comparison with different algorithms on the RR1 dataset. The best results are bold-faced, and runner-ups are underlined. (MPSNR ↑, MSSIM ↑, SAM ↓, ERGAS ↓).*

Metric
GS
GSA
AWLP
MTF-GLP
HySure
R-PNN
CLoRF

Noise-free
MPSNR
27.40
32.25
29.64
30.13
29.56
30.05
32.87

MSSIM
0.84
0.92
0.90
0.90
0.88
0.90
0.91

SAM
9.70
4.46
4.90
4.41
6.49
4.78
5.81

ERGAS
5.27
3.06
3.92
3.75
4.27
4.28
2.95

Metric
GS
GSA
AWLP
MTF-GLP
HySure
R-PNN
CLoRF

Noisy
MPSNR
27.25
31.90
29.23
29.74
29.44
29.33
32.61

MSSIM
0.82
0.90
0.87
0.88
0.88
0.88
0.89

SAM
10.05
5.51
5.96
5.76
7.07
6.05
6.50

ERGAS
5.35
3.16
4.08
3.90
4.12
4.60
2.99

![Refer to caption](2405.17818v2/Fig30.png)

![Refer to caption](2405.17818v2/Fig29.png)

*Figure 16: The first and second rows show the error images (Noise-free and Noisy) between the estimated HR-HSI and GT.*

### 5 Conclusion

In this work, we introduced an innovative continuous low-rank factorization representation for HSI-MSI fusion, which incorporates two INRs into the decomposition to capture spatial and spectral information, respectively. Theoretical analysis reveals that this continuous function representation adeptly portrays the low-rank and smoothness priors of HSIs. Extensive numerical experiments conducted for HSI-MSI fusion confirm its effectiveness and wide applicability. Nonetheless, our method still faces certain limitations. Given its unsupervised manner, CLoRF demands an extensive number of training epochs. Another limitation is that we solely utilize SIREN, which is widely used in INR. There’s a need to explore diverse continuous representations for both spatial and spectral domains.
The adoption of continuous low-rank factorization representation for processing and analyzing HSIs shows potential for future applications across various tasks, e.g., HSI unmixing and single RGB-HSI super-resolution. Our future efforts will be dedicated to expanding the versatility of CLoRF to address diverse HSI tasks.

##### Acknowledgements

This work was partly supported by
the Natural Science Foundation
of China (Nos. 12201286, 52303301), the Shenzhen Science and Technology Program (20231115165836001), HKRGC Grant No.CityU11301120, CityU Grant No. 9229120, HKRGC GRF 17201020 and 17300021, HKRGC C7004-21GF, and Joint NSFC and RGC N-HKU769/21, National Key R&D Program of China (2023YFA1011400), and the Shenzhen Fundamental Research Program (JCYJ20220818100602005).

##### Data Availability

Data will be made available on reasonable request.

### 6 Proof of Proposition 1

Since 𝐗∈S⁡[f]\mathbf{X}\in S[f], we directly get MF-rank[f]≥rank⁡(𝐗)[f]\geq\mathrm{rank}({\mathbf{X}}).
Now, we aim to prove the other side: MF-rank[f]≤rank⁡(𝐗)[f]\leq\mathrm{rank}({\mathbf{X}}).

Let 𝐌\mathbf{M} be any matrix within the set S⁡[f]S[f]. Each column vector of 𝐌\mathbf{M} is denoted by 𝐌(:,p)\mathbf{M}_{(:,p)} for p∈{1,2,…,n2}p\in\{1,2,\ldots,n_{2}\}. According to the definition of S⁡[f]S[f], there exists an index lp∈{1,2,…,n1}l_{p}\in\{1,2,\ldots,n_{1}\} dependent on pp such that 𝐌(:,p)\mathbf{M}_{(:,p)} is a permutation of the elements in 𝐗(:,lp)\mathbf{X}_{(:,l_{p})}, allowing for repeated sampling.
In other words, for each 𝐌(:,p)\mathbf{M}_{(:,p)}, there exists a permutation matrix 𝐏∈{0,1}n1×n1\mathbf{P}\in\{0,1\}^{n_{1}\times n_{1}} and a corresponding column of 𝐗\mathbf{X} depending on pp (specifically 𝐗(:,lp)\mathbf{X}_{(:,l_{p})}), such that 𝐌(:,p)=𝐏𝐗(:,lp)\mathbf{M}_{(:,p)}=\mathbf{P}\mathbf{X}_{(:,l_{p})}.
Additionally, the permutation matrix 𝐏\mathbf{P} is consistent across all columns 𝐌(:,p)\mathbf{M}_{(:,p)} for p=1,2,…,n2p=1,2,\ldots,n_{2}, i.e., 𝐌(:,p)=𝐏𝐗(:,lp)\mathbf{M}_{(:,p)}=\mathbf{P}\mathbf{X}_{(:,l_{p})} for each pp.

Define 𝐗~:=[𝐗(:,l1),𝐗(:,l2),…,𝐗(:,ln2)]∈ℝm1×n2\tilde{\mathbf{X}}:=[\mathbf{X}_{(:,l_{1})},\mathbf{X}_{(:,l_{2})},\ldots,\mathbf{X}_{(:,l_{n_{2}})}]\in\mathbb{R}^{m_{1}\times n_{2}}, we have that the rank of 𝐗~\tilde{\mathbf{X}} is less than or equal to the rank of 𝐗\mathbf{X}: rank⁡(𝐗~)≤rank⁡(𝐗)\mathrm{rank}(\tilde{\mathbf{X}})\leq\mathrm{rank}(\mathbf{X}).
Finally, since 𝐌=𝐏​𝐗~\mathbf{M}=\mathbf{P}\tilde{\mathbf{X}}, it follows that rank⁡(𝐌)≤rank⁡(𝐗)\mathrm{rank}(\mathbf{M})\leq\mathrm{rank}(\mathbf{X}), which leads to MF-rank[f]≤rank⁡(𝐗)[f]\leq\mathrm{rank}({\mathbf{X}}).

### 7 Proof of Theorem 3

First, we establish a linear representation for each factor function f⁡(𝐬,b)f(\mathbf{s},b) with fixed inputs 𝐬\mathbf{s} and bb, employing a set of basis functions. Then, we focus on presenting the continuous low-rank factorization and demonstrating that this factorization preserves the matrix factorization rank (MF-rank).

Suppose that MF-rank[f]=K[f]=K with K<∞K<\infty,
thus there exist a matrix 𝐌∈ℝn1×K\mathbf{M}\in\mathbb{R}^{n_{1}\times K} with rank(M)=K(M)=K. Denote: 𝒮={𝐬i|𝐌i,j=f(𝐬i,bj),i=1,…,n1},\mathcal{S}=\{\mathbf{s}_{i}\ |\ \mathbf{M}_{i,j}=f(\mathbf{s}_{i},b_{j}),i=1,\dots,n_{1}\}, and 𝒯={bj|𝐌i,j=f(𝐬i,bj),j=1,…,K}.\mathcal{T}=\{b_{j}\ |\ \mathbf{M}_{i,j}=f(\mathbf{s}_{i},b_{j}),j=1,\dots,K\}. It is easy to see that {𝐌(:,i)}i=1K\{\mathbf{M}_{(:,i)}\}_{i=1}^{K} are the column basis of S⁡[f]​⋃ℝn1S[f]\bigcup\mathbb{R}^{n_{1}}.

Given any matrix 𝐔∈ℝn1×n2​(n2≥K)\mathbf{U}\in\mathbb{R}^{n_{1}\times n_{2}}(n_{2}\geq K) as 𝐔(i,j)=f⁡(𝐬i,bj)\mathbf{U}_{(i,j)}=f(\mathbf{s}_{i},b_{j}) with 𝐬i∈𝒮,bj∈𝒵f\mathbf{s}_{i}\in\mathcal{S},b_{j}\in\mathcal{Z}_{f}, we have 𝐔∈S⁡[f]\mathbf{U}\in S[f] and rank⁡(𝐔)≤K\mathrm{rank}(\mathbf{U})\leq K. Furthermore, the column vector in 𝐔\mathbf{U} is a linear combination of the column basis {𝐌(:,i)}i=1K\{\mathbf{M}_{(:,i)}\}_{i=1}^{K}:

𝐔(:,j)=∑k=1Kck(bj)𝐌(:,k), for j=1,2,…,n2.\mathbf{U}_{(:,j)}=\sum_{k=1}^{K}c_{k}^{(b_{j})}\mathbf{M}_{(:,k)},\text{ for }j=1,2,\dots,n_{2}.

(9)

Here, we utilize Eq.(9) and rewrite f⁡(𝐬,b)f(\mathbf{s},b) by 𝐜(b)=[c1(b),c2(b),…,cK(b)]\mathbf{c}^{(b)}=[c^{(b)}_{1},c^{(b)}_{2},\dots,c^{(b)}_{K}]:

f⁡(𝐬,b)=∑k=1Kck(b)​𝐌(i,k)=∑k=1Kck(b)​f​(𝐬,bk),f(\mathbf{s},b)=\sum_{k=1}^{K}c_{k}^{(b)}\mathbf{M}_{(i,k)}=\sum_{k=1}^{K}c_{k}^{(b)}f(\mathbf{s},b_{k}),

(10)

for any 𝐬∈𝒮,b∈𝒵f.\mathbf{s}\in\mathcal{S},b\in\mathcal{Z}_{f}.

Next, we will generalize this conclusion from 𝐬∈𝒮\mathbf{\mathbf{}}\mathbf{s}\in\mathcal{S} to any 𝐬∈𝒜f\mathbf{s}\in\mathcal{A}_{f}.
Given 𝐬~∈𝒜f/𝒮\mathbf{\tilde{s}}\in\mathcal{A}_{f}/\mathcal{S} and we construct a matrix: 𝐓∈ℝ(n1+1)×n2\mathbf{T}\in\mathbb{R}^{(n_{1}+1)\times n_{2}}, where 𝐓(i,j)=f⁡(𝐬i,bj)\mathbf{T}_{(i,j)}=f(\mathbf{s}_{i},b_{j}) and 𝐬i∈𝒮\mathbf{s}_{i}\in\mathcal{S} for i=1,2,…,n1i=1,2,\dots,n_{1} and 𝐬n1+1=𝐬~\mathbf{s}_{n_{1}+1}=\mathbf{\tilde{s}}. Assume there exist KK column vectors {𝐓(:,jk)}k=1K\{\mathbf{T}_{(:,j_{k})}\}_{k=1}^{K} such that
𝐓(1:n1,jk)=𝐌(:,k),\mathbf{T}_{(1:n_{1},j_{k})}=\mathbf{M}_{(:,k)}, for k=1,2,…,Kk=1,2,\dots,K.
Hence, we get that rank⁡(𝐓)=K\mathrm{rank}(\mathbf{T})=K and for j=1,2,…,n2,j=1,2,\dots,n_{2},

𝐓(:,j)=∑k=1Kdk(bj)𝐓(:,jk),\mathbf{T}_{(:,j)}=\sum_{k=1}^{K}d_{k}^{(b_{j})}\mathbf{T}_{(:,j_{k})},

𝐓(1:n1,j)=∑k=1Kck(bj)𝐌(:,k).\mathbf{T}_{(1:n_{1},j)}=\sum_{k=1}^{K}c_{k}^{(b_{j})}\mathbf{M}_{(:,k)}.

Owing to the uniqueness of the coefficient vector, we get that 𝐝(bj)=𝐜(bj)\mathbf{d}^{(b_{j})}=\mathbf{c}^{(b_{j})}. Hence, we have

𝐓(n1+1,j)=∑k=1Kck(bj)​𝐓(n1+1,k),\mathbf{T}_{(n_{1}+1,j)}=\sum_{k=1}^{K}c_{k}^{(b_{j})}\mathbf{T}_{(n_{1}+1,k)},

(11)

which leads
f⁡(𝐬~,b)=∑k=1Kck(b)​f​(𝐬~,bk)f(\mathbf{\tilde{s}},b)=\sum_{k=1}^{K}c_{k}^{(b)}f(\mathbf{\tilde{s}},b_{k})
for any 𝐬~∈𝒜f/𝒮\mathbf{\tilde{s}}\in\mathcal{A}_{f}/\mathcal{S}. This gives the linear representation form of the factor function f⁡(𝐬,b)f(\mathbf{s},b) (with fixed 𝐬\mathbf{s} and bb) using some basis functions f⁡(𝐬,bk)f(\mathbf{s},b_{k}) with bk∈𝒯b_{k}\in\mathcal{T}.

We define the factor function fspatial​(⋅):𝒜f→ℝKf_{\mathrm{spatial}}(\cdot):\mathcal{A}_{f}\rightarrow\mathbb{R}^{K} as

fspatial​(𝐬~):=[f⁡(𝐬~,b1),f⁡(𝐬~,b2),…,f⁡(𝐬~,bK)]T.f_{\mathrm{spatial}}(\mathbf{\tilde{s}}):=[f(\mathbf{\tilde{s}},b_{1}),f(\mathbf{\tilde{s}},b_{2}),\dots,f(\mathbf{\tilde{s}},b_{K})]^{T}.

Also, define the matrix function h⁡(⋅):𝒩(K)×𝒵f→ℝh(\cdot):\ \mathcal{N}^{(K)}\times\mathcal{Z}_{f}\rightarrow\mathbb{R} as

h⁡(i,b):=ci(b),h(i,b):=c_{i}^{(b)},

where 𝒩(K)={1,2,…,K}.\mathcal{N}^{(K)}=\{1,2,\dots,K\}.
From the above analysis, we see that for any (𝐬,b)∈𝒟f=𝒜f×𝒵f(\mathbf{s},b)\in\mathcal{D}_{f}=\mathcal{A}_{f}\times\mathcal{Z}_{f}, it holds that

f⁡(𝐬,b)=∑k=1Kh⁡(k,b)​(fspatial​(𝐬))(k).f(\mathbf{s},b)=\sum_{k=1}^{K}h(k,b)(f_{\mathrm{spatial}}(\mathbf{s}))_{(k)}.

(12)

Denote
fspectral​(⋅):𝒵f→ℝKf_{\mathrm{spectral}}(\cdot):\mathcal{Z}_{f}\rightarrow\mathbb{R}^{K} as

fspectral​(b):=[h⁡(1,b),h⁡(2,b),⋯,h⁡(K,b)]T∈ℝK×1,f_{\mathrm{spectral}}(b):=[h(1,b),h(2,b),\cdots,h(K,b)]^{T}\in\mathbb{R}^{K\times 1},

then Eq.(12) is rewritten as

f⁡(𝐬,b)=fspatial​(𝐬)⋅fspectralT​(b).f(\mathbf{s},b)=f_{\mathrm{spatial}}(\mathbf{s})\cdot f^{T}_{\mathrm{spectral}}(b).

(13)

### 8 Proof of Theorem 4

For any (𝐬1,b1),(𝐬2,b2)∈𝒟f(\mathbf{s}_{1},b_{1}),(\mathbf{s}_{2},b_{2})\in\mathcal{D}_{f}, we have

|f⁡(𝐬1,b1)−f⁡(𝐬2,b1)|\displaystyle|f(\mathbf{s}_{1},b_{1})-f(\mathbf{s}_{2},b_{1})|

(14)

=\displaystyle=
|Φα​(𝐬1)⋅ΨθT​(b1)−Φα​(𝐬2)⋅ΨθT​(b1)|\displaystyle|\Phi_{\alpha}(\mathbf{s}_{1})\cdot\Psi_{\theta}^{T}(b_{1})-\Phi_{\alpha}(\mathbf{s}_{2})\cdot\Psi_{\theta}^{T}(b_{1})|

≤\displaystyle\leq
|(Φα​(𝐬1)−Φα​(𝐬2))⋅ΨθT​(b1)|\displaystyle|(\Phi_{\alpha}(\mathbf{s}_{1})-\Phi_{\alpha}(\mathbf{s}_{2}))\cdot\Psi_{\theta}^{T}(b_{1})|

≤\displaystyle\leq
‖Φα​(𝐬1)−Φα​(𝐬2)‖1​‖ΨθT​(b1)‖1.\displaystyle\|\Phi_{\alpha}(\mathbf{s}_{1})-\Phi_{\alpha}(\mathbf{s}_{2})\|_{1}\|\Psi_{\theta}^{T}(b_{1})\|_{1}.

Note that σ⁡(⋅)\sigma(\cdot) is Lipschitz continuous, i.e., |σ⁡(x)−σ⁡(y)|≤κ​|x−y||\sigma(x)-\sigma(y)|\leq\kappa|x-y| holds for any x,y,x,y, and letting
y=0y=0 derives σ⁡(x)≤κ​|x|\sigma(x)\leq\kappa|x| since σ⁡(0)=0\sigma(0)=0.
On the other hand, denote ψ(1)​(b)=𝐖11​t\psi^{(1)}(b)=\mathbf{W}_{1}^{1}t and ψ(k)​(b)=𝐖k1​σ​(ψ(k−1)​(b))\psi^{(k)}(b)=\mathbf{W}_{k}^{1}\sigma(\psi^{(k-1)}(b)), and ‖𝐖i1‖1,‖𝐖i2‖1\|\mathbf{W}_{i}^{1}\|_{1},\|\mathbf{W}_{i}^{2}\|_{1} are bounded by a positive constant η\eta for all ii. So we get:

‖Ψθ​(b)‖1=‖ψ(d)​(b)‖1\displaystyle\|\Psi_{\theta}(b)\|_{1}=\|\psi^{(d)}(b)\|_{1}

(15)

≤\displaystyle\leq
‖𝐖d1‖1​‖σ⁡(ψ(d−1)​(b))‖1\displaystyle\|\mathbf{W}_{d}^{1}\|_{1}\|\sigma(\psi^{(d-1)}(b))\|_{1}

≤\displaystyle\leq
η​κ​‖ψ(d−1)​(b)‖1≤ηd​κd−1​|b|.\displaystyle\eta\kappa\|\psi^{(d-1)}(b)\|_{1}\leq\eta^{d}\kappa^{d-1}|b|.

Meanwhile, we denote ϕ(1)​(𝐬)=𝐖12​𝐬\phi^{(1)}(\mathbf{s})=\mathbf{W}_{1}^{2}\mathbf{s} and ϕ(k)​(𝐬)=𝐖k2​σ​(ϕ(k−1)​(𝐬))\phi^{(k)}(\mathbf{s})=\mathbf{W}_{k}^{2}\sigma(\phi^{(k-1)}(\mathbf{s})). Then
it holds that

‖Φα​(𝐬1)−Φα​(𝐬2)‖1\displaystyle\|\Phi_{\alpha}(\mathbf{s}_{1})-\Phi_{\alpha}(\mathbf{s}_{2})\|_{1}

(16)

=\displaystyle=
‖ϕ(d)​(𝐬1)−ϕ(k)​(𝐬2)‖1\displaystyle\|\phi^{(d)}(\mathbf{s}_{1})-\phi^{(k)}(\mathbf{s}_{2})\|_{1}

=\displaystyle=
∥𝐖d2(σ(ϕ(d−1)(𝐬1))−σ(ϕ(d)(𝐬2))∥1\displaystyle\|\mathbf{W}_{d}^{2}(\sigma(\phi^{(d-1)}(\mathbf{s}_{1}))-\sigma(\phi^{(d)}(\mathbf{s}_{2}))\|_{1}

≤\displaystyle\leq
η​κ​‖ϕ(d−1)​(𝐬1)−ϕ(d−1)​(𝐬2)‖1\displaystyle\eta\kappa\|\phi^{(d-1)}(\mathbf{s}_{1})-\phi^{(d-1)}(\mathbf{s}_{2})\|_{1}

≤\displaystyle\leq
ηd​κd−1​‖𝐬1−𝐬2‖1,\displaystyle\eta^{d}\kappa^{d-1}\|\mathbf{s}_{1}-\mathbf{s}_{2}\|_{1},

Let ζ=max⁡{‖𝐬1‖1,|b1|}\zeta=\max\{\|\mathbf{s}_{1}\|_{1},|b_{1}|\}. Combining the inequalities Eq.(15) and Eq.(16), we have

|f⁡(𝐬1,b1)−f⁡(𝐬2,b1)|\displaystyle|f(\mathbf{s}_{1},b_{1})-f(\mathbf{s}_{2},b_{1})|

(17)

≤\displaystyle\leq
η2​d+1​κ2​d−2​|b1|​‖𝐬1−𝐬2‖1\displaystyle\eta^{2d+1}\kappa^{2d-2}|b_{1}|\|\mathbf{s}_{1}-\mathbf{s}_{2}\|_{1}

≤\displaystyle\leq
η2​d+1​κ2​d−2​ζ​‖𝐬1−𝐬2‖1.\displaystyle\eta^{2d+1}\kappa^{2d-2}\zeta\|\mathbf{s}_{1}-\mathbf{s}_{2}\|_{1}.

Similarly, we prove that

|f⁡(𝐬2,b1)−f⁡(𝐬2,b2)|≤η2​d+1​κ2​d−2​ζ​|b1−b2|.|f(\mathbf{s}_{2},b_{1})-f(\mathbf{s}_{2},b_{2})|\leq\eta^{2d+1}\kappa^{2d-2}\zeta|b_{1}-b_{2}|.

Combining the above two inequalities, we get

|f⁡(𝐬1,b1)−f⁡(𝐬2,b2)|≤|f⁡(𝐬1,b1)−f⁡(𝐬2,b1)|+|f⁡(𝐬2,b1)−f⁡(𝐬2,b2)|≤δ​‖𝐬1−𝐬2‖1+δ​|b1−b2|,\begin{split}|f(\mathbf{s}_{1},b_{1})-f(\mathbf{s}_{2},b_{2})|\leq&|f(\mathbf{s}_{1},b_{1})-f(\mathbf{s}_{2},b_{1})|\\
&+|f(\mathbf{s}_{2},b_{1})-f(\mathbf{s}_{2},b_{2})|\\
\leq&\delta\|\mathbf{s}_{1}-\mathbf{s}_{2}\|_{1}+\delta|b_{1}-b_{2}|,\end{split}

where δ=η2​d+1​κ2​d−2​ζ.\delta=\eta^{2d+1}\kappa^{2d-2}\zeta.

## References

- Aiazzi et al. ((2006))

Aiazzi, B., Alparone, L., Baronti, S., Garzelli, A. & Selva, M.

(2006).

MTF-tailored multiscale fusion of high-resolution MS
and Pan imagery.

Photogrammetric Engineering & Remote
Sensing 72 5 591–596.

 doi:10.14358/PERS.72.5.591

- Aiazzi et al. ((2007))

Aiazzi, B., Baronti, S. & Selva, M.

(2007).

Improving component
substitution pansharpening through multivariate regression of MS ++ pan
data.

IEEE Transactions on Geoscience and Remote
Sensing 45 10 3230–3239.

 doi:10.1109/TGRS.2007.901007

- Anokhin et al. ((2021))

Anokhin, I., Demochkin, K., Khakhulin, T., Sterkin, G., Lempitsky, V. & Korzhenkov, D.

(2021).

Image generators with conditionally-independent pixel
synthesis.

In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) ( 14278–14287).

 doi:10.1109/CVPR46437.2021.01405

- Baumgardner et al. ((2015))

Baumgardner, M.F., Biehl, L.L. & Landgrebe, D.A.

(2015).

220 band aviris hyperspectral image data set:
June 12, 1992 indian pine test site 3.

Purdue University Research Repository 10 7 991.

- Bioucas-Dias & Nascimento ((2008))

Bioucas-Dias, J.M. & Nascimento, J.M.

(2008).

Hyperspectral
subspace identification.

IEEE Transactions on Geoscience and Remote
Sensing 46 8 2435–2445.

- Chen et al. ((2021))

Chen, Y., Liu, S. & Wang, X.

(2021).

Learning continuous image representation with local
implicit image function.

In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) ( 8628–8638).

 doi:10.1109/CVPR46437.2021.00852

- Chen et al. ((2022))

Chen, Y., Zeng, J., He, W., Zhao, X-L. & Huang, T-Z.

(2022).

Hyperspectral and multispectral
image fusion using factor smoothed tensor ring decomposition.

IEEE Transactions on Geoscience and Remote
Sensing 60  1-17.

 doi:10.1109/TGRS.2021.3114197

- Debals et al. ((2017))

Debals, O., Van Barel, M. & De Lathauwer, L.

(2017).

Nonnegative matrix factorization using
nonnegative polynomial approximations.

IEEE Signal Processing Letters 24 7 948–952.

 doi:10.1109/LSP.2017.2697680

- Dell’Acqua et al. ((2004))

Dell’Acqua, F., Gamba, P., Ferrari, A., Palmason, J.A., Benediktsson, J.A. & Árnason, K.

(2004).

Exploiting spectral and
spatial information in hyperspectral urban data with high resolution.

IEEE Geoscience and Remote Sensing
Letters 1 4 322–326.

 doi:10.1109/LGRS.2004.837009

- Deng et al. ((2023))

Deng, S., Wu, R., Deng, L-J., Ran, R. & Jiang, T-X.

(2023).

Implicit neural feature fusion
function for multispectral and hyperspectral image fusion.

arXiv preprint arXiv:2307.07288   .

- Dian et al. ((2017))

Dian, R., Fang, L. & Li, S.

(2017).

Hyperspectral image super-resolution via
non-local sparse tensor factorization.

In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) ( 5344–5353).

 doi:10.1109/CVPR.2017.411

- Dian et al. ((2019))

Dian, R., Li, S., Fang, L., Lu, T. & Bioucas-Dias, J.M.

(2019).

Nonlocal sparse tensor
factorization for semiblind hyperspectral and multispectral image
fusion.

IEEE transactions on cybernetics 50 10 4469–4480.

 doi:10.1109/TCYB.2019.2951572

- Dian et al. ((2021))

Dian, R., Li, S. & Kang, X.

(2021).

Regularizing hyperspectral and multispectral image
fusion by CNN denoiser.

IEEE Transactions on Neural Networks and Learning
Systems 32 3 1124–1135.

 doi:10.1109/TNNLS.2020.2980398

- Dong et al. ((2016))

Dong, W., Fu, F., Shi, G., Cao, X., Wu, J., Li, G. & Li, X.

(2016).

Hyperspectral image super-resolution via
non-negative structured sparse representation.

IEEE Transactions on Image
Processing 25 5 2337–2352.

 doi:10.1109/TIP.2016.2542360

- Dupont et al. ((2021))

Dupont, E., Teh, Y.W. & Doucet, A.

(2021).

Generative models as distributions of functions.

In International Conference on Artificial Intelligence
and Statistics (AISTATS) ( 2989–3015).

- Gao et al. ((2014))

Gao, L., Li, J., Khodadadzadeh, M., Plaza, A., Zhang, B., He, Z. & Yan, H.

(2014).

Subspace-based support vector machines for
hyperspectral image classification.

IEEE Geoscience and Remote Sensing
Letters 12 2 349–353.

 doi:10.1109/LGRS.2014.2341044

- Guarino et al. ((2024))

Guarino, G., Ciotola, M., Vivone, G. & Scarpa, G.

(2024).

Band-wise hyperspectral image pansharpening using CNN
model propagation.

IEEE Transactions on Geoscience and Remote
Sensing 62  1-18.

 doi:10.1109/TGRS.2023.3339337

- Guo et al. ((2014))

Guo, Q., Zhang, B., Ran, Q., Gao, L., Li, J. & Plaza, A.

(2014).

Weighted-rxd and linear filter-based rxd: Improving background
statistics estimation for anomaly detection in hyperspectral imagery.

IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing 7 6 2351–2366.

 doi:10.1109/JSTARS.2014.2302446

- Han et al. ((2018))

Han, X-H., Shi, B. & Zheng, Y.

(2018).

SSF-CNN: Spatial and spectral fusion
with CNN for hyperspectral image super-resolution.

In 2018 25th IEEE International Conference on Image
Processing (ICIP) ( 2506–2510).

- He et al. ((2024))

He, L., Fang, Z., Li, J., Chanussot, J. & Plaza, A.

(2024).

Two spectral–spatial
implicit neural representations for arbitrary-resolution hyperspectral
pansharpening.

IEEE Transactions on Geoscience and Remote
Sensing 62  1-21.

 doi:10.1109/TGRS.2024.3380067

- He et al. ((2021))

He, L., Zhu, J., Li, J., Plaza, A., Chanussot, J. & Yu, Z.

(2021).

CNN-based hyperspectral pansharpening with arbitrary
resolution.

IEEE Transactions on Geoscience and Remote
Sensing 60  1–21.

 doi:10.1109/TGRS.2021.3132997

- Jia et al. ((2023))

Jia, S., Min, Z. & Fu, X.

(2023).

Multiscale spatial-spectral
transformer network for hyperspectral and multispectral image fusion.

Information Fusion 96  117–129.

 doi:10.1016/j.inffus.2023.03.011

- Khader et al. ((2023))

Khader, A., Yang, J. & Xiao, L.

(2023).

Model-guided deep unfolded fusion network with nonlocal spatial-spectral
priors for hyperspectral image super-resolution.

IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing 16  4607-4625.

 doi:10.1109/JSTARS.2023.3272370

- Laben & Brower ((2000))

Laben, C.A. & Brower, B.V.

(2000).

Process for enhancing the spatial resolution
of multispectral imagery using pansharpening.

: Google Patents.

US Patent 6,011,875

- Le Saux et al. ((2018))

Le Saux, B., Yokoya, N., Hansch, R. & Prasad, S.

(2018).

2018 IEEE GRSS data fusion
contest: Multimodal land use classification [technical committees].

IEEE Geoscience and Remote Sensing
Magazine 6 1 52–54.

 doi:10.1109/MGRS.2018.2798161

- N. Liu et al. ((2021))

Liu, N., Li, L., Li, W., Tao, R., Fowler, J.E. & Chanussot, J.

(2021).

Hyperspectral restoration and
fusion with multispectral imagery via low-rank tensor-approximation.

IEEE Transactions on Geoscience and Remote
Sensing 59 9 7817-7830.

 doi:10.1109/TGRS.2020.3049014

- X. Liu et al. ((2020))

Liu, X., Liu, Q. & Wang, Y.

(2020).

Remote sensing image fusion based on two-stream fusion
network.

Information Fusion 55  1–15.

 doi:10.1007/978-3-319-73603-7˙35

- Loncan et al. ((2015))

Loncan, L., De Almeida, L.B., Bioucas-Dias, J.M., Briottet, X., Chanussot, J., Dobigeon, N.others

(2015).

Hyperspectral
pansharpening: A review.

IEEE Geoscience and Remote Sensing
Magazine 3 3 27–46.

 doi:10.1109/MGRS.2015.2440094

- Luo et al. ((2024))

Luo, Y., Zhao, X., Li, Z., Ng, M.K. & Meng, D.

(2024).

Low-rank tensor function representation for
multi-dimensional data recovery.

IEEE Transactions on Pattern Analysis and Machine
Intelligence 46 5 3351-3369.

 doi:10.1109/TPAMI.2023.3341688

- Mildenhall et al. ((2021))

Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R. & Ng, R.

(2021).

NeRF: Representing scenes as neural radiance fields
for view synthesis.

Communications of the ACM 65 1 99–106.

 doi:10.1145/3503250

- Nguyen et al. ((2022))

Nguyen, H.V., Ulfarsson, M.O., Sveinsson, J.R. & Dalla Mura, M.

(2022).

Deep SURE for unsupervised remote sensing image fusion.

IEEE Transactions on Geoscience and Remote
Sensing 60  1–13.

 doi:10.1109/TGRS.2022.3215902

- Piziak & Odell ((1999))

Piziak, R. & Odell, P.L.

(1999).

Full rank
factorization of matrices.

Mathematics Magazine 72 3 193–201.

 doi:10.2307/2690882

- Qu et al. ((2017))

Qu, Y., Qi, H., Ayhan, B., Kwan, C. & Kidd, R.

(2017).

DOES multispectral/hyperspectral
pansharpening improve the performance of anomaly detection.

In 2017 IEEE International Geoscience and Remote Sensing
Symposium (IGARSS) ( 6130–6133).

- Ramasinghe & Lucey ((2022))

Ramasinghe, S. & Lucey, S.

(2022).

Beyond periodicity: Towards a unifying
framework for activations in coordinate-mlps.

In European Conference on Computer Vision (ECCV) ( 13693,  142–158).

 doi:10.1007/978-3-031-19827-4˙9

- Shah & Sitawarin ((2024))

Shah, K. & Sitawarin, C.

(2024).

SPDER: Semiperiodic damping-enabled object
representation.

In International Conference on Learning Representations (ICLR).

- Simoes et al. ((2014))

Simoes, M., Bioucas-Dias, J., Almeida, L.B. & Chanussot, J.

(2014).

A convex formulation for
hyperspectral image super-resolution via subspace-based
regularization.

IEEE Transactions on Geoscience and Remote
Sensing 53 6 3373–3388.

 doi:10.1109/TGRS.2014.2375320

- Sitzmann et al. ((2020))

Sitzmann, V., Martel, J., Bergman, A., Lindell, D. & Wetzstein, G.

(2020).

Implicit neural representations with periodic activation
functions.

In Advances in Neural Information Processing Systems (NeurIPS) ( 33,  7462–7473).

- Sun et al. ((2021))

Sun, L., Dong, W., Li, X., Wu, J., Li, L. & Shi, G.

(2021).

Deep maximum a posterior estimator for video denoising.

International Journal of Computer
Vision 129 10 2827–2845.

 doi:10.1007/S11263-021-01510-7

- Takikawa et al. ((2021))

Takikawa, T., Litalien, J., Yin, K., Kreis, K., Loop, C., Nowrouzezahrai, D.Fidler, S.

(2021).

Neural geometric level of detail: Real-time
rendering with implicit 3D shapes.

In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) ( 11358–11367).

 doi:10.1109/CVPR46437.2021.01120

- Vivone et al. ((2014))

Vivone, G., Alparone, L., Chanussot, J., Dalla Mura, M., Garzelli, A., Licciardi, G.A.Wald, L.

(2014).

A
critical comparison among pansharpening algorithms.

IEEE Transactions on Geoscience and Remote
Sensing 53 5 2565–2586.

 doi:10.1109/TGRS.2014.2361734

- Vivone et al. ((2023))

Vivone, G., Garzelli, A., Xu, Y., Liao, W. & Chanussot, J.

(2023).

Panchromatic and
hyperspectral image fusion: Outcome of the 2022 WHISPERS hyperspectral
pansharpening challenge.

IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing 16  166–179.

 doi:10.1109/JSTARS.2022.3220974

- K. Wang et al. ((2020))

Wang, K., Wang, Y., Zhao, X-L., Chan, J.C-W., Xu, Z. & Meng, D.

(2020).

Hyperspectral
and multispectral image fusion via nonlocal low-rank tensor decomposition and
spectral unmixing.

IEEE Transactions on Geoscience and Remote
Sensing 58 11 7654–7671.

 doi:10.1109/TGRS.2020.2983063

- T. Wang et al. ((2024))

Wang, T., Li, J., Ng, M.K. & Wang, C.

(2024).

Nonnegative matrix
functional factorization for hyperspectral unmixing with nonuniform spectral
sampling.

IEEE Transactions on Geoscience and Remote
Sensing 62  1-13.

 doi:10.1109/TGRS.2023.3347414

- W. Wang et al. ((2024))

Wang, W., Deng, L-J., Ran, R. & Vivone, G.

(2024).

A general paradigm with
detail-preserving conditional invertible network for image fusion.

International Journal of Computer
Vision 132 4 1029–1054.

 doi:10.1007/S11263-023-01924-5

- X. Wang, Cheng et al. ((2023))

Wang, X., Cheng, C., Liu, S., Song, R., Wang, X. & Feng, L.

(2023).

SS-INR: Spatial-spectral implicit neural representation network for
hyperspectral and multispectral image fusion.

IEEE Transactions on Geoscience and Remote
Sensing 61  1-14.

 doi:10.1109/TGRS.2023.3317413

- X. Wang, Wang et al. ((2023))

Wang, X., Wang, X., Song, R., Zhao, X. & Zhao, K.

(2023).

MCT-Net: Multi-hierarchical
cross transformer for hyperspectral and multispectral image fusion.

Knowledge-Based Systems 264  110362.

 doi:10.1016/j.knosys.2023.110362

- Z. Wang et al. ((2023))

Wang, Z., Ng, M.K., Michalski, J. & Zhuang, L.

(2023).

A self-supervised deep denoiser for
hyperspectral and multispectral image fusion.

IEEE Transactions on Geoscience and Remote
Sensing 61  1-14.

 doi:10.1109/TGRS.2023.3303921

- Wu et al. ((2024))

Wu, H., Wu, S., Zhang, K., Liu, X., Shi, S. & Bian, C.

(2024).

Unsupervised
blind spectral–spatial cross-super-resolution network for HSI and MSI
fusion.

IEEE Transactions on Geoscience and Remote
Sensing 62  1-14.

 doi:10.1109/TGRS.2024.3362862

- Xu & Jiao ((2023))

Xu, W. & Jiao, J.

(2023).

Revisiting implicit neural representations in low-level
vision.

In International Conference on Learning
Representations Workshop (ICLR Workshop).

- Yang et al. ((2018))

Yang, J., Zhao, Y-Q. & Chan, J.C-W.

(2018).

Hyperspectral and multispectral
image fusion via deep two-branches convolutional neural network.

Remote Sensing 10 5 800.

 doi:https://doi.org/10.3390/rs10050800

- Yang et al. ((2017))

Yang, J., Zhao, Y-Q., Yi, C. & Chan, J.C-W.

(2017).

No-reference hyperspectral image
quality assessment via quality-sensitive features learning.

Remote Sensing 9 4 305.

 doi:10.3390/rs9040305

- Yasuma et al. ((2010))

Yasuma, F., Mitsunaga, T., Iso, D. & Nayar, S.K.

(2010).

Generalized assorted pixel
camera: postcapture control of resolution, dynamic range, and
spectrum.

IEEE Transactions on Image
Processing 19 9 2241–2253.

 doi:10.1109/TIP.2010.2046811

- Yokota et al. ((2015))

Yokota, T., Zdunek, R., Cichocki, A. & Yamashita, Y.

(2015).

Smooth nonnegative matrix and tensor
factorizations for robust multi-way data analysis.

Signal Processing 113  234–249.

 doi:10.1016/j.sigpro.2015.02.003

- Yokoya et al. ((2012))

Yokoya, N., Yairi, T. & Iwasaki, A.

(2012).

Coupled nonnegative matrix
factorization unmixing for hyperspectral and multispectral data
fusion.

IEEE Transactions on Geoscience and Remote
Sensing 50 2 528–537.

 doi:10.1109/TGRS.2011.2161320

- J. Zhang et al. ((2024))

Zhang, J., Zhu, L., Deng, C. & Li, S.

(2024).

Hyperspectral and
multispectral image fusion via logarithmic low-rank tensor ring
decomposition.

IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing 17  11583-11597.

 doi:10.1109/JSTARS.2024.3416335

- K. Zhang et al. ((2022))

Zhang, K., Zhu, D., Min, X. & Zhai, G.

(2022).

Implicit neural representation
learning for hyperspectral image super-resolution.

IEEE Transactions on Geoscience and Remote
Sensing 61  1–12.

 doi:10.1109/TGRS.2022.3230204

- L. Zhang et al. ((2018))

Zhang, L., Wei, W., Bai, C., Gao, Y. & Zhang, Y.

(2018).

Exploiting clustering manifold
structure for hyperspectral imagery super-resolution.

IEEE Transactions on Image
Processing 27 12 5969–5982.

 doi:10.1109/TIP.2018.2862629

- X. Zhang et al. ((2020))

Zhang, X., Huang, W., Wang, Q. & Li, X.

(2020).

SSR-NET: Spatial–spectral
reconstruction network for hyperspectral and multispectral image
fusion.

IEEE Transactions on Geoscience and Remote
Sensing 59 7 5953–5965.

 doi:10.1109/TGRS.2020.3018732

- Zheng et al. ((2020))

Zheng, K., Gao, L., Liao, W., Hong, D., Zhang, B., Cui, X. & Chanussot, J.

(2020).

Coupled convolutional neural network with adaptive response function
learning for unsupervised hyperspectral super resolution.

IEEE Transactions on Geoscience and Remote
Sensing 59 3 2487–2502.

 doi:10.1109/TGRS.2020.3006534

- Zhuang & Bioucas-Dias ((2018))

Zhuang, L. & Bioucas-Dias, J.M.

(2018).

Fast hyperspectral image denoising
and inpainting based on low-rank and sparse representations.

IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing 11 3 730–742.

 doi:10.1109/JSTARS.2018.2796570

- Zhuang & Ng ((2021))

Zhuang, L. & Ng, M.K.

(2021).

FastHyMix: Fast and parameter-free hyperspectral
image mixed noise removal.

IEEE Transactions on Neural Networks and Learning
Systems 34 8 4702–4716.

 doi:10.1109/TNNLS.2021.3112577

<!-- END-OF-PAPER:f844f69e099d -->
