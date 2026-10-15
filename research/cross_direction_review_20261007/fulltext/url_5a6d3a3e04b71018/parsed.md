<!-- PDF PAGE 1 -->

Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution
by Image Fusion with the Panchromatic Image
Shuang Xu1 Zixiang Zhao2* Haowen Bai3 Chang Yu3
Jiangjun Peng1 Xiangyong Cao3∗ Deyu Meng3,4
1Northwestern Polytechnical University 2ETH Z¨urich 3Xi’an Jiaotong University
4Macau University of Science and Technology
xs@nwpu.edu.cn
Abstract
Hyperspectral images (HSIs) are frequently noisy and
of low resolution due to the constraints of imaging de-
vices. Recently launched satellites can concurrently ac-
quire HSIs and panchromatic (PAN) images, enabling the
restoration of HSIs to generate clean and high-resolution
imagery through fusing PAN images for denoising and
super-resolution. However , previous studies treated these
two tasks as independent processes, resulting in accumu-
lated errors. This paper introduces Hyperspectral Image
Joint Pandenoising and Pansharpening (Hipandas), a novel
learning paradigm that reconstructs high quality HSIs from
noisy low-resolution HSIs (NLRHS) and high-resolution
PAN images. The proposed unsupervised Hipandas frame-
work consists of a guided denoising network, a guided
super-resolution network, and a PAN reconstruction net-
work, utilizing an HSI low-rank prior and a newly intro-
duced detail-oriented low-rank prior . The interconnection
of these networks complicates the training process, neces-
sitating a two-stage training strategy to ensure effective
training. Experimental results on both simulated and real-
world datasets indicate that the proposed method surpasses
state-of-the-art algorithms, yielding more accurate and vi-
sually pleasing HRHS images. The code is available at
https://github.com/shuangxu96/Hipandas.
1. Introduction
Hyperspectral imaging has emerged as a powerful tool in
remote sensing [30]. Nevertheless, the acquisition of hyper-
spectral images (HSIs) is often impeded by practical limi-
tations. Factors such as atmospheric turbulence and limited
sensor apertures commonly result in noise corruption and
low spatial resolution for HSIs.
Denoising [53, 54] and super-resolution [5, 8, 34] have
been developed to enhance the quality. In contrast to
*Corresponding authors.
single-image restoration, the fusion of auxiliary images
from other modalities potentially leads to improvement
[45, 55]. The recent launch of satellites equipped with
hyperspectral-panchromatic (HS-PAN) imaging systems,
e.g. the PRISMA satellite, presents a new opportunity. PAN
images, distinguished by their high spatial resolution and
lack of noise, serve as an ideal data source for fusion [9, 12].
As illustrated in Fig. 1(a) and (b), guided denoising and
super-resolution can be achieved through pansharpening
and pandenoising techniques, respectively. However, the
images captured by the HS-PAN satellites ( e.g. PRISMA)
are typically noisy and low resolution. Currently, enhancing
image quality requires sequentially applying pandenoising
and pansharpening to noisy low-resolution HS (NLRHS)
images. This sequential strategy poses several challenges:
• Denoising and super-resolution are treated as independent
steps, requiring highly precise algorithms for both pan-
denoising and pansharpening. Inadequate denoising can
lead to noise amplification during super-resolution.
• Denoising may inadvertently smooth essential details,
thereby complicating texture recovery for the super-
resolution process.
• The inherent interdependence of denoising and super-
resolution implies that their disjoint execution may not
fully exploit their synergistic potential.
Additional motivation comes from the experimental
findings in [15], which indicate that the performance of net-
works trained on clean LRHS images is significantly de-
graded when applied to NLRHS images. This further con-
firms that solely relying on a model’s robustness to mitigate
the impact of noise in NLRHS images is challenging. How-
ever, despite the observation of this phenomenon, no re-
search has been conducted to carefully model noise specifi-
cally for HS pansharpening.
Despite the challenges, as shown in Fig. 1(c), the study
explores a novel technique named Hyperspectral Image
Joint Pandenoising and Pan sharpening (Hipandas) , de-
This ICCV paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
12002


<!-- PDF PAGE 2 -->

Hipandas (Hyperspectral Image Pandenoising and Pansharpening)
LRHS PAN Output
(a) Pansharpening (b) Pandenoising
NHS PAN Output
 NLRHS PAN Output
(c) Hipandas
LRHS images are low-resolution. LRHS images are low-resolution. NHS images are noisy. NHS images are noisy. NLRHS images are noisy and low-
resolution. 
NLRHS images are noisy and low-
resolution. 
Super-resolution
(Ignoring noise)
Super-resolution
(Ignoring noise)
Denoising
(Ignoring resolution)
Denoising
(Ignoring resolution) Joint Denoising and Super-resolutionJoint Denoising and Super-resolution
DataGoal Task
Conventional Tasks Proposed New Task
Figure 1. The differences among (a) pansharpening, (b) pandenoising and (c) Hipandas. Pansharpening addresses super-resolution, while
pandenoising focuses on denoising. In contrast, Hipandas is specifically designed to tackle both problems in a unified framework.
signed to reconstruct clean HRHS images from NLRHS
inputs, fused with high-resolution PAN (HRPAN) images.
Since real-world satellites tend to produce noisy and low-
resolution HSIs, the Hipandas task is more aligned with
practical scenarios than pandenoising or pansharpening
tasks alone. Due to the lack of large-scale datasets for
the Hipandas task, an unsupervised learning method is pro-
posed. This method comprises a guided denoising net-
work, a guided super-resolution network, and a PAN re-
construction network. The first two networks are utilized
for image enhancement, while the third ensures modality fi-
delity. These networks are interconnected, with the outputs
of the upstream networks serving as inputs to the down-
stream networks, thereby complicating the training pro-
cess. Consequently, a two-stage training strategy is intro-
duced. It is initially pretrained on low-resolution images
and then finetuned on high-resolution images to mitigate
learning bias towards the noise. Empirical results indicate
that the two-stage training strategy overwhelmingly outper-
forms the one-stage strategy. In summary, the contributions
are as follows:
• This paper proposes the Hipandas task, which is more
aligned with practical scenarios. An unsupervised
learning-based Hipandas (UHipandas) is formulated for
this task.
• A detail-oriented low-rank prior is introduced for the
Hipandas task and is applied to the network architecture
design of UHipandas.
• A two-stage training strategy is proposed to effectively
train UHipandas, which yields more accurate and visu-
ally appealing HRHS images compared to state-of-the-art
(SOTA) methods.
The organization of this paper is as follows. Section 2 re-
views related work. Section 3 delves into the detailed de-
scription of the proposed Hipandas method. Section 4 re-
ports experimental results. Section 5 concludes the paper.
2. Related Work
2.1. HSI super-resolution and pansharpening
Before deep learning, techniques such as sparse coding
[1, 44], wavelets [28], and Gaussian processes [2] were
employed for single HSI super-resolution. The success of
deep RGB image super-resolution [6] led to the emergence
of convolutional neural networks (CNNs) as a powerful
tool in the domain of single HSI super-resolution. This
includes 3D CNNs [19, 24], spectral information preserv-
ing networks [14, 21], spatial-spectral prior-based networks
[16, 38], and diffusion-based networks [25, 40, 42, 43].
Despite the achievements, the fusion of additional
modalities has the potential to yield better results [37]. PAN
images, characterized by the high spatial resolution and
signal-to-noise ratio, serve as effective guides [4, 23, 35].
Qu et al . developed a 3D CNN to fuse HS-PAN informa-
tion [32]. Dong et al. implemented adversarial learning for
spectrum preserved HS-PAN fusion [7]. Guarino et al. in-
troduced a rolling pansharpening CNN (RPNN) using band-
wise training to address challenges such as limited training
data and variations of band number [10]. Li et al. proposed
a dual conditional diffusion models-based fusion network
that operates in an iterative denoising manner [20]. Rui et
al. introduced a low-rank diffusion model for hyperspectral
pansharpening (PLRD), taking the strengths of a pretrained
diffusion model for enhanced generalization [33].
2.2. HSI denoising and pandenoising
For HSI denoising, a range of techniques have been de-
veloped, including noise modeling [47], total variation
[29, 39], nuclear norm [31, 36, 52] and subspace methods
[13]. In the era of deep learning, CNNs [3, 41, 50] and
transformers [17, 18, 26] have demonstrated promise. More
12003


<!-- PDF PAGE 3 -->

Table 1. Main notations in this paper.
Variable Denotation
P High-resolution PAN (HRPAN) image
Q Low-resolution PAN (LRPAN) image
H High-resolution HS (HRHS) image
L Low-resolution HS (LRHS) image
N Noisy LRHS (NLRHS) image
recently, diffusion models have emerged as effective tools
in HSI denoising. Zeng et al. utilized a transformer-based
backbone for spectral unmixing, with abundance maps de-
noised using a self-supervised generative diffusion network
[51]. Pang et al . developed a unified HSI restoration net-
work based on diffusion models (HIRD), which recon-
structs clean HSI data from the product of two low-rank
components, employing a pretrained diffusion model for
base image sampling [27].
Extensive research has been conducted on single HSI de-
noising, but there is a lack of work on pandenoising. To the
best of our knowledge, the study by Xuet al. [48] represents
the sole study on this topic. Based on the assumption that
PAN and HS images share similar textures, Xu et al . pro-
posed a PAN-weighted total variation (PWTV) regulariza-
tion [48]. Fusing PAN images enables PWTV to outperform
single HSI denoising methods.
2.3. Hipandas
Pansharpening aims to improve spatial resolution, while
pandenoising is devoted to remove noise. However, real-
world satellites tend to produce noisy and low-resolution
imagery, and sequential implementations of pansharpening
and pandenoising is at the risk of accumulated errors. To
the best of our knowledge, this paper is the early explo-
ration on end-to-end HSI denoising and super-resolution by
image fusion with PAN images.
3. Unsupervised Hipandas
This section proposes the Hipandas technique, and main no-
tations are summarized in Tab. 1.
3.1. Problem formulation for Hipandas
In HS imaging, the acquired data tends to be noisy and of
low resolution, which can be modeled by corrupting the
downsampled HRHS image with noise ε: N = H ↓ +ε,
where N ∈ Rh×w×b and H ∈ Rsh×sw×b are the noisy
LRHS and HRHS images, respectively, and ↓ is the down-
sampling operation. Here, s denotes the ratio of spatial
resolution, h, w, and b denote the height, width and band
number, respectively. For the HS-PAN imaging device, the
system also acquires HRPAN images P ∈ Rsh×sw.
PAN images, which offer complementary information,
make PAN-guided HSI restoration a promising technique
for achieving high-fidelity HSIs. This study introduces a
method termed Hyperspectral Image Joint Pandenoising
and Pansharpening (Hipandas), which aims to reconstruct
the noise-free HRHS image H from an NLRHS image N
fused with the HRPAN image P .
3.2. Proposed method
Due to the limited availability of large-scale datasets with
paired images {N, P, H}, this study proposes an unsu-
pervised learning framework for Hipandas (UHipandas) .
As depicted in Fig. 2, UHipandas consists of three com-
ponents: a Guided Denoising Network (GDN), a Guided
Super-Resolution Network (GSRN), and a Panchromatic
Reconstruction Network (PRN).
The GDN component is designed to eliminate noise from
the LRHS image N, with the guidance of the LRPAN image
to produce a denoised LRHS image ˆL. Given that the ob-
served PAN image P possesses a higher spatial resolution
than N, the GDN component employs the LRPAN image
Q = P ↓ to align with the spatial resolution of N. This
process is formalized as
ˆL = GDN (N, Q) . (1)
The GSRN component improves spatial resolution through
a detail injection approach. Initially, ˆL is upsampled ( e.g.,
via bicubic interpolation) to generate a coarse HRHS image.
Subsequently, ˆL ↑ and P restore the missing details, and the
final HRHS image ˆH is expressed as
ˆH = GSRN(ˆL, P) = ˆL ↑ +f(ˆL ↑, P), (2)
where f(ˆL ↑, P) represents the predicted detail map. In ad-
dition to the aforementioned guidance during the restoration
process, the strong relationship between PAN and HS im-
ages can provide valuable information. Consequently, the
PRN component is employed to ensure that PAN images
can be reconstructed from HSIs, formalized as
ˆP = PRN( ˆH), ˆQ = PRN(ˆL), (3)
where ˆP and ˆQ denote the reconstructed PAN images.
3.3. Loss functions and training details
The joint training of the three networks presents specific
challenges:
• The training data consists solely of the NLRHS image
N, HRPAN image P , and LRPAN image Q. Lacking
ground truth supervision, the formulation of rational loss
functions is crucial for the training of UHipandas.
• The output of the GDN component is utilized as the in-
put for the GSRN component. However, during the early
steps of training, the GDN component cannot generate a
satisfactorily denoised image, which would introduce a
training bias towards noise in the GSRN’s training.
To mitigate these problems, loss functions tailored to the
three networks are proposed, accompanied by a two-stage
training strategy:
12004


<!-- PDF PAGE 4 -->

LRPAN
NLRHS
DeLRHS
LRPAN
Downsampled
DeLRHS Reconstructed 
DeLRHS
Reconstructed 
LRPAN
LRPAN
NLRHS
DeLRHS
HRPAN
Output 
HRHS
Reconstructed 
LRPAN
 Reconstructed 
HRPAN
GDN
 GSRN
PRN
 Data Flow
Loss Computation
Notations
(a) Stage 1: Pretraining (b) Stage 2: Finetuning
Stage 1 Loss: Stage 2 Loss:
Figure 2. The framework for the proposed UHipandas. It is pretrained in stage 1 on LR scale images, and then finetuned in stage 2 on HR
scale images. The GDN and GSRN components are used for image enhancement, while the PRN component models relationship between
HS and PAN images.
Stage 1. In this stage, the networks are pretrained at the LR
scale. The reasons are twofold. LR scale images can reduce
computational burden. Additionally, neural networks’ spec-
tral bias property implies that training on LR images is less
complex than on HR ones [49].
a) For the GDN component , since ground truths are un-
available in the unsupervised learning, to achieve robust
denoising, an ℓ1 loss is utilized instead of the ℓ2 one be-
tween the NLRHS image N and the denoised LRHS image
ˆL = GDN (N, Q), as expressed by
LD = ∥N − ˆL∥1. (4)
This is due to the well-established assumption [29, 31, 52]
that the noise follows a heavy-tailed distribution, typically
modeled by Laplacian distributions, leading to an ℓ1 loss
according to the maximum likelihood principle.
b) For the PRN component, it is imperative that the recon-
structed LRPAN image maintains textures. By revising the
total variation regularization, this goal is achieved by the
PAN image’s high-frequency information loss,
LQ = ∥∇Q − ∇ ˆQ∥1, (5)
where ∇ represents the Sobel operator.
c) For the GSRN component, which involves super-
resolution, the network cannot be trained on the observed
dataset since there is no HR-LR image pair. Thus, the de-
noised LRHS image ˆL ∈ Rh×w×b is deemed as the HR
data, and its downsampled version ˆL ↓∈ Rh/s×w/s×b is re-
garded as the LR data. Then, the image could be recon-
structed by ˆL′ = GSRN(ˆL ↓, Q) ∈ Rh×w×b. In this way,
ˆL can be used as ground truth, and it compels the GSRN
component to learn the recovery of lost details by
L(1)
S = ∥ˆL − ˆL′∥2
F . (6)
In summary, the loss function in stage 1 is:
L(1) = LD + L(1)
S + LQ. (7)
Stage 2. In this stage, the networks are finetuned with
weights initialized from the pretraining phase. The loss
functions LD and LQ remain unchanged. However, the
objective of the GSRN component in this stage is to re-
construct an HRHS image, denoted as ˆH = GSRN(ˆL, P).
In the absence of a ground truth HRHS image to supervise
the learning of detail reconstruction, it has to ensure that
the downsampled HRHS image retains the same content as
the LRHS image. Consequently, the loss function for the
GSRN component is defined as
L(2)
S = ∥ ˆH ↓ − ˆL∥2
F . (8)
Furthermore, given that the HRHS image has been recon-
structed in this stage, a PAN image’s high-frequency infor-
mation loss is also introduced at the HR scale:
LP = ∥∇P − ∇ ˆP ∥1. (9)
In summary, the loss function in stage 2 is:
L(2) = LD + L(2)
S + LQ + LP . (10)
Remark: a) Although the loss functions contain many
terms, the multitask multimodal unsupervised learning
places a high reliance on loss design, unlike supervised
learning. LD and LS are used for data fidelity of denois-
ing and SR respectively, and LQ and LP are used for tex-
ture preservation from PAN images. b) Ablation studies in
supplementary materials show that adjusting the weights of
loss terms does not lead to better performances, so they are
assigned with the same importance in Eq. (7) & Eq. (10).
12005


<!-- PDF PAGE 5 -->

/uni00000013/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013
/uni00000025/uni00000044/uni00000051/uni00000047/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b
/uni00000014/uni00000011/uni00000013/uni00000028/uni00000051/uni00000048/uni00000055/uni0000004a/uni0000005c
/uni00000026/uni0000004f/uni00000048/uni00000044/uni00000051
= /uni00000014
= /uni00000015
= /uni00000016
= /uni00000017
= /uni00000018
(a) The Gaussian noise.
/uni00000013/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013
/uni00000025/uni00000044/uni00000051/uni00000047/uni00000003/uni0000002c/uni00000051/uni00000047/uni00000048/uni0000005b
/uni00000013/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000015
/uni00000013/uni00000011/uni00000017
/uni00000013/uni00000011/uni00000019
/uni00000013/uni00000011/uni0000001b
/uni00000014/uni00000011/uni00000013/uni00000028/uni00000051/uni00000048/uni00000055/uni0000004a/uni0000005c
/uni00000026/uni0000004f/uni00000048/uni00000044/uni00000051
p = /uni00000013/uni00000011/uni00000014
p = /uni00000013/uni00000011/uni00000015
p = /uni00000013/uni00000011/uni00000016
p = /uni00000013/uni00000011/uni00000017
p = /uni00000013/uni00000011/uni00000018 (b) The mixture noise.
Figure 3. The energy curve of clean and corrupted detail maps
(i.e., D and ˜D). σ and p denote the noise intensity for Gaussian
and mixture noise, respectively.
3.4. Network structure
It is important to incorporate data prior into network struc-
tures. The low-rank nature of HSIs is well-established and
widely applied in HSI denoising. However, the prior for
HSI pansharpening is less explored. As Eq. (2) is based on
the detail injection framework, the focus of this paper is on
modeling the prior for the detail obtained by f(·, ·).
Low-rank Prior for detail maps. The detail represents the
information lost from an LRHS image during its reconstruc-
tion to an HRHS image. Given an HRHS image H, when
downsampled to L = H ↓ and subsequently upsampled to
L ↑, the clean detail map is derived as D = H − L ↑. Ac-
counting for noise, the practical detail is the noisy version,
˜D = H − N ↑, where N = H ↓ +ε.
One of the main contributions of the work is the identi-
fication of the detail map D as being inherently low-rank.
This insight is based on an analysis of the energy curves for
both clean and noisy detail maps. The k-th energy value
is computed as Pk
i=1 λi/ Pb
i=1 λi, where λi represents the
singular value of the detail map. A steeper energy curve
signifies a higher degree of low-rankness in the correspond-
ing detail map. Through simulations of noisy detail maps
corrupted by Gaussian or mixed noise at varying intensi-
ties, as depicted in Fig. 3, it is observed that the clean detail
map exhibits a stronger low-rank characteristic, whereas the
noisy detail map tends to disrupt this property. Thus, in-
corporating the low-rank prior of the detail map into f(·, ·)
facilitates the recovery of more accurate detail maps.
Network design. For GDN and GSRN components, their
architectures are designed as a guided low-rank matrix
factorization network, leveraging low-rank priors for both
HSIs and detail maps. As depicted in Fig. 4(a), the net-
work output is represented as the product of two factor ma-
trices. Taking the GDN component as an instance, the de-
noised HSI is expressed as ˆL = V ×3 U, where ×3 denotes
the mode-3 product, V ∈ Rh×w×r constitutes r base im-
ages, and U ∈ Rb×r represents the expression coefficients.
Specifically, the i-th band of ˆL is a linear combination of r
base images, given by ˆL:,:,i = Pr
k=1 Ui,kV:,:,k. By setting
(c) HS&PAN Fusion (HPF) Layer
Input
HSI
Feature
Input
PAN
Feature
C
C
ConvConv
Output
PAN
Feature
Output
HSI
Feature
HPF Layer x5 SConv x2
AAP
Conv Conv
Conv
SoftmaxSigmoid
Output
HSI
PAN
(a) GDN/GSRN
(b) PRN
Conv x5 
HSI PAN
Figure 4. The architecture for the (a) GDN/GSRN component,
(b) PRN component and (c) HPF layer. Conv and SConv denote
convolutional and strided convolutional units, respectively. AAP
denotes the adaptive average pooling.
r ≪ b, the model enforces a low-rank structure on network
outputs.
Conventional matrix factorization models typically ad-
dress the optimization problem, minU,V {L(Y, V ×3 U) +
R1(V ) + R2(U)}, where Y denotes the noisy observa-
tion, and L(·, ·) and Ri(·) denote the loss and regularization
functions, respectively. The estimations of U and V are
computed via iterative algorithms and expressed as func-
tions of the observed data. However, this approach is heav-
ily reliant on manually designed objective functions.
In contrast, this work parametrizes V and U as functions
of the observation data via neural networks, with the for-
ward propagation of the network approximating the com-
putational flow of iterative algorithms in a data-driven man-
ner, thereby potentially obviating the need for hand-crafted
objective functions.
The proposed architecture comprises two distinct
branches. The U-branch models spectral information. It ap-
plies two strided convolution (SConv) units to the HSI, each
reducing the spatial resolution by a factor of two, resulting
in a feature of size h/4 × w/4 × b. An adaptive average
pooling operation is applied along the spatial dimensions,
followed by a softmax activation, to yield the transposed
coefficient matrix U T ∈ Rr×b. The V -branch is designed
to generate base images. Since PAN and base images share
similar structures, PAN and HS images are fused in this
branch. It begins with two head convolution (Conv) units
that embed both HS and PAN images into shallow features
with identical channels. Five HS and PAN fusion (HPF)
layers are then utilized, with a tail Conv unit transforming
the deep HS feature into V with r channels, activated by a
sigmoid function. To mitigate computational intensity, the
12006


<!-- PDF PAGE 6 -->

Table 2. Averaged metrics on simulated data between ground truthH and restored image ˆH for the Hipandas task in Gaussian noise cases,
where σ denotes the intensity of Gaussian noise. Best and second-best values are highlighted and underlined.
Methods
i.i.d. Gaussian (σ = 10) i.i.d. Gaussian ( σ = 30) non-i.i.d. Gaussian ( σ ∈[10,50])
PSNR↑ SSIM↑ ERGAS↓ SAM↓ PSNR↑ SSIM↑ ERGAS↓ SAM↓ PSNR↑ SSIM↑ ERGAS↓ SAM↓
PLRD+HIRD 37.93 0.9229 11.97 4.61 28.40 0.6866 31.04 10.78 27.89 0.7499 30.92 13.37
PLRD+PWTV 34.53 0.7881 17.11 7.23 27.42 0.5212 37.73 17.46 28.84 0.6198 35.44 19.81
SwinIR+HIRD 29.89 0.8569 52.50 7.90 29.80 0.8497 31.98 8.62 28.10 0.8267 43.11 16.08
SwinIR+PWTV 27.93 0.6855 212.50 13.16 18.63 0.1659 984.41 44.46 19.60 0.2446 966.29 44.54
RPNN+HIRD 36.60 0.9280 12.43 4.53 32.11 0.8711 19.98 7.28 32.43 0.8765 19.68 8.06
RPNN+PWTV 37.19 0.9342 12.27 4.31 29.73 0.6570 27.38 12.85 31.37 0.7440 26.13 13.25
HIRD+PLRD 34.36 0.8308 450.92 8.34 27.60 0.6303 569.13 19.14 26.99 0.5976 557.02 31.35
HIRD+SwinIR 36.24 0.8904 13.15 3.61 34.09 0.8700 16.65 5.40 32.20 0.8603 20.17 8.12
HIRD+RPNN 37.40 0.9314 11.95 4.10 35.06 0.9194 15.61 5.74 33.03 0.9072 18.97 8.22
PWTV+PLRD 36.56 0.9059 13.65 4.88 34.09 0.8865 18.21 7.20 32.48 0.8777 21.49 10.07
PWTV+SwinIR 37.65 0.9080 11.18 3.09 34.15 0.8393 16.53 5.38 34.26 0.8370 17.71 6.33
PWTV+RPNN 38.34 0.9415 10.88 3.87 36.70 0.9314 13.20 4.76 36.70 0.9299 13.46 4.97
UHipandas (ours)40.61 0.9547 8.50 3.00 36.88 0.9271 12.35 4.63 37.24 0.9313 11.87 4.52
Figure 5. Restoration results for Gaussian noise with σ = 30.
HPF layer is designed with a simple structure, comprising
two separate Conv units applied to the concatenated HS and
PAN features to produce the output features. Throughout
this paper, all Conv or SConv units are activated by leaky
rectified linear units, unless otherwise specified.
As for the PRN component, it is commonly postulated
that there exists a spectral transform matrix Φ ∈ R1×B,
and spectral degradation is expressed as P = H ×3 Φ [11,
46]. In other words, the PAN image is the weighted sum of
hyperspectral bands, akin to applying a 1 × 1 Conv unit to
H. However, this does not hold true for numerous satellite
systems. The PRN component thus is designed as 5 stacked
Conv units, and this can be regarded as a general nonlinear
version of P = H ×3 Φ. Ablation results in Sec. 4.3 show
that PRN gets better performance than the vanilla one.
4. Experiments
4.1. Results on simulated datasets
Datasets. Acquiring paired NLRHS, HRHS and HRPAN
images is challenging, so NLRHS and HRPAN images are
simulated from HRHS images. A 32-band clean HSI cap-
tured over Dongying, Shandong, China, by the Zhuhai-1
satellite, is selected. This image is divided into 81 patches
of 256 × 256 pixels, which are regarded as HRHS images
H. The corresponding NLRHS images N are obtained by
N = H ↓ +ε, leading to images with a size of64 ×64 ×32.
The HRPAN image P is synthesized by applying the spec-
tral response function of the IKONOS satellite to H.
Noise configuration. The first group comprises three
scenarios: independent and identically distributed (i.i.d.)
Gaussian noise with σ = 10 and 30, and non-i.i.d. Gaussian
noise with σ ∈ [10, 50]. The second group involves mixed
noise, where non-i.i.d. Gaussian noise affects 2/3 bands,
and impulse, stripe and deadline noise affect 1/3 bands with
intensities of p = 0.15, 0.35, and 0.55.
Evaluation metrics. They include peak signal-to-noise ra-
tio (PSNR), structural similarity (SSIM), Erreur Relative
Globale Adimensionnelle de Synthese (ERGAS), and spec-
tral angle mapper (SAM). Higher PSNR and SSIM, along
with lower ERGAS and SAM, signify better results.
Implementation details. Experiments were conducted on
a computer with a RTX 3090 GPU. The training comprised
12007


<!-- PDF PAGE 7 -->

Table 3. Averaged metrics on simulated data between ground truth H and restored image ˆH for the Hipandas task in mixture noise cases,
where p denotes the intensity of mixture noise. Best and second-best values are highlighted and underlined.
Methods
Mixture (p = 0.15) Mixture ( p = 0.35) Mixture ( p = 0.55)
PSNR↑ SSIM↑ ERGAS↓ SAM↓ PSNR↑ SSIM↑ ERGAS↓ SAM↓ PSNR↑ SSIM↑ ERGAS↓ SAM↓
PLRD+HIRD 29.38 0.8394 26.88 12.84 28.81 0.8275 28.55 13.04 28.20 0.7876 34.12 13.80
PLRD+PWTV 30.54 0.7028 32.25 18.27 29.70 0.6709 34.25 18.85 28.45 0.6183 85.80 21.01
SwinIR+HIRD 29.73 0.8410 37.83 11.66 29.38 0.8335 42.06 12.44 29.11 0.8255 51.47 13.60
SwinIR+PWTV 25.05 0.4912 766.41 37.41 23.80 0.4505 821.29 38.79 22.15 0.3873 1415.73 39.44
RPNN+HIRD 33.87 0.8931 17.47 6.90 32.75 0.8798 21.79 8.40 31.94 0.8656 31.09 9.86
RPNN+PWTV 34.91 0.8537 19.38 9.50 32.64 0.8045 24.91 11.98 30.71 0.7371 76.43 14.97
HIRD+PLRD 28.27 0.6445 605.39 34.60 29.18 0.6677 543.26 28.21 24.99 0.5323 561.31 41.36
HIRD+SwinIR 32.28 0.8628 20.69 8.77 31.91 0.8582 24.04 9.62 31.61 0.8500 36.18 10.81
HIRD+RPNN 33.11 0.9071 19.41 8.84 32.72 0.9037 22.84 9.68 32.44 0.8970 35.26 10.88
PWTV+PLRD 32.42 0.8689 24.60 11.72 31.86 0.8595 27.06 13.10 31.37 0.8512 38.51 13.67
PWTV+SwinIR 36.64 0.8855 14.52 5.25 35.77 0.8749 16.83 6.01 33.42 0.8255 335.51 11.53
PWTV+RPNN 37.79 0.9372 12.11 4.51 37.20 0.9336 13.71 5.18 35.41 0.9035 194.38 10.30
UHipandas (ours)40.24 0.9541 9.07 3.23 38.66 0.9448 10.79 3.91 36.61 0.9227 45.26 8.68
Figure 6. Restoration results for mixture noise with p = 0.55.
two stages, each consisting of 400 and 600 epochs, with
the Adam optimizer of a learning rate 10−3. The channel
number in networks was fixed at 128. The ranks r for GDN
and GSRN components were set to 3 and 12.
Compared methods. UHipandas is compared with SOTA
methods including HIRD [27], PWTV [48], SwinIR [22],
PLRD [33], and RPNN [10]. The first two are denois-
ing or pandenoising algorithms, and the others are super-
resolution or pansharpening algorithms. They are executed
in sequence, and two kinds of execution order are con-
sidered: super-resolution+denoising, and denoising+super-
resolution.
Quantitative Analysis. Tab. 2 and Tab. 3 present the evalu-
ation metrics for the Gaussian and mixture noise scenarios,
respectively. Compared with PWTV+RPNN (the second
best method), UHipandas could achieve improvement by
2.3 dB for the case of i.i.d. Gaussian noise withσ = 10, and
by 1.3 dB for the case of mixture noise with p = 0.55. The
data illustrate that our proposed UHipandas demonstrates a
marked enhancement in performance.
Visual Assessment. Fig. 5 provides a visual comparison
of the restoration results for images corrupted by Gaus-
sian noise at an intensity of σ = 30 . It is evident that
the images restored by HIRD+SwinIR, PWTV+PLRD, and
PWTV+SwinIR still exhibit noticeable noise, and other
methods suffer from considerable spectral distortion. Fig. 6
depicts the restoration results for images affected by mix-
ture noise at an intensity of p = 0 .55. It reveals that all
the compared methods fail to effectively restore the images,
with severe noise remaining and significant details being
lost. However, the proposed UHipandas consistently ex-
hibits good performance across different cases. More visual
results are reported in supplementary materials.
4.2. Results on the real-world dataset
Real-world datasets are publicly unavailable. Here we
construct a dataset containing five samples captured by
the PRISMA satellite. In this case, RPNN+PWTV ,
HIRD+RPNN and PWTV+RPNN are carried out, while
others are unavailable due to the CUDA out-of-memory
problem. However, these three methods are best perform-
ers in simulated experiments, and thus their results are very
competitive. The no-reference metric, Natural Image Qual-
12008


<!-- PDF PAGE 8 -->

Figure 7. Results on a real-world image sample. NIQE is displayed in the left-bottom corner, and lower NIQE indicates better results.
Table 4. NIQE metric values on real-world datasets.
Methods RPNN+PWTV HIRD+RPNN PWTV+RPNN UHipandas
NIQE↓ 10.40 7.19 6.80 6.05
ity Evaluator (NIQE), averaged over five samples is re-
ported in Tab. 4. Visual results of a sample are displayed
in Fig. 7. UHipandas produces a good result with clear tex-
tures and satisfactory color. More visual results are reported
in supplementary materials.
4.3. Ablation experiments
Ablation experiments are conducted to validate the rational-
ity of the proposed method. Results are listed in Tab. 5 for
a noise case, i.i.d. Gaussian noise with σ = 10.
Three components. To comprehensively evaluate the ef-
fectiveness of each component in UHipandas, Exp. I-III re-
spectively remove the GDN, GSRN and PRN components
by deleting the corresponding loss functions. The results
indicate that the removal of any component significantly
degrades performance, which shows the indispensability of
the three components.
Two-stage training. Exp. IV is the scenario without pre-
training to validate the rationality of two-stage training. The
results show that direct training cannot yield satisfactory
performance, suggesting that pretraining is a critical step
in preventing the training bias towards noise.
Low-rank modeling for GDN/GSRN components. The
architecture of GDN/GSRN components is based on low-
rank priors. Exp. V studies the architecture without low-
rank prior by revising it as a plain CNN for the fusion of
HS and PAN images. The results show that performance is
degraded by 3.6 dB without this prior.
Fusion of PAN images. Exp. VI investigates the scenario
without PAN images’ fusion. It is evident that this case
results in substantial degradation, suggesting that the pro-
posed model effectively fuses PAN images.
PRN components. Exp. VII replaces the PRN components
as P = H ×3 Φ, where Φ is regarded as the training param-
eters. It is shown that the metrics were slightly reduced, in-
dicating that the PRN components could better capture rela-
tionship between HS&PAN images than the linear spectral
transform P = H ×3 Φ.
4.4. Interaction of pandenoising and pansharpening
This subsection evaluates the impact of pansharpening on
the performance of pandenoising. To address this, we in-
Table 5. Results of six ablation experiments. Bold indicates the
best value.
Configurations PSNR SSIM ERGAS SAM
I w/o LD 19.51 0.5823 57.25 19.51
II w/o L(1)S andL(2)S 31.49 0.9210 20.37 4.41III w/o LPandLQ 33.32 0.7529 17.44 3.57IV w/o pretraining 35.94 0.9284 31.92 4.19V w/o low-rank modeling for GDN/GSRN 36.99 0.9325 13.35 3.83VI w/o fusion of PAN 33.76 0.7706 16.91 3.54
VII PRN →P=H×3Φ 40.47 0.9532 8.70 3.06
Ours 40.61 0.9547 8.50 3.00
Table 6. Evaluation metrics between ground truth L and denoised
image ˆL for the denoising task in the i.i.d. Gaussian noise case
with σ = 10.
PSNR SSIM ERGAS SAM
UHipandas 41.68 0.9610 7.05 2.32
Only denoising 40.67 0.9522 8.60 2.93
dependently train a GDN component with supervision from
LD. Subsequently, we compare its denoising capability for
LRHS images with UHipandas. Tab. 6 illustrates that em-
ploying only the GDN component yields a PSNR value of
40.67. However, when denoising and super-resolution are
integrated, as in UHipandas, there is a 1 dB enhancement in
performance, illustrating the synergy between pandenoising
and pansharpening. More analysis can be found in supple-
mentary materials.
5. Conclusion
This paper explores the Hipandas problem for HSI joint
restoration by image fusion, which is more practical than
the sequential implementation of pansharpening and pande-
noising. The proposed UHipandas is designed by fully ex-
ploiting the HSI and detail-oriented low-rank priors. With
elaborate loss functions and the two-stage training strategy,
UHipandas outperforms SOTA methods.
Acknowledgment
This work has been supported by the National Natu-
ral Science Foundation of China under Grant 12201497,
62272375, and 12401674, partly supported by Young Tal-
ent Fund of Xi’an Association for Science and Technology
under Grant 0959202513207, partly supported by Tianyuan
Fund for Mathematics of the National Natural Science
Foundation of China under Grant 12426105, and partly sup-
ported by Fundamental Research Funds for the Central Uni-
versities under Grant D5000240095.
12009


<!-- PDF PAGE 9 -->

References
[1] Naveed Akhtar, Faisal Shafait, and Ajmal S. Mian. Bayesian
sparse representation for hyperspectral image super resolu-
tion. In CVPR, pages 3631–3640, 2015. 2
[2] Naveed Akhtar, Faisal Shafait, and Ajmal S. Mian. Hier-
archical beta process with gaussian process prior for hyper-
spectral image super resolution. In ECCV, pages 103–120,
2016. 2
[3] Xiangyong Cao, Xueyang Fu, Chen Xu, and Deyu Meng.
Deep spatial-spectral global reasoning network for hyper-
spectral image denoising. IEEE TGRS, 60:1–14, 2022. 2
[4] Renwei Dian, Anjing Guo, and Shutao Li. Zero-shot hy-
perspectral sharpening. IEEE TPAMI, 45(10):12650–12666,
2023. 2
[5] Renwei Dian, Tianci Shan, Wei He, and Haibo Liu. Spec-
tral super-resolution via model-guided cross-fusion network.
IEEE TNNLS, 35(7):10059–10070, 2024. 1
[6] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou
Tang. Learning a deep convolutional network for image
super-resolution. In ECCV, pages 184–199, 2014. 2
[7] Wenqian Dong, Yufei Yang, Jiahui Qu, Weiying Xie, and
Yunsong Li. Fusion of hyperspectral and panchromatic im-
ages using generative adversarial network and image seg-
mentation. IEEE TGRS, 60:5508413, 2022. 2
[8] Jian Fang, Jingxiang Yang, Abdolraheem Khader, and Liang
Xiao. MIMO-SST: multi-input multi-output spatial-spectral
transformer for hyperspectral and multispectral image fu-
sion. IEEE TGRS, 62:5510020, 2024. 1
[9] Pedram Ghamisi, Behnood Rasti, Naoto Yokoya, Qunming
Wang, Bernhard Hofle, Lorenzo Bruzzone, Francesca Bo-
volo, Mingmin Chi, Katharina Anders, Richard Gloaguen,
Peter M. Atkinson, and Jon Atli Benediktsson. Multisource
and multitemporal data fusion in remote sensing: A compre-
hensive review of the state of the art. IEEE Geoscience and
Remote Sensing Magazine, 7(1):6–39, 2019. 1
[10] Giuseppe Guarino, Matteo Ciotola, Gemine Vivone, and
Giuseppe Scarpa. Band-wise hyperspectral image pansharp-
ening using CNN model propagation. IEEE TGRS , 62:
5500518, 2024. 2, 7
[11] Jiang He, Jie Li, Qiangqiang Yuan, Huanfeng Shen, and
Liangpei Zhang. Spectral response function-guided deep
optimization-driven network for spectral super-resolution.
IEEE TNNLS, 33(9):4213–4227, 2022. 6
[12] Lin He, Zhou Fang, Jun Li, Jocelyn Chanussot, and Antonio
Plaza. Two spectral-spatial implicit neural representations
for arbitrary-resolution hyperspectral pansharpening. IEEE
TGRS, 62:5513621, 2024. 1
[13] Wei He, Quanming Yao, Chao Li, Naoto Yokoya, Qibin
Zhao, Hongyan Zhang, and Liangpei Zhang. Non-local
meets global: An iterative paradigm for hyperspectral image
restoration. IEEE TPAMI, 44(4):2089–2107, 2022. 2
[14] Jing Hu, Xiuping Jia, Yunsong Li, Gang He, and Minghua
Zhao. Hyperspectral image super-resolution via intrafusion
network. IEEE TGRS, 58(10):7459–7471, 2020. 2
[15] Tao Huang, Weisheng Dong, Jinjian Wu, Leida Li, Xin
Li, and Guangming Shi. Deep hyperspectral image fusion
network with iterative spatio-spectral regularization. IEEE
Trans. Computational Imaging, 8:201–214, 2022. 1
[16] Junjun Jiang, He Sun, Xianming Liu, and Jiayi Ma. Learn-
ing spatial-spectral prior for super-resolution of hyperspec-
tral imagery. IEEE TCI, 6:1082–1096, 2020. 2
[17] Miaoyu Li, Ying Fu, and Yulun Zhang. Spatial-spectral
transformer for hyperspectral image denoising. In AAAI,
pages 1368–1376, 2023. 2
[18] Miaoyu Li, Ji Liu, Ying Fu, Yulun Zhang, and Dejing Dou.
Spectral enhanced rectangle transformer for hyperspectral
image denoising. In CVPR, pages 5805–5814, 2023. 2
[19] Qiang Li, Qi Wang, and Xuelong Li. Exploring the rela-
tionship between 2d/3d convolution for hyperspectral image
super-resolution. IEEE TGRS, 59(10):8693–8703, 2021. 2
[20] Shuangliang Li, Siwei Li, and Lihao Zhang. Hyperspectral
and panchromatic images fusion based on the dual condi-
tional diffusion models. IEEE TGRS, 61:5526315, 2023. 2
[21] Yunsong Li, Jing Hu, Xi Zhao, Weiying Xie, and Jiaojiao
Li. Hyperspectral image super-resolution using deep convo-
lutional neural network. Neurocomputing, 266:29–41, 2017.
2
[22] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang,
Luc Van Gool, and Radu Timofte. Swinir: Image restora-
tion using swin transformer. In ICCVW, pages 1833–1844,
2021. 7
[23] Xiaochen Lu, Junping Zhang, Dezheng Yang, Longting Xu,
and Fengde Jia. Cascaded convolutional neural network-
based hyperspectral image resolution enhancement via an
auxiliary panchromatic image. IEEE TIP , 30:6815–6828,
2021. 2
[24] Shaohui Mei, Xin Yuan, Jingyu Ji, Yifan Zhang, Shuai Wan,
and Qian Du. Hyperspectral image spatial super-resolution
via 3d full convolutional neural network. Remote. Sens., 9
(11):1139, 2017. 2
[25] Yuchun Miao, Lefei Zhang, Liangpei Zhang, and Dacheng
Tao. DDS2M: self-supervised denoising diffusion spatio-
spectral model for hyperspectral image restoration. In ICCV,
pages 12052–12062, 2023. 2
[26] Li Pang, Weizhen Gu, and Xiangyong Cao. Trq3dnet: A
3d quasi-recurrent and transformer based network for hyper-
spectral image denoising. Remote. Sens., 14(18):4598, 2022.
2
[27] Li Pang, Xiangyu Rui, Long Cui, Hongzhong Wang, Deyu
Meng, and Xiangyong Cao. Hir-diff: Unsupervised hyper-
spectral image restoration via improved diffusion models. In
CVPR, pages 3005–3014, 2024. 3, 7
[28] Rakesh C. Patel and Manjunath V . Joshi. Super-resolution of
hyperspectral images: Use of optimum wavelet filter coeffi-
cients and sparsity regularization. IEEE TGRS, 53(4):1728–
1736, 2015. 2
[29] Jiangjun Peng, Qi Xie, Qian Zhao, Yao Wang, Yee Leung,
and Deyu Meng. Enhanced 3dtv regularization and its ap-
plications on HSI denoising and compressed sensing. IEEE
TIP, 29:7889–7903, 2020. 2, 4
[30] Jiangtao Peng, Weiwei Sun, Heng-Chao Li, Wei Li, Xi-
angchao Meng, Chiru Ge, and Qian Du. Low-rank and sparse
representation for hyperspectral image processing: A review.
12010


<!-- PDF PAGE 10 -->

IEEE Geoscience and Remote Sensing Magazine , 10(1):10–
43, 2022. 1
[31] Jiangjun Peng, Yao Wang, Hong-Ying Zhang, Jianjun Wang,
and Deyu Meng. Exact decomposition of joint low rankness
and local smoothness plus sparse matrices. IEEE TPAMI, 45
(5):5766–5781, 2023. 2, 4
[32] Jiahui Qu, Yanzi Shi, Weiying Xie, Yunsong Li, Xianyun
Wu, and Qian Du. MSSL: hyperspectral and panchro-
matic images fusion via multiresolution spatial-spectral fea-
ture learning networks. IEEE TGRS, 60:5504113, 2022. 2
[33] Xiangyu Rui, Xiangyong Cao, Li Pang, Zeyu Zhu, Zong-
sheng Yue, and Deyu Meng. Unsupervised hyperspectral
pansharpening via low-rank diffusion model. Inf. Fusion ,
107:102325, 2024. 2, 7
[34] Yanli Shang, Jianjun Liu, Jingyi Zhang, and Zebin Wu.
MFT-GAN: A multiscale feature-guided transformer net-
work for unsupervised hyperspectral pansharpening. IEEE
TRGS, 62:5518516, 2024. 1
[35] Xin Tian, Wei Zhang, Yuerong Chen, Zhongyuan Wang, and
Jiayi Ma. Hyperfusion: A computational approach for hy-
perspectral, multispectral, and panchromatic image fusion.
IEEE TGRS, 60:1–16, 2022. 2
[36] Hailin Wang, Jiangjun Peng, Wenjin Qin, Jianjun Wang, and
Deyu Meng. Guaranteed tensor recovery fused low-rankness
and smoothness. IEEE TPAMI, 45(9):10990–11007, 2023. 2
[37] Wu Wang, Liang-Jian Deng, Ran Ran, and Gemine Vivone.
A general paradigm with detail-preserving conditional in-
vertible network for image fusion. IJCV, 132(4):1029–1054,
2024. 2
[38] Xinya Wang, Jiayi Ma, and Junjun Jiang. Hyperspectral im-
age super-resolution via recurrent feedback embedding and
spatial-spectral consistency regularization. IEEE TGRS, 60:
1–13, 2022. 2
[39] Yao Wang, Jiangjun Peng, Qian Zhao, Yee Leung, Xi-Le
Zhao, and Deyu Meng. Hyperspectral image restoration via
total variation regularized low-rank tensor decomposition.
IEEE JSTARS, 11(4):1227–1243, 2018. 2
[40] Zhaoyang Wang, Dongyang Li, Mingyang Zhang, Hao Luo,
and Maoguo Gong. Enhancing hyperspectral images via dif-
fusion model and group-autoencoder super-resolution net-
work. In AAAI, pages 5794–5804, 2024. 2
[41] Kaixuan Wei, Ying Fu, and Hua Huang. 3-d quasi-recurrent
neural network for hyperspectral image denoising. IEEE
TNNLS, 32(1):363–375, 2021. 2
[42] Chanyue Wu, Dong Wang, Yunpeng Bai, Hanyu Mao, Ying
Li, and Qiang Shen. Hsr-diff: Hyperspectral image super-
resolution via conditional diffusion models. In ICCV, pages
7060–7070, 2023. 2
[43] Yi Xiao, Qiangqiang Yuan, Kui Jiang, Jiang He, Xianyu Jin,
and Liangpei Zhang. Ediffsr: An efficient diffusion prob-
abilistic model for remote sensing image super-resolution.
IEEE TGRS, 62:5601514, 2024. 2
[44] Fengchao Xiong, Jiantao Zhou, Jun Zhou, Jianfeng Lu, and
Yuntao Qian. Multitask sparse representation model-inspired
network for hyperspectral image denoising. IEEE TGRS, 61:
5518515, 2023. 2
[45] Shuang Xu, Ouafa Amira, Junmin Liu, Chun-Xia Zhang,
Jiangshe Zhang, and Guanghai Li. HAM-MFN: hyperspec-
tral and multispectral image multiscale fusion network with
RAP loss. IEEE TGRS, 58(7):4618–4628, 2020. 1
[46] Shuang Xu, Jiangshe Zhang, Zixiang Zhao, Kai Sun, Junmin
Liu, and Chunxia Zhang. Deep gradient projection networks
for pan-sharpening. In CVPR, pages 1366–1375, 2021. 6
[47] Shuang Xu, Xiangyong Cao, Jiangjun Peng, Qiao Ke, Cong
Ma, and Deyu Meng. Hyperspectral image denoising by
asymmetric noise modeling. IEEE TGRS, 60:5545214, 2022.
2
[48] Shuang Xu, Qiao Ke, Jiangjun Peng, Xiangyong Cao, and
Zixiang Zhao. Pan-denoising: Guided hyperspectral image
denoising via weighted represent coefficient total variation.
IEEE TGRS, 62:5528714, 2024. 3, 7
[49] Zhi-Qin John Xu, Yaoyu Zhang, and Tao Luo. Overview
frequency principle/spectral bias in deep learning. arXiv,
2201.07395, 2022. 4
[50] Qiangqiang Yuan, Qiang Zhang, Jie Li, Huanfeng Shen, and
Liangpei Zhang. Hyperspectral image denoising employ-
ing a spatial-spectral deep residual convolutional neural net-
work. IEEE TGRS, 57(2):1205–1218, 2019. 2
[51] Haijin Zeng, Jiezhang Cao, Kai Zhang, Yongyong Chen,
Hiep Luong, and Wilfried Philips. Unmixing diffusion for
self-supervised hyperspectral image denoising. In CVPR,
pages 27820–27830, 2024. 3
[52] Hongyan Zhang, Wei He, Liangpei Zhang, Huanfeng Shen,
and Qiangqiang Yuan. Hyperspectral image restoration using
low-rank matrix recovery. IEEE TGRS , 52(8):4729–4743,
2014. 2, 4
[53] Qiang Zhang, Yushuai Dong, Yaming Zheng, Haoyang Yu,
Meiping Song, Lifu Zhang, and Qiangqiang Yuan. Three-
dimension spatial-spectral attention transformer for hyper-
spectral image denoising. IEEE TGRS , 62:5531213, 2024.
1
[54] Qiang Zhang, Yaming Zheng, Qiangqiang Yuan, Meiping
Song, Haoyang Yu, and Yi Xiao. Hyperspectral image de-
noising: From model-driven, data-driven, to model-data-
driven. IEEE TNNLS, 35(10):13143–13163, 2024. 1
[55] Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang,
Shuang Xu, Zudi Lin, Radu Timofte, and Luc Van Gool.
Cddfuse: Correlation-driven dual-branch feature decomposi-
tion for multi-modality image fusion. In CVPR, pages 5906–
5916, 2023. 1
12011

<!-- END-OF-PAPER:3c8dd44958a0 -->
