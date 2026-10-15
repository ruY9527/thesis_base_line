# A Selective Re-learning Mechanism for Hyperspectral Fusion Imaging

## sec:preamble preamble
_Pages 1-1_

A Selective Re-learning Mechanism for Hyperspectral Fusion Imaging
Yuanye Liu1
Jinyang Liu3
Renwei Dian1,2 *
Shutao Li3
1The School of Robotics, Hunan University, China
2Industrial Design and Machine Intelligence Innovation Research Institute,
Quanzhou-Hunan University, China
3The College of Electrical and Information Engineering, Hunan University, China
{yuanye liu, jinyangliu, drw, shutao li}@hnu.edu.cn
https://github.com/YuanyeLiu/SRLF-Net

## sec:preliminary Preliminary
_Pages 1-1_

fusion feature
Pseudo-MSI
Pseudo-LRHSI
High spectral
fidelity
Spatial
re-learning
Spatially distorted
feature points
Spectrally distorted
feature points
Value
Bands
Value
Bands
Rresponse
Bands
Spectral
response
Spatial
response
Spectral
re-learning
High spatial
fidelity
(a) Spatial-Spectral Structure-Guided Selective Re-learning Mechanism
(b) PSNR-FLOPs-Params Comparisons
PSNR (dB)
Params (M)
47.5
48.5
49.5
46.5
SRLF-Net
(ours)
Mog-DCN
MIMO-SST
LRTN
DSPNet
FusionMamba
DAEM
DHIF-Net
Figure 1. The diagram of the Spatial-Spectral Structure-Guided Selective Re-learning Mechanism (SSG-SRL) and PSNR-FLOPs-Params
Comparisons. (a) SSG-SRL adaptively filters out distorted feature points from the preliminary fusion feature for targeted learning. (b) The
PSNR-FLOPs-Params Comparisons evaluate different methods from three aspects: the quality of the fused image (vertical axis), model
size (horizontal axis), and computational complexity (circle radius). SRLF-Net outperforms other state-of-the-art methods while requiring
fewer floating-point operations (FLOPs).

## sec:abstract Abstract
_Pages 1-1_

Hyperspectral fusion imaging is challenged by high com-
putational cost due to the abundant spectral information.
We find that pixels in regions with smooth spatial-spectral
structure can be reconstructed well using a shallow net-
work, while only those in regions with complex spatial-
spectral structure require a deeper network. However, ex-
isting methods process all pixels uniformly, which ignores
this property. To leverage this property, we propose a Se-
lective Re-learning Fusion Network (SRLF) that initially ex-
tracts features from all pixels uniformly and then selectively
refines distorted feature points.
Specifically, SRLF first
employs a Preliminary Fusion Module with robust global
*Corresponding author. This paper is supported by the National Nat-
ural Science Foundation of China (No.
62422108, No.
62221002,
and No.
62201205), the Natural Science Foundation of Chongqing
(No. CSTB2024NSCQ-MSX0240), the Fujian Provincial Natural Science
Foundation of China (No. 2024J01227), and by the 2024 open project fund
of the National Key Laboratory of Electromagnetic Space Security.
modeling capability to generate a preliminary fusion fea-
ture. Afterward, it applies a Selective Re-learning Module
to focus on improving distorted feature points in the pre-
liminary fusion feature. To achieve targeted learning, we
present a novel Spatial-Spectral Structure-Guided Selective
Re-learning Mechanism (SSG-SRL) that integrates the ob-
servation model to identify the feature points with spatial or
spectral distortions. Only these distorted points are sent to
the corresponding re-learning blocks, reducing both com-
putational cost and the risk of overfitting. Finally, we de-
velop an SRLF-Net, composed of multiple cascaded SRLFs,
which surpasses multiple state-of-the-art methods on sev-
eral datasets with minimal computational cost.

## sec:introduction 1. Introduction
_Pages 1-2_

Hyperspectral imaging technology is widely used in remote
sensing [18, 19], environmental monitoring [27, 47], agri-
culture [21, 34], and various other fields due to its rich
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
spectral information. However, the limitations of imaging
principles result in a trade-off between spatial and spec-
tral resolution.
Achieving high spectral resolution often
requires sacrificing spatial resolution, limiting the practi-
cal applications of hyperspectral imaging technology. Re-
cently, computational imaging methods, such as spectral
super-resolution [3, 5, 10, 31], coded aperture snapshot
spectral imaging [4, 42, 46], and spatial-spectral fusion
[15, 23], have received significant attention for their ca-
pability to reconstruct high-resolution hyperspectral im-
ages (HRHSI) through advanced algorithms.
This paper
focuses on spectral-spatial fusion, which attempts to ob-
tain HRHSI by fusing low-resolution hyperspectral images
(LRHSI) with multispectral images (MSI).
Spectral-spatial fusion methods can be broadly catego-
rized into traditional and deep learning (DL)-based ap-
proaches.
Traditional methods usually rely on manual
priors to establish relationships among LRHSI, MSI, and
HRHSI. The nonlinear capabilities of such models are lim-
ited, making it challenging to obtain high-quality HRHSI.
Benefiting from the powerful nonlinear modeling capacity
of deep learning techniques, both convolutional neural net-
work (CNN)-based methods and Transformer-based meth-
ods have demonstrated performance far superior to tradi-
tional methods. Despite these advances, DL-based meth-
ods still face significant challenges, with one of the most
prominent being the trade-off between computational cost
and fusion performance. While many CNN-based methods
offer lower computational cost but limited fusion accuracy,
Transformer-based methods achieve better fusion accuracy
at the expense of higher computational cost. Reducing com-
putational cost while maximizing fusion accuracy is a crit-
ical practical challenge, especially important for deploying
models on edge computing platforms or compact cameras.
Recently, some promising lightweight frameworks [22,
38] for visible light image super-resolution have been pro-
posed. These frameworks employ classifiers to categorize
the image into complex and smooth regions. They then pro-
cess these regions with different network structures, effec-
tively reducing the overall computational cost. However,
applying a classifier to hyperspectral images (HSIs) incurs
significant computational cost, as it needs to account for the
spatial and spectral structure simultaneously. Additionally,
directly processing the image patches or pixels obtained
through classification disrupts their inherent position infor-
mation.
To address the issues mentioned above, we propose a
novel network, SRLF-Net, which primarily consists of sev-
eral cascaded SRLFs. Each SRLF comprises two modules:
a Preliminary Fusion Module and a Selective Re-learning
Module. The Preliminary Fusion Module aims to prelim-
inarily extract feature before breaking the spatial position
relationship. It employs a multi-scale Mamba architecture,
providing strong global perception capability. The Prelim-
inary Fusion Module outputs a preliminary fusion feature,
with most feature points accurately extracted and only a few
feature points distorted. The Selective Re-learning Mod-
ule consists of two Spectral Transformer blocks and one
Mamba block. The Spectral Transformer blocks are respon-
sible for re-learning the feature points with spatial and spec-
tral distortions, respectively, while the Mamba block is re-
sponsible for integrating the re-learned results. To achieve
selective learning, we propose a Spatial-Spectral Guided
Selective Re-learning Mechanism (SSG-SRL), as shown in
Fig. 1. This mechanism incorporates the observation model
to generate pseudo-MSI and pseudo-LRHSI. It then iden-
tifies distorted feature points from both spectral and spa-
tial perspectives, requiring only a few parameters and in-
curring extremely low computational cost. By focusing on
distorted feature points and avoiding the over-processing of
well-extracted ones, this selective re-learning approach re-
duces the computational cost and mitigates overfitting.
Our contribution can be summarized as follows:
• We propose a novel SRLF integrating Mamba and Trans-
former, which first performs preliminary fusion and then
adaptively selects distorted feature points for targeted
learning, thereby avoiding computational waste and miti-
gating overfitting.
• We propose a Spatial-Spectral Guided Selective Re-
learning Mechanism that integrates the observation mod-
els of LRHSI and MSI, efficiently enabling selective
learning with few parameters (Params) and low compu-
tational cost.
• Based on SRLF, we construct an SRLF-Net for spatial-
spectral fusion, which balances fusion accuracy and com-
putational cost, surpassing current state-of-the-art meth-
ods on several public datasets with minimal computa-
tional cost.

## sec:related-work 2. Related Work
_Pages 2-4_

2.1. Traditional Methods
Traditional spatial-spectral fusion methods can be catego-
rized into two types: PAN-sharpening-based methods and
decomposition-based methods. Early approaches, such as
component substitution (CS)-based methods [2, 35] and
multiresolution analysis (MRA)-based methods [1, 26],
were initially designed for panchromatic sharpening tasks,
but later were also used for fusing LRHSIs and MSIs. The
decomposition-based methods mainly refer to matrix fac-
torization (MF)and tensor decomposition (TD). The MF-
based approach unfolds hyperspectral data into a two-
dimensional matrix and decomposes it into a spectral fea-
ture matrix and a mixing coefficient matrix. For instance,
a representative MF-based method was proposed by La-
naras et al. [24], which transformed the spatial-spectral fu-
sion optimization problem into an estimation problem for a
constrained spectral feature matrix and a mixing coefficient
matrix. Li et al. [25] introduced regularization techniques
based on both external and internal priors in the matrix rep-
resentation. However, the unfolding operation disrupts the
three-dimensional structure of HSI, which is detrimental to
the reconstruction of HSI information. The TD-based meth-
ods treat the HSI as a three-dimensional tensor, overcoming
the limitations of MF-based methods and demonstrating a
certain level of advancement. For instance, Dian et al. [9]
proposed a generalized tensor nuclear norm to enforce low-
rank constraints in three spatial-spectral dimensions.
2.2. Deep Learning-based Methods
Deep Learning (DL)-based methods offer significant per-
formance improvements in the field of hyperspectral recon-
struction, thus gaining much attention [28, 29, 44, 45]. For
example, Palsson et al. [32] first employed the PCA tech-
nique for data dimensionality reduction and then fed the
reduced data into a fusion network based on 3D convo-
lution. In addition, some methods utilize the observation
model to improve the fusion performance. For example,
Xie et al. [41] and Dong [12] incorporated the observa-
tion model into the fusion network. Dian et al. [7] uti-
lized the observation model to generate training data, which
ensures the consistency of the imaging model between the
test data and the training data. CNN-based methods often
struggle with limited receptive fields, making them less ef-
fective in capturing global context. In contrast, the Multi-
Head Self-Attention (MHSA) mechanism in Transformers
allows for the effective capture of global information. Many
Transformer-based methods [30, 37] have been proposed
and have achieved better results. However, these methods
are limited by quadratic computational complexity.
Re-
cently, Mamba, derived from state-space equation models
in control theory, has also been introduced to the field of
HSI reconstruction. For example, Peng et al. [33] designed
the FusionMamba module capable of handling dual inputs,
and embedded it into two U-shaped networks to achieve hy-
perspectral reconstruction.
Existing methods treat all pixels equally. However, pix-
els in smooth regions can be learned by a shallow network,
whereas only those in complex regions require a deeper net-
work. To leverage this property, we propose a selective re-
learning mechanism that adaptively identifies distorted pix-
els for targeted refinement, thereby enhancing fusion qual-
ity while reducing computational cost.
3. Our Proposed Method
3.1. Motivation and Framework Overview
Motivation. The process of image reconstruction over a
network resembles how humans learn new things: first, ac-
quiring an overall understanding of the content, and then fo-
cusing on more complex details. Therefore, we propose that
the network should first perform an initial feature extraction
of the entire image and subsequently refine a few distorted
feature points. This strategy can reduce the excessive learn-
ing on low-difficulty feature points, thereby mitigating the
risk of overfitting and lowering FLOPs.
Framework Overview. The structure of SRLF-Net is
shown in Fig. 2. The input MSI and LRHSI are denoted as
Y and X, respectively. First, they are concatenated along
the channels and pass through an embedding layer to adjust
the channel dimensions, resulting in F0. Next, F0 under-
goes N SRLFs to produce [F1, F2, ..., FN]. Finally, a con-
volutional layer serves as the Refine layer to fuse these fea-
tures [F0, F1, F2, ..., FN], resulting in the fused HSI. The
SRLF consists of two parts: a Preliminary Fusion Mod-
ule and a Selective Re-learning Module. The Preliminary
Fusion Module generates a fusion feature, while the Selec-
tive Re-learning Module refines its distorted points based
on spatial-spectral structure.
3.2. Preliminary Fusion Module
Recently, Mamba [14] has been used to replace Trans-
former to capture global information with linear computa-
tional complexity, which can be discretized as:
y = Ku,
K = (CB, CAB, CA
L−1B),
(1)
where y is the output, K represents the convolutional ker-
nel, and L is the length of the input sequence.
¯A, ¯B, and
¯C are parameter matrices, with ¯B and ¯C varying according
to the input u. In practical computation, the Mamba mod-
ule employs kernel fusion and recomputation to improve the
efficiency of feature sequence scanning and reduce storage
requirements.
To extract feature efficiently, we apply the SS2D block
[16], an improved Mamba structure, as the base module.
It uses a cross-scan method to obtain the global receptive
field. The structure and Effective Receptive Field (ERF)
of the SS2D block are shown in Fig. 3(a). The figure illus-
trates that a single SS2D block primarily focuses on a cross-
shaped region, indicating that the module captures sequen-
tial information near the current position. The pixels outside
the cross shape have less impact on the central point because
they are farther away from it during the four traversal pro-
cesses. This is because each scanning path follows Eq. 1,
and the weights decay as the distance increases. Although
the SS2D block can capture global information, its percep-
tion capability is limited, primarily focusing on the pixels
that are closer along the scanning path. Owing to the prin-
ciple of SSM, the weights corresponding to each pixel are
not as independent as they are in the convolutional kernel.
...
Embedding
Embedding

## sec:preliminary-2 Preliminary
_Pages 4-4_

Fusion
Module

## sec:preliminary-3 Preliminary
_Pages 4-5_

Fusion
Module
Spa-RL
Spa-RL
Spe-RL
Spe-RL
Selective Re-learning Module
Selective Re-learning Fusion Network (SRLF)
...
Refine
Refine
F0 F1 F2
FN-1
F2 FN-1
Upsampling
Upsampling
LRHSI
MSI
HRHSI
Figure 2. The overview of the proposed SRLF-Net. The Embedding layer and the Refine layer are both individual convolutional layers.
SS2D
SS2D
Down
Down
SS2D
SS2D
SS2D
SS2D
Linear
Linear
Linear
Linear
SS2D
SS2D
SS2D
SS2D
SS2D
SS2D
DWConv
DWConv
Linear
Linear
SiLU
SiLU
2D-Selective Scan Operation
SUM
SUM
Cross-scan
LayerNorm
LayerNorm
Linear
Linear
SiLU
SiLU
(b) Cascaded Structure and Effective Receptive Field
(a) 2D-Selective Scan Block and Effective Receptive Field
(c) Multi-Scale Cascaded Structure and Effective Receptive Field
1 2
1 3 2 4
4 3 2 1
4 2 3 1
Feature
Figure 3. Different structures of Preliminary Fusion Modules and Effective Receptive Fields. (a) Due to the limitations of the SSM
principle, the global perception capability of a single SS2D block is limited. (b) The ERF of the cascade structure has slightly improved,
but not significantly. (c) The ERF of the multi-scale cascaded structure has significantly improved, resulting in a stronger global perception
capability.
Even after training, the ERF still exhibits a cross-shaped
pattern. Figure 3(b) shows the structure of simply stacking
three SS2D blocks and their corresponding ERF. The high-
lighted areas of the ERF are still concentrated on a few pix-
els, which is insufficient for broader global perception. To
enhance the global perception capability of the Preliminary
Fusion Module, we adopt a multi-scale structure, which has
a significantly larger highlighted central region in its ERF,
as shown in Fig. 3(c). Denote the input of this module as
Fin, then the process of the Preliminary Fusion Module can
be expressed as:
T1 = SS2D(Fin),
T d
1 = Down(T1),
T2 = SS2D(T d
1 ),
2 = PS(T2),
Fout = SS2D(Concat(T
2, T1)),
(2)
where SS2D(·) is the function of the SS2D block, and
Concat(·) represents the concatenation operation along the
channel dimension. Down(·)represents the downsampling
operation implemented by convolution, and PS(·) is the
PixelShuffle (PS) operation, which is an upsampling tech-
nique. T1 and T2 are the outputs of the first two SS2D
blocks, respectively. T d
1 is the outcome of the downsam-
pling opearation, and T
2 is the outcome of the PS operation.
Fout is the fusion result of two scale features.
3.3. Selective Re-learning Module
As shown in Fig. 2, the Selective Re-learning Module con-
sists of a Spatial Re-learning (Spa-RL) block, a Spectral Re-
learning (Spe-RL) block, and a Re-fusion (RF) block. Spa-
RL block and Spe-RL block utilize the same Spectral Trans-
former structure, refining spatially and spectrally distorted
feature points, respectively. The RF block is an SS2D block
that aims to integrate the results of spatial and spectral re-
learning. The input to the Selective Re-learning Module is a
fused feature, where the representation of spatially smooth
regions is accurate, with only a few points exhibiting signif-
icant distortion. Therefore, we propose a Spatial-Spectral
Structure-Guided Selective Re-learning Mechanism (SSG-
11.67G per fine-tuning, multiplied by the number of fine-tunings.
Spatial
response
Spatial
response
SAM-guided
selection
SAM-guided
selection
Re-learning
Re-learning
(b) Spectral Re-learning Block
Spectral response
Spectral response
SSIM-guided
selection
SSIM-guided
selection
Re-learning
Re-learning
(a) Spatial Re-learning Block
Rresponse
Bands
Figure 4. The process of the Spatial Re-learning (Spa-RL) block and Spectral Re-learning (Spe-RL) block. The Spa-RL block and the
Spe-RL block incorporate the SSG-SRL, enabling targeted re-learning of spatial-spectral distorted feature points for reconstruction.
Table 1. Quantitative indexes of the test methods on the CAVE dataset and the Harvard dataset. The best result is marked in bold font.

## sec:method Method
_Pages 5-6_

FLOPS
Params(M)
CAVE
Harvard
PSNR
SAM
UIQI
SSIM
PSNR
SAM
UIQI
SSIM
Hysure [36]
42.9303
8.1108
0.9321
0.9814
45.2041
3.6180
0.8689
0.9807
NSSR [11]
46.6246
3.2505
0.9563
0.9910
47.1459
2.9396
0.8899
0.9843
DAEM [17]
417.5G1
0.02
41.5822
4.1965
0.9283
0.9796
43.3745
4.3586
0.8467
0.9728
FusionMamba [33]
129.62G
2.58
46.8107
2.6511
0.9710
0.9937
47.6266
2.7612
0.8939
0.9851
DHIF-Net [20]
3.616T
22.6695
47.8613
2.3849
0.9732
0.9948
47.7735
2.6987
0.8953
0.9855
DSPNet [37]
422.189G
6.0551
47.5665
2.5881
0.9722
0.9941
47.5973
2.7785
0.8938
0.9851
MIMO-SST [13]
98.1653G
4.983
48.3609
2.5834
0.9706
0.9944
47.6915
2.7804
0.8941
0.9852
Mog-DCN [12]
4.390T
7.071
48.5910
2.2541
0.9748
0.9953
47.8910
2.6964
0.8961
0.9855
LRTN [30]
132.586G
3.535
47.7819
2.4387
0.9675
0.9939
47.7088
2.7330
0.8938
0.9855
Ours
81.22G
1.33
49.3881
2.1816
0.9752
0.9954
47.8954
2.6876
0.8956
0.9856
SRL) in the Spa-RL block and Spe-RL block.
Specifically, the output of the Preliminary Fusion Mod-
ule is denoted as Z′. We then use the Structural Similarity
Index (SSIM) to evaluate the quality of each feature point
in terms of spatial structure. However, directly comparing
Z′ with the Ground Truth (GT) Z is unfeasible. To address
this, we acquire the pseudo-MSI Y ′ using the spectral re-
sponse function of the observation model as:
Y ′ = RZ′,
(3)
where R represents the spectral response function, which
is assumed to be known. Next, we calculate the SSIM map
Mspa between Y ′ and Y to assess the spatial quality of each
feature point:
Mspa = SSIM(Y, Y ′).
(4)
We then select a few low-scoring feature points for re-
learning as:
M ′
spa = Sortp(Flatten(Mspa)),
VY ′, VY = Select(Y ′, Y ; M ′
spa, r),
V = RLspa(VY ′ + VY ),
spa = V + Y ′ + Y,
(5)
where Sortp(·) represents sorting in ascending order, and
Flatten(·) indicates unfolding along the spatial dimen-
sion. Select(·) selecting feature points based on M ′
spa, and
RLspa(·) denotes the re-learning block for spatial refine-
ment. M ′
spa is the sorted SSIM map. VY ′ and VY are spa-
tially distorted feature points in Y ′ and Y , respectively. V
represents the refined result of these distorted feature points,
and Z′
spa is the output of the Spa-RL block. r is the pro-
portion of spatially distorted feature points, which is deter-
mined based on the average SSIM score as:
r = λ(1 −Avg(Mspa)),
(6)
(a) GT
(b) NSSR
(c) Hysure
(d) DHIF-Net
(e) DSPNet
(f) MIMO-SST
(g) Mog-DCN
(h) LRTN
(i) Ours
Figure 5. The pseudo-color images and spectral angle error maps of reconstructed jelly beans ms (a test image of the CAVE dataset) by
different methods.
(a) GT
(b) NSSR
(c) Hysure
(d) DHIF-Net
(e) DSPNet
(f) MIMO-SST
(g) Mog-DCN
(h) LRTN
(i) Ours
Figure 6. The pseudo-color images spectral angle error maps of reconstructed imgf7 (a test image of the Harvard dataset) by different

## sec:methods methods.
_Pages 6-7_

where λ is a hyperparameter that adjusts the proportion of
feature points identified as distorted. Avg(·) denotes the av-
erage operation. A high Avg(Mspa) indicates a more accu-
rate feature representation, which means that fewer feature
points need to be re-learned. Conversely, a low Avg(Mspa)
suggests that more feature points need to re-learning.
Similarly, we degrade the result of spatial re-learning,
spa, into pseudo-LRHSI X′ as:
X′ = (Z′
spa ∗C) ↓,
(7)
where C is the Gaussian blur kernel. ∗and ↓denote convo-
lution and spatial downsampling operations. We then eval-
uate the spectral quality of each feature point via SAM map
Mspe as:
Mspe = SAM(X′, X).
(8)
We then select the feature points with high SAM scores,
at a ratio of r, as those exhibiting spectral distortion. We
then refine these feature points as:
M ′
spe = Sortr(Flatten(Mspe)),
UX′, UX = Select(X′, X; M ′
spe, r),
U = RLspe(UX′ + UX),
spe = U + X′ + X,
(9)
where Sortr
represents sorting in descending order.
RLspe(·) denotes the re-learning block for spectral refine-
ment, sharing the same structure as RLspa(·). M ′
spe is the
sorted SAM map. UX′ and UX are the spectrally distorted
feature points in X′ and X, respectively. U represents the
refined result of these distorted feature points. Z′
spe is the
output of the Spe-RL block. Finally, the RF block fuses
spe and Z′
spa to produce the output of the entire Selective
Re-learning Module.
3.4. FLOPs Analysis of SSG-SRL
The spectral response and spatial response can be consid-
ered as downsampling in the network, which is common.
Therefore, we ignore this part and only account for the
FLOPs caused by SSIM and SAM computations as:
FLOPsSSIM = C1(5k2 + 10)HW,
FLOPsSAM = 6HW
+ 2C2
s2 ,
(10)
where s represents the downsampling factor of the spatial
response, and k denotes the size of the Gaussian blur kernel.
C1 and C2 are the number of spectral channels for MSI and
HSI, respectively. H and W denote the spatial height and
width of the input MSI. In our experiments, H = W = 512,
(a) LRHSI
(b) NSSR
(c) Hysure
(d) DHIF-Net
(e) DSPNet
(f) MIMO-SST
(g) Mog-DCN
(h) LRTN
(i) Ours
Figure 7. The pseudo-color images of reconstructed Gaofen5 (part) by different methods.
Table 2. No-reference quality assessment for Gaofen5. The best result is marked in bold font.
metric

## sec:method-2 Method
_Pages 7-7_

Hysure [36]
NSSR [11]
FusionMamba [33]
DHIF-Net [20]
DSPNet [37]
MIMO-SST [13]
Mog-DCN [12]
LRTN [30]
Ours
QNR
0.9456
0.8944
0.9626
0.9717
0.9722
0.9714
0.9720
0.9718
0.9733
Table 3.
Ablation study on network depth.
The best result is
marked in bold font.

## sec:model Model
_Pages 7-7_

PSNR
SAM
UIQI
SSIM
N = 1
48.2703
2.3074
0.9731
0.9943
N = 2
48.7428
2.2172
0.9745
0.9950
N = 3
49.0562
2.2477
0.9750
0.9951
N = 4
49.3881
2.1816
0.9752
0.9954
N = 5
49.5214
2.1693
0.9753
0.9954
Table 4. Ablation study on key modules. The best result is marked
in bold font.

## sec:model-2 Model
_Pages 7-7_

FLOPs(G)
PSNR
SAM
UIQI
SSIM
w/o multi-scale
84.58
48.0473
2.4111
0.9713
0.9937
w/o selection
86.65
49.3135
2.2193
0.9751
0.9953
w/o B and R
81.22
49.3412
2.2054
0.9752
0.9953
Ours
81.22
49.3881
2.1816
0.9752
0.9954
C1 = 3, C2 = 31, s = 8, and k = 5. Therefore, the FLOPs
for a single SSG-SRL are less than 0.11 GFLOPs 2, which
is almost negligible.

## sec:experiments 4. Experiments
_Pages 7-8_

4.1. Datasets and Evaluation Metrics
We train the proposed SRLF-Net on three publicly avail-
able datasets: CAVE [43], Harvard [6], and Gaofen5. For
quantitative evaluation, we employ four image assessment
metrics: Peak Signal-to-Noise Ratio (PSNR), Spectral An-
gle Mapper (SAM), Universal Image Quality Index (UIQI
[39]), and Structural Similarity Index Measure (SSIM [40]).
4.2. Compared Methods
We compare SRLF-Net with two classic traditional meth-
ods: Hysure [36] and NSSR [11], as well as several state-
of-the-art DL-based methods: DAEM [17], FusionMamba
2The result of the FLOPs calculation only considers multiplication and
division operations.
SAM (deg)
FLOPs (G)
Figure 8. Robustness analysis of the ratio coefficient λ.
[33], LRTN [30], Mog-DCN [12], DHIF-Net [20], DSPNet
[37], and MIMO-SST [13].
4.3. Experimental Results on Simulated Data
Table 1 records the test results of all methods on the CAVE
and Harvard datasets. Our method achieves the best per-
formance with fewer FLOPs and Params. Figures 5 and 6
illustrate the pseudo-color images and spectral error maps
of the fusion results obtained from traditional approaches
and some competitive DL-based methods. The magnified
local regions of the images reveal that the results produced
by our method have clearer texture details and higher spec-
tral fidelity. Although Mog-DCN also achieves good visual
effects on the Harvard dataset, its computational cost is rel-
atively high.
4.4. Experimental Results on Real Data
For the Gaofen5 dataset, we perform spatial downsampling
on the existing LRHSI and MSI according to [15] and [8]
to obtain the input data for training, using the LRHSI as the
GT. During testing, we input the original LRHSI and MSI
to obtain the HRHSI. Since the Gaofen5 dataset doesn’t
have a corresponding HRHSI, we use the no-reference im-
age quality assessment metric QNR to quantitatively evalu-
ate the performance of different methods, as shown in Ta-
ble 2. Our method achieves the highest QNR. Additionally,
Fig. 7 presents the visual results of the reconstructed im-
ages generated by traditional approaches and some compet-
MSI
LRHSI
Visualization of selection based on spatial structure
Visualization of selection based on spectral structure
Stage 1
Stage 2
Stage 3
Stage 4
Stage 1
Stage 2
Stage 3
Stage 4
Figure 9. Visualization of the positions of distorted feature points under varied ratio coefficient λ for different stages. Visualization of
selection based on spatial structure shows the values of selected feature point positions in the original MSI, while the visualization based
on spectral structure displays the corresponding values in the original LRHSI. Black areas represent unselected feature points.
itive deep learning-based methods. The figure shows that
the reconstruction visual quality of deep learning methods
is significantly better than that of traditional methods.
4.5. Ablation Study
Impact of network depth N. The network depth N deter-
mines the number of SRLFs. Table 3 records the results for
N ranging from 1 to 5. It shows that when N <= 4, eval-
uation metrics improve significantly as N increases. For
instance, PSNR increases by at least 0.3134 dB (49.0562
dB −48.7428 dB). However, when N = 5, the quality im-
provement of the fused image is not substantial, with PSNR
increasing by only 0.1333 dB compared to N = 4. Consid-
ering the computational cost and fusion quality, N = 4 is
the optimal choice.
Impact of Preliminary Fusion Module.
In the Pre-
liminary Fusion Module, we employ a multi-scale Mamba
structure to enhance global perception ability. To assess the
performance of this structure, we substitute it with the struc-
ture shown in Fig. 3(b). As indicated in Table 4, the PSNR
of the model without multi-scale decreases by 1.3408 dB.
Impact of SSG-SRL. SSG-SRL is to filter out distorted
feature points for targeted learning. To verify the effect of
this selective re-learning, we retain the network structure
of the Selective Re-learning Module but removes the SSG-
SRL. The experimental results are presented in Table 4. Re-
moving the SSG-SRL not only decreases the PSNR but also
increases the FLOPs by 5.43 GFLOPs.
Impact of prior C and R. SRLF-Net involves the spec-
tral response function R and the spatial response function
B of the observation model (Eq. 3 and Eq. 7), which are
assumed to be known. Here, we test the fusion results when
R and C are unknown. We substitute a 5 × 5 strided con-
volutional layer for the spatial response function and a lin-
ear layer without bias for the spectral response function. As
shown in Table 4, SRLF-Net achieves satisfactory fusion re-
sults even when R and C are unknown, with PSNR reaches
49.3412 dB.
Impact of the ratio coefficient λ. The ratio coefficient
λ determines the number of feature points entering the re-
learning blocks. We adjust the value of λ during inference
from 0 to 1 to test its effect on network performance and
FLOPs. As shown in Fig. 8, SAM remains relatively sta-
ble when 0.1 ≤λ ≤0.3. When λ > 0.3, SAM begins to
fluctuate, and as λ > 0.5, SAM gradually increases. Mean-
while, the FLOPs grow linearly as λ increases. In Fig. 9, we
further visualize the selection results at different stages for
various values of λ. From the figure, it can be seen that in
all four stages, the feature points selected based on spatial
quality tend to be located in areas with significant texture
variation. In contrast, the feature points selected based on
spectral quality differ across stages.

## sec:conclusion 5. Conclusion
_Pages 8-8_

In this paper, we propose a novel Selective Re-learning
Fusion Network (SRLF), which selects distorted feature
points based on a preliminary fusion feature for re-
learning.
Specifically, SRLF first performs feature fu-
sion through the Preliminary Fusion Module and then
refines distorted feature points using the Selective Re-
learning Module.
To learn distorted feature points pur-
posefully, we introduce a Spatial-Spectral Structure-Guided
Selective Re-learning Mechanism (SSG-SRL), which mit-
igates the overfitting of accurate feature points, thereby
enhancing fusion accuracy while reducing computational
cost. Based on SRLF, we develop SRLF-Net for spatial-
spectral fusion, which outperforms several state-of-the-art
methods with minimal computational cost across multiple

## sec:references References
_Pages 9-10_

[1] Bruno Aiazzi, Luciano Alparone, Stefano Baronti, Andrea
Garzelli, and Massimo Selva. Mtf-tailored multiscale fusion
of high-resolution ms and pan imagery. Photogrammetric
Engineering & Remote Sensing, 72(5):591–596, 2006. 2
[2] Bruno Aiazzi, Stefano Baronti, and Massimo Selva.
Im-
proving component substitution pansharpening through mul-
tivariate regression of ms + pan data. IEEE Transactions on
Geoscience and Remote Sensing, 45(10):3230–3239, 2007.
[3] Naveed Akhtar and Ajmal Mian.
Hyperspectral recovery
from rgb images using gaussian processes. IEEE Transac-
tions on Pattern Analysis and Machine Intelligence, 42(1):
100–113, 2020. 2
[4] Yuanhao Cai, Jing Lin, Xiaowan Hu, Haoqian Wang, Xin
Yuan, Yulun Zhang, Radu Timofte, and Luc Van Gool.
Mask-guided spectral-wise transformer for efficient hy-
perspectral image reconstruction.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 17502–17511, 2022. 2
[5] Yuanhao Cai, Jing Lin, Zudi Lin, Haoqian Wang, Yulun
Zhang, Hanspeter Pfister, Radu Timofte, and Luc Van Gool.
Mst++: Multi-stage spectral-wise transformer for efficient
spectral reconstruction.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 745–755, 2022. 2
[6] Ayan Chakrabarti and Todd Zickler. Statistics of real-world
hyperspectral images. In Proceedings of the IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 193–200, 2011. 7
[7] Renwei Dian, Anjing Guo, and Shutao Li. Zero-shot hyper-
spectral sharpening. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 45(10):12650–12666, 2023. 3
[8] Renwei Dian, Anjing Guo, and Shutao Li. Zero-shot hyper-
spectral sharpening. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 45(10):12650–12666, 2023. 7
[9] Renwei Dian, Yuanye Liu, and Shutao Li.
Hyperspectral
image fusion via a novel generalized tensor nuclear norm
regularization. IEEE Transactions on Neural Networks and
Learning Systems, 2024. 3
[10] Renwei Dian, Yuanye Liu, and Shutao Li. Spectral super-
resolution via deep low-rank tensor representation.
IEEE
Transactions on Neural Networks and Learning Systems,
2024. 2
[11] Weisheng Dong, Fazuo Fu, Guangming Shi, Xun Cao, Jin-
jian Wu, Guangyu Li, and Xin Li.
Hyperspectral image
super-resolution via non-negative structured sparse represen-
tation. IEEE Transactions on Image Processing, 25(5):2337–
2352, 2016. 5, 7
[12] Weisheng Dong, Chen Zhou, Fangfang Wu, Jinjian Wu,
Guangming Shi, and Xin Li. Model-guided deep hyperspec-
tral image super-resolution.
IEEE Transactions on Image
Processing, 30:5754–5768, 2021. 3, 5, 7
[13] Jian Fang, Jingxiang Yang, Abdolraheem Khader, and Liang
Xiao. Mimo-sst: Multi-input multi-output spatial-spectral
transformer for hyperspectral and multispectral image fu-
sion. IEEE Transactions on Geoscience and Remote Sensing,
62:1–20, 2024. 5, 7
[14] Albert Gu and Tri Dao. Mamba: Linear-time sequence mod-
eling with selective state spaces, 2024. 3
[15] Anjing Guo, Renwei Dian, and Shutao Li. Unsupervised blur
kernel learning for pansharpening. In IGARSS 2020 - 2020
IEEE International Geoscience and Remote Sensing Sympo-
sium, pages 633–636, 2020. 2, 7
[16] Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong
Ren, and Shu-Tao Xia.
Mambair: A simple baseline for
image restoration with state-space model. In European con-
ference on computer vision, pages 222–241. Springer, 2024.
[17] Wen-jin Guo, Weiying Xie, Kai Jiang, Yunsong Li, Jie
Lei, and Leyuan Fang.
Toward stable, interpretable,
and lightweight hyperspectral super-resolution.
In 2023
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 22272–22281, 2023. 5, 7
[18] Xian Guo, Xin Huang, Lefei Zhang, Liangpei Zhang, An-
tonio Plaza, and J´on Atli Benediktsson. Support tensor ma-
chines for classification of hyperspectral remote sensing im-
agery. IEEE Transactions on Geoscience and Remote Sens-
ing, 54(6):3248–3264, 2016. 1
[19] Danfeng Hong, Bing Zhang, Xuyang Li, Yuxuan Li, Chenyu
Li, Jing Yao, Naoto Yokoya, Hao Li, Pedram Ghamisi, Xiup-
ing Jia, Antonio Plaza, Paolo Gamba, Jon Atli Benediktsson,
and Jocelyn Chanussot. Spectralgpt: Spectral remote sensing
foundation model. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 46(8):5227–5244, 2024. 1
[20] Tao Huang, Weisheng Dong, Jinjian Wu, Leida Li, Xin
Li, and Guangming Shi. Deep hyperspectral image fusion
network with iterative spatio-spectral regularization. IEEE
Transactions on Computational Imaging, 8:201–214, 2022.
5, 7
[21] Jianxin Jia, Jinsong Chen, Xiaorou Zheng, Yueming Wang,
Shanxin Guo, Haibin Sun, Changhui Jiang, Mika Kar-
jalainen, Kirsi Karila, Zhiyong Duan, Tinghuai Wang,
Chong Xu, Juha Hyypp¨a, and Yuwei Chen.
Tradeoffs in
the spatial and spectral resolution of airborne hyperspectral
imaging systems: A crop identification case study.
IEEE
Transactions on Geoscience and Remote Sensing, 60:1–18,
2022. 1
[22] Xiangtao Kong, Hengyuan Zhao, Yu Qiao, and Chao
Dong. Classsr: A general framework to accelerate super-
resolution networks by data characteristic. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 12016–12025, 2021. 2
[23] Zeqiang Lai, Ying Fu, and Jun Zhang. Hyperspectral im-
age super resolution with real unaligned rgb guidance. IEEE
Transactions on Neural Networks and Learning Systems,
2024. 2
[24] Charis
Lanaras,
Emmanuel
Baltsavias,
and
Konrad
Schindler.
Hyperspectral super-resolution by coupled
spectral unmixing. In 2015 IEEE International Conference
on Computer Vision (ICCV), pages 3586–3594, 2015. 2
[25] Shutao Li, Renwei Dian, and Haibo Liu. Learning the ex-
ternal and internal priors for multispectral and hyperspectral
image fusion. Science China Information Sciences, 66(4):
140303, 2023. 3
[26] JG Liu.
Smoothing filter-based intensity modulation: A
spectral preserve image fusion technique for improving spa-
tial details. International Journal of remote sensing, 21(18):
3461–3472, 2000. 2
[27] Jie Liu, Qisheng Feng, Tiangang Liang, Jianpeng Yin, Jin-
long Gao, Jing Ge, Mengjing Hou, Caixia Wu, and Wenlong
Li. Estimating the forage neutral detergent fiber content of
alpine grassland in the tibetan plateau using hyperspectral
data and machine learning algorithms. IEEE Transactions
on Geoscience and Remote Sensing, 60:1–17, 2022. 1
[28] Jinyang Liu, Renwei Dian, Shutao Li, and Haibo Liu. Sgfu-
sion: A saliency guided deep-learning framework for pixel-
level image fusion. Information Fusion, 91:205–214, 2023.
[29] Jinyang Liu, Shutao Li, Haibo Liu, Renwei Dian, and Xiao-
hui Wei. A lightweight pixel-level unified image fusion net-
work. IEEE Transactions on Neural Networks and Learning
Systems, 2023. 3
[30] Yuanye Liu, Renwei Dian, and Shutao Li.
Low-rank
transformer for high-resolution hyperspectral computational
imaging. International Journal of Computer Vision, pages
1–16, 2024. 3, 5, 7
[31] Shaohui Mei, Ge Zhang, Nan Wang, Bo Wu, Mingyang Ma,
Yifan Zhang, and Yan Feng.
Lightweight multiresolution
feature fusion network for spectral super-resolution. IEEE
Transactions on Geoscience and Remote Sensing, 61:1–14,
2023. 2
[32] Frosti Palsson, Johannes R. Sveinsson, and Magnus O. Ul-
farsson. Multispectral and hyperspectral image fusion using
a 3-d-convolutional neural network. IEEE Geoscience and
Remote Sensing Letters, 14(5):639–643, 2017. 3
[33] Siran Peng, Xiangyu Zhu, Haoyu Deng, Liang-Jian Deng,
and Zhen Lei. Fusionmamba: Efficient remote sensing im-
age fusion with state space model. IEEE Transactions on
Geoscience and Remote Sensing, 62:1–16, 2024. 3, 5, 7
[34] Mar´ıa D. Raya-Sereno,
Mar´ıa Alonso-Ayuso,
Jos´e L.
Pancorbo, Jos´e L. Gabriel, Carlos Camino, Pablo J. Zarco-
Tejada, and Miguel Quemada. Residual effect and n fertilizer
rate detection by high-resolution vnir-swir hyperspectral im-
agery and solar-induced chlorophyll fluorescence in wheat.
IEEE Transactions on Geoscience and Remote Sensing, 60:
1–17, 2022. 1
[35] Vittala K Shettigara. A generalized component substitution
technique for spatial enhancement of multispectral images
using a higher resolution data set. Photogrammetric Engi-
neering and remote sensing, 58(5):561–567, 1992. 2
[36] Miguel Simoes, Jos´e Bioucas-Dias, Luis B Almeida, and
Jocelyn Chanussot.
A convex formulation for hyperspec-
tral image superresolution via subspace-based regularization.
IEEE Transactions on Geoscience and Remote Sensing, 53
(6):3373–3388, 2014. 5, 7
[37] Yucheng Sun, Han Xu, Yong Ma, Minghui Wu, Xiaoguang
Mei, Jun Huang, and Jiayi Ma. Dual spatial-spectral pyra-
mid network with transformer for hyperspectral image fu-
sion. IEEE Transactions on Geoscience and Remote Sensing,
2023. 3, 5, 7
[38] Yan Wang, Yi Liu, Shijie Zhao, Junlin Li, and Li Zhang.
Camixersr: Only details need more” attention”. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 25837–25846, 2024. 2
[39] Zhou Wang and A.C. Bovik. A universal image quality in-
dex. IEEE Signal Processing Letters, 9(3):81–84, 2002. 7
[40] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli.
Image quality assessment: from error visibility to structural
similarity. IEEE Transactions on Image Processing, 13(4):
600–612, 2004. 7
[41] Qi Xie, Minghao Zhou, Qian Zhao, Zongben Xu, and Deyu
Meng. Mhf-net: An interpretable deep network for multi-
spectral and hyperspectral image fusion. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 44(3):1457–
1473, 2022. 3
[42] Kazuhiro Yamawaki and Xian-Hua Han. Lightweight hy-
perspectral image reconstruction network with deep feature
hallucination. In Proceedings of the Asian Conference on
Computer Vision, pages 164–178, 2022. 2
[43] Fumihito Yasuma, Tomoo Mitsunaga, Daisuke Iso, and
Shree K. Nayar. Generalized assorted pixel camera: Post-
capture control of resolution, dynamic range, and spectrum.
IEEE Transactions on Image Processing, 19(9):2241–2253,
2010. 7
[44] Chong Zhang, Wenjing Liu, Juntao Li, Siqi Li, Lizhi Wang,
Hua Huang, Yuanjin Zheng, Yongtian Wang, Jinli Suo, and
Weitao Song. Tunable optimally-coded snapshot hyperspec-
tral imaging for scene adaptation. Laser & Photonics Re-
views, page 2401921, 2025. 3
[45] Chong Zhang, Xianglei Liu, Lizhi Wang, Shining Ma, Yuan-
jin Zheng, Yue Liu, Hua Huang, Yongtian Wang, and Weitao
Song. Lensless efficient snapshot hyperspectral imaging us-
ing dynamic phase modulation. Photonics Research, 13(11):
511, 2025. 3
[46] Shipeng Zhang, Lizhi Wang, Lei Zhang, and Hua Huang.
Learning tensor low-rank prior for hyperspectral image re-
construction. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 12006–
12015, 2021. 2
[47] Yaqiong Zhang, Yongming Xu, Wencheng Xiong, Ran Qu,
Jiahua Ten, Qijia Lou, and Na Lv. Inversion study of heavy
metals in soils of potentially polluted sites based on uav hy-
perspectral data and machine learning algorithms. In 2021
11th Workshop on Hyperspectral Imaging and Signal Pro-
cessing: Evolution in Remote Sensing (WHISPERS), pages
1–5, 2021. 1
