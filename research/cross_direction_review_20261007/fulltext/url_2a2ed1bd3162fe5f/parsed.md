<!-- PDF PAGE 1 -->

Enpowering Your Pansharpening Models with Generalizability:
Unified Distribution is All You Need
Yongchuan Cui 1, 2 Peng Liu 1, 2, * Hui Zhang 3, *
1 Aerospace Information Research Institute, Chinese Academy of Sciences, 100094 Beijing, China
2 School of Electronic, Electrical and Communication Engineering,
University of Chinese Academy of Sciences, 101408 Beijing, China
3 School of Engineering Medicine, Beihang University, 100191 Beijing, China
yongchuancui@gmail.com, liupeng202303@aircas.ac.cn, hui.zhang@buaa.edu.cn
Abstract
Existing deep learning-based models for remote sensing
pansharpening exhibit exceptional performance on train-
ing datasets. However, due to sensor-specific characteristics
and varying imaging conditions, these models suffer from
substantial performance degradation when applied to un-
seen satellite data, lacking generalizability and thus limiting
their applicability. We argue that the performance drops
stem primarily from distributional discrepancies from dif-
ferent sources and the key to addressing this challenge lies
in bridging the gap between training and testing distribu-
tions. To validate the idea and further achieve a “train once,
deploy forever” capability, this paper introduces a novel
and intuitive approach to enpower any pansharpening mod-
els with generalizability by employing a unified distribution
strategy (UniPAN). Specifically, we construct a distribution
transformation function that normalizes the pixels sampled
from different sources to conform to an identical distribution.
The deep models are trained on the transformed domain,
and during testing on new datasets, the new data are also
transformed to match the training distribution. UniPAN aims
to train and test the model on a unified and consistent dis-
tribution, thereby enhancing its generalizability. Extensive
experiments validate the efficacy of UniPAN, demonstrat-
ing its potential to significantly enhance the performance of
deep pansharpening models across diverse satellite sensors.
Codes: https://github.com/yc-cui/UniPAN .
1. Introduction
Due to the inherent limitations of sensor imaging technol-
ogy, advanced remote sensing satellites such as IKONOS,
QuickBird, GaoFen-2, and WorldView-3, necessitate a com-
promise between the acquisition of extensive spectral infor-
*Corresponding author.
0.714 
0.756 
0.840 
0.917 
0.903 
0.938 
0.815 
0.956 
0.931 
0.934 
0.919 
0.972 
0.600.700.800.901.00IKQBGF1WV2WV3WV4
QNR
w/ UniPANw/o UniPAN
↑
0.1001 
0.0629 
0.0479 
0.0201 
0.0154 
0.0194 
0.0765 
0.0108 
0.0273 
0.0185 
0.0153 
0.0098 
0.000.030.060.09IKQBGF1WV2WV3WV4w/ UniPANw/o UniPAN
𝑫𝝀↓
0.2097 
0.1941 
0.1194 
0.0643 
0.0828 
0.0435 
0.1208 
0.0335 
0.0442 
0.0491 
0.0672 
0.0188 
0.000.060.120.18IKQBGF1WV2WV3WV4w/ UniPANw/o UniPAN
𝑫𝑺↓
Figure 1. Comparison of generalization metrics w/ and w/o UniPAN
trained on WorldView-3 (WV3) using FusionNet [19].
IKONOSQuickBirdGaofen-1WorldView-2WorldView-3WorldView-4
tSNE-1tSNE-2(i) w/o UniPAN(ii) w/ UniPANtSNE-1tSNE-2
Figure 2. Comparison of tSNE visualization w/ and w/o UniPAN.
mation and maintaining fine spatial details in their captured
imagery [20, 49, 68]. Consequently, these satellites typically
capture either high spatial resolution single-band panchro-
matic (PAN) images or low spatial resolution multispectral
(LRMS) images. Pansharpening is a crucial technique in re-
mote sensing image processing, aiming to combine PAN and
LRMS images to generate fused images with high spatial
resolution and abundant spectra (HRMS).
Traditional pansharpening methods have been exten-
sively studied over the past few decades. However, these
This ICCV paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
11850


<!-- PDF PAGE 2 -->

methods may introduce spectral distortions, require sig-
nificant computational resources, and are sensitive to the
quality of input data [ 20, 42, 49, 68]. In contrast, deep
learning (DL)-based methods have shown great potential
in pansharpening. Notably, various innovative architec-
tures have emerged, starting from early convolutional mod-
els [19, 47, 82, 83], progressing to Transformer-based ap-
proaches [8, 21, 22, 24, 50, 62, 65], then generative meth-
ods [43, 46, 48, 58, 78], and continuing with the emergence
of the latest novel designs [9, 23, 31, 53, 86]. The networks
are evolving towards larger parameters capacity, stronger
representation ability, and increasingly complex and diverse
structures.
However, as deep pansharpening thrives, the overly pow-
erful fitting capabilities of these networks have concurrently
presented several ongoing concerns, e.g. generalizability, in-
terpretability, over-reliance on high-quality data, etc. This
paper specifically delves into the widely criticized issue of
generalizability. Specifically, although deep models achieve
excellent performance on training datasets, they often fail to
generalize well to new data due to distribution shift such as
different noise levels, varying spectral signatures and sensor
conditioning characteristics. While models can achieve satis-
factory performances on new datasets through retraining or
fine-tuning, the process is time-consuming and fine-tuning
on every new dataset is impractical and cumbersome. The
ideal scenario would involve developing a pansharpening
model that can be trained once and deployed effectively
across diverse datasets and sensors, thereby providing atrain
once, deploy forever capability.
To this end, this paper introduces a novel approach that
aims to enpower any pansharpening models with generaliz-
ability by employing a unified distribution strategy (dubbed
as UniPAN). The proposed method aims to bridge the gap
between training and testing distributions, thereby improving
the model’s ability to generalize to unseen data. The core
idea is to normalize the distribution of spatial pixels from
the training dataset to conform to a predefined distribution
(e.g. Gaussian distribution) and deep models are then trained
on the transformed domain. This process is model-agnostic
and requires no additional trainable parameters. When gen-
eralizing to new data, testing datasets are also transformed
to match the training distribution, ensuring that the mod-
els are evaluated under the same distributional assumptions.
Fig. 1 presents the cross-satellite generalization performance
of a well-known pansharpening model FusionNet [ 19] on
NBUPansharpRSData [49] trained w/ and w/o UniPAN. The
model utilizing UniPAN demonstrates a significant improve-
ment on full-resolution metrics, which validates the strong
generalization ability for real-world scenes. To further elu-
cidate the efficacy of UniPAN, Fig. 2 presents tSNE [ 64]
dimensionality reduction clustering of multispectral images
from diverse satellites using NBUPansharpRSData [49] w/
and w/o UniPAN. The pre-UniPAN visualization reveals
distinct clustering patterns, with data points from different
satellites forming separate groups, indicative of substantial
inter-satellite distributional disparities. In contrast, follow-
ing the application of UniPAN, a notable transformation is
observed: the previously separated clusters become substan-
tially intermingled, demonstrating a significant reduction in
distributional discrepancies across different satellite datasets
(e.g. the purple scatters of QuickBird). This visual evidence
underscores UniPAN’s capability to harmonize heteroge-
neous satellite data, effectively creating a more cohesive and
unified representation.
The contributions of this paper could be summarized as
follows:
• We propose UniPAN, a model-agnostic approach that
employs a unified distribution strategy to enpower any
deep pansharpening models with generalizability, achiev-
ing a train once, deploy forever capability. This strategy
provides two advantages: 1) Architectural independence:
Compatible with any existing or future network archi-
tecture without structural modifications. 2) Training effi-
ciency: Eliminates the need for retraining or fine-tuning
when adapting to new sensors.
• UniPAN is implemented via inverse transform sampling,
and we prove its equivalence to the optimal solution of
optimal transport in the one-dimensional case. UniPAN
requires no additional trainable parameters and only needs
to be fitted once before training begins, introducing almost
no additional computational overhead. It serves as a viable
alternative to vanilla data normalization techniques such
as min-max and z-score normalization, offering a more
robust and unified approach to data distribution alignment.
• We conduct extensive experiments using a variety of deep
pansharpening models across numerous satellites. The
simple yet intuitive and effective strategy provides a uni-
versal solution to the generalization challenge in DL-based
pansharpening and experimental results demonstrate its ef-
fectiveness, showcasing significant improvements in cross-
dataset generalization across a wide range of models and
satellite sensors.
2. Related Work
2.1. Pansharpening Methods
Traditional Methods. Traditional pansharpening methods
can be broadly categorized into three paradigms: component
substitution (CS), multi-resolution analysis (MRA), and vari-
ational optimization (VO) [20, 42, 49, 68]. CS-based meth-
ods [4, 10, 27, 38, 55, 61] operate by substituting the spatial
component of the LRMS image with the PAN image. Rep-
resentative algorithms include Gram-Schmidt (GS) [4, 38],
principal component analysis (PCA) [10, 61], and intensity-
hue-saturation (IHS) [27, 55]. While computationally effi-
11851


<!-- PDF PAGE 3 -->

cient, CS methods often introduce spectral distortions due
to mismatches between PAN and substituted components.
MRA-based methods [1–3, 13, 26, 41] employ multi-scale
decomposition tools (e.g. wavelets [26], contourlets [13]) to
inject high-frequency details from PAN into LRMS. Typical
implementations include MTF-GLP [1–3] and SFIM [41].
These methods generally preserve spectral fidelity better
than CS approaches but may exhibit limited spatial enhance-
ment. VO-based approaches [7, 52, 63, 67, 74, 75] formulate
pansharpening as an optimization problem with handcrafted
priors, such as the variational wavelet-based methods [7] and
Bayesian approaches [52]. Despite their mathematical ele-
gance, traditional methods often require manual parameter
tuning and struggle with complex spatial-spectral relation-
ships [20, 68].
Deep Learning-Based Methods. The advent of deep learn-
ing (DL) have revolutionized pansharpening by enabling
end-to-end learning of spatial-spectral relationships. Early
pioneering convolutional neural networks (CNNs) such as
PNN [47], PanNet [82], and MSDCNN [83] established the
feasibility of learning fusion rules from a mount of simu-
lated data. Subsequent innovations introduced multi-scale
architectures [33, 72, 76], attention mechanisms [ 39, 73],
and kernel custimizing [12, 18, 25, 34, 85], etc. The CNN-
based methods demonstrated superior performance over tra-
ditional approaches but faced limitations in modeling long-
range dependencies. In contrast, Transformer [ 24, 45, 65]
addressed this limitation with self-attention mechanisms.
Although various Transformer architectures specifically de-
signed for pansharpening [8, 21, 22, 50, 62] have achieved
impressive performance, they face criticism for quadratic
computational complexity w.r.t. sequence length [ 16]. To
address this limitation, recent studies have explored linear-
complexity models [ 16, 31, 53, 86] that maintain perfor-
mance while improving efficiency. To further enhance fi-
delity preservation, generative models like generative ad-
versarial networks (GANs) [ 17, 43, 44, 46, 79] and diffu-
sion models [48, 58, 77, 78, 88] have emerged as promising
solutions. Beyond conventional configurations, contempo-
rary research has expanded into more challenging frontiers
to address practical constraints, including zero-shot learn-
ing paradigms [ 9, 23, 57, 71], frequency domain integra-
tion [30, 40, 84, 89, 90], and interpretable fusion mecha-
nisms [32, 70, 80, 81].
2.2. Generalization in DL-Based Pansharpening
Despite remarkable progress in DL-based pansharpening,
generalization capability remains a persistent challenge. The
issue manifests conspicuously in two critical scenarios: 1)
Models trained under the Wald protocol [69] using spatially
degraded data often fail to maintain performance when ap-
plied to full-resolution inference, frequently exhibiting se-
vere spectral distortions. 2) Networks optimized for specific
satellite sensors demonstrate substantial performance degra-
dation when deployed on unseen satellite platforms due to
distribution shifts in spectral signatures and sensor charac-
teristics. Existing approaches to address these challenges
primarily focus on unsupervised learning [14, 15, 46], gen-
eralized module design [ 77], or advanced data augmenta-
tion [11]. However, such methods typically require either
meticulous architectural redesign or computationally inten-
sive retraining/fine-tuning procedures. These requirements
pose significant practical limitations for real-world deploy-
ment.
In contrast to model-centric solutions, our approach
adopts a fundamentally different perspective by addressing
the generalization challenge through data-space interven-
tion. The proposed UniPAN framework introduces a model-
agnostic preprocessing technique that operates on raw sensor
data. Compared to traditional linear data normalization such
as min-max scaling or z-score standardization, our method
employs a non-linear mapping to project the input data onto a
unified distribution, effectively harmonizing the distribution
mismatches from all sensors.
3. Method
3.1. Distribution Transformation
The core challenge in cross-domain pansharpening stems
from the distribution discrepancy between source (training)
and target (testing) domains. Let XS denote the source do-
main (training data) and XT the target domain (testing data).
Our objective is to learn a transformation T : X → Z
that maps both domains to a shared space Z with unified
distribution ρt, e.g. standard normal distribution.
Inverse Transform Sampling.Given a univariate r.v.X ∼
ρs, we seek a transformation T : R → R such that
Z = T (X) ∼ ρt. The inverse transform sampling theo-
rem provides a theoretically grounded solution:
T (x) = F −1
T (FS(x)), (1)
where FS(x) = ρs(−∞, x] is the cumulative distribution
function (CDF) of X, and F −1
T (u) = inf{z ∈ R : FT (z) ≥
u} is the inverse CDF of the target distribution. By the uni-
versality of the uniform, U = FS(X) follows a uniform
distribution, and Z = F −1
T (U) inherits ρt exactly when FS
is continuous. To provide more intuitively analyze of the al-
gorithm’s operational process and efficiency, we establish a
fundamental connection between inverse transform sampling
and the well-studied optimal transport theory [51, 54].
Optimal Transport Perspective. Given two probability dis-
tributions ρs (source domain) and ρt (target domain) defined
on Rn and Rm respectively, we consider the Monge prob-
lem [51, 54]:
C(ρs, ρt) ≜ min
T : Rn → Rm
T♯ρs = ρt
Z
Rn
c(x, T(x))ρs(x) dx, (2)
11852


<!-- PDF PAGE 4 -->

where c : Ω s × Ωt → R+ is a cost function and for any
measurable Borel set E ∈ Ωs, T : Ω s → Ωt transfers
measures from one space Ωs to another space Ωt,
T♯ρs(E) = ρs(T −1(E)), (3)
here T♯ is the pushforward operator. The Monge problem
seeks a transport plan T from ρs to ρt that minimizes the
transportation cost. The optimal T ∗ is also known as the
Monge map of Eq. (2).
Let FS(x) = ρs(−∞, x] and FT (z) = ρt(−∞, z] de-
note the CDFs of the source and target distributions respec-
tively. In the univariate setting (n = m = 1, Ωs = Ωt = R),
when the source measure ρs is continuous and the cost func-
tion takes the form c(x, z) = h(|x − z|) with h being strictly
convex, by cyclical monotonicity [66], the Monge map T ∗
satisfies:
FS(x) = FT (T ∗(x)), (4)
yielding the explicit expression:
T ∗(x) = F −1
T (FS(x)). (5)
This remarkable result coincides exactly with the inverse
transform sampling formula Eq. (1), i.e. T ∗ = T . The trans-
port cost in this setting [ 37, 56, 59] can be computed as
C(ρs, ρt) =
Z 1
0
c(F −1
S (u), F −1
T (u))du. (6)
For the L1 cost function c(x, z) = |x − z| , this corre-
sponds to the W1 Wasserstein distance [ 36, 59] between
the distributions. In the one-dimensional (1D) scenario, Uni-
PAN operates independently on spatial pixels within each
spectral band. This implementation can be achieved through
interpolation-based inverse transform sampling without fram-
ing it as an optimization problem. However, given the in-
herent correlations between spectral bands and interdepen-
dencies among spatial patches, multidimensional extension
of this framework may be promising, albeit with increased
implementation complexity. The theoretical derivation es-
tablishing connections with optimal transport theory serves
to facilitate future generalization of UniPAN to multidimen-
sional configurations. Building upon the theoretical founda-
tion of distribution transformation through optimal transport,
in the following, we present a practical implementation of
1D UniPAN.
3.2. UniPAN Implementation
We present a practical implementation for 1D finite discrete
distributions of UniPAN through a quantile-based distribu-
tion transformation algorithm.
Finite Discrete Distribution Transformation.Given pixel
values X = {x1, x2, ..., xn} sampled from a spectral band,
Algorithm 1 UniPAN Training and Testing Procedure
Training Phase
Input: Training dataset Dtrain = {Li, Pi, Hi}N
i=1, inverse CDF
F −1
T of target distribution ρt, model M
Output: Trained model M
1: Initialize transformed LRMS ˜L and PAN ˜P
2: for each band b of LRMS {Li}N
i=1 & PAN {Pi}N
i=1 do
3: Sample n pixels Xb ← {x1, . . . , xn} from band b
4: Compute m quantiles Q = {q0, . . . , qm−1} ▷ Eq. (7)
5: Linspace uniform CDF U = {0, 1
m−1 , ..., 1}
6: for each pixel x in band b do
7: Forward and backward interp by u↑ and u↓ ▷ Eq. (9)
8: Compute eCDF value u(x) ← (u↑ + u↓) / 2 ▷ Eq. (8)
9: Transform x to z ← F −1
T (u(x)) and assign z to cor-
responding band of ˜L and ˜P ▷ Eq. (11)
10: end for
11: end for
12: while not converged do
13: Train M with L = ∥H − M( ˜L, ˜P ) + L ↑ ∥ 1
14: end while
15: return Trained model M
Testing Phase
Input: Testing dataset Dtest = {Li, Pi}N
i=1, inverse CDF F −1
T of
target distribution ρt, trained model M
Output: Pansharpening result ˆHtest
16: for each band b of LRMS & PAN do
17: Transform L, P to ˜L, ˜P using the training transformation
18: end for
19: Generate ˆHtest ← M( ˜L, ˜P ) + L ↑
20: return Fused result ˆHtest
we first compute m empirical quantiles Q = {q0, ..., qm−1}
where:
qi = inf

x ∈ X
P (X ≤ x) ≥ i
m − 1

, i = {0, 1, ..., m − 1}.
(7)
These quantiles discretize the empirical cumulative distri-
bution function (eCDF) of the source data. The number of
quantiles m controls the approximation fidelity.
Given any data point x to be transformed, the next step
involves determining its value in the eCDF through inter-
polation. However, when the source distribution contains
repeated values or sparse regions, quantile estimation may
yield non-unique or overlapping quantile boundaries. This
can result in abrupt jumps in the eCDF, thereby violating
the properties of monotonicity and smoothness. Therefore,
we use bidirectional interpolation to estimate a robust eCDF
value of x:
u(x) = u↑(x)/2 + u↓(x)/2
= interp(x, Q, U) / 2 +
(1 − interp (−x, −Qrev, −U rev)) / 2,
(8)
where U = {0, 1
m−1 , ...,1}, Qrev is the reversed quantiles.
11853


<!-- PDF PAGE 5 -->

QNR↑ ρt=N(0,1) ρt=U(0,1)
GF1 IK QB WV2 WV3 WV4 GF1 IK QB WV2 WV3 WV4
GF1 0.980(+0.005)0.946(−0.010)0.971(+0.000)0.963(−0.003)0.964(−0.004)0.974(−0.000)0.985(+0.010)0.961(+0.004)0.979(+0.007)0.984(+0.017)0.982(+0.013)0.987(+0.013)IK 0.950(+0.005)0.912(+0.039)0.915(+0.045)0.924(+0.011)0.916(+0.022)0.958(+0.011)0.956(+0.011)0.919(+0.045)0.940(+0.070)0.950(+0.037)0.945(+0.051)0.968(+0.021)QB 0.957(+0.006)0.936(+0.016)0.931(+0.047)0.953(+0.017)0.944(+0.014)0.970(+0.016)0.965(+0.014)0.944(+0.024)0.966(+0.082)0.969(+0.032)0.969(+0.038)0.970(+0.017)WV20.880(+0.026)0.811(+0.029)0.832(+0.062)0.946(+0.008)0.916(+0.012)0.951(+0.012)0.872(+0.018)0.775(−0.006)0.825(+0.055)0.944(+0.006)0.914(+0.011)0.940(+0.001)WV30.862(−0.006)0.804(+0.035)0.795(+0.067)0.921(−0.011)0.919(+0.011)0.950(+0.001)0.880(+0.011)0.805(+0.036)0.813(+0.084)0.943(+0.010)0.924(+0.016)0.952(+0.003)WV40.924(+0.009)0.898(+0.045)0.875(+0.036)0.948(−0.000)0.929(+0.003)0.972(+0.006)0.939(+0.024)0.889(+0.035)0.903(+0.064)0.952(+0.002)0.935(+0.009)0.973(+0.007)
Dλ↓ ρt=N(0,1) ρt=U(0,1)
GF1 IK QB WV2 WV3 WV4 GF1 IK QB WV2 WV3 WV4
GF1 0.007(−0.003)0.022(+0.008)0.005(−0.000)0.017(+0.004)0.015(+0.005)0.011(+0.000)0.004(−0.005)0.012(−0.001)0.004(−0.001)0.004(−0.008)0.004(−0.005)0.004(−0.005)IK 0.011(−0.000)0.020(−0.015)0.018(−0.011)0.031(+0.004)0.022(−0.003)0.014(−0.003)0.008(−0.003)0.020(−0.015)0.013(−0.016)0.013(−0.014)0.009(−0.016)0.010(−0.008)QB 0.013(−0.000)0.011(−0.004)0.015(−0.010)0.016(−0.004)0.014(+0.000)0.010(−0.005)0.009(−0.003)0.009(−0.006)0.007(−0.017)0.008(−0.013)0.006(−0.007)0.012(−0.003)WV20.042(−0.006)0.066(−0.007)0.037(−0.019)0.008(−0.000)0.014(−0.000)0.016(−0.002)0.046(−0.002)0.086(+0.012)0.048(−0.008)0.008(+0.000)0.015(+0.001)0.022(+0.002)WV30.050(+0.010)0.067(−0.011)0.049(−0.024)0.027(+0.011)0.014(+0.000)0.016(+0.002)0.042(+0.002)0.072(−0.006)0.051(−0.022)0.013(−0.002)0.011(−0.002)0.014(+0.000)WV40.024(−0.000)0.031(−0.012)0.029(−0.009)0.017(+0.008)0.018(+0.008)0.005(−0.001)0.016(−0.008)0.032(−0.011)0.022(−0.016)0.014(+0.004)0.015(+0.005)0.004(−0.002)
DS↓ ρt=N(0,1) ρt=U(0,1)
GF1 IK QB WV2 WV3 WV4 GF1 IK QB WV2 WV3 WV4
GF1 0.012(−0.003)0.032(+0.003)0.023(+0.000)0.020(+0.000)0.021(+0.000)0.015(−0.000)0.010(−0.005)0.026(−0.002)0.016(−0.006)0.010(−0.009)0.013(−0.007)0.007(−0.008)IK 0.039(−0.005)0.068(−0.026)0.068(−0.036)0.046(−0.014)0.063(−0.019)0.027(−0.007)0.035(−0.008)0.062(−0.033)0.046(−0.058)0.037(−0.024)0.045(−0.036)0.022(−0.013)QB 0.030(−0.005)0.053(−0.012)0.054(−0.039)0.031(−0.012)0.042(−0.014)0.020(−0.012)0.025(−0.011)0.046(−0.018)0.026(−0.067)0.023(−0.020)0.024(−0.031)0.020(−0.012)WV20.081(−0.022)0.134(−0.024)0.137(−0.049)0.045(−0.008)0.071(−0.012)0.032(−0.010)0.085(−0.018)0.153(−0.005)0.133(−0.053)0.047(−0.006)0.071(−0.012)0.039(−0.003)WV30.096(−0.000)0.141(−0.027)0.167(−0.050)0.054(+0.002)0.067(−0.011)0.033(−0.004)0.083(−0.013)0.134(−0.033)0.146(−0.072)0.043(−0.008)0.064(−0.014)0.033(−0.003)WV40.053(−0.009)0.073(−0.036)0.099(−0.029)0.035(−0.006)0.053(−0.011)0.021(−0.005)0.045(−0.018)0.082(−0.027)0.076(−0.052)0.036(−0.004)0.052(−0.012)0.021(−0.005)
Dρ↓ ρt=N(0,1) ρt=U(0,1)
GF1 IK QB WV2 WV3 WV4 GF1 IK QB WV2 WV3 WV4
GF1 0.043(+0.003)0.110(+0.030)0.026(−0.008)0.079(+0.032)0.062(+0.015)0.041(−0.002)0.019(−0.020)0.059(−0.021)0.022(−0.013)0.015(−0.032)0.017(−0.029)0.017(−0.026)IK 0.082(−0.006)0.095(−0.038)0.090(−0.058)0.126(+0.024)0.110(−0.010)0.061(−0.008)0.064(−0.024)0.092(−0.040)0.066(−0.083)0.051(−0.049)0.061(−0.059)0.045(−0.024)QB 0.061(−0.003)0.054(−0.020)0.075(−0.038)0.052(−0.026)0.060(−0.019)0.044(−0.017)0.036(−0.028)0.048(−0.026)0.029(−0.083)0.021(−0.057)0.023(−0.056)0.028(−0.034)WV20.166(−0.027)0.198(−0.039)0.199(−0.059)0.115(+0.001)0.130(−0.045)0.093(−0.026)0.164(−0.029)0.235(−0.002)0.200(−0.058)0.100(−0.013)0.126(−0.048)0.112(−0.007)WV30.205(+0.018)0.214(−0.041)0.234(−0.053)0.231(+0.109)0.173(+0.011)0.105(−0.015)0.176(−0.010)0.226(−0.029)0.219(−0.068)0.117(−0.004)0.131(−0.030)0.101(−0.018)WV40.105(−0.001)0.112(−0.042)0.115(−0.044)0.146(+0.080)0.169(+0.077)0.038(−0.014)0.072(−0.034)0.118(−0.035)0.105(−0.055)0.053(−0.012)0.062(−0.029)0.035(−0.017)
Table 1. Average results of12 models. Rows correspond to training satellites, and columns correspond to testing satellites. Green indicates
performance improvement, while red indicates performance degradation. The values in parentheses represent the average delta compared to
the baseline without UniPAN.
The function interp represents a linear interpolation be-
tween the quantile point Q and the reference point U, with
the interpolation formula given by:
interp(x, Q, U) =



U[0], x ≤ Q[0]
f(x, i, Q, U), Q [i] < x ≤ Q[i + 1]
U[m − 1], x > Q [m − 1]
(9)
where f computes the Q to U mapping,
f(x, i, Q, U) = U[i] + x − Q[i]
Q[i + 1] − Q[i](U[i + 1] − U[i])
(10)
The last step is to apply inverse CDF of target distribution,
T (x) = F −1
T (u(x)) (11)
Training and Testing Procedure.The training and testing
procedures of UniPAN are described in Algorithm 1. Here,
L ↑ denotes the upsampled LRMS image using bicubic in-
terpolation to match the size of the PAN image. The model
M is trained to predict the residual of LRMS with an l1
loss function that minimizes the discrepancy between the
predicted HRMS image and the ground truth.
As for the target distribution ρt, we test two widely used
distributions, i.e. standard normal distribution N (0, 1) and
standard uniform distribution U(0, 1), the transformation
T (x) becomes:
Tn(x) = Φ−1(FS(x)), Tu(x) = FS(x), (12)
where Φ−1 denotes the inverse CDF of standard normal
distribution. Since the inverse CDF of standard uniform dis-
tribution is an identical mapping, Tu becomes the estimated
empirical distribution itself. The standard normal distribu-
tion transformation non-linearly warps the input space to
match Gaussian statistics, while the uniform transformation
directly maps pixel intensities to their empirical quantiles.
4. Experiment
4.1. Setups
Datasets. We conducted experiments utilizing the Red,
Green, Blue, and NIR bands of the NBUPansharpRSData
dataset [49]. The variations in resolution among different
satellites, coupled with significant differences in land covers
and land uses, render this dataset particularly suitable for
assessing the generalization capabilities of pansharpening
models. The dataset comprises data from 6 satellites. For
each satellite, we partitioned the data into training, validation,
and testing sets in a ratio of 7 : 1 : 2.
Implementation Details. To validate the effectiveness
and generalization capability of UniPAN, we conducted
comparative experiments between baseline (w/o UniPAN)
and enhanced (w/ UniPAN) configurations using diverse
commonly-recognized models, including FusionNet [ 19],
PanNet [ 82], MSDDN [ 28, 29], GPPNN [ 80], MSD-
CNN [ 83], FeINFN [ 40], PNN [ 47], LAGConv [ 34],
SFIIN [89, 90], PreMix [ 18], UAPN [ 87], MDCUN [ 81].
11854


<!-- PDF PAGE 6 -->

LRMS
w/o UniPAN
PanNet
 MSDDN
 GPPNN
 MSDCNN
 PreMix
 FeINFN
 LAGConv
 SFIIN
 UAPN
 MDCUN
0.7191
0.0655/0.2305
0.6601
0.0968/0.2691
0.7636
0.0500/0.1962
0.7248
0.0616/0.2276
0.7443
0.0562/0.2113
0.6866
0.0753/0.2575
0.6293
0.0990/0.3015
0.6315
0.0878/0.3077
0.6717
0.0723/0.2760
0.7157
0.0713/0.2294
ρt = N(0, 1) 
 w/ UniPAN 
0.7006
0.0640/0.2515
0.7362
0.0414/0.2320
0.7506
0.0824/0.1819
0.6813
0.0773/0.2616
0.6909
0.0724/0.2551
0.6964
0.0735/0.2484
0.6571
0.0787/0.2867
0.6112
0.1068/0.3158
0.7309
0.0465/0.2335
0.8671
0.0227/0.1128
ρt = U(0, 1) 
 w/ UniPAN
0.7336
0.0578/0.2214
0.7164
0.0581/0.2394
0.7885
0.0655/0.1562
0.6965
0.0705/0.2507
0.7031
0.1037/0.2155
0.7153
0.0666/0.2337
0.6616
0.0833/0.2783
0.6538
0.0837/0.2865
0.7791
0.0306/0.1963
0.7290
0.0636/0.2215
Figure 3. Generalization testing on GaoFen-1, the networks are trained on WorldView-3. The bright green box indicates the zoomed-in area.
Below the zoomed-in image, the first line is the QNR value, and the second line is the Dλ/DS value. In most cases, the use of UniPAN
yields superior spectral and spatial preservation. For instance, MDCUN [81] exhibits a highly close alignment with the LRMS in spectral
characteristics with UniPAN of ρt = N (0, 1).
All implementations followed identical experimental pro-
cedures: a batch size of 8 (except for MDCUN [ 81] and
FeINFN [ 40] which used 4), the Adam [ 35] optimizer
(β1 = 0.9, β2 = 0.999) with an initial learning rate of 10−3,
and StepLR scheduling which reduces the learning rate by a
factor of 0.8 every 100 epochs. Training was performed on
an NVIDIA GeForce RTX 4090 GPU for 600 epochs using
the Wald protocol [69] with l1 loss, with best model selected
based on optimal validation performance. Note that, to pre-
vent data leakage of testing set, the transformation fitting
process is conducted on the training set of each satellite.
Evaluation Metrics. The full-reference metrics based on
spatial degradation (e.g. PSNR, SSIM) can only reflect per-
formance in simulated reduced-resolution experiments and
fail to demonstrate effectiveness in real-world full-resolution
scenarios [6, 16, 46]. In contrast, no-reference metrics like
QNR explicitly quantify model performance in both spectral
preservation and spatial enhancement, enabling direct eval-
uation of fusion results at native resolution while avoiding
theoretical biases introduced by downsampling. To better
align with practical applications where reference images
are unavailable, this paper adopts widely-used no-reference
spectral and spatial metrics, i.e. QNR [5], Dλ [5], DS [5],
and Dρ [60], to comprehensively assess cross-resolution and
cross-satellite generalization capabilities.
4.2. Quantitative Evaluation
Tab. 1 presents the average generalization performance of
various models enhanced with UniPAN under different target
distributions. Green indicate metric improvements, while red
denote performance degradation. Slight yet observable dif-
ferences exist between the two target distributions. Overall,
the uniform distribution achieves better performance than the
normal distribution. The normal distribution demonstrated a
modest decline in performance across the WV2 and WV3
datasets. This phenomenon may stem from the peaked char-
acteristics of the normal distribution, which could hinder
model training, whereas the uniform distribution provides
smoother normalization (see Fig. 5 in Sec. 4.3).
Notably, UniPAN withρt = U(0, 1) achieves significant
improvements across most metrics, with only marginal de-
clines in rare cases. For instance, when generalizing from
WV3 to QB, UniPAN improves the QNR metric by over
0.084 (from 0.729 to 0.813) averaged on all 12 models, un-
derscoring its effectiveness. Beyond QNR, UniPAN also
enhances spectral preservation (Dλ) and spatial consistency
11855


<!-- PDF PAGE 7 -->

LRMS
w/o UniPAN
PanNet
 MSDDN
 GPPNN
 MSDCNN
 PreMix
 FeINFN
 LAGConv
 SFIIN
 UAPN
 MDCUN
0.8575
0.0333/0.1129
0.8420
0.0603/0.1040
0.9086
0.0261/0.0670
0.8658
0.0286/0.1087
0.8882
0.0281/0.0862
0.8634
0.0329/0.1072
0.8616
0.0340/0.1081
0.8477
0.0350/0.1215
0.8461
0.0362/0.1221
0.8725
0.0266/0.1037
ρt = N(0, 1) 
 w/ UniPAN 
0.8632
0.0306/0.1095
0.8830
0.0279/0.0917
0.8757
0.0313/0.0960
0.8699
0.0276/0.1054
0.9168
0.0100/0.0740
0.8798
0.0264/0.0963
0.8687
0.0289/0.1054
0.8757
0.0286/0.0985
0.8823
0.0239/0.0961
0.8808
0.0245/0.0971
ρt = U(0, 1) 
 w/ UniPAN
0.8782
0.0246/0.0996
0.8909
0.0260/0.0853
0.8916
0.0324/0.0786
0.8996
0.0207/0.0814
0.8975
0.0205/0.0838
0.8934
0.0182/0.0900
0.8829
0.0234/0.0960
0.9060
0.0277/0.0681
0.8925
0.0218/0.0876
0.9207
0.0144/0.0659
Figure 4. Full-resolution generalization testing on QuickBird. The bright green box indicates the zoomed-in area. Below the zoomed-in
image, the first line is the QNR value, and the second line is the Dλ/DS value.
(DS, Dρ), further validating the efficacy of the unified dis-
tribution strategy. These results confirm that aligning het-
erogeneous satellite data to a unified distribution effectively
mitigates domain shifts, enabling robust cross-sensor gener-
alization. However, while UniPAN improves spatial fidelity
greatly, it may cause slight spectral distortion, as evidenced
by the minor performance degradation observed in the Dλ
metric. This is reasonable since the transformation applied
to each spectral band maintains the relative spatial structure
but might result in minimal loss of spectral information. Nev-
ertheless, such information loss is extremely limited, as the
overall enhancement in performance remains predominant.
4.3. Visual Comparison
Comparison of Fusion Results. To illustrate the practical
impact of UniPAN, Figs. 3 and 4 compare fusion results w/
and w/o UniPAN. In most cases, UniPAN achieves highly
favorable results in both spectral and spatial preservation.
Furthermore, the remarkably high QNR values demonstrate
its superior capability in balancing spectral and spatial fi-
delity, e.g. in Fig. 3, MDCUN [81] applying UniPAN with
normal distribution enhanced the QNR by 0.1514 (from
0.7157 to 0.8671) and aligned fused results closely with the
ground truth spectra. In the full-resolution generalization
test depicted in Fig. 4, even with minimal visual differences,
UniPAN achieves metrically significant improvements in
spectral and spatial preservation. This capability is critical
for downstream applications requiring accurate spectral in-
formation. Additional visual comparison results are included
in the supplementary materials.
Comparison of Data Distribution. Fig. 5 further analyzes
data distribution alignment. The kernel density estimation
(KDE) curves (top right area and diagonals) reveal that Uni-
PAN successfully transforms the original complex and het-
erogeneous distributions into a unified target distribution.
The quantile-quantile (Q-Q) plots (bottom left area) demon-
strate harmonized distributions across satellites after apply-
ing UniPAN (The greater the deviation from the y = x line,
the more significant the disparity between the two distribu-
tions). For example, the original Q-Q plots between WV4
and QB show significant dispersion, whereas post-UniPAN
points close tightly around y = x. This visual evidence
corroborates UniPAN’s ability to bridge distribution gaps.
4.4. Parameter Analysis
UniPAN introduces two hyperparameters: the number of
sampled pixels n and the number of quantiles m. Figs. 6
and 7 investigate their impact on performance of Fusion-
Net [19] trained on WV2 and tested on GF1. As n and m
vary, the metrics stabilize. Generally,U(0, 1) demonstrates
11856


<!-- PDF PAGE 8 -->

(a) w/o UniPAN
 (b) w/ UniPAN, ρt = N (0, 1)
 (c) w/ UniPAN, ρt = U (0, 1)
Figure 5. Comparison of data distributions before and after applying UniPAN. The upper-right and diagonals of each subplot display the
kernel density estimation (KDE) curves of Green band for each satellite. The scatter plots in the lower-left areas of each subplots represent
quantile-quantile (Q-Q) plots of green band, with the dashed line indicating y = x. The greater the deviation from the y = x line, the more
significant the disparity between the two distributions. Zoom in for better view.
0.711
0.761
0.811
0.861
0.911
0.961
QNR ↑
0.000
0.010
0.020
0.030
0.040
0.050
Dλ ↓
1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
Number of quantiles
0.000
0.030
0.060
0.090
0.120
0.150
DS ↓
ρt = N(0, 1) ρt = U(0, 1)   w/o UniPAN
Figure 6. Parameter analysis of number of quantiles, with number
of sampled pixels fixed to 10000.
superior performance compared to N (0, 1). Both configu-
rations achieve statistically significant improvements over
baseline methods without applying UniPAN. This parametric
robustness demonstrates that optimal performance can be
attained without exhaustive hyperparameter tuning, signifi-
cantly reducing computational overhead.
5. Conclusion and Discussion
We proposed UniPAN, a unified distribution strategy ad-
dressing generalization challenges in deep learning-based
pansharpening. UniPAN eliminates spectral discrepancies
by normalizing multi-source remote sensing imagery into
predefined target distributions, effectively bridging the gap
between training and testing distributions. Extensive quanti-
tative and visual comparisons across 6 satellite sensors and
0.711
0.761
0.811
0.861
0.911
0.961
QNR ↑
0.000
0.010
0.020
0.030
0.040
0.050
Dλ ↓
10k 11k 12k 13k 14k 15k 16k 17k 18k 19k 20k 30k
Number of sampled pixels
0.000
0.030
0.060
0.090
0.120
0.150
DS ↓
ρt = N(0, 1) ρt = U(0, 1)   w/o UniPAN
Figure 7. Parameter analysis of number of sampled pixels , with
number of quantiles fixed to 1000.
12 models demonstrate that UniPAN significantly enhances
cross-sensor generalization. Notably, this enhancement is
achieved without introducing any additional trainable param-
eters or the need for retraining or fine-tuning the models.
While this work aligns theoretically with 1D optimal
transport, considering interdependencies of spectral bands
or spatial patches, future research could explore extending
the method to the multi-dimensional case with optimal trans-
port theories. Additionally, the experimental results indicate
that the choice of target distribution has a notable impact
on performance, thus it is worthwhile investigating adaptive
target domain selection. To further validate the generalizabil-
ity of the proposed unified distribution strategy, the ideas
presented in this work could be extended to other remote
sensing image processing tasks, such as multi-modal fusion
and hyperspectral pansharpening, etc.
11857


<!-- PDF PAGE 9 -->

Acknowledgment. This research was partially supported
by the National Natural Science Foundation of China un-
der Grant 32441114, 41971397, 61731022, and the project
Y1H103101A.
References
[1] B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli. Context-
driven fusion of high spatial and spectral resolution images
based on oversampled multiresolution analysis. IEEE TGRS,
40(10):2300–2312, 2002. 3
[2] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva.
An MTF-based spectral distortion minimizing model for pan-
sharpening of very high resolution multispectral images of
urban areas. In 2003 2nd GRSS/ISPRS Joint Workshop on
Remote Sensing and Data Fusion over Urban Areas, pages
90–94, 2003.
[3] Bruno Aiazzi, L. Alparone, Stefano Baronti, Andrea Garzelli,
and Massimo Selva. MTF-tailored multiscale fusion of high-
resolution MS and Pan imagery. Photogramm. Eng. Remote
Sens., 72:591–596, 2006. 3
[4] Bruno Aiazzi, Stefano Baronti, and Massimo Selva. Im-
proving component substitution pansharpening through mul-
tivariate regression of MS+Pan data. IEEE TGRS, 45(10):
3230–3239, 2007. 2
[5] Luciano Alparone, Bruno Aiazzi, Stefano Baronti, Andrea
Garzelli, Filippo Nencini, and Massimo Selva. Multispectral
and panchromatic data fusion assessment without reference.
Photogramm. Eng. Remote Sens., 74(2):193–200, 2008. 6
[6] Alberto Arienzo, Gemine Vivone, Andrea Garzelli, Luciano
Alparone, and Jocelyn Chanussot. Full-resolution quality
assessment of pansharpening: Theoretical and hands-on ap-
proaches. IEEE GRSM, 10(3):168–201, 2022. 6
[7] Coloma Ballester, Vicent Caselles, and Laura Igual. A vari-
ational model for P+XS image fusion. IJCV, 69(1):43–58,
2006. 3
[8] Wele Gedara Chaminda Bandara and Vishal M Patel. Hy-
pertransformer: A textural and spectral feature fusion Trans-
former for pansharpening. In CVPR, pages 1767–1777, 2022.
2, 3
[9] Qi Cao, Liang-Jian Deng, Wu Wang, Junming Hou, and Gem-
ine Vivone. Zero-shot semi-supervised learning for pansharp-
ening. Inf. Fusion., 101:102001, 2024. 2, 3
[10] Pat S. Chavez and Andy Yaw Kwarteng. Extracting spectral
contrast in landsat thematic mapper image data using selective
principal component analysis. Photogramm. Eng. Remote
Sens., 55:339–348, 1989. 2
[11] Lihui Chen, Gemine Vivone, Zihao Nie, Jocelyn Chanussot,
and Xiaomin Yang. Spatial data augmentation: Improving the
generalization of neural networks for pansharpening. IEEE
TGRS, 61:1–11, 2023. 3
[12] Zhi-Xuan Chen, Cheng Jin, Tian-Jing Zhang, Xiao Wu, and
Liang-Jian Deng. SpanConv: A new convolution via spanning
kernel space for lightweight pansharpening. In IJCAI, pages
841–847, 2022. Main Track. 3
[13] Minhyeok Choi, Rae Young Kim, and Myeong-Ryong Nam.
Fusion of multispectral and panchromatic satellite images
using the curvelet transform. IEEE GRSL, 2(2):136–140,
2005. 3
[14] Matteo Ciotola, Sergio Vitale, Antonio Mazza, Giovanni
Poggi, and Giuseppe Scarpa. Pansharpening by convolu-
tional neural networks in the full resolution framework. IEEE
TGRS, pages 1–1, 2022. 3
[15] Matteo Ciotola, Giovanni Poggi, and Giuseppe Scarpa. Un-
supervised deep learning-based pansharpening with jointly
enhanced spectral and spatial fidelity. IEEE TGRS, 61:1–17,
2023. 3
[16] Yongchuan Cui, Peng Liu, Yan Ma, Lajiao Chen, Mengzhen
Xu, and Xingyan Guo. Pixel-wise ensembled masked autoen-
coder for multispectral pansharpening. IEEE TGRS, 62:1–22,
2024. 3, 6
[17] Yongchuan Cui, Peng Liu, Bingze Song, Lingjun Zhao, Yan
Ma, and Lajiao Chen. Reconstruction of large-scale missing
data in remote sensing images using Extend-GAN. IEEE
GRSL, 21:1–5, 2024. 3
[18] Yongchuan Cui, Peng Liu, Yan Ma, Lajiao Chen, Mengzhen
Xu, and Xingyan Guo. Pansharpening via predictive filtering
with element-wise feature mixing. ISPRS J. Photogramm.
Remote Sens., 219:22–37, 2025. 3, 5
[19] Liang-Jian Deng, Gemine Vivone, Cheng Jin, and Jocelyn
Chanussot. Detail injection-based deep convolutional neural
networks for pansharpening. IEEE TGRS, 59(8):6995–7010,
2021. 1, 2, 5, 7
[20] Liang-jian Deng, Gemine Vivone, Mercedes E. Paoletti,
Giuseppe Scarpa, Jiang He, Yongjun Zhang, Jocelyn Chanus-
sot, and Antonio Plaza. Machine learning in pansharpening:
A benchmark, from shallow to deep networks. IEEE GRSM,
10(3):279–315, 2022. 1, 2, 3
[21] Shangqi Deng, Liang-Jian Deng, Xiao Wu, Ran Ran, and Rui
Wen. Bidirectional dilation Transformer for multispectral and
hyperspectral image fusion. In IJCAI, 2023. 2, 3
[22] Shang-Qi Deng, Liang-Jian Deng, Xiao Wu, Ran Ran, Dan-
feng Hong, and Gemine Vivone. PSRT: Pyramid shuffle-
and-reshuffle Transformer for multispectral and hyperspectral
image fusion. IEEE TGRS, 61:1–15, 2023. 2, 3
[23] Renwei Dian, Anjing Guo, and Shutao Li. Zero-shot hy-
perspectral sharpening. IEEE TPAMI, 45(10):12650–12666,
2023. 2, 3
[24] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is
worth 16x16 words: Transformers for image recognition at
scale. ICLR, 2021. 2, 3
[25] Yule Duan, Xiao Wu, Haoyu Deng, and Liang-Jian Deng.
Content-adaptive non-local convolution for remote sensing
pansharpening. In CVPR, pages 27738–27747, 2024. 3
[26] Andrea Garzelli and Filippo Nencini. Wavelet-based fusion
of optical and SAR image data in remote sensing applications.
Int. J. Remote Sens., 23(23):5025–5039, 2002. 3
[27] Morteza Ghahremani and Hassan Ghassemian. Nonlinear
IHS: A promising method for pan-sharpening. IEEE GRSL,
13(11):1606–1610, 2016. 2
11858


<!-- PDF PAGE 10 -->

[28] Xuanhua He, Keyu Yan, Rui Li, Chengjun Xie, Jie Zhang,
and Man Zhou. Pyramid dual domain injection network for
pan-sharpening. In ICCV, pages 12862–12871, 2023. 5
[29] Xuanhua He, Keyu Yan, Jie Zhang, Rui Li, Chengjun Xie,
Man Zhou, and Danfeng Hong. Multiscale dual-domain guid-
ance network for pan-sharpening. IEEE TGRS, 61:1–13, 2023.
5
[30] Xuanhua He, Keyu Yan, Rui Li, Chengjun Xie, Jie Zhang, and
Man Zhou. Frequency-adaptive pan-sharpening with mixture
of experts. AAAI, 38(3):2121–2129, 2024. 3
[31] Xuanhua He, Ke Cao, Jie Zhang, Keyu Yan, Yingying Wang,
Rui Li, Chengjun Xie, Danfeng Hong, and Man Zhou. Pan-
Mamba: Effective pan-sharpening with state space model. Inf.
Fusion., 115:102779, 2025. 2, 3
[32] Junming Hou, Xinyang Liu, Chenxu Wu, Xiaofeng Cong,
Chentong Huang, Liang-Jian Deng, and Jian Wei You. Bido-
main uncertainty gated recursive network for pan-sharpening.
Inf. Fusion., 118:102938, 2025. 3
[33] Cheng Jin, Liang-Jian Deng, Ting-Zhu Huang, and Gemine
Vivone. Laplacian pyramid networks: A new approach for
multispectral pansharpening. Inf. Fusion., 78:158–170, 2022.
3
[34] Zi-Rong Jin, Tian-Jing Zhang, Tai-Xiang Jiang, Gemine
Vivone, and Liang-Jian Deng. LAGConv: Local-context adap-
tive convolution kernels with global harmonic bias for pan-
sharpening. AAAI, 36(1):1113–1121, 2022. 3, 5
[35] Diederik P Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. arXiv preprint arXiv:1412.6980 ,
2014. 6
[36] Soheil Kolouri, Se Rim Park, Matthew Thorpe, Dejan Slepcev,
and Gustavo K. Rohde. Optimal mass transport: Signal pro-
cessing and machine-learning applications. IEEE SPM, 34
(4):43–59, 2017. 4
[37] Soheil Kolouri, Kimia Nadjahi, Umut Simsekli, Roland
Badeau, and Gustavo K. Rohde. Generalized sliced wasser-
stein distances, 2019. 4
[38] Craig A Laben and Bernard V Brower. Process for enhancing
the spatial resolution of multispectral imagery using pan-
sharpening. 2000. US Patent 6,011,875. 2
[39] Zhixuan Li, Jinjiang Li, Fan Zhang, and Linwei Fan. CADUI:
Cross-attention-based depth unfolding iteration network for
pansharpening remote sensing images. IEEE TGRS, 61:1–20,
2023. 3
[40] YuJie Liang, ZiHan Cao, Shangqi Deng, Hong-Xia Dou, and
Liang-Jian Deng. Fourier-enhanced implicit neural fusion
network for multispectral and hyperspectral image fusion. In
NeurIPS, pages 63441–63465. Curran Associates, Inc., 2024.
3, 5, 6
[41] J. G. Liu. Smoothing filter-based intensity modulation: A
spectral preserve image fusion technique for improving spatial
details. Int. J. Remote Sens., 21(18):3461–3472, 2000. 3
[42] Peng Liu, Jun Li, Lizhe Wang, and Guojin He. Remote sens-
ing data fusion with generative adversarial networks: State-of-
the-art methods and future research directions. IEEE GRSM,
10(2):295–328, 2022. 2
[43] Qingjie Liu, Huanyu Zhou, Qizhi Xu, Xiangyu Liu, and Yun-
hong Wang. PSGAN: A generative adversarial network for
remote sensing image pan-sharpening. IEEE TGRS, 59(12):
10227–10242, 2020. 2, 3
[44] Xiaobo Liu, Xiang Liu, Haoran Dai, Xudong Kang, Antonio
Plaza, and Wenjie Zu. Mun-GAN: a multiscale unsupervised
network for remote sensing image pansharpening. IEEE
TGRS, 61:1–18, 2023. 3
[45] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng
Zhang, Stephen Lin, and Baining Guo. Swin Transformer:
Hierarchical vision Transformer using shifted windows. In
ICCV, 2021. 3
[46] Jiayi Ma, Wei Yu, Chen Chen, Pengwei Liang, Xiaojie Guo,
and Junjun Jiang. Pan-GAN: An unsupervised pan-sharpening
method for remote sensing image fusion. Inf. Fusion., 62:
110–120, 2020. 2, 3, 6
[47] Giuseppe Masi, Davide Cozzolino, Luisa Verdoliva, and
Giuseppe Scarpa. Pansharpening by convolutional neural
networks. In Remote Sens., page 594, 2016. 2, 3, 5
[48] Qingyan Meng, Wenxu Shi, Sijia Li, and Linlin Zhang. Pan-
Diff: A novel pansharpening method based on denoising dif-
fusion probabilistic model. IEEE TGRS, 61:1–17, 2023. 2,
3
[49] Xiangchao Meng, Yiming Xiong, Feng Shao, Huanfeng Shen,
Weiwei Sun, Gang Yang, Qiangqiang Yuan, Randi Fu, and
Hongyan Zhang. A large-scale benchmark data set for evalu-
ating pansharpening performance: Overview and implementa-
tion. IEEE GRSM, 9(1):18–52, 2021. 1, 2, 5
[50] Xiangchao Meng, Nan Wang, Feng Shao, and Shutao Li.
Vision Transformer for pansharpening. IEEE TGRS, 60:1–11,
2022. 2, 3
[51] Gaspard Monge. M ´emoire sur la th´eorie des d´eblais et des
remblais. Mem. Math. Phys. Acad. Royale Sci., pages 666–
704, 1781. 3
[52] Fjola Palsson, Johannes R. Sveinsson, and Magnus O. Ulfars-
son. A new pansharpening algorithm based on total variation.
IEEE GRSL, 11(1):318–322, 2014. 3
[53] Siran Peng, Xiangyu Zhu, Haoyu Deng, Liang-Jian Deng,
and Zhen Lei. FusionMamba: Efficient remote sensing image
fusion with state space model. IEEE TGRS, 2024. 2, 3
[54] Gabriel Peyr ´e and Marco Cuturi. Computational optimal
transport. Found. Trends Mach. Learn., 11(5-6):355–607,
2019. 3
[55] Sheida Rahmani, Melissa Strait, Daria Merkurjev, Michael
Moeller, and Todd Wittman. An adaptive IHS pan-sharpening
method. IEEE GRSL, 7(4):746–750, 2010. 2
[56] Aaditya Ramdas, Nicolas Garcia, and Marco Cuturi. On
wasserstein two sample testing and related families of non-
parametric tests, 2015. 4
[57] Xiangyu Rui, Xiangyong Cao, Yining Li, and Deyu Meng.
Variational zero-shot multispectral pansharpening. IEEE
TGRS, 62:1–16, 2024. 3
[58] Xiangyu Rui, Xiangyong Cao, Li Pang, Zeyu Zhu, Zongsheng
Yue, and Deyu Meng. Unsupervised hyperspectral pansharp-
ening via low-rank diffusion model. Inf. Fusion., 107:102325,
2024. 2, 3
[59] Filippo Santambrogio. One-dimensional issues, pages 59–85.
Springer International Publishing, Cham, 2015. 4
11859


<!-- PDF PAGE 11 -->

[60] Giuseppe Scarpa and Matteo Ciotola. Full-resolution quality
assessment for pansharpening. Remote Sens., 14(8), 2022. 6
[61] Vijay P. Shah, Nicolas H. Younan, and Roger L. King. An
efficient pan-sharpening method via a combined adaptive
PCA approach and contourlets. IEEE TGRS, 46(5):1323–
1335, 2008. 2
[62] Xunyang Su, Jinjiang Li, and Zhen Hua. Transformer-based
regression network for pansharpening remote sensing images.
IEEE TGRS, 60:1–23, 2022. 2, 3
[63] Weiwei Sun, Jun Zhou, Xiangchao Meng, Gang Yang, Kai
Ren, and Jiangtao Peng. Coupled temporal variation infor-
mation estimation and resolution enhancement for remote
sensing spatial-temporal-spectral fusion. IEEE TGRS, 61:
1–18, 2023. 3
[64] Laurens van der Maaten and Geoffrey Hinton. Visualizing
data using t-SNE. J. Mach. Learn. Res., 9(86):2579–2605,
2008. 2
[65] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In NeurIPS, 2017. 2, 3
[66] C´edric Villani et al.Optimal transport: old and new. Springer,
2008. 4
[67] Gemine Vivone, Miguel Sim˜oes, Mauro Dalla Mura, Rocco
Restaino, Jos´e M. Bioucas-Dias, Giorgio A. Licciardi, and
Jocelyn Chanussot. Pansharpening based on semiblind decon-
volution. IEEE TGRS, 53(4):1997–2010, 2015. 3
[68] Gemine Vivone, Mauro Dalla Mura, Andrea Garzelli, Rocco
Restaino, Giuseppe Scarpa, Magnus O. Ulfarsson, Luciano
Alparone, and Jocelyn Chanussot. A new benchmark based on
recent advances in multispectral pansharpening: Revisiting
pansharpening with classical and emerging pansharpening
methods. IEEE GRSM, 9(1):53–81, 2021. 1, 2, 3
[69] Lucien Wald, Thierry Ranchin, and Marc Mangolini. Fusion
of satellite images of different spatial resolutions: Assessing
the quality of resulting images. Photogramm. Eng. Remote
Sens., 63:691–699, 1997. 3, 6
[70] Hebaixu Wang, Meiqi Gong, Xiaoguang Mei, Hao Zhang, and
Jiayi Ma. Deep unfolded network with intrinsic supervision
for pan-sharpening. AAAI, 38(6):5419–5426, 2024. 3
[71] Hebaixu Wang, Hao Zhang, Xin Tian, and Jiayi Ma. Zero-
Sharpen: A universal pansharpening method across satellites
for reducing scale-variance gap via zero-shot variation. Inf.
Fusion., 101:102003, 2024. 3
[72] Jun-Da Wang, Liang-Jian Deng, Chen-Yu Zhao, Xiao Wu,
Hong-Ming Chen, and Gemine Vivone. Cascadic multirecep-
tive learning for multispectral pansharpening. IEEE TGRS,
61:1–16, 2023. 3
[73] Yingying Wang, Xuanhua He, Yuhang Dong, Yunlong Lin,
Yue Huang, and Xinghao Ding. Cross-modality interaction
network for pan-sharpening. IEEE TGRS, 62:1–16, 2024. 3
[74] Zhong-Cheng Wu, Ting-Zhu Huang, Liang-Jian Deng, Jin-
Fan Hu, and Gemine Vivone. VO+Net: An adaptive approach
using variational optimization and deep learning for panchro-
matic sharpening. IEEE TGRS, 60:1–16, 2022. 3
[75] Jin-Liang Xiao, Ting-Zhu Huang, Liang-Jian Deng, Zhong-
Cheng Wu, Xiao Wu, and Gemine Vivone. Variational pan-
sharpening based on coefficient estimation with nonlocal re-
gression. IEEE TGRS, 61:1–15, 2023. 3
[76] Guangxu Xie, Rencan Nie, Jinde Cao, He Li, and Jintao
Li. A deep multiresolution representation framework for
pansharpening. IEEE TGRS, 62:1–16, 2024. 3
[77] Yinghui Xing, Litao Qu, Shizhou Zhang, Jiapeng Feng, Xi-
uwei Zhang, and Yanning Zhang. Empower generalizability
for pansharpening through text-modulated diffusion model.
IEEE TGRS, 2024. 3
[78] Yinghui Xing, Litao Qu, Shizhou Zhang, Kai Zhang, Yan-
ning Zhang, and Lorenzo Bruzzone. CrossDiff: Exploring
self-supervised representation of pansharpening via cross-
predictive diffusion model. IEEE TIP, 2024. 2, 3
[79] Qizhi Xu, Yuan Li, Jinyan Nie, Qingjie Liu, and Mengyao
Guo. UPanGAN: Unsupervised pansharpening based on the
spectral and spatial loss constrained generative adversarial
network. Inf. Fusion., 91:31–46, 2023. 3
[80] Shuang Xu, Jiangshe Zhang, Zixiang Zhao, Kai Sun, Junmin
Liu, and Chunxia Zhang. Deep gradient projection networks
for pan-sharpening. In CVPR, pages 1366–1375, 2021. 3, 5
[81] Gang Yang, Man Zhou, Keyu Yan, Aiping Liu, Xueyang
Fu, and Fan Wang. Memory-augmented deep conditional
unfolding network for pansharpening. In CVPR, pages 1778–
1787, 2022. 3, 5, 6, 7
[82] Jie Yang, Xiao-Yang Fu, Yan Hu, Yue Huang, Xing Ding,
and John Paisley. PanNet: A deep network architecture for
pan-sharpening. ICCV, pages 1753–1761, 2017. 2, 3, 5
[83] Qiangqiang Yuan, Yancong Wei, Xiangchao Meng, Huanfeng
Shen, and Liangpei Zhang. A multiscale and multidepth
convolutional neural network for remote sensing imagery
pan-sharpening. IEEE JSTARS, 11(3):978–989, 2018. 2, 3, 5
[84] Jie Zhang, Xuanhua He, Keyu Yan, Ke Cao, Rui Li, Chengjun
Xie, Man Zhou, and Danfeng Hong. Pan-sharpening with
wavelet-enhanced high-frequency information. IEEE TGRS,
62:1–14, 2024. 3
[85] Chen-Yu Zhao, Tian-Jing Zhang, Ran Ran, Zhi-Xuan Chen,
and Liang-Jian Deng. LGPConv: Learnable gaussian pertur-
bation convolution for lightweight pansharpening. In IJCAI,
pages 4647–4655, 2023. Main Track. 3
[86] Mengjiao Zhao, Mengting Ma, Xiangdong Li, Xiaowen Ma,
Wei Zhang, and Siyang Song. Supervised detail-guided multi-
scale state space model for pan-sharpening. IEEE TGRS,
2024. 2, 3
[87] Kaiwen Zheng, Jie Huang, Man Zhou, Danfeng Hong, and
Feng Zhao. Deep adaptive pansharpening via uncertainty-
aware image fusion. IEEE TGRS, 61:1–15, 2023. 5
[88] Yu Zhong, Xiao Wu, Liang-Jian Deng, ZiHan Cao, and Hong-
Xia Dou. SSDiff: Spatial-spectral integrated diffusion model
for remote sensing pansharpening. In NeurIPS, pages 77962–
77986. Curran Associates, Inc., 2024. 3
[89] Man Zhou, Jie Huang, Keyu Yan, Hu Yu, Xueyang Fu, Aiping
Liu, Xian Wei, and Feng Zhao. Spatial-frequency domain
information integration for pan-sharpening. In ECCV, pages
274–291, Cham, 2022. Springer Nature Switzerland. 3, 5
[90] Man Zhou, Jie Huang, Keyu Yan, Danfeng Hong, Xiuping
Jia, Jocelyn Chanussot, and Chongyi Li. A general spatial-
frequency learning framework for multimodal image fusion.
IEEE TPAMI, pages 1–18, 2024. 3, 5
11860

<!-- END-OF-PAPER:6e6a3ccaf7f6 -->
