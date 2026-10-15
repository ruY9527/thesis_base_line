<!-- PDF PAGE 1 -->

Automatic Spectral Calibration of Hyperspectral Images:
Method, Dataset and Benchmark
Zhuoran Du1,2, Shaodi You1, Cheng Cheng2, Shikui Wei2
1University of Amsterdam, Netherlands
2Visual Intelligence +X International Cooperation Joint Laboratory of MOE, Beijing Jiaotong University
Abstract
Hyperspectral images (HSI) densely sample the world in
both the space and frequency domains and, therefore, are
more distinctive than RGB images. Usually, HSI needs to
be calibrated to minimize the impact of various illumination
conditions. The traditional way to calibrate HSI utilizes a
physical reference, which involves manual operations, oc-
clusions, and/or limits camera mobility. These limitations
inspire this paper to automatically calibrate HSIs using a
learning-based method. Towards this goal, a large-scale
HSI calibration dataset, which has 765 high-quality HSI
pairs covering diversified natural scenes and illuminations,
is created. The dataset is further expanded to 7650 pairs
by combining with 10 different physically measured illumi-
nations. A spectral illumination transformer (SIT) together
with an illumination attention module is proposed. Exten-
sive benchmarks demonstrate the SoTA performance of the
proposed SIT. The benchmarks also indicate that low-light
conditions are more challenging than normal conditions.
The dataset and codes are available online: https://
github.com/duranze/Automatic- spectral-
calibration-of-HSI.
1. Introduction
Hyperspectral image (HSI) densely samples the world in
both spatial and frequency domains. It is more informa-
tive than RGB images for recognizing material. Therefore
it is widely used in remote sensing and laboratory analysis
[21][4][6][17].
However, HSI appearance is highly dependent on global
illumination. While in remote sensing and lab environ-
ments, global illuminations are highly uniform in a large
spatial and temporal range or can be well controlled; global
illuminations in natural scenes are highly varied spatially
and temporally. Therefore, to capture HSI in natural scenes,
global illumination needs to be measured simultaneously.
There are two traditional ways to measure the global il-
lumination. The synchronous method [3] (Fig. 1.a) utilizes
Whiteboard Recording
(a) Synchronous Spectral Calibration
(c) Automatic Spectral Calibration (Proposed)
Spectral Illumination Transformer (SIT)
(b) Asynchronous Spectral Calibration
Devision
Devision
Figure 1. Comparison of HSI Calibration Methods: (a) Syn-
chronous, (b) Asynchronous, and (c) Automatic (Proposed).
a reference such as a barium sulfate panel (an ideal white
panel) [16] in the scene. The disadvantage is the panel
partly occludes the scene. The asynchronous method [2, 8]
(Fig. 1.b) takes two HSIs of the same scenes quickly with
and without the reference. The disadvantage is that this
method requires double shots, manual operations, and an
unmoved camera.
These disadvantages inspire this paper: can we learn to
automatically calibrate the HSI without a physical refer-
ence? If the accuracy is close to the physical methods, then
the learning-based method has the advantages of less man-
ual operations, no occlusion, and more mobility.
To enable this research, we have created the first HSI
auto calibration dataset BJTU-UVA Dataset. The dataset is
captured using a 204 bands 400-1000nm hyperspectral cam-
era with a 512 × 512 spatial resolution, and 12 bits quan-
tization. The dataset has 765 calibrated and uncalibrated
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
28081


<!-- PDF PAGE 2 -->

image pairs which are obtained using the asynchronous
method. The dataset covers diversified scenes and illumi-
nations. To further study the illumination calibration, an
expansion dataset BJTU-UVA-E is created, which compos-
ites calibrated HSIs with 10 different physically measured
illuminations. In such a way, the BJTU-UV A-E has 7650
uncalibrated HSIs. It is worth noticing that BJTU-UV A is
not only the first HSI calibration dataset, it is also one of the
largest public HSI datasets.
After that, we propose the Spectral Illumination Trans-
former (SIT), which is the first learning-based HSI calibra-
tion method to the best of our knowledge. SIT is based on
a U-shaped transformer. To allow SIT to learn illumina-
tion properly, we propose theillumination attention module
which is inspired by the classic Gray-World[5] algorithm.
We consider SIT the baseline of learning-based methods
which may inspire future research in this new topic.
Detailed benchmarks are provided using the BJTU-UV A
dataset. Given that there are no other learning-based meth-
ods for this novel task, methods for HSI denoising and
reconstruction are compared. And the classic method of
Gray-World is compared. It is shown that SIT is the state-
of-the-art. It is interesting to find that Gray-World performs
comparable to other learning-based methods on real illumi-
nations. It is also noticed that low-light scenes/areas are
likely to have larger errors; which can be a future research
topic.
The novelty of the paper can be summarized as:
• We propose a novel research topic, automatic HSI cali-
bration, which has no occlusion, less manual operations,
and more mobility than traditional methods.
• We propose the large-scale HSI dataset BJTU-UV A,
which is the first dataset for HSI auto-calibration.
• We propose the spectral illumination transformer (SIT)
with the illumination attention module, which is the very
first transformer-based method on this topic.
2. Related Work
2.1. Hyperspectral Image Calibration
In the literature, several spectral calibrations using physi-
cal references are developed for remote sensing and indoor
optics.
In remote sensing, calibration is done using a physi-
cal reference (Fig. 1.a). Pan et al. in 2001 [25] used a
spectroradiometer calibrated against a barium sulfate panel
(the physical reference), assuming constant solar irradiance;
Ferreira et al. in 2004 [11] used linear interpolation of pre-
and post-flight reference panel readings; and Miura et al.
in 2008 [22] used continuous ground-based panel readings
during flight to adjust data. The limitation of applying this
method to natural scenes is that there always exists occlu-
sion from the white reference.
In laboratory settings, where illumination can be pre-
cisely controlled, calibration can be performed asyn-
chronously (Fig. 1.b). Under these conditions, illumination
only needs to be measured once and can then be applied
in subsequent measurements. Recently, Moghadam et al.
[23] proposed a data-driven method to enable indoor hyper-
spectral imaging using affordable and commonly available
LED and fluorescent lighting sources. However, in the nat-
ural outdoor scenes, controlling illumination is infeasible.
Therefore, to apply asynchronous methods, after measuring
the illumination, the HSI has to be taken as soon as possible.
This requires manual operations and a stable camera.
2.2. RGB White Balancing
RGB white balancing is a similar topic but with major dif-
ferences. First, each of the RGB channels is a wide-band
channel whereas in HSI the channel is a narrow band. Sec-
ond, RGB bands usually overlap with each other whereas
HSI bands do not or have limited overlaps. Third, many
RGB-based methods aim for optimizing human perception
while HSI calibration aims for physical accuracy.
Recent advances in white balancing for sRGB images
have primarily leveraged deep learning techniques to en-
hance color correction. The two-stage framework by
Farghaly et al. [10] combined global color mapping with
local adjustments, achieving perceptually improved results.
Stability across varying color temperatures has been ad-
dressed by Li et al. [18] with their SWBNet, which learned
color temperature-insensitive features and applied a novel
contrastive loss for consistent outcomes. Addressing the
complexity of mixed-illuminant scenes, Afifi et al. [1]
blended images with predefined white-balance settings us-
ing learned weighting maps, while Kinli et al. [15] mod-
eled lighting variations as style factors, enhancing cor-
rection capabilities without explicit illuminant detection.
Lastly, efficiency in processing was significantly enhanced
by [14] through deterministic illumination mapping, facili-
tating real-time applications.
2.3. HSI Restoration and Reconstruction
Learning-based research on HSI is mainly focusing on de-
noising and band reconstruction. This research explores
many new deep-learning architectures for HSI because it is
not trivial to extend RGB-based architectures. Notice that
this research does not need the HSI to be calibrated and can-
not directly calibrate the HSI.
Puria et al. proposed the DivIll [23] method, which
used CNN layers and residual blocks to address reflectance
abnormalities caused by illumination variations in indoor
scenes. Li et al. [20] introduced a local-area-based ap-
proach that linearly represented each pixel by its neighbors
without requiring parameterized weight calculations, em-
phasizing spectrum preservation. Zhang et al. [34] provided
28082


<!-- PDF PAGE 3 -->

a comprehensive review of HSI denoising techniques rang-
ing from model-driven to hybrid model-data-driven meth-
ods. Lastly, Pan et al. [24] integrated multiscale con-
textual information with a coattention mechanism in the
Multiscale Adaptive Fusion Network (MAFNet), showcas-
ing improved denoising capabilities. SERT [19] introduced
a rectangle self-attention mechanism to capture non-local
spatial similarity and a spectral enhancement module to ex-
tract global low-rank properties of hyperspectral images for
effective noise suppression. Hu et al . [13] proposed the
HCANet by extending the U-shaped transformer to hyper-
spectral imaging (HSI) and leveraged a spectral transformer
to capture inter-channel relationships effectively.
3. HSI Calibration
In this paper, we denote an HSI as: I(x, y, λ) ∈ RH×W ×C,
where I is the intensity, x, y are the spatial coordinates and
λ the wavelength; H, W, C are the height, width, and chan-
nels correspondingly. In this paper we focus on global illu-
mination and assume it is uniform, therefore, for simplicity,
in most of this paper, we write only the wavelength I(λ).
HSI Illumination Calibration Physically, an HSI I cap-
tures the production of scene reflectanceR and illumination
L wavelength-wisely:
I(λ) = R(λ)L(λ). (1)
Unlike RGB, scene reflectance R(λ) is highly distinc-
tive if measured in an HSI way. And therefore it is
widely used in remote sensing [21][4][6][17] and labora-
tory analysis[27][9]. Notice the reflectance is not albedo but
still varies from shading and inter-flection and non-uniform
illumination.
The goal of HSI calibration is to linearly adjust the inten-
sity as if the global illumination is ideal white in the sam-
pled wavelength range:
L(λ) ≡ l, (2)
where l is a constant which usually set to be l = 1 , then
R(λ) ≡ I(λ).
If the illumination I(λ) is not ideal white, then we can
calibrate the image as if the illumination is ideal white by:
R(λ) = I(λ)
L(λ) . (3)
Synchronous method In a synchronized method
(Fig. 1.a), L(λ) is measured directly using a reference such
as a barium sulfate panel [16]. The disadvantage is the
reference has to be in the image which causes occultation.
Asynchronous method In the asynchronous method, we
first measure Lp,t1 (λ) on the reference at position p and
time t1. Then, take an HSI Ip,t2 (λ) without the reference
at the same position but at a later time. Assuming the dif-
ference between Lp,t1 (λ) and Lp,t2 (λ) is negligible, then:
R2(λ) = Ip,t2 (λ)
Lp,t1 (λ). The disadvantages are more manual op-
erations, and in an outdoor natural scene, the illumination
can be very dynamic and therefore needs to be remeasured
frequently; it also makes moving the camera infeasible.
Learning-based method (proposed) Considering the
above disadvantages, we propose a novel research question:
can the scene radiance be inferred directly through a learn-
ing method f without a physical reference and have satisfy-
ing accuracy as a physical measurement?
˜R(λ) = f (I(λ)). (4)
If possible, it has at least three advantages: first, there is no
occlusion, second, the camera can move freely, third, the
imaging only needs to be done once.
4. BJTU-UV A Dataset
Our work introduces the first-ever dataset for automatic
spectral calibration. The hyperspectral automatic spectral
calibration imagedataset (BJTU-UV A) comprises 765 pairs
of real HSI pairs; each pair includes one original HSI and its
asynchronous calibration. To further diversify the illumina-
tions, we combine the scene radiance with ten different real
illuminations to generate more uncalibrated images. The
expanded dataset (BJTU-UV A-E) comprises 7650 pairs of
uncalibrated and calibrated HSIs.
4.1. BJTU-UV A Dataset
For the HSI camera, we use Specim IQ [28], which offers a
spectral resolution of 3nm across a wavelength range from
400nm to 1000nm. For reference, we use the IQ White Ref-
erence Board [28], which can reflect all bands of the spec-
trum from 400nm to 1000nm evenly.
For caption, we use the asynchronous method as de-
scribed in Sec.3. Specifically, we capture the scene Ip,t(λ)
at position p and time t without the reference. And right
after that, we capture the illumination Lp,t+δ(λ) by putting
the reference in the scene. We kept the time difference small
and the camera position unchanged so as to minimize the
change in the illumination. The images are stored in the
linear format without gamma correction in 12 bits.
In practice, the scene and illumination are not noise-free.
The major source of the noise is the dark currentd(λ). Dark
current refers to the electronic noise recorded by the cam-
era’s sensor even when no light is present. d(λ) is also
recorded.
28083


<!-- PDF PAGE 4 -->

The ground truth is thus generated with the consideration
of d(λ):
Rgt(λ) = Ip,t(λ)
Lp,t+δ(λ) = I d
p,t(λ) − dp,t(λ)
Ld
p,t+δ(λ) − dp,t+δ(λ) . (5)
The dataset involves diversified scenes and illuminations
as shown in Fig. 2. These scenes feature urban and nat-
ural environments under different weather conditions and
times of day, including roads, buildings, statues, flowers,
and other colored materials. The variety in scene composi-
tion ensures that the dataset can adequately test the efficacy
of spectral calibration methods.
4.2. BJTU-UV A-E Dataset
To enable further study of the illumination. We expand the
dataset by combining the ground truth reflectance Rgt(λ)
with 10 different physically measured illuminationsLSi (λ).
ISi (λ) = Rgt(λ)LSi (λ), i ∈ 1, 2, · · · 10, (6)
We record these ten different illuminations with the
whiteboard as shown in Fig. 3. To minimize the dark cur-
rent, we also deduct the noise dSi (λ) from actual illumi-
nance recording Ld
Si (λ) by
LSi (λ) = Ld
Si (λ) − dSi (λ), i ∈ 1, 2, · · · 10. (7)
As shown in Fig. 3, the 10 illuminations include five
natural illuminations: sunny, cloudy, rainy, evening, and
shadow; which are measured using the whiteboard in the
scenes. The rest five are colored filtered, which are mea-
sured by placing the color filters (red, blue, yellow, purple,
and green) in front of the whiteboard.
4.3. 31 Channels Subset
Since much of the existing HSI research is conducted using
31 channels spanning 400–700 nm, we resample our HSIs
to match this standard, covering 31 channels from 400–700
nm, for both the BJTU-UV A and BJTU-UV A-E datasets.
This approach ensures compatibility with legacy methods.
4.4. Specifications
The dataset specifications are summarized in Fig. 2.d. No-
tably, this is not only the first HSI illumination calibration
dataset but also one of the largest HSI datasets focused on
natural scenes (further details are provided in the supple-
mentary materials).
4.5. Evaluation Metrics
We introduce four commonly used metrics: PSNR[30],
RMSE[12], ERGAS[29], and SAM[33].
The Peak Signal-to-Noise Ratio (PSNR) for HSI is de-
fined as:
PSNR = 1
B
BX
k=1
10 log10


max
 
Rk
gt
2
1
HW



Rk
gt − ˆRk




2
2

 , (8)
in which B denotes the number of bands and ∥·∥2 refers to
the two-norm.
The Root Mean Squared Error (RMSE) is calculated by:
RMSE =
vuut 1
N
NX
i=1
( ˆRgt
i
− Ri)2, (9)
in which N denotes the number of pixels in each image.
The Error Relative Global Dimensionless Synthesis (ER-
GAS) metric is specifically designed for evaluating the
quality of high-resolution synthesized images and can be
calculated by:
ERGAS =
vuuut 1
B
BX
k=1



Rk
gt − ˆRk




2
2
µ2 
Rk
gt
 , (10)
in which µ denotes the mean operation, and B number of
bands.
The Spectral Angle Mapper (SAM) is commonly em-
ployed to assess the extent of spectral information preserva-
tion at each pixel, which can be calculated by:
SAM = 1
N
NX
i=1
arccos
 
⟨ ˆRi, Ri
gt⟩
∥ ˆRi∥2∥Ri
gt∥2
!
, (11)
where N is the number of pixels.
5. Spectral Illumination Transformer
With the dataset ready, we propose the very first learning-
based method which focuses on automatic illumination
calibration for HSI: the Spectral illumination transformer
(SIT). We consider SIT a baseline and will inspire future
research.
5.1. Framework Overview
As shown in Fig.4, the overall framework of our SIT is a
U-shaped transformer proposed by [13] with residual con-
nections. The framework is inspired by the commonly used
U-Net for RGB images [26], and later extended to the trans-
former by Wang et al. [31]. Hu et al. [13] further extends
it to HSI. The U-shaped encoder-decoder structure enables
exploring more contextual information by increasing recep-
tion fields. And they use the spectral transformer to explore
the relationship between channels.
28084


<!-- PDF PAGE 5 -->

plants
buildings
water
metal
 facade
statues
(a) Dataset recording (b) Scene variety (c) Illumination variety
Bands 204
Images 765
Spectral Range 400-1000nm
Spatial Resolution 512 x 512
Bit Depth 12
Year 2024
(d) Specification
A
B
C
D
E
A
B C
D
E
Figure 2. Overview of the BJTU-UV A Dataset: (a) Dataset Recording Setup, (b) Variety of Scenes, (c) Illumination Variety, and (d) HSI
Camera Specifications and Dataset Specification
sunny
cloudy
dusk
rainy
shadowy
puple
blue
green
red
(a) Ten Pre-defined illumination recording
yellow
Natural Color filtered
(b) Spectral curves of whiteboard in ten illumination
Figure 3. Measurement of Ten Different Illuminations Using
a Whiteboard: Five Natural Conditions (Sunny, Cloudy, Rainy,
Evening, and Shadow) and Five Color-Filtered Conditions (Red,
Blue, Yellow, Purple, and Green)
However, as will be shown later by the experiments, this
structure was not specially designed for HSI calibration and
cannot properly capture the global illumination. Therefore,
we develop a novel spectral illumination transformer unit
(SIT-U).
5.2. Spectral Illumination Transformer Unit
The SIT-U is inspired by the classic RGB white balanc-
ing algorithm Gray-World [5]. The Gray-World method as-
sumes the global mean values of all pixels in each channel
are the same (e.g. 0.5).
As shown in Fig.4, The SIT-U is anM-layer transformer.
On the mth layer of SIT-U, the input feature xm
0 of the mth
layer of SIT-U goes through the layer normalization LN
and a convolutional layer f m
c0 by
xm
1 = f m
c0 (LN (xm
0 )), m = 1 · · · M. (12)
Then xm
1 is injected into the two branches: spectral atten-
tion (SA) calculation and illumination attention (IA) calcu-
lation. For SA calculation, we maintain the same structure
as HCANet[13]. We use SA to calculate the attention map
Am
S and the attention “value” xm
v in the mth layer of SIT.
Instead of immediately multiplying Am
S with xm
v directly
as HCANet, we design the illumination attention module to
import the factors of illumination into the attention process,
which will be introduced in detail in the following para-
graph.
Illumination Attention Inspired by the Gray-World algo-
rithm, we propose the Illumination Attention (IA). Specif-
ically, in the second branch “IA”, we use the illumination
extractor to gradually capture global illumination features
xm
I .
First, it goes through the CNN layerf m
c1 keeping the spa-
tial dimension by
xm
2 = f m
c1 (xm
1 ). (13)
Then, we gradually extract local illumination features
through down-scaling convolution and average poolingAP ,
by
xm
3 = AP (f m
c2 (xm
2 )), (14)
xm
4 = AP (f m
c3 (xm
3 )), (15)
where the convolutional layer f m
c2 and f m
c3 have a stride of
t and the pooling size of AP is p. Finally, we calculate the
global mean value in each channel by
xm
I = favg (xm
4 ), (16)
where favg is used to scale down the feature map xm
4 into a
tensor xm
I . This averaging operation can implicitly explore
the illumination of the entire image as the Gray-World[5]
method.
Then, we linearly project the illumination feature into
attentive hidden space using f m
l by
qm
I = km
I = f m
l (xm
I ), (17)
where qm
I and km
I are the query and key of attention model.
The attention matrix Am
I in the IA branch of mth layer can
be calculated by
Am
I = (qm
I )T km
I . (18)
Through the IA branch, we explore the features of global
illumination and calculate the other attention matrix Am
I .
28085


<!-- PDF PAGE 6 -->

Hyperspectral Intensity Image 
SIT-U
SIT-U
SIT-U
SIT-U
Encoder 1 Encoder N
Decoder N
Hyperspectral Reflectance Image 
CC
m+1th layer
LayerNorm
mth layer
convolution pixel-
unshuffle
pixel-
shuffle C
dot product
concat
matrix productlayer
norm add
Decoder 1
c
V K Q
hw
Conv
LayerNorm
FeedForward
AP
Spectral Attention Illumination Attention
AP
Figure 4. Structure of Spectral Illumination Transformer (SIT) framework and its unit (SIT-U) with Spectral and Illumination Attention
branches for improved HSI calibration. The Illumination Attention branch mimics the Gray-World method to capture illumination features.
Spectral Illumination Attention Then we combine the
two attention matrices Am
S and Am
I by
Am
SI = softmax(Am
S · Am
I ), (19)
where the Am
SI is the final attention matrix in the spectral il-
lumination attention. Then, we calculate the output of spec-
tral illumination attention xm
a by
xm
a = Am
SI xm
v , (20)
where the xm
v is calculated in the spectral attention branch.
Then, as shown in Fig.4, the output xm
v will go through
layer normalization and feed-forward layers to calculate the
final output of the mth layer of SIT-U.
6. Experiments
6.1. Experiment Settings
The BJTU-UV A dataset is split into three sets: 535 for train-
ing, 114 for validation, and 116 for testing. Similarly, for
the expansion BJTU-UV A-E: 5350 for training, 1140 for
validation, and 1160 for testing.
Then we propose two tracks of benchmarks. The first is
the full spectrum test (400-1000nm, 204 channels); and the
second is the 31-channel test, which is to align with existing
HSI-enhancement dataset and methods [32][7][3][8].
Since there are no existing learning-based methods for
direct comparison, we select a classic baseline, Gray-World
[5]; a recent indoor spectral restoration method, DivIll [23];
and two state-of-the-art HSI denoising methods, SERT [19]
and HCANet [13].
All models in our experiments are trained on Nvidia
3090 GPUs. We randomly crop the image with a size of
256 and train with a batch size of 4. We train our model and
other models for 500 epochs on the BJTU-UV A dataset and
50 epochs on the BJTU-UV A-E dataset. We use L1 loss for
optimizing pixel-level differences and Adam optimizer with
an initial learning rate of 10−4.
6.2. Benchmarks
Comparison on Full-Spectrum HSI Table 1 presents the
results. The first part shows the results on the original
dataset. We observe that the Gray-World method shows
relatively good performance, particularly with stable SAM
scores. Among deep learning-based approaches, DivIll,
SERT, and HCANet improve on metrics like PSNR, RMSE,
and ERGAS; however, they struggle to significantly en-
hance SAM scores. In contrast, our proposed SIT method
achieves the highest SAM scores across both tracks, with
substantial improvements in other evaluation metrics as
well.
The lower part shows the results on the expanded dataset.
The Gray-World method is not comparable to the other deep
learning methods on all three tracks. Although deep learn-
ing methods improve their performance on all three tracks,
our proposed method SIT still gets the best performance on
28086


<!-- PDF PAGE 7 -->

0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
Intensity GT
(a)
(b)
(c)
(d)
(e)
(f)
Gray-World HCANet DivIll SERT SIT (Proposed)
Figure 5. Visual Comparison of Absolute Error Using Heat Maps: The first two columns show the RGB rendering of the input and ground
truth (GT) for visualization. Rows 1-3 display samples from BJTU-UV A, and Rows 4-6 show samples from BJTU-UV A-E.
Val Test
P↑ S ↓ R↓ E↓ P↑ S ↓ R↓ E↓
BJTU-UV A
Gray-World[5] 22.6 2.9 9.1 11.3 23.3 3.1 8.4 11.6
DivIll[23] 23.5 5.5 8.3 10.2 24.1 6.1 7.6 10.0
SERT[19] 22.3 10.4 8.3 13.2 22.8 11.1 7.9 13.5
HCANet[13] 25.1 4.4 6.8 9.1 25.7 3.6 6.4 9.7
SIT(proposed) 26.1 2.9 6.2 8.7 26.3 3.1 5.8 9.6
BJTU-UV A-E
Gray-World[5] 22.5 2.6 9.3 11.5 23.2 2.8 8.6 11.7
DivIll[23] 35.2 3.1 2.2 2.7 35.2 3.4 2.2 3.0
SERT[19] 36.4 2.7 1.8 2.6 36.4 2.9 1.7 2.7
HCANet[13] 37.9 2.1 1.4 2.0 37.3 2.5 1.5 2.3
SIT(proposed) 39.3 1.8 1.2 1.7 39.1 2.0 1.3 1.9
Table 1. Automatic spectral calibration evaluation on the full-
spectrum HSI: PSNR(P), SAM(S), RMSE (R%), and ERGAS
(E%).
all metrics.
We visualize the experimental results in Fig. 5. As can
be seen, the proposed SIT is dedicated to this novel task and
consistently outperforms existing methods. More results are
provided in the supplementary materials.
Comparison on 31-Channel HSI Table 2 shows the per-
formance on 31-channel HSI data. The results align with
those from the full-spectrum experiments, demonstrating
that the proposed SIT method achieves the best performance
across all settings. Notably, all methods perform better than
in the full-spectrum experiments, likely due to the reduced
number of channels simplifying the tasks.
Discussions It is worth noticing that for the original
datasets, which are all-natural illuminations, Gray-World is
more reliable than some of the learning-based methods; it is
28087


<!-- PDF PAGE 8 -->

Val Test
P↑ S ↓ R↓ E↓ P↑ S ↓ R↓ E↓
BJTU-UV A
Gray-World[5] 25.2 3.3 5.8 9.6 24.2 4.1 6.5 14.2
DivIll[23] 22.4 12.7 7.7 14.3 21.7 13.3 7.5 16.5
SERT[19] 23.6 10.1 6.6 13.4 23.0 10.3 6.4 15.2
HCANet[13] 26.4 3.5 5.6 9.1 26.3 3.4 5.0 11.0
SIT(proposed) 26.8 2.9 5.3 9.2 26.7 2.7 5.1 11.7
BJTU-UV A-E
Gray-World[5] 25.3 3.1 5.8 9.5 24.2 3.6 6.5 14.1
DivIll[23] 37.2 2.6 1.6 3.0 37.8 2.7 1.3 2.9
SERT[19] 36.6 3.1 1.7 3.3 37.0 3.3 1.5 3.6
HCANet[13] 37.8 2.4 1.3 2.5 38.1 2.5 1.2 2.8
SIT(proposed) 40.3 1.8 1.0 1.8 41.1 1.9 0.8 1.9
Table 2. Automatic spectral calibration evaluation on 31-Channel
HSI:PSNR(P), SAM(S), RMSE (R%), and ERGAS (E%).
Validation Test
P ↑ S ↓ R ↓ E ↓ P ↑ S ↓ R ↓ E ↓
W/O SA & IA 21.4 11.5 9.3 15.1 22.1 8.5 11.9 15.6
IA 25.5 4.1 6.3 9.2 25.8 4.3 5.9 9.6
SA 25.1 4.4 6.8 9.1 25.7 3.6 6.4 9.7
IA & SA 26.1 2.9 6.2 8.7 26.3 3.1 5.8 9.6
Table 3. Ablation Study on BJTU-UV A Full-Spectrum Data:
Performance Comparison of Different Attention Configurations
(Spectral Attention (SA) and Illumination Attention (IA))
reasonable because the ”Gray-World” assumption is an as-
sumption on the illumination of natural images. As for the
expansion dataset, when the illuminations are not directly
captured from the natural scenes, the performance is lower
than other learning-based methods.
It is also noticed that learning-based methods perform
significantly better on the expansion dataset than on the
original dataset, whereas Gray-World does not have such
improvements. This is probably because: first, the expan-
sion dataset is much larger; second, the diversified illumina-
tions allow the learning-based methods to better learn and
model the illuminations.
6.3. Ablation Study
We study how useful the proposed illumination attention
(IA) module is. Four experimental setups are evaluated:
“W/O SA & IA”, where only CNN layers are used; Only
“SA”, is a setting which falls back to HCANet [13]; “IA”,
where attention similarity calculation uses only IA; and “IA
& SA”, our complete SIT approach, where the attention
maps of IA and SA are combined in each layer.
As shown in Table 3, IA can significantly improve the
performance with and without the presence of SA. When
both SA and IA are present, the performance is optimal.
Besides, it is interesting to find that although the structure
is different, the performance of SA and IA are similar when
working separately.
(a)
(b)
Figure 6. Challenging Cases: Comparison of Calibrated Spectra
by Different Methods
6.4. Challenging Cases
As can be seen in Fig. 6, although the proposed SIT per-
forms relatively better than other methods, it still faces chal-
lenges in low-light scenes and areas. This challenge is more
significant in the infrared bands, where the illumination is
generally less intense than the visible bands. We consider
handling low-light scenes as a potential future work.
7. Conclusion
This paper introduces a learning-based approach for the
automatic calibration of hyperspectral images (HSIs). A
large-scale HSI calibration dataset, BJTU-UV A, has been
developed, comprising 765 high-quality HSI pairs that en-
compass diverse natural scenes and illumination conditions.
This dataset is further augmented to 7,650 pairs by incor-
porating 10 distinct, physically measured illuminations. We
propose a spectral illumination transformer (SIT) with an il-
lumination attention module, achieving state-of-the-art per-
formance, as demonstrated by extensive benchmarks. An
ablation study highlights the effectiveness of the illumina-
tion attention module. The benchmark also reveals limi-
tations in low-light areas across both visible and infrared
bands, suggesting a direction for future work.
Acknowledgement
This work was conducted during the author’s visit to the
University of Amsterdam, Netherlands (UvA), partly sup-
ported by the Chinese Scholarship Council (CSC). Addi-
tionally, this work was supported by the National Key R&D
Program of China (2024YFC3014900).
28088


<!-- PDF PAGE 9 -->

References
[1] Mahmoud Afifi, Marcus A Brubaker, and Michael S Brown.
Auto white-balance correction for mixed-illuminant scenes.
In Proceedings of the IEEE/CVF Winter Conference on Ap-
plications of Computer Vision, pages 1210–1219, 2022. 2
[2] Boaz Arad and Ohad Ben-Shahar. Sparse recovery of hyper-
spectral signal from natural rgb images. In European Con-
ference on Computer Vision , pages 19–34. Springer, 2016.
1
[3] Boaz Arad, Radu Timofte, Rony Yahel, Nimrod Morag,
Amir Bernat, Yuanhao Cai, Jing Lin, Zudi Lin, Haoqian
Wang, Yulun Zhang, et al. Ntire 2022 spectral recovery chal-
lenge and data set. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
863–881, 2022. 1, 6
[4] E Ben-Dor, S Chabrillat, and Jos ´e AM Demattˆe. Character-
ization of soil properties using reflectance spectroscopy. In
Fundamentals, sensor systems, spectral libraries, and data
mining for vegetation, pages 187–247. CRC press, 2018. 1,
3
[5] Gershon Buchsbaum. A spatial processor model for object
colour perception. Journal of the Franklin institute, 310(1):
1–26, 1980. 2, 5, 6, 7, 8
[6] Simon Carter, Andy Fisher, Bridget Gibson, John Marshall,
Ben Russell, and Ian Whiteside. Atomic spectrometry up-
date: review of advances in the analysis of metals, chemicals
and materials. Journal of Analytical Atomic Spectrometry ,
32(11):2068–2117, 2017. 1, 3
[7] Ayan Chakrabarti and Todd Zickler. Statistics of real-world
hyperspectral images. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
193–200. IEEE, 2011. 6
[8] Zhuoran Du, Shikui Wei, Ting Liu, Shunli Zhang, Xiaotong
Chen, Shiyin Zhang, and Yao Zhao. Exploring the applica-
bility of spectral recovery in semantic segmentation of rgb
images. IEEE Transactions on Multimedia, 2023. 1, 6
[9] Gamal M ElMasry and Shigeki Nakauchi. Image analysis
operations applied to hyperspectral images for non-invasive
sensing of food quality–a comprehensive review.Biosystems
engineering, 142:53–82, 2016. 3
[10] Marwa Farghaly, Romany F Mansour, and Adel A Sewisy.
Two-stage deep learning framework for srgb image white
balance. Signal, Image and Video Processing , 17(1):277–
284, 2023. 2
[11] LG Ferreira, H Yoshioka, A Huete, and EE Sano. Opti-
cal characterization of the brazilian savanna physiognomies
for improved land cover monitoring of the cerrado biome:
preliminary assessments from an airborne campaign over an
lba core site. Journal of Arid Environments, 56(3):425–447,
2004. 2
[12] Rafael C Gonzalez and Richard E Woods. Digital Image
Processing. Prentice Hall, 2nd edition, 2002. 4
[13] Shuai Hu, Feng Gao, Xiaowei Zhou, Junyu Dong, and Qian
Du. Hybrid convolutional and attention network for hyper-
spectral image denoising. IEEE Geoscience and Remote
Sensing Letters, 2024. 3, 4, 5, 6, 7, 8
[14] Furkan Kınlı, Do˘ga Yılmaz, Barıs ¸¨Ozcan, and Furkan Kırac ¸.
Deterministic neural illumination mapping for efficient auto-
white balance correction. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 1139–
1147, 2023. 2
[15] Furkan Kınlı, Do˘ga Yılmaz, Barıs ¸¨Ozcan, and Furkan Kırac ¸.
Modeling the lighting in scenes as style for auto white-
balance correction. In Proceedings of the IEEE/CVF Win-
ter Conference on Applications of Computer Vision , pages
4903–4913, 2023. 2
[16] George A Leshkevich. Goniometric measurements of a
spray-painted barium sulfate reference panel. Remote Sens-
ing of Environment, 24(2):287–296, 1988. 1, 3
[17] Chunyu Li, Yusuke Monno, and Masatoshi Okutomi. Spec-
tral mvir: Joint reconstruction of 3d shape and spectral re-
flectance. In 2021 IEEE International Conference on Com-
putational Photography (ICCP), pages 1–12. IEEE, 2021. 1,
3
[18] Chunxiao Li, Xuejing Kang, Zhifeng Zhang, and Anlong
Ming. Swbnet: a stable white balance network for srgb im-
ages. In Proceedings of the AAAI Conference on Artificial
Intelligence, pages 1278–1286, 2023. 2
[19] Miaoyu Li, Ji Liu, Ying Fu, Yulun Zhang, and Dejing Dou.
Spectral enhanced rectangle transformer for hyperspectral
image denoising. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
5805–5814, 2023. 3, 6, 7, 8
[20] Shouzhi Li, Liangliang Zhu, Luyan Ji, Xin Yi, Yongchao
Zhao, and Xiurui Geng. Hyperspectral image denoising
based on the similar spectra approaching. Infrared Physics
& Technology, 131:104667, 2023. 2
[21] Maria Losurdo, Michael Bergmair, Giovanni Bruno, De-
nis Cattelan, Christoph Cobet, Antonello De Martino,
Karsten Fleischer, Zorana Dohcevic-Mitrovic, Norbert
Esser, Melanie Galliet, et al. Spectroscopic ellipsometry
and polarimetry for materials and systems analysis at the
nanometer scale: state-of-the-art, potential, and perspectives.
Journal of Nanoparticle Research, 11:1521–1554, 2009. 1,
3
[22] Tomoaki Miura, Alfredo R Huete, Laerte Guima˜aes Ferreira,
Edson E Sano, and Hiroki Yoshioka. A technique for re-
flectance calibration of airborne hyperspectral spectrometer
data using a broad, multiband radiometer. In Hyperspectral
Remote Sensing of Tropical and Sub-tropical Forests, pages
213–232. CRC Press, 2008. 2
[23] Puria Azadi Moghadam, Neha Sharma, and Mohamed
Hefeeda. Enabling hyperspectral imaging in diverse illumi-
nation conditions for indoor applications. In Proceedings of
the 12th ACM Multimedia Systems Conference, pages 23–35,
2021. 2, 6, 7, 8
[24] Haodong Pan, Feng Gao, Junyu Dong, and Qian Du. Mul-
tiscale adaptive fusion network for hyperspectral image de-
noising. IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing, 16:3045–3059, 2023. 3
[25] Lorraine A Remer, Andrew E Wald, and Yoram J Kaufman.
Angular and seasonal variation of spectral surface reflectance
ratios: Implications for the remote sensing of aerosol over
28089


<!-- PDF PAGE 10 -->

land. IEEE Transactions on Geoscience and Remote Sens-
ing, 39(2):275–283, 2001. 2
[26] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-
net: Convolutional networks for biomedical image segmen-
tation. In Medical image computing and computer-assisted
intervention–MICCAI 2015: 18th international conference,
Munich, Germany, October 5-9, 2015, proceedings, part III
18, pages 234–241. Springer, 2015. 4
[27] Eric Ariel L Salas and Geoffrey M Henebry. A new approach
for the analysis of hyperspectral data: Theory and sensitivity
analysis of the moment distance method. Remote sensing, 6
(1):20–41, 2013. 3
[28] Specim, Spectral Imaging Ltd. Specim - Hyperspectral
Imaging Cameras and Systems. https://www.specim.
com/. 3
[29] Lucien Wald. Quality of high resolution synthesized images:
Is there a simple criterion? In Proceedings of the Third Con-
ference ”Fusion of Earth Data”, pages 99–103, 2000. 4
[30] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: From error visibility to
structural similarity. IEEE Transactions on Image Process-
ing, 13(4):600–612, 2004. 4
[31] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang
Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general
u-shaped transformer for image restoration. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 17683–17693, 2022. 4
[32] Fumihito Yasuma, Tomoo Mitsunaga, Daisuke Iso, and
Shree K Nayar. Generalized assorted pixel camera: post-
capture control of resolution, dynamic range, and spectrum.
IEEE transactions on image processing , 19(9):2241–2253,
2010. 6
[33] Richard H Yuhas, Alexander FH Goetz, and Joe W Board-
man. Discrimination among semi-arid landscape endmem-
bers using the spectral angle mapper (sam) algorithm. In
Summaries of the Third Annual JPL Airborne Geoscience
Workshop, pages 147–149, 1992. 4
[34] Qiang Zhang, Yaming Zheng, Qiangqiang Yuan, Meiping
Song, Haoyang Yu, and Yi Xiao. Hyperspectral image de-
noising: From model-driven, data-driven, to model-data-
driven. IEEE Transactions on Neural Networks and Learn-
ing Systems, 2023. 2
28090

<!-- END-OF-PAPER:559d12f3b934 -->
