<!-- PDF PAGE 1 -->

Degradation-Robust Fusion: An Efficient Degradation-Aware Diffusion
Framework for Multimodal Image Fusion in Arbitrary Degradation Scenarios
Yu Shi1 Yu Liu1* Zhong-Cheng Wu1 Juan Cheng1 Huafeng Li2 Xun Chen3
1Department of Biomedical Engineering, Hefei University of Technology
2Faculty of Information Engineering and Automation, Kunming University of Science and Technology
3School of Information Science and Technology, University of Science and Technology of China
yushi@mail.hfut.edu.cn yuliu@hfut.edu.cn
Abstract
Complex degradations like noise, blur, and low resolution
are typical challenges in real-world image fusion tasks,
limiting the performance and practicality of existing meth-
ods. End-to-end neural network–based approaches are gen-
erally simple to design and highly efficient in inference,
but their black-box nature leads to limited interpretabil-
ity. Diffusion-based methods alleviate this to some extent
by providing powerful generative priors and a more struc-
tured inference process. However, they are trained to learn
a single-domain target distribution, whereas fusion lacks
natural fused data and relies on modeling complementary
information from multiple sources, making diffusion hard
to apply directly in practice. To address these challenges,
this paper proposes an efficient degradation-aware diffu-
sion framework for image fusion under arbitrary degrada-
tion scenarios. Specifically, instead of explicitly predict-
ing noise as in conventional diffusion models, our method
performs implicit denoising by directly regressing the fused
image, enabling flexible adaptation to diverse fusion tasks
under complex degradations with limited steps. Moreover,
we design a joint observation model correction mechanism
that simultaneously imposes degradation and fusion con-
straints during sampling to ensure high reconstruction ac-
curacy. Experiments on diverse fusion tasks and degrada-
tion configurations demonstrate the superiority of the pro-
posed method under complex degradation scenarios. Code:
https://github.com/YShi-cool/DRFusion.
1. Introduction
With the rapid development of multimodal imaging tech-
nologies in fields such as medical imaging, industrial vision,
and video surveillance, image fusion techniques have be-
come critical for enhancing image quality and improving vi-
*Corresponding author: Yu Liu.
Challenging!
...
×(T -1)
(a) Neural network-based methods (b) Diffusion model-based methods
(c)  Proposed efficient degradation-aware diffusion framework
Neural 
Networks
Loss Function
Source images
Fusion image
Ground truth
Not exist
Denoising Network
Degradation 
Model
Loss Function
High-quality source image
High-quality fusion image
Degradation-aware 
dIffusion model
Figure 1. Comparison of fusion strategies under different degrada-
tion scenarios: (a) methods based on neural networks; (b) existing
diffusion-based methods; (c) the proposed degradation-aware dif-
fusion framework from this work.
sual understanding. However, most existing fusion methods
assume that source images are of high quality, which over-
looks the prevalent degradation phenomena in real-world
imaging environments, such as noise, motion blur, and in-
sufficient resolution. These degradation factors introduce
distortions and result in the loss of critical information, sig-
nificantly limiting both the accuracy of the fusion results
and the practical robustness of the system.
The traditional “restoration + fusion” paradigm makes
the fusion result highly dependent on restoration quality,
while its decoupled design can introduce error accumula-
tion across stages and complicate deployment. These issues
motivate designing restoration and fusion within a unified
framework [11, 16, 54]. As illustrated in Fig. 1 (a), end-to-
end neural network–based fusion methods construct a para-
metric mapping from multi-source inputs to a fusion output,
which is optimized using loss functions that jointly con-
strain restoration and fusion quality [4, 9, 12]. These meth-
ods are usually simple to design and computationally effi-
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
33848


<!-- PDF PAGE 2 -->

cient. However, their black-box nature limits interpretabil-
ity, and the achievable reconstruction accuracy is highly de-
pendent on the choice and design of the loss functions.
Diffusion models [5, 7, 8], a class of generative mod-
els, have demonstrated exceptional performance in image
generation and reconstruction tasks due to their superior
distribution modeling capabilities and stability. Unlike the
black-box mapping of deep neural networks, diffusion mod-
els offer superior interpretability and higher fusion accuracy
through their solid theoretical foundations. This iterative
mechanism is highly beneficial to image fusion, making the
progressive aggregation of multimodal information trans-
parent. Additionally, such step-wise refinement inherently
guarantees the stability of the fusion process. However,
the following inherent limitations of diffusion models make
it difficult to apply them to image fusion under degraded
conditions. First, diffusion models are trained to learn a
target data distribution, whereas image fusion lacks natu-
rally available fused data to support such training [37, 52],
as shown in Fig. 2 (b). Second, standard diffusion mod-
els operate on a single-domain distribution, whereas fusion
requires modeling complementary information from multi-
ple sources, which calls for an additional formulation that
explicitly links cross-modal information, fusion objectives,
and the probabilistic model [61]. In addition, the computa-
tional cost of iterative sampling cannot be ignored. Existing
methods either address only specific degradations [48] or
rely on independently pre-trained restoration models [33],
yet they still do not provide an effective and flexible frame-
work capable of handling diverse fusion tasks under com-
plex degradation scenarios.
To address these challenges, this paper proposes a
degradation-aware diffusion framework that unifies restora-
tion and fusion for multimodal image fusion under arbitrary
degradations. In contrast to the explicit noise-prediction
paradigm in standard diffusion models, we adopt a direct
fused-image regression formulation that implicitly encodes
the denoising process, enabling the model to flexibly handle
diverse fusion tasks under complex degradation conditions
like the conventional end-to-end frameworks. By remov-
ing explicit noise prediction, the framework can achieve
competitive results within a limited number of diffusion
steps, leading to a substantial improvement in inference ef-
ficiency. Moreover, we design a joint observation correc-
tion mechanism that injects both degradation and fusion
constraints into the diffusion sampling process, forcing in-
termediate samples to remain aligned with the degradation
model while preserving complementary cross-modal infor-
mation, thereby ensuring high reconstruction accuracy of
the fusion results. Our contributions can be summarized as
follows:
• We propose an efficient degradation-aware diffusion
framework that jointly models degradation and multi-
modal image fusion within a single process. By directly
regressing the fused image instead of predicting noise,
our approach flexibly adapts to diverse tasks and achieves
competitive performance in few diffusion steps.
• We design a joint observation correction mechanism that
injects both degradation and fusion constraints into the
diffusion sampling process, enforcing consistency with
the degradation model while preserving complementary
cross-modal information to improve the reconstruction
accuracy of the fusion results.
• We conduct extensive experiments across diverse fusion
tasks and complex degradation scenarios, demonstrating
superior quantitative and visual performance.
2. Related Work
2.1. Multimodal Image Fusion
Based on their decomposition approaches, early image
fusion methods primarily rely on multi-scale transforms
[10, 24] and sparse representation [14, 23, 25]. In recent
years, deep learning (DL)-based methods [26, 53, 63], have
gradually replaced traditional methods to become the main-
stream in research. With the evolution of DL, these methods
have progressed from convolutional neural network (CNN)-
based approaches [1, 21, 45] to attention-based architec-
tures, including Transformer [28, 60, 62] and the more re-
cent Mamba models [2, 44, 64]. Building upon these archi-
tectural advances, recent studies have shifted focus toward
task-oriented enhancements. For example, to improve the
practicality and ease of deployment of fusion models, sev-
eral unified frameworks have been proposed [20, 40]. In
addition, some approaches integrate downstream tasks into
the fusion process to enhance the overall applicability and
effectiveness of the method [22, 27, 51]. To address com-
mon degradation issues that may occur during the imaging
process, such as noise, low resolution, and imprecise align-
ment, some studies have integrated degradation modeling
and image fusion into a single framework [13, 16, 29].
Due to their strong generative capabilities, robust gen-
eralization, and structured inference, diffusion models are
increasingly popular in image fusion [30, 50, 52]. While di-
rectly applicable to supervised tasks [15], diffusion models
struggle in practical, unsupervised fusion scenarios due to
their reliance on target data distributions. To address this
challenge, existing methods have proposed several strate-
gies, such as incorporating task-specific mathematical for-
mulations into the diffusion framework [31, 49, 61], design-
ing dedicated decoders to reconstruct fusion results from
intermediate features [55], or generating pseudo labels to
guide the training [32, 48]. Despite recent advances, cur-
rent diffusion-based fusion methods remain constrained by
standard frameworks, limiting their effectiveness in com-
plex scenarios like joint restoration and fusion.
33849


<!-- PDF PAGE 3 -->

Figure 2. The proposed framework for multimodal image fusion under various degradation scenarios in this work.
2.2. Restoration Oriented Diffusion Models
Diffusion models have emerged as a powerful paradigm for
image restoration tasks, including denoising [41], deblur-
ring [43], and super-resolution [36].Compared to existing
algorithms [17, 18], their iterative denoising and strong gen-
erative priors offer greater flexibility in modeling complex
data distributions. Fei et al. proposed generative diffu-
sion prior [6], demonstrating how a pre-trained diffusion
model can serve as a unified unsupervised prior for mul-
tiple restoration and enhancement tasks. In the context of
super-resolution, SinSR [39] further shows that diffusion-
based methods can approach or surpass state-of-the-art per-
formance while requiring fewer inference steps. For deblur-
ring, DDNM [38] incorporates a pseudoinverse-based con-
sistency step that embeds the blur model into each denois-
ing iteration, keeping samples aligned with the degraded in-
put. However, unlike single-image restoration, fusion tasks
must integrate complementary information from multiple
degraded sources, necessitating more elaborate designs be-
yond standard pipelines.
3. Method
3.1. Fusion-Oriented Diffusion Framework
In standard diffusion models as shown in Supplementary
Material A, a noise prediction network is first pre-trained
so that it can estimate the noise injected at any diffusion
timestept. The noise added at timesteptand the corre-
sponding noisy image can typically be obtained as follows:
p(xt|x0) =N(x t; √¯αt ·x 0,(1−¯αt)·I),(1)
whereα t = 1−β t and¯αt = Qt
i=1 αt,β 1, β2, ..., βT ∈
[0,1)denote a set of hyperparameters that control the vari-
ance schedule over diffusion timesteps. To facilitate train-
ing, using the reparameterization trick, the above equation
can be rewritten as:
p(xt|x0) =N(x t; √¯αt ·x 0,(1−¯α t)·I).(2)
To achieve accurate noise estimation, the network typically
needs to be trained for a large number of iterations, and the
diffusion periodTis usually set to be relatively large, which
further increases the overall computational cost. This archi-
tectural design requires access to the target data distribu-
tion during training, effectively restricting diffusion models
to supervised settings where clean target samples are avail-
able. Moreover, they are typically trained to model a single-
domain distribution, making it nontrivial to directly extend
them to multi-input tasks such as multimodal image fusion.
To address the above issues, inspired by [3], we propose
a new diffusion framework specifically designed for image
fusion, as illustrated in Fig. 2. We discard the explicit noise-
prediction step in standard diffusion models and retain only
the reverse process, directly mapping the inputs to the fu-
sion output through a limited number of diffusion iterations.
Fθ =f T
θ →f T−1
θ → · · ·f 0
θ ,(3)
whereF θ denotes the proposed diffusion framework,f t
θ de-
notes the diffusion iteration at timestept. At each diffu-
sion stepf t
θ, an accelerated DDIM-based sampling scheme
is employed:
ˆx0|t =ˆ xt −
√
1−¯αtεθ(ˆ xt, t),(4)
ˆ xt−1 = √¯αt−1ˆ x0|t +
p
1−¯αt−1εθ(ˆ xt, t).(5)
This design makes the diffusion model closer in architec-
ture to an end-to-end neural network architecture and brings
several advantages. First, the direct input-to-output map-
ping allows the proposed method to handle fusion in a self-
supervised manner, alleviating the lack of fusion labels.
Second, multi-input fusion tasks can be jointly optimized
within a single framework by imposing appropriate loss
33850


<!-- PDF PAGE 4 -->

constraints on the fusion output with respect to each source
input. Third, since the model no longer predicts noise ex-
plicitly but hides it in intermediate representations, it can
be combined with accelerated samplers such as DDIM to
obtain high-quality results within a limited number of dif-
fusion steps, thereby improving inference efficiency.
3.2. Joint Observation Correction Mechanism
Although diffusion models are capable of producing high-
quality results, their sampling process is inherently stochas-
tic. This randomness is most pronounced when using stan-
dard stochastic samplers like DDPM. In contrast, acceler-
ated schemes such as DDIM can yield nearly determinis-
tic sampling trajectories. Nevertheless, it’s crucial to re-
member that the underlying model is still learning a data
distribution, not a single deterministic mapping. There-
fore, some works introduce additional constraints into the
DDIM sampling process to guide the generation towards
solutions that are better aligned with the desired task objec-
tives [3, 38]. Motivated by these constraint-based strategies
but aiming for a more unified and flexible solution, we pro-
pose a degradation-aware diffusion framework tailored for
multimodal image fusion under complex degradations.
Degradation-Aware Diffusion Correction.The classi-
cal image degradation model can be formulated as follows:
y=AX+n,(6)
wherey∈R d×1 denotes the observed image,A∈R d×D
is the degradation matrix (operator),X∈R D×1 represents
the underlying clean image, andn∈R d×1 is the noise
term. We temporarily ignore the noise term. In the previous
diffusion sampling step, we obtain a denoised image esti-
mate ˆx0|t, which, however, does not necessarily satisfy the
above observation constrainty=AX. Therefore, starting
from the estimate obtained in the previous diffusion step, we
aim to find a new solution that satisfies the above constraint,
which requires performing a projection onto the constraint
set:
x⋆ = arg min
z
||z−x 0|t||2 s.t.Az=y.(7)
Geometrically, this is equivalent to projecting the pointx0|t
onto the subspace defined byAz=y. The solution to this
problem is:
x⋆ =x−A †(Ax−y),(8)
whereA † is the Moore–Penrose pseudoinverse,r=Ax−y
is defined as the residual andδ=A †ras the correction
term. Therefore, the projection step is given by subtracting
the correction from the basic solutionx⋆ =x 0|t−δ. For tra-
ditional restoration tasks, the above formulation is straight-
forward to solve, since every term is known. However, for
the image fusion problem under complex degradation con-
ditions considered in this work, the situation becomes much
more complicated. Existing methods mainly address single-
image degradation, whereas fusion involves multiple inputs.
In addition, the fusion image does not have a corresponding
degraded observation. These factors together pose signifi-
cant challenges for our solution (see Supplementary Mate-
rial B for detailed analysis).
Joint Observation Model.This paper proposes a joint
observation model to address the above problems. First,
we rewrite the multiple inputs in the form of a joint vari-
able[X 1,X 2,X f], and the degradation constraints of the
two source images together with the fusion constraint can
be expressed as follows:
y1 =A 1X1,(9)
y2 =A 2X2,(10)
Xf =W 1 ∗X 1 +W 2 ∗X 2.(11)
By movingX f in Eq. (11) to the right-hand side, the origi-
nal position of the fused image is replaced by a zero matrix,
which indicates that we do not need to obtain the fused im-
age in advance. The above model can be reformulated in
the following joint matrix form:


y1
y2
0

 =


A1 0 0
0A 2 0
−W1 −W 2 I




X1
X2
Xf

 ,(12)
whereA 1 andA 2 denote the degradation operators asso-
ciated with the two source images,W 1 andW 2 are the
fusion operators, andIis the identity matrix. The above
joint observation model not only overcomes the limitation
of previous formulations that can only be applied to single-
image restoration but also eliminates the need to explicitly
obtain the fusion observation, further enabling the simul-
taneous execution of both restoration and fusion under a
unified framework. The resulting joint degradation matrix
takes the form of a block matrix:
ˆA=


A1 0 0
0A 2 0
−W1 −W 2 I

 .(13)
However, when applying it to the constrained procedure in
Eq. (8), one issue remains: the pseudoinverse of the joint
degradation matrix is difficult to compute. Although there
are many ways to obtain a pseudoinverse, explicitly com-
puting it is clearly impractical, as it would incur prohibitive
computational cost and memory usage. Therefore, we pro-
pose an equation-based approach to compute the pseudoin-
verse implicitly by solving the corresponding linear system.
By solving Eqs. (9)–(11) separately and then arranging the
resulting solutions into a joint matrix in the same form as
Eq. (12), the following formula can be obtained:


X1
X2
Xf

 =


A1
† 0 0
0A 2
† 0
W1A1
† W2A2
† I




y1
y2
0

 .(14)
33851


<!-- PDF PAGE 5 -->

In this way, we obtain a pseudoinverse of the joint degrada-
tion matrix ˆAthat satisfies the Moore–Penrose conditions:
ˆA† =


A1
† 0 0
0A 2
† 0
W1A1
† W2A2
† I

 .(15)
By substituting Eq. (15) into Eq. (8), we obtain the final
joint observation constraint, which is then embedded into
the DDIM framework in Eqs. (4)–(5) to derive the final
diffusion iteration form:
ˆx0|t =ˆ xt −
√
1−¯αtεθ(ˆ xt, t),(16)
¯x0|t =ˆ x0|t − ˆA†(ˆAˆ x0|t −y),(17)
ˆ xt−1 = √¯αt−1¯ x0|t +
p
1−¯αt−1εθ(ˆ xt, t).(18)
By sequentially performing the above three stages, we ob-
tain a degradation-aware diffusion framework that is specif-
ically designed for image fusion. It is worth noting that,
to achieve better fusion performance, we do not adopt fixed
fusion weights, instead, we learn them in a data-driven man-
ner. Specifically, as illustrated in Fig. 2, the noise predictor
is designed as a multi-task architecture that, in addition to
predicting the noise, also outputs a weight mapW 1. We
then enforceW 1 +W 2 =1to obtain the two complemen-
tary fusion weights. The overall workflow of our proposed
method is summarized in Algorithm 1 of the supplementary
material.
Managing More Complex Degradation Scenarios.
The above discussion is based on the noise-free case. For
the noisy setting, we only need to modify Eq. (17) as fol-
lows:
¯ x0|t =ˆ x0|t −Σ t ˆA†(ˆAˆ x0|t −y),(19)
whereΣ t ∈R D×D is used to scale the correction term in
order to reduce the influence of noise [38]. Moreover, for
more complex degradation scenarios, such as combinations
of noise, blur, and low resolution simultaneously present in
both source images, the proposed joint observation model
can still handle them effectively. For instance, if source im-
age is subjected to several types of composite degradation
A=A 1A2 · · ·A n, its corresponding pseudoinverse oper-
ator can be written accordinglyA † =A †
nA†
n−1 · · ·A †
1 and
directly substituted into Eq. (15) to obtain a valid solution,
which is both concise and efficient (see Supplementary Ma-
terial C for detailed analysis).
3.3. Loss Function
Unlike previous diffusion based fusion methods, the pro-
posed framework no longer explicitly predicts the noise but
directly regresses the reconstructed image. This design al-
lows us to adopt commonly used unsupervised losses in
the image fusion literature to accomplish the above task.
Specifically, the overall loss function consists of two com-
ponents: a reconstruction loss on the source images and a
fusion loss:
Ltotal =L rec +λL f ,(20)
whereL rec is defined as:
Lrec =||X 1 − ¯X1||1 +||X 2 − ¯X2||1,(21)
whereX 1 andX 2 represent the two reconstructed source
images, ¯X1 and ¯X2 denote the labels of the two source im-
ages, and|| · || 1 denotes theL1norm. For different fusion
tasks, we adopt different fusion losses. In the case of in-
frared–visible image fusion, the loss is defined as follows:
Lrec =||X 1 − ¯X1||1 +||X 2 − ¯X2||1.(22)
For different fusion tasks, we adopt different fusion losses.
In the case of infrared–visible image fusion, the loss is de-
fined as follows:
Lf =||X f −max( ¯X1, ¯X2)||1
+γ||∇X f −max(∇ ¯X1,∇ ¯X2)||1, (23)
where∇denotes the gradient calculation operation,γis
the hyperparameter trade-off. For the medical image fusion
problem, the fusion loss is defined as follows:
Lf =
2X
i=1
||Xf − ¯Xi||1 +ϕ(1−SSIM(X f , ¯Xi)),(24)
whereSSIM(·)represents structural similarity calculation,
ϕis a hyperparameter. The proposed framework flexibly
combines different loss terms according to the fusion task at
hand, enabling task-oriented optimization instead of being
tied to a single noise-prediction objective. This breaks the
rigidity of traditional diffusion-based fusion frameworks.
4. Experiments
4.1. Setup
Implementation Details.The proposed model is imple-
mented in PyTorch and optimized using the Adam opti-
mizer. The batch size is set to 8, and the model is trained
for a total of 100 epochs. The initial learning rate is fixed to
0.0001 and then decayed using a multi-step scheduler. The
hyperparameterλis set to 10, whileγandϕare set to 20
and 10, respectively. All experiments were implemented on
a computer equipped with dual NVIDIA RTX 4090 GPUs.
Datasets.We evaluate the proposed method on the
infrared–visible image fusion dataset M3FD [19] and on
PET–MRI pairs from the Harvard medical image fusion
dataset. Three degradation scenarios are considered: noise,
blur, and a more challenging composite degradation case. In
the first two scenarios, we assume that both source images
33852


<!-- PDF PAGE 6 -->

IR VIS IFCNN U2Fusion MURF DDFM Text-Difuse VDMUFusion RFfusion
Noise
Blur
Composite 
degradation
Mask-DiFuser Ours
Figure 3. Qualitative results of different fusion methods on M3FD dataset. For the comparison methods, we first use corresponding
restoration algorithms (e.g., denoising, deblurring, super-resolution) to restore the images, and then apply the respective fusion methods.
The proposed method directly reconstructs the fusion results from the degraded source images. In the third degradation scenario, the
infrared image is of low resolution and we have enlarged it, the original image is shown in the top-left corner.
Table 1. Objective fusion metrics of various methods under different degradation scenarios on the M3FD dataset (Bold and gray back-
ground: best result in each column; underline: second best).
Methods Noise Blur Composite degradation
QM I QN CIEQAB/F QP QCB QW QM I QN CIEQAB/F QP QCB QW QM I QN CIEQAB/F QP QCB QW
IFCNN 0.2205 0.8036 0.4056 0.2115 0.4015 0.77780.3378 0.8051 0.1042 0.10570.3491 0.49690.3176 0.8048 0.0965 0.0571 0.3635 0.4708
U2Fusion0.2454 0.80390.42810.18170.48040.72720.2891 0.8045 0.1389 0.0696 0.42050.50350.2817 0.8044 0.1215 0.04910.44010.5507
MURF 0.2270 0.8038 0.2630 0.0975 0.4383 0.59070.3335 0.8054 0.1413 0.0793 0.3966 0.46070.3100 0.8050 0.1159 0.0416 0.40930.4817
DDFM 0.2685 0.8043 0.3631 0.1543 0.45300.77470.3320 0.8054 0.1022 0.0631 0.3277 0.42240.3202 0.8052 0.0885 0.0488 0.3488 0.4443
Text-DiFuse0.2896 0.8044 0.2109 0.0576 0.3807 0.34020.2905 0.8044 0.1195 0.0493 0.3350 0.25890.2929 0.8044 0.1103 0.0462 0.3381 0.2928
VDMUFusion0.2367 0.8039 0.3002 0.1580 0.4198 0.71050.3381 0.8053 0.0920 0.0712 0.3191 0.38600.3318 0.80520.0791 0.0474 0.3381 0.3985
RFfusion0.30210.80490.2831 0.1217 0.4147 0.69590.37710.80650.0972 0.0564 0.3855 0.38840.36460.8043 0.0836 0.0371 0.3660 0.3822
Mask-DiFuser0.2343 0.8039 0.33430.21610.4050 0.72710.2572 0.8042 0.21140.0860 0.3850 0.54030.2492 0.8041 0.19860.06110.3686 0.5845
Ours 0.35050.80520.40830.18250.42060.78100.44770.80680.36980.16710.45670.72330.37320.80550.21990.07550.37900.6237
are affected by the same type of degradation. In the com-
posite case for infrared–visible fusion, the infrared image is
corrupted by noise, blur, and low resolution, while the vis-
ible image suffers only from noise and blur. For PET–MRI
fusion, the PET image is subjected to all three degradations,
whereas the MRI image is degraded by noise and blur only.
Metrics and SOTA Competitors.We adopt six widely
used objective metrics for image fusion to evaluate the per-
formance of the proposed method, includingQM I,Q N CIE ,
QAB/F ,Q P ,Q CB , andQ W . In addition, we compare
against eight representative fusion approaches, namely the
CNN based methods IFCNN [59], U2Fusion [46], and
MURF [47], as well as the diffusion based methods DDFM
[61], Text-DiFuse [57], VDMUFusion [31], RFfusion [42]
and Mask-DiFuser [34]. For different degradation scenar-
ios, we first apply existing image restoration algorithms cor-
responding to each type of degradation, including denoising
[35], deblurring [56], and super-resolution [58] methods, to
recover the source images. The restored images are then
fused using the aforementioned competing methods. For
the composite degradation scenario, we sequentially apply
these restoration algorithms in the above order to recon-
struct the source images before fusion.
4.2. Infrared and Visible Image Fusion
Qualitative Comparison.The qualitative results for the in-
frared–visible image fusion task are presented in Fig. 3. For
each degradation scenario, we show two representative ex-
amples and provide a zoomed-in patch in the bottom-right
corner to better inspect local details. It can be observed that
cascaded restoration and fusion pipelines fail to fully elimi-
nate degradations, leaving residual artifacts such as blurred
structures and detail loss. In contrast, our proposed method
directly reconstructs high-quality fused images from de-
graded inputs. As shown in the visual comparisons, our ap-
proach clearly demonstrates superior preservation of color
fidelity and detail information. Specifically, it effectively
avoids the color distortion introduced by competing meth-
33853


<!-- PDF PAGE 7 -->

MRI PET IFCNN U2Fusion MURF DDFM Text-Difuse VDMUFusion Ours
Noise
Blur
Composite
degradation
RFfusion Mask-DiFuser
Figure 4. Qualitative comparison of different fusion methods on the Harvard dataset. For the comparison methods, we first apply the
corresponding restoration algorithms (e.g., denoising, deblurring, super-resolution) to process the images, and then perform fusion. The
proposed method directly reconstructs the fusion results from the degraded source images. In the third degradation scenario, the PET image
is of low resolution and we have enlarged it, the original image is shown in the top-left corner.
Table 2. Objective fusion metrics of various methods under different degradation scenarios on the Harvard dataset (Bold and gray back-
ground: best result in each column; underline: second best).
Methods Noise Blur Composite degradation
QM I QN CIEQAB/F QP QCB QW QM I QN CIEQAB/F QP QCB QW QM I QN CIEQAB/F QP QCB QW
IFCNN 0.5164 0.8067 0.3902 0.1791 0.4611 0.76150.6472 0.8073 0.2692 0.1613 0.5891 0.67520.5231 0.8065 0.1853 0.0515 0.4613 0.5999
U2Fusion0.5830 0.8069 0.3379 0.1658 0.4095 0.68270.6505 0.8074 0.2373 0.1523 0.3587 0.57600.5601 0.8067 0.1761 0.0628 0.4451 0.5374
MURF 0.4456 0.8061 0.3199 0.1380 0.3484 0.64310.6061 0.8072 0.3003 0.1273 0.3615 0.69360.4349 0.8059 0.1771 0.0447 0.3757 0.5462
DDFM 0.6025 0.8075 0.3840 0.19080.60780.75320.66530.8077 0.2507 0.16900.5810 0.62080.63530.8071 0.1971 0.07150.58780.6032
Text-DiFuse0.5324 0.8074 0.4026 0.1916 0.2982 0.68100.5547 0.8073 0.2839 0.1215 0.3256 0.58200.5382 0.8073 0.1810 0.0773 0.2458 0.4893
VDMUFusion0.5737 0.8074 0.3451 0.1858 0.3617 0.72450.65170.80790.2138 0.1450 0.4642 0.58250.5438 0.8071 0.1773 0.0635 0.3720 0.5611
RFfusion0.60850.80760.2525 0.1235 0.51130.55320.5831 0.8075 0.1917 0.1182 0.59200.41300.5931 0.8055 0.1508 0.0577 0.4635 0.4115
Mask-DiFuser0.5500 0.8075 0.42100.20650.3418 0.76020.5998 0.8076 0.29140.1621 0.3029 0.65910.5151 0.80720.21140.06500.3233 0.6148
Ours 0.61710.80780.42580.24360.45950.78920.6469 0.80770.38550.20160.62810.78850.60190.80730.29560.12580.48730.7344
ods like Mask-DiFuser (e.g., the tree regions in the first row)
and extracts more precise fine details (e.g., the text and car
regions in the third and last rows). Finally, both the thermal
radiation from infrared images and the rich textures from
visible images are optimally preserved in our outputs.
Quantitative Comparison.The corresponding quan-
titative results are summarized in Table 1. It can be ob-
served that the proposed method achieves the best or near-
best performance on most metrics across all three degra-
dation scenarios. In particular, under the more challeng-
ing deblurring and composite degradation settings, our ap-
proach significantly outperforms existing methods in terms
ofQ M I QN CIE ,Q AB/F ,Q P , andQ W . These results
demonstrate that the proposed degradation-aware diffusion
framework not only effectively mitigates the adverse effects
of noise, blur, and resolution mismatch, but also delivers
higher-quality fusion in terms of information preservation
and structural detail reconstruction.
4.3. Medical Image Fusion
Qualitative Comparison.Fig. 4 shows the qualitative re-
sults for the proposed PET-MRI fusion method across dif-
ferent degradation scenarios. Although restoration algo-
rithms reduce some degradation effects, they cannot com-
pletely eliminate issues such as blurring and detail loss. Ex-
isting fusion methods are also unable to fully optimize fu-
sion results under these conditions. In contrast, the pro-
posed method effectively reconstructs high-quality fused
33854


<!-- PDF PAGE 8 -->

images, preserving both the PET’s metabolic information
and the MRI’s structural details. The results demonstrate
that the proposed approach outperforms the competing
methods in terms of both information fidelity and structural
clarity, even in the presence of degradation.
Figure 5. Comparison of time efficiency and model parameters
across different fusion methods.
Quantitative Comparison.The quantitative results,
summarized in Table 2, demonstrate that the proposed
method outperforms existing fusion approaches across all
degradation scenarios. Specifically, our method achieves
the highest scores in terms ofQ AB/F ,Q P , andQ W in
all three degradation conditions (denoising, deblurring, and
composite degradation). In particular, under the more
challenging composite degradation scenario, the proposed
method shows significant improvements in metrics such as
QAB/F andQ P , highlighting its robustness in handling
complex degradations. These results validate the effective-
ness of the proposed degradation-aware diffusion frame-
work in maintaining high fusion quality while preserving
both structural and informational integrity.
4.4. Efficiency and Model Parameters Analysis
The time efficiency and model parameters analysis is shown
in Fig. 5. Neural network-based methods, such as IFCNN
and U2Fusion, exhibit fast run times, making them highly
efficient for tasks requiring speed. However, diffusion-
based methods like DDFM, Text-DiFuse, and VDMUFu-
sion generally have higher time costs due to their itera-
tive nature, resulting in slower performance. Our proposed
method strikes a balance by significantly improving time
efficiency compared to traditional diffusion models. Al-
though it cannot match the speed of neural networks, it is
much faster than conventional diffusion approaches, mak-
ing it a competitive choice for tasks requiring both quality
and efficiency. Despite being slower than neural networks,
our method offers a good trade-off between computational
cost and fusion accuracy. Furthermore, regarding the num-
ber of parameters, the proposed model delivers strong per-
formance without the need for a highly complex architec-
ture, and it is not particularly demanding in terms of com-
putational resources.
4.5. Ablation Experiments
In the ablation study, we compared the performance of
the model with and without the joint constraint across two
datasets and various degradation scenarios. The results,
shown in Fig. 6, demonstrate that adding the joint con-
straint significantly improves the performance in most of
the evaluation metrics, includingQ M I,Q N CIE ,Q AB/F ,
QP ,Q CB , andQ W . The method with the joint constraint
outperforms the unconstrained model, especially in terms of
reconstruction accuracy and detail preservation. This high-
lights the effectiveness of the joint constraint in enhancing
the fusion quality and recovery of fine details, proving its
importance in handling complex degradation scenarios. De-
tails of the ablation experiments and the performance vali-
dation on downstream tasks can be found in Supplementary
Material D and E.
(a)
 (b)
Figure 6. Ablation study results on two datasets. (a) represents the
M3FD dataset, and (b) represents the Harvard dataset.
5. Conclusion
In this work, we propose a degradation-aware diffusion
framework for image fusion that effectively addresses vari-
ous degradation scenarios, such as noise, blur, and low res-
olution. Extensive qualitative and quantitative evaluations
demonstrate that our method outperforms existing fusion
techniques, particularly in terms of preserving fine details
and structural integrity in the fused images. While tra-
ditional diffusion-based methods are typically slower than
neural network-based approaches, our framework achieves
a notable improvement in time efficiency, approaching the
speed of neural networks while maintaining superior fu-
sion quality. This makes our approach a highly competi-
tive solution for applications requiring both computational
efficiency and high-fidelity image fusion. Furthermore, this
work provides a novel perspective on diffusion-based im-
age fusion, offering a more efficient and accessible way to
leverage diffusion models for image fusion tasks.
33855


<!-- PDF PAGE 9 -->

Acknowledgement
This work was supported in part by the National Natural
Science Foundation of China under Grant 62576132 and
Grant U23A20294, in part by the Fundamental and Inter-
disciplinary Disciplines Breakthrough Plan of the Ministry
of Education of China under Grant JYB2025XDXM109,
and in part by the Yunnan Fundamental Research Projects
under Grants 202301A V070004 and 202501AS070123.
References
[1] Mostafa Amin-Naji, Ali Aghagolzadeh, and Mehdi Ezoji.
Ensemble of cnn for multi-focus image fusion.Information
Fusion, 51:201–214, 2019. 2
[2] Zihan Cao, Xiao Wu, Liang-Jian Deng, and Yu Zhong. A
novel state space model with local enhancement and state
sharing for image fusion. InProceedings of the 32nd ACM
International Conference on Multimedia, pages 1235–1244,
2024. 2
[3] Bin Chen, Zhenyu Zhang, Weiqi Li, Chen Zhao, Jiwen Yu,
Shijie Zhao, Jie Chen, and Jian Zhang. Invertible diffusion
models for compressed sensing.IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, 2025. 3, 4
[4] Jun Chen, Wei Yu, Xin Tian, Jun Huang, and Jiayi Ma. Mdb-
fusion: a visible and infrared image fusion framework ca-
pable for motion deblurring. In2024 IEEE International
Conference on Image Processing (ICIP), pages 1019–1025.
IEEE, 2024. 1
[5] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu,
and Mubarak Shah. Diffusion models in vision: A survey.
IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, 45(9):10850–10869, 2023. 2
[6] Ben Fei, Zhaoyang Lyu, Liang Pan, Junzhe Zhang, Weidong
Yang, Tianyue Luo, Bo Zhang, and Bo Dai. Generative dif-
fusion prior for unified image restoration and enhancement.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 9935–9946, 2023. 3
[7] Chunming He, Yuqi Shen, Chengyu Fang, Fengyang Xiao,
Longxiang Tang, Yulun Zhang, Wangmeng Zuo, Zhenhua
Guo, and Xiu Li. Diffusion models in low-level vision: A
survey.IEEE Transactions on Pattern Analysis and Machine
Intelligence, 2025. 2
[8] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffu-
sion probabilistic models.Advances in Neural Information
Processing Systems, 33:6840–6851, 2020. 2
[9] Jingxue Huang, Xiaosong Li, Haishu Tan, Lemiao Yang,
Gao Wang, and Peng Yi. Dednet: Infrared and visible image
fusion with noise removal by decomposition-driven network.
Measurement, 237:115092, 2024. 1
[10] Guofa Li, Yongjie Lin, and Xingda Qu. An infrared and
visible image fusion method based on multi-scale transfor-
mation and norm optimization.Information Fusion, 71:109–
129, 2021. 2
[11] Huafeng Li, Xiaoge He, Dapeng Tao, Yuanyan Tang, and
Ruxin Wang. Joint medical image fusion, denoising and
enhancement via discriminative low-rank sparse dictionaries
learning.Pattern Recognition, 79:130–146, 2018. 1
[12] Huafeng Li, Yueliang Cen, Yu Liu, Xun Chen, and Zhengtao
Yu. Different input resolutions and arbitrary output resolu-
tion: A meta learning-based deep framework for infrared and
visible image fusion.IEEE Transactions on Image Process-
ing, 30:4070–4083, 2021. 1
[13] Huafeng Li, Zengyi Yang, Yafei Zhang, Wei Jia, Zheng-
tao Yu, and Yu Liu. Mulfs-cap: Multimodal fusion-
supervised cross-modality alignment perception for unreg-
istered infrared-visible image fusion.IEEE Transactions
on Pattern Analysis and Machine Intelligence, 47(5):3673–
3690, 2025. 2
[14] Liangliang Li, Yan Shi, Ming Lv, Zhenhong Jia, Minqin Liu,
Xiaobin Zhao, Xueyu Zhang, and Hongbing Ma. Infrared
and visible image fusion via sparse representation and guided
filtering in laplacian pyramid domain.Remote Sensing, 16
(20):3804, 2024. 2
[15] Mining Li, Ronghao Pei, Tianyou Zheng, Yang Zhang, and
Weiwei Fu. Fusiondiff: Multi-focus image fusion using de-
noising diffusion probabilistic models.Expert Systems with
Applications, 238:121664, 2024. 2
[16] Xiaosong Li, Fuqiang Zhou, and Haishu Tan. Joint image fu-
sion and denoising via three-layer decomposition and sparse
representation.Knowledge-Based Systems, 224:107087,
2021. 1, 2
[17] Xingyuan Li, Jinyuan Liu, Zhixin Chen, Yang Zou, Long
Ma, Xin Fan, and Risheng Liu. Contourlet residual for
prompt learning enhanced infrared image super-resolution.
InEuropean Conference on Computer Vision, pages 270–
288. Springer, 2024. 3
[18] Xingyuan Li, Zirui Wang, Yang Zou, Zhixin Chen, Jun Ma,
Zhiying Jiang, Long Ma, and Jinyuan Liu. Difiisr: A diffu-
sion model with gradient guidance for infrared image super-
resolution. InProceedings of the Computer Vision and Pat-
tern Recognition Conference, pages 7534–7544, 2025. 3
[19] Jinyuan Liu, Xin Fan, Zhanbo Huang, Guanyao Wu, Risheng
Liu, Wei Zhong, and Zhongxuan Luo. Target-aware dual
adversarial learning and a multi-scenario multi-modality
benchmark to fuse infrared and visible for object detection.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 5802–5811, 2022. 5
[20] Jinyang Liu, Shutao Li, Haibo Liu, Renwei Dian, and Xiao-
hui Wei. A lightweight pixel-level unified image fusion net-
work.IEEE Transactions on Neural Networks and Learning
Systems, 2023. 2
[21] Jinyuan Liu, Runjia Lin, Guanyao Wu, Risheng Liu,
Zhongxuan Luo, and Xin Fan. Coconet: Coupled con-
trastive learning network with multi-level feature ensemble
for multi-modality image fusion.International Journal of
Computer Vision, 132(5):1748–1775, 2024. 2
[22] Risheng Liu, Zhu Liu, Jinyuan Liu, Xin Fan, and Zhongxuan
Luo. A task-guided, implicitly-searched and meta-initialized
deep model for image fusion.IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, 46(10):6594–6609,
2024. 2
[23] Yu Liu and Zengfu Wang. Simultaneous image fusion and
denoising with adaptive sparse representation.IET Image
Processing, 9(5):347–357, 2015. 2
33856


<!-- PDF PAGE 10 -->

[24] Yu Liu, Shuping Liu, and Zengfu Wang. A general frame-
work for image fusion based on multi-scale transform and
sparse representation.Information Fusion, 24:147–164,
2015. 2
[25] Yu Liu, Xun Chen, Rabab K Ward, and Z Jane Wang. Image
fusion with convolutional sparse representation.IEEE Signal
Processing Letters, 23(12):1882–1886, 2016. 2
[26] Yu Liu, Xun Chen, Hu Peng, and Zengfu Wang. Multi-focus
image fusion with a deep convolutional neural network.In-
formation Fusion, 36:191–207, 2017. 2
[27] Yu Liu, Yu Shi, Fuhao Mu, Juan Cheng, and Xun Chen.
Glioma segmentation-oriented multi-modal mr image fusion
with adversarial learning.IEEE/CAA Journal of Automatica
Sinica, 9(8):1528–1531, 2022. 2
[28] Yu Liu, Chen Yu, Juan Cheng, Z Jane Wang, and Xun
Chen. Mm-net: A mixformer-based multi-scale network for
anatomical and functional image fusion.IEEE Transactions
on Image Processing, 33:2197–2212, 2024. 2
[29] Jiayi Ma, Han Xu, Junjun Jiang, Xiaoguang Mei, and Xiao-
Ping Zhang. Ddcgan: A dual-discriminator conditional gen-
erative adversarial network for multi-resolution image fu-
sion.IEEE Transactions on Image Processing, 29:4980–
4995, 2020. 2
[30] Chengyi Pan, Xiuliang Xi, Xin Jin, Huangqimei Zheng,
Puming Wang, and Qiang Jiang. Dif-gan: A generative
adversarial network with multi-scale attention and diffusion
models for infrared-visible image fusion. In2024 IEEE In-
ternational Symposium on Parallel and Distributed Process-
ing with Applications (ISPA), pages 1960–1967. IEEE, 2024.
2
[31] Yu Shi, Yu Liu, Juan Cheng, Z Jane Wang, and Xun Chen.
Vdmufusion: A versatile diffusion model-based unsuper-
vised framework for image fusion.IEEE Transactions on
Image Processing, 2024. 2, 6
[32] Yu Shi, Yu Liu, Juan Cheng, Huafeng Li, and Xun Chen.
Semantic-guided diffusion sampling: A generalized strategy
for enhancing object segmentation oriented multimodal im-
age fusion.IEEE Journal of Selected Topics in Signal Pro-
cessing, pages 1–13, 2025. 2
[33] Linfeng Tang, Yuxin Deng, Xunpeng Yi, Qinglong Yan, Yix-
uan Yuan, and Jiayi Ma. Drmf: Degradation-robust multi-
modal image fusion via composable diffusion prior. InPro-
ceedings of the 32nd ACM International Conference on Mul-
timedia, pages 8546–8555, 2024. 2
[34] Linfeng Tang, Chunyu Li, and Jiayi Ma. Mask-difuser: A
masked diffusion model for unified unsupervised image fu-
sion.IEEE Transactions on Pattern Analysis and Machine
Intelligence, 2025. 6
[35] Gregory Vaksman, Michael Elad, and Peyman Milanfar.
Lidia: Lightweight learned image denoising with instance
adaptation. InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition Workshops, pages
524–525, 2020. 6
[36] Jiangang Wang, Qingnan Fan, Jinwei Chen, Hong Gu, Feng
Huang, and Wenqi Ren. Rap-sr: restoration prior enhance-
ment in diffusion models for realistic image super-resolution.
InProceedings of the AAAI Conference on Artificial Intelli-
gence, pages 7727–7735, 2025. 3
[37] Xiangxiang Wang, Lixing Fang, Junli Zhao, Zhenkuan Pan,
Hui Li, and Yi Li. Uud-fusion: An unsupervised universal
image fusion approach via generative diffusion model.Com-
puter Vision and Image Understanding, 249:104218, 2024. 2
[38] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot im-
age restoration using denoising diffusion null-space model.
arXiv preprint arXiv:2212.00490, 2022. 3, 4, 5
[39] Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang,
Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C
Kot, and Bihan Wen. Sinsr: diffusion-based image super-
resolution in a single step. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 25796–25805, 2024. 3
[40] Zhishe Wang, Junyao Wang, Yuanyuan Wu, Jiawei Xu, and
Xiaoqin Zhang. Unfusion: A unified multi-scale densely
connected network for infrared and visible image fusion.
IEEE Transactions on Circuits and Systems for Video Tech-
nology, 32(6):3360–3374, 2021. 2
[41] Zhixin Wang, Ziying Zhang, Xiaoyun Zhang, Huangjie
Zheng, Mingyuan Zhou, Ya Zhang, and Yanfeng Wang. Dr2:
Diffusion-based robust degradation remover for blind face
restoration. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 1704–
1713, 2023. 3
[42] Zirui Wang, Jiayi Zhang, Tianwei Guan, Yuhan Zhou,
Xingyuan Li, Minjing Dong, and Jinyuan Liu. Efficient recti-
fied flow for image fusion.arXiv preprint arXiv:2509.16549,
2025. 6
[43] Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xing-
long Wu, Yapeng Tian, Wenming Yang, and Luc Van Gool.
Diffir: Efficient diffusion model for image restoration. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 13095–13105, 2023. 3
[44] Xinyu Xie, Yawen Cui, Tao Tan, Xubin Zheng, and Zitong
Yu. Fusionmamba: Dynamic feature enhancement for mul-
timodal image fusion with mamba.Visual Intelligence, 2(1):
37, 2024. 2
[45] Han Xu and Jiayi Ma. Emfusion: An unsupervised enhanced
medical image fusion network.Information Fusion, 76:177–
186, 2021. 2
[46] Han Xu, Jiayi Ma, Junjun Jiang, Xiaojie Guo, and Haibin
Ling. U2fusion: A unified unsupervised image fusion net-
work.IEEE Transactions on Pattern Analysis and Machine
Intelligence, 44(1):502–518, 2020. 6
[47] Han Xu, Jiteng Yuan, and Jiayi Ma. Murf: Mutually re-
inforcing multi-modal image registration and fusion.IEEE
Transactions on Pattern Analysis and Machine Intelligence,
45(10):12148–12166, 2023. 6
[48] Yushen Xu, Xiaosong Li, Yuchan Jie, and Haishu Tan.
Simultaneous tri-modal medical image fusion and super-
resolution using conditional diffusion model. InInter-
national Conference on Medical Image Computing and
Computer-Assisted Intervention, pages 635–645. Springer,
2024. 2
[49] Yushen Xu, Xiaosong Li, Yuchun Wang, Xiaoqi Cheng,
Huafeng Li, and Haishu Tan. Flexid-fuse: Flexible number
of inputs multi-modal medical image fusion based on diffu-
33857


<!-- PDF PAGE 11 -->

sion model.Expert Systems with Applications, page 128895,
2025. 2
[50] Bo Yang, Zhaohui Jiang, Dong Pan, Haoyang Yu, Gui Gui,
and Weihua Gui. Lfdt-fusion: A latent feature-guided diffu-
sion transformer model for general image fusion.Informa-
tion Fusion, 113:102639, 2025. 2
[51] Zengyi Yang, Yafei Zhang, Huafeng Li, and Yu Liu.
Instruction-driven fusion of infrared–visible images: Tailor-
ing for diverse downstream tasks.Information Fusion, 121:
103148, 2025. 2
[52] Xunpeng Yi, Linfeng Tang, Hao Zhang, Han Xu, and Ji-
ayi Ma. Diff-if: Multi-modality image fusion via diffusion
model with fusion knowledge prior.Information Fusion,
110:102450, 2024. 2
[53] Xunpeng Yi, Han Xu, Hao Zhang, Linfeng Tang, and Ji-
ayi Ma. Text-if: Leveraging semantic text guidance for
degradation-aware and interactive image fusion. InProceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 27026–27035, 2024. 2
[54] Haitao Yin, Shutao Li, and Leyuan Fang. Simultaneous im-
age fusion and super-resolution using sparse representation.
Information Fusion, 14(3):229–240, 2013. 1
[55] Jun Yue, Leyuan Fang, Shaobo Xia, Yue Deng, and Jiayi Ma.
Dif-fusion: Toward high color fidelity in infrared and visible
image fusion with diffusion models.IEEE Transactions on
Image Processing, 32:5705–5720, 2023. 2
[56] Syed Waqas Zamir, Aditya Arora, Salman Khan, Mu-
nawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang.
Restormer: Efficient transformer for high-resolution image
restoration. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 5728–
5739, 2022. 6
[57] Hao Zhang, Lei Cao, and Jiayi Ma. Text-difuse: An inter-
active multi-modal image fusion framework based on text-
modulated diffusion model.Advances in Neural Information
Processing Systems, 37:39552–39572, 2024. 6
[58] Leheng Zhang, Yawei Li, Xingyu Zhou, Xiaorui Zhao, and
Shuhang Gu. Transcending the limit of local window:
Advanced super-resolution transformer with adaptive token
dictionary. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 2856–
2865, 2024. 6
[59] Yu Zhang, Yu Liu, Peng Sun, Han Yan, Xiaolin Zhao, and
Li Zhang. Ifcnn: A general image fusion framework based
on convolutional neural network.Information Fusion, 54:
99–118, 2020. 6
[60] Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang,
Shuang Xu, Zudi Lin, Radu Timofte, and Luc Van Gool.
Cddfuse: Correlation-driven dual-branch feature decompo-
sition for multi-modality image fusion. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5906–5916, 2023. 2
[61] Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang,
Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu
Timofte, and Luc Van Gool. Ddfm: denoising diffusion
model for multi-modality image fusion. InProceedings of
the IEEE/CVF International Conference on Computer Vi-
sion, pages 8082–8093, 2023. 2, 6
[62] Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang,
Kai Zhang, Shuang Xu, Dongdong Chen, Radu Timofte, and
Luc Van Gool. Equivariant multi-modality image fusion. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 25912–25921, 2024. 2
[63] Man Zhou, Jie Huang, Keyu Yan, Danfeng Hong, Xiuping
Jia, Jocelyn Chanussot, and Chongyi Li. A general spatial-
frequency learning framework for multimodal image fusion.
IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, 2024. 2
[64] Chunyu Zhu, Shangqi Deng, Xuan Song, Yachao Li, and Qi
Wang. Mamba collaborative implicit neural representation
for hyperspectral and multispectral remote sensing image fu-
sion.IEEE Transactions on Geoscience and Remote Sensing,
2025. 2
33858

<!-- END-OF-PAPER:b1233abd0c32 -->
