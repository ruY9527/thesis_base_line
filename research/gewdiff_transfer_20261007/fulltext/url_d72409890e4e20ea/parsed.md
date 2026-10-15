

<!-- PDF PAGE 1 -->

       Diffusion Model with Disentangled Modulations for Sharpening Multispectral and
                                                             Hyperspectral Images

                        Zihan Caoa, Shiqi Caoa, Liang-Jian Denga, *, Xiao Wua, Junming Houb, Gemine Vivonec, d

                                      aUniversityofElectronicScienceandTechnologyofChina,Chengdu,611731,China
                                                         bSoutheastUniversity,Nanjing,210000,China
                                     cInstituteofMethodologiesforEnvironmentalAnalysis(CNR-IMAA),Tito,85050,Italy
                                              dNationalBiodiversityFutureCenter(NBFC),Palermo,90133,Italy





Abstract
The denoising diffusion model has received increasing attention in the field of image generation in recent years, thanks to its
powerful generation capability. However, diffusion models should be deeply investigated in the field of multi-source image fusion,
such as remote sensing pansharpening and multispectral and hyperspectral image fusion (MHIF). In this paper, we introduce a
novel supervised diffusion model with two conditional modulation modules, specifically designed for the task of multi-source
image fusion. These modules mainly consist of a coarse-grained style modulation (CSM) and a fine-grained wavelet modulation
(FWM), which aim to disentangle coarse-grained style information and fine-grained frequency information, respectively, thereby
generating competitive fused images.  Moreover, some essential strategies for the training of the given diffusion model are well
discussed, e.g., the selection of training objectives. The superiority of the proposed method is verified compared with recent state-
of-the-art (SOTA) techniques by extensive experiments on two multi-source image fusion benchmarks, i.e., pansharpening and
MHIF. In addition, sufficient discussions and ablation studies in the experiments are involved to demonstrate the effectiveness of
our approach. Code will be available after possible acceptance.
Keywords:  Denoising diffusion model, wavelet transformation, pansharpening, multi-source image fusion, multispectral and
hyperspectral image fusion, end-to-end network, remote sensing


1. Introduction

   Recently, multi-source image fusion (MSIF) has attracted
much attention in the area of image processing and computer
vision. The MSIF leverages on different images with domain-
specific information to generate a fused image that cannot be
obtained under practical conditions, such as a high-resolution
multi-spectral image (HRMS), or a high-resolution hyperspec-
tralimage(HRHS).Inthiswork,wemainlyfocusontwoprac-
tical MSIF tasks, i.e., pansharpening [1] and    multispectral and
hyperspectralimagefusion(MHIF)[2],toproposeourmethod.
   Pansharpening, as a relevant problem in remote sensing im-
age processing, is attracting more and more interest from the
research community and commercial companies. Specifically,
pansharpening requires the fusion of a high spatial resolu-
tion panchromatic (PAN) image and a low spatial resolution
multispectral (LRMS) image to obtain a high spatial resolu-
tion multispectral image (HRMS), which preserves the advan-
tages of the two images belonging to two different domains.                        Figure1: (a)PreviousDL-basedmethodsdirectlylearnanon-linearmappingto
Most satellites can simultaneously capture PAN and MS im-                          fuse under the one-step preconditioned framework. The M,P, andbH indicate
ages,  such as WorldView-3 and GaoFen-2.    Pansharpening                          theLRMS,thePAN,andthefusedHRMSinpansharpening. (b)Thedenoising
methods can be divided into four categories [3], i.e., compo-                      diffusion model is enabled with a multistep denoising process while breaking
nent substitution (CS) methods [4,5,6,7], multi-resolution                         the preconditioned learning process. The q(xt|xt−1), pθ(xt−1|xt,c), and c repre-
analysis (MRA) methods [8,9,10,11], variational optimized                          sentthenoiseaddingforwardprocess,thedenoisingbackwardprocess,andthe
                                                                                   condition, respectively.


    * Corresponding Author.

PreprintsubmittedtoInformationFusion                                                                                                                                                                November21,2023

<!-- PDF PAGE 2 -->

(VO) techniques [12,13,14,15], and deep learning (DL) ap-                         cussionsaboutunsupervisedmethodscanbefoundinSect.5.5.
proaches [16,17,18,19,20,21]. The first three conventional                           Diffusion denoising probability model (DPM), as proposed
approaches have their own advantages and disadvantages, but                       for unconditional image generation [23,55], conditional image
they frequently result in spectral or spatial distortions because                 generation [23,56], text-to-image generation [57,58], image-
of the limitations of the various techniques.    CS methods of-                   to-image translation [59], and discrete nature language gener-
ten consider a linear projection into a transformed domain of                     ation [60] tasks, has shown its power providing extra feature
the MS image by substituting the separated MS spatial infor-                      details and good generation ability.  Another advantage is that
mationwiththePANimage. MRAmethodsrelieduponamul-                                  DPM owns a more stable training process and no model col-
tiscale decomposition (often exploiting linear filtering) to the                  lapse compared to the GAN-based model.  More specifically,
PAN image to extract its spatial components and inject them                       the training and testing phases of DPM are dubbed as the for-
into the MS image. Thus, both are based on linear operations                      ward process and backward process [61,62].  In the forward
and, hence, they cannot accurately extract spectral/spatial in-                   process,theinputimageiscorruptedbyapureGaussiannoise,
formationsufferingfromspectraland/orspatialdistortions. VO                        then the model tries to denoise and recover the original input
methods introduce regularization terms to take into account of                    duringthetraining. Inthebackwardprocess,theinputisapure
prior information.  However, regularization terms that are not                    Gaussiannoiseandthetrainedmodelisresponsibleforremov-
well-designedcancausedegradedpansharpeningperformance.                            ing, step-by-step, a little noise. Finally, after large enough time
About the DL-based approaches, which are mostly related to                        steps, the noise is recovered to the generated images. Further-
this work, we will discuss their advantages and disadvantages                     more, to control the generated images, Ho et al. [63] proposed
in conjunction with the DL-based methods for MHIF.                                the conditional DPM where the condition is fed into the diffu-
   MHIF techniques aim to fuse multispectral and hyperspec-                       sion model to control the trajectory of the stochastic differen-
tral images captured from the same scene to get information                       tial equation (SDE) [23,61,62] ordinary di             fferential equation
from phenomena that cannot be detected by the sole HRMS.                          (ODE)[64,65]. DPM,whichhasamultistepforward         /backward
More specifically, MHIF can also be seen as a preprocessing                       process (see Fig.1-b), produces intermediate time-dependent
method for further high-level applications, such as change de-                    variables and breaks the one-step preconditioned framework.
tection [22,23], mineral exploration [24,25], and segmenta-                       However, a few works noted that DPM can be applied to the
tion [26].  The recent literature of MHIF [2] mainly involves                     task of MSIF. The methods that are most closely related to our
model-based methods [27,28,29,30] and DL-based meth-                              workareDDPM-CD[66],Dif-fuse[67],andPanDi         ff[68]. The
ods [31,32,33].  Although the model-based methods utilize                         common design of the first two works is that they both utilize
various image priors, it is still difficult to obtain high-fidelity,              a trained diffusion model and an additional task-specific path-
less distorted high-resolution hyperspectral fusion images due                    way, i.e., detection head and fusion head that involve a tedious
to the lack of training with large-scale datasets.                                two-stage training. PanDiffdirectly inputs the PAN and LRMS
   Over  recent  years,  supervised  DL-based  methods  have                      images as conditions for diffusion model and designs a task-
achieved significant advances in various vision tasks, such as                    agnostic architecture to complete the pansharpening task, but
convolutionneuralnetworks(CNNs)[34,35]andVisionTrans-                             neglecting the entangled conditions.
formers (ViTs) [36,37].  In the task of MSIF, supervised DL-                         To address the aforementioned issues, we propose a novel
based methods can fuse an image by learning a non-linear                          diffusion  model  with  two  conditional  modulation  modules,
functional mapping from spatial and spectral degradation re-                      named DDIF, specifically designed for the task of MSIF. The
latedtopansharpeningorMHIFtask. PrevioussupervisedDL-                             two modules mainly include a coarse-grained style modulation
based methods can be regarded as the preconditioned fusion                        (CSM) and a fine-grained wavelet modulation (FWM), aim-
framework,whichinjectstheimagefeaturesandpriorsintothe                            ing to disentangle coarse-grained style information and fine-
learnable mapping process.  To promote the non-linear capa-                       grained frequency information, thus overcoming the limitation
bility, these DL-based works mainly focus on better architec-                     thattheentangledconditionsarenotsuitableforthefusiontask.
ture designing (e.g., [17,38,39,19,40] for pansharpening and                      The issue of the degraded learning ability is addressed by the
[41,42,32,43]forMHIF),moree                  ffectivemodules[44,45,46],           multistep denoising process inherited by the diffusion model.
and more novel model-driven networks [33,47,48,49]. How-                          Experiments show that our DDIF can fuse images with bet-
ever, the learnable mapping process tries to fuse the image fea-                  ter visual quality and performance metrics.  Moreover, DPM
tures and priors in just one step (i.e., model evaluation) based                  withSDEsamplingcanintroducenewnoisethatcontainsmany
on the preconditioned fusion framework (see Fig.1-a). More-                       high-frequency components when sampling for better fusion.
over, supervised DL-based methods cannot leverage the learn-                      Compared with existing diffusion models, our DDIF is specifi-
ingabilityofthenetworkbasedontheone-steppreconditioned                            cally designed for the multi-source fusion task, as well as it is
framework, exacerbating the domain gap of the multi-source                        able to yield better fusion outcomes.
image. Additionally, the conditions for the network are entan-                       The contribution of this work is three-fold:
gledandmaynotbesuitableforthefusiontask. Recently,many
unsupervisedmethodshavebeenproposedbyexploitingadver-                               1.Weproposeanovelsuperviseddi  ffusionmodelwithdisen-
sarial learning [50,51], novel spatial         /spectral loss [52], cycle               tangled modulations for sharpening multispectral and hy-
consistency [53], and learning degraded processes [54]. In this                         perspectral images (dubbed as DDIF), which can address
work, we mainly focus on supervised methods, and some dis-                              the degraded learning ability brought by the one-step pre-
                                                                              2

<!-- PDF PAGE 3 -->

     conditioned framework in previous DL-based models.  A                      on the input.   Building on this idea, AKD [44] further ex-
     preliminary manuscript can be found on the preprint web-                   plored the concept of LAGNet and proposed two dynamically
     site1.                                                                     generated branches responsible for spectral and spatial details
  2.The given DDIF contains two novel modulation mod-                           extraction,  respectively.   SpanConv [71] presented an inter-
     ules,i.e.,coarse-grainedstylemodulation(CSM)andfine-                       pretable span strategy, which effectively constructed a kernel
     grained wavelet modulation (FWM). These two modules                        space and reduced the redundancy of the convolution while
     disentangle style information and frequency components                     maintaining good performance.   Besides, PMACNet [72], a
     from different domain conditions, thus suiting the fusion                  parallel convolutional neural network structure, has been em-
     task and bringing improved fusion results.                                 ployed with a pixel-wise attention constraint module and a
  3.Our DDIF gets state-of-the-art (SOTA) performance on                        novel multireceptive-field attention block.   This architecture
     the pansharpening task considering two widely used pan-                    successfullyaddressedthesmallreceptivefieldissuecausedby
     sharpening datasets, as well as yields competitive perfor-                 CNNs.
     mance on one benchmark MHIF dataset. Discussions and                          Moving on to MHIF, CNN-FUSE [31] learned the subspace
     ablation studies assess the effectiveness of the proposed                  from the low-resolution hyperspectral (LRHS) image via sin-
     method.                                                                    gular value decomposition and then approximated the high-
                                                                                resolution hyperspectral (HRHS) image with subspace coef-
  The remaining of the paper is organized as follows. Sect.2                    ficients.   Furthermore,  SSRNet [42] proposed three models
describes the related works. The proposed methodology is pre-                   for MHIF, including a cross-mode message inserting model
sented in Sect.3.  A broad experimental analysis is shown in                    for producing preliminary fused HRSR, a spatial reconstruc-
Sect.4. Discussions are provided to the readers in Sect.5. Fi-                  tion model,  and a spectral reconstruction model.   Recently,
nally, conclusions are drawn in Sect.7.                                         DHIF [33] analyzed the importance of physical imaging mod-
                                                                                els for MHIF and introduced a spatio-spectral regularized deep
2. RelatedWorks                                                                 hyperspectral fusion model.
                                                                                   The aforementioned works conducted deep research on fea-
  ThissectionmainlyintroducessupervisedDL-basedanddif-                          turesofDL-basedmethodshelpingmodelsingeneratingbetter
fusion methods that are mostly related to our work, as well as                  fusion results. However, these DL-based models require fixed
mainstream generative tasks and methods related to diffusion                    priors and then fuse the image features, which means they are
models.                                                                         underthepreconditionedframework. Incontrast,denoisingdif-
                                                                                fusion models produce a predicted distribution as close as pos-
2.1. SupervisedPreconditionedModelsforImageFusion                               sible to the posterior distribution in each step of optimization,
                                                                                breaking the preconditioned learning process.
  Previous  DL-based  methods  are  mainly  based  on  a  pre-                  2.2. DiffusionModels
conditioned fusion framework.   Ma et al [69] first proposed
transformer-based framework for image fusion, in which ex-                         Diffusion models have recently been proposed for the gen-
plains the importance of transformer’s long distance depen-                     eration task,  including conditional or unconditional genera-
dence on image fusion tasks. Tang et al [70] integrated image                   tion [23,55,61],  text-to-image translation [58,73],  image
matching, fusion and semantic awareness into a unified frame-                   super-resolution [74],  image restoration [75,76],  and other
work,andgetspromisingoutcomes. Whileunsupervisedmeth-                           high-level image manipulation tasks [77,78,59].   Wherein,
ods have led to significant advancements in recent years, it is                 Song et al. [79] introduced first a score-based model that pro-
important to note that our work is related to supervised learn-                 duces samples via Langevin dynamics using gradients of the
ing. Thus, in this paper, we mainly focus on supervised meth-                   data distribution estimated with score matching. Ho et al. [23]
ods.  For pansharpening, PNN [17], inspired by a related sin-                   proposed DDPM from the direction of weighted variational
gle image super-resolution technique, proposed first a convo-                   bound,andtheirequivalenceisprovenin[62]. Toacceleratethe
lutional neural network for pansharpening.  To enable the net-                  sampling of DPM, DDIM [64] designed a non-Markov chain
work to handle the high-frequency information, PanNet [18]                      sampling process. In addition, DPM-solver [65] simplified the
was proposed to explicitly inject high-passed information from                  solutiontoanexponentiallyweightedintegraloftheneuralnet-
the PAN image into the upsampled LRMS image, resulting in                       work by computing the linear part of the ODE and applying
a better performance. Furthermore, DiCNN [38] has been pre-                     change-of-variable, further accelerating the sampling process.
sentedasadetails-inject-basedCNN,whichispowerfulinpre-                          TofreethedesignofDPMfromcumbersomemathematicalre-
serving frequency details. Another approach, the so-called Fu-                  quirements, EDM [56] decoupled various design components
sionNet [19], implemented an end-to-end residual network, al-                   of the diffusion model and designed a second-order ODE sam-
lowing it to explicitly learn high-frequency details.                           pler, which further improved the performance of the diffusion
  To deal with the static kernel of conventional convolutional                  model to reach state-of-the-art performance.
operators, LAGNet [45] introduced a dynamic kernel based                           Besides,  generative  approaches  mainly  consist  of  GAN-
                                                                                basedandflow-basedmodels(beyonddiffusionmodels). GAN-
                                                                                based models [80,81,82,83] generate a sample following a
   1https://arxiv.org/abs/2304.04774                                            datadistributionusingadiscriminatemodel, D,havingtherole

                                                                             3

<!-- PDF PAGE 4 -->

ofestimatingtheprobabilitythatasamplecomesfromthetrain-                                          Table 1: Some notations used in this work.
ing data rather than the generative model, G.  Instead, flow-                       Notation      Explanation
based models [84,85] learn the underlying distribution of data
bytransformingasimpleinputdistribution(e.g.,Gaussian)into                           t             The timestep
the target distribution through a series of invertible transforma-                  xt            An image at diffusion timestept
tions. Compared to the two above-mentioned generative mod-                          ϵ             Gaussian noise
els,diffusionmodelscangenerateimageswithmoredetailsand                              v             Training objective defined in Eq. (11)
higherfidelity. Theadvantagesofdiffusionmodelsincludesta-                           M             The upsampled LRMS
ble training, minimal mode collapse, and the ability to train                       P             The PAN image
with only a single mean squared error (MSE) loss. In compari-                       H             The original HRMS
son to GAN-based models, which suffer from instability issues                       bH            The fused HRMS
in their adversarial training, and flow-based models, which are                     Fl            Thel-th layer’s feature map
limitedintheirnetworkperformanceduetothereversibilityre-                            c             The diffusion condition
quirement, diffusion models are easier to train and design.                         LL,LH,        The wavelet coefficients
  Despite being so powerful in the field of image generation,                       HL,HH
diffusion models have not received much attention in the field                      Z             The scale used for style modulation in CSM
of MSIF. The most related works are as follows.   DDPM-                             S             The shift used for style modulation in CSM
CD [66] utilized di  ffusion models for landform change detec-                      (·)↓l         Bilinear downsample with factorl
tion.  Firstly, an unconditional diffusion model is trained on a
large dataset, and the features of a specific layer of the diffu-
sion model during the sampling process are used as additional                   thePANimage). Basedonthis,aconditioncanbedisentangled
informationinputtothesegmentationhead,producingsegmen-                          into the style codes and frequency information.  It is feasible
tationresults. Asaresult,DDPM-CDhasachievedsatisfactory                         to employ the diffusion model with two conditioning modules
performance in the field of landform change detection. Similar                  that handle coarse-grained style information and fine-grained
to DDPM-CD, Dif-fuse [67] fused red-green-blue (RGB) and                        frequency information, respectively.  Benefiting from this dis-
near-infraredimagesgettingclearadvances. Theseworksfocus                        entanglement, we can make the diffusion U-Net encoder and
onusingdeepsemanticfeaturemapsofDPMfeedingtheminto                              decoder responsible for style and frequency information sepa-
another segmentation or fusion head, which is not straightfor-                  rately, thus easing the learning process.  Finally, the combina-
ward, also (unnecessary) requiring a two-stage training. These                  tionofthemulti-sourceimagesisinjectedintothefinal(gener-
previous works raise a question that is how and if we can fuse                  ated) fused image.
images from two different domains by exploiting just a unique
diffusion model, training it in an end-to-end manner.
2.3. Motivations                                                                3. Methodology
  Diffusion models in the MSIF task are mostly based on a
straightforward approach.   They exploit an ideal image cor-                      This section introduces first some notations that will be used
rupted by noise as input, i.e., the noised version of the ground-               inthedescriptionoftheproposedmethod. Then,wewillbriefly
truth(GT)image. Afterwards,thecombinationofmulti-source                         review the mechanism of the diffusion denoising model. Sub-
images is fed into the diffusion model as the condition.  Ob-                   sequently, a detailed introduction to the architecture of the dif-
viously, the noised GT can be concatenated with them to fuse                    fusion model and the two conditional disentangled modulation
thetwodomains,whichhasbeenproventobeeffectivein[86].                            modules will be shown.  Finally, the effects of the proposed
However, we believe that this condition, just using concatena-                  twomodulesandsomediscussionsabouttrainingandsampling
tion, is entangled and will lead to insufficient convergence and                techniques will be provided.
poor results (see Sect.4for details). To achieve better results,
we considerconditional styleand frequencymodulation todis-
entangleconditionsinthedenoisingdiffusionprocess,thuswell                       3.1. Notations
adapting to the fusion task.
  About the style modulation, the related style information is                    Tab.1reports all the related notations used in the paper,
encoded as a condition to help the model control the general                    focusing on the pansharpening task.  From the table, we de-
features of objects, such as, shapes and colors, which has been                 note the PAN image, the upsampled LRMS image, the original
inspired by the extracting style code proposed in [82].In the                   HRMS image, and the fused HRMS image as P ∈ RH×W×c,
context of sharpening multispectral and hyperspectral images,                   M ∈ RH×W×C, H ∈ RH×W×C, and bH ∈ RH×W×C, respectively,
we describe style information as the coarse-grained spectral                    where, H, W, and C (or c) indicate the image spatial size and
changes that are intertwined with low-frequency spatial infor-                  the channel (or spectral) number, respectively.
mation as shown in Fig.4-a.The frequency modulation can be                        Considering the MHIF task, P correspondingly represents
introduced by the added Gaussian noise in the backward pro-                     the HRMS image, M is the upsampled LRHS image, H is the
cessofthe diffusionmodeland thedomain-relatedimage(e.g.,                        original HRHS image, andbH is the fused HRHS image.
                                                                            4

<!-- PDF PAGE 5 -->

Figure 2: The flowchart of our DDIF. The input is the residual related to the difference between HRMS and LRMS images. A Gaussian noise ϵt = √1− ¯αtϵ is
sampled to be added to the residual.  The diffusion U-Net exploiting our effective disentangled modulations (i.e., CSM and FWM) produces the undegraded x0.
Finally, x0 is added to LRMS to get the HRMS image. “wavelet coeffs.” areLLM,LHP,HLP,HHP as presented in Eq. (10), and (    ·)l is the bilinear downsample
operator by a factor ofl.


3.2. ABriefReviewofDiffusionModel                                                       The training of a diffusion model, by maximizing its varia-
   DiffusionmodelscangeneratearealisticimagefromaGaus-                               tional lower bound (VLB), is done exploiting a simple super-
sian distribution by reversing a noise process.  The diffusion                       vised loss [23], written as follows:

modelaccomplishesitstaskintwosteps,i.e.,forwardandback-                                                 min
ward processes, which are illustrated in Fig.1.                                                           θ    Lsimple =E[∥ϵ−ϵθ(xt,t)∥22].                   (4)
   The forward process aims to make the origin image, x0 ∼                           Please, note that in Eq. (4), the model prediction of the added
pdata(x0), noisy due to a T step Markov chain that gradually                         Gaussian noise, ϵθ(xt,t), is used as the training objective, but it
converts it into a Gaussian distribution.  The forward step is                       can be substituted by the original input,x0, or the “velocity”,v
defined as:                                                                          (see[57]andtheablationstudyinSect.4formoreinformation).
                                                                                        After training the model, we can sample data starting from a
                 q(xt|xt−1)=N(xt;p1−βtxt−1,βtI),                 (1)                 standard Gaussian noise, xT ∼ N(0,I), and, according to Eq.
where t ∈ [0,T], βt ∈ (0,1) is a pre-defined variance, that is                       (3), the mean and variance can be computed as follows:
function of the steps,N is a Gaussian distribution with mean,                                                          xt−      βt          ,              (5)
p1−βtxt−1, and standard deviation, βtI, and I is the identity                                       µθ(xt,t)=    1√            √
matrix.  Through the reparameterization trick, it can get xt as                                                     αt           1− ¯αtϵθ(xt,t)

follows:
                    q(xt|x0)= √      ¯αtx0+ p1− ¯αtϵ,                     (2)                                  Σθ(xt,t)= 1− ¯αt−11− ¯αt βt.                            (6)

where ϵ ∼ N(0,1) is the standard Gaussian noise, and  ¯αt  =                            To this aim, the image xt can be sampled from the previous
Qti=0αi withαt = 1−βt.                                                               step. Through T iterative sampling steps, the initial image, x0,
   Then, the backward process is related to the denoising of xt                      can be obtained.
by the following procedure:                                                             In the MSIF task, there are usually two (or even more) input
                                                                                     images from different domains.  Focusing on pansharpening,
               pθ(xt−1|xt)=N(xt−1;µθ(xt,t),Σθ(xt,t)),              (3)               thePANandLRMSimagesarefromdifferentdomains(specif-
                                                                                     ically from different sensors) and the HRMS image is from the
where θ denotes the learnable model parameters, and µθ(xt,t)                         targetdomain. Toguidethediffusionmodel,weempiricallyset
and Σθ(xt,t) are the mean and the standard deviation, respec-                        the PAN, the LRMS, and the constructed wavelet coefficients
tively, of the Gaussian distribution.                                                (presentedinSect.3.5)toconditionsdenotedas            c. Afterwards,
   In summary, the forward process degrades the data distribu-                       Eqs. (3)-(6) have to be conditioned to         c.
tion into a standard Gaussian distribution.  Instead, the back-                      3.3. OverallModelArchitecture
ward process, modeled by a neural network, aims to learn how
toremovethedegradationgeneratedintheforwardprocess,i.e.,                                As shown in Fig.2, the overall network architecture of our
the denoising task.                                                                  DDIF consists of an encoder, decoder, and two middle self-
                                                                                 5

<!-- PDF PAGE 6 -->

attentionmodules,whichresembleaU-Netarchitectureofcon-
volutional layers. There are two downsampling operations be-
tweenthetwolayersoftheencodertoreducethespatialresolu-
tionandincreasethenumberofchannels. Similarly,thereisan
upsampling operation by a factor of 2 between the two layers
of the decoder to increase the spatial resolution and reduce the
number of channels.
  The encoder and decoder have the same number of layers.
The encoder takes in input, not only the encoded features from
the previous layer, but also the modulated coarse-grained style
conditions (see Sect.3.4).  The features of the encoder at the
corresponding layer will be concatenated with the input fea-
tures of the decoder at the corresponding layer and the addi-
tionalwaveletfeatures(seeSect.3.5)alongthechanneldimen-
sion, feeding them into the decoder. The related output can be
modeledastheaddedGaussiannoise,ϵ,theoriginalimage,x0,
or the “velocity”,v, which will be discussed in Sect.3.7.

3.4. Coarse-grainedStyleModulation(CSM)
  Asmentionedabove,weconsiderdisentanglingthestyleand
the detailed information in the MS and PAN images proposing
a CSM as shown in Fig.3-a.  More specifically, the MS and
PANimagesareconcatenatedalongthechanneldimensionand
are considered conditional guidance.  We add this conditional                   Figure 3: Two conditioning modulations that have been proposed: (a) coarse-
guidance into CSM, then feeding the output feature into each                    grained style modulation (CSM), (b) fine-grained wavelet modulation (FWM).
layer of the encoder to modulate the encoder feature, Fl. It is                 Q,K,V are query, key, and value defined in Eq.10.  For the other notations,
                                                                                please refer to Tab.1.
worthnotingthatweavoidencodingatoomuchdetailedspatial
information, but reconstructing it in the decoder.  To this end,
theencoderonlyneedstoconsidertheoverallstyleinformation                         the features encoded from the encoder to support style infor-
without the detailed spatial information.  It is also meaningful                mation and frequency components disentanglement. However,
for the encoder to progressively decrease the spatial size of the               relying solely on the features outputted by the encoder is not
featuremapwhileincreasingthenumberofchannels. Byincor-                          enough for the decoder. We found that introducing appropriate
poratingstyleinformationintotheencoder,themodelcanfocus                         high-frequency spatial components during the decoding stages
on style-related aspects without being influenced by irrelevant                 is instead advantageous to produce images with richer details.
spatial information.                                                            An intuitive approach is to concatenate the high-frequency de-
  For CSM, it is feasible to generate the corresponding style                   tailed components extracted from the PAN image with the fea-
featurebyoutputtingthescale, Z,andtheshift,S,basedonthe                         tures outputted by the encoder along the channel dimension,
concatenation of the PAN and the MS images. Thus, we have:                      then feeding the concatenated features into the decoder for de-
                     Z,S= Split(MLP([P,M])),                                    coding.      Considering that previous traditional methods of-
                     Fl = Fl·(I+Z)+S,                                 (7)       ten exploit wavelet information [89,90,91] to complement
                                                                                additional frequency details, we can decompose images into
where “Split” splits the feature into two parts with equal size                 wavelets and incorporate them into the diffusion model.
along the channel dimension.  The MLP is implemented with                          Based on this cue, we apply wavelet decomposition on the
several convolutional layers, where a SiLU [87] activation and                  PAN image to extract the horizontal,  vertical,  and diagonal
a GroupNorm [88] are added in-between.  More specifically,                      high-frequency components and to use a cross-attention mech-
the scale, Z ∈Rh×w×d, and the shift, S∈Rh×w×d, are produced                     anism in the decoder to introduce these high-frequency de-
by an MLP. Besides, Fl ∈ Rh×w×d is the CSM output, linearly                     tails into the decoding process.   However, considering only
modulatedby Z andS,whereh,w,d aredeterminedbythesize                            high-frequency details could be inadequate, thus, we also in-
of the feature outputted from the previous encoder layer.                       troduced the low-frequency main component of the LRMS,
3.5. Fine-grainedWaveletModulation(FWM)                                         which is also extracted using wavelet decomposition.    The
                                                                                high-frequency and low-frequency components are concate-
  The encoder is responsible for encoding style information                     nated along the channel dimension, then passing them through
into a high dimensional, low spatial resolution feature by treat-               thefine-grainedattentionmoduleontheU-Netskipconnection.
ing the LRMS and PAN images as conditions.  The decoder                         WerefertothisprocessasFWM,seeFig.3-b. Theadvantageof
(whoseobjectiveistodecodehigh-dimensional,lowspatialres-                        thismodulationis thatitseparatesthe low-frequencyandhigh-
olutionfeaturesfromtheencoderasfaithfullyaspossible)uses                        frequencycomponents,withPANprovidingthehigh-frequency
                                                                            6

<!-- PDF PAGE 7 -->

details and LRMS providing the low-frequency spatial compo-
nent, thus making the learning process easier for the decoder.
  To decompose both the LRMS and PAN images into four
components, we used the DB1 wavelet decomposition [92],
whichincludesonemain(low-frequency)componentandthree
details components in the horizontal, vertical, and diagonal di-
rections. Thus, we have:                                                                             (a) Features by CSM modulation
              LLM,LHM,HLM,HHM = DB1(M),
                 LLP,LHP,HLP,HHP = DB1(P),                (8)

                  H
where LL ∈ R      2×W2×d denotes the main (low-frequency) com-
                    H                   H                         H
ponent, LH ∈ R      2×W2×d, HL ∈ R       2×W2×d, and HH ∈ R       2×W2×d
indicate the details components in the horizontal, vertical, and
diagonal directions, respectively.                                                                   (b) Features by FWM modulation
  Afterward, the attention operation can be used in FWM for
modulation. However, the memory complexity of the attention                              Figure 4: Feature maps of the two proposed modulations.
isO(n2), where n = HW.  When approaching the head of the
decoder, the output image size increases and it can become un-
acceptablewhendealingwithlargeimagesduetothequadratic                          3.6. TheEffectsofCSMandFWM
complexity with respect to the image size.  Hence, we adopt                       CSM generates the scale, Z, and the shift, S, for modula-
a linear-memory cross-attention mechanism toincorporate the                    tion.   The scale term controls the amplitude of the encoder
frequencycomponentsintothedecoder. Thelinear-memoryat-                         feature, Fl, and the shift directly injects the condition into the
tention mechanism is as follows:                                               feature. As depicted in Fig.4-a, the input feature has low con-
                   Q= Reshape(Softmax(Q,1)),                                   trast(buildings,forest,andlandshavesimilarfeatureintensity).
                   K = Reshape(Softmax(K,2)),                                  Moreover, the CSM feature has higher contrast than the input
                   V = Reshape(V),                                    (9)      feature under the control of the scale, as well as the color and
                   A= K⊙V⊤,                                                    shape (e.g., buildings) share some similarities with the shift.
                                                                               Exploiting this strategy, the style information is decomposed
                   O= A⊤⊙Q,                                                    into the scale and shifted to coarse-grained modulating the en-
where the “Reshape” operation is employed to flatten the spa-                  coder feature.
tial dimensions of Q, K, and V, into a single dimension, i.e.,                    FWM uses wavelet coefficients extracted from the MS and
RH×W×d → RHW×d, and Softmax(·,i) represents the Softmax                        PAN images. The MS image contributes to the low-frequency
operation along the i-th dimension. As a result, we reduce the                 coefficient, LLM, and the PAN image contributes to the high-
memory complexity toO d2 , where d is much smaller than n.                     frequency coefficients, LHP,HLP,HHP. As shown in Fig.4-
Thissignificantlyreducesmemoryusagewhileensuringtheef-                         b,theinputencoderfeaturegetsmeaningfulcontrastiveregions
fective introduction of the conditions. In the above equations,                due to CSM, but still lacking fine-grain frequency information.
Q is the query tensor, and K andV are the tensors obtained by                  Thanks to the help of the wavelet coefficients, the modulated
concatenating the LLM,LHP,HLP and the HHP components                           FWM feature has higher spatial fidelity.
along the channel dimension. These latter terms are projected                     In summary, the style and frequency information are well
by several convolutional layers as follows:                                    disentangled with the proposed two modulation modules and
    Q=WQ⊗F+bQ,                                                                 separately handled in the encoder and decoder. This can solve
                                                                               the preconditioned fusion problem of the previously developed
    [K,V]=WK,V⊗[LLM,LHP,HLP,HHP]+bK,V,      (10)                               DL-based methods.
where⊗istheconvolutionoperation,W∗ istheweight,b∗ isthe                        3.7. BoostthePerformanceFurther
bias,andFisthefeaturefromthecorrespondingencoderlayer.
This linear-memory cross-attention in the proposed FWM can                     Residual  Learning.       Because  of  CNNs  tend  to  learn
be used to learn a global response related to the frequency in-                low-frequency  information,  they  lack  in  representing  high-
formation. However, it exhibits some limitations in adequately                 frequency information.  Previous works attempted to address
capturingspatialinformation. Recognizingthatthevanillaself-                    this issue by filtering the input with high-pass filters and di-
attention mechanism has enough representational capability to                  rectly incorporating the related details as input [18] by design-
recover spatial details and offers the advantage of a quadratic                ing specialized high-frequency injection modules or supervis-
computationaloverload,wechosetoharnessthelinear-memory                         ing the process in the frequency domain [93].   Here, we do
cross-attention mechanism along the spectral dimension.   In                   not aim to design complex high-frequency modules that signif-
doing so, we introduced frequency components extracted by                      icantly increase the number of parameters in the network, nor
wavelet coefficients to modulate features from the correspond-                 do we want to perform complex operations such as computing
ing encoder layer, employing a cross-attention approach.                       loss in the frequency domain.
                                                                           7

<!-- PDF PAGE 8 -->

 Algorithm1: Training stage of our method.                                         Algorithm2: Inference stage of our method.
   Data: LRMS image, M, PAN image, P, GT image,x0,                                  Data: LRMS image, M, PAN image, P, trained
           diffusion model,xθ, with its parameters,θ,                                        diffusion model,xθ∗, with its parameters,θ∗,
           timestep,t, and denoised objective, ˆx0.                                          sampled image,xt, at timestept.
   Result: Optimized diffusion modelx∗θ.                                            Result: Sampled imagex0.
1 c← P,M,DB1(P,M);                        //Modulation                            1 t← T;
2 while   until  convergencedo                                                    2 c← P,M,DB1(P,M);                        //Modulation
3       t← Uniform(0,T);                                                          3 while   t > 0do
4       ϵ ∼N(0,I);                                                                4       xt−1← DDIM        Sample(xt,c);               //Eq.      (13)
5       xt ← √   ¯αt(x0− M)+ √1− ¯αtϵ;                                            5       t← t−1.
6        ˆx0←xθ(xt,c)+ M;          //Residual  learning                           6 end
7       θ←∇θLsimple(ˆx0,x0).                           //Eq.      (12)            7 x0←x0+ M.
8 end

                                                                                 From Eq. (13), the denoised     xt−1 can be a combination of the
   Inspired by FusionNet [19], we change the input of the dif-                   predicted xθ, the inferred noise, ϵθ, based on Eq. (14), and the
fusion model during the training process from HRMS to the                        new noise,ϵ ∼N(0,I). Therefore, we can accelerate sampling
differencebetweenHMRSandLRMS,i.e.,HRMS-LRMS,and                                  by τ = [τ1,τ2,··· ,τdim(τ)].  More specifically, τ is a subset of
then adding Gaussian noise as in Eq. (2).  We find that using                    [1,2,··· ,T], and we can make a fast sampling after usingτ:
noisy residuals as input, it converges faster and produces better                                                         1− ¯ατiϵθ(xτi,τi,c)
resultsthanwhenHRMSisusedasinput. Regardingtotheuse                                           xθ(xτi,τi,c) := xτi− p          √¯ατi                .         (15)
ofresiduallearning,therelatedablationexperimentsareshown
in Sect.4.8.2.                                                                   Accordingtotheabovesamplingequations,thesamplingspeed
TrainingObjectiveTherearethreechoicesforthetrainingob-                           can be improved. The overall training and sampling stages are
jective of the diffusion model2, i.e.,ϵ,x0, andv [57]. Note that                 presented in Algs.1and2.The existing e                 fficient ODE solvers
v is a combination ofϵ andx0 defined as follows:                                 can still be employed to accelerate our DDIF, such as the DPM
                       v= √¯αtϵ− p1− ¯αtx0.                        (11)          solver[65],PNDMsolver[94],andHeunsolver[56]. However,
                                                                                 we opted for the straightforward DDIM approach to accelerate
In the previous diffusion works [23,55], the training objective                  the sampling process, ensuring both simplicity and excellent
has often been selected as the added Gaussian noise, ϵ, which                    sharpening performance.
issuitableforlarge-scaledatasets. However,datasetsforMSIF
are much smaller than the ones related to natural images.  We                    4. Experiments
willshow,inSect.3.7,thatforsmalldatasets,predicting            x0 isa               In this section, we will provide a comprehensive overview
better choice thanϵ andv.                                                        oftheimplementationdetails,includingthemethodologiesand
   Thus, consideringx0 as goal, we have that the loss is:                        the technical details.    Furthermore,  we will delve into the
                   Lsimple =Eh∥x0−xθ(xt,t,c)∥22i.                  (12)
                                                                                 datasets, aiming to highlight their characteristics and relevance
Fast Iterative Sampling Regarding to the backward process                        tothiswork. Additionally,wewillpresentthebenchmarksem-
(sampling process), the diffusion model needs to iterate hun-                    ployedtoevaluatetheperformanceofourapproachontwoim-
dreds or thousands of times to generate an image, leading to                     age fusion tasks, i.e., pansharpening, and multispectral and hy-
a slow generation speed.  To solve this problem, we convert                      perspectral image fusion (MHIF). Finally, the main results and
theoriginalDDPMSDEsampler[23]oftheoriginalbackward                               ablation studies will be provided to quantitatively illustrate the
sampling process into the DDIM ODE sampler [64], which al-                       effectiveness of the proposed method.

lows for fast sampling in a non-Markov chain form as:                            4.1. ImplementationDetails
                               q                                                    The  proposed  DDIF  method  is  implemented  in  PyTorch
 xt−1 = √   ¯αt−1xθ(xt,t,c)+      1− ¯αt−1−σ2tϵθ(xt,t,c)+σ2tϵ, (13)              1.13.1 and Python 3.10.9 using AdamW [95] optimizer with
whereσt is an established function oft,xθ(xt,t,c) is the model                   alearningrateof1×10−4 tominimize Lsimple onaLinuxoper-
prediction ofx0, and                                                             atingsystemwithtwoNVIDIAGeForceRTX4090GPUs. The
                                                                                 initialization of convolution modules is based on a Kaiming
                  ϵθ(xt,t,c)= xt− √√¯αtxθ(xt,t,c)                                initialization [96].  To obtain wavelet coe    fficients, we use the
                                           1− ¯αt       .                  (14)  DB1 wavelets decomposition method decomposing the image
                                                                                 into four wavelet coefficients. The chosen diffusion denoising
                                                                                 model is a cosine schedule [55] for     αt:
   2We do not discuss the score function [62]    ∇xt logp(xt) =−  ϵθ(xt)√                                                    t/T +α         !
                                                                1− ¯αt, since                    ¯αt =  f(t)                                  ,             (16)
it has a similar form asϵ.                                                                              f(0),   f(t)= cos      1+α  · π2
                                                                              8

<!-- PDF PAGE 9 -->

where we setα to 8×10−3. The model does not learn the vari-                    patches for simulated data.   We applied the provided modu-
ance term Σθ introduced in [55].  The total training di    ffusion             lated transfer functions (MTFs) to the patched HRHSI and the
timestep is set to 500 for both pansharpening and MHIF exper-                  patched HRMSI following Wald’s protocol.  To get the final
iments.  Additionally, the exponential moving average (EMA)                    HRMSI, we adjusted the simulated HRMSI by using the mod-
ratio is set to 0.995. Moreover, the total training iterations for             ified M = (M− B· R)/A (as proposed in [98]), where      A and
theWorldView-3(WV3),GaoFen-2(GF2),andCAVEdatasets                               B are the correlated weight tensors, and R is the spectral re-
are set to 100k, 100k, and 300k iterations, respectively. Ac-                  sponse function.  Finally, we obtained 150 simulated LRHSI
cording to Eqs. (14) and (15), we set the number of sampling                   and HRMSI patches with sizes 40×40×150 and 80×80×4,
steps to 25 for the pansharpening task, and 100 for the MHIF                   respectively, with the original LRHSI serving as ground-truth.
task,ratherthan1000or2000DDPMsamplingstepsasin[97]                             We divided the 150 patches into train/validation/test sets using
and[68].Regardingourdi      ffusionmodelconfiguration,weem-                    the following percentages 80%/10%/10%.
ployed 4 layers each for both the encoder and decoder.  The
initialnumberofchannelsintheencoderlayerissetto32. Fol-
lowing each encoder/decoder layer, the number of channels is
adjusted by multiplication or division using the factors 1, 2, 2,
and 4, which correspond to the sequence of 4 encoder/decoder
layers.

4.2. Datasets
  To show the effectiveness of our DDIF, we conduct experi-
ments over a standard pansharpening data-collection (i.e., Pan-
Collection dataset3), which includes WV3 (8 bands) and GF2
(4 bands) data.  To evaluate the performance, we perform the
reduced-resolution and full-resolution experiments to compute
thereferenceandnon-referencemetrics,respectively. Notethat
thesametraininganddataaugmentationstrategyareappliedto
Pancollection for a fair comparison.
  For the MHIF, we choose the CAVE indoor dataset4 to fur-
ther evaluate our DDIF. The CAVE dataset contains 31 hyper-
spectral images (HSIs) captured under controlled illumination
with a spatial size of 512× 512 and 31 spectral bands ranging
from 400nm to 700nm at 10 nm steps.  The multispectral im-
ages(consistingofRGBbands)canbegeneratedbyusingHSIs                            Figure 5: Testing images from the CAVE dataset: (a) Balloons, (b) Compact
and the spectral response functions of the Nikon D700 camera,                  disc(CD),(c)Jellybeans,(d)Chartandstuffedtoy,(e)Clay,(f)Fakeandreal
                                                                               beers, (g) Fake and real lemon slices, (g) Fake and real tomatoes, (i) Feathers.
which has a spatial size of 512× 512× 3. 20 images are ran-                    (j) Flowers, (k) Hairs. The true color representation is used to depict the im-
domly selected for training and validation, and the remaining                  ages.
11 images are used for testing. The test set is shown in Fig.5.
Afterward, we cropped 20 selected HSIs and MSIs into 3920
overlapping patches for training and validation with the size of               4.3. Benchmark
64×64. AGaussianblurredkernel(size3×3)withastandard
deviation of 0.5 is used to get blurred HSIs that are decimated                   To  assess  the  performance  of  our  DDIF,  we  compare  it
by a factor of 4 to produce the LRHS images.  The cropped                      with previous state-of-the-art pansharpening methods (on the
MSIsareusedasHRMSandtheoriginalcroppedHSIsareused                              WV3 and GF2 datasets) and MHIF approaches (on the CAVE
asGT.Finally,thepairsaredividedintotraining(80%)andval-                        dataset).
idation (20%).   To verify the performance of each method on                      For the pansharpening task,  we choose some representa-
hyperspectralrealremotesensingdata,weutilizetheGF5-GF1                         tive model-based methods as optimized Brovey transform with
public dataset [98].  The GF5-GF1 dataset contains HSIs and                    haze correction method (BT-H) [99], band-dependent spatial-
MSIs,wherethespatialsizeofMSIsistwicethatofHSIs(i.e.,                          detail with physical constraints approach (BDSD-PC) [100],
1161×1120×150forHSIsand2332×2258×4forMSIs). We                                 generalizedLaplacianpyramid(GLP)withmodulationtransfer
randomly cropped HSIs and MSIs into patches of size 40×40                      function-matched filters and a full scale (FS) regression-based
and 80×80 with an overlap of 10 and 20, respectively, to gen-                  injection model (MTF-GLP-FS) [101] and a set of some com-
erate real data.  Based on the same patching scheme, further-                  petitive DL-based methods including PNN [17], DiCNN [38],
more, we can get the HRHSI (80×80) and HRMSI (160×160)                         MSDCNN [39], MMNet [102], FusionNet [19], CTINN [103],
                                                                               LAGConv [45],  and DCFNet [40].       A recently proposed

   3https://liangjiandeng.github.io/PanCollection.html                         diffusion-based pansharpening method, PanDiff[68], is also

   4https://www.cs.columbia.edu/CAVE/databases/                                compared. We compare them with our DDIF under the bench-
multispectral/                                                                 mark in [16]. Note that we do not compare our approach with
                                                                            9

<!-- PDF PAGE 10 -->

otherGAN-basedmodelsaswefoundthatalmostalltheGAN-                              with three traditional methods and some recent state-of-the-art
based models are designed for unsupervised pansharpening. A                    DL-based methods. The reduced-resolution and full-resolution
comparison with them leads to unfairness, but we still discuss                 results are reported in Tab.2.  To clearly demonstrate the ad-
on pros and cons in Sect.5.5.                                                  vantages of our method, we present the visual comparisons in
  For the MHIF task, we also chose model-based and DL-                         Fig.6, proposing some close-ups to better show some details.
based  methods  for  comparisons,  where  model-based  meth-                   Additionally, the error maps are shown accordingly.
ods include the coupled sparse tensor factorization (CSTF)                        On average, it can be seen that DL-based methods have sig-
method  [30],  the  fast  fusion  of  multi-band  images  based                nificantly better performance than traditional methods. Among
on solving a Sylvester equation approach (FUSE) [104], the                     DL-based  methods,  our  approach  achieves  the  best  perfor-
modulation transfer function-matched generalized Laplacian                     mance on the reduced-resolution dataset and competitive per-
pyramid  hyper-sharpening  (MTF-GLP-HS)  [101],  the  itera-                   formance on the full-resolution dataset.  Our DDIF can reach
tive regularization method based on tensor subspace repre-                     the values of 2.73 (SAM) and 2.02 (ERGAS) on the reduced
sentation method (IR-TenSR) [29], the low tensor-train rank-                   dataset, which outperforms all the compared DL-based tech-
based (LTTR) method [27] and the subspace-based low ten-                       niques.  The error maps also indicate that the image fused by
sor multi-rank regularization method (LTMR) [105].       For                   DDIF is closer to the GT (as it has darker blue colors).  The
real MHIF data, we chose some other competitive methods in-                    full-resolution images obtained by DDIF on the WV3 dataset
cluding the coupled nonnegative matrix factorization unmix-                    canachievestate-of-the-artperformance. Fig.9showsthefull-
ing method (CNMF) [106], the subspace-based regularization                     resolution outcomes, as well as their HQNR maps, where a
method (Hysure) [107], and a component substitution method,                    HQNR score closer to 1 indicates better fusion quality for the
i.e.,  the  Gram-Schmidt  adaptive  (GSA)  [108].    Regarding                 full-resolution image.   The obtained results indicate that our
to DL-based methods, we perform a comparison with CNN-                         DDIFcanfuseHRMSimagesreducingspatialandspectraldis-
FUSE [31],  SSRNet [42],  ResTFNet [41],  FusFormer [32],                      tortions, thus demonstrating that it has a good generalization at
HSRNet [109], and the recent state-of-the-art DHIF [33] tech-                  full-resolution.
nique. To ensure a fair comparison, all the DL-based methods
are retrained using identical input pairs and trained until con-               4.6. ResultsonGaoFen-2Dataset
vergence.                                                                         In this section, we test the performance of the compared ap-
4.4. QualityMetricsandRuntime                                                  proaches on 20 GaoFen-2 test samples from the PanCollection
  Forreduced-resolutiondatasetsinpansharpening,weusethe                        dataset. As shown in Tab.3, our DDIF outperforms the bench-
spectral angle mapper (SAM) [110], the relative dimensionless                  mark for almost all the reference and the non-reference met-
global error in synthesis (ERGAS) [111], the universal image                   rics for reduced-resolution and full-resolution datasets.  More
qualityindex(Q2n)[112],andthespatialcorrelationcoe      fficient               specifically, we observe an improvement of≈ 18%/17%/0.5%
(SCC) [113] as metrics.  The ideal values for Q2          n and SCC            intheSAM/ERGAS/Q4metricswhencomparingwiththesec-
are 1, and for SAM and ERGAS are 0.  Since these metrics                       ond best method, i.e., LAGConv [45].   Compared with the
require the GT, they are considered reference metrics and are                  third best method, i.e., DCFNet [40], our DDIF improves     ≈
used in the reduced-resolution test case. For the full-resolution              28%/29%/1.3% on the SAM/ERGAS/Q4 metrics.  The high
datasets, since there is no GT available, we use non-reference                 performanceonthefull-resolutiondatasetreflectstheoneatre-
metricstovalidatetheaccuracyofourDDIF.Thesemetricsin-                          duced resolution. Besides, Fig.7depicts some close-ups in the
cludeDλ,Ds,andhybridqualitywithnoreference(HQNR)[1].                           rectangularboxesandtheerrormapsamongthecomparedDL-
TheHQNRindexhasanidealvalueof1,whileDλ andDs have                              based methods.  The colors and edge details of the objects in
ideal values of 0.                                                             the figure are closer to the GT. Moreover, the error maps in-
  For  the  MHIF  dataset,  we  use  another  two  commonly-                   dicate that the residual between the outcome of DDIF and the
used metrics for the evaluation, i.e., the peak signal-to-noise                GT is minimal since the color of the map is the darkest blue
ratio  (PSNR)  and  the  structural  similarity  index  measure                one.In terms of runtime, our method achieves state-of-the-art
(SSIM)[114]. TheoptimalvaluesforPSNRandSSIMare       +∞                        performancewhilemaintainingfavorableexecutiontimescom-
and 1, respectively.                                                           pared to other DL-based methods.  Since DDIF is an iterative
  The runtime of traditional pansharpening methods is evalu-                   approach, an increase in runtime is expected.  However, our
ated using an Intel 12900k CPU on a reduced-resolution im-                     method significantly reduces the sampling time by employing
age from the WorldView-3 dataset, while DL-based methods                       Alg.2for accelerated sampling, enhancing its practical utility.
aretestedonanNVIDIA3090GPU.Likewise,theruntimesof                              In contrast, PanDiff[68] as a di    ffusion model, utilizing 2000
the MHIF algorithms are assessed on the CAVE dataset using                     steps of DDPM sampling, incurs substantial time consumption
the same hardware devices.                                                     for sampling.

4.5. ResultsonWorldView-3Dataset                                               4.7. ResultsonQuickBirdDataset
  Inthissection,weconductexperimentstoevaluateourDDIF                             We also conducted experiments on the QB dataset assessing
on the WV3 dataset assessing performance on 20 testing im-                     the performance both at reduced-resolution and full-resolution.
agesfromthePanCollectiondataset. Wecomparedourmethod                           Similarly, thereferenceand non-referencemetrics areobtained
                                                                           10

<!-- PDF PAGE 11 -->

           Figure 6: Visual comparisons with pansharpening methods on the WV3 dataset. The first and third rows show the RGB bands of the fused images. The second and
           fourth rows show the error maps between the GT and fused images. Some close-ups are depicted in red and yellow rectangles.  The numerical errors are displayed
           on the color bar, with darker colors indicating closer proximity to the ground-truth (GT) (i.e., better fusion performance).


           Table 2:  Results on the WV3 reduced-resolution and full-resolution datasets.  Some conventional methods (the first four rows) and DL-based approaches are
           compared. The best DL-based results are in red and the second best results are in blue.



















           on 20 testing samples, randomly selected from the QB dataset.                                   at reduced resolution, leading to a substantial improvement in
           Tab.4reports the quality indexes.  It is clear that our DDIF                                    SAM scores, ranging from 4.538 to 4.349. Also, our DDIF can
           method significantly outperforms all the compared approaches                                    get competitive performance for full-resolution test cases.  To
method                                                      Reduced                                                                   Full                              Runtime(s)
                           SAM(±std)     ERGAS(±std)       Q4(±std)         SCC(±std)11                       Dλ(±std)          Ds(±std)        HQNR(±std)
BDSD-PC[100]             5.4675±1.7185   4.6549±1.4667   0.8117±0.1063   0.9049±0.0419                     0.0625±0.0235   0.0730±0.0356   0.8698±0.0531                   0.059
MTF-GLP-FS[101]          5.3233±1.6548   4.6452±1.4441   0.8177±0.1014   0.8984±0.0466                     0.0206±0.0082   0.0630±0.0284   0.9180±0.0346                   0.023
BT-H[99]                 4.8985±1.3028   4.5150±1.3315   0.8182±0.1019   0.9240±0.0243                     0.0574±0.0232   0.0810±0.0374   0.8670±0.0540                   0.321
PNN[17]                  3.6798±0.7625   2.6819±0.6475   0.8929±0.0923   0.9761±0.0075                     0.0213±0.0080   0.0428±0.0147   0.9369±0.0212                   0.042
DiCNN[38]                3.5929±0.7623   2.6733±0.6627   0.9004±0.0871   0.9763±0.0072                     0.0362±0.0111   0.0462±0.0175   0.9195±0.0258                   0.083
MSDCNN[39]               3.7773±0.8032   2.7608±0.6884   0.8900±0.0900   0.9741±0.0076                     0.0230±0.0091   0.0467±0.0199   0.9316±0.0271                   0.112
FusionNet[19]            3.3252±0.6978   2.4666±0.6446   0.9044±0.0904   0.9807±0.0069                     0.0239±0.0090   0.0364±0.0137   0.9406±0.0197                   0.065
CTINN[103]               3.2523±0.6436   2.3936±0.5194   0.9056±0.0840   0.9826±0.0046                     0.0550±0.0288   0.0679±0.0312   0.8815±0.0488                   1.329
LAGConv[45]              3.1042±0.5585   2.2999±0.6128   0.9098±0.0907   0.9838±0.0068                     0.0368±0.0148   0.0418±0.0152   0.9230±0.0247                   1.381
MMNet[102]               3.0844±0.6398   2.3428±0.6260   0.9155±0.0855   0.9829±0.0056                     0.0540±0.0232   0.0336±0.0115   0.9143±0.0281                   0.348
DCFNet[40]               3.0264±0.73972.1588      ±0.45630.9051      ±0.08810.9861      ±0.0038            0.0781±0.0812   0.0508±0.0342   0.8771±0.1005                   0.548
PanDiff[68]              3.2968±0.6010   2.4667±0.5837   0.8980±0.0880   0.9800±0.0063                     0.0273±0.0123   0.0542±0.0264   0.9203±0.0360                  261.410
DDIF(ours)               2.7386±0.50802.0165      ±0.45080.9202      ±0.08240.9882      ±0.0031            0.0258±0.0187   0.0231±0.00750.9517      ±0.0173                2.602
Idealvalue                       0                   0                   1                   1                     0                   0                   1                  0

<!-- PDF PAGE 12 -->

          Figure 7:  Visual comparisons on the GF2 dataset. The first and third rows show the RGB bands of the fused images. The second and fourth rows show the error
          mapsbetweentheGTandfusedimages. Someclose-upsaredepictedinredandyellowrectangles.  Thenumericalerrorsaredisplayedonthecolorbar,withdarker
          colors indicating closer proximity to the ground-truth (GT) (i.e., better fusion performance).


          Table 3: Result on the GF2 reduced-resolution and full-resolution datasets.  Some conventional methods (the first four rows) and the DL-based approaches are
          compared. The best results are in red and the second best results are in blue.




















          better visualize the performance gap, Fig.8depicts the fused
          images and the error maps.                            Reduced                                                                        Full
method                       SAM(±std)      ERGAS(±std)        Q4(±std)          SCC(±std)                            Dλ(±std)           Ds(±std)         HQNR(±std)
BDSD-PC[100]               1.7110±0.3210    1.7025±0.4056    0.9932±0.0308    0.9448±0.016612                     0.0759±0.0301    0.1548±0.0280    0.7812±0.0409
MTF-GLP-FS[101]            1.6757±0.3457    1.6023±0.3545    0.8914±0.0256    0.9390±0.0197                       0.0336±0.0129    0.1404±0.0277    0.8309±0.0334
BT-H[99]                   1.6810±0.3168    1.5524±0.3642    0.9089±0.0292    0.9508±0.0150                       0.0602±0.0252    0.1313±0.0193    0.8165±0.0305
PNN[17]                    1.0477±0.2264    1.0572±0.2355    0.9604±0.0100    0.9772±0.0054                       0.0367±0.0291    0.0943±0.0224    0.8726±0.0373
DiCNN[38]                  1.0525±0.2310    1.0812±0.2510    0.9594±0.0101    0.9771±0.0058                       0.0413±0.0128    0.0992±0.0131    0.8636±0.0165
MSDCNN[39]                 1.0472±0.2210    1.0413±0.2309    0.9612±0.0108    0.9782±0.0050                       0.0269±0.0131    0.0730±0.0093    0.9020±0.0128
FusionNet[19]              0.9735±0.2117    0.9878±0.2222    0.9641±0.0093    0.9806±0.0049                       0.0400±0.0126    0.1013±0.0134    0.8628±0.0184
CTINN[103]                 0.8251±0.1386    0.6995±0.1068    0.9772±0.0117    0.9803±0.0015                       0.0586±0.0260    0.1096±0.0149    0.8381±0.0237
LAGConv[45]                0.7859±0.14780.6869       ±0.11250.9804       ±0.00850.9906       ±0.0019              0.0324±0.0130    0.0792±0.0136    0.8910±0.0204
MMNet[102]                 0.9929±0.1411    0.8117±0.1185    0.9690±0.0204    0.9859±0.0024                       0.0428±0.0300    0.1033±0.0129    0.8583±0.0269
DCFNet[40]                 0.8896±0.1577    0.8061±0.1369    0.9727±0.0100    0.9853±0.0024                       0.0234±0.01160.0659       ±0.00960.9122       ±0.0119
PanDiff[68]                0.8881±0.1197    0.7461±0.1032    0.9792±0.0097    0.9887±0.0020                       0.0265±0.0195    0.0729±0.0103    0.9025±0.0209
DDIF(ours)                 0.6408±0.12030.5668       ±0.10100.9855       ±0.00780.9859       ±0.0035              0.0201±0.01090.0408       ±0.01030.9398       ±0.0137
Idealvalue                          0                    0                    1                    1                       0                    0                    1

<!-- PDF PAGE 13 -->

          Figure 8: Visual comparisons on the QB dataset. The first and third rows show the RGB bands of the fused images. The second and fourth rows show the error
          mapsbetweentheGTandfusedimages. Someclose-upsaredepictedinredandyellowrectangles.  Thenumericalerrorsaredisplayedonthecolorbar,withdarker
          colors indicating closer proximity to the ground-truth (GT) (i.e., better fusion performance).


          Table 4: Quantitative results on the QuickBird reduced-resolution and full-resolution datasets.  Some conventional methods (the first three rows) and DL-based
          methods are compared. The best results are in red and the second best results are in blue.




















          4.8. AblationStudy                                                                           ify their effectiveness.
             In  this  section,  ablation  studies  are  conducted  on  CSM,Reduced                                                          Full
method    FWM, and the other performance-boosting techniques to ver-SAM(±std)      ERGAS(±std)        Q4(±std)          SCC(±std)Dλ(±std)           Ds(±std)         HQNR(±std)
BDSD-PC[100]               8.2620±2.0497    7.5420±0.8138    0.8323±0.1013    0.9030±0.018113                    0.1975±0.0334    0.1636±0.0483    0.6722±0.0577
MTF-GLP-FS[101]            8.1131±1.9553    7.5102±0.7926    0.8296±0.0905    0.8998±0.0196                      0.0489±0.0149    0.1383±0.0238    0.8199±0.0340
BT-H[99]                   7.1943±1.5523    7.4008±0.8378    0.8326±0.0880    0.9156±0.0152                      0.2300±0.0718    0.1648±0.0167    0.6434±0.0645
PNN[17]                    5.2054±0.9625    4.4722±0.3734    0.9180±0.0938    0.9711±0.0123                      0.0569±0.0112    0.0624±0.0239    0.8844±0.0304
DiCNN[38]                  5.3795±1.0266    5.1354±0.4876    0.9042±0.0942    0.9621±0.0133                      0.0920±0.0143    0.1067±0.0210    0.8114±0.0310
MSDCNN[39]                 5.1471±0.9342    4.3828±0.3400    0.9188±0.0966    0.9689±0.0121                      0.0602±0.0150    0.0667±0.0289    0.8774±0.0388
FusionNet[19]              4.9226±0.9077    4.1594±0.3212    0.9252±0.0902    0.9755±0.0104                      0.0586±0.0189   0.0522±0.00880.8922       ±0.0219
CTINN[103]                 4.6583±0.7755    3.6969±0.2888    0.9320±0.0072    0.9829±0.0072                      0.1738±0.0332    0.0731±0.0237    0.7663±0.0432
LAGConv[45]                4.5473±0.8296   3.8259±0.41960.9335       ±0.08780.9807       ±0.0091                 0.0844±0.0238    0.0676±0.0136    0.8536±0.0178
MMNet[102]                 4.5568±0.7285    3.6669±0.3036    0.9337±0.0941    0.9829±0.0070                      0.0890±0.0512    0.0972±0.0382    0.8225±0.0319
DCFNet[40]                 4.5383±0.7397    3.8315±0.2915    0.9325±0.0903    0.9741±0.0101                      0.0454±0.0147    0.1239±0.0269    0.8360±0.0158
PanDiff[68]                4.5754±0.7359    3.7422±0.3099    0.9345±0.0902    0.9818±0.0902                      0.0587±0.0223    0.0642±0.0252    0.8813±0.0417
DDIF(ours)                 4.3496±0.73133.5223       ±0.27030.9375       ±0.09040.9845       ±0.0069             0.0583±0.0126   0.0492±0.01030.8954       ±0.0206
Idealvalue                         0                    0                    1                    1                      0                    0                    1

<!-- PDF PAGE 14 -->

            Figure 9: Fused WV3 full-resolution data and their corresponding HQNR map. The high value in the HQNR map means better full-resolution fusion performance.


            4.8.1. DisentangledModulation                                                               Table5: Ablationstudyonthetwomodulationmodules. Bestresultsareinred.
                We propose two disentangled conditional modulation mod-
            ules, dubbed CSM and FWM, which are respectively responsi-
            ble for coarse-grain style condition modulation and fine-grain
            frequency modulation. Without using them, one intuitive way
            is to concatenate conditions (i.e., the LRMS and PAN images)
            with the noisy input along the channel dimension.  We ablate
            the proposed two modulations by training the diffusion U-Net                                  Table 6: Ablation study on using residual learning. Best results are in red.
            in the following forms:
               1.Only concatenating conditions and input together as pro-
                   posed in [86,74].
               2.Only using CSM.
               3.Conducting two modulations (i.e., CSM and FWM) in the
                   diffusion U-Net.                                                                     Table 7:  Ablation study on training objectives of the diffusion model.  Best
                                                                                                        results are in red.
                Each of the above-mentioned models has been trained until
            convergence for a fair comparison.  The performance on the
            WV3 reduced-resolution test set is reported in Tab.5. With the
            addition of the style modulation and the wavelet modulation,
            the fusion performance exhibits a monotonically increase.  It
            should be noted that without these two modulations, our DDIF
            degrades to DDPM [23] just adding the residual learning, thus                               4.8.2. ResidualLearning
            indicating that directly applying DDPM to the fusion task has                                  Toverifytheeffectivenessoftheresiduallearningtechnique,
            poor performance.                                                                           we return the input to the noisy GT asxt. Then, we retrain the
                                                                                                   14






StyleTransferObjectiveResidualWaveletSAM(±std)      ERGAS(±std)       Q8(±std)         SCC(±std)SAM(±std)      ERGAS(±std)       Q8(±std)         SCC(±std)SAM(±std)     ERGAS(±std)      Q8(±std)        SCC(±std)
 ModulationLearningModulation
     ✗         ✗ϵ3.7702±0.6397   2.7954±0.6516   0.8388±0.1181   0.9794±0.00643.2851±0.6828  2.5501±0.6141  0.8981±0.0904  0.9796±0.0067
     ✓         ✗✗v3.2397±0.4546    3.2096±1.003    0.9061±0.0837   0.9729±0.01133.4853±0.6080   2.6624±0.5912   0.8715±0.1025   0.9808±0.00523.1418±0.5789  2.3527±0.5116  0.8971±0.0940  0.9837±0.0047
     ✓         ✓✓x02.7386±0.50802.0165      ±0.45080.9202      ±0.08240.9882      ±0.00312.7386±0.50802.0165      ±0.45080.9202      ±0.08240.9882      ±0.00312.7386±0.50802.0165     ±0.45080.9202     ±0.08240.9882     ±0.0031

<!-- PDF PAGE 15 -->

                                                                                  tropy. We computed the pixel entropy of the GT and the resid-
                                                                                  ualimage(i.e.,HRMS−LRMS)andwefoundthattheentropy
                                                                                  oftheGTis6.498bits-per-pixel(bpp)andtheoneoftheresid-
                                                                                  ual image is only 4.461 bpp, which indicates that the residual
                                                                                  image distribution is simpler.    The entropy, HI, is defined as
                                                                                  follows:                           CX X
                                                                                                         HI = 1C             plog2(p),                      (17)
                                                                                                                    c=0 p∈Ic
                                                                                  whereC is the number of channels, Ic denotes the channel im-
                                                                                  age, and p is the value of a pixel.

                                                                                  4.8.3. TrainingObjectives
                                                                                     Thetrainingobjectivesofthemodelcanbeϵ,x0,orv,where
                                                                                  ϵ indicates that the model needs to predict the added Gaussian
                                                                                  noise, x0 denotes that the model predicts the original (noise-
                                                                                  less) image from the noisy image, and v reflects the fact that
                                                                                  themodelpredictstheweightedsumofϵ andx0. Weseparately
                                                                                  treat these three training objectives, ensuring the convergence
                                                                                  of the diffusion model in all three cases on the WV3 dataset.
                                                                                  Their performance is reported in Tab.7.  As we can see, pre-
                                                                                  dictingx0 gets the best performance. The backward process of
                                                                                  the three objectives is illustrated in Fig.11.  Starting from the
                                                                                  same xT, the x0 objective denoises much faster than the other
                                                                                  two options. We guess that for small-scale datasets,x0 is more
                                                                                  straightforward than ϵ and v since the sample density is more
Figure 10: Changes in loss (a) and SAM metric (b) on the WV3 dataset of our       concentrated, but, for large-scale datasets, predicting ϵ and v
DDIF over iterations with and without the residual learning technique (w/ res,    may force the network to learn detailed denoising trajectory,
w/o res).                                                                         which is useful to generate images with high-level conditions,
                                                                                  such as, classification labels [116], texts [58], and bounding
                                                                                  boxes [117].


                                                                                  5. Discussion

                                                                                     Inthissection,wewillanalyzethegeneralizationability,and
                                                                                  then extend our DDIF to the MHIF task. Finally, we will com-
                                                                                  paretheproposedapproachwithtraditionaldeepregressiveand
                                                                                  GAN-based models.

                                                                                  5.1. GeneralizationAbilityonWorldView-2Dataset
                                                                                     Thankstothepowerfulfittingabilityofneuralnetworks,they
                                                                                  often perform well on data sharing the same domain.  How-
                                                                                  ever,oncetestdataareshiftedtoadifferentdomain(neverseen
                                                                                  by the network during the training), the network usually per-
                                                                                  forms poorly. To assess the generalization ability, we evaluate
Figure  11:   The  denoised xt  and  the  related  SAM  metrics  over  different  our DDIF model, trained on the WV3 dataset, in a zero-shot
timesteps for different training objectives focusing on pansharpening. (a), (b),  manner5 on WorldView-2 data.  We also compare our method
and (c) represent the training objectivesx0,v, andϵ, respectively.                with other DL-based methods as reported in Tab.8.  The pro-
                                                                                  posed method achieves competitive results, thus demonstrating
diffusion model until the model converges calculating its per-                    its good generalization.  These results are in agreement with
formanceontheWV3datasetasshowninTab.6. Withresidual                               somerelatedworksthatapplythediffusionmodelinazero-shot
learning,thediffusionmodelcangetfusedimagesclosertothe                            manner to tasks as image inverse problem [76], image classifi-
GT.Besides,thelossconvergenceisfaster,asshowninFig.10.                            cation [118], and semantic segmentation [119].
One hypothesis is that the residual distribution is simpler than
theGTdistribution,thuslearninginamoreefficientmanner. A                              5“Zero-shot”meansthatwedirectlyusethetrainedmodeltotestonanother
direct way to express the complexity of a distribution is the en-                 dataset without any fine-tuning.
                                                                              15

<!-- PDF PAGE 16 -->

          Figure 12: The first and third rows show the natural color results on “chart and stuffed toy”. Some close-ups are depicted in the red and yellow rectangles. The
          second and fourth rows show the error maps between the GT and the fused images. The darker blue error maps indicate better performance.


                                 Table 8: Generalization ability of DL-based methods. The best results are in red and the second best results are in blue.
















          5.2. ExtensiontotheMHIFTask                                                              higher PSNR and a higher Q2n, as well as competitive perfor-
                                                                                                   mance on the other metrics. Fusion images and error maps are
             We additionally conduct experiments on the CAVE dataset.                              depicted in Fig.12. To analyze the spectral accuracy, the spec-
          The results and comparisons with other methods are provided                              tral signatures, obtained by extracting two different pixels in
          in Tab.9. It is worth noting that, compared to the state-of-the-                         two testing images, are compared in Fig.13showing that our
          artMHIFmethod,i.e.,DHIF[33],ourmethodachievesa0.1dB
method                                                   Reduced                              16                                       Full
                       SAM(±std)      ERGAS(±std)        Q4(±std)          SCC(±std)                          Dλ(±std)            Ds(±std)         HQNR(±std)
PNN[17]              7.1158±1.6812    5.6152±0.9431    0.7619±0.0928    0.8782±0.0175                      0.1484±0.0957    0.0771±0.0169    0.7869±0.0959
DiCNN[38]            6.9216±0.7898    6.2507±0.5745    0.7205±0.0746    0.8552±0.0289                      0.1412±0.0661    0.1023±0.0195    0.7700±0.0505
MSDCNN[39]           6.0064±0.6377    4.7438±0.4939   0.8241±0.0799    0.8972±0.0109                       0.0589±0.0421   0.0290±0.01380.9143       ±0.0516
FusionNet[19]        6.4257±0.8602    5.1363±0.5151    0.7961±0.0737    0.8746±0.0134                      0.0519±0.0292    0.0559±0.0146    0.8948±0.0187
CTINN[103]           6.4103±0.5953    4.6435±0.3792    0.8172±0.0873   0.9147±0.0102                       0.1722±0.0373    0.0375±0.0065    0.7967±0.0360
LAGConv[45]          6.9545±0.4739    5.3262±0.3185    0.8054±0.0837    0.9125±0.0101                      0.1302±0.0856    0.0547±0.0159    0.8229±0.0884
MMNet[102]           6.6109±0.3209    5.2213±0.2133    0.8143±0.0790    0.9136±0.0201                      0.0897±0.0340    0.0688±0.0209    0.8476±0.0569
DCFNet[40]           5.6194±0.6039   4.4887±0.37640.8292       ±0.08150.9154       ±0.0083                 0.0649±0.0357    0.0700±0.0219    0.8690±0.0233
DDIF(ours)           5.3827±0.57374.6712       ±0.4155    0.8217±0.0777    0.8993±0.0129                   0.0313±0.03760.0312       ±0.01110.9388       ±0.0453
Idealvalue                   0                     0                     1                     1                   0                     0                     1

<!-- PDF PAGE 17 -->

          Table 9: Quantitative results on the CAVE (×4) dataset. Some conventional methods and DL-based approaches are compared. The best results are in red and the
          second best results are in blue.


















                                                                                                 pared to other DL-based methods, considering that DDIF is
                                                                                                 an iterative approach, the proposed method’s runtime remains
                                                                                                 within an acceptable range.   It is worth noting that the run-
                                                                                                 timeofFusFormer[32]issignificantlyhigherthanthatofother
                                                                                                 methods. This is attributed to FusFormer’s inability to directly
                                                                                                 operateona512×512image. Itrequirestheimagetobeparti-
                                                                                                 tionedintosmallerpatches,andthenetworkperformsfusionon
                                                                                                 each patch before assembling them back together, resulting in
                                                                                                 anextendedruntime. Ourdesignednetworkdoesnotencounter
                                                                                                 this issue.

                                                 (a)                                             5.3. ComparisonswithOtherUnsupervisedMethods
                                                                                                    Unsupervised methods are directly trained at full-resolution,
                                                                                                 thus getting good performance when full-resolution metrics
                                                                                                 are considered.  In contrast, our DDIF only accesses reduced-
                                                                                                 resolution data during its training.   This train-test resolution
                                                                                                 mismatch poses challenges for supervised methods.   We se-
                                                                                                 lectedarecentstate-of-the-artunsupervisedregressivemethod,
                                                                                                 LDPNet [54], and a representative GAN-based method, Zer-
                                                                                                 GAN [51], for comparison.   Tab.11shows the comparisons
                                                                                                 with LDPNet and ZerGAN at full-resolution.  It can be seen
                                                                                                 that,  even without accessing full-resolution data during the
                                                                                                 training, our DDIF can still generalize well and obtain high
                                                                                                 performance on full-resolution data in comparison with state-
                                                 (b)                                             of-the-art unsupervised methods.

          Figure 13: Spectral vectors of the compared approaches: (a) spectral vectors           5.4. DifferenceswithRegressiveModels
          in “feather” located at position (400, 200), (b) spectral vectors in “chart and           Regressive models can be trained in both the supervised and
          stuffed toy” located at position (400, 200).  The horizontal and vertical axes         unsupervisedmanner. Themainstreamworks,suchas[19,40],
          represent the band number and the pixel values, respectively.                          focus  on  designing  powerful  architectures  or  neural  opera-
                                                                                                 tors [45] and then training the model in a supervised manner.
          DDIF has a more accurate spectral consistency.                                         Other works [120] chose to bound the regressive mapping by
                                                                                                 adding a regularized loss on fused outcomes training in an un-
             To better illustrate the superiority of our DDIF on real hy-                        supervised way. The proposed DDIF is still a supervised train-
          perspectraldata,weconductedfusionexperimentsontheGF5-                                  ing model, which benefits from the noiseless supervised sig-
          GF1dataset. AsshowninTab.10,ourapproachachievesstate-                                  nal from the GT rather than more complex regularized terms.
          of-the-artperformanceonreduced-resolutiondata,alsodemon-                               However,ourmethodcanalsobetransformedintoanapproach
          strating excellent results at full-resolution.                                         that relies upon unsupervised training by adding constraints on
             WetesttheruntimeofvariousmethodsontheCAVEdataset.                                   the training loss, Lsimple (as done by most of the unsupervised
          Although our method exhibits a slightly longer runtime com-                            works).
method                   PSNR(±std)       SSIM(±std)        Q2n(±std)         SAM(±std)      ERGAS(±std)              Runtime(s)
CSTF-FUS[30]            34.4632±4.2806   0.8662±0.0747   0.6659±0.1586   14.3683±5.3020   8.2885±5.284817                12.766
IR-TenSR[29]            35.6081±3.4461   0.9451±0.0267   0.7774±0.1228   12.2950±4.6825   5.8969±3.0455                  14.073
LTTR[27]                35.8505±3.4883   0.9562±0.0288   0.8404±0.0979    6.9895±2.5542    5.9904±2.9211                241.207
LTMR[105]               36.5434±3.2995   0.9631±0.0208   0.8416±0.1031    6.7105±2.1934    5.3868±2.5286                174.807
MTF-GLP-HS[115]         37.6920±3.8528   0.9725±0.0158   0.8716±0.0847    5.3281±1.9119    4.5749±2.6605                 12.766
UTV[28]                 38.6153±4.0640   0.9410±0.0434   0.7752±0.1416    8.6488±3.3764    4.5189±2.8173                613.535
CNN-FUSE[31]            42.4281±3.1994   0.9782±0.0079   0.9419±0.0240    5.7599±2.1509    2.8420±1.7590                 0.481
ResTFNet[41]            45.5842±5.4647   0.9939±0.0055   0.9581±0.0315    2.7643±0.6988    2.3134±2.4377                 0.485
SSRNet[42]              48.6196±3.9182   0.9954±0.0024   0.9598±0.0309    2.5415±0.8369    1.6358±1.2191                 0.594
Fusformer[32]           49.9831±8.0965   0.9943±0.0114   0.9624±0.0362    2.2033±0.8510    2.5337±5.3052                 17.143
HSRNet[109]             50.3805±3.3802   0.9970±0.0015   0.9666±0.0290    2.2272±0.6575    1.2002±0.7506                 0.690
DHIF[33]                51.0721±4.16480.9973      ±0.00170.9695      ±0.02672.0080      ±0.63041.2216      ±0.9653       0.837
DDIF(ours)              51.1758±4.61480.9971      ±0.00260.9737      ±0.01062.0952      ±0.6471    1.2996±1.2822         10.706
Idealvalue                     ∞                   1                   1                   0                   0           0

<!-- PDF PAGE 18 -->

          Table 10: Result on the GF5-GF1 reduced-resolution and full-resolution datasets. Some conventional methods (the first six rows) and the DL-based approaches are
          compared. The best results are in red and the second best results are in blue.






















          Table 11:  Quantitative comparisons with recent state-of-the-art unsupervised                FWM,toleverageonthelearningabilitybyusingDPM.Exper-
          methods on the WV3 full-resolution dataset.                                                  iments conducted on widely used pansharpening datasets and
                                                                                                       an additional MHIF set of data demonstrated that the proposed
                                                                                                       approachcanoutperform(bothqualitativelyandquantitatively)
                                                                                                       recent state-of-the-art image fusion approaches.  Furthermore,
                                                                                                       we provided to the readers some discussions on the proposed
                                                                                                       method, even showing ablation studies to verify the effective-
                                                                                                       ness of the proposed technique.
          5.5. DifferenceswithUnsupervisedGAN-basedModels
              GAN-based pansharpening models can be seen as an exten-                                  References
          sionofanunsupervisedregressivemodelbyaddingoneormul-
          tiple discriminators and a GAN loss6 [50,51]. Benefiting from                                  [1]A. Arienzo, G. Vivone, A. Garzelli, L. Alparone, J. Chanussot,  Full-
          additional supervised discriminator signals, those models can                                       resolution quality assessment of pansharpening: Theoretical and hands-
          get more realistic images. However, GAN-based models suffer                                         on approaches,   IEEE Geoscience and Remote Sensing Magazine 10
          from training instability and need careful tuning because of the                                    (2022) 168–201.
                                                                                                         [2]G.Vivone, Multispectralandhyperspectralimagefusioninremotesens-
          adversarial training. Our DDIF does not use adversarial train-                                      ing: A survey, Information Fusion 89 (2023) 405–417.
          ing exploiting a corrupt-reconstruct supervised training, which                                [3]G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O.
          avoids unstable training but still produces high-quality fused                                      Ulfarsson,L.Alparone,J.Chanussot, Anewbenchmarkbasedonrecent
          images.                                                                                             advancesinmultispectralpansharpening: Revisitingpansharpeningwith
                                                                                                              classical and emerging pansharpening methods,  IEEE Geoscience and
                                                                                                              Remote Sensing Magazine 9 (2020) 53–81.
                                                                                                         [4]P.S.Chavez,A.Y.Kwarteng, Extractingspectralcontrastinlandsatthe-
          6. Acknowledge                                                                                      matic mapper image data using selective principal component analysis,
                                                                                                              PhotogrammetricEngineeringandRemoteSensing55(1989)339–348.
              ThisresearchissupportedbyNationalKeyResearchandNa-                                         [5]C. A. Laben, B. V. Brower, Process for enhancing the spatial resolu-
          tional Natural Science Foundation of China (12271083), De-                                          tion of multispectral imagery using pan-sharpening, 2000. US Patent
          velopment Program of China (Grant No.  2020YFA0714001),                                             6,011,875.
                                                                                                         [6]B.Aiazzi,S.Baronti,M.Selva, Improvingcomponentsubstitutionpan-
          andNaturalScienceFoundationofSichuanProvince(2022NS-                                                sharpening through multivariate regression of ms + pan data,   IEEE
          FSC0501, 2023NSFSC1341, 2022NSFSC1821).                                                             TransactionsonGeoscienceandRemoteSensing45(2007)3230–3239.
                                                                                                         [7]J. Qu, Y. Li, W. Dong, Hyperspectral pansharpening with guided filter,
                                                                                                              IEEE Geoscience and Remote Sensing Letters 14 (2017) 2152–2156.
          7. Conclusions                                                                                 [8]X. Otazu, M. Gonz    ´alez-Aud´ıcana, O. Fors, J. N´u˜nez,    Introduction
                                                                                                              of sensor spectral response into image fusion methods. application to
              In this paper, we proposed a denoising diffusion model, the                                     wavelet-based methods, IEEE Transactions on Geoscience and Remote
          so-called DDIF, for two MSIF tasks. Our motivation is that su-                                      Sensing 43 (2005) 2376–2385.
                                                                                                         [9]J.Liu, Smoothingfilter-basedintensitymodulation: Aspectralpreserve
          pervisedDL-basedmethodssufferfromdegradedlearningabil-                                              imagefusiontechniqueforimprovingspatialdetails, InternationalJour-
          ity under the preconditioned framework and the conditions are                                       nal of Remote Sensing 21 (2000) 3461–3472.
          entangled, which is not suitable for the fusion task. Hence, we                               [10]B.Aiazzi,L.Alparone,S.Baronti,A.Garzelli,M.Selva, MTF-tailored
          designedtwonovelfeaturemodulationmodules,i.e.,CSMand                                                multiscale fusion of high-resolution ms and pan imagery,  Photogram-
                                                                                                              metric Engineering and Remote Sensing 72 (2006) 591–596.
                                                                                                        [11]G.Vivone,R.Restaino,M.DallaMura,G.Licciardi,J.Chanussot, Con-
                                                                                                              trast and error-based fusion schemes for multispectral image pansharp-
              6Note that GAN-based models can also be trained in a supervised manner,                         ening,  IEEE Geoscience and Remote Sensing Letters 11 (2013) 930–
          but,unfortunately,wedidnotfindthesekindsofworksintherelatedliterature.                              934.
method                                                           Simulated                        18                                             Real
                          PSNR(±std)       SSIM(±std)       Q2n(±std)        SAM(±std)     ERGAS(±std)                     Dλ(±std)          Ds(±std)       HQNR(±std)
CNMF[106]               44.2525±3.8884   0.9823±0.0122   0.7420±0.1767   0.8509±0.2133   2.7609±0.7674                  0.0447±0.0830   0.0592±0.0500   0.8979±0.0842
Hysure[107]             42.5184±4.5175   0.9728±0.0137   0.7323±0.1464   1.3046±0.4059   3.6770±1.3169                  0.0414±0.0760   0.0741±0.1046   0.8865±0.1169
GSA[108]                44.9948±5.3369   0.9795±0.0119   0.7544±0.1313   1.2003±0.3322   2.8978±0.9557                  0.0526±0.1029   0.0674±0.0615   0.8818±0.1006
LTTR[27]method          47.1451±2.9068   0.9897±0.0028   0.8442±0.0983   2.1593±0.2858   5.8080±2.7321Dλ                     Ds                  HQNR0.0989±0.1231   0.0465±0.0233   0.8596±0.1063
LTMR[105]               45.5163±2.4245   0.9898±0.0030   0.8494±0.1079   1.5950±0.3435   2.7199±1.2766                  0.0567±0.1029   0.0361±0.0177   0.9098±0.1048
MTF-GLP-HS[115]LDPNet[54]0.0235±0.0085    0.0364±0.0192    0.9411±0.019245.5954±5.8205   0.9837±0.0120   0.7772±0.1462   0.8561±0.2646   2.8475±1.15410.0303±0.0482   0.0747±0.1054   0.8966±0.1063
ZerGAN[51]          0.0221±0.0092    0.0210±0.0082    0.9574±0.0170
ResTFNet[41]            46.9763±2.1124   0.9934±0.0022   0.8504±0.1000   0.9060±0.1301   3.3234±3.1232                  0.0423±0.0782   0.0880±0.0587   0.8742±0.0994
SSRNet[42]DDIF(ours)0.0258±0.0187    0.0231±0.0075    0.9517±0.017345.4872±2.6866   0.9880±0.0047   0.8500±0.0942   1.0389±0.2101   4.8631±4.16050.1169±0.1401   0.0538±0.0190   0.8357±0.1341
Fusformer[32]           49.7373±4.6446   0.9914±0.0031   0.8908±0.0762   0.6382±0.1552   4.7612±0.5921                  0.0302±0.0558   0.0401±0.0250   0.9312±0.0636
HSRNet[109]             49.8109±3.0477   0.9964±0.0016   0.8883±0.0806   0.6925±0.1386   0.9006±0.4513                  0.0377±0.0734   0.0473±0.0201   0.9170±0.0748
DHIF[33]                55.3538±4.20070.9982      ±0.00090.9285      ±0.07580.3088      ±0.06220.8852      ±0.3882      0.0312±0.05740.0335      ±0.02150.9365      ±0.0624
DDIF(ours)              56.4022±3.81940.9984      ±0.00070.9382      ±0.06380.2726      ±0.04890.8449      ±0.5053      0.0327±0.0571   0.0350±0.02130.9333      ±0.0573
Idealvalue                     +∞                 1                  1                  0                  0                   0                  0                  1

<!-- PDF PAGE 19 -->

[12]X. He, L. Condat, J. M. Bioucas-Dias, J. Chanussot, J. Xia, A new pan-                                (2022) 1–5.
      sharpening method based on spatial and spectral sparsity priors,  IEEE                       [33]T. Huang, W. Dong, J. Wu, L. Li, X. Li, G. Shi,  Deep hyperspectral
      Transactions on Image Processing 23 (2014) 4160–4174.                                               imagefusionnetworkwithiterativespatio-spectralregularization, IEEE
[13]M. Moeller, T. Wittman, A. L. Bertozzi,  A variational approach to hy-                                Transactions on Computational Imaging 8 (2022) 201–214.
      perspectral image fusion,   in:  Algorithms and Technologies for Mul-                        [34]K.He,X.Zhang,S.Ren,J.Sun, Deepresiduallearningforimagerecog-
      tispectral, Hyperspectral, and Ultraspectral Imagery XV, volume 7334,                               nition, in: ProceedingsoftheIEEEConferenceonComputerVisionand
      SPIE, 2009, pp. 502–511.                                                                            Pattern Recognition (CVPR), 2016, pp. 770–778.
[14]T.Wang,F.Fang,F.Li,G.Zhang, High-qualitybayesianpansharpening,                                 [35]O. Ronneberger, P. Fischer, T. Brox,   U-net:  Convolutional networks
      IEEE Transactions on Image Processing 28 (2018) 227–239.                                            for biomedical image segmentation, in: Medical Image Computing and
[15]J. Duran, A. Buades, B. Coll, C. Sbert,  A nonlocal variational model                                 Computer-Assisted Intervention (MICCAI), Springer, 2015, pp. 234–
      for pansharpening image fusion, SIAM Journal on Imaging Sciences 7                                  241.
      (2014) 761–796.                                                                              [36]Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin
[16]L.-J. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang,                                    transformer: Hierarchical vision transformer using shifted windows, in:
      J. Chanussot, A. Plaza,  Machine learning in pansharpening: A bench-                                Proceedings of the IEEE/CVF International Conference on Computer
      mark, from shallow to deep networks,  IEEE Geoscience and Remote                                    Vision (ICCV), 2021, pp. 10012–10022.
      Sensing Magazine 10 (2022) 279–315.                                                          [37]W. Wang, E. Xie, X. Li, D.-P. Fan, K. Song, D. Liang, T. Lu, P. Luo,
[17]G. Masi, D. Cozzolino, L. Verdoliva, G. Scarpa, Pansharpening by con-                                 L. Shao,  Pyramid vision transformer: A versatile backbone for dense
      volutional neural networks, Remote Sensing 8 (2016) 594.                                            prediction without convolutions,  in: Proceedings of the IEEE/CVF In-
[18]J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. W. Paisley,  PanNet:  A                                  ternationalConferenceonComputerVision(ICCV),2021,pp.568–578.
      deep network architecture for pan-sharpening,  in: Proceedings of the                        [38]L.He,Y.Rao,J.Li,J.Chanussot,A.Plaza,J.Zhu,B.Li, Pansharpening
      IEEE/CVFInternationalConferenceonComputerVision(ICCV),2017,                                         via detail injection based convolutional neural networks, IEEE Journal
      pp. 1753–1761.                                                                                      of Selected Topics in Applied Earth Observations and Remote Sensing
[19]L.-J.Deng,G.Vivone,C.Jin,J.Chanussot, Detailinjection-baseddeep                                       12 (2019) 1188–1204.
      convolutionalneuralnetworksforpansharpening, IEEETransactionson                              [39]Q. Yuan, Y. Wei, X. Meng, H. Shen, L. Zhang,  A multiscale and mul-
      Geoscience and Remote Sensing 59 (2020) 6995–7010.                                                  tidepth convolutional neural network for remote sensing imagery pan-
[20]T.-J. Zhang, L.-J. Deng, T.-Z. Huang, J. Chanussot, G. Vivone,    A                                   sharpening, IEEEJournalofSelectedTopicsinAppliedEarthObserva-
      triple-double convolutional neural network for panchromatic sharpen-                                tions and Remote Sensing 11 (2018) 978–989.
      ing,   IEEE Transactions on Neural Networks and Learning Systems                             [40]X. Wu, T.-Z. Huang, L.-J. Deng, T.-J. Zhang,    Dynamic cross fea-
      (2022).                                                                                             ture fusion for remote sensing pansharpening,  in:  Proceedings of the
[21]Z.-C. Wu, T.-Z. Huang, D. Liang-Jian, J. Hu, G. Vivone,   VO     +Net:                                IEEE/CVFInternationalConferenceonComputerVision(ICCV),2021,
      An adaptive approach using variational optimization and deep learning                               pp. 14687–14696.
      for panchromatic sharpening,   IEEE Transactions on Geoscience and                           [41]X. Liu, Q. Liu, Y. Wang,  Remote sensing image fusion based on two-
      Remote Sensing (2021) 1–16.                                                                         stream fusion network, Information Fusion 55 (2020) 1–15.
[22]C. Wu, B. Du, X. Cui, L. Zhang, A post-classification change detection                         [42]X. Zhang, W. Huang, Q. Wang, X. Li,  SSR-NET: Spatial–spectral re-
      methodbasedoniterativeslowfeatureanalysisandbayesiansoftfusion,                                     construction network for hyperspectral and multispectral image fusion,
      Remote Sensing of Environment 199 (2017) 241–255.                                                   IEEETransactionsonGeoscienceandRemoteSensing59(2020)5953–
[23]J. Ho, A. Jain, P. Abbeel,   Denoising di     ffusion probabilistic models,                           5965.
      in: AdvancesinNeuralInformationProcessingSystems(NeurIPS),vol-                               [43]X. Wang, Q. Hu, J. Jiang, J. Ma,  A group-based embedding learning
      ume 33, 2020, pp. 6840–6851.                                                                        andintegrationnetworkforhyperspectralimagesuper-resolution, IEEE
[24]C. A. Bishop, J. G. Liu, P. J. Mason, Hyperspectral remote sensing for                                Transactions on Geoscience and Remote Sensing 60 (2022) 1–16.
      mineral exploration in pulang, yunnan province, china,   International                       [44]S. Peng, L.-J. Deng, J.-F. Hu, Y.-W. Zhuo,   Source-adaptive discrim-
      Journal of Remote Sensing 32 (2011) 2409–2426.                                                      inative kernels based network for remote sensing pansharpening,   in:
[25]T. A. Carrino, A. P. Cr      ´osta, C. L. B. Toledo, A. M. Silva,  Hyperspec-                         International Joint Conference on Artificial Intelligence (IJCAI), 2022.
      tral remote sensing applied to mineral exploration in southern peru: A                       [45]Z.-R. Jin, T.-J. Zhang, T.-X. Jiang, G. Vivone, L.-J. Deng,  LAGConv:
      multipledataintegrationapproachinthechapichiaragoldprospect, In-                                    Local-context adaptive convolution kernels with global harmonic bias
      ternational Journal of Applied Earth Observation and Geoinformation                                 for pansharpening,    in:  AAAI Conference on Artificial Intelligence
      64 (2018) 287–300.                                                                                  (AAAI), volume 36, 2022, pp. 1113–1121.
[26]M. D. Hossain, D. Chen, Segmentation for object-based image analysis                           [46]X.Wang,Y.Cheng,X.Mei,J.Jiang,J.Ma, Groupshu     ffleandspectral-
      (obia): A review of algorithms and challenges from remote sensing per-                              spatial fusion for hyperspectral image super-resolution,  IEEE Transac-
      spective,  ISPRS Journal of Photogrammetry and Remote Sensing 150                                   tions on Computational Imaging 8 (2022) 1223–1236.
      (2019) 115–134.                                                                              [47]X. Tian, K. Li, W. Zhang, Z. Wang, J. Ma,  Interpretable model-driven
[27]R. Dian, S. Li, L. Fang, Learning a low tensor-train rank representation                              deep network for hyperspectral, multispectral, and panchromatic image
      for hyperspectral image super-resolution, IEEE Transactions on Neural                               fusion,  IEEE Transactions on Neural Networks and Learning Systems
      Networks and Learning Systems 30 (2019) 2672–2683.                                                  (2023) 1–14.
[28]T. Xu, T.-Z. Huang, L.-J. Deng, X.-L. Zhao, J. Huang,  Hyperspectral                           [48]X. Tian, K. Li, Z. Wang, J. Ma, VP-Net: An interpretable deep network
      image superresolution using unidirectional total variation with tucker                              for variational pansharpening,  IEEE Transactions on Geoscience and
      decomposition,  IEEE Journal of Selected Topics in Applied Earth Ob-                                Remote Sensing 60 (2022) 1–16.
      servations and Remote Sensing 13 (2020) 4381–4398.                                           [49]D. Liu, J. Li, Q. Yuan, L. Zheng, J. He, S. Zhao, Y. Xiao,  An e     fficient
[29]T. Xu, T.-Z. Huang, L.-J. Deng, N. Yokoya, An iterative regularization                                unfolding network with disentangled spatial-spectral representation for
      methodbasedontensorsubspacerepresentationforhyperspectralimage                                      hyperspectralimagesuper-resolution, InformationFusion94(2023)92–
      super-resolution,IEEETransactionsonGeoscienceandRemoteSensing                                       111.
      60 (2022) 1–16.                                                                              [50]J. Ma, W. Yu, C. Chen, P. Liang, X. Guo, J. Jiang,  Pan-gan: An unsu-
[30]S. Li, R. Dian, L. Fang, J. M. Bioucas-Dias,   Fusing hyperspectral                                   pervised pan-sharpening method for remote sensing image fusion,  In-
      and multispectral images via coupled sparse tensor factorization, IEEE                              formation Fusion 62 (2020) 110–120.
      Transactions on Image Processing 27 (2018) 4118–4130.                                        [51]W. Diao, F. Zhang, J. Sun, Y. Xing, K. Zhang, L. Bruzzone,  Zergan:
[31]R. Dian, S. Li, X. Kang,  Regularizing hyperspectral and multispectral                                Zero-referenceganforfusionofmultispectralandpanchromaticimages,
      image fusion by cnn denoiser,  IEEE Transactions on Neural Networks                                 IEEETransactionsonNeuralNetworksandLearningSystems(2022)1–
      and Learning Systems 32 (2021) 1124–1135.                                                           15.
[32]J.-F. Hu, T.-Z. Huang, L.-J. Deng, H.-X. Dou, D. Hong, G. Vivone,                              [52]Q. Xu, Y. Li, J. Nie, Q. Liu, M. Guo,   Upangan:  Unsupervised pan-
      Fusformer:  A transformer-based fusion network for hyperspectral im-                                sharpening based on the spectral and spatial loss constrained generative
      age super-resolution, IEEE Geoscience and Remote Sensing Letters 19                                 adversarial network, Information Fusion 91 (2023) 31–46.

                                                                                            19

<!-- PDF PAGE 20 -->

[53]S. Shi, L. Zhang, Y. Altmann, J. Chen,   Unsupervised hyperspectral                                  Pattern Recognition (CVPR), 2022, pp. 10619–10629.
      and multispectral images fusion based on the cycle consistency,  arXiv                       [74]C.Saharia,J.Ho,W.Chan,T.Salimans,D.J.Fleet,M.Norouzi, Image
      preprint arXiv:2307.03413 (2023).                                                                  super-resolution via iterative refinement, IEEE Transactions on Pattern
[54]J. Ni, Z. Shao, Z. Zhang, M. Hou, J. Zhou, L. Fang, Y. Zhang, Ldp-net:                               Analysis and Machine Intelligence 45 (2022) 4713–4726.
      Anunsupervisedpansharpeningnetworkbasedonlearnabledegradation                                [75]B. Kawar, M. Elad, S. Ermon, J. Song,   Denoising di     ffusion restora-
      processes,  IEEE Journal of Selected Topics in Applied Earth Observa-                              tion models,  in: Advances in Neural Information Processing Systems
      tions and Remote Sensing 15 (2022) 5468–5479.                                                      (NeurIPS), volume 35, 2022, pp. 23593–23606.
[55]A. Q. Nichol, P. Dhariwal,  Improved denoising di     ffusion probabilistic                    [76]Y. Wang, J. Yu, J. Zhang,  Zero-shot image restoration using denoising
      models,   in:  International Conference on Machine Learning (ICML),                                diffusion null-space model,  in:  International Conference on Learning
      PMLR, 2021, pp. 8162–8171.                                                                         Representations (ICLR), 2023.
[56]T. Karras, M. Aittala, T. Aila, S. Laine, Elucidating the design space of                      [77]W. Wang, J. Bao, W. Zhou, D. Chen, D. Chen, L. Yuan, H. Li,  Sindif-
      diffusion-based generative models, in: Advances in Neural Information                              fusion: Learning a diffusion model from a single natural image,  arXiv
      Processing Systems (NeurIPS), volume 35, 2022, pp. 26565–26577.                                    preprint arXiv:2211.12445 (2022).
[57]C.  Saharia,  W.  Chan,  S.  Saxena,  L.  Li,  J.  Whang,  E.  L.  Denton,                     [78]J.Zhang,J.Guo,S.Sun,J.-G.Lou,D.Zhang, Layoutdi     ffusion: Improv-
      K.Ghasemipour,R.GontijoLopes,B.KaragolAyan,T.Salimans,etal.,                                       inggraphiclayoutgenerationbydiscretediffusionprobabilisticmodels,
      Photorealistic text-to-image diffusion models with deep language un-                               arXiv preprint arXiv:2303.11589 (2023).
      derstanding,  in: Advances in Neural Information Processing Systems                          [79]Y. Song, S. Ermon, Generative modeling by estimating gradients of the
      (NeurIPS), volume 35, 2022, pp. 36479–36494.                                                       datadistribution, in: Advancesinneuralinformationprocessingsystems
[58]R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer,   High-                                     (NeurIPS), volume 32, 2019.
      resolutionimagesynthesiswithlatentdiffusionmodels, in: Proceedings                           [80]I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley,
      oftheIEEE/CVFConferenceonComputerVisionandPatternRecogni-                                          S. Ozair, A. Courville, Y. Bengio,   Generative adversarial nets,   in:
      tion (CVPR), 2022, pp. 10684–10695.                                                                Advances in Neural Information Processing Systems (NeurIPS), vol-
[59]C. Saharia, W. Chan, H. Chang, C. Lee, J. Ho, T. Salimans, D. Fleet,                                 ume 27, 2014.
      M. Norouzi, Palette: Image-to-image diffusion models, in: ACM SIG-                           [81]M. Arjovsky, S. Chintala, L. Bottou, Wasserstein generative adversarial
      GRAPH 2022 Conference Proceedings (ACM SIGGRAPH), 2022, pp.                                        networks,  in: International Conference on Machine Learning (ICML),
      1–10.                                                                                              PMLR, 2017, pp. 214–223.
[60]X.Li,J.Thickstun,I.Gulrajani,P.S.Liang,T.B.Hashimoto, Di     ffusion-                          [82]T. Karras, S. Laine, T. Aila,  A style-based generator architecture for
      lm improves controllable text generation,  in: Advances in Neural In-                              generativeadversarialnetworks, in: ProceedingsoftheIEEE/CVFCon-
      formation Processing Systems (NeurIPS), volume 35, 2022, pp. 4328–                                 ferenceonComputerVisionandPatternRecognition(CVPR),2019,pp.
      4343.                                                                                              4401–4410.
[61]Y.Song,J.Sohl-Dickstein,D.P.Kingma,A.Kumar,S.Ermon,B.Poole,                                    [83]T.Karras,S.Laine,M.Aittala,J.Hellsten,J.Lehtinen,T.Aila, Analyz-
      Score-based generative modeling through stochastic differential equa-                              ing and improving the image quality of StyleGAN,  in: Proceedings of
      tions, in: InternationalConferenceonLearningRepresentations(ICLR),                                 theIEEE/CVFConferenceonComputerVisionandPatternRecognition
      2021.                                                                                              (CVPR), 2020.
[62]Y. Song, S. Ermon,   Generative modeling by estimating gradients of                            [84]L. Dinh, J. Sohl-Dickstein, S. Bengio,   Density estimation using real
      the data distribution,  in:  Advances in Neural Information Processing                             NVP,in: InternationalConferenceonLearningRepresentations(ICLR),
      Systems (NeurIPS), volume 32, 2019.                                                                2017.
[63]J. Ho, T. Salimans, Classifier-free di     ffusion guidance, in: Advances in                   [85]D. P. Kingma, P. Dhariwal,  Glow: Generative flow with invertible 1x1
      Neural Information Processing Systems (NeurIPS), Workshop on Deep                                  convolutions,  in: Advances in Neural Information Processing Systems
      Generative Models and Downstream Applications, 2022.                                               (NeurIPS), volume 31, 2018.
[64]J. Song, C. Meng, S. Ermon,  Denoising di     ffusion implicit models,  in:                    [86]H. Li, Y. Yang, M. Chang, S. Chen, H. Feng, Z. Xu, Q. Li, Y. Chen,
      International Conference on Learning Representations (ICLR), 2021.                                 Srdiff: Single image super-resolution with diffusion probabilistic mod-
[65]C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, J. Zhu, Dpm-solver: A fast ode                               els, Neurocomputing 479 (2022) 47–59.
      solver for diffusion probabilistic model sampling in around 10 steps,                        [87]S.Elfwing,E.Uchibe,K.Doya, Sigmoid-weightedlinearunitsforneu-
      in: AdvancesinNeuralInformationProcessingSystems(NeurIPS),vol-                                     ral network function approximation in reinforcement learning,  Neural
      ume 35, 2022, pp. 5775–5787.                                                                       Networks 107 (2018) 3–11.
[66]W.G.C.Bandara,N.G.Nair,V.M.Patel, DDPM-CD:Remotesensing                                        [88]Y. Wu, K. He,  Group normalization,  in: Proceedings of the European
      change detection using denoising diffusion probabilistic models, arXiv                             Conference on Computer Vision (ECCV), 2018, pp. 3–19.
      preprint arXiv:2206.11892 (2022).                                                            [89]Y. Zhang, G. Hong, An ihs and wavelet integrated approach to improve
[67]J.Yue,L.Fang,S.Xia,Y.Deng,J.Ma, Dif-Fusion: Towardshighcolor                                         pan-sharpeningvisualqualityofnaturalcolourikonosandquickbirdim-
      fidelityininfraredandvisibleimagefusionwithdiffusionmodels, arXiv                                  ages, Information fusion 6 (2005) 225–234.
      preprint arXiv:2301.08072 (2023).                                                            [90]R. B. Gomez, A. Jazaeri, M. Kafatos, Wavelet-based hyperspectral and
[68]Q. Meng, W. Shi, S. Li, L. Zhang,   Pandi     ff:  A novel pansharpening                             multispectralimagefusion,in: Geo-SpatialImageandDataExploitation
      method based on denoising diffusion probabilistic model, IEEE Trans-                               II, volume 4383, SPIE, 2001, pp. 36–42.
      actions on Geoscience and Remote Sensing 61 (2023) 1–17. doi:10.                             [91]H. Phung, Q. Dao, A. Tran, Wavelet di     ffusion models are fast and scal-
      1109/TGRS.2023.3279864.                                                                            ableimagegenerators, in: ProceedingsoftheIEEE/CVFConferenceon
[69]J. Ma, L. Tang, F. Fan, J. Huang, X. Mei, Y. Ma,  Swinfusion: Cross-                                 Computer Vision and Pattern Recognition (CVPR), 2023, pp. 10199–
      domain long-range learning for general image fusion via swin trans-                                10208.
      former, IEEE/CAA Journal of Automatica Sinica 9 (2022) 1200–1217.                            [92]I. Daubechies, Ten lectures on wavelets, SIAM, 1992.
[70]L. Tang, Y. Deng, Y. Ma, J. Huang, J. Ma,   Superfusion:  A versa-                             [93]M. Zhou, J. Huang, K. Yan, H. Yu, X. Fu, A. Liu, X. Wei, F. Zhao,
      tile image registration and fusion network with semantic awareness,                                Spatial-frequency domain information integration for pan-sharpening,
      IEEE/CAA Journal of Automatica Sinica 9 (2022) 2121–2137.                                          in:   Proceedings  of  the  European  Conference  on  Computer  Vision
[71]Z.-X. Chen, C. Jin, T.-J. Zhang, X. Wu, L.-J. Deng,  Spanconv: A new                                 (ECCV), Springer, 2022, pp. 274–291.
      convolutionviaspanningkernelspaceforlightweightpansharpening,in:                             [94]L. Liu, Y. Ren, Z. Lin, Z. Zhao,  Pseudo numerical methods for di     ffu-
      International Joint Conference on Artificial Intelligence (IJCAI), 2022.                           sion models on manifolds,   in:  International Conference on Learning
[72]Y. Liang, P. Zhang, Y. Mei, T. Wang,  Pmacnet: Parallel multiscale at-                               Representations (ICLR), 2022.
      tention constraint network for pan-sharpening,  IEEE Geoscience and                          [95]I. Loshchilov, F. Hutter,   Decoupled weight decay regularization,   in:
      Remote Sensing Letters 19 (2022) 1–5.                                                              International Conference on Learning Representations (ICLR), 2019.
[73]K. Preechakul, N. Chatthee, S. Wizadwongsa, S. Suwajanakorn, Di     ffu-                             URL:https://openreview.net/forum?id=Bkg6RiCqY7.
      sion autoencoders: Toward a meaningful and decodable representation,                         [96]K.He,X.Zhang,S.Ren,J.Sun, Delvingdeepintorectifiers: Surpassing
      in: Proceedings of the IEEE/CVF Conference on Computer Vision and                                  human-levelperformanceonimagenetclassification, in: Proceedingsof

                                                                                            20

<!-- PDF PAGE 21 -->

        the IEEE international conference on computer vision, 2015, pp. 1026–                     [119]W. Wu, Y. Zhao, M. Z. Shou, H. Zhou, C. Shen, Di       ffuMask: Synthesiz-
        1034.                                                                                             ingimageswithpixel-levelannotationsforsemanticsegmentationusing
 [97]X. Rui, X. Cao, Z. Zhu, Z. Yue, D. Meng, Unsupervised pansharpening                                  diffusion models, arXiv preprint arXiv:2303.11681 (2023).
        via low-rank diffusion model, ArXiv abs/2305.10925 (2023).                                [120]H.Zhang,H.Wang,X.Tian,J.Ma, P2sharpen: Aprogressivepansharp-
 [98]A. Guo, R. Dian, S. Li, A deep framework for hyperspectral image fu-                                 eningnetworkwithdeepspectraltransformation, InformationFusion91
        sionbetweendifferentsatellites, IEEETransactionsonPatternAnalysis                                 (2023) 103–122.
        and Machine Intelligence 45 (2023) 7939–7954. doi:10.1109/TPAMI.
        2022.3229433.
 [99]S. Lolli, L. Alparone, A. Garzelli, G. Vivone,    Haze correction for
        contrast-based multispectral pansharpening,  IEEE Geoscience and Re-
        mote Sensing Letters 14 (2017) 2255–2259.
[100]G.  Vivone,      Robust  band-dependent  spatial-detail  approaches  for
        panchromatic sharpening,  IEEE Transactions on Geoscience and Re-
        mote Sensing 57 (2019) 6421–6433.
[101]G. Vivone, R. Restaino, J. Chanussot,  Full scale regression-based in-
        jection coefficients for panchromatic sharpening, IEEE Transactions on
        Image Processing 27 (2018) 3418–3431.
[102]K. Yan, M. Zhou, L. Zhang, C. Xie, Memory-augmented model-driven
        networkforpansharpening, in: ProceedingsoftheEuropeanConference
        on Computer Vision (ECCV), Springer, 2022, pp. 306–322.
[103]M. Zhou, J. Huang, Y. Fang, X. Fu, A. Liu,  Pan-sharpening with cus-
        tomized transformer and invertible neural network,  in: AAAI Confer-
        ence on Artificial Intelligence (AAAI), 2022.
[104]Q.Wei,N.Dobigeon, J.-Y.Tourneret, Fastfusionofmulti-bandimages
        based on solving a sylvester equation,   IEEE Transactions on Image
        Processing 24 (2015) 4109–4121.
[105]R.Dian,S.Li, Hyperspectralimagesuper-resolutionviasubspace-based
        low tensor multi-rank regularization, IEEE Transactions on Image Pro-
        cessing 28 (2019) 5135–5146.
[106]N.Yokoya,T.Yairi,A.Iwasaki, Couplednon-negativematrixfactoriza-
        tion(cnmf)forhyperspectralandmultispectraldatafusion: Application
        to pasture classification,  in: 2011 IEEE International Geoscience and
        Remote Sensing Symposium, 2011, pp. 1779–1782.
[107]M. Simoes, J. Bioucas-Dias, L. B. Almeida, J. Chanussot,  A convex
        formulationforhyperspectralimagesuperresolutionviasubspace-based
        regularization,  IEEE Transactions on Geoscience and Remote Sensing
        53 (2014) 3373–3388.
[108]B.Aiazzi,S.Baronti,M.Selva, Improvingcomponentsubstitutionpan-
        sharpening through multivariate regression of ms + pan data,   IEEE
        TransactionsonGeoscienceandRemoteSensing45(2007)3230–3239.
[109]J.-F. Hu, T.-Z. Huang, L.-J. Deng, T.-X. Jiang, G. Vivone, J. Chanussot,
        Hyperspectral image super-resolution via deep spatiospectral attention
        convolutional neural networks, IEEE Transactions on Neural Networks
        and Learning Systems 33 (2021) 7251–7265.
[110]R.H.Yuhas,A.F.Goetz,J.W.Boardman, Discriminationamongsemi-
        arid landscape endmembers using the spectral angle mapper (sam) al-
        gorithm,  in: JPL, Summaries of the Third Annual JPL Airborne Geo-
        science Workshop. Volume 1: AVIRIS Workshop, 1992.
[111]L. Wald, Data fusion: definitions and architectures: fusion of images of
        different spatial resolutions, Presses des MINES, 2002.
[112]A.  Garzelli,   F.  Nencini,       Hypercomplex  quality  assessment  of
        multi/hyperspectralimages, IEEEGeoscienceandRemoteSensingLet-
        ters 6 (2009) 662–665.
[113]J. Zhou, D. L. Civco, J. A. Silander,  A wavelet transform method to
        merge landsat tm and spot panchromatic data,  International Journal of
        Remote Sensing 19 (1998) 743–757.
[114]Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli,  Image quality
        assessment: from error visibility to structural similarity, IEEE Transac-
        tions on Image Processing 13 (2004) 600–612.
[115]M. Selva, B. Aiazzi, F. Butera, L. Chiarantini, S. Baronti,    Hyper-
        sharpening: A first approach on sim-ga data, IEEE Journal of Selected
        Topics in Applied Earth Observations and Remote Sensing 8 (2015)
        3008–3024.
[116]P. Dhariwal, A. Nichol,  Di       ffusion models beat gans on image synthe-
        sis, in: AdvancesinNeuralInformationProcessingSystems(NeurIPS),
        volume 34, 2021, pp. 8780–8794.
[117]L. Zhang, M. Agrawala,  Adding conditional control to text-to-image
        diffusion models, arXiv preprint arXiv:2302.05543 (2023).
[118]A. C. Li, M. Prabhudesai, S. Duggal, E. Brown, D. Pathak,    Your
        diffusion  model  is  secretly  a  zero-shot  classifier,     arXiv  preprint
        arXiv:2303.16203 (2023).

                                                                                             21

<!-- END-OF-PAPER:172dba74541a -->
