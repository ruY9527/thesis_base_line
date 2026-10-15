

<!-- PDF PAGE 1 -->

GEWDiff:GeometricEnhancedWavelet-basedDiffusionModelforHyperspectral
                                                      Image Super-resolution

                       Sirui Wang1, Jiang He1,2, Nat`alia Blasco Andreo3, Xiao Xiang Zhu 1,2*
                               1Technical University of Munich, Arcisstraße 21, 80333 Munich, Germany
                                     2Munich Center for Machine Learning, 80333 Munich, Germany
                         3Universitat Aut`onoma de Barcelona, 08193 Cerdanyola del Vall`es, Barcelona, Spain
                       sirui.wang@tum.de, Natalia.Blasco@uab.cat, jiang.he@tum.de,xiaoxiang.zhu@tum.de


                              Abstract                                           beendeveloped.However,thefusionmodelcannotgenerate
   Improving the quality of hyperspectral images (HSIs), such                    HSIs in any region of interest without any prior knowledge
   as through super-resolution, is a crucial research area. How-                 offered by VHR RGB data. Most fusion models can obtain
   ever, generative modeling for HSIs presents several chal-                     10 m resolution hyperspectral data with MSIs, such as Sen-
   lenges. Due to their high spectral dimensionality, HSIs are                   tinel 2, but to get a spatial resolution from 10 to 2.5 m, we
   too memory-intensive for direct input into conventional dif-                  focusedonthesingle-image-generationmethodsforHSIsat
   fusion models. Furthermore, general generative models lack                    4-times super-resolution.
   an understanding of the topological and geometric struc-                         Traditional hyperspectral image (HSI) super-resolution
   tures of ground objects in remote sensing imagery. In ad-                     approaches typically rely on interpolation techniques, such
   dition,  most  diffusion  models  optimize  loss  functions  at               asnearestneighbororbilinearinterpolation.Whilestraight-
   the noise level, leading to a non-intuitive convergence be-                   forward and computationally efficient, these methods fail to
   havior and suboptimal generation quality for complex data.                    capturethecomplex,nonlinearrelationshipspresentinhigh-
   To address these challenges, we propose a Geometric En-                       dimensional  spectral  data.  Recently,  deep  learning-based
   hanced Wavelet-based Diffusion Model (GEWDiff), a novel                       methods,suchasconvolutionalneuralnetworks(CNNs)and
   framework  for  reconstructing  hyperspectral  images  at  4-
   times super-resolution. A wavelet-based encoder-decoder is                    generative adversarial networks (GANs) (Sidorov and Yn-
   introduced  that  efficiently  compresses  HSIs  into  a  latent              gve Hardeberg 2019; Li, Wang, and Li 2020; Hou et al.
   spacewhilepreservingspectral-spatialinformation.Toavoid                       2022; Shi et al. 2022; Li et al. 2020), have emerged as pow-
   distortion  during  generation,  we  incorporate  a  geometry-                erful alternatives. These models are capable of learning ex-
   enhanced diffusion process that preserves the geometric fea-                  pressive spectral-spatial representations directly from data,
   tures. Furthermore, a multi-level loss function was designed                  leading to significant improvements in reconstruction accu-
   to guide the diffusion process, promoting stable convergence                  racy over classical methods. Recent studies have shown the
   and  improved  reconstruction  fidelity.  Our  model  demon-                  strong potential of transformers (Zhang et al. 2023; Chen,
   stratedstate-of-the-artresultsacrossmultipledimensions,in-                    Zhang, and Zhang 2023; Yu et al. 2023; Su et al. 2025) in
   cluding fidelity, spectral accuracy, visual realism, and clarity.
                                                                                 vision tasks, due to their ability to model long-range spatial
Code— https://github.com/zhu-xlab/GEWDiff                                        and spectral dependencies. However, a common limitation
                                                                                 shared by the aforementioned methods is their difficulty in
                          Introduction                                           generating rich textures and complex spatial structures, de-
                                                                                 spitetheirstrongperformanceinpreservingspectralfidelity.
Hyperspectral images (HSIs) offer a unique perspective by                           Diffusion models have recently demonstrated remarkable
capturing continuous spectral features of ground objects.                        success in generating high-quality natural images, as seen
Despite advancements in research, the high costs and low                         in models such as the Stable Diffusion (Rombach et al.
coverage of super-resolution (SR) hyperspectral data limit                       2021) and DiT (Peebles and Xie 2023) frameworks. Mo-
their applications. Currently, open-access hyperspectral air-                    tivated by their strong generative capabilities and robust-
borne data are predominantly regional, typically focused on                      ness in modeling complex distributions, researchers have
several cities, increasing their exclusivity and cost. Hyper-                    begun exploring their applicability to hyperspectral image
spectral satellites have better coverage but suffer from in-                     super-resolution.  For  instance,  SpectralDiff  (Chen  et  al.
sufficientspatialresolution.Improvingthespatialresolution                        2023) and HSR-Diff (Wu et al. 2023) extend the diffusion
of hyperspectral satellite images is, therefore, crucial to al-                  paradigm directly to the hyperspectral domain, aiming to
low for fully harnessing the potential of hyperspectral data                     better model spectral-spatial correlations. However, despite
in Earth observation (EO). Many fusion models that com-                          recent progress, adapting diffusion models to hyperspectral
bine hyperspectral and multispectral images (MSIs) have                          image generation remains a significant challenge. Unlike

   *Corresponding author.                                                        natural or multispectral images, HSI data typically exhibit a
Copyright © 2026, Association for the Advancement of Artificial                  much lowersignal-to-noise ratio and higherspectral dimen-
Intelligence (www.aaai.org). All rights reserved.                                sionality,makingitdifficulttodesigndiffusionarchitectures

<!-- PDF PAGE 2 -->

                    Figure 1: Illustration of the Geometric Enhanced Wavelet-based Diffusion Model pipeline.


thatbalancespatialfidelity,spectralaccuracy,andvisualre-                     for large-scale diffusion models. Building on this, Zhao et
alism.Existingarchitecturesoftensufferfromaslowmodel                         al. (Zhao et al. 2024) proposed a parallel diffusion strat-
convergencespeed,excessivesamplingsteps,andaneedfor                          egy that separates high-frequency and low-frequency com-
high computational GPU memory, which limits their practi-                    ponents for underwater image restoration. Shi et al. (Shi
cality in real-world scenarios. We propose a Geometric En-                   etal.2024)proposedWaveDiffUR,awavelet-domaindiffu-
hanced Wavelet-based Diffusion Model (GEWDiff) to ad-                        sion model for remote sensing ultra-resolution (UR), which
dressthesechallenges,asshowninFigure1.Themaincon-                            involved reformulating high-magnification SR as a condi-
tributions of this study are summarized as follows:                          tional stochastic differential equation (SDE) solved via it-
 •  An efficient wavelet-based encoder-decoder will al-                      erative wavelet decomposition, integrating pre-trained SR
   mostlosslesslytransformhyperspectraldataintoalatent                       modules for scalability and a cross-scale pyramid (CSP)
   space by decomposing the input into multiple frequency                    constraint to preserve spectral-spatial fidelity. Si et al. (Si
   levels. This method preserves spectral-spatial informa-                   et al. 2025) proposed CASSIDiff, the first diffusion model
   tionwhilereducingchanneldimensionalitywithoutlong-                        for CASSI hyperspectral reconstruction, which integrated a
   term training.                                                            DWT-basedfeaturefusionmechanismtoreducenoiseanda
 •  Structural invariance control via the diffusion pro-                     spectral-spatialattentionmoduletocapturespectralcorrela-
   cess:Anedge-awarenoiseschedulerwasdesignedtoim-                           tions.Despitethesuccessofvariouswavelet-baseddiffusion
   provegenerationefficiencyandaccuracyduringtraining.                       models in natural and remote sensing images, their applica-
   Maskconditioningensurespreservationofthegeometric                         tion to hyperspectral image generation remains unexplored.
   integrity and prevents distortion.                                        Diffusion model for hyperspectral image
 •  Amulti-levellossfunctioncomprisingpixel-wiseloss,                        super-resolution
   perceptualloss,andgradientloss:Eachpartoftheloss                          Recent work has proposed adapting the diffusion process
   function contributes to a balanced convergence speed.                     to better fit the characteristics of HSI data. As such, ex-
   The loss function also enables the evaluation and align-                  isting  approaches  can  be  broadly  categorized  into  three
   ment of predicted and ground-truth images on specific                     paradigms:two-stagemodels,groupedautoencodermodels,
   details and semantic understanding.                                       and end-to-end frameworks (Wang et al. 2023). Two-stage
                        Related Work                                         models decompose the super-resolution task into two sep-
Properties of wavelet-based diffusion models                                 arate subtasks handled by distinct networks. For example,
                                                                             HSI-Gene(Pangetal.2024b)firstgenerateshigh-resolution
Wavelet-based diffusion models have recently gained in-                      RGB bands from the input HSI, and then fuses them with
creasing attention due to their suitability for image gener-                 the low-resolution HSI to reconstruct the final output. This
ation. For example, WaveDiff (Phung, Dao, and Tran 2023)                     modular design helps reduce computational complexity and
demonstrated that applying diffusion in the wavelet domain                   allows for flexible training. Grouped models partition the
allows images to be compressed into a structured latent                      spectral bands into groups and process them in parallel, of-
space, enabling almost-lossless reconstruction while signif-                 tenusingautoencoder-stylearchitectures.DMGASR(Wang
icantly reducing computational overheads. This approach                      et al. 2024), for instance, employs spectral grouping and
not only preserves high-fidelity details but also offers sub-                trains separate VAE-based diffusion modules for different
stantial memory savings, making it especially advantageous                   groups, enabling scalable training across high-dimensional

<!-- PDF PAGE 3 -->

spectrawhilepreservinginter-bandcorrelations.End-to-end
frameworks perform full-spectrum reconstruction in a sin-
gle model, often incorporating various strategies to manage
complexity and improve the expressiveness. For example,
HIR-Diff(Pangetal.2024a)integratesacodebookwithsin-
gular value decomposition to compress and guide the gen-
eration process. MTLSC-Diff (Qu et al. 2024) uses clas-
sification maps as spatial priors to improve the generation
accuracy. LSDiff (Cheng et al. 2024) applies the diffusion
process in a compressed latent space, which allows reduc-
ingmemoryusagewhilemaintainingthegenerationquality.
Although some methods have been explored, most rely on
two-stage training or fail to simultaneously ensure spectral                    Figure 2: Illustration of the wavelet-based encoder-decoder.
fidelity and visual quality, while our model addresses both
challenges effectively.
                                                                                coefficients into a dense and orthogonal matrix, facilitating
                              Method                                            more coherent spectral analysis:
Wavelet-based encoder and decoder                                                                    (zLR,RLR)= PCA(       VJLR),                      (5)
Our training-free encoder-decoder is based on Regression                        where zLR is the feature that will be input to the diffusion
wavelet analysis (RWA), first proposed for lossless hyper-                      latent space, and RLR represents the remaining components
spectral image compression by (Amrani et al. 2016). RWA                         that will be kept and reused in the decoder equation (6).
applies a predefined number of Haar wavelet (Haar 1910)                         Decoder.   Forthereconstruction,theinversePCArecovers
decompositions intercalated with a linear regression of the                     predictedfeaturesˆz0 fromthediffusionprocesstothesuper-
spectral dimension to exploit the redundancy that still re-                     resolution HSI main coefficients ˆVJ
mains in the discrete wavelet transform (DWT) domain to                                                                   SR.
further compress the data. RWA can compress HSIs with a                                              (ˆVJ       (         ˆz0,RLR).                      (6)
moreefficient,lossless,ornear-losslesstransformbystoring                                                SR)= I-PCA
the prediction error. The structure is shown in Figure 2.                       The final super-resolved image is obtained by an inverse
Encoder.   For the super-resolution task, RWA allows us to                      RWA, having set the residuals WjLR to zero, since these are
reduce the number of bands given to the diffusion model                         not available for the super-resolution HSI image:
by using the Haar wavelet. Let ILR be the input 10 m high-                                     (ˆISR)= I-RWA(  ˆVJSR,BLR,WjLR,J).              (7)
resolution hyperspectral image, the J-th level RWA trans-                       The details ˆwj
form can be represented as:                                                     gression model BLR to recover the information lost in theSR will be predicted by the adjusted linear re-
             (VJLR,(wjLR)1⩽ j⩽J)= RWA(  ILR,J),             (1)                 wavelet transform. Once the diffusion outputs the super-
              ˆwj                                                               resolution components ˆVJSR, inverse-RWA reconstructs the
                i =βji,0  +βji,1  Vj1+...+βji,kVjk,              (2)            predicted main coefficients to the spectral dimension.
                         min||      wji − ˆwji||2,                             (3)Geometric enhanced diffusion process
where wji represents the i-th details (high-coefficient) of the                 Hyperspectral image generation usually requires larger re-
j-th level wavelet transform and ˆwji its prediction; Vjk rep-                  verse sampling time steps. To solve this problem, we used
resentsthek-thlow-coefficient(main-coefficient)ofthej-th                        EDM(ElucidatingDiffusionModels)(Karrasetal.2022)as
level wavelet transform and βji,k the linear regression coef-                   ourbaselinemodel.EDMaddsnoiseinonestepinthetrain-
ficients that will be learned to adjust the linear regression.                  ing process. A probability flow ordinary differential equa-
Contrary to traditional RWA, where the residuals                                tion (ODE) then continuously increases the noise level of
                                                                                theimagewhenmovingforwardintime(Karrasetal.2022).
                       WjLR = wjLR− ˆwjLR,                          (4)         Instead of using a discrete time step to add noise, we used a
are computed in order to fully recover the original signal,                     concrete numberσ to represent the strength of the noise:
the proposed encoder will only store VJLR and the weights                                   σ∼ exp(    N(  Pmean =−1. 2,P  std =1. 2)),           (8)
of all the adjusted linear models BLR = [β1⩽ j⩽J], whereJ                       where Pmean represents the mean value, Pstd is the standard
is the level of wavelet transforms applied. The main coeffi-                    deviation, and N is a Gaussian distribution.
cients VJLR, which contain the most critical information, are                      The “time variable”  t used in our model is a continuous
used as input for the principal component analysis (PCA).                       variable,whichhastheadvantageofmappingthenoisescale
The following PCA transformation enables a more efficient                       to an approximately linear interval. In this way, the noise
compression of hyperspectral imagery (HSI) by achieving                         strength that will be added at the t moment can correspond
a higher compaction factor while preserving more informa-                       to the size of t, and the relationship can be represented as:
tion. Furthermore, it can convert the sparse wavelet-based                                                  t=−log(    σt).                               (9)

<!-- PDF PAGE 4 -->

                                                                              tn+1  −tn,n = 0,  1,...,N  −1. The initial noisy image and
                                                                              noise strength at step n can then be calculated with:

                                                                                                          zT =σT ·ϵ,                               (13)
                                                                                                                                      ρ
                                                                                        σn =     σ1/ρmax +    nN−1   σ1/ρmin −σ1/ρmax     ,      (14)
      Figure 3: Edge perturbed noisy image over time.
                                                                              whereρistheschedulingcurvatureparameter,andσmax/minis
                                                                              the maximum/minimum noise strength. The n+1 step de-
Edge-aware noise scheduler.   General diffusion models                        noised featureszn+1   can be calculated with:
haveanequalgenerationabilityforeachpixel.Intheremote
sensing scenario, we wanted to clarify the contour of build-                                        γ =−12· tn+1  −tntn−tn−  1,                       (15)
ings and other ground objects. Inspired by (Vandersanden
et al. 2024), we designed an edge-aware noise scheduler in                       ˜fθ =(1−γ)fθ(ˆzn,C,σn)+γfθ(ˆzn−  1,C,σ   n−  1),   (16)
our training stage to increase the generation ability of the
diffusion model for the pixels around the edges. The edge                                zn+1  = σn+1σn  ˆzn−σn+1 (e−∆t  −1)· ˜fθ.       (17)
ispreservedduringtheforwarddiffusionprocess.Thenoise
around the edge is smaller than the general noise:                            Multi-level loss function
            zt = z0+σtϵ⊙(1−E(1  −σ2norm)η),          (10)                     Multi-level loss equations, such as equation 18, can ensure
where zt represents the noisy features at moment t, t ∈                       that the generated image is accurate in all aspects.
(0,T),z0 isthefeaturesfromthegroundtruth,σt represents                                  L=λ(  t) ·(λ1Lpixel+λ2Lperc+λ3Lgrad),      (18)
thenoisestrengthatmomentt,ϵ∼ N(0  ,1)israndomnoise,                           To balance the convergence speed, we set λ1 = 0.8,λ  2 =
E is the binary edge map obtained from the input image,                       0. 1,λ  3 = 0. 1.  λ(  t)  indicates the loss weighting based on
σnorm isthenormalizedsigmaatmomentt,⊙highlightsthat                           t. Considering pixel loss can ensure that the absolute value
theoperationisamatrixmultiplication,andη =0.5adjusts                          of the spectral information of each pixel is accurate. Here
the perturbation strength influenced by the edge.                             weusethecombinationofL2normlossandSpectralAngle
Maskcontrollabletrainingandsampling.   Inourprelim-                           Mapper (SAM) (Yuhas, Goetz, and Boardman 1992) loss:
inary research, generating geometric objects without distor-
tion was a big challenge. To address this, a mask is intro-                              Lpixel =(∥z  0−ˆz0∥2+SAM(        z0,ˆz0))/2.            (19)
duced as a condition that improves the ability to generate                    Perceptron loss (Johnson, Alahi, and Fei-Fei 2016) can en-
buildings. Segmentation is calculated from low-resolution                     surethatthegeneratedimageissimilarinhigh-levelfeature
RGB channels from hyperspectral images with a segment-                        space:
anythingmodel(Kirillovetal.2023).Weusedoneminusthe
average of NDVInorm,NDVI norm ∈ [0,1] (Kriegler et al.                                     Lperc =∥ϕ  VGG(    ˆz0)−ϕVGG(     z0)∥2
1969) as the value of the mask to highlight the attention of                                                                         2.          (20)
buildings. LetSs be the pixel number of thes th segmenta-                     Gradient loss (Lu and Chen 2022) can ensure that the im-
tion region, then the valueMs of the mask is defined as:                      age gradient information is consistent with the real image.
                                X                                             TheimagesgeneratedbyDPMSolver++havehigh-contrast
          Ms =1−   1|Ss|               NDVInorm(x,y).       (11)              characteristics. Gradient loss is defined as follows:

                             (x,y)∈Ss
  In the training stage, the predicted resultˆz0 is calculated                  Lgrad= 12(∥▽xˆz0−▽xz0∥1+∥▽yˆz0−▽yz0∥1),  (21)
in one step:                                                                  where▽x/y represents the gradient of the image in the x/y
              ˆz0 =fθ(zt,C,σ   t),C=[zLR, M]   ,            (12)              direction.
where ˆz0 represents the predicted features when t = 0; fθ                                              Experiments
represents the objective function 3D U-Net with spectral fi-
delity enhancer (SFE) (Dong et al. 2021), as shown in fig-                    Datasets and implementation details
ure 1; zLR represents the low-resolution condition; M ∈                       A  group  of  EeteS  simulated  EnMap  hyperspectral  data,
(0,1)H×W is the mask condition; and C is the concatena-                       called the EnMap Campaign (realistic EnMAP-like high-
tion of all conditions.                                                       resolution data), was used for training. EnMap is a hyper-
  During  the  sampling  stage,  DPM-Solver++  (Lu  et  al.                   spectral satellite managed by the DLR Earth Observation
2022) accelerates the generation by employing a second-                       Center,offering30mspatialresolutionsinceJune2022.Ad-
order approximation to solve the underlying ODE, while                        ditionally, the EnMAP Campaign Portal (access via supple-
utilizing adaptive time stepping to significantly reduce the                  mentarymaterial)capturesaerialhyperspectralimageryand
number of function evaluations. In our sampler, t ∈ [0,T]                     simulates EnMap-like data, boasting a spatial resolution of
will be separated into N  steps. The step size is ∆t     =                    2.5-4mandcontainingdatafrom2009to2016.Wegathered

<!-- PDF PAGE 5 -->

           Metric           MCNet          MSDFormer     ESSAFormer       DMGASR           HIRDiff             SNLSR                Ours
           PSNR↑               28.300±0.0480  28.284±0.0000  27.483±0.1392  26.986±0.2052  24.833±0.1079  28.531±0.0001                                                                                       28.863±0.2940
           SSIM↑               0.6658±0.0025  0.6592±0.0000  0.5915±0.0191  0.5831±0.0118  0.6401±0.0024  0.6718±0.0000                                                                                       0.7104±0.0212
           SAM↓              8.3332±0.1243  8.7442±0.0000  9.2114±0.2482  11.340±0.0743  8.9538±0.0053                                                                         7.8911±0.0000                8.4283±0.3073
  (a)        CC↑             0.7440±0.0000  0.7645±0.0000  0.7374±0.0004  0.6767±0.0128  0.7543±0.0011  0.7527±0.0000                                                                                       0.7945±0.0165
          RMSE↓               0.0557±0.0002  0.0544±0.0000  0.0560±0.0002  0.0627±0.0006  0.0810±0.0015  0.0552±0.0000                                                                                       0.0548±0.0030
            FID↓             116.14±0.0856  103.74±0.0000  97.438±14.779  49.026±2.3984  50.596±1.0436  125.75±0.0691                                                                                       44.464±17.627
             LV↑          0.0004±0.0000  0.0004±0.0000  0.0004±0.0000  0.0037±0.0004  0.0021±0.0002  0.0003±0.0000                                                                                       0.0041±0.0022
           PSNR↑               24.216±0.0157  24.359±0.0000  24.103±0.0132  23.021±0.0146  21.567±0.5351  24.305±0.0000                                                                                       24.933±0.0079
           SSIM↑               0.5355±0.0008  0.5536±0.0000  0.5210±0.0010  0.4925±0.0025  0.4987±0.0133  0.5404±0.0000                                                                                       0.6337±0.0106
           SAM↓              11.663±0.0482  11.912±0.0000  12.166±0.0156  16.158±0.0117  12.348±0.1154  11.418±0.0001                                                                                       11.323±0.0456
  (b)        CC↑             0.7050±0.0004  0.7238±0.0000  0.7077±0.0004  0.6575±0.0007  0.7347±0.0003  0.7102±0.0000                                                                                       0.7771±0.0003
          RMSE↓               0.0685±0.0001  0.0669±0.0000  0.0682±0.0002  0.0779±0.0011  0.0940±0.0061  0.0680±0.0000                                                                                       0.0668±0.0020
            FID↓             257.45±0.1949  272.78±0.0000  288.85±7.1990  120.06±11.769  375.96±23.877  267.88±0.0032                                                                                       64.333±4.2810
             LV↑          0.0007±0.0000  0.0006±0.0000  0.0007±0.0000  0.0034±0.0005  0.0005±0.0000  0.0005±0.0000                                                                                       0.0087±0.0003
           PSNR↑               33.389±0.2641  28.709±0.0000  25.504±0.1340  32.864±0.2049  34.473±0.0069  35.734±0.0000                                                                                       35.837±0.1176
           SSIM↑               0.7441±0.0029  0.4766±0.0000  0.4120±0.0312  0.6802±0.0159  0.7362±0.0017  0.7525±0.0000                                                                                       0.7747±0.0045
           SAM↓              8.5500±0.0896  12.213±0.0000  18.724±0.5899  11.476±0.3843  8.3601±0.0446  7.6613±0.0000                                                                                       7.4735±0.0532
  (c)        CC↑             0.6495±0.0145  0.6300±0.0000  0.6326±0.0090  0.5001±0.0161  0.7102±0.0001  0.7733±0.0000                                                                                       0.7906±0.0055
          RMSE↓               0.0476±0.0006  0.0525±0.0000  0.0690±0.0006  0.0542±0.0013                                                           0.0420±0.0001                0.0471±0.0000  0.0468±0.0006
            FID↓             464.13±5.9021  738.62±0.0000  701.35±16.290  245.38±63.176  363.23±6.9705  470.34±0.0000                                                                                       238.12±16.970
             LV↑          0.0003±0.0000  0.0003±0.0000  0.0010±0.0000  0.0031±0.0015                                        0.0002±0.0000  0.0002±0.0000                              0.0011±0.0000
         Tr time (s)      1.33×104         3.99×104         7.65×104         3.16 ×105                 –                2.80×104         3.10×105
  (d)    Te time (s)          18.13                  10.40                   7.98                  334.00                212.90                  4.10                   28.70
             NFE               2562                   2562                   2562                 20×8                    20                     2562                     50
        Model size        6.50 MB             57.7 MB             3.70 MB             1.18 GB              1.56 GB             7.70 MB             4.55 GB

Table1:QuantitativecomparisonwithSOTASRmodelsofPSNR,SSIM,SAM,CC,RMSE,FID,andLVon(a)MDASsample
1,(b)MDASsample2,and(c)WDCdataset.(d)Modelefficiencywasevaluatedwiththetraining/testingtime,numberoffunc-
tion evaluations (NFE), and model size. (Best performance value is highlighted in bold. Noise-affected values are underlined.)


































Figure4:4-timesvisualcomparisonswithSOTASRmodelson(a)MDASsample1,(b)MDASsample2,and(c)WDCdataset.

<!-- PDF PAGE 6 -->

Figure 5: Reflectance of a random pixel and mean band-wise differences for (a–b) MDAS 1, (c–d) MDAS 2, and (e–f) WDC.











Figure 6: Qualitative comparison with different numbers ofρ and time steps for (a) PSNR, (b) SAM, and (c) FID. Qualitative
comparison with different numbers ofσmax andσmin for (d) PSNR, (e) SAM, and (f) FID on validation dataset.


                                                                            tures structural consistency. Both were averaged over spec-
                                                                            tral bands. Realism and clarity were addressed using the
                                                                            Fr´echet Inception Distance (FID) (Heusel et al. 2017), not-
                                                                            ing that it is computed in an RGB-trained feature space and
                                                                            thus was only used for relative comparisons. We also in-
                                                                            cludedLocalVariation(LV)(Pertuz,Puig,andGarcia2013)
                                                                            to evaluate the local texture sharpness. Spectral accuracy
                                                                            was measured via Spectral Angle Mapper (SAM) (Yuhas,
                                                                            Goetz, and Boardman 1992), Cross-Correlation (CC), and
Figure7:Encoder–decoderreconstructionqualitycompared                        Root Mean Square Error (RMSE). These metrics assess the
with the performance of the whole SR process for the vali-                  angular, correlational, and pixel-wise spectral alignment.
dation dataset.                                                             State-of-the-art image generation
                                                                            We evaluated the performance and efficiency of our model
                                                                            by comparing it with the six SOTA models: MCNet (Li,
8000 pairs of 256 × 256 × 242 patches of super-resolution                   Wang,andLi2020),MSDFormer(Chen,Zhang,andZhang
HSI images and aligned them with 4-times downsampled                        2023), ESSAFormer (Zhang et al. 2023), DMGASR (Wang
imagesfromEnMapCampaignandMDAS(Huetal.2023).                                et al. 2024), HIR Diff (Pang et al. 2024a), and SNLSR (Hu
This dataset covers 15 cities in Europe and the Americas,                   etal.2024).Wetrainedthesemodelsonourdatasetwiththe
representingdiversegroundobjects.Among8,000pairs,one                        implementation details provided in each paper, except for
pairwassplitintothevalidationdatasetandusedforparam-                        theunsupervisedmethodHIRDiff.TheHIRDiffmodelpro-
eter selection, and a group of ablation study experiments.                  vides a pre-trained checkpoint based on their fully prepared
Two pairs are selected for EnMap simulation testing. The                    HSIdataset.Ourmodeldemonstratedstrongperformancein
WDC (Biehl et al. 2015) dataset is used as one of the test                  generating medium to large-scale ground objects. In Figure
datasets to see the transmission on other datasets. We de-                  4 and Table 1, we can see that both the visualization result
ployedthemodelonfourNVIDIAA100GPUswithalearn-                               and quantitative result outperformed the other models. The
ing rate of1×10−4 and trained it for 200 epochs.                            WDC dataset result shows our model can adapt to another
                                                                            datasetwithadifferentspectralprofile.However,ourmodel
Quantitative metrics                                                        may struggle with accuracy when the input conditional im-
Fidelitywasevaluatedusingtwostandardmetrics:thePeak                         age lacks sufficient semantic information; for example, the
Signal-to-Noise Ratio (PSNR) (Huynh-Thu and Ghanbari                        rooftop of the MDAS sample 1 reconstructed image.
2012),whichmeasuresthepixel-wisequality,andtheStruc-                        Effect  of  the  encoder-decoder  alone  vs.  with  super-
turalSimilarityIndex(SSIM)(Wangetal.2004),whichcap-                         resolution.   We compared our encoder-decoder (RWA +

<!-- PDF PAGE 7 -->

              Method    Baseline       A           B          C          D          E          F          G          H           I            J         Ours
              w/RWA                     ✓                  ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓
              w/PCA                               ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓
              w/Mask               ✓      ✓                ✓                 ✓      ✓      ✓      ✓      ✓      ✓      ✓
              w/Edge               ✓      ✓                         ✓      ✓      ✓      ✓      ✓      ✓                 Inverse   ✓
              w/L pix      ✓       ✓      ✓      ✓      ✓                    ✓                          ✓      ✓      ✓      ✓
              w/L perc              ✓      ✓      ✓      ✓                            ✓                 ✓      ✓      ✓      ✓
              w/L geo               ✓      ✓      ✓      ✓                                     ✓      ✓      ✓      ✓      ✓
             w/Unet3D              ✓      ✓      ✓      ✓      ✓      ✓      ✓                          ✓                  ✓      ✓
               w/SFE                ✓      ✓      ✓      ✓      ✓      ✓      ✓                                  ✓      ✓      ✓
              PSNR↑                2.0476    15.788   25.640  26.579  26.681  26.101  21.664  26.467  26.388  26.181  26.8713  27.013
               SSIM↑               -0.0150   -0.0173  0.5267  0.6503  0.6530  0.6140  0.1580  0.6443  0.6240  0.6131   0.6667   0.6573
              SAM↓      124.15    85.239   15.125  11.766  12.160  12.804  25.528  12.445  12.788  14.085  11.7688  11.501
              RMSE↓     1.1879    0.8687   0.0850  0.0758  0.0749  0.0804  0.1342  0.0769  0.0783  0.0803   0.0739   0.0726CC↑              0.2643    0.4763   0.5530  0.6803  0.6882  0.6441  0.0145  0.6741  0.6681  0.6500   0.6999   0.7008
               FID↓       5019.1    484.16   83.627  43.445  36.269  40.303  701.94  40.195  47.475  65.985  34.9402  30.110
                LV↑           7.5344    0.6231   0.0160  0.0079  0.0077  0.0096  0.1684  0.0093  0.0074  0.0089   0.0080   0.0083

Table 2: Ablation study. Quantitative comparison for the effect of each module on the validation dataset. Results are averaged
over4runs.ThebaselineusedanEDMbackboneandDPM-Solver++sampleron242bands.A:noPCAintheencoder;B:no
RWAintheencoder;C:noedgeperturbation;D:nomaskconditioning;E/F/G:retainedpixel/perceptual/geometricloss;H:no
spectral fidelity module in the encoder; I: used 2D U-Net instead of 3D; J: more noise on edge (Zhang et al. 2025).


                                                                               Real-world  application.   We  tried  to  combine  EnMAP
                                                                               and Sentinel-2 imagery to 10 m resolution hyperspectral
                                                                               data via the unsupervised method HySure (Sim˜oes et al.
                                                                               2015). Leveraging our 4-times super-resolution generation
                                                                               model, GEWDiff, we could finally produce EnMAP hy-
                                                                               perspectral  images with a  2.5-meter resolution. The  no-
                                                                               reference image quality assessment MetaIQA (Zhu et al.
                                                                               2020) was improved from 0.1997 to 0.2029 (Figure 8 (b)).

Figure 8: (a)Visualization of EnMap, Sentinel 2, Fusion im-                    Ablation study
age, Super resolution image. (b) Effect of super-resolution.                   The results of our ablation studies, presented in Table 2, of-
                                                                               fersignificantinsightsintothecontributionsofvariouscom-
                                                                               ponents of the GEWDiff model. The use of a suitable en-
                                                                               coder plays a crucial role in our model design. The multi-
PCA)  against  RWA-only  and  PCA-only  baselines  (Fig-                       level loss function achieves better performance than an L2
ure 7), under 3- and 20-band compression. Without super-                       loss. The design of a 3D objective function, U-Net, and its
resolution,ourencoder-decoderachievedalmostlosslessre-                         spectral fidelity enhancer makes progress toward stabilizing
construction with a PSNR up to 56. With super-resolution,                      the results. The geometric enhancement strategies, such as
using RWA + PCA compression to 20 bands outperformed                           edgeperturbationandmaskconditioning,didnotshowasig-
all the baselines, demonstrating strong spectral compaction                    nificant improvement in the global metrics. However, some
and generation. As shown, retaining more bands improves                        effects could be observed from Figure 4, whereby the edges
the quality but increases the cost. We chose 20 bands to en-                   are clear, and there was no obvious building distortion.
sure a balance between performance and efficiency.
                                                                                                           Conclusion
Effect of ρ, number of timesteps   N, σmax and σmin on                         We proposed GEWDiff, which improves the spatial resolu-
sampling.   TheEDMnoisescheduleisshapedbyρ,which                               tionofhyperspectralimagesbyafactorof4.Ourmethodin-
controls how noise levels decay from σmax to σmin via N                        tegrates wavelet-domain transforms and geometric priors to
steps. Higher ρ sharpens early denoising but increases the                     effectivelypreservebothspectralfidelityandspatialtextures
randomness in later steps, potentially degrading spectral                      while accelerating convergence. The experimental results
consistency. We found that lowerρ values [0.6–0.7] yielded                     showedthatGEWDiffoutperformedthecurrentSOTAbase-
smoothernoiseschedulesvia50steps,whichalsobetterpre-                           lines.OnelimitationofGEWDiffisthattheresultreliestoo
servespectralfidelity(Figure6).Wealsostudiedtheeffects                         much on the input conditions. This limitation could be ad-
of σmax and σmin. A larger σmax allows more diversity but                      dressed in future work through the integration of classifier-
risks over-noising; a smaller one limits the detail. We set                    free guidance, which would enable the model to better gen-
σmax = 80 for a balance. For σmin, smaller values extend                       eralize under weak or ambiguous conditioning. Moreover,
the denoising phase, improving the detail but increasing ar-                   we would also contribute to model distillation for further
tifacts. We found the best results whenσmin∈[0.02,    0. 2].                   lightweight alternatives.

<!-- PDF PAGE 8 -->

                     Acknowledgments                                         IEEE Transactions on Neural Networks and Learning Sys-
Please      refer      to      the      extended      version      via       tems, 33(12): 7303–7317.
https://arxiv.org/html/2511.07103v1.                                         Foerster, S.; Brosinsky, A.; Wilczok, C.; and Bauer, M.
  This work was supported in part by Munich Center for                       2015. Is´abena 2011 - An EnMAP Preparatory Flight Cam-
Machine Learning and in part by German Federal Min-                          paign (Datasets).
istryforEconomicAffairsandClimateActionintheframe-                           Haar, A. 1910.  Zur Theorie der orthogonalen Funktionen-
workofthe“nationalcenterofexcellenceML4Earth”(grant                          systeme. Mathematische Annalen, 69: 331–371.
number: 50EE2201C). The authors sincerely thank GFZ                          Hank, T. B.; Locherer, M.; Richter, K.; and Mauser, W.
Helmholtz-Zentrum for providing the EnMAP Campaign                           2015.  Neusling (Landau a.d. Isar) 2012 - A Multitemporal
datasets (Buddenbaum and Hill 2020; Beamish et al. 2020;                     and Multisensoral Agricultural EnMAP Preparatory Flight
Brell et al. 2020; Milewski et al. 2020; Cooper et al. 2020;                 Campaign (Datasets).
Hank et al. 2015; Jarmer and Siegmann 2017; Neumann,                         Heusel,  M.;  Ramsauer,  H.;  Unterthiner,  T.;  Nessler,  B.;
Weiss,  and  Itzerott  2015;  Okujeni,  van  der  Linden,  and               and  Hochreiter,  S.  2017.      GANs  Trained  by  a  Two
Hostert2016;Foersteretal.2015;Boescheetal.2016)used                          Time-Scale Update Rule Converge to a Local Nash Equi-
in this paper, and Dr. Jingliang Hu for providing the MDAS                   librium. InAdvancesinNeuralInformationProcessingSys-
dataset.                                                                     tems (NeurIPS).
                          References                                         Hou, J.; Zhu, Z.; Hou, J.; Zeng, H.; Wu, J.; and Zhou, J.
Amrani,  N.;  Serra-Sagrist`a,  J.;  Laparra,  V.;  Marcellin,               2022. Deep posterior distribution-based embedding for hy-
M. W.; and Malo, J. 2016.   Regression Wavelet Analysis                      perspectral image super-resolution.  IEEE Transactions on
for Lossless Coding of Remote-Sensing Data. IEEE Trans-                      Image Processing, 31: 5720–5732.
actions on Geoscience and Remote Sensing, 54(9): 5616–                       Hu, J.; Liu, R.; Hong, D.; Camero, A.; Yao, J.; Schneider,
5627.                                                                        M.; Kurz, F.; Segl, K.; and Zhu, X. X. 2023. MDAS: a new
Beamish,A.; Chabrillat,S.; Brell,M.;Heim, B.;and Sachs,                      multimodal benchmark dataset for remote sensing.   Earth
T. 2020.  Toolik Lake Research Natural Area AISA-Eagle                       System Science Data, 15(1): 113–131.
hyperspectral Mosaic.                                                        Hu, Q.; Wang, X.; Jiang, J.; Zhang, X.-P.; and Ma, J. 2024.
Biehl, L.; Maud, A. R.; Hsu, W.-K.; and Yeh, T. T. 2015.                     ExploringtheSpectralPriorforHyperspectralImageSuper-
MultiSpec.                                                                   Resolution.   IEEE Transactions on Image Processing, 33:
                                                                             5260–5272.
Boesche,N.K.;Mielke,C.;Segl,K.;Chabrillat,S.;Rogass,                         Huynh-Thu, Q.; and Ghanbari, M. 2012.  The accuracy of
C.; Thomson, D.; Lundeen, S.; Brell, M.; and Guanter, L.                     PSNR in predicting video quality for different video scenes
2016.  EnGeoMAP Test Data: Simulated EnMAP Satellite                         and frame rates. Telecommunication Systems, 49(1): 13–27.
Data for Mountain Pass, USA and Rodalquilar, Spain.                          Jarmer,T.;andSiegmann,B.2017. K¨othen2011/2012-An
Brell, M.; Spengler, D.; Ruhtz, T.; Ward, K.; Chabrillat, S.;                EnMAP Preparatory Flight Campaign.
Segl, K.; Foerster, S.; and Itzerott, S. 2020.  Demmin, Ger-                 Johnson, J.; Alahi, A.; and Fei-Fei, L. 2016.   Perceptual
many (October 2015) - an EnMAP Preparatory Flight Cam-                       Losses for Real-Time Style Transfer and Super-Resolution.
paign.                                                                       arXiv:1603.08155.
Buddenbaum, H.; and Hill, J. 2020. Gerolstein, 2016-09-08                    Karras, T.; Aittala, M.; Aila, T.; and Laine, S. 2022. Eluci-
- An EnMAP Preparatory Flight Campaign.                                      dating the design space of diffusion-based generative mod-
Chen, N.; Yue, J.; Fang, L.; and Xia, S. 2023.  SpectralD-                   els. Advancesinneuralinformationprocessingsystems,35:
iff: A generative framework for hyperspectral image classi-                  26565–26577.
fication with diffusion models. IEEE Transactions on Geo-                    Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.;
science and Remote Sensing, 61: 1–16.                                        Gustafson,L.;Xiao,T.;Whitehead,S.;Berg,A.C.;Lo,W.-
Chen, S.; Zhang, L.; and Zhang, L. 2023.   MSDformer:                        Y.; Doll´ar, P.; and Girshick, R. 2023.  Segment Anything.
Multiscale deformable transformer for hyperspectral image                    arXiv:2304.02643.
super-resolution. IEEETransactionsonGeoscienceandRe-                         Kriegler, F. J.; Malila, W. A.; Nalepka, R. F.; and Richard-
mote Sensing, 61: 1–14.                                                      son, W. 1969.  Preprocessing transformations and their ef-
Cheng,Y.;Ma,Y.;Fan,F.;Ma,J.;Yao,Y.;andMei,X.2024.                            fects on multispectral recognition.   In Proceedings of the
Latentspectral-spatialdiffusionmodelforsinglehyperspec-                      Sixth International Symposium on Remote Sensing of Envi-
tral super-resolution.  Geo-spatial Information Science, 1–                  ronment, 97–131.
16.                                                                          Li, J.; Cui, R.; Li, B.; Song, R.; Li, Y.; Dai, Y.; and Du, Q.
Cooper, S.; Okujeni, A.; J¨anicke, C.; Segl, K.; van der Lin-                2020.  Hyperspectral Image Super-Resolution by Band At-
den, S.; and Hostert, P. 2020. 2013 Simulated EnMAP Mo-                      tention Through Adversarial Learning.  IEEE Transactions
saics for the San Francisco Bay Area, USA.                                   on Geoscience and Remote Sensing, 58(6): 4304–4318.
Dong,W.;Hou,S.;Xiao,S.;Qu,J.;Du,Q.;andLi,Y.2021.                             Li, Q.; Wang, Q.; and Li, X. 2020.  Mixed 2D/3D convo-
Generative dual-adversarial network with spectral fidelity                   lutional network for hyperspectral image super-resolution.
and spatial enhancement for hyperspectral pansharpening.                     Remote sensing, 12(10): 1660.

<!-- PDF PAGE 9 -->

Lu,C.;Zhou,Y.;Bao,F.;Chen,J.;Li,C.;andZhu,J.2022.                           Sim˜oes, M.; Bioucas-Dias, J.; Almeida, L. B.; and Chanus-
DPM-Solver++:Fastsolverforguidedsamplingofdiffusion                         sot,J.2015. AConvexFormulationforHyperspectralImage
probabilistic models. ArXiv preprint, arXiv:2211.01095.                     Superresolution via Subspace-Based Regularization.  IEEE
Lu, Z.; and Chen, Y. 2022.  Single image super-resolution                   Transactions on Geoscience and Remote Sensing, 53(6):
basedonamodifiedU-netwithmixedgradientloss. signal,                         3373–3388.
image and video processing, 16(5): 1143–1151.                               Su, X.; Shen, X.; Wan, M.; Nie, J.; Chen, L.; Liu, H.;
Milewski, R.; Chabrillat, S.; Brell, M.; Behling, R.; and                   and Zhou, X. 2025.   EigenSR: Eigenimage-Bridged Pre-
Eichstaedt, H. 2020. Omongwa Pan, Namibia (June 2015) -                     Trained  RGB  Learners  for  Single  Hyperspectral  Image
an EnMAP Preparatory Flight Campaign.                                       Super-Resolution.  In Proceedings of the AAAI Conference
Neumann, C.; Weiss, G.; and Itzerott, S. 2015.  D¨oberitzer                 on Artificial Intelligence, volume 39, 7033–7041.
Heide 2008/2009 - An EnMAP Preparatory Flight Cam-                          Vandersanden, J.; Holl, S.; Huang, X.; and Singh, G. 2024.
paign (Datasets).                                                           Edge-preservingnoisefordiffusionmodels. ArXivpreprint,
Okujeni,  A.;  van  der  Linden,  S.;  and  Hostert,  P.  2016.             arXiv:2410.01540.
Berlin-Urban-Gradient dataset 2009 - An EnMAP Prepara-                      Wang, X.; Hu, Q.; Cheng, Y.; and Ma, J. 2023. Hyperspec-
tory Flight Campaign (Datasets).                                            tral image super-resolution meets deep learning: A survey
Pang, L.; Rui, X.; Cui, L.; Wang, H.; Meng, D.; and Cao,                    and perspective.  IEEE/CAA Journal of Automatica Sinica,
X. 2024a.    HIR-Diff: Unsupervised hyperspectral image                     10(8): 1668–1691.
restoration via improved diffusion models.  In Proceedings                  Wang,Z.;Bovik,A.C.;Sheikh,H.R.;andSimoncelli,E.P.
of the IEEE/CVF Conference on Computer Vision and Pat-                      2004.   Image quality assessment: from error visibility to
tern Recognition, 3005–3014.                                                structural similarity. IEEE Transactions on Image Process-
Pang, L.; Tang, D.; Xu, S.; Meng, D.; and Cao, X. 2024b.                    ing, 13(4): 600–612.
HSIGene: A Foundation Model For Hyperspectral Image                         Wang, Z.; Li, D.; Zhang, M.; Luo, H.; and Gong, M. 2024.
Generation. arXiv:2409.12470.                                               Enhancing hyperspectral images via diffusion model and
Peebles, W.; and Xie, S. 2023.  Scalable Diffusion Models                   group-autoencoder super-resolution network.   In Proceed-
with Transformers. arXiv:2212.09748.                                        ings of the AAAI Conference on Artificial Intelligence, vol-
Pertuz,S.;Puig,D.;andGarcia,M.A.2013. Analysisoffo-                         ume 38, 5794–5804.
cusmeasureoperatorsforshape-from-focus. PatternRecog-                       Wu, C.; Wang, D.; Bai, Y.; Mao, H.; Li, Y.; and Shen,
nition, 46(5): 1415–1432.                                                   Q. 2023.  HSR-Diff: Hyperspectral image super-resolution
Phung, H.; Dao, Q.; and Tran, A. 2023.  Wavelet diffusion                   via conditional diffusion models.    In Proceedings of the
models are fast and scalable image generators. In Proceed-                  IEEE/CVF International Conference on Computer Vision,
ings of the IEEE/CVF conference on computer vision and                      7083–7093.
pattern recognition, 10199–10208.                                           Yu, D.; Li, Q.; Wang, X.; Zhang, Z.; Qian, Y.; and Xu, C.
                                                                            2023.  Dstrans: Dual-stream transformer for hyperspectral
Qu, J.; Xiao, L.; Dong, W.; and Li, Y. 2024. MTLSC-Diff:                    image restoration.  In Proceedings of the IEEE/CVF Win-
Multitask learning with diffusion models for hyperspectral                  ter Conference on Applications of Computer Vision, 3739–
imagesuper-resolutionandclassification.Knowledge-Based                      3749.
Systems, 303: 112415.                                                       Yuhas, R. H.; Goetz, A. F. H.; and Boardman, J. W. 1992.
Rombach,R.;Blattmann,A.;Lorenz,D.;Esser,P.;andOm-                           Discriminationamongsemi-aridlandscapeendmembersus-
mer,B.2021. High-ResolutionImageSynthesiswithLatent                         ing the Spectral Angle Mapper (SAM) algorithm.  In Sum-
Diffusion Models. arXiv:2112.10752.                                         mariesoftheThirdAnnualJPLAirborneGeoscienceWork-
Shi,Y.;Han,L.;Dancy,D.;andHan,L.2024. WaveDiffUR:                           shop, volume 1, 147–149. Pasadena, CA: JPL.
AdiffusionSDE-basedsolverforultramagnificationsuper-                        Zhang, L.; You, W.; Shi, K.; and Gu, S. 2025. Uncertainty-
resolution in remote sensing images. arXiv:2412.18996.                      guided Perturbation for Image Super-Resolution Diffusion
Shi, Y.; Han, L.; Han, L.; Chang, S.; Hu, T.; and Dancey,                   Model. arXiv:2503.18512.
D. 2022.  A Latent Encoder Coupled Generative Adversar-                     Zhang, M.; Zhang, C.; Zhang, Q.; Guo, J.; Gao, X.; and
ial Network (LE-GAN) for Efficient Hyperspectral Image                      Zhang, J. 2023.  Essaformer: Efficient transformer for hy-
Super-Resolution.   IEEE Transactions on Geoscience and                     perspectral image super-resolution.  In Proceedings of the
Remote Sensing, 60: 1–19.                                                   IEEE/CVF International Conference on Computer Vision,
Si, Y.; Lin, Z.; Wang, X.; and He, S. 2025.  A New Hyper-                   23073–23084.
spectralReconstructionMethodWithConditionalDiffusion                        Zhao, C.; Cai, W.; Dong, C.; and Hu, C. 2024.   Wavelet-
Model for Snapshot Spectral Compressive Imaging.  IEEE                      based Fourier Information Interaction with Frequency Dif-
Transactions on Instrumentation and Measurement, 74: 1–                     fusion Adjustment for Underwater Image Restoration.   In
14.                                                                         Proceedings of the IEEE/CVF Conference on Computer Vi-
Sidorov, O.; and Yngve Hardeberg, J. 2019.  Deep hyper-                     sion and Pattern Recognition (CVPR), 8281–8291.
spectral prior: Single-image denoising, inpainting, super-                  Zhu,  H.;  Li,  L.;  Wu,  J.;  Dong,  W.;  and  Shi,  G.  2020.
resolution.  In Proceedings of the IEEE/CVF International                   MetaIQA:  Deep  Meta-learning  for  No-Reference  Image
Conference on Computer Vision Workshops, 0–0.                               Quality Assessment. arXiv:2004.05508.

<!-- END-OF-PAPER:2e50a8e97214 -->
