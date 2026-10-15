

<!-- PDF PAGE 1 -->

   Structure-Aware Riemannian Flow Matching for Registration and Fusion of
                                  Hyperspectral and Multispectral Images




                 Quan Zhang1  Jun Li2  Weilong Zhu1  Mingyang Li2  Qinmu Shen2  Yuanxi Peng1

                           Abstract



     Precisealignmentisaprerequisiteforhyperspec-
     tral and multispectral image fusion, yet existing
     methods struggle with complex non-rigid defor-
     mations. Existing techniques either suffer from
     inter-task error accumulation by treating regis-
     tration and fusion as disjoint processes or ne-
     glect the geometric nature of distortions by re-
     lyingonisotropicEuclideanmetrics. Wepropose
     Structure-AwareRiemannianFlowMatching(SA-
     RFM), a geometry-informed framework for joint                       Figure 1. Conceptual comparison of different deep-learning-based
     registrationandfusionofhyperspectralandmul-                         registration and fusion methods.  (A) Discriminative methods:
     tispectralimages. SA-RFMreformulatesregistra-                       Theserequireexplicitmodelingofthedeformationfield,wherethe
     tion as dynamic optimal transport on a structure-                   two-stageoptimizationispronetolocaloptima. (B)Conditional
     induced Riemannian manifold, where anisotropic                      Diffusion: ThesemethodsrelyonisotropicGaussiannoise,which
     costsarederivedfromMSIstructuralcues. Tocir-                        lacks physical priors. (C) Ours: By integrating texture-informed
     cumventthecomplexityofexplicitOTsolvers,we                          anisotropicRiemannianmanifoldswithoptimaltransportoptimiza-
                                                                         tion, our approach achieves unified one-step modeling.
     incorporatethisgeometryinto aconditionalflow
     matchingframeworkviaalocalcostapproxima-
     tion. Thisformulationisrealizedthroughtwosyn-                       1. Introduction
     ergistic mechanisms: a Riemannian Flow Match-
     ing objective that enforces structure-aware error                   Hyperspectral images (HSIs), which capture scenes across
     measurement,andanoptimaltransportdirection                          hundredsofnarrowandcontiguousspectral bands,provide
     regularization that aligns the velocity field with                  richspectral informationthat isinvaluable fora widerange
     the induced metric, thereby resolving the funda-                    ofapplications,includingprecisionagriculture(Sahooetal.,
     mental mismatch between anisotropic costs and                       2015;Lu et al.,2020;Ang & Seng,2021), environmen-
     conventionalEuclideansupervision. Extensiveex-                      talmonitoring(Stuartet al.,2019;Peyghambari&Zhang,
     perimentsonfourdatasetsdemonstratethesuperi-                        2021), food safety (Feng & Sun,2012;Pu et al.,2023),
     orityofourmethod. Codeisavailableat:https:                          mineral exploration (Bishop et al.,2011;Pour et al.,2021),
     //github.com/ZhangQuan-hub.                                         and remote sensing (Zhu et al.,2023;Dian et al.,2026;
                                                                         Liu et al.,2025). However, due to fundamental hardware
                                                                         constraints, hyperspectralimaging systemsinevitablyface
                                                                         a trade-off between spatial and spectral resolution.  As a
                                                                         result, HSIs typically exhibit low spatial resolution, which
   1College ofComputer Scienceand Technology, National Uni-
versity of Defense Technology, Changsha, China 2College of Elec-         severely limits their practical usability.

tronic Science and Technology, National University of Defense            To alleviate this limitation, multispectral and hyperspec-
Technology, Changsha, China.  Correspondence to: Jun Li <li-             tralimage fusion(MHIF) hasemerged asa keytechnique.
jun14@nudt.edu.cn>.                                                      By combining the rich spatial details of a high-resolution
Proceedings of the 43rd International Conference on Machine              multispectralimage(HR-MSI)withtheabundantspectral
Learning,Seoul,SouthKorea. PMLR306,2026. Copyright2026                   informationofalow-resolutionHSI(LR-HSI),MHIFaims
by the author(s).                                                        toreconstructahigh-resolutionHSI(HR-HSI)(Luoetal.,

                                                                      1

<!-- PDF PAGE 2 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

2025;Maetal.,2024). Crucially,thesuccessofMHIFcriti-                      incorporate the induced geometry into a conditional flow
callydependsonaccuratepixel-levelalignmentbetweenthe                      matching framework via a locally approximated transport
MSI and HSI.In real-world scenarios,however, such image                   cost. SA-RFM introduces two complementary geometric
pairsareoftenaffectedbycomplexnon-rigidmisalignments                      mechanisms.  First, a Riemannian Flow Matching objec-
caused by platform vibrations, sensor parallax, and atmo-                 tive enforces structure awareness at the error-measurement
spheric turbulence.  Consequently, robust cross-modality                  level by penalizing velocity deviations along high-cost di-
registration remains a fundamental yet highly challenging                 rections.   Second, we identify a fundamental mismatch
prerequisite for reliable hyperspectral image fusion.                     between anisotropic transport costs and conventional Eu-
Traditionalregistration methods(Lowe,2004;Zhouet al.,                     clideandisplacementsupervision,andresolveitbyintroduc-
2020)oftenfailunderseveremodalitygapsandhavelargely                       ing a structure-aware optimal transport direction regulariza-
been superseded by deep learning approaches.  Existing                    tion. This regularization explicitly aligns the learned veloc-
learning-based methods can be broadly divided into two                    ityfieldwithminimum-costtransportdirectionsunderthe
paradigms.  The first paradigm consists of discriminative                 inducedRiemannianmetric,leadingtostructure-consistent
regression modelsbased onconvolutionalneural networks                     deformations without sacrificing computational efficiency.
(Dong et al.,2021;Zheng et al.,2021;Qu et al.,2025a;c),                   Main Novelty and Contributions.  1) We formulate the
transformers (Dong et al.,2024) or Mamba-based meth-                      unaligned HSI and MSI fusion task as a geometry-aware
ods (Li et al.,2025). As shown in Figure1, these meth-                    dynamic optimal transport problem, and demonstrate how
ods explicitly reconstruct pixel-level deformation fields for             optimaltransport principlescan beintegratedinto learning-
registration,uponwhichimagefusionisperformedsubse-                        basedregistrationwithoutexplicitlysolvingOTorcomput-
quently. However,such two-stage optimizationis proneto                    inggeodesics. 2)Weproposeastructure-awareRiemannian
task-switching errors, which can cause the model to fall                  flowmatchingframeworkthatintroducestwocomplemen-
into local minima and render it unable to handle complex                  tarygeometricmechanisms: aRiemannianflowmatching
distortion types.  The second paradigm leverages genera-                  objectivethatenforcesstructure-awareerrormeasurement,
tive models (Ho et al.,2020;Song et al.,2020;Rombach                      and an optimal transport direction regularization that re-
et al.,2022), which offer stronger image prior modeling                   solvesthefundamentalmismatchbetweenanisotropictrans-
capabilities.  Diffusion models are one of the more pop-                  portcostsandconventionalEuclideandisplacementsuper-
ular methods among them (Qu et al.,2025b;Diao et al.,                     vision. 3)Extensiveexperiments onmultiplehyperspectral
2025). However, standard diffusion-based formulations rely                benchmarks show that SA-RFM consistently achieves su-
on isotropic Gaussian noise priors. Whenapplied to regis-                 perior alignment accuracy and spectral fidelity, while main-
tration,thisassumptionintroducesafundamentalgeometric                     taining high computationalefficiency compared to existing
mismatch: structured, directionalmisalignments aretreated                 discriminative and generative registration and fusion ap-
asisotropicnoise. Consequently,thedenoisingtrajectoryis                   proaches.
forced to traverse a noise-dominated space that ignores im-
age topology, often resulting in spectral mixing and blurred              ConflictofInterestDisclosure.   Theauthorsdeclareno
structural boundaries.                                                    financial conflicts of interest related to this work.
Wearguethatimageregistrationshouldinsteadbeviewed
asatransportproblemgovernedbygeometry-awarecosts.                         2. Preliminaries
Specifically, transporting pixels across strong structural                We provide a brief review of Optimal Transport and Con-
boundaries should incur significantly higher costs than mo-               ditional Flow Matching. Related work is discussed in Ap-
tionalongcoherentstructures. Thismotivatesaformulation                    pendixC.
based on Optimal Transport (OT) with anisotropic, data-
dependent costs. However, explicitly solving OT or com-                   2.1. Optimal Transport
putinggeodesics onhigh-dimensional imagemanifoldsis
computationally intractable.                                              OptimalTransport(OT)providesaprincipledframeworkfor
To address these challenges, we propose Structure-Aware                   comparingandtransformingprobabilitymeasuresbymini-
Riemannian  Flow  Matching  (SA-RFM),  a  geometry-                       mizing transport costs. In the context of image registration,
informedlearningframeworkfortheregistrationandfusion                      Monge OTis theoretically superiorto stochasticrelaxations
of HSIsand MSIs. SA-RFM modelsimage registration asa                      (such as the Kantorovich problem) because physical reg-
dynamic optimal transport process on a structure-induced                  istrationnecessitatesa point-to-point correspondence(i.e.,
Riemannian manifold, where anisotropic transport costs                    deterministic deformation) rather than mass splitting or dif-
are derived from MSI structural cues. Rather than explic-                 fusion.  We focus on the Monge problem, which seeks
itly solving optimal transport or computing geodesics, we                 a deterministic mapping T  : Ω → Ω (where Ω ⊂ R2
                                                                          denotes the spatial domain and u ∈ Ω is the spatial coor-

                                                                      2

<!-- PDF PAGE 3 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images






























Figure2. The proposedSA-RFM framework. (a) Architecture ofthe fusion modules andvelocity network, where boththe encoders and
decoder are constructed usingConvBlocks. (b)The training scheme involving datasampling at various time steps. (c) Illustration of the
proposedRiemannianmetricformulatedinEq. (7)   ,whichassignshighertransportcostsacrosstexturedirectionsandlowercostsalong
them.


dinatevector) thatpushesthesource distributionP toward                     efficient optimization on large-scale image data via flow
the target distributionQ. This process is described by the                 matching schemes.
push-forwardoperatorT#: T#P =Q,representingtheevo-
lution ofthemeasure underthe mapping. Theoptimization                      2.2. Conditional Flow Matching
objective is to minimize the total transport cost:                         Conditional Flow Matching (CFM) is a method for learn-
                           Z                                               ing the velocity fieldvt that generates a probability path
                 C(T) =                                                    between the starting distribution P and the target distri-
                             Ωc(u,T(u))dP(u),                  (1)
where c(u,T(u)) encodes the geometric cost of moving                       bution Q.  While directly learning the total vector field
mass from u to T(u), typically taken as the squared Eu-                    of marginal probability paths is extremely difficult, we
clidean distance in traditional frameworks.                                can learn a conditional vector field conditioned on spe-
                                                                           cific sample pairs(s0,s1). For a linear interpolation path
TheBenamou-Brenierformulaestablishesanequivalent                           ψt(u|s0,s1) = (1− t)s0 + ts1, the corresponding con-
dynamicperspectivefortheMongeproblem. Itrepresents                         ditional vector field (i.e., conditional velocity) is constant:
the transport process through a time-varying velocity field                ut(st|s0,s1) =s1−s0. Byintegratingoverallsamplepairs
vt, governed by the ordinary differential equation (ODE)                   inthe trainingset,the CFMobjectivefunction isdefinedas:
dst
 dt  = vt(st), wherest denotes the particle trajectory and
t∈ [0,1] is the time parameter. In this dynamic view, the
Monge problem is equivalent to minimizing the average                        LCFM(θ) =Et,q(s0,s1),st       ∥vθ(st,t)−(s1−s0)∥2,  (3)
kinetic energy of the flow over the entire time interval:
                  Z 1  Z    1
              infvt         2∥vt(u)∥22ρt(u)dudt,               (2)
                    0    Ω                                                 where q(s0,s1) is the joint distribution of the source and
subject to the continuity equation ∂tρt +∇·(ρtvt) = 0,                     target samples, and vθ(st,t) represents the velocity field
whereρt(u) denotes the probability density function ofthe                  approximatedbyaneuralnetworkwithparametersθ. CFM
transported measureat timet and positionu. This dynamic                    converts the complex optimization into a regression task for
perspective bridges OT with generative models, enabling                    a network representing the time-dependent velocity.

                                                                        3

<!-- PDF PAGE 4 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images
3. Methodology                                                           from the MSI as
We propose Structure-Aware Riemannian Flow Matching                                          cX
(SA-RFM),ageometry-informedlearningframeworkthat                                SY(u) =           ∇Y(k)(u)∇Y(k)(u)⊤            ∗Gk,σ,    (6)
explicitlypreservesfineimagestructuresduringflowmatch-                                      k=1
ingbyconstructingastructure-inducedRiemanniangeome-                      where∇ denotes the gradient,which is extracted using the
try. Our framework is shown in Figure2.                                  Sobel operator. The ablation study evaluating various gra-
                                                                         dient computation methods is illustrated in Table6.        Y(k)
3.1. Problem Formulation                                                 denotes thek-th MSI band andGk,σ is a Gaussian smooth-
GeometricModelingofUnregisteredImageFusion. Con-                         ingkernel withkernel sizek andstandard deviationσ. The
sider an unregistered low-resolution hyperspectral image                 ablationstudiesondifferentconstructionmethodsofSY(u)
(LR-HSI)X↓∈Rh×w×C and a high-resolution multispec-                       and the parameters of the Gaussian kernel are shown in
tral image (HR-MSI) Y ∈ RH×W×c.  Our goal is to re-                      Table6and Figure5.
construct an aligned high-resolution hyperspectral image                 Based on this tensor, we define a position-dependent Rie-
Z∈RH×W×C thatpreservesbothspectralinformationand                         mannian metric:
spatial structures.
From a transport perspective, we treat the upsampled LR-                                      M(u) =I +βSY(u),                        (7)
HSIX∈RH×W×C asthesourcedistributions0∼PX,the                             whereI denotes the2×2 identity matrix representing the
desiredaligned HR-HSIas thetarget distributions1∼QZ,                     isotropic baseline and β > 0 is a scalar parameter con-
and the MSI as a reference distributiony∼PY providing                    trolling the sensitivity to scene structures.  The exact for-
reliablestructuralpriors. Theregistration–fusionoperator                 mulation of the structure tensor is provided in Appendix
is defined as:                                                           B.4.
                     QZ =F(T#PX,PY),                        (4)          Ratherthannumericallysolvingthestaticassignmentprob-
                                                                         lem, we adopt a dynamic formulation to characterize the
where T : Ω → Ω denotes the spatial transport map and                    transformation flow. The transport map is represented by a
F represents the fusion operator.  In this formulation, T                continuous deformationst governed by a time-dependent
correctsgeometric distortionsinPX viathe push-forward                    velocity fieldvt:
operator T#, whileF subsequently fuses high-frequency
spatial details from the HR-MSI.                                                     dst
                                                                                      dt  =vt(st),   s0 =u,   s1 =T(u).           (8)
3.2. Structure-Aware Riemannian Flow Matching                            Under the geometric view induced by M(u), the optimal
Due to the high dimensionality and severe ill-posedness of               transport task is equivalent to finding a velocity field that
image registration, we adopt the minimum-cost principle as               minimizes theaccumulated structure-awarekinetic energy:
a physical constraint. Registration is interpreted as trans-                               Z 1  Z
porting spectral vectors in the spatial domain, seeking a                       J(vt) =
smooth andinvertible mapT∗ that minimizesa cumulative                                        0    Ωvt(u)⊤M(u)vt(u)ρt(u)dudt.     (9)
transport cost:                                                          This formulation ensures that the registration path is not
                                Z                                        only smooth and invertible but also respects the underlying
               T∗ = argmin                                               semanticboundariesoftheHR-MSI.DerivationofEq. (9)
                          T       Ωc(u,T(u))du.                (5)       is provided in AppendixB.1.

Thedesignofthecostfunctionciscentraltopreservingthe                      Under this metric, the transport cost between two spatial
semantic integrityof the scene. We observethat traditional               locations: sourcelocationu atLR-HSIanddestination lo-
isotropic Euclidean transport costs fail to perceive complex             cationw at HR-HSI is naturally defined by the Riemannian
scene structures. In unregistered scenarios, if pixels cross             energy:
significantphysicalboundaries,suchasroad edgesorbuild-                                                  Z 1
ing contours, during the transport process, severe spectral                 c(u,w) =        inf
mixingandghostingartifactsmayoccur. Therefore,anideal                                      γ:[0,1]→Ω      0  ˙γ(t)⊤M(γ(t))˙γ(t)dt,   (10)
                                                                                         γ(0)=u, γ(1)=w
transport process should be anisotropic: movements across
structuralboundariesshouldincurhighcosts,whilemotions                    whereγ(t) denotes a continuous pathconnectingu andw,
alongstructuraldirectionsshouldbeencouraged. Tocapture                   and ˙γ(t) = dγdt ∈Tγ(t)Ωdenotesthetangentvelocityofthe
this spatial heterogeneity, we construct a structure tensor              path at timet.

                                                                      4

<!-- PDF PAGE 5 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

Algorithm1TheTrainingProcessofOurStructure-Aware                        vector difference between target and source at location u.
Riemannian Flow Matching (SA-RFM)                                       Wethendefineastructure-awareoptimalone-steptransport
Require:  DatasetD of tuples (X↓,Y,Z) (LR-HSI, HR-                      directionv†(u) as the solution to:
    MSI, HR-HSI); epochs E;  metric params (β,k,σ);                                      v†(u) = argmin
    weightλOTD.                                                                                         v     v⊤M(u)v,                 (13)
Ensure: Trained velocity networkfθ.                                                       s.t.  ⟨v,d(u)⟩ =∥d(u)∥2.
 1: Initializeθ and optimizer.
 2: fore = 1,...,E do                                                   This formulation preserves the global direction toward the
 3:     Sample mini-batch(X↓,Z,Y)∼D.                                    target(fixedbytheinnerproductconstraint)whileselecting
 4:     x0← Bicubic(X↓),   x1←Z.                                        the minimum-cost direction under the Riemannian metric.
 5:     Sample t ∼ U(0,1),    xt ← (1− t)x0 + tx1,
       u←x1−x0.                                                         The above problem admits a closed-form solution:
 6:     ComputestructuretensorSY fromY (Sobel+Gaus-
       sianGk,σ); metricM ←I +βSY.                                                  v†(u) =     M(u)−1d(u)
 7:     Predict velocityvθ←fθ(xt,t,X↓,Y).                                                     d(u)⊤M(u)−1d(u)∥d(u)∥2.         (14)
 8:    LRFM←E(vθ−u)⊤M(vθ−u).
 9:     v†  ←  argminvv⊤Mv     s.t.   ⟨v,u⟩  = ∥u∥2;.                  The derivation from Eq. (13)      to Eq. (14)      is provided in
       LOTD←E∥vθ−v†∥22                                                 AppendixB.2. Geometrically,                   v† can be seen as a metric-
10:     Updateθ by minimizingL =LRFM+λOTDLOTD.                          correction of the Euclidean displacement that favors low-
11: end for                                                             cost structural directions.
                                                                        We incorporate the optimal transport direction as an addi-
Forsmalldisplacements∆u =w−u,weapproximate the                          tional regularization term:
above energy by freezing the metric at u and assuming a                              LOTD =Eu∥vθ(u,t)−v†(u)∥22,           (15)
straight-line path, leading to the following local quadratic
form:                                                                   wherevθ(u,t) denotesthe neuralnetworkprediction atlo-
              c(u,u+∆u) ≈  ∆u⊤M(u)∆u.             (11)                  cation u and arbitrary time t sampled uniformly in [0,1]
This approximation preserves directional selectivity while              during training.
remainingcomputationallyefficient. Moreablationresults                  The overall training objective is:
on different metric constructions are reported in Table7.
Tolearnthedeformationflow,weadoptConditionalFlow                                          L =LRFM+λOTDLOTD,                  (16)
Matching (CFM). Inone-step CFM, the conventional super-                 whereλOTD is the regularization coefficient balancing Eu-
vision uses the Euclidean displacement:v∗(u) = s1(u)−                   clideanconsistencyandstructure-awaredirectionaloptimal-
s0(u),herev∗(u) representsthetargetvelocityvectoratlo-                  ity (set to 1.0 in our experiments). Sensitivity analysis is
cationu. Toincorporatestructuralconstraints,weintroduce                 shown in Figure4. Our implementation is summarized in
the Riemannian Flow Matching (RFM) loss:                                Algorithm1.
                     h
 LRFM =Et,s0,s1
     vθ(st,t)−v∗(u)⊤M(st) vθ(st,t)−v∗(u)i                   (12)      4. Experiments
                                                            .           4.1. Dataset Preparation

This loss penalizes errors along high-cost directions and               Datasets. Weconductedextensiveexperimentsonfourhy-
encourages the learned velocity field to respect structures.            perspectral datasets with varying spectral bands: Botswana,
                                                                        Pavia University, Chikusei, and Xiongan. The Botswana
3.3. Optimal Transport Direction Regularization                         dataset contains 1476× 256 pixels with a 30-m spatial
WhileLRFM introduces structure awareness at the error-                  resolution, covering 145 bands in the 0.4–2.5µm range.
measurement level, its supervision target remains the Eu-               The Pavia University dataset consists of610×340 pixels
clideandisplacementv∗(u). Inananisotropicspaceinduced                   witha1.3-m spatialresolution,comprising103 bandsrang-
byM(u), however, the Euclidean direction is not necessar-               ing from 0.43 to 0.86µm.  The Chikusei dataset features
ily the minimum-cost transport direction.                               2517×2335 pixels with a 2.5-m spatial resolution, cover-
                                                                        ing 128 bands across0.36–1.02µm. The Xiongan dataset
Toaddressthisissue,wefirstdefinetheEuclideandisplace-                   includes3750×1580pixelswitha0.5-mspatialresolution,
mentfieldasd(u) =s1(u)−s0(u),representingthedirect                      spanning 256 spectral bands from0.39 to1.00µm.

                                                                     5

<!-- PDF PAGE 6 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

Table 1. Quantitative comparison on the Pavia University dataset under Elastic, Homography, and Turbulence distortions. The best
metrics are highlighted inBoldand the second-best are Underlined. The gray column highlights our proposed method.

                                               MoGDCN       NonRegSR       M2DTN         MoEPNP        IR&ArF       PSRFDif        RFDif        PRFCo
 DISTORTION   METRIC     Integrated(TGRS’20)    (TIP’21)    (TGRS’22)     (AAAI’24)    (TNNLS’24)     (TIP’25)     (TGRS’25)    (TGRS’25)     (TGRS’25)       Ours
                    PSNR↑       40.66        36.22        39.43        37.19        40.13        39.43        41.49        37.77        42.33                42.69
     Elastic         SSIM↑       0.981        0.961        0.976        0.968        0.979        0.978        0.982        0.971        0.983               0.987
     (α=600,         SAM↓       2.163        3.610        2.629        3.219        2.376        2.697        2.015        2.897        1.759                1.735
      σ=15)        ERGAS↓     1.167        1.941        1.339        1.736        1.252        1.332        1.067        1.607        0.970                  0.931
                    RMSE↓      2.363        3.942        2.724        3.522        2.512        2.724        2.148        3.298        1.951                 1.872
                    PSNR↑       40.85        38.39        39.16        35.59        40.96        36.73        41.03        38.88        42.33                42.57
 Homography          SSIM↑       0.981        0.975        0.975        0.957        0.981        0.968        0.982        0.974        0.983               0.987
   (MPO=10%)         SAM↓       2.165        2.873        2.676        3.833        2.115        3.522        2.078        2.684        1.758                1.758
                   ERGAS↓     1.142        1.508        1.381        2.073        1.134        1.837        1.118        1.419        0.972                  0.941
                    RMSE↓      2.312        3.071        2.811        4.239        2.283        3.718        2.266        2.900        1.951                 1.898
                    PSNR↑       40.99        38.39        39.99        39.61        41.18        39.86        41.41        39.03        42.62                43.10
  Turbulence         SSIM↑       0.981        0.975        0.977        0.977        0.981        0.980        0.982        0.977        0.984               0.988
   (D/r0=2.0)        SAM↓       2.137        2.873        2.459        2.543        2.078        2.468        2.051        2.622        1.722                1.658
                   ERGAS↓     1.120        1.508        1.256        1.316        1.108        1.267        1.073        1.394        0.939                  0.886
                    RMSE↓      2.275        3.071        2.554        2.666        2.225        2.593        2.169        2.851        1.886                 1.785
         Params (M)                   –            7.918        1.900       17.64        45.42        23.20        66.24        14.72        8.367           1.078
         FLOPs (G)                   –            567.6        62.32        169.9        905.0         1260         34.67         113.6        181.5         21.29


     Deformation         Integrated          MoGDCN         NonRegSR           M2DTN            MoEPNP            IR&ArF             PSRFDif              RFDif                PRFCo                 Ours          Colorbar














Figure 3. Visual comparison on the Pavia University dataset. The first column displays the deformation fields, followed by the recovery
results of various methods and the error Colorbar. The top-left corner displays the difference map, while the bottom-right corner shows
the RGB (bands 67-29-1) image.


DatasetSplitting. Following(Zhangetal.,2020),wedesig-                                  deformationfieldsareshowninFigure3. Pleaserefertothe
natedthecentral128×128regionofeachdatasetasthetest                                     Appendix for additional experimental results.
set. Duringtraining,thistestregionwasmaskedtozero,and
trainingsamplesof size128×128wererandomlycropped                                       4.2. Benchmark and Evaluation Metrics
from the remaining areas.  Training data were generated                                Benchmark. We compared our method with publicly avail-
followingWald’s protocol(Waldetal.,1997). TheLR-HSI                                    able state-of-the-art approaches, categorizing them into:
wascreatedbyapplyinga5×5Gaussianblur(standardde-                                       (1) Traditional method: Integrated (Zhou et al.,2020); (2)
viationσ = 2)followedby4×bilineardownsampling. The                                     CNN-based methods: MoGDCN (Dong et al.,2021), Non-
HR-MSIwasgeneratedusingthespectralresponsefunction                                     RegSR(Zhengetal.,2021),andMoEPNP(Quetal.,2025a);
of the Gaofen-1 satellite (Chen et al.,2022).                                          IR&ArF (Qu et al.,2025c); (3) Transformer-based method:
Distortion Simulation. To evaluate robustness,we applied                               M2DTN (Dong et al.,2024), (4) Mamba-based method:
different degrees of distortion to the LR-HSI, including:                              PRFCo(Lietal.,2025);and(5)Diffusion-basedmethods:
(i) Elastic distortion with parameters (α,σ) = (600,15);                               PSRFDif (Qu et al.,2025b) and RFDif (Diao et al.,2025).
(ii) Homography distortion with Maximum Perturbation                                   Metrics. Performanceisevaluatedusingfivestandardquan-
Offset(MPO)ratesof10%;and(iii)Turbulencedistortion                                     titative metrics: Peak Signal-to-Noise Ratio (PSNR) (Hore
(Maoetal.,2021)withtheratioofaperturediametertothe                                     &Ziou,2010),StructuralSimilarityIndex(SSIM)(Wang
Fried parameter, D/r0 = 2.0.  Visualizations of specific

                                                                                   6

<!-- PDF PAGE 7 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

Table 2. Quantitative comparison on the Botswana dataset under Elastic, Homography, and Turbulence distortions. Bold indicates best,
Underline   indicates second-best. The gray column highlights our proposed method.

                                            MoGDCN      NonRegSR      M2DTN        MoEPNP       IR&ArF       PSRFDif       RFDif       PRFCo
 DISTORTION  METRIC     Integrated(TGRS’20)  (TIP’21)   (TGRS’22)    (AAAI’24)    (TNNLS’24)    (TIP’25)    (TGRS’25)    (TGRS’25)   (TGRS’25)       Ours
                   PSNR↑      41.17        42.71        40.16        38.29        43.32        45.70          39.57        45.15        44.23       46.45
    Elastic        SSIM↑       0.955        0.987        0.980        0.970        0.989        0.993         0.978        0.992        0.988       0.995
    (α=600,        SAM↓       5.985        3.651        4.232        5.252        3.580        2.573          5.381        2.797        3.657       2.411
     σ=15)        ERGAS↓     1.360        1.142        1.530        1.872        1.066        0.806           1.650        0.867        0.966       0.743
                  RMSE↓      2.229        1.867        2.503        3.105        1.739        1.322           2.679        1.410        1.567       1.214
                   PSNR↑      40.90        40.00        40.81        39.31        43.85          40.74        39.01        41.46        43.28       43.93
 Homography        SSIM↑       0.970        0.976        0.982        0.978        0.990        0.960        0.974        0.967        0.986        0.977
  (MPO=10%)        SAM↓       4.629        4.813        4.121        4.810        3.376          4.342        5.999        4.014        3.897       3.235
                  ERGAS↓     1.408        1.550        1.420        1.713        1.003           1.457        1.737        1.319        1.061       0.983
                  RMSE↓      2.299        2.549        2.323        2.761        1.637           2.342        2.859        2.157        1.748       1.622
                   PSNR↑      41.42        43.51        41.48        41.84        43.27        45.92          40.28        45.34        44.14       46.69
  Turbulence       SSIM↑       0.961        0.988        0.984        0.984        0.989        0.993         0.979        0.993       0.988        0.995
   (D/r0=2.0)      SAM↓       4.965        3.374        3.837        4.167        3.536        2.519          5.504        2.647        3.660        2.38
                  ERGAS↓     1.330        1.039        1.318        1.268        1.067        0.790           1.522        0.861        0.976        0.72
                  RMSE↓      2.165        1.702        2.150        2.063        1.750        1.290           2.469        1.379        1.582        1.18
       Params (M)↓                 –            9.015        1.907     27.79        45.97        29.30        94.98        16.27        16.49       1.203
       FLOPs (G)↓                 –            588.4        62.53      323.9         1038         1444         65.46        134.8        357.8      24.02


et al.,2004), Spectral Angle Mapper (SAM) (Yuhas et al.,                          errors for MoGDCN and IR&ArF are visually more pro-
1992), Dimensionless Global Error in Synthesis (ERGAS)                            nounced. In comparison, our smaller difference indicates
(Wald,2000), and Root Mean Square Error (RMSE).                                   that CNN-based models, when coupled with precise loss
                                                                                  function design, can achieve superior convergence. From
4.3. Implementation Details                                                       Table2andTable3,itcanstillbeobservedthatourmodel
Wetrained thenetworkfor 400iterations,utilizing24 ran-                            achievesthebestperformanceinmostcaseswiththefewest
domlycropped regionsper iteration. The optimizationwas                            parameters and floating-point operations.  More analyses
performed using the Adam optimizer with a learning rate of                        and visual results are provided in AppendixG.

1×10−4. All experiments were implemented in PyTorch                               4.5. Ablation and Parameter Sensitivity Studies
2.6.0 on a server running Ubuntu 22.04 equipped with an
NVIDIA RTX 4090 GPU.                                                              Tovalidatetheeffectivenessofourspecificdesignchoices,
                                                                                  we further conduct comprehensive ablation studies and sen-
4.4. Quantitative and Visual Results                                              sitivityanalysesbasedontheresultsobtainedfromthePavia
In Table1, we present the quantitative comparison be-                             University dataset under elastic distortions.
tween our proposed method and existing state-of-the-art                           Impact of the two loss functions. As illustrated in Table
approaches on the Pavia University dataset. While the per-                        4,the configuration”without LRFM”denotes thescenario
formanceofthetraditionalIntegratedmethodisnotparticu-                             where our objective degenerates into the standard velocity
larly remarkable, it remains stable across various distortion                     loss used in vanilla flow matching.  The results indicate
types. Incontrast,MoGDCN,M2DTN,andIR&ArFexhibit                                   that whileLRFM exerts a more pronounced influence on
high sensitivity to the specific type of distortion. RFDif per-                   performance,LOTD alsoprovidesadiscerniblecontribution.
formspoorlyonthePaviaUniversitydataset,yetasshown                                 Crucially,bothlossfunctionsareshowntofacilitatesuperior
in theAppendix, it yieldssignificantly better resultson the                       modelconvergence,therebyvalidatingtheeffectivenessand
Botswanadataset;PSRFDifdemonstratestheoppositetrend.                              necessity of our proposed optimization objectives.
This discrepancy suggests that these two diffusion-based                          Impactof thesampling steps. Asdemonstrated inTable5,
architecturessufferfrompoorgeneralization. Furthermore,                           we compareour method withtwo representative diffusion-
although the PRFCo method achieves competitive perfor-                            based approaches.  The results indicate that our method
mance, it entails substantial computational overhead. Our                         achieves superior performance using significantly fewer
method, designed based on a CNN architecture, maintains                           sampling steps. While the sampling accuracy further im-
the lowest computational cost while achieving optimal per-                        proves as the number of steps increases, it concurrently
formance due to the strategic design of the optimization                          introduces additional computational latency. Specifically,
objective.  As illustrated in Figure3, the reconstruction

                                                                              7

<!-- PDF PAGE 8 -->

                       Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

                Table 3. Quantitative comparison on the Xiongan dataset under Elastic, Homography, and Turbulence distortions. Bold indicates best,
                Underline    indicates second-best. The gray column highlights our proposed method.

                                                               MoGDCN        NonRegSR      M2DTN         MoEPNP       IR&ArF       PSRFDif        RFDif        PRFCo
                 DISTORTION   METRIC     Integrated(TGRS’20)    (TIP’21)     (TGRS’22)    (AAAI’24)    (TNNLS’24)     (TIP’25)    (TGRS’25)     (TGRS’25)    (TGRS’25)       Ours
                                     PSNR↑       41.26        41.75        43.64        38.34        42.74        48.90             43.31        48.08        47.06          50.61
                     Elastic         SSIM↑       0.981        0.986        0.989        0.980        0.986        0.997             0.991        0.996        0.996          0.997
                     (α=600,         SAM↓       2.455        2.359        1.959        2.943        2.272        1.149              2.040        1.246        1.400          0.999
                      σ=15)         ERGAS↓     1.112        1.047        0.843        1.551        0.937        0.462               0.878        0.509        0.570          0.381
                                     RMSE↓      2.206        2.085        1.678        3.086        1.860        0.915              1.742        1.006        1.131          0.752
                                     PSNR↑       40.69        41.08        43.89        39.67        45.39        46.42        42.52        47.05        47.08               49.43
                 Homography          SSIM↑       0.972        0.988        0.989        0.981        0.993        0.995        0.991        0.994        0.996               0.997
                   (MPO=10%)         SAM↓       2.557        2.458        1.940        2.786        1.696        1.416        2.063        1.371               1.377         1.083
                                    ERGAS↓     1.186        1.127        0.820        1.331        0.692        0.615        0.960        0.571        0.568                 0.435
                                     RMSE↓      2.356        2.251        1.630        2.649        1.371        1.217        1.908        1.133        1.129                0.861
                                     PSNR↑       41.63        46.33        44.00        40.67        40.26        49.62        43.75        50.15              48.07         51.98
                   Turbulence        SSIM↑       0.981        0.993        0.989        0.987        0.979        0.997             0.991        0.996        0.996          0.998
                   (D/r0=2.0)        SAM↓       2.299        1.476        1.937        2.437        2.716        1.050              2.051        1.059        1.321          0.909
                                    ERGAS↓     1.069        0.620        0.809        1.182        1.245        0.423        0.831        0.400                0.508         0.326
                                     RMSE↓      2.114        1.230        1.609        2.360        2.474        0.843        1.655        0.793               1.007         0.642













                     Figure 4. Parameter sensitivity analyses ofλOTD,β and number of channels of hidden layers regarding the model performance.


                Table4. Ablationstudyoftheproposedlossfunctionsunderelastic                           Table 5. Quantitative comparison and step analysis on the Pavia
                distortiononthePaviaUniversitydataset. Thegrayrowhighlights                            University dataset under elastic distortion. Units: Time (ms).
                our full model.









                the early PSRFDif requires 33,624 ms for a full sampling                               usedforHR-MSI.Thethirdrowcorrespondstoutilizingthe
                cycle,whereasourone-stepimplementationtakesonly9.66                                    averagegradientacrossallspectralbands. Inthefourthrow,
                ms. Combined with the significantly lower parameter count                              the model incorporates gradient information via isotropic
                and FLOPs reported in Table2, our method is far more                                   modeling,injectingitintothenetworkasastandardfeature
                conducive to practical real-world deployment.                                          map. ThefifthrowinvolvesreplacingtheSobeledgedetec-
                                                                                                       tion operator with the more sophisticated Scharr operator,
                ConstructionoftheRiemannianmetricM inEq. (7)   . We                                    while the sixth row employs the simpler Prewitt operator.
                conductablationexperimentsonthespecificconfigurations                                  Our analysis indicates that the utilization of gradients sig-
                of M, and the results are presented in Table6.  Specifi-                               nificantly enhances model performance, which is consistent
                cally, the first row represents the baseline scenario where                            with our empirical observations. Furthermore, the specific
                optimizationis performedentirelyinEuclidean space. The                                 choice of gradient operators and the method of utilizing
                second row evaluates a variation where gradient informa-                               gradient information have a discernible but minor impact
                tionis obtainedbytakingthe maximumgradientacrossall                                    on performance. Finally, theselection between Riemannian
                bands,ratherthantheband-by-bandaccumulationtypically                                   and Euclideanspace affects performancein a mannercom-

                                                                                                   8


METHOD          PSNR↑    SSIM↑    SAM↓    ERGAS↓    Time↓    NFE↓
LRFM  LOTD          PSNR↑   SSIM↑   SAM↓   ERGAS↓   RMSE↓
PSRFDif×      ×  41.492     0.982     2.015       1.067       33624     200041.107     0.983     2.164      1.106       2.245
RFDif            37.774     0.971     2.897       1.607      524.92      25
Ours(20-step)✓      ×42.897     0.988     1.689       0.908      117.43      2042.142     0.986     1.884      0.986       1.993
Ours(10-step)×     ✓42.827     0.988     1.694       0.914      59.667      1041.704     0.987     1.821      1.036       2.096
Ours(5-step)✓✓   42.734     0.988     1.693       0.925      28.126       542.6930.9871.7350.9311.872
Ours(1-step)     42.693     0.987     1.735      0.931     9.6574       1

<!-- PDF PAGE 9 -->

                     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

               Table 6. Ablation study of theRiemannian metric constructionM                  Table 8. Ablation study on different backbone networks on the
               on the Pavia University dataset under elastic distortion.                      Pavia University dataset under elastic distortion.











               Table 7.  Ablation study on different numerical approximations
               of the Riemannian transportcost on the Pavia University dataset
               under elastic distortion.  The gray row highlights our selected
               strategy.






                                                                                              Figure 5. Parameter sensitivity analyses of standard deviationσ
                                                                                              and kernel sizek of Gaussian kernel.
               parableto theinfluenceofthe isotropicmodelingapproach.
               Approximations of the Riemannian Transport Cost. We
               conductedablationexperimentsonthenumericalapproxi-
               mations for Eq. (11)     , with results summarized in Table7.                  channels = 220.  Additionally, ablation studies on the
               The first row evaluates the midpoint approximation, which                      GaussianblurkernelinEq. (6)    asshowninFigure5reveal
               computes the Riemannian metric at the trajectory’s cen-                        thatperformancesufferswhenboundariesareoverlyblurred
               ter. The second row represents the discrete path integral,                     or when the blurring effect is insufficient. We finally set the
               which accumulates transport costs along the entire trajec-                     kernel size tok = 5 and the standard deviation toσ = 1.0.
               tory.  In contrast, our approach freezes the metric at the
               source location; this strategy achieves competitive perfor-                    5. Conclusion
               mance—particularly in spectral mapping (SAM)—while
               significantly reducing computational overhead in terms of                      In this paper, we address the challenge of joint registra-
               both FLOPs and inference time.                                                 tion and fusion of hyperspectral and multispectral images
               Backbone Networks.   Given our focus on Riemannian                             under complex non-rigid distortions from the perspective
               space modeling and objective optimization, we evaluate                         ofRiemannianmanifoldmodeling. WeproposeStructure-
               different velocity and fusion network backbones in Table                       Aware Riemannian Flow Matching (SA-RFM), a geometry-
               8. OurCNN-basedchoiceprovidesan optimaltrade-off be-                           informed framework that reformulates image registration
               tween efficiency and performance. While ViT architectures                      as dynamicoptimal transport ona structure-induced mani-
               can offer superior accuracy, they entail significant com-                      fold. SA-RFM integrates anisotropic, structure-dependent
               putational overhead, justifying our selection for efficient                    transport costs into a conditional flow matching formula-
               deployment.                                                                    tion through a local cost approximation, and further intro-
                                                                                              ducesanoptimaltransportdirectionregularizationtoalign
               ParameterSensitivity. Comprehensivesensitivityanalyses                         thelearneddeformationfieldwithminimum-costtransport
               were conducted on the balancing weightλOTD, anisotropy                         directions under the induced geometry. Extensive experi-
               intensity β, and hidden layer channels (shown in Figure                        ments on multiple benchmarks under different distortions
               4). Results indicate that performance follows an inverted                      demonstratethatSA-RFMconsistentlyoutperformsexisting
               U-curve with respect to λOTD, suggesting that an exces-                        methodsin terms ofaccuracyandcomputational efficiency.
               sive weight dominates and hinders the optimization of the
               primary lossLRFM. The best weight isλOTD = 1.0. Re-                            Acknowledgements
               garding anisotropy, excessively high β values cause ex-
               treme spatial distortion, leading to performance degrada-                      This work was supported in part by the Young Scientists
               tion.  We select β = 0.5 in our paper.  While increasing                       FundoftheNationalNaturalScienceFoundationofChina
               the channelcount improvesperformance, thegains eventu-                         underNo. 62401578andinpartbytheShandongProvincial
               ally saturate asthe parameter volume grows; we thus select                     Natural Science Foundation under ZR2023LZH014.

                                                                                           9

METRICFORMULATION        PSNR↑   SSIM↑   SAM↓   ERGAS↓   RMSE↓
M=I(Euclidean)           41.933    0.987     1.782      1.010       2.041
c(u,u+∆u)BACKBONE u+(u+∆u)PSNR↑   SSIM↑   SAM↓   ERGAS↓ PSNR↑   SAM↓   ERGAS↓Params(M)↓   FLOPs(G)↓Time(ms)   FLOPs(M)
maximummaxc(Ic)          42.256    0.987     1.768      0.975       1.967
ResNet-18   41.286    0.985    1.933     1.065    14.53          18.10
∆u⊤M       2     ∆u    42.821   1.716     0.914    4.6551       9.6338
averageavgc(Ic)          42.323    0.987     1.777      0.967       1.952
L−1P
U-Net       39.849    0.981    2.380     1.275    31.51          16.03
IsotropicStructureMetric42.918   1.707     0.90442.056    0.987     1.774      0.997       2.01313.856       32.440
ViT-Base    43.800    0.990    1.563     0.826    130.5          56.27
i=0∆u⊤iM(xi)∆ui
ScharrOperator           42.564    0.987     1.749      0.942       1.898
Ours∆u⊤M(u)∆u42.693  0.98742.6931.7351.7350.9310.9311.0784.24956.488121.29
PrewittOperator          42.294    0.987     1.770      0.971       1.958
Ours                     42.693    0.987    1.735    0.931     1.872

<!-- PDF PAGE 10 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images
Impact Statement                                                         Dong, W., Xu, Y., Qu, J., and Hou, S.  Learning multi-
This paperpresents work whosegoal isto advancethe field                     modal cross-scale deformable transformer network for
of Machine Learning.  There are many potential societal                     unregistered hyperspectral image super-resolution.  In
consequences of our work, none which we feel must be                        Proceedings of the AAAI Conference on Artificial Intelli-
specifically highlighted here.                                              gence, volume 38, pp. 1573–1581, 2024.

                                                                         Feng, Y.-Z. and Sun, D.-W. Application of hyperspectral
References                                                                  imaginginfoodsafetyinspectionandcontrol: areview.
Ang, K. L.-M. and Seng, J. K. P.  Big data and machine                      Critical reviews in food science and nutrition, 52(11):
   learning with hyperspectral information in agriculture.                  1039–1058, 2012.

  IEEE Access, 9:36699–36718, 2021.                                      Guo,Y.,Fu, X.,Xu, M.,andJia, S. Stereo cross-attention
Arad, B. and Ben-Shahar, O. Sparse recovery of hyperspec-                   networkforunregisteredhyperspectralandmultispectral
   tral signal fromnatural rgb images. In European confer-                  image fusion.  IEEE Transactions on Geoscience and
   ence on computer vision, pp. 19–34. Springer, 2016.                      Remote Sensing, 61:1–15, 2023.

Arvanitidis, G., Hansen, L. K., and Hauberg, S.  Latent                  Ho,J.,Jain,A.,andAbbeel,P. Denoisingdiffusionproba-
   space oddity: on the curvature of deep generative models.                bilisticmodels. Advancesinneural informationprocess-
   arxiv. Preprint posted online October, 31, 2017.                         ing systems, 33:6840–6851, 2020.

Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., and              Hore, A. and Ziou, D.   Image quality metrics:  Psnr vs.
   Dalca,A.V. Voxelmorph: alearningframeworkforde-                          ssim. In 2010 20th international conference on pattern
   formable medical image registration. IEEE transactions                   recognition, pp. 2366–2369. IEEE, 2010.
   on medical imaging, 38(8):1788–1800, 2019.
                                                                         Jaderberg,M.,Simonyan,K.,Zisserman,A.,etal. Spatial
Beg, M. F., Miller, M. I., Trouv´e, A., and Younes, L. Com-                 transformer networks. Advances in neural information
   puting large deformation metric mappings via geodesic                    processing systems, 28, 2015.
   flowsofdiffeomorphisms. Internationaljournalofcom-
   puter vision, 61(2):139–157, 2005.                                    Li, Z., Wen, Y., Xiao, S., Qu, J., Li, N., and Dong, W. A
Bishop, C. A., Liu, J. G., and Mason, P. J. Hyperspectral                   progressive registration-fusion co-optimization a-mamba
   remotesensingformineralexplorationinpulang,yunnan                        network:  Toward deep unregistered hyperspectral and
   province,china. InternationalJournalofRemoteSensing,                     multispectralfusion. IEEETransactionsonGeoscience
   32(9):2409–2426, 2011.                                                   andRemoteSensing,63:1–15,2025. doi: 10.1109/TGRS.
                                                                            2025.3570954.
Chen, L., Letu, H., Fan, M., Shang, H., Tao, J., Wu, L.,                 Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and
   Zhang,Y.,Yu,C.,Gu,J.,Zhang,N.,etal.Anintroduction                        Le, M. Flow matching for generative modeling. arXiv
   to the chinese high-resolution earth observation system:                 preprint arXiv:2210.02747, 2022.
   Gaofen-1˜7civiliansatellites.JournalofRemoteSensing,
   2022.                                                                 Liu, X., Gong, C., and Liu, Q.   Flow straight and fast:
Dian, R., Wang, N., Guo, A., and Li, S.   Equivariant                       Learning to generateand transfer data with rectifiedflow.
   bayesian hyperspectral imaging via mosaiced and pan                      arXiv preprint arXiv:2209.03003, 2022.
   image fusion.   IEEE Transactions on Pattern Analy-                   Liu, Y., Liu, J., Dian, R., and Li, S. A selective re-learning
   sis and Machine Intelligence,  pp. 1–16,  2026.    doi:                  mechanism for hyperspectral fusion imaging.  In 2025
  10.1109/TPAMI.2026.3682543.                                               IEEE/CVF Conference on Computer Vision and Pattern

Diao,  W.,  Hu,  L.,  Zhang,  K.,  and  Xiao,  L.     Dual-                 Recognition (CVPR), pp. 7437–7446. IEEE, 2025.
   conditionally guided diffusion models for fusion of un-
   registered multisource remote sensing images.   IEEE                  Lowe,D.G. Distinctiveimagefeaturesfromscale-invariant
  Transactions onGeoscience andRemote Sensing, 63:1–                        keypoints. Internationaljournal ofcomputer vision, 60
  17, 2025. doi: 10.1109/TGRS.2025.3577073.                                 (2):91–110, 2004.

Dong, W., Zhou, C., Wu, F., Wu, J., Shi, G., and Li, X.                  Lu, B., Dao, P. D., Liu, J., He, Y., and Shang, J.  Recent
   Model-guideddeephyperspectralimagesuper-resolution.                      advances of hyperspectral imaging technology and ap-
  IEEETransactionsonImageProcessing,30:5754–5768,                           plications in agriculture. Remote Sensing, 12(16):2659,
   2021.                                                                    2020.

                                                                     10

<!-- PDF PAGE 11 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

Luo,S.,Qian,Y.,Bai,L.,Fan,Y.,Wang,Y.,andKong,W.                         Sahoo, R. N., Ray, S., and Manjunath, K.  Hyperspectral
  Deep learning-based hyperspectral and multispectral fu-                  remotesensingofagriculture. Currentscience,pp.848–
  siontechniques: Review,optimization,andperspectives.                     859, 2015.
  Information Fusion, pp. 103291, 2025.                                 Song, J., Meng, C., and Ermon, S.  Denoising diffusion

Ma, Q., Jiang, J., Liu, X., and Ma, J.  Reciprocal trans-                  implicitmodels. arXivpreprintarXiv:2010.02502,2020.
  formerforhyperspectralandmultispectralimagefusion.                    Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consis-
  Information Fusion, 104:102148, 2024.                                    tency models. 2023.
Mao, Z., Chimitt, N., and Chan, S. H. Accelerating atmo-                Stuart,M.B.,McGonigle,A.J.,andWillmott,J.R. Hyper-
  sphericturbulencesimulationvialearnedphase-to-space                      spectral imaging in environmental monitoring: A review
  transform. InProceedingsoftheIEEE/CVFinternational                       of recent developments and technological advances in
  conference on computer vision, pp. 14759–14768, 2021.                    compact fielddeployable systems. Sensors, 19(14):3071,

Peyghambari, S. and Zhang, Y. Hyperspectral remote sens-                   2019.
  ing in lithological mapping, mineral exploration, and                 Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y.,
  environmental geology: an updated review. Journal of                     Rector-Brooks, J., Wolf, G., and Bengio, Y. Improving
  Applied Remote Sensing, 15(3):031501–031501, 2021.                       andgeneralizingflow-basedgenerativemodelswithmini-
Pour, A. B., Zoheir, B., Pradhan, B., and Hashim, M. Edito-                batchoptimaltransport.arXivpreprintarXiv:2302.00482,
  rialforthespecialissue: Multispectralandhyperspectral                    2023.
  remotesensingdataformineralexplorationandenviron-
  mental monitoring of mined areas, 2021.                               Wald, L. Quality ofhighresolution synthesisedimages: Is
                                                                           there a simple criterion?  In Third conference” Fusion
Pu,H.,Wei,Q.,andSun,D.-W. Recentadvancesinmuscle                           ofEarthdata: mergingpointmeasurements,rastermaps
  foodsafetyevaluation: Hyperspectralimaginganalyses                       andremotelysensedimages”,pp.99–103.SEE/URISCA,
  andapplications. CriticalReviewsinFoodScienceand                         2000.
  Nutrition, 63(10):1297–1313, 2023.
Qu, J., Cui, J., Dong, W., Du, Q., Wu, X., Xiao, S., and                Wald, L., Ranchin, T., and Mangolini, M. Fusion of satel-
  Li, Y.  A principle design of registration-fusion consis-                lite imagesof different spatial resolutions: Assessing the
  tency: Towardinterpretabledeepunregisteredhyperspec-                     quality ofresulting images. Photogrammetric engineer-
                                                                           ing and remote sensing, 63(6):691–699, 1997.
  tralimagefusion. IEEETransactionsonNeuralNetworks
  and Learning Systems, 36(5):9648–9662, 2025a.  doi:                   Wang,Z., Bovik, A.C., Sheikh,H.R., andSimoncelli, E.P.
  10.1109/TNNLS.2024.3412528.                                              Imagequality assessment: from errorvisibility tostruc-
                                                                           tural similarity. IEEE transactions on image processing,
Qu, J., He,J., Li, Y., Dong, W., and Liu, S. Progressive syn-              13(4):600–612, 2004.
  ergisticregistrationandfusiondiffusionnetworkforun-
  registered hyperspectral and multispectral image fusion.              Ying, J.,Shen, H.-L., and Cao,S.-Y. Unalignedhyperspec-
  IEEE Transactions on Geoscience and Remote Sensing,                      tralimagefusionviaregistrationandinterpolationmodel-
  63:1–14, 2025b. doi: 10.1109/TGRS.2025.3564055.                          ing. IEEETransactionsonGeoscienceandRemoteSens-
Qu,J.,Wu,X.,Dong,W.,Cui,J.,andLi,Y. Ir&arf: Toward                         ing, 60:1–14, 2022. doi: 10.1109/TGRS.2021.3081136.

  deep interpretable arbitrary resolution fusion of unreg-              Yuhas, R. H., Goetz, A. F., and Boardman, J. W. Discrim-
  istered hyperspectral and multispectral images.  IEEE                    ination among semi-arid landscape endmembers using
  TransactionsonImageProcessing,34:1934–1949,2025c.                        the spectral angle mapper (sam) algorithm. In JPL, Sum-
  doi: 10.1109/TIP.2025.3551531.                                           maries of the Third Annual JPL Airborne Geoscience
Qu,Y.,Qi,H.,Kwan,C.,Yokoya,N.,andChanussot,J. Un-                          Workshop. Volume 1: AVIRIS Workshop, 1992.

  supervised and unregistered hyperspectral image super-                Zhang,X.,Huang,W.,Wang,Q.,andLi,X.Ssr-net: Spatial–
  resolution with mutual dirichlet-net. IEEE Transactions                  spectral reconstruction network for hyperspectral and
  on Geoscience and Remote Sensing, 60:1–18, 2021.                         multispectralimagefusion. IEEETransactionsonGeo-
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and                     science and Remote Sensing, 59(7):5953–5965, 2020.

  Ommer, B. High-resolution imagesynthesis withlatent                   Zheng, K., Gao, L., Hong, D., Zhang, B., and Chanussot, J.
  diffusion models. In Proceedings of the IEEE/CVFcon-                     Nonregsrnet: Anonrigidregistrationhyperspectralsuper-
  ferenceoncomputervisionandpatternrecognition,pp.                         resolution network. IEEE Transactions on Geoscience
  10684–10695, 2022.                                                       and Remote Sensing, 60:1–16, 2021.

                                                                    11

<!-- PDF PAGE 12 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

Zhou, Y., Rangarajan, A., and Gader, P. D. Nonrigid reg-
  istration of hyperspectral and color images with vastly
  differentspatialandspectralresolutionsforspectralun-
  mixingandpansharpening. InProceedingsoftheIEEE
  Conference on Computer Visionand Pattern Recognition
  (CVPR) Workshops, July 2017.

Zhou, Y., Rangarajan, A., and Gader, P. D. An integrated
  approach to registration and fusion of hyperspectral and
  multispectralimages. IEEETransactions onGeoscience
  and Remote Sensing, 58(5):3020–3033, 2020.  doi: 10.
  1109/TGRS.2019.2946803.

Zhu, C., Deng, S., Zhou, Y., Deng, L.-J., and Wu, Q. Qis-
  gan: A lightweight adversarial network with quadtree im-
  plicitsamplingformultispectralandhyperspectralimage
  fusion. IEEE Transactions on Geoscience and Remote
  Sensing, 61:1–15, 2023.













































                                                                     12

<!-- PDF PAGE 13 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images
A. Appendix Outline
Inthissupplementarymaterial,weprovidemoredetailsandcomprehensiveexperimentalresultsofourproposedStructure-
Aware Riemannian Flow Matching (SA-RFM) framework as follows:

   •Sec.Bpresents more theoretical derivations and proofs.

   • Sec.Cprovides a comprehensive review of Related Work, covering traditional paradigms, deep learning-based
      approaches, and the theoretical foundations of Riemannian one-step flow models.
   •Sec.Dintroduces the benchmarks used for comparison in our experiments.

   •Sec.Edefines the fusion metrics employed to quantitatively assess the registration and fusion performance.

   •Sec.Fdescribes the details of the datasets.

   • Sec.Greports more quantitative and visualization results to further demonstrate the generalization capability and
      robustness of our method.

   • Sec.Hpresents robustness experiments under realistic degradation scenarios, such as noise, artifacts, and spectral
      variations.

   •Sec.Iconducts a theoretical complexity analysis of the proposed network, including both time and space complexity.

B. Additional theoretical supplements and explanations
B.1. Derivation of Eq. (9)   .
Eq. (9)    is obtained by combining the standard dynamic optimal transport formulation in Eq. (2)    with the structure-induced
Riemannian metric defined in Eq.  (7)   .  Specifically, the standard dynamic optimal transport objective minimizes the
Euclidean kinetic energy:                                  Z 1  Z
                                                       infvt         1
                                                             0    Ω  2∥vt(u)∥22ρt(u)dudt.                                                               (17)

To incorporate the structural prior from the reference image, we equip the spatial domain Ω with a position-dependent
Riemannian metric tensor:
                                                              M(u) =I +βSY(u).                                                                         (18)
Accordingly, the Euclidean norm of the velocity field is replaced by the squared Riemannian norm:

                                                      ∥vt(u)∥2M(u) =vt(u)⊤M(u)vt(u).                                                              (19)

Substituting this metric-induced kinetic energy into the dynamic optimal transport objective gives:
                                                            Z 1  Z   1
                                           J(vt) = infvt             2vt(u)⊤M(u)vt(u)ρt(u)dudt.                                                (20)
                                                             0    Ω
This structure-aware formulation penalizes transport across strong structural boundaries, whereM(u) has large eigenvalues,
and thus encourages the learned transport trajectory to align with the underlying image geometry.

B.2. Proof of the closed-form solution for Eq. (13)      and Eq. (14)
Eq.  (13) is a quadratic program with a linear constraint:

                                                minv   v⊤M(u)v,    s.t.   d(u)⊤v =∥d(u)∥2.                                                       (21)


We construct the LagrangianL(v,λ) as

                                              L(v,λ) =v⊤M(u)v+λ ∥d(u)∥2−d(u)⊤v.                                                    (22)

                                                                           13

<!-- PDF PAGE 14 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

Setting its gradient with respect tov to zero gives

                                                         ∇vL = 2M(u)v−λd(u) = 0.                                                                  (23)

Using the invertibility ofM(u), we obtain
                                                               v = λ2M(u)−1d(u).                                                                          (24)

Substituting this expression forv into the original constraintd(u)⊤v =∥d(u)∥2 allows us to solve for the scalarλ/2:
                               λ   d(u)⊤M(u)−1d(u) =∥d(u)∥2    =⇒    λ
                               2                                                          2 =       ∥d(u)∥2d(u)⊤M(u)−1d(u).                                (25)

Pluggingλ/2 back into the expression forv yields the exact closed-form solution (Eq. (14)):

                                                    v†(u) =     M(u)−1d(u)d(u)⊤M(u)−1d(u)∥d(u)∥2.                                                            (26)


B.3. Rationale for Using OTD Regularization Instead of Fully Riemannian Flow Matching
While a fully Riemannian Flow Matching (RFM) framework following exact geodesics is mathematically elegant, its
application to high-dimensional image registration faces a severe computational bottleneck.
Calculating exact geodesics requires solving the second-order ODE

                                                                 ¨γk +Γkij˙γi˙γj = 0,                                                                           (27)

whichdepends onthe ChristoffelsymbolsΓkij derivedfrom thespatial derivatives ofthe metricM(u). Since ourstructure-
inducedmetricishighlydata-dependentandderivedfromdiscretehigh-resolutiongradients,solvingthisglobalBoundary
Value Problem (BVP) across millions of pixels at every training iteration is computationally intractable.
OurOTDregularizationisexplicitlydesignedasacomputationallyefficientsurrogateforthesegeodesics. Insteadofan
expensiveglobalBVP,OTDperformsalocal,first-ordergeometriccorrection. ByanalyticallysolvingEq.(13)     ,theresulting
v†(u) acts as a tangent approximation: it maintains the global Euclidean direction to ensure convergence, while its local
trajectory is warped byM(u)−1 to favor minimum-cost structural paths.

B.4. Exact Formulation of the Structure Tensor
We compute the MSI grayscale gradients∇I = [Ix,Iy]⊤ and the base tensor

                                                                   J0 =∇I∇I⊤.                                                                               (28)

We extract its principal eigenvalueλ1 and unit eigenvectore1 = [ex,ey]⊤, with the orthogonal vectore2 = [−ey,ex]⊤.
To penalize transport across strong edges, we non-linearly amplify the principal eigenvalue as

                                                                   λ′1 = 1+αλk1,                                                                              (29)

whereα = 10 andk = 2, while keepingλ′2 = 1. The exact metric tensor used in our code is:
                                                                             λ′1e2x+e2y      (λ′1−1)exey
                                     M(u) =λ′1e1e⊤1 +e2e⊤2  =              (λ′1−1)exey     λ′1e2y +e2x             .                                        (30)


B.5. Theoretical bounds and approximation error for the simplified Riemannian energy
By freezing the metric at u and assuming a straight-line path for displacement ∆u, we perform a first-order Taylor
approximation of the Riemannian energy. The approximation error is bounded by:

                                                       Error≤O          ∥∇M(u)∥∥∆u∥3             .                                                               (31)

                                                                           14

<!-- PDF PAGE 15 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

Since M(u) = I +βSY(u) (Eq.  (7)   ), this bound is directly constrained by the smoothness, i.e., the Hessian, of the
multispectral image structure tensorSY(u).
A straight-line assumption for large, single-step displacements may cause the approximation error to blow up in other
approaches. However, our approach avoids this by operating within a continuous dynamic Flow Matching ODE paradigm.
Rather than a single static leap, the total deformation is governed by

                                                                 dst
                                                                 dt  =vt(st).                                                                               (32)

Duringinference,theODEsolverbreakstheglobaldeformationintodiscrete,infinitesimalintegrationstepsdt,wherethe
instantaneous displacement is exactly
                                                                  ∆u =vtdt.                                                                                 (33)

Since a smalldt keeps the localdisplacement∥∆u∥ minimal, our quadratic approximation becomesasymptotically exact as
dt→ 0. Thus,theintegrationprocesspreventsprohibitiveerroraccumulationevenunderextrememacroscopicdeformations.
Furthermore, performance can be improved by increasing the number of steps, rather than being restricted to one-step
inference.

C. Related Work
C.1. Traditional Paradigms for Hyperspectral and Multispectral Image Registration
Traditionally, the fusion of hyperspectral images (HSIs) and multispectral images (MSIs) has predominantly relied on
a two-stage paradigm, where geometric alignment is treated as a prerequisite preprocessing step decoupled from the
subsequent fusion task(Ying et al.,2022;Zhou et al.,2020). To address the spatial misalignment between LR-HSIs and
HR-MSIs, early approaches largely relied on feature-based techniques, extracting distinct keypoints via Scale-Invariant
Fourier Transform (SIFT) or Harris detectors, and subsequently approximating non-rigid geometric distortions through
Thin-PlateSplines(TPS)orGaussianradialbasisfunctions. Parallelly,area-basedstrategieshavebeenemployedtooptimize
similaritymetricsacrossmodalities;forinstance,Zhouetal. proposedminimizingaleast-squaresobjectivefunctionbased
on the Point Spread Function (PSF)(Zhou et al.,2017), while Ying et al. introduced the Normalized Edge Difference (NED)
toalign imagesbyminimizingedge discrepancies(Yinget al.,2022). However, thisdecoupledstrategyignores theintrinsic
correlation between registration and fusion, inevitably leading to error propagation where misregistration artifacts cannot
berectifiedduringthefusionstage. Consequently,theseconventionalmethodsexhibitlimitedrobustnesswhenhandling
complex non-rigid deformations induced by sensor dynamics or terrain irregularities in remote sensing scenarios.

C.2. Deep Learning-Based Joint Registration and Fusion for Hyperspectral and Multispectral Images
The advent of deep learning has precipitated a paradigmatic shift in HSI processing from independent preprocessing
stepsto end-to-endjointoptimization frameworks. Initialefforts adaptedSpatialTransformerNetworks(STN)(Jaderberg
et al.,2015) and unsupervised deformation fields akin to VoxelMorph(Balakrishnan et al.,2019) to estimate affine or
non-rigid transformations.  To circumvent the ”error accumulation” inherent in cascaded pipelines, diverse strategies
have emerged.  For instance, u2-MDN(Qu et al.,2021) circumvents explicit registration by disentangling the feature
extraction process, specifically deriving spectral representations from HSIs and spatial details from MSIs, respectively.
However,themainstream hasevolvedtowardsunified architectureswhereregistrationand fusionaremutually reinforcing.
Zheng et al.  pioneered this with NonRegSRNet, integrating dense registration and fusion into a unified CNN-based
model(Zheng et al.,2021). While effective, CNN-based methods are often constrained by local receptive fields. Addressing
this, Transformer-based architectures have been introduced to capture long-range dependencies, such as SCANet which
simulatesstereo-visionprocesses(Guoetal.,2023),andM2DTNwhichleveragesdeformableattentiontoalignmulti-scale
features(Dongetal.,2024). Furthermore,tomitigatethecomputationalburdenofself-attentionmechanisms,recentstudies
have leveraged Mamba-based architectures to achieve efficient progressive registration-fusion co-optimization(Li et al.,
2025).  Parallel to these architecture innovations, model-driven networks have gained prominence by mathematically
unfolding the physical registration-fusion process into deep layers, thereby enhancing interpretability. Notable approaches
includeintegratingphysicalperceptionmodelsforregistration-fusionconsistency(Quetal.,2025a)orconstructingarbitrary-
resolution observation models(Qu et al.,2025c).  The aforementioned schemes are typically two-stage, assigning the
registration andfusion tasksto separatenetworks; consequently, theyare proneto introducingcross-task errors duringthe

                                                                       15

<!-- PDF PAGE 16 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

optimizationprocess. Mostrecently,GenerativeAIhasalsobeenadaptedforthistask,wherediffusionmodelsareemployed
to model the joint distribution of registration and fusion in a probabilistic manner(Qu et al.,2025b;Diao et al.,2025).

C.3. Riemannian Manifold Learning and One-Step Flow Generative Models
While diffusion models have shown promise in joint distribution modeling, their reliance on flat Euclidean metrics
and computationally intensive iterative sampling necessitates a shift toward geometry-aware and efficient generative
paradigms. From a geometric standpoint, hyperspectral data intrinsically resides on non-linear manifolds where Euclidean
measures fail to capturegeodesic distances essential for accuratedeformation modeling. Thishas motivatedthe adoption
of Riemannian geometry, tracing back to variational methods like LDDMM(Beg et al.,2005) that ensure topological
preservation, and evolving into modern geometric deep learning that learns structure-aware metric tensors to guide image
restoration(Arvanitidisetal.,2017). Complementingthisgeometricperspective,FlowMatching(FM)andOptimalTransport
(OT)haveemergedtoaddressthesamplinginefficiencyofstandarddiffusion. Unlikenoise-driventrajectories,FMconstructs
deterministic Ordinary Differential Equation (ODE) paths that minimize transport costs(Lipman et al.,2022;Tong et al.,
2023).
To further bridge the gap between high-fidelity generation and real-time inference, recent advancements in one-step
generation,exemplifiedbyRectifiedFlow(Liuetal.,2022)andConsistencyModels(Songetal.,2023),haveenabledthe
distillation of complex ODE trajectories into a single forward pass. These methods straighten the transport paths in the
probability space, allowing for efficient coupling between source and target distributions.  However, existing one-step
frameworks predominantly operate under isotropic Euclidean assumptions. Integrating Riemannian metric constraints
into one-stepflow matchingfor the specificchallenge ofunregistered HSI-MSIfusion remains anunderexplored frontier,
promising to combine the geometric fidelity of manifold learning with the computational efficiency of rectified flows.

D. The Benchmarks
IntegratedproposesajointoptimizationframeworkwhereregistrationutilizesB-splinefree-formdeformationandfusionis
solvedviaaclosed-formsolutionderivedfromamanifold-invariantassumption,effectivelyhandlingrigidandnon-rigid
distortions.
MoGDCN unfolds an iterative optimization algorithm into a model-guided deep network. By explicitly integrating the HSI
observation model and a learned spatial-spectral denoiser, it combines the interpretability of model-based methods with the
learning capability of CNNs to reconstruct HR-HSI.
NonRegSR introduces a novel unsupervised framework composed of a spatial transformation network and a triplet
architecture. Itsimultaneouslyperformsnon-rigidregistrationandspectralunmixing-basedsuper-resolutionwithoutrelying
on ground-truth labels, using a multitask learning strategy to ensure the two tasks complement each other.
M2DTNaddressesthelimitationofCNNsincapturingglobalfeaturesbyproposingaMulti-ModalCross-ScaleDeformable
Transformer.  Built upon a Spectrum-Preserving Spatial-Guided Registration-SR Unified (SSRU) model, it leverages
deformable cross-attention to align features across scales, effectively correcting distortions while preserving spectral fidelity
in unregistered scenarios.
MoEPNPpresentsatheoreticallygroundedapproachbydesigningaRegistration-FusionConsistencyModel(RFCM)that
mathematically couples the two tasks to reduce sensitivity to registration errors. To solve this, it employs a Mixture-of-
Experts(MoE)Plug-and-Playframework,whereanMoE-basedhyper-solverdynamicallyadjustsparameters,ensuringhigh
interpretability and robust performance against misalignments.
IR&ArFtargetsthechallengingproblemofarbitraryresolutionfusion. ItconstructsaSpatial-SpectralConsistentArbitrary
ScaleObservationModelandunfoldsitsoptimizationintoadeepinterpretablenetwork. Thisdesignallowsformodel-data
dual-drivenfusion, enablingthe generationof high-qualityHSIs atcontinuous resolutionscales evenunder unregistered
conditions.
PSRFDifleveragesthegenerativepowerofdiffusionmodelsthroughaProgressiveSynergisticRegistrationandFusion
Diffusion Network. It introduces a Fine Registration and Fusion Bridging (FRFB) model to guide the reverse diffusion
process, iteratively co-optimizing registration and fusion to handle complex, non-linear misalignments that traditional
methods fail to capture.


                                                                      16

<!-- PDF PAGE 17 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

RFDif integrates registrationand fusioninto aunified dual-conditionaldiffusion framework. Itestablishes aclosed-loop
mechanismwheretheregistrationmodulecorrectsmisalignmenttoguidefusion,andthefusedresultfeedbackstorefine
registration, allowing the two tasks to mutually enhance each other iteratively.
PRFCo explores the efficiency of State Space Models by utilizing an A-Mamba network. It employs a progressive co-
optimizationstrategywithaModal-AlignedProgressiveRegistration-Fusion(MAPRF)module,whichcorrectsdeformations
from coarse to fine levels while refining fusion results, offering a computationally efficient alternative to Transformer-based
methods.

E. Fusion Metrics
Four widely used fusion metrics are used to evaluate the performance of each method.
(1) Peak Signal to Noise Ratio: The PSNR is one of the most commonly used indicators to measure image quality (Hore &
Ziou,2010):                                                                                     !
                                               PSNR = 10·log10                max(Rc)2                                                       (34)
                                                                            1
                                                                           HW ∥Rc−Zc∥22
whereRc andZc represent thecth channel of the reference image and the predicted image, respectively.∥·∥2 means the
secondnorm. After addingupthePSNRs ofallchannels,we obtainthePSNRmetric ofthewholeimage. The largerthe
PSNR, the better the performance.
(2) Structural Similarity Index Measure: The SSIM is an indicator used to measure the degree of similarity between two
digital images (Wang et al.,2004). Structural similarity aligns more closely with the human visual perception of image
quality, providing a metric that better reflects the discernment of human eyes. The SSIM is formulated as


                                              SSIM = [l(R,Z)]α·[c(R,Z)]β·[s(R,Z)]γ                                                       (35)

                                                         l(R,Z) =  2µRµZ+C1µ2R+µ2Z+C1                                                                      (36)


                                                        c(R,Z) =  2σRσZ+C2σ2R+σ2Z+C2                                                                      (37)

                                                          s(R,Z) =  σRZ+C3σRσZ+C3                                                                        (38)

wherel, c, ands represent a comparison of the brightness, contrast, and structure between the reference image and the
predictedimage,respectively. α,β,γ areweightingcoefficients,typicallysetto1.µR andµZ denotetheaveragevalueof
R andZ,whileσR andσZ denotethestandard deviation. σRZ denotesthe covarianceofR andZ. Thehigherthe SSIM,
the better the performance.
(3) Spectral Angle Mapper: The SAM is a spectral angle mapping thattreats the spectrum of each pixel in an imageas a
high-dimensionalvector(Yuhasetal.,1992). Thesimilarity betweenspectraismeasured bycalculatingtheangle between
the two vectors. The SAM is defined as
                                                                          PC                   
                                               SAM = cos−1       qPC         c=1rczcqPC                                                                (39)

                                                                          c=1r2c        c=1z2c
whererc andzc refertothespectralvectorsofthereferenceandpredictedHR-HSIs. ThesmallertheSAM,themoresimilar
the two spectra.
(4) RelativeDimensionless Global Errorin Synthesis: ERGASis an indicatorused toevaluatethe qualityof remote sensing
images(Wald,2000);itconsidersthemeansquareerrorandbrightnessinformationoftheimagetoprovideacomprehensive
evaluation of the performance of the image. The definition formula is as follows:
                                                                     vuut
                                                                         1   CX  ∥Rc−Zc∥2
                                                 ERGAS = 100                                   2
                                                                  r     C   c=1    µ2(Rc)                                                               (40)

                                                                       17

<!-- PDF PAGE 18 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

wherer refersto thespatialdownsamplingratio whenweobtain theLR-HSI.µ(Rc) representsthe meanvalueof thecth
channel of the reference image. The lower the ERGAS value, the higher the image quality.
(5) Mean Relative Absolute Error: The MRAE measures the relative deviation between the reconstructed image and the
groundtruth,providinganassessmentofspectralfidelityrelativetothesignalintensity(Arad&Ben-Shahar,2016). Itis
defined as the mean of the absolute differences normalized by the ground truth values. The MRAE is formulated as

                                                                        NX  |Ri−Zi|
                                                      MRAE =  1N                 Ri                                                                          (41)
                                                                        i=1

whereN denotes the total number of pixels in the image. Ri andZi represent the spectral value of thei-th pixel in the
referenceimageandthe predictedimage,respectively. Thesmaller theMRAE,thebettertheperformance. Thismetricis
only used in visual results.

F. Datasets
Here, we provide detailed descriptions of the datasets used in the main text, as well as those used in the supplementary
experiments. All the datasets can be downloaded online.
(1) Pavia University: The Pavia University dataset was captured by ROSIS sensors over Pavia University in Italy. After
removingsampleswithoutanyinformation,thedatasetconsistsof610×340pixelsand103spectralbands,withwavelengths
ranging from 0.43 to 0.86µm at a spatial resolution of 1.3 m.
(2) Botswana: The Botswana datasetcontains a sequenceof data capturedby the Hyperion sensoron the EO-1satellite over
OkavangoDelta,Botswana. Afterremovingbandsthatcoverthewaterabsorptioncharacteristics,thedatasetconsistsof
1476× 256 pixels and 145 bands, with spectral wavelengths ranging from 0.4 to 2.5µm at a spatial resolution of 30 m.
(3)Chikusei: TheChikuseidatasetwasacquiredbytheHeadwallHyperspec-VNIR-Cimagingsensorovertheagricultural
andurban areasofChikusei, Ibaraki,Japan. Thisdataset featuresalargescenesize of2517×2335 pixels withaspatial
resolution of 2.5-m. It covers 128 spectral bands across the wavelength range of0.36–1.02µm.
(4) Xiongan: The Xiongan dataset was captured by an airborne sensor over the Xiongan New Area, China. This high-
resolution dataset includes3750×1580 pixels with a fine spatial resolution of 0.5-m. It spans 256 spectral bands ranging
from0.39to1.00µm.

G. Additional Quantitative and Visual Results of Comparative Experiments
As the detailed analysis of the Pavia University dataset is presented in the main manuscript, this section focuses on the
evaluations of the Botswana, Chikusei, and Xiongan datasets under three distinct deformation types: Elastic, Homography,
andTurbulence. Table2,Table9,andTable3presentthecomprehensiveresults,wherethebest andsecond-bestresultsare
highlightedin boldfaceandunderlined,respectively. Inthe quantitativeanalysisresults,the traditionalmethodIntegrated
exhibitsnotablestabilityacrossvaryingscenariosdespitenotachievingtop-tiernumericalscores. Amongdeeplearning-
basedapproaches,IR&ArFdemonstratesconsistentlyfavorableperformanceacrossallthreedatasets. However,generative
models exhibit inconsistent performance; specifically, RFDif performs exceptionally well on the Botswana and Xiongan
datasets but degrades noticeably on the Chikusei dataset, indicating insufficient robustness and limited generalization
capability. Conversely, PSRFDif, while delivering less outstanding peak performance, maintains a more stable output
comparedtoRFDif. Furthermore,whilePRFCoachievescompetitiverestorationquality,itincursasubstantialcomputational
burden. Insharpcontrast,ourproposedframework,designedbasedonalightweightCNNarchitecture,requirestheminimal
computational overhead (1.203 M parameters). However, attributed to our carefully designed optimization objective, it
achieves optimal outcomesacross almost all metrics (e.g., PSNR,SSIM, and SAM), confirming its superiorefficiency and
generalization capability compared to both traditional and state-of-the-art deep learning-based methods.
Additionally, we conduct qualitative analysis to visually assess the registration and fusion quality. The visual comparisons
on the Pavia University, Botswana, Chikusei, and Xiongan datasets are illustrated in Figures7,8,9, and10. From the
visual comparisons, particularly on the Pavia University and Botswana datasets, it is evident that competitive methods either
suffer severe ghosting artifacts caused by misregistration, such as Integrated and NonRegSR, or exhibit pronounced spectral
distortions, as seen with PSRFDif and RFDif. On the Chikusei and Xiongan datasets, even though the visual disparities

                                                                      18

<!-- PDF PAGE 19 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

Table 9. Quantitative comparison on theChikuseidataset underElastic, Homography, andTurbulence distortions. Bold indicates best,
Underline   indicates second-best. The gray column highlights our proposed method.

                                             MoGDCN       NonRegSR       M2DTN        MoEPNP       IR&ArF       PSRFDif        RFDif        PRFCo
 DISTORTION  METRIC     Integrated(TGRS’20)   (TIP’21)    (TGRS’22)     (AAAI’24)   (TNNLS’24)     (TIP’25)    (TGRS’25)     (TGRS’25)    (TGRS’25)       Ours

                   PSNR↑      38.65        38.74        38.25        38.54        39.74        40.77             38.28        39.57        39.74         41.77
    Elastic         SSIM↑       0.982        0.983        0.980        0.983        0.987        0.989           0.983        0.988        0.988         0.991
     (α=600,        SAM↓       3.157        3.282        3.555        3.171        2.549        2.311            3.320        2.494        2.378         2.153
     σ=15)        ERGAS↓     1.378        1.365        1.438        1.388        1.208        1.076              1.436        1.230        1.208         0.962

                   RMSE↓      2.979        2.948        3.118        3.017        2.627        2.334             3.107        2.680        2.626         2.080
                   PSNR↑      38.34        38.21        38.58        37.02        39.42        38.47        38.53        39.54        42.33              41.21
 Homography         SSIM↑       0.982        0.983        0.982        0.975        0.986        0.983        0.983        0.988            0.983        0.991
  (MPO=10%)         SAM↓       3.158        3.352        3.379        4.143        2.759        3.189        3.292        2.515        1.758             2.232
                  ERGAS↓     1.427        1.444        1.387        1.652        1.256        1.398        1.396        1.234        0.972               1.026
                   RMSE↓      3.087        3.134        3.003        3.595        2.727        3.042        3.021        2.689        1.951              2.218
                   PSNR↑      40.20        40.50        39.70        39.71        41.07        42.74        39.38        42.79              41.13        43.61
  Turbulence        SSIM↑       0.984        0.985        0.983        0.986        0.987        0.991           0.984        0.991         0.989        0.993
   (D/r0=2.0)       SAM↓       2.846        2.855        3.196        2.742        2.506        2.045        3.140        1.964             2.205        1.735
                  ERGAS↓     1.156        1.110        1.221        1.209        1.042        0.867        1.267        0.858               1.034        0.782
                   RMSE↓      2.492        2.408        2.641        2.636        2.256        1.860        2.740        1.849              2.239        1.683

Table 10. Performance evaluation andsensitivity analysison the Pavia University dataset underelastic distortion. Weevaluatethe impact
ofspectralvariationsandocclusionsinLR-HSI,alongsidevariousMSI-sidedistortionssuchasadditivenoise,spatialmisregistration, and
imaging artifacts.

 Setting                            PSNR↑   SSIM↑   SAM↓   ERGAS↓   RMSE↓   Experimental Description

 Ours (Baseline)                42.69      0.987      1.735        0.931         1.872     Standard Elastic (α = 600,σ = 15)
 Registered LR-HSI          42.97      0.988      1.689        0.898         1.811     ResultsofundistortedLR-HSIandHR-MSI
 Large Distortion              42.45      0.987      1.773        0.955         1.924     Strong Elastic (α = 1000,σ = 15)
 HSI Occlusion                 41.40      0.986      1.933        1.066         2.169     Robustness against partial HSI occlusion
 HSI Spectral Variation     38.59      0.958      2.037        1.491         3.000     Impact of HSI spectral signatures variation
 MSI Noise Injection        40.97      0.982      2.221        1.124         2.281     MSI with additive Gaussian noise
 MSI Misalignment           28.69      0.830      9.536        4.526         9.375     Effect of spatial misregistration in MSI
 MSI Artifact                    36.72      0.965      3.604        1.813         3.719     Effect of imaging artifacts in MSI


betweenstate-of-the-artmethodsandoursappearrelativelysubtle,ourmethodmaintainsadistinctadvantageinquantitative
fidelity.  Compared with these CNN and Diffusion-based methods, our method models the deformation flow based on
structure-induced Riemanniangeometry, which offers significantadvantages in respecting object boundaries and ensuring
pixel-level accuracy during transport. Overall, our method demonstrates outstanding performance in both quantitative and
qualitative experiments across diverse datasets, indicating its strong generalizability.

H. Robustness Experiments on LR-HSI and HR-MSI

Inpracticalapplications,LR-HSIandHR-MSIareoftenaffectedby variousenvironmentalnoises, artifacts,andspectral
distortions. To evaluate the robustness of our method under these realistic challenges, we design a series of degradation
experiments onthe PaviaUniversity dataset,targeting both LR-HSIand HR-MSI. Thevisual results areshown in Figure6,
andthequantitativeresultsarereportedinTable10. Inthetable,weadditionallyincludeabaselineexperimentconductedon
thealreadyregisteredLR-HSI.Itcanbeobservedthatthefusionperformanceisthebestwhennodegradationisintroduced.
Moreover, the fourth row of Table10shows the influence of stronger elastic distortion. The results indicate that, as the
distortion level increases, the degradation in fusion performance is not significant. Notably, we observe that introducing

                                                                                19

<!-- PDF PAGE 20 -->

     Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images

                   GT                         Occlusion                     Spectral                         Noise                      Misaligned                    Artifact






















Figure 6. Visual comparison of the fusion results on the Pavia University dataset under various degradation scenarios before elastic
distortion (α = 600,σ = 15). The toprowillustrates the simulatedlow-quality inputs: OcclusionandSpectralvariations are applied
to theoriginal LR-HSI, whileNoise, Misaligned, and Artifact interferences areintroduced to theMSI. (Note thatelastic distortionis
applied after degradations.) The middle row displays the reconstructed HR-HSI (visualized as RGB), and the bottom row shows the
corresponding difference maps between the reconstructed results and the reference Ground Truth.


degradations such as noise, misalignment, and artifacts into HR-MSI leads to a more pronounced performance drop. This
suggeststhatourmethodreliesonthequalityofHR-MSI,whichisconsistentwithourstructuretensordesignbasedonthe
texture details of HR-MSI and therefore matches our experimental expectation. In addition, spectral variation in LR-HSI
alsocauses arelativelylargeimpact,indicating thatourmethod dependslesson thespatialquality ofLR-HSIandmore on
the spectral quality of LR-HSI and the spatial quality of HR-MSI.

I. Complexity Analysis
Firstly for convenience, we setH andW to denote the spatial dimensions of the high-resolution image,C the number of
spectral bands of the hyperspectral image, andc the number of spectral bands of the multispectral image, where typically
c≪C. We set theD as the hidden layer dimensions,K as the number of steps andk as the kernel size of convolutions (a
littledifferentfromthemaintext). Thetimecomplexityofconstructingthestructure-inducedRiemanniangeometryinvolves
gradientcalculationandtensorsmoothingontheMSI,whichisO(HWc). ThetimecomplexityoftheStructure-AwareFlow
Matchingnetworkisdominatedbythefeatureencodingandfusionmodules. Specifically,thespectralandauxiliaryencoders
involve convolutions with complexity O(HW(CD +cD)k2), and the lightweight fusion module with point-wise and
depth-wiseconvolutionshasacomplexityofapproximatelyO(HWD2). ThecalculationoftheRiemannianFlowMatching
loss and structure-aware optimal transport direction requires pixel-wise matrix operations and inversions over the spectral
vectors,witha timecomplexityofO(HWC). Therefore,the totaltimecomplexityof ourSA-RFMtraining periterationis
O HW(CD+D2). Forinference,givenK stepsoftheODEsolver,thetotaltimecomplexityisO KHW(CD+D2).
TheinputoftheframeworkincludestheLR-HSI,HR-MSI,andtheconstructedRiemannianmetrictensorM ∈RH×W×2×2,
andthespacecomplexityforstoringtheseinputsandthegeneratedHigh-ResolutionHSIisO(HWC). Theweightmatrices
of the network, primarily in the convolutional layers of the encoders and the fusion block, contain parameters proportional
toO(CD+D2). The intermediate feature maps generated during the forward pass require memory space ofO(HWD).
Therefore, the total space complexity of SA-RFM is O HW(C + D) + D2, which demonstrates that the memory
consumption grows linearly with the spatial resolution and channel dimension, ensuring scalability.








                                                                       20

<!-- PDF PAGE 21 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images













           Deformation    Integrated       MoGDCN      NonRegSR        M2DTN         MoEPNP         IR&ArF          PSRFDif           RFDif             PRFCo              Ours            Reference



      RGB






      Diff.






      SAM






     MRAE







      RGB






      Diff.






      SAM






     MRAE






      RGB






      Diff.






      SAM






     MRAE


Figure 7. Visual performance comparison under three types of distortions on the Pavia University dataset. The first row shows estimated
RGB(bands67-29-1)images,thesecondrowshowsthedifferencemapsaveragedoverallbandswithrespecttothereference,thethird
rowshowstheSpectralAngleMapper(SAM)errormaps,andthefourthrowshowstheMeanRelativeAbsoluteError(MRAE)maps,the
first column displays the Deformation field. Far-right colorbars indicate the error ranges.









                                                                                   21

<!-- PDF PAGE 22 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images













           Deformation    Integrated       MoGDCN      NonRegSR        M2DTN         MoEPNP         IR&ArF         PSRFDif           RFDif             PRFCo              Ours            Reference



       RGB






       Diff.






       SAM






      MRAE







       RGB






       Diff.






       SAM






      MRAE






       RGB






       Diff.






       SAM






      MRAE


Figure 8. Visual performancecomparison under threetypes of distortionson the Botswana dataset. The firstrow shows estimatedRGB
(bands48-15-4) images, thesecondrowshowsthe differencemapsaveraged overall bandswithrespectto thereference,thethird row
showstheSpectralAngleMapper(SAM)errormaps,andthefourthrowshowstheMeanRelativeAbsoluteError(MRAE)maps,thefirst
column displays the Deformation field. Far-right colorbars indicate the error ranges.









                                                                                   22

<!-- PDF PAGE 23 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images













           Deformation    Integrated       MoGDCN      NonRegSR        M2DTN         MoEPNP         IR&ArF         PSRFDif           RFDif             PRFCo              Ours            Reference



      RGB






       Diff.






      SAM






      MRAE







      RGB






       Diff.






      SAM






      MRAE






      RGB






       Diff.






      SAM






      MRAE


Figure9. VisualperformancecomparisonunderthreetypesofdistortionsontheChikuseidataset. ThefirstrowshowsestimatedRGB
(bands101-71-37)images,thesecondrowshowsthedifferencemapsaveragedoverallbandswithrespecttothereference,thethirdrow
showstheSpectralAngleMapper(SAM)errormaps,andthefourthrowshowstheMeanRelativeAbsoluteError(MRAE)maps,thefirst
column displays the Deformation field. Far-right colorbars indicate the error ranges.









                                                                                 23

<!-- PDF PAGE 24 -->

      Structure-Aware Riemannian Flow Matching for Registration and Fusion of Hyperspectral and Multispectral Images













           Deformation    Integrated       MoGDCN      NonRegSR        M2DTN         MoEPNP         IR&ArF         PSRFDif           RFDif             PRFCo              Ours            Reference



      RGB






       Diff.






      SAM






      MRAE







      RGB






       Diff.






      SAM






      MRAE






      RGB






       Diff.






      SAM






      MRAE


Figure 10. Visual performance comparisonunder threetypes ofdistortionson theXiongan dataset. Thefirst rowshows estimatedRGB
(bands103-77-52)images,thesecondrowshowsthedifferencemapsaveragedoverallbandswithrespecttothereference,thethirdrow
showstheSpectralAngleMapper(SAM)errormaps,andthefourthrowshowstheMeanRelativeAbsoluteError(MRAE)maps,thefirst
column displays the Deformation field. Far-right colorbars indicate the error ranges.









                                                                                  24

<!-- END-OF-PAPER:4b86a82f8070 -->
