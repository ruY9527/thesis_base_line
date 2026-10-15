

<!-- PDF PAGE 1 -->

                                                                                                                                                             1
         A convex formulation for hyperspectral image
    superresolution via subspace-based regularization

       Miguel Sim˜oes, Jos´e Bioucas-Dias, Member, IEEE, Luis B. Almeida, Jocelyn Chanussot, Fellow, IEEE,





   Abstract—Hyperspectral remote sensing images (HSIs) usually                  parts of the electromagnetic (EM) spectrum. They ﬁnd appli-
have high spectral resolution and low spatial resolution. Con-                  cationsintheﬁeldsofremotesensing(agriculture,mineralogy,
versely, multispectral images (MSIs) usually have low spectral                  etc.), astronomy, and biomedicine, for example [1]. Our focus
and high spatial resolutions. The problem of inferring images                   will be on the remote sensing ﬁeld, where spectral images are
which  combine  the  high  spectral  and  high  spatial  resolutions            typically generated from air- or spaceborne sensors.
of HSIs and MSIs, respectively, is a data fusion problem that
has been the focus of recent active research due to the increasing                 In this context, it is common to distinguish between hy-
availability of HSIs and MSIs retrieved from the same geograph-                 perspectral  and  multispectral  images  (HSIs  and  MSIs,  re-
ical area.                                                                      spectively). The difference is application-dependent, but HSIs
   We formulate this problem as the minimization of a convex ob-                typically  have  high  spectral  resolution  in  the  visible,  near-
jective function containing two quadratic data-ﬁtting terms and                 infrared,andshortwaveinfraredspectralranges[1].Asaresult
an edge-preserving regularizer. The data-ﬁtting terms account for
blur, different resolutions, and additive noise. The regularizer,               of this high resolution, HSIs have a large number of bands,
a  form  of  vector  Total  Variation,  promotes  piecewise-smooth              each one corresponding to a somewhat narrow part of the EM
solutions  with  discontinuities  aligned  across  the  hyperspectral           spectrum. For example, the Hyperion Imaging Spectrometer
bands.                                                                          has  about  200  spectral  bands,  each  covering 10nm  of  the
   The downsampling operator accounting for the different spa-                  spectrum, with a spatial resolution of 30m [2].1 On the other
tial resolutions, the non-quadratic and non-smooth nature of the
regularizer,andtheverylargesizeoftheHSItobeestimatedlead                        hand, MSIs generally offer a higher spatial resolution, but each
to a hard optimization problem. We deal with these difﬁculties                  band covers a larger range of the spectrum, resulting in a much
by  exploiting  the  fact  that  HSIs  generally  “live”  in  a  low-           smaller number of bands. For example, the IKONOS satellite
dimensional  subspace  and  by  tailoring  the  Split  Augmented                collects multispectral images covering four bands (blue, green,
Lagrangian Shrinkage Algorithm (SALSA), which is an instance                    redandnear-infrared)withaspatialresolutionof3.2m[3].2 In
of  the  Alternating  Direction  Method  of  Multipliers  (ADMM),
to this optimization problem, by means of a convenient variable                 other words, HSIs have comparatively high spectral and low
splitting.Thespatialblurandthespectrallinearoperatorslinked,                    spatial resolutions, while MSIs have low spectral and high
respectively, with the HSI and MSI acquisition processes are also               spatial resolutions.
estimated, and we obtain an effective algorithm that outperforms                   It is of interest to fuse the information from these two data
the state-of-the-art, as illustrated in a series of experiments with            sources, to synthesize images with simultaneously high spec-
simulated and real-life data.                                                   tral and high spatial resolutions. A related problem that has
   Index  Terms—Hyperspectral  imaging,  superresolution,  data                 been extensively studied is pansharpening, which addresses
fusion,  vector  total  variation  (VTV),  convex  non-smooth  opti-            the fusion of multispectral and panchromatic images, the latter
mization, Alternating Direction Method of Multipliers (ADMM).                   of which are single-band images usually covering the visible

                                                                                and  the  near-infrared  spectral  ranges  [4]–[6].  Panchromatic
                                                                                images (PANs) typically have a spatial resolution that is even
                          I.  INTRODUCTION                                      higher than the one of MSIs. The HSI-MSI fusion problem is
   MAGES are an efﬁcient way to describe and store visual                       signiﬁcantly more difﬁcult to solve than pansharpening, owing
Iinformation about our world. This work will deal with a                        to three factors: a) although both are ill-posed, there is a much
special kind of them, the so-called spectral images. A spectral                 larger number of variables to estimate in HSI-MSI fusion, b)
image, or data cube, is a set of 2D images, also termed bands,                  the hyperspectral data typically have a large dimensionality,
representing the reﬂectance or radiance of a scene in different                 which  can  act  computationally  more  as  a  crutch  than  an
                                                                                asset, and c) often, the spectral range covered by the HSI
  Miguel Sim˜oes is with Instituto de Telecomunicac¸˜oes, Instituto Superior    is signiﬁcantly larger than the one covered by the MSI, and,
T´ecnico, Universidade de Lisboa, Lisbon, Portugal and GIPSA-Lab, Univer-       therefore, many bands of the HSI are not included in any band
sit´e de Grenoble, France.                                                      of the MSI.
  Jos´e   Bioucas-Dias   and   Luis   B.   Almeida   are   with   Instituto   de
Telecomunicac¸˜oes, Instituto Superior T´ecnico, Universidade de Lisboa, Lis-      Recently, some techniques dedicated to the fusion of HSIs
bon, Portugal.                                                                  and MSIs have been proposed. A common trend is to associate
  Jocelyn Chanussot is with GIPSA-Lab, Grenoble Institute of Technology,        this  problem  with  the  linear  spectral  unmixing  one,  which
France and with the Faculty of Electrical and Computer Engineering, Univer-
sity of Iceland.                                                                assumesthattheunderlyingdatacanbedescribedbyamixture
  This  work  was  partially  supported  by  the  Fundac¸˜ao  para  a  Ciˆencia    1More information at http://eo1.usgs.gov/sensors/hyperion.
e   Tecnologia,   Portuguese   Ministry   of   Science   and   Higher   Education,
projects   PEst-OE/EEI/0008/2013,   PTDC/EEI-PRO/1470/2012,   and   grant          2More  information  at  http://www.digitalglobe.com/sites/default/ﬁles/DG
SFRH/BD/87693/2012.                                                             IKONOS     DS.pdf.

<!-- PDF PAGE 2 -->

                                                                                                                                                        2


of  a  relatively  small  number  of  “pure”  spectral  signatures,           the images with data reduction techniques, to formulate the
corresponding to the materials that are present in the scene [1],             problem  in  a  computationally  efﬁcient  way.  This  method,
[7]. Since both HSIs and MSIs capture the same scene, the                     which  we  term  HySure,  for  Hyperspectral  Superresolution,
underlying materials (the so-called endmembers) should be the                 allows us to fuse hyperspectral data with either multispectral
same. Therefore, a spectral dictionary extracted from one of                  or panchromatic images.
the images should also be able to explain the other one. Due                     In the literature, the HSI-MSI fusion problem is very often
to the high spectral resolution of the HSIs, the dictionary is                dealt with as a non-blind one, in the sense that the spatial and
extracted from these data, and is then used to reconstruct the                spectral responses of the sensors are assumed to be known
multispectral data via sparse regression. The estimate of the                 (see [13], [15]–[17], for example). In practice, however, the
original high resolution HSI is then obtained from the regres-                information that is available about these responses is often
sion coefﬁcients and from the dictionary. This technique was                  scarce and/or somewhat inaccurate. In this work, we take a
introduced in [8] for HSIs, but there are older works exploiting              blind approach, assuming that these responses are unknown,
similar ideas for MSIs [9]. For example, Kawakami et al. [10]                 and we formulate another convex problem to estimate them,
fused hyperspectral images with images from RGB cameras,                      making only minimal assumptions: we assume that the spatial
starting by estimating the endmember mixing matrix from the                   response  has  limited  support  and  that  both  responses  are
hyperspectral data through aℓ1-minimization problem, solved                   relatively smooth. The estimate of the spectral response can be
via a non-smooth Gauss-Newton algorithm. The endmember                        improvedbyusinginformationonthecorrespondencebetween
matrix, jointly with the spectral responses of the RGB sensor,                bands from the two images, if that information is available—it
was then used as a basis to reconstruct the RGB image, by                     is often easily obtained from data on the spectral coverage of
formulating an optimization problem that imposed sparsity.                    the various bands from the two sensors.
In [11], Huang et al. unmixed the hyperspectral data via the K-                  This work extends [26] in several different directions: it
SVD algorithm, and reconstructed the MSI using orthogonal                     details the optimization process more clearly, it establishes the
matching pursuit to induce sparsity. The method was tested                    framework used to estimate the spatial and spectral responses
with Landsat/ETM+ and Aqua/Modis images. Song et al. [12]                     of the sensors, and it presents a number of new experimental
ﬁrst  learned  two  dictionaries  from  the  two  different  data,            results.
and then used a dictionary-pair learning method to establish
the correspondence between them. Again, their method was                      B. Outline
tested using Landsat/ETM+ with Aqua/Modis data, but only                         The remainder of this work is organized as follows. Sec-
taking into account the spectrally overlapping bands. A similar               tion II describes the data fusion method, including the pro-
and  older  technique  is  the  one  from  Yokoya  et  al.,  which            posed model and the formulation of the optimization problem.
alternately unmixes both sources of data to ﬁnd the signatures                The approach followed to perform the optimization is pre-
and the abundances of the endmembers [13].                                    sented in Section III. Section IV deals with the estimation of
  AdifferentframeworkwasproposedbyHardieetal.in[14],                          the sensors’ spatial and spectral responses. Section V presents
in which a fully Bayesian approach was followed, by imposing                  experimental results. Section VI concludes.
prior distributions on the problem. This work was the founda-
tion for other works: Zhang et al. introduced a method which
works  in  the  wavelet  domain  [15],  and  later  published  an                                 II.  DATA FUSION METHOD
expectation-maximization algorithm to maximize the posterior                  A. Observation Model
distribution [16]. Wei et al. used a Hamilton Monte Carlo                        Multispectral  and  hyperspectral  images  can  be  thought
algorithm  to  deal  with  the  high-dimensional  space  of  the              of as three-dimensional arrays or tensors, often called data
posterior distribution [17]. In [18], Chen et al. introduced a                cubes. However, for notational convenience, the representation
method that treats image registration and image fusion as a                   followed  in  this  work  will  consider  HSIs  and  MSIs  to  be
joint process. The fusion of HSIs with just the panchromatic                  two-dimensional matrices, where each line corresponds to a
band is a different, but related, problem [19]–[21]. Using only               spectral band, containing the lexicographically ordered pixels
the hyperspectral image, different authors [22], [23] treated                 of that band. We use bold lowercase to denote vectors (e.g.,
this problem as a simple superresolution one.                                 x, y) and bold uppercase to denote matrices (e.g., H, M).
                                                                                 Let the matrix representing the observed hyperspectral data
A. Contributions                                                              be Yh∈RLh×nh, withLh bands and spatial dimensionnh,
  This work is built around the standard linear inverse prob-                 and let Ym∈ RLm×nm  denote the observed multispectral
lemmodelforHSIsandMSIs.Thismodelisusedtoformulate                             data, withLm<Lh bands and spatial dimensionnm>nh.
data fusion as a convex optimization problem. We use a form                   Matrix Z∈ RLh×nm  denotes the high spatial and spectral
of vector Total Variation-based regularization [24], taking into              resolution data to be estimated.
account both the spatial and the spectral characteristics of the                 With this representation, we model the hyperspectral mea-
data. In order to perform the optimization, we follow an Alter-               surements as
nating Direction Method of Multipliers (ADMM) approach by                                              Yh =ZBM+Nh                             (1)
using the Split Augmented Lagrangian Shrinkage Algorithm                      where  matrix B∈ Rnm×nm  is  a  spatial  blurring  matrix
(SALSA) [25], and we explore the inherent redundancy of                       representing the hyperspectral sensor’s point spread function

<!-- PDF PAGE 3 -->

                                                                                                                                                      3


in  the  spatial  resolution  of Z;  it  is  assumed  to  be  band-          dimensionality. As an illustration of the amount of reduction
independent  and  to  be  under  circular  boundary  conditions.             that is possible, assume that the hyperspectral image has 200
These two assumptions are made for simplicity. When dealing                  bands. WithLs = 10, which is a typical value, only 5% of
with non-blind data fusion, allowing the blur to vary across                 the number of original variables need to be inferred.
bands would not change the complexity of the algorithm. In                      Different approaches can be followed to factorize matrix
the blind case, the increase in complexity would be relatively               Z, and two of them will be brieﬂy mentioned here. One is
small.  Regarding  the  boundary  conditions,  assuming  them                to take into account the physical process that gave origin to
to be periodic has two main advantages. First, it allows to                  Yh. In the linear unmixing approach [29], it is assumed that
use  Fast  Fourier  Transforms  (FFTs)  to  compute  convolu-                the spectral response of each pixel is a linear combination of
tions.  Second,  matrix  inversion,  usually  a  costly  operation,          the pure spectral signatures of the underlying endmembers. In
is easily performed, again through the use of FFTs, under                    this case, E would be the spectral signature matrix obtained
certain conditions that are met in this case. Although periodic              from Yh, and X would represent the abundance fractions of
boundaryconditionsarenottotallyrealistic,weexperimentally                    the endmembers for every pixel of Z. There are numerous
found  that  they  do  not  lead  to  any  signiﬁcant  artifacts  in         algorithms in the literature that address the unmixing problem
the fused image, while allowing a dramatic reduction in the                  (for  example,  Vertex  Component  Analysis  –  VCA  [31]).
amount of computation. A technique based on ADMM that                        Several of the methods discussed in Section I use the linear
makesnoassumptionsabouttheboundarieshasbeenproposed                          mixing model.
in [27], [28], but we did not ﬁnd the corresponding increase                    Another approach is to use Singular Value Decomposition
in complexity justiﬁed for the images we have worked on.                     (SVD) to obtain the factorization Yh = UΣVT, where U
  Matrix M∈ Rnm×nh,  whose  columns  are  a  subset  of                      andV are orthogonal matrices andΣ is a rectangular diagonal
the columns of the identity matrix, accounts for a uniform                   matrix containing the singular values, which are assumed to be
subsampling of the image, to yield the lower spatial resolution              in non-increasing order. Denote by ˆΣ, ˆU and ˆV, respectively,
of the hyperspectral image. Nh  represents independent and                   the truncated matrices obtained by discarding the rows and
identically distributed (i.i.d.) noise. The assumption that the              columns with the smallest singular values from Σ and the
noise  is  identically  distributed  across  bands  is  also  made           corresponding  columns  of U  and V.  A  low-dimensional
for simplicity. Accommodating statistically independent noise                approximationofYh isgivenby ˆUˆΣˆVT.Inthisapproach,we
across bands and pixels, but with band-dependent variance,                   make E = ˆU. Due to the low intrinsic dimensionality of the
would be straightforward.                                                    hyperspectraldata,mostofthesingularvaluesarerathersmall,
  We model the multispectral measurements as                                 allowing  a  very  signiﬁcant  dimensionality  reduction  while
                         Ym =RZ+Nm,                              (2)         retainingaratherfaithfulapproximationofYh.IfNh =0and
                                                                             all discarded singular values are zero, this representation spans
where R∈RLm×Lh holds in its rows the spectral responses                      the true signal subspace. If the former condition onNh is not
of the multispectral instrument, one per multispectral band,                 obeyed but Nh is i.i.d., this representation corresponds to the
and Nm represents i.i.d. noise.                                              maximum likelihood estimate of that subspace. However, if
  In this work, matricesB andR are estimated from the data,                  the noise is non-i.i.d., the estimation of the subspace is more
by formulating a quadratic optimization problem. Section IV                  complex; see, for example, [32] for details, and for algorithms
will address that topic.                                                     oriented to subspace estimation in hyperspectral applications.
                                                                                With any of these two factorizations, we replace Eq. (1)
B. Dimensionality reduction                                                  with
  Hyperspectral data normally have a large correlation be-                                           Yh =EXBM+Nh,                           (4)

tween bands: the spectral vectors, of sizeLh, usually “live”                 where the error due to the dimensionality reduction has been
in a subspace of dimension much lower thanLh [29], [30].                     incorporated into Nh.
Therefore, we can write                                                         Remote sensing images often are somewhat noisy. The use
                               Z =EX,                                     (3)of truncated SVD is also a very common approach to perform
                                                                             denoising, a topic that we shall address in Section V.
where E∈RLh×Ls is a matrix whoseLs columns span the
same subspace as the columns ofZ, andX∈RLs×nm are the                        C. Regularization
representation coefﬁcients. Small values ofLs, i.e.,Ls≪Lh,                      Theproblemthatwearetryingtosolveisstronglyill-posed,
translate into a description of the data in a relatively low-                and therefore needs adequate regularization. The regularizer
dimensional space.                                                           that we use is given by
  This dimensionality reduction has two advantages. One is                                                √
that it is computationally more efﬁcient to work in a lower                                           nm∑     Ls∑{[(XDh)ij]2+[(XDv)ij]2}
dimensional space than in the original space of Z, making                    ϕ(XDh,XDv)def=                                                            ,
algorithms which use these representations comparatively fast.                                        j=1     i=1
The other advantage is that, since the number of variables                                                                                          (5)
to  be  estimated  is  signiﬁcantly  reduced,  the  estimates  will          where  (A)ij  denotes  the  element  in  theith  row  andjth
normally be more accurate than if we worked in the original                  column of matrixA, and the products by matricesDh andDv

<!-- PDF PAGE 4 -->

                                                                                                                                                       4


compute the horizontal and vertical discrete differences of an                experience. The next section and the Appendix describe the
image, respectively, with periodic boundary conditions. This                  details of the optimization method.
regularizer is a form of vector Total Variation (VTV) [24]. Its
purpose is to impose sparsity in the distribution of the absolute                                III.  OPTIMIZATION METHOD
gradient of an image, meaning that transitions between the                      ADMM  involves  the  introduction  of  auxiliary  variables
pixels of an image should be smooth in the spatial dimension,                 into the optimization problem, through the so-called variable
exceptforasmallnumberofthem,whichshouldcoincidewith                           splitting technique. We split the original optimization variable
details such as edges. Total Variation was proposed for the ﬁrst              X into a total of ﬁve variables: one which we still callX, and
time in [33] and is extensively used in image restoration [25],               four auxiliary variables,V1 toV4. The optimization problem
[27],  [34]–[39].  It  has  two  different  discrete  formulations,           becomes
the  anisotropic  and  isotropic  ones  [40];  in  this  work,  we                                1 Yh−EV1M2                 Ym−REV22
use the isotropic formulation. In [41], Zhao et al. proposed                     minimize
an isotropic TV scheme for hyperspectral image deblurring                     X,V1,V2,V3,V4       2                     F +λm2                         F
in a band-by-band manner. This means that each band was                                               +λϕϕ(V3,V4)
regularized independently from the other ones. This approach                  subject to         V1 =XB,                                                  (7)
has a shortcoming: it does not take into account that edges                                       V2 =X,
and other details normally have the same locations in most                                        V3 =XDh,
bands. The vector form of the regularizer, which we use in this                                   V4 =XDv.
work, promotes solutions in which edges and other details are
aligned among the different bands. VTV has previously been                    For notational simplicity, we deﬁne the matricesV and H,
used in a pansharpening application [42] and in the denoising                                          VT1VT2VT3VT4        BT   
of hyperspectral images [43].                                                                              ,   H def=  I  ,
  We apply the regularizer to the reduced-dimensionality data                                  V def=                         DThDTv
X, and not to Z itself. This is indeed reasonable, since the
subspace spanned byE is the same as the one whereZ resides
(or an approximation, when using truncated SVD), and by                       and the cost function as follows:
regularizing X we are indirectly regularizing Z.                                                Yh−EV1M2                Ym−REV22
                                                                                 f(V)def=  12                       F +λm2                         F
D. Optimization problem                                                                          +λϕϕ(V3,V4).
  Let∥X∥F       def= √Tr(XXT) denote the Frobenius norm of                    We can express (7) as
X, and (·)T  denote the transposition operator. We can now                                           minimize
formulate an optimization problem based on our model with                                               X,V   f(V)                                   (8)
the proposed regularizer:                                                                            subject to   V =HXT.
                 1 Yh−EXBM2                 Ym−REX2               This problem has the following augmented Lagrangian [46]
  minimize
       X         2                      F +λm2                       F                                                HXT−V−A2
                    +λϕϕ(XDh,XDv).                                (6)                 L(X,V,A)=f(V)+µ2                                       F,      (9)

The ﬁrst two terms are data-ﬁtting terms, imposing that the                   where A  is  the  so-called  scaled  dual variable  [47],  andµ
estimated image should be able to explain the observed data                   is a positive constant, called penalty parameter. We are now
according to the model deﬁned in (4) and (2). The last term                   ready to apply the ADMM method, which yields the algorithm
is  the  regularizer.  The  parametersλm  andλϕ  control  the                 shown in Fig. 1. As we can see, SALSA solves the original,
relative importances of the various terms. We shall discuss                   complex optimization problem through an iteration on a set
the selection of these parameters in Section V-C.                             of  much  simpler  problems.  The  constraints  are  taken  into
  Problem (6) is convex, but is rather hard to solve, due to                  account, in an approximate way, by minimizing the augmented
the nature of the regularizer, which is non-quadratic and non-                Lagrangian of the problem relative to the auxiliary variables.
smooth.  Additional  difﬁculties  are  raised  by  the  large  size             The minimization with respect to X is a quadratic problem
of X (the variable to be estimated) and by the presence of                    with a block cyclic system matrix, which can be efﬁciently
the downsampling operator M in one of the quadratic terms,                    solved by means of the Fast Fourier Transform (FFT). Min-
preventing a direct use of the Fourier transform in optimiza-                 imizing  with  respect  to  the  auxiliary  variables  is  done  by
tions involving this term. We deal with these difﬁculties by                  solving three different problems, whose solutions correspond
using  SALSA [25].  An  alternative  approach  would  consist                 to three Moreau proximity operators [48]. The minimization
in  employing  a  primal-dual  method  [44],  [45].  Unlike  our              with respect toV1 is a quadratic problem which is efﬁciently
approach,  primal-dual  methods  do  not  require  the  solution              solved via FFTs, and the minimization relative to V2 is also
of  linear  systems  of  equations  on  each  iteration.  However,            quadratic; these two problems involve matrix inverses which
since the system matrix in our problem is diagonalizable using                can be computed in advance. Finally, the minimization with
light  computations,  SALSA  yields  much  faster  algorithms                 respect toV3 andV4 corresponds to a pixel-wise vector soft-
than those based on primal-dual methods, according to our                     thresholding operation.

<!-- PDF PAGE 5 -->

                                                                                                                                                     5


Require:  data: Yh, Ym; regularization parameters:λm,λϕ;                    directly addressed the estimation of responses for the fusion
   penalty parameter:µ; matrices R, B and E; initializations:               of HSIs and MSIs from the Hyperion and ASTER sensors,
   V(0) and A(0)                                                            respectively, which are aboard two different satellites. Their
   k := 0                                                                   method estimates both the relative spatial and relative spectral
   repeat                       (                                           responses of the sensors. The spatial response is assumed to
      X(k+1)∈ argmin              X,V(k),A(k))                              correspond to a Gaussian blur and its variance is estimated by
                      X L(                                                  using a template-matching technique. In order to determine
      V(k+1)∈ argmin              X(k+1),V,A(k))                            the spectral response, the authors use the so-called pre-launch
                       V L(                                                 response, with information obtained from measurements per-
      A(k+1) :=A(k)−           HX(k+1)T−V(k+1))                             formed on the sensors before they were launched into space.
      k :=k+1                                                               The method tries to ﬁnd a spectral response that is able to
   until stopping criterion is satisﬁed.                                    describe the observed data and that is close to the pre-launch
Fig. 1.   Pseudocode for the HySure algorithm. For details, see the Appendix.response. In a different approach, Huang et al. estimated the
                                                                            spectral response directly form the data, without requiring a
                                                                            priori information [11].
  The details of the optimization, as well as an analysis of                   Recall that, without noise,
the  algorithm’s complexity,  are  presented  in the  Appendix.                                 Yh =ZBM,   Ym =RZ,
The number of splitting variables could have been reduced,                  which implies that
by eliminating V1, for example. This could have been done
via a scheme similar to the one proposed in [41], working                                              RYh =YmBM.                             (10)
with Kronecker products. We chose not to do so, since the                      Taking (10) into account, we infer R and B by solving the
form of the algorithm that we presented above is simpler to                 optimization problem
derive, and the computational and memory gains of doing one                                 RYh−YmBM2+λbφB(B)+λRφR(R),
less splitting did not seem to be very signiﬁcant.                            minimize
  The algorithm described above satisﬁes the conditions for                       B,R                                                            (11)
the convergence of SALSA established in [25], which require                 whereφB(·) andφR(·) are quadratic regularizers that will be
matrixH to have full column rank (which is true in our case,                discussed in detail below, andλb,λR≥ 0 are the respective
due to the presence of identity matrix I), and functionf(·)                 regularization parameters. Matrix B, and possibly also matrix
to be closed, proper, and convex (which is also true, since                 R, are subject to some constraints discussed below.
it is a sum of closed, proper, and convex functions). Under                    A  special  consideration  needs  to  be  made  regarding  the
these conditions, and for arbitraryµ> 0, V(0) and A(0), if                  estimation of the spectral response. This is due to the fact
problem (8) has a solutionX∗, then the sequence{X(k)} will                  that, when using the observed data, it is not possible to fully
converge to X∗; if a solution does not exist, then at least                 estimate matrix R. The reason for this is that, as discussed
one  of  the  sequences{V(k)}  or{A(k)}  will  diverge.  The                in Section II-B, the hyperspectral data normally span only a
actual value of the penalty parameterµ is not important as                  low-dimensional subspace of the full spectral space. Only the
a condition for convergence, but can have a strong inﬂuence                 component of R parallel to that subspace can be estimated.
on the convergence speed of the algorithm. The choice ofµ                   This is not a drawback, however, since the component of R
is discussed in Section V-C.                                                orthogonal to that subspace has essentially no inﬂuence on the
 IV.  ESTIMATING THE SPATIAL BLUR AND THE SPECTRAL                          result of the image fusion. In fact, if we writeR =R∥+R⊥,
                                                                            where R∥ =RP∥ and R⊥ =RP⊥, and P∥ and P⊥ denote
                    RESPONSE FROM THE DATA                                  theprojectionmatricesontothesubspacesspannedbytheorig-
  As previously mentioned, matrices B and R are estimated                   inal hyperspectral vectors and onto the subspace orthogonal to
from the observed images. The advantages of doing so are                    it, respectively, we have RYh = R∥Yh +R⊥Yh = R∥Yh,
threefold. First, as previously mentioned, the available infor-             since R⊥Yh is zero. For the product RZ, which is involved
mation about the sensors can be rather scarce. Second, it may               in the fusion problem, we have RZ≈ R∥Z, since Z will
be hard to precisely adapt that information to the model that is            span  approximately  the  same  subspace  as Yh,  because  it
being used for data fusion. Third, there may be discrepancies               corresponds to an image containing the same endmembers.
between the real spatial and spectral responses and the data                   According to the observation model presented in Section
supplied by the manufacturers. These can be due to several                  II-A,  matrix B  accounts  for  a  2D  cyclic  convolution.  In
causes, such as atmospheric conditions, postprocessing arti-                addition,  we  assume  that  the  convolution  kernel  has  ﬁnite
facts, and even the variability within the observed scene [49].             support  contained  in  a  square  window  of  size√nb,  thus
  As  already  mentioned,  in  [15],  [16],  Zhang  et  al.  as-            containingnb pixels, centered at the origin.
sumed the spatial response to be known. However, they also                     Let [YmB]:j  denote thejth column of YmB, b∈ Rnb
suggested  using  Gaussian  blurs  with  different  variances  as           denote the columnwise ordering of the convolution kernel, and
spatial  responses  when  this  was  not  the  case,  arguing  that         Pj∈Rnm×nb denote a matrix which selects fromYm a patch
their fusion method did not require a strict knowledge of the               such that
spatial response of the sensor. In [50], Yokoya et al. have                                        [YmB]:j = (YmPj)b.

<!-- PDF PAGE 6 -->

                                                                                                                                                      6


With these deﬁnitions in place, a slight modiﬁcation of the                  of the HSI and the MSI, we constrain the elements of R that
optimization (11) is                                                         correspond to non-overlapping bands to zero.
                          nh∑ RYh,:j−Ym,jb2                               The estimation ofR can be made independently for each of
          minimize                                                           the MSI bands. LetrTi  denote a row vector containing theith
              b,R         j=1                                                rowofRwithouttheelementsthatareknowntocorrespondto
                                  +λbφb(b)+λRφR(R)                   (12)    hyperspectral bands that do not overlap theith multispectral
          subject to   bT1 = 1,                                              band, and by Yh,i  denote the matrix Yh  without the rows
                                                                             corresponding to those same bands. The optimization of (12)
where    Yh,:j     denotes     the  jth     column     of    Yh,             is decoupled with respect to the rows ofR and may be written
Ym,jdef=[(YmPcj)] ∈   RLm×nb,   with cj   denoting   the                     as
column  of Ym  corresponding  to  thejth  column  of Yh,                                       rTi Yh,i−Ym,i:BM2                Dri2,    (15)
φb(b)def=φB(B), and the normalization condition bT1 = 1                          minimize                                  F +λR
imposes unit DC gain of the blur.                                                    ri
  We note that (12) is a quadratic program with only equality                in  which Ym,i:  is  theith  row  of Ym,  and  the  product
constraints and, therefore, using Lagrange multipliers, its solu-            by D computes the differences between the elements in ri
tion can be obtained by solving a linear system of equations.                corresponding to contiguous hyperspectral bands. The solution
However, even though we have a closed-form solution, be-                     of (15) is given by
cause the size of the optimization variables (i.e.,nb+Lm×Lh)
is usually of the order of thousands, it may be useful to solve                        [                          ]−1Yh,i[                ]T.     (16)
problem (12) via alternated minimization with respect to b                       r∗i =   Yh,iYTh,i+λRDTD                      Ym,i:BM
and R.
  The optimization with respect to b leads to the following                     The  estimation  of  each  of  the  matrices B  and R,  as
regularized least squares problem:                                           presented so far, requires the knowledge of the other matrix.
                                                                             In  order  to  estimate  both,  and  instead  of  using  alternating
                          nh∑  RYh,:j−Ym,jb2                           optimization as proposed before, we adopt an even simpler
           minimizeb                                                         technique.  We  start  by  estimating R.  To  do  this  without
                          j=1      (Dhb2+Dvb2)                       knowing B, we ﬁrst blur both spectral images with a spatial
                             +λb                                     (13)    blur that is much stronger than the one produced by B, so

           subject to   bT1 = 1,                                             that the effect of B becomes negligible. This, conveniently,
                                                                             also minimizes the effect of possible misregistration between
  The two last terms of the function being minimized in (13)                 the hyperspectral and multispectral images. Following this, we
correspond toφb(·), which is a noise-removing regularizer that               estimate the spectral responseR using (16), setting the kernel
smooths the estimated convolution kernel by promoting that                   of the spatial blur between the strongly blurred multispectral
the values of the differences between neighboring pixels be                  and hyperspectral images to a delta impulse. Finally, we esti-
small. As  before, Dh  and Dv  compute the horizontal  and                   mate the spatial blur B using (13) on the original (unblurred)
vertical discrete differences of the convolution kernel, with                images, with the value of R just found. Fig. 2 summarizes
dimensions adjusted for this particular case.                                the estimation method. In the tests presented in Section V, we
  An  approximate  solution  for  (13)  is  computed  by  ﬁrst               have used, for the strong spatial blur, an averaging in a square
relaxing the constraint, estimating the ﬁlter without the nor-               of 9× 9 pixels for the MSI, and a correspondingly smaller
malization condition, and then normalizing the result to unit                averaging for the HSI.
DC gain. The solution of the unconstrained problem is given                     We now discuss the set of solutions of (12), which is an
by                                                                           important issue in our approach to the estimation ofb andR,
                                                                             closelyrelatedtothatofidentiﬁability.Giventhattheobjective
           [ nh∑                                                             function is quadratic, a sufﬁcient condition for it to have a
   b∗ =          YTm,jYm,j +λb(DThDh+DTvDv)]−1                               unique solution is that its Hessian matrix be positive deﬁnite.
             j=1                                                             Assuming thatλb,λR> 0, the null space associated with the
                                        [ nh∑                 ]      (14)    regularization terms is the set
                                               YTm,jRYh,:j      .               Adef={(R,b) : R =c1TLh, b =d1nb,c∈RLm,d∈R},
                                          j=1
  The support covered byb is user-speciﬁed. We have found,                   where we have assumed that the spectral response of the MS
experimentally, that the choice of this support does not have                channels spans over the entireLh HS spectral bands, and the
much inﬂuence on the blur estimate, as long as it encompasses                HS bands are contiguous in frequency. The case in which the
the support of the actual blur.                                              spectral response of the MS channels spans over subsets of the
  Concerning  the  estimation  of R,  we  use  the  regularizer              Lh HS spectral bands corresponds to a minor modiﬁcation of
φR(·) in order to deal with the indetermination of the orthog-               the reasoning provided below. The case in which the HS bands
onal component, and to reduce estimation noise. In the cases                 are not contiguous is somewhat more elaborate, but would
in which there is information about the overlap between bands                follow the same line of reasoning.

<!-- PDF PAGE 7 -->

                                                                                                                                                             7


   For any (R,b)∈A, we may write                                                 Administration (NASA) Airborne Visible/Infra-Red Imaging
    nh∑  RYh,:j−Ym,jb2 =          nh∑yh,jc−ym,jd2,     (17)          Spectrometer (AVIRIS), which is capable of delivering cali-
                                                                                 brated images in 224 contiguous spectral channels within the
    j=1                                j=1                                       0.4-2.5µm range [3].4 Five signatures from this library were
                                                        def=1TLhYh,:j  and       randomly selected as endmembers, and the image was built
for some c∈ RLm,d∈ R and whereyh,j                                               under the linear mixing model.
ym,jdef=Ym,j1nb. Let us suppose that there exits a nonzero                          We created an image with high resolution both in the spatial
couple (c,d) nulling all thenh quadratic terms in the right                      and  in  the  spectral  domains,  to  serve  as  ground  truth.  To
hand  side  of  (17).  In  this  case,  all  vectors ym,j,  forj  =              create a hyperspectral image, we spatially blurred the ground
1,...,nh would be collinear withc. Having into consideration                     truth one, and then downsampled the result by a factor of 4
that the components of ym,j represent the average intensities                    in each direction. Three different spatial blurs (block ﬁlter
intheLm MSbandsinthepatchPcj,suchascenarioishighly                               with  dimensions  5× 5,  Gaussian  ﬁlter  withσ  =  2  and
unlikely, implying that the intersection of the subspaceA with                   support 5×5, and the Starck-Murtagh ﬁlter [51]) were used
the null space associated with the data term shown in the left                   to synthesize three different HSIs. A false color representation
hand side of (17) is empty, except for the origin. We conclude,                  of a hyperspectral image can be seen in Fig. 6b, in which
therefore, that the Hessian of the quadratic objective function                  different colors correspond to different materials. To create
present in (12) is positive deﬁnite and, thus, the solution of the               panchromatic and multispectral images, the spectral response
corresponding optimization exists and is unique. An important                    of  the  IKONOS  satellite  was  used.  This  satellite  captures
consequence of this uniqueness is that the subproblems (13)                      both  a  panchromatic  (0.45-0.90µm)  and  four  multispectral
and (15) have unique solutions; moreover, the system matrices                    bands (0.45-0.52, 0.52-0.60, 0.63-0.69 and 0.76-0.90µm) [3].
present in the expressions (14) and (16) are nonsingular.                        Unless  otherwise  noted,  Gaussian  noise  was  added  to  the
                                                                                 hyperspectral image (SNR=30 dB) and to the multispectral
Require:  data: Yh  and Ym; regularization parameters:λR                         image (SNR=40 dB).
   andλB.                                                                           Dataset B was semi-synthetic. It was based on a standard
   Blur Ym with a strong blur.                                                   hyperspectral image (Pavia University, see Fig. 7). This image
   Blur Yh with a correspondingly scaled blur.                                   was obtained with the Reﬂective Optics System Imaging Spec-
   Estimate R using (16) on the blurred data.                                    trometer  (ROSIS),  which  has  115  spectral  bands,  spanning
   Estimate B using (14) on the original observed data.                          the 0.43-0.86µm spectral range, and a spatial resolution of
   Normalize b to unit DC gain.                                                  1.3m [3].5 This image was used as ground truth. Hyperspec-

Fig. 2.     Summary of the method to estimate the spectral response R and        tral, multispectral and panchromatic images were generated
the spatial blur; note thatB andb are just two different ways of expressing      from it as described for dataset A.
this spatial blur. Ym  and Yh  refer to the multispectral and hyperspectral
observations, respectively.                                                         Dataset  C  consisted  of  images  taken  above  Paris  (see
                                                                                 Fig. 8a), and was obtained by two instruments on board the
                                                                                 Earth Observing-1 Mission (EO-1) satellite, the Hyperion in-
                     V.  EXPERIMENTAL STUDY                                      strument and the Advanced Land Imager (ALI). Hyperion is a
   In  this  section,  we  ﬁrst  describe  the  datasets  that  were             hyperspectralimagerwithaspatialresolutionof30meters;the
used in the experimental tests, and the indices that were used                   ALI instrument provides both multispectral and panchromatic
to  evaluate  the  quality  of  the  results.  We  then  give  some              images at resolutions of 30 and 10 meters, respectively [2].6
details on the implementation of our algorithm and, ﬁnally, we                   The  hyperspectral  and  panchromatic  images  were  directly
present the experimental results, which include comparisons                      used for experiments on hyperspectral+panchromatic fusion,
with several other fusion methods.                                               and  therefore  we  had  no  access  to  the  ground  truth.  For
                                                                                 experiments on the fusion of hyperspectral and multispectral
                                                                                 images, we needed the HSI to have lower resolution than the
A. Data sets                                                                     MSI, and therefore we ﬁrst reduced the spatial resolution of
   Three datasets were used to test the different algorithms.                    the hyperspectral image by blurring with the Starck-Murtagh
Dataset A was purely synthetic. The ground truth image was                       ﬁlter  and  downsampling,  as  described  above  for  dataset  A
a collection of simple geometric shapes composed of different                    (using a downsampling factor of 3, in this case). The original
hypothetical materials. In order to simulate the different ma-                   hyperspectral image, before blurring and downsampling, was
terials, the U.S. Geological Survey Digital Spectral Library                     used as ground truth.
splib06  was  used.3  This  library  assembles  the  reﬂectance
values of different materials (e.g., minerals, plants, microor-
ganisms, man-made materials) as measured by different in-
struments, covering the wavelength range from ultraviolet to                       4More information is available at http://aviris.jpl.nasa.gov/.
                                                                                   5More        information        is        available        at        http://messtec.dlr.
far infrared. One of the instruments used to spectroscopically                   de/en/technology/dlr-remote-sensing-technology-institute/
analyze  the  data  was  the  National  Aeronautics  and  Space                  hyperspectral-systems-airborne-rosis-hyspex/index.php.
                                                                                   6More information is available at http://eo1.gsfc.nasa.gov/, http://eo1.usgs.
  3Available at http://speclab.cr.usgs.gov/spectral-lib.html.                    gov/sensors/ali and http://eo1.usgs.gov/sensors/hyperioncoverage.

<!-- PDF PAGE 8 -->

                                                                                                                                                        8


B. Quality indices                                                            ground truth. We hence only show false color representations
  To evaluate the quality of fusion results, three indices taken              of the estimated images, for visual inspection.
from  the  literature  were  used,  when  a  ground  truth  image
was available, as was the case for datasets A and B, and for                  C. Implementation details
HS+MS fusion on dataset C. The ﬁrst index was theErreur                          In the experimental tests, we performed two preprocessing
Relative Globale Adimensionnelle de Synth`ese (ERGAS), pro-                   steps  on  the  hyperspectral  data:  First,  uncalibrated  or  very
posed in [52] and deﬁned, for an estimated imageZ and a                       noisy bands were removed: when information on which bands
ground truth image ˆZ, as                                                     were uncalibrated was available (as in the case of Hyperion), it
                                  √
                                      1   Lh∑  MSE(Zl:,ˆZl:)                  was used; very noisy bands were identiﬁed manually. Second,
    ERGAS(Z,ˆZ)def= 1001S            Lh              µ2ˆZl:      ,     (18)   the  data  were  denoised  by  projecting Yh  onto  a  subspace
                                          l=1                                 of  dimensionLs  =  10  found  through  truncated  SVD;  the
whereS  is the ratio between the resolutions of the hyper-                    ground truth images, when available, were also projected onto
spectral image and of the multispectral or panchromatic one,                  this subspace. MakingLs = 10 allowed us preserve at least
i.e.,S  = √nm/nh; Zl:  and ˆZl:  are  thelth  bands  of  the                  99.95% of the energy of the original images from all datasets.
estimated image and of the ground truth image, respectively;                  Dataset C consisted of raw data, in which the energy per band
MSE(Zl:,ˆZl:) is the mean squared error betweenZl: andˆZl:;                   strongly varied across the spectrum. For this reason, before
andµˆZl: is the mean of ˆZl:.                                                 denoising,  we  normalized  all  bands  of  this  dataset  so  that
  The second index was the Spectral Angle Mapper (SAM),                       the 0.999 intensity quantile corresponded to a value of 1. A
which is the mean, among all pixels, of the angle between                     summary of these two steps can be seen in Fig. 3.
the vectors formed by the spectral representation of the pixel
in the estimated image and the spectral representation of the                  Require: observed hyperspectral and multispectral data.
same pixel in the ground truth image,                                             Remove uncalibrated/noisy bands.
                            nm∑          (     ZT:jˆZ:j       )                   Normalize of each band.
   SAM(Z,ˆZ)def=   1             arccos     Z:j2 ˆZ:j2   ,    (19)        Denoise the data.
                       nm   j=1                                                   return  Yh and Ym .
where Z:j denotes the spectral representation of thejth pixel                 Fig. 3.    Summary of the preprocessing steps.
of  the  estimated  image  and ˆZ:j  denotes  the  same  for  the
ground truth image. This index is an indicator of the spectral                   After  the  preprocessing,  we  estimated  the  spectral  and
quality of the estimated image. In this paper we report the                   spatialresponsesasdescribedinSectionIV.Wethenestimated
value of the SAM index in degrees.                                            matrix E using VCA;8 since the subspace estimated by VCA
  The third index was based on the Universal Image Quality                    shares the dimension of the subspace estimated by SVD [31],
Index (UIQI), proposed by Wang et al. [53]. It was computed                   we also madeLs = 10 in this step. Since VCA has a random
on a sliding window of size 32× 32 pixels, and averaged                       component, we performed ten runs of our algorithm in each
over all window positions. Denoting by zi theith windowed                     case, and we report the average of the corresponding results.
segment of a single-band image and byˆzi the corresponding                    Their standard deviation was negligible.
segmentofasingle-bandgroundtruthimage,theUIQIisgiven                             The tuning of the values of the algorithm’s parameters is an
by                                                                            interesting and complex topic, with a number of techniques

                     M∑   σziˆzi                                              that  can  be  adapted  to  problems  such  as  this.  Two  exam-
  Q(z,ˆz) def=   1M       σziσˆzi×  2µziµˆziµ2zi +µ2ˆzi×  2σziσˆziσ2zi +σ2ˆzi,   (20)ples of these techniques are Stein’s Unbiased Risk Estimator
                     i=1                                                      (SURE) [54] and Generalized Cross-Validation (GCV) [55].
whereM  is the number of window positions,σziˆzi  is the                      We  veriﬁed  experimentally,  however,  that  as  long  as  our
covariance betweenzi andˆzi,σzi is the standard deviation of                  remote sensing images were preprocessed as described earlier,
zi, andσˆzi is the standard deviation ofˆzi. This index has a                 constant values for these parameters tended to lead to near-
range of [−1,1], being equal to 1 when z =ˆz.                                 optimal results. To choose these values, we ﬁrst found the
  The deﬁnition of the UIQI index was extended to multiband                   optimal values for each situation, and computed the corre-
hyperspectral images by simple averaging:                                     sponding quality indices. We then chose a set of parameter
                                        Lh∑                                   values that were the same for all situations, but that yielded
               UIQI(Z,ˆZ)def=   1           Q(Zl:,ˆZl:).              (21)    quality indices that were very close to the previously found
                                   Lh   l=1                                   optimal ones. These values wereλm = 1 andµ = 5×10−2.
Q was computed using the MATLAB code provided by Wang                         We usedλϕ = 10−2 when fusing a HSI with a panchromatic
et al.7                                                                       image andλϕ = 5×10−4 when fusing a HSI with a MSI. We
  When   working   with   the   fusion   of   hyperspectral   and             usedλB =λR = 10 (see Section V-D1 for how these values
panchromatic images from dataset C, we had no access to the                   were chosen). To illustrate the inﬂuence of different values of
                                                                              λϕ on the ERGAS, SAM and UIQI quality indices (including
  7Available from https://ece.uwaterloo.ca/∼z70wang/research/quality index/
demo.html.                                                                      8Available from http://www.lx.it.pt/∼bioucas/code/demo  vca.zip.

<!-- PDF PAGE 9 -->

                                                                                                                                                            9
















                      (a) ERGAS.                                         (b) SAM.                                           (c) UIQI.
Fig. 4.    Quality indices for different values of λϕ, for dataset B (HSI+PAN) fusion.


















    (a) Spatial blur between the HSI and        (b) Spectral relationship between the HSI and the      (c) Spectral relationship between the HSI and the
    the panchromatic band.                      panchromatic band.                                     MSI. Different multispectral bands are shown in
                                                                                                       different colors.
Fig. 5.    Spectral and spatial blur estimates for dataset C.


the situation when there is no regularization, i.e.,λϕ = 0),                    estimation as described in Fig. 2. The estimated blurs, which
we performed a series of experiments on the HSI+PAN fusion                      look quite reasonable, can be seen in Fig. 5.
of dataset B (see Section V-D2 for more details). The results                      2) Fusion of hyperspectral and panchromatic images:  A
of these experiments can be seen in Fig. 4; forλϕ = 0, the                      number of methods for the fusion of multiband images with
values of ERGAS, SAM and UIQI were 5.046, 6.587 and                             panchromatic ones, drawn from the pansharpening literature,
0.882, respectively.                                                            were used for comparison with HySure. Those methods were
   In  [47],  a  stopping  criterion  was  proposed  for  problems              originally built having in mind the fusion of panchromatic
solved via ADMM. We veriﬁed that this criterion worked well,                    images with multispectral ones, i.e., they were built for a small
always yielding less than 200 iterations. Given this, we ran the                number of bands, and not for the large number of bands of
algorithm for 200 iterations in every case.                                     a typical hyperspectral image. The methods can, however, be
                                                                                extended in a straightforward manner to hyperspectral images,
D. Experimental results                                                         since they have no restrictions on the number of bands. In
                                                                                what follows, a quick rundown of those methods is given.
   1) Estimation of the spatial and spectral responses of the                   A criterion used to choose the methods for comparison was
sensors:  Our ﬁrst experiments were aimed at testing the esti-                  that they should not impose restrictions on the ratio between
mation of the spectral and spatial responses of the sensors on                  the resolutions of the high spatial resolution image and the
real-life data. After checking that the results on datasets A and               low spatial resolution one. Since our method does not impose
B were rather accurate, we chose the values ofλB andλR that                     such restrictions, we only compared it against similarly built
yielded the highest-quality results on those datasets. We then                  methods.
tested the estimation method, with those parameter values, on                      In [6], Amro et al. divided the pansharpening methods into
dataset C. In the estimation of the spectral response, we took                  several categories. One of them is the Component Substitution
into account the available information on the overlap between                   family;  different  methods  from  this  family  were  tested  in
the hyperspectral bands and the multispectral and panchro-                      this work. They are characterized by the transformation of
matic ones. Since the original hyperspectral and multispectral                  the  multispectral  bands  into  a  set  of  components,  usually
images of this dataset have the same resolution, for that pair                  through a linear transformation. After this, a component of
of images we have set B = I, corresponding to no spatial                        the transformed multiband image is replaced with an image
blur,  and  we  have  just  estimated R,  without  applying  any                derived  from  the  panchromatic  one,  and  then  the  transfor-
additional  spatial  blur.  For  the  hyperspectral+panchromatic                mation is undone. These methods work well only when the
images, which have different resolutions, we performed the                      spectra of the two data sources almost overlap, a condition

<!-- PDF PAGE 10 -->

                                                                                                                                                       10



















                 (a) Panchromatic image.                  (b) Hyperspectral image (false color).            (c) HySure’s result (false color).
Fig. 6.    Hyperspectral + panchromatic fusion on dataset A.

which  may  not  be  fulﬁlled  when  fusing  panchromatic  and                                                 TABLE I
                                                                                            RESULTS FOR DATASET A (HSI+PAN FUSION).
hyperspectral  images.  The  Gram-Schmidt  adaptive  (GSA)
method from Aiazzi et al. [56] is an adaptation of the Gram-                                                 ERGAS      SAM       UIQI
Schmidt spectral sharpening method (GS). The latter is based                                     GS           1.330     1.136    0.868
on  the  Gram-Schmidt  transformation  of  the  different  low                                   GSA          1.268     1.156    0.870
spatial resolution bands, followed by the substitution of the                                    FIHS         1.788     1.456    0.863
ﬁrst band of the transformed image with a modiﬁed version                                        PCA          1.451     1.149    0.865
of the panchromatic band. This modiﬁed version is given by                                       BT           1.832     1.427    0.875
a weighted sum of the multispectral bands, expanded to the                                       HPF          3.277     1.688    0.845
spatial resolution of the panchromatic image. The weights are                                    HySure       0.717     0.524    0.895
obtained in different ways, and that is the main difference
between GS and GSA. In GS, they are assumed to be the
same for all bands, while in GSA they are estimated from                         Fig. 6 and Table I show the results of the various methods
the observed data, usually guaranteeing better results. In the                for dataset A. We only show the results for the Starck-Murtagh
case of fusion with hyperspectral images, GSA involves the                    blur, since the results for the other two blurs were very similar
inversionofamatrixthatisclosetosingular,possiblyaffecting                     to these. The results for dataset B (again, just for the Starck-
the quality of the results. Nevertheless, as will be seen later,              Murtagh blur)can be seenin Fig. 7and Table II.The evolution
the  experiments  showed  an  improvement  of  GSA  relative                  of the cost function (6) during the optimization is shown in
to GS. The Fast Intensity-Hue-Saturation Fusion Technique                     Fig. 7e. Fig. 7f shows the root-mean-square error (RMSE)
(FIHS) is another method included in this family. It is similar               betweentheestimatedimageandthegroundtruthasafunction
to GSA and GS, with the difference that the processing is                     of  band  wavelength  for  the  three  best  methods.  Dataset  C
made in the IHS color space, with the panchromatic image                      allowed us to evaluate the methods on real-life data. Fig. 8
replacingtheintensitycomponentofthemultibandimage[57].                        shows the results.
Another method relies on the Principal Component Analysis                        The proposed method outperformed the other ones in all
(PCA) of the multiband image, and replaces the ﬁrst principal                 cases, except for the SAM index in dataset B, in which it was
component with the panchromatic image [57].                                   surpassed by BT. We found that most published pansharpening

  Another family of methods is the Relative Spectral Contri-                  methods seem to not deal well with the fact that the panchro-
bution family. An example is the Brovey Transform method                      matic image’s spectral range does not overlap a large number
(BT),basedonthechromaticitytransform[57].Inthismethod,                        of hyperspectral channels.
each pixel of the estimated image is given by the correspond-                    3) Fusion of hyper- and multispectral images:  The litera-
ing pixel of the panchromatic image, weighted by a linear                     ture on the fusion of hyperspectral and multispectral images
combination of the values of the different spatially expanded                 is much sparser than the one on pansharpening. As a con-
multispectral bands for this same pixel. Finally, another family              sequence, we were only able to perform comparisons with
is the High-Frequency Injection one, from which we used the                   one published method: we had access to an implementation
Box High-pass Filtering method (HPF). It is characterized by                  of  a  method  by  Zhang  et  al.  [15]  (henceforth,  designated
the extraction of high frequency information from the high                    by ZBS) and used it for comparisons on datasets B and C.
spatial  resolution  image,  followed  by  the  injection  of  this           This implementation needed the input HSIs and MSIs to be
information  into  the  multiband  image  [6].  To  perform  the              represented with the same spatial resolution. Therefore, we
high frequency extraction, the method starts by producing a                   upsampled  the  HSIs  to  the  resolution  of  the  MSIs,  using
low-pass version of the panchromatic image through a box                      bicubic  interpolation,  for  input  to  ZBS.  This  method  does
ﬁltering operation. This blurred image is then subtracted from                not estimate the spatial blur, needing it to be speciﬁed; we
the original one, yielding a high frequency version of it.                    estimated it as in our method, with the difference that we

<!-- PDF PAGE 11 -->

                                                                                                                                                                   11

                                                                             TABLE II
                                                         RESULTS FOR DATASET B (HSI+PAN FUSION).

                           SNR(Ym)                    40 dB                           30 dB                             20 dB
                           SNR(Yh)                    30 dB                           20 dB                             20 dB
                                           ERGAS        SAM      UIQI       ERGAS       SAM       UIQI      ERGAS        SAM        UIQI
                           GS               4.960      5.494     0.897       5.112      5.914     0.885       5.879      6.303      0.838
                           GSA              4.587      5.116     0.905       4.733      5.538     0.893       5.448      5.804      0.848
                           FIHS             4.813      5.255     0.905       4.962      5.669     0.894       5.661      5.908      0.848
                           PCA              7.609      9.448     0.774       7.712      9.711     0.766       8.280      10.020     0.730
                           BT               4.533      4.550     0.926       4.684      4.989     0.915       5.311      4.983      0.874
                           HPF              5.573      6.151     0.880       5.731      6.631     0.867       6.527      7.111      0.814
                           HySure           3.813      4.856     0.937       3.894      4.938     0.933       4.630      5.507      0.897


























       (a) Observed HSI (false color).        (b) Observed panchromatic im-           (c)    HySure’s    result    (false      (d) BT’s result (false color).
                                              age.                                    color).





















              (e) Evolution of the cost function during the optimization.                 (f)  Root-mean-square  error  (RMSE)  between  the  estimated
                                                                                          image and the ground truth, for the different bands (for the
                                                                                          three best methods).
Fig. 7.    Results for dataset B (HSI+PAN fusion).


worked with the upsampled version of the HSI. Following the                         to  us,  we  only  worked  on  a  section  of  the  image  with
lead of that method’s authors, we chose the decomposition                           200× 200 pixels, corresponding to the bottom left corner.
level of the Nondecimated Wavelet Transform to be three.                            For this dataset, as an illustration of the processing speed,
   The  results  of  these  tests  are  shown  in  Figs.  9  and  10                the proposed method took about 35 seconds to perform the
and  in  Tables  III  and  IV.  The  proposed  method  surpassed                    fusion in a MATLAB implementation running on an Intel R⃝
the other one in all tests. For dataset B, and due to input                         Xeon R⃝ CPU at 3.20 GHz with 16 GB of RAM. For dataset C,
restrictions of the implementation of ZBS that was available

<!-- PDF PAGE 12 -->

                                                                                                                                                                 12

















   (a)  Observed  panchromatic           (b) Observed HSI.              (c) HySure’s result.             (d) GSA’s result.               (e) GS’s result.
   image.















               (f) HPF’s result.                      (g) BT’s result.                     (h) FIHS’s result.                     (i) PCA’s result.
Fig. 8.    Results for dataset C (HSI+PAN fusion). All images, except (a), are in false color.















       (a) Observed multispectral im-         (b) Observed hyperspectral im-               (c) HySure’s result.                    (d) ZBS’s result.
       age.                                   age.
Fig. 9.    Results for dataset B (HSI+MSI fusion). All images are in false color. Figs. 9c and 9d are very similar to Fig. 9a due to the false color rendering,
but they have 93 bands, while Fig. 9a has only four.


                                  TABLE III                                                                          TABLE IV
              RESULTS FOR DATASET B (HSI+MSI FUSION).                                             RESULTS FOR DATASET C (HSI+MSI FUSION).

                                ERGAS       SAM       UIQI                                                          ERGAS       SAM       UIQI
                   ZBS           5.919      4.375     0.881                                            ZBS           5.011      3.672     0.725
                   HySure        1.213      1.956     0.995                                            HySure        4.101      3.092     0.840


                                                                                                              VI.  CONCLUSIONS
                                                                                      We have proposed a method, termed HySure, to perform
we worked on a section with 72×72 pixels. Fig. 11b shows                           the fusion of hyperspectral images with either panchromatic
the RMSE between the ground truth and the results of both                          or multispectral ones, with the goal of obtaining images which
methods, for each pixel, with the pixels sorted in order of                        have  high  resolution  in  both  the  spatial  and  the  spectral
ascending error; we are only showing results corresponding to                      domains. This problem is closely related to the pansharpening
theﬁrst99%oftheerrors,sincetheotherpixelsareverynoisy.                             one, but presents new challenges due to the much larger size
Figs. 11c, 11d and 11e compare the reﬂectance values of the                        of hyperspectral images when compared with the multispectral
resultsofthetwomethodswiththegroundtruthones,forthree                              images normally used in pansharpening and to the fact that
pixels, corresponding to the 10th, 50th and 90th percentiles of                    the different images do not normally have a complete spectral
the error of our method, respectively.                                             overlap. In addition to performing the fusion, the proposed

<!-- PDF PAGE 13 -->

                                                                                                                                                                   13












              (a)   Observed   multi-               (b)  Observed  hyper-                  (c) HySure’s result.                    (d) ZBS’s result.
              spectral image.                       spectral image.
Fig. 10.   Results for dataset C (HSI+MSI fusion). All images are in false color. Figs. 10c and 10d are very similar to Fig. 10a due to the false color rendering,
but they have 128 bands, while Fig. 10a has only nine.

















              (a) RMSE between the the results of both methods                                   (b) RMSE between the the results of both methods
              and the ground truth, per pixel, for dataset B.                                    and the ground truth, per pixel, for dataset C.













     (c) Reﬂectance of the pixel corresponding               (d) Reﬂectance of the pixel corresponding              (e) Reﬂectance of the pixel corresponding
     to the 10th percentile of the RMSE (dataset             to the 50th percentile of the RMSE (dataset            to the 90th percentile of the RMSE (dataset
     B).                                                     B).                                                    B).
Fig. 11.   Results for datasets B and C (HSI+MSI fusion). The results in Figs. 11a and 11b are in ascending order. Figs. 11c, 11d and 11e show the reﬂectance
values of the pixels corresponding to the 10th, 50th and 90th percentiles, respectively, for dataset B.


method is also able to estimate the relative spectral and spatial                   expanding (9) in its different components:
responses of the sensors from the data.
   We formulated the fusion problem as a convex program,
solved  via  the  Split  Augmented  Lagrangian  Shrinkage  Al-
gorithm (SALSA)—an instance of the Alternating Direction
MethodofMultipliers(ADMM).Theestimationoftherelative
responses of the sensors was formulated as a convex quadratic                         L(X,V1,V2,V3,V4,A1,A2,A3,A4)
program. Taking advantage of the low intrinsic dimensionality                                          Yh−EV1M2                 XB−V1−A12
of  hyperspectral  images  by  working  on  a  subspace  of  the                                 = 12                               2
space where those images are deﬁned, and using an adequate                                                   Ym−REV2F +µ              X−V2−A2F
                                                                                                                                   2                              2
variable splitting, we obtained an effective algorithm which                                        +λm2                           F +µ2                          F
compares quite favorably with several published methods on                                                                       XDh−V3−A32
both simulated and real-life data.                                                                  +λϕϕ(V3,V4)+µ2                                          F
                                                                                                          XDv−V4−A42
                                                                                                    +µ2                              F.
                                 APPENDIX                                                                                                                       (22)

   In  this  Appendix  we  show  in  detail  how  to  solve  the                        The optimization algorithm, which was given in condensed
optimization problem described in Section III. We start by                          form in Fig. 1, is given in more detail in Fig. 12.

<!-- PDF PAGE 14 -->

                                                                                                                                   14


   k := 0                                                          and                         (                  )
   repeat                  (                                                     V(k+1)M =      X(k+1)B−A(k)        M.             (25)
                                                                                   1                           1
      X(k+1)∈argmin         X,V(k)1 ,···,V(k)4 ,
                    X L            )                               [ETE+µI]−1 and ETYh can be precomputed. The compu-
                   A(k)1 ,···,A(k)4                                tations can be efﬁciently via the FFT, and have complexity
     fori = 1,···,4 do       (                                     O(Ls×nmlognm).
        V(k+1)i  ∈argmin       X(k+1),V(k)1 ,···,                     The minimization
                      Vi L                           )                                            Ym−REV22
                   V(k)                                                    V(k+1)              λm
                     i ,···,V(k)4 ,A(k)1 ,···,A(k)4                          2  ∈ argminV2      2                   F
     end for             (                       )                                                X(k+1)−V2−A(k)2
     A(k+1)                X(k+1)B−V(k+1)                                                     +µ2                       2   F
        1       :=A(k)1− (                1   )
     A(k+1)2       :=A(k)2−(X(k+1)−V(k+1)2        )                has the solution
     A(k+1)3       :=A(k)3−X(k+1)Dh−V(k+1)3                                      [                     ]−1
                         (                        )                   V(k+1)2       =λmETRTRE+µI
     A(k+1)4       :=A(k)4−X(k+1)Dv−V(k+1)4                                         [                    (                )]    (26)
     k :=k+1                                                                         λmETRTYm+µ           X(k+1)−A(k)2       ,
   until stopping criterion is satisﬁed.                                         (                )
Fig. 12.    Optimization algorithm.                                where only     X(k+1)−A(k)2       cannot be precomputed. The
                                                                   complexity of this part isO(Ls×nm).
                                                                      V3  and V4  are  computed  by  solving  the  minimization
  The ﬁrst minimization problem is                                 problem
                             XB−V(k)                                {                  }
       X(k+1)∈ argmin      µ                         2                      V(k+1)3  ,V(k+1)4  ∈ argmin
                            2           1−A(k)1      F                                             V3,V4λϕϕ(V3,V4)
                      X      X−V(k)                                                      X(k+1)Dh−V3−A(k)2
                                                   2                                      +µ
                         +µ2          2−A(k)2      F                                         2 X(k+1)Dv−V3−A(k)2       F
                             XDh−V(k)             2                                 +µ                               2
                         +µ               3−A(k)2                                            2                         3   F,
                            2XDv−V(k)             F
                                                      2            whosesolutionisgivenbyacolumn-wisevector-softthreshold
                         +µ2              4−A(k)3     F,           function [54],

which has the solution                                                          {(         )   (        )  }
             [                               ]−1                                   V(k+1)3      V(k+1)4       =
  X(k+1) =    BBT +I+DhDTh +DvDTv                                                           :j,{CF−λϕ:j    }   C            (27)
             [(              )       (              )                                = max                µ,0    CF ,
                V(k)1   +A(k)1 BT +    V(k)2   +A(k)2
                (              )        (              )    ]
             +    V(k)3   +A(k)3 DTh +   V(k)4   +A(k)4 DTv  .     where
                                                            (23)            {(                     )   (                    )  }
The computation can be efﬁciently performed through the use           C =      X(k+1)Dh−A(k)3       :j,  X(k+1)Dv−A(k)4       :j ,
of  the  Fast  Fourier  Transform,  having  complexityO(Ls×        and (.):j denotes thejth column of a matrix. We follow the
nmlognm). The ﬁrst term on the right hand side, including          convention that 0/||0||F = 0. The complexity of computing
the inverse, can be computed in advance, before the iteration.     V3 andV4 isO(Ls×nmlognm), being dominated by FFTs.
  To solve the minimization problem involving V1,                     After performing these optimizations, the following equa-
                                1Yh−EV1M2                    tions are used to update the Lagrange multipliers:
            V(k+1)1  ∈ argmin   2                  F
                          V1                                                                     (                     )
                      X(k+1)B−V1−A(k)2                                 A(k+1)               X(k+1)B−V(k+1)         ,
                 +µ                          1                                  1       =A(k)1−  (                1  )
                    2                             F,                         A(k+1)               X(k+1)−V(k+1)        ,
we can take advantage of the masking matrix M to separate                       2       =A(k)2−  (             2         )
V1 into V1M and V1M, where         M is the matrix that selects              A(k+1)3       =A(k)3−X(k+1)Dh−V(k+1)3        ,
the pixels not selected by M. We then have                                                       (                       )
                [            ]−1                                             A(k+1)4       =A(k)4−X(k+1)Dv−V(k+1)4        .
   V(k+1)1     M =ETE+µI[       (                   )]      (24)
                                                                      The complexity of the algorithm is dominated by the FFTs,
                    ETYh+µ       X(k+1)B−A(k)1        M            and isO(Ls×nmlognm) per iteration.

<!-- PDF PAGE 15 -->

                                                                                                                                                                                            15


                              ACKNOWLEDGMENT                                                     [21]G. Licciardi, M. Khan, J. Chanussot, A. Montanvert, L. Condat, and
   We  gratefully  acknowledge  Prof.  Paolo  Gamba  for  pro-                                          C.  Jutten,  “Fusion  of  hyperspectral  and  panchromatic  images  using
                                                                                                        multiresolution analysis and nonlinear PCA band reduction,” EURASIP
viding dataset B, Dr. Giorgio Licciardi for providing dataset                                           J. Adv. Signal Process., vol. 2012, no. 1, p. 207, 2012.
C, and Qi Wei for providing dataset D. We also gratefully                                        [22]T. Akgun, Y. Altunbasak, and R. Mersereau, “Super-resolution recon-
acknowledge Dr. Yifan Zhang for providing the source code                                               struction of hyperspectral images.” IEEE Trans. Image Process., vol. 14,
                                                                                                        no. 11, pp. 1860–1875, 2005.
for [15] and Dr. Gemine Vivone for providing the source code                                     [23]Y. Zhao, J. Yang, Q. Zhang, L. Song, Y. Cheng, and Q. Pan, “Hyper-
for the different pansharpening algorithms (GS, GSA, FIHS,                                              spectral imagery super-resolution by sparse representation and spectral
PCA, BT an HPF) [58].                                                                                   regularization,”EURASIPJ.Adv.SignalProcess.,vol.2011,no.1,p.87,
                                                                                                        2011.
                                                                                                 [24]X. Bresson and T. Chan, “Fast dual minimization of the vectorial total
                                    REFERENCES                                                          variation  norm  and  applications  to  color  image  processing,”  Inverse
                                                                                                        Probl. and Imag., vol. 2, no. 4, pp. 455–484, 2008.
 [1]J. Bioucas-Dias, A. Plaza, G. Camps-Valls, P. Scheunders, N. Nasrabadi,                      [25]M. Afonso, J. Bioucas-Dias, and M. Figueiredo, “An augmented La-
      andJ.Chanussot,“Hyperspectralremotesensingdataanalysisandfuture                                   grangian approach to the constrained optimization formulation of imag-
      challenges,” IEEE Geosci. Remote Sens. Mag., vol. 1, no. 2, 2013.                                 ing inverse problems.” IEEE Trans. Image Process., vol. 20, no. 3, pp.
 [2]E.  Middleton,  S.  Ungar,  D.  Mandl,  L.  Ong,  S.  Frye,  P.  Campbell,                          681–95, 2011.
      D. Landis, J. Young, and N. Pollack, “The Earth Observing One (EO-                         [26]M. Sim      ˜oes, J. Bioucas-Dias, L. Almeida, and J. Chanussot, “Hyperspec-
      1) satellite mission: over a decade in space,” IEEE J. Sel. Topics Appl.                          tral image superresolution: An edge-preserving convex formulation,” in
      Earth Observ. and Remote Sens., vol. 6, no. 2, pp. 243–256, 2013.                                 IEEE Int. Conf. Image Processing, Paris, 2014.
 [3]H. Kramer,      Observation of the Earth and its Environment.     Berlin:                    [27]M. Almeida and M. Figueiredo, “Deconvolving images with unknown
      Springer, 1994.                                                                                   boundaries using the alternating direction method of multipliers.” IEEE
 [4]L.Alparone,L.Wald,J.Chanussot,C.Thomas,P.Gamba,andL.Bruce,                                          Trans. Image Process., vol. 22, no. 8, pp. 3074–86, Aug. 2013.
      “Comparison of pansharpening algorithms: Outcome of the 2006 GRS-S                         [28]A. Matakos, S. Ramani, and J. Fessler, “Accelerated edge-preserving im-
      data-fusion contest,” IEEE Trans. Geosci. Remote Sens., vol. 45, no. 10,                          age restoration without boundary artifacts,” IEEE Trans. Image Process.,
      pp. 3012–3021, 2007.                                                                              vol. 22, no. 5, pp. 2019–2029, 2013.
 [5]C.  Thomas,  T.  Ranchin,  L.  Wald,  and  J.  Chanussot,  “Synthesis  of                    [29]J. Bioucas-Dias, A. Plaza, N. Dobigeon, M. Parente, Q. Du, P. Gader,
      multispectral  images  to  high  spatial  resolution:  A  critical  review  of                    and  J.  Chanussot,  “Hyperspectral  unmixing  overview:  Geometrical,
      fusion methods based on remote sensing physics,” IEEE Trans. Geosci.                              statistical, and sparse regression-based approaches,” IEEE J. Sel. Topics
      Remote Sens., vol. 46, no. 5, pp. 1301–1312, 2008.                                                Appl. Earth Observ. and Remote Sens., vol. 5, no. 2, pp. 354–379, 2012.
 [6]I. Amro, J. Mateos, M. Vega, R. Molina, and A. Katsaggelos, “A survey                        [30]K.Cawse-Nicholson,S.Damelin,A.Robin,andM.Sears,“Determining
      of classical methods and new trends in pansharpening of multispectral                             the intrinsic dimension of a hyperspectral image using random matrix
      images,” EURASIP J. Adv. Signal Process., vol. 2011, no. 1, pp. 79–100,                           theory.” IEEE Trans. Image Process., vol. 22, no. 4, pp. 1301–1310,
      2011.                                                                                             2013.
 [7]J. Greer, “Sparse demixing of hyperspectral images.”      IEEE Trans. Image                  [31]J. Nascimento and J. Bioucas-Dias, “Vertex component analysis: A fast
      Process., vol. 21, no. 1, pp. 219–228, 2012.                                                      algorithm to unmix hyperspectral data,” IEEE Trans. Geosci. Remote
 [8]A. Charles, B. Olshausen, and C. Rozell, “Learning sparse codes for                                 Sens., vol. 43, no. 4, pp. 898–910, 2005.
      hyperspectralimagery,”IEEEJ.Sel.TopicsSignalProcess.,vol.5,no.5,                           [32]J. Bioucas-Dias and J. Nascimento, “Hyperspectral subspace identiﬁca-
      pp. 963–978, 2011.                                                                                tion,” IEEE Trans. Geosci. Remote Sens., vol. 46, no. 8, pp. 2435–2445,
 [9]R. Zurita-Milla, J. Clevers, and M. Schaepman, “Unmixing-based Land-                                2008.
      sat TM and MERIS FR data fusion,” IEEE Geosci. Remote Sens. Lett.,                         [33]L. Rudin, S. Osher, and E. Fatemi, “Nonlinear total variation based noise
      vol. 5, no. 3, pp. 453–457, 2008.                                                                 removal algorithms,” Phys. D, vol. 60, pp. 259–268, 1992.
[10]R.  Kawakami,  J.  Wright,  Y.  Tai,  Y.  Matsushita,  M.  Ben-Ezra,  and                    [34]A.  Chambolle,  “An  algorithm  for  total  variation  minimization  and
      K. Ikeuchi, “High-resolution hyperspectral imaging via matrix factor-                             applications,” J. Math. Imag. Vis., no. 20, pp. 89–97, 2004.
      ization,” in Proc. 2011 IEEE Computer Society Conf. Computer Vision,                       [35]Y. Wang, J. Yang, W. Yin, and Y. Zhang, “A new alternating minimiza-
      Providence, RI, Jun. 2011, pp. 2329–2336.                                                         tionalgorithmfortotalvariationimagereconstruction,”SIAMJ.Imaging
[11]B. Huang, H. Song, H. Cui, J. Peng, and Z. Xu, “Spatial and spectral                                Sci., vol. 1, no. 3, pp. 248–272, 2008.
      imagefusionusingsparsematrixfactorization,”IEEETrans.Geo.Remo.                             [36]M. Tao and J. Yang,       Alternating direction algorithms for total variation
      Sens., vol. 52, no. 3, pp. 1693–1704, 2013.                                                       deconvolution in image reconstruction.     Dept. Math., Nanjing Univ.,
[12]H. Song, B. Huang, K. Zhang, and H. Zhang, “Spatio-spectral fusion of                               Tech. Rep.: TR0918, 2009.
      satellite images based on dictionary-pair learning,” in Informat. Fusion.                  [37]F.        ˇSroubek and P. Milanfar, “Robust multichannel blind deconvolution
      http:// dx.doi.org/10.1016/j.inffus.2013.08.005, 2013.                                            via fast alternating minimization.” IEEE Trans. Image Process., vol. 21,
[13]N.  Yokoya,  T.  Yairi,  and  A.  Iwasaki,  “Coupled  nonnegative  matrix                           no. 4, pp. 1687–700, Apr. 2012.
      factorization unmixing for hyperspectral and multispectral data fusion,”                   [38]L.  Xu  and  J.  Jia,  “Two-phase  kernel  estimation  for  robust  motion
      IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528–537, 2012.                              deblurring,” in Proc. 11th European Conference on Computer Vision,
[14]R. Hardie, M. Eismann, and G. Wilson, “MAP estimation for hyper-                                    Crete, Greece, 2010, pp. 157–170.
      spectral image resolution enhancement using an auxiliary sensor,” IEEE                     [39]P. Campisi and K. Egiazarian,        Blind image deconvolution: theory and
      Trans. Image Process., vol. 13, no. 9, pp. 1174–1184, 2004.                                       applications.   Boca Raton, FL: CRC Press, 2007.
[15]Y. Zhang, S. De Backer, and P. Scheunders, “Noise-resistant wavelet-                         [40]A.  Beck  and  M.  Teboulle,  “Fast  gradient-based  algorithms  for  con-
      based bayesian fusion of multispectral and hyperspectral images,” IEEE                            strained total variation image denoising and deblurring problems,” IEEE
      Trans. Geosci. Remote Sens., vol. 47, no. 11, pp. 3834–3843, 2009.                                Trans. Image Process., vol. 18, no. 11, pp. 2419–2434, 2009.
[16]Y.  Zhang,  A.  Duijster,  and  P.  Scheunders,  “A  bayesian  restoration                   [41]X. Zhao, F. Wang, T. Huang, M. Ng, and R. Plemmons, “Deblurring and
      approach for hyperspectral images,” IEEE Trans. Geosci. Remote Sens.,                             sparse unmixing for hyperspectral images,” IEEE Trans. Geosci. Remote
      vol. 50, no. 9, pp. 3453–3462, 2012.                                                              Sens., vol. 51, no. 7, 2013.
[17]Q. Wei, N. Dobigeon, and J. Tourneret, “Bayesian fusion of hyperspec-                        [42]X. He, L. Condat, J. Bioucas-Dias, J. Chanussot, and J. Xia, “A new
      tral and multispectral images,” in Proc. 2014 IEEE Int. Conf. Acoustics,                          pansharpening method based on spatial and spectral sparsity priors.”
      Speech and Signal Processing, Florence, Italy, 2014, pp. 3176–3180.                               IEEE Trans. Image Process., vol. PP, no. 99, pp. 1–1, 2014.
[18]S. Chen, Q. Guo, H. Leung, and E. Bosse, “A maximum likelihood                               [43]Q.  Yuan,  L.  Zhang,  and  H.  Shen,  “Hyperspectral  image  denoising
      approach to joint image registration and fusion.” IEEE Trans. Image                               employing  a  spectral–spatial  adaptive  total  variation  model,”  IEEE
      Process., vol. 20, no. 5, pp. 1363–1372, 2011.                                                    Trans. Geosci. Remote Sens., vol. 50, no. 10, pp. 3660–3677, 2012.
[19]M. Khan, J. Chanussot, and L. Alparone, “Pansharpening of hyperspec-                         [44]A.  Chambolle  and  T.  Pock,  “A  ﬁrst-order  primal-dual  algorithm  for
      tral images using spatial distortion optimization,” in Proc. 2009 IEEE                            convex problems with applications to imaging,” J. Math. Imaging Vis.,
      Int. Conf. Image Processing, Cairo, 2009, pp. 2853–2856.                                          vol. 40, no. 1, pp. 120–145, 2011.
[20]A. Garzelli, B. Aiazzi, S. Baronti, M. Selva, and L. Alparone, “Hyper-                       [45]L.  Condat,  “A  primal-dual  splitting  method  for  convex  optimization
      spectral image fusion,” in Proc. Hyperspectral 2010 Workshop, Frascati,                           involving  Lipschitzian,  proximable  and  linear  composite  terms,”  J.
      Italy, 2010, pp. 17–19.                                                                           Optimiz. Theory Appl., vol. 158, no. 2, pp. 460–479, 2013.

<!-- PDF PAGE 16 -->

                                                                                                                                                                                               16


[46]J.NocedalandS.Wright,       NumericalOptimization.  NewYork:Springer,
      2006.
[47]S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, “Distributed
      optimization and statistical learning via the alternating direction method
      of multipliers,” Found. Trends Mach. Learn., vol. 3, no. 1, pp. 1–122,
      2011.
[48]P.  Combettes  and  V.  Wajs,  “Signal  recovery  by  proximal  forward-
      backward splitting,” Multiscale Model. Sim., vol. 4, no. 4, pp. 1168–
      1200, 2005.
[49]T. Wang, G. Yan, H. Ren, and X. Mu, “Improved methods for spectral
      calibration  of  on-orbit  imaging  spectrometers,”  IEEE  Trans.  Geosci.
      Remote Sens., vol. 48, no. 11, pp. 3924–3931, 2010.
[50]N. Yokoya, N. Mayumi, and A. Iwasaki, “Cross-calibration for data
      fusion of EO-1/Hyperion and Terra/ASTER,” IEEE J. Sel. Topics Appl.
      Earth Observ. and Remote Sens., vol. 6, no. 2, pp. 419–426, 2013.
[51]J. Starck and F. Murtagh, “Image restoration with noise suppression
      using the wavelet transform,” Astron. Astrophys., vol. 288, pp. 342–348,
      1994.
[52]L.  Wald,  “Quality  of  high  resolution  synthesised  images:  Is  there  a
      simple criterion?” in Proc. 3rd Conf. Fusion of Earth Data: Merging
      Point Measurements, Raster Maps and Remotely Sensed Images, Sophia
      Antipolis, 2000, pp. 99–103.
[53]Z. Wang and A. Bovik, “A universal image quality index,”        IEEE Signal
      Process. Lett., vol. 9, no. 3, pp. 81–84, 2002.
[54]D. Donoho and I. Johnstone, “Adapting to unknown smoothness via
      wavelet shrinkage,” J. Am. Statist. Assoc., vol. 90, no. 432, pp. 1200–
      1224, 1995.
[55]G. Golub, M. Heath, and G. Wahba, “Generalized cross-validation as a
      method for choosing a good ridge parameter,” Technometrics, vol. 21,
      no. 2, pp. 215–223, 1979.
[56]B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution
      pansharpening through multivariate regression of MS+Pan data,” IEEE
      Trans. Geosci. Remote Sens., vol. 45, no. 10, pp. 3230–3239, 2007.
[57]T. Tu, P. Huang, C. Hung, and C. Chang, “A fast intensity hue saturation
      fusion technique with spectral adjustment for IKONOS imagery,” IEEE
      Geosci. Remote Sens. Lett., vol. 1, no. 4, pp. 309–312, 2004.
[58]G. Vivone,        Multispectral and Hyperspectral Pansharpening: A Critical
      Examination  and  New  Developments.      University  of  Salerno:  PhD
      dissertation, 2014.

<!-- END-OF-PAPER:ae722f549f12 -->
