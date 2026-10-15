

<!-- PDF PAGE 1 -->

      Hyperspectral super-resolutionof locally low rank images
       fromcomplementary multisourcedata (extended report)
   MiguelAngelVeganzones,MiguelSimoes,GiorgioLicciardi,JoséM.Bioucas-Dias,
                                             Jocelyn Chanussot



      Tocite this version:

    MiguelAngelVeganzones,MiguelSimoes,GiorgioLicciardi,JoséM.Bioucas-Dias,JocelynChanussot. Hyper-
    spectralsuper-resolutionoflocallylowrankimagesfromcomplementarymultisourcedata(extendedreport).
    2014. ⟨hal-00960076v2⟩







                                      HAL Id: hal-00960076
                             https://hal.science/hal-00960076v2

                                         Submitted on 19 Mar 2014




    HAL is a multi-disciplinary open access archive        L’archiveouvertepluridisciplinaireHAL,estdes-
for the deposit and dissemination of scientific re-    tinée au dépôt et à la diffusion de documents scien-
searchdocuments,whethertheyarepublishedornot.          tifiquesdeniveaurecherche,publiésounon,émanant
Thedocumentsmaycomefromteachingandresearch             des établissements d’enseignement et de recherche
institutionsinFranceorabroad,orfrompublicorpri-        français  ou  étrangers,  des  laboratoires  publics  ou
vateresearchcenters.                                   privés.


                                            HALAuthorization

<!-- PDF PAGE 2 -->

HYPERSPECTRAL SUPER-RESOLUTION OF
      LOCALLY LOW RANK IMAGES FROM
  COMPLEMENTARY MULTISOURCE DATA

         M.A. Veganzones1∗, M. Sim˜oes1,2, G. Licciardi1, J. Bioucas2 and J. Chanussot1,3
                         1GIPSA-lab, Grenoble-INP, Saint Martin d’H`eres, France
           2Instituto de Telecomunica¸c˜oes, Instituto Superior T´ecnico, Lisbon, Portugal
 3Faculty of Electrical and Computer Engineering, University of Iceland, Reykjavik, Iceland

                                    March 19, 2014




                                         Abstract
          Remote sensing hyperspectral images (HSI) are quite often locally low
      rank, in the sense that the spectral vectors acquired from a given spa-
      tial neighborhood belong to a low dimensional subspace/manifold.  This
      has been recently exploited for the fusion of low spatial resolution HSI
      with high spatial resolution multispectral images (MSI) in order to ob-
      tain super-resolution HSI. Most approaches adopt an unmixing or a ma-
      trix factorization perspective.  The derived methods have led to state-of-
      the-art results when the spectral information lies in a low dimensional
      subspace/manifold.   However,  if  the  subspace/manifold  dimensionality
      spanned by the complete data set is large, the performance of these meth-
      ods decrease mainly because the underlying sparse regression is severely
      ill-posed.  In this paper, we propose a local approach to cope with this
      diﬃculty.   Fundamentally,  we exploit the fact that real world HSI are
      locally low rank, to partition the image into patches and solve the data
      fusion problem independently for each patch. This way, in each patch the
      subspace/manifold dimensionality is low enough to obtain useful super-
      resolution.  We explore two alternatives to deﬁne the local regions, using
      sliding windows and binary partition trees.  The eﬀectiveness of the pro-
      posed approach is illustrated with synthetic and semi-real data.

1    Introduction
In recent years,  there has been a huge improvement of spectral and spatial
resolutions in the design of remote sensing sensors.  However, it seems that it
is not possible to achieve both of them simultaneously.   This is due, on the

  ∗This work was supported by the Delegation G´en´erale de l’Armement (DGA) under con-
tract PDE2012600079.


                                              1

<!-- PDF PAGE 3 -->

one hand, to the system trade-oﬀ related to data volume and signal-to-noise
ratio (SNR) limitations and, on the other hand, to the speciﬁc requirements of
diﬀerent applications [1]. Then, there is a need for super-resolution techniques
that fuse high spectral resolution images, such as hyperspectral images (HSI),
with high spatial images such as multispectral images (MSI) or panchromatic
images, in order to obtain high spectral and spatial (super-resolution) images.
    Recently, some techniques dedicated to the fusion of HSIs and MSIs have
been proposed.  A general trend is to associate this problem with either linear
spectral unmixing, which assumes that the underlying data can be described
by a mixture of a relatively small number of ”pure” spectral signatures, corre-
sponding to the materials present in the scene or, with the learning of a spec-
tral dictionary that codiﬁes the information present on the images. Since both
HSIs and MSIs capture the same scene, the underlying materials (the so-called
endmembers) or the dictionaries should be the same.  Therefore, the spectral
information extracted from one of the images should also be able to explain the
other one. Due to the high spectral resolution of the HSIs, the endmembers or
the dictionary are extracted from these data, and are then used to reconstruct
the MSI. Since MSIs have high spatial resolution, the ﬁnal reconstructed image
will be at super-resolution.

1.1    Related work
Zurita et al. [2], introduced one of the ﬁrst unmixing-based approaches to the fu-
sion of remote sensing multiband images. A related approach is proposed in [3],
where a very high-resolution hyperspectral image is estimated from a lower-
resolution hyperspectral image and a high-resolution RGB image. The method
starts by identifying an unmixing matrix used to represent the hyperspectral
spectra and then uses this matrix in conjunction with the RGB input to com-
pute, via sparse regression, representation coeﬃcients for the high-resolution
hyperspectral image. This methodology can be viewed as a factorization of the
input into a mixing matrix and a set of maximally sparse coeﬃcients.  An ap-
proach with similar ﬂavour is proposed in [4].  The main diﬀerence is that the
mixing matrix is replaced by a dictionary learnt using a non-negative matrix
factorization with sparsity regularization on the code.  In [5], the hyperspec-
tral data is unmixed via the K-SVD algorithm, and the multispsectral data is
reconstructed using orthogonal matching pursuit to induce sparsity.  Authors
in [1] proposed a method where two dictionaries were learned from the two
diﬀerent data,  and then used a dictionary-pair learning method to establish
the correspondence between them.  A similar and older technique introduced
in [6] alternately unmixes both sources of data to ﬁnd the signatures and the
abundances of the endmembers.

1.2    Contribution
In real world HSI, it is very likely that in a small spatial neighbourhood the
number of diﬀerent materials is small, i.e., these images are locally low rank. We


                                               2

<!-- PDF PAGE 4 -->

take advantage of this property to propose a local super-resolution methodology.
Firstly, we propose two approaches, one using a square sliding window of ﬁxed
size and a second using a binary partition tree representation, to obtain a set
of patches from the image,  such that the set of pixels with indices in each
patch span a subspace of lower dimensionality than that of the whole image.
Secondly, we exploit the locally low rank property of the pixels of each patch
in two diﬀerent ways:  by using an endmembers induction algorithm (EIA) to
ﬁnd the low rank subspace spanned by the spectral signatures of the materials
present in each patch;  and,  by a singular value decomposition (SVD)-based
approach to ﬁnd the local low-rank subspace of the data of each patch.  The
proposed methodology does not need to estimate the spatial blur, as far as it is
constant across bands.

1.3    Outline
Theremainderofthepaperisorganizedasfollows: inSec.2thesuper-resolution
problemisformulated,inSec.3theproposedlocalsuper-resolutionmethodology
is introduced and ﬁnally, in Sec. 4 experimental results and some conclusion
remarks are provided.

2    Problem formulation

LetX ∈ Rnh×n denoteaHSIwithnh spectralbands(rowsofX)andn = nx× ny
pixels(columnsofX). WemayinterpretXeitherasacollectionofnh images(or
bands) of size nx× ny, each one associated to a given wavelength interval, or as a
collection of n spectral vectors of size nh, each one associated with a given pixel.
In this work, we are concerned with the estimation of X, which we term the
original HSI, from two degraded observations of X: a) a low spatial resolution
HSI, Yh  ∈  Rnh×(nx/d)×(ny/d), where d > 1 denotes a spatial downsampling
factor, and b) a MSI, Ym ∈ Rnm×nx×ny, where nm ≪   nh. We assume that Yh
is generated as

                                   Yh = XBM+Nh,                                                (1)

where B ∈  Rn×n  is a matrix modeling band independent sensor blur, M ∈
Rn×(n/d2) is a masking matrix accounting for spatial downsampling of size d on
both spatial dimensions, and Nh is an additive perturbation.  Concerning the
MSI, Ym, we assume the generation model

                                    Ym = RX+Nm,                                                 (2)

where the matrix R ∈  Rnm×nh holds in its columns the nh spectral responses
of the multispectral sensor, and Nm is an additive perturbation.
    Let us suppose that it is possible to learn a dictionary D ∈  Rnh×nd  from
the hyperspectral image Yh, and that the columns X, denoted by xi, for i =
1,...,n     , may be sparsely represented as linear combinations of the columns of


                                              3

<!-- PDF PAGE 5 -->

D.  That is, given xi for i ∈ S, there is a sparse vector αi ∈  Rnd (i.e., only a
few components of αi are non-zero) such that:

                                          xi = Dαi.                                                       (3)

By replacing (3) in (2), we obtain:

                                    Ym = RDA+Nm,                                               (4)

where A ≡   [α1,...,      αn], is often termed as code  in dictionary learning and
sparse regression applications. If equation (4) can be solved with respect to A,
then we may plug its solution into (3) and thereby obtain an estimate of X.
    The success of a dictionary-based approach depends fundamentally on the
ability to solve (4) with respect to A. The diﬃculty in solving this system comes
from the fact that the system matrix RD ∈ Rnm×nd is often fat, i.e., nm < nd,
yielding an undetermined system of equations.  A typical multispectral sensor
has less than 10 bands, quite often 4 in the wavelength interval [0. 4, 2. 5]microns
where most hyperspectral sensors operate, whilst nd is often of the order of a
few tens.  The ill-posedness of (2) when nm < nd may be cured by exploiting
the sparsity of the codes αi, for i ∈ S, which opens the door to all sort of
sparse regression techniques, many of them recently introduced in compressive
sensing [7] applications. Nevertheless, we still face a diﬃculty: in hyperspectral
applications, the columns of D tend to be highly correlated, implying that the
mutual coherence of the columns of RD is close to 1.  This makes (2) ill-posed
even when codes αi, for i ∈ S, are sparse [8, 9].
    In the next section, we introduce two local dictionary-based techniques con-
ceived to cope with the ill-posedness with origin in the matrix system RD. The
main idea, in the vein of the local approaches to image restoration, is to de-
compose the HSI into patches and build patch-dependent dictionaries such that
nm ≥  nd in each patch.  Thereby, the inverse problem of solving (2) in each
patch is well-posed.

3    Local dictionary based HSI super-resolution

3.1    Local spatial patches deﬁnition
Let Pj  denote the set of patches obtained from an image, Yh, such that the
set of pixels with indices in each patch, Yj ≡  [yi, i  ∈ Pj], span a subspace of
lower dimensionality than that of Yh.  Next, we introduce two approaches to
obtain such patches.  The ﬁrst approach uses a sliding window and the second
approach relies on a binary partition tree (BPT) representation [10].

3.1.1    Using a sliding window of ﬁxed size
The sliding window methodology has been broadly used in image processing.
It consists in deﬁning a square window of ﬁxed size so the pixels lying inside
the window form a patch.  The window then slides over the whole image in a


                                               4

<!-- PDF PAGE 6 -->

standard zig-zag way, be it with some overlapping or not, eventually covering
the whole image and deﬁning the image patches, Pj.

3.1.2    Using a BPT representation
The BPT is a hierarchical region-based representation of an image in a tree
structure [10].   In the BPT representation,  the leaf nodes correspond to an
initial partition of the image, which can be the individual pixels, or a coarser
segmentation map.  From this initial partition, an iterative bottom-up region
merging algorithm is applied until only one region remains.  This last region
represents the whole image and corresponds to the root node.  All the nodes
between the leaves and the root result of the merging of two adjacent children
regions.
    TwonotionsareofprimeimportancewhendeﬁningaBPT,theregion model,
MR, which speciﬁes how a region R is modelled, and the merging criterion,
O(MRα, MRβ), which is a distance measure between the region models of any
two regions Rα and Rβ. Each merging iteration involves the search of the two
neighbouring regions which achieve the lowest pair-wise similarity among all
the pairs of neighbouring regions in the current segmentation map. Those two
regions are consequently merged.  To build the BPT representation from the
hyperspectral image [11, 12, 13], Yh, we use the ﬁrst-order parametric model
MR:
                                                     ∑NR
                                 MR     d= ¯x =   1      xi,
                                                NR   i=1

where NR is the number of pixels on the region; and, in order to merge regions,
the spectral angle distance:
                               )     d= dSAM(¯xα, ¯xβ) = arccos(       ¯xα¯xβ )
            O( MRα, MRβ                                           ∥¯xα∥∥¯xβ∥     .

Once the BPT representation is built, the BPT is pruned to achieve a partition
of the image such that the regions of the partition deﬁne the image patches,
Pj.  We use the vote-based pruning approach from [14], where each region of
the BPT representation is voted by its children. A region receives a number of
votes equal to the leaves of a children node if the distance between the region
modelandthechildrenmodelisabovesomethreshold. Oncealltheregionshave
collected their number of votes, the BPT is pruned so the partition cardinality
is maximized constrained to the regions of the partition having received at least
half the total number of votes.

3.2    Local dictionary-based super-resolution
3.2.1    EIA-based approach
Given a set of patches Pj, j = 1,...,P      ; for each Pj, we identify a mixing matrix
Dj by applying an EIA to extract a set of endmembers from the set of pixels


                                              5

<!-- PDF PAGE 7 -->

Table 1:  Mean and standard deviation values of the average RMSE, average
SAM, ERGAS and average Q quality measures for super-resolution methodolo-
gies comparison using the synthetic dataset.
                                                                     Synthetic
                                       RMSE                  SAM                ERGAS                   Q
        HSR-GDL-EIA               0.0552±0.0230        4.7777±3.5114        4.7583±1.6699        0.9629±0.0252
 HSR-LDL-EIA        Windows       0.0294±0.0007        2.3944±0.0610        2.1254±0.0450        0.9877±0.0007
                       BPT       0.0290±0.0008        2.2067±0.0714        2.0574±0.0447         0.9878±0.0008
 HSR-LDL-SVD        Windows       0.0295±0.0009        2.3877±0.0751        2.1093±0.0468        0.9875±0.0009
                       BPT       0.0288±0.0008        2.1810±0.0691        2.0429±0.0535         0.9881±0.0008

Xj, identiﬁed by the corresponding patch.  Having identiﬁed the dictionaries,
Dj, the code A is estimated by solving the following constrained least squares
(CLS) problems:

                        min                        F     j = 1,...,P,                                   (5)
                       Aj≥0∥Ym,j −  RDjAj∥2

where Aj ∈  Rnd×|Pj| and Ym,j ∈  Rnm×|Pj| gathers, respectively, the columns
of A and Ym with indices corresponding to the pixels Xj in Pj. The inequality
Aj  ≥  0 is to be understood in the component-wise sense, and ∥ · ∥F  denotes
the Frobenius norm.   The constraint Aj  ≥   0 in (5) is used because,  in the
linear mixing model,  the codes Aj  represent abundances of materials which
are necessarily non-negative [15].   It is possible to add the abundances sum-
to-one constraint, ATj I = 1|Pj|, but this constraint is usually dropped due to
possible scale model mismatches [9].  We remark that, since our observations
are spectral vectors and thus non-negative (apart from the noise contribution),
the non-negativity constraint in (5) is equivalent to a form of constrained ℓ1
regularization and therefore, to some kind of sparsity enforcement (see [16] for
the details).
    In order to solve (5), we use the SUnSAL algorithm [17], which is an in-
stance of the C-SALSA methodology introduced in [18] to eﬀectively solve a
large number of CLS problems sharing the same matrix system.  Fig. 1 shows
the pseudocode of the proposed HSI Super Resolution via Local Dictionary
Learning using EIAs (HSR-LDL-EIA). The algorithm operates on each patch
independently. Each patch indexes the set of pixels in the hyperspectral image,
Yh,j, and in the multispectral image, Ym,j. From the hyperspectral set of pix-
els, the subspace matrix, Dj, is obtained using any EIA. Then, the SUnSAL
algorithm is used to estimate the fractional abundances,  ˆAj, from the multi-
spectral set of pixels.   The fractional abundances are linearly combined with
the estimated subspace to estimate the super-resolution pixels of the patch, ˆXj.
The estimated super-resolution pixels of all the patches are combined to form
the estimated super-resolution image, X.







                                               6

<!-- PDF PAGE 8 -->

Algorithm HSR-LDL-EIA
Input:
1.    Yh, Ym, R, Pj, nd
2.    for j = 1 to P
3.                  Ym,j := [Ym,i,i  ∈ Pj]
4.                  Yh,j := [Yh,i,i  ∈ Pj]
5.                  Dj := EIA(Yh,j,n d)
6.                   ˆAj := argminAj≥0∥Ym,j − RDjAj∥2F
7.                   ˆXj := DjˆAj{ˆXj}
8.     ˆX :=         ,  j  = 1,...,P


Figure 1: HSI Super Resolution via Local Dictionary Learning using Endmem-
ber Induction Algorithms (HSR-LDL-EIA).


Table 2:  Mean and standard deviation values of the average RMSE, average
SAM, ERGAS and average Q quality measures for super-resolution methodolo-
gies comparison using the real Pavia University dataset.
                                                                   Pavia University
                                        RMSE                  SAM                 ERGAS                    Q
        HSR-GDL-EIA               0.0136±0.0017          2.9654±0.3068        2.2056±0.2994        0.9897±0.0024
  HSR-LDL-EIA        Windows       0.0169±0.0001         2.9804±0.0077        2.6302±0.0082         0.9832±0.0001
                       BPT         0.0138±0.0030        2.6766±0.4147        2.1924±0.3118          0.9887±0.0051
 HSR-LDL-SVD         Windows       0.0901±0.1101         2.7874±0.1449        15.042±19.988         0.8445±0.1345
                       BPT         0.0153±0.0015         3.1503±0.1907        2.9803±0.3274         0.9844±0.0037

3.2.2    SVD-based approach
In this method, we carry out the dictionary learning by solving the following
optimization problem:
                      minrank(Dj) |  min        Yh,j −  Djβj2
                                          βj,Dj                    F ≤  δ,                           (6)

where Yh,j  are the observed hyperspectral vectors with indexes in Pj,  and
βj  are weighting coeﬃcients (the code).   The optimization problem (6) aims
at ﬁnding the dictionary Dj of lower rank, desirably lesser than nm, yielding a
representation error for Yh,j smaller than δ. Using the local low rank argument,
the range of the subspace spanned by the super-resolution pixels Xj in patch
Pj should be close to that of Dj and thus, we may obtain the code for Xj by
solving:

                  minAj  ∥Ym,j −  RDjAj∥2F +τ∥Aj∥2F    j = 1,...,P.                           (7)

Comparing (7) with (5), we have discarded the non-negativity constraint and
have introduced the quadratic term τ∥Aj∥2F, where τ ≥  0 is a regularization pa-
rameter. We have discarded the non-negativity constraint because the columns


                                                7

<!-- PDF PAGE 9 -->

Algorithm HSR-LDL-SVD
Input:
1.    Yh, Ym, R, Pj
2.    for j = 1 to P
3.                  Ym,j := [Ym,i,i  ∈ Pj]
4.                  Yh,j := [Yh,i,i  ∈ Pj]
5.                  Ukmax := SVD(Yh,j,k max){                          }
6.                  Dj := Uˆk,      ˆk := min k : UTkYh,j2F ≤  δ
7.                   ˆAj := ( DTj RTRDj +τI)  −1DTj RTYm,j
8.                   ˆXj := DjˆAj{ˆXj}
9.     ˆX :=         ,  j  = 1,...,P


Figure2: HSISuperResolutionviaLocalDictionaryLearningusingtheSingular
Value Decomposition (HSR-LDL-SVD).


of Dj are no more interpretable as spectra of pure materials and thus Aj are no
more interpretable as abundances.  The introduction of a quadratic regularizer
aims at improving the noise-ampliﬁcation we may have owing to the very low
singular vectors of RDj.
    Although problem (6) is nonconvex, it can easily be solved based on the
SVD of Yh,j.  Let Uk ∈ Rnh×k be a matrix holding the k left singular vectors
of Yh,j ordered by decreasing value of the corresponding singular values. Then,
for rank(Di) = k, it holds that the solution to:
                          min   Yh,j −  Djβj2
                          βj,Dj                     F = UTkYh,j2F                               (8)

is achieved for Dj = Uk [19]. Therefore, the solution of (6) is given by:

                       Dj := Uˆk |ˆk = argkminUTkYh,j2F ≤  δ

    Fig. 2 shows the pseudocode of the proposed HSI Super Resolution via Local
Dictionary Learning using Single Value Decomposition (HSR-LDL-SVD). The
algorithm works in a similar way to the HSR-LDL-EIA algorithm.  The main
diﬀerence is that the subspace matrix, Dj, is obtained from the ˆk columns of the
singular value decomposition of the hyperspectral pixels.  Then, the fractional
abundances, ˆAj, are calculated from the multispectral pixels using the analytic
solution to the regularized inversion problem (8).  Finally, the super-resolution
pixels of the patch,  ˆXj,  are calculated;  and,  the estimated super-resolution
pixels of all the patches are combined to form the estimated super-resolution
image, X.




                                                 8

<!-- PDF PAGE 10 -->

4    Experimental Results and conclusions
In order to assess the performance of the proposed local super-resolution meth-
ods,weusedthefollowingexperimentalmethodology: 1)Givenasuper-resolution
image, X, we simulate a low-spatial resolution hyperspectral image, Yh, by ap-
plying a Gaussian blurring (B) and downsampling the blurred image by a factor
of four (M). 2) We also simulate a high-spatial resolution multispectral image,
Ym, by applying a spectral response matrix (R) to the original super-resolution
image.  3) White noise was added to both images, Yh and Ym, with 30db and
40db respectively.  4) Then, we estimate the super-resolution image,  ˆX, by ap-
plying a super-resolution methodology to both images, Yh and Ym. 5) Finally,
we compare the original and the estimated super-resolution images, using the
RMSE, the SAM, the ERGAS and the Q quality measures [1].
    We compared the two proposed local super-resolution approaches, the HSR-
LDL-EIA and the HSR-LDL-SVD, to the global approach based on EIAs, HSR-
GDL-EIA. The SVD-based approach has been deﬁned explicitly for the local
approach.   Its global version performs poorly and thus, there is no point in
showing the comparison. For the proposed local approaches, we also compared
the sliding window and BPT-based approaches to obtain the image patches. In
the case of the EIA-based approach, we used the Vertex Component Analysis
(VCA) EIA [20]. We made the comparison over two datasets:

   1.  A synthetic image composed of multiple geometric shapes (ellipses and
      rectangles) of diﬀerent sizes and orientations, where each geometrical el-
      ement and the background are formed using a diﬀerent linear mixture of
      5 endmembers from the USGS spectral library, for a total of 45 diﬀerent
      endmembers in the image. This image simulates a high spectral variability
      scenario.  In this case, we ﬁxed the number of induced endmembers and
      selected SVD components to p = 2 for the local approaches, and p = 45
      for the global one.

   2.  The well-known Pavia University dataset.  This dataset is a real image
      representing an urban area mainly composed of buildings, urban vegeta-
      tion and parking lots.  The multi-spectral image was simulated using the
      spectral response of the ﬁrst three bands of the Ikonos sensor. For the lo-
      cal approaches, we ﬁxed the number of induced endmembers and selected
      SVD components to p = 3. For the global approach, this value was set to
      p = 10.

    Tables 1-2 show the results obtained for the synthetic and Pavia University
datasets respectively. For each quality measure, it is shown the mean and stan-
dard deviation of Monte Carlo runs.  Results show that the local approaches
outperform the global one in the synthetic scene, and that the EIA-based local
approach does the same in the real scene. From the point of view of the patches
deﬁnition, the BPT representation shows a slightly higher performance, hinting
at the relevance of the patching deﬁnition process.   However, the windowing
approach shows slightly lower standard deviation probably due to the eﬀects


                                              9

<!-- PDF PAGE 11 -->

of noise in the BPT partitioning.  The lack of space prevents us from showing
the obtained super-resolution images for visual comparison.  We encourage the
interested reader to access the results extended report at ﬁrst author’s website1.
We’ll further work on the validation of the proposed methodologies and on the
study of the patches deﬁnition by means of other BPT pruning strategies. We’ll
also face the problem of estimate the spectral response matrix, R, from the
data.
    Figs. 3 and  4 show false color representations of the actual Pavia University
dataset and the super-resolution estimated images obtained by the compared
approaches.



























     Figure 3: False color representations of the Pavia University dataset.


Acknowledges

Thanks for funding to Funda¸c˜ao para a Ciˆencia e Tecnologia (FCT), Portuguese
Ministry of Science and Higher Education, projects PEst-OE/EEI/0008/2013
and PTDC/EEI-PRO/1470/2012, and grant SFRH/BD/87693/2012.

   1http://www.gipsa-lab.grenoble-inp.fr/page   pro.php?vid=1728






                                              10

<!-- PDF PAGE 12 -->

         (a)               (b)               (c)              (d)               (e)












         (f)               (g)               (h)               (i)               (j)


Figure 4:  False color representations of the super-resolution images obtained
from the simulated hyperspectral and multispectral observations of the Pavia
University dataset.  Top and bottom rows respectively show the best and the
worst  images  obtained  by:   (a,f)  HSR-GDL-EIA,  (b,g)  HSR-LDL-EIA (win-
dows), (c,h) HSR-LDL-EIA (BPT), (d,i) HSR-LDL-SVD (windows) and, (e,j)
HSR-LDL-SVD (BPT)

References

 [1]  H. Songa, B. Huanga, K. Zhangb, and H. Zhanga, “Spatio-spectral fusion
     of satellite images based on dictionary-pair learning,” Information Fusion,
     vol. 18, pp. 148 – 160, 2014.
 [2]  R. Zurita-Milla, J. Clevers, and M. Schaepman, “Unmixing-based landsat
     tmandmerisfrdatafusion,”IEEE Geoscience and Remote Sensing Letters,
     vol. 5, no. 3, pp. 453–457, 2008.

 [3]  R. Kawakami,  J. Wright,  Y.-W. Tai,  Y. Matsushita,  M. Ben-Ezra,  and
     K. Ikeuchi, “High-resolution hyperspectral imaging via matrix factoriza-
     tion,” in 2011 IEEE Conference on Computer Vision and Pattern Recog-
     nition (CVPR), 2011, pp. 2329–2336.



                                             11

<!-- PDF PAGE 13 -->

 [4]  A. S. Charles, B. A. Olshausen, and C. J. Rozell, “Learning sparse codes
     for hyperspectral imagery,” IEEE Journal of Selected Topics in Signal Pro-
     cessing, vol. 5, no. 5, pp. 963–978, 2011.

 [5]  B. Huang, H. Song, H. Cui, J. Peng, and Z. Xu, “Spatial and spectral im-
     age fusion using sparse matrix factorization,” IEEE Transactions on Geo-
     science and Remote Sensing, vol. 52, no. 3, pp. 1693–1704, 2014.

 [6]  N. Yokoya, T. Yairi, and A. Iwasaki, “Coupled nonnegative matrix factor-
     ization unmixing for hyperspectral and multispectral data fusion,” IEEE
     Transactions on Geoscience and Remote Sensing, vol. 50, no. 2, pp. 528–
     537, 2012.

 [7]  R. G. Baraniuk, “Compressive sensing [lecture notes],” IEEE Signal Pro-
     cessing Magazine, vol. 24, no. 4, pp. 118–121, 2007.

 [8]  M.-D. Iordache, J. M. Bioucas-Dias, and A. Plaza, “Sparse unmixing of hy-
     perspectral data,” IEEE Transactions on Geoscience and Remote Sensing,
     vol. 49, no. 6, pp. 2014–2039, 2011.
 [9]  J. M. Bioucas-Dias, A. Plaza, N. Dobigeon, M. Parente, Q. Du, P. Gader,
     and J. Chanussot, “Hyperspectral unmixing overview: Geometrical, statis-
     tical, and sparse regression-based approaches,” IEEE Journal of Selected
     Topics in Applied Earth Observations and Remote Sensing, vol. 5, no. 2,
     pp. 354–379, 2012.

[10]  P. Salembier and L. Garrido, “Binary partition tree as an eﬃcient repre-
     sentation for image processing, segmentation, and information retrieval,”
     IEEE Transactions on Image Processing, vol. 9, no. 4, pp. 561–576, 2000.

[11]  S. Valero, P. Salembier, and J. Chanussot, “Comparison of merging orders
     and pruning strategies for binary partition tree in hyperspectral data,”
     in 17th IEEE International Conference on Image Processing (ICIP), Sept
     2010, pp. 2565–2568.

[12]  ——, “Hyperspectral image representation and processing with binary par-
     tition trees,” IEEE Transactions on Image Processing, vol. 22, no. 4, pp.
     1430–1443, April 2013.

[13]  M. Veganzones, G. Tochon, M. Dalla Mura, A. Plaza, and J. Chanussot,
     “Hyperspectral image segmentation using a new spectral mixture-based bi-
     nary partition tree-representation,” in 20th IEEE International Conference
     on Image Processing (ICIP), 2013.
[14]  G. Tochon, J. Feret, R. Martin, R. Tupayachi, J. Chanussot, and G. Asner,
     “Binary partition tree as a hyperspectral segmentation tool for tropical
     rainforests,” in 2012 IEEE International Geoscience and Remote Sensing
     Symposium (IGARSS), 2012, pp. 6368–6371.



                                              12

<!-- PDF PAGE 14 -->

[15]  W.-K. Ma, J. Bioucas-Dias, T. Chan, N. Gillis, P. Gader, A. Plaza, A. Am-
     bikapathi, and C. Chi, “A signal processing perspective on hyperspectral
     unmixing:  Insights from remote sensing,” IEEE Signal Processing Maga-
     zine, vol. 31, no. 1, pp. 67–81, 2014.

[16]  A. M. Bruckstein,  M. Elad,  and M. Zibulevsky,  “On the uniqueness of
     nonnegative sparse solutions to underdetermined systems of equations,”
     IEEE Transactions on Information Theory, vol. 54, no. 11, pp. 4813–4820,
     2008.

[17]  J.M.Bioucas-DiasandM.A.Figueiredo, “Alternatingdirectionalgorithms
     for constrained sparse regression: Application to hyperspectral unmixing,”
     inIEEE 2nd Workshop on Hyperspectral Image and Signal Processing: Evo-
     lution in Remote Sensing (WHISPERS), 2010, pp. 1–4.

[18]  M. V. Afonso, J. M. Bioucas-Dias, and M. A. Figueiredo, “An augmented
     lagrangianapproachtotheconstrainedoptimizationformulationofimaging
     inverse problems,” IEEE Transactions on Image Processing, vol. 20, no. 3,
     pp. 681–695, 2011.
[19]  C. Eckart and G. Young, “The approximation of one matrix by another of
     lower rank,” Psychometrika, vol. 1, no. 3, pp. 211–218, 1936.

[20]  J. Nascimento and J. Dias, “Vertex component analysis: a fast algorithm to
     unmix hyperspectral data,” IEEE Transactions on Geoscience and Remote
     Sensing, vol. 43, no. 4, pp. 898 – 910, april 2005.




























                                              13

<!-- END-OF-PAPER:358766021635 -->
