<!-- PDF PAGE 1 -->

GSNR: Graph Smooth Null-Space Representation for Inverse Problems
Romario Gualdr´on-Hurtado*, Roman Jacome*, Rafael S. Su´arez, Henry Arguello
Universidad Industrial de Santander, Colombia, 680002
{yesid2238324,rajaccar,rafael2269082}@correo.uis.edu.co,henarfu@uis.edu.co
Abstract
Inverse problems in imaging are ill-posed, leading to in-
finitely many solutions consistent with the measurements
due to the non-trivial null-space of the sensing matrix.
Common image priors promote solutions on the general
image manifold, such as sparsity, smoothness, or score
function. However, as these priors do not constrain the
null-space component, they can bias the reconstruction.
Thus, we aim to incorporate meaningful null-space infor-
mation in the reconstruction framework. Inspired by smooth
image representation on graphs, we propose Graph-Smooth
Null-Space Representation (GSNR), a mechanism that im-
poses structure only into the invisible component. Par-
ticularly, given a graph Laplacian, we construct a null-
restricted Laplacian that encodes similarity between neigh-
boring pixels in the null-space signal, and we design a low-
dimensional projection matrix from thep-smoothest spec-
tral graph modes (lowest graph frequencies). This approach
has strong theoretical and practical implications: i) im-
proved convergence via a null-only graph regularizer, ii)
better coverage, how much null-space variance is captured
bypmodes, and iii) high predictability, how well these
modes can be inferred from the measurements. GSNR is
incorporated into well-known inverse problem solvers, e.g.,
PnP , DIP , and diffusion solvers, in four scenarios: image
deblurring, compressed sensing, demosaicing, and image
super-resolution, providing consistent improvement of up to
4.3 dB over baseline formulations and up to 1 dB compared
with end-to-end learned models in terms of PSNR.
1. Introduction
Recovering an image from a limited number of noisy mea-
surements requires estimating a solution to an undetermined
inverse problem, which is of the form
y=Hx ∗ +ω,ω∼ N(0, σ 2I). (1)
where y∈R m denotes the measurement vector, x∗ ∈R n
is the unknown target image, H∈R m×n is the sensing
matrix with m≤n , and ω models additive Gaussian noise.
This formulation is used in a broad range of imaging tasks
*Equal contribution.
by using a proper structure ofH. Compressed sensing (CS)
tasks employ dense sensing matrices [5]. Image restora-
tion problems, such as deblurring [53], inpainting [51], and
super-resolution [46], are modeled by structured Toeplitz
or block-diagonal matrices. In medical imaging, modalities
like magnetic resonance imaging (MRI) [26] and computed
tomography (CT) [2] use structured physics-based opera-
tors (e.g., undersampled Fourier or Radon transforms). The
formulation in (1) becomes ill-posed due to matrix instabil-
ity and/or the undersampling scheme. Therefore, incorpo-
rating prior knowledge or regularization is essential to pro-
mote reconstructions that are consistent with the expected
structure of x∗ [4]. The estimation of x∗ is commonly
formulated as a variational problem that combines a data-
fidelity term and a regularization term:
bx= arg min
˜ x
1
2 ∥H˜ x−y∥2
2| {z }
data fidelity:g(˜ x)
+η f(˜ x)|{z}
prior
(2)
where g(˜ x)enforces consistency with the measurements
and f(˜ x)incorporates prior knowledge about the struc-
ture of x∗. Recent advances have introduced data-driven
regularizers for f(˜ x), most notably Plug-and-Play (PnP)
[6, 20, 48, 52] and Regularization by Denoising (RED)
[9, 34, 35], which have achieved remarkable success across
a wide range of imaging tasks [20] by implicitly promot-
ing solutions within the training image dataset manifold i.e.,
well-trained denoiser learns the data score-function [27].
PnP and RED use learned denoisers, allowing the inte-
gration of powerful deep restoration networks within algo-
rithms [27, 44]. Despite their versatility and success, these
priors operate in the image domain and do not explicitly ex-
ploit the structure of H, including its null-space (NS). Only
a portion of x∗ is observable throughH. The range-null-
space decomposition (RNSD) states that anyx∈R n vector
is decomposed asx=x r +x n, wherex r =P rxde-
notes the component lying in the range space ofH, with
Pr =H †H, andx n =P nxdenotes the component lying
in its NS, with the NS projector Pn =I−H †H.x n is
invisible toHbecause it belongs its NS, defined as
Null(H) ={x∈R n :Hx=0}
={x:x⊥h j,∀j∈ {1, . . . , m}},
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
12469


<!-- PDF PAGE 2 -->

whereh j ∈R n is thej-th row ofH. This interpretation
has inspired approaches that explicitly exploit Null(H).
Null-Space Networks (NSN) [41] and Deep Decomposi-
tion [7] operate across the entire Null(H), treating all in-
visible directions as equally relevant, neglecting the fact
that natural, perceptually plausible images occupy only a
low-dimensional manifold within that space [17]. To ad-
dress this gap, the recent Nonlinear Projections of the Null-
Space (NPN) framework [17] introduces a learnable low-
dimensional NS projection matrix S∈R p×n, withp≤
(n−m).Sis designed depending on the inverse prob-
lem, making its rows orthogonal toH, thenSis opti-
mized jointly a NS component predictor G(y)≡Sx ∗ to
adapt thespan(S)into a subspace ofNull(H). During
reconstruction, NPN penalizes deviations ∥G(y)−S˜ x∥2
2.
However, natural images occupy a low-dimensional, struc-
tured subset insideNull(H), so blindly learning an arbitrary
span(S⊤)⊂Null(H)can waste capacity and induce bias.
In this work, we present a principled framework that pro-
vides a new representation of the NS, enabling the optimal
selection of informative NS directions. To address this is-
sue, we took into account two main optimality criteria:
Coverage:How much of the null-space ofHis repre-
sented with the projection matrixS.
Predictability:How easily can the projectionsSxbe
estimated from the measurementsyonly.
Inspired by the graph (smooth) representation of im-
ages [28, 29], we introduce Graph-Smooth Null-Space Rep-
resentation (GSNR) to overcome the shortcomings of NS
representation and prediction. Different from the well-
studied graph-structure preserving approach in imaging in-
verse problems [19, 28] that promotes smoothness over the
imagex, our approach is the first to endow the NS compo-
nentx n with a graph structure, defined through a sparse,
positive semidefinite graph LaplacianL⪰0, such as
4/8-nearest-neighbor (NN) image grid whose edge weights
encode local pixel similarity. We form thenull-restricted
LaplacianT:=P n L Pn onNull(H). Via spectral
graph theory, we construct the graph-smooth NS projection
matrixSas thep-smoothest NS modes, which corresponds
to thepfirst graph Fourier modes. The proposed GSNR
changes how we solve inverse problems by providing ex-
act structure information where the sensor is blind. By se-
lecting thep-smoothest eigenmodes of the null-restricted
Laplacian, our method (i) reduces hallucinations and bias
by constraining only the invisible component, not the whole
image; (ii) improves conditioning, leading to faster and
more stable convergence across PnP, DIP, and diffusion
solvers; (iii) increases data efficiency by capturing most
null-space variation with smallp(high coverage) and by
targeting the modes that are easiest to infer from measure-
ments (high predictability); and (iv) offers operational di-
agnostics coverage/predictability curves that allows select-
ingpand justify regularization objectively. Because it is
plug-compatible with standard pipelines and agnostic to the
forward operator, the approach scales across different imag-
ing tasks, turning an ill-posed ambiguity into a structured,
measurable, and learnable component. The proposed con-
struction has the following benefits:
1. Selected graph-smooth null-space components produce
high coverage at low dimensionality projection, i.e.,
smallp(Theorem 1 and Theorem 2).
2. Predicting thepnull-space components fromyusing a
GSNR is easier than the ones obtained with a common
null-space basis (Proposition 1).
3. Predicted graph null-space components can be incorpo-
rated via regularization into solvers such as PnP, Deep
Image Prior, and Diffusion Models.
2. Related Work
Variational and learned priors.A common scheme
to solve (2) is proximal gradient descent (PGD) [30]. Re-
cently, PnP [20] and RED [35] methods have replaced the
proximal operator of f(˜ x)with learned denoisers, enabling
the incorporation of data-driven priors without requiring an
explicit analytic form [17]. Although these frameworks
achieve state-of-the-art performance, they leaveNull(H)
largely unconstrained; denoisers may freely modify com-
ponents that are invisible to the sensing operator.
Nonlinear Projections of the Null-Space (NPN).
To explicitly regularize image unobserved components,
[17] introduced a task-specific low-dimensional projection
withinNull(H), enforcing consistency between learned NS
predictions and reconstructions. Building on this principle,
we propose GSNR, which incorporates graph-smoothness
analysis to guide the selection of a structured null-subspace.
Our design ensures that the learned projection aligns with
the intrinsic geometry of the data, yielding NS components
that are both predictable from the measurements and con-
sistent with the graph-induced smoothness prior.
Graph theory in imaging inverse problems. Varia-
tional regularization methods address inverse problems by
imposing smoothness or sparsity priors that constrain the
solution space. Total variation [38] enforces pixel-wise
smoothness, whereas wavelet-based sparsity priors [13]
promote sparse representations of images in the transform
domain via soft-thresholding. Although these priors im-
prove stability, they are typically limited to local structures
and often oversmooth fine textures or complex geometries.
To better capture local dependencies, graph-based and non-
local regularization methods have been reintroduced. [32]
proposed an adaptive nonlocal regularization framework
that jointly estimates the image and the graph that encodes
patch similarities. However, like classical priors, this frame-
work operates solely on the reconstruction and doesn’t con-
strain the image component lying inNull(H).
12470


<!-- PDF PAGE 3 -->

3. Graph-Smooth Null-Space Representation
3.1. Preliminaries: Graphs
Graphs provide a geometric representation of data, where
vertices correspond to data elements (e.g., pixels or image
patches) and weighted edges encode pairwise similarities
between them [42]. Signals defined on weighted undirected
graphs are mathematically denoted byG= (V,E,W),
whereVis the vertex set with|V|=n,E ⊆ V × Vis
the edge set, andW∈R n×n is the symmetric weighted
adjacency matrix withW ij >0if(i, j)∈ EandW ij = 0
otherwise. The unnormalized graph Laplacian is
L=D−W,(Lx) i =
X
j
Wij
 
xi −x j

, (3)
where the diagonal degree matrixD= diag(d 1, . . . , dn)
accomplishesd i =P
j Wij. Its Dirichlet energy is
x⊤Lx= 1
2
X
i,j
Wij
 
xi −x j
2
, (4)
which penalizes discrepancies across edges with large
weights. For symmetricW,Lis real symmetric pos-
itive semidefinite (SPSD) with eigenpairs{(λ ℓ,v ℓ)}n−1
ℓ=0 ,
0 =λ 0 ≤λ 1 ≤ · · ·,v 0 ∝1on connected graphs. Eigen-
vectors with smallλ ℓ vary slowly across high-weight edges,
motivating the interpretation ofλas a graph frequency. For
the usage ofL, we focus on two graph topologies [29]:
1.4NN:This promotes piecewise smoothness aligned
with the coordinate axes and attenuates horizontal/ver-
tical fluctuations, yielding an anisotropic (axis-biased)
smoothing effect. Here,E 4 ={(i, j) :∥p i −p j∥1 = 1},
Wij =1 (i,j)∈E4, withp i ∈Z 2 pixel coordinates.
2.8NN:This reduces orientation bias and more closely
approximates an isotropic Laplace operator, promoting
rotation-invariant piecewise smoothness and suppressing
fluctuations uniformly across directions. Here,E 8 =
{(i, j) :∥p i −p j∥∞ = 1},W ij =w o1∥·∥2=1 +
wd1∥·∥2=
√
2, wherew d =w o/
√
2.
Nevertheless, we extend the analysis of the proposed ap-
proach to other topologies in Supp. A4.1.
3.2. Graph-smooth null modes
We study smoothness within the NS by restricting the
Laplacian to
T=P nLPn, (5)
the reason for using the null-graph operatorT∈R n×n
stems from the need to pay attention to certain geometric
properties of the image’s NS, which are extracted byL.
LetTx=P nLPnx=P nLxn, whereLx n extracts, ar-
eas of greatest difference in the NS, and then re-project it
to the NS withP n, thus ensuring that this operation re-
sults in a projection from the NS. In Fig. 1, from left to
right, we show the ground-truthx ∗, back-projectionH ⊤y,
true null componentP nx∗, and two graph NS projections
PnL8nnPnx∗ andP nL4nnPnx∗, for super-resolution (SR).
Figure 1. For image SR task withSRF= 4, n= 3·128 2, we
show ground-truth, adjoint reconstruction, NS projection, projec-
tion onto graph-smooth NS withL 4nn andL 8nn.
The mapP nx∗ isolates signal content invisible to the sen-
sor, whileP nLPnx∗ highlights whereLx n falls into the
NS. These graph projections highlight the smoothestblind
signal components, e.g., textures near edges.
Then, by applying eigenvalue decomposition, we see that
T=Vdiag (µ 1, . . . , µn)V ⊤ is SPSD, with0≤µ 1 ≤
· · · ≤µ n, whereV∈R n×n is orthogonal. Note thatV
diagonalizesT, so its columns{v i}are thegraph-Fourier
modes withinNull(H). Ifµ i is smaller, then a smoother
graph-mode is selected.Vhas the following properties: i)
Vhas orthonormal columnsv i ∈Null(H),µ i-ordered. ii)
These{v i}aregraph-smooth null modesprobingx. The
graph-smooth NS projectorSwithprows is obtained by
taking the firstpcolumns ofV:
S=
V⊤
p
z }| {
V[:,1:p] ⊤ = [v1, . . . ,vp]⊤ ∈R p×n. (6)
Note thatSkeeps only thep-smoothest null directions and
Sxexpresses the NS part ofxin an orthonormal basis
sorted by increasing graph frequency. Particularly, based
on RNSD:Sx=S
 
Pr +P n

x=SP rx+SP nx,since
rows ofSlie inNull(H), henceSP r =0, leading to
Sx=0+SP nx=V ⊤
p Pn x=


⟨v1,x n⟩
...
⟨vp,x n⟩

 .Here{v i}p
i=1
are thepLaplacian-smoothest orthonormal null modes; thus
Sxcollects the coefficients ofxalong thesepdirections.
3.3. Learning low-dimensional GSNR
We train the GSNR predictor G following:
G∗ = min
G
E

∥G(y)−Sx ∗∥2
2

. (7)
Here, we aim to predict the low-dimensional GSNR using
the neural networkG. Subsequently, the theoretical foun-
dations will be laid that greater predictability will guarantee
better convergence in the task.
3.4. Reconstruction objective
Thus, we incorporate our method as
min
˜ x
g(˜ x) +η f(˜ x) +γ∥G∗(y)−S˜ x∥2
2 + γg
2
ϕ(˜ x)
z }| {
˜ x⊤T˜ x,(8)
which includes the learned GSNR term, plus a graph regu-
larizer acting only on the null component of the reconstruc-
tion, weighted byγandγ g, respectively. This retains com-
patibility with proximal/PnP/DM solvers.
12471


<!-- PDF PAGE 4 -->

4. Theory and Analysis
We develop a theoretical analysis on GSNR based on
null-restricted spectralformulation that turns GSNR de-
sign/performance prediction into ordered-eigenvalue crite-
ria onTfor optimal subspace selection, obtaining a princi-
pled eigen-ordering and criteria used to chooseS, p,L. The
theoretical results are summarized as:
Coverage: the first p eigenmodes of T provide an opti-
mal graph-smooth representation of Null(H), by cov-
ering a high null-space spectrum energy withp≪
n−m.
Minimax optimality: the same GSNR basis is worst-
case optimal over a graph-energy ellipsoid inNull(H),
giving a principled ordering of null directions
Predictability: via block-precision/Schur-complement
analysis, we show that GSNR has a higher per-mode
recoverability from measurementsythan plain null-
space projections.
Convergence: the effect of the graph regularizer pro-
vides better conditioning and convergence properties
based on fixed-point analysis.
4.1. Null-space coverage
Before developing the formal analysis on the coverage
properties of the NS projections, we validate the represen-
tation performance of three settings: i)L=I, ii)L 4nn, and
iii)L 8nn. Note that the first setting is equivalent to just tak-
ing the NS basis. In Fig. 2 (a), each row shows projections
(x= (H ⊤H+S ⊤S)x∗) as the dimension of the null–space
subspace increases, p∈ {153,1536,4608,9216,15360} ,
for three graph choices in the design of S: (top, orange)
L=I ; (middle, green) grid L8nn; (bottom, blue) grid L4nn.
For small p, the grid Laplacians already yield visually plau-
sible faces, while L=I still produces noisy, poorly struc-
tured estimates. The number printed above each image is
the MSE of the representation. The plot at the bottom shows
the MSE for all values ofp={1, . . . , n−m}, showing
that the graph-null-space projections provide better signal
representation with smallerpthan just the NS basis. In gen-
eral, to measure the amount of energy from the NS that the
low-dimensional operatorScan capture, we use the spectral
coverage defined as:
C(p) = tr
 
SCov(xn)S⊤
tr (Cov(xn)) ∈[0,1], (9)
whereCov(x)is the covariance matrix ofx. View the
Laplacian eigenvectors as a graph–Fourier basis: the first
modes are smooth, low “graph–frequency” patterns, while
later modes are increasingly oscillatory [29, 42]. TheC(p)
indicates what fraction of the total NS variance lies in the
span of the first p graph–Fourier modes retained byS, sim-
ilarly as explained variance in PCA [18]. Intuitively, if the
signal is graph–smooth, most variability concentrates in low
Figure 2. Coverage and spectral analysis for SR task.(a)x ∗ =
H⊤Hx∗ +S ⊤Sx∗ forSgiven by (6) usingL=I(orange),
L=L 4nn (green) andL=L 8nn (blue) for different values of
p.(b)RNSD representation error, varyingp.(c)Variation ofT
normalized eigenvalues with respect to their index.
frequencies, so coverage climbs rapidly with smallp; if not,
coverage grows more slowly, like in PCA. A4.2 shows cov-
erage results and eigenvalue analysis that validate the use-
fulness of graph Laplacians, rather than a learnable matrix.
Mathematical insights for the selection of the appropriatep
value depending on the coverage are given in A5.
Definition 1 (Gaussian Markov Random Field [39])A
GMRF on the graphG= (V,E)is a zero-mean Gaussian
vectorx∈R n with precision matrixQ=Σ −1 ∈R n×n,
withΣ= Cov(x), is sparse and encodes the conditional
independence of the graph
p(x)∝exp

−1
2x⊤Qx

,Q i,j = 0← →(i, j)/∈ E,
where smoothness is required in neighboring pixels, we de-
fine the precision matrixQ=αL+ϵIwithα >0, ϵ >0.
Under this setting, we derive the following guarantee on im-
proved NS coverage.
Theorem 1 (Coverage for graph-smooth null-space)
Consider the construction ofTin(5). The covariance
of the NS,Cov(x n)is a spectral function ofT, i.e.,
Cov(xn) =Vdiag(λ 1, . . . , λn)V⊤, whereλ i = 1
αµi+ϵ,
λ1 ≥λ 2 · · · ≥λ n. With the construction ofSin(6), the
coverage of the NS using GMRF with a Laplacian matrixL
asC L(p)and the coverage whenL=Iis denotedC I(p),
satisfies for everyp= 1, . . . , n−m.
CL(p)≥C I(p).
12472


<!-- PDF PAGE 5 -->

The proof can be found in Supp. A1.
We illustrate this effect in Fig. 2(b)-(c). In 2(b), the rep-
resentation error of the RNSD decreases more rapidly when
usingL 8nn andL 4nn compared toI. In 2(c), the eigen-
value spectra ofTreveal that, unlike the flat spectrum of the
pure NS projectorP n, graph Laplacians exhibit smoothly
increasing eigenvalues. From Theorem 1, a flat spectrum
(µi = 1for allias withL=I) yieldsC I(p) =p/q, so cov-
erage grows only linearly withp, whereas Laplacians with
increasing{µ i}achieve faster coverage growth.
Our second theoretical result is a minimax optimality on
the NS coverage by the NS scheme. First, define the feasible
set of NS components
Mτ :=

xn ∈Null(H) :x ⊤
n L xn ≤τ
	
Theorem 2 (Minimax optimality)Let Vp =
span{v1, . . . ,vp} and PVp be its projector. Then, among
all p-dimensional V⊂Null(H) , provided µp+1 >0 ,
min
dim(V)=p
sup
xn∈Mτ
∥(I−P V )xn∥2
2 =
sup
xn∈Mτ
∥(I−P Vp)xn∥2
2 = τ
µp+1
.
The proof of this theorem is in Supp. A2.
Remark 1SelectingSso that its rows are{v ⊤
1 , . . . ,v⊤
p }
is provably optimal for covering the NS under the graph
energyx ⊤Tx. Aspgrows, the guaranteed worst-case er-
ror decays with1/µ p+1, which is fast for graph Laplacians
(where manyµ j are small). WithL=Ione getsT=P n
(flat spectrum), henceµ p+1 = 1and no improvement in the
bound, explaining why geometry-free designs cannot cover
the NS efficiently.
4.2. Null-space projections predictability
It is also important to quantify the predictability level of the
NS projectionsSx ∗ from the acquired measurementsy. We
develop astatistical couplinganalysis via linear estimation.
Proposition 1 (Per–mode predictability bound)Letx∈
Rn be zero–mean Gaussian with covarianceCand preci-
sionQ=C −1. LetH∈R m×n and denote the orthogonal
projectors ontoR:= Range(H ⊤)andN:= Null(H)by
Pr andP n, respectively. BlockQandCwith respect to the
decompositionR n =R ⊕ N:
Q=
Qrr Qrn
Qnr Qnn

,C=
Crr Crn
Cnr Cnn

.
AssumeQ nn ≻0onNand let{v j}be an orthonor-
mal eigenbasis ofQ nn withQ nnvj =µ j vj andµ j >
0. Define thej-th null coefficienta j :=v ⊤
j xn where
xn :=P nx. Consider measurements from(1)and denote
Cy :=HC rrH⊤ +σ 2Im. Then the populationR 2 of the
optimal linear predictor ofa j fromysatisfies
ρ2
j := Cov(aj,y) ⊤ C−1
y Cov(y, aj)
Var(aj) ≤ cj
cj +µ j
,
cj :=v ⊤
j
 
QrnCrrQnr

vj.
Figure 3. Per-mode predictability for each case whenL∈
{I,L 8nn,L 4nn}for SR case withSRF= 4withn= 64 2.
The proof of this proposition is found in Supp. A3.
Remark 2If the prior is a GMRF with precisionQ=
αL+εIonL, thenQ nn =αT+εP n withT:=P nLPn,
and one can take{vj}as theT–eigenmodes (graph-smooth
null modes). The bound becomes
ρ2
j ≤ cj
cj +α λ j(T) +ε , c j =v ⊤
j
 
QrnCrrQnr

vj,
withQ rn =αL rn. Thus smoother null modes (small
λj(T)) are more predictable. Now, ifL=I(isotropic
prior), thenQ=γIandQ rn =P r Q Pn =0, soc j = 0
andρ 2
j ≤0for allj: there isnostatistical coupling between
RandN.
To evaluate the developed theory, Fig. 3, the empirical
per–mode predictabilityρ 2
j (one dot per eigenmode), for
three choices ofL∈ {L 8nn,L 4nn,I}, whereL=Ihas
same index ordering across panels. From indices[0, m−1],
the predictability is high for all settings, since they corre-
spond to the eigenpairs related to the Range space, thus, the
predictability is high. The observations are: (i) forL=I,
the spectrum is constant over the NS and then drops to0
outside of it, while both grid Laplacians exhibit a smooth,
monotone growth ofµ j; (ii) correspondingly, with grid
Laplacians, a long tail of modes has non-zero predictability,
(ρ2
j spread over a wide index range), whereas withL=I
only a small initial block of modes shows any predictability,
and the rest are essentially0. Note also thatL 4nn has higher
predictability thanL 8nn in more eigenmodes, following the
fast increase eigenvalue distribution Fig 2 (c).
Note that this analysis was using a linear estimation
framework, harnessing the NS GMRF prior. In practice,
thestatistical couplingcan also be achieved with the super-
vised learning of the predictorGin Eq. (7), but the trend of
GS prior holds on this setting. To evaluate the predictability
of the learned predictor, we used
R2(p)≜1−
E
h
∥G∗(y)−Sx ∗∥2
2
i
E
h
∥Sx∗∥2
2
i . (10)
12473


<!-- PDF PAGE 6 -->

a) b)
Figure 4. a) Coverage and b) Predictability vs.pfor CS with
CIFAR-10 dataset. In this case,m/n= 0.1andp= 1· · ·n−m.
4.3. Graph-based regularizer
Consider the PnP–PGD iteration solving the optimization
problem (8) without the term∥G ∗(y)−Sx∥(for more de-
tailed analysis of this term, see [17, Theorem 1]),
xk+1 = Dσ

xk −α
 
∇g(xk) +γ g∇ϕ(xk)

(11)
= Dσ

xk −α
 
H⊤(Hxk −y) +γ gTxk

, (12)
whereD σ is a bounded denoiser with∥D σ(u)−
Dσ(v)∥ ≤
√
1 +δ∥u−v∥(as in [17, Assumption 2]),
since∇ϕ(x k) =T ⊤xk =Tx k. Letλ min >0andλ max
be the extreme eigenvalues ofA γg =H ⊤H+γ gTon
span(Range(H⊤)∪Null(H)) =R n. Then for any step
sizeα∈(0,2/λ max),
∥xk+1 −x ∗∥ ≤(1 +δ)∥I−αA γg ∥2 ∥xk −x ∗∥(13)
≤ρ∥x k −x ∗∥, (14)
whereρ= (1 +δ) max{1−αλ min, αλ max −1}. In par-
ticular, ifρ <1, The iteration is a contraction and con-
verges linearly to a fixed point. The optimalα(for this lin-
ear bound) isα ∗ = 2
λmin+λmax
with rateρ ∗ = (1 +δ) κ−1
κ+1 ,
κ= λmax
λmin
.Compared with the baseline (γ g = 0), adding
γgT: (i) makesA γg ≻0even onNull(H), guaranteeing
uniqueness of the quadratic subproblem, and (ii)reduces
the condition numberκbecauseTacts precisely on direc-
tions whereH ⊤His small, improving convergence andρ ∗.
5. Experiments
To validate the proposed approach, we implemented it in
PyTorch [31] and used the DeepInverse library [43] to
benchmark different solvers, as detailed below. Since our
work does not propose a specific networkG, we validate
the approach using a U-Net model [37] with a four-level 2D
encoder–decoder architecture: the encoder applies double
convolution blocks withC,2C,4C,and8Cchannels (with
C= 64) and2×2max-pooling, while the decoder uses
nearest-neighbor upsampling with skip connections, refin-
ing features back toCchannels before a final1×1convo-
lution produces the output. The network is optimized using
the Adam optimizer [21] with a learning rate of10 −3 for
50 epochs. For all experiments, we used a noise variance
σ2 = 0.05. Code at 1. We consider the inverse problems:
1github.com/yromariogh/GSNR
Compressed Sensing (CS):H∈ {−1,1} m×n are the
first 10% rows of a Hadamard matrixA∈ {−1,1} n×n [5].
We validate with CIFAR-10 [23] and CelebA [25] datasets.
Super-Resolution (SR):We setn= 3·128 2 and use an
SR factor (SRF) of 4, i.e.,m= 3·32 2. We validate on the
CelebA dataset [25], using 10000 training images resized to
128×128. We also show results for the Synthetic Aperture
Radar (SAR) image dataset [11].
Demosaicing:We used the CelebA dataset, resized to
64×64, and the Bayer filter [3] acquisition pattern.
Deblurring:We setn= 64 2 ·3and used a 2-D Gaus-
sian kernel with a bandwidthσ k = 1. We validated our
experiments on the CelebA [25] dataset and the Places365
[54] dataset, both resized to64×64, using 10000 images
for training and a batch size of 32.
For all of these imaging tasks, we construct the null-
restricted LaplacianTand the matrixS, leveraging prop-
erties ofH. Details on the computation of these matrices in
A5. We provide comprehensive ablations of GSNR in A6.
5.1. Predictability and Coverage
The coverage definition in (9), has a general expression for
the covariance matrix computation. In practice, we com-
puted as follows. Given centered imagesx i and their null
projectionsx n,i =P n(xi − ¯x), where ¯x= 1
N
PN
i=1 xi
and the NS covarianceCov(x n) = 1
N
PN
i=1 xn,ix⊤
n,i. Fig.
4 shows the results in terms of (a) Coverage and (b) Pre-
dictability in CS for different operatorsS, under the same
acquisition matrixH. We used theN= 10000image of
the CIFAR-10 dataset. In this case, NPN (L=I) is ob-
tained following [17, Eq. 3], and GSNR usesL=L 8nn.
Note that here we take the firstprows of eachS. Supp.
A9 shows additional results for CS. A7 shows the coverage
curves for the SR task.
5.2. Diffusion Solvers
We used two diffusion model (DM) solvers, DPS [8] and
DiffPIR [55]. See Supp. A8 for more details on the adap-
tation of these models with the proposed approach. Fig. 5
compares DPS and DiffPIR with and without the proposed
GSNR term. In all cases, the baseline DM already pro-
duces plausible faces, but it often leaves residual blur and
aliasing around high–frequency structures such as hair, eye-
brows, and facial contours. When we replace the generic
NS basis by the GSNR, the reconstructions become visibly
sharper and more faithful to the ground truth: edges around
the jawline and nose are better defined, textures are less
washed out, and blocky artifacts from the low-resolution
observationyare further suppressed. These qualitative im-
provements are consistent with the quantitative SSIM/P-
SNR numbers printed in each panel: the gains over the
baseline DiffPIR/DPS are modest in dB but systematic (e.g.,
30.19→30.31;30.19→30.48), indicating that the GSNR
term helps the diffusion prior resolve NS ambiguities rather
than hallucinating arbitrary details. In other words, the DM
12474


<!-- PDF PAGE 7 -->

Figure 5. Results of DM-based solvers (DPS [8] & DiffPIR [55]) for Baseline, NPN [17], and GSNR withL4nn andL 8nn. Here,p= 0.1n.
Table 1. Quantitative comparison on the super-resolution and de-
mosaicing tasks (higher is better) using the CelebA test set. For
each task, the bestresult and second-best results.
Method Task
Super-Resolution Demosaicing
PnP [20] 27.37 39.35
NPN [17] 29.21 39.77
DDN-Cascade [7] 28.92 37.83
DDN-Independent [7] 28.81 38.92
DNSN [41] 26.52 39.33
Unrolling [50] 28.97 38.82
GSNR-PnP w.L8nn 29.38 39.88
GSNR-PnP w.L4nn 29.42 39.89
still provides the powerful image prior, but GSNR steers it
along graph-smooth NS directions that are predictable from
yand compatible with the forward model, yielding more
accurate and stable reconstructions. Supp. A8 shows the
integration of GSNR in latent-space DMs [15].
5.3. Comparison with End-to-End Methods
Table 1 reports PSNR on SR and demosaicing, comparing
classical PnP, the proposed GSNR-PnP variants, NPN, and
several strong deep-learning baselines. Across both tasks,
GSNR-PnP with graph LaplaciansL 4nn andL 8nn achieves
the best performance. For SR, GSNR-PnP yields roughly
a2dB gain over vanilla PGD-PnP and a clear margin over
both NPN (withL=I) and task-specific unrolled/cascade
architectures. For demosaicing, the problem is less ill-
posed, so the absolute gains are smaller, but GSNR-PnP still
provides a consistent improvement over PGD-PnP, NPN,
and the deep baselines. Overall, these results show that
injecting graph-smooth NS structure into a standard PnP
solver consistently improves upon (i) geometry-free Pn-
P/NPN baselines and (ii) specialized deep unrolled or cas-
cade architectures, with particularly strong benefits on the
most underdetermined SR task. In A10 and A11, we show
additional results for Demosaicking and SR, respectively.
Table 2. Final PSNR (dB) for deblurring (σ k = 1) withn=
642 ·3andp= 0.8·n. We report results on Places365 [11]
and CelebA [25] test sets using Lip-DnCNN [40] and a wavelet
denoiser [12]. For each column, bestand second-best results.
Methodγ g Places365 CelebA
Lip-DnCNN Wavelet Lip-DnCNN Wavelet
Baseline (PGD-PnP) – 31.58 30.78 35.26 33.64
NPN [17] – 33.22 33.17 37.86 37.68
GSNR w.L8nn ✘ 33.60 33.59 38.18 37.99
GSNR w.L8nn ✓ 33.60 33.59 38.18 37.99
5.4. Plug-and-Play
We used the PGD-PnP [16, 20], but GSNR can be easily
adapted to other formulations, such as ADMM [6] or HQS
[33]. We used three denoisers: DnCNN with Lipchitz train-
ing (Lip-DnCNN) [40], DRUNet [52], Wavelet denoiser
(Daubechies 8 with 4 levels) using soft-thresholding prox-
imal [12]. For improving stability and performance of the
PGD-PnP, we include the equivariant denoising technique
[45] with two random90rotations and two reflections.
Deblurring: Table 2 reports the final PSNR for deblur-
ring on Places365 and CelebA under different graph vari-
ants and denoisers. In all settings, both NPN and GSNR
substantially improve over the baseline PGD-PnP solver:
on CelebA, the gain is on the order of+2dB for both Lip-
DnCNN and the wavelet denoiser, and on Places365, the
gain is around+1.5–2dB. The GSNR variant withL 8nn
provides a small but consistent improvement over NPN on
both datasets and for both denoisers, indicating that inject-
ing NS structure into the NS helps the solver recover sharper
details without overfitting. Interestingly, turning the addi-
tional graph penalty on or off (γ g ✓vs.✘) has a negligi-
ble effect on the final PSNR in this experiment, suggesting
that the learned GSNR already captures most of the useful
graph structure for deblurring. Figure 6 shows the deblur-
ring convergence behaviour with the DnCNN denoiser. The
baseline PnP-PGD curve increases steadily but saturates at a
lower PSNR plateau. In contrast, both GSNR variants (with
γg = 0andγ g = 0.1) converge to a noticeably higher final
PSNR, confirming that the graph-smooth null-space repre-
12475


<!-- PDF PAGE 8 -->

Figure 6. Deblurring convergence curves for PnP-PGD baseline
using Lip-DnCNN and GSNR variants withL 8nn with and with-
out the null-restricted Laplacian regularization.
sentation improves the fixed point of the algorithm. The ex-
plicit null-only graph penalty (γg = 0.1) mainly affects the
transient regime: it accelerates convergence in the first few
hundred iterations, reaching near-peak PSNR significantly
earlier than GSNR withγ g = 0(aligned with the theoreti-
cal analysis conducted in Sec. 4.3), while all three methods
eventually stabilize. Additional visual results in A12.
SR: Fig. 7 shows an SR example for different graph de-
signs and denoisers. Columns correspond to the choice of
LaplacianLor to the baseline PnP without GSNR, while
rows correspond to different denoisers. In all three rows,
moving from the baseline andL=Icolumns to the graph-
basedL 4nn andL 8nn columns produces visibly sharper
details: facial contours are crisper, hair strands are bet-
ter resolved, and blocky artifacts inherited from the low-
resolution backprojectionH ⊤yare reduced. The improve-
ments are most pronounced for the DnCNN and DRUNet
denoisers. Still, they are also clearly present for the simpler
Wavelet denoiser, indicating that the benefit comes from
the GSNR itself rather than from the denoiser. Among the
graph designs,L 8nn yields the visually clearest reconstruc-
tion, suggesting that richer graph connectivity enables the
solver to exploit structure in the null space rather than hallu-
cinating high-frequency content. Results with Deep Image
Prior [49] and SAR images in SR can be found in A11.
6. Discussion
We demonstrate that GSNR is a versatile approach for a
broad range of inverse problems, data modalities, and neu-
ral networks (A14). Nevertheless, the method requires pre-
cise knowledge of the sensing matrix, which is unavail-
able in some applications, e.g., real-world image deblurring.
In A15, we provide analysis on the effect of uncertainty
in the sensing operator, where a performance drop is evi-
dent, but the improvement compared with the baseline algo-
rithm is maintained. Follow-up ideas will harness the null-
space construction under sensing matrix uncertainty for cal-
ibrated null-space regularization functions. Another aspect
Figure 7. Results of PnP, NPN, and GSNR varyingLand denois-
ers. Best results for each denoiser are in bold. Here,p= 0.1n.
of GSNR is the requirement to perform EVD on the null-
restricted LaplacianT, which, at large image scales, neces-
sitates substantial computational resources. However, this
computation is performedofflineonce per(H,L, n, p), and
reused for learning the networkGand the reconstruction
step. In A13, we provide a detailed runtime analysis of the
approach. Additionally, our approach is designed for lin-
ear inverse problems; the adaptation for nonlinear inverse
problems, e.g., phase retrieval, requires careful modeling of
the invertibility properties of the sensing operator and its
input-dependent ambiguities. In recent work [14], a gener-
alization of the null-space property was developed for neu-
ral networks (non-linear functions), opening new frontiers
for learning optimal null-space components.
7. Conclusion and Future Outlook
We introducedGSNR: Graph-Smooth Null-Space Repre-
sentation, which equips an inverse problem solver not only
withwhatthe measurements constrain, but also withhowto
use the degrees of freedom that the sensor never sees (null-
space). Instead of regularizing the full image, GSNR builds
a low-dimensional basis forNull(H)aligned with graph
smoothness via the null-restricted LaplacianT=P nLPn,
and can optionally add a null-only graph penalty at infer-
ence. Theoretically, the smoothestT-modes are minimax-
optimal for null-space components, and under a GMRF
prior, they provably capture more null-space variance than
geometry-free choices while enjoying stronger statistical
coupling to the measurements. In practice, this yields bet-
ter coverage of plausible solutions, higher predictability of
null-space signal components, and improved conditioning,
translating into faster and more stable convergence of PnP
and diffusion-based solvers. Looking ahead, rather than fix-
ingLto a hand-crafted grid Laplacian, learning the graph
structure or employing interpretable graph neural operators
promises data-adaptive GSNR bases that better reflect se-
mantic structure. Also, a more rigorous study of GSNR
within generative solvers could provide a principled way to
inject graph-smooth null constraints into these frameworks
without breaking their probabilistic interpretation.
12476


<!-- PDF PAGE 9 -->

Acknowledgements
This work was supported by the OFFICE OF NA V AL
RESEARCH under Grant W911NF2510165, Project ti-
tled “ADV ANCED ALGORITHMS FOR DISTRIBUTED
BEAMFORMING”. The views and conclusions contained
in this document are those of the authors and should not
be interpreted as representing the official policies, either
expressed or implied, of the OFFICE OF NA V AL RE-
SEARCH or the U.S. Government.
References
[1] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge
on single image super-resolution: Dataset and study. InThe
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR) Workshops, 2017. 11
[2] Daniel Otero Baguer, Johannes Leuschner, and Maximilian
Schmidt. Computed tomography reconstruction using deep
image prior and learned reconstruction methods.Inverse
Problems, 36(9):094004, 2020. 1
[3] Bryce E Bayer. Color imaging array, 1976. US Patent
3,971,065. 6
[4] Martin Benning and Martin Burger. Modern regularization
methods for inverse problems.Acta numerica, 27:1–111,
2018. 1
[5] E. J. Candes and M. B. Wakin. An introduction to com-
pressive sampling.IEEE Signal Processing Magazine, 25
(2):21–30, 2008. 1, 6
[6] Stanley H Chan, Xiran Wang, and Omar A Elgendy. Plug-
and-play admm for image restoration: Fixed-point conver-
gence and applications.IEEE Transactions on Computa-
tional Imaging, 3(1):84–98, 2016. 1, 7
[7] Dongdong Chen and Mike E Davies. Deep decomposition
learning for inverse imaging problems. InComputer Vision–
ECCV 2020: 16th European Conference, Glasgow, UK, Au-
gust 23–28, 2020, Proceedings, Part XXVIII 16, pages 510–
526. Springer, 2020. 2, 7
[8] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mc-
cann, Marc Louis Klasky, and Jong Chul Ye. Diffusion pos-
terior sampling for general noisy inverse problems. InThe
Eleventh International Conference on Learning Representa-
tions, 2023. 6, 7, 1
[9] Regev Cohen, Michael Elad, and Peyman Milanfar. Regu-
larization by denoising via fixed-point projection (red-pro).
SIAM Journal on Imaging Sciences, 14(3):1374–1406, 2021.
1
[10] Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll,
Mubarak Shah, and Fahad Shahbaz Khan. AdaIR: Adap-
tive all-in-one image restoration via frequency mining and
modulation. InThe Thirteenth International Conference on
Learning Representations, 2025. 11
[11] Emanuele Dalsasso, Xiangli Yang, Lo ¨ıc Denis, Florence
Tupin, and Wen Yang. Sar image despeckling by deep neural
networks: from a pre-trained model to an end-to-end training
strategy.Remote Sensing, 12(16), 2020. 6, 7
[12] D.L. Donoho. De-noising by soft-thresholding.IEEE Trans-
actions on Information Theory, 41(3):613–627, 1995. 7
[13] David L Donoho and Iain M Johnstone. Ideal spatial adapta-
tion by wavelet shrinkage.Biometrika, 81(3):425–455, 1994.
2
[14] Yamit Ehrlich, Nimrod Berman, and Assaf Shocher. Pseudo-
invertible neural networks, 2026. 8
[15] Yutong He, Naoki Murata, Chieh-Hsin Lai, Yuhta Takida,
Toshimitsu Uesaka, Dongjun Kim, Wei-Hsiang Liao, Yuki
Mitsufuji, J Zico Kolter, Ruslan Salakhutdinov, and Ste-
fano Ermon. Manifold preserving guided diffusion. InThe
Twelfth International Conference on Learning Representa-
tions, 2024. 7, 1, 8
[16] Samuel Hurault, Antonin Chambolle, Arthur Leclaire, and
Nicolas Papadakis. A relaxed proximal gradient descent
algorithm for convergent plug-and-play with proximal de-
noiser. InInternational Conference on Scale Space and
Variational Methods in Computer Vision, pages 379–392.
Springer, 2023. 7
[17] Roman Jacome, Romario Gualdr ´on-Hurtado, Le ´on Su ´arez-
Rodr´ıguez, and Henry Arguello. NPN: Non-Linear Projec-
tions of the Null-Space for Imaging Inverse Problems. In
The Thirty-ninth Annual Conference on Neural Information
Processing Systems, 2025. 2, 6, 7, 8, 9, 11
[18] Ian T Jolliffe and Jorge Cadima. Principal component analy-
sis: a review and recent developments.Philosophical trans-
actions of the royal society A: Mathematical, Physical and
Engineering Sciences, 374(2065):20150202, 2016. 4
[19] Vassilis Kalofolias. How to learn a graph from smooth sig-
nals. InProceedings of the 19th International Conference on
Artificial Intelligence and Statistics, pages 920–929, Cadiz,
Spain, 2016. PMLR. 2
[20] Ulugbek S. Kamilov, Charles A. Bouman, Gregery T. Buz-
zard, and Brendt Wohlberg. Plug-and-play methods for inte-
grating physical and learned models in computational imag-
ing: Theory, algorithms, and applications.IEEE Signal Pro-
cessing Magazine, 40(1):85–97, 2023. 1, 2, 7
[21] DP Kingma and J Ba. Adam: a method for stochastic opti-
mization, 2014. 6
[22] Lingshun Kong, Jiangxin Dong, Jinhui Tang, Ming-Hsuan
Yang, and Jinshan Pan. Efficient visual state space model for
image deblurring. InProceedings of the computer vision and
pattern recognition conference, pages 12710–12719, 2025.
11
[23] A. Krizhevsky. Learning multiple layers of features from tiny
images.Master’s thesis, University of Tront, page 1, 2009. 6
[24] Richard B Lehoucq, Danny C Sorensen, and Chao Yang.
ARPACK users’ guide: solution of large-scale eigenvalue
problems with implicitly restarted Arnoldi methods. SIAM,
1998. 4
[25] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.
Deep learning face attributes in the wild. InProceedings of
International Conference on Computer Vision (ICCV), 2015.
6, 7
[26] Michael Lustig, David L Donoho, Juan M Santos, and
John M Pauly. Compressed sensing mri.IEEE signal pro-
cessing magazine, 25(2):72–82, 2008. 1
[27] Peyman Milanfar and Mauricio Delbracio. Denoising: a
powerful building block for imaging, inverse problems and
12477


<!-- PDF PAGE 10 -->

machine learning.Philosophical Transactions A, 383(2299):
20240326, 2025. 1
[28] Sunil K Narang, Yung Hsuan Chao, and Antonio Ortega.
Graph-wavelet filterbanks for edge-aware image processing.
In2012 IEEE Statistical Signal Processing Workshop (SSP),
pages 141–144, 2012. 2
[29] Antonio Ortega, Pascal Frossard, Jelena Kovaˇcevi´c, Jos´e MF
Moura, and Pierre Vandergheynst. Graph signal processing:
Overview, challenges, and applications.Proceedings of the
IEEE, 106(5):808–828, 2018. 2, 3, 4
[30] Neal Parikh and Stephen Boyd. Proximal algorithms.Foun-
dations and Trends in Optimization, 1(3):127–239, 2014. 2
[31] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison,
Andreas Kopf, Edward Yang, Zachary DeVito, Martin Rai-
son, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An
imperative style, high-performance deep learning library. In
Advances in Neural Information Processing Systems. Curran
Associates, Inc., 2019. 6
[32] Gabriel Peyr ´e, S ´ebastien Bougleux, and Laurent Cohen.
Non-local regularization of inverse problems. InComputer
Vision – ECCV 2008, pages 57–68, Berlin, Heidelberg, 2008.
Springer Berlin Heidelberg. 2
[33] Arash Rasti-Meymandi, Aboozar Ghaffari, and Emad
Fatemizadeh. Plug and play augmented hqs: Convergence
analysis and its application in mri reconstruction.Neuro-
computing, 518:1–14, 2023. 7
[34] Edward T Reehorst and Philip Schniter. Regularization by
denoising: Clarifications and new interpretations.IEEE
transactions on computational imaging, 5(1):52–67, 2018.
1
[35] Yaniv Romano, Michael Elad, and Peyman Milanfar. The
little engine that could: Regularization by denoising (red).
SIAM journal on imaging sciences, 10(4):1804–1844, 2017.
1, 2
[36] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj ¨orn Ommer. High-resolution image
synthesis with latent diffusion models. InProceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 8
[37] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-
net: Convolutional networks for biomedical image segmen-
tation. InInternational Conference on Medical image com-
puting and computer-assisted intervention, pages 234–241.
Springer, 2015. 6
[38] Leonid I. Rudin, Stanley Osher, and Emad Fatemi. Nonlinear
total variation based noise removal algorithms.Physica D:
Nonlinear Phenomena, 60(1):259–268, 1992. 2
[39] Havard Rue and Leonhard Held.Gaussian Markov random
fields: theory and applications. Chapman and Hall/CRC,
2005. 4
[40] Ernest Ryu, Jialin Liu, Sicheng Wang, Xiaohan Chen,
Zhangyang Wang, and Wotao Yin. Plug-and-play methods
provably converge with properly trained denoisers. InIn-
ternational Conference on Machine Learning, pages 5546–
5557. PMLR, 2019. 7, 9
[41] Johannes Schwab, Stephan Antholzer, and Markus Halt-
meier. Deep null space learning for inverse problems:
convergence analysis and rates.Inverse Problems, 35(2):
025008, 2019. 2, 7
[42] David I Shuman, Sunil K Narang, Pascal Frossard, Antonio
Ortega, and Pierre Vandergheynst. The emerging field of sig-
nal processing on graphs: Extending high-dimensional data
analysis to networks and other irregular domains.IEEE sig-
nal processing magazine, 30(3):83–98, 2013. 3, 4
[43] Juli ´an Tachella, Matthieu Terris, Samuel Hurault, Andrew
Wang, Leo Davy, J ´er´emy Scanvic, Victor Sechaud, Ro-
main V o, Thomas Moreau, Thomas Davies, Dongdong Chen,
Nils Laurent, Brayan Monroy, Jonathan Dong, Zhiyuan Hu,
Minh-Hai Nguyen, Florian Sarron, Pierre Weiss, Paul Es-
cande, Mathurin Massias, Thibaut Modrzyk, Brett Levac,
Tob´ıas I. Liaudat, Maxime Song, Johannes Hertrich, Se-
bastian Neumayer, and Georg Schramm. Deepinverse: A
python package for solving imaging inverse problems with
deep learning.Journal of Open Source Software, 10(115):
8923, 2025. 6
[44] Hong Ye Tan, Subhadip Mukherjee, and Junqi Tang. Solv-
ing imaging inverse problems using plug-and-play denoisers:
Regularization and optimization perspectives, 2025. 1
[45] Matthieu Terris, Thomas Moreau, Nelly Pustelnik, and Ju-
lian Tachella. Equivariant plug-and-play image reconstruc-
tion. InProceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 25255–25264,
2024. 7
[46] Jing Tian and Kai-Kuang Ma. A survey on super-resolution
imaging.Signal, Image and Video Processing, 5:329–342,
2011. 1
[47] Xiangpeng Tian, Xiangyu Liao, Xiao Liu, Meng Li, and
Chao Ren. Degradation-aware feature perturbation for all-in-
one image restoration. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
28165–28175, 2025. 11
[48] Singanallur V . Venkatakrishnan, Charles A. Bouman, and
Brendt Wohlberg. Plug-and-play priors for model based re-
construction. In2013 IEEE Global Conference on Signal
and Information Processing, pages 945–948, 2013. 1
[49] Lizhi Wang, Chen Sun, Ying Fu, Min H Kim, and Hua
Huang. Hyperspectral image reconstruction using a deep
spatial-spectral prior. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
8032–8041, 2019. 8
[50] Jinxi Xiang, Yonggui Dong, and Yunjie Yang. Fista-net:
Learning a fast iterative shrinkage thresholding network for
inverse problems in imaging.IEEE Transactions on Medical
Imaging, 40(5):1329–1339, 2021. 7
[51] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and
Thomas S Huang. Generative image inpainting with con-
textual attention. InProceedings of the IEEE conference on
computer vision and pattern recognition, pages 5505–5514,
2018. 1
[52] Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc
Van Gool, and Radu Timofte. Plug-and-play image restora-
tion with deep denoiser prior.IEEE Transactions on Pat-
12478


<!-- PDF PAGE 11 -->

tern Analysis and Machine Intelligence, 44(10):6360–6376,
2021. 1, 7, 9
[53] Kaihao Zhang, Wenqi Ren, Wenhan Luo, Wei-Sheng Lai,
Bj¨orn Stenger, Ming-Hsuan Yang, and Hongdong Li. Deep
image deblurring: A survey.International Journal of Com-
puter Vision, 130(9):2103–2130, 2022. 1
[54] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva,
and Antonio Torralba. Places: A 10 million image database
for scene recognition.IEEE Transactions on Pattern Analy-
sis and Machine Intelligence, 2017. 6
[55] Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bi-
han Wen, Radu Timofte, and Luc Van Gool. Denoising dif-
fusion models for plug-and-play image restoration. InPro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) Workshops, pages 1219–
1229, 2023. 6, 7, 1, 8
12479

<!-- END-OF-PAPER:e56a8daee065 -->
