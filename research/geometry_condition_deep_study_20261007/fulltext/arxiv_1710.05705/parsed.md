Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation

    ![](/static/base/1.0.1/images/icons/smileybones-small.svg)
    arXiv is now an independent nonprofit!
    Learn more
    ×

    License: arXiv.org perpetual non-exclusive license

arXiv:1710.05705v4 [cs.CV] 09 Apr 2018

# Blind Image Fusion for Hyperspectral Imaging with the Directional Total Variation

Journal: Inverse Problems

Leon Bungert

  
David A. Coomes

  
Matthias J. Ehrhardt

  
Jennifer Rasch

  
Rafael Reisenhofer

Affiliation:  University of Bremen, Fachbereich 3, Postfach 330440, 28334 Bremen, Germany

  
Carola-Bibiane Schönlieb

Email: m.j.ehrhardt@damtp.cam.ac.uk

###### Abstract

Hyperspectral imaging is a cutting-edge type of remote sensing used for mapping vegetation properties, rock minerals and other materials. A major drawback of hyperspectral imaging devices is their intrinsic low spatial resolution. In this paper, we propose a method for increasing the spatial resolution of a hyperspectral image by fusing it with an image of higher spatial resolution that was obtained with a different imaging modality. This is accomplished by solving a variational problem in which the regularization functional is the directional total variation. To accommodate for possible mis-registrations between the two images, we consider a non-convex blind super-resolution problem where both a fused image and the corresponding convolution kernel are estimated. Using this approach, our model can realign the given images if needed. Our experimental results indicate that the non-convexity is negligible in practice and that reliable solutions can be computed using a variety of different optimization algorithms. Numerical results on real remote sensing data from plant sciences and urban monitoring show the potential of the proposed method and suggests that it is robust with respect to the regularization parameters, mis-registration and the shape of the kernel.

###### ams

49M37, 65K10, 90C30, 90C90

###### pacs

42.30.Va, 42.68.Wt, 95.75.Pq, 95.75.Rs

Keywords: remote sensing, super-resolution, pansharpening, blind deconvolution, hyperspectral imaging

## 1 Introduction

Hyperspectral imaging is an earth observation technique that measures the energy of light reflected off the earth’s surface within many narrow spectral bands, from which reflectance curves can be calculated. The shape of these reflectance curves provides information about the chemical and physical properties of materials such as minerals in rocks, vegetation, synthetic materials and water, which allows these materials to be classified and mapped remotely (see [1, 2], for instance).

Usually, hyperspectral images of the earth are being recorded by imaging devices that are mounted on planes or satellites. In either case, the speed of the carrier and narrowness of the spectral bands result in limited energy reaching the sensor for each spectral band. Therefore, most hyperspectral cameras have an intrinsic trade-off between spatial and spectral resolution [3, 4]. A standard approach to overcoming this trade-off is to record a second image with low spectral but high spatial resolution (e.g. a photograph or a laser scan) [5] and to fuse these data together in order to create an image that has the high spectral resolution of the hyperspectral image and the high spatial resolution of the photograph [5, 6, 7, 8, 3, 9, 4, 10, 11, 12, 13]. These techniques have become known as *pansharpening* or *image fusion*. The problem is illustrated by three example data sets in figure 1.

![Refer to caption](1710.05705v4/fig_1a.png)

![Refer to caption](1710.05705v4/fig_1b.png)

![Refer to caption](1710.05705v4/fig_1c.png)

*Figure 1: Three example data for image fusion in remote sensing. They each consist of a hyperspectral image (small image, only one channel shown) and an image of higher spatial resolution (large image). The goal is to create an image that has both high spatial and high spectral resolution.*

Most image fusion techniques are based on certain assumptions regarding the data and the corresponding imaging devices. First, it is often assumed that the image of high spatial resolution is a linear combination of the spectral channels with known weights [4]. Second, the loss of resolution is usually modeled as a linear operator which consists of a subsampled convolution with known kernel (point spread function). While both assumptions may be justified in some applications, it may be difficult to measure or estimate the weights and the convolution kernel in a practical situation.

Instead of having accurate descriptions or measurements about the system response of our cameras [6, 4, 12], we make use of the fact that both images are acquired from the same scenery. While the actual image intensities will depend on the wavelengths that are being recorded and are therefore dissimilar, the geometric structure of those images is likely to be very similar as they are both taken from the same terrain [6, 7, 9, 12]. This approach has been studied extensively in the context of medical imaging where traditionally an anatomical imaging modality with high spatial resolution is used to aid the reconstruction of a functional imaging modality, cf. e.g. [14, 15, 16, 17, 18, 19, 20].

Note that the hyperspectral data image and the high-resolution photograph usually undergo a georectification procedure which aligns them using a digital elevation map of the study area. However, this adjustment is often imprecise, particularly in topographically complex regions, so further steps are required to co-register the images precisely [21]. Moreover, having a good description of the point spread function of the system is very unnatural. Thus, we take a blind approach and estimate the point spread function within our reconstruction and thereby learn this information from the recorded data. This approach is intrinsically related to blind deconvolution [22, 23, 24, 25]. Figure 2 shows an example of blind vs non-blind image fusion applied to the real remote sensing data with unknown blurring kernel. It can be seen that the shape of the kernel that is estimated during the reconstruction clearly differs from the fixed kernel. Equally important is the translation of the estimated kernel, which can compensate for shifts between the given image pair. To the best of our knowledge this is the first contribution where structural side information / anatomical priors are combined with registration which removes the unnatural assumption of perfectly registered images that is usually made, cf. [14, 15, 16, 17, 18, 19, 20], for instance.

![Refer to caption](1710.05705v4/fig_2.png)

****

*Figure 2: Comparison of non-blind versus blind image fusion. From left to right: hyperspectral image (one channel, four times enlarged), result of non-blind and blind image fusion together with the kernels that are used or estimated, respectively. The plotted kernels are of size 41×4141\times 41 and have been enlarged for better visibility. The dotted green lines highlight the center of mass of the obtained kernel and indicate the amount of translational displacement between the fused images.*

## 2 Mathematical Model and Notation

In this work we consider the problem of simultaneous image fusion and blind image deconvolution which can be cast as the optimization problem

(u∗,k∗)∈arg⁡min(u,k)∈U×K​{12​‖𝐀k​u−f‖2+ℛu​(u)+ℛk​(k)}(u^{\ast},k^{\ast})\in\operatorname{arg}\underset{(u,k)\in U\times K}{\min}\;\left\{\frac{1}{2}\|\mathbf{A}_{k}u-f\|^{2}+\mathcal{R}_{u}(u)+\mathcal{R}_{k}(k)\right\}

(1)

where the *forward operator* 𝐀k\mathbf{A}_{k} is a sampled blur with kernel kk. Thus, it relates an image u∈U:=ℝmu\in U:=\mathbb{R}^{m} of size m=(m1,m2)m=(m_{1},m_{2}) to blurred and subsampled data f∈ℝn,n=(n1,n2)f\in\mathbb{R}^{n},n=(n_{1},n_{2}). Note that we used *multi-index notation* which means ℝm=ℝm1×m2\mathbb{R}^{m}=\mathbb{R}^{m_{1}\times m_{2}}, for instance. Here, we consider the case of *super-resolution* n<mn<m (true if and only if n1<m1n_{1}<m_{1} and n2<m2n_{2}<m_{2}). The data discrepancy is measured in terms of the Euclidean / Frobenius norm ‖x‖2:=∑ixi2\|x\|^{2}:=\sum_{i}x_{i}^{2} and the optimal solution is *regularized* through the penalty functionals ℛu\mathcal{R}_{u} and ℛk\mathcal{R}_{k}.

Since—up to now—the ingredients of our model are only vaguely defined, the remainder of this section is devoted to an detailed explanation of the operators and regularization functionals involved.

### 2.1 The Forward Operator

The forward operator 𝐀k≔𝐒∘𝐁∘𝐂k\mathbf{A}_{k}\;\coloneqq\;\mathbf{S}\circ\mathbf{B}\circ\mathbf{C}_{k} is the composition of three operators: 𝐂k\mathbf{C}_{k} represents a *convolution* with a kernel kk, 𝐁\mathbf{B} performs *boundary treatment* and 𝐒\mathbf{S} is a *sampling operator*.

The convolution operator 𝐂k:U→U\mathbf{C}_{k}:U\to U with kernel k∈K:=ℝr,r:=(r1,r2)k\in K:=\mathbb{R}^{r},r:=(r_{1},r_{2}) is defined by

𝐂k​u≔𝐉​k∗u\mathbf{C}_{k}u\;\coloneqq\;\mathbf{J}k\ast u

(2)

where ∗\ast is the cyclic convolution of images in U=ℝmU=\mathbb{R}^{m}. Since the kernel kk is generally assumed to have smaller support than the image u{u}, i.e. r<mr<m, we introduced the embedding operator 𝐉:K→U\mathbf{J}:K\rightarrow U which embeds the ‘small’ convolution kernel k∈Kk\in K into the image space UU by padding it with zeros. By assuming periodic boundary conditions on uu and 𝐉​k\mathbf{J}k, the convolution can be efficiently computed via the discrete Fourier transform 𝐅\mathbf{F}, i.e. 𝐂k​u=𝐅−1​(𝐅𝐉​k⊙𝐅​u)\mathbf{C}_{k}u=\mathbf{F}^{-1}\left(\mathbf{F}\mathbf{J}k\odot\mathbf{F}u\right) where ⊙\odot denotes the Hadamard product (pointwise multiplication). With a slight abuse of notation, we can exploit the symmetry of the convolution to infer

𝐂k​u=𝐉​k∗u=u∗𝐉​k=𝐂u​𝐉​k,\mathbf{C}_{k}u=\mathbf{J}k\ast u=u\ast\mathbf{J}k=\mathbf{C}_{u}\mathbf{J}k\,,

(3)

and thus

𝐀k​u=𝐀u​𝐉​k,\mathbf{A}_{k}u=\mathbf{A}_{u}\mathbf{J}k\,,

(4)

which will prove helpful later-on.

While periodic boundary conditions are computationally useful, any kind of boundary treatment is artificial and may result in artifacts. For instance, boundary artifacts are expected in the boundary region with margins l≔(r−1)/2l\;\coloneqq\;(r-1)/2 where rr is the diameter of the convolution kernel. To avoid these artefacts, we follow the *boundary-layer approach* (cf. [26] and references therein; also described by Almeida and Figueiredo in [27]) and introduce a boundary clipping (margin deletion) operator 𝐁:U→ℝm−2​l\mathbf{B}:U\to\mathbb{R}^{m-2l},

(𝐁​u)i≔ui+l(\mathbf{B}u)_{i}\;\coloneqq\;u_{i+l}

(5)

that maps a large image uu to a *meaningful image* 𝐁​u∈ℝm−2​l\mathbf{B}u\in\mathbb{R}^{m-2l} which is independent of the type of boundary conditions used for the precedent convolution. Thus, after computation of an optimal solution u∗u^{\ast} of (1) we only use the meaningful part 𝐁​u∗\mathbf{B}u^{\ast}.

Finally, the third component of our forward operator is the sampling 𝐒:ℝm−2​l→ℝn\mathbf{S}:\mathbb{R}^{m-2l}\to\mathbb{R}^{n} which mimics an integration sensor of size s×ss\times s, i.e.

(𝐒​u)i≔1s2​∑j∈Δiuj(\mathbf{S}u)_{i}\;\coloneqq\;\frac{1}{s^{2}}\sum_{j\in\Delta_{i}}u_{j}

(6)

where the detector indices are given by the set Δi:={j=(j1,j2)|i≤j<i+s}\Delta_{i}:=\{j=(j_{1},j_{2})\,|\,i\leq j<i+s\}, which contains s2s^{2} elements. Consequently, for a kernel of odd dimensions the relation m−2​l=s​nm-2l=sn connects the sizes of kernel, image and data.

Let us now turn to the regularization and prior knowledge on the image uu and the kernel kk.

### 2.2 Regularization Functionals

#### 2.2.1 Image Regularization

A popular prior to regularize images is the *total variation* [28] as it allows the recovery of smooth images whilst preserving edges. It is defined as the global 1-norm of the discretized gradient

TV⁡(u)≔∑i‖∇ui‖.\mathrm{TV}(u)\;\coloneqq\;\sum_{i}\|\nabla u_{i}\|\,.

(7)

Due to the periodic boundary conditions needed for the forward operator, the discretized gradient (with forward differences) ∇u∈U2\nabla u\in U^{2} of an image u∈Uu\in U is

(∇ui)j≔umod⁡(i+ej,m)−ui,j∈{1,2},(\nabla u_{i})_{j}\;\coloneqq\;u_{\operatorname{mod}(i+e_{j},m)}-u_{i}\,,\quad j\in\{1,2\},

(8)

where e1=(1,0)e_{1}=(1,0) and e2=(0,1)e_{2}=(0,1) are the standard unit vectors in ℝ2\mathbb{R}^{2}.

In the task at hand, the high-resolution photograph can be viewed as a source of additional a-priori knowledge about the sought solution. As both the hyperspectral data and the high-resolution photograph depict the same scenery, it is reasonable to assume that they show the similar geometrical structures albeit having very different intensities. However, the TV functional (7) is not able to incorporate this kind of additional prior information. Instead, we will apply the so-called directional TV (dTV), which can fuse the structural information of a high-resolution image with the intensities of a low-resolution channel of a hyperspectral image. Figure 3 provides a visual comparison of reconstructions obtained from applying the TV and the dTV functional, respectively.

###### Definition (Directional Total Variation [18]).

Let ξ∈U2,‖ξi‖≤γ<1\xi\in U^{2},\|\xi_{i}\|\leq\gamma<1 be a vector-field that determines directions and denote by 𝐏∈U2×2,𝐏i:=𝐈−ξi⊗ξi\mathbf{P}\in U^{2\times 2},\mathbf{P}_{i}:=\mathbf{I}-\xi_{i}\otimes\xi_{i} an associated matrix-field where 𝐈\mathbf{I} is the 2×22\times 2 identity matrix and ⊗\otimes marks the outer product of vectors. Then the *directional total variation* dTV:U→ℝ\mathrm{dTV}:U\to\mathbb{R} is defined as as

dTV(u)≔∑i∥𝐏i∇ui∥.\mathrm{dTV}(u)\;\coloneqq\;\sum_{i}\|\mathbf{P}_{i}\nabla u_{i}\|\,.

(9)

Note that due to the linearity of 𝐏i\mathbf{P}_{i} the directional total variation is *convex*.

![Refer to caption](1710.05705v4/fig_3.png)

*Figure 3: Comparison of total variation (TV\mathrm{TV}) versus directional total variation (dTV\mathrm{dTV}) regularization for image fusion. From left to right: hyperspectral image (one channel, four times enlarged), results of TV\mathrm{TV} and dTV\mathrm{dTV} together with the estimated kernels.*

Let us shortly comment on the interpretation of the directional total variation. The vector-field ξ\xi allows the definition of directions that should be less penalized and therefore get promoted. It is easy to see that the quantity 𝐏i∇ui\mathbf{P}_{i}\nabla u_{i} can be expanded as

𝐏i∇ui=∇ui−⟨ξi,∇ui⟩ξi\mathbf{P}_{i}\nabla u_{i}=\nabla u_{i}-\langle\xi_{i},\nabla u_{i}\rangle\xi_{i}

(10)

where ⟨⋅,⋅⟩\langle\cdot,\cdot\rangle denotes the inner product on ℝ2\mathbb{R}^{2}. This expression reduces to (1−∥ξi∥2)∇ui(1-\|\xi_{i}\|^{2})\nabla u_{i} in regions where ∇ui\nabla u_{i} is collinear to ξi\xi_{i}, and to ∇ui\nabla u_{i} where ∇ui\nabla u_{i} is orthogonal to ξi\xi_{i}. Thus, gradients that are aligned / collinear ξi\xi_{i} are favored as long as ‖ξi‖>0\|\xi_{i}\|>0. In the extreme case ‖ξi‖=1\|\xi_{i}\|=1, aligned gradients are not penalized any more. It is important to note, that this prior *does not enforce* gradients in the direction of ξi\xi_{i}, as a gradient in the direction of ξi\xi_{i} is never ‘cheaper’ than a zero gradient.

The uniform upper bound γ<1\gamma<1 ensures that the directional total variation is equivalent to the total variation in a semi-norm sense, i.e.

(1−γ2)​TV​(u)≤dTV⁡(u)≤TV⁡(u).(1-\gamma^{2})\mathrm{TV}(u)\leq\mathrm{dTV}(u)\leq\mathrm{TV}(u)\,.

(11)

Similar versions of the directional total variation have been used before [29, 17, 19] and it is related to other anisotropic priors [14, 30, 31] and the notion of parallel level sets [32, 33, 34]. Note that the directional total variation generalizes the usual total variation (7) for ξ=0\xi=0 and other versions of the directional total variation [35, 36] where the direction ξ\xi is constant and not depending in the pixel location ii. With the isotropic choice 𝐏i=wi​𝐈, 0≤wi≤1\mathbf{P}_{i}=w_{i}\mathbf{I},\,0\leq w_{i}\leq 1 it reduces to weighted total variation [37, 38, 39, 29, 18].

![Refer to caption](1710.05705v4/fig_4a.png)

![Refer to caption](1710.05705v4/fig_4b.png)

****

*Figure 4: Side information for Directional Total Variation. The side information is given as an RGB image (left) and converted into gray scales (center) from which a vector-field may be computed (right), e.g. with (12) (shown for γ=1,ε=0.3\gamma=1,\varepsilon=0.3). The close-ups show that the vector-field does not only carry detailed information about edge locations but also about edge directions.*

Having this definition of the directional total variation at hand, the question is how to define the vector-field ξ∈U2\xi\in U^{2}. As we want to favor images having a similar structure as the high-resolution photograph, we let v∈Uv\in U denote a gray-scale image which is generated from the color high-resolution image and define the vector-field

ξi≔γ​∇vi‖∇vi‖ε,\xi_{i}\;\coloneqq\;\gamma\frac{\nabla v_{i}}{\|{\nabla v}_{i}\|_{\varepsilon}}\,,

(12)

where ‖x‖ε≔‖x‖2+ε2\|x\|_{\varepsilon}\;\coloneqq\;\sqrt{\|x\|^{2}+\varepsilon^{2}} and the scaling γ∈[0,1)\gamma\in[0,1) ensures that ‖ξi‖≤γ\|\xi_{i}\|\leq\gamma. The parameter ε>0\varepsilon>0 guarantees that the vector-field ξ\xi is well-defined, even if ∇vi=0\nabla v_{i}=0. Moreover, it allows to control which gradient magnitudes ‖∇vi‖\|\nabla v_{i}\| should be considered meaningful and which are merely due to noise. Thus, both parameters ε\varepsilon and γ\gamma in general will depend on the given side information vv but can be chosen without considering a particular data set ff. Note that in practice, one chooses γ\gamma very close or even equal to 1 (cf. [18] for the latter) such that the side information image has sufficiently much influence. Extensive studies of this parameter are performed in [36] where the authors obtain their best results for small values of a≔ 1−γ2a\;\coloneqq\;1-\gamma^{2}. An example photograph, its gray scale version and the corresponding vector-field are shown in figure 4.

In addition to the structural a-priori knowledge given by the photograph and encoded in the directional total variation, it is reasonable to assume that the solution we are looking for is non-negative. Thus, the regularization functional for the image becomes

ℛu≔λudTV+ı[0,∞)m\mathcal{R}_{u}\;\coloneqq\;\lambda_{u}\mathrm{dTV}+\imath_{[0,\infty)^{m}}

(13)

where the characteristic function of the non-negative quadrant is defined as

ı[0,∞)m(u)≔{0if u∈[0,∞)m∞else.\imath_{[0,\infty)^{m}}(u)\;\coloneqq\;\begin{cases}0&if $u\in[0,\infty)^{m}$\\
\infty&else\end{cases}\,.

(14)

The regularization parameter for the image λu>0\lambda_{u}>0 controls the trade-off between regularity and data fidelity.

### 2.3 Kernel Regularization

Similarly as for the image uu, it is reasonable to assume that the kernel is non-negative and regular. Motion blurs that arise from ideal diffraction-free lenses have typically a compactly supported kernel with sharp cut-off boundaries [26, p. 234]. Thus, the desired regularity of the kernel may be achieved by utilizing the total variation. Of course, other smoothness priors such as the H1H^{1}-semi-norm or total generalized variation [40] are possible, too, and will be subject of future work. In addition, blind deconvolution has the intrinsic non-uniqueness in the problem formulation that does not allow to estimate the correct global scaling factors for both the kernel and the image. To circumvent this issue, a common choice (e.g. [24, 41, 25, 26]) is to normalize the kernel with respect to the 1-norm. Thus, together with the non-negativity, we assume that the kernel is in the unit simplex

𝕊≔{k∈K:ki≥0,∑iki=1}.\mathbb{S}\;\coloneqq\;\left\{k\in K\,:\,k_{i}\geq 0,\,\sum_{i}k_{i}=1\right\}\,.

(15)

Letting ı𝕊\imath_{\mathbb{S}} denote the characteristic function of the unit simplex, defined analogously to (14), the regularization functional for the kernel becomes

ℛk≔λk​TV+ı𝕊\mathcal{R}_{k}\;\coloneqq\;\lambda_{k}\mathrm{TV}+\imath_{\mathbb{S}}

(16)

where we again can trade-off regularity and data fidelity by the regularization parameter λk>0\lambda_{k}>0 which in general needs to be chosen differently than the regularization parameter for the image λu\lambda_{u}.

## 3 Algorithms

Given the abstract mathematical model (1) which models the solution that we are after, this section discusses algorithms that will attempt to numerically compute this solution. For a concise presentation, we cast problem (1) in the form

(u∗,k∗)∈arg⁡minu,k​Ψ​(u,k)(u^{\ast},k^{\ast})\in\operatorname{arg}\underset{u,k}{\min}\;\Psi(u,k)

(17)

by defining Ψ⁡(u,k)≔𝒟⁡(u,k)+ℛu​(u)+ℛk​(k)\Psi(u,k)\;\coloneqq\;\mathcal{D}(u,k)+\mathcal{R}_{u}(u)+\mathcal{R}_{k}(k) and abbreviating the data fidelity term in (1) by 𝒟⁡(u,k)≔12​‖𝐀k​u−f‖2\mathcal{D}(u,k)\;\coloneqq\;\frac{1}{2}\|\mathbf{A}_{k}u-f\|^{2}. While the function Ψ\Psi is not convex in the joint variable (u,k)(u,k), it is bi-convex, i.e. it is convex in each of the variables uu and kk. The latter holds true as the regularization functions ℛu\mathcal{R}_{u} and ℛk\mathcal{R}_{k} are convex and the data fidelity 𝒟\mathcal{D} is bi-convex.

An abstract algorithm for finding solutions of (17) is the *proximal alternating minimization (PAM)* algorithm [42]. Given an image-kernel pair (u,k)(u,k) and step sizes τu,τk>0\tau_{u},\tau_{k}>0, it computes the next iterate (u+,k+)(u^{+},k^{+}) by the update scheme

u+=arg⁡min𝑥​{12​‖x−u‖2+τu​Ψ​(x,k)},\displaystyle u^{+}=\operatorname{arg}\underset{x}{\min}\;\left\{\frac{1}{2}\|x-u\|^{2}+\tau_{u}\Psi(x,k)\right\}\,,

(18a)

k+=arg⁡min𝑥​{12​‖x−k‖2+τk​Ψ​(u+,x)}.\displaystyle k^{+}=\operatorname{arg}\underset{x}{\min}\;\left\{\frac{1}{2}\|x-k\|^{2}+\tau_{k}\Psi(u^{+},x)\right\}\,.

(18b)

Under reasonably mild conditions on the regularity of Ψ\Psi (see [42] for more details), PAM monotonously decreases the objective and its iterates converge to a critical point of Ψ\Psi. As the overall problem is non-convex, we cannot hope to find global solutions with this strategy.

### 3.1 Proximal alternating linearized minimization (PALM)

In addition, we may exploit the fact that the data term 𝒟\mathcal{D} is continuously differentiable with respect to uu and kk and that it is not very difficult to compute the *proximal operators* with respect to the regularization functionals ℛu\mathcal{R}_{u} and ℛk\mathcal{R}_{k}. The proximal operator of a functional ℛ\mathcal{R} is defined as the solution operator to a denoising problem with regularization term ℛ\mathcal{R}

proxℛ⁡(y)≔arg⁡min𝑥​{12​‖x−y‖2+ℛ⁡(x)}.\operatorname{prox}_{\mathcal{R}}(y)\;\coloneqq\;\operatorname{arg}\underset{x}{\min}\;\left\{\frac{1}{2}\|x-y\|^{2}+\mathcal{R}(x)\right\}\,.

(19)

These extra information can be utilized with the *proximal alternating linearized minimization (PALM)* [43]. Linearizing the differentiable part 𝒟\mathcal{D} of the objective function Ψ\Psi in the update of (18) yields the new update

u+=proxτu​ℛu⁡(u−τu​∇u𝒟​(u,k)),\displaystyle u^{+}=\operatorname{prox}_{\tau_{u}\mathcal{R}_{u}}\Bigl(u-\tau_{u}\nabla_{u}\mathcal{D}(u,k)\Bigr)\,,

(20a)

k+=proxτk​ℛk⁡(k−τk​∇k𝒟​(u+,k)).\displaystyle k^{+}=\operatorname{prox}_{\tau_{k}\mathcal{R}_{k}}\Bigl(k-\tau_{k}\nabla_{k}\mathcal{D}(u^{+},k)\Bigr)\,.

(20b)

It can be interpreted both as a linearized version of PAM and as an alternating form of forward-backward splitting [44, 45]. Note that PALM also monotonously decreases the objective and has the same convergence guarantees as long as the step size parameters are well-chosen which we will discuss later in this section.

### 3.2 Inertial PALM (iPALM)

Local methods for non-convex optimization may lead to critical points that are not global minimizers. The authors of [46] proposed an inertial variant of forward-backward splitting which next to the gradient direction gives inertia to steer the iterates in the previously chosen direction. They empirically showed on a simple two dimensional problem that inertia may help to avoid spurious critical points. An inertial version of PALM [41], called *iPALM*, follows the updates

uα\displaystyle u_{\alpha}
=u+α⁡(u−u−),\displaystyle=u+\alpha(u-u^{-})\,,

(21a)

u+\displaystyle u^{+}
=proxτu​ℛu⁡(uα−τu​∇u𝒟​(uα,k)),\displaystyle=\operatorname{prox}_{\tau_{u}\mathcal{R}_{u}}\Bigl(u_{\alpha}-\tau_{u}\nabla_{u}\mathcal{D}(u_{\alpha},k)\Bigr)\,,

(21b)

kα\displaystyle k_{\alpha}
=k+α⁡(k−k−),\displaystyle=k+\alpha(k-k^{-})\,,

(21c)

k+\displaystyle k^{+}
=proxτk​ℛk⁡(kα−τk​∇k𝒟​(u+,kα)),\displaystyle=\operatorname{prox}_{\tau_{k}\mathcal{R}_{k}}\Bigl(k_{\alpha}-\tau_{k}\nabla_{k}\mathcal{D}(u^{+},k_{\alpha})\Bigr)\,,

(21d)

with inertia parameter α∈[0,1)\alpha\in[0,1). Next to the current iterate (u,k)(u,k) it also depends on the previous iterate (u−,k−)(u^{-},k^{-}). We remark that PALM can be recovered by vanishing inertial parameter α=0\alpha=0. Also note that one may define iPALM not with one inertial parameter α\alpha but with four different ones [41]. While iPALM is *not* guaranteed to decrease the objective Ψ\Psi, it monotonously decreases a surrogate functional that relates to Ψ\Psi [41] and its iterates are guaranteed to converge to a critical point of Ψ\Psi for well-chosen step size and inertia parameter.

*Algorithm 1 PALM / iPALM with backtracking. PALM is recovered by setting the inertial parameter α=0\alpha=0. The parameters L¯,L¯,η,θ\underline{L},\overline{L},\eta,\theta can all be chosen different for the image uu and kernel kk but for simplicity we chose not to.*

              1:

            initial iterates u,ku,k, inertial parameter α∈[0,1)\alpha\in[0,1), initial step size parameters Lu,LkL_{u},L_{k}, constant for step size θ>1\theta>1, bounds for step size parameters 0<L¯≤L¯<∞0<\underline{L}\leq\overline{L}<\infty, backtracking constant η>1\eta>1

              2:

            function iPALM

              3:

               u−←u,k−←ku^{-}\leftarrow u,\;k^{-}\leftarrow k

              4:

               for t=0,1,…t=0,1,\ldots do

              5:

                % update image uu

              6:

                uα←u+α⁡(u−u−)u_{\alpha}\leftarrow u+\alpha(u-u^{-})

              7:

                (u+,Lu)←(u^{+},L_{u})\leftarrow backtracking(uα,∇u𝒟​(uα,k),Luu_{\alpha},\nabla_{u}\mathcal{D}(u_{\alpha},k),L_{u})

              8:

                u−←u,u←u+u^{-}\leftarrow u,\;u\leftarrow u^{+}

              9:

                % update kernel kk

              10:

                kα←k+α⁡(k−k−)k_{\alpha}\leftarrow k+\alpha(k-k^{-})

              11:

                (k+,Lk)←(k^{+},L_{k})\leftarrow backtracking(kα,∇k𝒟​(u+,kα),Lkk_{\alpha},\nabla_{k}\mathcal{D}(u^{+},k_{\alpha}),L_{k})

              12:

                k−←k,k←k+k^{-}\leftarrow k,\;k\leftarrow k^{+}
   

              13:

               return (u+,k+)(u^{+},k^{+})

              14:

            function backtracking(xα,g,Lx_{\alpha},g,L)

              15:

               while True do

              16:

                % select step size

              17:

                τ←1−α1+2​α​2θ​L\tau\leftarrow\frac{1-\alpha}{1+2\alpha}\frac{2}{\theta L}

              18:

                % forward-backward step where the proximal operator

              19:

                % is approximated by a fixed number of warm-started iterations

              20:

                x+←proxτ​ℛ⁡(xα−τ​g)x^{+}\leftarrow\operatorname{prox}_{\tau\mathcal{R}}(x_{\alpha}-\tau g)

              21:

                if Descent inequality (24) does not hold then

              22:

                  % increase Lipschitz constant and try again

              23:

                  L←min⁡(η​L,L¯)L\leftarrow\min(\eta L,\overline{L})

              24:

                  continue
    

              25:

                if Proximal descent inequality (25) does hold then

              26:

                  % continue to next iteration with larger step size

              27:

                  L←max⁡(L/η,L¯)L\leftarrow\max(L/\eta,\underline{L})

              28:

                  break
       

              29:

               return (x+,L)(x^{+},L)

### 3.3 Step sizes and backtracking

The convergence of PALM and iPALM depend on the step size parameters τu,τk\tau_{u},\tau_{k}. Following [41], we choose the step size parameters as

τx≔1−α1+2​α​2θ​Lx,x∈{u,k},\tau_{x}\;\coloneqq\;\frac{1-\alpha}{1+2\alpha}\frac{2}{\theta L_{x}}\,,\qquad x\in\{u,k\}\,,

(22)

where θ>1\theta>1 is a global constant and LxL_{x} is the local Lipschitz constant of the derivative of the data fit 𝒟\mathcal{D} with respect to xx. Using the relation (4), its derivatives with respect to the image and kernel are

∇u𝒟​(u,k)\displaystyle\nabla_{u}\mathcal{D}(u,k)
=𝐀k∗​(𝐀k​u−f),\displaystyle=\mathbf{A}_{k}^{\ast}(\mathbf{A}_{k}u-f)\,,

(23a)

∇k𝒟​(u,k)\displaystyle\nabla_{k}\mathcal{D}(u,k)
=𝐉∗​𝐀u∗​(𝐀u​𝐉​k−f)\displaystyle=\mathbf{J}^{\ast}\mathbf{A}_{u}^{\ast}(\mathbf{A}_{u}\mathbf{J}k-f)\,

(23b)

which shows that LuL_{u} will depend on kk and LkL_{k} on uu. While it is possible to conservatively estimate these constants, standard estimates turn out to be highly pessimistic and one can do significantly better by determining LuL_{u} and LkL_{k} with a backtracking scheme [46, 47]. The actual local Lipschitz constants LuL_{u} and LkL_{k} satisfy the descent inequalities (also called descent Lemma) [47, 46, 41, 43]

𝒟⁡(u+,k)≤𝒟⁡(uα,k)+⟨∇u𝒟​(uα,k),u+−uα⟩+Lu2​‖u+−uα‖2,\displaystyle\mathcal{D}(u^{+},k)\leq\mathcal{D}(u_{\alpha},k)+\left\langle\nabla_{u}\mathcal{D}(u_{\alpha},k),u^{+}-u_{\alpha}\right\rangle+\frac{L_{u}}{2}\|u^{+}-u_{\alpha}\|^{2}\,,

(24a)

𝒟⁡(u+,k+)≤𝒟⁡(u+,kα)+⟨∇k𝒟​(u+,kα),k+−kα⟩+Lk2​‖k+−kα‖2.\displaystyle\mathcal{D}(u^{+},k^{+})\leq\mathcal{D}(u^{+},k_{\alpha})+\left\langle\nabla_{k}\mathcal{D}(u^{+},k_{\alpha}),k^{+}-k_{\alpha}\right\rangle+\frac{L_{k}}{2}\|k^{+}-k_{\alpha}\|^{2}\,.

(24b)

If during the iterations these inequalities are not satisfied, we increase the Lipschitz estimates by a constant factor η>1\eta>1 and repeat the iteration.

In addition to the descent inequality, an inexact evaluation of the proximal operators (e.g. due to finite iterations) may also hinder convergence. Thus, we will evaluate the proximal operators to a precision such that the proximal descent inequalities

ℛu​(u+)\displaystyle\mathcal{R}_{u}(u^{+})
≤ℛu​(u)+⟨∇u𝒟​(uα,k),u−u+⟩+12​τu​(‖u−uα‖2−‖u+−uα‖2),\displaystyle\leq\mathcal{R}_{u}(u)+\left\langle\nabla_{u}\mathcal{D}(u_{\alpha},k),u-u^{+}\right\rangle+\frac{1}{2\tau_{u}}\left(\|u-u_{\alpha}\|^{2}-\|u^{+}-u_{\alpha}\|^{2}\right)\,,

(25a)

ℛk​(k+)\displaystyle\mathcal{R}_{k}(k^{+})
≤ℛk​(k)+⟨∇k𝒟​(u+,kα),k−k+⟩+12​τk​(‖k−kα‖2−‖k+−kα‖2)\displaystyle\leq\mathcal{R}_{k}(k)+\left\langle\nabla_{k}\mathcal{D}(u^{+},k_{\alpha}),k-k^{+}\right\rangle+\frac{1}{2\tau_{k}}\left(\|k-k_{\alpha}\|^{2}-\|k^{+}-k_{\alpha}\|^{2}\right)

(25b)

hold. Equations (24), (25) and the parameters Lu,LkL_{u},L_{k} guarantee the monotonic descent of PALM and iPALM with respect to the objective or its surrogate [41, 43].

The pseudo code for PALM and iPALM with this backtracking can be found in algorithm 1.

## 4 Numerical Setting

##### Data

In this work we use data from two separate data sets. The first data set [48] is from environmental remote sensing to study vegetation in the Mediterranean woodlands in Alta Tajo natural park, northeast of Madrid, Spain [49]. It was acquired in October to November 2014 using an AISA Eagle hyperspectral sensor and a Leica RCD105 39 megapixel digital camera mounted on a Dornier 228 aircraft. The hyperspectral imagery consisting of 126 spectral bands almost equidistantly covering the wavelengths 400 nm to 970 nm with a spatial resolution of 1 m x 1 m. The aerial photograph has a resolution of 0.25 m x 0.25 m but only captures channels corresponding to the visible red, green and blue. We refer to images from this data set as environment1, environment2.

The second data set [50, 51] shows an urban area of Vancouver, Canada. It was acquired in March to May 2015 by the DEIMOS-2 satellite operating on a mean altitude of 620 km. The satellite carried a very highly resolving push-broom camera with one panchromatic channel of 1 m x 1 m resolution and the four multispectral channels red, green, blue and near-infrared with a resolution of 4 m x 4 m. Images from this data set are marked urban1, urban2.

Next to the already described real data sets we simulated an environmental data set where we know the ground truth image and kernel. We test two different kernels, one which is disk-shaped and one which is an off-centered Gaussian and refer to the corresponding data sets as groundtruth_disk and groundtruth_Gaussian. The red channel of an RGB image is convolved with these two kernels, subsampled by a factor of s=4s=4 and Gaussian noise of variance 0.001 is added. The data is shown at the top left of figures 13 and 13. For the data set groundtruth_Gaussian we use the RGB image itself as side information and for groundtruth_disk we use a shifted version of it. This situation occurs frequently in applications since usually data and side information are acquired by different imaging devices which can lead to a relative translation between the observed images. Note that we used replicated instead of periodic boundary conditions for the convolution in order to avoid *inverse crime*. As the ground truth is known in these cases, we evaluate the result in terms of the *structural similarity index (SSIM)* [52] and the *Haar wavelet-based perceptual similarity index (HPSI)* [53]. As compared to measures like the peak-signal-to-noise ratio (PSNR) or the mean squared error (MSE), the SSIM and the HPSI aim at reproducing how image similarity is perceived by human viewers. The SSIM is one of the most widely used image similarity measures and obtained by comparing local statistical parameters such as local means and local variances. The HPSI is based on a simple discrete Haar wavelet transform and incorporates basic assumptions about the human visual system. The HPSI was recently shown to yield state-of-the-art correlations with human opinion scores on large benchmarking databases of differently distorted images.

In all numerical experiments the data ff are scaled to [0,1][0,1].

##### Visualization

Throughout the results, the data and reconstructed images are shown with MATLAB’s ‘parula’ colormap where yellow represents values ≥1\geq 1 and dark blue corresponds to values ≤0\leq 0, respectively. The kernels are visualized with gray scales where black corresponds to the smallest value and white to the largest. In addition, we draw a red hair cross on the center, i.e. the pixel with indices l+1l+1, where ll denotes the radius of the kernel. The kernel’s centroid ∑ii​ki\sum_{i}ik_{i} is visualized by a green hair cross in order to visualize the off-set between data and side information.

##### Algorithms

The numerical results will be computed with the algorithms PALM, iPALM and PAM. For PAM, the inner problems require solving a deconvolution problem with convex regularization which is implemented with ADMM [54, 55, 56] following a similar strategy as in [18]. All three—PALM, iPALM and PAM—require the computation of proximal operators for the total variation and directional total variation under a convex constraint. These are implemented with a finite number of warm started Fast Gradient Projection/FISTA [57, 47] iterations. Within these, projections onto the constraint sets are computed by proxı[0,∞)m(u)=max(0,u)\operatorname{prox}_{\imath_{[0,\infty)^{m}}}(u)=\max(0,u) (to be understood componentwise) and the fast algorithm of [58] for the projection onto the unit simplex. A MATLAB implementation of the algorithms and the data itself can be found at [59].

##### Parameters

The numerical experiments require model parameters and algorithm parameters. For the mathematical model, the parameters of the directional total variation are chosen as γ=0.9995\gamma=0.9995 and ε=0.003\varepsilon=0.003 for all experiments, cf. (12) and figure 18 in the appendix. The low resolution images are of size 100×100100\times 100 with an subsampling rate of s=4s=4. Since we fix the kernel size to be 41×4141\times 41, we will consequently obtain reconstructions of size 440×440440\times 440 where we only visualize the meaningful part of size 400×400400\times 400.

Except for the algorithm comparison, we use PALM with 2000 iterations and a step size constant θ=1.1\theta=1.1, cf. (22). The backtracking parameters in algorithm 1 are chosen as η=2\eta=2, L¯=1\underline{L}=1 and L¯=1030\overline{L}=10^{30}.

##### Initialization

Both algorithm 1 and our implementation of PAM require initial guesses for image and kernel. While the kernel is initialized with a compactly supported Gaussian kernel, we choose the initial image to be 𝐇​f\mathbf{H}f, where 𝐇:ℝn→U\mathbf{H}:\mathbb{R}^{n}\to U is a right inverse of the operator 𝐒∘𝐁\mathbf{S}\circ\mathbf{B}, i.e 𝐇\mathbf{H} is an upsampling operator (cf. the top-left images in figures 6 and 8).

## 5 Numerical Results

### 5.1 Comparison of Algorithms

![Refer to caption](1710.05705v4/fig_5.png)

*Figure 5: Iterates of PALM for data set groundtruth_disk with λu=0.1\lambda_{u}=0.1, λk=10\lambda_{k}=10 with objective function values Ψ\Psi and similarity to ground truth.*

![Refer to caption](1710.05705v4/fig_6.png)

*Figure 6: Reconstructions by three different algorithms—PAM and iPALM with inertial parameters α=0.2\alpha=0.2 and α=0.5\alpha=0.5—after thousands of iterations for the same setting as figure 6. The visual impression of the images and kernels and the objective values indicate that all algorithms converge to a similar critical point.*

![Refer to caption](1710.05705v4/fig_7.png)

*Figure 7: Iterates of PALM for the data set environment1 with λu=1\lambda_{u}=1 and λk=10\lambda_{k}=10 with the objective function values given in brackets.*

![Refer to caption](1710.05705v4/fig_8.png)

*Figure 8: Reconstructions by PAM and iPALM with inertial parameters α=0.2\alpha=0.2 and α=0.5\alpha=0.5 for the setting of figure 8. Visually, the images and kernels and quantitatively, the objective function values indicate that all algorithms converge to the same critical point.*

![Refer to caption](1710.05705v4/trees1_shift_5px_disk__lambda_u__0-1__lambda_k_10__gamma_0-9995__eta_0-003__ObjectiveDecay.png)

![Refer to caption](1710.05705v4/trees1_shift_5px_disk__lambda_u__0-1__lambda_k_10__gamma_0-9995__eta_0-003__ObjectiveTracking.png)

![Refer to caption](1710.05705v4/trees1_shift_5px_disk__lambda_u__0-1__lambda_k_10__gamma_0-9995__eta_0-003__Legend.png)

![Refer to caption](1710.05705v4/trees1_shift_5px_disk__lambda_u__0-1__lambda_k_10__gamma_0-9995__eta_0-003__DataFidelityTracking.png)

![Refer to caption](1710.05705v4/trees1_shift_5px_disk__lambda_u__0-1__lambda_k_10__gamma_0-9995__eta_0-003__RuTracking.png)

![Refer to caption](1710.05705v4/trees1_shift_5px_disk__lambda_u__0-1__lambda_k_10__gamma_0-9995__eta_0-003__RkTracking.png)

*Figure 9: Algorithm statistics over the course of the iterations for the same setting as figure 6. The relative objective function values (top left) show that all algorithms have about the same speed on this example. The objective function values (top center) and its summands (bottom) indicate that also all algorithms converge to the same critical point. Overall, the plots show that more inertia leads to slower convergence.*

![Refer to caption](1710.05705v4/trees2_ch108_NW__lambda_u__1__lambda_k_10__gamma_0-9995__eta_0-003__ObjectiveDecay.png)

![Refer to caption](1710.05705v4/trees2_ch108_NW__lambda_u__1__lambda_k_10__gamma_0-9995__eta_0-003__ObjectiveTracking.png)

![Refer to caption](1710.05705v4/trees1_shift_5px_disk__lambda_u__0-1__lambda_k_10__gamma_0-9995__eta_0-003__Legend.png)

![Refer to caption](1710.05705v4/trees2_ch108_NW__lambda_u__1__lambda_k_10__gamma_0-9995__eta_0-003__DataFidelityTracking.png)

![Refer to caption](1710.05705v4/trees2_ch108_NW__lambda_u__1__lambda_k_10__gamma_0-9995__eta_0-003__RuTracking.png)

![Refer to caption](1710.05705v4/trees2_ch108_NW__lambda_u__1__lambda_k_10__gamma_0-9995__eta_0-003__RkTracking.png)

*Figure 10: Algorithm statistics for PALM and iPALM with a variety of inertial parameters for the same setting as figure 8. The objective function values (top center) and its summands (bottom) indicate that all algorithms converge to the same critical point. The relative objective function values (top left) show that for this example a higher inertial parameter leads to faster convergence.*

Before we focus on the actual image fusion reconstructions, we compare the algorithms PALM, iPALM and PAM on two different data sets. The results are shown in figures 6, 6, 10 for the data set groundtruth_disk and in figures 8, 8, 10 for environment1. Please refer to the captions of each figure for some detailed observations. We would like to highlight two important observations. First, both examples show that all three algorithms converge to practically the same critical point and that a visually pleasing reconstruction is already obtained after about 500 iterations. Thus, even though problem (1) is non-convex, the algorithm itself does not seem to have much influence on the final reconstructions. Second, the examples show that inertia may slow the convergence down as for groundtruth_disk or may make it faster as for environment1. Other results—which are not shown here for brevity—indicate that this is correlated with the regularization parameter λu\lambda_{u}. For large values, e.g. λu≥1\lambda_{u}\geq 1, we found that inertia consistently sped up convergence while for smaller values, e.g. λu≤0.1\lambda_{u}\leq 0.1 we found that it was always slowing down the convergence. Thus, for simplicity and due to the lack of supporting examples that would justify other decisions, all further reconstructions are performed by PALM.

### 5.2 Simulated Data

*Figure 11: Similarity measures SSIM and HPSI for different values of the regularization parameters λu\lambda_{u} and λk\lambda_{k}. Top: data set groundtruth_disk with shifted side information. Bottom: data set groundtruth_Gaussian. Yellow corresponds to high and blue to low similarity. All four plots indicate that the reconstructions are not very sensitive with respect to the kernel regularization.*

![Refer to caption](1710.05705v4/fig_12.png)

*Figure 12: Reconstructions for groundtruth_disk with shifted side information. Top: Low resolution data and side information. Bottom: Reconstruction with varying image kernel regularization and image regularization chosen as λu=0.1\lambda_{u}=0.1. SSIM and HPSI are given in percentages and higher values indicating better performance.*

![Refer to caption](1710.05705v4/fig_13.png)

*Figure 13: Reconstructions for groundtruth_Gaussian. Top: Low resolution data and side information. Bottom: Reconstruction for varying image regularization and kernel regularization being chosen as λk=10\lambda_{k}=10. SSIM and HPSI are given in percentages and higher values indicating better performance.*

This section is dedicated to simulated data where the ground truth image is known. Figure 11 shows the similarity indices SSIM and HPSI of the reconstructed images when varying the image and kernel regularization parameters λu\lambda_{u} and λk\lambda_{k}. The results show that the proposed model is stable with respect to the choice of parameters as long as the image regularization λu\lambda_{u} is not too small. It is remarkable that the success of the model depends only mildly on the choice of λk\lambda_{k} and a vanishing kernel regularization λk=0\lambda_{k}=0 has only modest impact. However, it can also be seen that best reconstructions are obtained for non-zero λk\lambda_{k} which indicates that a mild kernel regularization should be performed.

In figure 13 we show reconstruction results for the groundtruth_disk test case where we fix the image regularizing parameter λu=0.1\lambda_{u}=0.1 and vary the kernel regularization λk\lambda_{k}. Note that we complicated the situation by using a side information which is shifted by several pixels and, thus, *not aligned* to the data. The proposed approach is well-fit to deal with this problem since blind deconvolution can implicitly perform rigid registration of data and side information by producing a blurring kernel that compensates for the translation by having an off-centered centroid. The results show that our method is able to compensate for unaligned side information images and can achieve very good results, nevertheless. The influence of the amount of kernel regularization is noticeably small although over-regularization clearly leads to different contrasts in the reconstructed image. Note that the barycentric translations of the kernels in figure 13 correspond exactly—up to pixel accuracy—to the number of pixels that the shifted side information deviates from the un-shifted one.

Figure 13 shows in a similar way the reconstruction results for the test case groundtruth_Gaussian, using the un-shifted side information. In contrast to the previous example, here we fix the parameter λk=10\lambda_{k}=10 and vary λu\lambda_{u} instead. Here, the effects of different regularization strengths are considerably larger; particularly, in the case λu=0.01\lambda_{u}=0.01 the shift between data and side information—which was induced by convolving with the off-centered kernel—is not fully detected. Consequently, the resulting similarity values, being not translationally invariant, are relatively poor. In the opposite scenario of over-regularizing the image, small structures in the image are not captured well.

### 5.3 Hyperspectral Data

![Refer to caption](1710.05705v4/fig_14.png)

*Figure 14: Varying both regularization parameters for environment1 which corresponds to a near-infrared channel with wavelength of around 900 nm. Too small image regularization λu\lambda_{u} leads to point artefacts and too large λu\lambda_{u} to a loss of contrast and wrong estimation of the kernel. While no kernel regularization does not seem to have negative effects on the reconstruction, too large kernel regularization changes the image contrast.*

![Refer to caption](1710.05705v4/fig_15.png)

*Figure 15: Results for urban1 (near-infrared) for varying image and kernel regularization. Over- and underregularization have similar effects like in figure 14. However, due to a lower noise level of the data, the parameters can be chosen smaller.*

![Refer to caption](1710.05705v4/fig_16.png)

*Figure 16: Reconstructions of several channels corresponding to different wavelengths given in nanometer (nm) of environment2. The regularization parameters for this experiment were chosen constant across the channels as λu=1\lambda_{u}=1 and λk=1\lambda_{k}=1. For visualization, reconstructed images are scaled back to their physical range. Note that despite the reconstruction being performed channel-by-channel, all kernels have about the same shape and the same shift.*

![Refer to caption](1710.05705v4/fig_17.png)

*Figure 17: Reconstruction of the multispectral data set urban2. The regularization parameters were chosen the same for all channels as λu=0.01\lambda_{u}=0.01 and λk=1\lambda_{k}=1. Kernels are being estimated consistently over the range of the channels up to an accuracy of one pixel. The resulting RGB and false color images have well-defined boundaries and show no color smearing. (NIR = near-infrared)*

In this section we apply our method to real data and—similarly to the previous section— investigate the effects of varying image and kernel regularization. To this end, we apply our method to one channel of the hyperspectral environmental and of the multispectral urban data set. Figures 14 and 15 show data, side information, and the corresponding reconstruction results for different values of the regularization parameters. Since no ground truth solutions are known in this case, we rely on our visual impression to choose a set of parameters. Figure 14, which corresponds to near-infrared light, suggests that a good compromise among data fidelity and regularization is achieved for λu=λk=1\lambda_{u}=\lambda_{k}=1. However, even if no TV\mathrm{TV} regularity of the kernel is required, that is in the case λk=0\lambda_{k}=0, the resulting images do hardly differ. Bearing in mind the results of figure 13, we will regularize the kernel slightly in the sequel. The situation is similar in figure 15, which also corresponds to near-infrared light. However, a smaller image regularization parameter, e.g. λu=0.01\lambda_{u}=0.01, seems necessary in order to avoid over-smoothing.

Finally, we apply the proposed strategy to several hyperspectral channels of the data set environment2, using the same parameters for the reconstruction of each channel. Here, we scaled the reconstructed images back to their physical range to allow inter-channel comparisons. Figure 17 shows the reconstruction of six hyperspectral channels ranging from blue to near-infrared light. Note that the resulting kernels only change slightly with respect to the channel. We would like to point out that we were able to use the same set of parameters, namely λu=λk=1\lambda_{u}=\lambda_{k}=1, for all channels. In figure 17 we show the reconstruction of all channels of the multispectral urban data set urban2 using λu=0.01\lambda_{u}=0.01 and λk=1\lambda_{k}=1. Additionally, we visualized the results by re-combining the four channels into an RGB and a false color image which has the channels near-infrared, red and green. We would like to highlight that the kernels are consistently estimated up to a pixel accuracy and that no color smearing is visible in the RGB and false color images despite the fact that all channels have been processed independently.

## 6 Conclusions and Outlook

We have presented a novel model for simultaneous image fusion and blind deblurring of hyperspectral images based on the directional total variation. We showed that this approach yields sharp images for both environmental and urban data sets that are either acquired by a plane or by a satellite. In addition, the numerical comparison of different algorithms showed that—despite the non-convexity of the problem—the computed solutions are stable with respect to the algorithms. The implicitly included registration by estimating the kernel yields further robustness to imperfections in the image acquisitions. Thus, the proposed approach is appealing for automated processing of large data sets.

Future work will be devoted to multiple directions. First, the proposed approach is channel-by-channel which makes it computationally appealing. However, a coupling of the spectral channels via their kernel and/or via spectral fingerprint preservation may yield further improvements. Second, higher order regularization such as the total generalized variation [40] has been proven useful for natural images. This naturally leads to the directional total generalized variation which combines higher order smoothness with structural a-priori knowledge as presented in this work, thereby generalizing the definition of [36]. Similarly, other kernel regularization functionals such as the total generalized variation or the H1H^{1}-semi-norm will be considered in future investigations to better model the smoothness of the kernel. Finally, it is currently not clear how the computational cost of the proposed method depends on the size of the considered data. In particular, it is difficult to predict to which degree larger images also require a higher number of iterations during optimization for the method to yield satisfactory results. However, due to the fact that our approach mostly depends on operations that are of a local nature, it should be highly suitable for parallelization. The development and analysis of implementations that utilize the benefits of distributed computing architectures is also a topic for future research.

We would like to express our gratitude to Juheon Lee and Qi Wei for their help with the environmental and urban data sets, respectively. Furthermore, the authors would like to thank Deimos Imaging for acquiring and providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee.
M.J.E. and C.-B.S. acknowledge support from Leverhulme Trust project ‘Breaking the non-convexity barrier’, EPSRC grant ‘EP/M00483X/1’, EPSRC centre ‘EP/N014588/1’, the Cantab Capital Institute for the Mathematics of Information, and from CHiPS (Horizon 2020 RISE project grant). Moreover, C.-B.S. is thankful for support from the Alan Turing Institute.
D.A.C. acknowledges the support of NERC (grant number NE/K016377/1). We are grateful to NERC’s Airborne Research Facility and Data Analysis Node https://arsf-dan.nerc.ac.uk/ for conducting the airborne survey and pre-processing the environmental data collected from Alto Tajo, Spain (survey CAM11/03).

## Appendix

We test the influence of parameter γ\gamma in (12) on a simulated and a real data set. Figure 18 shows the corresponding reconstruction results for γ∈{0.995,0.9995,1}\gamma\in\{0.995,0.9995,1\} where the middle value is used in all other numerical experiments in this work. Note that the results for γ=1\gamma=1 suffer from stronger point artifacts whereas those for γ=0.995\gamma=0.995 do not capture local structures well enough.

![Refer to caption](1710.05705v4/fig_A1a.png)

![Refer to caption](1710.05705v4/fig_A1b.png)

*Figure 18: Reconstructions for varying vector field parameters γ\gamma. Top: Results for groundtruth_disk with λu=0.1\lambda_{u}=0.1 and λk=10\lambda_{k}=10. Bottom: Results for environment1 with λu=1\lambda_{u}=1, λk=1\lambda_{k}=1.*

## References

- [1]

Ustin S L, Roberts D A, Gamon J A, Asner G P and Green R O 2004 AIBS
Bulletin 54 523–534

- [2]

Homolova L, Malenovskỳ Z, Clevers J G, García-Santos G and Schaepman
M E 2013 Ecological Complexity 15 1–16

- [3]

Amro I, Mateos J, Vega M, Molina R and Katsaggelos A K 2011 EURASIP
Journal on Advances in Signal Processing 2011 1–22

- [4]

Loncan L, De Almeida L B, Bioucas-Dias J M, Briottet X, Chanussot J, Dobigeon
N, Fabre S, Liao W, Licciardi G A, Simoes M, Tourneret J Y, Veganzones M A,
Vivone G, Wei Q and Yokoya N 2015 IEEE Geoscience and Remote Sensing
Magazine 3 27–46

- [5]

Mowle E W and Dennehy C J 1991 The LANDSAT-6 Satellite: An Overview Telesystems Conference, 1991. Proceedings. Vol.1., NTC ’91., National

- [6]

Ballester C, Caselles V, Igual L, Verdera J and Rougé B 2006 Int. J.
Comput. Vision 69 43–58

- [7]

Möller M, Wittman T and Bertozzi A L 2009 A Variational Approach to
Hyperspectral Image Fusion SPIE: Algorithms and Technologies for
Multispectral, Hyperspectral, and Ultraspectral Imagery XV vol 7334 ed
Shen S S and Lewis P E

- [8]

Padwick C, Deskevich M, Pacifici F and Smallwood S 2010 WorldView-2
Pan-sharpening ASPRS vol 48 pp 26–30

- [9]

Möller M, Wittman T, Bertozzi A L and Burger M 2012 SIAM J. Imaging
Sci. 5 150–178

- [10]

Simoes M, Bioucas-Dias J, Almeida L B and Chanussot J 2015 IEEE
Transactions on Geoscience and Remote Sensing 53 3373–3388

- [11]

Zhang Q, Cao Z, Hu Z, Jia Y and Wu X 2015 IEEE Geosci. Remote Sens.
Lett. 12 467–471

- [12]

Hou L and Zhang X 2016 Journal of Mathematical Imaging and Vision 55 36–49

- [13]

Duran J, Buades A, Coll B, Sbert C and Blanchet G 2017 ISPRS Journal of
Photogrammetry and Remote Sensing 125 78–105

- [14]

Kaipio J P, Kolehmainen V, Vauhkonen M and Somersalo E 1999 Inverse
Probl. 15 713–729

- [15]

Bowsher J E, Yuan H, Hedlund L W, Turkington T G, Akabani G, Badea A, Kurylo
W C, Wheeler C T, Cofer G P, Dewhirst M W and Johnson G A 2004 Utilizing MRI
Information to Estimate F18-FDG Distributions in Rat Flank Tumors Proc.
IEEE NSS-MIC pp 2488–2492

- [16]

Vunckx K, Atre A, Baete K, Reilhac A, Deroose C M, Van Laere K and Nuyts J
2012 IEEE Trans. Med. Imag. 31 599–612

- [17]

Ehrhardt M J, Markiewicz P J, Liljeroth M, Barnes A, Kolehmainen V, Duncan J,
Pizarro L, Atkinson D, Hutton B F, Ourselin S, Thielemans K and Arridge S R
2016 IEEE Trans. Med. Imag. 35 2189–2199

- [18]

Ehrhardt M J and Betcke M M 2016 SIAM J. Imaging Sci. 9
1084–1106

- [19]

Kolehmainen V, Ehrhardt M J and Arridge S 2017 (submitted to) Inverse
Problems 1–20

- [20]

Mehranian A, Belzunce M A, Niccolini F, Politis M, Prieto C, Turkheimer F,
Hammers A and Reader A J 2017 Phys. Med. Biol.

- [21]

Lee J, Cai X, Schönlieb C B and Coomes D A 2015 IEEE Transactions on
Geoscience and Remote Sensing 53 6073–6084

- [22]

Amizic B, Spinoulas L, Molina R and Katsaggelos A K 2013 IEEE Trans. Image
Process. 22 3994–4006

- [23]

Perrone D and Favaro P 2016 Int. J. Comput. Vision 117 159–172

- [24]

Kundur D and Hatzinakos D 1996 IEEE Signal Process. Mag. 13
43–64

- [25]

Vorontsov S V and Jefferies S M 2017 Inverse Probl. 33 055004

- [26]

Chan T F and Shen J 2005 Image processing and analysis: variational, PDE,
wavelet, and stochastic methods (SIAM)

- [27]

Almeida M S and Figueiredo M 2013 IEEE Trans. Image Process. 22
3074–3086

- [28]

Rudin L I, Osher S and Fatemi E 1992 Physica D 60 259–268

- [29]

Grasmair M and Lenzen F 2010 Appl. Math. Opt. 62 323–339

- [30]

Estellers V, Thiran J and Bresson X 2013 IEEE Trans. Image Process.
22 2611–2626

- [31]

Estellers V, Soatto S and Bresson X 2015 IEEE Trans. Image Process.
24 1777–1790

- [32]

Ehrhardt M J and Arridge S R 2014 IEEE Trans. Image Process. 23
9–18

- [33]

Ehrhardt M J, Thielemans K, Pizarro L, Atkinson D, Ourselin S, Hutton B F and
Arridge S R 2015 Inverse Probl. 31 015001

- [34]

Ehrhardt M J 2015 Joint Reconstruction for Multi-Modality Imaging with
Common Structure Ph.D. thesis University College London

- [35]

Bayram I and Kamasak M E 2012 IEEE Signal Process. Lett. 19
265–269

- [36]

Kongskov R D, Dong Y and Knudsen K 2017 Directional Total Generalized
Variation Regularization (Preprint arXiv:1701.02675)

- [37]

Arridge S R, Kolehmainen V and Schweiger M J 2008 Reconstruction and
Regularisation in Optical Tomography Mathematical Methods in Biomedical
Imaging and Intensity-Modulated Radiation Therapy (IMRT) ed Censor, Y and
Jiang, M and Louis A (Scuola Normale Superiore)

- [38]

Grasmair M 2009 Locally Adaptive Total Variation Regularization Proc.
SSVM pp 331–342

- [39]

Knoll F, Dong Y, Langskammer C, Hintermüller M and Stollberger R 2010
Total Variation Denoising with Spatially Dependent Regularization Proceedings of ISMRM vol 18 p 5088

- [40]

Bredies K, Kunisch K and Pock T 2010 SIAM J. Imaging Sci. 3
492–526

- [41]

Pock T and Sabach S 2016 SIAM J. Imaging Sci. 9 1756–1787

- [42]

Attouch H, Bolte J, Redont P and Soubeyran A 2010 Math. Oper. Res. 35 438–457

- [43]

Bolte J, Sabach S and Teboulle M 2014 Math. Program. 146 459–494

- [44]

Lions P L and Mercier B 1979 SIAM J. Numer. Anal. 16 964–979

- [45]

Combettes P L and Wajs V R 2005 Multiscale Modeling & Simulation 4 1168–1200

- [46]

Ochs P, Chen Y, Brox T and Pock T 2014 SIAM J. Imaging Sci. 7
1388–1419

- [47]

Beck A and Teboulle M 2009 IEEE Trans. Image Process. 18
2419–2434

- [48]

Arnold R, Johnson J, Joseph C and Llewellyn G 2014 NERC Airborne Research
Facility URL http://arsf.nerc.ac.uk

- [49]

Simonson W, Ruiz-Benito P, Valladares F and Coomes D 2016 Biogeosciences 13 961–973 ISSN 17264189

- [50]

2016 IEEE GRSS Data Fusion Contest
URL http://www.grss-ieee.org/community/technical-committees/data-fusion

- [51]

Mou L, Zhu X, Vakalopoulou M, Karantzalos K, Paragios N, Le Saux B, Moser G and
Tuia D 2017 IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing 10 3435–3447

- [52]

Wang Z, Bovik A C, Sheikh H R and Simoncelli E P 2004 IEEE Trans. Image
Process. 13 600–612

- [53]

Reisenhofer R, Bosse S, Kutyniok G and Wiegand T 2018 Signal Processing:
Image Communication 61 33–43

- [54]

Gabay D and Mercier B 1976 Comput. Math. Appl. 2 17–40

- [55]

Boyd S P, Parikh N, Chu E, Peleato B and Eckstein J 2010 Foundations and
Trends in Machine Learning 3 1–122

- [56]

Afonso M V, Bioucas-Dias J M and Figueiredo M A T 2010 IEEE Trans. Image
Process. 19 2345–2356

- [57]

Beck A and Teboulle M 2009 SIAM J. Imaging Sci. 2 183–202

- [58]

Duchi J, Shalev-Shwartz S, Singer Y and Chandra T 2008 Efficient projections
onto the l1-ball for learning in high dimensions Proc. 25th ICML
(ACM) pp 272–279

- [59]

Bungert L, Coomes D A, Ehrhardt M J, Rasch J, Reisenhofer R and Schönlieb C B
2017 Supplementary Material for Blind Image Fusion for Hyperspectral Imaging
with the Directional Total Variation
URL https://doi.org/10.17863/CAM.17104

<!-- END-OF-PAPER:a4aac5004727 -->
