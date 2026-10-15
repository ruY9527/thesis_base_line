# Net 扩展文献补充：近两年证据与可迁移边界

检索日期：2026-10-07。近两年窗：2024-10-07 至 2026-10-07。基线对象：CAVE、31 通道 LR-HSI、3 通道 HR-MSI、×8、监督双分支 CNN。本次是原有 cross_direction_review_20261007 的有界补充，不是穷尽性系统综述。最近邻旧论文用于查新，不计入近两年推荐数量。

结论先行：这轮扩展没有得到可以直接宣布“原创”的模块。保留主线 I13：用 GT 定义 MSI 细节相对 HSI-only 重建的实际光谱误差收益，再监督细节注入门控。其研究价值应来自收益目标的定义、可识别性和跨退化有效性；“置信融合”“区域不确定性加权”“观测一致性”“光谱校正”这些宽泛贡献均已有先例。尤其 CoPGNet、VPG-Net 全文尚未取得，使查新不能闭合。本补充不改变这个证据边界，也未运行训练实验。

## 本次阅读回执

已按 research-papers 的逐段 read_plan 完成下列六篇正文至参考文献的读取，针对方法图/公式另作视觉核查。前三篇复用旧缓存，但下列阅读是本次新增核查；后三篇是本次新获取全文。全文可用范围指获取的主论文，未取得的 CVPR supplementary 不作已读声明。

|论文、身份日期|本次文本覆盖|匹配的结束 sentinel|阅读边界|
|---|---|---|---|
|[UAFL: Enhancing Unregistered Hyperspectral Image Super-Resolution via Unmixing-based Abundance Fusion Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)，CVPR 2026 主会，2026-06，41573–41583|1044 行全部；一次输出中段截断后补读 410–764|END-OF-PAPER:2593c6d9b523|主文及参考文献；检查方法图|
|[UMNet: Uncertainty-guided Memory Network for Hyperspectral Pansharpening](https://ojs.aaai.org/index.php/AAAI/article/view/37988)，AAAI 2026 主会，2026-03-14，10199–10206，DOI 10.1609/aaai.v40i12.37988|0–439、440–879，共 880 行|END-OF-PAPER:3e70f9ba059e|主文及参考文献；检查记忆及不确定性流程图；PDF首页题名与官方页面一致|
|[CARL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html)，ICLR 2026 主会|0–339、340–679、680–1019、1020–1360，共 1361 行|END-OF-PAPER:e217733d253b|23 页论文及 appendix；检查前端图|
|[Automatic Spectral Calibration of Hyperspectral Images: Method, Dataset and Benchmark](https://openaccess.thecvf.com/content/CVPR2025/html/Du_Automatic_Spectral_Calibration_of_Hyperspectral_Images_Method_Dataset_and_Benchmark_CVPR_2025_paper.html)，CVPR 2025 主会，2025-06，28081–28090|0–504、505–1006，共 1007 行|END-OF-PAPER:559d12f3b934|主文及参考文献；检查 SIT 图|
|[Degradation-Robust Fusion: An Efficient Degradation-Aware Diffusion Framework for Multimodal Image Fusion in Arbitrary Degradation Scenarios](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Degradation-Robust_Fusion_An_Efficient_Degradation-Aware_Diffusion_Framework_for_Multimodal_Image_CVPR_2026_paper.html)，CVPR 2026 主会，2026-06，33848–33858|0–354、355–709、710–1064，共 1065 行|END-OF-PAPER:b1233abd0c32|主文及参考文献；检查联合观测图及式 (15)|
|[Real Noise Decoupling for Hyperspectral Image Denoising](https://ojs.aaai.org/index.php/AAAI/article/view/38291)，AAAI 2026 主会，2026-03-14，12925–12933，DOI 10.1609/aaai.v40i15.38291|0–399、400–787，共 788 行|END-OF-PAPER:ad7a95a11468|主文及参考文献；检查 EMNet/IMNet 时序|

新增三篇全文、parsed text、元数据与中央方法页已由主任务保存到 fulltext/，附 SHA-256；原有研究包没有被覆写。机器可读身份、深度与读取计划见 [sources-addendum.json](./sources-addendum.json)。

## 三个跨方向可迁移机制

### 1. 真实噪声分解 → HSI 分支恢复与细节注入协同

RND 的 EMNet 先从已标定的合成噪声学习，随后冻结 EMNet 训练处理残余真实噪声的 IMNet，最后联合微调。推理顺序是 IMNet→EMNet。它加入中间噪声分布与光谱一致性约束；这并不证明两种噪声在物理上唯一可分解。论文解决 HSI 去噪，不是 HSI–MSI 融合。[RND 官方论文](https://ojs.aaai.org/index.php/AAAI/article/view/38291)

可改的不只是 Net：数据加载器加入有记录的噪声标定参数，LR-HSI 分支前放一个很小的预恢复模块，融合头联合优化。将 I13 收益目标扩展到“清洁/含噪 HSI 上，MSI 高频究竟改善哪类光谱误差”，再检查预恢复是否改变收益标签。需要同参数、同训练预算比较：仅噪声增强、独立去噪再融合、联合去噪融合、带收益门控。

最近邻是 RND 本身，以及已读 Hipandas 的去噪/引导超分联合建模。因此“先去噪再融合”“两阶段噪声学习”不足以构成新贡献。没有真实配对噪声数据时只能称合成噪声实验；若预恢复抹掉窄谱弱信号，或仅噪声增强就能取得同样收益，应停止该方向。

### 2. 全局照明建模 → 有目标定义的光谱幅度稳健性

SIT 使用 I(λ)=R(λ)L(λ) 的全局照明假设，将光谱注意力与全局照明注意力结合来估计反射率；其数据含实测照明和反射率对应。这里的 “spectral calibration” 指照明/反射率校准，不能改写成传感器 SRF 校准或波长校准。[SIT 官方论文](https://openaccess.thecvf.com/content/CVPR2025/html/Du_Automatic_Spectral_Calibration_of_Hyperspectral_Images_Method_Dataset_and_Benchmark_CVPR_2025_paper.html)

可迁移一个小型全局照明条件编码器到 Net 的融合头，同时把数据合成、归一化和目标定义改成一致的观测过程。先选清楚重建目标是辐射强度还是反射率，再决定是否允许全局光谱增益变化。当前 CAVE 归一化强度不能自动提供真实照明/反射率分解监督。

最近邻为 SIT、已有光照分解/增强工作，以及 DIFIV 的跨观测光谱可变性处理。该方向的可检验问题是：一致照明变化下，门控能否保住绝对光谱幅度，避免仅改善 SAM。若没有实测或可信生成的照明目标、全局增益增强足够、或者先归一化把待研究变化消掉，应停止。

### 3. 联合退化约束 → 小步观测残差修正

DRFusion 在少步扩散恢复中同时处理两源观测退化，并直接预测融合图像。其实验是 IR–VIS 和 PET–MRI；源图像和输出目标与 HSI–MSI 定量重建不同，原文最大强度/梯度式融合损失不能直接迁移为 HSI 监督损失。[DRFusion 官方论文](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Degradation-Robust_Fusion_An_Efficient_Degradation-Aware_Diffusion_Framework_for_Multimodal_Image_CVPR_2026_paper.html)

建议只迁移“联合观测纠偏”思想：在 Net 尾部用真实训练退化算子计算 LR-HSI 与 HR-MSI 两路残差，经有界步长修正公共高光谱输出。先做确定性一到两步模块；与普通一致性损失、单次梯度修正、等参数残差头比较，观察 SAM、观测残差和失配退化的共同变化。

最近邻 DRFusion、DDNM、PINO、GSNR 及展开式 HSI 融合已覆盖联合观测和一致性修正。只有在算子失配下有明确机制且收益超出简单纠偏时，才值得继续；“加入物理一致性”本身不是新点。若只在准确已知合成算子下有效、错误算子带来更大光谱偏差、计算预算大幅增加，应降级为工程模块。

**式 (15) 的独立核查。** DRFusion 将分块矩阵写出的逆称为 Moore–Penrose 伪逆。一般秩亏情形下这并不自动成立。原 PDF 与解析文本均有这一形式，故非解析错误。取下列简单分块退化例：

```text
A      = [[0, 0, 0], [0, 1, 0], [-0.5, -0.5, 1]]
Bpaper = [[0, 0, 0], [0, 1, 0], [0, 0.5, 1]]
```

本次用 NumPy 检查，ABA=A、BAB=B、AB 对称成立，但 ‖(BA)ᵀ−BA‖F=0.7071，第四个 Penrose 条件失败。该反例说明复用闭式公式前必须验证秩与算子条件；不等于证明作者代码或论文实验失效，也不等于当前基线算子必然触发此例。安全实现应分别验证四个条件，或使用稳定线性求解/迭代更新，不能因论文用了 † 就视为无条件正确。

## 三篇缓存核心论文的本次核查结论

- **UAFL：**对上采样 LR-HSI 做 SVD，取低维基并学习残差丰度后加回上采样 HSI；最终输出并非简单的 rank-3 硬约束。流、相似性、可变形融合和注意力动态门已经存在。适合基线加入受控失配/配准实验，但单纯“低秩+门控+对齐”已被覆盖。[官方论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)
- **UMNet：**双 U-Net、双向记忆融合及空间×通道不确定性分解；先估计 Laplace 尺度，再冻结权重强调难区。Laplace 尺度 θ 的方差为 2θ²；该尺度不等于经校准的后验不确定性。其 PAN 单通道设置、潜在特征通道与本基线 MSI3/物理波长存在差异。可作门控/加权监督的最近邻，不能把新名称当新机制。[官方论文](https://ojs.aaai.org/index.php/AAAI/article/view/37988)
- **CARL：**共享单波段 patch 前端、波长编码、学习光谱 query 汇聚与空间编码器，服务于大规模光谱自监督表示和下游语义任务。可迁移小前端处理变通道输入，但 CAVE 小样本不足以支持照搬预训练规模；语义指标也不能替代光谱重建证据。PINO、SpecAware、CARL 已共同压缩“波长感知通用前端”的原创空间。[官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html)

## 直接领域查新威胁：未全文者严格保留深度标签

|论文与官方身份|这次读取深度|对提案的影响|
|---|---|---|
|[CoPGNet: Exploring Structured Uncertainty for External Prior-Guided Hyperspectral Image Fusion](https://www.sciencedirect.com/science/article/pii/S1566253525010590)，Information Fusion 129:103997，2026-05；DOI 10.1016/j.inffus.2025.103997|官方索引摘要、引言、章节片段；canonical 完整/abs HTML 均 403，**未取得全文**|已有 clean-HSI 外部先验、内部结构先验、区域调制和 SAM 引导两阶段不确定性加权。广义“结构不确定性融合”已有直接先例。具体收益标签、注入定义与其方法的区别仍待全文闭合。|
|[VPG-Net: Uncertainty-Driven Generative Prior Learning for Sparse Model-Guided Hyperspectral Image Fusion](https://ieeexplore.ieee.org/document/11520599)，TIP 35:5224–5238，2026；DOI 10.1109/TIP.2026.3689415|出版者 DOI 元数据、作者登记与索引摘要；全文级联失败，IEEE 验证页|不确定性驱动的干净码本匹配、生成先验与稀疏展开已有先例。暂不能断言与 I13 的精确差异。确切出版日未核实，Crossref created 时间不是出版日。|
|[Heterospectral Structure Compensation Sampling for Hyperspectral Fusion Computational Imaging](https://ieeexplore.ieee.org/abstract/document/11276863)，TIP 34:7930–7942，2025-12-03；DOI 10.1109/TIP.2025.3636676|官方出版记录/摘要、[作者仓库](https://github.com/1318133/HSC-Sampling) README；全文级联失败|跨波段结构互补与多阶段交互已提出，不能声称首次利用 LR-HSI 其他波段的空间细节。|
|[ASSR-Net](https://arxiv.org/abs/2604.05742)，arXiv 2026-04-07，**未确认正式主会/期刊**|作者摘要|方向性空间增强再以 LR-HSI 锚定光谱重校准已有公开先例；不算近两年顶会推荐。|
|[ARGS-Diff](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.html)，CVPR 2025 主会，2025-06|官方摘要与元数据；发现候选|自学习 HSI–MSI、低维因子、残差引导扩散已有方案。|
|[DFPIR](https://openaccess.thecvf.com/content/CVPR2025/html/Tian_Degradation-Aware_Feature_Perturbation_for_All-in-One_Image_Restoration_CVPR_2025_paper.html)，CVPR 2025 主会，2025-06|官方摘要与元数据；发现候选|退化引导特征扰动已覆盖宽泛“退化条件化”命题。|
|[Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Towards_a_Universal_Image_Degradation_Model_via_Content-Degradation_Disentanglement_ICCV_2025_paper.html)，ICCV 2025 主会，2025-10|官方摘要与元数据；发现候选|内容与退化解耦已被提出。|
|[UMFNet](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Uncertainty-Aware_Modality_Fusion_for_Unaligned_RGB-T_Salient_Object_Detection_CVPR_2026_paper.html)，CVPR 2026 主会，2026-06|官方摘要与元数据；发现候选|非配准 RGB-T 的不确定性调制也是门控类迁移的跨方向先例，语义输出不同。|
|[Multispectral Demosaicing via Dual Cameras](https://openaccess.thecvf.com/content/ICCV2025/html/Tedla_Multispectral_Demosaicing_via_Dual_Cameras_ICCV_2025_paper.html)，ICCV 2025 主会，2025-10|官方摘要与元数据；发现候选|双传感器引导重建已存在；马赛克观测不同，不能照搬比较。|
|[Robust Depth Estimation Under Sensor Degradations: A Multi-Sensor Fusion Perspective](https://ieeexplore.ieee.org/document/11045210)，TPAMI 47(10):8691–8707，2025-10；DOI 10.1109/TPAMI.2025.3581311|出版者 DOI 元数据；IEEE 验证页；发现候选|是传感器失效下融合可靠性的相关领域威胁，未读全文，不推定其详细算法。|

发现候选不属于“已精读推荐”，表内重叠判断仅限官方可见摘要或作者说明。Information Fusion 是用户列举以外的同领域期刊；这里只用作直接查新证据，不用“顶刊”标签替代方法比较。

## 对 I13 主线的收口

保留一个狭义的实验性贡献候选：**在相同观测与干预条件下，监督预测 MSI 细节对 HSI-only 专家的光谱误差收益，并据此控制注入。**本次未发现足以证明这一精确定义完全相同的已读论文；这只是当前检索结果，不是原创证明。CoPGNet/VPG-Net 全文缺口、ASSR-Net 光谱重校准，以及收益门控在图像复原中的既有工作都必须进入最终查新。

需要明确区别：

1. 预测误差尺度/置信度，与预测“使用 MSI 比不用 MSI 改善多少”不同；这个区别必须落实为同算子同扰动下的标签和消融，不能只改用语。
2. 高频纹理是否可靠，与高光谱重建是否有收益不同。波段 SAM、绝对误差、窄谱峰和失配退化需同时验证，不能只报视觉改善。
3. 优于 baseline 不足以支持必要机制。收益标签打乱、仅误差预测、普通注意力、基于残差的一致性门控应成为强对照；去掉收益监督后若仍有同样收益，应撤销该贡献。
4. 最终结果若仅来自新增参数、额外 GT 教师或不一致干预，需缩减为工程改进。跨传感器/真实噪声/实测照明没有数据时，不写成已验证泛化贡献。

未建议同时叠加六个新模块。优先把 I13 标签、干预一致性和最近邻对照做扎实；RND 的噪声支线、SIT 的照明支线、DRFusion 的小步残差支线按数据条件逐项决定。
