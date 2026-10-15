# 核心论文阅读卡与迁移边界

所有“迁移”均为本次探索提出的假设，尚未训练验证。来源结论仅概括已读内容；作者报告的分数不与本仓库直接拼表。

## srlf — A Selective Re-learning Mechanism for Hyperspectral Fusion Imaging

CVPR 2025 main · 选择性计算 / HSI–MSI 融合 · [原文](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html)

机制与贡献：先初步融合，再用回投影后的空间与光谱偏差选择需要重新学习的位置。

对 baseline 的迁移设想：在融合特征后增加小型修正头，先以固定比例选择测试是否有收益。

查新后必须承认的重叠：回投影残差选点、SAM/SSIM 驱动重新学习已经存在，不能作为新贡献。

证据边界：FLOPs 的下降不能直接换算成 4090 实际延迟下降。

## uafl — Enhancing Unregistered Hyperspectral Image Super-Resolution via Unmixing-based Abundance Fusion Learning

CVPR 2026 main · 未配准引导 / 丰度空间 · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)

机制与贡献：SVD 子空间内学习丰度残差，结合多尺度形变聚合与空间、通道交互。

对 baseline 的迁移设想：丰度残差可代替当前 31 通道尾部残差；先与直接残差在相同训练预算下比较。

查新后必须承认的重叠：低秩残差、丰度融合、匹配置信度及流引导偏移均已有直接先例。

证据边界：输出包含原始上采样 HSI 跳连，不能把其整体输出误说成严格 rank-3；SVD 分量也不自动满足非负端元约束。

## umnet — UMNet: Uncertainty-guided Memory Network for Hyperspectral Pansharpening

AAAI 2026 main · 结构化不确定性 / HS–PAN · [原文](https://ojs.aaai.org/index.php/AAAI/article/view/37988)

机制与贡献：空间、光谱因子化不确定性配合递归融合；先学习不确定性，再提高困难位置的训练权重。

对 baseline 的迁移设想：用独立小头估计训练误差分布，研究其能否区分可纠正错误与不可辨识误差。

查新后必须承认的重叠：空间×光谱不确定性、异方差损失和困难样本加权已经存在。

证据边界：PAN 单通道结果不能直接证明三通道 MSI 融合有效；误差尺度也不等于已校准的真实后验方差。

## hipandas — Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion

ICCV 2025 main · 联合去噪 / 低秩细节 · [原文](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Hipandas_Hyperspectral_Image_Joint_Denoising_and_Super-Resolution_by_Image_Fusion_ICCV_2025_paper.html)

机制与贡献：分阶段处理噪声与超分辨率，使用低秩网络建模 HSI 及细节，并学习 PAN 响应。

对 baseline 的迁移设想：保留有监督 L1，将低秩细节头作为消融，增加噪声压力测试。

查新后必须承认的重叠：低秩细节先验、联合去噪与超分辨率不是新的迁移点。

证据边界：原方法是按场景无监督优化；低秩头可能损伤稀少材料是待验证假设，不能当作已发现的缺陷。

## unipan — Enpowering Your Pansharpening Models with Generalizability: Unified Distribution is All You Need

ICCV 2025 main · 分布迁移 / 分位数变换 · [原文](https://openaccess.thecvf.com/content/ICCV2025/html/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.html)

机制与贡献：逐波段经验分位数变换，把输入映射到统一分布；在原始域监督并保留原始 LR 跳连。

对 baseline 的迁移设想：仅作为泛化对照，变换参数只能在训练场景拟合。

查新后必须承认的重叠：原始辐射域损失与原始跳连已在 Algorithm 1 中，不能包装成我们的改进。

证据边界：边际对齐不保证联合光谱关系；论文已指出多维扩展方向，不能声称首次提出。

## pino — Physics-informed Neural Operator for Pansharpening

NeurIPS 2025 main · 连续表示 / 成像算子 · [原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/89af8e4eb696738f2c9e589522968a09-Paper-Conference.pdf)

机制与贡献：编码后查询连续空间、波长表示，通过 SRF 积分形成输出，并保留 LR 跳连。

对 baseline 的迁移设想：共享连续光谱残差解码器可替换固定 31 通道 tail；输入 SRF 应先使用已知标定值。

查新后必须承认的重叠：它已经包含 CAVE HSI–MSI 融合实验，不能把“从 pansharpening 迁移到融合”本身作为创新。

证据边界：文中不同传感器实验不等于已证明任意通道数、任意 SRF 单模型泛化；联合学习辐射场和 SRF 还涉及可辨识性。

## diff_spectra — Unraveling Metameric Dilemma for Spectral Reconstruction: A High-Fidelity Approach via Semi-Supervised Learning

NeurIPS 2025 main · 同色异谱 / 生成先验 · [原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf)

机制与贡献：结合光照、色度处理，同色异谱增强及光谱扩散先验进行 RGB→HSI 重建。

对 baseline 的迁移设想：优先迁移歧义压力测试和训练增强，暂不搬入完整多步扩散链。

查新后必须承认的重叠：同色异谱增强、范围/零空间分解、光谱扩散均已有先例。

证据边界：任务没有 LR HSI；同样的输入不能由确定性模型区分真实零空间分量。印刷公式存在需核对之处，见正文。

## carl — CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis

ICLR 2026 main · 跨相机表示 / 波长条件化 · [原文](https://proceedings.iclr.cc/paper_files/paper/2026/file/282967f8abaae52a452a97ee961410f3-Paper-Conference.pdf)

机制与贡献：物理波长编码与光谱聚合生成固定宽度表示，再学习空间表示和跨相机特征。

对 baseline 的迁移设想：仅借用共享波段编码与聚合，不在 CAVE 从头复现大规模基础模型预训练。

查新后必须承认的重叠：通道数无关表示、波长编码和跨相机表示学习已经存在。

证据边界：主要验证分类与分割，不能直接推断高保真光谱重建；语义不变性可能损失幅值是研究问题。

## gsnr — GSNR: Graph Smooth Null-Space Representation for Inverse Problems

CVPR 2026 main · 图先验 / 不可观测分量 · [原文](https://openaccess.thecvf.com/content/CVPR2026/html/Gualdron-Hurtado_GSNR_Graph_Smooth_Null-Space_Representation_for_Inverse_Problems_CVPR_2026_paper.html)

机制与贡献：在测量零空间内用图平滑构造低维结构，并学习其系数作为逆问题先验。

对 baseline 的迁移设想：作为零空间正则化的强对照；融合任务应使用两个观测组成的联合算子。

查新后必须承认的重叠：“在不可见分量上加入图先验/学习低维零空间”已经存在，NPN 也是直接前驱。

证据边界：全图特征分解可能昂贵；已读主 PDF，另附补充材料未取得，因此不据此作完整理论审计。

## specaware — SpecAware: A Spectral-Content Aware Foundation Model for Unifying Multi-Sensor Learning in Hyperspectral Remote Sensing Mapping

ISPRS JPRS 2026 journal · 传感器条件超网络 / 多传感器 · [原文](https://doi.org/10.1016/j.isprsjprs.2026.02.024)

机制与贡献：元数据与内容联合生成每波段矩阵因子，动态编码、动态解码可变通道 HSI。

对 baseline 的迁移设想：轻量共享波段前端可适配 Net；需要检验完整 SRF 形状是否比波长/FWHM 更有效。

查新后必须承认的重叠：可变通道解码器、重建预训练、Charbonnier+SAM 保谱损失都已经存在。

证据边界：主要下游任务是语义解译；全文为作者 arXiv v2，期刊身份另经 DOI 和作者代码核实，未逐字比对期刊最终排版。

## 关键否定性先例

**Task Decoupled Framework for Reference-based Super-Resolution**（CVPR 2022 main，历史先例）。单图 SR 与参考纹理迁移分开；冻结单图 SR 后，以 HR−SISR 残差监督纹理分支，再用匹配置信度控制迁移。 [原文](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.pdf)

**Lightweight Interpretable RGB-Guided Hyperspectral Super-Resolution under Real Cross-resolution Misalignment**（arXiv 2026-09-01; 未核实正式发表）。配准可靠性加权细节注入；低置信度回退到上采样 HSI。 [原文](https://arxiv.org/abs/2609.01060)

**Separating Intrinsic Ambiguity from Estimation Uncertainty in Deep Generative Models for Linear Inverse Problems**（2026; ICML FoGen workshop，非 ICML 主会）。范围/零空间级联，分开内禀歧义与传播不确定性，并以 oracle 条件作校准诊断。 [原文](https://arxiv.org/abs/2605.15050)

**Deep Hyperspectral and Multispectral Image Fusion Accounting for Spectral Variability**（TGRS 2023，历史先例）。两个时相的潜在 HR 图像与局部变化先验；不能声称首次处理跨模态局部变化。 [原文](https://www.cedric-richard.fr/Articles/wang2023deep.pdf)

**Uncertainty-Aware Null Space Networks for Data-Consistent Image Reconstruction**（arXiv 2023，历史先例）。数据一致零空间网络加输入相关误差尺度，已经结合重建与不确定性。 [原文](https://arxiv.org/abs/2304.06955)
