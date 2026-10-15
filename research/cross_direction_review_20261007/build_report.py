"""Build the source-grounded research report. Does not import or change baseline code."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]

def write(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

# Concise source paraphrases; transfer proposals are our exploratory inferences.
PAPERS = [
 dict(id='srlf', title='A Selective Re-learning Mechanism for Hyperspectral Fusion Imaging', venue='CVPR 2025 main', year=2025, field='选择性计算 / HSI–MSI 融合', url='https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html', mechanism='先初步融合，再用回投影后的空间与光谱偏差选择需要重新学习的位置。', transfer='在融合特征后增加小型修正头，先以固定比例选择测试是否有收益。', overlap='回投影残差选点、SAM/SSIM 驱动重新学习已经存在，不能作为新贡献。', limit='FLOPs 的下降不能直接换算成 4090 实际延迟下降。'),
 dict(id='uafl', title='Enhancing Unregistered Hyperspectral Image Super-Resolution via Unmixing-based Abundance Fusion Learning', venue='CVPR 2026 main', year=2026, field='未配准引导 / 丰度空间', url='https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html', mechanism='SVD 子空间内学习丰度残差，结合多尺度形变聚合与空间、通道交互。', transfer='丰度残差可代替当前 31 通道尾部残差；先与直接残差在相同训练预算下比较。', overlap='低秩残差、丰度融合、匹配置信度及流引导偏移均已有直接先例。', limit='输出包含原始上采样 HSI 跳连，不能把其整体输出误说成严格 rank-3；SVD 分量也不自动满足非负端元约束。'),
 dict(id='umnet', title='UMNet: Uncertainty-guided Memory Network for Hyperspectral Pansharpening', venue='AAAI 2026 main', year=2026, field='结构化不确定性 / HS–PAN', url='https://ojs.aaai.org/index.php/AAAI/article/view/37988', mechanism='空间、光谱因子化不确定性配合递归融合；先学习不确定性，再提高困难位置的训练权重。', transfer='用独立小头估计训练误差分布，研究其能否区分可纠正错误与不可辨识误差。', overlap='空间×光谱不确定性、异方差损失和困难样本加权已经存在。', limit='PAN 单通道结果不能直接证明三通道 MSI 融合有效；误差尺度也不等于已校准的真实后验方差。'),
 dict(id='hipandas', title='Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion', venue='ICCV 2025 main', year=2025, field='联合去噪 / 低秩细节', url='https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Hipandas_Hyperspectral_Image_Joint_Denoising_and_Super-Resolution_by_Image_Fusion_ICCV_2025_paper.html', mechanism='分阶段处理噪声与超分辨率，使用低秩网络建模 HSI 及细节，并学习 PAN 响应。', transfer='保留有监督 L1，将低秩细节头作为消融，增加噪声压力测试。', overlap='低秩细节先验、联合去噪与超分辨率不是新的迁移点。', limit='原方法是按场景无监督优化；低秩头可能损伤稀少材料是待验证假设，不能当作已发现的缺陷。'),
 dict(id='unipan', title='Enpowering Your Pansharpening Models with Generalizability: Unified Distribution is All You Need', venue='ICCV 2025 main', year=2025, field='分布迁移 / 分位数变换', url='https://openaccess.thecvf.com/content/ICCV2025/html/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.html', mechanism='逐波段经验分位数变换，把输入映射到统一分布；在原始域监督并保留原始 LR 跳连。', transfer='仅作为泛化对照，变换参数只能在训练场景拟合。', overlap='原始辐射域损失与原始跳连已在 Algorithm 1 中，不能包装成我们的改进。', limit='边际对齐不保证联合光谱关系；论文已指出多维扩展方向，不能声称首次提出。'),
 dict(id='pino', title='Physics-informed Neural Operator for Pansharpening', venue='NeurIPS 2025 main', year=2025, field='连续表示 / 成像算子', url='https://proceedings.neurips.cc/paper_files/paper/2025/file/89af8e4eb696738f2c9e589522968a09-Paper-Conference.pdf', mechanism='编码后查询连续空间、波长表示，通过 SRF 积分形成输出，并保留 LR 跳连。', transfer='共享连续光谱残差解码器可替换固定 31 通道 tail；输入 SRF 应先使用已知标定值。', overlap='它已经包含 CAVE HSI–MSI 融合实验，不能把“从 pansharpening 迁移到融合”本身作为创新。', limit='文中不同传感器实验不等于已证明任意通道数、任意 SRF 单模型泛化；联合学习辐射场和 SRF 还涉及可辨识性。'),
 dict(id='diff_spectra', title='Unraveling Metameric Dilemma for Spectral Reconstruction: A High-Fidelity Approach via Semi-Supervised Learning', venue='NeurIPS 2025 main', year=2025, field='同色异谱 / 生成先验', url='https://proceedings.neurips.cc/paper_files/paper/2025/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf', mechanism='结合光照、色度处理，同色异谱增强及光谱扩散先验进行 RGB→HSI 重建。', transfer='优先迁移歧义压力测试和训练增强，暂不搬入完整多步扩散链。', overlap='同色异谱增强、范围/零空间分解、光谱扩散均已有先例。', limit='任务没有 LR HSI；同样的输入不能由确定性模型区分真实零空间分量。印刷公式存在需核对之处，见正文。'),
 dict(id='carl', title='CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis', venue='ICLR 2026 main', year=2026, field='跨相机表示 / 波长条件化', url='https://proceedings.iclr.cc/paper_files/paper/2026/file/282967f8abaae52a452a97ee961410f3-Paper-Conference.pdf', mechanism='物理波长编码与光谱聚合生成固定宽度表示，再学习空间表示和跨相机特征。', transfer='仅借用共享波段编码与聚合，不在 CAVE 从头复现大规模基础模型预训练。', overlap='通道数无关表示、波长编码和跨相机表示学习已经存在。', limit='主要验证分类与分割，不能直接推断高保真光谱重建；语义不变性可能损失幅值是研究问题。'),
 dict(id='gsnr', title='GSNR: Graph Smooth Null-Space Representation for Inverse Problems', venue='CVPR 2026 main', year=2026, field='图先验 / 不可观测分量', url='https://openaccess.thecvf.com/content/CVPR2026/html/Gualdron-Hurtado_GSNR_Graph_Smooth_Null-Space_Representation_for_Inverse_Problems_CVPR_2026_paper.html', mechanism='在测量零空间内用图平滑构造低维结构，并学习其系数作为逆问题先验。', transfer='作为零空间正则化的强对照；融合任务应使用两个观测组成的联合算子。', overlap='“在不可见分量上加入图先验/学习低维零空间”已经存在，NPN 也是直接前驱。', limit='全图特征分解可能昂贵；已读主 PDF，另附补充材料未取得，因此不据此作完整理论审计。'),
 dict(id='specaware', title='SpecAware: A Spectral-Content Aware Foundation Model for Unifying Multi-Sensor Learning in Hyperspectral Remote Sensing Mapping', venue='ISPRS JPRS 2026 journal', year=2026, field='传感器条件超网络 / 多传感器', url='https://doi.org/10.1016/j.isprsjprs.2026.02.024', mechanism='元数据与内容联合生成每波段矩阵因子，动态编码、动态解码可变通道 HSI。', transfer='轻量共享波段前端可适配 Net；需要检验完整 SRF 形状是否比波长/FWHM 更有效。', overlap='可变通道解码器、重建预训练、Charbonnier+SAM 保谱损失都已经存在。', limit='主要下游任务是语义解译；全文为作者 arXiv v2，期刊身份另经 DOI 和作者代码核实，未逐字比对期刊最终排版。'),
]

EXTRA = [
 ('EMR-Diff','CVPR 2026 main','边缘感知、多模态残差扩散','https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_EMR-Diff_Edge-aware_Multimodal_Residual_Diffusion_Model_for_Hyperspectral_Image_Super-resolution_CVPR_2026_paper.html'),
 ('GEWDiff','AAAI 2026 main','几何约束、波形/小波与扩散相关方向的直接对照','https://ojs.aaai.org/index.php/AAAI/article/view/37978'),
 ('CO²IF','AAAI 2026 main','语言引导、跨模态最优传输与融合','https://ojs.aaai.org/index.php/AAAI/article/view/38261'),
 ('OTIAS','AAAI 2025 main','自适应采样与隐式表示','https://ojs.aaai.org/index.php/AAAI/article/view/32275'),
 ('CLoRF','IJCV 2025 journal','任意分辨率、自监督连续表示融合','https://link.springer.com/article/10.1007/s11263-025-02540-1'),
 ('MossFuse','IJCV 2026 journal','自监督模态解耦、无监督 HSI 超分辨率','https://link.springer.com/article/10.1007/s11263-026-02757-8'),
 ('SSC-HSR','IJCV 2025 journal','未对齐 RGB 引导与空间光谱一致性','https://link.springer.com/article/10.1007/s11263-025-02466-8'),
 ('HyperSIGMA','TPAMI 2025 journal','HSI 基础模型，多任务包括重建方向','https://github.com/WHU-Sigma/HyperSIGMA'),
 ('CDPS','TIP 2026 journal','耦合扩散后验采样、无监督 HSI–MSI 融合','https://github.com/Zhu1116/CDPS-train'),
 ('NPN','NeurIPS 2025 main','传感算子专属的低维零空间学习先验','https://proceedings.neurips.cc/paper_files/paper/2025/hash/acb1891d79e617134ed604084ebbc919-Abstract-Conference.html'),
 ('CoPGNet','Information Fusion 2026 journal','外部先验引导与结构化不确定性','https://doi.org/10.1016/j.inffus.2025.103997'),
 ('BUGPan','Information Fusion 2025 journal','双域不确定性门控与可变递归计算','https://doi.org/10.1016/j.inffus.2025.102938'),
]

OPPONENTS = [
 ('task_decoupled','Task Decoupled Framework for Reference-based Super-Resolution','CVPR 2022 main，历史先例','https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.pdf','单图 SR 与参考纹理迁移分开；冻结单图 SR 后，以 HR−SISR 残差监督纹理分支，再用匹配置信度控制迁移。'),
 ('lightweight_unaligned','Lightweight Interpretable RGB-Guided Hyperspectral Super-Resolution under Real Cross-resolution Misalignment','arXiv 2026-09-01; 未核实正式发表','https://arxiv.org/abs/2609.01060','配准可靠性加权细节注入；低置信度回退到上采样 HSI。'),
 ('ambiguity_separation','Separating Intrinsic Ambiguity from Estimation Uncertainty in Deep Generative Models for Linear Inverse Problems','2026; ICML FoGen workshop，非 ICML 主会','https://arxiv.org/abs/2605.15050','范围/零空间级联，分开内禀歧义与传播不确定性，并以 oracle 条件作校准诊断。'),
 ('difiv','Deep Hyperspectral and Multispectral Image Fusion Accounting for Spectral Variability','TGRS 2023，历史先例','https://www.cedric-richard.fr/Articles/wang2023deep.pdf','两个时相的潜在 HR 图像与局部变化先验；不能声称首次处理跨模态局部变化。'),
 ('uq_nullspace','Uncertainty-Aware Null Space Networks for Data-Consistent Image Reconstruction','arXiv 2023，历史先例','https://arxiv.org/abs/2304.06955','数据一致零空间网络加输入相关误差尺度，已经结合重建与不确定性。'),
]

write('sources.json', {'searched_on':'2026-10-07','window':['2024-10-07','2026-10-07'],'core_fulltext':PAPERS,'extended_catalog':[dict(id=x[0],venue=x[1],purpose=x[2],url=x[3],depth='official abstract / metadata or author repository; not full-text read in this pass') for x in EXTRA],'opponents_fulltext':[dict(id=x[0],title=x[1],venue=x[2],url=x[3],overlap=x[4]) for x in OPPONENTS]})

cards = '# 核心论文阅读卡与迁移边界\n\n所有“迁移”均为本次探索提出的假设，尚未训练验证。来源结论仅概括已读内容；作者报告的分数不与本仓库直接拼表。\n'
for p in PAPERS:
    cards += f"\n## {p['id']} — {p['title']}\n\n{p['venue']} · {p['field']} · [原文]({p['url']})\n\n机制与贡献：{p['mechanism']}\n\n对 baseline 的迁移设想：{p['transfer']}\n\n查新后必须承认的重叠：{p['overlap']}\n\n证据边界：{p['limit']}\n"
cards += '\n## 关键否定性先例\n'
for key,title,venue,url,overlap in OPPONENTS:
    cards += f'\n**{title}**（{venue}）。{overlap} [原文]({url})\n'
write('paper-cards.md', cards)

report = r'''# 基于当前 CAVE baseline 的近两年论文迁移、查新与研究路线

检索日：2026-10-07。滚动时间窗：2024-10-07 至 2026-10-07。训练偏好：保持有监督、优先方法创新；用户硬件：4090，未指定时间预算。

## 1. 最终判断

**先研究“监督细节注入的实际光谱收益”；跨传感器连续重建作为储备；不推荐用通用模块替换、置信度门控、低秩残差、零空间不确定性或扩散拼装作为独立创新。**

本次没有找到可以负责任地宣称“已确认无人提出”的方案。保留的一条主线是查新后仍可检验的窄方法假设；另一条储备路线有明确用途，但目前方法新颖性不足。任何“顶刊潜力”都必须经过强对照与实验兑现，不能由论文名称或模块组合推出。

已阅读全文的核心集包含 9 篇指定类型顶会主会论文，加上 ISPRS JPRS 的 SpecAware；另外完整读取 5 篇关键先例。扩展目录包含 IJCV、TIP、TPAMI、AAAI、NeurIPS 等正式论文，但逐项标明摘要/元数据证据，不把它们计入全文主读。阅读、检索与未取得材料的记录均保留在本目录。

## 2. 你的 baseline 实际是什么

当前入口 [train_cave_M.py](D:/coding/github_self/thesis_base_line/train_cave_M.py:69) 使用 [model.py](D:/coding/github_self/thesis_base_line/model.py:57) 的 `Net`，并不是 `other_base/baseline.py` 的 Transformer。

```mermaid
flowchart LR
  A[LR HSI: 31 bands] --> B[两层卷积 + 特征双线性上采样]
  C[HR MSI: 3 bands] --> D[两层卷积]
  B --> E[拼接 + 128到64卷积]
  D --> E
  E --> F[8个残差块 + 潜在通道注意力]
  F --> G[64到31残差头]
  A --> H[原始HSI双线性上采样]
  G --> I[相加得到HR HSI]
  H --> I
```

注意力的 64 个通道是潜在特征，不是 31 个物理波段。[通道池化](D:/coding/github_self/thesis_base_line/model.py:31)、[融合入口](D:/coding/github_self/thesis_base_line/model.py:112) 和 [输出残差](D:/coding/github_self/thesis_base_line/model.py:116) 是具体改造接口。

退化使用固定、行归一化的 3×31 SRF，以及 8×8、σ=3 的高斯核和 ×8 抽样；训练使用 HR 真值 L1。当前缺少传感器变化、错配/遮挡、多时相变化和噪声专项评价。因此方法方向应先建立可测量的失败条件，而不是假定 CNN 一定缺乏某种能力。

**一个会影响结论的协议问题：训练脚本用 `cave_test` 周期评估并选择最佳 PSNR/RMSE 权重。** [相关代码](D:/coding/github_self/thesis_base_line/train_cave_M.py:163)。研究时必须另设场景级验证集，保留最终测试场景；这是保证结论可靠的基础，不是方法贡献。当前 20 个训练、12 个测试文件名没有交集，仍需防止按重叠 patch 划分导致泄漏。

## 3. 核心论文地图：不同方向如何迁移

下表的“迁移判断”是本次推论，不是原论文已证明的结论。论文详细边界见 [paper-cards.md](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/paper-cards.md)。

|方向|论文与正式场所|可借用机制|落到 Net 的位置|查新后的判断|
|---|---|---|---|---|
|选择性修正|[SRLF](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html)，CVPR 2025|回投影偏差选点，再重新学习|body 后的小修正头|可当强对照；残差选点不是新点|
|未配准引导|[UAFL](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)，CVPR 2026|丰度残差、形变聚合|fuse 前与 tail|直接搬低秩/置信度已重复|
|不确定性训练|[UMNet](https://ojs.aaai.org/index.php/AAAI/article/view/37988)，AAAI 2026|空间光谱误差尺度、分阶段训练|独立误差头与损失|异方差和困难加权已重复|
|噪声与细节|[Hipandas](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Hipandas_Hyperspectral_Image_Joint_Denoising_and_Super-Resolution_by_Image_Fusion_ICCV_2025_paper.html)，ICCV 2025|低秩细节与去噪阶段|tail 与训练退化|有监督迁移可做消融，不是新贡献|
|分布泛化|[UniPAN](https://openaccess.thecvf.com/content/ICCV2025/html/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.html)，ICCV 2025|分位数映射到共同分布|输入归一化|保留原始 skip 和原始域监督也已有|
|连续成像|[PINO](https://proceedings.neurips.cc/paper_files/paper/2025/file/89af8e4eb696738f2c9e589522968a09-Paper-Conference.pdf)，NeurIPS 2025|连续辐射表示、SRF 积分|共享光谱解码器|已做 CAVE 融合，不能声称首次迁移|
|同色异谱|[Diff-Spectra](https://proceedings.neurips.cc/paper_files/paper/2025/file/c34ddd05eb089991f06f3c5dc36836e0-Paper-Conference.pdf)，NeurIPS 2025|歧义增强、光谱生成先验|压力测试，后续可研究先验|无 LR HSI；相同输入歧义仍存在|
|相机无关表示|[CARL](https://proceedings.iclr.cc/paper_files/paper/2026/file/282967f8abaae52a452a97ee961410f3-Paper-Conference.pdf)，ICLR 2026|波长编码、变通道聚合|输入两头|语义任务收益不自动等于保谱重建收益|
|不可见分量先验|[GSNR](https://openaccess.thecvf.com/content/CVPR2026/html/Gualdron-Hurtado_GSNR_Graph_Smooth_Null-Space_Representation_for_Inverse_Problems_CVPR_2026_paper.html)，CVPR 2026|图平滑零空间、学习低维系数|联合观测正则项|零空间图先验已经有强先例|
|元数据条件化|[SpecAware](https://doi.org/10.1016/j.isprsjprs.2026.02.024)，ISPRS JPRS 2026|动态波段编码与动态解码|输入头、tail|动态解码与重建保谱损失都已有|

优先读 UAFL、最新未配准先例、UMNet、SRLF，以判断主路线；再读 PINO、SpecAware、CARL，理解储备路线。Diff-Spectra、GSNR 与歧义分解论文用于辨识方法能解决的边界。

## 4. 查新中淘汰了什么

|最初看似合理的想法|最强重叠|决定|
|---|---|---|
|共享/私有模态解耦|MossFuse，且本仓库已有相同 ideation|淘汰宽泛主张|
|低秩或丰度空间残差|UAFL、Hipandas|淘汰“首次低秩残差”|
|几何/边缘/小波加扩散|GEWDiff、EMR-Diff、CDPS|不能靠拼装支撑主贡献|
|回投影残差选择难点|SRLF|作为必须比较的基线|
|置信度门控与低置信度回退|UAFL；2026-09 的 RGB 引导先例|核心主张被直接覆盖|
|不确定性门控、困难加权、动态递归|UMNet、CoPGNet、BUGPan|不能声称首次；后两篇全文未得，保留高优先级查新缺口|
|波长/带宽条件、可变通道编码与解码|CARL、SpecAware，连续成像另有 PINO|储备工程路线，新颖性不足|
|在零空间上学先验|NPN、GSNR|被覆盖|
|内禀歧义与估计不确定性分离并校准|Guo 等 2026、Angermann 等 2023|被直接先例覆盖|

两篇尤其值得注意的否定性先例：

- [Lightweight Interpretable RGB-Guided HSR](https://arxiv.org/abs/2609.01060)：已经用配准可靠性控制细节注入，并回退到 HSI 上采样。它是预印本，不计入顶会正式清单，但一样能够否定宽泛的新颖性。
- [Separating Intrinsic Ambiguity…](https://arxiv.org/abs/2605.15050)：已经采用范围/零空间级联、内禀歧义与传播不确定性分解，以及 oracle 条件下的校准诊断。其 ICML 2026 FoGen **workshop** 身份不能写成 ICML 主会。

老论文也必须用于查新：[DIFIV](https://www.cedric-richard.fr/Articles/wang2023deep.pdf) 已考虑局部光谱变化；[Uncertainty-Aware Null Space Networks](https://arxiv.org/abs/2304.06955) 已把数据一致性与输入相关不确定性结合；[TTSR](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Learning_Texture_Transformer_Network_for_Image_Super-Resolution_CVPR_2020_paper.html) 早已通过相关性控制参考纹理迁移。它们不是近两年推荐集，却是不能跳过的先例。

## 5. 保留主线：监督细节注入的实际光谱收益

### 5.1 研究问题与贡献边界

**几何匹配得好、测量残差小的 MSI 细节，是否仍可能增加 HSI 重建误差？能否直接学习“注入后光谱误差降低多少”，据此决定注入强度？**

这是新生成的窄假设 I13。它在看到通用置信度回退被覆盖后形成；不把被淘汰的 I12 偷换成已经通过查新的方案。新颖性定位评分 **3/5（仅窄范围探索）**，判定置信度 **2/5**。

候选贡献是“训练目标与决策依据”从匹配可靠性改为条件光谱收益，配合能区分二者的反例和强对照。双分支、门控、小头、oracle 教师、最优凸混合公式本身都是通用工具，均不声称首次。

### 5.2 可以直接映射到 baseline 的设计

保留 CNN 主体，设置 HSI-only 专家 (B) 和 HSI–MSI 专家 (F)：

\[
B=f_H(L),\quad F=f_{HM}(L,M),\quad d=F-B,\quad \hat X=B+g_\psi(o)\,d.
\]

$f_{HM}$ 可以从当前 `Net` 开始；$f_H$ 必须用 HSI-only 输入明确训练。不能把当前模型的 MSI 直接置零就当成经过训练的回退专家。首轮允许共享 LR 编码器，但要比较等预算双专家，隔离额外容量收益。

训练时，在每个局部空间块 $p$ 上定义三种信号：

1. 几何匹配可靠性：warp 后结构或特征相关性，仅作对照。
2. 可见域吻合：由 $R\hat X$ 与 $\hat XD^\top$ 的误差得到，仅作输入证据。
3. 真正的监督收益：
   \[
   \Delta_p=\ell(B_p,X_p)-\ell(F_p,X_p).
   \]
   其中 $X$ 只用于训练标签与测试评价；推理头输入 $o$ 仅包含观测、已知退化、专家预测和中间特征。

先学习收益回归与正负收益分类，不把输出自动解释成校准概率。可用损失：

\[
\mathcal L=\|\hat X-X\|_1+
\lambda_v\,\operatorname{Huber}(v_\psi(o),\operatorname{stopgrad}(\Delta))+
\lambda_s\,\operatorname{BCE}(a_\psi(o),\mathbf1[\Delta>\tau]).
\]

门控具体取 $g_\psi=\sigma(a_\psi)$，其中 $a_\psi$ 是收益正负分类的 logit；$v_\psi$ 为同一小头共享特征上的收益回归输出。BCE 使用 logits 形式；Huber 回归标签可用训练集尺度归一化。两项共同监督收益决策，不能让一个与门控无关的辅助回归头替代真正的决策改造。阈值 $\tau$ 根据训练尺度和独立验证集设置，记录灵敏度；第一轮设置空间块门控而不是 31 个波段分别随意门控，以减少不连续谱线风险。

平方误差下，一个冻结专家对的单块最优凸权重可作为教学标签：

\[
g_p^*=\operatorname{clip}\!\left(
\frac{\langle X_p-B_p,d_p\rangle}{\|d_p\|_2^2+\epsilon},0,1\right).
\]

这是标准凸组合最小二乘构造，不是新理论；当 $d_p\ne0$ 且 $\epsilon=0$ 时是平方误差下的精确解，加入正 $\epsilon$ 后是稳定化近似，$d_p=0$ 时可取 0。它不是 L1 或 SAM 的通用最优解。可把预测 $g^*$ 的小头列为简单强对照。若拟议收益模型不能超过这个普通门控方案，主张应降为常规工程。

训练顺序：先独立训练专家；冻结专家生成收益标签；按场景分折或保留训练内子集训练收益头，避免只拟合专家在自身训练 patch 上的过低误差；最后尝试小学习率联合优化。联合阶段标签必须 detach 或来自固定教师，否则专家可能通过改变自身输出来操纵标签。

推理时不读取 HR 真值。测试中使用 HR 的 oracle 门控只展示上限，不能计入部署模型得分。

### 5.3 与最接近论文逐项比较

|方法|控制什么|训练目标/证据|我们的差异假设|仍可能被否定的原因|
|---|---|---|---|---|
|UAFL|对齐与丰度特征聚合|配准信号、特征相似性、有监督重建|直接监督注入对整条光谱误差的收益|损失功能等价或收益只是对齐置信度代理|
|2026-09 轻量 RGB 引导先例|可靠区域细节注入与回退|配准有效性、结构一致性|同样可靠性下区分正/负光谱收益|其可靠性设计足以完成同样筛选|
|UMNet|重建误差尺度与困难位置训练|异方差与分阶段损失|预测两专家差额，而不是单模型误差尺度|差额只是 UQ 的简单变形|
|SRLF|重新学习位置|回投影的空间/光谱偏差|小回投影偏差但有负迁移的位置|测量残差足以解释全部收益|
|DIFIV|跨时相局部变化|双潜变量与变化先验|当前有监督单目标任务的注入收益决策|只是另一种鲁棒权重|
|TTSR / 通用 MoE|纹理相关性、专家选择|相关性与端到端生成|针对 HSI 辐射误差的收益监督|通用专家门控已经覆盖基本算法|
|CVPR 2022 Task Decoupled|单图 SR 与参考纹理任务|冻结 SISR，真值残差监督纹理分支，置信度控制迁移|显式监督两种预测的误差收益差额|真值残差学习已存在，收益头可能只是常规附加监督|

其中 [Task Decoupled Framework](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.pdf) 是全文已核查的强先例。必须比较迁移到 HSI 的“HSI-only 重建 + HR 残差监督的引导纹理分支 + 相关性门控”；不得声称首次双分支、冻结回退专家或真值残差指导纹理。CoPGNet 全文未取得，是未关闭的高优先级威胁；不能用“没看到同样公式”认定其没有覆盖这一目标。

### 5.4 能判定这条路线值不值得继续的实验

**首个实验是测“收益与可靠性是否真的不同”，不是先训新模块。** 在严格验证场景上比较 $B,F$，分块画出匹配/回投影分数与真实 $\Delta$；统计高可靠但负收益区域。若没有有规模的此类区域，路线动机不足。

之后比较同一专家输出下的：不门控、常数权重、结构置信度门控、回投影残差门控、异方差门控、普通端到端门控、直接预测 $g^*$ 的门控，以及拟议收益模型。另做上述 Task Decoupled 式真值残差监督强对照，以检验收益目标的必要性。所有方法使用同样的输入信息；有配准模块时共用配准输出，避免把更好的配准误当作新决策的收益。

压力组分别包含：正确配准；平移/非刚性错位；局部遮挡或不对应；局部光度差异；错误 SRF；噪声。**遮挡或参考替换只干预 MSI 引导，HR HSI 目标保持原场景；这是配对引导干预，不是因果识别的保证。** 合成扰动只用于受控诊断，真实结论还需要独立真实数据。

报告全图 PSNR、SAM、RMSE，同时报告负迁移区域比例、接受注入区域的光谱误差、拒绝区域的恢复质量和门控覆盖率。全部拒绝可能减少错误注入，也可能完全失去 MSI 的收益，因此必须同时测覆盖与收益。

预先采用探索性停止条件：oracle 混合相较较好专家的收益很小；普通门控已获得同样效果；增益只来自更大容量或特殊合成遮挡；净收益随 seed 反转；干净场景持续明显退化。发生任一情况，应停止将其当作主创新。

## 6. 储备路线：完整传感器响应条件下的连续保谱重建

研究问题：仅有中心波长和 FWHM 的条件表示，能否区分具有同中心、同宽度却不同响应形状的传感器？一个保持原始辐射信息的轻量连续残差解码器，是否改善未见 SRF 家族的融合？

在 Net 两个输入头中用共享单波段编码与 SRF 编码形成固定 64 宽度特征，保留 body；输出由连续残差 $q_\theta(z_p,\lambda)$ 在目标波段响应上积分：

\[
\hat X_b(p)=U(L)_b(p)+
\frac{\sum_j w_j\,q_\theta(z_p,\lambda_j)\,r_b(\lambda_j)}
{\sum_j w_j\,r_b(\lambda_j)+\epsilon}.
\]

输入 MSI 的完整响应函数也参与条件化；不同 band count 通过共享编码和集合聚合处理。目标如果仍为 CAVE 固定 31 波段，可以先固定输出查询，避免把输入变通道和输出变通道混为一个已实现能力。只在 CAVE 的可见光采样范围内做查询；不能把离散 31 波段插值当作已知连续真实光谱，不能声称可靠外推到 NIR/SWIR。

损失先保留原始域 L1，可加 SAM 和已知观测一致性，全部属于已有工具。传感器响应优先固定为已知值，不能把一切自由 SRF 参数交给网络后宣称恢复了真实物理响应。CAVE 的归一化强度不等于绝对辐射标定；这里幅值保真指统一评价尺度下的强度误差，不能宣称已恢复真实辐射单位。

**查新评分 2/5，需重大差异化；判定置信度 3/5。** PINO 已有连续成像与 CAVE 融合；SpecAware 已有动态波段编码、动态解码及保谱重建损失；CARL 已有相机无关表征。因此这条路线目前适合作为能力扩展和机理研究，不满足独立主创新的证据条件。

最低比较应包括：固定头 Net；仅随机 SRF 训练；仅中心波长条件；中心波长+FWHM 条件；完整响应条件；PINO 式连续头；轻量 SpecAware 式动态头。按独立场景×留出传感器家族测试，区分目标传感器有标定、有少量训练适配和完全未见三种设定。合成留出 SRF 只能支持合成泛化；必须用真实响应及真实数据扩展主结论。

若完整响应条件不能超过同预算元数据条件或普通随机 SRF 增强，就没有必要增加这一结构。它只有在发现可重复、非平凡的响应形状失败模式，并提出能解决它的必要机制后，才值得重新进入查新。

## 7. 不可辨识性：防止给方法不可能兑现的目标

令 HR 光谱矩阵 $X\in\mathbb R^{31\times N}$，观测为 $M=RX$、$L=XD^\top$。若存在非零扰动

\[
R\Delta=0,\quad \Delta D^\top=0,
\]

那么 $X$ 与 $X+\Delta$ 的两种观测相同。可以取 $\Delta=vq^\top$，其中 $Rv=0,Dq=0$。这不是新定理，而是线性逆问题的基本事实；学习先验能够降低分布内平均误差，不能给所有相同观测的真实对象提供唯一恢复保证。

本次用当前 `create_F()` 的实际数值做了小型解析检查：SRF 的 rank 为 3，光谱零空间维数为 28。构造两个有界 HR cube，其 MSI 最大差异约 $2.22\times10^{-16}$、LR 差异约 $3.33\times10^{-16}$，而 HR RMSE 不为零。**这里空间算子是示意性的 ×8 块均值，不是 baseline 的高斯算子；合成光谱也不声称是真实材料。** [完整结果](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/analytic-checks.json)。

这项检查只说明“观测一致”不足以证实真值，不是实验性能或新理论。收益头在完全相同的观测上仍输出相同决策，不能保证识别每个歧义实例；评价应针对给定数据分布的平均收益与失败率。

## 8. 4090 下的执行优先级

第一阶段只做当前 CNN、HSI-only 专家及小型收益头。沿用 64×64 patch 起步，先用少量 batch 测真实峰值显存，再选 batch size 与混合精度；“4090”并不提供本任务实测显存，不能在未运行时承诺训练时长或全套模型都能装下。

先用一个种子、固定更新次数完成动机与 oracle 上限试验。若信号成立，再做至少 3 个 seed 和场景级复测；增加第二个独立数据集。只在这些通过后考虑更复杂对齐/连续表示。CARL 大规模预训练、完整扩散训练与多步采样不作为起点；SpecAware 原文预训练使用多卡，不能用其时间估算单卡复现。

建议的阶段顺序：

|阶段|回答的关键问题|交付|停止条件|
|---|---|---|---|
|P0 协议冻结|结果是否可比较？|场景清单、退化、指标、验证规则|不能独立验证或数据泄漏|
|P1 两专家与上限|错误注入是否存在且可修正？|正负收益分布、oracle 上限|没有有效空间|
|P2 简单强对照|收益监督是否超过普通门控？|同预算、同配准消融|常规门控同样有效|
|P3 外部验证|收益是否超出合成训练分布？|独立数据集、多 seed、场景区间|只在一种合成扰动改善|
|P4 储备能力|完整 SRF 是否有不可替代价值？|留出传感器家族比较|普通增强足够|

尚未执行 GPU 训练、尚未测得增益。这些是探索性实验建议，不是已注册的确认性实验协议。

## 9. 指标、数据与复现细节

- 固定排序与显式场景 manifest；当前 dataloader 的 `os.listdir` 不排序。可从现有 20 训练场景暂分 16/4 作探索训练/验证，最终仍保留 12 测试场景；这只是建议划分，不是领域统一标准。可在开发后冻结规则并重训，但不能再通过测试集选择超参数。
- 当前 [PSNR](D:/coding/github_self/thesis_base_line/calculate_metrics.py:29) 是每波段误差、固定 data_range 的 dB 平均；RMSE 带 ×255 尺度，SAM 为度。还存在输出 clamp 与 patch 重叠平均。与论文比数值之前必须统一这些细节、退化边界及场景清单。
- PINO 文中的 CAVE 20/11 划分与本地 20/12 不同，PSNR 定义也需统一。不能把它的表格分数与本地结果直接拼为 SOTA 排名。
- 同一场景的 patch 和像素不独立。区间和统计应按场景聚合/重采样；少量场景导致区间宽，不能用大量 patch 人为提高显著性。
- 用全 HR cube 的最大值归一化是当前合成协议的一部分；真实推理若没有 HR 真值，必须说明可获得的标定尺度，而不能偷用未知 HR 最大值。
- 分位数、归一化统计、收益阈值及校准参数均在训练/验证场景拟合；传感器元数据不等于允许使用测试 HR 标签。

## 10. 原文核查发现与证据限度

Diff-Spectra 打印的 Eq.6–10 在 $A\in\mathbb R^{C\times c},C>c$ 的设定下使用 $(AA^\top)^{-1}$；它不可能通常可逆。若 $A$ 满列秩，应使用 $A^+=(A^\top A)^{-1}A^\top$；最小范数谱解是 $XA^+$。其增强中的投影 $A(A^\top A)^{-1}A^\top$ 则维度合理。官方 PDF 的 Eq.13 在渲染页也为空白，已排除仅由文本解析丢失的解释。[p6 图像](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/evidence-pages/diff_spectra-p6.png)、[p7 图像](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/evidence-pages/diff_spectra-p7.png)。这些是打印公式的复现核对问题，未审其代码，不声称代码一定错误或整篇结论失效。

全文检索使用 research-papers 的 canonical fetch helper，PDF 文本解析失败时才使用 pypdf 回退，并保留页标记。逐份完成解析阅读计划与 END-OF-PAPER 标记；部分长输出被截断后重新分段读取。核心 PDF 的 9 个关键页面做了可视核对。

CoPGNet、BUGPan：DOI/作者来源可核实，但 OA 检索未成功，未得全文。无邮件参数时未伪造 Unpaywall 联系方式；有 429 与无 OA 返回。GSNR 主 PDF 之外的补充材料未取得。OpenReview 曾返回浏览器验证页面并被解析器错误标为成功；该文件明确排除，改用 arXiv 全文，不把状态 ok 当成阅读证据。

这不是具有穷尽性保证的系统综述，也没有独立第二审查者。查新是同一上下文中的自审；搜索未找到不等于没有先例。2024 年 ECCV 在滚动窗起点之前，只可作历史先例；Findings、workshop 与预印本均不混记为主会。未来 2027 论文不纳入近两年正式推荐。

## 11. 扩展阅读目录

以下以正式页面、DOI 或作者代码确认，当前仅摘要/元数据级；不据此复述未读实验或断言其没有某种机制。

'''
for title,venue,purpose,url in EXTRA:
    report += f'- [{title}]({url})（{venue}）：{purpose}。\n'
report += r'''

## 12. 交付索引与继续研究时的入口

|文件|用途|
|---|---|
|[paper-cards.md](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/paper-cards.md)|逐篇贡献、迁移位置与限制|
|[idea-bank.md](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/ideation/idea-bank.md)|15 个候选与筛选理由|
|[selected-idea.md](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/ideation/selected-idea.md)|主线的最小验证与停止条件|
|[novelty-report.md](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/novelty-review/novelty-report.md)|查新结论、评分与未关闭威胁|
|[prior-art-matrix.md](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/novelty-review/prior-art-matrix.md)|最接近先例逐维比较|
|[search-log.md](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/novelty-review/search-log.md)|实际查询、目的、命中与限制|
|[reading-ledger.md](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/reading-ledger.md)|阅读深度、全文标记、未得材料|
|[sources.json](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/sources.json)|来源与正式发表类别|
|[baseline-snapshot.json](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/baseline-snapshot.json)|代码摘要哈希、数据清单与偏好|
|[analytic-checks.json](D:/coding/github_self/thesis_base_line/research/cross_direction_review_20261007/analytic-checks.json)|解析反例的实际数值与限定条件|

最适合当前深入研究的题目表述是：**“以监督光谱收益学习可靠的 MSI 细节注入决策”**。当前只允许把它写为待验证研究假设；验证后贡献句应围绕实际发现及其必要机制重写，不能先承诺首创或性能提升。
'''
write('research-report.md', report)

ledger = '# 阅读与证据账本\n\n证据截至 2026-10-07。全文指完成本地解析阅读计划与终止标记，不代表额外出版补充材料均已取得。\n\n|论文|深度|解析行数|终止标记|限制|\n|---|---|---:|---|---|\n'
for p in PAPERS:
    f=json.loads((ROOT/f"fetch-{p['id']}.json").read_text(encoding='utf-8'))
    ledger += f"|{p['id']}|全文|{f['parsed_lines']}|{f['sentinel']}|{p['limit']}|\n"
for key,title,venue,url,overlap in OPPONENTS:
    f=json.loads((ROOT/f'fetch-{key}.json').read_text(encoding='utf-8'))
    ledger += f"|{key}|全文|{f['parsed_lines']}|{f['sentinel']}|{venue}|\n"
ledger += '\n扩展目录为官方摘要/元数据/作者代码级。CoPGNet、BUGPan 的全文尝试见各 fetch JSON；不从未获取全文推断不存在重叠。\n\n视觉核查：rendered-pages.json 中 9 页均已查看；Diff-Spectra Eq.13 空白在 PDF 图像中确认。SpecAware 读作者 arXiv v2（2026-02-24），正式期刊 DOI 单独核实。\n\n长输出被截断后补读：Hipandas、Diff-Spectra、PINO、UAFL、UniPAN、DIFIV、最新未配准先例、Guo 分解论文、旧 UQ 网络、SpecAware 方法段。Guo 的 126–175 与 SpecAware 的 121–435 已重新完整输出。旧 OpenReview 浏览器验证页面虽有 sentinel，排除为无效正文；fetch-ambiguity_separation.json 已替换为真实 arXiv 正文记录。\n\n未发生独立审稿；未运行 GPU 训练或性能实验。原有 research 目录中的 geometry_condition 与 MossFuse 等 ideation 已读取，并在筛选时视为已存在的本地思路，不能再次宣称本次原创。\n'
write('reading-ledger.md', ledger)

if __name__=='__main__':
    print(json.dumps({'core_fulltext':len(PAPERS),'opponent_fulltext':len(OPPONENTS),'catalog_entries':len(EXTRA),'report_chars':len(report)},ensure_ascii=False))
