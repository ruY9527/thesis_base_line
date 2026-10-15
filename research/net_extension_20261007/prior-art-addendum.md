# I13 先例补检与贡献边界

检索日：2026-10-07；近两年正式论文窗：2024-10-07 至 2026-10-07。此次为针对决策关键缺口的有界补检，不是高召回系统综述。研究目标是找功能重叠，不是只检索相同模型名或相同公式。全部性能是原论文陈述；本仓库没有复现实验。

## 新增最强重叠

|论文|场所与时间|直接支持的机制|对 I13 的威胁|仍需区分|
|---|---|---|---|---|
|[EnsIR: An Ensemble Algorithm for Image Restoration via Gaussian Mixture Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f12d4e75bb8c62aba3e88d0586af96d3-Abstract-Conference.html)|NeurIPS 2024 Main Conference；arXiv 初稿 2024-10-30，窗内|预训练恢复模型，带 GT 参考集，按专家预测值区间估计局部组合权重，test LUT 推理|**5/5 对“用真实误差指导冻结专家局部加权”的宽泛主张**；对特定 HSI 差额目标为 4/5|GMM/range-bin 权重与光谱收益辅助头不完全相同；须同 B/F 比较，不能把命名不同当新颖性|
|[Complexity Experts are Task-Discriminative Learners for Any Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Zamfir_Complexity_Experts_are_Task-Discriminative_Learners_for_Any_Image_Restoration_CVPR_2025_paper.html)|CVPR 2025 主会；初稿 2024-11-27|异复杂度恢复专家与复杂度偏置路由，按输入选择处理容量|3/5；否定“自适应专家路由本身”新点|其主要选择复杂度/任务，不能仅摘要断言已用两专家光谱收益标签|
|[PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_PatchScaler_An_Efficient_Patch-Independent_Diffusion_Model_for_Image_Super-Resolution_ICCV_2025_paper.html)|ICCV 2025 主会；正式发表窗内，初稿 2024-05-27 在窗外|粗恢复 confidence 与 patch 难度分组，按难度分配不同采样路径|3/5；误差/难度预测指导处理已经存在|一个粗模型的难度与两动作损失差额不同；不能声称首次 GT 监督 confidence|
|[WCGE-Net: Wavelet-Decoupled Complementarity-Guided Expert Network for Hyperspectral and Multispectral Image Fusion](https://ieeexplore.ieee.org/abstract/document/11664275/)|IEEE JSTARS 19:28546–28561；2026-08-24；DOI 10.1109/JSTARS.2026.3726856|wavelet 特征解耦，空间/光谱专家，跨模态校准与跨频率动态聚合|4/5 对同任务“互补专家融合”主张|只有官方摘要，全文细节未知；不是用户列举的顶刊类型，但仍是有效先例|
|[Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html)|ICML 2024 主会，2024-07；**窗外历史先例**|回归问题中冻结预测器与两阶段 deferral、依赖 label/instance 的成本、一般有界回归损失|**5/5 对“以真实收益选专家是全新原则”**；4/5 对 endpoint cost policy|硬专家选择与连续空间门控不同；不能直接搬其理论为本网络担保|

这里威胁评分针对指定宽/窄主张，不代表论文质量或任务性能排名。没有确认“完全等价于 I13 的逐块光谱差额辅助头”的单一论文；但组合的功能层面已有明显先例。要支持独立方法贡献，必须找出必要机制及其对等对照下的证据。

## 阅读到什么、没有读到什么

**EnsIR。** 阅读了作者 [arXiv HTML](https://arxiv.org/html/2410.22959v1) 的摘要、§3.1–3.3、实验对照与有关 appendix 段落。§3.2 使用成对参考集，假定 reference/test 同分布；§3.3 在 range-bin 中求 GMM 权重并存 LUT，推理仅用专家预测。不能把它描述成不需要任何 GT，也不能把论文“training-free”理解为不估计权重。其 GMM 参数估计不同于训练一个 neural gate，但已充分覆盖后训练专家加权的基本框架。原 NeurIPS PDF 因工具长度上限未成功打开，改用作者全文版本核对；未声称本次逐页读完整 PDF。

**PatchScaler。** 读取了官方主会摘要、检索到的主文对应段落及 [补充 A.1](https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Liu_PatchScaler_An_Efficient_ICCV_2025_supplemental.pdf)。粗特征与 confidence 两头；补充 Eq. 8 为 L1 重建加 `C ||y_HR-x_HR||² - eta log C` 类型监督。它依赖误差学习 confidence，并非几何匹配 confidence；这削弱了“改成真实误差就新”的论证。它没有在已读取段落中使用两专家收益差额；该观察不等于全文中不存在其他相近机制。

**MoCE-IR。** 正式场所由 CVF 官方条目和主 PDF 核验；读取作者摘要及主文相关专家/路由段落。不是完整逐节阅读，未提取全部训练超参数，不能宣称已排除所有 overlap。

**WCGE-Net。** 官方 Xplore 可检索摘要与出版元数据；直接 open 出错，未取得全文。其摘要足以否定宽泛的空间/光谱专家与动态互补融合主张，不足以判定其路由是否已有收益教师。应列为新增高优先级全文缺口。

**Regression with Multi-Expert Deferral。** PMLR 官方摘要核验了两阶段/冻结预测器、label-dependent costs、bounded regression loss 的覆盖；未将摘要的 H-consistency guarantee 引用到我们的神经门控。它在 rolling window 之前，只计查新历史先例。

## 同领域已有关键威胁复核

1. [UMNet: Uncertainty-guided Memory Network for Hyperspectral Pansharpening，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37988)。官方页面确认 **PAN–LRHS pansharpening**，不是直接的 RGB 三通道 HSI–MSI fusion。它有空间光谱 recurrent fusion、NMF 状态记忆和两步 uncertainty-guided loss。可迁移作误差尺度强对照；不能说任务完全相同，也不能从一个模型的不确定性等同推出两动作收益差额已完全覆盖。
2. [CoPGNet，Information Fusion 2026](https://doi.org/10.1016/j.inffus.2025.103997)。官方摘要/作者所在大学的出版记录确认外部/内部先验与结构 uncertainty 加权；本次再次检索 pdf/code 没有取得全文，publisher open 403。**其是否已用先验引入后的误差差额监督权重仍未关闭。** 摘要没有写出的机制不可当成“不存在”。[作者机构出版记录](https://researchconnect.suny.edu/en/publications/exploring-structured-uncertainty-for-external-prior-guided-hypers/)。
3. [Task Decoupled Framework for Reference-based Super-Resolution，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.pdf)。既有包已全文核查：冻结 SISR、真值残差纹理分支与相关性控制迁移。这里只继承既有证据，未重做完整阅读。它直接限制冻结回退专家/GT residual 贡献表述。
4. [Lightweight Interpretable RGB-Guided HSR，2026-09 预印本](https://arxiv.org/abs/2609.01060)。既有包已读，可靠性控制细节并回退到 HSI-preserving estimate。不是正式顶会论文，但宽泛可信度回退已经覆盖。

新检索还发现 [VPG-Net: Uncertainty-Driven Generative Prior Learning for Sparse Model-Guided Hyperspectral Image Fusion，TIP 2026](https://doi.org/10.1109/TIP.2026.3689415)，35:5224–5238。作者原摘要（由 PubMed 索引）列出 VQ 清洁先验、uncertainty probabilistic matching 与 dynamic modulation。**本次 publisher DOI 访问失败、无全文，只作为需取原文的相关论文线索，不以此判定 I13 标签已被覆盖。** 不采用第三方博客的技术解读作证据。[原论文摘要索引](https://pubmed.ncbi.nlm.nih.gov/42133515/)。

## 与当前可实现方案的功能对比

|比较维度|I13 原方案|常规/已有替代|该维度能否独立称贡献|
|---|---|---|---|
|动作集合|HSI-only B 与 fusion F 的凸混合|恢复集成、reference-SR 纹理回退|否|
|训练真值|两专家 endpoint loss 差额|GT 参考集权重估计、回归成本/deferral|一般原则否|
|推理信息|观测、专家预测、中间特征、已知退化|EnsIR 专家预测值、普通 gate 特征、回投影|仅信息扩充否|
|优化目标|BCE+Huber gain aux，加重建 policy|ordinary reconstruction、cost-sensitive choice、oracle weight teacher|需实验必要性；标准目标叠加本身不足|
|物理控制|SRF 与 PSF 残差|既有 model-guided/回投影恢复|否|
|经验现象|同等可靠性下仍有不同光谱收益|预期普通集成互补现象|可能形成窄实证贡献，但必须跨场景/扰动验证|

重点对照应该共用 **完全相同的 B/F checkpoint**。若一个方法额外改 experts/alignment/operator，而另一个只改门控，无法隔离收益标签的作用。EnsIR LUT 原算法按值分桶；迁移到 31 波段时需报告共享权重还是 band-wise 权重。原文为 RGB 恢复，HSI 的全谱一致性不是其已证结论；沿用逐波段权重可能影响谱形，必须同时看 SAM。

## 决策关键查询记录

下面保留此次有界补检中影响判断的原查询。其余宽泛查询噪声较多，不作为“无先例”的证据。

|原查询|要否定的主张|结果|
|---|---|---|
|`"super-resolution" "oracle" "gating" experts`|oracle 教师+专家路由首次|引向后训练恢复集成|
|`"image restoration" "error" "expert selection" 2025 2024`|误差指导选择首次|恢复集成/MoE 线索|
|`EnsIR An Ensemble Algorithm for Image Restoration via Gaussian Mixture Models NeurIPS 2024`|真实误差参考集加权首次|官方主会+作者全文，强重叠|
|`"RefESR" ensemble super-resolution`|观察一致性+参考集权重首次|历史 RefESR 线索；此补检未取得 primary 原全文，不用于独立机制判断|
|`"image restoration" "selective" "ensemble" CVPR 2025 NeurIPS 2025`|选择性专家恢复首次|MoCE-IR 与 EnsIR|
|`"regression" "learning to defer" ICLR 2025 NeurIPS`|监督真实收益选择是新原则|PMLR 多专家回归 deferral；保留历史强先例|
|`"super-resolution" "confidence" "ground-truth" "selection" CVPR2025 ICCV2025`|GT confidence 导向处理首次|PatchScaler 官方主会和补充|
|`"CoPGNet" "github"`|CoPGNet 未知训练权重是否覆盖|未取得源码；缺口保持|
|`"Exploring structured uncertainty for external prior-guided hyperspectral image fusion" pdf`|先验收益监督 exact overlap|作者机构摘要；全文未得|
|`"Wavelet-Decoupled Complementarity-Guided Expert Network" hyperspectral`|同任务空间光谱专家首次|WCGE-Net，Xplore 官方摘要|
|`"WCGE-Net" arxiv github`|门控/收益标签细节|未取得全文/源码；不能排除 overlap|
|`"Uncertainty-Driven Generative Prior Learning" hyperspectral TIP 2026`|uncertainty matching+先验调制首次|TIP 正式论文线索；全文待取|

没有完成引用追踪召回审计、跨数据库去重筛选或完整专家收益文献覆盖，因此不能把检索未命中包装成已证明新颖。

## 下一轮可能改变判断的证据

先确认高几何可靠性、低测量残差条件下是否有规模可预测的收益异质性；比较同专家下的 EnsIR、ordinary gate、cost-weighted endpoint、oracle-weight teacher 与收益辅助 gate。只有辅助监督有跨 seed/场景的独立增量，才值得补齐上述全文后重审窄方法定位。

若 CoPGNet/WCGE-Net 出现功能等价差额教师，或普通成本敏感/集成对照同效，则架构定位降到 1–2；如果发现必要机制与真实数据的可重复失败模式，则另行查新，不能提前给当前组合打 4–5。既有 I13 没有看到训练结果后筛选，本次也没有训练结果；所有建议仍属探索。
