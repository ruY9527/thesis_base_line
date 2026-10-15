# 有监督高光谱—多光谱图像融合：最新证据与选题建议

检索截止：**2026-10-10**。任务：低空间分辨率 HSI + 高空间分辨率 MSI/RGB → 高空间分辨率 HSI；训练可使用 HR-HSI 参考真值，推理只能使用观测与可获得的传感器信息。

本次采用批判性证据图谱，核心时间窗口为 2025–2026，少量 2024 前作只用于排查新颖性。核心出版范围按你的要求限定在 TIP、TPAMI、IJCV、CVPR、ICCV、ECCV、NeurIPS、AAAI、ICLR；会议按正式主会或官方接收目录确认，不能把 workshop、投稿版本或标题中的年份当作主会录用。文献矩阵保留 **27 项选题相关记录**，并不代表检索只得到 27 篇，也不代表这些都是直接有监督融合论文。

**我的建议：先围绕“传感器响应存在偏差时，如何控制 MSI 细节注入带来的光谱误差”做诊断实验，再决定是否扩展成方法。** 以“可归因的融合增益”为辅助诊断，比立即搭一个更复杂的网络更有价值。跨传感器连续谱重建可以作为第二条路线，但不要同时开展三个大方向。

这是一项待验证的研究建议，不是已经证实的新颖贡献。本次没有训练模型、没有复现论文数值、没有改动你的 baseline；已完成的是文献发现、证据分层、重点方法核对、选题调整和实施计划。

## 1. 最新文献怎样改变选题

2026 年的进展已经把几个容易想到的创新点挤得很满：

- **扩散效率和细节增强**：CVPR 2026 EMR-Diff 已结合多模态残差、边缘相关噪声、双路融合 UNet 和多尺度监督；TIP 2026 DifferentialDiff 又涉及条件调制、小波和差分注意力。因此，“少步扩散 + 小波/边缘模块”很难单独支撑新论文。[EMR-Diff 正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_EMR-Diff_Edge-aware_Multimodal_Residual_Diffusion_Model_for_Hyperspectral_Image_Super-resolution_CVPR_2026_paper.html)，[DifferentialDiff DOI](https://doi.org/10.1109/TIP.2026.3723234)
- **退化模型更贴近实际**：TIP 2026 BHSR-Net 已显式考虑 MSI 模糊；TPAMI 2025 NUD/PAF 已研究位置相关的非均匀退化。仅加入一个 MSI blur 分支，或随机使用不同 Gaussian kernel，研究距离不足。[BHSR-Net 作者代码](https://github.com/Dou0405/BHSR-Net)，[NUD/PAF 作者代码](https://github.com/jeline0110/NUD_PAF)
- **未配准融合**：IJCV 2025 SSC-HSR、TIP 2025 IR&ArF、CVPR 2026 UAFL 已覆盖流、可变形聚合、任意尺度、SVD 系数学习和调制融合。“对齐 + 门控 + 子空间”已是明确的前作组合。[SSC-HSR](https://doi.org/10.1007/s11263-025-02466-8)，[IR&ArF](https://doi.org/10.1109/TIP.2025.3551531)，[UAFL](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)
- **MSI 无法提供全部窄带细节**：TIP 2025 HSC-Sampling 已提出异谱结构补偿，直接针对 MSI 光谱分辨率不足，利用 HSI 内部跨波段结构。不能把“MSI 高频并非每个 HSI 波段都适用”写成首次发现。[正式记录](https://ieeexplore.ieee.org/document/11276863)
- **通用谱表示与生成先验**：TIP DSES-Diff 的 Early Access 元数据日期为 **2026-10-01**，已提出可变波段的谱扩散与空间专家先验；ECCV 2026 OneHSI、ICLR 2026 CARL、NeurIPS 2026 官方目录中的 LESSViT，也覆盖波长感知和可变通道表示。增加 wavelength embedding 或 band mask 本身不足以构成创新。[DSES-Diff](https://doi.org/10.1109/TIP.2026.3736733)，[OneHSI](https://eccv.ecva.net/virtual/2026/poster/4477)，[NeurIPS 2026 官方目录](https://neurips.cc/Downloads/2026)
- **语义先验**：AAAI 2026 CO²IF 已引入文本、跨模态最优传输和协同 Mamba。一般的 CLIP/文本辅助融合也不是空白。[CO²IF 正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/38261)

这些进展仍未自动回答一个更严格的问题：**当提供给网络的 SRF/PSF 信息不准确时，哪些空间—光谱修正可以可靠使用，哪些应该减弱？** 我建议围绕这一问题建立实验，而不预先假定一个 gate 就能解决它。

## 2. 优先阅读与复现的直接相关工作

证据 A：核心方法与实验已读，并目视核验关键图表；B：部分全文或作者训练代码已核验；C：正式元数据/作者摘要证据。A 也不表示数值已复现。完整标题、任务边界与代码状态见 [文献矩阵](literature-matrix.md)。

| 工作 | 出版 | 对当前研究的作用 | 监督与证据状态 |
|---|---|---|---|
| SRLF-Net | CVPR 2025 | 常规对齐融合强基线；重投影选点最邻近 | HR-HSI L1；A；有官方代码 |
| HSC-Sampling/MRC | TIP 2025 | 跨波段细节补偿最邻近 | 官方训练代码确认 GT L1；B；全文未取得 |
| EMR-Diff | CVPR 2026 | 近期监督扩散比较对象 | 多尺度 GT L1；A；有官方代码 |
| BHSR-Net | TIP 2026 | 模糊 MSI 与双退化首要比较对象 | 官方代码确认 GT L1；B；全文未取得 |
| UAFL | CVPR 2026 | 未配准与低秩残差路线首要比较对象 | GT L1；A；有官方代码 |
| IR&ArF | TIP 2025 | 未配准 + 任意尺度前作 | 官方代码确认 GT L1；B |
| SSC-HSR | IJCV 2025 | 对齐/融合交互前作 | 作者仓库与正式元数据；C；损失待完整核验 |
| CO²IF | AAAI 2026 | 文本语义路线最邻近 | GT L1 + 辅助对比损失；A |
| CAUWT | TIP 2026 卷年 | 内容自适应展开 + 小波前作 | C；监督细节待全文；DOI 年份为 2025 |
| DifferentialDiff | TIP 2026 | 最新差分注意力条件扩散 | C；不能仅凭摘要认定完整训练条件 |
| VPG-Net | TIP 2026 | 不确定性生成先验 + 稀疏展开 | HR-HSI 先验训练明确；融合配对监督待全文；C |

PINO（NeurIPS 2025）主任务是 pansharpening，但正文已经包含 **CAVE 的 HSI/MSI 融合实验**，不能把“把连续物理神经算子用于高光谱融合”当作首次迁移。FeINFN（NeurIPS 2024）也是直接融合的频域隐式表示前作。[PINO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/89af8e4eb696738f2c9e589522968a09-Abstract-Conference.html)，[FeINFN](https://proceedings.neurips.cc/paper_files/paper/2024/hash/74088c68894b99383c12399c9c637be9-Abstract-Conference.html)

**我不会把各篇论文里的最高 PSNR 拼成统一排行榜。** 数据划分、退化、归一化、谱响应和 SAM 单位不同，跨论文数字直接比较会误导。复现时应同时保留作者原设置和统一设置。

## 3. 四篇重点核对带来的具体判断

### SRLF-Net：选点机制已有，收益需要分解

其流程是多尺度 Mamba 初始融合 → 按已知 R 重投影到 MSI，以 SSIM 挑出空间问题位置 → 重投影到 LR-HSI，以 SAM 挑出谱问题位置 → 重学习与再融合。故“根据观测误差选择位置”“Mamba 与 spectral Transformer 配合”都不能作为我们的主要创新。

2026-10-11 的正式精读补充确认：Table 4 也将已知退化替换为可学习步长卷积和无偏置线性层，达到 49.3412 dB，仅比固定先验版本低 0.0469 dB。因此不能把“学习未知退化”本身当作空白。需要额外比较这个代理版本，并区分它与“真实成像不变、提供的标定信息错误”的实验。

论文 Table 1 中 Harvard 的 PSNR 为 47.8954，Mog-DCN 为 47.8910，差值只有 0.0044 dB；Table 4 中移除 selection 后为 49.3135，完整模型为 49.3881，差值 0.0746 dB。它的多尺度建模与整网设计很重要，不能把全部提升归因于选点。补充材料还说明 FLOPs 省略部分 attention 计算，复现时需要实际延迟和完整计算口径。[正文](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.pdf)，[补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liu_A_Selective_Re-learning_CVPR_2025_supplemental.pdf)

研究启示是：重投影误差能帮助定位观测不一致，但**观测一致不能证明 HR-HSI 真值正确**。我们应比较真值误差、重投影误差和 MSI 相对增益之间的关系。

### EMR-Diff：需要比较完整系统，不能只借一个模块

它训练的状态包含 HR-HSI 与其前三个波段构成的 pseudo-MSI，观测端则拼接上采样 LR-HSI 与 HR-MSI。边缘噪声由 MSI 提取；反向网络同时接收两种观测，使用多尺度 L1。

Harvard 消融中，普通噪声 48.36 dB、边缘噪声 49.28 dB；这是作者的同论文结果，并未在本地复现。论文正文 kappa=1，公开 5-step 配置为 2.0，且退化设置与本地 baseline 不同，复现必须记录版本与配置。测试脚本虽然构造了 GT 的 pseudo-MSI 变量，但该变量未进入所核对的推理调用；不能据此指控真值泄漏。[论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_EMR-Diff_Edge-aware_Multimodal_Residual_Diffusion_Model_for_Hyperspectral_Image_Super-resolution_CVPR_2026_paper.html)，[代码](https://github.com/luocz55/EMR-Diff)

待验证的缺口：从 MSI 平均图取边缘，再加强边缘噪声，并不等于证明每个窄谱带都应增强同一条边。对于不可见于 MSI 的窄带结构，需要单独构造反例。

### UAFL：低秩残差很有用，但不是物理端元证明

其方法从上采样 LR-HSI 的 SVD 中取 K=3 个左奇异向量，网络在系数域学习残差，并经可变形聚合、空间/通道调制利用未配准 RGB。输出为上采样 HSI 加基底乘残差，因此不能直接说“整个输出被限制为 rank 3”。SVD 分量也不自动满足物理端元与丰度的非负、和为一约束。

REAL 数据有实际相机不配准，但方法仍使用 Gaussian 和下采样生成 LR-HSI。它支持未配准场景的性能结论，不能单独支持真实低分辨率 PSF、噪声和响应偏差的全面结论。Table 2 在 REAL ×8 上为 37.23 dB，对照 SSC-HSR 为 36.19 dB，属于作者同设置比较。[论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)

### CO²IF：语言路线需承担额外证据成本

其跨模态 OT 对齐文本与谱特征，CoMamba 结合 MSI、HSI 与文本，直接使用 HR-HSI L1。论文 CAVE 划分为 22/10，与常见 20/12 不同；不能直接对比另一篇的 CAVE 数字。

若走语言路线，必须检验错误文本、通用文本、只由推理可得 MSI 生成的文本，以及去掉人工修订后的效果。文本能表达类别先验，但不证明新增窄带细节受到测量支持。本次不推荐把它作为主线。[论文](https://ojs.aaai.org/index.php/AAAI/article/view/38261)

## 4. 必须划清的邻近任务

| 邻近工作 | 为什么重要 | 为什么不能混入“直接有监督 MSI 融合 SOTA” |
|---|---|---|
| DSES-Diff / CDPS | 谱/空间先验与扩散后验 | 先验数据及融合监督机制不同 |
| MossFuse / CLoRF / ARGS-Diff / SHOTUN | 解耦、任意尺度、自学习 | 不使用常规 HR-HSI 配对监督 |
| NUD/PAF | 非均匀退化与真实数据 | 本次核对的公开融合脚本用观测及再退化损失 |
| OneHSI / CARL / LESSViT | 跨传感器谱表示 | 分析任务的鲁棒性不等于辐射值重建保真 |
| LGD-Net / UMNet | 模态交互与不确定性 | 输入是 PAN，谱信息约束与多通道 MSI 不同 |
| GEWDiff | 几何与小波生成机制 | 正文任务是单幅 HSI ×4 超分辨率 |
| RGB→HSI 谱重建 | 谱歧义与先验偏置 | 没有 LR-HSI 观测，不能同等讨论可辨识性 |

NeurIPS 2026 截至本次检索已有官方 Poster 目录，不能机械地只检索到 NeurIPS 2025。本次已补查其中的 hyperspectral/multispectral 标题，发现 LESSViT 等邻近任务；仅有接收目录和预印本证据时，不把它写成已经完整审阅的 proceedings 论文。[官方目录](https://neurips.cc/Downloads/2026)，[LESSViT 作者预印本](https://arxiv.org/abs/2605.18541)

## 5. 建议保留的研究方向及排序

详细机制、最强反对意见、决定性实验和停止条件见 [选题与实验设计](directions-and-experiments.md)。

| 排序 | 研究问题 | 与现有工作的研究距离 | 最先验证的事情 |
|---|---|---|---|
| **主线** | SRF/PSF 有误时，如何控制 MSI 细节注入的光谱风险？ | 需要超越 BHSR 的双退化、NUD 的位置相关退化和普通数据一致性 | 固定观测，改变提供给模型的算子信息；检验错误校准导致的额外损失 |
| **辅助诊断/可能独立成文** | MSI 相比 HSI-only，在哪些空间—光谱模式上有实际收益？ | HSC、SRLF、VPG、UMNet 都是强近邻；单个 gate 创新不足 | 是否存在稳定的“MSI 边缘强但 HSI 某些谱带收益低/为负”的可复现群体 |
| **第二路线** | 用完整 SRF 形状支持未见传感器的监督融合 | PINO、FeINFN、DSES、OneHSI、CARL、LESSViT 构成前作 | 同中心波长、同宽度、不同响应形状时，完整 SRF 是否比坐标摘要更有用？ |

主线与辅助诊断可以组成同一个研究：先量化错误响应下的负迁移，再做有监督的修正控制。**第一阶段固定空间配准、固定波段数与尺度，只研究 SRF 误差。** 只有出现稳定效果后，再加入 MSI 模糊/PSF 误差；同时引入任意尺度、未知响应、非刚性形变、扩散先验会让归因变得困难。

不推荐的独立选题：把 ChannelAttention 换成 spectral attention；Mamba/Transformer 替换；小波损失；Gaussian 随机化；普通残差 gate；通用 CLIP 辅助；“首个 wavelength-aware 融合”；没有独立校准证据的“完全盲物理融合”。这些可作工程组件，不能靠名称组成主要贡献。

## 6. 对本地 baseline 的判断

它是双输入残差 CNN：LR-HSI 编码后上采样，与 MSI 编码特征拼接，经过残差块与潜特征通道注意力，输出对上采样 HSI 的残差。它适合机制诊断，暂时不足以独立证明超过近期强方法。

必须先处理的实验问题：

1. 当前用 cave_test 指标挑选最佳 checkpoint。独立测试集应只用于最终报告，模型选择使用固定验证集；若过去已反复查看测试结果，需要声明并重新冻结协议。
2. 数据文件来自未排序的目录枚举，缺乏固定 scene manifest。数据集先按 scene 划分，再裁 patch；空间单图数据需有隔离区域及 PSF halo。
3. PSNR 当前先对每个空间位置取对数、再平均，等价于对空间误差取几何平均，不是常见逐波段或整 cube MSE 的口径。
4. 固定 3×31 SRF 与单 Gaussian 退化，无法支撑跨传感器或响应鲁棒性主张。
5. 64 维潜特征注意力不是 31 个物理波段的可信度，不能将其可视化解释为“传感器可观测性”。
6. 每个 HR-HSI 场景按其最大值归一化可以是明确的合成协议，但不能默认为真实推理中已知的辐射校准方式。

逐文件调整、优先级、验收与回退策略已写在 [baseline 修改计划](baseline-modification-plan.md)。**这里只给计划；现有代码未改。**

## 7. 怎样才有足够强的实验论证

至少需要三层证据，且每层标明测量来源：

- 合成可控实验：以 HR-HSI 为参考，已知真算子生成观测，训练/验证/测试 scene 严格隔离；SRF 误差实验中，观测生成使用真 R，提供给网络的是另一个 R，二者不能混用。
- 跨数据集与跨传感器实验：训练域和完整传感器家族留出，不能只随机抽波段；高光谱范围相差较大时，把域改变和响应改变分开报告。
- 真实观测证据：真实未配准配对、真实传感器 LR-HSI/MSI、真实校准/噪声分别标明。没有 HR-HSI 真值时，观测一致性、降尺度指标和可视化只能支持有限结论，不能当作 native HR-HSI 的真值精度证明。

报告 PSNR、SAM（明确度或弧度）、ERGAS、逐波段误差、场景均值与不确定性；增加真实延迟、峰值显存、参数量。比较同参数/同训练预算的增强 CNN、普通 gate、SRLF、HSC、BHSR 和一个监督扩散方法。校准误差的失败曲线、最差场景与干净场景收益损失比只展示最佳平均值更关键。

统计单位是 scene，不是高度相关的 patch。预先选定主终点，对多个训练种子和按场景配对的指标差值报告区间；不把同一图上的几十万像素当作独立样本。

## 8. 尚未闭合的证据

本次对 EMR-Diff、SRLF、UAFL、CO²IF 的核心方法/实验和关键图表进行了核对，对 BHSR、HSC、IR&ArF、NUD/PAF 的训练代码做了静态核验。**TIP 新论文中仍有全文获取障碍**，尤其 DSES-Diff、DifferentialDiff、VPG、CAUWT；因此其具体实验覆盖范围仍是新颖性审查的保留项，而不是已确认的空白。

检索使用官方会议目录、出版/作者页面、Crossref、OpenAlex 和 Europe PMC。索引可能滞后，关键词也可能漏掉更换任务名称的论文；本报告不声称穷尽全部文献或给出已证实的“首创”结论。[检索与证据审计](search-and-evidence-audit.md)

用户于 2026-10-11 确认中文和当前项目 research 保存设置后，已完成 [EMR-Diff 正式精读](notes/高光谱图像融合/EMR-Diff/EMR-Diff.md) 与 [SRLF-Net 正式精读](notes/高光谱图像融合/SRLF-Net/SRLF-Net.md)。两篇均通过 Grounding Lint、Final Note Lint、模型质量审阅和可读性审阅，并由正式保存脚本写入工作区。原有证据卡仍保留，用于调研横向比较。广泛检索截止日期仍为 2026-10-10；本轮补充的代码和书目核对在 [笔记索引](notes/README.md) 及 notes-verification.json 中单独记录。

