# 文献矩阵

截至 2026-10-10；27 项选题相关记录。主会/顶刊出版身份、监督类型、阅读程度分开记录。“代码确认”是静态检查，不代表成功复现。

- A：已读核心方法与实验，并目视核验关键图表。
- B：部分全文或作者训练代码已核验。
- C：正式元数据/作者摘要层，完整训练或实验细节仍待核验。

| ID | 工作与出版 | 任务/监督 | 证据 | 代码与全文 | 对选题的约束 |
|---|---|---|---|---|---|
| emrdiff | [EMR-Diff: Edge-aware Multimodal Residual Diffusion Model for Hyperspectral Image Super-resolution](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_EMR-Diff_Edge-aware_Multimodal_Residual_Diffusion_Model_for_Hyperspectral_Image_Super-resolution_CVPR_2026_paper.html)<br>CVPR 2026；main | LR-HSI + HR-MSI<br>确认：HR-HSI、多尺度 L1 | A | [作者代码](https://github.com/luocz55/EMR-Diff)；[本地 PDF](sources/emrdiff/source.pdf) | 少步扩散、边缘噪声、多模态残差已覆盖 |
| uafl | [Enhancing Unregistered Hyperspectral Image Super-Resolution via Unmixing-based Abundance Fusion Learning](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)<br>CVPR 2026；main | 未配准 LR-HSI + RGB<br>确认：HR-HSI L1 | A | [作者代码](https://github.com/yingkai-zhang/UAFL)；[本地 PDF](sources/uafl/source.pdf) | SVD 残差系数、可变形对齐、动态门控已覆盖 |
| srlf | [A Selective Re-learning Mechanism for Hyperspectral Fusion Imaging](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html)<br>CVPR 2025；main | LR-HSI + MSI<br>确认：合成 HR-HSI L1；Gaofen5 另用再退化监督 | A | [作者代码](https://github.com/YuanyeLiu/SRLF-Net)；[本地 PDF](sources/srlf/source.pdf) | 观测重投影错误选点、Mamba/Transformer 已覆盖 |
| hsc | [Heterospectral Structure Compensation Sampling for Hyperspectral Fusion Computational Imaging](https://ieeexplore.ieee.org/document/11276863)<br>TIP 2025；在线 2025-12-03；34:7930–7942 | LR-HSI + MSI<br>代码确认：HR-HSI L1 | B | [作者代码](https://github.com/1318133/HSC-Sampling)；本次未取得已核验全文 | MSI 谱覆盖不足与跨波段结构补偿已有直接研究 |
| bhsr | [Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model](https://doi.org/10.1109/TIP.2026.3714832)<br>TIP 2026；35:8338–8353；不采用索引缺省 1 月 1 日为首发日 | LR-HSI + 模糊 HR-MSI<br>代码确认：HR-HSI L1 | B | [作者代码](https://github.com/Dou0405/BHSR-Net)；本次未取得已核验全文 | MSI 模糊、双退化、半光滑 Newton 展开已覆盖 |
| irarf | [IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images](https://doi.org/10.1109/TIP.2025.3551531)<br>TIP 2025；34:1934–1949 | 未配准、任意尺度 LR-HSI + MSI<br>代码确认：HR-HSI L1 | B | [作者代码](https://github.com/Jiahuiqu/IR-ArF)；本次未取得已核验全文 | 任意尺度与未配准不能单独作为新问题 |
| ssc | [Unaligned RGB Guided Hyperspectral Image Super-Resolution with Spatial-Spectral Concordance](https://doi.org/10.1007/s11263-025-02466-8)<br>IJCV 2025；在线 2025-06-17；133(9):6590–6610 | 未配准 LR-HSI + RGB<br>作者仓库支持监督框架；本次未核验完整训练损失 | C | [作者代码](https://github.com/yingkai-zhang/SSC-HSR)；本次未取得已核验全文 | 对齐与融合交互、流与可变形聚合已有先例 |
| co2if | [CO²IF: Language-Bridging Hyperspectral-Multispectral Image Fusion with Coordinated and Cross-modal Optimal Transport](https://ojs.aaai.org/index.php/AAAI/article/view/38261)<br>AAAI 2026；main；40(15):12654–12662 | LR-HSI + MSI + 文本<br>确认：HR-HSI L1 与辅助对比损失 | A | [作者代码](https://github.com/yzkLearning/CO2IF)；[本地 PDF](sources/co2if/source.pdf) | 文本语义 + Mamba + OT 已有直接融合工作 |
| cauwt | [Content-Adaptive Unfolding Wavelet Transformer for Hyperspectral Image Super-Resolution](https://doi.org/10.1109/TIP.2025.3636789)<br>TIP 2026；35:1923–1936；DOI 为 2025，不等于 2026 首发 | LR-HSI + MSI<br>HR-HSI 监督细节待全文确认 | C | 未核验可用作者实现；本次未取得已核验全文 | 内容自适应展开参数 + 小波 Transformer 已覆盖 |
| differentialdiff | [A Two-Stage Conditional Diffusion Model With Differential Attention for Hyperspectral and Multispectral Image Fusion](https://doi.org/10.1109/TIP.2026.3723234)<br>TIP 2026；DOI 已登记；IEEE SPS 2026 年 9 月页面 | LR-HSI + MSI<br>需全文确认配对训练与目标 | C | [作者代码](https://github.com/Ruijie2580/DifferentialDiff)；本次未取得已核验全文 | FiLM + 小波 + 差分注意力扩散已覆盖；仓库快照仅 README |
| vpg | [Uncertainty-Driven Generative Prior Learning for Sparse Model-Guided Hyperspectral Image Fusion](https://doi.org/10.1109/TIP.2026.3689415)<br>TIP 2026；35:5224–5238 | LR-HSI + MSI；预训练 HR-HSI 先验<br>洁净 HR-HSI 先验学习明确；融合配对监督待全文确认 | C | 未核验可用作者实现；本次未取得已核验全文 | 不确定性 + 生成先验 + 稀疏展开不能泛泛声称新颖 |
| pino | [Physics-informed Neural Operator for Pansharpening](https://proceedings.neurips.cc/paper_files/paper/2025/hash/89af8e4eb696738f2c9e589522968a09-Abstract-Conference.html)<br>NeurIPS 2025；main；含 CAVE MHIF 实验 | PAN/MS 主任务；HSI/MSI 扩展<br>有参考监督；具体训练细节仅部分核对 | B | 未核验可用作者实现；[本地 PDF](sources/pino/source.pdf) | 连续辐射场与 SRF 积分、迁移 HSI 融合已有先例 |
| nud | [Building Non-Uniform Degradation Model for Position-Aware Hyperspectral Image Fusion](https://doi.org/10.1109/TPAMI.2025.3604234)<br>TPAMI 2025；47(12):11464–11482 | 空间非均匀 LR-HSI + MSI<br>公开融合脚本：观测再退化自监督，不使用原 HR-HSI 真值损失 | B | [作者代码](https://github.com/jeline0110/NUD_PAF)；本次未取得已核验全文 | 物理与真实数据挑战基线，不能标成普通配对有监督 |
| dses | [DSES-Diff: A Dynamic-Spectral and Efficient-Spatial Diffusion-Based Foundation Model for Scalable Hyperspectral and Multispectral Image Fusion](https://doi.org/10.1109/TIP.2026.3736733)<br>TIP 2026；Early Access 元数据：2026-10-01 | LR-HSI + MSI；分别预训练谱/空间先验<br>大量非配对谱/空间数据先验学习；不是常规配对监督训练 | C | 未核验可用作者实现；本次未取得已核验全文 | 最新边界：可变波段谱 DiT + 稀疏空间 MoE；全文尚不可得 |
| mossfuse | [Unsupervised Hyperspectral Image Super-Resolution via Self-Supervised Modality Decoupling](https://doi.org/10.1007/s11263-026-02757-8)<br>IJCV 2026；在线 2026-03-06；预印本早于 2026 | LR-HSI + MSI<br>自监督/无监督 | C | [作者代码](https://github.com/dusongcheng/MossFuse)；本次未取得已核验全文 | 共享与互补解耦、低参数融合已有先例 |
| clorf | [Hyperspectral and Multispectral Image Fusion with Arbitrary Resolution Through Self-Supervised Representations](https://doi.org/10.1007/s11263-025-02540-1)<br>IJCV 2025；在线 2025-08-03 | 任意尺度 LR-HSI + MSI<br>自监督 | C | 未核验可用作者实现；本次未取得已核验全文 | 连续隐式表示及任意尺度已有无监督强邻居 |
| args | [Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.html)<br>CVPR 2025；main；17862–17871 | LR-HSI + MSI<br>自学习；与配对监督分开 | C | 未核验可用作者实现；本次未取得已核验全文 | 子空间残差引导扩散不能泛化为未研究 |
| cdps | [Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion](https://doi.org/10.1109/TIP.2025.3647207)<br>TIP 2026；35:69–84；DOI 2025.3647207 | LR-HSI + MSI<br>无监督融合；存在先验数据成本 | C | 未核验可用作者实现；本次未取得已核验全文 | 联合后验采样与耦合先验已有直接工作 |
| shotun | [Self-Expressive High-Order Tensor Unrolling Network for Unsupervised Hyperspectral and Multispectral Image Fusion](https://doi.org/10.1109/TIP.2026.3695389)<br>TIP 2026；35:5714–5729 | LR-HSI + MSI<br>无监督 | C | 未核验可用作者实现；本次未取得已核验全文 | 高阶张量展开邻近基线 |
| bfctn | [Bayesian Fully-Connected Tensor Network for Hyperspectral-Multispectral Image Fusion](https://doi.org/10.1109/TIP.2026.3696124)<br>TIP 2026；DOI 已登记 | LR-HSI + MSI<br>VB/EM 模型推断；不列为 HR-HSI 配对神经监督 | C | [作者代码](https://github.com/LinsongShan/BFCTN)；本次未取得已核验全文 | 传统概率建模强邻居；避免将观测似然称作有监督 |
| onehsi | [OneHSI: A Unified Hyperspectral Foundation Model with Physical Consistency](https://eccv.ecva.net/virtual/2026/poster/4477)<br>ECCV 2026；官方主会 poster #4477；论文 PDF 已公开 | 单 HSI 表征；六种分析任务<br>MAE 预训练 + 下游监督；不是 HSI/MSI 辐射重建实验 | B | 未核验可用作者实现；[本地 PDF](sources/onehsi_pdf/source.pdf) | 波长坐标、物理尺度、跨域表示已覆盖，但不等于重建保真 |
| lgd | [LGD-Net: Leader-Guided Cross-Modal Dynamics for Hyperspectral and Panchromatic Image Fusion](https://eccv.ecva.net/virtual/2026/poster/3288)<br>ECCV 2026；官方主会 poster #3288；论文 PDF 已公开 | LR-HSI + PAN<br>PAN 融合邻近任务；本次监督损失未完整核验 | B | 未核验可用作者实现；[本地 PDF](sources/lgdnet_pdf/source.pdf) | 图传播与连续模态动力学属于 PAN 邻居，不能当直接 MSI SOTA |
| um | [Uncertainty-guided Memory Network for Hyperspectral Pansharpening](https://ojs.aaai.org/index.php/AAAI/article/view/37988)<br>AAAI 2026；main；article 37988 | LR-HSI + PAN<br>邻近 PAN 任务；监督目标待全文核验 | C | 未核验可用作者实现；本次未取得已核验全文 | 普通不确定性门控/记忆不能单独作为融合创新 |
| gew | [GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution](https://ojs.aaai.org/index.php/AAAI/article/view/37978)<br>AAAI 2026；main；40(12):10109–10117 | 单幅 HSI ×4 SR<br>单幅任务；不是双输入融合 | B | 未核验可用作者实现；[本地 PDF](sources/gewdiff/source.pdf) | 几何与小波可借鉴；不进入直接融合排名 |
| carl | [CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis](https://openreview.net/pdf/06c03489864236829ee6008cde9cee87d1b41a15.pdf)<br>ICLR 2026；正式版本标题页 Published as a conference paper；全文下载 403 | 谱图像分析/表征<br>邻近表征学习；不是 HR-HSI 重建 | C | 未核验可用作者实现；本次未取得已核验全文 | 可变通道和波长感知已有工作 |
| lessvit | [LESSViT: Robust Hyperspectral Representation Learning under Spectral Configuration Shift](https://neurips.cc/Downloads/2026)<br>NeurIPS 2026；官方 Downloads 列为 Poster #154948；会议未举行；arXiv 2605.18541 | 谱配置改变下的 HSI 表征<br>MAE + 下游分析；非双输入融合 | C | 未核验可用作者实现；本次未取得已核验全文 | 最新接收目录已检查；低秩空间/光谱注意力 + 波长感知不是空白 |
| feinfn | [Fourier-enhanced Implicit Neural Fusion Network for Multispectral and Hyperspectral Image Fusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/74088c68894b99383c12399c9c637be9-Abstract-Conference.html)<br>NeurIPS 2024；main；历史新颖性锚点 | LR-HSI + MSI<br>本次只复核正式摘要，训练细节不做新断言 | C | [作者代码](https://github.com/294coder/Efficient-MIF)；本次未取得已核验全文 | Fourier、隐式函数、Gabor 解码的直接前作 |

## 候选与排除的界线

A/B/C 是阅读证据等级，不是论文质量打分。C 级新论文也可能使选题失去新颖性，但不能仅凭摘要断言其完整能力。

NUD/PAF 的论文讨论学习式融合与训练退化，其公开 main_fus_syn.py/main_fus_real.py 的融合损失分别对 LR-HSI、再退化数据和 MSI 构造；原 HR-HSI 用于评价。故按本次代码证据，将它放在观测/再退化监督邻近组，而非普通 HR-HSI 配对监督组。

DSES-Diff 的非配对先验学习不能称为完全没有训练数据；OneHSI/LESSViT/CARL 的表征分析能力不能等同于辐射值融合。PINO 已包含 CAVE MHIF 扩展，不能只根据标题排除。

另行记录但不作为“已录用顶会”核心工作：

- [Lightweight Interpretable RGB-Guided Hyperspectral Super-Resolution under Real Cross-resolution Misalignment，arXiv:2609.01060](https://arxiv.org/abs/2609.01060)：作者预印本检索证据显示已有 HSI-preserving 与置信门控设计；本次未确认指定顶会/顶刊出版。可以反驳过宽新颖性声明，不能纳入顶会 SOTA 名单。
- [FSRVL 的 ICLR 投稿 PDF](https://openreview.net/pdf?id=l7frq7NyFX)：可检索版本标为 Under review，未确认最终主会接收。任务为无监督融合，不能写成 ICLR 2026 已接收的有监督融合。
- NeurIPS 2026 官方目录还出现“Manifold Prior Guided Deep Unfolding for Hyperspectral Image Reconstruction”等标题；在正文与输入任务未核验前仅保留检索线索，不推定是 HSI/MSI 融合。
- HSIGene、HSI restoration、RGB-only 光谱重建、HSI/LiDAR 分类等在检索中出现，但不为数量而加入直接融合组。

补查的邻近资源：[DANet/MixShuffle/HAL，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wen_Regulating_Rather_than_Constraining_Adaptive_Guidance_for_Complex_Spectral_Reconstruction_CVPR_2026_paper.html)，主会 pp.34832–34842。正式摘要证实已有数据混合与样本/通道/像素梯度重分配，输入为 PAN/MS；已加入选题的廉价强对照，不推定为直接 HSI/MSI 融合。

未进入此矩阵的出版物并不等于质量低或与领域无关；这里只遵守本次选题与出版范围。

