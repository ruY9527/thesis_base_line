# 核验来源与阅读顺序

建议先读R01/R02界定文本与区域先验最近邻，再读R04理解HSI进入VLM的证据接口，之后按是否愿意新增训练选择R05/R07或R11/R12/R13。若研究CoT，先读R14/R15/R23；若研究扩散，补R16/R17/R18/R24。R19/R20是项目记录，不能当正式已复现论文。

每条记录注明核验层级；公开论文或仓库不代表当前环境已运行。完整结构化记录见sources.json。

### R01. CO²IF: Language-Bridging Hyperspectral-Multispectral Image Fusion with Coordinated and Cross-modal Optimal Transport

- 年份/状态：2026；AAAI 2026。
- 主来源：[原始记录](https://ojs.aaai.org/index.php/AAAI/article/view/38261)；[全文或摘要核验入口](https://ojs.aaai.org/index.php/AAAI/article/download/38261/42223)；[作者仓库](https://github.com/yzkLearning/CO2IF)。
- 核验：full_text_targeted。
- 相关机制：直接HSI–MSI文本融合；CLIP adapter、跨模态最优传输和CoMamba，需要训练；COT不是思维链。
- 限制：当前可见仓库仅README/pipeline图片，未确认实现、权重或文本数据可用。

### R02. Exploring structured uncertainty for external prior-guided hyperspectral image fusion

- 年份/状态：2026；Information Fusion 129, 103997。
- 主来源：[原始记录](https://www.sciencedirect.com/science/article/pii/S1566253525010590)。
- 核验：publisher_abstract_highlights。
- 相关机制：直接HSI–MSI融合，SAM外部先验及结构不确定性；SAM+区域加权已有先例。
- 限制：出版社搜索可读摘要/highlights，直接打开失败；未获取全文或核验代码。

### R03. CLIPPan: Adapting CLIP as a Supervisor for Unsupervised Pansharpening

- 年份/状态：2026；AAAI 2026。
- 主来源：[原始记录](https://ojs.aaai.org/index.php/AAAI/article/view/37451)。
- 核验：publisher_abstract。
- 相关机制：文本化融合协议和CLIP监督；先做轻量适配，再训练融合网络。
- 限制：MSI–PAN任务；无GT不等于不训练。

### R04. Beyond RGB: Benchmarking and Enhancing MLLMs for Hyperspectral Image Understanding via Training-Free Reasoning Framework

- 年份/状态：2026；arXiv预印本，v2 2026-08-23。
- 主来源：[原始记录](https://arxiv.org/abs/2604.08884)；[全文或摘要核验入口](https://arxiv.org/html/2604.08884v2)。
- 核验：full_text_targeted。
- 相关机制：VSR²用RGB/PCA/定量文本报告暴露HSI证据，是理解任务而非HR-HSI重建。
- 限制：推理收益因模型和任务变化，不能直接迁移成融合PSNR收益。

### R05. HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model

- 年份/状态：2025；IEEE TPAMI 47(8), 6427–6444；初始预印本2024。
- 主来源：[原始记录](https://arxiv.org/abs/2406.11519)；[作者仓库](https://github.com/WHU-Sigma/HyperSIGMA)。
- 核验：author_abstract_official_repository。
- 相关机制：HSI空间/光谱预训练表征，官方含超分和去噪任务。
- 限制：下游接口、波段和波长需适配；非文本LLM，非直接无训练融合器。

### R06. SpectralGPT: Spectral Remote Sensing Foundation Model

- 年份/状态：2024；IEEE TPAMI 46(8), 5227–5244。
- 主来源：[原始记录](https://ieeexplore.ieee.org/document/10490262)。
- 核验：publisher_abstract。
- 相关机制：光谱遥感MAE类预训练，非对话语言模型。
- 限制：已报道下游任务不能作为当前融合效果证据。

### R07. HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery

- 年份/状态：2025；CVPR 2025。
- 主来源：[原始记录](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_HyperFree_A_Channel-adaptive_and_Tuning-free_Foundation_Model_for_Hyperspectral_Remote_CVPR_2025_paper.pdf)。
- 核验：official_paper_targeted。
- 相关机制：通道自适应HSI基础模型，可研究冻结结构先验。
- 限制：tuning-free有原任务边界，未证明直接HSI–MSI融合无需训练。

### R08. EigenSR: Eigenimage-Bridged Pre-Trained RGB Learners for Single Hyperspectral Image Super-Resolution

- 年份/状态：2025；AAAI 2025；预印本2024。
- 主来源：[原始记录](https://arxiv.org/abs/2409.04050)；[作者仓库](https://github.com/enter-i-username/EigenSR)。
- 核验：author_abstract_official_repository。
- 相关机制：本征图桥接RGB预训练SR，LoRA/head/tail适配及迭代光谱正则。
- 限制：单HSI超分；需训练，不能写严格training-free。

### R09. HSIGene: A Foundation Model for Hyperspectral Image Generation

- 年份/状态：2024/2025；arXiv初稿2024；本报告依据v2，不强行指定正式venue。
- 主来源：[原始记录](https://arxiv.org/abs/2409.12470)；[全文或摘要核验入口](https://arxiv.org/html/2409.12470v2)。
- 核验：full_text_targeted。
- 相关机制：多条件HSI生成包含文本，48波段400–1000nm配置；有训练与数据扩增流程。
- 限制：生成任务；不能把合成谱当真实观测或将未观测NIR当CAVE监督。

### R10. Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model

- 年份/状态：2025；CVPR 2025；ARGS-Diff。
- 主来源：[原始记录](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.pdf)。
- 核验：official_paper_existing_local_review。
- 相关机制：直接融合，残差引导子空间扩散。
- 限制：自学习扩散不等于直接复用冻结通用扩散。

### R11. Unsupervised Hyperspectral Pansharpening via Low-rank Diffusion Model

- 年份/状态：2024；Information Fusion 2024；PLRDiff。
- 主来源：[原始记录](https://github.com/earth-insights/PLRDiff)；[作者仓库](https://github.com/earth-insights/PLRDiff)。
- 核验：official_repository。
- 相关机制：LR-HSI估计低秩系数，复用预训练扩散。
- 限制：HSI–PAN；换成多通道MSI需改观测模型。

### R12. Diffusion Posterior Sampling for General Noisy Inverse Problems

- 年份/状态：2023；ICLR 2023；预印本2022。
- 主来源：[原始记录](https://arxiv.org/abs/2209.14687)；[作者仓库](https://github.com/DPS2022/diffusion-posterior-sampling)。
- 核验：author_paper_official_repository。
- 相关机制：预训练扩散与观测引导的通用逆问题机制。
- 限制：近似后验方法；RGB先验不能原封不动作用31波段。

### R13. Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model

- 年份/状态：2023；ICLR 2023；DDNM。
- 主来源：[原始记录](https://openreview.net/forum?id=mRieQgMtNTQ)；[作者仓库](https://github.com/wyhuai/DDNM)。
- 核验：conference_paper_official_repository。
- 相关机制：线性观测范围/零空间分工的零样本恢复。
- 限制：噪声、先验分布、RGB通道与融合联合算子均需重新考虑。

### R14. Q-Agent: Quality-Driven Chain-of-Thought Image Restoration Agent through Robust Multimodal Large Language Model

- 年份/状态：2025；arXiv预印本；不在本报告推断正式venue。
- 主来源：[原始记录](https://arxiv.org/abs/2504.07148)。
- 核验：author_abstract。
- 相关机制：CoT退化感知加客观IQA的恢复工具选择，MLLM做微调。
- 限制：自然图恢复；IQA不能保证高光谱真实，非严格免训练。

### R15. RIRF: Reasoning Image Restoration Framework

- 年份/状态：2026；arXiv预印本，2026-04-10。
- 主来源：[原始记录](https://arxiv.org/abs/2604.09511)。
- 核验：author_abstract。
- 相关机制：Reason and Restore；微调Qwen3-VL做结构化诊断并结合恢复训练/RL。
- 限制：不能把结构化CoT接恢复作为首次。

### R16. Dual Modality Prompted Diffusion Priors for Zero Shot Hyperspectral Pansharpening

- 年份/状态：2026；arXiv预印本，2026-08；DIDM。
- 主来源：[原始记录](https://arxiv.org/abs/2608.11748)；[全文或摘要核验入口](https://arxiv.org/html/2608.11748v2)。
- 核验：full_text_targeted_by_researcher。
- 相关机制：冻结扩散骨干，但逐输入对优化prompt/injection/reconstructor参数。
- 限制：zero-shot不等于无梯度、无参数更新；HSI–PAN，非直接MSI。

### R17. DSES-Diff: A Dynamic-Spectral and Efficient-Spatial Diffusion-Based Foundation Model for Scalable Hyperspectral and Multispectral Image Fusion

- 年份/状态：2026；IEEE TIP online ahead of print 2026-10-01，DOI 10.1109/TIP.2026.3736733。
- 主来源：[原始记录](https://doi.org/10.1109/TIP.2026.3736733)；[全文或摘要核验入口](https://pubmed.ncbi.nlm.nih.gov/42821724/)。
- 核验：author_abstract_index_metadata。
- 相关机制：直接融合foundation model，分别预训练光谱DiT与空间稀疏MoE，采用无配对大数据。
- 限制：仅摘要与正式索引元数据；全文、权重和训练适配细节未核验。

### R18. Modulated Diffusion with Spatial–Spectral Disentangled Guidance for Hyperspectral Image Super-Resolution

- 年份/状态：2026；Remote Sensing 18(10), 1582。
- 主来源：[原始记录](https://doi.org/10.3390/rs18101582)。
- 核验：publisher_abstract。
- 相关机制：需要训练的动态条件扩散骨干，其SSDG采样引导可不重训。
- 限制：training-free仅限定推理引导组件，非整条方法。

### R19. Lib-SRDM (repository)

- 年份/状态：date_unverified；仅官方公开项目记录，不推断论文venue。
- 主来源：[原始记录](https://github.com/Dmsw/Lib-SRDM)；[作者仓库](https://github.com/Dmsw/Lib-SRDM)。
- 核验：official_repository。
- 相关机制：RAM++/GroundingDINO/SAM-HQ构建语义光谱库，并训练HSI扩散；含RGB→HSI/relighting。
- 限制：库来自训练划分；不是已验证直接HSI–MSI融合，论文元数据未核验。

### R20. OFA-HSI (repository)

- 年份/状态：2026_claim_date_unverified；作者仓库标注投稿IEEE GRSM，未确认接收。
- 主来源：[原始记录](https://github.com/zhu-xlab/OFA-HSI)；[作者仓库](https://github.com/zhu-xlab/OFA-HSI)。
- 核验：official_repository_announcement。
- 相关机制：对齐HSI/MSI/文本等多模态数据与基准计划。
- 限制：页面资源均Coming soon、命令模板；不能作为已到手数据或权重。

### R21. SAMSA: Segment Anything Model Enhanced with Spectral Angles for Hyperspectral Interactive Medical Image Segmentation

- 年份/状态：2025；MICCAI 2025。
- 主来源：[原始记录](https://papers.miccai.org/miccai-2025/paper/1453_paper.pdf)；[作者仓库](https://github.com/CVRS-Hamlyn/SAMSA)。
- 核验：official_paper_abstract_official_repository。
- 相关机制：SAM与光谱角先验在HSI医疗分割有先例。
- 限制：分割任务；故SAM+spectral-angle门控仍需更严格差异核验。

### R22. MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration

- 年份/状态：2025；ICCV 2025。
- 主来源：[原始记录](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Wu_MP-HSIR_A_Multi-Prompt_Framework_for_Universal_Hyperspectral_Image_Restoration_ICCV_2025_paper.pdf)；[作者仓库](https://github.com/ZhehuiWu/MP-HSIR)。
- 核验：official_repository_paper_targeted_by_researcher。
- 相关机制：冻结CLIP编码退化文本，训练prompt适配和恢复网络；文本退化调制已有HSI近邻。
- 限制：通用HSI恢复，具体融合任务覆盖未核验。

### R23. Degradation Alchemy: Self-Supervised Unknown-to-Known Transformation for Blind Hyperspectral Image Fusion

- 年份/状态：2025；arXiv预印本；本报告未确认正式venue。
- 主来源：[原始记录](https://arxiv.org/abs/2503.14892)；[全文或摘要核验入口](https://arxiv.org/html/2503.14892v1)。
- 核验：full_text_targeted_by_researcher。
- 相关机制：U2K保留冻结融合网络，通过需要训练的包装/转换模块适配未知退化。
- 限制：不是文本方法，也非严格免训练；是控制器必须考虑的非LLM近邻。

### R24. Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion

- 年份/状态：2026；IEEE TIP 35:69–84；DOI 10.1109/TIP.2025.3647207。
- 主来源：[原始记录](https://doi.org/10.1109/TIP.2025.3647207)；[全文或摘要核验入口](https://pubmed.ncbi.nlm.nih.gov/41477791/)；[作者仓库](https://github.com/Zhu1116/CDPS-train)。
- 核验：author_abstract_index_official_repository。
- 相关机制：直接HSI–MSI融合，从输入测试图像对训练空间/光谱扩散先验，双观测耦合采样。
- 限制：无HR监督，但有test-pair网络训练。

### R25. Controlling Vision-Language Models for Multi-Task Image Restoration

- 年份/状态：2024；ICLR 2024；DA-CLIP。
- 主来源：[原始记录](https://arxiv.org/abs/2310.01018)。
- 核验：author_abstract_conference_metadata。
- 相关机制：冻结CLIP视觉编码器，训练控制器/恢复模型区分内容与退化。
- 限制：自然RGB恢复，不是当前HSI–MSI融合的效果证据。

