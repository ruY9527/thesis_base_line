# Idea Bank

## Generation Setup

方向：现有31/3波段CAVE×8 CNN与冻结大模型、文本、training-free结合。优先机制而非模型堆叠；按失配、假设攻击、表征迁移、诊断四种视角生成12个候选。文献地图见landscape-map.md。未知GPU/目标venue，因此按小缓存模型和局部修正起步，不承诺具体硬件可运行。已有baseline诊断参与动机形成，但未运行任何候选。

训练方式：IDEA-001为冻结大模型的小模块适配；IDEA-002为冻结全网络仅优化图像；IDEA-003为冻结控制器候选。分数是探索性判断，失败条件优先于总分。

## Candidate Ideas

### IDEA-001: 观测证据校验的可拒绝文本先验

- 摘要：用区域谱证据校验冻结VLM关系，在fuse后有界注入且保留退出路径。
- 可检验假设：可靠区域关系有助细节迁移；异质或冲突关系降权可减少谱失真。
- 动机与故事：同类语义与连续光谱错位；文本影响需由实际观测限制。
- 来源：[R01](https://ojs.aaai.org/index.php/AAAI/article/view/38261)；[R02](https://www.sciencedirect.com/science/article/pii/S1566253525010590)；[R21](https://papers.miccai.org/miccai-2025/paper/1453_paper.pdf)；[R22](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Wu_MP-HSIR_A_Multi-Prompt_Framework_for_Universal_Hyperspectral_Image_Restoration_ICCV_2025_paper.pdf)。
- 最近邻：CO²IF: Language-Bridging Hyperspectral-Multispectral Image Fusion with Coordinated and Cross-modal Optimal Transport；Exploring structured uncertainty for external prior-guided hyperspectral image fusion；SAMSA: Segment Anything Model Enhanced with Spectral Angles for Hyperspectral Interactive Medical Image Segmentation；MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration。
- 与简单“X用于Y”的区别：相对文本对齐/困难区域权重，研究推断时先验冲突、拒绝和错误prompt鲁棒性。
- 假设条件：文本区域定位与LR观测足迹能对应；固定R/D可信；小模块训练被允许。
- 最小验证：正确、空、打乱、错误描述×纯视觉/有谱校验，对照同参数的小模块。
- 否定观察：纯视觉或空文本达到同等表现，或门控全归零，否定文本机制。
- 廉价kill test /停止条件：若无法同时观察可靠关系利用和错误关系拒绝，或SAM劣化且无补偿意义，撤下文本主贡献。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：CoPGNet或CO²IF及后续是否已系统做推断拒绝/语义冲突；SAMSA谱角组合差异？
- 风险/成本：medium / weeks；未读付费近邻全文；GPU预算和允许训练程度未知。
- 状态：selected；明确机制与廉价反事实检验，最直接匹配用户需求；仅新颖性信号，不能称已创新。
- 当前证据等级：exploratory_hypothesis。

### IDEA-002: 无新增网络训练的冻结先验残差图修正

- 摘要：Net与外部模型固定，仅优化有幅度约束的图像残差；文本/谱证据筛选关系图。
- 可检验假设：观测支持的关系图比RGB图更适合共享残差，且保守幅度限制减少先验失真。
- 动机与故事：冻结先验容易过度补偿；把其影响限制在可核验关系与残差预算中。
- 来源：[R02](https://www.sciencedirect.com/science/article/pii/S1566253525010590)；[R11](https://github.com/earth-insights/PLRDiff)；[R12](https://arxiv.org/abs/2209.14687)；[R13](https://openreview.net/forum?id=mRieQgMtNTQ)；[R21](https://papers.miccai.org/miccai-2025/paper/1453_paper.pdf)。
- 最近邻：Exploring structured uncertainty for external prior-guided hyperspectral image fusion；Unsupervised Hyperspectral Pansharpening via Low-rank Diffusion Model；Diffusion Posterior Sampling for General Noisy Inverse Problems；Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model；SAMSA: Segment Anything Model Enhanced with Spectral Angles for Hyperspectral Interactive Medical Image Segmentation。
- 与简单“X用于Y”的区别：图约束作用于修正，显式区分视觉/文本/谱支持与无法核验区域；不宣称图正则本身全新。
- 假设条件：可获得匹配的Net权重；可微D及伴随准确；稀疏图可计算；网络完全冻结。
- 最小验证：固定可复测checkpoint，比仅一致性、TV、RGB图、视觉+谱图、视觉+文本+谱图；同迭代预算。
- 否定观察：普通TV/局部图等效，或观测残差改善但HR谱恶化。
- 廉价kill test /停止条件：若同预算简单图足够，撤下大模型；若语言无额外作用，撤下文本；若谱失真持续则停止该修正。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：观测谱筛选的冻结残差图与传统图正则/PnP/CoPGNet到底相差什么？
- 风险/成本：medium / weeks；对应旧最佳权重尚未核验；原有选模与坐标协议需冻结。
- 状态：selected；适合无新增网络训练要求；将欠定性风险显式纳入测试。
- 当前证据等级：exploratory_hypothesis。

### IDEA-003: 有限动作的证据驱动CoT融合控制器

- 摘要：冻结LLM根据数值诊断卡选择核候选、先验强度或修正策略，工具反馈接受/回退。
- 可检验假设：同输入、工具和预算下，证据组织与反馈能减少跨退化场景中的无效动作。
- 动机与故事：残差不能直接说明退化或内容哪一方错误；诊断必须变成可执行假设。
- 来源：[R04](https://arxiv.org/abs/2604.08884)；[R14](https://arxiv.org/abs/2504.07148)；[R15](https://arxiv.org/abs/2604.09511)；[R23](https://arxiv.org/abs/2503.14892)。
- 最近邻：Beyond RGB: Benchmarking and Enhancing MLLMs for Hyperspectral Image Understanding via Training-Free Reasoning Framework；Q-Agent: Quality-Driven Chain-of-Thought Image Restoration Agent through Robust Multimodal Large Language Model；RIRF: Reasoning Image Restoration Framework；Degradation Alchemy: Self-Supervised Unknown-to-Known Transformation for Blind Hyperspectral Image Fusion。
- 与简单“X用于Y”的区别：双传感器兼容诊断与无GT选择；等预算比较，非首次CoT恢复。
- 假设条件：观测诊断具备足够区分度；工具库可靠；冻结LLM能稳定输出合法动作。
- 最小验证：固定正确R和有限PSF族，比规则、随机/网格、数值控制器、LLM一次建议、LLM反馈。
- 否定观察：等预算规则/网格已匹配，或仅增加工具次数才有收益。
- 廉价kill test /停止条件：若数值诊断已足够或不能在新失配场景泛化，撤下CoT必要性；不使用GT决策。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：Q-Agent/RIRF/U2K是否已覆盖同类诊断反馈，语言控制是否多余？
- 风险/成本：high / months；需可靠工具库、混合退化集与可用LLM；目前不是已具备的实验能力。
- 状态：needs_research；科学问题有价值，但工程与必要性风险高，暂不主推。
- 当前证据等级：exploratory_hypothesis。

### IDEA-004: 语义先验幻觉与可辨识性诊断基准

- 摘要：构造同颜色异谱、同语义异谱、错误prompt、混合像元与共同零空间控制组。
- 可检验假设：均值PSNR可能掩盖材料区域的文本诱发失真；分层反事实诊断可揭示机制。
- 动机与故事：语义先验被当成光谱事实时，其失效并非普通随机误差。
- 来源：[R01](https://ojs.aaai.org/index.php/AAAI/article/view/38261)；[R02](https://www.sciencedirect.com/science/article/pii/S1566253525010590)；[R04](https://arxiv.org/abs/2604.08884)；[R13](https://openreview.net/forum?id=mRieQgMtNTQ)。
- 最近邻：CO²IF: Language-Bridging Hyperspectral-Multispectral Image Fusion with Coordinated and Cross-modal Optimal Transport；Exploring structured uncertainty for external prior-guided hyperspectral image fusion；Beyond RGB: Benchmarking and Enhancing MLLMs for Hyperspectral Image Understanding via Training-Free Reasoning Framework；Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model。
- 与简单“X用于Y”的区别：衡量先验接纳/拒绝及不确定性边界，而非只增加质量指标。
- 假设条件：能构造幅值合法的反事实谱和边界，并核实区域谱差异。
- 最小验证：先在合法输入生成的描述上做配对干预，用独立HR-HSI仅评价；数学歧义组不要求恢复无证据真相。
- 否定观察：分层诊断完全被传统SAM/区域误差解释，或合成现象不出现在真实样本。
- 廉价kill test /停止条件：若无额外机制洞见或规模不足，降为A/B的辅助评价而非独立benchmark论文。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：是否已有文本融合robustness benchmark覆盖这些控制组？
- 风险/成本：medium / weeks；独立贡献需要足够真实验证；旧测试数据仍只能用于探索。
- 状态：shortlisted；可支持主方法且失败也有信息，保留为诊断，不另占主选名额。
- 当前证据等级：exploratory_hypothesis。

### IDEA-005: 语言检索实测光谱库的软材料先验

- 摘要：文本提供材料候选查询，真实测量库经波长/单位匹配，再由LR-HSI证据软筛选。
- 可检验假设：实测谱检索比LLM编谱更可追溯，证据筛选可能减少错误材料匹配。
- 动机与故事：材料名称不足以给出反射率，检索必须回到可测量谱。
- 来源：[R01](https://ojs.aaai.org/index.php/AAAI/article/view/38261)；[R19](https://github.com/Dmsw/Lib-SRDM)。
- 最近邻：CO²IF: Language-Bridging Hyperspectral-Multispectral Image Fusion with Coordinated and Cross-modal Optimal Transport；Lib-SRDM (repository)。
- 与简单“X用于Y”的区别：未知类退出、波长单位一致与双观测筛选；需区别现有语义光谱库。
- 假设条件：存在与目标场景、波长和照明匹配的合法测量库。
- 最小验证：小型训练划分谱库，对比无类别检索、文本检索、有/无观测筛选，测试未知材料。
- 否定观察：正确语言也检不到适配光谱，或纯谱检索达到同等效果。
- 廉价kill test /停止条件：库覆盖/照明偏差无法解决则停止当前CAVE路线；文本不增益则撤下语言主张。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：Lib-SRDM与其他语义谱库已做哪些融合筛选和未知类处理？
- 风险/成本：high / months；可用谱库未确认；CAVE室内物体材质难匹配遥感库。
- 状态：needs_research；机制具体但存在数据可用性阻碍。
- 当前证据等级：exploratory_hypothesis。

### IDEA-006: 传感器元数据文本化条件融合

- 摘要：把波长、SRF、倍率和噪声报告变成文本条件控制CNN。
- 可检验假设：如果文本编码能跨传感器组织数值信息，应改善未见响应泛化。
- 动机与故事：跨传感器条件需要明确，但自然语言是否优于数值输入未知。
- 来源：[R03](https://ojs.aaai.org/index.php/AAAI/article/view/37451)；[R06](https://ieeexplore.ieee.org/document/10490262)；[R22](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Wu_MP-HSIR_A_Multi-Prompt_Framework_for_Universal_Hyperspectral_Image_Restoration_ICCV_2025_paper.pdf)。
- 最近邻：CLIPPan: Adapting CLIP as a Supervisor for Unsupervised Pansharpening；SpectralGPT: Spectral Remote Sensing Foundation Model；MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration。
- 与简单“X用于Y”的区别：只有同等数值信息下文本表示的独立优势才有新意义。
- 假设条件：完整传感器元数据可用且精确。
- 最小验证：相同信息的数值MLP、波长编码和文本条件比较。
- 否定观察：数值条件匹配或超过文本。
- 廉价kill test /停止条件：若收益来自新增元数据而非文本表示，停止作为语言创新。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：是否已被protocol prompt/退化文本/波长条件工作覆盖？
- 风险/成本：medium / weeks；目前只有固定R/D；跨传感器设定尚未搭建。
- 状态：rejected；作为单独贡献太接近已有条件适配，暂作对照。
- 当前证据等级：exploratory_hypothesis。

### IDEA-007: 大模型教师蒸馏到轻量Net

- 摘要：训练期缓存可信大模型区域表征，推理学生仅保留小CNN。
- 可检验假设：若教师确有光谱收益，学生可保留部分收益并减少部署开销。
- 动机与故事：大模型推理成本可能大于0.83M主干，需要明确精度—代价曲线。
- 来源：[R05](https://arxiv.org/abs/2406.11519)；[R25](https://arxiv.org/abs/2310.01018)。
- 最近邻：HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model；Controlling Vision-Language Models for Multi-Task Image Restoration。
- 与简单“X用于Y”的区别：需先有经过观测验证的教师作用，蒸馏本身不新。
- 假设条件：教师确有经独立测试支持的收益且可缓存。
- 最小验证：教师有/无可信筛选、学生等参数普通训练，报告整图时间显存。
- 否定观察：教师无谱收益或学生不优于常规训练。
- 廉价kill test /停止条件：先确认教师价值，否则停止；作为单独创新新颖性不足。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：是否存在同任务相同先验筛选蒸馏机制？
- 风险/成本：medium / weeks；教师尚未验证。
- 状态：rejected；可作A的效率扩展，拒绝成为独立主线。
- 当前证据等级：exploratory_hypothesis。

### IDEA-008: 冻结光谱与视觉基础模型的冲突分工

- 摘要：HyperSIGMA/HyperFree提供谱或HSI结构，RGB模型提供空间关系，冲突时限制先验共享。
- 可检验假设：HSI专用先验能在RGB误判区域补充证据，但需要保持测量单位。
- 动机与故事：自然图先验与光谱分布不一致，两类预训练可能互补也可能重复。
- 来源：[R05](https://arxiv.org/abs/2406.11519)；[R06](https://ieeexplore.ieee.org/document/10490262)；[R07](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_HyperFree_A_Channel-adaptive_and_Tuning-free_Foundation_Model_for_Hyperspectral_Remote_CVPR_2025_paper.pdf)；[R17](https://doi.org/10.1109/TIP.2026.3736733)。
- 最近邻：HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model；SpectralGPT: Spectral Remote Sensing Foundation Model；HyperFree: A Channel-adaptive and Tuning-free Foundation Model for Hyperspectral Remote Sensing Imagery；DSES-Diff: A Dynamic-Spectral and Efficient-Spatial Diffusion-Based Foundation Model for Scalable Hyperspectral and Multispectral Image Fusion。
- 与简单“X用于Y”的区别：具体研究冲突及通道/波长适配，不能只拼两个encoder。
- 假设条件：公开权重接口能支持31波段或正确映射；域差异可控。
- 最小验证：冻结小版本HSI基础模型，对比RGB先验、HSI先验、双先验和随机初始化等预算适配。
- 否定观察：双先验只增加开销或收益来自新增参数而非预训练。
- 廉价kill test /停止条件：输入映射严重改变谱信息或显存不可承受则不保留；无互补性降为单先验。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：DSES-Diff和多基础模型适配是否已有相同分工与冲突研究？
- 风险/成本：high / weeks；具体权重下载、接口与CAVE领域适配未运行。
- 状态：shortlisted；有信号但接口成本明显，保留为A/B后续消融。
- 当前证据等级：exploratory_hypothesis。

### IDEA-009: 本征图桥接预训练RGB超分

- 摘要：用固定光谱基将HSI变为少量空间图，调用预训练RGB SR再恢复光谱。
- 可检验假设：空间—光谱分工能缓解通道差异并保留大模型纹理能力。
- 动机与故事：RGB模型无法直接处理31波段，需要表征桥接。
- 来源：[R08](https://arxiv.org/abs/2409.04050)；[R11](https://github.com/earth-insights/PLRDiff)。
- 最近邻：EigenSR: Eigenimage-Bridged Pre-Trained RGB Learners for Single Hyperspectral Image Super-Resolution；Unsupervised Hyperspectral Pansharpening via Low-rank Diffusion Model。
- 与简单“X用于Y”的区别：必须增加明确的新失配问题才区别EigenSR/低秩diffusion。
- 假设条件：固定低秩能覆盖目标数据且不丢稀有材料。
- 最小验证：同基、同RGB模型，比较已知EigenSR类适配与简单双观测修正。
- 否定观察：现有桥接方法覆盖机制，或稀有谱被截断。
- 廉价kill test /停止条件：当前方案只有复用已有桥接，停止作为独立新颖贡献。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：是否只是EigenSR加MSI观测的增量移植？
- 风险/成本：medium / weeks；预训练接口和倍率兼容待核验。
- 状态：rejected；作为参考实现有用，当前独立创新不足。
- 当前证据等级：exploratory_hypothesis。

### IDEA-010: 冻结扩散的有界候选修正与观测选择

- 摘要：只在低维空间生成有限候选残差，双观测和幅度约束筛选，限制文本引导强度。
- 可检验假设：候选先验可能在弱观测场景有用，但有界影响比自由生成更稳健。
- 动机与故事：生成细节与真实光谱可能冲突，不能只看视觉锐化。
- 来源：[R09](https://arxiv.org/abs/2409.12470)；[R10](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.pdf)；[R11](https://github.com/earth-insights/PLRDiff)；[R12](https://arxiv.org/abs/2209.14687)；[R13](https://openreview.net/forum?id=mRieQgMtNTQ)；[R16](https://arxiv.org/abs/2608.11748)；[R17](https://doi.org/10.1109/TIP.2026.3736733)；[R18](https://doi.org/10.3390/rs18101582)。
- 最近邻：HSIGene: A Foundation Model for Hyperspectral Image Generation；Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model；Unsupervised Hyperspectral Pansharpening via Low-rank Diffusion Model；Diffusion Posterior Sampling for General Noisy Inverse Problems；Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model；Dual Modality Prompted Diffusion Priors for Zero Shot Hyperspectral Pansharpening；DSES-Diff: A Dynamic-Spectral and Efficient-Spatial Diffusion-Based Foundation Model for Scalable Hyperspectral and Multispectral Image Fusion；Modulated Diffusion with Spatial–Spectral Disentangled Guidance for Hyperspectral Image Super-Resolution。
- 与简单“X用于Y”的区别：必须比已存在子空间/双prompt扩散提出可辨识或可靠性差异。
- 假设条件：有兼容目标波段和表示的可用生成权重；采样成本可接受。
- 最小验证：先比较无扩散的B，再在同时间预算加冻结扩散候选，测谱形/采样变异。
- 否定观察：确定性修正已匹配，或候选好看却SAM变差。
- 廉价kill test /停止条件：若缺兼容权重或单场景成本不合理则暂停；未胜简单修正则不保留扩散。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：PLRDiff/DIDM/DSES-Diff已覆盖多少可靠性/候选选择？
- 风险/成本：high / months；GPU和权重接口未确认；2026近邻全文不全。
- 状态：needs_research；保留为远期机制，当前不承担实现主线。
- 当前证据等级：exploratory_hypothesis。

### IDEA-011: 直接拼接CLIP文本替换通道注意力

- 摘要：把caption embedding拼到fuse或由文本生成通道权重。
- 可检验假设：文本应提升融合表现。
- 动机与故事：局部CNN可能缺语义，但拼接没有说明光谱作用机制。
- 来源：[R01](https://ojs.aaai.org/index.php/AAAI/article/view/38261)；[R22](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Wu_MP-HSIR_A_Multi-Prompt_Framework_for_Universal_Hyperspectral_Image_Restoration_ICCV_2025_paper.pdf)。
- 最近邻：CO²IF: Language-Bridging Hyperspectral-Multispectral Image Fusion with Coordinated and Cross-modal Optimal Transport；MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration。
- 与简单“X用于Y”的区别：当前没有可核验差异；CO²IF已经直接文本融合。
- 假设条件：文本与输入对应。
- 最小验证：空/随机/正确文本和同参数卷积对照。
- 否定观察：无文本同等，或最近邻已有更完整文本融合。
- 廉价kill test /停止条件：不在没有新机制时作为主创新推进。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：已经明显被CO²IF覆盖。
- 风险/成本：medium / days；无具体新失配与贡献。
- 状态：rejected；实现容易但创新不足，明确拒绝。
- 当前证据等级：exploratory_hypothesis。

### IDEA-012: 让LLM长CoT直接生成31波段或从多个结果挑最清楚

- 摘要：通过长描述推理直接预测连续光谱，或让VLM凭显示效果选融合图。
- 可检验假设：推理文本越充分，光谱恢复越准确。
- 动机与故事：语言没有直接连续光谱测量，RGB清晰度无法判定谱真值。
- 来源：[R04](https://arxiv.org/abs/2604.08884)；[R14](https://arxiv.org/abs/2504.07148)；[R15](https://arxiv.org/abs/2604.09511)。
- 最近邻：Beyond RGB: Benchmarking and Enhancing MLLMs for Hyperspectral Image Understanding via Training-Free Reasoning Framework；Q-Agent: Quality-Driven Chain-of-Thought Image Restoration Agent through Robust Multimodal Large Language Model；RIRF: Reasoning Image Restoration Framework。
- 与简单“X用于Y”的区别：没有可支持的恢复机制；隐式推理内容不作为科学证据。
- 假设条件：LLM输出可追溯精确谱这一前提未成立。
- 最小验证：同颜色异谱控制组与数值重建比较。
- 否定观察：长CoT不能识别观测不可辨识谱，或选视觉好看而谱更差。
- 廉价kill test /停止条件：不作为重建器推进；只允许结构化有限动作诊断替代。
- 正结果说明：只有排除简单对照与额外预算后，才支持上述具体机制；不自动证明新颖性。
- 负结果说明：应接受简单方法已足够、先验不可核验或语言没有额外作用；不要靠更大模型挽救未证假设。
- 新颖性待审：与已有CoT恢复相比有何新测量或机制？
- 风险/成本：high / months；可辨识性与连续数值可靠性不成立。
- 状态：rejected；不可测/不可靠主张与已有CoT近邻并存，拒绝。
- 当前证据等级：exploratory_hypothesis。

