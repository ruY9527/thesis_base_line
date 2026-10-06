# 面向当前 baseline 的自监督高光谱融合论文研究清单

核验日期：2026-10-05。范围：IJCV、TIP、CVPR、NeurIPS、AAAI、ICLR、ICCV、ECCV、TPAMI 的正式期刊论文与主会论文；兼顾直接 HSI–MSI 融合、相邻高光谱恢复、自监督逆问题及可迁移结构。

## 检索与核验说明

本次使用用户指定的 [firecrawl-research-papers](C:/Users/Administrator/.codex/skills/firecrawl-research-papers/SKILL.md) 的检索、扩展和核验流程。当前没有可调用的 Firecrawl MCP/CLI 或 API key，因此实际检索由网页工具完成，依据会议官网、出版社、作者论文及作者 GitHub；没有执行 Firecrawl Research 的语义索引或引用图接口。

下文共收录 **62 篇去重候选，均核实存在实际公开源码**。这是一份针对 baseline 的研究导航，不是宣称穷尽九个 venue 的系统综述，也没有逐篇复现指标。每篇注明训练设置、源码文件证据和迁移限制。代码公开不代表工程完整、权重可下载或 Win11 上直接运行：CESST、UMNet 是部分工程源码；FeatUp 的不同组件公开程度不同；CDPS 的完整耦合采样入口仍有核验限制。

收录标准：论文与指定 venue 对应；有实际实现而非只有占位 README；方法能对应本项目的具体瓶颈。初筛中也使用题目索引和相关论文列表发现线索，最终条目以主来源为依据。相同论文的预印本、正式版不重复计数；DIP 的 CVPR 2018 与 IJCV 2020 扩展作为一个工作家族。UAL 与 UTAL 是两篇不同论文，学习时应视作同一路线的演进。

## 先明确你的 baseline 与研究目标

当前 [model.py](D:/coding/github_self/thesis_base_line/model.py) 是双分支卷积融合网络：

```text
LR-HSI -> hsi_head -> 特征双线性放大 --+
                                     +-> concat -> fuse -> f
HR-MSI -> msi_head -------------------+

深层特征 = f + body_out(8×ResBlock(f))
图像修正 = tail(深层特征)
输出HSI  = 双线性放大的原始LR-HSI + 图像修正
```

`ChannelAttention` 是全局空间平均、LayerNorm、两层瓶颈 MLP 与 sigmoid 门控。64 个通道是网络学到的特征通道，不等于 31 个物理波段。它给每个特征通道一个全图共享权重，不能单独识别图像哪里失真，也不直接实现 HSI/MSI 间的信息交互。

[train_cave_M.py](D:/coding/github_self/thesis_base_line/train_cave_M.py:143) 当前用 `L1Loss(output, a1.cuda())`，其中 `a1` 是 HR-HSI 目标，因此当前训练属于监督训练。`model.py` 可以作为自监督研究的网络骨架；自监督性质取决于训练数据和损失，增加注意力、Transformer 或 Mamba 不会自动改变监督方式。

令观测为 LR-HSI \(Y_h\) 与 HR-MSI \(Y_m\)，目标为 HR-HSI \(X\)。在已配准、线性且空间退化跨波段一致的简化模型下：

\[
Y_h=\mathcal D_P(X)+n_h,\qquad Y_m=R X+n_m,
\]

其中 \(\mathcal D_P\) 是 PSF 模糊和空间采样，\(R\) 是 SRF 光谱响应。现有 `create_F()` 与 Gaussian PSF 可作为已知退化实验的起点，但必须核对真实数据生成代码的卷积边界、核中心和采样起点。

最小自监督对照可以保留 Net，并使用：

\[
\widehat X=f_\theta(Y_h,Y_m),\qquad
\mathcal L_{obs}=\operatorname{mean}|\mathcal D_P(\widehat X)-Y_h|
+\lambda\operatorname{mean}|R\widehat X-Y_m|.
\]

这只使用实际观测作为目标；需要用 PyTorch 实现可微退化，避免 NumPy/SciPy 或 detach 切断重建损失梯度。纯一致性依然是欠定问题：联合算子 \(A(X)=(\mathcal D_P(X),RX)\) 的共同零空间可能允许多种重建结果。因此还要研究低秩、内部学习、合适的等变/拆分目标或先验；这一判断可参照 [EI](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.html) 和 [MOI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/203e651b448deba5de5f45430c45ea04-Abstract-Conference.html)，不能把它写成单靠损失就保证唯一恢复。

## 最建议先读的论文与源码

此顺序是针对你当前学习阶段的研究建议，不是论文性能排名。

| 顺序 | 工作 | 首先学习什么 | 对 baseline 的位置 | 论文与实现 |
|---|---|---|---|---|
| 1 | ZSL，TPAMI 2023 | 观测内降尺度训练、退化响应估计 | 数据生成、训练目标 | [论文](https://doi.org/10.1109/TPAMI.2023.3279050) · [代码](https://github.com/renweidian/ZSL) |
| 2 | CS2DIPs，TIP 2024 | 两种观测如何共同约束重建 | 自监督损失、低秩输出 | [论文](https://doi.org/10.1109/TIP.2024.3390582) · [代码](https://github.com/Ambitionfy/CS2DIPs) |
| 3 | CUCaNet，ECCV 2020 | 解混、跨注意力、SRF/PSF 学习 | fuse、退化模块 | [论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740205.pdf) · [代码](https://github.com/danfenghong/ECCV2020_CUCaNet) |
| 4 | MossFuse，IJCV 2026 | 共享/互补模态信息如何解耦 | 双分支、concat+fuse | [论文](https://doi.org/10.1007/s11263-026-02757-8) · [代码](https://github.com/dusongcheng/MossFuse) |
| 5 | CLoRF，IJCV 2025 | 连续空间/光谱表示与低秩结构 | 插值、tail、输出形式 | [论文](https://link.springer.com/article/10.1007/s11263-025-02540-1) · [代码](https://github.com/wangting1907/CLoRF-Fusion) |
| 6 | EI、MOI、Equivariant Splitting | 为什么一致性不足，怎样设计自监督目标 | 训练循环、联合算子 | [EI代码](https://github.com/edongdongchen/EI) · [MOI代码](https://github.com/edongdongchen/MOI) · [ES代码](https://github.com/vsechaud/Equivariant-Splitting) |
| 7 | SRLF-Net，CVPR 2025 | 用观测误差找需要细化的位置 | body 后的修正分支 | [论文](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html) · [代码](https://github.com/YuanyeLiu/SRLF-Net) |
| 8 | FcaNet / Restormer / NAFNet | 注意力摘要、特征交互和轻量门控 | ChannelAttention、ResBlock | [FcaNet](https://github.com/cfzd/FcaNet) · [Restormer](https://github.com/swz30/Restormer) · [NAFNet](https://github.com/megvii-research/NAFNet) |
| 9 | ARGS-Diff、DDS2M | 当前观测上的空间/光谱扩散先验 | 更大的后续研究路线 | [ARGS训练](https://github.com/Zhu1116/ARGS-Diff-train) · [DDS2M](https://github.com/miaoyuchun/DDS2M) |

先阅读损失的计算路径，再阅读模型模块。对每个仓库追踪：输入来自哪里、目标是什么、退化是否已知/可学习、是否依赖外部预训练、是否使用真值选最佳迭代。比只照搬一段 Attention 更能帮助你理解方法为何成立。

## 共性、趋势与尚未解决的问题

**共性。** 直接无监督融合路线常把空间/光谱退化显式写入重建，把 HSI 的低维相关性与 MSI 的空间细节分别利用。CUCaNet、CS2DIPs、CLoRF 和 ARGS-Diff 实现方式各异；它们支持研究这些建模原则，不能由此推断任意两种模块相加都会提高精度。详见各条目的原论文。

**近期趋势。** MossFuse 研究模态解耦，CLoRF 研究连续表示，ARGS-Diff/CDPS 研究子空间与后验扩散，CARL 研究跨相机自监督表征，Equivariant Splitting 研究不完整观测下的训练风险。它们分别对应融合信息如何组织、输出如何表示、先验如何学习和训练信号如何建立，而非同一种创新方向。

**核心问题。** 未知退化、噪声、错位和光谱不可辨识性会叠加。固定 R/P 的合成 CAVE 成绩不能直接说明真实传感器表现；语义特征不变性可能牺牲精确光谱幅值；生成式先验可能补出看起来清晰但不符合真实光谱的细节；较低观测误差也不必然代表较高 HR-HSI 精度。这些都需要专门实验，而不是只看 RGB 可视化。

## 可以探索的六条 idea 路线

以下是结合当前代码的研究假设，**并未验证有效，也没有完成新颖性审查**。已存在的论文要作为最近邻比较，不能把直接移植命名为原创。

### 1. 用观测失配指导局部修正

当前 CA 是全图共享通道权重，尾部一次性预测修正量。可先得到 \(\widehat X_0\)，计算 \(e_h=Y_h-\mathcal D_P(\widehat X_0)\) 和 \(e_m=Y_m-R\widehat X_0\)，把 LR 残差放大、MSI 残差映射到特征空间，再预测空间与通道联合门控，输出第二次修正。

最近邻是 [SRLF-Net](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html) 的观测结构选择和 [ARGS-Diff](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.pdf) 的残差引导。可研究的具体问题：在自监督、含噪或 R/P 失配时，怎样区分噪声残差和真正缺失细节。对照 CA、无残差门控、全图额外卷积及相同参数预算；同时测空间误差、SAM 与双观测误差，避免只因网络变深而改善。

### 2. 先校准退化，再条件化融合

参考 [ZSL](https://doi.org/10.1109/TPAMI.2023.3279050)、[CUCaNet](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740205.pdf)、[UTAL](https://doi.org/10.1109/TPAMI.2024.3361894) 和 [KernelGAN](https://github.com/sefibk/KernelGAN)。对非负、归一化 PSF 及物理合理 SRF 建受约束参数化，在简化模型中利用低尺度 MSI 的兼容关系 \(R Y_h\approx\mathcal D_P(Y_m)\)，先估计退化再冻结或交替更新，最后把退化编码作为融合条件。

这一关系要求配准、线性响应、相容的采样/边界，以及近似跨波段一致的空间退化；不是适用于任何真实图像的恒等式。测试已知退化、固定错误退化、估计后冻结、全联合训练四组。若只优化观测损失，重建和退化可能互相补偿，必须同时检查响应估计误差与最终重建，不能只报告损失更小。

### 3. 共享/私有信息分支与跨传感器学习

参考 [MossFuse](https://arxiv.org/abs/2412.04802) 和 [CARL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html)。两路各拆为共享结构与模态私有信息，只在共享分支做对应约束，保留 HSI 私有光谱和 MSI 私有高频。训练可利用不同已知 SRF 的合法模拟观测；是否有跨传感器收益要在未见 SRF 上测试。

最小实现先改 `fuse`，不要同时换整个编码器。对照仅拆分、加共享约束、加私有约束，检查共享与私有分支是否坍塌。把全部特征强行一致可能丢掉融合需要的互补信息。MossFuse 已研究解耦，必须提出更具体的失配问题或训练机制才可能形成新的贡献。

### 4. 在低秩系数域学习连续残差

参考 [CLoRF](https://link.springer.com/article/10.1007/s11263-025-02540-1)、[CS2DIPs](https://doi.org/10.1109/TIP.2024.3390582)、[LRSDN](https://doi.org/10.1109/TIP.2024.3354127)、[OTIAS](https://ojs.aaai.org/index.php/AAAI/article/view/32275) 和 [UAFL](https://github.com/yingkai-zhang/UAFL)。让网络先预测少量空间系数，再与光谱基相乘；进一步以坐标、像素覆盖范围与真实波长为条件查询输出。你的 `tail:64→31` 和固定双线性特征插值是切入点。

先验证固定网格低秩版本，再增加坐标解码器，区分低秩收益与连续表示收益。测试未见倍率、波段缺失与真实波长变化，并检查稀有材料误差。波段序号不能替代 nm；SVD 正交基也不自动等于非负物理端元。连续表示不能凭空创造观测中没有的信息。

### 5. 面向含噪观测改进图像残差基线

当前 `hsi_up` 把 LR-HSI 的噪声一起放大。参考 [Hipandas](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Hipandas_Hyperspectral_Image_Joint_Denoising_and_Super-Resolution_by_Image_Fusion_ICCV_2025_paper.html)、[REI](https://github.com/edongdongchen/REI)、[UNSURE](https://github.com/tachella/unsure) 与 Noise2Void 等方法，研究去噪后的 LR-HSI 作为跳跃基线，或在两种观测上设置分别适用的噪声风险目标。

普通掩码不是通用无标签方案：目标噪声必须满足相应独立性假设，且插值基线、另一模态或特征路径不得泄漏被留出的目标。跨模态共享干净信号本身可以是有效条件；真正要排查的是目标值/噪声泄漏。实验需覆盖白噪声、跨波段相关噪声和条带，并检查去噪是否抹掉低强度材料。

### 6. 自监督信息补充：等变、拆分与内部学习

参考 [EI](https://github.com/edongdongchen/EI)、[MOI](https://github.com/edongdongchen/MOI)、[Equivariant Splitting](https://github.com/vsechaud/Equivariant-Splitting) 与 [ZSSR](https://github.com/assafshocher/ZSSR)，优先在训练循环中建立合法的新约束，网络可以暂时保持不变。对联合算子 A 分析变换是否改变不可见信息结构；普通旋转/翻转可能与低通退化近似交换，从而不能消除共同零空间。

对照纯观测一致性、普通几何等变、满足任务条件的拆分/尺度训练，以及结构先验。不要把这些通用论文的理论保证直接搬到 HSI/MSI。范围外补充 [Scale-Equivariant Imaging，TCI 2026](https://ieeexplore.ieee.org/document/11370202/) 对低通问题很有启发，但不计入指定 venue 的62篇。

## 建议的实验顺序与判断标准

1. **建立三种清楚命名的对照**：现有监督 Net（性能参考）、只用双观测的一致性 Net、ZSL 式内部降尺度训练 Net。严格自监督实验不得以 HR-HSI 真值参与损失或逐图最佳迭代选择。
2. **一次只验证一个假设**：固定数据划分、尺度、R/P、训练预算和随机种子；增加模型容量时设置参数接近的卷积对照。先比较简单模型，再考虑完整扩散/基础模型。
3. **从合成已知退化到真实困难因素**：跨图训练与单图优化分开报告；逐项加入未知核、SRF变化、噪声、错位。CAVE与Harvard近景、Pavia/WDC/Chikusei遥感不能不加说明地合并指标。
4. **不只比较PSNR**：测试集有真值时报告PSNR、SSIM、SAM、ERGAS及逐波段/材料误差；没有真值时报告双观测重投影、空间/光谱可靠性和传感器一致性，明确这些不能完全替代HR精度。训练/验证/测试图像要独立，避免同一大图相邻patch泄漏。
5. **自监督验证要说明协议**：若使用独立有标签验证集调参，应明确为训练无需标签但调参使用标签；若声称全程无HR真值，则停止规则和超参数选择也要符合这一设定。真值可在协议冻结后作测试评估。

## 代码学习和 Win11 环境提示

这些仓库不宜直接合并到本项目的 requirements.txt：不同工作使用不同 PyTorch/TensorFlow、CUDA扩展和数据接口。逐个建立环境；移植模块时先统一 `(B,C,H,W)`、输入通道、归一化、退化采样和输出含义。

uSDN/DBIN/MHF-Net 等有旧 TensorFlow 工程，SENet 官方实现是 Caffe；MambaIR 系列及部分重采样/可变形算子可能需要编译；GEWDiff 的训练源码使用 NCCL和多GPU硬编码；UMNet缺一些工程依赖。对3060/4090的可运行性须以具体配置和实测显存为准，本次没有做安装或GPU测试。MambaIR 的FLOPs应核对作者修正后的统计方法，不能只用现有 thop 给扩展算子记零计算量。

## 未计入主清单的候选与原因

| 候选 | 原因 | 来源 |
|---|---|---|
| CO²IF，AAAI 2026 | 论文存在，但本次仓库只有README与pipeline.png，未见源码 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/38261) · [仓库](https://github.com/yzkLearning/CO2IF) |
| FusionNet，TIP 2020 | 找到作者论文，尚未核实对应公开实现 | [作者全文](https://haozhangxidian.github.io/Papers/TIP_FusionNet.pdf) |
| HSR-Diff，ICCV 2023 | 论文已核实，本轮未找到可核验的作者实现，故不满足本次代码条件 | [论文](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_HSR-Diff_Hyperspectral_Image_Super-Resolution_via_Conditional_Diffusion_Models_ICCV_2023_paper.html) |
| Degradation Alchemy / U2K | 找到作者预印本；本轮未核实指定venue和对应实现，不计入正式清单 | [预印本](https://arxiv.org/abs/2503.14892) |
| SwinIR、MST++ | 分别ICCVW 2021、CVPRW 2022，不算ICCV/CVPR主会 | [SwinIR正式记录](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.html) · [MST++正式记录](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Cai_MST_Multi-Stage_Spectral-Wise_Transformer_for_Efficient_Spectral_Reconstruction_CVPRW_2022_paper.pdf) |
| Noise2Noise、Noise2Self | ICML 2018/2019，超出用户指定venue | 不计入本次篇数 |

UAFL 初筛时CVF访问失败，后续从正式CVF索引核实CVPR2026，已转入主清单；“未核实”不应误写成“没有代码”。

## 重跑输入与维护

```yaml
workflow: firecrawl-research-papers
actual_collection: public web tools; publisher/conference/author papers and official code
verified_on: 2026-10-05
topic: self-supervised HSI-MSI fusion for a dual-branch residual channel-attention CNN
venues: [IJCV, TIP, CVPR, NeurIPS, AAAI, ICLR, ICCV, ECCV, TPAMI]
count_policy: no requested fixed limit; expand relevant families and retain verified implementations
records: 62
outputs: [markdown review, UTF-8 CSV catalog, JSON evidence records]
queries:
  - unsupervised self-supervised hyperspectral multispectral fusion official code
  - blind degradation spectral response spatial response zero-shot HSI fusion
  - continuous low-rank implicit HSI MSI representations
  - equivariant imaging measurement splitting unsupervised inverse problems
  - spectral attention content-aware upsampling restoration Mamba
  - HSI diffusion restoration pansharpening camera-agnostic spectral pretraining
```

来源为核验当时的公开页面；仓库会更新。正式引用时再核对publisher BibTeX、卷期、DOI、作者名单和代码版本，不直接复制仓库中的过时预印本引用。MossFuse正式IJCV2026题名、CDPS正式TIP2026年份、MHF-Net正式TPAMI2022年份已专门核对。下方给出每篇的论文与代码来源、具体实现文件和方法边界。


## 62篇论文总索引


| Venue | Count |
|---|---:|
| IJCV | 4 |
| TIP | 6 |
| CVPR | 20 |
| NeurIPS | 5 |
| AAAI | 4 |
| ICLR | 6 |
| ICCV | 7 |
| ECCV | 5 |
| TPAMI | 5 |

P1/P2/P3 是本次的阅读优先级，不是对论文质量或精度的打分。部分代码限制请看每篇明细。


### A. 直接 HSI–MSI 融合 (21)

| ID | Work | Venue | Priority | Training setting | Paper / Code |
|---|---|---|---|---|---|
| P01 | MossFuse | IJCV 2026 | P1 | 自监督/无监督 HSI-MSI 融合；含退化估计 | [Paper](https://doi.org/10.1007/s11263-026-02757-8) / [Code](https://github.com/dusongcheng/MossFuse) |
| P02 | ARGS-Diff | CVPR 2025 | P1 | 只用当前 LR-HSI/MSI 学习；无外部数据 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.pdf) / [Code](https://github.com/Zhu1116/ARGS-Diff) |
| P03 | CLoRF | IJCV 2025 | P1 | 自监督连续表示 HSI-MSI 融合 | [Paper](https://link.springer.com/article/10.1007/s11263-025-02540-1) / [Code](https://github.com/wangting1907/CLoRF-Fusion) |
| P04 | SRLF-Net | CVPR 2025 | P1 | 监督 HSI/MSI 融合；成像模型引导的选择性细化 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html) / [Code](https://github.com/YuanyeLiu/SRLF-Net) |
| P05 | CS2DIPs | TIP 2024 | P1 | 无需外部训练数据，观测对内优化 | [Paper](https://doi.org/10.1109/TIP.2024.3390582) / [Code](https://github.com/Ambitionfy/CS2DIPs) |
| P06 | UTAL | TPAMI 2024 | P1 | 监督合成数据预训练 + 无监督测试时适配；非全程零标签 | [Paper](https://doi.org/10.1109/TPAMI.2024.3361894) / [Code](https://github.com/JiangtaoNie/UAL) |
| P07 | ZSL | TPAMI 2023 | P1 | 当前 HSI/MSI 观测构造低尺度训练对；零样本融合 | [Paper](https://doi.org/10.1109/TPAMI.2023.3279050) / [Code](https://github.com/renweidian/ZSL) |
| P08 | CUCaNet | ECCV 2020 | P1 | 无监督盲融合/耦合解混 | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740205.pdf) / [Code](https://github.com/danfenghong/ECCV2020_CUCaNet) |
| P09 | CDPS | TIP 2026 | P2 | 无监督；仅输入测试对学习扩散先验 | [Paper](https://doi.org/10.1109/TIP.2025.3647207) / [Code](https://github.com/Zhu1116/CDPS-train) |
| P10 | SHOTUN | TIP 2026 | P2 | 无监督张量展开；未知退化估计含预训练策略 | [Paper](https://doi.org/10.1109/TIP.2026.3695389) / [Code](https://github.com/Shawn-H-Wang/SHOTUN) |
| P11 | UAFL | CVPR 2026 | P2 | 监督、未配准 RGB 引导 HSI 超分 | [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html) / [Code](https://github.com/yingkai-zhang/UAFL) |
| P12 | IR&ArF | TIP 2025 | P2 | 监督未配准/任意尺度融合 | [Paper](https://doi.org/10.1109/TIP.2025.3551531) / [Code](https://github.com/Jiahuiqu/IR-ArF) |
| P13 | LRTN | IJCV 2025 | P2 | 监督 HSI/MSI 融合 | [Paper](https://link.springer.com/article/10.1007/s11263-024-02203-7) / [Code](https://github.com/YuanyeLiu/Low-Rank-Transformer-For-High-Resolution-Hyperspectral-Computational-Imaging) |
| P14 | OTIAS | AAAI 2025 | P2 | 监督 HSI-MSI 融合 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32275) / [Code](https://github.com/shangqideng/OTIAS) |
| P15 | SSC-HSR | IJCV 2025 | P2 | 监督、未对齐 RGB 参考引导 HSI 超分 | [Paper](https://link.springer.com/article/10.1007/s11263-025-02466-8) / [Code](https://github.com/yingkai-zhang/SSC-HSR) |
| P16 | FeINFN | NeurIPS 2024 | P2 | 监督 HSI–MSI 融合；隐式表示 | [Paper](https://neurips.cc/virtual/2024/poster/96117) / [Code](https://github.com/294coder/Efficient-MIF) |
| P17 | MHF-Net | TPAMI 2022 | P2 | 监督学习；consistent/blind两种退化设置 | [Paper](https://doi.org/10.1109/TPAMI.2020.3015691) / [Code](https://github.com/XieQi2015/MHF-net) |
| P18 | MoG-DCN | TIP 2021 | P2 | 监督、成像模型引导的深度展开融合 | [Paper](https://doi.org/10.1109/TIP.2021.3078058) / [Code](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution) |
| P19 | UAL | CVPR 2020 | P2 | 监督预训练 + 无监督逐样本适配 | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.html) / [Code](https://github.com/JiangtaoNie/UAL) |
| P20 | uSDN | CVPR 2018 | P2 | 无监督耦合自编码器/解混 | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.html) / [Code](https://github.com/aicip/uSDN) |
| P21 | DBIN | ICCV 2019 | P3 | 盲融合，监督学习/迭代估计观察模型；不可等同无监督 | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Blind_Hyperspectral_Image_Fusion_ICCV_2019_paper.pdf) / [Code](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion) |

### B. 高光谱恢复、全色融合与预训练 (9)

| ID | Work | Venue | Priority | Training setting | Paper / Code |
|---|---|---|---|---|---|
| P22 | Hipandas | ICCV 2025 | P1 | 无监督 HSI–PAN 联合去噪与超分 | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Hipandas_Hyperspectral_Image_Joint_Denoising_and_Super-Resolution_by_Image_Fusion_ICCV_2025_paper.html) / [Code](https://github.com/shuangxu96/Hipandas) |
| P23 | LRSDN | TIP 2024 | P1 | 无外部训练数据的 CASSI 自监督重建 | [Paper](https://doi.org/10.1109/TIP.2024.3354127) / [Code](https://github.com/ChenYong1993/LRSDN) |
| P24 | DDS2M | ICCV 2023 | P1 | 只用退化 HSI 的自监督去噪/补全/单图超分；不是双模态融合 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Miao_DDS2M_Self-Supervised_Denoising_Diffusion_Spatio-Spectral_Model_for_Hyperspectral_Image_Restoration_ICCV_2023_paper.pdf) / [Code](https://github.com/miaoyuchun/DDS2M) |
| P25 | GEWDiff | AAAI 2026 | P2 | 监督 HSI 单图超分；扩散模型 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37978) / [Code](https://github.com/zhu-xlab/GEWDiff) |
| P26 | UMNet | AAAI 2026 | P2 | 监督 HSI/PAN 融合；两阶段不确定性训练 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37988) / [Code](https://github.com/EchoPhD/UMNet) |
| P27 | DM-ZS | CVPR 2025 | P2 | HSI/PAN、观测内零样本引导+外部预训练RGB扩散先验 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Hyperspectral_Pansharpening_via_Diffusion_Models_with_Iteratively_Zero-Shot_Guidance_CVPR_2025_paper.html) / [Code](https://github.com/Jin-liangXiao/DM-ZS) |
| P28 | HIR-Diff | CVPR 2024 | P2 | 使用外部预训练扩散先验的无监督 HSI 恢复/单图超分 | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.pdf) / [Code](https://github.com/LiPang/HIRDiff) |
| P29 | UniPAN | ICCV 2025 | P3 | 监督 MS–PAN 融合；跨传感器泛化 | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.html) / [Code](https://github.com/yc-cui/UniPAN) |
| P30 | SpectralGPT | TPAMI 2024 | P3 | 大规模自监督光谱遥感表征预训练；下游理解任务 | [Paper](https://doi.org/10.1109/TPAMI.2024.3362475) / [Code](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT) |

### C. 自监督逆问题、噪声与内部学习 (14)

| ID | Work | Venue | Priority | Training setting | Paper / Code |
|---|---|---|---|---|---|
| P31 | Equivariant Splitting | ICLR 2026 | P1 | 单一不完整观测算子、观测拆分与等变重建 | [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34a5f638617ce32700e235e50dff9c8a-Abstract-Conference.html) / [Code](https://github.com/vsechaud/Equivariant-Splitting) |
| P32 | UNSURE | ICLR 2025 | P1 | 未知噪声水平，仅带噪观测训练 | [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d36dfcdb14473a8526111c221660f2ab-Abstract-Conference.html) / [Code](https://github.com/tachella/unsure) |
| P33 | MOI | NeurIPS 2022 | P1 | 多观测算子的无监督逆问题；MOI | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/203e651b448deba5de5f45430c45ea04-Abstract-Conference.html) / [Code](https://github.com/edongdongchen/MOI) |
| P34 | REI | CVPR 2022 | P1 | 有噪声的不完整观测；Gaussian、Poisson、混合噪声 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Robust_Equivariant_Imaging_A_Fully_Unsupervised_Framework_for_Learning_To_CVPR_2022_paper.html) / [Code](https://github.com/edongdongchen/REI) |
| P35 | EI | ICCV 2021 | P1 | 仅不完整观测，自监督逆问题；原文实验 CT 与图像修复 | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.html) / [Code](https://github.com/edongdongchen/EI) |
| P36 | KernelGAN | NeurIPS 2019 | P1 | 单幅 LR 图像的无监督空间退化核估计 | [Paper](https://proceedings.neurips.cc/paper/2019/hash/5fd0b37cd7dbbb00f97ba6ce92bf5add-Abstract.html) / [Code](https://github.com/sefibk/KernelGAN) |
| P37 | ZSSR | CVPR 2018 | P1 | 单图像内部样本的零样本超分辨率 | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf) / [Code](https://github.com/assafshocher/ZSSR) |
| P38 | DDNM | ICLR 2023 | P2 | 预训练扩散先验的零样本线性逆问题；DDNM/DDNM+ | [Paper](https://openreview.net/pdf?id=mRieQgMtNTQ) / [Code](https://github.com/wyhuai/DDNM) |
| P39 | Neighbor2Neighbor | CVPR 2021 | P2 | 由单幅带噪图像随机邻域子采样构造训练对 | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.html) / [Code](https://github.com/TaoHuang2018/Neighbor2Neighbor) |
| P40 | Noise2Same | NeurIPS 2020 | P2 | 单次带噪观测的自监督去噪 | [Paper](https://proceedings.neurips.cc/paper/2020/hash/ea6b2efbdd4255a9f1b3bbc6399b58f4-Abstract.html) / [Code](https://github.com/divelab/Noise2Same) |
| P41 | Noise2Void | CVPR 2019 | P2 | 单次带噪图像的自监督去噪，盲点/掩码训练 | [Paper](https://arxiv.org/abs/1811.10980) / [Code](https://github.com/juglab/n2v) |
| P42 | DIP | CVPR 2018 | P2 | 单图像优化，无外部预训练数据的神经网络结构先验 | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf) / [Code](https://github.com/DmitryUlyanov/deep-image-prior) |
| P43 | DPS | ICLR 2023 | P3 | 预训练扩散，带噪线性/非线性逆问题 | [Paper](https://arxiv.org/abs/2209.14687) / [Code](https://github.com/DPS2022/diffusion-posterior-sampling) |
| P44 | DDRM | NeurIPS 2022 | P3 | 预训练扩散先验，带噪线性逆问题后验采样 | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf) / [Code](https://github.com/bahjat-kawar/ddrm) |

### D. 特征结构、上采样与表征迁移 (14)

| ID | Work | Venue | Priority | Training setting | Paper / Code |
|---|---|---|---|---|---|
| P45 | CARL | ICLR 2026 | P1 | 自监督光谱表征预训练；下游分类/分割/回归；不是 HSI-MSI 融合重建方法 | [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html) / [Code](https://github.com/IMSY-DKFZ/CARL) |
| P46 | CESST | AAAI 2024 | P1 | 有监督 RGB→HSI 光谱重建；不是 LR-HSI+HR-MSI 融合 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28478) / [Code](https://github.com/AlexYangxx/CESST) |
| P47 | FeatUp | ICLR 2024 | P1 | 自监督高分辨率特征上采样；语义特征任务，不是光谱图像重建 | [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c5601d99ed028448f29d1dae2e4a926d-Abstract-Conference.html) / [Code](https://github.com/mhamilton723/FeatUp) |
| P48 | MST | CVPR 2022 | P1 | 有监督 CASSI 编码孔径光谱压缩重建 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Mask-Guided_Spectral-Wise_Transformer_for_Efficient_Hyperspectral_Image_Reconstruction_CVPR_2022_paper.html) / [Code](https://github.com/caiyuanhao1998/MST) |
| P49 | NAFNet | ECCV 2022 | P1 | 有监督自然图像去噪/去模糊；NAFNet | [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3043_ECCV_2022_paper.php) / [Code](https://github.com/megvii-research/NAFNet) |
| P50 | Restormer | CVPR 2022 | P1 | 有监督自然图像去噪/去模糊/去雨 | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html) / [Code](https://github.com/swz30/Restormer) |
| P51 | HyperSIGMA | TPAMI 2025 | P2 | 高光谱基础模型预训练及多任务适配，包含 HSI 超分/去噪；不是直接 label-free HSI-MSI 融合方案 | [Paper](https://doi.org/10.1109/TPAMI.2025.3557581) / [Code](https://github.com/WHU-Sigma/HyperSIGMA) |
| P52 | MambaIRv2 | CVPR 2025 | P2 | 有监督图像超分/去噪/JPEG恢复 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html) / [Code](https://github.com/csguoh/MambaIR) |
| P53 | MambaIR | ECCV 2024 | P2 | 有监督图像恢复 | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2740_ECCV_2024_paper.php) / [Code](https://github.com/csguoh/MambaIR/tree/mambair) |
| P54 | HAT | CVPR 2023 | P2 | 有监督图像超分辨率；HAT | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html) / [Code](https://github.com/XPixelGroup/HAT) |
| P55 | Uformer | CVPR 2022 | P2 | 有监督自然图像恢复 | [Paper](https://arxiv.org/abs/2106.03106) / [Code](https://github.com/ZhendongWang6/Uformer) |
| P56 | LIIF | CVPR 2021 | P2 | 从已有高分辨率RGB图像合成低分辨率输入的任意尺度重建；作者称自监督SR训练，不能等同于无HR-HSI真值融合 | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html) / [Code](https://github.com/yinboc/liif) |
| P57 | CARAFE | ICCV 2019 | P2 | 内容感知特征上采样；原实验检测/分割/修复为有监督 | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.html) / [Code](https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/carafe.py) |
| P58 | RCAN | ECCV 2018 | P2 | 有监督单图像超分辨率 | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf) / [Code](https://github.com/yulunzhang/RCAN) |

### E. 通道注意力原理与对照 (4)

| ID | Work | Venue | Priority | Training setting | Paper / Code |
|---|---|---|---|---|---|
| P59 | FcaNet | ICCV 2021 | P1 | 监督分类/检测；可迁移注意力 | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.html) / [Code](https://github.com/cfzd/FcaNet) |
| P60 | ECA-Net | CVPR 2020 | P2 | 监督分类/检测；可迁移注意力 | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf) / [Code](https://github.com/BangguWu/ECANet) |
| P61 | CBAM | ECCV 2018 | P2 | 监督分类/检测；可迁移注意力 | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper) / [Code](https://github.com/Jongchan/attention-module) |
| P62 | SENet | CVPR 2018 | P3 | 监督分类；注意力基础 | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) / [Code](https://github.com/hujie-frank/SENet) |

## 逐篇方法、迁移位置与代码依据


### P01 | MossFuse | IJCV 2026

**Unsupervised Hyperspectral Image Super-Resolution via Self-Supervised Modality Decoupling**

Authors: Songcheng Du; Yang Zou; Zixu Wang; Xingyuan Li; Ying Li; Changjing Shang; Qiang Shen

[Paper](https://doi.org/10.1007/s11263-026-02757-8) | [Official code](https://github.com/dusongcheng/MossFuse)

**原作训练设定**: 自监督/无监督 HSI-MSI 融合；含退化估计

**方法核心**: 将模态共享与互补信息分解，使用子空间聚类损失引导分解后再融合。

**对当前 baseline 的迁移建议（研究假设）**: 针对简单 concat+fuse 的冗余，将两路特征拆为共享/私有分支，再设计跨模态一致性与互补保留损失。

**限制与适配边界**: 不能只复制模块而保留 HR-HSI 标签损失称为自监督；环境 README 有重复/缺失版本项。

**已观察到的实现证据**: 已核对公开树：MossFuseNet.py、train.py、test.py、degradation_net.py；README 提供 python test.py 与合成过程。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: arxiv v1 标题 Modality Decoupling is All You Need；v4 明确 Journal reference IJCV134,152(2026)，DOI一致；Springer直读超时。

Verification sources: [Source 1](https://arxiv.org/abs/2412.04802) / [Source 2](https://github.com/dusongcheng/MossFuse)


### P02 | ARGS-Diff | CVPR 2025

**Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model**

Authors: Jian Zhu; He Wang; Yang Xu; Zebin Wu; Zhihui Wei

[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.pdf) | [Official code](https://github.com/Zhu1116/ARGS-Diff)

**原作训练设定**: 只用当前 LR-HSI/MSI 学习；无外部数据

**方法核心**: 分别从 HSI/MSI 学习光谱与空间扩散先验，在低维子空间中重建，采样每步使用自适应观测残差引导。

**对当前 baseline 的迁移建议（研究假设）**: 先迁移可计算的观测残差/数据一致性校正到 CNN 输出或迭代块；完整扩散方案需大改。

**限制与适配边界**: 网络修正 residual 与观测退化 residual 不同；扩散训练与采样开销需实测。

**已观察到的实现证据**: 推理 sample_subspace.py、diffusion/、示例ckpt；训练仓库 ARGS-Diff-train 有 spatial_train.py/spectral_train.py/ guided_diffusion。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 训练仓库 https://github.com/Zhu1116/ARGS-Diff-train；README HR-HSI optional。

Verification sources: [Source 1](https://arxiv.org/abs/2505.11800) / [Source 2](https://github.com/Zhu1116/ARGS-Diff) / [Source 3](https://github.com/Zhu1116/ARGS-Diff-train)


### P03 | CLoRF | IJCV 2025

**Hyperspectral and Multispectral Image Fusion with Arbitrary Resolution Through Self-Supervised Representations**

Authors: Ting Wang; Zipei Yan; Jizhou Li; Xile Zhao; Chao Wang; Michael Ng

[Paper](https://link.springer.com/article/10.1007/s11263-025-02540-1) | [Official code](https://github.com/wangting1907/CLoRF-Fusion)

**原作训练设定**: 自监督连续表示 HSI-MSI 融合

**方法核心**: 连续低秩分解 CLoRF，将空间和光谱连续神经表示与低秩结构结合，按坐标查询任意分辨率。

**对当前 baseline 的迁移建议（研究假设）**: 将 tail 固定网格输出改为低秩系数+光谱基/坐标解码器，探索连续空间尺度与波长建模。

**限制与适配边界**: 属于输出表示和优化框架改造，非普通注意力直接替换；示例需要模拟观测与已知响应。

**已观察到的实现证据**: 公开 inr/、train_tv_spatial.py、train_tv_spatial.sh、utils.py；README 有模拟 LR-HSI/MSI 与执行命令。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: Publisher opened，IJCV133(11):7515-7535；DOI10.1007/s11263-025-02540-1。

Verification sources: [Source 1](https://link.springer.com/article/10.1007/s11263-025-02540-1) / [Source 2](https://github.com/wangting1907/CLoRF-Fusion)


### P04 | SRLF-Net | CVPR 2025

**A Selective Re-learning Mechanism for Hyperspectral Fusion Imaging**

Authors: Yuanye Liu; Jinyang Liu; Renwei Dian; Shutao Li

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html) | [Official code](https://github.com/YuanyeLiu/SRLF-Net)

**原作训练设定**: 监督 HSI/MSI 融合；成像模型引导的选择性细化

**方法核心**: 先初步融合，再根据空间-光谱结构与观测模型识别失真特征点，只对这些点再学习。

**对当前 baseline 的迁移建议（研究假设）**: 在 fuse/body 后增加轻量误差反馈细化；比较全图更新与选择性局部更新。

**限制与适配边界**: 观测引导不等于无真值训练；已有选择性修正方法必须作为新 idea 的近邻。参数/FLOPs 统计要覆盖选择和扫描操作。

**已观察到的实现证据**: deep_select_mamba.py、mambair.py、train_cave.py、train_nc.py、test.py、train_dataloader.py；官方提供实际训练实现。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liu_A_Selective_Re-learning_CVPR_2025_supplemental.pdf) / [Source 2](https://github.com/YuanyeLiu/SRLF-Net)


### P05 | CS2DIPs | TIP 2024

**CS2DIPs: Unsupervised HSI Super-Resolution Using Coupled Spatial and Spectral DIPs**

Authors: Yuan Fang; Yipeng Liu; Chong-Yung Chi; Zhen Long; Ce Zhu

[Paper](https://doi.org/10.1109/TIP.2024.3390582) | [Official code](https://github.com/Ambitionfy/CS2DIPs)

**原作训练设定**: 无需外部训练数据，观测对内优化

**方法核心**: 非负矩阵-向量张量分解结合耦合空间/光谱 DIP，生成丰度与光谱矩阵并物理重组。

**对当前 baseline 的迁移建议（研究假设）**: 用双观测退化重投影损失替代 HR-HSI L1；探索把输出分解为光谱基与空间系数。

**限制与适配边界**: 逐图优化有迭代成本；已知退化/非负混合假设；必须避免以 GT 指标选停止时刻。

**已观察到的实现证据**: demo.py 已打开：总损失明确比较 Y 与 PZ、X 与 H_z(Z)，执行反传；modules/含实现。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: TIP33:3090-3101；作者大学官网公开全文。

Verification sources: [Source 1](https://www.ee.nthu.edu.tw/cychi/publications/journal_articles2011/CS2DIPs_unsupervised_HSI_super-resolution_using_coupled_spatial_and_spectral_DIPs.pdf) / [Source 2](https://github.com/Ambitionfy/CS2DIPs/blob/main/demo.py)


### P06 | UTAL | TPAMI 2024

**Unsupervised Test-Time Adaptation Learning for Effective Hyperspectral Image Super-Resolution With Unknown Degeneration**

Authors: Lei Zhang; Jiangtao Nie; Wei Wei; Yanning Zhang

[Paper](https://doi.org/10.1109/TPAMI.2024.3361894) | [Official code](https://github.com/JiangtaoNie/UAL)

**原作训练设定**: 监督合成数据预训练 + 无监督测试时适配；非全程零标签

**方法核心**: 共享互指导融合先验预训练，测试时自指导适配与未知退化联合估计；元训练与交替优化提高适配效率。

**对当前 baseline 的迁移建议（研究假设）**: 保留现有监督 baseline 作为初始化，加入轻量 adapter 与观测一致性测试时更新，比较跨 PSF/SRF 的泛化。

**限制与适配边界**: 不能宣称无监督从头；每样本适配成本和测试时可用观测需报告。

**已观察到的实现证据**: 公开 Adaptor.py、Train_FusionModel.py、Spa_downs.py、meta-al-UTAL/；README 分清 UAL 与 UTAL/meta 训练流程。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: TPAMI46(7):5008-5025，2024；代码仓库同时含 CVPR2020 UAL。

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/38315602/) / [Source 2](https://github.com/JiangtaoNie/UAL)


### P07 | ZSL | TPAMI 2023

**Zero-Shot Hyperspectral Sharpening**

Authors: Renwei Dian; Anjing Guo; Shutao Li

[Paper](https://doi.org/10.1109/TPAMI.2023.3279050) | [Official code](https://github.com/renweidian/ZSL)

**原作训练设定**: 当前 HSI/MSI 观测构造低尺度训练对；零样本融合

**方法核心**: 先估计空间/光谱响应，再对当前观测降尺度训练，结合降维和成像模型损失。

**对当前 baseline 的迁移建议（研究假设）**: 优先迁移到 train_cave_M.py 的数据构造和损失；可保留 Net，训练目标改为原始 LR-HSI，输入为进一步降尺度的 HSI/MSI。

**限制与适配边界**: 降尺度学习依赖跨尺度规律；原始 LR-HSI 噪声会成为伪目标；官方退化估计需要 MATLAB。

**已观察到的实现证据**: ZSL_Training.py、ZSL_Testing.py、model1.py、demo_kernel_estimation.m；README 给出运行顺序。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/37235456/) / [Source 2](https://github.com/renweidian/ZSL)


### P08 | CUCaNet | ECCV 2020

**Cross-Attention in Coupled Unmixing Nets for Unsupervised Hyperspectral Super-Resolution**

Authors: Jing Yao; Danfeng Hong; Jocelyn Chanussot; Deyu Meng; Xiaoxiang Zhu; Zongben Xu

[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740205.pdf) | [Official code](https://github.com/danfenghong/ECCV2020_CUCaNet)

**原作训练设定**: 无监督盲融合/耦合解混

**方法核心**: 两路卷积自编码器学习基与系数，交叉注意力传递空间光谱信息，用一致性学习 SRF/PSF。

**对当前 baseline 的迁移建议（研究假设）**: 替换 concat-only 融合，令 HSI 与 MSI 互相生成指导信息；加入可约束的 SRF/PSF 模块。

**限制与适配边界**: 交叉注意力与当前全局通道门控不同；光谱解混约束需一并理解。

**已观察到的实现证据**: 公开 model/ 与 Main_CAVE.py；README 提供 Windows Python3.7 PyTorch demo、逐图运行和输出路径。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: ECCV pp208-224；会议官网全文与代码仓库互相对应。

Verification sources: [Source 1](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740205.pdf) / [Source 2](https://github.com/danfenghong/ECCV2020_CUCaNet)


### P09 | CDPS | TIP 2026

**Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion**

Authors: Yang Xu; Jian Zhu; Danfeng Hong; Zhihui Wei; Zebin Wu

[Paper](https://doi.org/10.1109/TIP.2025.3647207) | [Official code](https://github.com/Zhu1116/CDPS-train)

**原作训练设定**: 无监督；仅输入测试对学习扩散先验

**方法核心**: 分别学习光谱与空间扩散先验，并在耦合后验采样中同时约束两类观测保真。

**对当前 baseline 的迁移建议（研究假设）**: 启发双分支先验和双退化观测一致性；CNN 可先做两路不同强度的数据保真反馈。

**限制与适配边界**: 与 ARGS-Diff 相近但不同工作；本文代码链接核对到训练/分支采样仓库，完整联合采样入口还需进一步复现审计。

**已观察到的实现证据**: 公开 guided_diffusion、spatial_train.py、spectral_train.py、spatial_sample.py、spectral_sample.py；README 训练步骤，HRHSI optional。

**代码核验程度**: Training and component sampling observed; full coupled sampling entry not confirmed

**补充核验**: 仓库称 TIP2025 accepted；正式 TIP2026 vol35 pp69-84。不可直接写 TIP2025。

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/41477791/) / [Source 2](https://github.com/Zhu1116/CDPS-train)


### P10 | SHOTUN | TIP 2026

**Self-Expressive High-Order Tensor Unrolling Network for Unsupervised Hyperspectral and Multispectral Image Fusion**

Authors: He Wang; Yang Xu; Zhihui Wei; Zebin Wu

[Paper](https://doi.org/10.1109/TIP.2026.3695389) | [Official code](https://github.com/Shawn-H-Wang/SHOTUN)

**原作训练设定**: 无监督张量展开；未知退化估计含预训练策略

**方法核心**: 重叠图像块间自表达关系形成高阶张量表示，保持空间结构；将交替子问题优化展开为模块。

**对当前 baseline 的迁移建议（研究假设）**: 探索重叠 patch 非局部关系或低秩约束，或把 ResBlock 变为观测校正+学习先验的迭代块。

**限制与适配边界**: 并非直接替换 CA 即复现；高阶张量/非局部 patch 内存和复杂度需评估。

**已观察到的实现证据**: 公开 LowRankNetwork.py、blind.py、两个 unfolded_net_sparse_representation...py；README有执行 demo。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 正式 TIP35:5714-5729；原README旧 BibTeX编号10705122与新 DOI3695389不同，不复制旧编号。

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/42202190/) / [Source 2](https://github.com/Shawn-H-Wang/SHOTUN)


### P11 | UAFL | CVPR 2026

**Enhancing Unregistered Hyperspectral Image Super-Resolution via Unmixing-based Abundance Fusion Learning**

Authors: Yingkai Zhang; Tao Zhang; Jing Nie; Ying Fu

[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html) | [Official code](https://github.com/yingkai-zhang/UAFL)

**原作训练设定**: 监督、未配准 RGB 引导 HSI 超分

**方法核心**: 用 SVD 保留 LR-HSI 光谱基，学习丰度修正；粗细可变形聚合和丰度跨注意力处理未对齐参考。

**对当前 baseline 的迁移建议（研究假设）**: 将 hsi_up+tail(f) 改成固定光谱基乘丰度残差；研究在保持光谱基时注入 MSI 细节。

**限制与适配边界**: SVD基不能自动解释成非负物理端元；低秩截断可能漏掉稀有光谱。原作使用HR目标，需换自监督目标。mmcv_full/mmflow/cupy/torch_dwconv 有编译依赖。

**已观察到的实现证据**: source/、run.py、torchlight/、requirements.txt；2026-07-04已发布实现与示例权重；论文承诺将发布的旧描述不能代表当前状态。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.pdf) / [Source 2](https://github.com/yingkai-zhang/UAFL)


### P12 | IR&ArF | TIP 2025

**IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images**

Authors: Jiahui Qu; Xiaoyang Wu; Wenqian Dong; Jizhou Cui; Yunsong Li

[Paper](https://doi.org/10.1109/TIP.2025.3551531) | [Official code](https://github.com/Jiahuiqu/IR-ArF)

**原作训练设定**: 监督未配准/任意尺度融合

**方法核心**: 构建空间光谱一致的任意尺度观察模型并展开优化，联合处理注册误差与动态分辨率需求。

**对当前 baseline 的迁移建议（研究假设）**: 当前假设完美配准，可在融合前加入对齐与跨尺度查询；自监督版本需改成观测域注册/重投影损失。

**限制与适配边界**: 不能直接照搬训练称为自监督；官方 train.py L258使用HR-HSI真值L1，L263-265也用GT。

**已观察到的实现证据**: model.py/train.py/test.py/dataloader.py 已公开；raw train.py 已核对GT损失。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: TIP34:1934-1949；README标题Towards，正式title Toward。

Verification sources: [Source 1](https://github.com/Jiahuiqu/IR-ArF) / [Source 2](https://raw.githubusercontent.com/Jiahuiqu/IR-ArF/main/train.py)


### P13 | LRTN | IJCV 2025

**Low-Rank Transformer for High-Resolution Hyperspectral Computational Imaging**

Authors: Yuanye Liu; Renwei Dian; Shutao Li

[Paper](https://link.springer.com/article/10.1007/s11263-024-02203-7) | [Official code](https://github.com/YuanyeLiu/Low-Rank-Transformer-For-High-Resolution-Hyperspectral-Computational-Imaging)

**原作训练设定**: 监督 HSI/MSI 融合

**方法核心**: 跨模态互补特征提取、先验引导注意力、空间低秩交叉注意力降低全局建模成本。

**对当前 baseline 的迁移建议（研究假设）**: 研究用低秩跨注意力替换 concat+fuse 或部分 ResBlock，保持双分支和图像残差。

**限制与适配边界**: 作者代码要求测试图像尺寸与训练一致；需要移除固定空间尺寸假设。2024在线、正式卷期2025，按2025计篇。

**已观察到的实现证据**: LRTN.py、LRTN_gf5.py、main.py、train_cave.py、train_dataloader.py；已查看作者仓库。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://link.springer.com/article/10.1007/s11263-024-02203-7) / [Source 2](https://github.com/YuanyeLiu/Low-Rank-Transformer-For-High-Resolution-Hyperspectral-Computational-Imaging)


### P14 | OTIAS | AAAI 2025

**OTIAS: OcTree Implicit Adaptive Sampling for Multispectral and Hyperspectral Image Fusion**

Authors: Shangqi Deng; Jun Ma; Liang-Jian Deng; Ping Wei

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32275) | [Official code](https://github.com/shangqideng/OTIAS)

**原作训练设定**: 监督 HSI-MSI 融合

**方法核心**: 八叉树在空间与通道维建模，按坐标生成与光谱信息有关的自适应插值核。

**对当前 baseline 的迁移建议（研究假设）**: 替换固定双线性 feature interpolation，让上采样同时考虑波段/特征与空间坐标。

**限制与适配边界**: 完整算法并非简单三线性插值；官方用 GT L1，迁入 self-supervised需替换训练目标。

**已观察到的实现证据**: 公开 model/、train.py、eval.py、tools/；raw train.py L98-101确认out与ref监督损失；README提供权重与数据链接。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: AAAI39(3):2708-2716；Publisher DOI10.1609/aaai.v39i3.32275，2025-04-11。

Verification sources: [Source 1](https://ojs.aaai.org/index.php/AAAI/article/view/32275) / [Source 2](https://raw.githubusercontent.com/shangqideng/OTIAS/main/train.py)


### P15 | SSC-HSR | IJCV 2025

**Unaligned RGB Guided Hyperspectral Image Super-Resolution with Spatial-Spectral Concordance**

Authors: Yingkai Zhang; Zeqiang Lai; Tao Zhang; Ying Fu; Chenghu Zhou

[Paper](https://link.springer.com/article/10.1007/s11263-025-02466-8) | [Official code](https://github.com/yingkai-zhang/SSC-HSR)

**原作训练设定**: 监督、未对齐 RGB 参考引导 HSI 超分

**方法核心**: 两阶段对齐、迭代可变形特征聚合与参考引导光谱注意力。

**对当前 baseline 的迁移建议（研究假设）**: 在两路特征拼接前做可靠性加权的参考聚合；真实双传感器错位实验可作为扩展方向。

**限制与适配边界**: 需要几何模型和额外对齐训练；原任务参考 RGB 与目标可存在视差/遮挡。自监督损失也需显式处理坐标变换。

**已观察到的实现证据**: source/、torchlight/、run.py、create_env.sh、requirements.txt；README 提供对齐和融合训练、权重。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://link.springer.com/article/10.1007/s11263-025-02466-8) / [Source 2](https://github.com/yingkai-zhang/SSC-HSR)


### P16 | FeINFN | NeurIPS 2024

**Fourier-enhanced Implicit Neural Fusion Network for Multispectral and Hyperspectral Image Fusion**

Authors: Yu-Jie Liang; Zihan Cao; Shangqi Deng; Hong-Xia Dou; Liang-Jian Deng

[Paper](https://neurips.cc/virtual/2024/poster/96117) | [Official code](https://github.com/294coder/Efficient-MIF)

**原作训练设定**: 监督 HSI–MSI 融合；隐式表示

**方法核心**: 空间/频率隐式融合和 Gabor 激活解码器，改善高频表示。

**对当前 baseline 的迁移建议（研究假设）**: 研究用频域条件分支或坐标解码器替换固定上采样及 tail。

**限制与适配边界**: 不是无 HR 标签训练的证明；README 训练入口名称需与当前仓库核对。

**已观察到的实现证据**: model/FeINFN.py；readmes/FeINFN.md；训练入口与配置

**代码核验程度**: Actual public source observed; not executed or reproduced


### P17 | MHF-Net | TPAMI 2022

**MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion**

Authors: Qi Xie; Minghao Zhou; Qian Zhao; Zongben Xu; Deyu Meng

[Paper](https://doi.org/10.1109/TPAMI.2020.3015691) | [Official code](https://github.com/XieQi2015/MHF-net)

**原作训练设定**: 监督学习；consistent/blind两种退化设置

**方法核心**: 低秩先验与观测退化建模，展开近端梯度算法构造可解释网络。

**对当前 baseline 的迁移建议（研究假设）**: 在 ResBlock 前后显式加入空间/光谱残差校正，降低纯黑盒重构；低秩输出减少通道开销。

**限制与适配边界**: 盲模型仍不是无监督模型；官方为 TensorFlow1.x。

**已观察到的实现证据**: CMHF-net/BMHF-net公开 MHFnet.py、CAVEmain.py、CAVE_dataReader.py；README有训练/测试开关。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: DOI早期在线2020；正式 TPAMI44(3):1457-1473(2022)。CVPR2019是前作不重复计算。

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/32780695/) / [Source 2](https://github.com/XieQi2015/MHF-net/tree/master/CMHF-net)


### P18 | MoG-DCN | TIP 2021

**Model-Guided Deep Hyperspectral Image Super-Resolution**

Authors: Weisheng Dong; Chen Zhou; Fangfang Wu; Jinjian Wu; Guangming Shi; Xin Li

[Paper](https://doi.org/10.1109/TIP.2021.3078058) | [Official code](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution)

**原作训练设定**: 监督、成像模型引导的深度展开融合

**方法核心**: 将基于HSI去噪先验的迭代重建展开，网络内保留观测模型更新。

**对当前 baseline 的迁移建议（研究假设）**: 把当前单次残差输出改成少量观测梯度更新+ResBlock先验更新，研究有限步展开。

**限制与适配边界**: 模型引导不自动等于自监督；原始作者网页404，公开作者实现仍有代码。完整网络更换与轻量迭代分支需分别对照。

**已观察到的实现证据**: sf_8_CAVE/network_31.py、network_31_v1.py、network_31_v2.py、train.py、tst.py，已查看源码树。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/33979283/) / [Source 2](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution/tree/main/sf_8_CAVE)


### P19 | UAL | CVPR 2020

**Unsupervised Adaptation Learning for Hyperspectral Imagery Super-Resolution**

Authors: Lei Zhang; Jiangtao Nie; Wei Wei; Yanning Zhang; Shengcai Liao; Ling Shao

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.html) | [Official code](https://github.com/JiangtaoNie/UAL)

**原作训练设定**: 监督预训练 + 无监督逐样本适配

**方法核心**: 将跨图像的共有先验与当前观测的私有特性分开学习，融合模块预训练后训练适配器。

**对当前 baseline 的迁移建议（研究假设）**: 实现难度较完整 diffusion 低，可将当前 Net 外接自适应修正网络并以重投影损失更新。

**限制与适配边界**: 属于 UTAL 的前作，共享仓库，优先读 TPAMI扩展再回溯。

**已观察到的实现证据**: 公开 Train_FusionModel.py 与 Adaptor.py；README明确两阶段。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: CVF搜索返回官方摘要，直接打开403；不以非官方摘要为证。

Verification sources: [Source 1](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.html) / [Source 2](https://github.com/JiangtaoNie/UAL)


### P20 | uSDN | CVPR 2018

**Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution**

Authors: Ying Qu; Hairong Qi; Chiman Kwan

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.html) | [Official code](https://github.com/aicip/uSDN)

**原作训练设定**: 无监督耦合自编码器/解混

**方法核心**: 用 Dirichlet 表示学习丰度，结合稀疏与光谱一致性，把空间/光谱信息通过共享解码结构连接。

**对当前 baseline 的迁移建议（研究假设）**: 学习自监督观测重构与物理约束基础；可增加非负、和为1的丰度分支作为结构先验。

**限制与适配边界**: TensorFlow1.8旧实现，需迁移；对混合模型与丰度假设有依赖。

**已观察到的实现证据**: 公开 uSDN.py、pbAuto_uSDN.py、utils.py；作者 README 明确执行 uSDN.py 和 CVPR2018引用。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 基础历史论文；CVF直读403，官方作者代码与arxiv可回溯。

Verification sources: [Source 1](https://github.com/aicip/uSDN)


### P21 | DBIN | ICCV 2019

**Deep Blind Hyperspectral Image Fusion**

Authors: Wu Wang; Weihong Zeng; Yue Huang; Xinghao Ding; John Paisley

[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Blind_Hyperspectral_Image_Fusion_ICCV_2019_paper.pdf) | [Official code](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion)

**原作训练设定**: 盲融合，监督学习/迭代估计观察模型；不可等同无监督

**方法核心**: 融合和 PSF/SRF 观察模型估计交替优化，同时追求空间与光谱精度。

**对当前 baseline 的迁移建议（研究假设）**: 设计退化估计网络与 fusion 交替更新，研究固定 PSF/R 在真实传感器失配时的局限。

**限制与适配边界**: 作者仓库同时含 TNNLS2021 EDBIN；旧 TensorFlow依赖；需区分两版本。

**已观察到的实现证据**: 公开 models_boost_res/、models_ibp_sn22/、train_boost_res.py、test_boost_res.py及EDBIN代码。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: CVF官方PDF已搜索核对；说明 blind 不代表 label-free。

Verification sources: [Source 1](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Blind_Hyperspectral_Image_Fusion_ICCV_2019_paper.pdf) / [Source 2](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion)


### P22 | Hipandas | ICCV 2025

**Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image**

Authors: Shuang Xu; Zixiang Zhao; Haowen Bai; Chang Yu; Jiangjun Peng; Xiangyong Cao; Deyu Meng

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Hipandas_Hyperspectral_Image_Joint_Denoising_and_Super-Resolution_by_Image_Fusion_ICCV_2025_paper.html) | [Official code](https://github.com/shuangxu96/Hipandas)

**原作训练设定**: 无监督 HSI–PAN 联合去噪与超分

**方法核心**: 去噪/超分/PAN 重建联合约束，细节低秩先验与两阶段训练。

**对当前 baseline 的迁移建议（研究假设）**: 让残差基线来自去噪后的 LR-HSI；设计 MSI 再投影约束与含噪训练。

**限制与适配边界**: 辅助输入是 PAN 不是多通道 MSI，观测算子需改；两阶段的低分辨率监督须仔细核对。

**已观察到的实现证据**: main.py；model.py；simulate_data.py；utils/

**代码核验程度**: Actual public source observed; not executed or reproduced


### P23 | LRSDN | TIP 2024

**Hyperspectral Compressive Snapshot Reconstruction via Coupled Low-Rank Subspace Representation and Self-Supervised Deep Network**

Authors: Yong Chen; Wenzhen Lai; Wei He; Xi-Le Zhao; Jinshan Zeng

[Paper](https://doi.org/10.1109/TIP.2024.3354127) | [Official code](https://github.com/ChenYong1993/LRSDN)

**原作训练设定**: 无外部训练数据的 CASSI 自监督重建

**方法核心**: 正交光谱基与空间系数低秩分解，DGSAN 学习系数先验，并嵌入 ADMM 交替优化。

**对当前 baseline 的迁移建议（研究假设）**: 将 tail 改为较少系数通道，再与光谱基相乘；用双观测算子重写数据项，学习交替更新与低秩约束。

**限制与适配边界**: CASSI 编码与 HSI/MSI 退化不同；固定秩可能丢失稀有材料；仓库使用混合优化代码且说明较少。

**已观察到的实现证据**: run/main.py、run/dgsan.py、run/net.py、run/downsampler.py、initial/、model/，真实源码目录已打开。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://chenyong1993.github.io/yongchen.github.io/papers/2024/Hyperspectral_Compressive_Snapshot_Reconstruction_via_Coupled_Low-Rank_Subspace_Representation_and_Self-Supervised_Deep_Network.pdf) / [Source 2](https://github.com/ChenYong1993/LRSDN/tree/main/run)


### P24 | DDS2M | ICCV 2023

**DDS2M: Self-Supervised Denoising Diffusion Spatio-Spectral Model for Hyperspectral Image Restoration**

Authors: Yuchun Miao; Lefei Zhang; Liangpei Zhang; Dacheng Tao

[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Miao_DDS2M_Self-Supervised_Denoising_Diffusion_Spatio-Spectral_Model_for_Hyperspectral_Image_Restoration_ICCV_2023_paper.pdf) | [Official code](https://github.com/miaoyuchun/DDS2M)

**原作训练设定**: 只用退化 HSI 的自监督去噪/补全/单图超分；不是双模态融合

**方法核心**: 在反向扩散中用变分空间-光谱模块学习当前图像的后验，空间和光谱网络无需外部训练数据。

**对当前 baseline 的迁移建议（研究假设）**: 研究空间系数与光谱基分解；把其单观测似然替换为 LR-HSI 与 HR-MSI 联合观测，再比较与普通 CNN 内部学习的差异。

**限制与适配边界**: 不能直接复制 main_sisr.py 就得到融合；采样和逐图优化增加时间。CVF 直读遇到403，摘要由作者 arXiv 核验。

**已观察到的实现证据**: main_sisr.py、main_denoising.py、main_completion.py、runners/、models/、functions/；README 给出执行命令。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://arxiv.org/abs/2303.06682) / [Source 2](https://github.com/miaoyuchun/DDS2M)


### P25 | GEWDiff | AAAI 2026

**GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution**

Authors: Sirui Wang; Jiang He; Natàlia Blasco Andreo; Xiao Xiang Zhu

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37978) | [Official code](https://github.com/zhu-xlab/GEWDiff)

**原作训练设定**: 监督 HSI 单图超分；扩散模型

**方法核心**: 小波压缩空间-光谱表示、几何增强扩散、多层重建目标。

**对当前 baseline 的迁移建议（研究假设）**: 先迁移轻量小波高低频分支到 body/tail，比较物理自监督下的细节与光谱误差；完整扩散作为后续路线。

**限制与适配边界**: train.py 读取 img_hr_hf 作为目标，不能标为无HR标签。源码硬编码多GPU和NCCL，Win11单卡直接运行需调整。

**已观察到的实现证据**: model/edm.py、RWT.py、unet3d.py、train.py、test.py、run.sh；已打开raw训练代码，提供数据/权重链接。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://raw.githubusercontent.com/zhu-xlab/GEWDiff/main/train.py) / [Source 2](https://github.com/zhu-xlab/GEWDiff)


### P26 | UMNet | AAAI 2026

**UMNet: Uncertainty-guided Memory Network for Hyperspectral Pansharpening**

Authors: Xiaozheng Wang; Yong Yang; Shuying Huang; Nayu Liu; Ziyang Liu

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37988) | [Official code](https://github.com/EchoPhD/UMNet)

**原作训练设定**: 监督 HSI/PAN 融合；两阶段不确定性训练

**方法核心**: 相邻波段递归融合、NMF状态记忆交互与空间/光谱不确定性损失。

**对当前 baseline 的迁移建议（研究假设）**: 在物理波段而非任意隐藏通道上建递归依赖；研究两观测残差的可靠性权重。

**限制与适配边界**: train.py 显式 outputs-reference 且模型接收 reference，不能直接迁入无标签训练。仓库缺 dataloaders/utils/configs等依赖，README让参照DFCFN，属于部分工程实现。

**已观察到的实现证据**: HFRUNet.py、train.py、trainpre.py 实际存在；已打开train.py确认reference用于损失；About的publish soon文字已过时。

**代码核验程度**: Partial engineering source; dataloaders/utils/configs missing

Verification sources: [Source 1](https://github.com/EchoPhD/UMNet/blob/main/train.py) / [Source 2](https://github.com/EchoPhD/UMNet)


### P27 | DM-ZS | CVPR 2025

**Hyperspectral Pansharpening via Diffusion Models with Iteratively Zero-Shot Guidance**

Authors: Jin-Liang Xiao; Ting-Zhu Huang; Liang-Jian Deng; Guang Lin; Zihan Cao; Chao Li; Qibin Zhao

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Hyperspectral_Pansharpening_via_Diffusion_Models_with_Iteratively_Zero-Shot_Guidance_CVPR_2025_paper.html) | [Official code](https://github.com/Jin-liangXiao/DM-ZS)

**原作训练设定**: HSI/PAN、观测内零样本引导+外部预训练RGB扩散先验

**方法核心**: 用当前观测训练引导网络，反复引导生成RGB细节，再通过神经空间-光谱分解映射回HSI。

**对当前 baseline 的迁移建议（研究假设）**: 学习将空间细节生成与光谱映射分开；先在现有 tail 前加入低维细节分支和双观测约束。

**限制与适配边界**: 不是所有参数仅用当前观测学习；需要外部 diffusion checkpoint。PAN换MSI后需改变观测与引导，不可强迫31波段直接使用RGB扩散。

**已观察到的实现证据**: Demo_main.py、guide_hyperpan.py、model1.py、guided_diffusion/、models/、base.json；README 指定预训练权重与引导训练。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://github.com/Jin-liangXiao/DM-ZS)


### P28 | HIR-Diff | CVPR 2024

**HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models**

Authors: Li Pang; Xiangyu Rui; Long Cui; Hongzhong Wang; Deyu Meng; Xiangyong Cao

[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.pdf) | [Official code](https://github.com/LiPang/HIRDiff)

**原作训练设定**: 使用外部预训练扩散先验的无监督 HSI 恢复/单图超分

**方法核心**: 低维图像与光谱系数矩阵分解，SVD/RRQR估计系数，TV引导低维扩散采样。

**对当前 baseline 的迁移建议（研究假设）**: 学习把光谱映射与空间生成分离；为扩散后验加入MSI观测项，或先做低秩CNN对照。

**限制与适配边界**: 是使用预训练先验的任务零样本方案，不是从零只用当前观测；低维RGB先验可能抹掉异常材料。CVF页面抓取失败，作者arXiv与实现可访问。

**已观察到的实现证据**: main.py、guided_diffusion/、configs/、matlab/、utility/；提供去噪/超分/补全命令与预训练模型链接。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://arxiv.org/abs/2402.15865) / [Source 2](https://github.com/LiPang/HIRDiff)


### P29 | UniPAN | ICCV 2025

**Enpowering Your Pansharpening Models with Generalizability: Unified Distribution is All You Need**

Authors: Yongchuan Cui; Peng Liu; Hui Zhang

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.html) | [Official code](https://github.com/yc-cui/UniPAN)

**原作训练设定**: 监督 MS–PAN 融合；跨传感器泛化

**方法核心**: 训练和测试映射到统一像素分布以缓解传感器分布偏移。

**对当前 baseline 的迁移建议（研究假设）**: 数据预处理/反变换对照；检验固定归一化和不同 SRF 的泛化。

**限制与适配边界**: 逐波段映射可能改变真实光谱比例，必须保留可逆变换并检验 SAM；不是直接 HSI 方法。

**已观察到的实现证据**: examples/FeINFN.py 等12模型训练测试；env.yaml

**代码核验程度**: Actual public source observed; not executed or reproduced


### P30 | SpectralGPT | TPAMI 2024

**SpectralGPT: Spectral Remote Sensing Foundation Model**

Authors: Danfeng Hong et al.

[Paper](https://doi.org/10.1109/TPAMI.2024.3362475) | [Official code](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)

**原作训练设定**: 大规模自监督光谱遥感表征预训练；下游理解任务

**方法核心**: 3D空间-光谱token、遮挡重建、多目标光谱学习和渐进式预训练。

**对当前 baseline 的迁移建议（研究假设）**: 借鉴光谱遮挡预训练 hsi_head，或冻结编码器加轻量适配；重建训练仍需双观测一致性。

**限制与适配边界**: 基础模型的预训练并非当前图像对零样本；CAVE与卫星MSI域差异显著。完整预训练成本大，官方示例8张4090；不能据此推断单卡无法适配。

**已观察到的实现证据**: main_pretrain.py、engine_pretrain.py、models_mae_spectral.py、models_vit_tensor.py、downstream_tasks/；有真实预训练和下游入口。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://www.bgd.ed.tum.de/pdf/2024_SpectralGPT_Hao.pdf) / [Source 2](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)


### P31 | Equivariant Splitting | ICLR 2026

**Equivariant Splitting: Self-supervised learning from incomplete data**

Authors: Victor Sechaud; Jérémy Scanvic; Quentin Barthélemy; Patrice Abry; Julián Tachella

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34a5f638617ce32700e235e50dff9c8a-Abstract-Conference.html) | [Official code](https://github.com/vsechaud/Equivariant-Splitting)

**原作训练设定**: 单一不完整观测算子、观测拆分与等变重建

**方法核心**: 结合观测拆分损失与新的重建等变定义，在给定假设下得到监督损失的无偏估计。

**对当前 baseline 的迁移建议（研究假设）**: 推断：对 HSI/MSI 联合观测设计留出与重投影，并测试等变网络/训练约束；优先学习自监督损失实现。

**限制与适配边界**: 理论假设不自动覆盖高光谱融合；训练时被留出的目标不能通过 MSI、插值基线或跨模态交互泄漏。

**已观察到的实现证据**: 官方仓库 mri/、tomography/、cs+inp/ 均有训练说明；README 明确 run_train.py 调用 train.py，并链接 DeepInverse 支持。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 正式 ICLR 2026 论文，官方 proceedings 与实际代码双重确认；OpenReview 页面有浏览器验证，使用 proceedings 核验。


### P32 | UNSURE | ICLR 2025

**UNSURE: self-supervised learning with Unknown Noise level and Stein's Unbiased Risk Estimate**

Authors: Julián Tachella; Mike Davies; Laurent Jacques

[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d36dfcdb14473a8526111c221660f2ab-Abstract-Conference.html) | [Official code](https://github.com/tachella/unsure)

**原作训练设定**: 未知噪声水平，仅带噪观测训练

**方法核心**: 通过约束/对偶优化构造无需预先知道噪声水平的 SURE 类学习目标。

**对当前 baseline 的迁移建议（研究假设）**: 推断：对 HSI/MSI 观测分别建风险估计，研究波段相关与异方差噪声；修改 loss 与训练循环，架构可先保持。

**限制与适配边界**: 未知噪声水平不等于任意未知噪声；须选择适用变体与协方差参数化，纯风险估计也不自动解决融合的联合零空间。

**已观察到的实现证据**: 实际官方仓库有 unsure.py、main.py、models.py、generate_dataset.py；README 链接 DeepInverse 损失及可运行 notebook。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 官方 ICLR proceedings 已确认；README 标题曾使用简写 Unknown Noise level Stein's Unbiased Risk Estimator，应以 proceedings 全题为准。


### P33 | MOI | NeurIPS 2022

**Unsupervised Learning From Incomplete Measurements for Inverse Problems**

Authors: Julián Tachella; Dongdong Chen; Mike Davies

[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/203e651b448deba5de5f45430c45ea04-Abstract-Conference.html) | [Official code](https://github.com/edongdongchen/MOI)

**原作训练设定**: 多观测算子的无监督逆问题；MOI

**方法核心**: 分析不同算子、测量数和信号模型维度的可辨识条件，并以跨算子重建构造无监督训练损失。

**对当前 baseline 的迁移建议（研究假设）**: 推断：把不同传感器 SRF/PSF、尺度和采样模式作为条件，研究跨算子观测约束与退化条件化网络。

**限制与适配边界**: 双模态输入不自动满足 MOI 的所有条件；需要有真实或合法获得的不同算子测量，不能任意编造观测以声称识别保证。

**已观察到的实现证据**: 官方 MOI 仓库存在 moi/、physics/、demo_train.py、demo_test.py；README 指向 moi.py。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 高度适合理解自监督融合的可辨识性与传感器泛化；不同论文同作者不代表同一训练设置。


### P34 | REI | CVPR 2022

**Robust Equivariant Imaging: A Fully Unsupervised Framework for Learning to Image From Noisy and Partial Measurements**

Authors: Dongdong Chen; Julián Tachella; Mike E. Davies

[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Robust_Equivariant_Imaging_A_Fully_Unsupervised_Framework_for_Learning_To_CVPR_2022_paper.html) | [Official code](https://github.com/edongdongchen/REI)

**原作训练设定**: 有噪声的不完整观测；Gaussian、Poisson、混合噪声

**方法核心**: 用 SURE 修正带噪测量一致性，同时加入鲁棒等变约束，避免把噪声当作准确目标。

**对当前 baseline 的迁移建议（研究假设）**: 推断：为 HSI/MSI 两传感器分别建噪声模型，在训练脚本中替换简单观测 L1，并保留双输入 Net。

**限制与适配边界**: 噪声模型和参数必须适用；偏置、条带及跨波段相关噪声不能一律用独立白噪声处理；仍受 EI 可辨识条件限制。

**已观察到的实现证据**: 官方仓库明确核心 rei_end2end.py 与 rei_end2end_ct.py，存在 rei/、demo_scripts/ 和训练测试入口。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 论文 PDF 一次访问被 CVF 403 阻止；arXiv 摘要与官方仓库提供同一论文和代码；代码存在，未实际运行。


### P35 | EI | ICCV 2021

**Equivariant Imaging: Learning Beyond the Range Space**

Authors: Dongdong Chen; Julián Tachella; Mike E. Davies

[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.html) | [Official code](https://github.com/edongdongchen/EI)

**原作训练设定**: 仅不完整观测，自监督逆问题；原文实验 CT 与图像修复

**方法核心**: 将观测一致性与重建结果的群变换等变性结合，为观测算子零空间引入训练约束。

**对当前 baseline 的迁移建议（研究假设）**: 推断：保留 Net 双输入，将联合算子 A(X)=(空间模糊降采样 X, 光谱响应 R X) 写成可微模块，添加 f(A(T f(y)))≈T f(y) 的等变循环。

**限制与适配边界**: 必须核对联合算子与变换群的可辨识性；普通翻转/旋转可能不增加低通与光谱缺失的信息。不是直接照搬旋转损失便保证高光谱恢复。

**已观察到的实现证据**: 官方 README 指定核心 ei/closure/ei.py；仓库存在 get_started.py、demo_train.py、physics/、transforms/。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: EI 是 ICCV 2021，不能误记为 ICML。原文没有验证该 HSI+MSI baseline。


### P36 | KernelGAN | NeurIPS 2019

**Blind Super-Resolution Kernel Estimation Using an Internal-GAN**

Authors: Sefi Bell-Kligler; Assaf Shocher; Michal Irani

[Paper](https://proceedings.neurips.cc/paper/2019/hash/5fd0b37cd7dbbb00f97ba6ce92bf5add-Abstract.html) | [Official code](https://github.com/sefibk/KernelGAN)

**原作训练设定**: 单幅 LR 图像的无监督空间退化核估计

**方法核心**: 通过跨尺度 patch 分布匹配，用图像内部 GAN 学习图像特定的降采样核。

**对当前 baseline 的迁移建议（研究假设）**: 推断：为固定 PSF 的 baseline 增加可学习空间退化核，结合 HSI/MSI 两路观测一致性，研究盲融合。

**限制与适配边界**: 只估空间核，不直接估光谱响应 R；小图块、噪声、跨尺度分布不匹配会导致不稳定，联合估计存在退化解需约束。

**已观察到的实现证据**: 官方仓库存在 kernelGAN.py、learner.py、networks.py、loss.py、train.py，README 明确 NeurIPS 2019。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 为 PSF 加非负/和为一/支撑/中心约束及独立核精度评估；迁移建议是推断。


### P37 | ZSSR | CVPR 2018

**"Zero-Shot" Super-Resolution Using Deep Internal Learning**

Authors: Assaf Shocher; Nadav Cohen; Michal Irani

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf) | [Official code](https://github.com/assafshocher/ZSSR)

**原作训练设定**: 单图像内部样本的零样本超分辨率

**方法核心**: 从当前 LR 图像进一步降采样构造内部训练对，以图像内部跨尺度重复训练小网络。

**对当前 baseline 的迁移建议（研究假设）**: 推断：构造低一级 HSI/MSI 配对尺度，用观测自身建立 reduced-resolution 自监督融合训练，再应用到原尺度。

**限制与适配边界**: 内部重复性与跨尺度退化一致性可能不成立；原图像内容变成低尺度标签不能等同目标 HR-HSI 真值；噪声和 aliasing 会传播。

**已观察到的实现证据**: 官方仓库真实存在 ZSSR.py、run_ZSSR.py、run_ZSSR_single_input.py、imresize.py，README 明确 CVPR 2018。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 官方旧代码 TensorFlow；适合借鉴数据构造思路，不能原样接入双输入 torch Net。


### P38 | DDNM | ICLR 2023

**Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model**

Authors: Yinhuai Wang; Jiwen Yu; Jian Zhang

[Paper](https://openreview.net/pdf?id=mRieQgMtNTQ) | [Official code](https://github.com/wyhuai/DDNM)

**原作训练设定**: 预训练扩散先验的零样本线性逆问题；DDNM/DDNM+

**方法核心**: 将解分为观测确定的值域与先验决定的零空间，在反向扩散中校正测量一致性。

**对当前 baseline 的迁移建议（研究假设）**: 推断：先借鉴输出投影/数据一致性层而非迁入整个扩散模型；联合算子为 HSI 空间降采样和 MSI 光谱降维，修正其共同零空间。

**限制与适配边界**: RGB 预训练先验不能直接处理 31 波段，任务零样本不等于无需干净先验训练数据；噪声下硬投影会拟合噪声，联合伪逆计算有成本。

**已观察到的实现证据**: 作者仓库含 main.py、functions/、guided_diffusion/、configs/，提供 SVD 与 simplified 两实现；论文直接给作者代码链接。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 当前插值基线加残差不保证观测一致性。作者拼写 GitHub/arXiv 为 Yinhuai，OpenReview PDF 文本曾显示 Yinhui，须引用规范 BibTeX。


### P39 | Neighbor2Neighbor | CVPR 2021

**Neighbor2Neighbor: Self-Supervised Denoising From Single Noisy Images**

Authors: Tao Huang; Songjiang Li; Xu Jia; Huchuan Lu; Jianzhuang Liu

[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.html) | [Official code](https://github.com/TaoHuang2018/Neighbor2Neighbor)

**原作训练设定**: 由单幅带噪图像随机邻域子采样构造训练对

**方法核心**: 相邻像素近似共享干净信号，随机邻域配对加正则项，训练普通去噪网络。

**对当前 baseline 的迁移建议（研究假设）**: 推断：先研究 LR-HSI 或 MSI 的自监督去噪，再将子采样一致性适配双模态；代码便于学习随机 mask。

**限制与适配边界**: 邻近像素并不严格相同，边缘/纹理有偏差；相关噪声破坏配对假设；它不直接生成 HR-HSI 真值。

**已观察到的实现证据**: 官方 PyTorch 仓库存在 train.py、arch_unet.py、dataset_tool.py；README 明确 CVPR 2021，DeepInverse 官方文档也注明从此仓库适配。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 不要误用搜索结果中的第三方 pminhtam/Neigh2Neigh；本记录使用作者仓库。


### P40 | Noise2Same | NeurIPS 2020

**Noise2Same: Optimizing A Self-Supervised Bound for Image Denoising**

Authors: Yaochen Xie; Zhengyang Wang; Shuiwang Ji

[Paper](https://proceedings.neurips.cc/paper/2020/hash/ea6b2efbdd4255a9f1b3bbc6399b58f4-Abstract.html) | [Official code](https://github.com/divelab/Noise2Same)

**原作训练设定**: 单次带噪观测的自监督去噪

**方法核心**: 优化监督去噪风险的自监督上界，以重建项和掩码一致性项减轻严格 J-invariant 架构要求。

**对当前 baseline 的迁移建议（研究假设）**: 推断：baseline 原样/掩码输入共享权重，构造观测域预测一致性，研究保留残差结构时的噪声泄漏控制。

**限制与适配边界**: 不是一般融合损失的无条件替代；上界和噪声假设要重新核对；不能只对两个输出强制相同而没有非平凡约束。

**已观察到的实现证据**: 官方 TensorFlow 仓库有 models.py、network.py、network_configure.py、notebook；README 给出双前向及损失伪代码。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 移植损失思想较容易，原代码框架与当前 torch 不同。


### P41 | Noise2Void | CVPR 2019

**Noise2Void - Learning Denoising From Single Noisy Images**

Authors: Alexander Krull; Tim-Oliver Buchholz; Florian Jug

[Paper](https://arxiv.org/abs/1811.10980) | [Official code](https://github.com/juglab/n2v)

**原作训练设定**: 单次带噪图像的自监督去噪，盲点/掩码训练

**方法核心**: 遮蔽待预测像素，用邻域预测它，以带噪像素为训练目标而不直接复制其噪声。

**对当前 baseline 的迁移建议（研究假设）**: 推断：在 HSI 去噪分支或观测域构造掩码重建，为无干净标签的融合增加抗噪训练。

**限制与适配边界**: 需要合适的噪声独立性/零条件均值假设；HSI 跨波段相关噪声与 MSI 交叉泄漏会破坏简单盲点；原始残差基线若包含目标像素可绕过掩码。

**已观察到的实现证据**: 作者 juglab/n2v 实际包含 n2v/models/n2v_standard.py，代码实现 N2V_DataWrapper 掩码数据；官方 README 引用 CVPR 2019。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 官方实现 TensorFlow/CSBDeep，不能直接当 PyTorch 模块粘贴；当前仓库还包含 N2V2，复现时需区分版本。


### P42 | DIP | CVPR 2018

**Deep Image Prior**

Authors: Dmitry Ulyanov; Andrea Vedaldi; Victor Lempitsky

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf) | [Official code](https://github.com/DmitryUlyanov/deep-image-prior)

**原作训练设定**: 单图像优化，无外部预训练数据的神经网络结构先验

**方法核心**: 随机初始化的生成网络本身带有图像先验，通过观测拟合优化参数进行恢复。

**对当前 baseline 的迁移建议（研究假设）**: 推断：用双观测退化一致性做每场景优化/适配对照；保留 baseline 时称场景自适应，若严格实现 DIP 需改成固定随机输入生成器。

**限制与适配边界**: 原 baseline 条件双输入并非严格 DIP；单场景迭代慢、可能过拟合噪声，早停必须不依赖 HR 真值。

**已观察到的实现证据**: 作者项目页直接链接官方仓库，实际存在 denoising.ipynb、super-resolution.ipynb、flash-no-flash.ipynb 与 models/。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 同题扩展版发表于 IJCV 2020, 128(7):1867–1888, DOI 10.1007/s11263-020-01303-4；同一工作家族不要重复计数。作者 Oxford PDF https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/ulyanov20deep.pdf 。


### P43 | DPS | ICLR 2023

**Diffusion Posterior Sampling for General Noisy Inverse Problems**

Authors: Hyungjin Chung; Jeongsol Kim; Michael Thompson McCann; Marc Louis Klasky; Jong Chul Ye

[Paper](https://arxiv.org/abs/2209.14687) | [Official code](https://github.com/DPS2022/diffusion-posterior-sampling)

**原作训练设定**: 预训练扩散，带噪线性/非线性逆问题

**方法核心**: 以近似测量似然梯度引导扩散后验采样，避免在带噪场景强制硬一致性。

**对当前 baseline 的迁移建议（研究假设）**: 推断：为联合 HSI/MSI 建可微测量似然，比较软/硬数据一致性；进一步可探索 baseline 输出作为迭代初始化。

**限制与适配边界**: 需要适配的先验与传感器噪声模型，采样成本高且可能产生光谱失真，不能当作一个普通注意力模块直接插入。

**已观察到的实现证据**: 官方 PyTorch 仓库存在 sample_condition.py、guided_diffusion/condition_methods.py、guided_diffusion/measurements.py 和 task configs。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 实际会议 ICLR 2023，某些二手列表误标 CVPR 2023，不应沿用。


### P44 | DDRM | NeurIPS 2022

**Denoising Diffusion Restoration Models**

Authors: Bahjat Kawar; Michael Elad; Stefano Ermon; Jiaming Song

[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf) | [Official code](https://github.com/bahjat-kawar/ddrm)

**原作训练设定**: 预训练扩散先验，带噪线性逆问题后验采样

**方法核心**: 在退化算子的谱分解空间中按噪声水平调整扩散恢复，避免特定任务的配对再训练。

**对当前 baseline 的迁移建议（研究假设）**: 推断：理解双传感器联合算子的可观测/不可观测成分，为 baseline 引入噪声感知的一致性修正；全扩散方案属于大改。

**限制与适配边界**: 需预训练生成先验、已知算子和噪声信息；RGB checkpoints 不支持原生 HSI；谱分解与多步采样开销，感知合理不等于光谱准确。

**已观察到的实现证据**: 官方仓库存在 main.py、runners/、functions/、guided_diffusion/，README 提供预训练模型和采样命令，明确 NeurIPS 2022。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 归类为预训练先验迁移，不归类为从当前 HSI/MSI 观测零数据训练。


### P45 | CARL | ICLR 2026

**CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis**

Authors: Alexander Baumann, Leonardo Ayala, Silvia Seidlitz, Jan Sellner, Alexander Studier-Fischer, Berkin Özdemir, Lena Maier-Hein, Slobodan Ilic

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html) | [Official code](https://github.com/IMSY-DKFZ/CARL)

**原作训练设定**: 自监督光谱表征预训练；下游分类/分割/回归；不是 HSI-MSI 融合重建方法

**方法核心**: 以 self-attention/cross-attention 光谱编码器将不同波长及通道数的 RGB/MSI/HSI 转成可共享表征，采用特征层面的时空光谱自监督。

**对当前 baseline 的迁移建议（研究假设）**: 优先研究替换 hsi_head 的光谱编码与两路共享表征；结合不同 R 的模拟 MSI 做跨传感器一致性预训练，再保留现有 tail 重建。

**限制与适配边界**: 当前 baseline 固定31/3通道；需额外提供真实波长，且分类/分割特征的传感器不变性不必然保留精确光谱幅值。需以观测一致性防止丢失光谱细节。

**已观察到的实现证据**: 官方 proceedings 明确链接作者仓库；已打开仓库，实际存在 carl/、main_ssl.py、main_cls.py、main_seg.py、configs/，提供 SSL checkpoint 和训练入口。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 2026 正式会议记录已核实。不是仅 arXiv。未执行代码；不能将其下游指标当成融合证据。


### P46 | CESST | AAAI 2024

**Hyperspectral Image Reconstruction via Combinatorial Embedding of Cross-Channel Spatio-Spectral Clues**

Authors: Xingxing Yang, Jie Chen, Zaifeng Yang

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28478) | [Official code](https://github.com/AlexYangxx/CESST)

**原作训练设定**: 有监督 RGB→HSI 光谱重建；不是 LR-HSI+HR-MSI 融合

**方法核心**: 分别构建色彩通道的光谱嵌入，组合挖掘跨通道线索，以时空光谱注意力和 spectrum-fusion attention 融合互补信息。

**对当前 baseline 的迁移建议（研究假设）**: 研究 spectrum-fusion attention 替换 Concat+fuse，或为 msi_head 分通道生成光谱假设并以 LR-HSI 锚定；融合后 body 引入时空光谱注意力。

**限制与适配边界**: RGB 色彩通道假设与任意 MSI 响应不等价；要适配 R、波段数和双路输入。仓库公开了模型/损失，但当前目录未见完整训练入口。

**已观察到的实现证据**: AAAI 正式页直接指向作者仓库；已逐层打开 train_code/architecture/，存在 CESST.py、__init__.py，并有 Loss.py 和 utils.py。

**代码核验程度**: Partial model/loss source; full training entry not observed

**补充核验**: 源代码已释放但属于部分训练实现；适合模块学习，不承诺开箱复现整篇。DOI 10.1609/aaai.v38i7.28478。


### P47 | FeatUp | ICLR 2024

**FeatUp: A Model-Agnostic Framework for Features at Any Resolution**

Authors: Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, William T. Freeman

[Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c5601d99ed028448f29d1dae2e4a926d-Abstract-Conference.html) | [Official code](https://github.com/mhamilton723/FeatUp)

**原作训练设定**: 自监督高分辨率特征上采样；语义特征任务，不是光谱图像重建

**方法核心**: 使用多视图一致性恢复特征空间分辨率，提供逐图 implicit upsampler 与通用 joint bilateral upsampler。

**对当前 baseline 的迁移建议（研究假设）**: 研究用HR-MSI指导LR-HSI特征上采样，替换 f_hsi 的固定插值；多视图变换前后上下采样一致性作为辅助自监督信号。

**限制与适配边界**: RGB语义特征上的边缘引导不保证光谱保真；需改为HSI/MSI特征与物理观测损失。仓库仍标记自训JBU入口Coming Soon，不宣称所有训练路径完整。

**已观察到的实现证据**: 已打开 official repo，featup/ 源文件、hubconf.py、train_implicit_upsampler.py 路径及预训练JBU可用；ICLR proceedings 正式页面核实。

**代码核验程度**: Implicit trainer/JBU inference observed; self-training JBU marked Coming Soon

**补充核验**: 可学习现有隐式训练/推理代码，通用JBU自行训练完整性有公开限制；JBU使用自定义CUDA内核。已读 arXiv HTML 方法3节验证多视图一致性；ICLR完整PDF因16.9MB超过抓取限制，HTML可访问。


### P48 | MST | CVPR 2022

**Mask-Guided Spectral-Wise Transformer for Efficient Hyperspectral Image Reconstruction**

Authors: Yuanhao Cai, Jing Lin, Xiaowan Hu, Haoqian Wang, Xin Yuan, Yulun Zhang, Radu Timofte, Luc Van Gool

[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Mask-Guided_Spectral-Wise_Transformer_for_Efficient_Hyperspectral_Image_Reconstruction_CVPR_2022_paper.html) | [Official code](https://github.com/caiyuanhao1998/MST)

**原作训练设定**: 有监督 CASSI 编码孔径光谱压缩重建

**方法核心**: S-MSA 将光谱特征作为 token，在光谱维计算注意力；物理编码 mask 引导高保真区域。

**对当前 baseline 的迁移建议（研究假设）**: 提取 S-MSA 替换 ChannelAttention 或部分 ResBlock；原来每通道单个权重升级为特征间交互，配合 LR-HSI 光谱约束。

**限制与适配边界**: CASSI 物理 mask 不是 MAE 随机遮挡，也不是融合中的 SRF/PSF；不能直接照搬测量入口、mask-guidance 和训练损失。64隐藏通道不等于31物理波段。

**已观察到的实现证据**: CVF 正式页直接列官方 repo；repo 有 simulation/、real/、requirements.txt 和训练/测试教程，官方新闻明确 MST 代码和权重释放。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: MST 本体是 CVPR 主会；MST++ 是 CVPR Workshops 2022，不能混淆。


### P49 | NAFNet | ECCV 2022

**Simple Baselines for Image Restoration**

Authors: Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun

[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3043_ECCV_2022_paper.php) | [Official code](https://github.com/megvii-research/NAFNet)

**原作训练设定**: 有监督自然图像去噪/去模糊；NAFNet

**方法核心**: SimpleGate 特征乘法与 simplified channel attention，去除常用非线性激活；残差缩放 beta/gamma 零初始化，构成简洁高效 block。

**对当前 baseline 的迁移建议（研究假设）**: 替换 ResBlock 的 Conv-GELU-Conv+CA 为 NAFBlock；单独比较 SimpleGate、SCA 和残差缩放，适合搭建更强且轻的结构参照。

**限制与适配边界**: 去掉显式激活不意味着整个映射线性：特征相乘依然非线性。原训练有 clean target；移植不会自动成为自监督融合。

**已观察到的实现证据**: ECVA 正式页链接作者仓库；已打开 NAFNet_arch.py，含 SimpleGate、NAFBlock、sca、beta/gamma 和训练文件。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 对当前 LayerNorm+MLP+Sigmoid 注意力很直接的容量/成本对照。


### P50 | Restormer | CVPR 2022

**Restormer: Efficient Transformer for High-Resolution Image Restoration**

Authors: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang

[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html) | [Official code](https://github.com/swz30/Restormer)

**原作训练设定**: 有监督自然图像去噪/去模糊/去雨

**方法核心**: MDTA 使用沿通道的转置多头注意力，depthwise 卷积补局部信息；GDFN 用乘法门控处理特征，避免空间全局注意力的平方开销。

**对当前 baseline 的迁移建议（研究假设）**: 用一个 MDTA+GDFN block 替换部分 ResBlock，在64通道融合特征上处理；也可迁移 GDFN 到 fuse 后，保留图像残差与物理自监督损失。

**限制与适配边界**: 它学习的是隐藏通道交互，不能自动称为物理波长注意力。架构本身不产生自监督信号，必须改变训练目标。

**已观察到的实现证据**: 已打开 basicsr/models/archs/restormer_arch.py，真实源包含 Attention、FeedForward、TransformerBlock；官方有训练代码及四类任务配置和权重。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 比完整搬运 U 型网络更适合首先做单块迁移。自监督使用需重新训练。


### P51 | HyperSIGMA | TPAMI 2025

**HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model**

Authors: Di Wang, Meiqi Hu, Yao Jin, Yuchun Miao, Jiaqi Yang, Yichu Xu, Xiaolei Qin, Jiaqi Ma, Lingyu Sun, Chenxing Li, Chuan Fu, Hongruixuan Chen, Chengxi Han, Naoto Yokoya, Jing Zhang, Minqiang Xu, Lin Liu, Lefei Zhang, Chen Wu, Bo Du, Dacheng Tao, Liangpei Zhang

[Paper](https://doi.org/10.1109/TPAMI.2025.3557581) | [Official code](https://github.com/WHU-Sigma/HyperSIGMA)

**原作训练设定**: 高光谱基础模型预训练及多任务适配，包含 HSI 超分/去噪；不是直接 label-free HSI-MSI 融合方案

**方法核心**: 稀疏采样注意力 SSA 缓解高光谱空间与光谱冗余，空间与光谱双表征通过光谱增强模块结合。

**对当前 baseline 的迁移建议（研究假设）**: 学习 spectral enhancement 与 SSA，作为 hsi_head 或 body 的适配分支；先冻结小规模预训练编码器加轻量 adapter，与从头训练对照。

**限制与适配边界**: 模型规模、卫星预训练数据与 CAVE 近景光谱域差距；全量预训练成本高。预训练超分代码并不等于已有自监督融合。

**已观察到的实现证据**: 已打开官方仓库：Pretrain/、ImageSuperResolution/models/、main38_houston.py、train_houston.sh、ImageDenoising/；官方 citation 给出 TPAMI 47(8),6427-6444 与 DOI。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: IEEE 页面抓取正文受限，用作者仓库正式 citation 与 arXiv 核对；代码源实际存在，未运行。


### P52 | MambaIRv2 | CVPR 2025

**MambaIRv2: Attentive State Space Restoration**

Authors: Hang Guo, Yong Guo, Yaohua Zha, Yulun Zhang, Wenbo Li, Tao Dai, Shu-Tao Xia, Yawei Li

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html) | [Official code](https://github.com/csguoh/MambaIR)

**原作训练设定**: 有监督图像超分/去噪/JPEG恢复

**方法核心**: Attentive state-space 为扫描引入非因果信息，semantic-guided neighboring 让远距离相似像素交互，缓解单向扫描局限。

**对当前 baseline 的迁移建议（研究假设）**: 将部分 body 替换为轻量v2 blocks，比较局部CNN/通道门控/全局扫描；融合场景可研究以MSI结构指导邻居选择。

**限制与适配边界**: MSI引导邻居是迁移假设，不是原作验证结论；自然图像语义相似不保证光谱相同，错误邻居可能引入光谱污染。依赖特殊算子。

**已观察到的实现证据**: CVF正式页与作者repo一致；实际 basicsr/archs/mambairv2_arch.py、mambairv2light_arch.py 和options/train/mambairv2存在；2024-12-04已发布代码。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 用于结构迁移与低开销对照；不能直接替代自监督损失设计。


### P53 | MambaIR | ECCV 2024

**MambaIR: A Simple Baseline for Image Restoration with State-Space Model**

Authors: Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, Shu-Tao Xia

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2740_ECCV_2024_paper.php) | [Official code](https://github.com/csguoh/MambaIR/tree/mambair)

**原作训练设定**: 有监督图像恢复

**方法核心**: Residual State Space Block 结合选择性状态空间扫描、局部卷积增强与通道注意力，建模长程信息。

**对当前 baseline 的迁移建议（研究假设）**: 替换 body 中少数 ResBlock，保持hsi_head/msi_head/fuse/tail，检查长程空间依赖是否补足平均池化门控；先只做空间扫描。

**限制与适配边界**: 不自动建模物理波长，也不提供label-free学习信号。需要mamba_ssm/causal_conv1d等，官方主要测试Ubuntu，Windows现有环境需重新验证。

**已观察到的实现证据**: ECVA正式页列代码；已打开作者repo的mambair分支和main/basicsr/archs，有mambair_arch.py与训练/测试配置。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 官方2024-08-19声明旧thop复杂度统计漏项，已更新fvcore计数；比较效率应使用修正后实现。


### P54 | HAT | CVPR 2023

**Activating More Pixels in Image Super-Resolution Transformer**

Authors: Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, Chao Dong

[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html) | [Official code](https://github.com/XPixelGroup/HAT)

**原作训练设定**: 有监督图像超分辨率；HAT

**方法核心**: 结合通道注意力与窗口空间 self-attention，并使用 overlapping cross-attention 促进跨窗口交互。

**对当前 baseline 的迁移建议（研究假设）**: 在部分 ResBlock 中增加窗口空间交互，保留原通道注意力；用重叠窗口降低局部块边界限制，检查空间细节与 SAM。

**限制与适配边界**: overlapping cross-attention 是跨窗口，不是自动的 HSI-MSI 跨模态注意力；若改为双路 Q/K/V 属于新的迁移设计。全量 HAT 开销高。

**已观察到的实现证据**: CVF 正式页列作者仓库；已打开 hat/archs/，存在 hat_arch.py 等实现；官方 repo 提供训练配置与预训练模型。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 同repo也含 TPAMI 延伸稿；这里仅计 CVPR2023 原作，避免重复计数。


### P55 | Uformer | CVPR 2022

**Uformer: A General U-Shaped Transformer for Image Restoration**

Authors: Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, Houqiang Li

[Paper](https://arxiv.org/abs/2106.03106) | [Official code](https://github.com/ZhendongWang6/Uformer)

**原作训练设定**: 有监督自然图像恢复

**方法核心**: 局部增强窗口注意力 LeWin、FFN depthwise 卷积，以及 U 型多尺度编码解码与 skip connections。

**对当前 baseline 的迁移建议（研究假设）**: 将当前始终64×64的 body 改为浅层多尺度结构，或只换少量 LeWin blocks；每个尺度融合 HSI/MSI 特征。

**限制与适配边界**: 整体迁移会同时改变容量、深度、分辨率，难归因；窗口与下采样尺寸需处理，原RGB模型不带光谱约束。

**已观察到的实现证据**: 已打开 official repo，真实 model.py、train/、test/、script/ 存在；README 给正式 CVPR citation 与训练命令。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: CVF 正式链接抓取失败；官方 author repo 明确 CVPR2022 + BibTeX 与 arXiv相符。


### P56 | LIIF | CVPR 2021

**Learning Continuous Image Representation With Local Implicit Image Function**

Authors: Yinbo Chen, Sifei Liu, Xiaolong Wang

[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html) | [Official code](https://github.com/yinboc/liif)

**原作训练设定**: 从已有高分辨率RGB图像合成低分辨率输入的任意尺度重建；作者称自监督SR训练，不能等同于无HR-HSI真值融合

**方法核心**: 查询连续坐标和邻近局部特征，用局部隐式函数预测像素，实现连续/任意分辨率表示。

**对当前 baseline 的迁移建议（研究假设）**: 将 tail 改为接收融合局部特征、坐标与像素cell的解码器，输出31波段修正；在观测分辨率用可微PSF/SRF约束连续重建。

**限制与适配边界**: 原解码目标RGB，不是融合；需31维输出/光谱条件编码，且以连续积分模拟传感器观测，否则点采样不等于模糊降采样。按坐标查询增加实现复杂度。

**已观察到的实现证据**: 已打开官方repo和models目录，有 liif.py、mlp.py、train_liif.py、configs/、测试代码、官方CVPR2021 citation。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 不要将原论文 self-supervised 字样解释成无需任何高分辨率图像，公开训练使用DIV2K HR。


### P57 | CARAFE | ICCV 2019

**CARAFE: Content-Aware ReAssembly of FEatures**

Authors: Jiaqi Wang, Kai Chen, Rui Xu, Ziwei Liu, Chen Change Loy, Dahua Lin

[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.html) | [Official code](https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/carafe.py)

**原作训练设定**: 内容感知特征上采样；原实验检测/分割/修复为有监督

**方法核心**: 根据当前特征预测空间位置相关重组核，聚合较大邻域内容，替代固定双线性插值。

**对当前 baseline 的迁移建议（研究假设）**: 替换 hsi_head 之后的 F.interpolate(f_hsi)，优先保留原31波段 hsi_up 基线以隔离变量；可尝试MSI引导重组核。

**限制与适配边界**: 官方 operator 只依赖待上采样特征，不自动用MSI指导；大倍率重组与噪声放大需要验证。官方实现需编译扩展，现有Win11环境兼容性未测试。

**已观察到的实现证据**: CVF 论文指向 OpenMMLab mmdetection；实际开源实现迁至 mmcv ops，已打开 carafe.py，包含 CARAFENaiveFunction、CARAFEFunction、CARAFEPack 和扩展加载。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 源已公开；复现配置引用mmcv版本需锁定，不默认现有requirements可直接安装。


### P58 | RCAN | ECCV 2018

**Image Super-Resolution Using Very Deep Residual Channel Attention Networks**

Authors: Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, Yun Fu

[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf) | [Official code](https://github.com/yulunzhang/RCAN)

**原作训练设定**: 有监督单图像超分辨率

**方法核心**: Residual-in-Residual 多层跳连与全局平均池化通道注意力，将低频信息旁路，使主体学习修正。

**对当前 baseline 的迁移建议（研究假设）**: 作为 ChannelAttention 与三级残差的结构来源参照；可尝试 residual groups 和组级跳连，公平比较深度与参数量。

**限制与适配边界**: 当前 baseline 已经有非常接近的思想；简单再加 RCAN 注意力不构成新贡献。原始版本 Python/PyTorch 老，需要适配。

**已观察到的实现证据**: 作者 repo 实际含 RCAN_TrainCode/、RCAN_TestCode/，ECCV2018 citation 和训练/测试说明；已打开官网 repo。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 优先用于解释、引文和消融对照而非当成新 idea。


### P59 | FcaNet | ICCV 2021

**FcaNet: Frequency Channel Attention Networks**

Authors: Zequn Qin; Pengyi Zhang; Fei Wu; Xi Li

[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.html) | [Official code](https://github.com/cfzd/FcaNet)

**原作训练设定**: 监督分类/检测；可迁移注意力

**方法核心**: 以空间 DCT 多频分量推广全局平均池化的通道摘要。

**对当前 baseline 的迁移建议（研究假设）**: 替换 ChannelAttention 中 x.mean；保持输出 (B,64,H,W)。

**限制与适配边界**: 这里的 multi-spectral 指空间频率分量，不是 HSI 波长；仍需自监督损失。

**已观察到的实现证据**: model/layer.py；main.py；训练与评估脚本

**代码核验程度**: Actual public source observed; not executed or reproduced


### P60 | ECA-Net | CVPR 2020

**ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks**

Authors: Qilong Wang; Banggu Wu; Pengfei Zhu; Peihua Li; Wangmeng Zuo; Qinghua Hu

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf) | [Official code](https://github.com/BangguWu/ECANet)

**原作训练设定**: 监督分类/检测；可迁移注意力

**方法核心**: 用不降维的局部跨通道一维卷积替代两层瓶颈 MLP。

**对当前 baseline 的迁移建议（研究假设）**: 替换 ChannelAttention 的 MLP，比较权重预测容量和参数量。

**限制与适配边界**: 64 个学习通道的邻接关系不等于物理波段邻接；分类优势不能直接推断融合优势。

**已观察到的实现证据**: models/；main.py；light_main.py；模块代码

**代码核验程度**: Actual public source observed; not executed or reproduced


### P61 | CBAM | ECCV 2018

**CBAM: Convolutional Block Attention Module**

Authors: Sanghyun Woo; Jongchan Park; Joon-Young Lee; In So Kweon

[Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper) | [Official code](https://github.com/Jongchan/attention-module)

**原作训练设定**: 监督分类/检测；可迁移注意力

**方法核心**: 串联通道注意力和空间注意力；摘要使用平均/最大池化。

**对当前 baseline 的迁移建议（研究假设）**: 替换 ResBlock.att，构建平均摘要与局部空间门控对照。

**限制与适配边界**: 已经是成熟模块，直接加入不等于新方法；未证明自监督融合有效。

**已观察到的实现证据**: MODELS/；train_imagenet.py；scripts/

**代码核验程度**: Actual public source observed; not executed or reproduced


### P62 | SENet | CVPR 2018

**Squeeze-and-Excitation Networks**

Authors: Jie Hu; Li Shen; Gang Sun

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) | [Official code](https://github.com/hujie-frank/SENet)

**原作训练设定**: 监督分类；注意力基础

**方法核心**: 全局平均与瓶颈 MLP 生成逐通道门控。

**对当前 baseline 的迁移建议（研究假设）**: 作为当前 SE 类注意力的来源与结构参照；比较 LayerNorm/GELU 的改动。

**限制与适配边界**: 官方为 Caffe；你当前注意力并非完全相同的原始 SE，且并非原创范式。

**已观察到的实现证据**: src/caffe/layers/；include/caffe/layers/；models/

**代码核验程度**: Actual public source observed; not executed or reproduced

