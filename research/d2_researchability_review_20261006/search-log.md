# Search Log｜D2 定向反证检索

> 日期：2026-10-06。工具：web 搜索与原文／作者代码直读。无年份、语言或地域硬过滤，保留当前可见2026工作；最后核查原始出版状态。不是系统综述，不提供全领域召回率。
>
> 先检索原主张，再根据不利证据调整候选，最后检索调整版。没有因为不利命中而删除它们。没有声称不存在未命中的同类方法。

| ID | Query（实际查询） | Source / Filters | Purpose | Closest hits / Notes |
|---|---|---|---|---|
| Q01 | hyperspectral multispectral fusion spectral modulation spatial details decoupling orthogonality complementary features | web，无过滤 | 否定调制与解耦模块首次性 | HP-MFDN、空间引导融合；后续重点核查 SSMNet |
| Q02 | multimodal shared private disentanglement orthogonality synergy loss task relevant interaction CoMM | web，无过滤 | 检查分离与协同兼容已知工作 | FOCAL、CVPR2026 DMIL、COrAL线索 |
| Q03 | hyperspectral multispectral fusion coupled matrix factorization spectral signatures abundances Yokoya 2012 | web，无过滤 | 否定光谱×空间的新原理 | 作者WHISPERS2012稿；后续定位TGRS2012正文 |
| Q04 | hyperspectral multispectral fusion decoupling private features redundancy interaction 2025 2026 | web，无过滤 | 检查近期直接融合重叠 | MossFuse、CoFusion等；未将全部摘要算作已精读 |
| Q05 | "Information-Theoretic Decomposition for Multimodal Interaction Learning" github arxiv | web，无过滤 | 核查DMIL正式来源与代码 | arXiv2606.11614、CVF索引及声明代码 |
| Q06 | "Orthogonalized Multimodal Contrastive Learning" synergy | web，无过滤 | 反证正交必然破坏协同 | COrAL原稿；随后读方法与损失 |
| Q07 | "Rethinking Pan-Sharpening via Spectral-Band Modulation" github | web，无过滤 | 核查直接空间／波段调制近邻 | IEEE条目、作者页面；随后PDF确认卷年与代码 |
| Q08 | "spectral" "fusion" "disentanglement" "task" "orthogonality" | web，无过滤 | 检查任务分离训练的跨域近邻 | 分类等旁近工作；未作为直接融合贡献证据 |
| Q09 | hyperspectral multispectral fusion "counterfactual" "null" augmentation | web，无过滤 | 检查调整版物理组合构造是否已覆盖 | 未获得已确认完全同构命中；不是不存在证明 |
| Q10 | hyperspectral fusion "spectral" "abundance" "augmentation" self supervised | web，无过滤 | 否定合成材料／空间目标首次性 | Physics-guided supervision；丰度合成相关工作 |
| Q11 | hyperspectral fusion "orthogonal" "synergy" regularization | web，无过滤 | 检查正交与协同、可替代解释 | OSF及跨域方法 |
| Q12 | "Gradient Surgery for Multi-Task Learning" PCGrad NeurIPS 2020 | web，无过滤 | 排除普通优化协调的重新命名 | 原稿及作者仓库 |
| Q13 | "Orthogonal Sequential Fusion in Multimodal" github | web，无过滤 | 核查OSF与出版状态 | Oxford档案明确ISIF2025；OpenReview2024只是投稿稿 |
| Q14 | "Physics-guided supervision for highly generalisable" github code | web，无过滤 | 核查合成训练近邻代码 | 出版社来源，代码未确认 |
| Q15 | "hyperspectral" "counterfactual" "fusion" spatial spectral | web，无过滤 | 检查反事实谱空方案重叠 | CauseHSI分类工作，不将分类结果外推到融合 |
| Q16 | "hyperspectral" "null-space" "self-supervised" fusion | web，无过滤 | 检查物理分工与自监督近邻 | PhySpec／HyDiff-EI线索；出版／全文核查不足，保留未决 |
| Q17 | "GeWu-Lab/DMIL" github | web，无过滤 | 确认声明代码 | 原稿和CVF索引声明一致；仓库直读仍失败 |
| Q18 | "PhySpec" github | web，无过滤 | 核查零空间辅助训练近邻 | OpenReview验证页阻挡，未把线索提升为方法已全面审查 |
| Q19 | "Physics-guided supervision" "EnMAP" code authors | web，无过滤 | 核查合成监督近邻细节 | 出版社可索引§2内容；正文直读失败 |
| Q20 | "hyperspectral" "separation" "interaction" "regularization" fusion private | web，无过滤 | 否定简单分离加交互首次性 | 多种融合／分类旁近，未据摘要宣称核心已完全覆盖 |
| Q21 | Naoto Yokoya coupled nonnegative matrix factorization CNMF code github | web，无过滤 | 核查经典方法与公开实现 | 作者页面提供MATLAB/Python；排除无关单细胞cNMF命中 |
| Q22 | "Information-Theoretic Decomposition" "CVPR 2026" Yang html | web，无过滤 | 核查会议状态 | CVF索引、官方program线索；HTML直读失败 |
| Q23 | "self supervised" "hyperspectral fusion" "factorial" | web，无过滤 | 检查四组组合诊断近邻 | 无已核实完全同构结果，仍保持首次性未决 |

## 阅读、失败与证据范围

| 来源 | 结果与使用边界 |
|---|---|
| [CNMF期刊作者PDF](https://naotoyokoya.com/assets/pdf/NYokoyaTGRS2012.pdf)／[下载页](https://naotoyokoya.com/Publications.html) | 直读成功，确认期刊卷年与基本分解思想；未运行旧实现 |
| [MossFuse原稿](https://arxiv.org/html/2412.04802v4)／[损失源码](https://raw.githubusercontent.com/dusongcheng/MossFuse/main/utils.py) | 前序原文+当前源码；不把某负项存在等同于已证实损害 |
| [FOCAL原文](https://proceedings.neurips.cc/paper_files/paper/2023/file/93e98ddf39a9beb0a97fbbe56a986c80-Paper-Conference.pdf) | 直读成功，确认正交对象及任务领域 |
| [COrAL原稿](https://arxiv.org/html/2602.14983)／[仓库](https://github.com/Caro-Ci/COrAL) | 直读成功；未核实正式期刊卷期，不复述作者的首次性为事实 |
| [DMIL原稿](https://arxiv.org/html/2606.11614) | 直读成功；[CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Information-Theoretic_Decomposition_for_Multimodal_Interaction_Learning_CVPR_2026_paper.pdf)403，HTML亦失败；[代码](https://github.com/GeWu-Lab/DMIL)两次直读失败 |
| [SSMNet原文](https://www.haoshen.site/resources/LiuXinyang_SSMNet_TGRS_2023.pdf)／[代码](https://github.com/ez4lionky/SSMNet/) | 成功；用页眉TGRS2024卷62而非文件名2023作为卷年 |
| [FiLM](https://arxiv.org/abs/1709.07871)／[CoMM仓库](https://github.com/Duplums/CoMM)／[PCGrad原文](https://arxiv.org/abs/2001.06782) | 原始来源支持基础机制；没有重跑实验 |
| [Physics-guided supervision](https://www.tandfonline.com/doi/abs/10.1080/19479832.2026.2613383) | 出版社搜索索引可见方法段；full与abs直读失败，代码未确认；具体四组构造重叠仍需全文核查 |
| [OSF作者档案](https://ora.ox.ac.uk/objects/uuid%3A5d25fd68-bc13-43d2-b64b-64ba881acbdb) | 正式FUSION2025元数据可见；不误写ICLR2024录用 |
| [PhySpec OpenReview](https://openreview.net/forum?id=WISfJyOA6M) | 验证页阻挡；附件搜索线索不是已全面阅读的依据 |

## 覆盖限制与未解决项

- 未完成正式系统综述、引文图追踪、数据库全检索或召回率审计。23条定向查询足以发现强重叠，不能证明完全没有同构研究。
- NQ1 首次性受物理合成监督、零空间辅助任务、等变学习／因子组合训练等近邻影响，正式方法冻结后应继续专项核查。
- NQ2 与 NQ3 需要实验，不是继续搜索即可解决。没有实验时不得升级为“已有研究普遍存在的缺点”。
- 本轮数字校核仅验证代数；未作为指标收益、学习机制或独立审查证据。
- 输出的 JSON 标为 unlinked，定位最多为有限继续；结构校验通过不代表文献完整性或科学正确性得到认证。
