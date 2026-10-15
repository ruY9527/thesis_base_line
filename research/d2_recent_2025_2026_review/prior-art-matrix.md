# Prior Art Matrix：近期先行工作与证据矩阵

核验：2026-10-06。7 篇正式顶刊、2 篇补充顶会；均已访问公开作者仓库。代码可访问不等于已运行。本文为定向筛选，不声称穷尽两年文献。

| ID | 论文/正式年份 | 发表依据与 DOI | 作者代码 | 本轮读到的证据 | 与 D2 的重叠/边界 |
|---|---|---|---|---|---|
| J01 | MossFuse，IJCV 2026 | [作者机构](https://research.aber.ac.uk/en/publications/unsupervised-hyperspectral-image-super-resolution-via-self-superv/)；10.1007/s11263-026-02757-8；2026-03-06 | [MossFuse](https://github.com/dusongcheng/MossFuse) | [作者全文 v4](https://arxiv.org/html/2412.04802v4)、utils.py 的分离损失 | 共享/互补分离与后续融合已有；损失存在不证明有害；预印本起于 2024 |
| J02 | CLoRF-Fusion，IJCV 2025 | [出版社](https://link.springer.com/article/10.1007/s11263-025-02540-1)；2025-08-03 | [CLoRF-Fusion](https://github.com/wangting1907/CLoRF-Fusion) | 出版社方法、train_tv_spatial.py 完整入口 | 空间/光谱乘积已有；观测训练、TV；按真值 PSNR 保存最佳示例 |
| J03 | SHOTUN，TIP 2026 | [作者摘要登记](https://pubmed.ncbi.nlm.nih.gov/42202190/)；10.1109/TIP.2026.3695389；35:5714–5729 | [SHOTUN](https://github.com/Shawn-H-Wang/SHOTUN) | 摘要、HOTN/CoreTensorUpdateStage/训练入口 | 关系保留与张量组合已有；非局部核心已读，但 IEEE 全文未完整取得 |
| J04 | CDPS，TIP 2026 卷期 | [作者摘要登记](https://pubmed.ncbi.nlm.nih.gov/41477791/)；10.1109/TIP.2025.3647207；35:69–84 | [CDPS-train](https://github.com/Zhu1116/CDPS-train) | 作者摘要、两路文件树、spatial_sample.py | 当前观测学习两先验，耦合采样；完整后验代码未审完；仓库 TIP2025 不作第二篇 |
| J05 | NUD/PAF，TPAMI 2025 | [作者摘要登记](https://pubmed.ncbi.nlm.nih.gov/41191478/)；10.1109/TPAMI.2025.3604234；47(12):11464–11482 | [NUD_PAF](https://github.com/jeline0110/NUD_PAF) | 摘要、PAF/main_fus_syn.py | 位置相关退化与融合已有；排除退化失配；网络内部源码本轮未成功取得 |
| J06 | IR&ArF，TIP 2025 | [作者摘要登记](https://pubmed.ncbi.nlm.nih.gov/40126967/)；10.1109/TIP.2025.3551531；34:1934–1949 | [IR-ArF](https://github.com/Jiahuiqu/IR-ArF) | 摘要、README、train.py | 几何/任意尺度观测展开；gths 标签 L1，原训练有监督 |
| J07 | SSC-HSR，IJCV 2025 | [出版社](https://link.springer.com/article/10.1007/s11263-025-02466-8)；2025-06-17 | [SSC-HSR](https://github.com/yingkai-zhang/SSC-HSR) | 出版社摘要、README 训练与合成光流流程 | 配准与聚合已有；不能直接当作无标签训练；整体训练源码未全审 |
| C01 | UAFL，CVPR 2026，补充会议 | [会议论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html) | [UAFL](https://github.com/yingkai-zhang/UAFL) | CVF 摘要、作者仓库机制和训练说明 | SVD 基保持、丰度残差、跨注意力/调制已有；有标签参考超分 |
| C02 | Equivariant Splitting，ICLR 2026，补充会议 | [正式会议](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34a5f638617ce32700e235e50dff9c8a-Abstract-Conference.html) | [Equivariant-Splitting](https://github.com/vsechaud/Equivariant-Splitting) | 会议摘要、作者仓库 | splitting+equivariance 已有；不偏理论需条件，非直接 HSI 任务 |

## 不进入近期主表的工作

| 工作 | 处理 | 原因 |
|---|---|---|
| CNMF、FiLM | 仅保留历史定位 | 2012、2018，非本轮近期依据 |
| COrAL | 暂不作已确认顶刊 | 预印本 2026，正式期刊状态未充分核实；并非十多年前 |
| Low-Rank Transformer | 不作为最新上线工作 | IJCV 卷期可能写 2025，但首次正式上线 2024-08-20，早于本轮窗口；不能只按卷期漂移年份 |
| Variational Tensor Subspace Decomposition | 排除近期主表 | 检索到旧工作，不能因网页更新时间判为 2026 |
| DMIL | 暂不进入有代码核实的主表 | 近期会议线索，但作者代码访问未充分核实 |
| FreqFusion 2024、旧 PCGrad 等 | 历史背景 | 不承担本轮 2025–2026 主要机制论证 |

## 仍可能否定新颖性的近邻

- NQ1：辅助投影承担解耦、原始表示承担重建的训练接口设计。
- NQ2：可控物理组合、零空间合成、内部降尺度任务的联合响应训练。
- NQ3：近期张量展开与耦合后验方法是否已经具有等价的任务关系约束；目前全文/完整源码覆盖不足。

“本轮没找到完全同构”不是“没有人做过”。需在机制或指标出现初步证据后针对这些近邻扩大检索。
