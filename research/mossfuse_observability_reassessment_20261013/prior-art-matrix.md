# Prior Art Matrix

日期2026-10-13。近期以2024-10-13之后为主；旧工作用于概念先例。没有文献全覆盖结论。

| 工作与一手来源 | 已核实内容 | 重叠与边界 | 阅读层次 |
| --- | --- | --- | --- |
| [MossFuse / IJCV2026](https://link.springer.com/article/10.1007/s11263-026-02757-8)，[作者v4](https://arxiv.org/html/2412.04802v4) | 共享/互补、共同LR-MSI监督、退化估计、最终物理一致性、门控、响应扰动 | 直接覆盖宽泛旧设想；未证明一定跨传感器失败 | 作者全文已读；重点复核；固定代码静态核查 |
| [NUD/PAF / TPAMI2025](https://doi.org/10.1109/TPAMI.2025.3604234) | 摘要提出非均匀退化与位置感知融合匹配 | 退化指导结构已有近邻；具体模块重叠仍需全文 | 摘要+此前代码 |
| [PINO / NeurIPS2025](https://papers.neurips.cc/paper_files/paper/2025/hash/89af8e4eb696738f2c9e589522968a09-Abstract-Conference.html) | 连续辐射场、学习响应积分、CAVE融合实验 | 物理响应进入融合已有方法；不是Moss式Shared，响应不全为外部标定 | 前轮全文，本轮方法/表3复核 |
| [GSNR / CVPR2026](https://arxiv.org/abs/2602.20328) | 通用线性反问题的测量零空间结构化 | 可见/不可见分开不是新概念；非HSI–MSI直接结果 | 前轮正文全文，本轮理论/讨论复核 |
| [UQ Null Space Networks / arXiv2023](https://arxiv.org/abs/2304.06955) | 数据一致性、零空间与不确定性 | 零空间加不确定性已有先例；超出近期范围 | 前轮作者全文 |
| [LMN / SIAM2026](https://epubs.siam.org/doi/10.1137/25M1796448)，[作者稿](https://arxiv.org/abs/2512.19489) | 摘要声明端元变化下结构化张量模型及条件式恢复保证 | 同任务理论已有覆盖；不能推为任意场景均恢复 | 摘要/导言；缓存全文未完整读完 |
| [Unified Tensor Fusion / SIAM2026](https://epubs.siam.org/doi/abs/10.1137/25M1814098) | 摘要：联合估计HR-HSI、PSF、SRF | 盲退化联合重建已有覆盖；具体机制未评 | 摘要，全文获取失败 |
| [SRLF / CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html) | 观测失真驱动选择性重学习 | 残差选择不能首创；不同于尚待验证的RU功能分工 | 前轮全文 |
| [SpecAware / ISPRS2026](https://doi.org/10.1016/j.isprsjprs.2026.02.024)，[作者稿](https://arxiv.org/abs/2510.27219) | 波长、带宽条件化跨传感器表示 | 元数据输入不新；映射任务不是融合直接基线 | 前轮作者v2全文 |
| [RASD-FuNet / IF在线2026](https://doi.org/10.1016/j.inffus.2026.104498) | 低秩置信度、空间-光谱动态路由 | 路由机制近邻；未评完整物理可观测性重叠 | 摘要/代码，非全文 |
| [WCGE-Net / JSTARS2026](https://doi.org/10.1109/JSTARS.2026.3726856) | 小波解耦、跨模态专家、动态加权；2026-08-24上线 | 特征解耦与选择融合已覆盖；是否算子条件化仍待全文 | IEEE正式摘要 |
| [CoPGNet / IF2026](https://doi.org/10.1016/j.inffus.2025.103997) | 外部光谱/内部结构先验、不确定区域加权 | 训练难区域加权与输入信任度降权需区分 | 前轮摘要/引言 |
| [VPGNet / TIP2026](https://doi.org/10.1109/TIP.2026.3689415) | 概率匹配与不确定性先验 | 不确定性融合近邻，未排除所有重叠 | 前轮摘要 |

广泛关键词组合新颖性弱；窄版本需要逐条排除方法重叠。出版年份以正式来源为准，作者预印本年份另记。
