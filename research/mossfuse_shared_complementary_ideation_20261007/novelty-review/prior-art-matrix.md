# Prior Art Matrix

技术概括以作者全文/出版社摘要/固定代码为界；内容每篇保持简短，不复制原文。
| 工作 | 来源与代码 | 与候选的重叠 | 尚未排除的关键问题 |
|---|---|---|---|
| MossFuse | [原始来源](https://doi.org/10.1007/s11263-026-02757-8)；[代码](https://github.com/dusongcheng/MossFuse) | 信息组织与聚合；D2 泛化叙事重叠 | A/B 是否形成测量条件与输出功能增量 |
| SRLF | [原始来源](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.pdf)；[代码](https://github.com/YuanyeLiu/SRLF-Net) | 观测域失真选点再学习；A 的普通局部修正近邻 | 灵敏度条件是否超越误差条件 |
| RASD | [原始来源](https://www.sciencedirect.com/science/article/abs/pii/S1566253526003775)；[代码](https://github.com/YuanyeLiu/Rank-Aware-Routing-Decomposition-for-Hyperspectral-and-Multispectral-Image-Fusion) | 预测分数和秩相关双路线；内部降尺度构造，实际数据身份未核实 | A 的 RU 谱并非当前已读入口，但子空间近邻仍待查 |
| CoMM | [原始来源](https://proceedings.iclr.cc/paper_files/paper/2025/file/108030643e640ac050e0ed5e6aace48f-Paper-Conference.pdf)；[代码](https://github.com/Duplums/CoMM) | 共享、独有、协同表征 | 不以第三协同分支或 PID 名字为新颖性 |
| NUD_PAF | [原始来源](https://pubmed.ncbi.nlm.nih.gov/41191478/)；[代码](https://github.com/jeline0110/NUD_PAF) | 非均匀退化与位置属性融合 | C 的相容共享域是否有独立增量 |
| ES | [原始来源](https://openreview.net/forum?id=upMIVpe467)；[代码](https://github.com/vsechaud/Equivariant-Splitting) | 观测拆分与等变条件 | B 不能直接借用风险理论，是否只为拆分应用 |
| SpecAware | [原始来源](https://doi.org/10.1016/j.isprsjprs.2026.02.024)；[代码](https://github.com/busbyjrj/SpecAware) | 元信息/内容条件编码；mapping 任务 | B 的分工是否超越传感器条件化 |
| Optics | [原始来源](https://vccimaging.org/Publications/Fu2025Limitations/)；[代码](https://github.com/vccimaging/OpticsAwareHSI-Analysis) | RGB 重建同色异谱与光学影响 | A 的 LR-HSI 是否足以解除相关歧义；仅作相关背景 |
| PSF2026 | [原始来源](https://www.tandfonline.com/doi/full/10.1080/19479832.2026.2613383)；代码未核实，仅反证近邻 | 已考虑波长相关 PSF 的数据生成 | C 不能主张首次；代码未核实，出版社直读受限 |

## Source Coverage Limits
RASD 出版索引的卷期为 2026 年 12 月，当前尚未到该月份，在线首次日期未验证。SpecAware 已确认仓库目录/README，未读其全部核心代码；ES 复用此前源码核对记录与本轮官方仓库说明；无参考源码运行。NQ1–NQ3 仍开放。
