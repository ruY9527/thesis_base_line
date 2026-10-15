# 论文代码依据、迁移位置与证据边界

核对日期：2026-10-06。本轮直接只读获取作者公开仓库的关键代码，核对提交与文件摘要，没有运行参考代码，没有把参考源码复制为项目 Python 文件。文件记录见同目录 source-manifest.json。

**下面区分“作者已经实现了什么”“本方案借鉴什么”“本方案尚需证明什么”。** 不把作者已有机制改个名字作为创新，也不根据分离损失的存在就判断原方法有害。

## 1. MossFuse：动机一的直接代码依据

论文：[作者全文](https://arxiv.org/html/2412.04802v4)。期刊：IJCV 2026；正式出版依据沿用已有研究核对。本次代码提交：`d6c7eb4a487046aecae5265cabf81b0c23506465`。

| 代码位置 | 实际实现 | 对本方案的意义 |
|---|---|---|
| [MossFuseNet.py:10](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/MossFuseNet.py#L10) Encoder_MSI | baseFeature 和 detailFeature 两个分支，返回两种特征及主编码特征 | 双来源中进一步组织角色已有先例；本方案用轻量 1×1 投影，不搬整个编码器 |
| [MossFuseNet.py:46](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/MossFuseNet.py#L46) Encoder_HSI | 另一来源也返回 base/detail；对 HSI 先插值到 HR 后编码 | 本方案保留原 HSI 的 LR 编码，并在 LR 做新投影；尺寸路线不同 |
| [MossFuseNet.py:171](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/MossFuseNet.py#L171) Decoder_HR | 先合并两路 base，再空间聚合，再光谱聚合 | **作者已经考虑分离后的配合**，不能写成它只分离、没有联合重建 |
| [net.py:198](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/net.py#L198) Cross_Spatial_Attention | 互补信息对 query 做空间调制，注意力输出还用于调制另一支 | 交互/调制本身不是本方案的新颖性 |
| [net.py:245](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/net.py#L245) Cross_Spectral_Attention | 光谱权重调制 query，输出也与另一支聚合 | 类名含 spatial/spectral 不等于 N×N 全像素注意力；源码 reshape 后主要是头内通道相关矩阵 |
| [utils.py:75](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/utils.py#L75) cal_cos_loss | 将输入整体 view(-1)，再 cosine | 原实现把 batch 也包含进去；本方案按样本计算属于显式改动 |
| [utils.py:82](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/utils.py#L82) cal_decomp_loss | 共享相似项在分子，三类互补排斥项进入分母 | 作为 M1-B 的直接分离对照；损失优化不等于功能完整性 |
| [train.py:111](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/train.py#L111) 训练输出/损失 | 读取四种特征，计算多项重建/退化相关量 | 证明分离、融合和训练目标需要一起检查，不能只读编码器 |
| [train.py:150](https://github.com/dusongcheng/MossFuse/blob/d6c7eb4a487046aecae5265cabf81b0c23506465/train.py#L150) loss1 | 该提交实际 backward 的主项是 0.1×分离损失＋两项最终输出观测损失；blind 分支另用观测退化一致性 | **计算了某个 loss 不意味着它进入训练**。不要把被注释的完整损失当作此提交的实际训练 |

该提交第 133 行还计算 HR-HSI 的 L1 以记录，但它不进入第 150 行的 loss1。MSI/HSI 自重建量同样被计算，却未出现在当前活动的 loss1 中。这是论文论述、代码中可用模块、具体提交训练入口之间需要区分的事实；不据此推断作者所有实验设置。

原论文 §3.4 已讨论简单解耦可能影响信息完整性，并设计自监督重建；Fig.12 的分离权重实验体现重建质量不随分离目标单调增加。可以支持“存在目标取舍值得研究”，**尚不支持“光谱—空间必要联系被破坏就是指标下降的原因”**。

本方案的候选贡献应落到：具体分离位置/合理强度下的联合响应取舍，以及在相同观测角色目标与公平结构对照下的修复。主计划第 3 节只是低成本结构筛查，第 4 节才检验上述机制。

## 2. CLoRF-Fusion：组合关系与观测训练的代码依据

论文：[IJCV 2025 出版社](https://link.springer.com/article/10.1007/s11263-025-02540-1)。代码提交：`185fff41bccf4367b39a4f791c704b94268eee16`。

- [train_tv_spatial.py:103](https://github.com/wangting1907/CLoRF-Fusion/blob/185fff41bccf4367b39a4f791c704b94268eee16/train_tv_spatial.py#L103)：两个坐标网络分别输出空间表示和光谱表示，第 107 行以矩阵乘积组合。依据是光谱与空间共同决定重建；不能据此声称逐元素特征乘积是新的低秩分解。
- [train_tv_spatial.py:118](https://github.com/wangting1907/CLoRF-Fusion/blob/185fff41bccf4367b39a4f791c704b94268eee16/train_tv_spatial.py#L118)：预测经空间/光谱退化匹配两种观测，随后加空间 TV。迁移到主计划第 6.2 节的是双观测训练组织，不是整个 INR。
- [train_tv_spatial.py:175](https://github.com/wangting1907/CLoRF-Fusion/blob/185fff41bccf4367b39a4f791c704b94268eee16/train_tv_spatial.py#L175)：示例按真值 PSNR 保存最佳输出。本方案不沿用这种无标签主实验选模方式，应采用独立验证规则或冻结预算。

作者默认倍率/核、噪声和当前 CAVE ×8 设置不同，不引用其论文成绩作为本地同协议对照。主计划的低秩特征乘积属于借鉴组合关系的简化结构，不是复现 CLoRF。

## 3. SHOTUN：动机二的关系保持依据

论文：[TIP 2026 作者摘要登记](https://pubmed.ncbi.nlm.nih.gov/42202190/)。代码提交：`d0bec639d1ed77150c0e4c2aaac3da876c9886bf`。

| 代码位置 | 本轮确认的作用 | 迁移范围 |
|---|---|---|
| [nonlocal 入口:57](https://github.com/Shawn-H-Wang/SHOTUN/blob/d0bec639d1ed77150c0e4c2aaac3da876c9886bf/unfolded_net_sparse_representation_full_unfolded_nonlocal.py#L57) HOTN | 第四模式关系矩阵处理后参与高阶张量重建，同时有 identity 关系版本 | 关系参与组合重建已有，本方案不能把“增加关系矩阵”称为首创 |
| [nonlocal 入口:90](https://github.com/Shawn-H-Wang/SHOTUN/blob/d0bec639d1ed77150c0e4c2aaac3da876c9886bf/unfolded_net_sparse_representation_full_unfolded_nonlocal.py#L90) CoreTensorUpdateStage | 观测残差经过光谱/空间因子更新核心张量 | 关系保持与观测约束结合已有；本方案不移植整个迭代展开 |
| [nonlocal 入口:149](https://github.com/Shawn-H-Wang/SHOTUN/blob/d0bec639d1ed77150c0e4c2aaac3da876c9886bf/unfolded_net_sparse_representation_full_unfolded_nonlocal.py#L149) AggerationModule | 块聚合后乘光谱基回图像 | 需要区分潜在局部信息分配与最终拼块平均 |
| [nonlocal 入口:279](https://github.com/Shawn-H-Wang/SHOTUN/blob/d0bec639d1ed77150c0e4c2aaac3da876c9886bf/unfolded_net_sparse_representation_full_unfolded_nonlocal.py#L279) build_loss | 两种观测的重建 MSE | 同样支持 M3 观测训练，但其尺度/求和方式不可照搬 |
| [utils_blocks.py:24](https://github.com/Shawn-H-Wang/SHOTUN/blob/d0bec639d1ed77150c0e4c2aaac3da876c9886bf/utils_blocks.py#L24) _make_blocks | 分块、边界扩展与排列 | 局部关系实现需检查索引；其 reflect padding 不等于本地成像 D 的 symmetric boundary |
| [utils_blocks.py:119](https://github.com/Shawn-H-Wang/SHOTUN/blob/d0bec639d1ed77150c0e4c2aaac3da876c9886bf/utils_blocks.py#L119) _agregate_blocks | Col2Im 与重叠权重/计数聚合 | 不直接替换本地 reconstruction，避免混入拼接策略收益 |

LowRankNetwork.py 也已检查相关符号，其中 FMSAN 给出另一关系计算结构，但本次非局部入口未导入它，因此**不把 FMSAN 宣称为该入口实际运行的核心模块**。

本方案第 5 节采用两来源条件化的 3×3 路由，仅搬运交互中的 HSI 互补候选。借鉴“空间关系需要保留”，不是从作者源码逐行搬出一个同名模块。新的局部设计是否有价值，要与 MSI-only、HSI-only、均匀平滑和普通卷积对照。

## 4. CDPS：动机三中“先验来自观测”的代码依据

论文：[TIP 正式卷期为 2026，DOI 年份为 2025](https://pubmed.ncbi.nlm.nih.gov/41477791/)。作者仓库 README 标题为 TIP2025，不算另一篇论文。代码提交：`54cf99b92b872ee77b191045c8003851398a9065`。

- [image_datasets_spatial.py:29](https://github.com/Zhu1116/CDPS-train/blob/54cf99b92b872ee77b191045c8003851398a9065/guided_diffusion/image_datasets_spatial.py#L29)：空间训练数据读取 HRMSI，变换到 [-1,1]，逐通道取空间块。
- [image_datasets_spectral.py:29](https://github.com/Zhu1116/CDPS-train/blob/54cf99b92b872ee77b191045c8003851398a9065/guided_diffusion/image_datasets_spectral.py#L29)：光谱训练数据读取 LRHSI，将每个像素的光谱作为样本。
- [spatial_sample.py:27](https://github.com/Zhu1116/CDPS-train/blob/54cf99b92b872ee77b191045c8003851398a9065/spatial_sample.py#L27)：加载空间扩散权重并采样；所读入口没有把本地 HSI/MSI 成像模型嵌入完整耦合后验融合。

代码能支持“训练先验时不必读取 HRHSI”的事实；不支持“在当前小 Net 上加一个扩散 loss 就能复现 CDPS”。本轮只读了选定数据/采样入口和扩散文件，不宣称已完成 CDPS 全部方法的可运行复现。

迁移到本方案的是观测数据隔离和学习先验的组织原则。M3 的内部任务与生成扰动属于我们提出的候选，不能归给 CDPS 作者，也没有继承其性能保证。

## 5. Equivariant Splitting：动机三的训练目标边界

论文：[ICLR 2026 官方页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34a5f638617ce32700e235e50dff9c8a-Abstract-Conference.html)。代码提交：`1316067cd61f787284a3ddfca6ee5442dc7d4394`。

- [mri/losses.py:9](https://github.com/vsechaud/Equivariant-Splitting/blob/1316067cd61f787284a3ddfca6ee5442dc7d4394/mri/losses.py#L9)：SplitR2RLoss 在指定噪声模型和 mask 下计算分裂损失，adapt_model 包装原网络；训练目标与模型访问的数据需要一起调整。
- [cs+inp/train.py](https://github.com/vsechaud/Equivariant-Splitting/blob/1316067cd61f787284a3ddfca6ee5442dc7d4394/cs%2Binp/train.py)：通过 physics、losses 和模型组织训练，不能将其结论移到任意 Net 的任意增强上。
- [README](https://github.com/vsechaud/Equivariant-Splitting/blob/1316067cd61f787284a3ddfca6ee5442dc7d4394/README.md)：MRI 环境还指向 DeepInverse 的特定 fork；本方案首版不引入这一整套环境。

对本地代码最有用的启发是：先定义监督依据和成像算子，再决定结构/损失。不把“随机翻转输入”直接称为符合理论的等变训练，不为当前双模态无噪声设置宣称无偏。

## 6. 从代码依据到修改点的最终映射

| 动机 | 论文已有事实 | 本地改动位置 | 候选方法 | 必须新增的证据 |
|---|---|---|---|---|
| M1 | MossFuse 分离、融合和观测损失；CLoRF 组合表示 | 原 fuse 后、body 前 | 四路来源候选＋有回退的交互接口；随后比较具体分离约束 | 结构对照收益；合理强度的角色/联合行为取舍 |
| M2 | SHOTUN 关系参与高阶组合和块聚合 | bridge 内部的 HSI 联合项之前 | 两来源条件的局部光谱路由 | 超过平滑/卷积对照，局部光谱误差改善 |
| M3 | CLoRF/SHOTUN 观测重建，CDPS 观测先验，ES 模型与损失协同 | 同一 Net 的外部训练器及数据接口 | 内部尺度目标、同四组的联合训练对照 | 无 HR 泄漏、算子一致、同目标/预算下有效 |

这里只建立“为什么可以这样设计”的代码依据，没有建立“这样改一定能提高指标”或“首次提出”的证据。优先级按主方案的阶段门槛推进。
