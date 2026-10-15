# Novelty Report

## Contribution Under Review

- One-sentence claim: 已知正确成像算子下，检验观测灵敏度引导的共享与互补功能分工，是否改善未见SRF下的HSI–MSI融合并超出普通条件输入、增强与适应。
- Operating mode: standalone
- Upstream artifacts used: 用户组会设想、MossFuse作者v4全文、固定提交d6c7eb4a487046aecae5265cabf81b0c23506465。
- Literature-context artifacts used: prior-art-matrix.md、search-log.md、已有全文缓存与阅读账本。
- Boundary: 本轮评估完成；没有模型实验，候选未获证实。

## Claim Decomposition

| Dimension | Claimed novelty | Searchable formulation | 判断 |
| --- | --- | --- | --- |
| framing | 传感物理可观测性重新解释分工 | fusion observability shared complementary | 宽泛表述重叠强 |
| method | R/D与灵敏度指导分工 | RU conditioning; range/null-space; uncertainty | 候选待验证，SVD本身非创新 |
| protocol | 已知算子，未见响应，分支功能干预 | held-out SRF; known degradation; functional intervention | 合理协议，非自动的方法创新 |
| artifact | 证据账本、固定代码与数学核验 | pinned sources | 研究资料，不是科学发现 |
| finding | 低物理损失下可修复分工错误 | low residual; high HR error; branch roles | 尚无实验证据 |

## Strongest Overlaps

| Work | Threat rating (1-5) | Why it matters | 阅读层次 |
| --- | ---: | --- | --- |
| MossFuse / IJCV2026 | 5 | 已有Shared物理监督、退化估计、最终物理损失、扰动与门控 | 作者全文已读；静态代码核查 |
| NUD/PAF / TPAMI2025 | 4 | 退化指导融合结构 | 摘要/代码，非全文 |
| PINO / NeurIPS2025 | 4 | 响应积分物理融合，含CAVE | 前轮全文、本轮复核 |
| GSNR / CVPR2026 | 4 | 不可见零空间结构化 | 前轮正文全文、本轮复核 |
| UQ Null Space Networks / arXiv2023 | 4 | 零空间+不确定性已有概念 | 前轮作者全文；旧先例 |
| LMN / SIAM2026 | 4 | 同任务结构化可恢复性保证 | 摘要/导言，非全部定理 |
| SRLF / CVPR2025 | 3 | 观测失真引导选择性重学习 | 前轮全文 |
| SpecAware / ISPRS2026 | 3 | 传感器波长/带宽条件化 | 前轮作者全文，任务不同 |
| RASD-FuNet / IF2026 | 3 | 低秩置信度路由分工 | 摘要/代码，非全文 |
| WCGE-Net / JSTARS2026 | 3 | 解耦互补特征与专家动态融合 | 摘要 |

分数为主观定位，不代表发表概率。不同任务只作为概念先例，不能不加改造混作直接性能基线。

## Novelty-Killing Objections

1. 仅加SRF/PSF和物理一致性，重复MossFuse已有机制。
2. 仅加权或门控，与原模型及选择性重学习、置信度路由重叠；需证明新增依据。
3. 一般零空间反例不能证明Shared错误；共同LR-MSI在理想正确算子下已有参照。
4. 传感器改变后Shared可合理改变；特征漂移不是错误标签。
5. 公开入口未使用论文全部损失；补齐原项是实现修正。
6. MSI不支持的信息仍可由HSI支持；盲目降权会丢掉有效互补。
7. GT监督、参数量与适应预算不同会破坏收益归因。
8. RU灵敏度属于经典子空间逆问题；SVD/条件数不是独立创新。

## Decision

- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 3
- Narrowest defensible positioning: 已知正确成像算子下，检验观测灵敏度引导的共享与互补功能分工，是否改善未见SRF下的HSI–MSI融合并超出普通条件输入、增强与适应。
- What would change the decision: 完整基线在合法响应变化下出现可重复的功能与任务问题，候选方法在公平对照中修复该问题，且补齐最接近方法全文后仍存在明确差异。
- Literature assurance mode: unlinked
- Literature assurance verdict: targeted-search-only-not-systematic
- Unresolved high-priority novelty-critical questions: NQ1, NQ2, NQ3

评分针对收窄版本；宽泛旧版倾向2/5，需要重构。原模型已有物理约束的判断把握高；新增机制是否有效的把握低。这里不是新颖性通过，也没有科学结果晋级。

## Paper-Review Handoff

| Finding to add | Paper location / quote | Comment type | Impact | Confidence | Evidence basis |
| --- | --- | --- | ---: | ---: | --- |
| Shared已有物理监督 | §3.4、式16 | claim_accuracy | 5 | 5 | 共同LR-MSI重建 |
| 已做响应扰动 | §4.4、Fig.10 | claim_accuracy | 5 | 5 | 扰动算子合成输入 |
| Shared漂移不等于错误 | RH=MD^T | methodology | 5 | 5 | 算子相对目标 |
| 瓶颈未获证实 | 本轮无模型实验 | missing_information | 5 | 5 | 数学与实验分离 |
| 区分完整论文与公开入口 | train.py149–164 | methodology | 5 | 5 | 固定静态代码 |
| 近邻排查尚不完整 | prior-art-matrix.md | claim_accuracy | 5 | 4 | 阅读层次分别标注 |

NQ1：是否已有同任务方法用SRF×局部子空间的观测灵敏度约束共享/互补功能分工？
NQ2：是否已有“正确已知算子+未见响应+分支功能干预”的相同协议与发现？
NQ3：NUD/PAF、RASD-FuNet、WCGE-Net等完整方法是否与最终具体算法重叠？
