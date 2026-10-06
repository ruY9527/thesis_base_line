# Landscape Map

## Scope

- 方向：现有CAVE×8、31波段LR-HSI/3通道HR-MSI融合与预训练模型、文本和无新增训练增强。
- 贡献：先验可靠性/拒绝机制，以及其对光谱失真的可检验作用；不把模型替换当贡献。
- 投稿目标、GPU与是否严格零训练未知；采用小模型缓存/局部图修正的起步假设，保留两种训练路线。
- 非目标：从头训练基础模型、声称恢复未观测真相、同时自由估计任意SRF/PSF/warp、大量模块拼接。

## Existing Context Used

| 本地证据 | 用途 | 边界 |
|---|---|---|
| [model.py](D:/coding/github_self/thesis_base_line/model.py:57) | 确认双分支CNN、fuse/body/tail与插值残差接口 | 没有实际模型前向/新训练 |
| [train_cave_M.py](D:/coding/github_self/thesis_base_line/train_cave_M.py:69) | 固定R、8×8 σ=3 PSF、×8、HR监督L1 | 不把它称为自监督 |
| [utils.py](D:/coding/github_self/thesis_base_line/utils.py:17) | 确认SciPy边界/采样，未来需可微D及伴随 | 当前算子不能直接提供反传 |
| research/baseline_comparison_20261006/learning_report.md | 参数/坐标/测试选模/权重限制 | 既有探索结果，非本轮实测 |
| research/self_supervised_hsi_fusion_review.md 等旧笔记 | 已有inverse/self-supervised邻近地图，避免重复泛泛想法 | 旧62篇没有全部重新核验 |

候选生成发生在上述旧诊断阅读之后；未见新候选实验结果。选出的故事是exploratory，不是confirmatory证据。

## Subareas and Active Approaches

| 子方向 | 代表 | 常见假设 | 值得攻击的条件 |
|---|---|---|---|
| 文本引导融合 | CO²IF | 语义对齐有益于谱空间融合 | 同语义异谱，错误描述，自动caption的证据来源 |
| SAM区域/结构不确定性 | CoPGNet | 区域结构可组织融合误差 | 掩膜不等于材料区，混合LR足迹，外部先验失效 |
| 光谱预训练 | HyperSIGMA/HyperFree/SpectralGPT | 大数据表征能跨域迁移 | CAVE室内可见光与遥感波长/辐射分布差异 |
| RGB预训练桥接 | EigenSR/PLRDiff/HIR-Diff近邻 | 低维空间表征可复用视觉prior | 少数材料/谱比例丢失，先验不满足观测 |
| 输入对扩散与posterior | ARGS-Diff/CDPS | 无HR标签仍可学习两路prior | 参数仍更新，计算成本与prior过拟合 |
| 冻结扩散prompt | DIDM | 提示可转移遥感扩散能力 | “zero-shot”仍需逐样本训练适配模块 |
| CoT/恢复工具 | Q-Agent/RIRF/U2K近邻 | 诊断可指引恢复策略 | 语言是否优于相同数值证据与搜索预算 |

## Closest Work

本轮核验表及完整题名见sources.md；具体层级见sources.json。

- CO²IF（AAAI2026）是直接文本HSI–MSI近邻，全文核验；COT是optimal transport。
- CoPGNet（Information Fusion2026）是直接SAM/区域不确定性近邻，仅出版社摘要与highlights。
- MP-HSIR（ICCV2025）限制“退化文本调制HSI”的首次表述，官方仓库/论文定点核验。
- DIDM（2026预印本）全文显示冻结主干但优化prompt/injection/reconstructor。
- PLRDiff（Information Fusion2024）复用预训练diffusion但为PAN任务。
- DSES-Diff（TIP2026-10-01在线发表）直接无配对预训练融合foundation，仅作者摘要与索引元数据。
- U2K（2025预印本，本轮未确认正式venue）冻结baseline但训练转换模块，是控制器的重要非LLM近邻。
- Q-Agent/RIRF已做CoT恢复；VSR²做training-free HSI理解，均不能直接证明本任务收益。
- SAMSA在HSI分割中结合SAM和谱角，提示“谱角检验语义”也需机制差异审查。

## Gap Inventory

| Gap | 文献基础 | 候选问题 | 风险 |
|---|---|---|---|
| GAP-001：外部语义错误与实际谱证据冲突 | CO²IF/CoPGNet/SAMSA | 推断时限制/拒绝能否减少错误prompt损害，同时保留可靠prior收益？ | 相关全文和引文追踪仍可能找到同机制 |
| GAP-002：已有baseline如何安全调用冻结prior | PnP/PLRDiff/CoPGNet/DDNM | 谱筛选残差图是否超过普通TV和视觉图？ | 图正则本身已有；语言可能无额外作用 |
| GAP-003：诊断残差归因与动作选择 | VSR²/Q-Agent/RIRF/U2K | 同动作/信息/预算下，LLM组织证据有无收益？ | 数值规则可能足够，工具库耗时大 |
| GAP-004：均值质量掩盖材料失真 | CO²IF/CoPGNet/VSR² | 错语义、混合像元、不可辨识控制组能否揭示未见失败？ | 合成诊断与真实数据存在落差 |

“Gap”仅表示值得复查的候选缺口，不意味着已证明文献完全没有研究。

## Assumptions and Negative Evidence

1. 自动文本来自相同输入，不新增物理测量；它只组织证据并引入外部学习先验。
2. 语义类别、分割mask、SVD基分别不等于材料谱真值、材料区域或真实端元。
3. 两路重投影残差小不能保证HR谱真值；共同零空间可容纳错误。
4. 可见光CAVE不支持NIR/红边的实测研究主张。
5. 来源中的frozen/tuning-free/unsupervised/zero-shot各有边界；DIDM/CDPS仍训练参数。
6. 稀疏图与缓存减少成本是设计意图，实际整图耗时与显存必须测量。
7. 文本分支效果若可由纯视觉/同参数卷积解释，应撤下语言贡献。
8. 未确认对应旧最佳Net权重，后续不得用不匹配权重声称baseline提升。

## Disconfirmation Targets

| 假设 | 最廉价攻击 | 停止/收窄 |
|---|---|---|
| 文本提供额外机制 | 正确/空/打乱/错误prompt，视觉mask固定 | 空或视觉条件等效，则取消文本主张 |
| 光谱核验有选择性 | 可信和冲突区域同时测接受率/收益 | 全部关闭prior或错误常被接受则改机制 |
| 大模型图优于手工图 | 同预算TV/局部图/RGB图 | 无增益则保留简单方案 |
| 物理评分可选好重建 | 观测残差与独立HR谱误差配对 | 残差降而谱变差，不宣称质量保证 |
| CoT增加诊断价值 | 等预算规则/网格/数值控制器 | 规则已足够则取消CoT必要性 |
| 先验可解不可辨识样本 | 合法共同零空间控制组 | 限制/不确定性处理，不要求无证据真相恢复 |

## Search and Source Limits

- 实际工具：web检索、官方全文/摘要、作者仓库；Firecrawl MCP/CLI在本环境无可用接口。按用户技能的论文路由使用firecrawl-research-papers的证据工作流，以官方web作为替代。
- 日期：截至2026-10-06；特别包含2026-10-01已在线发表的DSES-Diff。仅检索可见记录，未做完整时间过滤召回验证。
- 搜索按HSI–MSI text/CLIP/foundation、training-free diffusion、SAM uncertainty、CoT restoration等表述，多轮核验直接近邻。
- CO²IF仓库目前可见仅README与pipeline图；论文存在不等于可复现代码到手。
- CoPGNet全文不可获取；若干CVF/PDF访问失败。只记录可验证层级。
- DSES-Diff权重、OFA-HSI发布资源尚未确认；OFA-HSI标Coming soon。
- 未实施系统综述、逐篇引文回溯、论文指标复现或新颖性成立证明。
- 默认下一步：research-novelty-review；不是直接冻结实验或论文贡献。

