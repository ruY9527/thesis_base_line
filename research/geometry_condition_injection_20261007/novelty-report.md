# Novelty Report

## Contribution Under Review

Standalone定向审查：局部光谱可辨性约束的方向几何条件注入。不是对GEWDiff整个论文的首次审稿；使用研究报告及工作区既有全文/诊断。宽泛几何注入方案2/5，窄候选3/5。

## Claim Decomposition

| Dimension | Claimed novelty | Searchable formulation |
|---|---|---|
| framing | 同空间几何下的光谱细节分配 | geometry + spectral sensitivity + fusion |
| method | LR局部基的连续RU灵敏度耦合张量几何修正 | local subspace + sensor response + conditional injection |
| protocol | 同几何/同秩/不同响应的干预及部署基对照 | metamerism + geometry + controlled comparison |
| artifact | 当前合成检查可重跑，但不是新数据集 | actual R/D operator examples |
| finding | 尚无自然机制或训练收益 | 不能预先提出已证实结论 |

## Strongest Overlaps

见prior-art-matrix.md。SDP-Net、GGAT、DDIF、Matrix-R为强重合；dTV和SA-RFM排除首次方向引导；HSDiff排除同色异谱增强/不确定性首创。全文缺口使缺失描述不能作为差异证据。

## Novelty-Killing Objections

- SDP-Net已明确控制MSI结构传播和光谱偏移，可靠性门控故事直接重合。
- GGAT、dTV和SA-RFM已覆盖显式几何、方向及结构条件生成。
- Matrix-R、DDSR与标准子空间正则化覆盖响应投影，公式是常规正则化。
- RU来自低分辨率局部估计，可能遗漏HR材料变化，只有oracle基有效不构成部署方法。
- baseline的原MSI路径会绕过新增门控，不能声称控制了全部结构注入。
- 两个观测都不可见的扰动不可唯一恢复，物理一致性不是细节正确证明。

## Decision

- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 2
- Narrowest defensible positioning: 在已知响应、配准及稳定LR局部基条件下，检验方向几何与连续有效光谱响应灵敏度的条件耦合能否减少边界光谱分配误差，优于等预算门控与同信息量子空间对照；不宣称首次或已证实。
- What would change the decision: 若完整近邻已同构，或可部署局部基无效、自然失败不存在、等预算门控/经典求解解释全部收益，则降至2/5或放弃；完整查新和独立实验支持后才考虑提高。
- Literature assurance mode: unlinked
- Literature assurance verdict: targeted-search-only; material-fulltext-gaps
- Unresolved high-priority novelty-critical questions: NQ1, NQ2, NQ3, NQ4

仅决定定位，不验证实证正确、独立性或确认性。选择为结果/文献告知的探索候选；无新训练结果。合成例子非原创定理，也不代表自然样本普遍发生失败。

## Paper-Review Handoff

没有待审稿论文。交接时必须限定局部基与传感器条件，报告原MSI通路旁路风险、弱响应与零响应区别、自然机制缺口及所有近邻全文问题。不得以此判定直接进入确认性实验；先补近邻与诊断。
