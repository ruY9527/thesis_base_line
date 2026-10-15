# Novelty Report — 本次查新决定

## Contribution Under Review

日期：2026-10-07；窗口 2024-10-07—2026-10-07。完整前序证据见 ../cross_direction_review_20261007。此文件保留本次补检后的当前判断，不覆盖旧报告。

## Decision

架构方法 novelty_decision_rating=2：双专家 + sigmoid policy + endpoint 差额回归/BCE + 观测残差尚需重大差异化。经验研究问题 rating=3：只保留“在相同专家/信息/预算下，光谱收益辅助是否提供配准/观测可靠性之外的可重复增量”这一窄探索。decision_confidence_rating=2。

## Strongest Overlaps

EnsIR（NeurIPS 2024 main）已用带真值参考集求局部恢复专家权重；历史多专家回归 deferral 已覆盖 label-dependent cost；UAFL/UMNet/SRLF 与最新 RGB-HSR 先例限制可信度、误差尺度与回退的新颖性。详见 prior-art-addendum.md。没有任何候选在本次达到“无人提出已确认”或 rating 4/5。

## Novelty-Killing Objections

端点负收益不等于所有部分注入有害；BCE 胜率不等于最优连续强度。该反例已进入测试，原端点门只留作局限对照。当前 gain 原型使用独立重建 policy 与收益辅助头，不宣称最优期望策略或已校准概率。

## 搜索和阅读边界

本次新增/复读的精确范围、全文 sentinel 与失败尝试见 literature-addendum.md、sources-addendum.json、search-addendum.md。EnsIR 审查采用已核对的方法段与官方主会记录，未宣称本次完成其整份 PDF 阅读。CoPGNet/WCGE-Net 等重要全文仍未齐，不把摘要未提及等同机制不存在。

这不是有高召回保证的系统综述。子任务分开执行但共享前序上下文；没有跨模型或完全盲审。当前 ratings 约束写作定位，不能认证性能、科学有效性、实现正确性或投稿可行性。

## 选择历史和下一门槛

I13 来自本地旧报告的查新后重构，本次继承其候选而非新生原创；本次又根据 EnsIR 与数学反例缩窄。未根据收敛实验结果选择方向。当前只有测试和一个优化步骤的流程证据，属于探索阶段。

先测同专家 oracle 空间，再加普通 policy、成本敏感、oracle teacher、常数混合、EnsIR/回投影等对照；控制两个专家的容量、数据与更新。若常规对照同效、真实数据无稳定增量或新取得论文覆盖功能，方法定位降为1–2并重构。若找到可重复的独立失败模式及必要机制，补齐全文后重新查新；当前不提前升级。

## Claim Decomposition

|维度|可检索主张与状态|
|---|---|
|framing|可靠性之外的光谱收益，待验证窄问题|
|method|固定双专家与收益辅助policy，标准组合，不宣称新机制|
|protocol|四角色场景隔离与干预，可靠评价基础|
|artifact|新增可切换研究代码，工程交付而非科学新颖性证明|
|finding|没有正式实验发现|

## Paper-Review Handoff

当前没有待审稿件。任何未来稿件须删除首次GT权重、BCE最优强度及已经提升性能等未经支持的表述。

## Decision Record

- Novelty decision rating (1-5): 2
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 2
- Narrowest defensible positioning: Under identical frozen HSI-only and HSI-MSI experts, information and budgets, test whether explicit endpoint spectral-gain auxiliary supervision adds reproducible reconstruction benefit beyond ordinary policy, cost-sensitive weighting, oracle-weight teachers and reference-set ensembles.
- What would change the decision: Ordinary policy/cost-sensitive/ensemble comparisons match gain arm: reduce method novelty to 1-2; Functionally equivalent gain teacher found: abandon current first-method claim; Necessary mechanism and repeatable independent failure pattern supported by experiments: rerun bounded novelty review before promotion
- Literature assurance mode: unlinked
- Literature assurance verdict: bounded-unlinked; no systematic coverage assurance
- Unresolved high-priority novelty-critical questions: CoPGNet-fulltext, WCGE-fulltext, EnsIR-matched-comparator, VPG-fulltext
