# Novelty Report

## Contribution Under Review

本次对象是 I13 的窄收益监督方法假设；I14 仅作储备评价。工作模式 standalone / unlinked；同上下文自审，没有独立审查者。已使用 baseline、原有本地探索、核心全文与正式来源；不是穷尽性系统综述。

## Claim Decomposition

|维度|主张|状态|
|---|---|---|
|framing|匹配可靠性与实际光谱收益可能不同|可检验问题；不是首次“可靠引导”|
|method|两专家差额标签训练决策头|窄候选；通用门控与教师框架可能等价|
|protocol|固定专家下分离几何/可见域吻合与光谱收益|探索性诊断；未证明新协议|
|artifact|可复用研究与阅读包|本次交付，不是新数据集|
|finding|高可靠区域仍有负收益且收益监督可修正|尚无训练证据，不可写成结论|

I11/I12 原始主张已否定。I13 是在否定后生成的新假设；I14 读到动态解码与保谱损失后降为储备。没有本次模型实验结果参与选题，没有把结果知情选择包装成确认性研究。

## Strongest Overlaps

最新未配准先例、UAFL、UMNet、SRLF、DIFIV、TTSR、CVPR2022 Task Decoupled 与通用 MoE。Task Decoupled 主文已完整读取，已有冻结单图 SR、真值残差监督纹理分支与相关性迁移；它是必须进行的强对照。详细重叠、威胁评分和来源见 prior-art-matrix.md。CoPGNet 全文未得，是不能绕过的威胁。I14 与 PINO/SpecAware 的结构重叠很强，评分 2/5，判定置信度 3/5。

## Novelty-Killing Objections

1. “收益监督”可能只是普通 gate 的 oracle 标签设计，不产生必要的新机制。
2. 高几何可靠却有负收益的区域可能稀少，或其问题由现有结构可靠性分数解释。
3. 净收益可能来自多一个 HSI-only 专家；必须比较同一专家对和同预算普通门控。
4. CoPGNet 或参考 SR 可能已直接学习外部先验的增量收益；缺全文或检索无命中不是不存在。
5. 相同观测的共同零空间歧义不能由观测头逐例解决；不得声称恢复保证。

## Decision

- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 2
- Narrowest defensible positioning: I13：仅探索在固定专家与同等配准条件下，以监督光谱误差差额训练细节注入决策，是否能超过匹配置信度、测量残差和普通专家门控。
- What would change the decision: 若 CoPGNet 或参考 SR/专家门控已有功能等价收益教师，或普通 gate/置信度基线达到同等效果，降为 1–2；若高召回补检与独立实验同时支持必要机制，才重审定位。
- Literature assurance mode: unlinked
- Literature assurance verdict: bounded-exploration-with-material-gaps
- Unresolved high-priority novelty-critical questions: Q1, Q2, Q3

评分只支撑窄探索定位，不认证科学原创、效果、实现正确性、确认性状态或投稿准备度。不能声称首次门控/回退/教师，不能声称已确认无人提出或独立查新。I14 不进入独立主贡献清单。

Q1：CoPGNet 的先验权重是否用收益目标监督？Q2：参考 SR/专家路由是否已有功能等价收益教师？Q3：普通 g* 教师和端到端 gate 是否已足够？Q4（储备）：完整 SRF 形状条件是否提供超出 PINO+SpecAware 常规组合的必要机制？

## Paper-Review Handoff

当前没有具体新稿件可审。后续实验计划应先做两专家失败模式与 oracle 上限，再公平比较简单强对照；只有验证后才起草贡献句。保留 I11/I12 的否定历史、Q1–Q3 和探索性证据身份。主报告中打印公式核查只涉及 Diff-Spectra 排版与维度，不扩展为代码或整体结论错误。
