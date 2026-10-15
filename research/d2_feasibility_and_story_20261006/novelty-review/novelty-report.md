# Novelty Report：D2 可行性再审查

## Contribution Under Review

具体方向为约束接口、受限物理任务联合响应及观测内部训练。先前通用交互与最近主文献已经审查，当前只补关键近邻；没有再次全面综述。

## Claim Decomposition

框架：特定训练取舍，不是解耦普遍有害。方法：投影接口+内部输出组合目标。评价：受限先验族、不可辨识冲突族与加性负对照。经验发现：全部待网络实验。

## Strongest Overlaps

MossFuse的角色分离、CLoRF的空间光谱组合、SHOTUN的关系保留、CDPS的观测先验；CoMM ICLR2025与SynCo2026预印本进一步覆盖通用协同训练。SynCo尚未核实正式期刊/代码，不影响其作为新颖性反证的相关性。

## Novelty-Killing Objections

联合零空间不可辨识；正交不等于无交互；辅助投影可逃避；内部降尺度项目本来已有；多数据/多预算可解释收益；只恢复损害不等于超过原baseline；通用协同训练已有近期先行工作。

## Decision

- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 3
- Narrowest defensible positioning: 在声明成像算子与结构先验的受限任务中，探索特定分离接口的正确联合响应取舍，以及观测内部组合训练是否超越同数据普通训练并改善自然融合。
- What would change the decision: 若合理分离没有功能取舍、简单弱约束或同数据普通训练解释全部收益、自然指标不改善或同构近期方法覆盖，则降级或重写；只有强对照与自然证据成立后才扩大定位。
- Literature assurance mode: unlinked
- Literature assurance verdict: targeted-search-only-not-systematic
- Unresolved high-priority novelty-critical questions: NQ1, NQ2, NQ3

通用协同或投影模块单独定位约1–2/5；3/5只支持有限探索，不是科学有效性、发表概率或收益认证。当前新代数检查说明边界与构造不空；没有训练新Net。已有负面定位与监督选模风险保留，不重新归零。

## Paper-Review Handoff

同一代理自审，不是独立审稿。主文和实验计划以C1/C2假设绑定；无实证audit的改进主张在paper-plan中blocked/omit，只能用研究问题语言。此前插值A1/A2审计并不支持这些新实验主张，未当作同scope证据引用。
