# Novelty Report
## Contribution Under Review
A/I05：连续局部响应灵敏度是否比秩/纹理更能指导细谱修正。
B/I06：观测撤除下的条件功能分工是否超越普通留出训练。
C/I07：共享域相容条件，列为储备，不视作当前统一 D 的既有失败。
## Claim Decomposition
定义层：共享/互补、协同、观测行空间和 SVD 均已有，不声称新定义。
机制层：A 的任务化灵敏度接口、B 的观测条件输出路径可能有增量，但 NQ1/NQ2 未排尽。
训练层：实际留出测量合法；mask 与内部低分辨率目标均已有；没有套用 ES 理论。
实验层：只有代数例子与代码检查；自然问题、性能和功能分工未证实。
## Strongest Overlaps
RASD 覆盖局部路由；SRLF 覆盖观测失真选择性修正；CoMM/MossFuse 覆盖共享互补及交互；ES 覆盖测量拆分；SpecAware 覆盖传感器条件化。NUD/PAF 与 2026 波长 PSF 监督限制 C 的宽泛主张。
## Novelty-Killing Objections
1. A 可能是标准岭子空间求解换神经接口；必须比较普通正则化及固定 R 投影。
2. B 可能是普通留出训练加 R token；必须同目标、同预算比较。
3. 角色探针可能被最终路径绕过；检查屏蔽后输出与补偿。
4. 不能通过人工极强分离制造缺陷再恢复它；原 Net 没有该损失。
5. C 可能只是修正 D，不是新解耦机制；正确算子无共享是关键对照。
6. 联合零空间不能用同观测目标唯一恢复；合成目标是先验。
## Decision
- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 2
- Literature assurance mode: unlinked
- Literature assurance verdict: targeted-search-only-not-systematic
- Unresolved high-priority novelty-critical questions: NQ1, NQ2, NQ3
- Narrowest defensible positioning: 在已知算子与可声明的局部先验下，探索连续测量灵敏度或真实观测撤除所约束的功能分工，是否超越秩/纹理门控、普通留出训练与普通算子条件网络。
- What would change the decision: 若最新同构方法覆盖、自然问题不存在、LR 基不可用或简单强对照解释全部收益，则降级；只有同协议正常融合与功能证据成立，并补全关键近邻审查后才能扩大贡献。
宽泛模块版仅 1–2 分。3 分表示值得继续查证的窄假设，不表示新颖性已通过。未绑定系统文献包，不给 4–5 分。
## Paper-Review Handoff
先完成关键近邻全文与协议，再做小型实验，不先拟定发表级结论。两个优先候选必须各自通过正常融合和简单替代对照；组合最后做 2×2 验证。本轮为同助手自审，没有独立证据。
