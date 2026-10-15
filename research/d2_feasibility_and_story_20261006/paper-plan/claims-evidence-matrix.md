# Claims-Evidence Matrix

经验发现暂时omit；开题可写成问题，不可改写为已支持结果。

| Paper claim ID | Claim | Type | Evidence mode | Support status | Manuscript action | Required assurance | Source claim IDs | Audit IDs | Excluded audit IDs | Planned sections | Exhibit IDs | Citation need IDs | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC0 | 近期已有角色解耦、空间光谱组合与协同训练，通用交互不是空白。 | context | citation | supported | assert | none |  |  |  | Introduction, Related Work |  | CN1, CN2, CN3 | 定向覆盖，不是穷尽综述。 |
| PC1 | 特定分离接口在合理设置下影响正确联合响应。 | primary | empirical | blocked | omit | exploratory | C1 |  |  | Introduction, Methods | F1, F2 | CN1, CN3 | 没有网络结果；只在声明先验和容量范围提出假设。 |
| PC2 | 内部组合训练超越同数据普通训练，并改善自然PSNR/RMSE/SAM。 | supporting | empirical | blocked | omit | exploratory | C2 |  |  | Introduction, Methods | F3, F4 | CN2, CN3 | 没有自然新指标；不保证跨尺度与跨先验泛化。 |
| PC3 | 联合零空间存在同观测不同真值，只能在声明结构先验内讨论恢复。 | limitation | theoretical | supported | limitation | none |  |  |  | Introduction, Methods | F0 |  | 程序样本检查不证明自然恢复；受限oracle知道生成常量。 |
