# Execution Bridge

计划交接；manual表示训练实现尚未完成，不是可以直接启动的命令。

### B1

- Claim IDs: C1
- Decision gate ID: G1
- Current/requested evidence class: exploratory
- Inputs: 显式split及观测算子、候选头、强对照、冻结evaluator
- Outputs: planned_results/B1.json
- Hidden information: GT不可用于条件；GT-hidden不可用于停止/选择
- Gate: 数学定义和实现一致；保留比较者可胜的无噪声满秩情形
- Failure: 算子或表示有错则先修复；反例无效则修改主张
- Retention: 全scene/seed与失败、skip和重试

### B2

- Claim IDs: C1
- Decision gate ID: G2
- Current/requested evidence class: exploratory
- Inputs: 显式split及观测算子、候选头、强对照、冻结evaluator
- Outputs: planned_results/B2.json
- Hidden information: GT不可用于条件；GT-hidden不可用于停止/选择
- Gate: 联合支持提供超出秩/纹理/RU/尺度ridge的解释，外部数据方向一致
- Failure: 不能解释自然失稳则收缩机制主张
- Retention: 全scene/seed与失败、skip和重试

### B3

- Claim IDs: C1, C2
- Decision gate ID: G3
- Current/requested evidence class: exploratory
- Inputs: 显式split及观测算子、候选头、强对照、冻结evaluator
- Outputs: planned_results/B3.json
- Hidden information: GT不可用于条件；GT-hidden不可用于停止/选择
- Gate: 边界收益且整体不劣，胜过同信息量学习门控或明确简化理由
- Failure: 只胜弱对照或靠额外损失则不能主张支持度机制
- Retention: 全scene/seed与失败、skip和重试

### B4

- Claim IDs: C2
- Decision gate ID: G4
- Current/requested evidence class: exploratory
- Inputs: 显式split及观测算子、候选头、强对照、冻结evaluator
- Outputs: planned_results/B4.json
- Hidden information: GT不可用于条件；GT-hidden不可用于停止/选择
- Gate: 在本任务明确范围内胜过或匹配合理强对照并展示关键边界优势
- Failure: 输给强方法则收缩为模块或诊断贡献
- Retention: 全scene/seed与失败、skip和重试

### B5

- Claim IDs: C1, C2
- Decision gate ID: G5
- Current/requested evidence class: exploratory
- Inputs: 显式split及观测算子、候选头、强对照、冻结evaluator
- Outputs: planned_results/B5.json
- Hidden information: GT不可用于条件；GT-hidden不可用于停止/选择
- Gate: 揭示明确的有效范围；两路线结果分开解释
- Failure: 算子错误下失效则限制结论；原生尺度失败则不能宣称无真值迁移成功
- Retention: 全scene/seed与失败、skip和重试

