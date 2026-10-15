# Execution Bridge

本轮是手动实施前方案，没有新增可运行训练入口。模型结构位置/张量契约见主方案；执行前必须核对当前源码摘要。原程序名称可以定位现有逻辑，但不能把未来model_d2.py的命令描述成当前已可运行。

### B0

- Claim IDs: C1, C2
- Decision gate: G0
- Evidence class: exploratory
- Declared inputs: model.py、utils.py、冻结split-manifest.csv、详细代码方案；实现后加新Net/训练器摘要。
- Declared evaluator artifacts: calculate_metrics.py；当前reconstruction流程统一使用，任何变化单列因素。
- Entrypoint: manual；按主方案对应章节由未来实现者创建/接入，不存在可直接执行的新命令。
- Required outputs: research/d2_net_modification_plan_20261006/future-results/B0/summary.json，以及config、checkpoint、逐场景CSV、failure与时间/显存。
- Lineage relation: baseline / ablation / replication / parameter_variation / negative_control；重试必须关联父run。
- Complete loss/outcome contract: 所有OOM/NaN/跳过/超时/未跑/失败seed保留；技术重试关联父run。
- Hidden-information controls: B1/B2/B3监督筛查允许HR训练但只验证选模；B4训练/初始化不得读HR；固定预算或观测HSI内部验证选模，冻结后HR只评估。
- Blockers: 新结构/训练器尚未实现；无新增源码时只能做计划审查；模型测试必须等实现。
- Downstream activation: B0协议先过；B2需G1科研门槛且用户查看首轮数值后继续；B3讲解耦取舍才必做；B4只在无HR协议和结构冻结后人工启用。
- Acceptance: 公共初值可对齐、Tensor接口一致、checkpoint可复测；D float64<1e-10/float32<1e-5才能开物理损失。
- Failure interpretation: 协议或实现不一致先修复；不运行科学比较。

### B1

- Claim IDs: C1
- Decision gate: G1
- Evidence class: exploratory
- Declared inputs: model.py、utils.py、冻结split-manifest.csv、详细代码方案；实现后加新Net/训练器摘要。
- Declared evaluator artifacts: calculate_metrics.py；当前reconstruction流程统一使用，任何变化单列因素。
- Entrypoint: manual；按主方案对应章节由未来实现者创建/接入，不存在可直接执行的新命令。
- Required outputs: research/d2_net_modification_plan_20261006/future-results/B1/summary.json，以及config、checkpoint、逐场景CSV、failure与时间/显存。
- Lineage relation: baseline / ablation / replication / parameter_variation / negative_control；重试必须关联父run。
- Complete loss/outcome contract: 所有OOM/NaN/跳过/超时/未跑/失败seed保留；技术重试关联父run。
- Hidden-information controls: B1/B2/B3监督筛查允许HR训练但只验证选模；B4训练/初始化不得读HR；固定预算或观测HSI内部验证选模，冻结后HR只评估。
- Blockers: 新结构/训练器尚未实现；原有监督HR-L1；64/8块、dim64/8blocks；所有组复制共同原Net初始化。
- Downstream activation: B0协议先过；B2需G1科研门槛且用户查看首轮数值后继续；B3讲解耦取舍才必做；B4只在无HR协议和结构冻结后人工启用。
- Acceptance: 单seed验证PSNR增量>=0.10dB、RMSE不升、SAM相对恶化<=1%；之后至少2/3seed正增益且简单对照不能解释全部收益。
- Failure interpretation: 若简单对照更好收窄结构解释；稳定下降则停止M2累加路线。

### B2

- Claim IDs: C1
- Decision gate: G2
- Evidence class: exploratory
- Declared inputs: model.py、utils.py、冻结split-manifest.csv、详细代码方案；实现后加新Net/训练器摘要。
- Declared evaluator artifacts: calculate_metrics.py；当前reconstruction流程统一使用，任何变化单列因素。
- Entrypoint: manual；按主方案对应章节由未来实现者创建/接入，不存在可直接执行的新命令。
- Required outputs: research/d2_net_modification_plan_20261006/future-results/B2/summary.json，以及config、checkpoint、逐场景CSV、failure与时间/显存。
- Lineage relation: baseline / ablation / replication / parameter_variation / negative_control；重试必须关联父run。
- Complete loss/outcome contract: 所有OOM/NaN/跳过/超时/未跑/失败seed保留；技术重试关联父run。
- Hidden-information controls: B1/B2/B3监督筛查允许HR训练但只验证选模；B4训练/初始化不得读HR；固定预算或观测HSI内部验证选模，冻结后HR只评估。
- Blockers: 新结构/训练器尚未实现；3x3窗口、query/key16；固定同模型/损失训练协议。
- Downstream activation: B0协议先过；B2需G1科研门槛且用户查看首轮数值后继续；B3讲解耦取舍才必做；B4只在无HR协议和结构冻结后人工启用。
- Acceptance: 相对E1验证PSNR>=0.10dB、RMSE不升、SAM相对恶化<=1%；边缘无系统恶化，再3seed和局部对照。
- Failure interpretation: 若均匀/卷积解释收益，不讲光谱分配机制；退化或OOM先修实现再判科学门槛。

### B3

- Claim IDs: C1
- Decision gate: G3
- Evidence class: exploratory
- Declared inputs: model.py、utils.py、冻结split-manifest.csv、详细代码方案；实现后加新Net/训练器摘要。
- Declared evaluator artifacts: calculate_metrics.py；当前reconstruction流程统一使用，任何变化单列因素。
- Entrypoint: manual；按主方案对应章节由未来实现者创建/接入，不存在可直接执行的新命令。
- Required outputs: research/d2_net_modification_plan_20261006/future-results/B3/summary.json，以及config、checkpoint、逐场景CSV、failure与时间/显存。
- Lineage relation: baseline / ablation / replication / parameter_variation / negative_control；重试必须关联父run。
- Complete loss/outcome contract: 所有OOM/NaN/跳过/超时/未跑/失败seed保留；技术重试关联父run。
- Hidden-information controls: B1/B2/B3监督筛查允许HR训练但只验证选模；B4训练/初始化不得读HR；固定预算或观测HSI内部验证选模，冻结后HR只评估。
- Blockers: 新结构/训练器尚未实现；先过D预检；分离梯度比0/0.01/0.1，极强值只压力测试。
- Downstream activation: B0协议先过；B2需G1科研门槛且用户查看首轮数值后继续；B3讲解耦取舍才必做；B4只在无HR协议和结构冻结后人工启用。
- Acceptance: 合理强度有可重复具体取舍且接口改善；弱约束、role-only可以竞争，投影不靠塌缩。
- Failure interpretation: 若没有取舍，放弃分离有害解释，保留经B1/B2支持的结构收益。

### B4

- Claim IDs: C2
- Decision gate: G4
- Evidence class: exploratory
- Declared inputs: model.py、utils.py、冻结split-manifest.csv、详细代码方案；实现后加新Net/训练器摘要。
- Declared evaluator artifacts: calculate_metrics.py；当前reconstruction流程统一使用，任何变化单列因素。
- Entrypoint: manual；按主方案对应章节由未来实现者创建/接入，不存在可直接执行的新命令。
- Required outputs: research/d2_net_modification_plan_20261006/future-results/B4/summary.json，以及config、checkpoint、逐场景CSV、failure与时间/显存。
- Lineage relation: baseline / ablation / replication / parameter_variation / negative_control；重试必须关联父run。
- Complete loss/outcome contract: 所有OOM/NaN/跳过/超时/未跑/失败seed保留；技术重试关联父run。
- Hidden-information controls: B1/B2/B3监督筛查允许HR训练但只验证选模；B4训练/初始化不得读HR；固定预算或观测HSI内部验证选模，冻结后HR只评估。
- Blockers: 新结构/训练器尚未实现；不使用HR训练权重；内部32x32目标来自完整观测L；先2000steps共同筛查再10000steps共同预算。
- Downstream activation: B0协议先过；B2需G1科研门槛且用户查看首轮数值后继续；B3讲解耦取舍才必做；B4只在无HR协议和结构冻结后人工启用。
- Acceptance: 无HR泄漏、算子一致、生成任务可学且完整误差与自然指标有改善；同四组/预算比较支持训练价值。
- Failure interpretation: Lmix独有收益不成立则去掉；HR泄漏/不可学/算子错则先修，绝不宣称唯一恢复。

## Restart and idempotency

每次run新目录；同run恢复原checkpoint/optimizer/scheduler/RNG/配置，不覆盖失败记录或静默换数据。改变因素创建新run并记录关系。不因计划结构校验通过而自动运行训练。
