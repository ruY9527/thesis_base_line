# Experiment Plan

## Context

Standalone exploratory规划；两路线同等；没有训练结果或独立审计。详见 deep-research-report.md。历史选择：CAVE既有测试已查看；人工反例是本轮设计验证。

## Claim Map

| Claim ID | Type | Claim | Falsifier |
| --- | --- | --- | --- |
| C1 | primary | 局部联合支持能解释超出rank/texture/RU/scale-ridge的自然几何失稳 | 控制纹理/尺度后无解释增量 |
| C2 | primary | 额外物理几何残差在两训练路线中改善边界光谱归属而整体不劣 | 强同信息对照等效或更好，或整体谱质量下降 |

## Experimental Storyline

| Block | Role | Paper placement | Why it exists |
| --- | --- | --- | --- |
| B1 | mechanism/control | main | 算子、模式与旋转不变性检查 |
| B2 | mechanism/control | main | 自然场景机制诊断 |
| B3 | mechanism/control | main | 共享骨干机制消融 |
| B4 | mechanism/control | main | 三类强方法公平比较 |
| B5 | mechanism/control | main | 失败边界与无真值尺度迁移 |

## Non-Vacuity Preflight

人工双零模式与粗空间零谱模式已执行；无噪声满秩伪逆和同信息门控可胜，禁止强制所有setting获益。仅B1数值子项完成，自然机制待测。

## Experiment Blocks

### B1

- block_id: B1
- paper_role: main
- claim_ids: C1
- decision_gate_id: G1
- anti_claims_ruled_out: 增益来自容量、尺度正则或额外观测信息
- why_this_block_exists: 算子、模式与旋转不变性检查
- dataset_split_task: 基线R/D、人工粗/细及谱可见/零模式；不使用自然指标
- systems_compared: MSI-only; joint support; classical restricted ridge
- fixed_factors: observation access; loss within each route; selection rules; budget
- variable_factors: spatial scale; spectral visibility
- metrics: operator residual; basis rotation error; joint spectrum
- setup_details: 当前CAVE CNN起点；局部谱基和模板定义先冻结；训练实现未完成
- seeds: 3
- success_criterion: 数学定义和实现一致；保留比较者可胜的无噪声满秩情形
- minimum_effect_size: 数值容差按float64设1e-10，非性能门槛
- failure_interpretation: 算子或表示有错则先修复；反例无效则修改主张
- expected_output_artifact: planned_results/B1.json
- compute_budget: 先小规模pilot，按实际硬件测量；本轮没有启动训练
- dependencies: 
- priority: must-run
- current_evidence_class: exploratory
- evidence_class: exploratory
- selection_rule: 预先manifest及案例规则；当前历史测试已被查看，不充当新盲测
- non_vacuity_check: 无噪声满秩下不收缩解可胜出；同信息学习门控可胜；保留这些情形
- complete_outcome_accounting: 保留每scene/seed的负结果、错误、skip、超时和技术重试；不得仅存best
- hidden_information_controls: 条件只能读观测；GT-hidden路线禁止GT用于条件/停止/选择；留出观测也从条件移除
- independence_requirements: 最终外部scene未参与方法选择; 评价与条件的数据访问分离
- independence_evidence: 当前没有独立复现或审查，仅规划要求
- predecessor_failures: RU-only不足以表示完整融合可辨性; 原测试选择历史需保留

### B2

- block_id: B2
- paper_role: main
- claim_ids: C1
- decision_gate_id: G2
- anti_claims_ruled_out: 增益来自容量、尺度正则或额外观测信息
- why_this_block_exists: 自然场景机制诊断
- dataset_split_task: CAVE训练/验证观测＋独立数据；冻结evaluator，显式scene manifest
- systems_compared: rank and texture predictor; RU-only predictor; joint-support predictor
- fixed_factors: observation access; loss within each route; selection rules; budget
- variable_factors: support grouping; template bandwidth
- metrics: scene-level boundary SAM; mode error; incremental predictive value
- setup_details: 当前CAVE CNN起点；局部谱基和模板定义先冻结；训练实现未完成
- seeds: 3
- success_criterion: 联合支持提供超出秩/纹理/RU/尺度ridge的解释，外部数据方向一致
- minimum_effect_size: 解释增量须场景配对区间不跨零；预实验前确定模型与复杂度
- failure_interpretation: 不能解释自然失稳则收缩机制主张
- expected_output_artifact: planned_results/B2.json
- compute_budget: 先小规模pilot，按实际硬件测量；本轮没有启动训练
- dependencies: B1
- priority: must-run
- current_evidence_class: exploratory
- evidence_class: exploratory
- selection_rule: 预先manifest及案例规则；当前历史测试已被查看，不充当新盲测
- non_vacuity_check: 无噪声满秩下不收缩解可胜出；同信息学习门控可胜；保留这些情形
- complete_outcome_accounting: 保留每scene/seed的负结果、错误、skip、超时和技术重试；不得仅存best
- hidden_information_controls: 条件只能读观测；GT-hidden路线禁止GT用于条件/停止/选择；留出观测也从条件移除
- independence_requirements: 最终外部scene未参与方法选择; 评价与条件的数据访问分离
- independence_evidence: 当前没有独立复现或审查，仅规划要求
- predecessor_failures: RU-only不足以表示完整融合可辨性; 原测试选择历史需保留

### B3

- block_id: B3
- paper_role: main
- claim_ids: C1; C2
- decision_gate_id: G3
- anti_claims_ruled_out: 增益来自容量、尺度正则或额外观测信息
- why_this_block_exists: 共享骨干机制消融
- dataset_split_task: 分别supervised及GT-hidden；观测条件完全相同，3seeds
- systems_compared: same-capacity learned geometry gate; RU-only gate; joint physical gate; same-information learned joint gate; scale-adaptive ridge
- fixed_factors: observation access; loss within each route; selection rules; budget
- variable_factors: gate rule; geometry condition
- metrics: PSNR; SAM; boundary SAM; spectral gradient error; runtime
- setup_details: 当前CAVE CNN起点；局部谱基和模板定义先冻结；训练实现未完成
- seeds: 3
- success_criterion: 边界收益且整体不劣，胜过同信息量学习门控或明确简化理由
- minimum_effect_size: boundary SAM >=5%相对改善；PSNR下降<=0.1dB；整体SAM恶化<=1%；规划阈值尚待冻结
- failure_interpretation: 只胜弱对照或靠额外损失则不能主张支持度机制
- expected_output_artifact: planned_results/B3.json
- compute_budget: 先小规模pilot，按实际硬件测量；本轮没有启动训练
- dependencies: B2
- priority: must-run
- current_evidence_class: exploratory
- evidence_class: exploratory
- selection_rule: 预先manifest及案例规则；当前历史测试已被查看，不充当新盲测
- non_vacuity_check: 无噪声满秩下不收缩解可胜出；同信息学习门控可胜；保留这些情形
- complete_outcome_accounting: 保留每scene/seed的负结果、错误、skip、超时和技术重试；不得仅存best
- hidden_information_controls: 条件只能读观测；GT-hidden路线禁止GT用于条件/停止/选择；留出观测也从条件移除
- independence_requirements: 最终外部scene未参与方法选择; 评价与条件的数据访问分离
- independence_evidence: 当前没有独立复现或审查，仅规划要求
- predecessor_failures: RU-only不足以表示完整融合可辨性; 原测试选择历史需保留

### B4

- block_id: B4
- paper_role: main
- claim_ids: C2
- decision_gate_id: G4
- anti_claims_ruled_out: 增益来自容量、尺度正则或额外观测信息
- why_this_block_exists: 三类强方法公平比较
- dataset_split_task: 每路线独立比较预算与checkpoint；测试GT仅供冻结evaluator
- systems_compared: geometry-gated neural family; local low-rank physical family; observation-self-supervised family
- fixed_factors: observation access; loss within each route; selection rules; budget
- variable_factors: method family
- metrics: scene PSNR/SAM/ERGAS; boundary SAM; offline/scene adaptation/inference time
- setup_details: 当前CAVE CNN起点；局部谱基和模板定义先冻结；训练实现未完成
- seeds: 3
- success_criterion: 在本任务明确范围内胜过或匹配合理强对照并展示关键边界优势
- minimum_effect_size: 沿用B3质量门槛，额外报告Pareto权衡
- failure_interpretation: 输给强方法则收缩为模块或诊断贡献
- expected_output_artifact: planned_results/B4.json
- compute_budget: 先小规模pilot，按实际硬件测量；本轮没有启动训练
- dependencies: B3
- priority: must-run
- current_evidence_class: exploratory
- evidence_class: exploratory
- selection_rule: 预先manifest及案例规则；当前历史测试已被查看，不充当新盲测
- non_vacuity_check: 无噪声满秩下不收缩解可胜出；同信息学习门控可胜；保留这些情形
- complete_outcome_accounting: 保留每scene/seed的负结果、错误、skip、超时和技术重试；不得仅存best
- hidden_information_controls: 条件只能读观测；GT-hidden路线禁止GT用于条件/停止/选择；留出观测也从条件移除
- independence_requirements: 最终外部scene未参与方法选择; 评价与条件的数据访问分离
- independence_evidence: 当前没有独立复现或审查，仅规划要求
- predecessor_failures: RU-only不足以表示完整融合可辨性; 原测试选择历史需保留

### B5

- block_id: B5
- paper_role: main
- claim_ids: C1; C2
- decision_gate_id: G5
- anti_claims_ruled_out: 增益来自容量、尺度正则或额外观测信息
- why_this_block_exists: 失败边界与无真值尺度迁移
- dataset_split_task: 预定noise/SRF/PSF/basis perturbations；真实无HR数据辅助评价
- systems_compared: geometry-gated neural family; local low-rank physical family; observation-self-supervised family
- fixed_factors: observation access; loss within each route; selection rules; budget
- variable_factors: noise; operator error; basis coverage; scale
- metrics: boundary and mode errors; held-out observation prediction; consistency; failure frequency
- setup_details: 当前CAVE CNN起点；局部谱基和模板定义先冻结；训练实现未完成
- seeds: 3
- success_criterion: 揭示明确的有效范围；两路线结果分开解释
- minimum_effect_size: 不得用无参考质量分数替代HR谱精度；不要求所有setting胜出
- failure_interpretation: 算子错误下失效则限制结论；原生尺度失败则不能宣称无真值迁移成功
- expected_output_artifact: planned_results/B5.json
- compute_budget: 先小规模pilot，按实际硬件测量；本轮没有启动训练
- dependencies: B4
- priority: must-run
- current_evidence_class: exploratory
- evidence_class: exploratory
- selection_rule: 预先manifest及案例规则；当前历史测试已被查看，不充当新盲测
- non_vacuity_check: 无噪声满秩下不收缩解可胜出；同信息学习门控可胜；保留这些情形
- complete_outcome_accounting: 保留每scene/seed的负结果、错误、skip、超时和技术重试；不得仅存best
- hidden_information_controls: 条件只能读观测；GT-hidden路线禁止GT用于条件/停止/选择；留出观测也从条件移除
- independence_requirements: 最终外部scene未参与方法选择; 评价与条件的数据访问分离
- independence_evidence: 当前没有独立复现或审查，仅规划要求
- predecessor_failures: RU-only不足以表示完整融合可辨性; 原测试选择历史需保留

## Run Order

| Order | Block | Purpose | Dependency | Gate ID | Stop / go gate | Est. cost |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | B1 | 算子、模式与旋转不变性检查 |  | G1 | 算子或表示有错则先修复；反例无效则修改主张 | 先小规模pilot，按实际硬件测量；本轮没有启动训练 |
| 2 | B2 | 自然场景机制诊断 | B1 | G2 | 不能解释自然失稳则收缩机制主张 | 先小规模pilot，按实际硬件测量；本轮没有启动训练 |
| 3 | B3 | 共享骨干机制消融 | B2 | G3 | 只胜弱对照或靠额外损失则不能主张支持度机制 | 先小规模pilot，按实际硬件测量；本轮没有启动训练 |
| 4 | B4 | 三类强方法公平比较 | B3 | G4 | 输给强方法则收缩为模块或诊断贡献 | 先小规模pilot，按实际硬件测量；本轮没有启动训练 |
| 5 | B5 | 失败边界与无真值尺度迁移 | B4 | G5 | 算子错误下失效则限制结论；原生尺度失败则不能宣称无真值迁移成功 | 先小规模pilot，按实际硬件测量；本轮没有启动训练 |

## Decision Gates

| Gate ID | Opens after | Decision question | Proceed if | Revise if | Stop if | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | B1 | 算子、模式与旋转不变性检查 | 数学定义和实现一致；保留比较者可胜的无噪声满秩情形 | 算子或表示有错则先修复；反例无效则修改主张 | 核心假设被强对照否定则停止扩张 | 研究者 |
| G2 | B2 | 自然场景机制诊断 | 联合支持提供超出秩/纹理/RU/尺度ridge的解释，外部数据方向一致 | 不能解释自然失稳则收缩机制主张 | 核心假设被强对照否定则停止扩张 | 研究者 |
| G3 | B3 | 共享骨干机制消融 | 边界收益且整体不劣，胜过同信息量学习门控或明确简化理由 | 只胜弱对照或靠额外损失则不能主张支持度机制 | 核心假设被强对照否定则停止扩张 | 研究者 |
| G4 | B4 | 三类强方法公平比较 | 在本任务明确范围内胜过或匹配合理强对照并展示关键边界优势 | 输给强方法则收缩为模块或诊断贡献 | 核心假设被强对照否定则停止扩张 | 研究者 |
| G5 | B5 | 失败边界与无真值尺度迁移 | 揭示明确的有效范围；两路线结果分开解释 | 算子错误下失效则限制结论；原生尺度失败则不能宣称无真值迁移成功 | 核心假设被强对照否定则停止扩张 | 研究者 |

## Risks and Confounds

GT条件泄漏、谱基遗漏、patch退化边界、噪声权重、尺度ridge等价、旧测试选择。措施和指标定义见主报告5–9节。规划效应门槛未经结果调优，开始训练前需要冻结，不是已预注册结论。
