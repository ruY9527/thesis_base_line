# Execution Bridge

这是后续实现交接，尚无训练入口，不虚构可运行训练命令。

## Block Hand-off

### B1

- Claim IDs: C1, C2
- Decision gate ID: G1
- Current evidence class: exploratory
- Requested evidence class: exploratory
- Expected implementation entrypoint: 新隔离目录中的训练/诊断入口，尚待实现。
- Expected command or notebook: 本轮仅feasibility_preflight.py可运行；主训练不可提交。
- Inputs required: model.py、utils.py、split-manifest.csv及只含观测的数据接口。
- Declared input snapshot paths: 与run-blocks.json相同；不是不可变保证。
- Declared evaluator snapshot paths: calculate_metrics.py。
- Output artifacts to produce: research/d2_feasibility_and_story_20261006/future-results/B1/summary.json，以及config、per_scene、per_seed、失败清单。
- Auditor-facing checks: float64算子差<1e-10、float32候选差<1e-5，非零可用梯度，split无重叠，目标与输入角色明确。
- Intended lineage relation: baseline、ablation、negative_control或replication，遵守声明。
- Parent run ID or rationale: 前驱科学门槛检查后的新运行；技术重试指向失败父运行。
- Hidden information unavailable to the evaluated system: 测试HR及其误差、研究者偏好结果。
- Failure, skip, null, timeout, and retry states to retain: 全部保留，禁止只保留最佳seed。
- Idempotency and restart requirements: 每次运行唯一ID；同配置恢复记录checkpoint与父运行，配置变化创建新ID。
- Known blockers: 可微算子与训练入口未实现，前驱门槛尚未通过；第二数据集未冻结。
- Downstream activation: 人工检查科学结果，不因文件存在或技术完成自动激活。

### B2

- Claim IDs: C1
- Decision gate ID: G2
- Current evidence class: exploratory
- Requested evidence class: exploratory
- Expected implementation entrypoint: 新隔离目录中的训练/诊断入口，尚待实现。
- Expected command or notebook: 本轮仅feasibility_preflight.py可运行；主训练不可提交。
- Inputs required: model.py、utils.py、split-manifest.csv及只含观测的数据接口。
- Declared input snapshot paths: 与run-blocks.json相同；不是不可变保证。
- Declared evaluator snapshot paths: calculate_metrics.py。
- Output artifacts to produce: research/d2_feasibility_and_story_20261006/future-results/B2/summary.json，以及config、per_scene、per_seed、失败清单。
- Auditor-facing checks: 联合参照能学；合理分离有可重复取舍且不是极强损失才出现；不将OOD普遍失败当损害。
- Intended lineage relation: baseline、ablation、negative_control或replication，遵守声明。
- Parent run ID or rationale: 前驱科学门槛检查后的新运行；技术重试指向失败父运行。
- Hidden information unavailable to the evaluated system: 测试HR及其误差、研究者偏好结果。
- Failure, skip, null, timeout, and retry states to retain: 全部保留，禁止只保留最佳seed。
- Idempotency and restart requirements: 每次运行唯一ID；同配置恢复记录checkpoint与父运行，配置变化创建新ID。
- Known blockers: 可微算子与训练入口未实现，前驱门槛尚未通过；第二数据集未冻结。
- Downstream activation: 人工检查科学结果，不因文件存在或技术完成自动激活。

### B3

- Claim IDs: C1
- Decision gate ID: G3
- Current evidence class: exploratory
- Requested evidence class: exploratory
- Expected implementation entrypoint: 新隔离目录中的训练/诊断入口，尚待实现。
- Expected command or notebook: 本轮仅feasibility_preflight.py可运行；主训练不可提交。
- Inputs required: model.py、utils.py、split-manifest.csv及只含观测的数据接口。
- Declared input snapshot paths: 与run-blocks.json相同；不是不可变保证。
- Declared evaluator snapshot paths: calculate_metrics.py。
- Output artifacts to produce: research/d2_feasibility_and_story_20261006/future-results/B3/summary.json，以及config、per_scene、per_seed、失败清单。
- Auditor-facing checks: 响应相对改善约10%且角色/完整误差没有明显恶化；简单替代不能解释全部效果。
- Intended lineage relation: baseline、ablation、negative_control或replication，遵守声明。
- Parent run ID or rationale: 前驱科学门槛检查后的新运行；技术重试指向失败父运行。
- Hidden information unavailable to the evaluated system: 测试HR及其误差、研究者偏好结果。
- Failure, skip, null, timeout, and retry states to retain: 全部保留，禁止只保留最佳seed。
- Idempotency and restart requirements: 每次运行唯一ID；同配置恢复记录checkpoint与父运行，配置变化创建新ID。
- Known blockers: 可微算子与训练入口未实现，前驱门槛尚未通过；第二数据集未冻结。
- Downstream activation: 人工检查科学结果，不因文件存在或技术完成自动激活。

### B4

- Claim IDs: C2
- Decision gate ID: G4
- Current evidence class: exploratory
- Requested evidence class: exploratory
- Expected implementation entrypoint: 新隔离目录中的训练/诊断入口，尚待实现。
- Expected command or notebook: 本轮仅feasibility_preflight.py可运行；主训练不可提交。
- Inputs required: model.py、utils.py、split-manifest.csv及只含观测的数据接口。
- Declared input snapshot paths: 与run-blocks.json相同；不是不可变保证。
- Declared evaluator snapshot paths: calculate_metrics.py。
- Output artifacts to produce: research/d2_feasibility_and_story_20261006/future-results/B4/summary.json，以及config、per_scene、per_seed、失败清单。
- Auditor-facing checks: 相对原架构内部任务与同四元普通训练PSNR暂以+0.10dB推进，RMSE不恶化、SAM差<=+0.02度，且重复方向一致。
- Intended lineage relation: baseline、ablation、negative_control或replication，遵守声明。
- Parent run ID or rationale: 前驱科学门槛检查后的新运行；技术重试指向失败父运行。
- Hidden information unavailable to the evaluated system: 测试HR及其误差、研究者偏好结果。
- Failure, skip, null, timeout, and retry states to retain: 全部保留，禁止只保留最佳seed。
- Idempotency and restart requirements: 每次运行唯一ID；同配置恢复记录checkpoint与父运行，配置变化创建新ID。
- Known blockers: 可微算子与训练入口未实现，前驱门槛尚未通过；第二数据集未冻结。
- Downstream activation: 人工检查科学结果，不因文件存在或技术完成自动激活。

### B5

- Claim IDs: C1, C2
- Decision gate ID: G5
- Current evidence class: exploratory
- Requested evidence class: exploratory
- Expected implementation entrypoint: 新隔离目录中的训练/诊断入口，尚待实现。
- Expected command or notebook: 本轮仅feasibility_preflight.py可运行；主训练不可提交。
- Inputs required: model.py、utils.py、split-manifest.csv及只含观测的数据接口。
- Declared input snapshot paths: 与run-blocks.json相同；不是不可变保证。
- Declared evaluator snapshot paths: calculate_metrics.py。
- Output artifacts to produce: research/d2_feasibility_and_story_20261006/future-results/B5/summary.json，以及config、per_scene、per_seed、失败清单。
- Auditor-facing checks: 强对照与另一个声明范围内场景集合重复成立，机制有效范围与失败先验清楚；近邻检索补全。
- Intended lineage relation: baseline、ablation、negative_control或replication，遵守声明。
- Parent run ID or rationale: 前驱科学门槛检查后的新运行；技术重试指向失败父运行。
- Hidden information unavailable to the evaluated system: 测试HR及其误差、研究者偏好结果。
- Failure, skip, null, timeout, and retry states to retain: 全部保留，禁止只保留最佳seed。
- Idempotency and restart requirements: 每次运行唯一ID；同配置恢复记录checkpoint与父运行，配置变化创建新ID。
- Known blockers: 可微算子与训练入口未实现，前驱门槛尚未通过；第二数据集未冻结。
- Downstream activation: 人工检查科学结果，不因文件存在或技术完成自动激活。

