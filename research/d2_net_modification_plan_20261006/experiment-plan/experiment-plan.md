# Experiment Plan

## Context

- Operating mode: standalone；本轮交付修改计划，不执行训练或维护多代理工作流。
- Problem: 来源区分是否保留光谱—空间配合，以及如何在无HR标签下训练。
- Evaluation goal: 先判断M1结构是否值得继续，再判断M2和M3；原始Net必须可以赢。
- Current/requested evidence class: exploratory。
- Main artifact: [详细代码方案](D:/coding/github_self/thesis_base_line/research/d2_net_modification_plan_20261006/d2_net_modification_plan.md)。
- Upstream: 原model.py、训练/数据/指标、已有D2与baseline诊断、作者固定提交代码。
- Selection history: 旧12场景和历史结果已查看，不包装为盲测。未有新Net结果。
- Material predecessor failures: 泛交互已有先例；当前Net没有直接分离；历史训练不是selfsup；不可辨识无法无条件解决。

## Claim Map

| Claim ID | Type | Why it matters | Minimum convincing evidence | Anti-claim to rule out | Falsifier | Decision if unproven |
|---|---|---|---|---|---|---|
| C1 | primary | 结构与功能联系 | B1/B2公平对照，讲取舍需B3 | 加参/平滑/极强损失解释 | 简单模型同样好 | reframe |
| C2 | supporting | 无HR训练依据 | B4同四组、同预算、无HR隔离 | 信息/预算/HR泄漏 | 自然无收益或普通目标解释 | defer |

## Experimental Storyline

| Block | Role | Paper placement | Why it exists |
|---|---|---|---|
| B0 | protocol | appendix | 公共协议、兼容性和算子 |
| B1 | anchor | main | 用户先跑的动机一 |
| B2 | allocation | main | 动机二局部分配 |
| B3 | mechanism | main if claimed | 因果故事不能由命名得到 |
| B4 | selfsup | main if successful | 动机三训练机制 |

## Non-Vacuity Preflight

新Net尚未实现，本轮不能报告网络预检通过。实施后：关闭和零尺度应等价；加性/普通MLP可胜；边界路由有效；同四组完整误差必须进入loss；内部目标可学但不要求恢复任意joint-null真值。
生成模式、场景清单不按想要的结果选择；失败/NaN/OOM/空区域保留。具体清单见主方案第8节。

## Experiment Blocks

### B0 — 公共协议与未来技术验收

- Decision gate ID: G0
- Claim tested: C1, C2
- Priority: must-run
- Systems compared: 原始Net；关闭/零尺度候选；原NumPy R/D 与未来torch R/D
- Setup details: 无新增源码时只能做计划审查；模型测试必须等实现。
- Metrics: 输出等价误差；形状/梯度；划分/保存；算子误差
- Seeds: 1；B1先1seed筛查，其余重复用于继续判断。
- Success criterion: 公共初值可对齐、Tensor接口一致、checkpoint可复测；D float64<1e-10/float32<1e-5才能开物理损失。
- Failure interpretation: 协议或实现不一致先修复；不运行科学比较。
- Expected output: research/d2_net_modification_plan_20261006/future-results/B0/summary.json

### B1 — M1-A：来源交互首轮筛查

- Decision gate ID: G1
- Claim tested: C1
- Priority: must-run
- Systems compared: E0；E1；E1-add；E1-mlp
- Setup details: 原有监督HR-L1；64/8块、dim64/8blocks；所有组复制共同原Net初始化。
- Metrics: PSNR；RMSE；SAM；eta/饱和/方差；时间/显存
- Seeds: 3；B1先1seed筛查，其余重复用于继续判断。
- Success criterion: 单seed验证PSNR增量>=0.10dB、RMSE不升、SAM相对恶化<=1%；之后至少2/3seed正增益且简单对照不能解释全部收益。
- Failure interpretation: 若简单对照更好收窄结构解释；稳定下降则停止M2累加路线。
- Expected output: research/d2_net_modification_plan_20261006/future-results/B1/summary.json

### B2 — M2：局部光谱分配

- Decision gate ID: G2
- Claim tested: C1
- Priority: defer
- Systems compared: E1；E2；E2-uniform；E2-msi；E2-hsi；E2-conv
- Setup details: 3x3窗口、query/key16；固定同模型/损失训练协议。
- Metrics: 整体PSNR/RMSE/SAM；固定输入边缘区RMSE/SAM；rho/邻居权重；时间/显存
- Seeds: 3；B1先1seed筛查，其余重复用于继续判断。
- Success criterion: 相对E1验证PSNR>=0.10dB、RMSE不升、SAM相对恶化<=1%；边缘无系统恶化，再3seed和局部对照。
- Failure interpretation: 若均匀/卷积解释收益，不讲光谱分配机制；退化或OOM先修实现再判科学门槛。
- Expected output: research/d2_net_modification_plan_20261006/future-results/B2/summary.json

### B3 — M1-B：分离原因机制验证

- Decision gate ID: G3
- Claim tested: C1
- Priority: nice-to-have
- Systems compared: D0；D1弱/中分离；D2投影分离；bridge关闭/开启
- Setup details: 先过D预检；分离梯度比0/0.01/0.1，极强值只压力测试。
- Metrics: 角色读出误差；样本方差/范数；分离梯度比；联合响应/局部分配；PSNR/RMSE/SAM
- Seeds: 3；B1先1seed筛查，其余重复用于继续判断。
- Success criterion: 合理强度有可重复具体取舍且接口改善；弱约束、role-only可以竞争，投影不靠塌缩。
- Failure interpretation: 若没有取舍，放弃分离有害解释，保留经B1/B2支持的结构收益。
- Expected output: research/d2_net_modification_plan_20261006/future-results/B3/summary.json

### B4 — M3：无HR的训练依据

- Decision gate ID: G4
- Claim tested: C2
- Priority: defer
- Systems compared: 原Net与冻结改进Net；obs-only；internal；same-quartet plain；same-quartet plus mix
- Setup details: 不使用HR训练权重；内部32x32目标来自完整观测L；先2000steps共同筛查再10000steps共同预算。
- Metrics: Lobs；四组完整误差；Lmix；自然PSNR/RMSE/SAM；前向/step/时间；失败/冲突
- Seeds: 3；B1先1seed筛查，其余重复用于继续判断。
- Success criterion: 无HR泄漏、算子一致、生成任务可学且完整误差与自然指标有改善；同四组/预算比较支持训练价值。
- Failure interpretation: Lmix独有收益不成立则去掉；HR泄漏/不可学/算子错则先修，绝不宣称唯一恢复。
- Expected output: research/d2_net_modification_plan_20261006/future-results/B4/summary.json

## Run Order

| Order | Block | Priority | Depends on | Gate | Proceed if | Stop/revise if |
|---|---|---|---|---|---|---|
| 0 | B0 | must-run | 初始 | G0 | 协议与接口一致 | 修复不一致 |
| 1 | B1 | must-run | B0 | G1 | 首seed数值有希望，再对照/重复 | 原模型或简单对照更好 |
| 2 | B2 | defer | B1科研门槛 | G2 | 局部分配及指标改善 | 平滑/坐标解释或退化 |
| 3 | B3 | nice-to-have | B0/B1；要讲解耦取舍则必做 | G3 | 合理强度可重复取舍 | 撤销因果叙述 |
| 4 | B4 | defer | B0/B1；结构冻结，B2可选 | G4 | 无HR同协议收益 | 撤销联合训练价值 |

## Decision Gates

见decision-gates.md。技术完成不是科学通过；文件存在不激活后续。门槛仅用于探索性投入判断。

## Risks and Confounds

公共初始化、test选模、相位、拼块、额外参数/预算、角色塌缩、局部误匹配、无HR泄漏与生成先验迁移。未来训练入口未实现，当前不提供可运行的新模型命令。主方案第7/9节给出修正与失败解释。
