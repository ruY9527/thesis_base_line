# Landscape Map
## Scope
已配准 HSI–MSI 自监督融合；优先已知 R、正确 D，原 Net 作为保留对照。以 2025–2026 近期论文排查为主，旧线性代数作为工具不包装成新理论。
## Existing Context Used
model.py、utils.py、train_cave_M.py；p01_decoupling_research_directions.md；D2 recent review、feasibility/story 与 d2_net_modification_plan_20261006。先前已有 D1–D10，避免重新命名后宣称新颖。
## Subareas and Active Approaches
共享/互补与聚合；观测失真选择性再学习；秩感知路由；子空间系数重建；测量拆分自监督；传感器元信息条件化；非均匀/波段退化。
## Closest Work
MossFuse、SRLF、RASD、CoMM、NUD/PAF、Equivariant Splitting、SpecAware、Optics-Aware。来源 URL 见 ../research_direction_analysis.md 与 ../novelty-review/prior-art-matrix.md。
## Gap Inventory
候选而非已证实空白：同秩不同 RU 灵敏度；观测撤除后功能分工变化；不相容退化下共同测量域是否存在。
## Assumptions and Negative Evidence
泛化交互有强近邻；原 baseline 没有显式分离损失；局部基误差与联合零空间制约恢复。RASD 入口已将输入 HSI 用作内部降尺度目标，但外部配置中的实际数据身份未核实；不能据此认证其无 HR 标签。
## Disconfirmation Targets
简单纹理/秩门控、标准正则化子空间求解、普通留出训练、普通 R 条件网络、正确退化模型普通训练能否解释全部效果？
## Search and Source Limits
36 个记录查询，网页/公开仓库定向检索；未做全数据库召回审计。部分出版社直接打开失败，以可检索原始摘要为边界。代码存在未等于已运行；自审非独立审查。
