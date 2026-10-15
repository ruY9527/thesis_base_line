# Selected Idea
## Recommendation
I05/A 先做实际 LR 基诊断，适合当前固定 R 的指标目标；I06/B 对应自监督训练贡献，先设普通留出训练强对照。I07/C 条件储备，不作为第三项并行实现。
## Idea Summary
A：同秩区域可能具有不同 MSI 响应灵敏度，研究错误细谱分配与连续条件修正。
B：撤除真实测量改变信息支持，利用留出测量训练条件分工，正常完整输入指标为主。
## Differentiation
A 收窄原 D5，不宣称 SVD/岭公式新颖；B 收窄原 D1/D4，不宣称 mask/测量拆分新颖。潜在增量需在任务错误、功能接口和公平训练对照共同成立时判断。
## Assumptions and Disconfirmation
A 需要可部署 LR 基；HR 基上界仅供诊断。B 需要移除全部目标输入路径，目标不能泄露。标准子空间/普通门控、普通留出训练若解释全部效果，放弃方法主张。
## Minimum Validation
A：自然灵敏度—误差诊断、同秩不同灵敏度小任务、普通扩容/纹理/秩/固定 R 对照。
B：固定三种通道撤除、同目标预算、条件/无条件与固定/条件角色比较、完整输入与功能屏蔽两类指标。
## Risks and Blocking Questions
NQ1/NQ2 近邻未排尽；没有新网络性能证据；联合零空间不可凭空恢复；辅助头可被绕过。当前自监督入口还需另建，现 train_cave_M.py 的 HR L1 不能用于主自监督实验。
## Handoff Notes
完整推导、故事、baseline 插入点与停止规则见 ../research_direction_analysis.md。先补针对候选的系统检索和小实验，不要求一次实现两方向。此前 D2 改造方案保留为工程对照。
