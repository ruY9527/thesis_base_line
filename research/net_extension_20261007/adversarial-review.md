# Net 扩展的独立有界反例审查

审查日：2026-10-07。范围：只读当前 `model.py`、`utils.py`、既有 cross_direction_review 研究包，以及补充检索；不运行训练，不实现模型。审查者由原研究技能的协作流程委派，未选择候选；但共享既有上下文，不能称完全盲审或系统性独立查新。

## 结论与定位

现存研究包实际选择 **I13：监督 MSI 细节注入的实际光谱收益**，并未选择 observation-nullspace calibrated network。零空间分解、误差尺度、可信度回退均已被既有研究包的先例否定。

**架构方法定位：2/5，需重大差异化；窄经验研究问题：3/5，只可探索；决定置信度：2/5。** 新增近两年强先例 EnsIR 已用带真值的参考集估计预训练恢复专家的局部组合权重。通用回归 learning-to-defer 也已覆盖按真实损失选择冻结专家。仅增加光谱收益差额、BCE、Huber 和小门控头，没有足够证据成为独立方法贡献。评分只约束贡献表述，不认证性能、代码或实验结论。[EnsIR，NeurIPS 2024 主会](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f12d4e75bb8c62aba3e88d0586af96d3-Abstract-Conference.html)；[Regression with Multi-Expert Deferral，ICML 2024 历史先例](https://proceedings.mlr.press/v235/mao24d.html)。

可保留的最窄问题是：**在相同冻结专家、相同配准、同等预算和相同输入信息下，几何可靠性及观测残差尚不能解释的局部光谱收益，是否可以预测；显式收益监督是否超过普通重建门控、oracle 权重教师和参考集集成？** 这是待验证的经验假设。现在不能写首次、已确认无人提出、因果收益保证或顶刊级创新。

## 1. 必须修正的端点与强度混淆

设 `B=f_H(L)`、`F=f_HM(L,M)`、`d=F-B`，输出为 `B+g d`。原 I13 标签 `Delta=ell(B,X)-ell(F,X)` 比较 g=0 与 g=1 两个端点，**并不衡量所有部分注入**。

反例：B=0、F=2、X=0.75。L1 端点收益为 0.75−1.25=−0.5；但 g=0.375 恰好恢复 X，零误差。平方误差端点收益为 0.5625−1.5625=−1，也仍有零误差混合。若要求所有值在 [0,1]，取 B=0、F=1、X=0.375，同样成立（L1 Delta=−0.25、g*=0.375）。故“端点负收益 ⇒ 应拒绝所有细节”不成立。不能把 negative endpoint gain 的比例写成“所有注入都产生负迁移”的比例。

平方损失下令 `a=<X-B,d>`、`b=||d||²`，则

\[
\ell(B+gd,X)=\|X-B\|^2-2ga+g^2b,\quad
\Delta=2a-b,\quad g^*=\operatorname{clip}(a/b,0,1).
\]

当 b>0、epsilon=0 时，最后一式是该块平方损失的精确凸权重；加入 epsilon 是稳定化近似，不是精确解。端点正收益只意味着未裁剪最优权重超过 1/2，而不是最优权重为 1。d=0 时 g 不可由重建目标辨认，取 0 即可。

若块内采用 L1，则最优 scalar g 是各 `t_i=(X_i-B_i)/d_i` 的以 `|d_i|` 加权中位数，再限制到 [0,1]；d_i=0 项不影响选择。平方 oracle 不能称 L1/SAM 的最优教师。SAM 有不同目标，且零谱向量必须有明确数值约定。

## 2. 胜率、期望收益与混合权重是不同对象

未经成本加权的 BCE 在理想条件下学习 `P(Delta>tau|o)`。它不学习 `E(Delta|o)`。例：90% 的块收益 +0.01，10% 的块收益 −1；胜率 0.9，但平均收益 −0.091。若按胜率批准全量注入，会增加预期误差。这是损失函数选择问题，不是概率校准能自动修好的问题。

对只允许 B/F 两个动作的部署策略，预测条件期望 Delta 再择优，或以 `|Delta|` 为权重训练 endpoint BCE，是应加入的普通成本敏感对照。权重需截断/稳定化并记录，以免极少数异常块主导训练。回归 deferral 的成本敏感理论已经存在，不能把这一补救称新贡献。[历史理论](https://proceedings.mlr.press/v235/mao24d.html)。

对允许连续 g 的平方目标，理想条件最优为

\[
g_{\rm opt}(o)=\operatorname{clip}\!
\left(\frac{\mathbb E[a\mid o]}{\mathbb E[b\mid o]},0,1\right).
\]

如果 o 已完整包括专家预测，b 在给定 o 后是已知的，则简化为 `clip(E[a|o]/b,0,1)`。直接无权 MSE 拟合逐样本 clipped g* 一般不等于最小化期望最终重建误差；裁剪、不同 b 和推理特征信息损失会改变关系。强对照应保留真实重建目标。

三种目标应该明确分开：endpoint gain（原假设的局限对照）、reconstruction policy（普通连续门控）、oracle_weight teacher（仅训练 HR 教师）。收益回归和端点分类可作辅助诊断，policy logit 应与 endpoint logit 分开。只看 `gain aux on/off` 才能检验差额监督的增量；整个双专家模型胜过一个旧 Net 不能证明收益头有效。

## 3. 两阶段训练的标签完整性

1. **专家先训并冻结。** `requires_grad=False`、eval 模式、固定 checkpoint 与哈希；若新增 BN/dropout，no_grad 不等价于 eval。HSI-only 专家必须明确训练，MSI 置零的旧融合 Net 不能自动当成回退专家。
2. **场景划分至少区分四个角色。** expert_train 训专家；gate_train 产生专家未见场景的收益监督；validation 调超参/阈值/early stopping；test 最终一次评价。数据少可交叉拟合训练内场景，但仍需独立验证和最终测试。不得把同一场景的重叠 patches 分配到不同角色。
3. **训练输入只来自可部署信息。** 真值只参与损失/教师标签；不得把 GT residual、真实误差图、oracle g* 或由 GT 推断的遮挡 mask 拼入推理 o。合成扰动的 mask 若拟部署不可得，只可监督/分析，不可当推理特征。
4. **标签冻结且可复现。** 记录教师 checkpoint、scene、crop、退化/扰动种子、波段尺度、block size、loss 定义、归一化尺度；收益尺度只能来自训练角色。
5. **detach 不是联合训练的充分保护。** 它阻断本 step 标签梯度，但下一 step 的专家可以因其自身损失而改变 B/F/Delta；回退专家还可能随 gate 联合目标退化。第一版固定专家。若日后联合，只从独立固定教师产生标签，另测专家本身质量，并明确固定教师收益与当前候选动作不再完全相同。

交叉拟合产生的 teacher experts 与最终全训练集 experts 可能分布不同；这不是泄漏，但 gate 可能失配。首版使用预定 expert_train/gate_train 的固定专家即可清晰验证机制，后续再研究全量重训。

## 4. 真实行动与标签必须一致

当前 `utils.py` 的 tile 推理是每块先 clamp 到 [0,1]，再重叠平均。若训练标签用 raw B/F，而最终评估使用 clamp/projection/blending，Delta 与实际行动的误差可能不同。必须定义标签针对哪个预测阶段；主评价在相同完整流水线上进行。clamp 是非线性，`clamp(B+gd)` 不等于 `clamp(B)+g(clamp(F)-clamp(B))`。

空间变化的 g 若先低分辨率估计再插值，需报告实际的门控空间尺度、tile 边界与 block sensitivity。仅因整体 PSNR 上升，不能认定所有被接受区域都有正收益。

已知 SRF 投影若按像素作用，可与同像素 scalar gate 的线性部分交换；涉及空间 blur/downsample 的投影通常与空间变化的 gate 不交换。物理校正若加在门控之后，oracle 与训练标签也应针对相同顺序。

`Gaussian_downsample` 的 `signal.convolve2d(...,boundary='symm',mode='same')`、8×8 偶数核与 `[0::s,0::s]` 相位需要逐值复现。PyTorch reflect 不等于 SciPy symmetric，偶数 padding 相位也可能错一像素。operator 验证是实验正确性基础，不是贡献。

## 5. 不能包装成贡献的常规组件

|组件/主张|判断|最低处理|
|---|---|---|
|双分支、残差 CNN、通道注意力|常规结构；当前 baseline 已有|作为固定骨架|
|HSI-only 与融合双专家、冻结训练|普通分工/后训练集成；参考 SR 也有冻结回退专家|比同预算单专家/双专家|
|sigmoid、BCE、Huber、stopgrad、小头|标准工具|消融，不称主要贡献|
|从 GT 学权重、oracle 凸混合|EnsIR 与通用集成；oracle LS 是标准优化|加入 reference-set ensemble|
|回投影挑难点、可信度回退|SRLF 与未配准 HSR 先例|作为强对照|
|空间/光谱专家与动态互补权重|WCGE-Net 已同任务实现|禁止首次专家融合|
|光谱零空间投影、UQ 校准|既有 null-space/UQ 先例；不能提供唯一恢复|仅作可解释诊断|
|场景留出、训练标签分折|避免泄漏的评价设计|必要协议，不独立方法贡献|

相关论文的来源和证据级别见同目录 prior-art-addendum.md；不能从仅摘要的论文推断其细节机制不存在。

## 6. 零空间边界保持不变

给定 M=RX、L=XD^T，P=R^dagger R 只是光谱行空间投影，Q=I−P 只是光谱零空间投影。取非零 v∈ker(R)、q∈ker(D)，DeltaX=vq^T，则 R DeltaX=0 且 DeltaX D^T=0。因此同观测的两种真值可有不同最佳 g/收益标签。**投影不会令 QX 可辨识；收益头也无法对全部歧义实例正确决策。** 可讨论给定场景分布中的平均风险，不能承诺逐实例可靠性或零负迁移。

此外，光谱 R 零空间不等于联合观测 A(X)=(RX,XD^T) 的零空间；仅投影 Q 保证 MSI 一致，并不保证 LR-HSI 一致。若 R/PSF 错配，硬投影还可能将错误假设强加给预测。

## 7. 必须通过的判别性实验与失败标准

优先做动机与 oracle 上限，随后才讨论复杂结构。验证场景中固定 B/F，分组观察结构匹配、观测残差与 Delta；另算连续 oracle 的实际收益，避免端点陷阱。高匹配/低残差区域的负端点收益不单独证明某个迁移机制；需要控制专家容量、退化难度与场景成分。

最低等信息对照：B、F、固定最优全局 g、简单平均、结构门控、回投影门控、重建 policy、endpoint BCE、`|Delta|` 加权 endpoint BCE、oracle_weight 教师 policy、收益辅助 policy、EnsIR 式参考集集成。前三个 learnable policy 使用同一头和数据预算。需要全部保留不代表首轮全部训练复杂论文；先用小头与可复现简化对照排除常规解释。

评价包括最终 PSNR/SAM/RMSE、逐场景差异、全/部分注入的 realized gain、负迁移 magnitude、门控覆盖率、收益排序、校准（若声称概率）及运行成本。净风险差应按 scene 统计；不能把高度相关的 patches 当独立重复。低覆盖的拒绝策略不能靠拒绝全部动作获得单独胜利。

发生以下任一结果，停止将当前实现当主方法创新：连续 oracle 改善很小；EnsIR/普通重建 policy/加权 endpoint 已同效；收益辅助的增量跨 seed/scene 不稳定；收益只由新增参数或一个合成遮挡分布解释；干净场景持续退化；改一次专家 checkpoint gate 即失效。若动机不成立，应保留消融发现并转向已证实的失败问题，不继续更名包装。

没有任何训练性能证据在本审查中产生。后续若通过这些测试，仍需补齐 CoPGNet、WCGE-Net 全文与高召回专家收益先例，才能重新审查方法新颖性。
