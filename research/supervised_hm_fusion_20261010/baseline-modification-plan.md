# baseline 修改计划（未执行）

范围：只读审查与修改计划。现有 model.py 在任务开始时已有本地修改，本次保留该状态。下列新文件名与接口都是建议，尚未创建或接入；research 文件夹里的取证辅助脚本不是训练实现。

## 1. 当前模型能承担什么

当前 Net 的默认配置为 spn=31、ch=3、dim=64、num_blocks=8。LR-HSI 两层编码后双线性上采样，MSI 两层编码，两路特征拼接、卷积融合，经过 8 个残差块与潜通道注意力，再预测 31 波段残差，叠加双线性上采样的 HSI。

适合：固定 31/3 波段、已配准、×8 设置下的控制实验，以及有无 MSI、算子条件输入、简单 gate 的增益归因。

不够：仅凭此模型的性能超过原版本，证明新方法优于近期顶会强模型；仅凭 64 维 ChannelAttention 权重，解释物理波段可信度；在没有响应输入的情况下宣称跨传感器融合。

## 2. P0：协议与评价修复，先于任何新模块

| 文件/位置 | 现状 | 计划 | 验收 |
|---|---|---|---|
| train_cave_M.py:66、163–200 | cave_test 在训练中反复评价，并按 PSNR/RMSE 选 best | train/val/test manifest；只用 val 选模；测试独立入口 | 日志与 checkpoint 元数据能证明测试不参与模型选择 |
| train_dataloader.py:190 起 | os.listdir 顺序取前 num 个场景 | scene 名称显式排序/冻结名单；记录排除项 | 重启或换文件系统后划分一致 |
| calculate_metrics.py:26 起 | PSNR 先逐位置取 log 再平均 | 新增标准指标，原指标留作 legacy 复核；论文主报告用统一定义 | 固定误差的数值正确，逐 band/cube PSNR 区分明确 |
| calculate_metrics.py:46 起 | RMSE 使用 0–255 尺度并截断 | 主指标固定范围与是否 clip；同时保留 raw 误差诊断 | 两种尺度转换可解释、记录越界率 |
| calculate_metrics.py:58 起 | SAM 为 HWC NumPy，零范数依赖 epsilon | 标明数据排列和角度单位；明确 zero-spectrum 策略 | NaN、黑背景与极低强度谱处理有文档 |
| utils.py:22 | symmetric padding、phase 0、偶数核卷积 | 记录 PSF、相位、边界、采样方式；同论文复现另存配置 | 整图与带 halo patch 的退化可对应 |
| train_dataloader.py:201 | HR 场景最大值归一化 | 合成实验明确声明；真实测量用推理可得标定 | 测试推理不隐含未知 HR 强度信息 |
| train_cave_M.py | 训练与设备路径写死 | 配置化数据、设备、seed、输出；保留原可复核配置 | 一个实验配置能重建训练与评估流程 |

**PSNR 的重要区别。** 当前输入 BCHW 经两次 dim=0 均值后仍保留 H×W，之后逐位置取 log。假设两个位置的误差分别为 0.1 和 0.01、峰值为 1，当前口径对应平均 20/40 dB，即 30 dB；整 cube MSE 口径则约 22.97 dB。误差分布越不均匀，差异越可能影响方法排名。

建议主报告至少同时保存：

\[
\mathrm{PSNR}_{cube}=10\log_{10}(L^2/\mathrm{MSE}_{cube}),
\]

以及逐波段 MSE 得到的 PSNR 后取波段平均。二者不等价；不混用。合成数据每个 scene 的指标先计算，再按 scene 汇总。

如果 CAVE 已有公认的 20/12 场景划分，保留 12 个最终测试场景，从 20 个训练场景中固定划出验证集（具体名单确定后写 manifest）。另做作者协议复现时，保持作者名单，不假称完全同一设置。不能从随机 patch 中划分验证集后仍宣称 scene 泛化。

P0 的数值检查有必要，因为指标与科学结论直接相关：identity、固定偏差、空间非均匀误差、已知夹角、零范数谱、退化相位和 patch halo。这里只列验收，没有新增测试或运行训练。

## 3. P1：独立诊断分支，保留原模型

建议未来新增实验入口 train_fusion_control.py、evaluate_fusion_control.py 和 experiments/configs/，原 model.py 暂不替换，以保留参考版本。

### 3.1 数据协议

建议新增 datasets/fusion_observations.py 和 operators/：

- 每个 sample 返回观测 Y、M，训练时另返回 X；保存 scene ID。
- 记录 phi_true 与 phi_provided，二者用途严格分离。
- R 的归一化、波长网格、PSF、MSI blur、噪声、采样相位都在配置中有版本。
- 真算子生成观测；提供给模型的算子按独立误差机制产生。
- 对测试样本只提供推理可得信息；phi_true 只进入离线评价/受控实验分析。
- augmentation 要同时变换观测、目标和空间算子；空间位置相关算子不能在翻转后沿用原坐标。
- 先 scene split、整图退化后切 patch，或使用与整图等价的 halo。不得把跨 split 的卷积邻域混合。

### 3.2 网络与对照

建议未来新增 models/hsi_only.py、models/fusion_control.py；尽量用原残差 CNN 作为公共 backbone。

1. HSI-only：从 Y 预测 X_H；用于 MSI 相对收益诊断。
2. 双输入原 CNN：f(Y,M)。
3. 容量匹配双输入 CNN：排除新增参数和训练预算解释。
4. 普通 gate：X_H + g(Y,M)Δ。
5. 重投影 gate：增加 R、D_H 下的观测残差。
6. 物理支持基线：固定 ridge/模式支持收缩，不假称创新。
7. 拟议控制模型：额外接收估计算子与可获得误差描述。

计划接口示例：

\[
\mathrm{forward}(Y,M,\widehat\phi,\mathrm{error\_info})\to
(\widehat X,\Delta,g,\mathrm{diagnostics}).
\]

这里 diagnostics 用于分析，不改变普通推理所需输入。物理谱模式建立在 31 波段/响应空间，潜特征仅作预测表示。若采用 LR-HSI 的 SVD 基底，需处理符号/基底变化，并在高阶残差上做消融；不称其为真实端元。

先固定目标波段数和配准条件。扩散、Mamba 或大型 Transformer 不是 P1 的前置条件。

### 3.3 收益标签

可新增 analysis/benefit_maps.py：按固定区域/谱模式计算 GT 下的候选修正收益，仅用于训练标签或离线评价。

教师模型按 scene 的 out-of-fold 生成标签，保存教师版本与 split。不能直接让教师拟合全数据再在同一训练 scene 上生成“可靠性真值”。测试集的 oracle gate 可以作为理论上限报告，明确标记为不可部署，不用于选择实际 gate。

### 3.4 损失选择

先以相同 L1 或 MSE 作为所有神经基线的重建目标。如果主诊断收益由平方误差定义，要标明它和 L1/SAM 的目标差异。

数据一致性作为辅助项，必须用提供给模型的算子，并记录算子错误可能引入偏差。不能在训练或推理中偷偷使用测试 phi_true。算子误差集合监督、gain/risk 辅助项逐项增加；每增加一项保留等训练预算的对照，不能一次叠加多个 loss 再归因。

## 4. P2：强方法复现与场景扩展

先选近期代表而非堆数量：

| 工作 | 作用 | 复现前提 |
|---|---|---|
| SRLF | 对齐、观测重投影选点 | 作者 main/supp、已知 R/PSF、统一指标与完整延迟 |
| HSC-Sampling/MRC | 跨波段结构补偿 | 补到全文；已知作者 GT L1 实现 |
| BHSR-Net | 模糊 MSI、双退化 | 补全文；确认估计/已知算子接口与不同 variant |
| EMR-Diff | 监督扩散代表 | 记录 kappa、5-step 配置、随机采样次数、训练裁剪 |
| UAFL / SSC-HSR / IR&ArF | 未配准路线 | 仅在增加几何误差后比较，配准协议一致 |

作者设置复现与统一设置重新训练分开存表。代码公开不代表可立即复现；硬编码路径、数据字段、依赖与 checkpoint 要先检查。禁止为省事只引用对方论文数值来证明自己的改进。

对扩散模型：同输入随机种子多次采样，报告平均误差和采样方差；不在测试真值上挑最佳采样。精度、训练/推理成本和额外预训练数据一起记录。

## 5. P3：若进入跨传感器连续谱路线

此时再计划新的响应曲线编码、共享通道 token、连续谱解码与目标响应积分，替代固定 31→3 通道假设。完整接口需输入 HSI/MSI/目标 HSI 的响应及波长网格，并验证积分权重、光谱范围与单位。

先对比 band center、center+FWHM、完整 SRF。若完整响应不胜过摘要条件，停止架构扩展。不要在没有真实响应曲线和设备留出时声称 sensor-agnostic。

原 CNN 可以作为固定网格参考，不要求它硬适配所有传感器；更复杂架构应作为新实验模型保留可追踪版本。

## 6. 产出与验收

未来每次训练需保存：代码版本、配置、数据 scene manifest、真/提供算子分布、随机种子、训练步数、checkpoint 选择依据、逐 scene 指标、运行时间与显存。

最终论证包应包括：干净精度、元信息错误曲线、真实/合成分层、最差场景、逐波段误差、模式收益、oracle 上限、容量匹配消融、训练数据成本。

本次未产生模型性能结果。五个主要源码的任务前后 SHA-256 对照保存在 baseline-integrity.json；任务开始时已有的 model.py 修改状态保持原样。

