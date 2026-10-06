# 从 62 篇论文到可落地的自监督高光谱融合候选贡献

撰写与补充检索日期：2026-10-05。基于 [原研究报告](D:/coding/github_self/thesis_base_line/research/self_supervised_hsi_fusion_review.md) 的“逐篇方法、迁移位置与代码依据”，保留 P01–P62 的顺序和原方法、迁移建议、代码证据，再逐篇增加问题、动机、机制、实现与证伪条件。

**这里给出的是有实现路径、可被实验推翻的候选贡献，不是已经验证有效或已经确认全球新颖的成果。** “最小实现”表示现有 PyTorch 工程可以向它演进，不表示原仓库已经实现了这个新想法。原论文结论与本文推断分别放置；相近候选需要合并，不能把 62 条思路当成 62 个互相独立的论文贡献。

## 先把研究问题写具体

最有价值的思考方式是：**哪一种失配，在什么条件下，让当前训练信号给出错误方向；我用什么机制纠正这个方向；怎样排除仅仅增加模型容量的解释。** 例如，“增加多频注意力”太宽；“在 PSF 错配与相关噪声同时出现时，区分不可观测高频和可修正的观测残差，避免注意力追逐噪声”才开始成为研究问题。

下面所有“可能失效”都指待检验的适配场景，不能据此断言原论文已有实验失败。每篇提出一条主候选，部分篇目另给延伸候选；最后只收敛到三个优先方向。优先级评价的是适合本项目先验证的程度，不是论文水平或预期录用概率。

## 与当前工程对齐的约定

### 1. 网络、数据和算子

[model.py](D:/coding/github_self/thesis_base_line/model.py:57) 的主干仍可保留：31 波段 LR-HSI、3 通道 HR-MSI，经两分支编码、特征插值、`fuse`、8 个 `ResBlock`、`tail`，最后加上 `hsi_up`。`ChannelAttention` 的 64 个隐藏通道不能直接当作按波长排序的 31 个波段。

当前 [train_cave_M.py](D:/coding/github_self/thesis_base_line/train_cave_M.py:143) 用 HR-HSI 的 L1 标签损失，仍属于监督训练。候选自监督方法必须另建训练入口；模拟实验可以用干净 HSI 生成观测，但训练、早停与调参不能读取目标分辨率的干净 HSI。它只用于独立评估。

记 \(Y_h\in\mathbb R^{B\times31\times h\times w}\)、\(Y_m\in\mathbb R^{B\times3\times H\times W}\)、\(H=sh,W=sw\)，输出 \(\hat X\in\mathbb R^{B\times31\times H\times W}\)。已配准简化模型为

\[
A(X)=(D_{P,s,\phi}X,RX),\quad Y_h=D_{P,s,\phi}X+n_h,\quad Y_m=RX+n_m.
\]

`R` 为 3×31 矩阵；可用固定权重的 1×1 卷积实现。空间算子先模糊再按指定相位 \(\phi\) 采样，可用分组卷积实现，但要匹配现有生成过程。增加配准后必须明确参考坐标系及 `warp → blur → sample` 顺序，不能把几种顺序视为等价。

[utils.py](D:/coding/github_self/thesis_base_line/utils.py:22) 实际用 `scipy.signal.convolve2d(..., boundary='symm', mode='same')` 后 `[0::s,0::s]`。偶数 8×8 核、same 对齐及对称扩展都可能造成相位差；PyTorch `reflect` 不能未经核对就当作 SciPy `symm`。先用脉冲、常数、斜坡核对算子输出，并核对离散伴随。否则所谓“创新收益”可能只是在补偿实现错误。

`create_F()` 的行归一化适用于当前模拟响应。真实传感器的增益不能被无条件抹掉：若把 SRF 行归一化，需要显式保留每通道增益，且固定尺度规范，避免增益与输出幅值共同漂移。

### 2. 可以复用，但不能直接照抄的内部训练数据

[train_dataloader.py](D:/coding/github_self/thesis_base_line/train_dataloader.py:64) 的 `RealDATAProcess2` 与 `RealDATAProcess` 已有“把传入 HSI/MSI 再降尺度”的雏形。它们返回 `(hsi_target, msi_input, hsi_input)`，与 CAVE 类的 `(hsi_target, hsi_input, msi_input)` 顺序不同。新入口宜用命名字段，不能继续靠 `a1/a2/a3` 猜含义。

若传入的是实际 LR-HSI，那么这里的 `hsi_target` 是观测自身，不是目标 HR-HSI 真值。不过，这个标签依然带噪，并不自动满足噪声独立性。64×64 HR 裁剪在 s=8 时仅有 8×8 LR-HSI；再降 8 倍得到 1×1，不能支撑空间内部学习。应先在完整 LR 图上构造内部对、再裁剪，或从 q=2/4 起步，同时研究 q 与部署倍率 s 不一致的问题。对 CAVE 原生 512×512 场景，s=8 的完整 LR 图为 64×64，是尺寸示例，不是所有数据集的固定尺寸。

### 3. 共用损失与留出规则

基础观测损失记为

\[
L_{obs}=\ell_h(D\hat X,Y_h)+\lambda_m\ell_m(R\hat X,Y_m).
\]

噪声已知且适合 Gaussian 模型时，可用白化二次项；明显离群点可用有明确假设的鲁棒项。`L1`、白化、SURE、盲点损失不互相等价，每条候选要说明适用条件。波长平滑、低秩、TV 只能提供先验，不能证明共同零空间中的真值唯一。

本文的“留出观测”有两种含义：**校准留出块**检验响应参数对未参与校准的区域是否适用；**训练目标留出**还必须从网络全部输入路径移除目标值，才能检验复制泄漏。邻域填充、`hsi_up`、全图归一化、教师网络与缓存特征都在排查范围内。空间相关噪声需要保护区，跨波段相关噪声需要联合遮挡；保护范围要依据测量或显式模拟的相关结构选择。

另一独立传感器提供同一位置的干净信号线索，是融合所需要的有效条件，并不天然是泄漏。危险的是共享目标噪声、直接使用目标值，或从未经遮挡的目标构造输入。

### 4. 验证与贡献边界

最小证伪验证统一要求：与相同训练设置、相近参数量/前向次数的对照比较；每次先改变一种失配因素，再测试组合失配；有真值的独立合成测试报告 PSNR、SAM、逐波段/稀有材料误差，真实无真值数据报告可审计的观测检验及限制。留出误差变小不能单独证明 HR-HSI 更准确。

“不保留条件”是指撤销该假设或降低它的贡献等级，不是看到一次负结果就丢弃整个方向。失败要先排除算子错配、数据泄漏与不公平预算；也不能通过反复看测试真值挑选最好超参数来挽救假设。

## 逐篇候选速查索引

每篇先保留原报告依据，再读“进一步思考”。下表按原报告顺序导航。

| ID | 论文 | 进一步的具体问题 |
|---|---|---|
| [P01](#p01) | MossFuse | 响应改变时，共享特征到底共享什么？ |
| [P02](#p02) | ARGS-Diff | 残差大，究竟是细节缺失还是算子不对？ |
| [P03](#p03) | CLoRF | 训练像素中心与真实传感器积分不一致 |
| [P04](#p04) | SRLF-Net | 高误差区域不一定值得反复学习 |
| [P05](#p05) | CS2DIPs | 全局低秩能省算力，也会消失稀有材料 |
| [P06](#p06) | UTAL | 测试时适配可能把错误传感器拟合得更好 |
| [P07](#p07) | ZSL | 内部再降尺度并不自动复现真实退化 |
| [P08](#p08) | CUCaNet | 注意力不应该替退化模块背锅 |
| [P09](#p09) | CDPS | 两条后验引导可能重复强化同一方向 |
| [P10](#p10) | SHOTUN | 相似 MSI patch 可能属于不同光谱材料 |
| [P11](#p11) | UAFL | 错位校正与低秩基修正会互相补偿 |
| [P12](#p12) | IR&ArF | 先 warp 再模糊，与先模糊再 warp 不相同 |
| [P13](#p13) | LRTN | 一个全图秩不适合所有区域 |
| [P14](#p14) | OTIAS | 插值核适应内容，却可能不适应成像覆盖 |
| [P15](#p15) | SSC-HSR | 颜色差不能直接当作配准误差 |
| [P16](#p16) | FeINFN | 高频不全是细节，弱响应频率也不全该删除 |
| [P17](#p17) | MHF-Net | 网络修正未必沿着物理误差下降方向 |
| [P18](#p18) | MoG-DCN | 错误前向模型下，似然下降也可能伤害真值 |
| [P19](#p19) | UAL | 逐图适配把一个场景的偏差写进整个网络 |
| [P20](#p20) | uSDN | 丰度和为一，可能替光照或增益误差背锅 |
| [P21](#p21) | DBIN | 全图一个 PSF 会把局部退化写成局部内容 |
| [P22](#p22) | Hipandas | 噪声从图像跳连直达输出 |
| [P23](#p23) | LRSDN | 正交基稳定了数值，未必稳定了物理光谱 |
| [P24](#p24) | DDS2M | 各波段噪声不同，扩散误差尺度不能统一 |
| [P25](#p25) | GEWDiff | 小波边缘增强会把扫描条带当作几何结构 |
| [P26](#p26) | UMNet | 无真值时，什么叫“可靠的误差图”？ |
| [P27](#p27) | DM-ZS | RGB 先验可能补出“好看但不存在”的材料细节 |
| [P28](#p28) | HIR-Diff | 信息量大的波段，也可能只是噪声大的波段 |
| [P29](#p29) | UniPAN | 分布统一可能破坏光谱比值和物理响应 |
| [P30](#p30) | SpectralGPT | 预训练的遮挡重建任务与融合测量缺失不同 |
| [P31](#p31) | Equivariant Splitting | 按图像像素遮挡，不等于按测量信息拆分 |
| [P32](#p32) | UNSURE | 未知噪声估计把响应偏差也当成噪声 |
| [P33](#p33) | MOI | 有多种退化条件，不代表有新的真实观测信息 |
| [P34](#p34) | REI | 等变循环会不会反复复制同一份噪声？ |
| [P35](#p35) | EI | 旋转和翻转可能并没有补足缺失信息 |
| [P36](#p36) | KernelGAN | 模糊核中心与采样相位容易混淆 |
| [P37](#p37) | ZSSR | 内部训练尺度越多，不一定越有帮助 |
| [P38](#p38) | DDNM | 两种带噪观测之间可能不存在共同硬一致解 |
| [P39](#p39) | Neighbor2Neighbor | 邻像素噪声独立，但干净信号并不相同 |
| [P40](#p40) | Noise2Same | 完整输入与遮挡输入的一致性会不会奖励复制？ |
| [P41](#p41) | Noise2Void | 遮一个波段的像素，其他波段仍可能暴露噪声 |
| [P42](#p42) | DIP | 没有 HR 真值，逐图优化何时停？ |
| [P43](#p43) | DPS | 大的似然梯度有时来自坏观测 |
| [P44](#p44) | DDRM | 两路观测的弱方向不等于分别的弱方向 |
| [P45](#p45) | CARL | 传感器不变特征，可能丢掉重建需要的绝对幅值 |
| [P46](#p46) | CESST | 从 MSI 生成一种光谱，可能掩盖多解歧义 |
| [P47](#p47) | FeatUp | MSI 边缘不一定是 HSI 的材料边缘 |
| [P48](#p48) | MST | 物理 mask 引导不能直接换成融合随机 mask |
| [P49](#p49) | NAFNet | 无标签适配的初始大修正可能破坏可靠基线 |
| [P50](#p50) | Restormer | 两路特征尺度不同，attention 相关性可能偏向噪声 |
| [P51](#p51) | HyperSIGMA | 稀疏采样节省计算，却可能漏掉低面积材料 |
| [P52](#p52) | MambaIRv2 | 语义邻居不能保证光谱邻居 |
| [P53](#p53) | MambaIR | 长程扫描会不会传播方向相关噪声？ |
| [P54](#p54) | HAT | 窗口位置与采样相位变化，会造成重建接缝 |
| [P55](#p55) | Uformer | 多尺度特征融合不等于多尺度物理一致 |
| [P56](#p56) | LIIF | 波段序号与真实光谱响应不是同一个坐标 |
| [P57](#p57) | CARAFE | 逐位置重组核会不会跨材料传播光谱？ |
| [P58](#p58) | RCAN | 深残差不仅学细节，也能改变已可靠的低频谱 |
| [P59](#p59) | FcaNet | 空间 DCT 摘要会同时看到条带与细节 |
| [P60](#p60) | ECA-Net | 隐藏通道的相邻，不代表物理波段相邻 |
| [P61](#p61) | CBAM | 同一个空间位置，两传感器可能给出冲突证据 |
| [P62](#p62) | SENet | 全图平均会稀释小面积的关键材料 |

## 逐篇展开：原方法依据与进一步候选贡献

<a id="p01"></a>

### P01 | MossFuse | IJCV 2026

#### 原报告方法、迁移位置与代码依据

**Unsupervised Hyperspectral Image Super-Resolution via Self-Supervised Modality Decoupling**

Authors: Songcheng Du; Yang Zou; Zixu Wang; Xingyuan Li; Ying Li; Changjing Shang; Qiang Shen

[Paper](https://doi.org/10.1007/s11263-026-02757-8) | [Official code](https://github.com/dusongcheng/MossFuse)

**原作训练设定**: 自监督/无监督 HSI-MSI 融合；含退化估计

**方法核心**: 将模态共享与互补信息分解，使用子空间聚类损失引导分解后再融合。

**对当前 baseline 的迁移建议（研究假设）**: 针对简单 concat+fuse 的冗余，将两路特征拆为共享/私有分支，再设计跨模态一致性与互补保留损失。

**限制与适配边界**: 不能只复制模块而保留 HR-HSI 标签损失称为自监督；环境 README 有重复/缺失版本项。

**已观察到的实现证据**: 已核对公开树：MossFuseNet.py、train.py、test.py、degradation_net.py；README 提供 python test.py 与合成过程。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: arxiv v1 标题 Modality Decoupling is All You Need；v4 明确 Journal reference IJCV134,152(2026)，DOI一致；Springer直读超时。

Verification sources: [Source 1](https://arxiv.org/abs/2412.04802) / [Source 2](https://github.com/dusongcheng/MossFuse)

#### 进一步思考：响应改变时，共享特征到底共享什么？

**具体失配与动机。** 同一材料经不同 SRF 成像，MSI 数值会变化；如果“共享”被训练成两路隐藏特征逐点相等，可能把 HSI 的细窄吸收峰当成跨模态差异去除。反过来，响应差异也可能被误存成“私有材料特征”。这不同于一般的模态冗余问题。

**候选机制。** 先用已知或校准后的 R，把 HSI 分解为 MSI 可见的光谱方向与其余方向。对满行秩 R，\(Q_R=R^\top(RR^\top+\epsilon I)^{-1}R\) 是正则化的近似值域映射；无正则且可逆时才是正交投影。在共同 LR 网格上约束共享分支重建 `R Y_h` 与 `D Y_m` 的相容结构，而不要求所有 64 维特征相同。私有分支承担 `(I-Q_R)Y_h` 的光谱信息，并保留局部空间先验。R 改变时随之改变“可见/不可见”的分工。

**最小实现与训练。** 在 `fuse` 前增加两个小 1×1 分支，R 的 31×31 映射只作用于物理波段，再送入编码器；共享辅助头输出 3 通道，私有辅助头在 LR 端重建余量。总目标为 `L_obs + 共同网格兼容项 + LR私有重建项`。先固定 R/P，不同时增加盲估计或大模型；ε 和辅助权重通过独立观测验证选择。

**最小证伪与不保留条件。** 比较普通 concat、原式共享/私有划分、R 条件划分；用训练未见 SRF 和 MSI 近似同色而 HSI 不同的材料对。若仅训练 SRF 上改善、未见响应或吸收峰误差不改善，不能主张响应泛化。分支全零或私有分支复制全输入也属于失败。

**贡献边界与定位。** 最近邻 P01、P08、P45，物理值域分解亦是已有思想。潜在贡献是“SRF 改变造成共享定义漂移”的诊断与响应条件化训练，而不是首次模态解耦。**优先：中高；额外依赖：无特殊 CUDA。** 延伸候选：当 R 只有名义值时，对一组受约束响应做分支稳定性检验，报告对响应不确定性敏感的波段，而非承诺恢复不可见光谱。

<a id="p02"></a>

### P02 | ARGS-Diff | CVPR 2025

#### 原报告方法、迁移位置与代码依据

**Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model**

Authors: Jian Zhu; He Wang; Yang Xu; Zebin Wu; Zhihui Wei

[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Self-Learning_Hyperspectral_and_Multispectral_Image_Fusion_via_Adaptive_Residual_Guided_CVPR_2025_paper.pdf) | [Official code](https://github.com/Zhu1116/ARGS-Diff)

**原作训练设定**: 只用当前 LR-HSI/MSI 学习；无外部数据

**方法核心**: 分别从 HSI/MSI 学习光谱与空间扩散先验，在低维子空间中重建，采样每步使用自适应观测残差引导。

**对当前 baseline 的迁移建议（研究假设）**: 先迁移可计算的观测残差/数据一致性校正到 CNN 输出或迭代块；完整扩散方案需大改。

**限制与适配边界**: 网络修正 residual 与观测退化 residual 不同；扩散训练与采样开销需实测。

**已观察到的实现证据**: 推理 sample_subspace.py、diffusion/、示例ckpt；训练仓库 ARGS-Diff-train 有 spatial_train.py/spectral_train.py/ guided_diffusion。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 训练仓库 https://github.com/Zhu1116/ARGS-Diff-train；README HR-HSI optional。

Verification sources: [Source 1](https://arxiv.org/abs/2505.11800) / [Source 2](https://github.com/Zhu1116/ARGS-Diff) / [Source 3](https://github.com/Zhu1116/ARGS-Diff-train)

#### 进一步思考：残差大，究竟是细节缺失还是算子不对？

**具体失配与动机。** 当前输出修正量 `tail(f)`、观测误差 `Y-A(X_hat)`、扩散过程的随机误差是三种不同的残差。在错误 PSF 或条带噪声下，大观测误差可能指向重复条带；直接增强残差引导可能把它写进重建。

**候选机制。** 用可解释诊断区分三类证据：白化后且跨传感器支持的局部结构残差、随采样相位/核宽变化显著减小的校准残差、单传感器特有的噪声残差。诊断不是保证正确的分类器：先对少量合法 P 候选计算共同 LR 域兼容误差，固定最优候选；再比较两路重投影的边缘一致性，为第二次残差修正分配不同步长。对可疑噪声只做受约束去噪，不强化高频。

**最小实现与训练。** 先不迁移扩散：`Net → 观测残差与诊断 → 2层修正头 → X_hat2`。门控使用冻结/停止梯度的校准诊断，`L_obs` 仍完整计算，不能乘一个可自由变零的置信度。两次前向的卷积对照必须保留。稳定后才将诊断映射到 ARGS 的引导权重。

**最小证伪与不保留条件。** 四类场景：已知干净、噪声、核偏差、两者叠加。检查诊断对已知模拟类别的混淆，以及 SAM/条带残留。若普通白化加同容量修正头已取得同等结果，复杂残差分型不保留为独立贡献。

**贡献边界与定位。** P02/P04 已有残差引导与再学习；候选贡献必须是混合失配下的残差归因与训练响应。**优先：高；先 CNN 后扩散。**

<a id="p03"></a>

### P03 | CLoRF | IJCV 2025

#### 原报告方法、迁移位置与代码依据

**Hyperspectral and Multispectral Image Fusion with Arbitrary Resolution Through Self-Supervised Representations**

Authors: Ting Wang; Zipei Yan; Jizhou Li; Xile Zhao; Chao Wang; Michael Ng

[Paper](https://link.springer.com/article/10.1007/s11263-025-02540-1) | [Official code](https://github.com/wangting1907/CLoRF-Fusion)

**原作训练设定**: 自监督连续表示 HSI-MSI 融合

**方法核心**: 连续低秩分解 CLoRF，将空间和光谱连续神经表示与低秩结构结合，按坐标查询任意分辨率。

**对当前 baseline 的迁移建议（研究假设）**: 将 tail 固定网格输出改为低秩系数+光谱基/坐标解码器，探索连续空间尺度与波长建模。

**限制与适配边界**: 属于输出表示和优化框架改造，非普通注意力直接替换；示例需要模拟观测与已知响应。

**已观察到的实现证据**: 公开 inr/、train_tv_spatial.py、train_tv_spatial.sh、utils.py；README 有模拟 LR-HSI/MSI 与执行命令。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: Publisher opened，IJCV133(11):7515-7535；DOI10.1007/s11263-025-02540-1。

Verification sources: [Source 1](https://link.springer.com/article/10.1007/s11263-025-02540-1) / [Source 2](https://github.com/wangting1907/CLoRF-Fusion)

#### 进一步思考：训练像素中心与真实传感器积分不一致

**具体失配与动机。** 连续函数在像素中心的预测不是 PSF 加权的区域测量。倍率、核宽和采样相位变化后，训练端的“一个像素”与测试端的“一个像素”具有不同覆盖范围；网络可能把这种差别错误学成纹理。

**候选机制。** 连续低秩表示 `X(u,v)=E C(u,v)`，用有限个采样点近似每个观测的加权积分：`Y_h[i] ≈ sum_j w_ij E C(u_ij,v_ij)`。查询位置由像素覆盖范围和相位决定，权重来自已知 PSF/像素响应。HR-MSI 的观测也经自己的像素响应积分和 R，而不是对空间编码器输出直接采点。未知相位先用共同 LR 域校准，避免坐标偏移与内容任意互补。

**最小实现与训练。** 第一步保持固定网格 Net，只核对离散 D；第二步仅把 `tail` 改成 r 维系数 MLP，E 从训练可见的 LR-HSI 估计并冻结；第三步增加分块坐标查询和积分。先用 2×2/4×4 求积作为精度与成本对照，积分点增多必须验证误差收敛，不能宣称四点精确模拟任意 PSF。

**最小证伪与不保留条件。** 构造相同场景、不同相位/核宽/倍率的观测，比较中心查询、只加 cell、积分且不校准、积分加相位条件。若改善只来自额外查询次数，或相位校准错误使纹理更糟，不能主张机制成立。

**贡献边界与定位。** P03/P14/P56 以及已有 IPE 都接近；“像素积分 INR”本身不能作为首次贡献。可研究的是内部自监督的训练—测试响应失配与相位条件化求积。**优先：中高；坐标解码比纯损失改动复杂。** 延伸候选：在带噪观测上用固定总查询预算，给大覆盖像素更多积分点，研究求积误差是否会被误判为观测噪声。

<a id="p04"></a>

### P04 | SRLF-Net | CVPR 2025

#### 原报告方法、迁移位置与代码依据

**A Selective Re-learning Mechanism for Hyperspectral Fusion Imaging**

Authors: Yuanye Liu; Jinyang Liu; Renwei Dian; Shutao Li

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html) | [Official code](https://github.com/YuanyeLiu/SRLF-Net)

**原作训练设定**: 监督 HSI/MSI 融合；成像模型引导的选择性细化

**方法核心**: 先初步融合，再根据空间-光谱结构与观测模型识别失真特征点，只对这些点再学习。

**对当前 baseline 的迁移建议（研究假设）**: 在 fuse/body 后增加轻量误差反馈细化；比较全图更新与选择性局部更新。

**限制与适配边界**: 观测引导不等于无真值训练；已有选择性修正方法必须作为新 idea 的近邻。参数/FLOPs 统计要覆盖选择和扫描操作。

**已观察到的实现证据**: deep_select_mamba.py、mambair.py、train_cave.py、train_nc.py、test.py、train_dataloader.py；官方提供实际训练实现。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liu_A_Selective_Re-learning_CVPR_2025_supplemental.pdf) / [Source 2](https://github.com/YuanyeLiu/SRLF-Net)

#### 进一步思考：高误差区域不一定值得反复学习

**具体失配与动机。** 自监督再学习最容易选中饱和像素、错位边缘、坏波段与条带。它们误差很大，却可能不是网络通过细节修正能解决的区域。监督场景的选择逻辑不能直接当作含噪自监督场景的可靠性判据。

**候选机制。** 对 `e_h/e_m` 做噪声尺度归一化，并比较冻结算子下的两路结构支持。选区分成“可靠但未拟合”和“观测异常”；前者送入再学习头，后者仍接受完整鲁棒观测损失，但不提升学习权重。每个块设最低覆盖率，防止选区只集中在高对比边缘；连续多轮选中且验证无进展的区域退出强化队列。

**最小实现与训练。** 在 `tail` 后加一个轻量修正头；选区由上一轮冻结输出计算，先用明确阈值及分位数，不训练自由置信度网络。`L_obs` 不删除未选区域，选区只控制辅助修正预算。实现缓存时记录算子版本、遮挡状态与图像编号。

**最小证伪与不保留条件。** 比较随机区域、误差最大区域、白化误差区域、可靠性加覆盖选择，保持选择比例与算力相同。若优点只是避开损失困难样本，却让坏波段重建持续恶化，撤销“更可靠训练”的表述。

**贡献边界与定位。** 最近邻就是 P04，并关联 P02/P26；需要证明选择策略在噪声/失配下为何不同。**优先：高；适合作为方向一的训练组件。**

<a id="p05"></a>

### P05 | CS2DIPs | TIP 2024

#### 原报告方法、迁移位置与代码依据

**CS2DIPs: Unsupervised HSI Super-Resolution Using Coupled Spatial and Spectral DIPs**

Authors: Yuan Fang; Yipeng Liu; Chong-Yung Chi; Zhen Long; Ce Zhu

[Paper](https://doi.org/10.1109/TIP.2024.3390582) | [Official code](https://github.com/Ambitionfy/CS2DIPs)

**原作训练设定**: 无需外部训练数据，观测对内优化

**方法核心**: 非负矩阵-向量张量分解结合耦合空间/光谱 DIP，生成丰度与光谱矩阵并物理重组。

**对当前 baseline 的迁移建议（研究假设）**: 用双观测退化重投影损失替代 HR-HSI L1；探索把输出分解为光谱基与空间系数。

**限制与适配边界**: 逐图优化有迭代成本；已知退化/非负混合假设；必须避免以 GT 指标选停止时刻。

**已观察到的实现证据**: demo.py 已打开：总损失明确比较 Y 与 PZ、X 与 H_z(Z)，执行反传；modules/含实现。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: TIP33:3090-3101；作者大学官网公开全文。

Verification sources: [Source 1](https://www.ee.nthu.edu.tw/cychi/publications/journal_articles2011/CS2DIPs_unsupervised_HSI_super-resolution_using_coupled_spatial_and_spectral_DIPs.pdf) / [Source 2](https://github.com/Ambitionfy/CS2DIPs/blob/main/demo.py)

#### 进一步思考：全局低秩能省算力，也会消失稀有材料

**具体失配与动机。** 在少量像素才出现的弱光谱成分，对全局能量占比很小。固定小秩可能把它删除；盲目增秩又可能把噪声纳入。问题不是简单选择更大 r，而是区分被截断信号与噪声。

**候选机制。** `X_hat=E_r C + U_perp V`，E_r 固定于观测稳健子空间，U_perp 只保留少量与 E_r 正交的候选方向，V 在空间上受组稀疏约束。候选方向只能来自训练可见 LR-HSI 的未解释部分；是否启用由独立观测留出收益决定，而不是重建训练损失下降。不能凭 MSI 的 3 个通道生成任意未知端元。

**最小实现与训练。** 将 `tail` 分成 r 个主系数和 k 个补充系数，用冻结 31×(r+k) 的 1×1 解码；先固定 r/k 做小网格观测验证。无需直接迁入两套 DIP。低秩基估计不能用测试 HR-HSI；严格目标留出时也不能从被留出的数值估计基。

**最小证伪与不保留条件。** 控制稀有材料面积和 SNR，比较固定秩、扩大秩、等参数自由残差、受约束补充方向，报告材料光谱而不仅全图 PSNR。若新增方向只改善训练拟合并放大低 SNR 波段，不保留。

**贡献边界与定位。** P05/P13/P23/P28，低秩加稀疏亦是成熟路线。贡献需聚焦无 HR 标签时的补充方向选择及罕见成分保留。**优先：中高。**

<a id="p06"></a>

### P06 | UTAL | TPAMI 2024

#### 原报告方法、迁移位置与代码依据

**Unsupervised Test-Time Adaptation Learning for Effective Hyperspectral Image Super-Resolution With Unknown Degeneration**

Authors: Lei Zhang; Jiangtao Nie; Wei Wei; Yanning Zhang

[Paper](https://doi.org/10.1109/TPAMI.2024.3361894) | [Official code](https://github.com/JiangtaoNie/UAL)

**原作训练设定**: 监督合成数据预训练 + 无监督测试时适配；非全程零标签

**方法核心**: 共享互指导融合先验预训练，测试时自指导适配与未知退化联合估计；元训练与交替优化提高适配效率。

**对当前 baseline 的迁移建议（研究假设）**: 保留现有监督 baseline 作为初始化，加入轻量 adapter 与观测一致性测试时更新，比较跨 PSF/SRF 的泛化。

**限制与适配边界**: 不能宣称无监督从头；每样本适配成本和测试时可用观测需报告。

**已观察到的实现证据**: 公开 Adaptor.py、Train_FusionModel.py、Spa_downs.py、meta-al-UTAL/；README 分清 UAL 与 UTAL/meta 训练流程。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: TPAMI46(7):5008-5025，2024；代码仓库同时含 CVPR2020 UAL。

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/38315602/) / [Source 2](https://github.com/JiangtaoNie/UAL)

#### 进一步思考：测试时适配可能把错误传感器拟合得更好

**具体失配与动机。** 若 X、P、R 同时自由更新，网络可以通过修改输出和响应相互补偿。观测误差下降并不证明响应更正确；一次适配也可能破坏预训练网络在其他区域的光谱结构。

**候选机制。** 校准与重建采用分开的更新证据：先固定名义 R，仅从训练观测块的 `R Y_h ≈ D_P Y_m` 校准低维 P/相位；用不参与校准的块检验，再冻结 P。随后只更新小 adapter，用目标移除后的预测留出监控停止。若要更新 R，先固定 P 并加物理波长支撑/尺度规范，不能同时全自由。

**最小实现与训练。** 在 `fuse` 后加零初始化 1×1 adapter；划分校准、适配和观测验证块，记录保护区和每阶段可见的数据。固定 R/P 的无外部预训练版本先作本项目对照，再评估 UTAL 的预训练设置，明确两者外部信息不同。

**最小证伪与不保留条件。** 对照不适配、全参数适配、全联合盲适配、校准冻结加小适配；同时检查核误差和 HR 误差。若分阶段方案只是更强正则、与等容量冻结对照无差别，则贡献缩小为工程稳定化。

**贡献边界与定位。** P06/P19/P21、HySure、U2K 接近；独立校准本身不是新事物。待证贡献是避免测试适配中退化—内容补偿的可审计训练机制。**优先：高。**

<a id="p07"></a>

### P07 | ZSL | TPAMI 2023

#### 原报告方法、迁移位置与代码依据

**Zero-Shot Hyperspectral Sharpening**

Authors: Renwei Dian; Anjing Guo; Shutao Li

[Paper](https://doi.org/10.1109/TPAMI.2023.3279050) | [Official code](https://github.com/renweidian/ZSL)

**原作训练设定**: 当前 HSI/MSI 观测构造低尺度训练对；零样本融合

**方法核心**: 先估计空间/光谱响应，再对当前观测降尺度训练，结合降维和成像模型损失。

**对当前 baseline 的迁移建议（研究假设）**: 优先迁移到 train_cave_M.py 的数据构造和损失；可保留 Net，训练目标改为原始 LR-HSI，输入为进一步降尺度的 HSI/MSI。

**限制与适配边界**: 降尺度学习依赖跨尺度规律；原始 LR-HSI 噪声会成为伪目标；官方退化估计需要 MATLAB。

**已观察到的实现证据**: ZSL_Training.py、ZSL_Testing.py、model1.py、demo_kernel_estimation.m；README 给出运行顺序。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/37235456/) / [Source 2](https://github.com/renweidian/ZSL)

#### 进一步思考：内部再降尺度并不自动复现真实退化

**具体失配与动机。** 当前内部数据雏形把同一个 Gaussian_downsample 再施加到已有 LR 图。不同网格上的同样 sigma 不代表相同物理模糊；二次模糊与采样相位组合，也不等于真实一级成像。网络在内部任务学到的逆过程因此可能不适用于部署倍率。

**候选机制。** 建立每一级的有效响应与坐标：物理覆盖宽度、网格间距、核中心及相位随尺度转换。对 Gaussian 连续近似可利用方差可加关系推导额外模糊；对离散核与采样必须显式检查复合算子，不能直接套 `sigma_total^2=sigma_1^2+sigma_2^2` 就宣布精确。内部训练从 q=2/4 到部署尺度，用尺度/响应编码调制小 adapter。

**最小实现与训练。** 新建观测内 Dataset，在完整 Y_h/Y_m 上生成 `(D_q Y_h, D_to_h Y_m) → Y_h`，先确定两个输入与目标的物理网格。内部标签 Y_h 带噪，使用噪声修正或目标遮挡版本，最后结合原尺度 `L_obs` 适配。记录每个内部样本的真实有效倍率、核与相位。

**最小证伪与不保留条件。** 比较直接复用固定核、响应匹配、只加倍率条件、响应和相位条件；跨 PSF 与部署倍率测试。若只在固定合成核上增益，不能主张解决跨尺度响应失配。

**贡献边界与定位。** P07/P37/P03 与 U2K 是必要最近邻。核心是内部自监督响应失配，不是首次 ZSL 数据构造。**优先：高。** 延伸候选：按 Y_h 的有效空间尺寸及噪声估计选择内部倍率，避免以少量 1×1/2×2 输入支撑高倍率任务；这可成为训练采样策略，但单独贡献较弱。

<a id="p08"></a>

### P08 | CUCaNet | ECCV 2020

#### 原报告方法、迁移位置与代码依据

**Cross-Attention in Coupled Unmixing Nets for Unsupervised Hyperspectral Super-Resolution**

Authors: Jing Yao; Danfeng Hong; Jocelyn Chanussot; Deyu Meng; Xiaoxiang Zhu; Zongben Xu

[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740205.pdf) | [Official code](https://github.com/danfenghong/ECCV2020_CUCaNet)

**原作训练设定**: 无监督盲融合/耦合解混

**方法核心**: 两路卷积自编码器学习基与系数，交叉注意力传递空间光谱信息，用一致性学习 SRF/PSF。

**对当前 baseline 的迁移建议（研究假设）**: 替换 concat-only 融合，令 HSI 与 MSI 互相生成指导信息；加入可约束的 SRF/PSF 模块。

**限制与适配边界**: 交叉注意力与当前全局通道门控不同；光谱解混约束需一并理解。

**已观察到的实现证据**: 公开 model/ 与 Main_CAVE.py；README 提供 Windows Python3.7 PyTorch demo、逐图运行和输出路径。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: ECCV pp208-224；会议官网全文与代码仓库互相对应。

Verification sources: [Source 1](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740205.pdf) / [Source 2](https://github.com/danfenghong/ECCV2020_CUCaNet)

#### 进一步思考：注意力不应该替退化模块背锅

**具体失配与动机。** 跨注意力能够改变两路特征，错误 SRF/PSF 同样会产生跨模态差异。让注意力与响应模块一起自由解释差异，可能得到“观测相符、物理参数不符”的解。

**候选机制。** 响应估计先依赖共同 LR 网格兼容关系，主干不参与该阶段；响应冻结后，跨注意力只处理校准之后剩下的内容差异。SRF 若未知，以名义响应为中心学习小幅、非负且受波长支撑的扰动；将响应置信区间而非单一数值作为注意力条件，让不确定响应降低强制共享约束。

**最小实现与训练。** 第一版固定 R，仅估核宽/相位等少量参数；第二版增加受约束 R。`fuse` 改为门控双分支，但保持两输入分辨率匹配方式。校准块验证通过后才启动网络训练，联合训练作为必要对照而非默认选择。

**最小证伪与不保留条件。** 固定正确、固定错误、自由联合、校准冻结四组，记录 R/P 精度与重建光谱。若冻结方案响应更准确但 HR 重建并未改善，应报告机制局限，不能把参数误差独自当作融合贡献。

**贡献边界与定位。** P08/P06/P21 和传统盲校准。关键差别应是响应不确定性与特征共享之间的训练耦合设计。**优先：中高。**

<a id="p09"></a>

### P09 | CDPS | TIP 2026

#### 原报告方法、迁移位置与代码依据

**Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion**

Authors: Yang Xu; Jian Zhu; Danfeng Hong; Zhihui Wei; Zebin Wu

[Paper](https://doi.org/10.1109/TIP.2025.3647207) | [Official code](https://github.com/Zhu1116/CDPS-train)

**原作训练设定**: 无监督；仅输入测试对学习扩散先验

**方法核心**: 分别学习光谱与空间扩散先验，并在耦合后验采样中同时约束两类观测保真。

**对当前 baseline 的迁移建议（研究假设）**: 启发双分支先验和双退化观测一致性；CNN 可先做两路不同强度的数据保真反馈。

**限制与适配边界**: 与 ARGS-Diff 相近但不同工作；本文代码链接核对到训练/分支采样仓库，完整联合采样入口还需进一步复现审计。

**已观察到的实现证据**: 公开 guided_diffusion、spatial_train.py、spectral_train.py、spatial_sample.py、spectral_sample.py；README 训练步骤，HRHSI optional。

**代码核验程度**: Training and component sampling observed; full coupled sampling entry not confirmed

**补充核验**: 仓库称 TIP2025 accepted；正式 TIP2026 vol35 pp69-84。不可直接写 TIP2025。

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/41477791/) / [Source 2](https://github.com/Zhu1116/CDPS-train)

#### 进一步思考：两条后验引导可能重复强化同一方向

**具体失配与动机。** MSI 约束某些光谱组合，HSI 约束低空间频率。二者既有互补方向也有重复方向；固定比例相加的后验梯度可能在重复方向过度修正，对弱约束方向又用力不够。

**候选机制。** 在低秩系数域分析白化联合算子 `A_E(C)=(D(EC),REC)` 的局部曲率，给重叠可观测方向降低重复步长，给弱但非零可观测方向设置有上限的预条件步长。共同零空间只能由先验约束，不能把它当作小奇异值直接无限放大。

**最小实现与训练。** 先对 Net 输出做 2–3 步加权最小二乘修正，用共轭梯度或对角近似，避免整个 HR 算子的 SVD；核/R 固定，噪声权重固定。若有效，再把同一预条件规则放到耦合扩散采样中。公开代码入口有核验限制，原方法复现与新增机制验证分开。

**最小证伪与不保留条件。** 双传感器 SNR 不平衡与 SRF 行高度相关的场景，对照固定权重、单独白化、曲率预条件，匹配迭代次数。若速度或质量收益无法超越简单白化，删去复杂曲率设计。

**贡献边界与定位。** P09/P38/P43/P44，预条件逆问题本身成熟。候选贡献是耦合融合中的重复约束诊断与受限引导预算。**优先：中；完整扩散成本高。**

<a id="p10"></a>

### P10 | SHOTUN | TIP 2026

#### 原报告方法、迁移位置与代码依据

**Self-Expressive High-Order Tensor Unrolling Network for Unsupervised Hyperspectral and Multispectral Image Fusion**

Authors: He Wang; Yang Xu; Zhihui Wei; Zebin Wu

[Paper](https://doi.org/10.1109/TIP.2026.3695389) | [Official code](https://github.com/Shawn-H-Wang/SHOTUN)

**原作训练设定**: 无监督张量展开；未知退化估计含预训练策略

**方法核心**: 重叠图像块间自表达关系形成高阶张量表示，保持空间结构；将交替子问题优化展开为模块。

**对当前 baseline 的迁移建议（研究假设）**: 探索重叠 patch 非局部关系或低秩约束，或把 ResBlock 变为观测校正+学习先验的迭代块。

**限制与适配边界**: 并非直接替换 CA 即复现；高阶张量/非局部 patch 内存和复杂度需评估。

**已观察到的实现证据**: 公开 LowRankNetwork.py、blind.py、两个 unfolded_net_sparse_representation...py；README有执行 demo。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 正式 TIP35:5714-5729；原README旧 BibTeX编号10705122与新 DOI3695389不同，不复制旧编号。

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/42202190/) / [Source 2](https://github.com/Shawn-H-Wang/SHOTUN)

#### 进一步思考：相似 MSI patch 可能属于不同光谱材料

**具体失配与动机。** 光谱异色同谱投影，即不同 HSI 光谱经 R 后近似相同，会让 MSI 驱动的 patch 相似性产生错误匹配。非局部自表达再传播这些匹配，可能把一种材料的光谱扩散到另一种材料。

**候选机制。** 在共同 LR 网格，用“MSI 结构相似 + LR-HSI 光谱相容”双证据构建候选邻居。只保留 top-k，加入噪声归一化与相互匹配检验；当 LR 支撑太弱或材料混合严重时，保留本地路径、降低非局部贡献，不把缺少证据视作光谱相同。

**最小实现与训练。** 不先迁入全部张量自表达：在两个 `ResBlock` 间加低分辨率候选检索与 HR 特征稀疏聚合，检索描述子可冻结，最终仍用 `L_obs`。按块流式计算相似度，记录 k 与显存峰值。

**最小证伪与不保留条件。** 在有已知光谱类别的模拟测试构造 MSI 近色材料，比较 MSI-only、HSI-only、双证据检索与无非局部，检查错误跨材料连边比例。若连边更准确却不改善光谱，或成本抵消收益，应降低优先级。

**贡献边界与定位。** P10/P52/P47；光谱引导匹配已很常见，需聚焦投影歧义下的拒绝匹配与噪声处理。**优先：中高。**

<a id="p11"></a>

### P11 | UAFL | CVPR 2026

#### 原报告方法、迁移位置与代码依据

**Enhancing Unregistered Hyperspectral Image Super-Resolution via Unmixing-based Abundance Fusion Learning**

Authors: Yingkai Zhang; Tao Zhang; Jing Nie; Ying Fu

[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html) | [Official code](https://github.com/yingkai-zhang/UAFL)

**原作训练设定**: 监督、未配准 RGB 引导 HSI 超分

**方法核心**: 用 SVD 保留 LR-HSI 光谱基，学习丰度修正；粗细可变形聚合和丰度跨注意力处理未对齐参考。

**对当前 baseline 的迁移建议（研究假设）**: 将 hsi_up+tail(f) 改成固定光谱基乘丰度残差；研究在保持光谱基时注入 MSI 细节。

**限制与适配边界**: SVD基不能自动解释成非负物理端元；低秩截断可能漏掉稀有光谱。原作使用HR目标，需换自监督目标。mmcv_full/mmflow/cupy/torch_dwconv 有编译依赖。

**已观察到的实现证据**: source/、run.py、torchlight/、requirements.txt；2026-07-04已发布实现与示例权重；论文承诺将发布的旧描述不能代表当前状态。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.pdf) / [Source 2](https://github.com/yingkai-zhang/UAFL)

#### 进一步思考：错位校正与低秩基修正会互相补偿

**具体失配与动机。** 固定 LR-HSI 基可能缺少小面积材料；错误配准又让材料边界的系数混合。自由改基和自由 warp 会给出两种解释，训练损失难判断到底是哪一种。

**候选机制。** 采用坐标系固定、分阶段更新：先冻结光谱基，只用共同 LR 网格相容区域估计低自由度平移/仿射；再冻结 warp，根据 LR 光谱留出残差决定是否启用少量正交补充基。warp 不可靠区域仅降低额外跨模态共享项，不删除主观测约束。用有效覆盖 mask 处理边界和缺失视野。

**最小实现与训练。** `grid_sample` 实现小位移，`tail` 改系数域输出；第一版不搬 deformable CUDA。基只用可见 LR 估计，记录参考坐标系。先让原尺度观测损失贯通再加候选基，避免同时改全部结构。

**最小证伪与不保留条件。** 单独错位、单独弱材料、二者叠加；比较固定基/可改基及 warp 固定/可学的组合，评估位移误差与材料光谱。若改善来自更大基而非分阶段识别机制，不能主张解决两者混淆。

**贡献边界与定位。** P11/P12/P15/P05；“配准+低秩”已有，候选是配准—基修正互补偿的约束与诊断。**优先：中；涉及实际坐标细节。** 延伸候选：以多组近似同样观测误差的 warp/基解生成不确定区域图，但该图是模型歧义，不是已校准真值误差。

<a id="p12"></a>

### P12 | IR&ArF | TIP 2025

#### 原报告方法、迁移位置与代码依据

**IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images**

Authors: Jiahui Qu; Xiaoyang Wu; Wenqian Dong; Jizhou Cui; Yunsong Li

[Paper](https://doi.org/10.1109/TIP.2025.3551531) | [Official code](https://github.com/Jiahuiqu/IR-ArF)

**原作训练设定**: 监督未配准/任意尺度融合

**方法核心**: 构建空间光谱一致的任意尺度观察模型并展开优化，联合处理注册误差与动态分辨率需求。

**对当前 baseline 的迁移建议（研究假设）**: 当前假设完美配准，可在融合前加入对齐与跨尺度查询；自监督版本需改成观测域注册/重投影损失。

**限制与适配边界**: 不能直接照搬训练称为自监督；官方 train.py L258使用HR-HSI真值L1，L263-265也用GT。

**已观察到的实现证据**: model.py/train.py/test.py/dataloader.py 已公开；raw train.py 已核对GT损失。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: TIP34:1934-1949；README标题Towards，正式title Toward。

Verification sources: [Source 1](https://github.com/Jiahuiqu/IR-ArF) / [Source 2](https://raw.githubusercontent.com/Jiahuiqu/IR-ArF/main/train.py)

#### 进一步思考：先 warp 再模糊，与先模糊再 warp 不相同

**具体失配与动机。** 非刚性形变下，PSF 在参考坐标的覆盖形状随位置变化；把低分辨率图简单 warp，并用固定模糊解释它，会使边界误差被网络当成光谱差异。大倍率会放大这种顺序误差。

**候选机制。** 显式使用各传感器坐标系：若 X 定义在 MSI 网格，HSI 预测为 `D_P(W_h X)`。先实现正确顺序，对照 `W_lr(D_P X)`；必要时在小形变近似下，以 warp 的局部 Jacobian 调整像素覆盖，不直接用同一核宽处理所有区域。

**最小实现与训练。** 新增外部 `Physics` 包装器，不先改 Net。先用已知仿射 warp 检验顺序，再允许小范围可学 warp；采用平滑、位移幅度限制及有效视野 mask，固定 P/R，避免三个未知模块同时自由补偿。

**最小证伪与不保留条件。** 对不同剪切、局部缩放和倍率，比较两种顺序与正确前向模型；报告边界、遮挡和内部区域分别的误差。若差别只来自边界填充错误，修复后应撤销“形变响应建模”的贡献表述。

**贡献边界与定位。** P12/P11/P15 与几何成像已有研究；新的问题应限定到自监督融合的算子顺序失配及可落地近似。**优先：中高；可以先做很小的算子实验。**

<a id="p13"></a>

### P13 | LRTN | IJCV 2025

#### 原报告方法、迁移位置与代码依据

**Low-Rank Transformer for High-Resolution Hyperspectral Computational Imaging**

Authors: Yuanye Liu; Renwei Dian; Shutao Li

[Paper](https://link.springer.com/article/10.1007/s11263-024-02203-7) | [Official code](https://github.com/YuanyeLiu/Low-Rank-Transformer-For-High-Resolution-Hyperspectral-Computational-Imaging)

**原作训练设定**: 监督 HSI/MSI 融合

**方法核心**: 跨模态互补特征提取、先验引导注意力、空间低秩交叉注意力降低全局建模成本。

**对当前 baseline 的迁移建议（研究假设）**: 研究用低秩跨注意力替换 concat+fuse 或部分 ResBlock，保持双分支和图像残差。

**限制与适配边界**: 作者代码要求测试图像尺寸与训练一致；需要移除固定空间尺寸假设。2024在线、正式卷期2025，按2025计篇。

**已观察到的实现证据**: LRTN.py、LRTN_gf5.py、main.py、train_cave.py、train_dataloader.py；已查看作者仓库。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://link.springer.com/article/10.1007/s11263-024-02203-7) / [Source 2](https://github.com/YuanyeLiu/Low-Rank-Transformer-For-High-Resolution-Hyperspectral-Computational-Imaging)

#### 进一步思考：一个全图秩不适合所有区域

**具体失配与动机。** 平坦区域低秩足够，材料交界或弱材料区域可能需要额外方向。空间上自由选择秩也可能把噪声区域选为高秩，造成局部谱波动和块边界。

**候选机制。** 使用共享最大基 E，预测系数门控而不是每个 patch 重新估基；秩门控来自 LR 的白化未解释残差和独立观测验证，要求空间平滑、相邻窗口一致。仅允许少量附加方向，形成“主低秩 + 有证据的局部补充”，不按训练误差无限扩秩。

**最小实现与训练。** `tail` 输出 r_max 系数及几个低分辨率门控，固定基后共享解码；辅助稀疏惩罚控制补充分量，主 `L_obs` 完整保留。门控第一次先用固定诊断规则，确认不是噪声后再学习小预测器。

**最小证伪与不保留条件。** 材料数量不均、面积极不均、不同噪声强度，对照全局小秩、全局大秩、自由空间门控、受约束门控。若选秩随噪声增长而非随真实复杂度变化，不保留该策略。

**贡献边界与定位。** P13/P05/P23，局部低秩/自适应秩不是新概念；贡献需要无标签、含噪情况下的秩选择证据。**优先：中。**

<a id="p14"></a>

### P14 | OTIAS | AAAI 2025

#### 原报告方法、迁移位置与代码依据

**OTIAS: OcTree Implicit Adaptive Sampling for Multispectral and Hyperspectral Image Fusion**

Authors: Shangqi Deng; Jun Ma; Liang-Jian Deng; Ping Wei

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32275) | [Official code](https://github.com/shangqideng/OTIAS)

**原作训练设定**: 监督 HSI-MSI 融合

**方法核心**: 八叉树在空间与通道维建模，按坐标生成与光谱信息有关的自适应插值核。

**对当前 baseline 的迁移建议（研究假设）**: 替换固定双线性 feature interpolation，让上采样同时考虑波段/特征与空间坐标。

**限制与适配边界**: 完整算法并非简单三线性插值；官方用 GT L1，迁入 self-supervised需替换训练目标。

**已观察到的实现证据**: 公开 model/、train.py、eval.py、tools/；raw train.py L98-101确认out与ref监督损失；README提供权重与数据链接。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: AAAI39(3):2708-2716；Publisher DOI10.1609/aaai.v39i3.32275，2025-04-11。

Verification sources: [Source 1](https://ojs.aaai.org/index.php/AAAI/article/view/32275) / [Source 2](https://raw.githubusercontent.com/shangqideng/OTIAS/main/train.py)

#### 进一步思考：插值核适应内容，却可能不适应成像覆盖

**具体失配与动机。** 学习插值权重可以追踪边缘，但不同倍率像素覆盖与抗混叠需求不同。只输入坐标或倍率可能仍把高频 aliasing 当作可迁移结构，尤其内部倍率与目标倍率不一致时。

**候选机制。** 插值核同时接收相对坐标、cell、有效 PSF 与相位；权重归一化并设置覆盖范围限制。用两个合法观测尺度验证核是否符合响应变化，而不是只检查一张最终图。响应不确定时输出多个受约束核假设，以观测验证决定其组合。

**最小实现与训练。** 只替换 `f_hsi` 的双线性上采样，保持原 `hsi_up` 与 `tail`；先固定单核，再加条件核。训练为内部对加原尺度 `L_obs`；倍率、核宽与相位需一起记录，不能分别随机却忽略几何网格。

**最小证伪与不保留条件。** 未见倍率、非整数缩放及相位变化；比较内容条件、坐标条件、响应条件，与同预算 CARAFE/双线性对照。若响应条件只相当于记住训练倍率，不主张跨响应泛化。

**贡献边界与定位。** P14/P03/P56/P57；候选是内容与响应条件冲突时的训练机制，直接学习插值已被原作覆盖。**优先：中。**

<a id="p15"></a>

### P15 | SSC-HSR | IJCV 2025

#### 原报告方法、迁移位置与代码依据

**Unaligned RGB Guided Hyperspectral Image Super-Resolution with Spatial-Spectral Concordance**

Authors: Yingkai Zhang; Zeqiang Lai; Tao Zhang; Ying Fu; Chenghu Zhou

[Paper](https://link.springer.com/article/10.1007/s11263-025-02466-8) | [Official code](https://github.com/yingkai-zhang/SSC-HSR)

**原作训练设定**: 监督、未对齐 RGB 参考引导 HSI 超分

**方法核心**: 两阶段对齐、迭代可变形特征聚合与参考引导光谱注意力。

**对当前 baseline 的迁移建议（研究假设）**: 在两路特征拼接前做可靠性加权的参考聚合；真实双传感器错位实验可作为扩展方向。

**限制与适配边界**: 需要几何模型和额外对齐训练；原任务参考 RGB 与目标可存在视差/遮挡。自监督损失也需显式处理坐标变换。

**已观察到的实现证据**: source/、torchlight/、run.py、create_env.sh、requirements.txt；README 提供对齐和融合训练、权重。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://link.springer.com/article/10.1007/s11263-025-02466-8) / [Source 2](https://github.com/yingkai-zhang/SSC-HSR)

#### 进一步思考：颜色差不能直接当作配准误差

**具体失配与动机。** HR-MSI 与 HSI 的光谱响应不同，跨模态亮度差可能来自材料而非位移。用原始通道强度驱动光流，会把某些光谱差异解释成形变，进而对齐错误材料。

**候选机制。** 先通过 R 与 D 将两路转换到可比较的共同低尺度光谱域，再估计小位移；HR 细化只使用结构梯度、前后向相容性与有效覆盖证据。对饱和、遮挡、细窄谱差区域保留“不确定”，不用大 warp 强行降低光度差。

**最小实现与训练。** 初版用平移/仿射加 `grid_sample`，不先引入 RAFT。参数估计从训练观测块获得，独立块检验；`L_obs` 采用明确 warp 顺序，不确定 mask 只控制额外对应约束且设最低有效比例。

**最小证伪与不保留条件。** 分离真实位移与纯光谱响应变化两组，检查“无真实位移时是否预测假位移”。若 R/P 校准误差主导位移估计，应先校准，不宣称对未知响应鲁棒。

**贡献边界与定位。** P15/P11/P12/P08；候选贡献是自监督配准中光谱差与几何差的区分。**优先：中高。**

<a id="p16"></a>

### P16 | FeINFN | NeurIPS 2024

#### 原报告方法、迁移位置与代码依据

**Fourier-enhanced Implicit Neural Fusion Network for Multispectral and Hyperspectral Image Fusion**

Authors: Yu-Jie Liang; Zihan Cao; Shangqi Deng; Hong-Xia Dou; Liang-Jian Deng

[Paper](https://neurips.cc/virtual/2024/poster/96117) | [Official code](https://github.com/294coder/Efficient-MIF)

**原作训练设定**: 监督 HSI–MSI 融合；隐式表示

**方法核心**: 空间/频率隐式融合和 Gabor 激活解码器，改善高频表示。

**对当前 baseline 的迁移建议（研究假设）**: 研究用频域条件分支或坐标解码器替换固定上采样及 tail。

**限制与适配边界**: 不是无 HR 标签训练的证明；README 训练入口名称需与当前仓库核对。

**已观察到的实现证据**: model/FeINFN.py；readmes/FeINFN.md；训练入口与配置

**代码核验程度**: Actual public source observed; not executed or reproduced

#### 进一步思考：高频不全是细节，弱响应频率也不全该删除

**具体失配与动机。** 高频增强既可能补充边缘，也可能放大噪声与 aliasing。HSI 低通不可见的高频仍可能由 MSI 提供部分线索，因此“低通不支持就全部删掉”同样错误。

**候选机制。** 在物理低秩域，把频率证据分成 HSI 支持、MSI 支持和两路均弱三类。已知 PSF 的频率响应、R E 的条件数与噪声水平共同确定有界增益；对两路均弱方向只给受限先验残差，输出其敏感性说明，不把增益当作光谱真实性保证。

**最小实现与训练。** 先在 `tail` 前加 DCT/小波分支与固定响应条件门控，不直接搬整个 Gabor INR。频率门控调节修正量，主物理损失不被门控；响应偏差作为专门测试维度。

**最小证伪与不保留条件。** 纹理、条带、aliasing 分别出现并组合，对照直接高频增强、噪声归一化、响应条件增益，检查逐波段高频误差。若清晰度提高但 SAM/弱材料恶化，不能称光谱保真改善。

**贡献边界与定位。** P16/P59/P25/P44，频率融合已有；潜在贡献是双观测对高频支持不对称时的受限增益。**优先：中高；属于可与方向一结合的组件。**

<a id="p17"></a>

### P17 | MHF-Net | TPAMI 2022

#### 原报告方法、迁移位置与代码依据

**MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion**

Authors: Qi Xie; Minghao Zhou; Qian Zhao; Zongben Xu; Deyu Meng

[Paper](https://doi.org/10.1109/TPAMI.2020.3015691) | [Official code](https://github.com/XieQi2015/MHF-net)

**原作训练设定**: 监督学习；consistent/blind两种退化设置

**方法核心**: 低秩先验与观测退化建模，展开近端梯度算法构造可解释网络。

**对当前 baseline 的迁移建议（研究假设）**: 在 ResBlock 前后显式加入空间/光谱残差校正，降低纯黑盒重构；低秩输出减少通道开销。

**限制与适配边界**: 盲模型仍不是无监督模型；官方为 TensorFlow1.x。

**已观察到的实现证据**: CMHF-net/BMHF-net公开 MHFnet.py、CAVEmain.py、CAVE_dataReader.py；README有训练/测试开关。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: DOI早期在线2020；正式 TPAMI44(3):1457-1473(2022)。CVPR2019是前作不重复计算。

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/32780695/) / [Source 2](https://github.com/XieQi2015/MHF-net/tree/master/CMHF-net)

#### 进一步思考：网络修正未必沿着物理误差下降方向

**具体失配与动机。** 当前 `hsi_up + tail(f)` 没有观测一致性步骤，某次特征修正可能在空间上更锐利，却破坏已经符合 MSI 的光谱组合。增加展开层也不能自动解决离散算子和伴随不一致。

**候选机制。** 把 CNN 先验修正与物理梯度拆开：`Z=Net(Y_h,Y_m)`，再以 `X_next=Z-η A* W(AZ-y)` 修正。η 取有界正值，W 为固定噪声精度。若加入可学修正，使用固定观测验证或候选步长回溯控制明显的观测恶化；这只约束观测方向，不证明整体 HR 误差下降。

**最小实现与训练。** 新增 2–3 个共享权重步骤，明确 `D*` 的边界累积与采样零插入；自动微分可先获得梯度，避免手写错误伴随。先固定 η，再允许少量响应条件步长。训练仍只用观测或合法内部目标，展开模块自身不提供标签。

**最小证伪与不保留条件。** 比较无修正、固定梯度、可学步长及同深度卷积，检查各阶段双观测误差与 HR 误差。若硬压观测误差只把噪声写回输出，改为噪声感知软约束，不保留无噪声一致性叙事。

**贡献边界与定位。** P17/P18/P38 已覆盖展开与数据一致性；候选必须围绕自监督下的观测方向保护及离散失配，而非首次展开。**优先：中高；先作为强对照。**

<a id="p18"></a>

### P18 | MoG-DCN | TIP 2021

#### 原报告方法、迁移位置与代码依据

**Model-Guided Deep Hyperspectral Image Super-Resolution**

Authors: Weisheng Dong; Chen Zhou; Fangfang Wu; Jinjian Wu; Guangming Shi; Xin Li

[Paper](https://doi.org/10.1109/TIP.2021.3078058) | [Official code](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution)

**原作训练设定**: 监督、成像模型引导的深度展开融合

**方法核心**: 将基于HSI去噪先验的迭代重建展开，网络内保留观测模型更新。

**对当前 baseline 的迁移建议（研究假设）**: 把当前单次残差输出改成少量观测梯度更新+ResBlock先验更新，研究有限步展开。

**限制与适配边界**: 模型引导不自动等于自监督；原始作者网页404，公开作者实现仍有代码。完整网络更换与轻量迭代分支需分别对照。

**已观察到的实现证据**: sf_8_CAVE/network_31.py、network_31_v1.py、network_31_v2.py、train.py、tst.py，已查看源码树。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://pubmed.ncbi.nlm.nih.gov/33979283/) / [Source 2](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution/tree/main/sf_8_CAVE)

#### 进一步思考：错误前向模型下，似然下降也可能伤害真值

**具体失配与动机。** 模型假设固定 PSF，但实际更宽或有相位偏差时，梯度会要求 CNN 修正一个不存在的内容差。单纯加深去噪器可能掩盖退化误差。

**候选机制。** 采用“校准可信范围 + 保守修正”：先独立估计响应及允许偏差，物理步只在该范围内运行；对不同允许响应计算候选修正，只有对观测验证块稳健的更新才接受。响应无法确定时保留软一致性，而非让一个错误算子的梯度主导所有阶段。

**最小实现与训练。** 使用 P17 的小型迭代包装器，加入 3–5 个低维核宽/相位候选；选候选不使用 HR 真值。网络和响应不在同一个梯度步骤中自由变化；噪声尺度与响应范围需独立设置或校准。

**最小证伪与不保留条件。** 核正确、轻微偏差、超出可信范围三组，比较固定似然、联合盲估计和受限更新。若收益完全来自更宽容的鲁棒损失，贡献应收窄为鲁棒目标，不包装成新的展开机制。

**贡献边界与定位。** P18/P06/P36，鲁棒逆问题与信赖域有先例。候选是含失配自监督融合中的更新接受证据。**优先：中。**

<a id="p19"></a>

### P19 | UAL | CVPR 2020

#### 原报告方法、迁移位置与代码依据

**Unsupervised Adaptation Learning for Hyperspectral Imagery Super-Resolution**

Authors: Lei Zhang; Jiangtao Nie; Wei Wei; Yanning Zhang; Shengcai Liao; Ling Shao

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.html) | [Official code](https://github.com/JiangtaoNie/UAL)

**原作训练设定**: 监督预训练 + 无监督逐样本适配

**方法核心**: 将跨图像的共有先验与当前观测的私有特性分开学习，融合模块预训练后训练适配器。

**对当前 baseline 的迁移建议（研究假设）**: 实现难度较完整 diffusion 低，可将当前 Net 外接自适应修正网络并以重投影损失更新。

**限制与适配边界**: 属于 UTAL 的前作，共享仓库，优先读 TPAMI扩展再回溯。

**已观察到的实现证据**: 公开 Train_FusionModel.py 与 Adaptor.py；README明确两阶段。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: CVF搜索返回官方摘要，直接打开403；不以非官方摘要为证。

Verification sources: [Source 1](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.html) / [Source 2](https://github.com/JiangtaoNie/UAL)

#### 进一步思考：逐图适配把一个场景的偏差写进整个网络

**具体失配与动机。** 适配图像光照、材料比例与噪声不同，更新全部层可能改善当前块，却让相邻材料谱幅漂移。共享网络容量太大时，“适应传感器”和“记住当前图像”难区分。

**候选机制。** 按适配目标区分参数：少量响应条件参数跨场景共享，内容修正 adapter 只在场景内启用。用观测块间的梯度方向一致性及验证收益选择 adapter 位置/容量；梯度冲突强时减少可更新层，不把所有早期光谱卷积都开放。保留一个冻结的参考输出，限制其已被观测支持的分量漂移。

**最小实现与训练。** 比较 `fuse` 后和 `body` 后两处小 adapter，不先搜索大量结构；同一传感器的校准结果可缓存，但不能缓存未经遮挡的目标特征用于盲点训练。明确跨图训练与逐图适配的评估协议。

**最小证伪与不保留条件。** 用不同材料比例场景与固定传感器，比较全量、仅 tail、不同 adapter、无适配。若减少参数即可解释收益，跨块诊断不是独立贡献。若适配改善当前块却恶化其他材料，限制共享范围。

**贡献边界与定位。** P19/P06/P30/P51，轻量适配已有；需证明“传感器参数共享与内容参数局部化”的训练分工。**优先：中高。**

<a id="p20"></a>

### P20 | uSDN | CVPR 2018

#### 原报告方法、迁移位置与代码依据

**Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution**

Authors: Ying Qu; Hairong Qi; Chiman Kwan

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Qu_Unsupervised_Sparse_Dirichlet-Net_CVPR_2018_paper.html) | [Official code](https://github.com/aicip/uSDN)

**原作训练设定**: 无监督耦合自编码器/解混

**方法核心**: 用 Dirichlet 表示学习丰度，结合稀疏与光谱一致性，把空间/光谱信息通过共享解码结构连接。

**对当前 baseline 的迁移建议（研究假设）**: 学习自监督观测重构与物理约束基础；可增加非负、和为1的丰度分支作为结构先验。

**限制与适配边界**: TensorFlow1.8旧实现，需迁移；对混合模型与丰度假设有依赖。

**已观察到的实现证据**: 公开 uSDN.py、pbAuto_uSDN.py、utils.py；作者 README 明确执行 uSDN.py 和 CVPR2018引用。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 基础历史论文；CVF直读403，官方作者代码与arxiv可回溯。

Verification sources: [Source 1](https://github.com/aicip/uSDN)

#### 进一步思考：丰度和为一，可能替光照或增益误差背锅

**具体失配与动机。** Dirichlet/非负混合很有解释性，但相同材料在不同亮度或传感器增益下，观测幅值不完全一致。若所有差异只能由丰度解释，网络会改变材料比例来追逐辐射差。

**候选机制。** 分开建模材料混合与缓变辐射量：`X=g(u,v) E a(u,v)+b(u,v)`，a 非负且和为一，g 正且缓变，b 在第一版固定为零；额外 MSI 增益用已知或受限通道参数。固定 E 的尺度与 g 的平均规范，防止任意缩放。若没有辐射标定，不能同时释放所有 g、b、E。

**最小实现与训练。** `tail` 输出 k 个丰度 logits 和 1 个增益，用 softmax/softplus 解码；E 仅从观测估计，不把 SVD 基直接称作物理端元。用 `L_obs + 增益平滑`，先在已知端元模拟数据验证，再迁到真实材料。

**最小证伪与不保留条件。** 独立控制材料比例、照度和 MSI 通道增益，检查丰度误差是否对纯照度改变更稳定。若 g 只是吸收谱形错误并使输出幅值任意漂移，说明规范不足，不能宣称物理可解释性。

**贡献边界与定位。** P20、解混中的可变光照/端元变异已有。候选仅是自监督融合中辐射失配与丰度误差的分离，需补查端元变异文献。**优先：中；物理假设比结构更关键。**

<a id="p21"></a>

### P21 | DBIN | ICCV 2019

#### 原报告方法、迁移位置与代码依据

**Deep Blind Hyperspectral Image Fusion**

Authors: Wu Wang; Weihong Zeng; Yue Huang; Xinghao Ding; John Paisley

[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Blind_Hyperspectral_Image_Fusion_ICCV_2019_paper.pdf) | [Official code](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion)

**原作训练设定**: 盲融合，监督学习/迭代估计观察模型；不可等同无监督

**方法核心**: 融合和 PSF/SRF 观察模型估计交替优化，同时追求空间与光谱精度。

**对当前 baseline 的迁移建议（研究假设）**: 设计退化估计网络与 fusion 交替更新，研究固定 PSF/R 在真实传感器失配时的局限。

**限制与适配边界**: 作者仓库同时含 TNNLS2021 EDBIN；旧 TensorFlow依赖；需区分两版本。

**已观察到的实现证据**: 公开 models_boost_res/、models_ibp_sn22/、train_boost_res.py、test_boost_res.py及EDBIN代码。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: CVF官方PDF已搜索核对；说明 blind 不代表 label-free。

Verification sources: [Source 1](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Blind_Hyperspectral_Image_Fusion_ICCV_2019_paper.pdf) / [Source 2](https://github.com/wwhappylife/Deep-Blind-Hyperspectral-Image-Fusion)

#### 进一步思考：全图一个 PSF 会把局部退化写成局部内容

**具体失配与动机。** 真实图像可能存在位置变化的模糊，单一全局核在某些区域正确、在另一些区域错误。全图降低平均观测误差后，局部错配仍可能由重建网络补偿。

**候选机制。** 先用独立观测块检验“一个全局核是否足够”；只有残差具有稳定空间模式时，采用两个固定中心、受限核的平滑混合 `P(u,v)=αP1+(1-α)P2`。α 在粗网格估计并平滑，R 固定；不能每个像素自由学习任意核。校准与重建分阶段，响应复杂度由留出兼容收益决定。

**最小实现与训练。** 外置两个分组卷积和粗 α 图，在新训练入口中使用；先评估纯局部响应模拟场景，不迁入旧 TensorFlow 主干。校准比较 `R Y_h` 与相应空间退化的 Y_m 时，要保证混合核应用顺序及网格一致。

**最小证伪与不保留条件。** 全局核、两区域核、连续渐变核三组；对照全局正确/错误核、自由局部核、受限混合。全局核场景中若混合模型仍显著“发明”变化，说明它在拟合噪声，应停用复杂化。

**贡献边界与定位。** P21/P08/P36，空间可变模糊已有大量研究。候选是何时升级响应模型、以及如何防止局部退化与内容互补偿。**优先：中高。**

<a id="p22"></a>

### P22 | Hipandas | ICCV 2025

#### 原报告方法、迁移位置与代码依据

**Hipandas: Hyperspectral Image Joint Denoising and Super-Resolution by Image Fusion with the Panchromatic Image**

Authors: Shuang Xu; Zixiang Zhao; Haowen Bai; Chang Yu; Jiangjun Peng; Xiangyong Cao; Deyu Meng

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Hipandas_Hyperspectral_Image_Joint_Denoising_and_Super-Resolution_by_Image_Fusion_ICCV_2025_paper.html) | [Official code](https://github.com/shuangxu96/Hipandas)

**原作训练设定**: 无监督 HSI–PAN 联合去噪与超分

**方法核心**: 去噪/超分/PAN 重建联合约束，细节低秩先验与两阶段训练。

**对当前 baseline 的迁移建议（研究假设）**: 让残差基线来自去噪后的 LR-HSI；设计 MSI 再投影约束与含噪训练。

**限制与适配边界**: 辅助输入是 PAN 不是多通道 MSI，观测算子需改；两阶段的低分辨率监督须仔细核对。

**已观察到的实现证据**: main.py；model.py；simulate_data.py；utils/

**代码核验程度**: Actual public source observed; not executed or reproduced

#### 进一步思考：噪声从图像跳连直达输出

**具体失配与动机。** 当前 `hsi_up` 保留光谱低频，也把 LR-HSI 噪声直接带到 HR 输出。网络既要生成空间细节，又要预测负噪声抵消跳连；普通观测一致性反而会奖励保留噪声。

**候选机制。** 将两种修正分开：`Z_h=g(Y_h,Y_m_coarse)` 负责 LR 噪声修正，`X_hat=U(Z_h)+delta_detail` 负责空间细节。用目标移除的 LR 留出训练 g；`delta_detail` 接受双观测软约束，避免把低频去噪任务藏在高频分支里。噪声下不能强制 `D(delta_detail)=0` 且 `D X_hat=Y_h` 同时成立，应允许有噪声容差。

**最小实现与训练。** g 先用 2–3 层小卷积，保持原 body；遮挡必须在 g、hsi_head 和跳连之前。独立 MSI 可补充结构，目标噪声若与 MSI 相关则必须联合遮挡/校准。第一版只测独立噪声，再扩展相关噪声。

**最小证伪与不保留条件。** 对照原 skip、去掉 skip、预去噪 skip、分工式 skip，匹配参数与训练。报告低频谱幅、弱材料和高频细节。若去噪收益来自过平滑，或无噪声场景普遍退化，不能主张更稳健的残差基线。

**贡献边界与定位。** P22/P41/P34，联合去噪超分本身已被原作覆盖；候选是条件输入 CNN 中跳连噪声泄漏及低/高尺度训练分工。**优先：高。** 延伸候选：观测已可靠的 LR 分量走直连，估计噪声大的分量走修正，但可靠性需独立估计且不能自由消除损失。

<a id="p23"></a>

### P23 | LRSDN | TIP 2024

#### 原报告方法、迁移位置与代码依据

**Hyperspectral Compressive Snapshot Reconstruction via Coupled Low-Rank Subspace Representation and Self-Supervised Deep Network**

Authors: Yong Chen; Wenzhen Lai; Wei He; Xi-Le Zhao; Jinshan Zeng

[Paper](https://doi.org/10.1109/TIP.2024.3354127) | [Official code](https://github.com/ChenYong1993/LRSDN)

**原作训练设定**: 无外部训练数据的 CASSI 自监督重建

**方法核心**: 正交光谱基与空间系数低秩分解，DGSAN 学习系数先验，并嵌入 ADMM 交替优化。

**对当前 baseline 的迁移建议（研究假设）**: 将 tail 改为较少系数通道，再与光谱基相乘；用双观测算子重写数据项，学习交替更新与低秩约束。

**限制与适配边界**: CASSI 编码与 HSI/MSI 退化不同；固定秩可能丢失稀有材料；仓库使用混合优化代码且说明较少。

**已观察到的实现证据**: run/main.py、run/dgsan.py、run/net.py、run/downsampler.py、initial/、model/，真实源码目录已打开。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://chenyong1993.github.io/yongchen.github.io/papers/2024/Hyperspectral_Compressive_Snapshot_Reconstruction_via_Coupled_Low-Rank_Subspace_Representation_and_Self-Supervised_Deep_Network.pdf) / [Source 2](https://github.com/ChenYong1993/LRSDN/tree/main/run)

#### 进一步思考：正交基稳定了数值，未必稳定了物理光谱

**具体失配与动机。** 基旋转和系数反旋转可以保持输出不变，优化过程中的“基发生变化”不必然表示新材料被发现。带噪输入的基漂移还会让不同迭代的系数含义不一致。

**候选机制。** 用可见 LR-HSI 形成参考基 E0，后续基仅在正交补空间允许小幅受限更新；定期以 Procrustes 对齐消除纯坐标旋转，比较子空间投影矩阵而非逐列基误差。新增方向需通过观测留出预测，而不是仅满足正交约束。对系数更新使用联合观测的有界预条件。

**最小实现与训练。** 先固定 E0，`tail:64→r` 后用 31×r 解码；再加入低自由度基扰动，保持两个阶段的损失完全一致。QR/SVD 可用 PyTorch，第一版基估计和对齐停止梯度，减少奇异值接近导致的训练不稳定。

**最小证伪与不保留条件。** 控制 SNR 与秩，比较固定基、自由正交基、锚定小扰动；检查投影矩阵稳定性、逐材料光谱和重建误差。若变化只是在同一子空间内旋转，不能称自适应发现新光谱。

**贡献边界与定位。** P23/P05/P28/P11；锚定和基对齐成熟，贡献需围绕单场景无标签学习的基漂移误判与训练稳定性。**优先：中。**

<a id="p24"></a>

### P24 | DDS2M | ICCV 2023

#### 原报告方法、迁移位置与代码依据

**DDS2M: Self-Supervised Denoising Diffusion Spatio-Spectral Model for Hyperspectral Image Restoration**

Authors: Yuchun Miao; Lefei Zhang; Liangpei Zhang; Dacheng Tao

[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Miao_DDS2M_Self-Supervised_Denoising_Diffusion_Spatio-Spectral_Model_for_Hyperspectral_Image_Restoration_ICCV_2023_paper.pdf) | [Official code](https://github.com/miaoyuchun/DDS2M)

**原作训练设定**: 只用退化 HSI 的自监督去噪/补全/单图超分；不是双模态融合

**方法核心**: 在反向扩散中用变分空间-光谱模块学习当前图像的后验，空间和光谱网络无需外部训练数据。

**对当前 baseline 的迁移建议（研究假设）**: 研究空间系数与光谱基分解；把其单观测似然替换为 LR-HSI 与 HR-MSI 联合观测，再比较与普通 CNN 内部学习的差异。

**限制与适配边界**: 不能直接复制 main_sisr.py 就得到融合；采样和逐图优化增加时间。CVF 直读遇到403，摘要由作者 arXiv 核验。

**已观察到的实现证据**: main_sisr.py、main_denoising.py、main_completion.py、runners/、models/、functions/；README 给出执行命令。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://arxiv.org/abs/2303.06682) / [Source 2](https://github.com/miaoyuchun/DDS2M)

#### 进一步思考：各波段噪声不同，扩散误差尺度不能统一

**具体失配与动机。** 同一个噪声调度与融合权重可能在低 SNR 波段过拟合、在高 SNR 波段过度去噪。跨波段相关噪声还会污染从观测提取的低维扩散系数。

**候选机制。** 先在低秩域定义两传感器各自的噪声精度；扩散似然与系数先验使用同一单位尺度，避免归一化前后噪声参数错位。相关协方差使用对角加低秩或短带状模型，并做收缩，不能从单个复杂场景自由估计全协方差就声称可靠。

**最小实现与训练。** 第一版只在现有 CNN 损失中加入固定已知白化，验证收益；第二版用独立校准或可信噪声元数据估参数；最后才改 DDS2M 的扩散目标。需要保留变分/采样公式适用假设，不能把加权 L1 叫作完整后验。

**最小证伪与不保留条件。** 模拟白噪声、波段异方差、跨波段相关噪声，对照同方差、对角白化、收缩协方差。若相关结构估计误差使效果不如对角模型，保留简单方案。

**贡献边界与定位。** P24/P32/P34/P44，噪声感知后验是已有方向；贡献应限定到子空间投影与双传感器噪声尺度不相容的训练修正。**优先：中；完整扩散后置。**

<a id="p25"></a>

### P25 | GEWDiff | AAAI 2026

#### 原报告方法、迁移位置与代码依据

**GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution**

Authors: Sirui Wang; Jiang He; Natàlia Blasco Andreo; Xiao Xiang Zhu

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37978) | [Official code](https://github.com/zhu-xlab/GEWDiff)

**原作训练设定**: 监督 HSI 单图超分；扩散模型

**方法核心**: 小波压缩空间-光谱表示、几何增强扩散、多层重建目标。

**对当前 baseline 的迁移建议（研究假设）**: 先迁移轻量小波高低频分支到 body/tail，比较物理自监督下的细节与光谱误差；完整扩散作为后续路线。

**限制与适配边界**: train.py 读取 img_hr_hf 作为目标，不能标为无HR标签。源码硬编码多GPU和NCCL，Win11单卡直接运行需调整。

**已观察到的实现证据**: model/edm.py、RWT.py、unet3d.py、train.py、test.py、run.sh；已打开raw训练代码，提供数据/权重链接。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://raw.githubusercontent.com/zhu-xlab/GEWDiff/main/train.py) / [Source 2](https://github.com/zhu-xlab/GEWDiff)

#### 进一步思考：小波边缘增强会把扫描条带当作几何结构

**具体失配与动机。** 条带和某方向真实边缘都在同一高频子带出现；仅看单幅图像的小波幅值，难以判断该增强哪一个。真实 MSI 的结构证据与 HSI 噪声方向可以提供区分线索。

**候选机制。** 在共同网格比较两路的方向子带：只有经过响应校准、噪声归一化且有独立传感器支持的子带获得较强修正。HSI 单路强而 MSI 无支持的方向保留小步长，并通过目标移除的 HSI 预测检验，避免把 MSI 不可见材料边界全部删除。

**最小实现与训练。** 先用固定 Haar DWT 或手写滤波分解 `tail` 修正，加入三个方向门控；不迁入多卡扩散训练。主 `L_obs` 与噪声目标完整保留，门控只控制细节分支。跨传感器支持在共同 LR 网格上计算，不能把不同分辨率频率直接对应。

**最小证伪与不保留条件。** 水平/垂直条带与同向材料边缘组合，对照普通小波、单路门控、双证据门控。若抑条带同时删除单传感器可见材料边界，机制需要重做。

**贡献边界与定位。** P25/P16/P59；候选是方向噪声与真实边缘的跨传感器证据分工，不是首次 wavelet diffusion。**优先：中高；可用普通 PyTorch 做小版本。**

<a id="p26"></a>

### P26 | UMNet | AAAI 2026

#### 原报告方法、迁移位置与代码依据

**UMNet: Uncertainty-guided Memory Network for Hyperspectral Pansharpening**

Authors: Xiaozheng Wang; Yong Yang; Shuying Huang; Nayu Liu; Ziyang Liu

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37988) | [Official code](https://github.com/EchoPhD/UMNet)

**原作训练设定**: 监督 HSI/PAN 融合；两阶段不确定性训练

**方法核心**: 相邻波段递归融合、NMF状态记忆交互与空间/光谱不确定性损失。

**对当前 baseline 的迁移建议（研究假设）**: 在物理波段而非任意隐藏通道上建递归依赖；研究两观测残差的可靠性权重。

**限制与适配边界**: train.py 显式 outputs-reference 且模型接收 reference，不能直接迁入无标签训练。仓库缺 dataloaders/utils/configs等依赖，README让参照DFCFN，属于部分工程实现。

**已观察到的实现证据**: HFRUNet.py、train.py、trainpre.py 实际存在；已打开train.py确认reference用于损失；About的publish soon文字已过时。

**代码核验程度**: Partial engineering source; dataloaders/utils/configs missing

Verification sources: [Source 1](https://github.com/EchoPhD/UMNet/blob/main/train.py) / [Source 2](https://github.com/EchoPhD/UMNet)

#### 进一步思考：无真值时，什么叫“可靠的误差图”？

**具体失配与动机。** 直接以观测残差训练不确定性头，可能让它把算子错误、噪声和重建偏差全装进“大方差”；若方差自由增大，网络可以降低损失却不改善图像。

**候选机制。** 分开预测传感器噪声与模型敏感性：噪声尺度由校准/适用风险估计得到；模型敏感性用目标移除后的多遮挡预测分歧估计。Gaussian NLL 包含 `r²/v + log v`，对 v 设物理合理范围，并用独立观测块检验覆盖率；高分歧只能表示不稳定，不能直接称 HR 真值误差。

**最小实现与训练。** `tail` 后增加小方差头，先只在观测域输出 31/3 个噪声尺度或分块尺度；以固定噪声白化为强对照。先完成观测不确定性校准，不同时宣称 HR 像素置信区间。原代码为部分工程，完整自监督训练需另写。

**最小证伪与不保留条件。** 比较预测方差与已知模拟噪声、留出残差分布和 HR 误差相关性。若方差只跟亮度/难例相关却不校准，或靠增大 v 掩盖错配，不保留“可靠不确定性”的贡献。

**贡献边界与定位。** P26/P32/P31；无标签可校准性的证据比再加 uncertainty head 更重要。**优先：中高。** 延伸候选：让适配停止依据留出风险与预测稳定性共同满足，而非由单一最大方差决定。

<a id="p27"></a>

### P27 | DM-ZS | CVPR 2025

#### 原报告方法、迁移位置与代码依据

**Hyperspectral Pansharpening via Diffusion Models with Iteratively Zero-Shot Guidance**

Authors: Jin-Liang Xiao; Ting-Zhu Huang; Liang-Jian Deng; Guang Lin; Zihan Cao; Chao Li; Qibin Zhao

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Hyperspectral_Pansharpening_via_Diffusion_Models_with_Iteratively_Zero-Shot_Guidance_CVPR_2025_paper.html) | [Official code](https://github.com/Jin-liangXiao/DM-ZS)

**原作训练设定**: HSI/PAN、观测内零样本引导+外部预训练RGB扩散先验

**方法核心**: 用当前观测训练引导网络，反复引导生成RGB细节，再通过神经空间-光谱分解映射回HSI。

**对当前 baseline 的迁移建议（研究假设）**: 学习将空间细节生成与光谱映射分开；先在现有 tail 前加入低维细节分支和双观测约束。

**限制与适配边界**: 不是所有参数仅用当前观测学习；需要外部 diffusion checkpoint。PAN换MSI后需改变观测与引导，不可强迫31波段直接使用RGB扩散。

**已观察到的实现证据**: Demo_main.py、guide_hyperpan.py、model1.py、guided_diffusion/、models/、base.json；README 指定预训练权重与引导训练。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://github.com/Jin-liangXiao/DM-ZS)

#### 进一步思考：RGB 先验可能补出“好看但不存在”的材料细节

**具体失配与动机。** RGB 扩散先验能产生自然纹理，却无法区分同 RGB 投影的不同光谱材料；先验输出经自由光谱映射后，可能获得低观测误差但错误窄谱结构。

**候选机制。** 把生成式细节视为候选，先检验其经过 R 与 D 后的支持。输出分成被观测约束的修正与共同弱约束残差，后者设置幅度/光谱平滑/子空间限制，并检验对随机种子和响应扰动的敏感性。不能只投影到 R 值域就删除所有高频：HSI 与 MSI 的联合约束才是判断依据。

**最小实现与训练。** 先用原 Net 做初始化，与不使用生成先验的版本比较；小图上运行少步先验修正，31 波段解码用观测基，明确外部 RGB 预训练数据的使用。做有限步软一致性，噪声下不硬投影。

**最小证伪与不保留条件。** 同 RGB 投影的不同材料纹理、不同随机种子、未见 R，报告视觉细节和材料谱误差分别的变化。若纹理稳定但光谱偏差增大，生成先验只可作为视觉辅助，不可主张高光谱真实性改善。

**贡献边界与定位。** P27/P38/P43/P44；候选是生成先验在融合共同弱约束方向的可审计限制。**优先：中低；不要作为第一轮实现。**

<a id="p28"></a>

### P28 | HIR-Diff | CVPR 2024

#### 原报告方法、迁移位置与代码依据

**HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models**

Authors: Li Pang; Xiangyu Rui; Long Cui; Hongzhong Wang; Deyu Meng; Xiangyong Cao

[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.pdf) | [Official code](https://github.com/LiPang/HIRDiff)

**原作训练设定**: 使用外部预训练扩散先验的无监督 HSI 恢复/单图超分

**方法核心**: 低维图像与光谱系数矩阵分解，SVD/RRQR估计系数，TV引导低维扩散采样。

**对当前 baseline 的迁移建议（研究假设）**: 学习把光谱映射与空间生成分离；为扩散后验加入MSI观测项，或先做低秩CNN对照。

**限制与适配边界**: 是使用预训练先验的任务零样本方案，不是从零只用当前观测；低维RGB先验可能抹掉异常材料。CVF页面抓取失败，作者arXiv与实现可访问。

**已观察到的实现证据**: main.py、guided_diffusion/、configs/、matlab/、utility/；提供去噪/超分/补全命令与预训练模型链接。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://arxiv.org/abs/2402.15865) / [Source 2](https://github.com/LiPang/HIRDiff)

#### 进一步思考：信息量大的波段，也可能只是噪声大的波段

**具体失配与动机。** SVD/RRQR 的能量或线性独立性选择，若不处理坏波段和异方差噪声，可能偏向高噪声方向。代表波段未覆盖窄吸收区，也可能丢失材料差异。

**候选机制。** 噪声白化后做稳健代表波段选择，同时约束真实波长覆盖；解码回物理单位时必须撤销白化。用独立观测预测选择子空间维数，避免将训练拟合当作信息量。真实波长未知时，只能做序号覆盖实验，不能声称物理波长优化。

**最小实现与训练。** 先改低秩基/代表波段构造，不迁入完整扩散；观测估计对角噪声尺度或使用模拟已知尺度，固定选择结果训练 Net 的低秩 tail。以不白化的 RRQR、仅白化、白化加覆盖对照。

**最小证伪与不保留条件。** 加入死波段、不同 SNR、窄谱特征，检查选中波段、子空间误差和材料 SAM。若覆盖约束只改善平均谱平滑却丢失稀有峰，不保留该权重设置。

**贡献边界与定位。** P28/P23/P05，以及已有 HySURE 子空间选择。贡献应是噪声与波长覆盖失配的稳健子空间构造。**优先：中。**

<a id="p29"></a>

### P29 | UniPAN | ICCV 2025

#### 原报告方法、迁移位置与代码依据

**Enpowering Your Pansharpening Models with Generalizability: Unified Distribution is All You Need**

Authors: Yongchuan Cui; Peng Liu; Hui Zhang

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.html) | [Official code](https://github.com/yc-cui/UniPAN)

**原作训练设定**: 监督 MS–PAN 融合；跨传感器泛化

**方法核心**: 训练和测试映射到统一像素分布以缓解传感器分布偏移。

**对当前 baseline 的迁移建议（研究假设）**: 数据预处理/反变换对照；检验固定归一化和不同 SRF 的泛化。

**限制与适配边界**: 逐波段映射可能改变真实光谱比例，必须保留可逆变换并检验 SAM；不是直接 HSI 方法。

**已观察到的实现证据**: examples/FeINFN.py 等12模型训练测试；env.yaml

**代码核验程度**: Actual public source observed; not executed or reproduced

#### 进一步思考：分布统一可能破坏光谱比值和物理响应

**具体失配与动机。** 各波段独立直方图匹配虽能让输入更像预训练分布，却不一定保留谱幅、SAM 或 `Y_m=RX`。网络可能在变换空间中拟合成功，在原单位上光谱失真。

**候选机制。** 只允许可追踪的辐射变换，并同步变换前向模型。例如共享 `X'=aX+b1` 时，`Y_m'=aY_m+bR1`；不能无条件给每个 MSI 通道加同一个 b。逐波段对角变换 G 要同步把响应改成 `R G^{-1}`，并记录逆变换。第一版使用全局 a、b，避免条件数和噪声同步变化过复杂。

**最小实现与训练。** Dataset 外加可逆包装，Net 处理变换后的观测，损失在原物理单位或严格等价的变换单位计算；变换参数只从可见观测/训练集获得，不从测试 HR 估计。保留未经变换的强对照。

**最小证伪与不保留条件。** 跨照度与传感器增益，比较直方图匹配、普通标准化、模型同步的可逆变换；检查物理兼容误差与光谱幅值。若变换本身不可逆或响应同步不准确，不能主张光谱保真。

**贡献边界与定位。** P29/P45/P20；候选是迁移分布统一时的响应协变训练规则，普通归一化不是贡献。**优先：中高；成本低，适合先诊断。**

<a id="p30"></a>

### P30 | SpectralGPT | TPAMI 2024

#### 原报告方法、迁移位置与代码依据

**SpectralGPT: Spectral Remote Sensing Foundation Model**

Authors: Danfeng Hong et al.

[Paper](https://doi.org/10.1109/TPAMI.2024.3362475) | [Official code](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)

**原作训练设定**: 大规模自监督光谱遥感表征预训练；下游理解任务

**方法核心**: 3D空间-光谱token、遮挡重建、多目标光谱学习和渐进式预训练。

**对当前 baseline 的迁移建议（研究假设）**: 借鉴光谱遮挡预训练 hsi_head，或冻结编码器加轻量适配；重建训练仍需双观测一致性。

**限制与适配边界**: 基础模型的预训练并非当前图像对零样本；CAVE与卫星MSI域差异显著。完整预训练成本大，官方示例8张4090；不能据此推断单卡无法适配。

**已观察到的实现证据**: main_pretrain.py、engine_pretrain.py、models_mae_spectral.py、models_vit_tensor.py、downstream_tasks/；有真实预训练和下游入口。

**代码核验程度**: Actual public source observed; not executed or reproduced

Verification sources: [Source 1](https://www.bgd.ed.tum.de/pdf/2024_SpectralGPT_Hao.pdf) / [Source 2](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)

#### 进一步思考：预训练的遮挡重建任务与融合测量缺失不同

**具体失配与动机。** MAE 的 token 缺失，不等价于 PSF 模糊、光谱响应积分与坏波段；卫星预训练特征也可能不适合 CAVE 近景材料。简单加载编码器不能证明它对实际融合有用。

**候选机制。** 保留冻结小型编码器，只学习响应条件 adapter；构造符合真实 R/P 的观测预测任务，而非随机 token 复原。让编码器预测可见的留出观测，解码仍受原单位 `L_obs` 约束；预训练特征约束限定于可见低尺度结构，不能让教师未经遮挡地看完整目标。

**最小实现与训练。** 先用小光谱编码器仿照训练机制，不复现超大模型预训练；若使用公开权重，核对输入波段、波长、归一化与 token 形状，评估冻结、小 adapter、从头训练三组。固定 31/3 版本先落地，变长波段是后续工程。

**最小证伪与不保留条件。** 同域与跨域、未见 R 与坏波段，对照随机初始化等规模编码器。若收益仅来自额外外部数据，贡献要明确为预训练迁移，不称只靠当前图像的零样本自监督。

**贡献边界与定位。** P30/P51/P45，物理任务适配已有相近方向；需证明确切预训练—测量失配及 adapter 的作用。**优先：中低；大模型后置。**

<a id="p31"></a>

### P31 | Equivariant Splitting | ICLR 2026

#### 原报告方法、迁移位置与代码依据

**Equivariant Splitting: Self-supervised learning from incomplete data**

Authors: Victor Sechaud; Jérémy Scanvic; Quentin Barthélemy; Patrice Abry; Julián Tachella

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34a5f638617ce32700e235e50dff9c8a-Abstract-Conference.html) | [Official code](https://github.com/vsechaud/Equivariant-Splitting)

**原作训练设定**: 单一不完整观测算子、观测拆分与等变重建

**方法核心**: 结合观测拆分损失与新的重建等变定义，在给定假设下得到监督损失的无偏估计。

**对当前 baseline 的迁移建议（研究假设）**: 推断：对 HSI/MSI 联合观测设计留出与重投影，并测试等变网络/训练约束；优先学习自监督损失实现。

**限制与适配边界**: 理论假设不自动覆盖高光谱融合；训练时被留出的目标不能通过 MSI、插值基线或跨模态交互泄漏。

**已观察到的实现证据**: 官方仓库 mri/、tomography/、cs+inp/ 均有训练说明；README 明确 run_train.py 调用 train.py，并链接 DeepInverse 支持。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 正式 ICLR 2026 论文，官方 proceedings 与实际代码双重确认；OpenReview 页面有浏览器验证，使用 proceedings 核验。

#### 进一步思考：按图像像素遮挡，不等于按测量信息拆分

**具体失配与动机。** 一个 LR-HSI 测量由多个 HR 位置经 PSF 聚合；同一目标也可能经跳连、邻域或缓存回到网络。把输出图遮一块再训练，无法保证被留出的观测目标没有参与输入。

**候选机制。** 在观测索引上拆分 HSI/MSI 测量，网络从保留测量重建，再预测留出测量。遮挡先作用于原始输入，所有填补与 `hsi_up` 都从保留部分生成。若目标噪声有空间/光谱相关性，扩张同传感器保护区；独立 MSI 提供干净结构可以保留。对联合观测的拆分目标先推导所需零条件均值/独立性及算子条件，不直接继承原论文无偏结论。

**最小实现与训练。** 写 `MaskedObservationDataset` 与输入包装器，31/3 通道可先零填加固定插补；若加 valid-mask 特征，则明确改变 head 的输入通道。损失只评价实际存在的留出测量，不能把零填区当作真实零目标。训练时动态遮挡，另用固定未参与调参的遮挡检验泛化；模型不应在最后无掩码适配中重新接触留出噪声。

**最小证伪与不保留条件。** 做依赖测试：改变某个留出目标数值，保持其他可见数据不变，网络输入/初始化/固定前向输出必须不变。比较无遮挡、像素遮挡、全路径观测拆分及相关保护区，检查噪声误差与弱材料。若保护区只靠删信息降噪且损害融合，应缩小或重设拆分。

**贡献边界与定位。** P31/P41/P40、AP-BSN/AT-BSN 都接近。候选贡献是双传感器聚合测量与跳连下的可审计拆分，而非首次 observation splitting。**优先：高。** 延伸候选：为不同噪声相关范围设置不同空间—光谱拆分形状，通过独立校准选择，而非凭最终 PSNR 调 mask。

<a id="p32"></a>

### P32 | UNSURE | ICLR 2025

#### 原报告方法、迁移位置与代码依据

**UNSURE: self-supervised learning with Unknown Noise level and Stein's Unbiased Risk Estimate**

Authors: Julián Tachella; Mike Davies; Laurent Jacques

[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d36dfcdb14473a8526111c221660f2ab-Abstract-Conference.html) | [Official code](https://github.com/tachella/unsure)

**原作训练设定**: 未知噪声水平，仅带噪观测训练

**方法核心**: 通过约束/对偶优化构造无需预先知道噪声水平的 SURE 类学习目标。

**对当前 baseline 的迁移建议（研究假设）**: 推断：对 HSI/MSI 观测分别建风险估计，研究波段相关与异方差噪声；修改 loss 与训练循环，架构可先保持。

**限制与适配边界**: 未知噪声水平不等于任意未知噪声；须选择适用变体与协方差参数化，纯风险估计也不自动解决融合的联合零空间。

**已观察到的实现证据**: 实际官方仓库有 unsure.py、main.py、models.py、generate_dataset.py；README 链接 DeepInverse 损失及可运行 notebook。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 官方 ICLR proceedings 已确认；README 标题曾使用简写 Unknown Noise level Stein's Unbiased Risk Estimator，应以 proceedings 全题为准。

#### 进一步思考：未知噪声估计把响应偏差也当成噪声

**具体失配与动机。** 当固定 R/P 不准确，观测残差包含系统偏差。让未知噪声模型解释全部残差，可能得到过大的估计方差，从而降低对真正可修正结构的惩罚。

**候选机制。** 校准响应和估计噪声采用分离证据：先检查共同 LR 域的系统性残差与相位敏感性，再用适用 UNSURE 变体处理剩余随机噪声。结构化协方差采用收缩、少量带状/低秩参数及尺度限制；不能自由拟合整个 31×31 协方差来吸收材料结构。风险估计目标需对应 `A f(y)` 的观测域预测或满足条件的投影风险，不能未经推导称 HR 风险无偏。

**最小实现与训练。** 从已知 A、独立 Gaussian、已知 σ 的 SURE 对照开始，再引入未知 σ；Monte Carlo 散度作用于正确单位的观测预测，可先用一个随机探针估计并检查方差。未知噪声的优化遵循原作约束/对偶机制，而非简单把 σ 当可学损失除数。之后才扩展双传感器和相关噪声。

**最小证伪与不保留条件。** 独立改变噪声水平和核误差，检查 σ 是否在仅有核误差时虚增。若未知噪声方案不优于固定正确 σ，或估计值与模拟噪声无关，不能宣称可靠噪声适应。

**贡献边界与定位。** P32/P34/P26，另有 S5P 自监督 SR 已结合 SURE 与 EI。贡献要聚焦校准误差污染风险估计的机制。**优先：中高；理论核对比加 loss 更重要。**

<a id="p33"></a>

### P33 | MOI | NeurIPS 2022

#### 原报告方法、迁移位置与代码依据

**Unsupervised Learning From Incomplete Measurements for Inverse Problems**

Authors: Julián Tachella; Dongdong Chen; Mike Davies

[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/203e651b448deba5de5f45430c45ea04-Abstract-Conference.html) | [Official code](https://github.com/edongdongchen/MOI)

**原作训练设定**: 多观测算子的无监督逆问题；MOI

**方法核心**: 分析不同算子、测量数和信号模型维度的可辨识条件，并以跨算子重建构造无监督训练损失。

**对当前 baseline 的迁移建议（研究假设）**: 推断：把不同传感器 SRF/PSF、尺度和采样模式作为条件，研究跨算子观测约束与退化条件化网络。

**限制与适配边界**: 双模态输入不自动满足 MOI 的所有条件；需要有真实或合法获得的不同算子测量，不能任意编造观测以声称识别保证。

**已观察到的实现证据**: 官方 MOI 仓库存在 moi/、physics/、demo_train.py、demo_test.py；README 指向 moi.py。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 高度适合理解自监督融合的可辨识性与传感器泛化；不同论文同作者不代表同一训练设置。

#### 进一步思考：有多种退化条件，不代表有新的真实观测信息

**具体失配与动机。** 从同一个 `X_hat` 合成许多 R/P 的观测，再训练恢复 `X_hat`，只是在蒸馏自身猜测；这些数据不能自动消除原观测共同零空间。跨传感器训练容易把这种闭环误称为多算子可辨识性。

**候选机制。** 明确区分三种数据：同场景真实多传感器测量、训练图像合法生成的多算子模拟测量、当前重建产生的伪测量。仅第一/第二类可用来检验额外真实信息或训练分布收益；第三类只作为一致性正则。用小型低秩离散联合矩阵计算各响应/采样模式覆盖的弱方向，再选择真正互补的训练算子，而非盲目增加数量。

**最小实现与训练。** 在训练模拟集生成不同已知 SRF/PSF，网络接收响应编码；严格保存目标来源和外部信息标签。固定内容、改变算子做独立测试；真实只有一对观测时不声称拥有同场景多算子保证。有限 toy 分析再决定是否研究真实追加测量。

**最小证伪与不保留条件。** 同数量冗余算子与互补算子、真实/合法模拟与伪测量分别比较，检查弱方向覆盖与未见 R 重建。若收益仅来自多次增强且与算子互补性无关，贡献应收窄为条件化泛化。

**贡献边界与定位。** P33/P35/P45，MOI 理论不能被任务输入数量替代。候选是融合训练中算子多样性和信息增益的区分。**优先：中高；需要严格数据协议。**

<a id="p34"></a>

### P34 | REI | CVPR 2022

#### 原报告方法、迁移位置与代码依据

**Robust Equivariant Imaging: A Fully Unsupervised Framework for Learning to Image From Noisy and Partial Measurements**

Authors: Dongdong Chen; Julián Tachella; Mike E. Davies

[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Robust_Equivariant_Imaging_A_Fully_Unsupervised_Framework_for_Learning_To_CVPR_2022_paper.html) | [Official code](https://github.com/edongdongchen/REI)

**原作训练设定**: 有噪声的不完整观测；Gaussian、Poisson、混合噪声

**方法核心**: 用 SURE 修正带噪测量一致性，同时加入鲁棒等变约束，避免把噪声当作准确目标。

**对当前 baseline 的迁移建议（研究假设）**: 推断：为 HSI/MSI 两传感器分别建噪声模型，在训练脚本中替换简单观测 L1，并保留双输入 Net。

**限制与适配边界**: 噪声模型和参数必须适用；偏置、条带及跨波段相关噪声不能一律用独立白噪声处理；仍受 EI 可辨识条件限制。

**已观察到的实现证据**: 官方仓库明确核心 rei_end2end.py 与 rei_end2end_ct.py，存在 rei/、demo_scripts/ 和训练测试入口。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 论文 PDF 一次访问被 CVF 403 阻止；arXiv 摘要与官方仓库提供同一论文和代码；代码存在，未实际运行。

#### 进一步思考：等变循环会不会反复复制同一份噪声？

**具体失配与动机。** `X_hat` 重新投影后作为新输入，如果同时保留原带噪目标，循环中噪声及模型偏差可能相互强化。两传感器噪声尺度不同，用同一噪声扰动也可能错误耦合它们。

**候选机制。** 先为两路分别建立适用噪声模型与风险项，再构造变换后的合法前向测量；循环目标从停止梯度的重建生成，而输入加入独立的新噪声实现。这个循环提供先验训练信号，不等于新真实测量。几何变换同步改变坐标、采样相位和有效视野，避免所谓等变损失其实在比较不对应像素。

**最小实现与训练。** 保持 Net，先做观测域 SURE/已知噪声对照，再加一次循环；Gaussian、Poisson 分别实现，不能把同一公式用于条带。记录随机扰动独立性与每一项梯度路径；循环权重先小范围验证。

**最小证伪与不保留条件。** 相同噪声复用和独立噪声、正确变换算子和未同步相位的变换分别对照。若收益来自普通数据增强，或者低通共同零空间未获得可解释改进，不能把 REI 理论保证直接搬来。

**贡献边界与定位。** P34/P35/P32，已有 REI 处理噪声等变；候选是融合双噪声与采样相位下的循环训练核对。**优先：中。**

<a id="p35"></a>

### P35 | EI | ICCV 2021

#### 原报告方法、迁移位置与代码依据

**Equivariant Imaging: Learning Beyond the Range Space**

Authors: Dongdong Chen; Julián Tachella; Mike E. Davies

[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.html) | [Official code](https://github.com/edongdongchen/EI)

**原作训练设定**: 仅不完整观测，自监督逆问题；原文实验 CT 与图像修复

**方法核心**: 将观测一致性与重建结果的群变换等变性结合，为观测算子零空间引入训练约束。

**对当前 baseline 的迁移建议（研究假设）**: 推断：保留 Net 双输入，将联合算子 A(X)=(空间模糊降采样 X, 光谱响应 R X) 写成可微模块，添加 f(A(T f(y)))≈T f(y) 的等变循环。

**限制与适配边界**: 必须核对联合算子与变换群的可辨识性；普通翻转/旋转可能不增加低通与光谱缺失的信息。不是直接照搬旋转损失便保证高光谱恢复。

**已观察到的实现证据**: 官方 README 指定核心 ei/closure/ei.py；仓库存在 get_started.py、demo_train.py、physics/、transforms/。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: EI 是 ICCV 2021，不能误记为 ICML。原文没有验证该 HSI+MSI baseline。

#### 进一步思考：旋转和翻转可能并没有补足缺失信息

**具体失配与动机。** 各向同性低通与普通旋转近似交换；若 T 总把原共同零空间映回相同零空间，等变约束很可能只是增强，不足以解决欠定问题。

**候选机制。** 先筛选变换：在小型离散模型找满足 `A v≈0` 的方向 v，检查 `A T v` 是否仍近零，以及所有允许变换后的算子联合覆盖如何变化。只研究符合图像分布与传感器条件的变换；任意波段置换不具有物理波长不变性，不能为增加秩而随便使用。空间尺度/平移也要核对相位及分布假设。

**最小实现与训练。** 先写小尺寸矩阵实验，不改主干；比较普通旋转、正确坐标平移、合法尺度变化的约束效果。训练一次循环即可，未满足条件的情况标记为经验正则，而不是可辨识性证明。

**最小证伪与不保留条件。** 人工注入已知共同零空间扰动，检验哪种约束能区分，再看独立融合测试。若 null perturbation 在所有候选下都不可见，停止“通过这些变换解决零空间”的主张，转向明确先验或额外测量。

**贡献边界与定位。** P35/P33、Scale-Equivariant Imaging 是最近邻。候选是 HSI–MSI 联合算子的变换适用性诊断；诊断本身需有实质新结论才足够。**优先：中高；适合先做理论/数值辨识。**

<a id="p36"></a>

### P36 | KernelGAN | NeurIPS 2019

#### 原报告方法、迁移位置与代码依据

**Blind Super-Resolution Kernel Estimation Using an Internal-GAN**

Authors: Sefi Bell-Kligler; Assaf Shocher; Michal Irani

[Paper](https://proceedings.neurips.cc/paper/2019/hash/5fd0b37cd7dbbb00f97ba6ce92bf5add-Abstract.html) | [Official code](https://github.com/sefibk/KernelGAN)

**原作训练设定**: 单幅 LR 图像的无监督空间退化核估计

**方法核心**: 通过跨尺度 patch 分布匹配，用图像内部 GAN 学习图像特定的降采样核。

**对当前 baseline 的迁移建议（研究假设）**: 推断：为固定 PSF 的 baseline 增加可学习空间退化核，结合 HSI/MSI 两路观测一致性，研究盲融合。

**限制与适配边界**: 只估空间核，不直接估光谱响应 R；小图块、噪声、跨尺度分布不匹配会导致不稳定，联合估计存在退化解需约束。

**已观察到的实现证据**: 官方仓库存在 kernelGAN.py、learner.py、networks.py、loss.py、train.py，README 明确 NeurIPS 2019。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 为 PSF 加非负/和为一/支撑/中心约束及独立核精度评估；迁移建议是推断。

#### 进一步思考：模糊核中心与采样相位容易混淆

**具体失配与动机。** 核偏心和采样起点偏移都能移动输出；若两者同时自由学习，存在参数互补偿。内部 patch 分布匹配也可能把条带或重复纹理当作退化证据。

**候选机制。** 固定核质心规范，单独估计相位；优先通过 `R Y_h≈D_{P,phi}Y_m` 的跨传感器关系校准，并在独立块检验。第一版 R 已知，P 仅少量宽度/椭圆参数；高维核只在低维模型被证伪后开放。对兼容残差做鲁棒处理，并报告可识别范围，而非认为所有核可恢复。

**最小实现与训练。** 小型离散相位候选或连续 `grid_sample` 偏移，核非负、和为一且中心受限；校准完成后冻结供 Net 使用。先比较传统观测校准与内部 GAN，不强制迁入判别器。相位和偶数核边界须匹配当前 SciPy 数据生成。

**最小证伪与不保留条件。** 单独核宽变化、相位变化、核偏心、条带四组，检查估计参数、留出相容与 HR 重建。若相位与质心仍自由漂移但重建变好，只能报告等效算子拟合，不声称真实 PSF 精确恢复。

**贡献边界与定位。** P36/P07/P08、HySure 已有响应估计。候选是盲融合的中心—相位规范与噪声污染诊断。**优先：高。** 延伸候选：多幅同传感器场景共享 P、允许场景内容独立，以跨场景稳定性拒绝被单幅重复纹理误导的核。

<a id="p37"></a>

### P37 | ZSSR | CVPR 2018

#### 原报告方法、迁移位置与代码依据

**"Zero-Shot" Super-Resolution Using Deep Internal Learning**

Authors: Assaf Shocher; Nadav Cohen; Michal Irani

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf) | [Official code](https://github.com/assafshocher/ZSSR)

**原作训练设定**: 单图像内部样本的零样本超分辨率

**方法核心**: 从当前 LR 图像进一步降采样构造内部训练对，以图像内部跨尺度重复训练小网络。

**对当前 baseline 的迁移建议（研究假设）**: 推断：构造低一级 HSI/MSI 配对尺度，用观测自身建立 reduced-resolution 自监督融合训练，再应用到原尺度。

**限制与适配边界**: 内部重复性与跨尺度退化一致性可能不成立；原图像内容变成低尺度标签不能等同目标 HR-HSI 真值；噪声和 aliasing 会传播。

**已观察到的实现证据**: 官方仓库真实存在 ZSSR.py、run_ZSSR.py、run_ZSSR_single_input.py、imresize.py，README 明确 CVPR 2018。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 官方旧代码 TensorFlow；适合借鉴数据构造思路，不能原样接入双输入 torch Net。

#### 进一步思考：内部训练尺度越多，不一定越有帮助

**具体失配与动机。** 小 LR 图继续缩小后材料和纹理消失；含噪图的内部标签又重复带噪。当前 patch 尺寸尤其容易让内部任务退化成几乎无空间结构的预测。

**候选机制。** 用有效 LR 尺寸、可靠梯度比例与 SNR 限制内部倍率；先从 q=2 开始，只在保留足够结构时进入 q=4/8。显式输入 q 与有效响应，用原尺度双观测适配缓解训练部署差距。若继续缩小已抹掉目标材料，就不把该尺度纳入训练。

**最小实现与训练。** 在完整 LR 图上生成内部对，再裁剪；每次记录最小输入尺寸和材料/梯度覆盖统计。固定 curriculum、随机尺度、结构约束 curriculum 同预算比较；内部 Y_h 标签含噪时配合 P31/P41 的目标移除或适用风险目标。

**最小证伪与不保留条件。** 同场景不同 LR 尺寸、倍率和 SNR，检查小图下是否更稳定；部署倍率固定，训练倍率变化。若结构筛选只是减少训练数据、与随机同量样本无差异，则不保留它作为贡献。

**贡献边界与定位。** P37/P07/P03，内部尺度 curriculum 有先例；候选是观测信息耗尽与倍率迁移之间的训练取舍。**优先：高；先确保工程尺寸正确。**

<a id="p38"></a>

### P38 | DDNM | ICLR 2023

#### 原报告方法、迁移位置与代码依据

**Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model**

Authors: Yinhuai Wang; Jiwen Yu; Jian Zhang

[Paper](https://openreview.net/pdf?id=mRieQgMtNTQ) | [Official code](https://github.com/wyhuai/DDNM)

**原作训练设定**: 预训练扩散先验的零样本线性逆问题；DDNM/DDNM+

**方法核心**: 将解分为观测确定的值域与先验决定的零空间，在反向扩散中校正测量一致性。

**对当前 baseline 的迁移建议（研究假设）**: 推断：先借鉴输出投影/数据一致性层而非迁入整个扩散模型；联合算子为 HSI 空间降采样和 MSI 光谱降维，修正其共同零空间。

**限制与适配边界**: RGB 预训练先验不能直接处理 31 波段，任务零样本不等于无需干净先验训练数据；噪声下硬投影会拟合噪声，联合伪逆计算有成本。

**已观察到的实现证据**: 作者仓库含 main.py、functions/、guided_diffusion/、configs/，提供 SVD 与 simplified 两实现；论文直接给作者代码链接。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 当前插值基线加残差不保证观测一致性。作者拼写 GitHub/arXiv 为 Yinhuai，OpenReview PDF 文本曾显示 Yinhui，须引用规范 BibTeX。

#### 进一步思考：两种带噪观测之间可能不存在共同硬一致解

**具体失配与动机。** 独立噪声、响应误差和错位会让两路观测不完全相容。轮流硬投影到 HSI/MSI 约束，可能振荡或把噪声加入输出；当前残差网络也并不满足任一硬约束。

**候选机制。** 用联合软一致性近端步骤替代两次独立硬投影：

\[
X_{dc}=\arg\min_X \frac12\|X-X_0\|^2+\frac\eta2\|W_h^{1/2}(DX-Y_h)\|^2+\frac{\eta\lambda}2\|W_m^{1/2}(RX-Y_m)\|^2.
\]

其线性系统可用少步共轭梯度解；W/η 固定或有界，不能通过让 W 变零逃避约束。共同零空间由 X0 的先验保留，软一致性不会凭空解决它。

**最小实现与训练。** `X0=Net(...)` 后放 3–5 步可微 CG，已知算子先验证伴随，再端到端训练。使用 autograd 也需验证 CG 截断与内存；第一版可固定修正模块，不搬整个扩散先验。

**最小证伪与不保留条件。** 无噪声、传感器 SNR 不平衡、轻微 R/P 错配，对照硬 HSI、硬 MSI、轮流硬投影、联合软修正；迭代预算一致。若软层只是额外平滑而在弱材料上更差，缩小 η 或撤销其核心地位。

**贡献边界与定位。** P38/P17/P09/P44，软数据一致性成熟。贡献需要自监督中不相容观测的诊断、权重标定与训练机制。**优先：高；也应是很多候选的强对照。**

<a id="p39"></a>

### P39 | Neighbor2Neighbor | CVPR 2021

#### 原报告方法、迁移位置与代码依据

**Neighbor2Neighbor: Self-Supervised Denoising From Single Noisy Images**

Authors: Tao Huang; Songjiang Li; Xu Jia; Huchuan Lu; Jianzhuang Liu

[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.html) | [Official code](https://github.com/TaoHuang2018/Neighbor2Neighbor)

**原作训练设定**: 由单幅带噪图像随机邻域子采样构造训练对

**方法核心**: 相邻像素近似共享干净信号，随机邻域配对加正则项，训练普通去噪网络。

**对当前 baseline 的迁移建议（研究假设）**: 推断：先研究 LR-HSI 或 MSI 的自监督去噪，再将子采样一致性适配双模态；代码便于学习随机 mask。

**限制与适配边界**: 邻近像素并不严格相同，边缘/纹理有偏差；相关噪声破坏配对假设；它不直接生成 HR-HSI 真值。

**已观察到的实现证据**: 官方 PyTorch 仓库存在 train.py、arch_unet.py、dataset_tool.py；README 明确 CVPR 2021，DeepInverse 官方文档也注明从此仓库适配。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 不要误用搜索结果中的第三方 pminhtam/Neigh2Neigh；本记录使用作者仓库。

#### 进一步思考：邻像素噪声独立，但干净信号并不相同

**具体失配与动机。** 材料边界、窄结构两侧可能有不同光谱；随机邻域配对会把谱差当噪声平滑。随着倍率增大，一个 LR 像素的混合内容差异更显著。

**候选机制。** 用独立 MSI 在共同网格的结构证据筛选内部邻对，优先同区域、避免跨边界；保留一定随机覆盖以减少采样偏差。若要预测邻对干净差值，只能用可见结构与低自由度光谱锚，不用目标 HSI 数值选择“相近对”，以免选择过程依赖目标噪声。

**最小实现与训练。** 数据层修改邻对采样，不先改 Net；LR 去噪小分支先单独验证，再作为跳连基线。相似性从 MSI 或独立校准特征得到，MSI 不可见边界标注为未知而非强认同材质。

**最小证伪与不保留条件。** 纯平坦、边界密集、MSI 近色但谱不同三组，比较随机邻对、MSI 筛选及盲点对照。若只改善平坦区而材料边界损失更大，该机制不适合作融合主训练目标。

**贡献边界与定位。** P39/P41/P47，边缘保持配对已有；候选是融合尺度下的干净邻对偏差控制。**优先：中。**

<a id="p40"></a>

### P40 | Noise2Same | NeurIPS 2020

#### 原报告方法、迁移位置与代码依据

**Noise2Same: Optimizing A Self-Supervised Bound for Image Denoising**

Authors: Yaochen Xie; Zhengyang Wang; Shuiwang Ji

[Paper](https://proceedings.neurips.cc/paper/2020/hash/ea6b2efbdd4255a9f1b3bbc6399b58f4-Abstract.html) | [Official code](https://github.com/divelab/Noise2Same)

**原作训练设定**: 单次带噪观测的自监督去噪

**方法核心**: 优化监督去噪风险的自监督上界，以重建项和掩码一致性项减轻严格 J-invariant 架构要求。

**对当前 baseline 的迁移建议（研究假设）**: 推断：baseline 原样/掩码输入共享权重，构造观测域预测一致性，研究保留残差结构时的噪声泄漏控制。

**限制与适配边界**: 不是一般融合损失的无条件替代；上界和噪声假设要重新核对；不能只对两个输出强制相同而没有非平凡约束。

**已观察到的实现证据**: 官方 TensorFlow 仓库有 models.py、network.py、network_configure.py、notebook；README 给出双前向及损失伪代码。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 移植损失思想较容易，原代码框架与当前 torch 不同。

#### 进一步思考：完整输入与遮挡输入的一致性会不会奖励复制？

**具体失配与动机。** `hsi_up` 若在遮挡之前缓存，遮挡 head 也阻止不了目标噪声直达输出。只让两次输出接近，又可能通过全局平滑得到平凡解。

**候选机制。** 把输入遮挡、基线插值和所有分支封装为同一个函数，盲点前向只能接触保留值；将完整/遮挡预测的一致性放在正确观测域，并与非平凡的留出预测结合。原 Noise2Same 上界的系数、假设与双模态推广需另行推导，未经推导只称掩码一致性正则。

**最小实现与训练。** 两次共享参数前向，去掉未经遮挡的缓存；对留出目标做数值依赖测试，检查归一化统计和子空间基也未读取目标。保持第二次前向预算的普通增强对照，避免双前向本身带来收益。

**最小证伪与不保留条件。** 故意保留未经遮挡 skip、正确移除 skip、全路径移除三组，验证复制噪声现象是否可重复；纯一致性若产生过平滑而无观测预测收益，应减少或删除。

**贡献边界与定位。** P40/P41/P22，候选是条件残差融合网络中跳连导致的目标泄漏及其可验证训练封装。**优先：高；可先作为机制审计。**

<a id="p41"></a>

### P41 | Noise2Void | CVPR 2019

#### 原报告方法、迁移位置与代码依据

**Noise2Void - Learning Denoising From Single Noisy Images**

Authors: Alexander Krull; Tim-Oliver Buchholz; Florian Jug

[Paper](https://arxiv.org/abs/1811.10980) | [Official code](https://github.com/juglab/n2v)

**原作训练设定**: 单次带噪图像的自监督去噪，盲点/掩码训练

**方法核心**: 遮蔽待预测像素，用邻域预测它，以带噪像素为训练目标而不直接复制其噪声。

**对当前 baseline 的迁移建议（研究假设）**: 推断：在 HSI 去噪分支或观测域构造掩码重建，为无干净标签的融合增加抗噪训练。

**限制与适配边界**: 需要合适的噪声独立性/零条件均值假设；HSI 跨波段相关噪声与 MSI 交叉泄漏会破坏简单盲点；原始残差基线若包含目标像素可绕过掩码。

**已观察到的实现证据**: 作者 juglab/n2v 实际包含 n2v/models/n2v_standard.py，代码实现 N2V_DataWrapper 掩码数据；官方 README 引用 CVPR 2019。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 官方实现 TensorFlow/CSBDeep，不能直接当 PyTorch 模块粘贴；当前仓库还包含 N2V2，复现时需区分版本。

#### 进一步思考：遮一个波段的像素，其他波段仍可能暴露噪声

**具体失配与动机。** HSI 同一空间位置的多波段噪声可能相关；一个波段留出，其他波段仍携带目标噪声。空间条带则可能跨长距离相关，单像素盲点不够。

**候选机制。** 按已知/独立估计的相关结构设计空间—光谱组遮挡，而非每个值独立遮挡。比如跨波段共同噪声可遮同位置的相关波段组；扫描条带可留出完整条带段并设保护区。独立 MSI 作为补充条件保留，若传感器存在共享噪声来源则同步处理。遮挡代价和材料信息损失必须同时衡量。

**最小实现与训练。** Dataset 输出 raw observations、valid masks 与 heldout targets；所有输入、跳连、基估计仅来自保留值。训练输出经 A 后评价真正留出观测；测试采用多个遮挡预测集成或经验证的方案，不能无条件改成完整输入就声称严格盲点保证仍成立。

**最小证伪与不保留条件。** 独立、跨波段相关、条带三种噪声，比较单值、同位置全波段、相关组/保护区；检查随相关长度变化的效果与弱材料损失。若过大遮挡依赖 MSI 猜出错误光谱，降低范围并报告不可解场景。

**贡献边界与定位。** P41/P31/P39 以及 AP-BSN/AT-BSN、相关高光谱噪声论文接近。核心应是传感器测量与双输入路径下的组拆分，不是扩大 blind spot。**优先：高。** 延伸候选：利用少量独立暗场/重复扫描估计噪声相关范围；这属于额外校准数据设置，需与纯单次观测设置分开报告。

<a id="p42"></a>

### P42 | DIP | CVPR 2018

#### 原报告方法、迁移位置与代码依据

**Deep Image Prior**

Authors: Dmitry Ulyanov; Andrea Vedaldi; Victor Lempitsky

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf) | [Official code](https://github.com/DmitryUlyanov/deep-image-prior)

**原作训练设定**: 单图像优化，无外部预训练数据的神经网络结构先验

**方法核心**: 随机初始化的生成网络本身带有图像先验，通过观测拟合优化参数进行恢复。

**对当前 baseline 的迁移建议（研究假设）**: 推断：用双观测退化一致性做每场景优化/适配对照；保留 baseline 时称场景自适应，若严格实现 DIP 需改成固定随机输入生成器。

**限制与适配边界**: 原 baseline 条件双输入并非严格 DIP；单场景迭代慢、可能过拟合噪声，早停必须不依赖 HR 真值。

**已观察到的实现证据**: 作者项目页直接链接官方仓库，实际存在 denoising.ipynb、super-resolution.ipynb、flash-no-flash.ipynb 与 models/。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 同题扩展版发表于 IJCV 2020, 128(7):1867–1888, DOI 10.1007/s11263-020-01303-4；同一工作家族不要重复计数。作者 Oxford PDF https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/ulyanov20deep.pdf 。

#### 进一步思考：没有 HR 真值，逐图优化何时停？

**具体失配与动机。** 逐图优化先拟合信号、后拟合噪声只是常见经验，不是每场景可靠规则。训练观测误差单调下降时，最佳 HR 迭代仍可能已经过去。

**候选机制。** 同时看目标移除后的观测预测风险与对输入轻微合法扰动的稳定性；两者不再改善时停止。校准块与 stopping 块分开；划分有相关保护区。即便稳定且留出风险低，共同零空间仍可能错，因此早停只能防过拟合部分观测，不能证明 HR 最优。

**最小实现与训练。** 当前 Net 做条件场景适配作为一种对照；严格 DIP 则另设固定随机输入生成器，不混淆名称。固定检查间隔、连续耐心轮数和总预算，规则只由训练/观测验证确定，不看测试 HR。

**最小证伪与不保留条件。** 在合成测试事后比较停止点与 HR 最佳点的偏差、不同噪声和核错误下的稳定性；HR 最佳点只能作为 oracle 评价。若留出规则系统性挑中谱失真的晚期，换停止依据，不能报告 oracle 结果冒充实际方法。

**贡献边界与定位。** P42/P31/P26，DIP 自监督早停有大量先例；候选是融合双观测、相关噪声与欠定性下的停止诊断。**优先：中高；更像训练机制组件。**

<a id="p43"></a>

### P43 | DPS | ICLR 2023

#### 原报告方法、迁移位置与代码依据

**Diffusion Posterior Sampling for General Noisy Inverse Problems**

Authors: Hyungjin Chung; Jeongsol Kim; Michael Thompson McCann; Marc Louis Klasky; Jong Chul Ye

[Paper](https://arxiv.org/abs/2209.14687) | [Official code](https://github.com/DPS2022/diffusion-posterior-sampling)

**原作训练设定**: 预训练扩散，带噪线性/非线性逆问题

**方法核心**: 以近似测量似然梯度引导扩散后验采样，避免在带噪场景强制硬一致性。

**对当前 baseline 的迁移建议（研究假设）**: 推断：为联合 HSI/MSI 建可微测量似然，比较软/硬数据一致性；进一步可探索 baseline 输出作为迭代初始化。

**限制与适配边界**: 需要适配的先验与传感器噪声模型，采样成本高且可能产生光谱失真，不能当作一个普通注意力模块直接插入。

**已观察到的实现证据**: 官方 PyTorch 仓库存在 sample_condition.py、guided_diffusion/condition_methods.py、guided_diffusion/measurements.py 和 task configs。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 实际会议 ICLR 2023，某些二手列表误标 CVPR 2023，不应沿用。

#### 进一步思考：大的似然梯度有时来自坏观测

**具体失配与动机。** 饱和、坏波段、条带与未知响应会产生大梯度。按独立 Gaussian 似然引导扩散，可能在这些方向过强更新，使先验细节和物理误差相互拉扯。

**候选机制。** 用校准后的异方差/鲁棒观测模型，并将更新限于可信步长；异常测量 mask 来自传感器有效值规则或独立诊断，不由网络任意删掉难例。比较两路似然梯度冲突及响应扰动敏感性，对强冲突区域降低先验引导步长、请求更保守的输出，而非只增大似然权重。

**最小实现与训练。** 先在 Net 后实现少步鲁棒物理修正，检查是否优于 P38；扩散阶段借鉴 `condition_methods.py` 的梯度路径，明确 RGB 先验到 31 波段的适配与外部信息。有限迭代引导不等于已精确采样后验。

**最小证伪与不保留条件。** 饱和、单波段损坏、R/P 偏差，比较普通 Gaussian、固定鲁棒、冲突受限引导。若鲁棒项已解释全部改善，取消额外冲突模块；输出“保守”仍需检测是否仅过平滑。

**贡献边界与定位。** P43/P09/P24/P26，候选是观测异常与先验冲突的具体更新机制。**优先：中低；完整扩散成本高。**

<a id="p44"></a>

### P44 | DDRM | NeurIPS 2022

#### 原报告方法、迁移位置与代码依据

**Denoising Diffusion Restoration Models**

Authors: Bahjat Kawar; Michael Elad; Stefano Ermon; Jiaming Song

[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf) | [Official code](https://github.com/bahjat-kawar/ddrm)

**原作训练设定**: 预训练扩散先验，带噪线性逆问题后验采样

**方法核心**: 在退化算子的谱分解空间中按噪声水平调整扩散恢复，避免特定任务的配对再训练。

**对当前 baseline 的迁移建议（研究假设）**: 推断：理解双传感器联合算子的可观测/不可观测成分，为 baseline 引入噪声感知的一致性修正；全扩散方案属于大改。

**限制与适配边界**: 需预训练生成先验、已知算子和噪声信息；RGB checkpoints 不支持原生 HSI；谱分解与多步采样开销，感知合理不等于光谱准确。

**已观察到的实现证据**: 官方仓库存在 main.py、runners/、functions/、guided_diffusion/，README 提供预训练模型和采样命令，明确 NeurIPS 2022。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 归类为预训练先验迁移，不归类为从当前 HSI/MSI 观测零数据训练。

#### 进一步思考：两路观测的弱方向不等于分别的弱方向

**具体失配与动机。** 单独分析 R 或 PSF，会忽略联合约束中的补充与重复信息；在噪声不平衡时，有些方向虽然数学上可见，实际却很不可靠。降采样还导致频率 aliasing，不能把整个 A 误认为 FFT 下逐频率对角。

**候选机制。** 用白化联合算子在低秩空间分析有效弱方向，把“严格零空间”“噪声下弱可见”“可靠可见”分开，给修正不同上限。第一阶段用小网格显式 SVD 验证；真实尺寸采用矩阵向量积、CG/局部谱近似，保留 aliasing 耦合，不建巨大 HR 矩阵。

**最小实现与训练。** 先以 Net 输出为先验解做 P38 软修正，再比较低秩预条件；不直接声称复现完整 DDRM 后验。固定噪声权重并提供不白化对照，避免把观测单位差异误认作可观测性。

**最小证伪与不保留条件。** 改变 MSI 噪声和 R 的条件数，注入已知弱方向扰动，比较算子诊断与实际恢复。若近似谱忽略 aliasing 后诊断失真，应退回 CG 或小尺寸结论，不推广到真实高倍率。

**贡献边界与定位。** P44/P38/P33/P09，联合算子分析已有基础；贡献需要针对融合与噪声的有效观测方向及可计算机制。**优先：中。**

<a id="p45"></a>

### P45 | CARL | ICLR 2026

#### 原报告方法、迁移位置与代码依据

**CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis**

Authors: Alexander Baumann, Leonardo Ayala, Silvia Seidlitz, Jan Sellner, Alexander Studier-Fischer, Berkin Özdemir, Lena Maier-Hein, Slobodan Ilic

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html) | [Official code](https://github.com/IMSY-DKFZ/CARL)

**原作训练设定**: 自监督光谱表征预训练；下游分类/分割/回归；不是 HSI-MSI 融合重建方法

**方法核心**: 以 self-attention/cross-attention 光谱编码器将不同波长及通道数的 RGB/MSI/HSI 转成可共享表征，采用特征层面的时空光谱自监督。

**对当前 baseline 的迁移建议（研究假设）**: 优先研究替换 hsi_head 的光谱编码与两路共享表征；结合不同 R 的模拟 MSI 做跨传感器一致性预训练，再保留现有 tail 重建。

**限制与适配边界**: 当前 baseline 固定31/3通道；需额外提供真实波长，且分类/分割特征的传感器不变性不必然保留精确光谱幅值。需以观测一致性防止丢失光谱细节。

**已观察到的实现证据**: 官方 proceedings 明确链接作者仓库；已打开仓库，实际存在 carl/、main_ssl.py、main_cls.py、main_seg.py、configs/，提供 SSL checkpoint 和训练入口。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 2026 正式会议记录已核实。不是仅 arXiv。未执行代码；不能将其下游指标当成融合证据。

#### 进一步思考：传感器不变特征，可能丢掉重建需要的绝对幅值

**具体失配与动机。** 分类希望不同相机看到同一种材料时特征接近，融合重建却需要准确的谱幅及响应差异。把两路所有特征强制不变，可能把辐射信息作为“相机因素”删除。

**候选机制。** 共享分支保留材料结构，另设幅值/响应条件旁路保留观测单位、真实波长与通道增益。跨传感器一致性仅作用于指定共享部分，重建旁路必须能还原原单位的 R/D 观测。SRF 编码与波长编码共同描述通道，中心波长相同但响应宽度不同的通道不能视为相同。

**最小实现与训练。** 第一版固定 31/3 通道，`hsi_head/msi_head` 增加小条件 MLP 与幅值 token，保持现有 tail；再考虑变长输入编码和可查询输出。比较全部不变、部分不变、有幅值旁路及普通 concat，外部预训练与从头训练分开。

**最小证伪与不保留条件。** 材料相同但照度/SRF 不同、材料不同但 MSI 近色两类，分别检查共享相似性与原单位重建。若特征看起来更聚类而谱幅误差增加，不能宣称融合表征更好。

**贡献边界与定位。** P45/P01/P29，候选是分析任务的不变性与重建任务的辐射保真冲突。**优先：中高。** 延伸候选：对缺失波段区分“没有采集”与“测得零值”，用 valid-mask 与响应元数据条件化，避免统一编码器把二者混淆；固定维输入可以先以零填加 mask 实现。

<a id="p46"></a>

### P46 | CESST | AAAI 2024

#### 原报告方法、迁移位置与代码依据

**Hyperspectral Image Reconstruction via Combinatorial Embedding of Cross-Channel Spatio-Spectral Clues**

Authors: Xingxing Yang, Jie Chen, Zaifeng Yang

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28478) | [Official code](https://github.com/AlexYangxx/CESST)

**原作训练设定**: 有监督 RGB→HSI 光谱重建；不是 LR-HSI+HR-MSI 融合

**方法核心**: 分别构建色彩通道的光谱嵌入，组合挖掘跨通道线索，以时空光谱注意力和 spectrum-fusion attention 融合互补信息。

**对当前 baseline 的迁移建议（研究假设）**: 研究 spectrum-fusion attention 替换 Concat+fuse，或为 msi_head 分通道生成光谱假设并以 LR-HSI 锚定；融合后 body 引入时空光谱注意力。

**限制与适配边界**: RGB 色彩通道假设与任意 MSI 响应不等价；要适配 R、波段数和双路输入。仓库公开了模型/损失，但当前目录未见完整训练入口。

**已观察到的实现证据**: AAAI 正式页直接指向作者仓库；已逐层打开 train_code/architecture/，存在 CESST.py、__init__.py，并有 Loss.py 和 utils.py。

**代码核验程度**: Partial model/loss source; full training entry not observed

**补充核验**: 源代码已释放但属于部分训练实现；适合模块学习，不承诺开箱复现整篇。DOI 10.1609/aaai.v38i7.28478。

#### 进一步思考：从 MSI 生成一种光谱，可能掩盖多解歧义

**具体失配与动机。** RGB/MSI 到 HSI 不是一一映射；同样 MSI 可由不同窄谱结构产生。单一谱假设的融合 attention 可能自信地选错，而 LR-HSI 才提供材料锚。

**候选机制。** 在观测低秩基内生成少量系数假设，先检验每个假设的 LR-HSI 重投影，再以有界权重组合。假设之间允许在 MSI 弱约束方向变化，但其低尺度均值须被 LR 支持；若所有假设同样兼容，应标记歧义并采用保守先验，不能假装观测已作唯一选择。

**最小实现与训练。** `msi_head` 或 `fuse` 后增加 K=2/3 个小系数头，基固定；用 `L_obs + 假设的受约束分散项`，分散项只防坍塌且不得奖励无限差异。留出观测选择/组合，不用 HR 的 best-of-K。代码为部分实现，新增训练需自行贯通。

**最小证伪与不保留条件。** 构造 MSI 近同色、LR 能区分与 LR 也不能区分两组；比较单头、等参数宽单头、多假设。若多头坍塌或只是增加容量，撤销歧义处理贡献。

**贡献边界与定位。** P46/P27/P10，多假设重建已有研究；候选是 LR 锚能否有效选择投影歧义下的谱假设。**优先：中。**

<a id="p47"></a>

### P47 | FeatUp | ICLR 2024

#### 原报告方法、迁移位置与代码依据

**FeatUp: A Model-Agnostic Framework for Features at Any Resolution**

Authors: Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, William T. Freeman

[Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c5601d99ed028448f29d1dae2e4a926d-Abstract-Conference.html) | [Official code](https://github.com/mhamilton723/FeatUp)

**原作训练设定**: 自监督高分辨率特征上采样；语义特征任务，不是光谱图像重建

**方法核心**: 使用多视图一致性恢复特征空间分辨率，提供逐图 implicit upsampler 与通用 joint bilateral upsampler。

**对当前 baseline 的迁移建议（研究假设）**: 研究用HR-MSI指导LR-HSI特征上采样，替换 f_hsi 的固定插值；多视图变换前后上下采样一致性作为辅助自监督信号。

**限制与适配边界**: RGB语义特征上的边缘引导不保证光谱保真；需改为HSI/MSI特征与物理观测损失。仓库仍标记自训JBU入口Coming Soon，不宣称所有训练路径完整。

**已观察到的实现证据**: 已打开 official repo，featup/ 源文件、hubconf.py、train_implicit_upsampler.py 路径及预训练JBU可用；ICLR proceedings 正式页面核实。

**代码核验程度**: Implicit trainer/JBU inference observed; self-training JBU marked Coming Soon

**补充核验**: 可学习现有隐式训练/推理代码，通用JBU自行训练完整性有公开限制；JBU使用自定义CUDA内核。已读 arXiv HTML 方法3节验证多视图一致性；ICLR完整PDF因16.9MB超过抓取限制，HTML可访问。

#### 进一步思考：MSI 边缘不一定是 HSI 的材料边缘

**具体失配与动机。** MSI 的阴影/颜色边缘可能无对应材料变化；反过来，不同材料经 R 投影近色时，HSI 有边界而 MSI 没有。完全由 MSI 引导上采样会复制前者、漏掉后者。

**候选机制。** 将上采样核拆成几何距离项、MSI 结构项与 LR-HSI 谱相容项；对双方冲突的邻域保留原局部插值，限制跨边界传播。谱相容证据只来自粗网格，不把它包装成未知 HR 边界真值。通过多视图/裁剪的一致性训练核，同时用完整物理损失约束输出。

**最小实现与训练。** 仅替换 `f_hsi` 的插值，用 `unfold` 实现小邻域核原型，不要求 JBU 自定义 CUDA；权重归一化，先共享给全部物理波段/相应系数。原 `hsi_up` 保留以隔离特征上采样影响。

**最小证伪与不保留条件。** 纯光照边缘、真实材料边缘、MSI 不可见边缘分别测试，对照双线性、MSI-only、双证据核及等参数卷积。若改善只是把图像更锐化但材料边界错位，不能主张保真上采样。

**贡献边界与定位。** P47/P57/P10/P52，引导滤波已成熟。候选是跨模态边缘不相容时的拒绝传播及训练证据。**优先：中高。**

<a id="p48"></a>

### P48 | MST | CVPR 2022

#### 原报告方法、迁移位置与代码依据

**Mask-Guided Spectral-Wise Transformer for Efficient Hyperspectral Image Reconstruction**

Authors: Yuanhao Cai, Jing Lin, Xiaowan Hu, Haoqian Wang, Xin Yuan, Yulun Zhang, Radu Timofte, Luc Van Gool

[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Mask-Guided_Spectral-Wise_Transformer_for_Efficient_Hyperspectral_Image_Reconstruction_CVPR_2022_paper.html) | [Official code](https://github.com/caiyuanhao1998/MST)

**原作训练设定**: 有监督 CASSI 编码孔径光谱压缩重建

**方法核心**: S-MSA 将光谱特征作为 token，在光谱维计算注意力；物理编码 mask 引导高保真区域。

**对当前 baseline 的迁移建议（研究假设）**: 提取 S-MSA 替换 ChannelAttention 或部分 ResBlock；原来每通道单个权重升级为特征间交互，配合 LR-HSI 光谱约束。

**限制与适配边界**: CASSI 物理 mask 不是 MAE 随机遮挡，也不是融合中的 SRF/PSF；不能直接照搬测量入口、mask-guidance 和训练损失。64隐藏通道不等于31物理波段。

**已观察到的实现证据**: CVF 正式页直接列官方 repo；repo 有 simulation/、real/、requirements.txt 和训练/测试教程，官方新闻明确 MST 代码和权重释放。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: MST 本体是 CVPR 主会；MST++ 是 CVPR Workshops 2022，不能混淆。

#### 进一步思考：物理 mask 引导不能直接换成融合随机 mask

**具体失配与动机。** CASSI 编码孔径的可观测性结构与 R/P 的融合测量不同。64 个隐藏通道的注意力也没有明确波长意义，直接叫“物理光谱注意力”容易混淆机制。

**候选机制。** 在 31 个物理波段或低秩光谱系数的显式分支上建立 attention，输入真实波长/响应列和噪声可靠性；以 R 对各谱方向的支持程度调节跨谱信息传播。不能用“R 某列大”简单等价为该波段可信，因为相关列可能导致光谱组合歧义，应结合 `R E` 的条件结构及 LR 锚。

**最小实现与训练。** 先在 `hsi_head` 前增加一个小物理波段 attention，输出仍为 31 通道，再送入现有 head；MSI 通过响应描述而非 CASSI mask 注入。比较隐藏通道 attention、物理波段 attention、增加响应/噪声条件三组。

**最小证伪与不保留条件。** SRF 重叠变化、坏波段、波长间距变化，检查注意力与实际可预测谱方向关系。若条件 attention 只记住通道编号，未见响应收益消失，不能主张物理泛化。

**贡献边界与定位。** P48/P45/P60，波长/物理条件 attention 已有相近方向。候选应限定到 CASSI 引导迁移至融合时的测量支持失配。**优先：中。**

<a id="p49"></a>

### P49 | NAFNet | ECCV 2022

#### 原报告方法、迁移位置与代码依据

**Simple Baselines for Image Restoration**

Authors: Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun

[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3043_ECCV_2022_paper.php) | [Official code](https://github.com/megvii-research/NAFNet)

**原作训练设定**: 有监督自然图像去噪/去模糊；NAFNet

**方法核心**: SimpleGate 特征乘法与 simplified channel attention，去除常用非线性激活；残差缩放 beta/gamma 零初始化，构成简洁高效 block。

**对当前 baseline 的迁移建议（研究假设）**: 替换 ResBlock 的 Conv-GELU-Conv+CA 为 NAFBlock；单独比较 SimpleGate、SCA 和残差缩放，适合搭建更强且轻的结构参照。

**限制与适配边界**: 去掉显式激活不意味着整个映射线性：特征相乘依然非线性。原训练有 clean target；移植不会自动成为自监督融合。

**已观察到的实现证据**: ECVA 正式页链接作者仓库；已打开 NAFNet_arch.py，含 SimpleGate、NAFBlock、sca、beta/gamma 和训练文件。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 对当前 LayerNorm+MLP+Sigmoid 注意力很直接的容量/成本对照。

#### 进一步思考：无标签适配的初始大修正可能破坏可靠基线

**具体失配与动机。** 随机初始化修正头可能让输出偏离 `hsi_up`，自监督欠定目标又难以判断哪些偏离合理。盲目变强的门控会快速拟合噪声或算子偏差。

**候选机制。** 用零初始化残差缩放启动适配，修正幅度逐步释放；释放依据为目标移除后的验证收益和两路物理稳定性，而非固定 epoch。物理可信方向与弱约束方向可设置不同上限，保留无噪声/低噪声跳连收益。零初始化本身属于 NAFNet 已有设计，不是创新。

**最小实现与训练。** 先给原 ResBlock/tail 增加可学小尺度，比较固定零启动、普通初始化与验证控制释放；确认初期缩放参数能收到梯度、主干不会长期被冻结。再比较 NAFBlock，避免同时改变激活、归一化与训练机制。

**最小证伪与不保留条件。** 不同噪声、核偏差与训练预算，检查初期稳定性和最终性能。若只是训练更慢但不降低过拟合，或固定 warmup 已取得同效，验证控制不是必要贡献。

**贡献边界与定位。** P49/P58/P19，候选是无 HR 标签适配中残差释放的观测证据。**优先：中高；成本低，独立发表潜力较弱。**

<a id="p50"></a>

### P50 | Restormer | CVPR 2022

#### 原报告方法、迁移位置与代码依据

**Restormer: Efficient Transformer for High-Resolution Image Restoration**

Authors: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang

[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html) | [Official code](https://github.com/swz30/Restormer)

**原作训练设定**: 有监督自然图像去噪/去模糊/去雨

**方法核心**: MDTA 使用沿通道的转置多头注意力，depthwise 卷积补局部信息；GDFN 用乘法门控处理特征，避免空间全局注意力的平方开销。

**对当前 baseline 的迁移建议（研究假设）**: 用一个 MDTA+GDFN block 替换部分 ResBlock，在64通道融合特征上处理；也可迁移 GDFN 到 fuse 后，保留图像残差与物理自监督损失。

**限制与适配边界**: 它学习的是隐藏通道交互，不能自动称为物理波长注意力。架构本身不产生自监督信号，必须改变训练目标。

**已观察到的实现证据**: 已打开 basicsr/models/archs/restormer_arch.py，真实源包含 Attention、FeedForward、TransformerBlock；官方有训练代码及四类任务配置和权重。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 比完整搬运 U 型网络更适合首先做单块迁移。自监督使用需重新训练。

#### 进一步思考：两路特征尺度不同，attention 相关性可能偏向噪声

**具体失配与动机。** HSI 与 MSI 特征的幅值、噪声与动态范围不同；直接拼接后计算隐藏通道相关性，可能偏向高方差噪声特征。完全归一化又可能抹去重建需要的辐射幅值。

**候选机制。** 在 attention 计算路径做传感器噪声尺度归一化，在值/残差路径保留原幅值与增益 token；两路温度/门控由校准噪声和响应条件生成，并限制范围。把相关性用于交互，幅值旁路用于物理重建，区分“相似度标准化”与“输出单位标准化”。

**最小实现与训练。** 只替换两个 ResBlock 为小 MDTA/GDFN block，增加轻量双路尺度参数，不迁整套 U 型网络；先固定校准尺度，再可学条件温度。比较简单 RMS 标准化、幅值旁路与完整条件交互。

**最小证伪与不保留条件。** 改变两传感器 SNR 比和增益，保持真实 X 不变，检查 attention 是否随噪声错误偏移、输出谱幅是否漂移。若简单输入标定足够，额外 attention 机制不保留。

**贡献边界与定位。** P50/P45/P29，候选是跨传感器相似度与辐射幅值的分工。**优先：中；不是把 MDTA 改名为 spectral attention。**

<a id="p51"></a>

### P51 | HyperSIGMA | TPAMI 2025

#### 原报告方法、迁移位置与代码依据

**HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model**

Authors: Di Wang, Meiqi Hu, Yao Jin, Yuchun Miao, Jiaqi Yang, Yichu Xu, Xiaolei Qin, Jiaqi Ma, Lingyu Sun, Chenxing Li, Chuan Fu, Hongruixuan Chen, Chengxi Han, Naoto Yokoya, Jing Zhang, Minqiang Xu, Lin Liu, Lefei Zhang, Chen Wu, Bo Du, Dacheng Tao, Liangpei Zhang

[Paper](https://doi.org/10.1109/TPAMI.2025.3557581) | [Official code](https://github.com/WHU-Sigma/HyperSIGMA)

**原作训练设定**: 高光谱基础模型预训练及多任务适配，包含 HSI 超分/去噪；不是直接 label-free HSI-MSI 融合方案

**方法核心**: 稀疏采样注意力 SSA 缓解高光谱空间与光谱冗余，空间与光谱双表征通过光谱增强模块结合。

**对当前 baseline 的迁移建议（研究假设）**: 学习 spectral enhancement 与 SSA，作为 hsi_head 或 body 的适配分支；先冻结小规模预训练编码器加轻量 adapter，与从头训练对照。

**限制与适配边界**: 模型规模、卫星预训练数据与 CAVE 近景光谱域差距；全量预训练成本高。预训练超分代码并不等于已有自监督融合。

**已观察到的实现证据**: 已打开官方仓库：Pretrain/、ImageSuperResolution/models/、main38_houston.py、train_houston.sh、ImageDenoising/；官方 citation 给出 TPAMI 47(8),6427-6444 与 DOI。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: IEEE 页面抓取正文受限，用作者仓库正式 citation 与 arXiv 核对；代码源实际存在，未运行。

#### 进一步思考：稀疏采样节省计算，却可能漏掉低面积材料

**具体失配与动机。** 高冗余区域易被代表 token 覆盖，弱小材料可能既不显著也不高能；注意力只按语义/幅值抽样时，未被选中的材料没有足够交互。按重建误差抽样又可能优先选噪声。

**候选机制。** 稀疏 token 配额同时覆盖空间、可靠光谱稀有性与观测留出困难区域；稀有性先在噪声归一化的 LR 描述子中估计，困难区需多轮稳定证据。固定一部分均匀覆盖，避免完全自强化选择；记录未选区域的误差，不能只评价被选部分。

**最小实现与训练。** 首先在小型 attention 分支实现 token 选择，冻结或不使用基础模型；若证据成立再接公开编码器与 adapter。选择基于可见输入，缓存按 mask/算子版本隔离，避免把目标留出值送回特征。

**最小证伪与不保留条件。** 改变稀有材料面积、SNR 和 token 预算，比较均匀、能量、纯误差、覆盖加可靠稀有性，报告少数材料误差与实际显存/速度。若收益仅增加 token 总数，不构成机制贡献。

**贡献边界与定位。** P51/P04/P05，稀疏 attention 与长尾采样都有先例；候选是可靠稀有光谱覆盖与算力预算的取舍。**优先：中；预训练成本另算。**

<a id="p52"></a>

### P52 | MambaIRv2 | CVPR 2025

#### 原报告方法、迁移位置与代码依据

**MambaIRv2: Attentive State Space Restoration**

Authors: Hang Guo, Yong Guo, Yaohua Zha, Yulun Zhang, Wenbo Li, Tao Dai, Shu-Tao Xia, Yawei Li

[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html) | [Official code](https://github.com/csguoh/MambaIR)

**原作训练设定**: 有监督图像超分/去噪/JPEG恢复

**方法核心**: Attentive state-space 为扫描引入非因果信息，semantic-guided neighboring 让远距离相似像素交互，缓解单向扫描局限。

**对当前 baseline 的迁移建议（研究假设）**: 将部分 body 替换为轻量v2 blocks，比较局部CNN/通道门控/全局扫描；融合场景可研究以MSI结构指导邻居选择。

**限制与适配边界**: MSI引导邻居是迁移假设，不是原作验证结论；自然图像语义相似不保证光谱相同，错误邻居可能引入光谱污染。依赖特殊算子。

**已观察到的实现证据**: CVF正式页与作者repo一致；实际 basicsr/archs/mambairv2_arch.py、mambairv2light_arch.py 和options/train/mambairv2存在；2024-12-04已发布代码。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 用于结构迁移与低开销对照；不能直接替代自监督损失设计。

#### 进一步思考：语义邻居不能保证光谱邻居

**具体失配与动机。** MSI 相似或语义接近的远处像素可能有不同材料谱。semantic-guided neighboring 若直接迁入，会沿状态传播错误光谱，影响范围大于一个局部卷积。

**候选机制。** 邻居匹配使用 MSI 结构候选与 LR-HSI 谱锚交集；对粗 HSI 混合严重的候选给低可信权重，并保留本地路径。加入响应变化下的邻居稳定性约束：同一训练场景经不同合法 R 生成 MSI，真实有谱锚支持的邻居应稳定，单凭颜色偶然近似的邻居应可被拒绝。

**最小实现与训练。** 先在普通 PyTorch 的稀疏聚合头验证选邻机制，再考虑 v2 state-space block；避免一开始解决 Windows 特殊算子编译。选择和聚合分别做消融，不能把全局模型容量收益归给选邻。

**最小证伪与不保留条件。** MSI 近色材料与重复几何纹理、未见响应、粗 HSI 严重混合，对照颜色/语义邻居、双证据邻居、无远程路径。若错误连边没有下降，停止光谱安全传播主张。

**贡献边界与定位。** P52/P10/P47/P01，候选是 SRF 导致邻居关系漂移的自监督稳定性训练。**优先：中高；Mamba 不是必要起点。**

<a id="p53"></a>

### P53 | MambaIR | ECCV 2024

#### 原报告方法、迁移位置与代码依据

**MambaIR: A Simple Baseline for Image Restoration with State-Space Model**

Authors: Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, Shu-Tao Xia

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2740_ECCV_2024_paper.php) | [Official code](https://github.com/csguoh/MambaIR/tree/mambair)

**原作训练设定**: 有监督图像恢复

**方法核心**: Residual State Space Block 结合选择性状态空间扫描、局部卷积增强与通道注意力，建模长程信息。

**对当前 baseline 的迁移建议（研究假设）**: 替换 body 中少数 ResBlock，保持hsi_head/msi_head/fuse/tail，检查长程空间依赖是否补足平均池化门控；先只做空间扫描。

**限制与适配边界**: 不自动建模物理波长，也不提供label-free学习信号。需要mamba_ssm/causal_conv1d等，官方主要测试Ubuntu，Windows现有环境需重新验证。

**已观察到的实现证据**: ECVA正式页列代码；已打开作者repo的mambair分支和main/basicsr/archs，有mambair_arch.py与训练/测试配置。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 官方2024-08-19声明旧thop复杂度统计漏项，已更新fvcore计数；比较效率应使用修正后实现。

#### 进一步思考：长程扫描会不会传播方向相关噪声？

**具体失配与动机。** 状态空间长程交互有助重复结构，也可能沿扫描方向传播条带。多方向扫描并不自动区分条带与细长真实材料；增大感受野不保证噪声更少。

**候选机制。** 使用独立校准的条带方向/相关长度条件，限制可疑方向上的状态传播；通过另一传感器结构支持保留真实长边缘。对方向门控施加有界与最低覆盖，避免关闭某方向就逃避困难。噪声相关诊断只参与传播规则，主观测约束完整保留。

**最小实现与训练。** 先以行/列轻量序列模块或普通卷积代理做方向传播实验，明确代理不是复现 MambaIR；确认机制后再接 RSSB。记录实际运行时间，不只报可能漏项的 FLOPs。

**最小证伪与不保留条件。** 条带平行/垂直真实边缘、方向变化与无条带场景，对照四向固定、单向、方向条件化。若条件化删掉真实线结构，不能宣称对方向噪声鲁棒。

**贡献边界与定位。** P53/P25/P41，候选是传感器相关噪声与长程扫描方向的冲突。**优先：中；Windows 编译成本后置。**

<a id="p54"></a>

### P54 | HAT | CVPR 2023

#### 原报告方法、迁移位置与代码依据

**Activating More Pixels in Image Super-Resolution Transformer**

Authors: Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, Chao Dong

[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html) | [Official code](https://github.com/XPixelGroup/HAT)

**原作训练设定**: 有监督图像超分辨率；HAT

**方法核心**: 结合通道注意力与窗口空间 self-attention，并使用 overlapping cross-attention 促进跨窗口交互。

**对当前 baseline 的迁移建议（研究假设）**: 在部分 ResBlock 中增加窗口空间交互，保留原通道注意力；用重叠窗口降低局部块边界限制，检查空间细节与 SAM。

**限制与适配边界**: overlapping cross-attention 是跨窗口，不是自动的 HSI-MSI 跨模态注意力；若改为双路 Q/K/V 属于新的迁移设计。全量 HAT 开销高。

**已观察到的实现证据**: CVF 正式页列作者仓库；已打开 hat/archs/，存在 hat_arch.py 等实现；官方 repo 提供训练配置与预训练模型。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 同repo也含 TPAMI 延伸稿；这里仅计 CVPR2023 原作，避免重复计数。

#### 进一步思考：窗口位置与采样相位变化，会造成重建接缝

**具体失配与动机。** 相同场景改变裁剪起点后，窗口划分和 LR 采样起点可能不同；如果 crop 仍按简单整数除法截 LR，两路坐标会错位。重叠 attention 只能缓解交互边界，不能修复错误数据对齐。

**候选机制。** 用全局坐标记录裁剪原点与采样相位，先确保数据一致，再对同一物理区域的不同窗口覆盖结果加重叠一致性。裁剪 offset 不是 s 的整数倍时，显式计算 LR 支撑或限制起点，不默默用 `int(j/s)`。重叠区域一致性与完整 `L_obs` 联用，避免常数图通过一致性。

**最小实现与训练。** 先修改新 Dataset 的坐标元数据与裁剪规则，再引入少量窗口块；同一区域两种 offset 前向产生一致性项，增加同预算双前向增强对照。整图与拼 patch 推理分别评估。

**最小证伪与不保留条件。** 遍历裁剪 offset 和窗口大小，测重叠区域误差与最终谱误差。若收益完全来自修复数据裁剪 bug，应把它作为工程修正，不能包装成网络创新。

**贡献边界与定位。** P54/P14/P07，候选需在正确坐标后仍证明相位条件窗口训练有效。**优先：中高；第一步可能只是必须做的正确性修复。**

<a id="p55"></a>

### P55 | Uformer | CVPR 2022

#### 原报告方法、迁移位置与代码依据

**Uformer: A General U-Shaped Transformer for Image Restoration**

Authors: Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, Houqiang Li

[Paper](https://arxiv.org/abs/2106.03106) | [Official code](https://github.com/ZhendongWang6/Uformer)

**原作训练设定**: 有监督自然图像恢复

**方法核心**: 局部增强窗口注意力 LeWin、FFN depthwise 卷积，以及 U 型多尺度编码解码与 skip connections。

**对当前 baseline 的迁移建议（研究假设）**: 将当前始终64×64的 body 改为浅层多尺度结构，或只换少量 LeWin blocks；每个尺度融合 HSI/MSI 特征。

**限制与适配边界**: 整体迁移会同时改变容量、深度、分辨率，难归因；窗口与下采样尺寸需处理，原RGB模型不带光谱约束。

**已观察到的实现证据**: 已打开 official repo，真实 model.py、train/、test/、script/ 存在；README 给正式 CVPR citation 与训练命令。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: CVF 正式链接抓取失败；官方 author repo 明确 CVPR2022 + BibTeX 与 arXiv相符。

#### 进一步思考：多尺度特征融合不等于多尺度物理一致

**具体失配与动机。** U 型下采样若没有抗混叠/网格记录，特征各尺度的 HSI/MSI 关系可能不再对应传感器关系。简单每层融合可能重复注入 MSI 边缘，并在 decoder 放大 aliasing。

**候选机制。** 用已知低通与相位一致的观测金字塔作为各层锚，而不是把任意 stride-conv 输出当作物理尺度。各尺度只约束真实观测的合法进一步退化，不能拿当前输出自合成的细节当作新真值。层间残差负责不同尺度任务，最深层处理粗谱结构，浅层补有观测支持的细节。

**最小实现与训练。** 先做两尺度 body，与当前单尺度同参数量对照；固定抗混叠下采样，辅助头输出各尺度观测预测。每个 D_q 的复合关系先核对，边界 mask 一致。多尺度观测来自同一对数据，不声称增加独立信息。

**最小证伪与不保留条件。** 高频 aliasing、倍率变化、相位变化，对照任意下采样、固定抗混叠、响应一致金字塔。若普通两尺度卷积已有同样收益，多尺度物理训练不是必要贡献。

**贡献边界与定位。** P55/P07/P37/P17，候选是网络金字塔与成像金字塔的尺度失配。**优先：中。**

<a id="p56"></a>

### P56 | LIIF | CVPR 2021

#### 原报告方法、迁移位置与代码依据

**Learning Continuous Image Representation With Local Implicit Image Function**

Authors: Yinbo Chen, Sifei Liu, Xiaolong Wang

[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html) | [Official code](https://github.com/yinboc/liif)

**原作训练设定**: 从已有高分辨率RGB图像合成低分辨率输入的任意尺度重建；作者称自监督SR训练，不能等同于无HR-HSI真值融合

**方法核心**: 查询连续坐标和邻近局部特征，用局部隐式函数预测像素，实现连续/任意分辨率表示。

**对当前 baseline 的迁移建议（研究假设）**: 将 tail 改为接收融合局部特征、坐标与像素cell的解码器，输出31波段修正；在观测分辨率用可微PSF/SRF约束连续重建。

**限制与适配边界**: 原解码目标RGB，不是融合；需31维输出/光谱条件编码，且以连续积分模拟传感器观测，否则点采样不等于模糊降采样。按坐标查询增加实现复杂度。

**已观察到的实现证据**: 已打开官方repo和models目录，有 liif.py、mlp.py、train_liif.py、configs/、测试代码、官方CVPR2021 citation。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 不要将原论文 self-supervised 字样解释成无需任何高分辨率图像，公开训练使用DIV2K HR。

#### 进一步思考：波段序号与真实光谱响应不是同一个坐标

**具体失配与动机。** “第 10 个波段”在不同相机可能对应不同中心波长与宽度。按序号查询的连续谱函数，即使支持变通道数，也可能对物理波长插值错误；MSI 通道更是一个响应积分，而不是三个点波长。

**候选机制。** 定义空间—波长函数 `x(u,v,lambda)`，HSI 波段与 MSI 通道都按其响应在 λ 上积分，再与空间像素覆盖组合。中心波长和带宽是条件，完整已知响应优于只给中心。若实际只有离散 R 与 31 个已知波段，先做离散加权求和，不擅自构造连续真光谱。

**最小实现与训练。** 保留融合特征编码器，仅改 tail 为坐标/波长条件 MLP；先固定 31 个 λ 的查询与离散 R 验证，再测试真实元数据支持的缺波段/变响应。查询分块，避免 HR×31×所有求积点同时驻留显存。

**最小证伪与不保留条件。** 波段子集、非等间距与响应宽度变化，比较序号、中心波长、响应积分。未提供 λ/带宽时只能报告序号实验；若任意缺波段靠训练记忆而非观测支撑，应明确泛化边界。

**贡献边界与定位。** P56/P03/P45/P14，连续谱/响应条件表示已有相近工作。候选是输入响应积分与输出波长查询的不相容。**优先：中；空间 INR 与光谱 INR 应分开验证。**

<a id="p57"></a>

### P57 | CARAFE | ICCV 2019

#### 原报告方法、迁移位置与代码依据

**CARAFE: Content-Aware ReAssembly of FEatures**

Authors: Jiaqi Wang, Kai Chen, Rui Xu, Ziwei Liu, Chen Change Loy, Dahua Lin

[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_CARAFE_Content-Aware_ReAssembly_of_FEatures_ICCV_2019_paper.html) | [Official code](https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/carafe.py)

**原作训练设定**: 内容感知特征上采样；原实验检测/分割/修复为有监督

**方法核心**: 根据当前特征预测空间位置相关重组核，聚合较大邻域内容，替代固定双线性插值。

**对当前 baseline 的迁移建议（研究假设）**: 替换 hsi_head 之后的 F.interpolate(f_hsi)，优先保留原31波段 hsi_up 基线以隔离变量；可尝试MSI引导重组核。

**限制与适配边界**: 官方 operator 只依赖待上采样特征，不自动用MSI指导；大倍率重组与噪声放大需要验证。官方实现需编译扩展，现有Win11环境兼容性未测试。

**已观察到的实现证据**: CVF 论文指向 OpenMMLab mmdetection；实际开源实现迁至 mmcv ops，已打开 carafe.py，包含 CARAFENaiveFunction、CARAFEFunction、CARAFEPack 和扩展加载。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 源已公开；复现配置引用mmcv版本需锁定，不默认现有requirements可直接安装。

#### 进一步思考：逐位置重组核会不会跨材料传播光谱？

**具体失配与动机。** 内容感知核能变锐，但 MSI 引导可能跨越光谱材料边界。若不同物理波段使用完全不同的空间重组权重，还可能改变像素谱形而观感正常。

**候选机制。** 让可解释空间核由 MSI 结构和 LR-HSI 谱相容共同决定，在物理波段或低秩系数域共享空间权重；对边界不确定区限制邻域与偏移。正权重、和为一有助防止幅值放大，但只是凸组合约束，不能保证真实材料不混合。特征域 64 通道共享与物理 31 波段共享的含义必须区分。

**最小实现与训练。** 优先用 `unfold + softmax + 加权求和` 做小核原型、分块上采样，先替换 `f_hsi`；第二步才验证直接图像上采样。官方 mmcv 编译扩展不是机制验证的前提，倍率 8 的中间张量需测显存。

**最小证伪与不保留条件。** MSI 可见/不可见材料边界、噪声与倍率变化，对照双线性、特征重组、谱相容共享重组。若共有核保谱却明显模糊，或自由核更好且无谱伤害，调整约束强度，不预设结论。

**贡献边界与定位。** P57/P47/P14，候选是光谱传播安全性与可学习上采样核的约束。**优先：中高；模块可落地，单独新颖性偏弱。**

<a id="p58"></a>

### P58 | RCAN | ECCV 2018

#### 原报告方法、迁移位置与代码依据

**Image Super-Resolution Using Very Deep Residual Channel Attention Networks**

Authors: Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, Yun Fu

[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf) | [Official code](https://github.com/yulunzhang/RCAN)

**原作训练设定**: 有监督单图像超分辨率

**方法核心**: Residual-in-Residual 多层跳连与全局平均池化通道注意力，将低频信息旁路，使主体学习修正。

**对当前 baseline 的迁移建议（研究假设）**: 作为 ChannelAttention 与三级残差的结构来源参照；可尝试 residual groups 和组级跳连，公平比较深度与参数量。

**限制与适配边界**: 当前 baseline 已经有非常接近的思想；简单再加 RCAN 注意力不构成新贡献。原始版本 Python/PyTorch 老，需要适配。

**已观察到的实现证据**: 作者 repo 实际含 RCAN_TrainCode/、RCAN_TestCode/，ECCV2018 citation 和训练/测试说明；已打开官网 repo。

**代码核验程度**: Actual public source observed; not executed or reproduced

**补充核验**: 优先用于解释、引文和消融对照而非当成新 idea。

#### 进一步思考：深残差不仅学细节，也能改变已可靠的低频谱

**具体失配与动机。** baseline 已经有块内、深层特征、图像三级残差；再加 residual groups 主要改变容量。自监督下的自由残差却可能在观测薄弱方向漂移，深跳连并没有自动限制这种漂移。

**候选机制。** 对输出修正进行尺度/光谱方向的预算分配：可靠低尺度成分只允许小修正，MSI 支持的高频方向允许较大修正，共同弱约束方向幅度受限。低/高通分解只是频率近似，不能称严格值域投影；精确方向应依据联合 A 或低秩近似分析。

**最小实现与训练。** 先在原 `tail` 后增加固定滤波分解和有界尺度，不增加 body 深度；从已知噪声/响应的预算开始，再尝试观测验证调节。原 RCAN、加深 CNN 和固定残差缩放都是必要对照。

**最小证伪与不保留条件。** 干净已知响应、噪声与核偏差，检查约束是否减少谱漂移同时保留真实细节。若只靠降低残差幅度得到过平滑，应降级为正则化对照。

**贡献边界与定位。** P58/P49/P16/P38，残差/频率分工成熟；候选是自监督修正预算与观测可信度的对应。**优先：中；直接 RCAN 移植不是创新。**

<a id="p59"></a>

### P59 | FcaNet | ICCV 2021

#### 原报告方法、迁移位置与代码依据

**FcaNet: Frequency Channel Attention Networks**

Authors: Zequn Qin; Pengyi Zhang; Fei Wu; Xi Li

[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.html) | [Official code](https://github.com/cfzd/FcaNet)

**原作训练设定**: 监督分类/检测；可迁移注意力

**方法核心**: 以空间 DCT 多频分量推广全局平均池化的通道摘要。

**对当前 baseline 的迁移建议（研究假设）**: 替换 ChannelAttention 中 x.mean；保持输出 (B,64,H,W)。

**限制与适配边界**: 这里的 multi-spectral 指空间频率分量，不是 HSI 波长；仍需自监督损失。

**已观察到的实现证据**: model/layer.py；main.py；训练与评估脚本

**代码核验程度**: Actual public source observed; not executed or reproduced

#### 进一步思考：空间 DCT 摘要会同时看到条带与细节

**具体失配与动机。** 当前 GAP 只保留零频空间摘要，FcaNet 扩展空间频率，但特征高频中的噪声也会进入权重预测。PSF/SRF 或倍率改变后，同一空间频率的可靠性不一样。

**候选机制。** 多频摘要分组接收有效 PSF、两路噪声和跨传感器结构支持条件，输出有界频率门控，再生成 64 个隐藏通道权重。高频不被 HSI 看见而被 MSI 支持时仍应保留；两路均弱时只允许受限先验增益。对摘要做噪声尺度归一化，但保留幅值旁路，避免谱辐射信息被全抹掉。

**最小实现与训练。** 只替换 `ChannelAttention` 的 `x.mean`，保留原 LayerNorm/MLP/Sigmoid；固定 DCT 频率、增加频率而无条件、响应条件三组，参数匹配。若需要物理 λ，另在 31 波段分支建模，不能把 DCT 空间频率称作高光谱波长。

**最小证伪与不保留条件。** 相同场景不同核宽、方向条带和 SNR，检查门控响应及 SAM/高频误差。若噪声也得到大门控，或条件收益不超过白化 GAP，撤销多频机制优势。

**贡献边界与定位。** P59/P16/P25/P02，条件多频注意力已有相近研究。候选是双传感器对空间频率支持不对称时的可靠摘要。**优先：中高；适合作为主方向组件。** 延伸候选：只从目标移除后的可见特征计算摘要，比较完整输入摘要是否把目标噪声通过全局权重广播到全部空间位置。

<a id="p60"></a>

### P60 | ECA-Net | CVPR 2020

#### 原报告方法、迁移位置与代码依据

**ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks**

Authors: Qilong Wang; Banggu Wu; Pengfei Zhu; Peihua Li; Wangmeng Zuo; Qinghua Hu

[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf) | [Official code](https://github.com/BangguWu/ECANet)

**原作训练设定**: 监督分类/检测；可迁移注意力

**方法核心**: 用不降维的局部跨通道一维卷积替代两层瓶颈 MLP。

**对当前 baseline 的迁移建议（研究假设）**: 替换 ChannelAttention 的 MLP，比较权重预测容量和参数量。

**限制与适配边界**: 64 个学习通道的邻接关系不等于物理波段邻接；分类优势不能直接推断融合优势。

**已观察到的实现证据**: models/；main.py；light_main.py；模块代码

**代码核验程度**: Actual public source observed; not executed or reproduced

#### 进一步思考：隐藏通道的相邻，不代表物理波段相邻

**具体失配与动机。** ECA 在通道序号上做局部卷积；在 64 个学习通道中，序号并无物理波长意义。这不证明 ECA 错误，网络可学习适合的排列；但不能据此声称它利用了真实相邻波段规律。

**候选机制。** 若目标是物理光谱局部交互，在 31 波段分支用真实波长距离建邻接，再加入波段有效性/噪声可靠性。非等间距/缺失波段时按 λ 距离重新归一化，而非仍用固定序号 kernel。隐藏域 ECA 保留为轻量容量对照，不作物理解释。

**最小实现与训练。** 在 `hsi_head` 前加入小带状光谱混合，输出保持 31；固定等间距版本与原 CA/ECA 同预算比较，真实元数据不足时仅做序号原型。坏波段零填时必须附 mask，不能当成正常暗波段。

**最小证伪与不保留条件。** 波段缺失、间距改变、坏波段相关噪声，检查 λ 邻接比序号邻接是否更稳。对隐藏通道置换敏感只是归纳偏置诊断，不是融合优势证明；若无泛化收益，不保留物理条件模块。

**贡献边界与定位。** P60/P48/P45，波长感知邻接已很常见。**优先：中低；更适合消融组件，独立贡献不足。**

<a id="p61"></a>

### P61 | CBAM | ECCV 2018

#### 原报告方法、迁移位置与代码依据

**CBAM: Convolutional Block Attention Module**

Authors: Sanghyun Woo; Jongchan Park; Joon-Young Lee; In So Kweon

[Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper) | [Official code](https://github.com/Jongchan/attention-module)

**原作训练设定**: 监督分类/检测；可迁移注意力

**方法核心**: 串联通道注意力和空间注意力；摘要使用平均/最大池化。

**对当前 baseline 的迁移建议（研究假设）**: 替换 ResBlock.att，构建平均摘要与局部空间门控对照。

**限制与适配边界**: 已经是成熟模块，直接加入不等于新方法；未证明自监督融合有效。

**已观察到的实现证据**: MODELS/；train_imagenet.py；scripts/

**代码核验程度**: Actual public source observed; not executed or reproduced

#### 进一步思考：同一个空间位置，两传感器可能给出冲突证据

**具体失配与动机。** 通道+空间门控能突出位置，却不知道该处是缺细节、错位、噪声还是饱和。直接用观测残差预测置信度并加权损失，可能通过把难区域权重归零作弊。

**候选机制。** 在校准共同网格上构造结构相容、单路异常与覆盖有效性三种诊断，给空间修正分配有界权重。诊断先冻结/停止梯度，仅调节残差路径；主观测损失不乘自由门控。异常区域可用固定鲁棒损失，须保留最低有效样本覆盖和明确规则。

**最小实现与训练。** `ResBlock.att` 加小空间分支，输入原特征和冻结诊断；比较 CBAM、残差大小门控、诊断门控。先固定 R/P，避免把未知响应的系统误差误解释为空间异常。

**最小证伪与不保留条件。** 噪声、饱和、错位单独/组合，检查全零门控、困难区域覆盖与谱误差。若只让训练 loss 好看却漏修困难区域，停止置信度贡献表述。

**贡献边界与定位。** P61/P04/P26/P02；候选是跨传感器冲突诊断与门控退化解防止。**优先：中高；可作为方向一的轻量实现。**

<a id="p62"></a>

### P62 | SENet | CVPR 2018

#### 原报告方法、迁移位置与代码依据

**Squeeze-and-Excitation Networks**

Authors: Jie Hu; Li Shen; Gang Sun

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) | [Official code](https://github.com/hujie-frank/SENet)

**原作训练设定**: 监督分类；注意力基础

**方法核心**: 全局平均与瓶颈 MLP 生成逐通道门控。

**对当前 baseline 的迁移建议（研究假设）**: 作为当前 SE 类注意力的来源与结构参照；比较 LayerNorm/GELU 的改动。

**限制与适配边界**: 官方为 Caffe；你当前注意力并非完全相同的原始 SE，且并非原创范式。

**已观察到的实现证据**: src/caffe/layers/；include/caffe/layers/；models/

**代码核验程度**: Actual public source observed; not executed or reproduced

#### 进一步思考：全图平均会稀释小面积的关键材料

**具体失配与动机。** 同一隐藏通道可能在大面积背景与稀有材料上有不同作用；全图共享一个权重，难以兼顾两者。高噪声小区域也可能异常激活，因此单纯 max/高分位池化并不可靠。

**候选机制。** 使用少量区域摘要，区域由 MSI 几何结构与 LR-HSI 可靠谱差共同定义，保留全局摘要作回退；各区域权重平滑映回空间，不把 64 隐藏通道直接对应材料。对小区域设置最低支持像素数与噪声检验，避免被极端噪点主导；观察材料面积变化时全局/区域门控的差别。

**最小实现与训练。** 从全局+规则网格区域开始，仅改变 CA 摘要和空间广播；确认区域机制有效后再用可见观测结构分区。比较 GAP、mean+max、固定区域、双证据区域，同参数与同计算预算。

**最小证伪与不保留条件。** 同一材料面积由大变小、SNR 改变及强背景，报告稀有材料谱误差而非只全图指标。若网格区域已同样有效，复杂分区不保留；若小区域噪声被增强，重新定义可靠性。

**贡献边界与定位。** P62/P59/P61/P51，局部/区域注意力成熟。候选是无 HR 标签下稀有材料摘要与噪声激活的区分。**优先：中；作为辅助机制更合适。**

## 将 73 条候选收敛到三个主方向

上面为 62 篇各给出一条主候选，另有 11 条延伸，共 **73 条候选思路**。许多条目针对同一矛盾，应该合并而不是一起堆进网络。下面是三条值得进入下一轮小规模机制验证的方向；先做正确的自监督对照，再决定其中哪一条值得正式实验与进一步新颖性审查。

评分为 1–5 的相对判断，5 为更清楚、更有差异线索、更易在当前工程验证；不是量化实验结果。“新颖性线索”只取 3 分，是因为已有近邻很多，差异需要进一步核验。

| 主方向 | 问题清晰度 | 新颖性线索 | 当前工程可行性 | 可证伪性 | 科学意义 | 平均 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| 噪声相关性与跳连路径联合约束的观测拆分 | 5 | 3 | 4 | 5 | 5 | 4.4 | 优先候选；未实现、未验证 |
| 采样相位与有效响应条件化的内部训练 | 5 | 3 | 4 | 5 | 4 | 4.2 | 优先候选；先固定网格后 INR |
| 独立校准约束下的残差归因与测试适配 | 5 | 3 | 4 | 5 | 5 | 4.4 | 优先候选；先固定 R，仅估低维 P |

### 主方向一：噪声相关性与跳连路径联合约束的观测拆分

**思考链条。** 没有 HR 标签 → 用观测当目标 → 网络输入恰好包含这些观测 → 原 `hsi_up` 可以直接复制目标噪声 → 单点遮挡仍可能从相关波段/邻域泄漏 → 需要由测量噪声相关结构与实际计算路径共同决定拆分，而不是随意扩大 blind spot。

**具体研究问题。** 在双传感器融合中，怎样遮掉目标噪声的所有相关输入，同时保留另一独立传感器对干净结构的帮助，减少遮挡对光谱/空间互补信息的损伤？范围先限定到已知 R/P、已配准、零条件均值、有限范围的空间/光谱相关噪声，不一开始包含全部真实噪声。

**最小机制。** 对留出索引 J，定义同传感器禁止输入集合 G(J)，含相关波段/空间保护区。若两传感器噪声条件独立，另一传感器相同位置可保留；若它们共享潜在噪声，则根据已知前向支持或估计相关结构扩展跨传感器禁止集合。mask 在原始观测处执行，之后才归一化、填补、估基、计算 head 和跳连。

一个容易忽略的模拟区别：`Y_h=D(X)+n_h, Y_m=RX+n_m` 且两噪声独立，与 `Y_h=D(X+n), Y_m=R(X+n)` 共享同一潜在噪声，具有不同泄漏条件。后一种情况下，一个 HSI 留出测量的噪声可能由其 PSF 支撑内的 MSI 观测暴露，需要按支持关系处理；不能把二者混为同一个训练协议。

**可执行目标。** 令 \(\tilde y\) 只包含保留观测，\(\hat X=f_\theta(\tilde y,\text{valid masks})\)，训练目标为

\[
L_{split}=\frac1{|J_h|}\|[D\hat X-Y_h]_{J_h}\|^2+\frac\lambda{|J_m|}\|[R\hat X-Y_m]_{J_m}\|^2.
\]

不同噪声可用固定精度白化。保护区不能参与监督目标与可见填补；损失索引和 valid-mask 要分别保存。可以交替只留出 HSI 或 MSI，避免同时遮掉所有互补结构；真正的跨传感器相关目标则需要联合策略。

**为什么可能成立。** 对固定预测函数，在 `E[n_J | 可见输入, X]=0` 等条件下，预测留出带噪测量的平方误差与预测干净测量的误差相差噪声能量常数。相关噪声使这个条件失效，禁止集合正是为了尽量恢复它。这一说明只涉及留出测量风险，不是 HR-HSI 风险或完整融合可辨识性的证明；单场景参数记忆、遮挡选择依赖目标等问题还需额外控制。

**最小工程路径。** 新建观测拆分 Dataset、输入包装器和训练入口；保留原模型作为对照。先使用固定插补且不改 head，随后再验证 mask embedding。至少一次依赖测试要确认：冻结权重后改留出目标，只会改目标张量，不会改输入、基、cache 或 skip。这个测试查输入路径，不足以排除训练参数记忆，所以还需要整块永久验证留出与独立测试场景。

**关键对照与否定证据。** 原观测训练、单点遮挡、只修 skip、空间—光谱组遮挡、完整机制；全部使用相同网络/预算。若完整机制不比“正确 mask + 简单去噪”更好，或关键增益只是去掉跳连，则缩小贡献。若条带范围太长导致禁止集几乎覆盖整幅图，单次观测方案可能信息不足，应承认需要独立校准/重复测量。

**最近邻与拟议贡献表述。** P22/P31/P40/P41，AP-BSN、AT-BSN，以及相关 HSI 噪声工作。若证据支持，可表述为：“识别双观测残差融合中的目标噪声传播路径，构造结合测量相关结构与计算路径的观测拆分，在保留独立跨传感器结构条件的同时减轻噪声复制。”不要写成首次盲点、首次相关噪声去除或首次自监督融合。

### 主方向二：采样相位与有效响应条件化的内部训练

**思考链条。** 当前观测能构造内部对 → 内部目标本身已经模糊/带噪 → 再缩小改变有效响应与像素覆盖 → 裁剪和采样起点又改变相位 → 训练与部署虽使用同一个网络，实际逆问题不同 → 应让内部任务明确携带尺度、响应与坐标条件，并用原尺度观测适配检查迁移。

**具体研究问题。** 在没有 HR 训练标签时，内部训练的倍率/相位/有效模糊条件不匹配部署，是否是高倍率融合误差的重要来源？先只研究已配准、固定已知 R、已知或小范围未知 P，避免把光流、全盲响应和连续光谱同时加入。

**最小内部对。** 原始 `Y_h` 位于 h×w，`Y_m` 位于 sh×sw。先用已校准的部署 D_s 把 Y_m 转到 h×w，再把 Y_h 以内部 D_q 降到 h/q×w/q：

\[
f_\theta(D_qY_h, D_sY_m; q,P_q,\phi_q)\approx Y_h.
\]

训练输出网格为 h×w，输入倍率为 q；部署使用 `(Y_h,Y_m; s,P_s,phi_s)`。这只是合法内部任务，不保证等同清晰 HR 任务：Y_h 已经模糊，材料/纹理分布也变了，条件化不能恢复其中已丢失的信息。内部标签带噪时用方向一或适用风险修正，不能直接当干净真值。

**候选机制。** 先核对每一级的离散前向算子；在物理坐标下记录 PSF 覆盖、网格间距与相位。构造内部尺度时优先保留足够的空间支撑，q 从 2/4 开始；对合成核可检查复合响应与抗混叠。用小 MLP 把 `(q, 核参数, 相位)` 变成 `fuse/body` 的有界调制，原图观测一致性用于部署适配。

**可执行目标。** `L_internal` 仅用观测 Y_h 作内部目标；加 `L_obs(original_pair)` 时分清训练内观测与永久验证留出。坐标 INR 是第二阶段：先在固定网格证实响应条件机制，再引入 P03 的积分查询；否则无法归因是“表示更强”还是“响应匹配”。

**最小工程路径。** 新 Dataset 使用命名张量和物理坐标；避免二次 8 倍把 8×8 patch 降成 1×1。先在原 Net 加几十/几百维以内的条件编码，不承诺具体参数量，实际实现后统计。保持原图尺度监督参考、固定核内部训练、倍率条件、响应/相位条件四类对照。

**关键对照与否定证据。** 同场景同 R，独立改变 q/s、核宽、相位与裁剪起点，观察失配是否造成稳定误差模式。若修复算子/裁剪后条件化没有额外收益，该方向应回退为正确性修复；若仅内部尺度改善、部署不改善，不能声称解决跨尺度训练。进一步求积应有精度—成本曲线，而不是仅报多查询版本的最高指标。

**最近邻与拟议贡献表述。** P07/P37/P03/P14/P56、IPE、Scale-Equivariant Imaging、U2K。若证据支持，可表述为：“诊断内部融合训练与部署之间的有效响应及采样相位失配，提出响应条件化的内部任务与原尺度适配，降低对固定模拟退化的依赖。”像素积分、任意尺度或内部学习本身都不应列为首次贡献。

### 主方向三：独立校准约束下的残差归因与测试适配

**思考链条。** 自监督只看到观测误差 → 误差可能来自噪声、算子或内容 → 联合更新 X/P/R 会相互补偿 → 最大残差再学习容易追错目标 → 先用与重建网络无关的兼容关系校准，再对剩余残差分配修正，才有希望获得可解释的适配。

**具体研究问题。** 在固定已知 R、小范围未知 PSF/相位和已配准观测下，独立校准能否减少网络对退化偏差的内容补偿？校准后的残差诊断能否优于直接“哪里误差大就修哪里”？这是可先落地的范围；任意未知 R/P/warp 同时估计暂不作为主张。

**校准阶段。** 以一组训练观测块拟合少量 \(\psi\)：

\[
L_{cal}(\psi)=\rho\big(RY_h-D_{P_\psi,s,\phi_\psi}Y_m\big)+\gamma\Omega(\psi).
\]

核非负归一化、质心固定，相位单独表示，增益规范固定。验证块不参与拟合；跨波段相同模糊、线性响应、相容边界与配准是使用该关系的条件。兼容残差的噪声是两路变换后噪声的组合，不能直接用原始单路 σ 解释。若条件不成立，优先诊断或扩展模型，不让融合头硬补偿。

**重建阶段。** 冻结响应训练/适配原 Net，再计算白化 `e_h/e_m`、兼容误差对合法响应扰动的敏感性与跨传感器结构支持。第一版规则可明确：响应扰动即可解释的大误差回到校准检查；单路、与噪声方向一致的误差不给额外高频预算；两路结构支持的剩余误差交给轻量再学习头。规则是经验诊断，需要模拟验证，不能宣称精确因果分解。

**可执行目标。** 第一阶段 `L_cal`，第二阶段固定算子的 `L_obs + 合法内部/拆分目标`；修正头追加一次前向，主损失不被自由置信度乘零。校准更新与网络更新分开日志记录；一旦校准验证退化，恢复上一可接受响应而非继续降低训练拟合。

**关键对照与否定证据。** 固定正确算子作 oracle 参考、固定错误算子、自由联合、低维校准冻结、校准冻结加诊断修正；额外设置同参数/同前向卷积。观察参数精度、验证兼容、HR 谱误差三者是否一致。若降观测误差却核估计更错、HR 更差，说明补偿仍存在；若简单鲁棒损失已有相同收益，贡献收窄。

**最近邻与拟议贡献表述。** P06/P08/P21/P36、HySure、U2K、P02/P04。若证据支持，可表述为：“针对盲适配中退化与内容相互补偿、残差强化噪声的问题，采用独立观测校准约束与校准后残差诊断，限制不可靠修正并提高跨失配稳定性。”不要写成首次 SRF/PSF 估计、首次交替训练或首次残差引导。

## 各类候选的归并与取舍

| 研究矛盾 | 主要条目 | 建议先保留的核心 | 暂不同时加入 |
|---|---|---|---|
| 噪声复制、目标泄漏、风险失真 | P22/P31/P32/P34/P39–P42 | 原始输入处的观测拆分与跳连核对 | 大扩散、全自由协方差、未经推导的 HR 无偏风险 |
| 内部尺度、像素覆盖、相位/窗口不一致 | P03/P07/P12/P14/P35/P37/P54–P56 | 正确离散算子、坐标元数据、响应条件训练 | 一次迁入空间 INR+光谱 INR+U 型 Transformer |
| 响应与内容互补偿、错误残差强化 | P02/P04/P06/P08/P18/P19/P21/P26/P36/P61 | 已知 R 的低维 P 校准，固定主损失的诊断修正 | 任意 R/P/warp 同时自由估计、自由删难例置信度 |
| 稀有材料、基/秩漂移 | P05/P11/P13/P20/P23/P28/P51/P62 | 固定观测基、材料级评价、受限补充方向 | 仅以训练误差增秩、把正交基叫端元 |
| 跨传感器表征、幅值与投影歧义 | P01/P29/P30/P45/P46/P48/P50/P60 | 响应条件与原单位重建旁路 | 全部特征强制不变、直接依赖超大预训练 |
| 错邻居/错边缘传播 | P10/P15/P47/P52/P57 | MSI 结构与 LR 光谱双证据及拒绝路径 | 无证据全局扫描、全部位置强制跨模态对齐 |
| 高频先验与联合弱方向 | P09/P16/P17/P24/P25/P27/P38/P43/P44/P49/P53/P58/P59 | 已知噪声的软一致性强对照、有界修正 | 把视觉清晰当光谱真实、巨大算子 SVD、全量扩散 |

CA 的替换、adapter、低秩 tail、轻量空间门控都可以服务主问题；它们的职责应该能被消融清楚。最有希望的论文论证是一条具体矛盾和一条解决机制，而不是同时带 6 种注意力的网络。

## 本轮新颖性初查：已经存在什么，还没有核验什么

本轮补查是关键词检索与主来源比对，目的为排除显然重复的贡献措辞；**没有完成逐篇全文新颖性审查、引文追踪或完整最新文献覆盖**。原 62 篇的实现证据保留原报告状态，本轮没有把候选机制当作已有开源实现，也没有运行这些仓库。

| 已核对的最近邻 | 主来源与代码 | 它限制了哪一种“首次”表述 | 对本轮候选的启发 |
|---|---|---|---|
| AP-BSN，CVPR 2022 | [CVF 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_AP-BSN_Self-Supervised_Denoising_for_Real-World_Images_via_Asymmetric_PD_and_CVPR_2022_paper.html) · [作者实现](https://github.com/wooseoklee4/AP-BSN) | 相关真实噪声与非对称 pixel-shuffle/盲点训练已经被研究 | 方向一必须说明双传感器、聚合测量及跳连路径的额外区别 |
| Exploring Efficient Asymmetric Blind-Spots，CVPR 2024 | [CVF 论文全文](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Exploring_Efficient_Asymmetric_Blind-Spots_for_Self-Supervised_Denoising_in_Real-World_Scenarios_CVPR_2024_paper.pdf) | 可调 blind-spot 与抑相关噪声/局部结构损失的取舍已有研究 | 不能把扩大遮挡或不同训练测试 mask 本身称为新机制；本轮未成功核对其仓库，不作代码可用承诺 |
| Enhancing Multi-Scale Implicit Learning with Integrated Positional Encoding | [作者 arXiv](https://arxiv.org/abs/2112.05756) | 将 SR 像素理解为局部区域信号聚合并使用 IPE 已有 | 方向二必须限定到内部任务、PSF/相位与部署响应失配；本轮未核验正式 venue/源码 |
| HySure，TGRS 2015 | [作者论文](https://arxiv.org/abs/1411.4005) · [作者 MATLAB 实现](https://github.com/alfaiate/HySure) | 低秩融合、空间/光谱响应估计并非新近才出现 | 方向三需要比传统独立校准多出可证伪的适配与残差处理机制 |
| Degradation Alchemy / U2K | [作者 arXiv](https://arxiv.org/abs/2503.14892) | 未知到已知退化转换及自监督交替优化已有接近方案 | 必须比较校准冻结/响应条件化与退化转换；本轮仍不承诺正式 venue 和代码完整性 |
| Scale-Equivariant Imaging，TCI 2026 | [作者论文链接入口及实现](https://github.com/jscanvic/Scale-Equivariant-Imaging) | 低通逆问题中普通旋转/平移不足、尺度等变的作用已有研究 | P35 与方向二应使用它作最近邻，不能再次声称首次尺度等变 |
| Self-Supervised Super-Resolution for Sentinel-5P Hyperspectral Images | [作者 arXiv](https://arxiv.org/abs/2604.17652) | HSI 自监督 SR 中结合 SURE/EI 与传感器噪声元数据已有 | P32/P34 不能把“HSI+SURE+EI”作为独立新贡献；本轮只核对摘要中的论文与代码声明，未运行实现 |
| Hyperspectral Subspace Identification Using SURE | [作者 arXiv](https://arxiv.org/abs/1606.00219) | 无干净标签的光谱子空间选择已有风险估计路线 | P05/P13/P28 需要比简单按风险挑秩多出特定失配与材料保护证据 |

这些补充用于候选新颖性约束，不改动原报告 62 篇统计，不宣称它们都属于原来指定的九个 venue。它们也提醒：搜索未找到完全一样的关键词，不能证明没人做过相同机制。

## 开始实现前，先回答的六个具体问题

1. **这次研究先固定什么？** 建议固定 R、配准与离散边界；已知 P 的噪声研究和未知 P 的校准研究分开起步。
2. **哪条路径可能暴露目标？** 检查原始观测、归一化、插补、子空间基、teacher/cache、`hsi_up`，不是只看最后 loss。
3. **新增机制纠正什么错误方向？** 必须能指向噪声复制、相位错配、校准补偿、错邻居、罕见成分截断等明确现象。
4. **最简单替代是什么？** 正确算子、白化损失、固定低秩、简单 adapter、软一致性层、同预算卷积都要作为可能足够的解释。
5. **什么结果会推翻想法？** 不仅比较均值，还检查目标失配场景、材料谱、参数估计与观测验证；每篇已经给出不保留条件。
6. **如果有效，贡献怎么写？** 写“在明确条件下识别了什么失配，并用什么训练机制缓解，证据与局限是什么”；不要写“结合若干先进模块，因此有创新”。

建议实际起步顺序是：正确可微算子与纯观测 Net → 跳连/目标依赖审计 → 已知算子的观测拆分小验证 → 内部尺度响应核对 → 单独低维校准验证。这里只给出推进判断，尚未冻结正式实验方案、超参数或论文贡献；最终选择应依据初步证据，再做更严格最近邻核验。

## 交付与核对记录

本文件保留了原报告 P01–P62 的原方法、迁移位置、限制和源码证据，并新增 62 条主候选、11 条延伸及三个收敛方向。每条主候选均有具体失配、候选机制、最小实现、验证/否定条件与贡献边界。原报告及项目 Python 文件未因本次研究改写；本轮没有训练模型或声称取得实验提升。
