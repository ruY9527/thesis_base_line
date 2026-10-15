# 重点论文证据卡与复现审查

这是本次调研的论断—证据记录，不是正式 DeepPaperNote 完成件。页码以本地 PDF 页号为准。阅读范围明确列出；未运行作者代码或训练，不将作者数字写成本地结果。

## EMR-Diff｜CVPR 2026｜A

- 证据：[官方论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_EMR-Diff_Edge-aware_Multimodal_Residual_Diffusion_Model_for_Hyperspectral_Image_Super-resolution_CVPR_2026_paper.html)，[PDF](sources/emrdiff/source.pdf)，[作者代码](https://github.com/luocz55/EMR-Diff)。
- 本轮范围：核心方法 pp.3–5、实验 pp.6–8，导论与结论；目视 pp.3/7 的架构与消融页。参考文献和所有可能的附加版本未逐条审计。
- 机制流：拼接训练 HR-HSI/其前三谱带，观测端拼接 upsample(Y)/MSI → 残差桥接扩散、按 MSI Sobel 强度调制噪声 → 双路径 BAF-UNet 预测并多尺度监督 → 5 步反向采样，取 HSI 部分。
- Eq.(3) 定义观测端减 clean state 的残差，图注存在相反描述；复现按方程与代码一致性核对。
- 边缘噪声以 MSI 平均谱带图为依据，不是每个 HSI 波段的真实高频。其有效性在作者设置中有消融，但跨 SRF/窄带结构的普遍适用性仍需验证。
- 数据：ICVL、Harvard、Chikusei，正文为 Gaussian + ×8 bicubic；与本地 symmetric convolution + decimation 不相同。
- 代码证据：model/ResShift_model.py:97–139 使用 GT pseudo-MSI 与多尺度 L1。测试中的 p_MSI 变量虽由 GT 构造，但所核验采样路径未使用它。
- 配置差异：正文 kappa=1，config/5_step_EMRDiff.yaml 为 2.0、5 steps、31+3 channels。发布 loader 用固定 (500,500) 裁 256 patch，且 gt/(max−min) 不减 min；应先确认数据预处理与论文对应版本。
- 实验事实：Harvard 噪声消融 48.36→49.28；step 表列 3/4/5/10，最优为 5。正文关于其它 step 的表述有不一致，报告时以表和具体运行配置为准。
- 复现优先级：高；目的为监督扩散与细节注入对照，不能仅比较参数量而省略采样开销。

## SRLF-Net｜CVPR 2025｜A

- 证据：[正文](sources/srlf/source.pdf)，[补充材料](sources/srlf_supp/source.pdf)，[作者仓库](https://github.com/YuanyeLiu/SRLF-Net)。
- 本轮范围：核心方法、Table 1–4、真实数据协议，补充材料训练/划分与 FLOPs；目视 pp.4/7 的表与消融。
- 机制流：多尺度 SS2D 初始融合 → 已知 R 下 pseudo-MSI/观测 MSI 的 SSIM 选点 → 空间退化后的 pseudo-LRHSI/观测 LRHSI 的 SAM 选点 → spectral Transformer 重学习与 SS2D 再融合。
- 监督：合成 CAVE/Harvard 用 L1；Gaofen5 用原观测再退化，以 LR-HSI 当缩尺度 GT，原分辨率无 HR 真值。不能把 QNR 结论当 native HR-HSI 真值精度证明。
- 划分：补充材料 CAVE 20/12、Harvard 30/20；Nikon 响应与 8×8 Gaussian。“最后 12 张”仍需要具体 scene 清单才能完全复现。
- 关键事实：Harvard Table 1 47.8954 vs Mog-DCN 47.8910；Table 4 w/o selection 49.3135 vs full 49.3881。需要分清选点、多尺度和整体模型的贡献。
- FLOPs 边界：补充材料称省略 Transformer attention matrix 等操作，不能直接把 81.22G 当完整所有操作成本。
- 它反驳的过宽声明：首次使用重投影一致性选择困难位置；首次融合 Mamba 与谱注意力；首次自适应纠错融合。
- 它没有自动证明的命题：观测误差低的位置具有正确 HR 谱细节；选点机制在未知或错误 R 时仍可靠。
- 2026-10-11 精读补充：原文已有学习退化代理消融，49.3412 vs 固定先验 49.3881；因此上项仅指选择可靠性及错误标定的证据边界，不表示论文未做未知退化实验。只读作者核心实现确认按通道注意力再学习、批次平均选点预算及特征通道映射。详见 [正式笔记](notes/高光谱图像融合/SRLF-Net/SRLF-Net.md)。

## UAFL｜CVPR 2026｜A

- 证据：[正文](sources/uafl/source.pdf)，[正式论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)，[作者仓库](https://github.com/yingkai-zhang/UAFL)。
- 本轮范围：pp.3–8 的方程、方法、设置、结果与消融；目视 pp.3/7。
- 机制流：upsample HSI 的 SVD 得 E 与系数 → RGB 流先验及亚像素 deformable 聚合 → 空间/通道系数交互和调制 → E 乘预测残差并加 HSI-up。
- K=3 指残差基底，而非所有输出的严格 rank 3。E 的 SVD 分量不自动满足物理端元/丰度约束。
- ICVL 留 20 测试、其余训练；REAL 60 对中 10 测试、其余训练；实际相机未配准，但 LR-HSI 是人工退化。
- Table 3 Unmix 单独从 41.26→41.41；加其它模块后 42.05。这说明完整聚合/表示很重要，不能将整网收益全归因于“解混”。
- CFDA Table 4 的 DCNv2 数字为 41.80，附近正文写 41.85；引用应以明确表格值为准，数值复现后再讨论。
- 最近邻约束：SVD/系数域融合、亚像素流与动态 gate 都需要与其区分。窄带/低能量高频残差可能不在前 3 个模式中，但这个失效假设需要专门实验，不能仅凭结构推定已发生。

## CO²IF｜AAAI 2026｜A

- 证据：[正文](sources/co2if/source.pdf)，[官方文章页](https://ojs.aaai.org/index.php/AAAI/article/view/38261)，[作者代码链接](https://github.com/yzkLearning/CO2IF)。
- 本轮范围：核心方法、Eq.(17)、设置、Table 1–4；目视 p.6 结果表。仓库实现本轮未完整审计。
- 机制流：文本特征与谱特征通过 COT 对齐 → MSI 与文本调制 CoMamba 状态 → 输出 HR-HSI → L1 GT 与对比目标训练。
- 论文文本数据由 GPT-4V 生成并人工核对；部署与评价需审查 caption 输入来源、费用与可重复性。不能凭此推定存在 GT 泄漏。
- CAVE 为 22/10，Harvard 为 30/20，Chikusei 16 个非重叠 512 patch 中 12/4。CAVE 划分与 SRLF 不同。
- Table 1 CAVE 50.25 vs 在该协议重新比较的 SRLF 49.19；不是与 SRLF 原文 49.3881 的统一排名。
- 需要的挑战：错误文本、恒定类别文本、MSI 自动文本、去掉人工修订，及文本对窄谱带细节的真实贡献。
- 选题结论：语义信息可作为先验，但本轮没有证据证明它解决校准偏差或可观测性。因此不作为推荐主线。

## BHSR-Net｜TIP 2026｜B

- 出版核验：[DOI](https://doi.org/10.1109/TIP.2026.3714832)，TIP 35:8338–8353。
- 作者代码快照：train.py、models.py、modules.py、loss.py、README 已下载。训练对 pred 与 HrHSI 调用 criterion；loss.py 对多阶段 Z_pred 与 Z 计算加权 L1。
- 确认“具有 HR-HSI 监督”，但尚未取得完整论文；对不同 blur、非均匀 PSF、算子未知程度和真实数据指标不作超出作者摘要的断言。
- D²-Fusion 已同时建模 HSI 退化与 MSI blur，半光滑 Newton 迭代展开。38K 参数是作者所称的紧凑 variant，并非本地复现的完整模型参数数。
- 需要核验：数据与算子是否使用估计值、variant 的训练预算、实际 latency、完全同设置的公平比较。
- 使用角色：主线最近邻，必须补全文，不可把“MSI 也有模糊”作为新增核心贡献。[作者仓库](https://github.com/Dou0405/BHSR-Net)

## HSC-Sampling/MRC｜TIP 2025｜B

- 正式记录：DOI 10.1109/TIP.2025.3636676，2025-12-03，34:7930–7942。
- 证据：[出版社摘要](https://ieeexplore.ieee.org/document/11276863)，[作者仓库](https://github.com/1318133/HSC-Sampling)。
- 本轮代码：train_caveL1_method2.py:162–172 明确用输出与 a1 的 L1；a3/a2 两观测进网络。完整论文仍未取得。
- 摘要明确问题是 MSI 低谱分辨率无法提供更精细谱范围的信息；HSC-Sampling 利用 HSI 各谱带结构互补，M2M 建模多阶段特征。
- 不能由摘要确定其全部退化泛化能力；但已足以反驳“第一次关注 MSI 细节不能覆盖所有 HSI 波段”的主张。
- 模型代码有大量空间/谱卷积与补偿实现，最终与新原型比较需保留容量匹配与相同数据口径。

## NUD/PAF｜TPAMI 2025｜B

- 正式核验：10.1109/TPAMI.2025.3604234，47(12):11464–11482。
- 证据：[作者仓库](https://github.com/jeline0110/NUD_PAF)，本地快照 main_fus_syn.py/main_fus_real.py。
- main_fus_syn.py:58–61 训练损失只对 lrhsi/hrmsi 构造；原 HRHSI 在第 78 起用于评价。real 脚本也使用观测损失。不能因摘要提到“fusion training supervision”就归类成普通配对 HR 真值监督。
- 作者提供真实 hypsen.mat 与仿真 Pavia 数据。没有下载大型数据或执行 60000 次训练。
- 方法边界：非均匀退化与位置感知已有。新方向需与“空间位置条件化”区分，不可只增加 positional encoding。

## PINO 与最新表征工作：迁移边界

[PINO 本地 PDF](sources/pino/source.pdf) 正文 Table 3 与附录明确已有 CAVE HSI/MSI 融合、×4/×8；附录叙述 CAVE 为 20 训练/11 测试，需要具体名单解释未覆盖样本。其 SRF 是可学习表示，不因输出曲线形状像真实响应就证明真实响应被唯一识别；这一判断来自一般反问题可辨识性，而非复现故障。

[OneHSI 本地 PDF](sources/onehsi_pdf/source.pdf) 的 pp.5–10 方法/任务部分已部分读取，明确下游为分类、分割、检测、变化、异常与显著性六类；没有据此推定它已验证 HR-HSI 辐射值融合。SCORE/SEAT/CORE 给出跨波长、尺度与域的设计启发，仍需重新验证重建值的校准。

[DSES-Diff 正式 DOI](https://doi.org/10.1109/TIP.2026.3736733) 与作者摘要表明分别学习谱与空间生成先验，不依赖大量配对数据；可变波段处理已是直接前作。全文下载尚不可得，推理融合过程、失败案例和算子泛化范围不能写成已验证。

[CVPR 2025 Automatic Spectral Calibration](https://openaccess.thecvf.com/content/CVPR2025/html/Du_Automatic_Spectral_Calibration_of_Hyperspectral_Images_Method_Dataset_and_Benchmark_CVPR_2025_paper.html) 是照明/反射率校准及 BJTU-UVA 数据的邻近资源，**不是 MSI SRF 误差估计器**。若未来扩展真实 radiometry，可参考其作者数据；不能混淆照明谱、反射率与相机响应三个未知因素。

