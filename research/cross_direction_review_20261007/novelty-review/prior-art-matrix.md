# Prior Art Matrix

威胁评分 1–5：5 为足以覆盖宽泛主张。下列“差异”都是候选假设，未证明效果。

|先例与来源|任务/场所|重叠维度|威胁|剩余差异假设|能否否定我们的主张|
|---|---|---|---:|---|---|
|[UAFL](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)|有监督未配准 HSI–RGB / CVPR 2026|丰度残差、对齐、相似性引导|5|I13 监督两专家注入后的光谱误差差额|若收益信号只是匹配置信度代理则没有必要差异|
|[轻量未配准先例](https://arxiv.org/abs/2609.01060)|真实错配 RGB 引导 / preprint 2026-09|可靠性加权细节、上采样回退|5|I13 同可靠性条件下测实际光谱收益|直接否定 I12；I13 需证明置信度对照不足|
|[UMNet](https://ojs.aaai.org/index.php/AAAI/article/view/37988)|HS–PAN / AAAI 2026|结构化误差尺度、困难区域训练|4|I13 两专家收益差额而非单模型尺度|可能只是 UQ 目标的简单改写|
|[SRLF](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html)|HSI–MSI / CVPR 2025|回投影偏差、选择性计算|4|I13 针对低偏差但负收益的位置|若回投影即可解释收益则主线被否定|
|[DIFIV](https://www.cedric-richard.fr/Articles/wang2023deep.pdf)|光谱变化 HSI–MSI / TGRS 2023|局部变化、鲁棒权重|4|I13 单目标有监督决策|局部内容差异处理不能声称首次|
|[TTSR](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Learning_Texture_Transformer_Network_for_Image_Super-Resolution_CVPR_2020_paper.html)|参考 SR / CVPR 2020|相关性控制纹理迁移|4|收益标签与 HSI 目标的必要性|通用门控已有；Q2 未关闭|
|[Task Decoupled](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.pdf)|参考 SR / CVPR 2022|冻结 SISR、HR−SISR 残差监督、相关性纹理迁移|5|I13 显式误差收益差额，而不是纹理残差标签|若真值残差监督已足够，收益头没有必要性；禁止首次双分支/真值指导的表述|
|[CoPGNet](https://doi.org/10.1016/j.inffus.2025.103997)|外部先验 HSI 融合 / IF 2026|结构化不确定性、先验引导|4|尚不能确定；全文未得|高优先级 Q1，可能直接覆盖收益/先验决策|
|[BUGPan](https://doi.org/10.1016/j.inffus.2025.102938)|pansharpening / IF 2025|不确定性门控、可变递归|5|I13 不是计算量分配|否定 I08 宽泛主张；全文缺口不能略过|
|[PINO](https://proceedings.neurips.cc/paper_files/paper/2025/file/89af8e4eb696738f2c9e589522968a09-Paper-Conference.pdf)|含 CAVE 融合 / NeurIPS 2025|连续表示、SRF 成像、LR 跳连|5|I14 已知完整响应家族的轻量同模型实验|不是首次从 PAN 迁移到 HSI–MSI|
|[SpecAware](https://doi.org/10.1016/j.isprsjprs.2026.02.024)|重建预训练与语义解译 / ISPRS 2026|元数据+内容编码、动态解码、保谱损失|5|I14 完整响应形状与融合约束|“动态解码保谱”本身被覆盖；当前 I14 仅 2/5|
|[CARL](https://proceedings.iclr.cc/paper_files/paper/2026/file/282967f8abaae52a452a97ee961410f3-Paper-Conference.pdf)|语义分析 / ICLR 2026|波长条件、相机无关聚合|5|辐射重建需另验证|仅任务迁移仍可能增量|
|[UniPAN](https://openaccess.thecvf.com/content/ICCV2025/html/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.html)|MS–PAN / ICCV 2025|分布对齐、原始域跳连与损失|5|尚未证明新的必要机制|否定 I06|
|[Guo 等](https://arxiv.org/abs/2605.15050)|线性逆问题 / ICML FoGen workshop 2026|内禀/传播不确定性、级联、校准|5|HSI 专项诊断有用途但不是宽泛新方法|直接否定 I11；不将 workshop 当主会|
|[旧 UQ 零空间](https://arxiv.org/abs/2304.06955)|CT/MRI / 2023|数据一致性加不确定性|5|暂无独立机制|零空间+UQ 已存在|
|[GSNR](https://openaccess.thecvf.com/content/CVPR2026/html/Gualdron-Hurtado_GSNR_Graph_Smooth_Null-Space_Representation_for_Inverse_Problems_CVPR_2026_paper.html) / [NPN](https://proceedings.neurips.cc/paper_files/paper/2025/hash/acb1891d79e617134ed604084ebbc919-Abstract-Conference.html)|逆问题 / CVPR 2026、NeurIPS 2025|低维零空间先验与图结构|5|双观测迁移有工程价值|不能声称首次零空间图先验|

本地先例：research/geometry_condition_* 与 mossfuse_shared_complementary_ideation_20261007 已包含相关思路。不是外部发表证据，但不能把已有本地探索重复计为本次原创。
