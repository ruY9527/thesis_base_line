# 候选研究点与筛选历史

筛选在阅读全文与先例检索之后完成，属探索性选择，没有本次模型性能结果参与选择。新颖性信号不是已确认原创。

|编号|候选|最近先例|状态|理由|
|---|---|---|---|---|
|I01|共享/私有模态解耦|MossFuse, local-mossfuse-ideation|rejected|模态解耦已被覆盖，仓库也已有同方向探索。|
|I02|把 CNN 块替换为 Mamba 或注意力|SRLF, CO2IF|rejected|模块替换缺少新的问题、机制与必要性。|
|I03|边缘/小波/几何约束残差扩散|GEWDiff, EMR-Diff, CDPS|rejected|直接拼装已有方向，算力和查新风险都高。|
|I04|低秩丰度残差尾部|UAFL, Hipandas|rejected|低秩细节与丰度残差已有直接先例。|
|I05|连续表示与可学习 SRF 融合|PINO, CLoRF, OTIAS|rejected|PINO 已包含 CAVE 融合，不能以任务迁移制造首创。|
|I06|分位数归一化并保留原始跳连|UniPAN|rejected|原始域监督和跳连已在原文 Algorithm 1。|
|I07|空间光谱不确定性与困难加权|UMNet, CoPGNet|rejected|结构化不确定性与阶段加权已有先例。|
|I08|残差选点与自适应递归|SRLF, BUGPan|rejected|选择性重新学习与可变递归已被覆盖。|
|I09|语义基础模型先验引导融合|CO2IF, HyperSIGMA, CoPGNet|rejected|方向已有强重叠，缺少本任务独有必要机制。|
|I10|联合去噪与超分辨率|Hipandas|rejected|有价值但转为有监督不是独立创新。|
|I11|内禀歧义与估计不确定性分离|ambiguity_separation, uq_nullspace, local-geometry-ideation|rejected|Guo 2026 已直接给出分解、级联与校准；不能保留宽泛新颖性主张。|
|I12|配准置信度门控与 HSI 回退|lightweight_unaligned, UAFL, TTSR|rejected|2026-09 先例直接覆盖可靠性控制细节注入与回退。|
|I13|监督 MSI 注入的实际光谱收益|UAFL, lightweight_unaligned, UMNet, SRLF, DIFIV, TTSR, Task Decoupled CVPR2022|selected|仅保留窄假设：在相同专家/配准条件下，收益标签是否比可靠性分数与真值残差监督更有效；无结果、低查新置信度。|
|I14|完整 SRF 条件的轻量连续保谱重建|PINO, SpecAware, CARL, UniPAN|needs_research|SpecAware 全文已含动态解码与保谱重建，现方案主要为已有方法组合；储备，不作为独立主贡献。|
|I15|两个观测的联合零空间图先验|GSNR, NPN, uq_nullspace, local-geometry-ideation|rejected|零空间学习先验被覆盖；联合观测与 Kronecker 实现本身仍属常规迁移。|

I11 与 I12 是被否定的原始主张，保留否定历史。I13 是在看到 I12 被覆盖后新形成的窄假设，不是给 I12 换名以维持原评分。I14 阅读 SpecAware 全文后降为储备，需要研究，不进入主贡献清单。
