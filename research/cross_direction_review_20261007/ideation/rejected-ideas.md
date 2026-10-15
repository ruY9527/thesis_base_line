# 淘汰候选

**I01 共享/私有模态解耦**：模态解耦已被覆盖，仓库也已有同方向探索。 最接近：MossFuse, local-mossfuse-ideation。

**I02 把 CNN 块替换为 Mamba 或注意力**：模块替换缺少新的问题、机制与必要性。 最接近：SRLF, CO2IF。

**I03 边缘/小波/几何约束残差扩散**：直接拼装已有方向，算力和查新风险都高。 最接近：GEWDiff, EMR-Diff, CDPS。

**I04 低秩丰度残差尾部**：低秩细节与丰度残差已有直接先例。 最接近：UAFL, Hipandas。

**I05 连续表示与可学习 SRF 融合**：PINO 已包含 CAVE 融合，不能以任务迁移制造首创。 最接近：PINO, CLoRF, OTIAS。

**I06 分位数归一化并保留原始跳连**：原始域监督和跳连已在原文 Algorithm 1。 最接近：UniPAN。

**I07 空间光谱不确定性与困难加权**：结构化不确定性与阶段加权已有先例。 最接近：UMNet, CoPGNet。

**I08 残差选点与自适应递归**：选择性重新学习与可变递归已被覆盖。 最接近：SRLF, BUGPan。

**I09 语义基础模型先验引导融合**：方向已有强重叠，缺少本任务独有必要机制。 最接近：CO2IF, HyperSIGMA, CoPGNet。

**I10 联合去噪与超分辨率**：有价值但转为有监督不是独立创新。 最接近：Hipandas。

**I11 内禀歧义与估计不确定性分离**：Guo 2026 已直接给出分解、级联与校准；不能保留宽泛新颖性主张。 最接近：ambiguity_separation, uq_nullspace, local-geometry-ideation。

**I12 配准置信度门控与 HSI 回退**：2026-09 先例直接覆盖可靠性控制细节注入与回退。 最接近：lightweight_unaligned, UAFL, TTSR。

**I15 两个观测的联合零空间图先验**：零空间学习先验被覆盖；联合观测与 Kronecker 实现本身仍属常规迁移。 最接近：GSNR, NPN, uq_nullspace, local-geometry-ideation。

I14 未淘汰为工程能力，但没有作为主方法创新保留；需要研究后重新审查。
