# 领域与方向地图

当前 baseline：固定 31-band LR HSI + 固定 3-band HR MSI → 单目标 HR HSI，有监督 L1。

|研究轴|已经覆盖的方向|本次决策|
|---|---|---|
|空间引导可靠性|UAFL、SSC-HSR、2026-09 轻量先例|不能声称首次匹配置信度或回退|
|实际注入价值|参考 SR/专家门控均构成威胁|仅窄收益监督假设，主探索线|
|计算分配|SRLF、BUGPan|强对照，不作主创新|
|误差与歧义|UMNet、CoPGNet、Guo 2026、旧 UQ 网络|宽泛分解/加权被覆盖|
|结构先验|Hipandas、UAFL、MossFuse、GSNR、NPN|低秩、解耦、图/零空间已有|
|连续与传感器|PINO、CLoRF、CARL、SpecAware、UniPAN|储备，需必要机制差异|
|生成与语义|Diff-Spectra、EMR-Diff、GEWDiff、CDPS、CO²IF、HyperSIGMA|优先借鉴压力测试；不从大模型拼装起步|

不能把 HS–PAN、RGB→HSI、分类/分割与 HSI–MSI 融合当成同一任务。任务迁移的贡献必须由任务特有约束与有效的新机制支持。
