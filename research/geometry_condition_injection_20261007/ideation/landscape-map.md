# Landscape Map

## Scope

HSI–MSI融合的空间边界/方向作为重建条件；当前CAVE31/RGB×8监督Net。非全领域系统综述。

## Existing Context Used

当前模型/训练/数据生成；../research-report.md；既有GEWDiff迁移、MossFuse共享互补研究。核心灵敏度思路来自前期候选，不宣称完全独立新发现。

## Subareas and Active Approaches

方向正则、细节注入、图几何、结构门控、小波条件扩散、联合配准、子空间响应投影、物理后验与不确定性。

## Closest Work

SDP-Net/GGAT最危险；DDIF/GEWDiff覆盖条件和小波生成；dTV/SA-RFM覆盖方向结构；Matrix-R/DDSR覆盖投影；HSDiff覆盖metamer增强与校准。来源见../sources.json。

## Gap Inventory

待验证缺口为几何条件对局部光谱变化的支持能力；不是声称已有方法都没有谱约束。局部连续RU灵敏度是否比内容门控/秩/固定投影更有解释力仍未知。

## Assumptions and Negative Evidence

当前理想数据不存在粗尺度相容残差；低分辨率局部基可能遗漏真实HR材料。物理一致性不排除共同零空间细节。公式是常见正则化，并非新理论。

## Disconfirmation Targets

同预算普通门控、经典子空间求解、HR-oracle依赖、自然机制缺失、近邻全文同构。失败时缩小或放弃故事。

## Search and Source Limits

截至2026-10-07的定向检索与自审；GGAT/SDP-Net/SA-RFM全文未完整获得。未有独立评审、召回率、训练提升或SOTA证据。
