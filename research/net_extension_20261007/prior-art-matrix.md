# Prior Art Matrix — 当前先例矩阵入口

详细逐维比较与威胁评分见 [新增先例矩阵](prior-art-addendum.md)；同领域前序矩阵见 [原先例矩阵](../cross_direction_review_20261007/novelty-review/prior-art-matrix.md)。

|主张|最强先例|当前决定|
|---|---|---|
|带真值参考集学习冻结专家局部权重|EnsIR NeurIPS2024|基本框架已覆盖，方法定位2/5|
|按真实成本选择回归专家|ICML2024 Multi-Expert Deferral（历史）|不能称新原则；加入固定成本尺度对照|
|可信度控制 HSI 细节、回退到HSI|UAFL、2026-09 RGB-HSR预印本|宽泛门控新颖性淘汰|
|误差尺度训练困难位置|UMNet AAAI2026|单模型误差与两动作差额不同，但标准辅助损失不能自动构成新机制|
|回投影挑选重建位置|SRLF CVPR2025|强对照，不能首次|
|互补空间/光谱专家|WCGE-Net JSTARS2026|同任务先例有效，全文未得仍是缺口|
|连续/动态光谱头|PINO、CARL、SpecAware|储备能力；目前不满足独立方法创新证据|
|光谱收益辅助在同专家下的独立增量|上述全部及CoPGNet（未得全文）|窄探索问题3/5；需完整对照与真实实验|
