# 阅读与证据账本

证据截至 2026-10-07。全文指完成本地解析阅读计划与终止标记，不代表额外出版补充材料均已取得。

|论文|深度|解析行数|终止标记|限制|
|---|---|---:|---|---|
|srlf|全文|1062|e11304668abd|FLOPs 的下降不能直接换算成 4090 实际延迟下降。|
|uafl|全文|1044|2593c6d9b523|输出包含原始上采样 HSI 跳连，不能把其整体输出误说成严格 rank-3；SVD 分量也不自动满足非负端元约束。|
|umnet|全文|880|3e70f9ba059e|PAN 单通道结果不能直接证明三通道 MSI 融合有效；误差尺度也不等于已校准的真实后验方差。|
|hipandas|全文|971|3c8dd44958a0|原方法是按场景无监督优化；低秩头可能损伤稀少材料是待验证假设，不能当作已发现的缺陷。|
|unipan|全文|1237|6e6a3ccaf7f6|边际对齐不保证联合光谱关系；论文已指出多维扩展方向，不能声称首次提出。|
|pino|全文|1722|1bb85f387467|文中不同传感器实验不等于已证明任意通道数、任意 SRF 单模型泛化；联合学习辐射场和 SRF 还涉及可辨识性。|
|diff_spectra|全文|1225|a4f3dfe88cb4|任务没有 LR HSI；同样的输入不能由确定性模型区分真实零空间分量。印刷公式存在需核对之处，见正文。|
|carl|全文|1361|e217733d253b|主要验证分类与分割，不能直接推断高保真光谱重建；语义不变性可能损失幅值是研究问题。|
|gsnr|全文|1143|e56a8daee065|全图特征分解可能昂贵；已读主 PDF，另附补充材料未取得，因此不据此作完整理论审计。|
|specaware|全文|2892|709b41987850|主要下游任务是语义解译；全文为作者 arXiv v2，期刊身份另经 DOI 和作者代码核实，未逐字比对期刊最终排版。|
|task_decoupled|全文|875|23dc2b585231|CVPR 2022 main，历史先例|
|lightweight_unaligned|全文|1086|1546e6659bba|arXiv 2026-09-01; 未核实正式发表|
|ambiguity_separation|全文|736|cdd680e21009|2026; ICML FoGen workshop，非 ICML 主会|
|difiv|全文|1545|20dec1dde813|TGRS 2023，历史先例|
|uq_nullspace|全文|933|f16507201b27|arXiv 2023，历史先例|

扩展目录为官方摘要/元数据/作者代码级。CoPGNet、BUGPan 的全文尝试见各 fetch JSON；不从未获取全文推断不存在重叠。

视觉核查：rendered-pages.json 中 9 页均已查看；Diff-Spectra Eq.13 空白在 PDF 图像中确认。SpecAware 读作者 arXiv v2（2026-02-24），正式期刊 DOI 单独核实。

长输出被截断后补读：Hipandas、Diff-Spectra、PINO、UAFL、UniPAN、DIFIV、最新未配准先例、Guo 分解论文、旧 UQ 网络、SpecAware 方法段。Guo 的 126–175 与 SpecAware 的 121–435 已重新完整输出。旧 OpenReview 浏览器验证页面虽有 sentinel，排除为无效正文；fetch-ambiguity_separation.json 已替换为真实 arXiv 正文记录。

未发生独立审稿；未运行 GPU 训练或性能实验。原有 research 目录中的 geometry_condition 与 MossFuse 等 ideation 已读取，并在筛选时视为已存在的本地思路，不能再次宣称本次原创。
