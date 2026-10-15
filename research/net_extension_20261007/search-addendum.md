# 补充检索与阅读审计

运行日期：2026-10-07。主题：CAVE 31-band LR-HSI + RGB/3-band HR-MSI 的监督融合网络；退化、噪声、光照、跨传感器表示、结构化误差与细节注入收益。近两年推荐窗为 2024-10-07 至 2026-10-07。旧文与未正式接收预印本仅作查新先例。以 CVF、AAAI、ICLR/NeurIPS 官方、IEEE/Elsevier 出版记录、作者论文/代码为主；未用二次 AI 文章建立方法结论。

## 起点与检索策略

先读取旧 sources.json、reading-ledger.md 与 research-report.md，复用已有 10 篇核心论文和 prior-art 清单；避免把旧 UAFL、UMNet、CARL 等重新算成新发现。已有主线是 I13 的光谱收益监督注入门控，本次不另推“泛化置信门控”为原创。

本次执行的检索组覆盖：

- HSI–MSI fusion + CVPR/ICCV/AAAI/TIP/TPAMI + 2025/2026，寻找直接任务最近邻。
- degradation-aware / arbitrary degradation / multimodal fusion，寻找可迁移退化约束。
- real noise decoupling + hyperspectral + AAAI 2026，补噪声方向。
- spectral calibration / illumination + hyperspectral + CVPR 2025，区分照明校准与 SRF 校准。
- wavelength-aware / sensor-agnostic / continuous spectral reconstruction，复查 CARL/PINO/SpecAware 对通用前端的覆盖。
- uncertainty / structure-aware / clean prior / hyperspectral fusion，查 CoPGNet、VPG-Net、HSC-Sampling。
- spectral recalibration / anisotropic structure / hyperspectral fusion，发现 ASSR-Net。
- wavelength-dependent blur / band-dependent PSF / hyperspectral fusion，发现 WACV/技术报告及旧 blind fusion 相关工作，仅作检索线索，没有据此形成已全文支持的新贡献。
- 最后精确检索 `"CoPGNet" hyperspectral fusion S1566253525010590`，确认官方索引显示的是 abstract/introduction/section snippets，而非已读完整方法。

上述为检索组审计，不冒充导出的完整逐次 API 请求日志。结果和纳入深度分别保存于 sources-addendum.json；范围有界，不宣称领域零遗漏。

## 纳入与阅读过程

新全文 shortlist 三篇：SIT（CVPR2025）、DRFusion（CVPR2026）、RND（AAAI2026）。三篇覆盖照明、联合退化约束、真实噪声方向，不以名称类似代替任务适配。通过 research-papers 的 canonical fetch_and_parse.py 获取官方 PDF，读取全部 read_plan，核对主文结束 sentinel。三篇 CVPR/AAAI supplementary 未取得，不声称已读。

缓存复核三篇：UAFL、UMNet、CARL。UAFL 首次输出中段截断，因此补读 410–764；其余分段输出覆盖全文件。CARL 包含 appendix。六篇文本行段和 sentinel 均已写 literature-addendum.md 及 JSON，不将旧阅读记录混为本次读取。末次再次读取 UMNet PDF 首页，题名与 AAAI 官方页面一致：UMNet: Uncertainty-guided Memory Network for Hyperspectral Pansharpening；此前补充稿的 Uncertainty-Aware...Hyperspectral Image Pansharpening 是代理转录错误，已修正，不存在此次证实的 PDF/官方题名差异。

视觉核查：UAFL 方法图；UMNet 记忆/两阶段学习图；CARL shared per-band front end 图；SIT 第6页方法图；DRFusion 第3页联合观测图与第5页式(15)；RND 第3页训练/推理图。本次 pypdf 文本提取的机制与 PDF 图一致。DRFusion 式(15)另用 NumPy 验证 Penrose 条件，见文献补充的反例和适用边界。

## Canonical 全文解析与失败回执

helper：`C:/Users/Administrator/.codex/skills/research-papers/scripts/fetch_and_parse.py`。Bundled Python 含 pypdf、pypdfium2、PIL、NumPy；系统 Python 缺少 pypdf。新下载及渲染均放临时 `net_literature_20261007` 缓存，未覆盖旧全文包。官方 PDF 获取后，canonical 默认解析不适用时，使用 bundled pypdf 按页转文本，保留页边界、生成完整读取计划和末尾 sentinel；不能把 PDF URL 获得成功本身当完成阅读。

- **SIT**：canonical `url:2a4c29da201f308d`，官方 CVF PDF，1007 行，END-OF-PAPER:559d12f3b934。
- **DRFusion**：canonical `url:3212353e2eb3d14c`，官方 CVF PDF，1065 行，END-OF-PAPER:b1233abd0c32。
- **RND**：canonical `url:48c20c1d8afb64a6`，AAAI 官方 PDF，788 行，END-OF-PAPER:ad7a95a11468。
- **HSC-Sampling**，DOI 10.1109/TIP.2025.3636676：Unpaywall 无授权 email 跳过；OpenAlex 无 OA；Semantic Scholar HTTP429；Crossref 无可用 PDF 链接；DOI landing HTTP202。官方 IEEE 页面和作者仓库保留为身份/摘要级证据，无 sentinel。
- **VPG-Net**，DOI 10.1109/TIP.2026.3689415：Unpaywall 同上；OpenAlex 无 OA；Semantic Scholar HTTP429 后可回收摘要但无 OA PDF；Crossref 无 helper 可用 PDF；DOI landing HTTP202；IEEE browser 验证页。无 sentinel；Crossref created 时间没有冒充发表日。
- **CoPGNet 完整 URL**：`https://www.sciencedirect.com/science/article/pii/S1566253525010590`，canonical `url:241ff27db960ec13`，direct_url HTTP403，status failed，无 parsed/read_plan/sentinel。
- **CoPGNet abs URL**：`https://www.sciencedirect.com/science/article/abs/pii/S1566253525010590`，canonical `url:4038ba08ea5dbd14`，direct_url HTTP403，status failed，无 parsed/read_plan/sentinel。web open 同样 403。
- **CoPGNet 官方索引结果**：可见 abstract、introduction、highlights、section snippets；条目还写 “View full text”，没有完整方法、公式和实验。已核验不能把这些片段标成全文。DOI 为 10.1016/j.inffus.2025.103997，卷129、article103997、2026-05；DOI 中的2025不替代卷期身份2026。
- **CSMSF TPAMI**：Crossref 出版者元数据可用，IEEE 验证页；仅发现候选，无全文声明。

未虚构 Unpaywall 联系 email；没有采用失效网页、验证页或搜索片段生成伪完整 sentinel。未用付费/身份验证绕过方式取得全文。

## 边界与待闭合事项

近两年已全文推荐之外，ARGS-Diff、DFPIR、ICCV 内容退化解耦、UMFNet、Dual Cameras demosaicing 等只作摘要/元数据候选。它们的算法细节和有效性不能以此次检索替代全文比较。ASSR-Net 是未验证正式主会身份的预印本，仍是公开先例，但不混入“顶会已接收”列表。

CoPGNet 对“结构不确定性 HSI 融合”存在直接重叠，VPG-Net 对“干净生成先验+不确定性”也有重叠。二者全文缺口是查新的实质限制；现阶段结论只能是“存在可实验的狭义收益监督假设”，不能写“检索证明原创”。根任务的模型实现和训练验证由主代理完成；本代理未改模型或训练，也未将文献报告中的潜在迁移写成实验已验证。

