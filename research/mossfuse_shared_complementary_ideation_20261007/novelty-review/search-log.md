# Search Log

## Scope and Date

2026-10-07；网页检索与作者公开仓库；36 个查询，定向反证检索。不是按发表日期过滤的全数据库系统检索；优先 2025–2026，核对实际来源日期。

## Exact Queries

1. "hyperspectral" "shared" "private" fusion 2025 2026

2. "hyperspectral" "Rank-Aware" routing decomposition

3. "hyperspectral fusion" "spectral response" generalization 2025 2026

4. "hyperspectral" "band-dependent" blur fusion

5. "hyperspectral fusion" "spectral response" "generalization" 2025

6. "hyperspectral" "shared private" 2026 fusion

7. "hyperspectral" "self-supervised" "masked" fusion 2025 2026

8. "hyperspectral fusion" "identifiability" "subspace" 2025 2026

9. "Fu" "Limitations" "spectral" CVPR 2025 metameric

10. "hyperspectral fusion" "non-uniform" "degradation" 2025 TPAMI

11. "hyperspectral" "shared" "sensor" fusion CVPR 2026

12. "hyperspectral" "band-dependent" "fusion" 2025 2026 PSF

13. "hyperspectral fusion" "sensor" "generalization" "2026" response

14. "hyperspectral fusion" "leave-one"

15. "hyperspectral fusion" "masked" "self-supervised" multispectral

16. "hyperspectral" "spectral response" "observable" "fusion"

17. "hyperspectral fusion" "commutativity"

18. "hyperspectral fusion" "wavelength-dependent" blur

19. "hyperspectral fusion" "shared" "observability"

20. "hyperspectral fusion" "sensor-invariant" 2025 2026

21. "Physics-guided supervision" "wavelength-dependent" EnMAP 2026

22. "Building Non-Uniform Degradation Model" position-aware hyperspectral fusion

23. "hyperspectral fusion" "spectral response" "channel" "mask" 2025 2026

24. "hyperspectral fusion" "singular values" "spectral response" 2025 2026

25. "hyperspectral fusion" "masking" 2025 2026

26. "spectral response" "hyperspectral fusion" "conditioning"

27. "Equivariant Splitting" "2026" self supervised inverse problems

28. "CoMM" multimodal "ICLR" 2025 code

29. "hyperspectral multispectral fusion" "observability"

30. "hyperspectral fusion" "condition number" "spectral"

31. "hyperspectral fusion" "shared" "commut"

32. "hyperspectral fusion" "sensor" "role" "masked"

33. "MossFuse" "2026" IJCV Springer

34. "hyperspectral fusion" "response-aware" 2025 2026

35. "hyperspectral fusion" "spectral visibility"

36. "hyperspectral fusion" "measurement splitting"

## Search Purposes and Outcomes

Q1–8 发现共享/私有、局部路由和响应/子空间近邻，RASD 是强重叠。Q9 将原先误指的 CVPR 限制研究纠正为 TCI2025 Optics-Aware。Q10–20 排查非均匀退化、留出观测与共同域。Q21–24 取得出版社可检索波长 PSF 与 NUD 摘要。Q25–28 确认 ES ICLR2026、CoMM ICLR2025 官方仓库。Q29–32 搜索 A/B/C 的关键词，并确认 SpecAware 条件编码来源。Q33–36 核对 MossFuse 正式日期及窄主张。若窄查询未命中，不能以此断言不存在近邻。

## Primary Sources Accepted

只用作者论文、出版社/正式会议、PubMed 摘要与作者仓库支持技术判断；未将二手 AI 解读、专利聚合页、搜索噪声当作方法证据。来源见 prior-art-matrix.md。

## Full Text and Code Depth

MossFuse 作者全文及此前固定源码核对；SRLF 原始 PDF、选定代码；RASD 原始可检索摘要及模型/内部降尺度训练代码（外部配置数据身份未核实）；CoMM 官方论文及 README；NUD PubMed 原始摘要及此前代码；ES 官方仓库及此前选读；SpecAware 出版摘要与仓库 README；Optics 作者论文/README；PSF2026 出版可检索正文，代码未核实。

## Access and Date Corrections

部分 ScienceDirect/TandF 直读 403/内部失败，以作者原始可检索文本为限；RASD 未来卷期与在线日期区分；旧论文近期抓取时间不作为 2025/26 发表依据。

## Unresolved Critical Questions

NQ1：近年子空间灵敏度条件融合，尤其实际 LR 基与条件数门控。NQ2：测量拆分+条件功能角色，缺失模态与 operator-conditioned reconstruction。NQ3：共同观测域、算子行空间相容性与波段 PSF。没有召回挑战集和全文系统筛查，保留低置信度。
