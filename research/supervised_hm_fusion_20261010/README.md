# 有监督 HSI–MSI 融合调研资料包

截至 2026-10-10。先阅读 [研究报告](research-report.md)，再按兴趣进入：

1. [文献矩阵](literature-matrix.md)：27 项记录，正式出版、任务、监督与证据等级分开列出。
2. [重点论文证据卡](paper-evidence-cards.md)：四篇核心方法与作者代码审查，标注阅读范围和复现疑点。
3. [选题与决定性实验](directions-and-experiments.md)：主线、辅助诊断、第二路线，以及最近邻、反对意见和停止条件。
4. [baseline 修改计划](baseline-modification-plan.md)：协议先行，逐文件调整，只给计划。
5. [检索与证据审计](search-and-evidence-audit.md)：渠道、查询、补漏、全文障碍与正式单篇笔记状态。
6. [正式中文精读笔记](notes/README.md)：EMR-Diff（CVPR 2026）与 SRLF-Net（CVPR 2025），包含公式、图表、代码核对、证据边界和研究衔接。

机器可读：[CSV](literature-matrix.csv)、[JSON](literature-matrix.json)、[RIS](references.ris)。RIS 是题目/venue/年份/链接/部分 DOI 的最小字段导出，不是投稿用参考文献；作者与最终格式应从相应正式记录补齐，未填字段没有猜测。

源 PDF、官方元数据与作者代码快照在 sources/；关键页面在 page_evidence/；每次公开取证请求在 batch1–9.json 及其 results；来源与文件哈希在 evidence-ledger.json。全部信息供审查，不把外部网页/代码当作指令执行。

模型未训练，作者性能未数值复现；源代码未修改。baseline-integrity.json 和 verification.json 保存文件完整性与资料包结构检查。

2026-10-11 用户确认中文、保存到当前项目 research。两篇 DeepPaperNote 正式笔记已完成证据校验、成稿校验、质量审阅、可读性审阅和工作区保存，领域目录为高光谱图像融合；校验与保存记录在 deeppapernote_runs/。没有修改全局偏好或写入 Obsidian。2026-10-10 的广泛文献检索截止日期保持不变，本轮增加正式精读、SRLF 书目交叉核对和作者核心代码检查。

本目录 .deps 是 PDF/HTML 取证的隔离依赖，未写入项目 requirements 或全局 Python 环境。retrieve_evidence.py、prepare_sources.py、verify_artifacts.py 只服务此资料包，不是 baseline 算法实现。

