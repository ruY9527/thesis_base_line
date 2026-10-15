# 解耦方向近期顶刊阅读

日期：2026-10-12。

- [研究报告：四层精读与课题迁移](研究报告_四层精读与课题迁移.md)：7 篇重点论文、1 篇最新跨传感器线索；按研究问题、方法、证据与迁移组织。
- [MossFuse 证据卡](cards/MossFuse_证据精读卡.md)：作者公开全文已读。
- [SpecAware 证据卡](cards/SpecAware_证据精读卡.md)：作者公开全文已读；邻近表征任务。
- [检索与阅读账本](检索与阅读账本.md)：19 个核验候选、访问边界、版本与阅读覆盖。
- `references.bib`：精选论文的可导入参考文献，由已核验 DOI 元数据生成。
- `access/crossref-metadata.json`：原始出版登记记录。
- `access/fetch-*.json`：合法全文获取尝试、解析阅读计划和失败原因。

采用 research-papers 与 nature-paper-card 工作流。两篇全文卡使用 structure-grounded 定位，运行 bundled audit_paper_card.py；自动审计仅检查结构和溯源，不证明科学结论正确。未运行训练或性能复现。
