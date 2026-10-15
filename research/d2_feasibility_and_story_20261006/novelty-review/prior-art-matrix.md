# Prior Art Matrix

| 近邻 | 一手来源 | 重叠威胁1–5 | 与候选差别及未解问题 |
|---|---|---|---|
| MossFuse IJCV2026 | https://arxiv.org/html/2412.04802v4 | 5 | 角色分离已有；特定功能取舍尚未验证 |
| CLoRF IJCV2025 | https://link.springer.com/article/10.1007/s11263-025-02540-1 | 4 | 因子组合已有，不能单独新称 |
| SHOTUN TIP2026 | https://pubmed.ncbi.nlm.nih.gov/42202190/ | 4 | 关系保留已有，完整全文覆盖待补 |
| CoMM ICLR2025 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/108030643e640ac050e0ed5e6aace48f-Abstract-Conference.html | 5 | 共享/独有/协同学习已有；代码https://github.com/Duplums/CoMM |
| SynCo 2026预印本 | https://arxiv.org/html/2609.32846v1 | 5 | 交互残差专门训练已有；正式venue与代码未确认，不作顶刊推荐 |
| Equivariant Splitting ICLR2026 | https://proceedings.iclr.cc/paper_files/paper/2026/hash/34a5f638617ce32700e235e50dff9c8a-Abstract-Conference.html | 4 | 自监督风险理论有条件；不能无条件迁移 |
| 本项目RealDATAProcess2 | train_dataloader.py | 5 | 内部再降尺度流程已有，不能当原创核心 |

NQ1接口近邻、NQ2物理组合内部训练、NQ3近期全文与协同机制待进一步核查。主表7篇顶刊的代码与正式时间沿用上一轮矩阵，不重复包装成新核验。
