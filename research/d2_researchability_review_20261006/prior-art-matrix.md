# Prior Art Matrix｜D2 最近邻与重叠风险

> 定向审查，2026-10-06。威胁分数针对原宽泛主张；不是论文质量。已查看代码页面不等于安装、复现或验证所有配置。

| Work | Venue / year | Closest overlap | Threat rating (1-5) | What still looks new | What could kill the claim |
|---|---|---|---:|---|---|
| [CNMF](https://naotoyokoya.com/assets/pdf/NYokoyaTGRS2012.pdf) | TGRS 2012，50(2),528–537 | 光谱与精细丰度联合重建；已知空间／光谱观测 | 5 | 特定神经分离约束的功能取舍尚不是这篇的主题 | 如果候选仅是光谱×空间系数，核心已存在 |
| [MossFuse](https://arxiv.org/html/2412.04802v4) | IJCV 2026；本次沿用已核实条目、读 v4 方法及公开损失 | 共享／互补分解与聚合，跨私有相似性抑制 | 5 | 可以检验哪一项具体影响哪个功能，但尚无损害证据 | 只移除惩罚或添加既有交互，不足形成新贡献 |
| [FiLM](https://arxiv.org/abs/1709.07871) | AAAI 2018；arXiv 2017 | 条件缩放与偏移 | 5 | 测量定义的功能诊断与约束边界 | 调制本身和门控模块没有新颖性 |
| [SSMNet](https://www.haoshen.site/resources/LiuXinyang_SSMNet_TGRS_2023.pdf) | TGRS 2024，卷62；在线2023 | 空间引导的波段调制和跨波段结合 | 5 | HSI–MSI 双测量下的特定训练问题 | 波段相关空间恢复已有近邻，不能只换成 HSI |
| [CoMM](https://github.com/Duplums/CoMM) | ICLR 2025 | 共享、独有、协同信息表示 | 5 | 融合测量约束、共同零空间与可重复重建取舍 | “首次保留协同”被否定 |
| [FOCAL](https://proceedings.neurips.cc/paper_files/paper/2023/file/93e98ddf39a9beb0a97fbbe56a986c80-Paper-Conference.pdf) | NeurIPS 2023 | 共享／私有分解，包含跨私有正交 | 4 | 时序表征不是融合图像重建；实际影响需独立验证 | 不能声称所有私有分离必然破坏联合功能 |
| [COrAL](https://arxiv.org/html/2602.14983) | arXiv 2026；正式卷期未核对 | 分离与促进联合使用同时存在；受控合成评价 | 5 | R、D 定义的测量分离四组诊断 | “解耦与协同兼容”的核心叙事已覆盖 |
| [DMIL](https://arxiv.org/html/2606.11614) | CVPR 2026 官方索引可检索，PDF直读403；读 arXiv正文 | 样本相关交互组成，显式分解与训练 | 4 | 专用于融合观测与光谱细节的功能检验 | “动态交互”或简单区域门控不是充分差异 |
| [PCGrad](https://arxiv.org/abs/2001.06782) | NeurIPS 2020 | 协调相互冲突的任务梯度 | 4 | 物理定义的任务目标可以与一般优化区别，但须对照 | 若普通梯度协调解释收益，专属训练机制减弱 |
| [Physics-guided supervision](https://www.tandfonline.com/doi/abs/10.1080/19479832.2026.2613383) | 出版社2026条目；本次仅索引内容 | 光谱／丰度构造目标及退化模拟 | 4 | 控制单路观测变化以诊断混合输出响应 | 仅以物理混合合成训练为贡献已有强近邻 |
| [Orthogonal Sequential Fusion](https://ora.ox.ac.uk/objects/uuid%3A5d25fd68-bc13-43d2-b64b-64ba881acbdb) | ISIF FUSION 2025；不是 ICLR 2024 已录用论文 | 正交与互补使用并存 | 4 | 不同任务／算子诊断可留空间 | 不能把“正交能够保留协同”包装为新发现 |

## 代码与阅读深度

| 工作 | 代码证据 | 本次实际阅读深度 |
|---|---|---|
| CNMF | [作者页面有 MATLAB/Python 下载](https://naotoyokoya.com/Publications.html) | 期刊正文、作者下载描述；未下载或迁移旧 Python |
| MossFuse | [仓库](https://github.com/dusongcheng/MossFuse)／[utils.py](https://raw.githubusercontent.com/dusongcheng/MossFuse/main/utils.py) | 前序原文分析与本次实际损失源码；未训练 |
| FiLM | [作者仓库](https://github.com/ethanjperez/film) | 原文摘要与前序代码页面；未移植 |
| SSMNet | [作者仓库](https://github.com/ez4lionky/SSMNet/) | 原文方法开头、卷年、代码页面 |
| CoMM | [作者仓库](https://github.com/Duplums/CoMM) | 原已读条目与当前 README 方法说明 |
| FOCAL | [原文声明代码](https://github.com/tomoyoshki/focal) | 原文 §§1–3 正交设置；本次未直读仓库 |
| COrAL | [可见仓库](https://github.com/Caro-Ci/COrAL) | 原稿 §§3–5 与仓库页面；未运行代码 |
| DMIL | [作者声明地址](https://github.com/GeWu-Lab/DMIL) | arXiv 正文；仓库两次直读失败，代码可用性未独立确认 |
| PCGrad | [可见仓库](https://github.com/tianheyu927/PCGrad) | 原稿与代码页面；未做优化对照 |
| Physics-guided supervision | 未确认 | 出版社搜索索引方法段；全文直读失败，细节重叠仍需核查 |
| Orthogonal Sequential Fusion | 未确认 | Oxford作者稿与正式元数据；原OpenReview是投稿版本，不当作ICLR已录用 |

这些未确认代码的条目用于审查重叠，不作为“已经核实可直接迁入 baseline”的推荐。代码不足不影响它们作为已公开思想对新颖性的约束。

## 明确区分已经否定的主张与尚未解决的问题

- 已否定作为首次主张：光谱×空间分配；首次考虑跨模态协同；首次结合正交与联合使用；普通乘性调制／门控；一般梯度冲突处理。
- 尚未证实而非被文献证明：MossFuse 私有负项会损害当前任务；条件分离比全局弱约束更好；特定物理组合诊断和混合输出训练尚无人做过；上述机制在自然场景提升指标。
- 如果近邻已覆盖相同四组算子构造，调整版需再次降级。当前矩阵不支持强首次性措辞。
