# 检索与证据审计

## 范围与计数

截至 2026-10-10，critical-evidence-map；不声称系统综述完整召回率。

本地保留 24 次 web 检索/定点打开工具调用，其中 83 条 search query；前两次初始探索没有完整 JSON 留存，不计入这两个审计数字。完整结果在 web-search-results-final.jsonl（S03–S25）与 web-search-last-check.json（S26），早期 web-search-results.json 是至 S23 的快照，不是最终全集。

公共接口/作者仓库/出版来源另有 53 个请求，见 batch1–9.json 和各自 results。OpenAlex 脚本返回 50 项；TIP/TPAMI 的 Europe PMC 关键词检索返回 37 项原始记录，其中含分类、追踪、单幅/其他成像任务。**这些渠道有重复，不能相加当作独立论文数。**

最终矩阵保留 27 项：A 4、B 8、C 15。另记录投稿/预印本挑战和校准/PAN 邻近资源。27 是人工筛选后的选题证据数量，不是所有有监督顶级融合论文的总数。

53 个取证请求中：48 个返回资料，4 个 IEEE PDF 请求未返回 PDF，1 个 CARL PDF 请求 403。返回资料也包括目录、元数据、代码；48 不等于 48 篇全文。当前有 6 个新取得 PDF，加 3 个此前本地官方 PDF 复用，共 9 个 PDF 文件，其中包含补充材料。文件存在也不等于已完整阅读。

## 检索渠道与用途

| 渠道 | 本次实际用途 | 边界 |
|---|---|---|
| CVF CVPR 2025/2026、ICCV 2025 官方整会目录 | 按 hyperspectral、spectral image、spectral reconstruction 抽取标题再筛选 | 标题措辞不同仍会漏；未按此推断会议没有其它相关论文 |
| ECCV 2026 官方目录/具体 poster 与 PDF | 核验 OneHSI、LGD-Net 主会身份 | 表征/PAN 与直接融合分开 |
| NeurIPS 2026 官方 Downloads | 检查最新接收目录，发现 LESSViT 等邻近工作 | virtual 页面 robots 限制；完整正文/track细节未全部可得 |
| NeurIPS proceedings 2024/2025 | PINO/FeINFN 正式身份与方法范围 | 年份区分，不把旧论文新版本重复计作新工作 |
| IEEE/IEEE SPS、Springer、AAAI | 正式发表、摘要、会议论文 | 部分期刊全文不可访问；未使用付费绕过来源 |
| Crossref | 核对 DOI、题目、卷年、作者及出版字段 | 缺字段即缺字段；候选首条需按标题核对，不自动视为相同论文 |
| OpenAlex | 近期主题发现 | skill 脚本从检索候选池排序，不是整个领域最新文献穷尽 |
| Europe PMC/PubMed 作者摘要 | TIP/TPAMI 检索补查与正式摘要 | 索引覆盖不完整；有些 January 1 日期是卷年/缺省，不当作真实首发 |
| 作者 GitHub | 核验训练损失、代码公开程度、版本差异 | 静态检查不等于运行成功；main 可变，保存了树对象 SHA 与文件哈希 |
| 本地已有 research | 复用官方 PDF，作为文献线索 | 前次 AI 报告不作为论文论断的独立证据 |

专用 nature-academic-search MCP 与 Zotero 工具未挂载，本次用了 skill 所支持的公共 OpenAlex/Crossref 等降级检索。最初受限网络失败，后续公开取证经环境审批执行成功；没有自动审批拒绝遗留。未联系作者、未发送任何外部消息、未下载大型数据集。

## 查询组与补漏

初始探索为 2025/2026 hyperspectral-multispectral supervised fusion 与指定 venues 的组合。

S03–S17 按直接任务、扩散、非均匀退化、未配准、跨传感器/谱坐标、神经算子、不确定性与相对收益分组，多次采用 exact title、DOI、site 限定。确切 query 见保存结果，不用一个长 OR 查询代替逐方向检索。

S18–S19 补查 HSC-Sampling、CAUWT、NUD、SRLF 作者仓库与正式记录；S20–S23 补查 IJCV/ICLR/NeurIPS，尤其发现 NeurIPS 2026 接收目录已可查，避免停留在 2025。S24–S25 核对 CVPR 校准邻近任务；S26 核对 DANet 与 UALNet，辨别 PAN 和 MSI-only 光谱重建。

**补漏测试结果：** OpenAlex 的 50 项不能单独保证找到最新相关 TIP；期刊索引补查补出 HSC、CAUWT 等直接近邻，改变了对“跨波段结构补偿/小波展开”新颖性的判断。对连续物理路线，PINO 的 CAVE 扩展又推翻了只凭 pansharpening 标题排除它的做法。这是本轮实际发生的检索修正。

没有执行可宣称完整高召回的 seed/challenge recall、所有引文网络前后向扩展或 PRISMA 流程。因此本包适用于选题研判，不冒称已经完成系统综述。

## 关键判定的证据锚点

| 判定 | 可核查证据 |
|---|---|
| EMR 有 HR-HSI 监督 | 正文多尺度监督；ResShift_model.py:97–139 |
| EMR 测试 p_MSI 不进入所核对推理 | ResShift_model.py:206–260 数据流 |
| EMR kappa 论文/代码不同 | 正文设置；config/5_step_EMRDiff.yaml:11 |
| SRLF 已用观测重投影选点 | 正文 Eq.(3)–(9)；supp §6 |
| SRLF 真实数据不是原生 HR 真值监督 | 正文 §4.4 再退化与 QNR 说明 |
| UAFL 的 rank 3 只限制预测残差 | 正文 Eq.(1)–(4) 与 K=3 设置 |
| UAFL REAL 有真实不配准但 LR 是合成退化 | 正文 §4.1 |
| CO²IF 使用 GT L1 | 正文 Eq.(17) |
| BHSR/HSC/IR&ArF 有配对参考监督 | 各作者 train/loss 文件 |
| NUD 不能直接归入普通 GT 监督 | main_fus_syn.py:58–61、main_fus_real.py:71–74 |
| DSES 不是常规大规模配对训练 | DOI 与作者正式摘要 |
| OneHSI 表征任务不能直接当融合精度证据 | 正文 downstream 六任务说明 |
| GEWDiff 是单幅 SR | 正文第一/二页任务定义 |
| DifferentialDiff 代码尚未公开可训练实现 | 本次 GitHub 递归 tree 快照只有 README |
| baseline 代码未改变 | baseline-integrity.json 五文件前后哈希 |

## 已取得全文与实际阅读范围

| PDF | 本次范围 | 未覆盖/未完成 |
|---|---|---|
| EMR-Diff | 核心方法、主实验与消融、关键图表目视 | 全引用审计、数值复现 |
| SRLF + supp | 核心方法、主结果、训练/划分/FLOPs，关键表目视 | 所有补充图逐例分析、数值复现 |
| UAFL | 核心方法、设置、主结果与消融、关键图表目视 | 完整实现/全引用核验、数值复现 |
| CO²IF | 核心方法、损失、设置、主结果与消融，关键表目视 | 完整仓库与文本数据溯源、数值复现 |
| PINO | 主要方法线索、CAVE 结果/附录训练/响应内容摘读 | 未标为全文精读 |
| OneHSI | 方法与下游任务部分阅读 | 完整附录与实验逐表复核 |
| LGD-Net | 任务与核心机制部分阅读 | 训练/全部实验复核 |
| GEWDiff | 导论任务边界及机制线索 | 不是直接融合重点精读 |

提取工具有过长表格/图中文字和双栏排序问题；对实际用于结论的数值，已渲染并目视关键页面，保存于 page_evidence/。未读或截断部分不视为完成全文阅读。

## 正式 DeepPaperNote 状态

用户于 2026-10-11 明确确认中文及当前项目 research 工作区保存设置，本轮以显式运行设置执行，没有更改全局偏好或 Obsidian 仓库。

两篇正式运行分别完成身份解析、元数据、官方本地 PDF、全页 canonical raw sections/source manifest、资产、图表决策及视觉检查、synthesis bundle、note plan、Grounding Lint、Final Note Lint、模型质量审阅、模型可读性审阅及 Formal Save。EMR-Diff 主文 11 页，SRLF 主文 10 页，原始抽取均无页数截断；机器节标题分组有误归类，以实际页码和原文内容为准。补充材料另核对 SRLF 第 1–2 页方法及表 5–6，未逐例分析图 11–12。

两篇最终脚本校验和模型审阅均无警告，保存字节与 Final Note Lint 的 SHA-256 一致。共插入 16 项图表，保留 8 项占位；插入前逐项目视，部分自动裁剪进行了一次修复。EMR-Diff 的表 6 候选误配为表 5，已阻止插入并按原始页面转录数值。2 项低优先级表格未作为图片插入，其关键数值仍在正文讨论。

本轮另只读获取 SRLF 作者 deep_select_mamba.py，核对通道映射、批次选点数、回填与注意力方向；快照日期为 2026-10-11，哈希及 URL 在相应 note plan 和 notes-verification.json。两次单篇元数据核对及这项代码获取不计入上一轮 batch1–9 的 53 项公开取证请求，也不把广泛检索截止日期扩写为本轮日期。

正式笔记见 [索引](notes/README.md)，当前校验与保存记录在 deeppapernote_runs/。paper-evidence-cards.md 仍是横向调研证据卡；正式笔记、资料包结构校验与模型数值复现是不同层次，本次没有执行模型复现实验。

## 未解决项

1. DSES、DifferentialDiff、VPG、CAUWT、BHSR、HSC 等部分期刊全文未取得；监督细节和实验范围按矩阵等级保留。
2. 刚接收的 NeurIPS 2026 正式全文/最终版本仍可能更新；只凭目录不判断完整贡献。
3. 本轮方向中的风险控制机制尚未证实新颖；泛用 gate、ridge、Gram 支持、不确定性和增广都有前作。
4. 实验算力、真实传感器响应可获得性与数据许可尚未做部署审查，因为本次未进入实现/下载数据。
5. 最终方法是否成立取决于对照实验；顶级出版身份并不免除其指标/数据/代码问题的核验。

