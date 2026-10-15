**定向检索、读取和证据限制记录**

日期：2026-10-07。目的：核实 GEWDiff，搜索会直接削弱迁移主张的近邻；不是全面系统综述。优先使用作者全文、出版商、arXiv、CVF 与作者仓库，二手索引只用于发现，不用于支撑技术结论。

**本轮记录的检索词与目的**

下表列出本轮工具记录中可核对的代表性原样检索词。没有把此表宣称为所有检索调用的逐字导出；前期标题检索和来源跟踪体现在全文缓存与 source-manifest 中。

| Query | 来源 | 过滤 | 试图否定的主张 | 最近命中 | 限制 |
| --- | --- | --- | --- | --- | --- |
| `"Wavelet" "Cross" "Geometric" "WCGE-Net" fusion` | web search | 无时间过滤 | 小波几何互补融合缺少直接近邻 | 初次未命中 WCGE 的有效主记录 | 改用缩写精确词 |
| `"WCGE-Net" fusion` | web search | 无时间过滤 | wavelet＋complementarity＋experts 新颖 | IEEE WCGE-Net | 只取得摘要，全文页脚不算论文 |
| `"Wavelet-Based Cross" fusion 2026` | web search | 2026 在检索词中 | 同名/近邻 wavelet cross-modal 方法遗漏 | 非 HSI 工作多 | 不作为 HSI 直接比较 |
| `"Hyperspectral and multispectral image fusion guided by latent diffusion and spectral expert models"` | web search | 标题精确 | 潜扩散＋高低频＋专家新颖 | ScienceDirect KBS 2026 | 抽象级证据 |
| `"A diffusion model for hyperspectral and multispectral fusion guided by prior knowledge"` | web search | 标题精确 | 谱频先验指导 fusion residual 新颖 | ScienceDirect PG-FDN | 抽象级证据 |
| `"Coupled Diffusion Posterior Sampling" fusion` | web search | 标题片段 | 无 HR-HSI 双先验 diffusion 新颖 | 作者 CDPS-train | 不将仓库 README 等同全文 |
| `"Coupled Diffusion Posterior Sampling" site:pubmed.ncbi.nlm.nih.gov` | web search | PubMed | 发表年份/无监督 claim 不清楚 | PMID 41477791 | 正式卷年份 2026 与作者 TIP2025 标签分别记 |

**主要全文获取与读取**

- GEWDiff：arXiv 2511.07103v1 HTML 与 AAAI 官方 PDF；完整阅读主体、附录/补充、参考文献，并核对末尾标识。
- HSR-Diff：arXiv 2306.12085v1 HTML；完整读取含补充材料。正式 CVF 入口部分调用返回 403，不因此假设该版本与预印本内容完全相同。
- DDIF：作者 liangjiandeng.github.io 的公开 PDF；完整读取。原始辅助工具缺少可用 PDF parser，使用 bundled pypdf 的 layout 提取补齐解析，保留原辅助工具的获取、缓存、读取计划与标识。
- ARGS-Diff：arXiv 2505.11800v1 HTML；完整读取包括补充消融。
- 主要全文末尾标识依次为 `2e50a8e97214`、`140badcce563`、`172dba74541a`、`c6ac3581a8c9`、`88c4203a3fa0`。逐文件位置见 analysis.md。

**补充全文尝试与失败**

| 目标 | helper 输入 | 返回 | 最终可用证据 |
| --- | --- | --- | --- |
| PG-FDN | ScienceDirect S1569843225005709 | HTTP 403，failed | web 获取出版商摘要 |
| Latent diffusion＋spectral experts | ScienceDirect S0950705126000638 | HTTP 403，failed | web 获取出版商摘要 |
| WCGE-Net | IEEE abstract/document/11664275 | helper ok HTML，但仅 56 行导航页脚 | web 获取 IEEE 摘要；本地 HTML 标识不能当作已读全文 |
| CDPS | DOI 10.1109/TIP.2025.3647207 | abstract_only；无 OA、部分 429/202，Crossref 无 PDF | Semantic Scholar/PubMed 摘要及作者仓库 |

辅助工具未设置 OPENALEX_EMAIL/UNPAYWALL_EMAIL，因为没有用户已授权且真实可用的邮箱；未伪造邮箱。没有绕过访问控制或付费下载。故最近近邻存在明确全文覆盖缺口；新颖性置信度保持低。

原始返回记录在 `secondary-fetch-0.log` 至 `secondary-fetch-3.log`。IEEE 自动解析到页脚是人工判定降级的例子，不把所有 `status=ok` 当作完整研究证据。

**作者代码与本地材料**

- 作者仓库 zhu-xlab/GEWDiff 固定 commit `1744732922fc74a61f210b7d5f65de2ad4fe1a45`。逐文件下载、URL、SHA-256 在 source-manifest.json。作者代码未执行。
- 已阅读关键 RWT、EDM、U-Net、数据与 mask/edge、训练/推理实现，发现幅值/尺寸/PCA/后处理约定需要适配；静态结论与运行验证分开。
- 本仓库未找到适用 AGENTS.md；GitNexus list_repos 未列出已索引本仓库，故直接源代码阅读。
- 已阅读本地 model.py、train_cave_M.py、主要 CAVE 数据处理与 utils/metrics 以及已有 baseline/D2 证据材料。
- 新诊断使用 baseline 环境，仅对三张训练图和当前线性退化做读取、投影/算子分析；无训练，无模型权重改写，无最终测试集选择。

**选择历史**

先由论文压缩与几何机制提出迁移候选，再通过近邻查新收缩贡献边界；随后训练场景 oracle 投影检查影响了 k 与旁路建议。最终方案属于结果告知的探索候选，不是预注册确认性设计。后续确认实验需固定独立验证/测试协议，禁止用当前测试目录反复挑参数。
