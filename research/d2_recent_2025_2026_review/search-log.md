# Search Log：实际检索与核验日志

日期：2026-10-06。目的：回应旧文献占主要论据的问题，以 2025–2026 正式顶刊重新审查 D2；顶会单列补充。使用 firecrawl-research-papers 的核验原则，但当前无可调用 Firecrawl 接口，实际使用 Web 检索与出版社/会议/作者机构/作者仓库。沿用 research-novelty-review 的定向审查产物结构。

## 检索范围与保证边界

主范围为 IJCV、TIP、TPAMI，题目与 HSI-MSI 融合、自监督、分解、局部关系、观测失配有关。旧 `papers.json` 是线索来源，不自动视作本轮验证。正式年份同时核对上线/卷期；证据不足不猜日期。此为 targeted-search-only-not-systematic，无召回审计、系统综述覆盖率或独立审稿。

## 本轮实际执行的 28 个查询

| 编号 | 原检索词 | 结果用途 |
|---|---|---|
| Q01 | hyperspectral multispectral fusion 2025 2026 "Transactions on Image Processing" github | 直接任务近期线索 |
| Q02 | hyperspectral fusion 2025 2026 "International Journal of Computer Vision" code | IJCV 线索 |
| Q03 | hyperspectral multispectral 2025 2026 "Pattern Analysis and Machine Intelligence" fusion github | TPAMI 线索 |
| Q04 | multimodal disentanglement synergy 2025 2026 "Transactions on Pattern" | 相邻领域近邻 |
| Q05 | "Coupled Diffusion Posterior Sampling" 2026 TIP 3647207 | CDPS 书目 |
| Q06 | "Self-Expressive High-Order Tensor" 2026 3695389 | SHOTUN 书目 |
| Q07 | "Hyperspectral and Multispectral Image Fusion via Variational Tensor Subspace Decomposition" | 排除旧工作 |
| Q08 | "Building Non-uniform Degradation Model for Position-aware" DOI | NUD/PAF 书目 |
| Q09 | "hyperspectral" "TIP" "2025" "decoupling" github | 分解线索 |
| Q10 | "hyperspectral" "TIP" "2026" "interaction" github | 交互线索 |
| Q11 | "hyperspectral" "IJCV" "2026" github fusion | 最新期刊线索 |
| Q12 | "multimodal" "disentanglement" "TPAMI" "2025" "2026" code | 相邻近邻 |
| Q13 | "3647207" | 编号检索有无关命中，不作支持 |
| Q14 | "3695389" | 编号检索有无关命中，不作支持 |
| Q15 | "Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral" | CDPS 作者摘要 |
| Q16 | "Self-Expressive High-Order Tensor Unrolling Network" paper pdf | 寻找全文 |
| Q17 | "Self-Expressive High-Order Tensor" "2026" "June" | 精确日期未确认，未写入 |
| Q18 | "10.1109/TIP.2025.3551531" publication | IR&ArF 日期 |
| Q19 | "10.1109/TPAMI.2025.3604234" date publication | NUD/PAF 日期 |
| Q20 | "10.1007/s11263-026-02757-8" Published | MossFuse 正式状态 |
| Q21 | "Self-Expressive High-Order Tensor Unrolling Network" site:pubmed.ncbi.nlm.nih.gov | 2026 卷期、作者摘要 |
| Q22 | "IR&ArF" site:pubmed.ncbi.nlm.nih.gov | 2025 卷期、作者摘要 |
| Q23 | "Building Non-Uniform Degradation" site:pubmed.ncbi.nlm.nih.gov | 2025 卷期、作者摘要 |
| Q24 | "Self-Expressive High-Order Tensor Unrolling" pdf -site:researchgate.net -site:eurekamag.com | 合法全文线索 |
| Q25 | site:openaccess.thecvf.com/content/CVPR2026 "Enhancing Unregistered" | UAFL 正式会议确认 |
| Q26 | site:pubmed.ncbi.nlm.nih.gov "IR&ArF" | 书目复核，无关医学同名排除 |
| Q27 | site:pubmed.ncbi.nlm.nih.gov "Building Non-Uniform Degradation" | 书目复核 |
| Q28 | site:pubmed.ncbi.nlm.nih.gov "Self-Expressive High-Order Tensor" | 书目复核 |

## 直接打开/源码核验

出版社：CLoRF、SSC-HSR 页面成功；MossFuse Springer 直读超时，转作者机构正式记录和 arXiv v4。IEEE 入口无法完整读取，书目与作者摘要使用 PubMed；部分打开出现验证码，但检索返回的相同登记页内容可核对。

作者仓库：主表 9 个仓库均可访问。实际查看的源码：MossFuse utils.py；CLoRF train_tv_spatial.py；SHOTUN 非局部展开文件；CDPS spatial_sample.py；NUD/PAF main_fus_syn.py；IR&ArF train.py。SSC-HSR、UAFL、Equivariant Splitting 主要查看 README/入口与正式摘要，未完成所有核心源码精读。

SHOTUN raw URL 首次失败，GitHub blob 页面成功，核对 HOTN/更新阶段/训练损失。NUD/PAF net_fus.py raw 失败，不据此编造网络细节。CLoRF inr/models.py 入口内容过少，机制以训练入口与出版社为准。UAFL 仓库的 CVF 链接点击失败，后续精确检索找到了正式 CVF 摘要与 PDF。

本机 Crossref 网络请求受限，Web Crossref 也失败；采用已有正常网页渠道，没有安装工具、没有要求额外权限。第三方元数据偶有作者字段冲突，以作者登记摘要与作者仓库为准。

## 筛选与可追踪结论

确认核心：7 篇顶刊、2 篇补充顶会。排除/降级：旧论文、正式发表未充分确认的 COrAL、代码未充分确认的 DMIL、2024 先上线的 Low-Rank Transformer 等。没有把检索失败解释为论文不存在。

新颖性判定由近期真实方法与历史边界共同约束。尚未训练模型、没有真实场景新指标；3/5 的判定只支持有限探索。需要继续解决 NQ1/NQ2/NQ3 才能扩大原创性表述。
