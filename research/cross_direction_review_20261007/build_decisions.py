"""Record exploratory selection history and a conservative self-review decision."""
from build_report import ROOT, PAPERS, EXTRA, OPPONENTS, write
import json

IDEAS = [
 ('I01','共享/私有模态解耦','representation',['MossFuse','local-mossfuse-ideation'],'rejected','模态解耦已被覆盖，仓库也已有同方向探索。',[4,1,4,4,3]),
 ('I02','把 CNN 块替换为 Mamba 或注意力','architecture',['SRLF','CO2IF'],'rejected','模块替换缺少新的问题、机制与必要性。',[4,1,4,4,2]),
 ('I03','边缘/小波/几何约束残差扩散','generative-prior',['GEWDiff','EMR-Diff','CDPS'],'rejected','直接拼装已有方向，算力和查新风险都高。',[3,1,2,3,3]),
 ('I04','低秩丰度残差尾部','spectral-structure',['UAFL','Hipandas'],'rejected','低秩细节与丰度残差已有直接先例。',[5,1,5,5,3]),
 ('I05','连续表示与可学习 SRF 融合','continuous-operator',['PINO','CLoRF','OTIAS'],'rejected','PINO 已包含 CAVE 融合，不能以任务迁移制造首创。',[4,1,3,4,4]),
 ('I06','分位数归一化并保留原始跳连','distribution-transfer',['UniPAN'],'rejected','原始域监督和跳连已在原文 Algorithm 1。',[5,1,5,5,2]),
 ('I07','空间光谱不确定性与困难加权','uncertainty',['UMNet','CoPGNet'],'rejected','结构化不确定性与阶段加权已有先例。',[5,1,5,5,3]),
 ('I08','残差选点与自适应递归','adaptive-computation',['SRLF','BUGPan'],'rejected','选择性重新学习与可变递归已被覆盖。',[4,1,4,4,3]),
 ('I09','语义基础模型先验引导融合','external-prior',['CO2IF','HyperSIGMA','CoPGNet'],'rejected','方向已有强重叠，缺少本任务独有必要机制。',[3,1,2,3,3]),
 ('I10','联合去噪与超分辨率','joint-restoration',['Hipandas'],'rejected','有价值但转为有监督不是独立创新。',[5,1,4,5,3]),
 ('I11','内禀歧义与估计不确定性分离','uncertainty-decomposition',['ambiguity_separation','uq_nullspace','local-geometry-ideation'],'rejected','Guo 2026 已直接给出分解、级联与校准；不能保留宽泛新颖性主张。',[5,1,3,4,4]),
 ('I12','配准置信度门控与 HSI 回退','robust-guidance',['lightweight_unaligned','UAFL','TTSR'],'rejected','2026-09 先例直接覆盖可靠性控制细节注入与回退。',[5,1,5,5,4]),
 ('I13','监督 MSI 注入的实际光谱收益','decision-objective',['UAFL','lightweight_unaligned','UMNet','SRLF','DIFIV','TTSR','Task Decoupled CVPR2022'],'selected','仅保留窄假设：在相同专家/配准条件下，收益标签是否比可靠性分数与真值残差监督更有效；无结果、低查新置信度。',[5,3,5,5,4]),
 ('I14','完整 SRF 条件的轻量连续保谱重建','sensor-generalization',['PINO','SpecAware','CARL','UniPAN'],'needs_research','SpecAware 全文已含动态解码与保谱重建，现方案主要为已有方法组合；储备，不作为独立主贡献。',[5,2,3,5,4]),
 ('I15','两个观测的联合零空间图先验','inverse-prior',['GSNR','NPN','uq_nullspace','local-geometry-ideation'],'rejected','零空间学习先验被覆盖；联合观测与 Kronecker 实现本身仍属常规迁移。',[4,1,3,4,3]),
]
records=[]
for key,title,kind,priors,status,why,scores in IDEAS:
    selected=key=='I13'
    reserve=key=='I14'
    records.append(dict(
        idea_id=key,title=title,summary=why,
        hypothesis=('同等几何可靠性下，光谱收益监督能够减少负迁移，且超过普通专家门控。' if selected else '完整 SRF 形状比中心波长/FWHM 在未见响应家族上提供必要重建信息。' if reserve else title+'能提供不同于所列先例的必要机制；当前证据不支持这一宽泛主张。'),
        contribution_type=kind,source_basis=priors,
        assumptions=(['存在足量高可靠但负收益的引导区域','冻结/分折专家标签避免训练误差标签偏差','推理不依赖 HR 标签'] if selected else ['已知响应函数可获取','严格场景与传感器留出','CAVE 采样谱不能自动代表连续真实谱'] if reserve else ['与原始方法的任务、损失和机制需公平比较']),
        novelty_questions=(['CoPGNet 是否已直接监督先验/引导带来的误差收益？','参考超分辨率和通用专家门控是否已有功能等价的收益教师？','普通 g* 标签或端到端 gate 是否足以达到相同收益？'] if selected else ['完整响应条件是否只是 PINO 与 SpecAware 的常规组合？','何种非平凡失效需要新的必要机制？'] if reserve else ['是否存在与所列最强先例的实质差异？']),
        minimum_validation=('先测两专家的负迁移与 oracle 上限，再比较同专家的多种门控；严格验证场景。' if selected else '同预算完整 SRF vs 波长/FWHM vs 普通 SRF 增强，独立场景×传感器家族留出。' if reserve else '在继续前逐维证明不同于最强先例；当前淘汰。'),
        disconfirming_evidence=('高可靠负收益区域不存在；简单 gate 同样有效。' if selected else 'PINO、SpecAware 已覆盖主要结构；完整响应条件没有额外价值。' if reserve else '已找到直接方法先例，详见先例矩阵。'),
        kill_criteria=('oracle 上限很小；普通门控持平；只来自容量/合成扰动；干净场景持续退化。' if selected else '普通随机 SRF 增强或元数据条件已足够；没有可重复的响应形状失败机制。' if reserve else '宽泛主张已被覆盖，不继续以此为主创新。'),
        closest_work=priors,differentiation=('决策训练标签从匹配/误差尺度改为两专家的监督光谱误差差额；其必要性尚待证实。' if selected else '用完整响应轮廓和已知成像积分检验幅值保真；现有构件差异不足。' if reserve else '尚无足够实质差异。'),
        scores=dict(zip(['clarity','novelty_signal','feasibility','testability','significance'],scores)),
        risk=('routine MoE; incomplete prior-art coverage; limited scene count' if selected else 'routine combination and synthetic sensor generalization only' if reserve else 'direct prior-art overlap'),
        estimated_effort=('low-to-medium CNN pilot on one 4090; runtime unmeasured' if selected else 'medium; variable-channel data and response metadata needed' if reserve else 'not scheduled'),
        status=status,rationale=why,
        blocking_questions=(['Q1','Q2','Q3'] if selected else ['Q4'] if reserve else [])))
write('ideation/idea-scores.json',dict(topic='有监督 CAVE HSI–MSI 融合的近两年跨方向迁移',scoring_scale='1-5',ideas=records))
bank='# 候选研究点与筛选历史\n\n筛选在阅读全文与先例检索之后完成，属探索性选择，没有本次模型性能结果参与选择。新颖性信号不是已确认原创。\n\n|编号|候选|最近先例|状态|理由|\n|---|---|---|---|---|\n'
for key,title,kind,priors,status,why,scores in IDEAS:
    bank+=f"|{key}|{title}|{', '.join(priors)}|{status}|{why}|\n"
bank+='\nI11 与 I12 是被否定的原始主张，保留否定历史。I13 是在看到 I12 被覆盖后新形成的窄假设，不是给 I12 换名以维持原评分。I14 阅读 SpecAware 全文后降为储备，需要研究，不进入主贡献清单。\n'
write('ideation/idea-bank.md',bank)
write('ideation/landscape-map.md',r'''# 领域与方向地图

当前 baseline：固定 31-band LR HSI + 固定 3-band HR MSI → 单目标 HR HSI，有监督 L1。

|研究轴|已经覆盖的方向|本次决策|
|---|---|---|
|空间引导可靠性|UAFL、SSC-HSR、2026-09 轻量先例|不能声称首次匹配置信度或回退|
|实际注入价值|参考 SR/专家门控均构成威胁|仅窄收益监督假设，主探索线|
|计算分配|SRLF、BUGPan|强对照，不作主创新|
|误差与歧义|UMNet、CoPGNet、Guo 2026、旧 UQ 网络|宽泛分解/加权被覆盖|
|结构先验|Hipandas、UAFL、MossFuse、GSNR、NPN|低秩、解耦、图/零空间已有|
|连续与传感器|PINO、CLoRF、CARL、SpecAware、UniPAN|储备，需必要机制差异|
|生成与语义|Diff-Spectra、EMR-Diff、GEWDiff、CDPS、CO²IF、HyperSIGMA|优先借鉴压力测试；不从大模型拼装起步|

不能把 HS–PAN、RGB→HSI、分类/分割与 HSI–MSI 融合当成同一任务。任务迁移的贡献必须由任务特有约束与有效的新机制支持。
''')
write('ideation/selected-idea.md',r'''# 选定的窄探索假设 I13

**监督学习 MSI 细节注入的光谱误差收益，并检验它是否优于几何可靠性和普通专家门控。**

状态：探索性候选；查新定位 3/5，置信度 2/5；没有性能证据。不是“已确认无人做过”的结论。

机制、接口、标签和损失见 research-report.md 第 5 节。必须使用显式训练的 HSI-only 专家，冻结/分折教师产生收益标签；推理只能看观测与预测。oracle HR 门控属于上限诊断。

首要试验：高匹配/低回投影误差条件下，是否存在足量负光谱收益？第二试验：同样专家、配准、容量与训练预算下，收益监督是否超过端到端普通 gate、g* 教师 gate、误差尺度 gate？

不成立即停止：上限很小；简单 gate 同样有效；收益只来自容量或一种合成遮挡；干净场景持续退化。之后才能规划独立确认性实验。仍需取得 CoPGNet 全文，以及更高召回的参考 SR/专家收益教师查新。

储备 I14：完整 SRF 形状条件的连续保谱融合；定位 2/5，目前不满足独立主创新标准。
''')
write('ideation/rejected-ideas.md','# 淘汰候选\n\n'+ '\n\n'.join(f'**{key} {title}**：{why} 最接近：{", ".join(priors)}。' for key,title,kind,priors,status,why,scores in IDEAS if status=='rejected')+'\n\nI14 未淘汰为工程能力，但没有作为主方法创新保留；需要研究后重新审查。\n')
write('ideation/ideation-decision.json',dict(decision='proceed_to_experiment_plan',selected_idea_ids=['I13'],confidence='low',novelty_check_status='reviewed',rationale='只有窄收益监督假设进入探索试验；原始置信度回退与歧义分解被否定，传感器路线降为储备。',next_skill='research-experiment-plan',required_handoffs=['research-report.md section 5','novelty-review/novelty-decision.json','baseline-snapshot.json','prior-art threats Q1-Q3','outcome-independent exploratory pilot; no trained results'],limits=['novelty self-review, not independent','not an exhaustive systematic review','missing CoPGNet/BUGPan full texts','no GPU training','not confirmatory or route-ready']))

QUERY_TEXT = '''hyperspectral multispectral fusion CVPR 2025 2026 spectral response fusion
hyperspectral multispectral image fusion ICCV 2025 AAAI 2026 TIP 2025
site:openaccess.thecvf.com 2025 "SpecAware"
site:openaccess.thecvf.com 2026 "hyperspectral" "fusion"
site:proceedings.neurips.cc 2025 hyperspectral inverse problems self supervised
site:proceedings.iclr.cc 2026 CARL spectral
"SpecAware" spectral response hyperspectral 2025
"hyperspectral" "CVPR 2026" fusion uncertainty misalignment
"hyperspectral" "ICLR 2026" CARL
"hyperspectral" "NeurIPS 2025" fusion
hyperspectral multispectral fusion uncertainty calibration conformal prediction
hyperspectral fusion sensor invariant equivariant spectral response learning 2025 2026
hyperspectral unregistered fusion occlusion confidence reliability 2025 2026
hyperspectral fusion adaptive computation early exit selective relearning 2025
"Physics-informed Neural Operator for Pansharpening" NeurIPS 2025
"Enpowering Your Pansharpening Models" ICCV2025
"hyperspectral" "fusion" "uncertainty" calibration reconstruction 2025 2026 -classification
"hyperspectral" "fusion" "joint denoising" multispectral 2025 2026
"hyperspectral" "fusion" "occlusion" 2025 2026
"hyperspectral" "fusion" "quantile" normalization
"Exploring structured uncertainty for external prior-guided hyperspectral image fusion"
"Bidomain uncertainty gated recursive network" pansharpening
"hyperspectral" "fusion" "ambiguity" "uncertainty"
"unregistered" "hyperspectral" "occlusion" fusion
"CoPGNet" github pdf
"BUGPan" pdf Bidomain uncertainty gated
"hyperspectral fusion" "joint null" uncertainty
"hyperspectral fusion" "occlusion" rejection confidence
"hyperspectral multispectral" "uncertainty" "aleatoric"
"hyperspectral fusion" "null space" "uncertainty"
"hyperspectral" "fusion" "non-corresponding"
"reference" "super-resolution" "negative transfer" "confidence" CVPR 2025
"hyperspectral" "TIP" "2026" "CDPS"
"SHOTUN" "2026" hyperspectral
"HyperSIGMA" "TPAMI" 2025
"Unaligned RGB-Guided Hyperspectral" IJCV 2025
"Coupled Diffusion Posterior Sampling" "TIP"
"Self-Expressive High-Order Tensor" "2026"
"Implicit Representation" "Arbitrary" "3551531"
"hyperspectral" "fusion" "selective prediction" uncertainty
"Exploring structured uncertainty for external prior-guided" Xu Fang
"Self-Expressive High-Order Tensor" Wang
"Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral"
"Fusion of Hyperspectral and Multispectral Images Accounting for Localized"
"null space" "uncertainty" "inverse problems" conformal
"hyperspectral fusion" "uncertainty" "identifiability"
"reference super-resolution" "confidence" "oracle"
"hyperspectral" "fusion" "counterfactual"
"Separating Intrinsic Ambiguity from Estimation Uncertainty"
"Non-Linear Projections of the Null-Space for Imaging Inverse Problems" NeurIPS 2025
"SpecAware" "ISPRS" 2026
"reference super resolution" "confidence" "negative transfer" oracle
"hyperspectral fusion" "utility" "gate"
"SpecAware" "ISPRS" 2026
"reference super resolution" "confidence" "negative transfer" oracle
site:openaccess.thecvf.com "reference" "super-resolution" "confidence" "2025"
"reference super-resolution" "oracle" gating
"hyperspectral fusion" "counterfactual" benefit
"spectral" "fusion" "utility" "confidence" hyperspectral
site:openaccess.thecvf.com "Learning Texture Transformer Network for Image Super-Resolution"
site:proceedings.neurips.cc "Non-Linear Projections" 2025
site:ecva.net 2026 "hyperspectral" "fusion"
site:openreview.net ICLR 2025 "hyperspectral" "fusion"
"TIP" "2025" "hyperspectral fusion" uncertainty
"reference-based super-resolution" "confidence" "ground truth" gate'''.splitlines()

log='# Search Log\n\n检索日期：2026-10-07。引擎：web search；最终事实只依赖官方会议/期刊页面、原文、作者资料与代码。网页抓取、DOI 和 canonical helper 的原始尝试另见 retrieval-manifest.json。查询中的年份是词项；不是统一数据库日期过滤。下表是人工相关性记录，不是引擎全量命中、去重计数或可复现召回率审计。重复查询保留。\n\n|序号|实际查询|目的与结果|\n|---|---|---|\n'
for i,q in enumerate(QUERY_TEXT,1):
    if i in [21,25,41]: outcome='寻找结构化不确定性杀伤先例：CoPGNet；全文未取得，保留 Q1。'
    elif i in [22,26]: outcome='动态计算杀伤先例：BUGPan DOI/代码；全文未取得。'
    elif i in [36,24,28]: outcome='查未对应/遮挡：SSC-HSR 与 2026-09 轻量先例；后者全文直接覆盖回退主张。'
    elif i in [45,49]: outcome='查歧义分解：GSNR、旧 UQ 网络、Guo 2026；完成关键全文核查。'
    elif i in [50,61]: outcome='核实 NPN：NeurIPS 2025 官方主会页面；摘要级。'
    elif i in [7,51,54]: outcome='核实 SpecAware：ISPRS JPRS 2026 DOI 和作者代码；全文改从 arXiv v2 读取。'
    elif i in [32,47,48,52,53,55,56,57,58,59]: outcome='查参考迁移/收益教师：没有本次确认的精确等价命中；检索较稀疏且噪声大，保留 Q2，不能据此声称不存在。'
    elif i==60: outcome='历史先例 TTSR：官方 CVPR 2020 页面确认相关性控制纹理迁移；摘要级。'
    elif i==65: outcome='找到 CVPR2022 Task Decoupled：完成主文全文阅读；已监督 HR−SISR 残差及相关性迁移，加入 I13 强对照。'
    elif i in [62,63]: outcome='补查 ECCV2026 与 ICLR2025：本次未确认可加入核心的直接命中；不宣称该场所没有相关论文。'
    elif i==64: outcome='发现 TIP 异谱结构补偿采样 DOI 线索；仅聚合索引，未提升为已核实正式推荐或方法证据。'
    elif i in [33,37,43]: outcome='核实 TIP CDPS：作者代码/DOI 记录；不提升为全文证据。'
    elif i in [34,38,42]: outcome='检索 SHOTUN：找到元数据但当前抓取遇验证，不作为核心方法证据。'
    elif i in [11,17,23,27,29,30,40,46]: outcome='不确定性/零空间杀伤查询：UMNet、相关逆问题先例；精确收益监督仍未穷尽。'
    else: outcome='发现与核实领域正式论文：对照 sources.json；官方内容优先，聚合网站仅作线索。'
    log+=f'|{i}|`{q}`|{outcome}|\n'
log+='\n来源排除：CVPR Findings、ICCV/CVPR workshop、未来 2027 命中不计主会；Guo 的 FoGen workshop 与最新未配准预印本作为先例单列。ECCV 2024 早于滚动窗起点。CoPGNet/BUGPan OA 尝试包括 Unpaywall 跳过无 email、OpenAlex、Semantic Scholar、Crossref 与 DOI 页；失败不意味着不存在相关机制。\n'
write('novelty-review/search-log.md',log)

matrix=r'''# Prior Art Matrix

威胁评分 1–5：5 为足以覆盖宽泛主张。下列“差异”都是候选假设，未证明效果。

|先例与来源|任务/场所|重叠维度|威胁|剩余差异假设|能否否定我们的主张|
|---|---|---|---:|---|---|
|[UAFL](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Enhancing_Unregistered_Hyperspectral_Image_Super-Resolution_via_Unmixing-based_Abundance_Fusion_Learning_CVPR_2026_paper.html)|有监督未配准 HSI–RGB / CVPR 2026|丰度残差、对齐、相似性引导|5|I13 监督两专家注入后的光谱误差差额|若收益信号只是匹配置信度代理则没有必要差异|
|[轻量未配准先例](https://arxiv.org/abs/2609.01060)|真实错配 RGB 引导 / preprint 2026-09|可靠性加权细节、上采样回退|5|I13 同可靠性条件下测实际光谱收益|直接否定 I12；I13 需证明置信度对照不足|
|[UMNet](https://ojs.aaai.org/index.php/AAAI/article/view/37988)|HS–PAN / AAAI 2026|结构化误差尺度、困难区域训练|4|I13 两专家收益差额而非单模型尺度|可能只是 UQ 目标的简单改写|
|[SRLF](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_A_Selective_Re-learning_Mechanism_for_Hyperspectral_Fusion_Imaging_CVPR_2025_paper.html)|HSI–MSI / CVPR 2025|回投影偏差、选择性计算|4|I13 针对低偏差但负收益的位置|若回投影即可解释收益则主线被否定|
|[DIFIV](https://www.cedric-richard.fr/Articles/wang2023deep.pdf)|光谱变化 HSI–MSI / TGRS 2023|局部变化、鲁棒权重|4|I13 单目标有监督决策|局部内容差异处理不能声称首次|
|[TTSR](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Learning_Texture_Transformer_Network_for_Image_Super-Resolution_CVPR_2020_paper.html)|参考 SR / CVPR 2020|相关性控制纹理迁移|4|收益标签与 HSI 目标的必要性|通用门控已有；Q2 未关闭|
|[Task Decoupled](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Task_Decoupled_Framework_for_Reference-Based_Super-Resolution_CVPR_2022_paper.pdf)|参考 SR / CVPR 2022|冻结 SISR、HR−SISR 残差监督、相关性纹理迁移|5|I13 显式误差收益差额，而不是纹理残差标签|若真值残差监督已足够，收益头没有必要性；禁止首次双分支/真值指导的表述|
|[CoPGNet](https://doi.org/10.1016/j.inffus.2025.103997)|外部先验 HSI 融合 / IF 2026|结构化不确定性、先验引导|4|尚不能确定；全文未得|高优先级 Q1，可能直接覆盖收益/先验决策|
|[BUGPan](https://doi.org/10.1016/j.inffus.2025.102938)|pansharpening / IF 2025|不确定性门控、可变递归|5|I13 不是计算量分配|否定 I08 宽泛主张；全文缺口不能略过|
|[PINO](https://proceedings.neurips.cc/paper_files/paper/2025/file/89af8e4eb696738f2c9e589522968a09-Paper-Conference.pdf)|含 CAVE 融合 / NeurIPS 2025|连续表示、SRF 成像、LR 跳连|5|I14 已知完整响应家族的轻量同模型实验|不是首次从 PAN 迁移到 HSI–MSI|
|[SpecAware](https://doi.org/10.1016/j.isprsjprs.2026.02.024)|重建预训练与语义解译 / ISPRS 2026|元数据+内容编码、动态解码、保谱损失|5|I14 完整响应形状与融合约束|“动态解码保谱”本身被覆盖；当前 I14 仅 2/5|
|[CARL](https://proceedings.iclr.cc/paper_files/paper/2026/file/282967f8abaae52a452a97ee961410f3-Paper-Conference.pdf)|语义分析 / ICLR 2026|波长条件、相机无关聚合|5|辐射重建需另验证|仅任务迁移仍可能增量|
|[UniPAN](https://openaccess.thecvf.com/content/ICCV2025/html/Cui_Enpowering_Your_Pansharpening_Models_with_Generalizability_Unified_Distribution_is_All_ICCV_2025_paper.html)|MS–PAN / ICCV 2025|分布对齐、原始域跳连与损失|5|尚未证明新的必要机制|否定 I06|
|[Guo 等](https://arxiv.org/abs/2605.15050)|线性逆问题 / ICML FoGen workshop 2026|内禀/传播不确定性、级联、校准|5|HSI 专项诊断有用途但不是宽泛新方法|直接否定 I11；不将 workshop 当主会|
|[旧 UQ 零空间](https://arxiv.org/abs/2304.06955)|CT/MRI / 2023|数据一致性加不确定性|5|暂无独立机制|零空间+UQ 已存在|
|[GSNR](https://openaccess.thecvf.com/content/CVPR2026/html/Gualdron-Hurtado_GSNR_Graph_Smooth_Null-Space_Representation_for_Inverse_Problems_CVPR_2026_paper.html) / [NPN](https://proceedings.neurips.cc/paper_files/paper/2025/hash/acb1891d79e617134ed604084ebbc919-Abstract-Conference.html)|逆问题 / CVPR 2026、NeurIPS 2025|低维零空间先验与图结构|5|双观测迁移有工程价值|不能声称首次零空间图先验|

本地先例：research/geometry_condition_* 与 mossfuse_shared_complementary_ideation_20261007 已包含相关思路。不是外部发表证据，但不能把已有本地探索重复计为本次原创。
'''
write('novelty-review/prior-art-matrix.md',matrix)

position='I13：仅探索在固定专家与同等配准条件下，以监督光谱误差差额训练细节注入决策，是否能超过匹配置信度、测量残差和普通专家门控。'
change='若 CoPGNet 或参考 SR/专家门控已有功能等价收益教师，或普通 gate/置信度基线达到同等效果，降为 1–2；若高召回补检与独立实验同时支持必要机制，才重审定位。'
decision=dict(schema_version='1.1',status='complete',novelty_decision_rating=3,impact_positioning_rating=3,decision_confidence_rating=2,narrowest_defensible_positioning=position,what_would_change_the_decision=change,
    top_kill_shot_objections=['普通 MoE 或 oracle convex-gate 教师可能功能等价','CVPR2022 Task Decoupled 已有冻结 SISR 加真值残差监督','2026-09 先例已覆盖可靠性细节注入与回退','CoPGNet 未得全文，存在高优先级覆盖风险','收益改善可能只来自额外专家容量或合成扰动'],
    literature_context_used=['../research-report.md','../paper-cards.md','../reading-ledger.md','../sources.json','prior-art-matrix.md','search-log.md','prior local geometry and MossFuse ideation'],paper_review_summary_used='',
    claims_to_qualify=['不得声称首次门控、回退、oracle 教师或不确定性融合','不得声称已确认无人提出或独立查新','不得声称保证恢复所有同观测真实谱','不得声称已提升性能或适合顶刊','I14 仅储备，新颖性不足'],
    missing_prior_work=['CoPGNet 全文及其收益/先验权重训练细节','BUGPan 全文用于完整动态计算对照','更高召回的参考 SR 和专家收益教师文献','GSNR 另附补充材料'],review_findings_to_add=[],
    literature_assurance=dict(mode='unlinked',paths={k:'' for k in ['protocol','search_log','recall_audit','corpus_manifest','screening_log','evidence','report']},file_sha256={k:'' for k in ['protocol','search_log','recall_audit','corpus_manifest','screening_log','evidence','report']},corpus_manifest_sha256='',corpus_version=None,review_profile='bounded exploratory discovery',assurance_verdict='bounded-exploration-with-material-gaps',unresolved_high_priority_novelty_question_ids=['Q1','Q2','Q3']),
    review_independence='same-context self-review; no materially independent reviewer',
    evidence_class='exploratory literature synthesis and analytic toy check; no trained outcomes',
    selection_history=['I11 broad uncertainty decomposition killed by Guo 2026','I12 broad confidence fallback killed by September 2026 preprint','I13 newly generated after I12 failure; no empirical outcome selection','CVPR2022 Task Decoupled subsequently added as strong residual-supervision comparator; two-branch/GT-residual claims excluded','I14 downgraded after full SpecAware read; not maintained as main contribution'],
    reserve_idea=dict(idea_id='I14',novelty_decision_rating=2,decision_confidence_rating=3,status='needs major differentiation; not main contribution'))
write('novelty-review/novelty-decision.json',decision)
novel=r'''# Novelty Report

## Contribution Under Review

本次对象是 I13 的窄收益监督方法假设；I14 仅作储备评价。工作模式 standalone / unlinked；同上下文自审，没有独立审查者。已使用 baseline、原有本地探索、核心全文与正式来源；不是穷尽性系统综述。

## Claim Decomposition

|维度|主张|状态|
|---|---|---|
|framing|匹配可靠性与实际光谱收益可能不同|可检验问题；不是首次“可靠引导”|
|method|两专家差额标签训练决策头|窄候选；通用门控与教师框架可能等价|
|protocol|固定专家下分离几何/可见域吻合与光谱收益|探索性诊断；未证明新协议|
|artifact|可复用研究与阅读包|本次交付，不是新数据集|
|finding|高可靠区域仍有负收益且收益监督可修正|尚无训练证据，不可写成结论|

I11/I12 原始主张已否定。I13 是在否定后生成的新假设；I14 读到动态解码与保谱损失后降为储备。没有本次模型实验结果参与选题，没有把结果知情选择包装成确认性研究。

## Strongest Overlaps

最新未配准先例、UAFL、UMNet、SRLF、DIFIV、TTSR、CVPR2022 Task Decoupled 与通用 MoE。Task Decoupled 主文已完整读取，已有冻结单图 SR、真值残差监督纹理分支与相关性迁移；它是必须进行的强对照。详细重叠、威胁评分和来源见 prior-art-matrix.md。CoPGNet 全文未得，是不能绕过的威胁。I14 与 PINO/SpecAware 的结构重叠很强，评分 2/5，判定置信度 3/5。

## Novelty-Killing Objections

1. “收益监督”可能只是普通 gate 的 oracle 标签设计，不产生必要的新机制。
2. 高几何可靠却有负收益的区域可能稀少，或其问题由现有结构可靠性分数解释。
3. 净收益可能来自多一个 HSI-only 专家；必须比较同一专家对和同预算普通门控。
4. CoPGNet 或参考 SR 可能已直接学习外部先验的增量收益；缺全文或检索无命中不是不存在。
5. 相同观测的共同零空间歧义不能由观测头逐例解决；不得声称恢复保证。

## Decision

'''
novel+=f'''- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 2
- Narrowest defensible positioning: {position}
- What would change the decision: {change}
- Literature assurance mode: unlinked
- Literature assurance verdict: bounded-exploration-with-material-gaps
- Unresolved high-priority novelty-critical questions: Q1, Q2, Q3

评分只支撑窄探索定位，不认证科学原创、效果、实现正确性、确认性状态或投稿准备度。不能声称首次门控/回退/教师，不能声称已确认无人提出或独立查新。I14 不进入独立主贡献清单。

Q1：CoPGNet 的先验权重是否用收益目标监督？Q2：参考 SR/专家路由是否已有功能等价收益教师？Q3：普通 g* 教师和端到端 gate 是否已足够？Q4（储备）：完整 SRF 形状条件是否提供超出 PINO+SpecAware 常规组合的必要机制？

## Paper-Review Handoff

当前没有具体新稿件可审。后续实验计划应先做两专家失败模式与 oracle 上限，再公平比较简单强对照；只有验证后才起草贡献句。保留 I11/I12 的否定历史、Q1–Q3 和探索性证据身份。主报告中打印公式核查只涉及 Diff-Spectra 排版与维度，不扩展为代码或整体结论错误。
'''
write('novelty-review/novelty-report.md',novel)

if __name__=='__main__':
    print(json.dumps({'ideas':len(records),'selected':['I13'],'reserve':['I14'],'exact_queries':len(QUERY_TEXT),'novelty_rating':3,'decision_confidence':2},ensure_ascii=False))
