"""Preserve bounded literature positioning and exploratory idea decisions."""
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
IDEA = BASE / 'ideation'
IDEA.mkdir(exist_ok=True)
SKILLS = Path('C:/Users/Administrator/.codex/skills')

def write(path, content):
    path.write_text(content.strip() + '\n', encoding='utf-8')

sources = [
 ('dTV', 'Inverse Problems 2018', 'https://arxiv.org/abs/1710.05705', '作者全文定义、方向解释及参数敏感性段落；未全篇逐页精读', '方向几何引导、盲核和失配', 4),
 ('HSR-Diff', 'ICCV 2023', 'https://arxiv.org/abs/2306.12085', '本地已有全文证据，本轮复用；本轮 CVF 403', '双观测条件扩散', 4),
 ('DDIF', 'Information Fusion 2024', 'https://liangjiandeng.github.io/papers/2024/zihan-infus2024.pdf', '本地作者全文，本轮重新核对 CSM/FWM 方法段', '残差扩散、粗细条件调制与空间小波', 5),
 ('GEWDiff', 'AAAI 2026', 'https://ojs.aaai.org/index.php/AAAI/article/download/37978/41940', '本地正式全文表2/结论，本轮在线作者扩展稿方法', 'mask条件与边缘噪声', 5),
 ('GGAT', 'Pattern Recognition 2026；卷期 November 2026，已在线', 'https://doi.org/10.1016/j.patcog.2026.113592', '出版商可检索摘要、引言；完整访问403；首次上线日期未核实', 'HSI–MSI显式高阶几何和MSI图注意力', 5),
 ('SDP-Net', 'Knowledge-Based Systems 2026', 'https://doi.org/10.1016/j.knosys.2026.116227', '出版商可检索摘要、引言和贡献段；完整访问403', '像素结构引导、控制注入、光谱保护、双路径门控', 5),
 ('SA-RFM', 'ICML 2026，PMLR 306', 'https://proceedings.mlr.press/v306/zhang26df.html', '会议摘要、作者仓库框架；PDF超时/不可访问', 'MSI结构诱导各向异性条件生成，联合配准融合', 4),
 ('ARGS-Diff', 'CVPR 2025', 'https://arxiv.org/abs/2505.11800', '本地既有作者全文和正式定位，本轮CVF403', '自学习双先验、自适应观测残差指导', 4),
 ('Matrix-R', 'Sensors 2025', 'https://www.mdpi.com/1424-8220/25/24/7662', '出版商全文投影定义和历史pansharpening段落，非全篇精读', '响应行空间/零空间分离', 5),
 ('DDSR', 'Remote Sensing 2024', 'https://doi.org/10.3390/rs16152692', '工作区既有方法段证据；本轮尝试全文429', '响应range/null分离及扩散', 5),
 ('HSDiff', 'arXiv预印本 2025', 'https://arxiv.org/abs/2511.18473', '作者全文相关方法、实验、局限和metameric augmentation段落', '纹理引导同色异谱增强、物理后验与不确定性', 4),
 ('Prior-consistent DIM', 'Geo-spatial Information Science，2026在线', 'https://doi.org/10.1080/10095020.2026.2694245', '出版商可检索摘要/动机；全文403', '退化先验注入及空间光谱一致性', 4),
]
write(BASE/'sources.json', json.dumps({'as_of':'2026-10-07','scope':'bounded targeted search, not systematic review',
 'records':[dict(zip(['title','venue','url','read_depth','overlap','threat'],s)) for s in sources]},ensure_ascii=False,indent=2))

matrix = '# Prior Art Matrix\n\n日期：2026-10-07。威胁针对宽主张“几何/小波条件注入融合网络”；1–5，5为直接覆盖。窄候选仍需排查同构实现。\n\n| 工作 | 年份/载体 | 直接重合 | 威胁 | 窄候选需证明的差异 | 证据深度 |\n|---|---|---|---|---|---|\n'
for title, venue, url, depth, overlap, threat in sources:
    diff = '局部连续RU灵敏度与几何修正耦合，胜过同信息量及同预算对照；不能首创已有组件'
    if title == 'SDP-Net': diff = '全文是否已有等价响应敏感度条件；普通控制门控不能成立为差异'
    if title == 'SA-RFM': diff = '已配准条件下的光谱归属，而非形变运输；全文仍缺'
    if title == 'Matrix-R': diff = '内容相关局部响应而非固定投影；投影本身不是贡献'
    if title == 'HSDiff': diff = '双观测融合的细节支持范围；不能声称首创metamer增强或校准'
    matrix += f'| [{title}]({url}) | {venue} | {overlap} | {threat} | {diff} | {depth} |\n'
matrix += '\n最强否定：首次几何融合、首次小波条件扩散、首次可靠性门控、首次响应投影均不能主张。摘要没有描述某个机制，不足以证明该机制不存在。\n'
write(BASE/'prior-art-matrix.md', matrix)

queries = [
 '"Geometric Enhanced Wavelet-based Diffusion Model"',
 'hyperspectral multispectral fusion geometric guidance structure tensor edge condition injection',
 '"GEWDiff" arxiv',
 'hyperspectral multispectral fusion "edge" "injection" network',
 '"High-frequency geometry enhanced graph attention network"',
 '"Spatial-aware guidance" "wavelet" "SDP" fusion',
 '"High-frequency geometry enhanced graph attention network" DOI GGAT',
 '"hyperspectral" "structure tensor" fusion network',
 'hyperspectral multispectral fusion edge guided network EFIN geometry gradient',
 '"hyperspectral" fusion "uncertainty" "detail" "injection"',
 'hyperspectral multispectral fusion "directional total variation"',
 'hyperspectral multispectral fusion "spectral" "observability" geometry',
 '"hyperspectral" fusion "uncertainty" edge guidance spectral response',
 '"High-frequency geometry" "GGAT" github',
 '"hyperspectral" fusion "spectral observability" geometry gate',
 '"hyperspectral" fusion "null space" "geometry"',
 '"hyperspectral" fusion "metameric" edge',
 '"A spatially guided dual-path network" "2026" "116227"',
 '"Structure-Aware Riemannian Flow Matching" "structure tensor"',
 '"Matrix-R Theory" "RGB-Guided" journal',
 '"GGAT" "Ziqing" geometry',
 'site:proceedings.mlr.press/v306/zhang26df "tensor" "metric"',
 '"Structure-Aware Riemannian Flow Matching" arxiv',
 '"A spatially guided dual-path" "SS2D" "limitation"',
 '"High-frequency geometry enhanced" "limitations"',
]
log = '# Search Log\n\n日期：2026-10-07（北京时间）。本轮25个原样查询；web search，无年份/域名过滤（词内年份和site限定除外）。技术结论只引用作者/出版商/会议来源，二手索引仅发现线索。非系统综述，没有召回率或穷尽性承诺。\n\n| Query | Source | Filters | Purpose | Closest hits / notes |\n|---|---|---|---|---|\n'
for q in queries:
    purpose = '寻找直接重合，否定宽泛首创及排查局部响应条件'
    log += f'| `{q}` | web search | 无额外过滤 | {purpose} | 来源与读取深度见 sources.json；不是每个查询都有有效命中 |\n'
log += '''
## Source follow-up and access limits

- GEWDiff正式表2/结论、DDIF CSM/FWM来自既有本地全文，本轮重新读相关段落；HSR-Diff/ARGS-Diff沿用可追溯工作区证据，不宣称本轮重读全部全文。
- GGAT/SDP-Net搜索可取得出版商摘要、引言/贡献。open正文403；不把检索片段称为完整方法和实验。
- SA-RFM PMLR摘要与作者README已读。raw GitHub正式PDF超时；PMLR另一PDF路径不可访问。没有据此断言其具体结构张量实现、计算量或不存在谱响应条件。
- dTV作者全文读定义、方向解释和参数敏感性相关段；Matrix-R读投影定义和历史融合段；HSDiff读相关方法/实验/局限段。未声明全篇完整阅读。
- DDSR公开入口429，本轮仅沿用前期方法证据；CVF HSR-Diff/ARGS-Diff返回403。
- prior-consistent DIM出版商摘要/动机可检索，全文403。
- 探索性访问了一个未确认的ScienceDirect PII S1566253524000769，返回403；未将该URL识别为DDIF或纳入技术证据。
- 没有绕过访问控制，未据网页抓取时间认定首次发表日期。GGAT已在线可检索，但卷期November2026在当前日期之后；不引用二手March2026日期。

## Local scope and selection history

读取当前model、训练、CAVE数据生成与utils，以及既有GEWDiff、MossFuse/解耦研究。新方向复用已有光谱可辨性候选，聚焦几何条件。未训练、未读取最终测试图像开展选择。合成诊断说明可定义问题与非唯一性，不作为自然场景效果。

证据级别为exploratory、self-review。没有七文件系统综述证据包，也没有独立审稿。下轮最关键问题：NQ1 SDP-Net完整方法是否同构；NQ2 GGAT/SA-RFM是否已有相同光谱条件；NQ3标准子空间/响应引导是否解释全部机制；NQ4近期metamer/不确定性融合是否覆盖延伸。
'''
write(BASE/'search-log.md',log)

position = '在已知响应、配准及稳定LR局部基条件下，检验方向几何与连续有效光谱响应灵敏度的条件耦合能否减少边界光谱分配误差，优于等预算门控与同信息量子空间对照；不宣称首次或已证实。'
change = '若完整近邻已同构，或可部署局部基无效、自然失败不存在、等预算门控/经典求解解释全部收益，则降至2/5或放弃；完整查新和独立实验支持后才考虑提高。'
objections = [
 'SDP-Net已明确控制MSI结构传播和光谱偏移，可靠性门控故事直接重合。',
 'GGAT、dTV和SA-RFM已覆盖显式几何、方向及结构条件生成。',
 'Matrix-R、DDSR与标准子空间正则化覆盖响应投影，公式是常规正则化。',
 'RU来自低分辨率局部估计，可能遗漏HR材料变化，只有oracle基有效不构成部署方法。',
 'baseline的原MSI路径会绕过新增门控，不能声称控制了全部结构注入。',
 '两个观测都不可见的扰动不可唯一恢复，物理一致性不是细节正确证明。'
]
spec=importlib.util.spec_from_file_location('novelty_template',SKILLS/'research-novelty-review/scripts/init_novelty_pack.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
decision=module.decision_template()
decision.update(status='complete',novelty_decision_rating=3,impact_positioning_rating=3,decision_confidence_rating=2,
 narrowest_defensible_positioning=position,what_would_change_the_decision=change,
 top_kill_shot_objections=objections,literature_context_used=['research-report.md','sources.json','prior-art-matrix.md',
 '../gewdiff_transfer_20261007/analysis.md','../mossfuse_shared_complementary_ideation_20261007/research_direction_analysis.md'],
 claims_to_qualify=['新颖性仅为定向自审暂定3/5；最近论文未获完整方法。','不能首创几何条件、门控、小波扩散、投影或同色异谱增强。',
 '未证明自然失败、可部署基、训练增益或SOTA。','附加路径控制不等于全网络受控，弱响应不等于全任务不可辨。'],
 missing_prior_work=['SDP-Net完整方法和代码比对','GGAT与SA-RFM完整方法','近期局部子空间/响应条件方法全文'],
 selection_history='文献及已有光谱可辨性候选告知；合成算子诊断告知恢复边界；未查看新训练结果。',
 evidence_class='exploratory hypothesis with synthetic operator checks',review_independence='self-review; no delegated or materially independent pass')
decision['literature_assurance'].update(assurance_verdict='targeted-search-only; material-fulltext-gaps',
 unresolved_high_priority_novelty_question_ids=['NQ1','NQ2','NQ3','NQ4'])
write(BASE/'novelty-decision.json',json.dumps(decision,ensure_ascii=False,indent=2))
report = '''# Novelty Report

## Contribution Under Review

Standalone定向审查：局部光谱可辨性约束的方向几何条件注入。不是对GEWDiff整个论文的首次审稿；使用研究报告及工作区既有全文/诊断。宽泛几何注入方案2/5，窄候选3/5。

## Claim Decomposition

| Dimension | Claimed novelty | Searchable formulation |
|---|---|---|
| framing | 同空间几何下的光谱细节分配 | geometry + spectral sensitivity + fusion |
| method | LR局部基的连续RU灵敏度耦合张量几何修正 | local subspace + sensor response + conditional injection |
| protocol | 同几何/同秩/不同响应的干预及部署基对照 | metamerism + geometry + controlled comparison |
| artifact | 当前合成检查可重跑，但不是新数据集 | actual R/D operator examples |
| finding | 尚无自然机制或训练收益 | 不能预先提出已证实结论 |

## Strongest Overlaps

见prior-art-matrix.md。SDP-Net、GGAT、DDIF、Matrix-R为强重合；dTV和SA-RFM排除首次方向引导；HSDiff排除同色异谱增强/不确定性首创。全文缺口使缺失描述不能作为差异证据。

## Novelty-Killing Objections

'''
report += '\n'.join('- '+o for o in objections)
report += f'''

## Decision

- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 2
- Narrowest defensible positioning: {position}
- What would change the decision: {change}
- Literature assurance mode: unlinked
- Literature assurance verdict: targeted-search-only; material-fulltext-gaps
- Unresolved high-priority novelty-critical questions: NQ1, NQ2, NQ3, NQ4

仅决定定位，不验证实证正确、独立性或确认性。选择为结果/文献告知的探索候选；无新训练结果。合成例子非原创定理，也不代表自然样本普遍发生失败。

## Paper-Review Handoff

没有待审稿论文。交接时必须限定局部基与传感器条件，报告原MSI通路旁路风险、弱响应与零响应区别、自然机制缺口及所有近邻全文问题。不得以此判定直接进入确认性实验；先补近邻与诊断。
'''
write(BASE/'novelty-report.md',report)

definitions = [
 ('G01','局部光谱可辨性约束的方向几何注入','method','selected',[5,3,4,5,4],'medium','weeks',
  '几何与连续RU灵敏度联合条件优于单独几何门控，减少边界光谱分配错误。',
  '自然边界分组诊断、同几何同秩不同响应模拟、几何×灵敏度2×2消融。',
  '仅oracle基有效，或同预算门控/固定投影/经典子空间求解解释全部收益。',
  ['SDP-Net','GGAT','Matrix-R','dTV'], '局部内容相关响应耦合几何修正；公式本身不是新意。'),
 ('G02','退化带宽条件下的尺度几何传递','method','shortlisted',[4,3,3,4,4],'high','weeks',
  '区分观测支持的粗几何和细几何，比普通多尺度拼接更少错误传播。',
  '同几何不同PSF/尺度干预，与同条件多尺度卷积对照。',
  '普通多尺度/物理一致性解释全部收益，或无法得到可部署的尺度支持量。',
  ['DDIF','SDP-Net','Prior-consistent DIM'], '把尺度支持能力作为可测条件，不将小波分解称为创新。'),
 ('G03','几何条件干预与光谱响应诊断','diagnostic','shortlisted',[4,3,4,5,3],'medium','days',
  '干预几何条件产生的光谱变化与有效观测灵敏度对应，可揭示错误注入。',
  '仅扰动辅助几何条件并检查局部输出方向，同时监测真实输入不变的合理先验变化。',
  '响应只反映任意辅助分支或通用平滑，无法对应自然误差或主重建路径。',
  ['SDP-Net','GEWDiff','Matrix-R'], '作为主线诊断；不能要求所有不可测几何扰动都严格不影响预测。'),
 ('G04','双观测弱支持细节的不确定性','mixed','needs_research',[4,3,2,4,4],'high','months',
  '弱约束区域输出校准区间比仅锐化更可靠。',
  '先比异方差头和ensemble，再考虑后验采样；报告覆盖率与区间宽度。',
  '普通不确定性模型相同效果，或HSDiff/CDPS近邻已覆盖。',
  ['HSDiff','ARGS-Diff','DDSR'], '融合双观测支持范围的校准，尚不声称空白。'),
 ('G05','几何配准与融合联合优化','method','rejected',[4,2,2,4,4],'high','months',
  '联合估计形变可减少错位边界。','比较联合与顺序配准、SA-RFM。',
  '近期联合方法已有核心机制，且当前数据无错位。',['SA-RFM','dTV'], '当前没有清晰区别，作为失配测试背景。'),
 ('G06','空间小波几何门控','method','rejected',[5,2,4,4,3],'medium','weeks',
  '小波高频门控改善边缘。','与DDIF/SDP-Net思想及同条件卷积对照。',
  '普通高频条件或门控已覆盖核心。',['DDIF','SDP-Net'], '单独研究不足，必要时作为尺度实现。'),
 ('G07','几何条件残差扩散','method','needs_research',[4,2,2,4,4],'high','months',
  '在弱支持区域多步条件恢复优于同条件确定性方法。',
  '同条件、实际时间与参数预算比较，并额外测量校准。',
  '确定性修正相同或更好，或条件组合与DDIF等同构。',['DDIF','HSR-Diff','GEWDiff','ARGS-Diff'],
  '扩散必要性尚无证据，不作主线先决条件。'),
 ('G08','语义区域mask注入','method','rejected',[4,1,3,4,2],'medium','weeks',
  '分割区域可能提供梯度之外的结构信息。','与梯度/张量同预算比较，控制额外预训练模型。',
  'GEWDiff已覆盖且CAVE领域动机弱，收益来自外部模型。',['GEWDiff'], '需要新任务证据，当前不优先。'),
]
records=[]
for ident,title,typ,status,scores,risk,effort,hyp,val,kill,closest,diff in definitions:
    rec=dict(idea_id=ident,title=title,summary=hyp,hypothesis=hyp,contribution_type=typ,
      source_basis=closest+['research-report.md','observation-probe.json'],
      assumptions=['条件仅来自推理可用Y/M及已知或已校准退化','机制能够在自然或物理合理诊断中检验'],
      novelty_questions=['最近完整方法是否已有相同条件、作用机制和训练/评估接口'],
      minimum_validation=val,disconfirming_evidence=kill,kill_criteria=kill,closest_work=closest,
      differentiation=diff,scores=dict(zip(['clarity','novelty_signal','feasibility','testability','significance'],scores)),
      risk=risk,estimated_effort=effort,status=status,rationale=diff,
      blocking_questions=['全文近邻覆盖仍不完整；未有自然场景机制和训练证据'])
    records.append(rec)
write(IDEA/'idea-scores.json',json.dumps({'topic':'当前CAVE监督baseline的几何条件融合',
 'scoring_scale':'1-5','ideas':records},ensure_ascii=False,indent=2))
bank='# Idea Bank\n\n## Generation Setup\n\n八候选来自当前结构、既有研究和定向查新，全部为探索假设。并非随机头脑风暴或性能承诺。\n\n## Candidate Ideas\n'
for rec in records:
    bank+=f"\n### {rec['idea_id']} {rec['title']}\n\n- 状态：{rec['status']}。假设：{rec['hypothesis']}\n- 最近工作：{', '.join(rec['closest_work'])}\n- 区别：{rec['differentiation']}\n- 最小检验：{rec['minimum_validation']}\n- 否证/停止：{rec['kill_criteria']}\n- 评分（清晰/新颖信号/可行/可检验/意义）：{list(rec['scores'].values())}。证据：仅探索，无训练。\n"
write(IDEA/'idea-bank.md',bank)
write(IDEA/'rejected-ideas.md','''# Rejected Ideas

## Rejection Categories

- 已有直接近邻：G05联合配准（SA-RFM等）；G06空间小波门控（DDIF/SDP-Net）；G08语义mask（GEWDiff）。
- 缺必要性/资源证据：G07残差扩散暂不入选；G04不确定性待额外查新及预算确认。
- G02尺度条件和G03条件干预保留作不同角度，但本轮只推荐G01。没有为凑三条主线而全量堆叠。
- 如果G01被等信息量经典方法解释、自然机制不存在或部署基无效，保留失败结论，不换指标拯救原主张。
''')
write(IDEA/'landscape-map.md','''# Landscape Map

## Scope

HSI–MSI融合的空间边界/方向作为重建条件；当前CAVE31/RGB×8监督Net。非全领域系统综述。

## Existing Context Used

当前模型/训练/数据生成；../research-report.md；既有GEWDiff迁移、MossFuse共享互补研究。核心灵敏度思路来自前期候选，不宣称完全独立新发现。

## Subareas and Active Approaches

方向正则、细节注入、图几何、结构门控、小波条件扩散、联合配准、子空间响应投影、物理后验与不确定性。

## Closest Work

SDP-Net/GGAT最危险；DDIF/GEWDiff覆盖条件和小波生成；dTV/SA-RFM覆盖方向结构；Matrix-R/DDSR覆盖投影；HSDiff覆盖metamer增强与校准。来源见../sources.json。

## Gap Inventory

待验证缺口为几何条件对局部光谱变化的支持能力；不是声称已有方法都没有谱约束。局部连续RU灵敏度是否比内容门控/秩/固定投影更有解释力仍未知。

## Assumptions and Negative Evidence

当前理想数据不存在粗尺度相容残差；低分辨率局部基可能遗漏真实HR材料。物理一致性不排除共同零空间细节。公式是常见正则化，并非新理论。

## Disconfirmation Targets

同预算普通门控、经典子空间求解、HR-oracle依赖、自然机制缺失、近邻全文同构。失败时缩小或放弃故事。

## Search and Source Limits

截至2026-10-07的定向检索与自审；GGAT/SDP-Net/SA-RFM全文未完整获得。未有独立评审、召回率、训练提升或SOTA证据。
''')
write(IDEA/'selected-idea.md','''# Selected Idea

## Recommendation

仅推荐G01：局部光谱可辨性约束的方向几何条件注入。新颖定位3/5、置信2/5；窄范围探索。

## Idea Summary

观察几何相似区域的RU响应不同，检验张量几何与连续局部响应条件是否减少边界的错误光谱分配。MSI提供空间条件，LR-HSI与R约束修正方向；先确定性附加修正，后再讨论路径重构和生成。

## Differentiation

不能首创几何、方向、小波、门控或投影。必须证明优于SDP式内容控制、固定R投影、同信息量子空间与等参数修正；近邻全文仍有高优先级缺口。

## Assumptions and Disconfirmation

LR可部署基稳定、已知配准和响应、条件无HR泄漏。只有HR基有效、经典求解解释全部收益、自然机制不存在、整体谱误差恶化或全文同构时停止/重构。

## Minimum Validation

自然边界分组诊断＋同几何/同秩/不同响应干预；几何×灵敏度2×2消融，统一训练与选模；PSNR/SAM/边界逐波段误差、观测误差和成本。现阶段无训练提升。

## Risks and Blocking Questions

NQ1–NQ4见../novelty-report.md；基估计、旁路和过度抑制风险。新增旁路不等于全模型受控。真实几何弱支持不代表应全部清零。

## Handoff Notes

证据级别exploratory；同一研究者自审。文献及既有候选告知，合成诊断用于恢复边界。定位核查已做，但先补完整近邻和自然诊断再冻结确认性方案。下一技能research-novelty-review（补缺口）；之后才research-experiment-plan。完整方法与实验细节见../research-report.md第5–7节。
''')
write(IDEA/'ideation-decision.json',json.dumps(dict(decision='proceed_to_novelty_review',selected_idea_ids=['G01'],
 confidence='low',novelty_check_status='reviewed',rationale='定向自审暂定3/5，需补全文和经典对照，尚未冻结确认性主张。',
 next_skill='research-novelty-review',required_handoffs=['selected-idea.md','idea-scores.json','landscape-map.md','../novelty-decision.json'],
 limits=['exploratory only; no training gains','self-review, material fulltext gaps','synthetic diagnostics do not establish natural failures']),ensure_ascii=False,indent=2))
print('Saved report, source/search records, eight-candidate ideation pack and bounded novelty decision.')
