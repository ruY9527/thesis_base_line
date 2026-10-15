"""Create auditable planning records; no model training or benchmark execution."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def write(name,text):
    (ROOT/name).write_text(text,encoding='utf-8')
def dump(name,value):
    write(name,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

specs=[
 ('B1',['C1'],'算子、模式与旋转不变性检查','基线R/D、人工粗/细及谱可见/零模式；不使用自然指标',
  ['MSI-only','joint support','classical restricted ridge'],['spatial scale','spectral visibility'],
  ['operator residual','basis rotation error','joint spectrum'],
  '数学定义和实现一致；保留比较者可胜的无噪声满秩情形','数值容差按float64设1e-10，非性能门槛','算子或表示有错则先修复；反例无效则修改主张'),
 ('B2',['C1'],'自然场景机制诊断','CAVE训练/验证观测＋独立数据；冻结evaluator，显式scene manifest',
  ['rank and texture predictor','RU-only predictor','joint-support predictor'],['support grouping','template bandwidth'],
  ['scene-level boundary SAM','mode error','incremental predictive value'],
  '联合支持提供超出秩/纹理/RU/尺度ridge的解释，外部数据方向一致','解释增量须场景配对区间不跨零；预实验前确定模型与复杂度','不能解释自然失稳则收缩机制主张'),
 ('B3',['C1','C2'],'共享骨干机制消融','分别supervised及GT-hidden；观测条件完全相同，3seeds',
  ['same-capacity learned geometry gate','RU-only gate','joint physical gate','same-information learned joint gate','scale-adaptive ridge'],['gate rule','geometry condition'],
  ['PSNR','SAM','boundary SAM','spectral gradient error','runtime'],
  '边界收益且整体不劣，胜过同信息量学习门控或明确简化理由','boundary SAM >=5%相对改善；PSNR下降<=0.1dB；整体SAM恶化<=1%；规划阈值尚待冻结','只胜弱对照或靠额外损失则不能主张支持度机制'),
 ('B4',['C2'],'三类强方法公平比较','每路线独立比较预算与checkpoint；测试GT仅供冻结evaluator',
  ['geometry-gated neural family','local low-rank physical family','observation-self-supervised family'],['method family'],
  ['scene PSNR/SAM/ERGAS','boundary SAM','offline/scene adaptation/inference time'],
  '在本任务明确范围内胜过或匹配合理强对照并展示关键边界优势','沿用B3质量门槛，额外报告Pareto权衡','输给强方法则收缩为模块或诊断贡献'),
 ('B5',['C1','C2'],'失败边界与无真值尺度迁移','预定noise/SRF/PSF/basis perturbations；真实无HR数据辅助评价',
  ['geometry-gated neural family','local low-rank physical family','observation-self-supervised family'],['noise','operator error','basis coverage','scale'],
  ['boundary and mode errors','held-out observation prediction','consistency','failure frequency'],
  '揭示明确的有效范围；两路线结果分开解释','不得用无参考质量分数替代HR谱精度；不要求所有setting胜出','算子错误下失效则限制结论；原生尺度失败则不能宣称无真值迁移成功')
]
blocks=[]
for i,(bid,cids,why,setting,systems,variables,metrics,success,effect,failure) in enumerate(specs):
    blocks.append(dict(block_id=bid,paper_role='main',claim_ids=cids,decision_gate_id=f'G{i+1}',
        anti_claims_ruled_out=['增益来自容量、尺度正则或额外观测信息'],why_this_block_exists=why,
        dataset_split_task=setting,systems_compared=systems,fixed_factors=['observation access','loss within each route','selection rules','budget'],
        variable_factors=variables,metrics=metrics,setup_details='当前CAVE CNN起点；局部谱基和模板定义先冻结；训练实现未完成',
        seeds=3,success_criterion=success,minimum_effect_size=effect,failure_interpretation=failure,
        expected_output_artifact=f'planned_results/{bid}.json',compute_budget='先小规模pilot，按实际硬件测量；本轮没有启动训练',
        dependencies=[] if i==0 else [f'B{i}'],priority='must-run',current_evidence_class='exploratory',evidence_class='exploratory',
        selection_rule='预先manifest及案例规则；当前历史测试已被查看，不充当新盲测',
        non_vacuity_check='无噪声满秩下不收缩解可胜出；同信息学习门控可胜；保留这些情形',
        complete_outcome_accounting='保留每scene/seed的负结果、错误、skip、超时和技术重试；不得仅存best',
        hidden_information_controls='条件只能读观测；GT-hidden路线禁止GT用于条件/停止/选择；留出观测也从条件移除',
        independence_requirements=['最终外部scene未参与方法选择','评价与条件的数据访问分离'],
        independence_evidence='当前没有独立复现或审查，仅规划要求',predecessor_failures=['RU-only不足以表示完整融合可辨性','原测试选择历史需保留'],
        execution=dict(mode='manual',entrypoint=None,declared_inputs=['deep-research-report.md','joint_support_probe.py'],
                       declared_evaluator_artifacts=[],required_outputs=[dict(path=f'planned_results/{bid}.json',kind='metrics')]),
        lineage_policy=dict(allowed_relations=['baseline','replication','ablation','parameter_variation','negative_control','sensitivity','alternative_hypothesis','technical_retry'])))
claims=[]
for cid,text,links,falsifier in [
 ('C1','局部联合支持能解释超出rank/texture/RU/scale-ridge的自然几何失稳',['B1','B2','B3','B5'],'控制纹理/尺度后无解释增量'),
 ('C2','额外物理几何残差在两训练路线中改善边界光谱归属而整体不劣',['B3','B4','B5'],'强同信息对照等效或更好，或整体谱质量下降')]:
    claims.append(dict(claim_id=cid,priority='primary',claim=text,why_it_matters='决定模块是否有独立研究价值',
        minimum_convincing_evidence='场景级独立数据和强对照，3seeds；GT-hidden流程冻结后评价',
        anti_claim='只是新增参数、额外物理损失或普通尺度ridge',falsifier=falsifier,decision_if_unproven='reframe',
        linked_blocks=links,current_evidence_class='exploratory',evidence_class='exploratory',decision_rule='按对应gate，不以执行完成代替科学支持',
        loss_contract='supervised与GT-hidden损失分别对齐，报告全部质量和成本',falsification_test=falsifier,
        selection_history=['前轮基线测试输出已查看；本轮仅检索及人工算子反例，未看候选训练结果'],
        predecessor_failures=['MSI-only门控不能代表HSI对粗空间零光谱方向的支持']))
dump('claim-map.json',claims);dump('run-blocks.json',blocks)
gates=table(['Gate ID','Opens after','Decision question','Proceed if','Revise if','Stop if','Owner'],
    [[b['decision_gate_id'],b['block_id'],b['why_this_block_exists'],b['success_criterion'],b['failure_interpretation'],'核心假设被强对照否定则停止扩张','研究者'] for b in blocks])
write('decision-gates.md','# Decision Gates\n\n'+gates)
plan='# Experiment Plan\n\n## Context\n\nStandalone exploratory规划；两路线同等；没有训练结果或独立审计。详见 deep-research-report.md。历史选择：CAVE既有测试已查看；人工反例是本轮设计验证。\n\n## Claim Map\n\n'
plan+=table(['Claim ID','Type','Claim','Falsifier'],[[c['claim_id'],'primary',c['claim'],c['falsifier']] for c in claims])
plan+='\n## Experimental Storyline\n\n'+table(['Block','Role','Paper placement','Why it exists'],[[b['block_id'],'mechanism/control','main',b['why_this_block_exists']] for b in blocks])
plan+='\n## Non-Vacuity Preflight\n\n人工双零模式与粗空间零谱模式已执行；无噪声满秩伪逆和同信息门控可胜，禁止强制所有setting获益。仅B1数值子项完成，自然机制待测。\n\n## Experiment Blocks\n\n'
for b in blocks:
    plan+=f"### {b['block_id']}\n\n"+'\n'.join(f'- {k}: '+('; '.join(map(str,v)) if isinstance(v,list) else str(v)) for k,v in b.items() if k not in {'execution','lineage_policy'})+'\n\n'
plan+='## Run Order\n\n'+table(['Order','Block','Purpose','Dependency','Gate ID','Stop / go gate','Est. cost'],[[i+1,b['block_id'],b['why_this_block_exists'],', '.join(b['dependencies']),b['decision_gate_id'],b['failure_interpretation'],b['compute_budget']] for i,b in enumerate(blocks)])
plan+='\n## Decision Gates\n\n'+gates+'\n## Risks and Confounds\n\nGT条件泄漏、谱基遗漏、patch退化边界、噪声权重、尺度ridge等价、旧测试选择。措施和指标定义见主报告5–9节。规划效应门槛未经结果调优，开始训练前需要冻结，不是已预注册结论。\n'
write('experiment-plan.md',plan)
write('experiment-tracker.md','# Experiment Tracker\n\n'+table(['Run ID','Block ID','Gate ID','Purpose','Priority','Status','Owner','Dependency','Output artifact','Notes'],
    [[f'R{i+1:03}',b['block_id'],b['decision_gate_id'],b['why_this_block_exists'],'must-run','analyzed' if i==0 else 'planned','研究者',','.join(b['dependencies']),b['expected_output_artifact'],'B1仅数值子项完成，完整gate仍待检查' if i==0 else '未执行'] for i,b in enumerate(blocks)]))
write('execution-bridge.md','# Execution Bridge\n\n计划交接；manual表示训练实现尚未完成，不是可以直接启动的命令。\n\n'+''.join(
    f"### {b['block_id']}\n\n- Claim IDs: {', '.join(b['claim_ids'])}\n- Decision gate ID: {b['decision_gate_id']}\n- Current/requested evidence class: exploratory\n- Inputs: 显式split及观测算子、候选头、强对照、冻结evaluator\n- Outputs: {b['expected_output_artifact']}\n- Hidden information: GT不可用于条件；GT-hidden不可用于停止/选择\n- Gate: {b['success_criterion']}\n- Failure: {b['failure_interpretation']}\n- Retention: 全scene/seed与失败、skip和重试\n\n" for b in blocks))

pc=[]
def paperclaim(pid,text,kind,mode,status,action,cids,figs,cites,limitations,missing,evidence):
    pc.append(dict(paper_claim_id=pid,claim=text,claim_type=kind,evidence_mode=mode,support_status=status,
                   manuscript_action=action,required_assurance_class='exploratory' if mode=='empirical' else 'none',
                   source_claim_ids=cids,audit_ids=[],audit_exclusions=[],evidence_artifacts=evidence,
                   planned_sections=['Introduction','Method','Experiments','Limitations'],exhibit_ids=figs,citation_need_ids=cites,
                   limitations=limitations,missing_evidence=missing,scope='已配准HSI-MSI；固定局部模型与声明的观测；两训练路线分别解释',
                   rationale='按证据边界，不能把计划或数值反例晋升为网络性能'))
paperclaim('P0','几何方向、局部低秩、观测融合与学习门控已有先例','context','citation','supported','assert',[],[],['N1'],['最近论文存在全文缺口'],[],['deep-research-report.md'])
paperclaim('P1','固定局部线性模型的法矩阵零空间为两类受限观测零空间交集','supporting','theoretical','supported','assert',[],['F1'],['N2'],['标准线性代数事实，不构成首次理论创新'],[],['deep-research-report.md','joint_support_probe.py'])
paperclaim('P2',claims[0]['claim'],'primary','empirical','blocked','omit',['C1'],['F2'],[],['尚无自然场景证据'],['B2及强机制对照和结果审计'],[])
paperclaim('P3',claims[1]['claim'],'primary','empirical','blocked','omit',['C2'],['F3','T1'],[],['尚未训练候选模型'],['B3-B5两路线实验及结果审计'],[])
paperclaim('P4','观测拟合不能单独证明共同零空间内的HR谱细节正确','limitation','limitation','supported','limitation',[],['F1'],['N2'],['先验能选择解，但不增加观测信息'],[],['deep-research-report.md'])
dump('claim-evidence-bindings.json',dict(schema_version='1.0',paper_id='joint-support-directional-fusion',identity_version=1,status='complete',claims=pc))
headers=['Paper claim ID','Claim','Type','Evidence mode','Support status','Manuscript action','Required assurance','Source claim IDs','Audit IDs','Excluded audit IDs','Planned sections','Exhibit IDs','Citation need IDs','Limitation']
rows=[[p['paper_claim_id'],p['claim'],p['claim_type'],p['evidence_mode'],p['support_status'],p['manuscript_action'],p['required_assurance_class'],','.join(p['source_claim_ids']),'','',';'.join(p['planned_sections']),','.join(p['exhibit_ids']),','.join(p['citation_need_ids']),';'.join(p['limitations'])] for p in pc]
write('claims-evidence-matrix.md','# Claims-Evidence Matrix\n\n'+table(headers,rows))
write('figure-plan.md','# Figure Plan\n\n'+table(['Exhibit ID','Purpose','Paper claim IDs','Priority','Status','Notes'],[
 ['F1','粗/细与谱可见/零模式反例','P1,P4','must-run','numeric only','joint-support-probe.json；图尚未制作'],
 ['F2','支持分组与谱误差的自然机制','P2','must-run','planned','同时展示对照胜出案例'],
 ['F3','物理残差头与两路线信息流','P3','must-run','planned','明确GT仅训练标签或评价'],
 ['T1','同信息消融与两路线主表','P3','must-run','planned','不能填入预测分数']]))
write('citation-plan.md','# Citation Plan\n\n'+table(['Citation need ID','Paper claim IDs','Citation need','Source status','Notes'],[
 ['N1','P0','HySure、dTV、CLoRF、SA-RFM及SDP/GGAT定位','verified with depth limits','主报告原始链接；SDP作者代码静态复核；部分全文缺口'],
 ['N2','P1,P4','双观测与标准受限最小二乘事实','standard derivation and primary sources','HySure全文；不宣称新定理']]))
write('paper-plan.md','# Paper Plan\n\n## Header\n\n- Paper ID: joint-support-directional-fusion\n- Identity version: 1\n- Plan status: complete（结构计划完成，实验主张仍blocked）\n- Working title: Joint-Observation-Supported Directional Geometry Injection for Hyperspectral–Multispectral Fusion\n- Operating mode: standalone\n- Target venue: 机制结果及应用证据完成后再定TIP/TGRS/IJCV取向\n\n## Structure\n\n1. Introduction：边界位置与光谱细节归属；承认最近门控几何工作。\n2. Related Work：几何条件、局部表示与物理算子、自监督。\n3. Method：受限观测定义、残差注入位置、共同条件与两种训练目标。\n4. Experiments：先机制，再同信息消融、强比较及两路线。\n5. Limitations：谱基遗漏、尺度ridge简化、算子误差、共同零空间。\n\n## Section Notes\n\nP0/P1可以按范围陈述；P2/P3仅研究假设，不可写已实现或显著提升。自然例子按预定规则选择；先无噪声理想对照，再困难情况；真实无真值数据不声称HR谱真值。核心叙事见主报告7节。\n\n## Evidence Boundary\n\nclaim-evidence-bindings.json为主张权威。没有results-audit.json，因为没有训练结果；blocked主张omit。结构验证不是科学验证或独立审稿。\n')

previous=json.loads((ROOT.parent/'geometry_condition_injection_20261007/sources.json').read_text(encoding='utf-8'))
records=previous['records']
updates={'dTV':'完整arXiv v4：END-OF-PAPER:a4aac5004727','SA-RFM':'完整会议PDF含附录：END-OF-PAPER:4b86a82f8070','SDP-Net':'摘要/引言＋固定提交作者模型全文与train函数；完整论文缺口'}
for r in records:
    r['origin']='previous bounded research; evidence not silently promoted'
    if r['title'] in updates:r['read_depth']=updates[r['title']]
records.extend([
 dict(title='HySure',venue='TGRS 2015',url='https://doi.org/10.1109/TGRS.2014.2375320',read_depth='完整作者预印本：END-OF-PAPER:ae722f549f12'),
 dict(title='CLoRF',venue='IJCV 2025',url='https://doi.org/10.1007/s11263-025-02540-1',read_depth='完整arXiv v2：END-OF-PAPER:f844f69e099d；核心train脚本，正式版差异未逐页核对'),
 dict(title='Local low rank HSI fusion',venue='TIP 2016',url='https://doi.org/10.1109/TIP.2015.2496263',read_depth='正式元数据/摘要；助手最后取得的PDF实际是2014扩展报告，不能视为正式版全文'),
 dict(title='Local low rank extended report',venue='HAL extended report 2014',url='https://hal.science/hal-00960076v2',read_depth='完整正文与参考文献：END-OF-PAPER:358766021635；显式小奇异值噪声放大与ridge逆解先例'),
 dict(title='Local low rank coupled unmixing',venue='TGRS 2017',url='https://doi.org/10.1109/TGRS.2017.2718728',read_depth='发现候选、摘要；全文获取失败，不用于细节否定'),
 dict(title='STEREO',venue='TSP 2018',url='https://doi.org/10.1109/TSP.2018.2876362',read_depth='发现与理论挑战候选；HTML为14行stub，已降级；不用于细节综合'),
 dict(title='IR&ArF',venue='TIP 2025',url='https://doi.org/10.1109/TIP.2025.3551531',read_depth='发现候选；SA-RFM参考文献交叉验证，不深读'),
 dict(title='Deep SURE',venue='TGRS 2022',url='https://doi.org/10.1109/TGRS.2022.3215902',read_depth='CLoRF参考文献发现；可作后续对照，不推断方法细节')])
dump('sources.json',dict(as_of='2026-10-07',scope='targeted deep research, not a systematic review or exhaustive novelty proof',records=records))
manifest=json.loads((ROOT/'retrieval-manifest.json').read_text(encoding='utf-8'))
for r in manifest:
    r['human_review_status']='full text read' if r['name'] in {'clorf','hysure','dtv','sa_rfm'} else ('stub downgraded; not full text' if r['name']=='stereo' else 'full text access gap')
dump('reading-ledger.json',manifest)
late=json.loads((ROOT/'fetch-local-tip.json').read_text(encoding='utf-8-sig'))
late['human_review_status']='full 2014 extended report read; NOT the requested 2016 journal version'
late['human_verified_version']='HAL hal-00960076v2, 19 March 2014; five authors; 14 PDF pages including cover'
manifest.append(late)
dump('reading-ledger.json',manifest)
write('search-and-access-log.md','# 检索与访问记录\n\n日期2026-10-07。定向检索，不是系统综述。查询覆盖geometry/graph/gating、local low rank/observability、self-supervision和近期registration/flow matching。发现集及来源深度见sources.json；前轮候选按原深度复用。\n\n标准research-papers fetch_and_parse助手执行合法全文级联。OpenAlex未找到部分OA，Semantic Scholar出现429，publisher全文存在访问限制；未提供真实联系邮箱，因此Unpaywall跳过，没有虚构邮箱。pypdf用于原PDF解析失败时的格式保留回退。HTML figure转换出现转义异常，仍取得正文者按实际覆盖读取；没有将返回ok当作全文保证。STEREO只有stub，人工降级。\n\n全文读取结束标记：ae722f549f12、a4aac5004727、f844f69e099d、4b86a82f8070。长输出曾截断的CLoRF方法/结果与dTV开头已补读。图表数值有PDF解析错列风险，主报告不做跨论文PSNR排名。正式期刊版本与预印本差异在sources记录。作者代码的提交和sha256见author_code/*/manifest.json；未执行作者训练。\n\n不作全国/全球首创结论；SDP/GGAT缺口是后续投稿前要关闭的最近邻风险。计划结构校验与本研究者自审不属于独立科学验证。\n')
print('Planning pack and source/reading ledgers written. No model training.')
