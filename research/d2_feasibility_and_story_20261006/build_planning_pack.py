"""Materialize exploratory planning views; does not execute planned experiments."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = HERE / "experiment-plan"
PAPER = HERE / "paper-plan"
NOV = HERE / "novelty-review"
NOV.mkdir(exist_ok=True)


def write(path, value):
    path.write_text(value, encoding="utf-8")


def dump(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(map(str,r)) + " |" for r in rows) + "\n"


history = ["Recent-paper reframing after prior-art inspection; broad interaction framing downgraded.",
           "Previous interpolation and phase outcomes inspected; no D2 training outcomes exist.",
           "Current numeric preflight is synthetic algebra only, not natural-image evidence."]
failures = ["Generic interaction, projection and synergy training have substantial prior art.",
            "Unrestricted joint-null targets are not identifiable from the two observations.",
            "Uploaded supervised checkpoint binding and test-based selection are unsuitable as the main comparator."]
claims = [
    dict(claim_id="C1",priority="primary",claim="在声明算子与生成先验的有限容量任务内，指定分离位置对正确联合响应存在可重复的取舍。",
         why_it_matters="区分表示几何、功能配合和不可辨识性。",minimum_convincing_evidence="合理权重、相同数据和角色目标的多种子控制；弱约束可以竞争。",
         anti_claim="效应只由极强损失、任务不可学、辅助头塌缩或额外数据造成。",falsifier="合理范围没有响应取舍，或弱约束解释全部结果。",
         decision_if_unproven="reframe",linked_blocks=["B1","B2","B3","B5"],
         decision_rule="仅在受控任务有效且公平控制不能解释时保留；自然关联另由 C2 判断。",
         loss_contract="完整四幅误差、混合误差、加性零目标绝对误差、角色误差与失败记录均保留。",
         falsification_test="加性任务、可加参照、无分离角色辅助、弱直接分离和零空间冲突。"),
    dict(claim_id="C2",priority="supporting",claim="接口与内部组合训练在不使用真实HR训练标签的情况下，超越同样数据与预算的普通训练并改善未见场景。",
         why_it_matters="机制需要转化为 PSNR/RMSE/SAM 的实际收益。",minimum_convincing_evidence="二因素消融、同四元数据普通L1、原架构自监督组与自然场景多种子比较。",
         anti_claim="收益来自额外目标、计算、参数、测试选模或改变退化与归一化。",falsifier="没有自然收益，或普通内部训练、弱约束及相同预算可完全解释。",
         decision_if_unproven="defer",linked_blocks=["B1","B4","B5"],
         decision_rule="只在冻结配置后的自然评估与同预算强对照成立时保留任务改进主张。",
         loss_contract="同checkpoint完整PSNR、RMSE、SAM、每场景与种子差值、训练成本和全部失败。",
         falsification_test="同样完整四元目标无混合损失、真实内部任务预算替代、CAVE冻结测试、先验偏差。")]
for c in claims:
    c.update(current_evidence_class="exploratory",evidence_class="exploratory",selection_history=history,predecessor_failures=failures)
dump(EXP / "claim-map.json", claims)

settings = [
    ("B1",["C1","C2"],"协议与非空预检",[],
     ["原NumPy R/D","候选可微R/D","受限族oracle","零空间冲突","加性目标"],
     ["边界/核/相位实现","标签与内部数据路径"],
     ["算子最大绝对误差","梯度与形状","冲突观测差","非零目标幅度"],
     "数值代数已通过；可微D一致性、自然数据标签隔离和内部任务仍待实现。",
     "float64算子差<1e-10、float32候选差<1e-5，非零可用梯度，split无重叠，目标与输入角色明确。",
     "算子不一致则修实现；受限族oracle失败则修先验；标签隔离失败不得进入主训练。",1),
    ("B2",["C1"],"受控任务与分离效应",["B1"],
     ["原联合Net无分离","同容量可加参照","角色辅助无分离G0","原表示直接分离"],
     ["分离强度0/0.01/0.1/0.3梯度比","加性或乘法目标"],
     ["E_mix","四幅完整误差","加性绝对差分","角色误差","非塌缩指标"],
     "先1种子筛查，3种子主档；明确生成先验，按基础谱/空间模式划分，所有模型同数据普通完整L1。",
     "联合参照能学；合理分离有可重复取舍且不是极强损失才出现；不将OOD普遍失败当损害。",
     "无合理效应则不保留解耦损害因果主张；不可学则调整诊断任务而非修改期待结论。",3),
    ("B3",["C1"],"约束接口必要性",["B2"],
     ["直接分离","投影分离","弱直接分离","去互补负项","角色辅助无分离"],
     ["分离施加位置及公平简单替代"],
     ["E_mix","完整误差","角色重建","投影方差/有效秩","训练参数与成本"],
     "相同角色目标、相同投影头参数与公共初始化；角色监督读取辅助映射以防常数逃避。",
     "响应相对改善约10%且角色/完整误差没有明显恶化；简单替代不能解释全部效果。",
     "弱约束同样好则优先简单方案；头塌缩则接口机制失败，保留负结果。",3),
    ("B4",["C2"],"训练机制与自然指标",["B1","B3"],
     ["B0-obs","B0-int","角色辅助G0","A0C0","A1C0","A0C1","A1C1","同预算真实内部任务","简单局部/高频误差加权"],
     ["分离位置A","是否混合目标C"],
     ["平均波段PSNR","RMSE×255","SAM度","E_mix","每场景差值","峰值显存/时间"],
     "四元数据/完整L1/四次前向预算相同；训练观测16、验证观测4、冻结测试12；主3种子。",
     "相对原架构内部任务与同四元普通训练PSNR暂以+0.10dB推进，RMSE不恶化、SAM差<=+0.02度，且重复方向一致。",
     "只合成有效则降为诊断；A无价值则删除A；只胜坏约束不算baseline收益；波动覆盖效应则不确定。",3),
    ("B5",["C1","C2"],"近期强对照与先验边界",["B4"],
     ["冻结最终候选","MossFuse同协议","分解或关系家族CLoRF/SHOTUN择一","公平简单最佳组"],
     ["已声明先验偏差","噪声/退化敏感性","另一个数据集"],
     ["PSNR/RMSE/SAM","场景与种子差值区间","失败比例","预算"],
     "关键结论建议5种子；第二数据集来源未冻结，不启动；场景优化与摊销模型成本单列。",
     "强对照与另一个声明范围内场景集合重复成立，机制有效范围与失败先验清楚；近邻检索补全。",
     "收益不重复或同构先行工作覆盖则降级定位；不可解释负结果不可省略。",5),
]
blocks=[]
for bid,cids,purpose,deps,systems,var,metrics,setup,success,negative,seeds in settings:
    output=f"research/d2_feasibility_and_story_20261006/future-results/{bid}/summary.json"
    b=dict(block_id=bid,paper_role="appendix" if bid=="B1" else "main",claim_ids=cids,
           decision_gate_id="G"+bid[1:],anti_claims_ruled_out=[c['anti_claim'] for c in claims if c['claim_id'] in cids],
           why_this_block_exists=purpose,dataset_split_task="受控合成族轨道T；CAVE训练16/验证4/测试12轨道N，清单见split-manifest.csv。",
           systems_compared=systems,fixed_factors=["原R/D及采样相位","数据与普通完整目标","公共权重复制初始化","冻结选模与指标"],
           variable_factors=var,metrics=metrics,setup_details=setup,seeds=seeds,success_criterion=success,
           minimum_effect_size="探索阈值见主文第8节；不得当显著性证据。",failure_interpretation=negative,
           expected_output_artifact=output,compute_budget="B1 CPU小检查；其余先实测2000步1种子，暂定10000步全组一致；不承诺运行时长。",
           dependencies=deps,priority="must-run" if bid!="B5" else "nice-to-have",current_evidence_class="exploratory",evidence_class="exploratory",
           selection_rule="按固定场景清单与生成模式划分；未看D2结果；不按测试错误选择案例。",
           non_vacuity_check="代数先验族可恢复、加性参照可胜已检查；网络可学性与可微D未检查。",
           complete_outcome_accounting="记录planned/completed/failed/skipped/timeout/retry；保留OOM、NaN、空分母及所有种子，重试与父run关联。",
           hidden_information_controls="模型只读观测与声明生成先验；测试HR只进冻结评估器，训练不读取HR；legacy归一化元数据另行披露。",
           independence_requirements=[],independence_evidence="本轮为同一代理自审；未来评估暂不宣称独立。",
           operational_threat_model="探索研究的标签泄漏、结果挑选和预算失配。",operational_harms="错误科研结论与无效计算投入。",
           predecessor_failures=failures,
           execution=dict(mode="manual",entrypoint=None,declared_inputs=[{"path":"model.py","snapshot":"digest-at-start"},{"path":"utils.py","snapshot":"digest-at-start"},{"path":"research/d2_feasibility_and_story_20261006/experiment-plan/split-manifest.csv","snapshot":"digest-at-start"}],
                          declared_evaluator_artifacts=[{"path":"calculate_metrics.py","snapshot":"digest-at-start"}],required_outputs=[{"path":output,"kind":"metrics_and_failures"}]),
           lineage_policy=dict(allowed_relations=["baseline","replication","ablation","parameter_variation","negative_control","sensitivity","alternative_hypothesis","technical_retry"]),
           activation_rule="人工检查前驱科学门槛后才激活；完成文件或通过结构校验不等于通过门槛。")
    blocks.append(b)
dump(EXP / "run-blocks.json", blocks)

manifest=list(csv.DictReader((ROOT/'research/baseline_comparison_20261006/dataset_manifest.csv').open(encoding='utf-8-sig')))
train=sorted((r for r in manifest if r['split']=='train'),key=lambda r:r['scene'])
assert len(train)==20
val={r['scene'] for r in train[-4:]}
with (EXP/'split-manifest.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=['planned_split','legacy_split','scene','path','sha256'])
    writer.writeheader()
    for r in manifest:
        writer.writerow(dict(planned_split='test' if r['split']=='test' else 'validation' if r['scene'] in val else 'train',legacy_split=r['split'],scene=r['scene'],path=r['path'],sha256=r['sha256']))

gate_rows=[]
for b in blocks:
    gate_rows.append([b['decision_gate_id'],b['block_id'],b['why_this_block_exists'],b['success_criterion'],"技术失败修复后再检查；不确定则缩小主张",b['failure_interpretation'],"研究者人工检查；无自动授权"])
gates="# Decision Gates\n\n各门槛尚未通过。代数预检已完成，不代表 B1 全部完成；主文第8节阈值仅是探索决策。\n\n"+table(['Gate ID','Opens after','Decision question','Proceed if','Revise if','Stop if','Owner'],gate_rows)
write(EXP/'decision-gates.md',gates)
tracker="# Experiment Tracker\n\n不伪造运行状态；仅当前预检为已有证据，网络块未运行。\n\n"+table(['Run ID','Block ID','Gate ID','Purpose','Priority','Status','Owner','Dependency','Output artifact','Notes'],[[f"R{i:03d}",b['block_id'],b['decision_gate_id'],b['why_this_block_exists'],b['priority'],'planned','研究者',','.join(b['dependencies']) or '协议检查',b['expected_output_artifact'],'B1部分代数检查已有，其余未运行' if i==1 else '科学前驱门槛未通过'] for i,b in enumerate(blocks,1)])
write(EXP/'experiment-tracker.md',tracker)
plan="""# Experiment Plan

## Context

Operating mode: standalone。当前及请求证据类别均 exploratory；不启动GPU训练、不宣称独立确认。
主问题为特定分离约束的功能取舍及其无HR训练标签的保护机制。上游为近期D2报告、baseline审计和本轮代数预检。局部关系B暂缓，最多两个核心主张、五个实验块、三个baseline家族。
先前查看过监督日志与插值结果；D2尚无网络结果。宽泛协同新意不足、联合零空间不可辨识及原日志选模问题必须保留。

## Claim Map

"""+table(['Claim ID','Type','Why it matters','Minimum convincing evidence','Anti-claim to rule out','Falsifier','Decision if unproven'],[[c['claim_id'],c['priority'],c['why_it_matters'],c['minimum_convincing_evidence'],c['anti_claim'],c['falsifier'],c['decision_if_unproven']] for c in claims])+"\n## Experimental Storyline\n\n"+table(['Block','Role','Paper placement','Why it exists'],[[b['block_id'],'诊断/接口/机制/边界',b['paper_role'],b['why_this_block_exists']] for b in blocks])+"""
## Non-Vacuity Preflight

5组原R/D代数已检查：受限先验族可恢复；无先验冲突不可恢复；加性参照可胜；目标非零。结果在../preflight-results.json，不是网络证据。
可微D匹配、网络可学性、标签隔离仍pending；当前门槛 exploratory-only，不宣称整个预检通过。
样本按生成模式和场景划分，错误、跳过、NaN、OOM、超时、重试均计入；普通训练可以胜，不人为削弱它。

## Experiment Blocks

"""
for b in blocks:
    plan+=f"### {b['block_id']}：{b['why_this_block_exists']}\n\n"
    for label,key in [('门槛','decision_gate_id'),('设置','setup_details'),('成功条件','success_criterion'),('负结果解释','failure_interpretation'),('预算','compute_budget'),('预期输出','expected_output_artifact')]:
        plan+=f"- {label}：{b[key]}\n"
    plan+='- 对照：'+ '；'.join(b['systems_compared'])+'\n- 指标：'+'；'.join(b['metrics'])+'\n\n'
plan+='## Run Order\n\n'+table(['Order','Block','Purpose','Dependency','Gate ID','Stop / go gate','Est. cost'],[[i,b['block_id'],b['why_this_block_exists'],','.join(b['dependencies']) or '无前驱',b['decision_gate_id'],'人工科学检查后激活',b['compute_budget']] for i,b in enumerate(blocks,1)])+'\n## Decision Gates\n\n详见 decision-gates.md；B1数值检查只覆盖其中部分。\n\n## Risks and Confounds\n\n联合零空间先验、投影塌缩、额外监督、普通内部训练、强损失人为损害、尺度/退化错配、测试选模、旧归一化元数据。控制规则见主文第3/5/6/8节。\n'
write(EXP/'experiment-plan.md',plan)
bridge="# Execution Bridge\n\n这是后续实现交接，尚无训练入口，不虚构可运行训练命令。\n\n## Block Hand-off\n\n"
for b in blocks:
    bridge+=f"### {b['block_id']}\n\n- Claim IDs: {', '.join(b['claim_ids'])}\n- Decision gate ID: {b['decision_gate_id']}\n- Current evidence class: exploratory\n- Requested evidence class: exploratory\n- Expected implementation entrypoint: 新隔离目录中的训练/诊断入口，尚待实现。\n- Expected command or notebook: 本轮仅feasibility_preflight.py可运行；主训练不可提交。\n- Inputs required: model.py、utils.py、split-manifest.csv及只含观测的数据接口。\n- Declared input snapshot paths: 与run-blocks.json相同；不是不可变保证。\n- Declared evaluator snapshot paths: calculate_metrics.py。\n- Output artifacts to produce: {b['expected_output_artifact']}，以及config、per_scene、per_seed、失败清单。\n- Auditor-facing checks: {b['success_criterion']}\n- Intended lineage relation: baseline、ablation、negative_control或replication，遵守声明。\n- Parent run ID or rationale: 前驱科学门槛检查后的新运行；技术重试指向失败父运行。\n- Hidden information unavailable to the evaluated system: 测试HR及其误差、研究者偏好结果。\n- Failure, skip, null, timeout, and retry states to retain: 全部保留，禁止只保留最佳seed。\n- Idempotency and restart requirements: 每次运行唯一ID；同配置恢复记录checkpoint与父运行，配置变化创建新ID。\n- Known blockers: 可微算子与训练入口未实现，前驱门槛尚未通过；第二数据集未冻结。\n- Downstream activation: 人工检查科学结果，不因文件存在或技术完成自动激活。\n\n"
write(EXP/'execution-bridge.md',bridge)

# Novelty pack: retain limited positioning, disclose latest counterevidence.
prior=json.loads((ROOT/'research/d2_recent_2025_2026_review/novelty-decision.json').read_text(encoding='utf-8'))
d=prior.copy()
d.update(narrowest_defensible_positioning="在声明成像算子与结构先验的受限任务中，探索特定分离接口的正确联合响应取舍，以及观测内部组合训练是否超越同数据普通训练并改善自然融合。",
         what_would_change_the_decision="若合理分离没有功能取舍、简单弱约束或同数据普通训练解释全部收益、自然指标不改善或同构近期方法覆盖，则降级或重写；只有强对照与自然证据成立后才扩大定位。",
         literature_context_used=prior['literature_context_used']+['research/d2_feasibility_and_story_20261006/d2_feasibility_steps_motivation_story.md'],
         missing_prior_work=['NQ1: 项目相关投影解耦与任务表示接口的完全同构近邻未排尽。','NQ2: 物理组合、内部目标和零空间辅助训练近邻未系统排尽。','NQ3: 近期IEEE全文及SynCo等协同训练对比仍需深入核验。'],
         claims_to_qualify=prior['claims_to_qualify']+['Joint-response diagnostics are not PID synergy estimates.','SynCo 2026 is preprint counterevidence, not a verified journal/code recommendation.','Only restricted-family recoverability is checked; no new Net metrics.'],
         evidence_scope={'network_training':False,'natural_image_evaluation':False,'numeric_preflight':'../preflight-results.json','scope':'five synthetic algebra families only'},
         next_step='B1协议与可微D、B2受控网络可学性和合理分离效应；不要先堆叠局部关系模块。')
d['literature_assurance']=json.loads(json.dumps(prior['literature_assurance']))
d['literature_assurance']['query_count']=6
dump(NOV/'novelty-decision.json',d)
report=f"""# Novelty Report：D2 可行性再审查

## Contribution Under Review

具体方向为约束接口、受限物理任务联合响应及观测内部训练。先前通用交互与最近主文献已经审查，当前只补关键近邻；没有再次全面综述。

## Claim Decomposition

框架：特定训练取舍，不是解耦普遍有害。方法：投影接口+内部输出组合目标。评价：受限先验族、不可辨识冲突族与加性负对照。经验发现：全部待网络实验。

## Strongest Overlaps

MossFuse的角色分离、CLoRF的空间光谱组合、SHOTUN的关系保留、CDPS的观测先验；CoMM ICLR2025与SynCo2026预印本进一步覆盖通用协同训练。SynCo尚未核实正式期刊/代码，不影响其作为新颖性反证的相关性。

## Novelty-Killing Objections

联合零空间不可辨识；正交不等于无交互；辅助投影可逃避；内部降尺度项目本来已有；多数据/多预算可解释收益；只恢复损害不等于超过原baseline；通用协同训练已有近期先行工作。

## Decision

- Novelty decision rating (1-5): 3
- Impact positioning rating (1-5): 3
- Decision confidence rating (1-5): 3
- Narrowest defensible positioning: {d['narrowest_defensible_positioning']}
- What would change the decision: {d['what_would_change_the_decision']}
- Literature assurance mode: unlinked
- Literature assurance verdict: targeted-search-only-not-systematic
- Unresolved high-priority novelty-critical questions: NQ1, NQ2, NQ3

通用协同或投影模块单独定位约1–2/5；3/5只支持有限探索，不是科学有效性、发表概率或收益认证。当前新代数检查说明边界与构造不空；没有训练新Net。已有负面定位与监督选模风险保留，不重新归零。

## Paper-Review Handoff

同一代理自审，不是独立审稿。主文和实验计划以C1/C2假设绑定；无实证audit的改进主张在paper-plan中blocked/omit，只能用研究问题语言。此前插值A1/A2审计并不支持这些新实验主张，未当作同scope证据引用。
"""
write(NOV/'novelty-report.md',report)
write(NOV/'prior-art-matrix.md',"""# Prior Art Matrix

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
""")
write(NOV/'search-log.md',"""# Search Log

2026-10-06；定向审查、同代理自审。复用前一轮7篇顶刊与2篇会议证据，只补协同/投影及训练监督依据近邻。以下6个查询实际执行：

1. 2025 2026 multimodal disentanglement projection head reconstruction synergy shared private contrastive learning
2. 2025 2026 hyperspectral fusion self supervised null space training spectral spatial consistency
3. "CoMM" "ICLR 2025" synergy
4. 2025 2026 "hyperspectral" "decoupling" "loss" "fusion"
5. "SynCo" "Yarici" github
6. "hyperspectral fusion" "projection" "disentanglement" 2025 2026

打开正式ICLR CoMM记录、作者GitHub及SynCo arXiv全文；SynCo检索未确认作者代码/正式venue，仍列作新颖性近邻。Scientific Reports、JSTARS、非直接HSI融合、未来卷期等结果不作为本轮用户指定顶刊核心。没有系统召回保证。

本机读取近期报告、baseline审计、原算子与数据loader。执行新preflight五组代数：观测冲突、声明先验内可恢复、加性参照可胜、几何/功能与常数头边界。无网络训练、无自然数据新指标。预检与结构校验不能作为实证支持。
""")

# Paper binding: untested empirical claims are explicitly omitted as findings.
pc=[]
def add_pc(pid,text,kind,mode,status,action,source,figs,cites,limitations,missing,artifacts=None):
    pc.append(dict(paper_claim_id=pid,claim=text,claim_type=kind,evidence_mode=mode,support_status=status,
                   manuscript_action=action,required_assurance_class='exploratory' if mode=='empirical' else 'none',
                   source_claim_ids=source,audit_ids=[],audit_exclusions=[],evidence_artifacts=artifacts or [],
                   planned_sections=['Introduction','Methods' if mode!='citation' else 'Related Work'],
                   exhibit_ids=figs,citation_need_ids=cites,limitations=limitations,missing_evidence=missing,
                   scope='已声明先验和算子中的受控任务及CAVE探索性自然评估',
                   rationale='经验主张无同scope审计，不能assert；仅计划检验。' if status=='blocked' else '引用或解析限制不代替经验收益。'))
add_pc('PC0','近期已有角色解耦、空间光谱组合与协同训练，通用交互不是空白。','context','citation','supported','assert',[],[],['CN1','CN2','CN3'],['定向覆盖，不是穷尽综述。'],[])
add_pc('PC1','特定分离接口在合理设置下影响正确联合响应。','primary','empirical','blocked','omit',['C1'],['F1','F2'],['CN1','CN3'],['没有网络结果；只在声明先验和容量范围提出假设。'],['B2/B3实际运行、角色非塌缩与强对照审计。'])
add_pc('PC2','内部组合训练超越同数据普通训练，并改善自然PSNR/RMSE/SAM。','supporting','empirical','blocked','omit',['C2'],['F3','F4'],['CN2','CN3'],['没有自然新指标；不保证跨尺度与跨先验泛化。'],['B4/B5配对种子与场景结果、成本与结果审计。'])
add_pc('PC3','联合零空间存在同观测不同真值，只能在声明结构先验内讨论恢复。','limitation','theoretical','supported','limitation',[],['F0'],[],['程序样本检查不证明自然恢复；受限oracle知道生成常量。'],[],['research/d2_feasibility_and_story_20261006/d2_feasibility_steps_motivation_story.md','research/d2_feasibility_and_story_20261006/preflight-results.json'])
dump(PAPER/'claim-evidence-bindings.json',dict(schema_version='1.0',paper_id='d2-exploratory-interface-20261006',identity_version=1,status='draft',claims=pc))
write(PAPER/'claims-evidence-matrix.md','# Claims-Evidence Matrix\n\n经验发现暂时omit；开题可写成问题，不可改写为已支持结果。\n\n'+table(['Paper claim ID','Claim','Type','Evidence mode','Support status','Manuscript action','Required assurance','Source claim IDs','Audit IDs','Excluded audit IDs','Planned sections','Exhibit IDs','Citation need IDs','Limitation'],[[c['paper_claim_id'],c['claim'],c['claim_type'],c['evidence_mode'],c['support_status'],c['manuscript_action'],c['required_assurance_class'],', '.join(c['source_claim_ids']),'','',', '.join(c['planned_sections']),', '.join(c['exhibit_ids']),', '.join(c['citation_need_ids']),'; '.join(c['limitations'])] for c in pc]))
write(PAPER/'figure-plan.md','# Figure Plan\n\n'+table(['Exhibit ID','Purpose','Paper claim IDs','Priority','Status','Notes'],[
    ['F0','可恢复先验族/不可辨识冲突/加性对照','PC3','mandatory','planned','代数已有，正式图未绘制'],
    ['F1','合理分离强度与混合/完整误差曲线','PC1','mandatory','planned','不能仅特征cos'],
    ['F2','接口公平对照、角色与非塌缩','PC1','mandatory','planned','不能只展示loss更低'],
    ['F3','A/C二因素与同数据普通训练','PC2','mandatory','planned','没有结果不能填数'],
    ['F4','自然场景多指标、失败与成本','PC2','mandatory','planned','同checkpoint报告']]))
write(PAPER/'citation-plan.md','# Citation Plan\n\n'+table(['Citation need ID','Paper claim IDs','Citation need','Source status','Notes'],[
    ['CN1','PC0, PC1','MossFuse/CLoRF/SHOTUN近期分解与关系','verified-bounded','上一轮主矩阵；IEEE全文不全'],
    ['CN2','PC0, PC2','CDPS/Equivariant Splitting的观测训练与边界','verified-bounded','不得转移不偏结论'],
    ['CN3','PC0, PC1, PC2','CoMM ICLR2025及SynCo2026预印本协同近邻','verified-bounded','SynCo不标正式顶刊/代码；主文含一手链接']]))
write(PAPER/'paper-plan.md',"""# Paper Plan

## Header

- Paper ID: d2-exploratory-interface-20261006（规划标识，不是已承诺投稿项目）
- Identity version: 1
- Plan status: draft
- Working title: 自监督高光谱融合中的约束接口与光谱—空间联合响应（候选）
- One-sentence contribution: 待检验的指定分离约束取舍及其观测内部训练保护。
- Operating mode: standalone
- Target venue: 尚未选定；不保证顶刊可发表。
- Page budget: 未设定，不虚构投稿限制。

## Structure

1. 任务要求：光谱信息需要正确空间分配，近期分解/关系已有基础。
2. 研究问题：指定约束是否与联合任务响应发生取舍，不声称普遍有害。
3. 诊断：声明先验的可学族、不可辨识冲突和加性负对照。
4. 方法：只保留实际证据支持的接口/内部组合训练。
5. 实验：合理权重、简单替代、角色不塌缩、同数据/预算、未见自然图像。
6. 边界：不可辨识、内部尺度泛化、先验偏差及失败案例。

## Section Notes

当前开题故事使用“研究/检验/尝试”，不是“发现/证明/显著改善”。正文主文第10节给出三动机与分结果叙事。
PC1/PC2现为blocked/omit，不能作为经验发现写进摘要；可以作为未检验问题提出。PC0为近期文献事实、PC3为解析限制。
若合理分离无影响，则删除损害因果故事；若A不必要则删除A；若普通训练解释全部收益则降级贡献。不得为保持故事而选结果。

## Evidence Boundary

claim-evidence-bindings.json是支持状态来源。没有C1/C2同scope结果审计，因此不借用旧插值审计A1/A2来填支持。预检与结构校验也不是网络收益。只能在真实运行、结果审计、近邻复核后更新状态；当前只是计划。
""")

immutable=['model.py','train_cave_M.py','train_dataloader.py','utils.py','calculate_metrics.py']
verification={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in immutable}
dump(HERE/'source-snapshot.json',dict(date='2026-10-06',hashes=verification,network_training=False,model_source_modified=False,old_reports_modified=False))
print(json.dumps({'experiment_blocks':len(blocks),'experiment_claims':len(claims),'paper_claims':len(pc),'split_counts':{'train':16,'validation':4,'test':12},'status':'planning_only'},ensure_ascii=False))
