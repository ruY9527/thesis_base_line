"""Build a source-grounded learning report from completed local diagnostics."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/baseline_comparison_20261006'
d = json.loads((OUT/'diagnostics.json').read_text(encoding='utf-8'))
methods = {row['method']:row for row in d['measured_methods']}
best = d['best_psnr_row']

with (OUT/'interpolation_per_scene.csv').open(encoding='utf-8-sig',newline='') as f:
    scenes = list(csv.DictReader(f))
paired = []
for scene in sorted({row['scene'] for row in scenes}):
    values = {row['method']:row for row in scenes if row['scene']==scene}
    paired.append({'scene':scene,
                   'bicubic_psnr_gain_alignment':float(values['bicubic_phase_aligned']['psnr'])-float(values['bicubic_full']['psnr']),
                   'bilinear_psnr_gain_alignment':float(values['bilinear_phase_aligned']['psnr'])-float(values['bilinear_full']['psnr'])})
with (OUT/'phase_alignment_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)

labels={'bilinear_full':'默认双线性（整图）','bicubic_full':'默认双三次（整图）',
        'bilinear_tiled':'默认双线性（分块）','bicubic_tiled':'默认双三次（分块）',
        'bilinear_phase_aligned':'坐标对齐双线性（整图）','bicubic_phase_aligned':'坐标对齐双三次（整图）'}
order=list(labels)
table='\n'.join(f"| {labels[key]} | {methods[key]['psnr_mean']:.4f} | {methods[key]['rmse_255_mean']:.4f} | {methods[key]['sam_deg_mean']:.4f} | {methods[key]['ergas_scale8_mean']:.4f} |" for key in order)
gain=methods['bicubic_phase_aligned']['psnr_mean']-methods['bicubic_full']['psnr_mean']
paired_wins=sum(row['bicubic_psnr_gain_alignment']>0 for row in paired)
report=f'''# Baseline 对比、诊断与学习记录

日期：2026-10-06。任务：CAVE ×8，高光谱/多光谱融合。

## 先掌握证据边界

本次实际运行了当前本地 12 个测试场景上的六种插值对照，核对了当前 Net 的参数量与计算量，分析了上传的训练记录。
没有重新训练 Fusformer、MoG-DCN、DSPNet；没有复现上传记录的第 1350 轮模型。
当前仅发现 Model/3 的第 1 轮模型，其记录与上传 CSV 的第 1 轮数值不一致，不能替代上传运行。
因此：插值结果属于本地实测；48.0453 dB 属于上传记录；论文结果属于外部参考，不能三者混为同条件排行榜。

## 本地实测：插值对照

采用 8×8、σ=3 高斯核，scipy 对称边界卷积，0::8 采样；逐场景归一化，重建裁剪到 [0,1]。
PSNR、RMSE、SAM 直接复用项目指标实现。RMSE 为 0–255 尺度；SAM 为角度。
ERGAS 使用倍率 8 的公式；数值必须与使用相同定义的结果比较。

| 方法 | PSNR ↑ | RMSE ↓ | SAM ↓ | ERGAS ↓ |
|---|---:|---:|---:|---:|
{table}

各指标先逐场景计算，再对 12 个场景等权平均。summary.csv 中的标准差是场景间差异，不是随机种子间差异。
插值仅使用 LR-HSI；融合网络额外使用 HR-MSI，因此插值是输入信息较少的基础对照，不能代替融合方法对照。

## 最值得学习的发现：坐标约定可以改变几 dB

固定同一份 LR-HSI，仅使双三次插值坐标与当前降采样位置对齐，平均 PSNR 从
{methods['bicubic_full']['psnr_mean']:.4f} 提高到 {methods['bicubic_phase_aligned']['psnr_mean']:.4f} dB，增量 {gain:.4f} dB。
在 {paired_wins}/12 个场景中 PSNR 提高；逐场景差值见 phase_alignment_pairs.csv。

原因：当前偶数 8×8 对称核配合 scipy 的 same 卷积与 0::8 采样，内部线性斜坡的 LR 样本中心在 HR 坐标 8j−0.5。
默认 align_corners=False 的插值把 LR 中心放在 8j+3.5，两者相差 4 个 HR 像素。
用线性斜坡验证：内部采样位置偏移为 −0.5，坐标对齐后内部重建最大误差为 0。
边界仍按具体插值实现处理。本次对齐仅适用于当前退化，不是所有 CAVE 实验的通用公式。

这不证明你的网络有 5 dB 改进空间：网络可能已通过残差学习补偿坐标偏移。
也不应直接替换已训练网络的上采样路径后宣称性能改善；应把坐标一致性作为独立、重新训练的实验因素。

## 当前模型与训练记录

- 当前 Net：{d['model']['parameters']:,} 个参数，即 {d['model']['parameters']/1e6:.4f} M。
- 输入 LR-HSI 8×8、HR-MSI 64×64、batch=1：约 3.104 G MACs（thop），不能直接当作论文 FLOPs。
- 参数规模较小，但不能仅凭参数量称为高效。推理使用重叠块，需要额外测量整图时间和显存。
- 512×512 场景当前每个方向生成 16 个位置，最后位置重复，实际网络前向 256 次；去重后为 15×15=225 次。
  这约减少 12.1% 的块前向次数，并非实测加速比例。去重会改变边缘块权重，应单独复测，不与结构改进混合。
- 上传记录最佳 PSNR：epoch {int(best['epoch'])}，PSNR {best['val_psnr']:.6f}，RMSE {best['val_rmse']:.6f}，SAM {best['val_sam']:.6f}°。
- 第 1000 轮 PSNR 48.005737，到最佳只增加 0.039547 dB；1000–2000 轮 PSNR 范围 {d['late_psnr_range']:.6f} dB。
  这是同一次训练中的收敛信息，不是跨种子重复性，也不能据此确定新的预设停止轮次。
- 不同指标的最佳轮次不同，必须固定同一 checkpoint 报告全部指标。

## 实验流程中需要优先修正的问题

| 优先级 | 当前证据 | 为什么影响结论 | 下一步 |
|---|---|---|---|
| P0 | cave_test 在训练期间被反复评价并用于保存候选模型 | 测试集参与选择，最佳结果具有选择偏倚 | 单独建立场景级验证集；测试只在方案冻结后评价 |
| P0 | 上传 CSV 未附场景清单、代码快照、权重 | 无法确认 48 dB 对应当前结构和当前 12 场景 | 找回对应权重及运行配置，逐场景复测 |
| P0 | 当前保存条件是 abs(best-current)<0.15 | 大幅提升可能不保存，接近最优但更差的也会以 best 命名 | 验证集指标严格改善时保存 best，另存 last |
| P1 | val_loss 是未裁剪块 MAE，指标是裁剪并融合后的整图结果 | loss 与最终输出的趋势可能不同 | 同时记录 reconstructed_MAE，明确两种口径 |
| P1 | 训练目录通过 os.listdir 取前 num 个文件 | 没有冻结场景清单，文件变动可改变训练样本 | 保存排序、筛选后的清单、随机种子和文件摘要 |
| P1 | 参数约 0.831M，但没有整图耗时/显存 | 无法建立轻量化优势 | 固定设备、预热、重复测量完整重建 |

文件字节 SHA-256 检查：20 个训练场景、12 个测试场景，跨划分相同字节文件 {len(d['cross_split_identical_file_hashes'])} 个。
这不能排除内容相近或重新保存后的重复场景，也不证明上传运行用了当前划分。
当前 12 个测试场景的原始最大值均为 1，因此此批数据上的 scene/scene.max() 不改变数值；原始预处理来源仍未验证。

## 文献参照：从 IJCV 的对照选择学习

以下是 DCINN 论文（IJCV 2024）Table 4 的均值，不是本地复现结果。
它使用 21/11 划分、8×8 方差 1 高斯核、16-bit 归一化，与当前设置有差异。

| 方法 | PSNR ↑ | SAM ↓ | ERGAS ↓ | 学习目的 |
|---|---:|---:|---:|---|
| SSRNet | 46.21 | 3.13 | 2.08 | 基础深度融合对照 |
| Fusformer | 47.95 | 2.74 | 1.42 | Transformer 融合对照 |
| DHIF | 48.46 | 2.50 | 0.84 | 模型引导重建对照 |
| MoG-DCN | 49.17 | 2.49 | 0.75 | 较强且有物理模型的对照 |
| 3DTNet | 49.22 | 2.39 | 0.73 | 较强 Transformer 对照 |
| DCINN（作者方法） | 50.05 | 2.28 | 0.71 | 看作者如何证明相对强对照的改进 |

来源：[IJCV 原文](https://liangjiandeng.github.io/papers/2024/wang-ijcv2024.pdf)。
复现入口：[MoG-DCN 作者代码](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution)、
[DSPNet 作者代码](https://github.com/syc11-25/DSPNet)。DSPNet README 提供的是 f4 预训练入口，不能直接当作 ×8 成绩。

## 按这个顺序学习与提升

| 实验 | 只改变什么 | 想回答的问题 | 判断依据 |
|---|---|---|---|
| E0 | 冻结划分、指标、种子和正确保存流程 | 原始 baseline 在可信流程下是多少？ | checkpoint 可复测；输出逐场景结果 |
| E1 | 接入对应第 1350 轮模型做诊断 | 上传均值是否能重现？哪些场景较弱？ | 同场景同流程复测，差异逐项解释；旧测试集只作探索 |
| E2 | 分别重新训练默认/坐标对齐残差插值路径 | 网络是否在花容量补偿坐标误差？ | 同其他因素、同种子配对；验证集选模 |
| E3 | 删除通道注意力，其他保持一致 | 通道注意力到底贡献多少？ | 精度、参数和时间一起比较 |
| E4 | 保持网络，单独加观测一致性损失 | 退化模型约束能否改善光谱/空间一致性？ | 同时看 PSNR、SAM、ERGAS、误差图 |
| E5 | 在统一数据与指标下复现强融合对照 | 是否真的接近或超过优质论文的方法？ | 所有方法同划分同退化，至少 3 个配对种子 |

E4 可从 L=L1(H_hat,H)+λs L1(R H_hat,M)+λp L1(D H_hat,L) 开始。
R 必须使用同一光谱响应；D 必须匹配核、边界、采样和整图/块处理。块边缘若无法一致，应使用有效区域而非直接比较。
λs、λp 只用验证集确定。这是候选实验，不是保证提升，也不构成已验证的新颖性。

现有 other_base/baseline.py 是另一套本地注意力网络，不能标注成 Fusformer 或 DSPNet 复现。
它把特征尺寸固定成 64×64，且需要当前环境未安装的 timm；本次未训练或评价它。

先完成 E0/E1，再做单因素实验，最后考虑组合改进。任何基于旧测试成绩挑出的方案，都需要新的独立确认评估。
本次没有设“必须提高 0.5 dB”这类人为门槛；效果应与配对重复的变异、场景一致性及计算代价一起判断。

## 结果文件

- diagnostics.json：训练记录、模型与本地实测汇总。
- interpolation_per_scene.csv：逐场景结果。
- interpolation_summary.csv：均值与场景间标准差。
- phase_alignment_pairs.csv：坐标对齐的逐场景配对差值。
- dataset_manifest.csv：数据划分和文件摘要。
- results-audit.json / results-audit.md：证据边界审查。
'''
(OUT/'learning_report.md').write_text(report,encoding='utf-8')

evidence_paths=[Path(d['record_path']),ROOT/'model.py',ROOT/'train_cave_M.py',
                ROOT/'train_dataloader.py',ROOT/'utils.py',ROOT/'calculate_metrics.py',
                ROOT/'research/baseline_diagnostics.py',OUT/'diagnostics.json',
                OUT/'dataset_manifest.csv',OUT/'interpolation_per_scene.csv',OUT/'phase_alignment_pairs.csv']
artifacts=[]
for path in evidence_paths:
    artifacts.append({'path':str(path),'kind':'source_or_result','source':'experiment',
                      'digest':hashlib.sha256(path.read_bytes()).hexdigest()})
def checks_for(phase=False):
    details={
        'protocol_integrity':('pass' if phase else 'inconclusive','Fixed local degradation and all 12 scenes; uploaded training used test for model selection.'),
        'metric_validity':('pass','Reused project metrics; analytic 0.01 constant error gives PSNR 40, RMSE 2.55, SAM 0; phase checked with an interior ramp.'),
        'baseline_fairness':('inconclusive','No deep comparator rerun; interpolation has less input information; literature uses different protocols.'),
        'outcome_accounting':('pass' if phase else 'inconclusive','All 12 local scenes included for all interpolation methods; history of uploaded training failures and other runs unknown.'),
        'inferential_support':('inconclusive','One diagnostic dataset and one uploaded training run; no independent split or seed replication.'),
        'confound_control':('pass' if phase else 'inconclusive','Phase comparison changes grid only; uploaded model architecture/data binding unknown.'),
        'provenance':('pass' if phase else 'inconclusive','Local diagnostic sources and result artifacts hashed; uploaded checkpoint absent.'),
        'snapshot_continuity':('not_assessed','No orchestrated immutable snapshot or authenticated runtime binding.'),
        'independence':('not_assessed','Same assistant implemented and audited diagnostics; no independent verifier.')}
    return [{'check_id':key,'status':value[0],'rationale':value[1],
             'evidence_paths':[str(OUT/'diagnostics.json')]} for key,value in details.items()]
audits=[]
for audit_id,phase in [('A1',False),('A2',True)]:
    verdict='supports_exploratory_follow_up' if phase else 'inconclusive'
    audits.append({'audit_id':audit_id,'claim_id':'phase_alignment' if phase else 'strong_baseline',
                   'claim_text':'Aligning interpolation sampling coordinates improves mean PSNR on current local CAVE scenes.' if phase else 'Uploaded run establishes a strong, competitive paper baseline.',
                   'scope':'Current 12 local CAVE scenes, x8, 8x8 sigma3 symmetric blur; byte hashes frozen. Uploaded run binding remains unverified.',
                   'source_mode':'standalone','requested_assurance_class':'exploratory' if phase else 'confirmatory',
                   'attained_assurance_class':'exploratory' if phase else 'none','verdict':verdict,
                   'audited_claim_effect':'strengthen' if phase else 'inconclusive','source_runs':[],
                   'run_selection':{'selection_rule':'All 12 diagnostic scenes and all 40 uploaded record points included. Local Model/1 and Model/2 are header-only; Model/3 has only epoch 1 and is a distinct run, so is not independent replication of the uploaded model.','excluded_runs':[{'run_id':'local_model3_epoch1','rationale':'Different epoch-1 metrics and no matching uploaded checkpoint; cannot substitute for the uploaded epoch1350 model.'}]},
                   'evidence_artifacts':artifacts,'check_results':checks_for(phase),
                   'independence':{'self_review':True,'dimensions':[],'evidence':'Diagnostic executor and auditor are the same assistant.'},
                   'limitations':['Uploaded checkpoint missing.','No deep method retraining.','Outcome-informed phase investigation is exploratory.','Scene standard deviations are not seed uncertainty.'],
                   'predecessor_failures':[],
                   'minimum_corrective_action':'Replicate phase-controlled experiment with new scene validation and frozen test.' if phase else 'Provide matching checkpoint and manifest; freeze validation/test and rerun strong deep comparators with paired seeds.',
                   'narrative_anchor':f'## Audit {audit_id}'})
(OUT/'results-audit.json').write_text(json.dumps({'schema_version':'1.0','paper_id':'standalone-cave-baseline','identity_version':1,'status':'complete','audits':audits},indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'results-audit.md').write_text('''# Baseline evidence audit

## Audit A1
- Bounded verdict: inconclusive
Uploaded metrics show a plateau, but the matching checkpoint, dataset manifest and strong same-protocol comparators are missing. They do not yet establish a competitive paper baseline. No independent verification is claimed.

## Audit A2
- Bounded verdict: supports_exploratory_follow_up
All 12 local scenes were evaluated with identical degradation and metrics. Matching interpolation coordinates improves the mean PSNR. This outcome-informed diagnostic is exploratory, not a confirmed neural-network improvement. Scene variation is not random-seed replication.

See learning_report.md for measured tables, limitations, and minimum corrective actions. Structural validation of this pack does not establish scientific or external verification.
''',encoding='utf-8')
print(json.dumps({'bicubic_alignment_gain_db':gain,'paired_scenes_improved':paired_wins,'report':str(OUT/'learning_report.md')},ensure_ascii=False))
