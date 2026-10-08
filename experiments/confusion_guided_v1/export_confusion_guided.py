"""Export reviewed endpoints and implementation after the final integrity audit."""
from pathlib import Path
import json,sys,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/confusion_guided_v1';U=R/'runs/confusion_guided_v1'
sys.path.insert(0,str(E/'a_sep/src'))
from kd_runtime import safe_path,atomic_json,now,digest
audit=json.loads((U/'completion_audit.json').read_text())
analysis=json.loads((U/'result_analysis.json').read_text())
assert audit['all_training_and_endpoint_requirements_complete']
assert analysis['status']=='complete_endpoint_analysis'
guard_audit=json.loads((U/'guard_completion_audit.json').read_text())
assert guard_audit['all_training_and_endpoint_requirements_complete']
pairkd_audit=json.loads((U/'pairkd_only_completion_audit.json').read_text())
assert pairkd_audit['all_training_and_endpoint_requirements_complete']
candidates=['a_sep','b_pairkd','c_newaware_sep','d_pairkd_only']
selected=analysis['recommendation']['recommended_retained_method']
if selected in candidates:
    source=E/selected/'src';checkpoint=U/f'formal/{selected}/10-5/step2/checkpoints/model_final.pth'
elif selected=='optimized_KD':
    source=R/'experiments/kd_parallel_v1/b_relational/src'
    checkpoint=R/'runs/kd_parallel_v1/formal/b_relational/10-5/step2/checkpoints/model_final.pth'
else:
    source=R/'runs/fixed_baseline_v1/source_snapshot.tar.gz'
    checkpoint=R/f'runs/fixed_baseline_v1/{selected}/10-5/step2/checkpoints/model_final.pth'
assert source.exists() and checkpoint.exists()
precision_report={'utc':now(),'scope':'Original-GT-space final endpoint pixel precision/recall; micro averages are distinct from class-mean IoU. No training change or extra evaluation.', 'methods':{}}
for method,record in analysis['endpoints'].items():
    h=record.get('histogram')
    if h is None:continue
    rows=[sum(row) for row in h];columns=[sum(row[c] for row in h) for c in range(21)]
    groups={}
    for group,ids in [('old15',range(1,16)),('new5',range(16,21))]:
        tp=sum(h[c][c] for c in ids);gt=sum(rows[c] for c in ids);pred=sum(columns[c] for c in ids)
        groups[group]={'true_positive_pixels':tp,'false_positive_pixels':pred-tp,'false_negative_pixels':gt-tp,
            'micro_precision_percent':100*tp/pred,'micro_recall_percent':100*tp/gt}
    precision_report['methods'][method]={'source':record['source'],'groups':groups,
        'per_class':[{'id':c,'name':analysis['scope']['classes'][c],'precision_percent':100*h[c][c]/columns[c] if columns[c] else None,
            'recall_percent':100*h[c][c]/rows[c] if rows[c] else None,'false_positive_pixels':columns[c]-h[c][c],'false_negative_pixels':rows[c]-h[c][c]} for c in range(21)]}
atomic_json(U/'precision_recall_report.json',precision_report)
decision={'utc':now(),'status':'complete','selected_method':selected,'selected_source':str(source),
    'selected_checkpoint':str(checkpoint),'selected_checkpoint_sha256':digest(checkpoint),
    'selection_rule':analysis['recommendation']['selection_rule'],'recommendation':analysis['recommendation'],
    'metrics':{method:record['metrics'] for method,record in analysis['endpoints'].items()},
    'candidate_changes_vs_optimized_KD':{arm:{'metrics_delta_pp':analysis['comparisons'][arm]['optimized_KD']['metric_delta_pp'],
        'highlight_directions':analysis['comparisons'][arm]['optimized_KD']['highlight_directions'],
        'aggregate_errors':analysis['comparisons'][arm]['optimized_KD']['aggregate_error_deltas']}
        for arm in candidates},
    'guard_changes_vs_parent_A':analysis['comparisons']['c_newaware_sep']['a_sep'],
    'focused_KD_changes_vs_parent_optimized_KD':analysis['comparisons']['d_pairkd_only']['optimized_KD'],
    'audit':str(U/'completion_audit.json'),'full_analysis':str(U/'result_analysis.json'),
    'mechanism_report':str(U/'mechanism_report.json'),'scope':analysis['scope'],
    'precision_recall_report':str(U/'precision_recall_report.json'),
    'guard_audit':str(U/'guard_completion_audit.json'),
    'focused_KD_audit':str(U/'pairkd_only_completion_audit.json'),
    'guard_diagnostic':str(U/'diagnostics/sep_new_conflict/merged.json'),
    'resource_policy':'8card: initially two mechanisms, 4 GPUs each/batch2/global8; targeted SEP followup C and teacher-domain KD followup D sequentially use all8 x batch1/global8 and resume existing stage2 iter2000. 4card inactive; step0/sharedwarmup/baselines/A or optimizedKD stage1 reused.'}
atomic_json(U/'final_decision.json',decision)
names={'optimized_KD':'已有优化 KD','base':'已有 BASE','kd':'原型 KD','sep':'原型 SEP','kd_sep':'原型 KD+SEP',
    'a_sep':'A：定向 SEP + 优化 KD','b_pairkd':'B：定向 SEP + 类别对 KD','c_newaware_sep':'C：定向 SEP 的新类保护 + 优化 KD',
    'd_pairkd_only':'D：教师已知类内的混淆指导 KD'}
lines=['EvoProto：混淆指导的 SEP / KD 实验结果',f'完成时间（UTC）：{now()}',
    f'按完整终点保留：{names.get(selected,selected)}',
    '', '实验范围：VOC 10-5，seed=0。整体 mIoU 包含背景；旧类为15个前景类，新类为5个前景类。',
    '以下都是阶段2第8000步、完整1449张验证图像的结果。单位为百分比。',
    '', '方法 | 整体 mIoU | 旧15类 mIoU | 新5类 mIoU']
for method in ['base','kd','optimized_KD',*candidates]:
    m=decision['metrics'][method]
    lines.append(f"{names[method]} | {m['all_miou']:.4f} | {m['previous_foreground_miou']:.4f} | {m['current_foreground_miou']:.4f}")
lines += ['', '方法实现：',
    '混淆矩阵仅用于排序和筛选每个类别需要重点处理的竞争类，保留方向 i→j；没有将矩阵比例直接当成学习权重。',
    'SEP 对可信 PAR/CAM 锚点增加有方向的像素 logit 间隔约束，同时作用于主分割头和原型分支；旧类锚点要求冻结 teacher 一致。实际强度由锚点可信度、间隔误差和固定外部系数控制。',
    'B 的类别对 KD 使用可信旧类像素上的 teacher 二分类软关系，替换最多一半原全旧类 KL；门控不满足时精确退回已有优化 KD。',
    'C 仅在阶段2给旧类 SEP 增加 (1−max_new_CAM)^2 保护因子，降低新类区域被错误旧类锚点压制的强度。新类锚点、方向选择、门控、类别计数和分母不变；KD 沿用已有优化实现。',
    'D 让混淆仅指导 KD：在 teacher 已知的旧类1–15之间选有方向的伙伴，避免最混淆的新类先被选择后再被KD丢弃。比例仍按全21类行归一化，阈值及支持要求不变。类别对蒸馏最多替换一半原全旧类 KL，SEP 外部权重为0，新类不直接蒸馏。',
    '', '复用与资源：',
    'A/B 在8卡机分别用4卡×单卡batch2并行。C/D 顺序使用8卡×单卡batch1，全局batch仍为8。4卡机未安排任务。',
    '所有方案复用 step0、已有基准和共同 warmup。C 进一步继承 A 的完整阶段1，以及 A 的阶段2第2000步模型、优化器、在线混淆和选择器状态，仅续训剩余6000步。没有新增种子搜索或参数网格。',
    'D 继承已有优化KD的阶段1 teacher和阶段2第2000步模型/优化器，只续训6000步；在线混淆和选择器从零积累，并按原100次观察/200次渐进要求启用。',
    '', '本轮结论：',
    '四个新候选都没有超过已有优化KD的整体终点，因此不替换当前最佳训练方案。',
    'C 将狗→猫误判率由12.43%降至10.95%、沙发→椅子由17.03%降至14.90%，但椅子→沙发由6.89%升至12.42%。方向改善需要同时检查反向代价。',
    'D 新类整体像素召回率由73.20%升至76.23%，但像素精度由67.99%降至65.99%，出现更多误检；这一汇总与新类平均IoU不同。单向错误减少不能替代完整IoU。',
    '观察到的继续优化重点是伪标签锚点误标、反向混淆和误检扩张，而不只是扩大重点类别对的学习强度。这是下一步的研究判断，不声称已验证新方案会提高效果。',
    '', 'GT 机制诊断与局限：',
    '固定128张图像上的诊断找到379个旧类 SEP 锚点实际属于新类，且直接压低正确新类别。静态保护预计削减其主头/原型局部 logit 推力92.60%/90.09%，正确旧类推力仅减少1.09%/1.31%。这不是参数梯度范数，也不是训练后的准确率收益。',
    '像素级 GT 仅用于诊断和评估，没有进入新增 SEP/KD 损失、混淆方向选择或保护门控。128张GT诊断用于设计C修正，因此完整验证结果属于开发期验证，并非新的独立测试。',
    '只有一个任务和一个固定种子，不声称统计显著性、跨种子稳定性或泛化。C 改为8卡训练并重启采样/增强，不能将相对A的全部差异严格归因于单个保护因子。',
    '完整终点决定保留方法；中间峰值和单个类别对误判下降不能替代整体、旧类、新类评估。',
    '', '验证：',
    '原始选择器和损失共36项行为检查、新保护14项行为检查通过；4卡原机制和8卡新保护的全局损失/梯度一致性通过；真实图像梯度路径及训练/评估 smoke 通过。',
    'D 的8项教师选择域行为检查通过，默认选择器仍与旧实现精确一致；8卡类别对KD全局损失/参数梯度一致性及实际阶段2图像梯度路径通过。',
    '完整审计确认：代码冻结、教师/恢复检查点来源、训练与评估成功退出、1449图像无遗漏或重复、GT行计数一致、指标重算、最终模型与实际评估模型/在线状态一致。',
    '', f'选中检查点：{checkpoint}',f'SHA256：{digest(checkpoint)}',
    '详细每类IoU、全部方向混淆变化及推荐依据见 result_analysis.json；实现与收据见 implementation.zip。']
safe_path(U/'sep_kd_result.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
safe_path(E/'implementation.zip')
with zipfile.ZipFile(E/'implementation.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for arm in candidates:
        for path in sorted((E/arm/'src').rglob('*.py')):
            if '__pycache__' not in path.parts:archive.write(path,str(path.relative_to(E)))
        study=U/'formal'/arm/'study.json';archive.write(study,f'receipts/{arm}/study.json')
        for step in ([2] if arm in ['c_newaware_sep','d_pairkd_only'] else [1,2]):
            stage=U/f'formal/{arm}/10-5/step{step}'
            for name in ['config.json','launch.json','training_complete.json','pair_metrics.jsonl','kd_metrics.jsonl','online_confusion.json']:
                path=stage/name
                if path.exists():archive.write(path,f'receipts/{arm}/step{step}/{name}')
            endpoint=U/f'formal/{arm}/evaluations/step{step}_iter8000/result.json'
            archive.write(endpoint,f'receipts/{arm}/step{step}/endpoint.json')
    for name in ['origin.json','protocol.json','integration.json','preflight.json','gradient_probe.json',
        'selector_tests.log','loss_tests.log','loss_ddp_test.log','run_confusion_guided.py',
        'report_confusion_guided.py','audit_confusion_guided.py','analyze_confusion_guided.py',
        'mechanism_confusion_guided.py','probe_confusion_guided.py','test_directed_pair_selector.py',
        'test_confusion_pair_losses.py','test_confusion_pair_ddp.py','pixel_kd_legacy_reference.py',
        'guard_protocol.json','guard_preflight.json','guard_cpu_tests.log','guard_ddp_tests.log',
        'run_sep_guard.py','prepare_sep_guard.py','audit_sep_guard.py','test_sep_new_guard.py',
        'test_sep_new_guard_ddp.py','sep_legacy_reference.py','diagnose_sep_new_conflict.py',
        'pairkd_only_protocol.json','pairkd_only_preflight.json','pairkd_only_cpu_tests.log','pairkd_only_ddp_tests.log',
        'pairkd_only_gradient_probe.json','run_pairkd_only.py','prepare_pairkd_only.py','audit_pairkd_only.py',
        'test_pairkd_selector_domain.py','selector_legacy_reference.py','test_pairkd_only_ddp.py','probe_pairkd_only.py']:
        archive.write(E/name,f'study/{name}')
    archive.write(U/'diagnostics/sep_new_conflict/merged.json','diagnostics/sep_new_conflict.json')
    for name in ['sep_kd_result.txt','final_decision.json','precision_recall_report.json','completion_audit.json','guard_completion_audit.json','pairkd_only_completion_audit.json','result_analysis.json','mechanism_report.json','report.json']:
        archive.write(U/name,f'results/{name}')
    if source.is_dir() and selected not in candidates:
        for path in sorted(source.rglob('*.py')):
            if '__pycache__' not in path.parts:archive.write(path,'selected_existing_method/'+str(path.relative_to(source)))
atomic_json(U/'export.json',{'utc':now(),'implementation_zip':str(E/'implementation.zip'),
    'sha256':digest(E/'implementation.zip'),'decision_sha256':digest(U/'final_decision.json')})
cluster_path=safe_path(R/'.runtime/kd_pixel_v1/cluster.json');cluster=json.loads(cluster_path.read_text())
cluster.update(utc=now(),status='completed',selected_source=str(source),selected_checkpoint=str(checkpoint),
    final_decision=str(U/'final_decision.json'))
cluster['8card'].update(assignments={},completed_confusion_guided_assignments={'0,1,2,3':'a_sep','4,5,6,7':'b_pairkd'},
    completed_guard_assignments={'0,1,2,3,4,5,6,7':'c_newaware_sep'},completed_focused_KD_assignments={'0,1,2,3,4,5,6,7':'d_pairkd_only'},status='idle; confusion-guided full training/evaluation completed')
for record in cluster.get('confusion_guided_records',{}).values():record['status']='completed'
cluster['4card']={'status':'inactive per user; no new work'};atomic_json(cluster_path,cluster)
print(json.dumps({'selected':selected,'metrics':decision['metrics'],'zip_sha256':digest(E/'implementation.zip')}))
