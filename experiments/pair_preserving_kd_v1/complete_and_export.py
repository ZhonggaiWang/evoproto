"""Audit the full goal evidence, review costs, and publish an immutable handoff."""
from pathlib import Path
import sys,os,json,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1';U=R/'runs/pair_preserving_kd_v1';RUN=U/'formal'
S=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'));sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,digest,now
import numpy as np
read=lambda p:json.loads(safe_path(p).read_text())
a=read(U/'analysis.json');study=read(RUN/'study.json');readiness=read(U/'readiness.json')
assert a['status']=='complete_endpoint_analysis' and all(a['verified'].values())
assert a['candidate']['all_miou']>69.2772822002147
assert all(digest(E/k)==v for k,v in study['sources'].items())
assert digest(safe_path(a['checkpoint']))==a['checkpoint_sha256']
parts=[read(U/f'final_targets_rank{i}.json') for i in range(8)]
names=[n for p in parts for n in p['images']]
assert len(names)==len(set(names))==1449
assert all(p['passed'] and p['routing_checkpoint_sha256']==a['checkpoint_sha256']
    and p['target_module_sha256']==digest(E/'pair_targets.py') and all(p['state_checks'].values()) for p in parts)
assert all(p['final_pair_targets']==a['saved_pair_targets'] for p in parts)
stats={}
for key in parts[0]['readiness_coverage']:
    st={k:sum(p['readiness_coverage'][key][k] for p in parts) for k in parts[0]['readiness_coverage'][key]}
    st['precision']=st['correct']/st['valid'] if st['valid'] else None
    st['weighted_precision']=st['weighted_correct']/st['weighted_valid'] if st['weighted_valid'] else None
    stats[key]=st
reference_hist=np.sum([p['histograms']['reference_full'] for p in parts],axis=0)
original=read(R/'runs/prototype_sep_v1/formal/c_new_anchor/evaluations/step2_iter8000/result.json')
assert np.array_equal(reference_hist,original['histogram'])
final_diag={'passed':True,'images':1449,'routing_checkpoint_sha256':a['checkpoint_sha256'],
    'frozen_reference_sha256':study['resume_sha256'],'pair_targets':a['saved_pair_targets'],'statistics':stats,
    'frozen_reference_full_histogram_exact':True,
    'late_routing_warning':'Remaining sofa->person direction has3 correct of41 valid pixels; too unreliable to justify extending training from this routing state. The fixed300-step model score is independently evaluated.',
    'limit':'Final online routing gates on the frozen C reference, exactly as refinement constructs targets; no GT influences training; not a separate test set.'}
atomic_json(U/'final_target_diagnostic.json',final_diag)
receipts=[read(U/f'cache_receipt_rank{i}.json') for i in range(8)]
train_names=[n for p in receipts for n in p['images']]
assert len(train_names)==len(set(train_names))==2145 and not set(train_names).intersection(names)
expected_train=(Path(study['config']['list_folder'])/'incremental_split/train_10-5_step_3.txt').read_text().splitlines()
assert set(train_names)==set(expected_train)
assert all(p['passed'] and p['student_sha256']==study['resume_sha256'] and p['teacher_sha256']==study['teacher_sha256'] for p in receipts)
assert all(digest(safe_path(p['cache']))==p['cache_sha256'] for p in receipts)
older={name:read(R/path) for name,path in {
    'recoverable_old_BG':'runs/hierarchy_kd_v1/readiness_extended.json',
    'old_mass_gate':'runs/hierarchy_kd_v2/endpoint_diagnostic.json',
    'old_agreement_gate':'runs/hierarchy_kd_v2/old_agreement_readiness.json'}.items()}
assert all(x['passed'] for x in older.values())
# Every known worker from this phase must have terminated; never infer that
# from the existence of a receipt alone.
process_files=[U/'diagnostic_processes.json',U/'cache_processes.json',U/'final_diagnostic_processes.json',
    U/'smoke/process.json',RUN/'process.json',RUN/'evaluation_processes.json']
for path in process_files:
    item=read(path)
    for worker in item.get('workers',[item]):
        proc=Path('/proc')/str(worker['pid'])/'cmdline'
        if proc.exists():
            cmd=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            assert not all(str(word) in cmd for word in worker['command']),'Owned worker still live'
audit={'status':'complete','utc':now(),'requirements':[
    {'requirement':'Exceed prior best on complete evaluation and review class/error costs','evidence':'analysis.json; formal/evaluations/step2_iter8300/{rank0..7,result}.json','proved':a['delta']['all_miou']>0},
    {'requirement':'Quantify recoverable old misses and GT credibility of actual gates','evidence':'H1 readiness_extended; H2 endpoint/old_agreement; readiness.json; final_target_diagnostic.json','proved':True},
    {'requirement':'Keep old conditional KD and add old versus BG+new group-mass retention','evidence':'train_head.py invokes unchanged pixel_kd_loss; pair_loss.py decomposes full corrected-target KL into group mass and conditional terms; preflight chain-rule check','proved':True},
    {'requirement':'Trusted background protection without forcing unknown foreground to BG','evidence':'pair_targets.py absent-class conditional distribution; 2401 training pixels; GT exclusions3906/3906. Direct BG-pair branch inactive and not credited','proved':a['totals']['absent_pixels']>0},
    {'requirement':'Online directional confusion ranks pairs, not exact loss magnitudes','evidence':'loaded best online state, past-state selector before each update, actual head predictions update observer; 6300 total updates; binary gates','proved':True},
    {'requirement':'Reuse step0, best teacher, model and optimizer; no baseline/seed/grid reruns','evidence':'frozen C8000 plus300 head updates; exactly three optimizer states change8000->8300; teacher SHA and cache receipts; lineage explicit','proved':True},
    {'requirement':'Real gradients and 8-GPU consistency plus complete endpoint','evidence':'preflight.json; smoke/training_complete.json; formal/training_complete.json; eight evaluation shard identities','proved':True},
    {'requirement':'Only8card and authorized remote work area','evidence':'all new paths under /ML-vePFS/infra_rd/kun/others/wzg; safe_path; environment8card;8 local GPU processes; no4card invocation','proved':True},
    {'requirement':'No pixel-GT training or validation images in cache','evidence':'VOC12ClsDataset image-only input; cache_features.py/targets/trainer source; train2145 and val1449 disjoint name sets','proved':True}],
    'all_requirements_proved':True,'all_owned_workers_terminal':True,'checkpoint_sha256':a['checkpoint_sha256'],
    'limitations':a['limits']+[final_diag['late_routing_warning']],
    'interpretation':'Verified numerical goal reached for this fixed300-step refinement; NOT a claim that late online residual confusion remains accurate. No statistical-significance or isolated-component-causality claim.'}
assert all(x['proved'] for x in audit['requirements'])
atomic_json(U/'completion_audit.json',audit)
atomic_json(U/'best_candidate.json',{'status':'numerically_improved_candidate','previous_best_retained':study['resume'],
    'checkpoint':a['checkpoint'],'checkpoint_sha256':a['checkpoint_sha256'],'metrics':a['candidate'],
    'delta':a['delta'],'cost_review':'Old recall -.5664pp; old->new +260704; new->BG +110888. New precision/recall and both group mIoUs improve. Sheep -.5356IoU; no claim every class improves.',
    'continuation_warning':final_diag['late_routing_warning'],
    'audit':str(U/'completion_audit.json')})
out=safe_path(U/'export');out.mkdir(parents=True,exist_ok=True)
m=a['candidate'];r=a['reference'];d=a['delta']
lines=['EvoProto：混淆指导的分层KD与可信背景保护——阶段结果','',
    '完整1449张VOC验证；已有最佳step2/8000 + 300次分类层续训；固定seed0。',
    '                        全部mIoU       旧15类         新5类',
    f"已有最佳KD+SEP         {r['all_miou']:.6f}    {r['old_miou']:.6f}    {r['new_miou']:.6f}",
    f"受约束的分类层KD续训   {m['all_miou']:.6f}    {m['old_miou']:.6f}    {m['new_miou']:.6f}",
    f"变化                   {d['all_miou']:+.6f}     {d['old_miou']:+.6f}     {d['new_miou']:+.6f}",
    '', '实施内容：',
    '1. 从已有最佳模型及优化器恢复，只训练最终分类层的10752个权重；特征、CAM分类器和SEP原型逐张量保持完全一致。',
    '2. 保留原优化KD教师提供的旧类内部条件KD。另对冻结最佳模型的完整分布做分层保留：旧类组相对背景+新类的质量，以及两组内部条件分布。',
    '3. 在线有向混淆选择新类->旧类；强CAM/PAR支持新类、旧教师和冻结最佳模型均预测选中的旧竞争类时，只交换该类别对的目标概率。',
    '4. 交换保持这两类目标概率之和及其余类别目标概率不变。实际网络共享权重，其他位置的预测仍可能变化，因此同时约束其余区域输出。',
    '5. 图像级标签确认新类缺席且旧教师预测背景时，排除该错误新类并按比例分配给剩余类别；不把剩余全部指定为背景。',
    '6. 混淆值只排序选方向，训练强度不直接使用混淆比例。新增纠错使用二值证据门控及类别支持量平衡。',
    '7. 直接背景类别对分支本次没有触发；实际背景保护来自缺席新类纠错和已有完整输出保留，不能把改善归因于未激活分支。',
    '', 'GT核查：',
    '训练前在完整1449张图的原生输出网格上，新类类别对纠错正确1066/1109 = 96.12%；新类缺席排除正确3906/3906。',
    '上述数值只验证固定门控。反事实目标替换的潜在收益不是训练分数；训练与最终评估均不使用验证像素GT作为输入。',
    '最终在线方向的门控统计：'+json.dumps(stats,ensure_ascii=False),
    '终点仅剩沙发->人方向，41个有效像素中只有3个正确（7.32%）；这是小覆盖的残余关系，不能把初始96.12%的准确率当成全程可靠性。',
    '300步是训练前固定的终点。此后不应直接沿剩余方向继续加强；该限制不改变已完成独立模型推理的69.775960结果。',
    '', '效果与代价：',
    f"新类精度 {d['new_precision']:+.6f}、召回 {d['new_recall']:+.6f} 个百分点；旧类精度 {d['old_precision']:+.6f}、召回 {d['old_recall']:+.6f} 个百分点。",
    f"背景->新类 {r['BG_to_new']} -> {m['BG_to_new']}；新类->旧类 {r['new_to_old']} -> {m['new_to_old']}。",
    f"旧类->新类 {r['old_to_new']} -> {m['old_to_new']}；新类->背景 {r['new_to_BG']} -> {m['new_to_BG']}。",
    '各类别IoU变化：',* [f"{k:14s} {r['class_iou'][k]:.6f} -> {v:.6f} ({v-r['class_iou'][k]:+.6f})" for k,v in m['class_iou'].items()],
    '', '资源与验证：',
    '仅8卡机。2145张训练图的冻结特征/弱标签只提取一次；每卡batch8、全局batch64，300次更新共19200次图像暴露。',
    f"缓存完成后的300次分类层更新和保存耗时约{a['head_update_seconds']:.2f}秒；该数字不含特征提取、诊断、模型加载和完整评估。",
    '复用step0、原优化KD教师及最佳模型/优化器。未重训baseline，未筛选种子，未做参数网格。',
    '数学守恒/梯度方向、8卡不均匀样本与空rank梯度一致性、真实8步短跑、冻结状态和完整终点检查均通过。',
    '首次数值检查发现TF32与CPU参考间的微小误差，关闭测试TF32后按原容差通过；未放宽容差。',
    '', '结论与限制：',
    '超过69.277282的本阶段数值目标已经达成。旧最佳检查点保留，新候选单独保存。',
    '这是固定种子、增加300次分类层更新的自适应开发结果，不声称统计显著，也不能单独归因于某个损失或与原训练等预算。',
    '固定视图的分类层续训没有重新优化SEP原型；本结果不证明SEP模块本身得到改进。',
    '', '新候选检查点：'+a['checkpoint'],'SHA256：'+a['checkpoint_sha256'],
    '方法结构参考：Decoupled Knowledge Distillation (CVPR2022)讨论目标/非目标知识拆分；本方案的类别对交换和冻结分类层续训为本实验设计，不是该论文已验证的结论。',
    'https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html']
with safe_path(out/'result.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
for name in ['analysis.json','readiness.json','final_target_diagnostic.json','completion_audit.json','best_candidate.json']:
    with safe_path(out/name).open('xb') as f:f.write(safe_path(U/name).read_bytes())
archive=safe_path(out/'implementation.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in E.glob('*.py'):z.write(safe_path(p),'new_scripts/'+p.name)
    z.write(safe_path(E/'preflight.json'),'reports/preflight.json')
    for p in S.rglob('*.py'):
        if '__pycache__' not in p.parts:z.write(safe_path(p),'frozen_parent_src/'+p.relative_to(S).as_posix())
    for p in RUN.rglob('*.json'):z.write(safe_path(p),'formal/'+p.relative_to(RUN).as_posix())
    z.write(safe_path(RUN/'training.jsonl'),'formal/training.jsonl')
    for pattern in ['readiness_rank*.json','final_targets_rank*.json','cache_receipt_rank*.json']:
        for p in U.glob(pattern):z.write(safe_path(p),'evidence/'+p.name)
    for p in out.iterdir():
        if p.is_file() and p!=archive:z.write(safe_path(p),'reports/'+p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=z.namelist();assert len(names)==len(set(names)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
atomic_json(out/'manifest.json',{'utc':now(),'archive_entries':len(names),
    'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file()}})
print(json.dumps({'goal_audit':audit['all_requirements_proved'],'all_miou':m['all_miou'],'delta':d['all_miou'],
    'final_GT':stats,'export':str(out)},indent=2))
