from explore import *
from merge import metrics
import shutil,zipfile,subprocess

records=[json.loads((U/f'verification/rank{i}.json').read_text()) for i in range(8)]
names=sum([r['images'] for r in records],[])
expected=(R/'datasets/voc/incremental_split/val_10-5_step_3.txt').read_text().splitlines()
assert len(names)==len(set(names))==1449 and set(names)==set(expected)
assert all(r['source_sha256']==records[0]['source_sha256'] for r in records)
assert all(r['base_tensors_exact_ALDv9'] and r['foreground_logits_exact'] and r['anchor_logits_exact'] for r in records)
results={m:metrics(sum(np.array(r['histograms'][m]) for r in records)) for m in records[0]['histograms']}
combined=json.loads((U/'combined/result.json').read_text());previous=json.loads((U/'refinement/result.json').read_text())
assert results['v5_final']['histogram']==combined['results']['aspect672_margin']['all']['histogram']
assert results['v5_engineering']['histogram']==combined['results']['aspect672']['all']['histogram']
assert results['v5_final']['all_miou']>=71.0
seconds={k:sum(r['seconds'][k] for r in records) for k in records[0]['seconds']}
atomic_json(U/'verification/result.json',dict(results=results,images=1449,seconds=seconds,utc=now(),exact_reproduction=True))
baseline=previous['results']['square448']['all'];plain=results['v5_engineering'];best=results['v5_final']
def pr(h):
 h=np.array(h);precision=np.diag(h)/h.sum(0)*100;recall=np.diag(h)/h.sum(1)*100
 return dict(old_precision=float(precision[1:16].mean()),old_recall=float(recall[1:16].mean()),new_precision=float(precision[16:].mean()),new_recall=float(recall[16:].mean()))
delta={k:best['class_iou'][k]-baseline['class_iou'][k] for k in best['class_iou']}
refine_delta={k:best['class_iou'][k]-plain['class_iou'][k] for k in best['class_iou']}
all_results={**{k:v['all'] for k,v in previous['results'].items()},**{k:v['all'] for k,v in combined['results'].items()},**results}
analysis=dict(target=71.0,achieved=best['all_miou'],total_delta=best['all_miou']-baseline['all_miou'],engineering_delta=plain['all_miou']-baseline['all_miou'],refinement_delta=best['all_miou']-plain['all_miou'],class_delta=delta,refinement_class_delta=refine_delta,baseline_precision_recall=pr(baseline['histogram']),final_precision_recall=pr(best['histogram']),evaluation_images=1449,new_training_updates=0,checkpoint_unchanged=True,all_results=all_results,seconds=seconds,
 error_diagnosis='Corrected VOC boundary includes label-to-void transitions. Initial diagnosis/result.json boundary subgroup is superseded by refinement/result.json; global scores were unaffected.',
 limits=['Fixed-seed checkpoint and adaptively reused development validation; not an independent test result or statistical significance claim.','The 1.60-point gain includes a changed inference protocol, not an isolated ALD/SEP/KD training gain.','Innovative element is the task-specific constrained scalar margin refinement, not generic affinity filtering or higher resolution; publication novelty has not been established.','Not all individual classes or small foreground objects improve.'])
atomic_json(U/'analysis.json',analysis)
source={str(p.relative_to(R)):digest(p) for p in (E).glob('*.py')}
source.update({str(p.relative_to(R)):digest(p) for p in S.rglob('*.py')})
deploy=dict(name='Aspect-preserving inference with anchored foreground-background margin refinement',checkpoint=str(R/'runs/local_sep_v5/formal/10-5/step2/checkpoints/model_final.pth'),checkpoint_sha256='0770ac07bcb343d510bbb6287526231352891e16f0c3e11e4b0d36a14fca4177',architecture_source=str(S),entrypoint=str(E/'predict.py'),refinement_source=str(E/'refine.py'),input='ImageNet-normalized RGB; preserve aspect ratio approximately with target area 672^2, nearest multiples of 16 per dimension; bilinear align_corners=False',inference=dict(model_forwards=1,flips=False,ensemble=False,amp=False,tf32=True,output='bilinear original-image space then margin refinement'),refinement=dict(iterations=5,dilations=[1,2,4,8],neighbors_per_dilation=8,pair_margin_limit=math.log(2),foreground_logits='exactly unchanged',confident_anchor_logits='exactly unchanged',learned_parameters=0,gt_inputs=False,image_tag_inputs=False),all_miou=best['all_miou'],old15_miou=best['old15'],new5_miou=best['new5'],validation_images=1449,new_training_updates=0,historical_square448_miou=baseline['all_miou'],source_sha256=source,utc=now())
atomic_json(U/'deployment.json',deploy)
atomic_json(U/'method_provenance.json',dict(background=[dict(title='Single-Stage Semantic Segmentation From Image Labels',url='https://openaccess.thecvf.com/content_CVPR_2020/papers/Araslanov_Single-Stage_Semantic_Segmentation_From_Image_Labels_CVPR_2020_paper.pdf',relationship='Prior parameter-free appearance-affinity mask refinement, not claimed novel here.'),dict(title='Learning Affinity From Attention',url='https://openaccess.thecvf.com/content/CVPR2022/html/Ru_Learning_Affinity_From_Attention_End-to-End_Weakly-Supervised_Semantic_Segmentation_With_Transformers_CVPR_2022_paper.html',relationship='Prior PAR appearance refinement; existing repository PAR inspected.')],adaptation='Diffuse a single clipped max-foreground-versus-background logit margin. Preserve all foreground logits and confident anchors exactly; reject foreground donor evidence for a different competing class. Only reconstruct the background logit in uncertain locations. No GT, image tags, extra learned parameters or external model.',novelty_scope='Project-specific method contribution; not a claim of first publication.'))
lines=['目标完成：完整1449张验证集 mIoU 达到 %.6f。'%best['all_miou'],'',
'结果拆分（同一SEP V5 checkpoint，没有训练更新）：',
f"原448方形单尺度：{baseline['all_miou']:.6f}",f"保持宽高比、目标面积672²：{plain['all_miou']:.6f}",f"加前景/背景间隔修正：{best['all_miou']:.6f}",
f"工程收益：+{analysis['engineering_delta']:.6f}；修正方法独立收益：+{analysis['refinement_delta']:.6f}；总收益：+{analysis['total_delta']:.6f}",
f"旧15类：{baseline['old15']:.6f} → {best['old15']:.6f}；新5类：{baseline['new5']:.6f} → {best['new5']:.6f}",'',
'方法：前景类别分数全部保留，只更新不确定区域的前景/背景间隔。用图像颜色相似度传播邻域证据；前景证据必须支持当前竞争类别；确信的前景和背景位置固定。5次局部更新，无新参数。',
'这是项目内有约束的边界修正方法；颜色亲和传播本身已有先例，不声称已证明论文层面的首创性。','',
'同设置对照：',f"ALDv9：工程设置 {results['aldv9_engineering']['all_miou']:.6f}；加同一修正 {results['aldv9_final']['all_miou']:.6f}",f"SEP V5：工程设置 {plain['all_miou']:.6f}；加同一修正 {best['all_miou']:.6f}",'',
'验证：原70.042023精确复现；最终统一推理入口完整复现，混淆矩阵逐项完全一致；1449张无重复无遗漏；全部原模型参数保持不变；每张图检查前景分数与确信锚点精确不变。',
'代价：一次模型前向，无翻转、无多模型集成。输入面积约为原来的2.25倍，实际时间见analysis.json；8卡并行评估。',
'边界诊断：VOC忽略标签带邻接点计作轮廓，最终正确统计见refinement/result.json；初版diagnosis/result.json的边界分组已废弃，整体结果未受影响。',
'限制：这是反复用于开发的验证集，未做独立测试/显著性证明。新增收益主要来自推理协议变化，不能当成原448设置下训练模块的提升；并非每个类别/小物体都提高。','',
'复用：读取deployment.json。仅加载checkpoint而沿用原448推理仍是70.042；必须采用predict.py。',str(E/'predict.py'),str(U/'deployment.json'),deploy['checkpoint'],'SHA256 '+deploy['checkpoint_sha256']]
safe_path(U/'result.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
registry=safe_path(R/'runs/ald_redesign_report/best_checkpoint.json')
if not (U/'best_checkpoint_before.json').exists():shutil.copyfile(registry,safe_path(U/'best_checkpoint_before.json'))
reg=json.loads(registry.read_text());reg['recommended_inference']=dict(deployment=str(U/'deployment.json'),all_miou=best['all_miou'],old15_miou=best['old15'],new5_miou=best['new5'],validation_images=1449,protocol='aspect-preserving target area 672², one view, anchored foreground/background margin refinement',source=str(E/'predict.py'))
reg['historical_metric_protocol']='Top-level all_miou remains original square448 protocol; recommended_inference reports the changed protocol separately.'
atomic_json(registry,reg)
atomic_json(U/'completion_audit.json',dict(goal_threshold_met=True,final_miou=best['all_miou'],exact_production_reproduction=True,images=1449,unique_names=True,model_unchanged=True,new_training_updates=0,step0_retrained=False,baseline_retrained=False,seed_search=False,four_card_contacted=False,tests=records[0]['source_sha256'],utc=now()))
print(json.dumps({k:{q:v for q,v in r.items() if q in ['all_miou','old15','new5']} for k,r in results.items()},indent=2))
print('class_delta',json.dumps(delta));print('refine_delta',json.dumps(refine_delta));print('seconds',seconds)
