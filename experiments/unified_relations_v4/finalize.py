from common import *
import json,subprocess,zipfile,shutil
import numpy as np
import torch

W=safe_path(R/'runs/unified_method_report');W.mkdir(exist_ok=True)
verification=json.loads((U/'verification/result.json').read_text());assert verification['exact_reproduction']
deploy=json.loads((U/'deployment.json').read_text());result=json.loads((U/'formal/result.json').read_text());assert result['images']==1449
parts=[json.loads((U/f'formal/eval_rank{i}.json').read_text()) for i in range(8)];assert len({p['checkpoint_sha256'] for p in parts})==1
assert all(json.loads((U/f'formal/audit_rank{i}.json').read_text())['passed'] for i in range(8))
assert all(json.loads((U/f'formal/audit_rank{i}.json').read_text())['encoder_and_classifiers_bitwise_unchanged'] for i in range(8))
study=json.loads((U/'formal/study.json').read_text());assert all(digest(E/k)==v for k,v in study['source_sha256'].items())
assert not set(study['train_images'])&set(sum([p['images'] for p in parts],[]))
hist=np.array(result['results']['aspect672']['histogram']);parent=json.loads((R/'runs/spatial_recovery_v1/verification/result.json').read_text())['results']['aldv9_engineering']
variants={f'v{i}':json.loads((R/f'runs/unified_relations_v{i}/formal/result.json').read_text())['results'] for i in range(1,5)}
last=json.loads((U/'formal/metrics.jsonl').read_text().splitlines()[-1]);assert last['cumulative_counts'][4]>0 and last['sep']>0 and last['kd']>0
graph=json.loads((U/'formal/graph.json').read_text());assert [16,0] in graph['supported_pairs'] and len(graph['supported_pairs'])==20
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert not processes.strip(),processes
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
analysis=dict(preferred_version='v4',selection_reason='User prioritizes a coherent simple four-module mechanism; v4 also retains ~71 performance.',results=result['results'],same_setting_parent=parent,variants=variants,delta_same_setting=result['results']['aspect672']['all_miou']-parent['all_miou'],teacher_count=1,graph_count=1,new_parameters=0,changed_tensors=verification['changed_tensors'],global_batch=32,updates=300,seed=0,seed_search=False,selected_pair_pixel_exposures=last['cumulative_counts'][3],active_SEP_pixel_exposures=last['cumulative_counts'][4],valid_pixel_exposures=last['cumulative_counts'][0],supported_pairs=graph['supported_pairs'],bg_to_fg=int(hist[0,1:].sum()),fg_to_bg=int(hist[1:,0].sum()),original_best_preserved=True,validation_protocol='1449 images, aspect672 single view, no postprocessing',scope=study['scope'],utc=now())
atomic_json(W/'analysis.json',analysis)
atomic_json(W/'preferred_method.json',dict(**deploy,priority='method_coherence_over_peak_metric',method_description=str(E/'method.txt'),historical_peak=dict(all_miou=71.6447728273661,deployment=str(R/'runs/spatial_recovery_v1/deployment.json'),retained=True),full_incremental_retraining_completed=False))
atomic_json(W/'implementation_audit.json',dict(passed=True,source_matches_training=True,complete_validation_exactly_reproduced=True,all8_DDP_weights_exact=True,encoder_and_classifiers_bitwise_unchanged=True,only5_existing_decoder_tensors_changed=True,all_four_modules_active=True,one_teacher_one_graph=True,separate_hard_CE_removed=True,graph_includes_BG=True,GT_role='Current-new image tags for training; pixel GT only diagnosis/evaluation; no GT inputs at inference',old_image_GT_passed_to_training=False,train_validation_disjoint=True,independent_all_stage_claim=False,all_gpus_released=True,gpu_snapshot=gpus,preflight=json.loads((E/'preflight.json').read_text()),utc=now()))
shutil.copyfile(safe_path(E/'method.txt'),safe_path(W/'method.txt'))
summary='''四模块统一目标已完成。

核心逻辑：ALD 与有向混淆矩阵定义允许修改的监督分布；KD 保留组间关系，SEP 纠正组内错误排序。两项是同一个 KL 约束的分解，而非独立加权的补丁。

最终完整1449张验证集：71.516129 mIoU。
旧15类：74.061501；新5类：59.668321。
原448方形协议：70.079566。
单尺度、保持宽高比、输入面积约672²；无翻转、集成、分离适配器或边界后处理。

统一训练只用一个冻结参考模型、一份矩阵、一个现有分割头；无新增可训练参数。背景也进入同一分布，避免前景整体失去约束。完整推理入口已精确复现结果。

重要范围：本轮是复用已有ALDv9模型的阶段内统一训练，300步，只更新分割头；不是从step0重新运行全部增量阶段。既有71.645高性能版本保留。没有筛选随机种子。

方法、公式、限制与实现对应关系见 method.txt；对照、失败版本和逐类数据见 analysis.json；复用入口与checkpoint见 preferred_method.json。
'''
safe_path(W/'result.txt').write_text(summary,encoding='utf-8')
registry=safe_path(R/'runs/ald_redesign_report/best_checkpoint.json')
shutil.copyfile(registry,safe_path(W/'registry_before.json'))
reg=json.loads(registry.read_text());reg['preferred_unified_method']=dict(registry=str(W/'preferred_method.json'),checkpoint=deploy['checkpoint'],checkpoint_sha256=deploy['checkpoint_sha256'],source=str(E),all_miou=deploy['all_miou'],square448_miou=deploy['square448_miou'],reason='Method coherence prioritized by user; single-scale without postprocessing; historic peak retained separately',training_scope=deploy['training_scope'])
atomic_json(registry,reg)
atomic_json(W/'progress.json',dict(status='complete',target='coherent four-module method first, ~71 second',achieved_miou=deploy['all_miou'],four_card_contacted=False,all_gpus_released=True,utc=now()))
atomic_json(W/'next_goal_state.json',dict(status='complete',preferred_method=str(W/'preferred_method.json'),caveat='No all-stage clean retraining has been done; next research validation requires a separately scoped request. Do not misattribute inherited warm-start performance.',utc=now()))
bundle=safe_path(W/'implementation_and_records.zip');paths=[]
for i in range(1,5):
 for base in [R/f'experiments/unified_relations_v{i}',R/f'runs/unified_relations_v{i}']:
  paths.extend(p for p in base.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.jsonl','.log','.txt'])
paths.extend(p for p in W.iterdir() if p.suffix in ['.json','.txt'] and p.name!='manifest.json')
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(set(paths)):safe_path(p);z.write(p,str(p.relative_to(R)))
with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
names=['method.txt','result.txt','analysis.json','preferred_method.json','implementation_audit.json','progress.json','next_goal_state.json','implementation_and_records.zip']
atomic_json(W/'manifest.json',dict(files={n:dict(bytes=(W/n).stat().st_size,sha256=digest(W/n)) for n in names},archive_files=len(set(paths)),weights_included=False,checkpoint=deploy['checkpoint'],checkpoint_sha256=deploy['checkpoint_sha256'],utc=now()))
print(json.dumps(dict(all_miou=deploy['all_miou'],files=len(set(paths)),archive_bytes=bundle.stat().st_size,method_report=str(W),gpus=gpus)))
