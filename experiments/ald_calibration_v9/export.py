from pathlib import Path
import os,sys,json,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v9';U=R/'runs/ald_calibration_v9';REPORT=R/'runs/ald_redesign_report';O=REPORT/'v9'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
read=lambda p:json.loads(safe_path(p).read_text())
a=read(U/'analysis.json');audit=read(U/'completion_audit.json');weak=read(U/'weak_input_audit.json')
assert a['target_exceeded'] and all(audit['requirements'].values()) and weak['passed']
assert digest(a['checkpoint'])==a['checkpoint_sha256']
safe_path(O).mkdir(parents=True,exist_ok=True)
for n in ['analysis.json','completion_audit.json','weak_input_audit.json','readiness.json']:atomic_json(O/n,read(U/n))
m=a['candidate'];r=a['reference'];d=a['delta'];cl=a['classification']
best={'utc':now(),'name':'ALD_v9_conflict_partial_set','checkpoint':a['checkpoint'],'checkpoint_sha256':a['checkpoint_sha256'],
    'all_miou':m['all_miou'],'old15_miou':m['old_miou'],'new5_miou':m['new_miou'],'validation_images':1449,
    'previous_best':str(R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth'),
    'previous_best_sha256':'3f43207d665ce6001aa149867435364e13baca5d47a0e2274dfb9187975dd28d',
    'previous_best_retained':True,'single_model_inference':True,'analysis':str(U/'analysis.json'),'completion_audit':str(U/'completion_audit.json'),
    'scope':'VOC10-5 step2; fixed-seed adaptive development; numerical improvement, no statistical significance claim'}
atomic_json(O/'best_checkpoint.json',best)
old=REPORT/'best_checkpoint.json'
if old.exists() and not (REPORT/'best_checkpoint_before_v9.json').exists():atomic_json(REPORT/'best_checkpoint_before_v9.json',read(old))
atomic_json(old,best)
lines=['ALD优化完成记录：冲突区域使用候选集合与条件前景关系',
    '日期：'+now(),'',
    f"最终完整1449张验证：{r['all_miou']:.6f} → {m['all_miou']:.6f} mIoU（+{d['all_miou']:.6f}）。",
    f"旧15类：{r['old_miou']:.6f} → {m['old_miou']:.6f}；新5类：{r['new_miou']:.6f} → {m['new_miou']:.6f}。",
    '已登记为本次探索的新最佳，原最佳和全部失败实验保留。','',
    '实际修改：',
    '1. 旧类图像级预测使用确认存在、确认缺失、不确定三种状态；未知不当负类，空间证据不支持的阳性分类冲突不再变为强硬标签。',
    '2. 旧类、当前新类与背景硬标签需要相应证据一致；新类CAM/PAR与固定参考冲突时，不再强制选择其中一个类别。',
    '3. 冲突像素保留候选集合，只排除可信的图像级缺失类别。关系损失只学习允许前景类别之间的相对分布，不指定前景/背景总强度。',
    '4. PTC和旧类KD同步处理不可信像素；保留有效的条件旧类KD。混淆矩阵重新累计，当前新类行仍由独立CAM/PAR证据提供，避免因参考一致性而人为消除混淆。',
    '5. 本次额外优化不再施加几何SEP，沿用已通过SEP和KD训练的模型，并保留原型语义监督。','',
    '效果与代价：',
    f"新类像素精确率：{r['new_precision']:.3f}% → {m['new_precision']:.3f}%；召回率：{r['new_recall']:.3f}% → {m['new_recall']:.3f}%。",
    f"旧类像素精确率：{r['old_precision']:.3f}% → {m['old_precision']:.3f}%；召回率：{r['old_recall']:.3f}% → {m['old_recall']:.3f}%。",
    f"背景误判新类：{r['BG_to_new']} → {m['BG_to_new']}像素；新类误判旧类：{r['new_to_old']} → {m['new_to_old']}。",
    f"代价包括旧类误判背景增加{d['old_to_BG']}像素，餐桌IoU下降{-a['class_iou_delta']['diningtable']:.3f}，羊下降{-a['class_iou_delta']['sheep']:.3f}，沙发下降{-a['class_iou_delta']['sofa']:.3f}。",
    f"旧类图像分类精确率：{cl['reference']['old']['precision']:.3f}% → {cl['candidate']['old']['precision']:.3f}%；召回率：{cl['reference']['old']['recall']:.3f}% → {cl['candidate']['old']['recall']:.3f}%。",'',
    '固定最佳检查点上的训练前GT诊断（非训练后指标）：硬标签精度91.960%→93.798%，覆盖96.418%；新类硬标签精度73.394%→76.584%；冲突候选集合保留99.844%的真实类别。',
    'GT仅用于诊断与验证，不进入训练标签构造。2145张训练图像与1449张验证图像ID无交集；旧类图像状态逐项核对为原始弱推理缓存。','',
    '训练与验证：只用8卡机，8卡×每卡4张，300次额外更新，共9600次样本曝光。复用最佳模型及对应Adam状态，编码器2e-6/分类与分割头2e-5余弦衰减。',
    '未重训step0或baseline，未筛随机种子或做参数网格。8卡梯度、真实短程参数一致性、检查点安全读取均通过；终点与额外分类诊断得到完全一致的像素混淆统计。',
    '固定种子的自适应开发结果；额外训练及损失策略均有变化，不作统计显著性或纯ALD因果收益声明。条件前景损失的梯度性质不等于共享参数更新后所有像素的前景概率严格不变。',
    '新版ALD直接在独立源码中启用；配置中ald=False指旧版ALD开关关闭。验证范围是VOC10-5的step2。',
    '部署仍是原来的单模型结构，不需要双模型集成或额外推理阶段。','',
    '新最佳模型：'+a['checkpoint'],'SHA256：'+a['checkpoint_sha256'],
    '实现：'+str(E/'src'),'训练、诊断与审计：'+str(U),'',
    '各类别IoU变化：']
lines += [f"{k}: {r['class_iou'][k]:.6f} → {m['class_iou'][k]:.6f} ({v:+.6f})" for k,v in a['class_iou_delta'].items()]
safe_path(O/'result.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
atomic_json(O/'progress.json',{'utc':now(),'status':'objective_verified','target_exceeded':True,'best':best,'goal_completion_evidence':str(U/'completion_audit.json')})
atomic_json(REPORT/'next_goal_state.json',{'utc':now(),'goal_status':'objective_verified_ready_for_native_completion','best_checkpoint':best,
    'requirements_audit':str(U/'completion_audit.json'),'next_action':'No further experiment required for this goal; complete native goal after local export integrity verification. Preserve all previous results.',
    '8card_jobs_terminal':True,'4card_untouched':True})
with zipfile.ZipFile(safe_path(O/'implementation_and_records.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in E.glob('*.py'):z.write(safe_path(p),'experiment/'+p.name)
    for p in (E/'src').rglob('*.py'):
        if '__pycache__' not in p.parts:z.write(safe_path(p),'experiment/src/'+str(p.relative_to(E/'src')))
    for p in E.glob('*.json'):z.write(safe_path(p),'experiment/'+p.name)
    z.write(safe_path(weak['image_evidence']),'inputs/image_evidence.pth')
    for n in ['evidence.py','cache_evidence.py']:z.write(safe_path(R/'experiments/ald_calibration_v1'/n),'inputs/weak_evidence_source/'+n)
    for p in U.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.log','.txt']:z.write(safe_path(p),'run/'+str(p.relative_to(U)))
    for p in O.iterdir():
        if p.is_file() and p.suffix in ['.json','.txt'] and p.name!='manifest.json':z.write(p,'report/'+p.name)
atomic_json(O/'manifest.json',{'utc':now(),'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in O.iterdir() if p.is_file() and p.name!='manifest.json'}})
print(json.dumps({'promoted':best,'output':str(O)}))
