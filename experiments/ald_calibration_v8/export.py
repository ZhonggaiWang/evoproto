from pathlib import Path
import os,sys,json,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v8';U=R/'runs/ald_calibration_v8';O=R/'runs/ald_redesign_report/v8'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
a=json.loads(safe_path(U/'analysis.json').read_text());assert a['status']=='complete_endpoint_analysis'
safe_path(O).mkdir(parents=True,exist_ok=True)
atomic_json(O/'analysis.json',a)
m=a['candidate'];r=a['reference'];d=a['delta'];p=a['pseudo_supervision_448_grid'];c=a['classification'];n=a['diagnostic_counts']
lines=['ALD 第8版：从伪监督入口进行早期校准',
    '日期：'+now(),'',
    '结果：'+('完整终点超过上一最佳；须结合新旧类代价审查。' if a['target_exceeded'] else '未超过上一最佳，不能认定这版ALD优化成功。'),
    f"完整1449张验证：{r['all_miou']:.6f} → {m['all_miou']:.6f} mIoU（{d['all_miou']:+.6f}）",
    f"旧15类：{r['old_miou']:.6f} → {m['old_miou']:.6f}；新5类：{r['new_miou']:.6f} → {m['new_miou']:.6f}",
    f"旧类像素精确率/召回率：{r['old_precision']:.3f}/{r['old_recall']:.3f} → {m['old_precision']:.3f}/{m['old_recall']:.3f}",
    f"新类像素精确率/召回率：{r['new_precision']:.3f}/{r['new_recall']:.3f} → {m['new_precision']:.3f}/{m['new_recall']:.3f}",
    f"背景误判新类像素：{r['BG_to_new']} → {m['BG_to_new']}；旧类误判新类：{r['old_to_new']} → {m['old_to_new']}",
    f"新类误判旧类：{r['new_to_old']} → {m['new_to_old']}",'',
    '方法：复用现有第2000步模型与Adam状态，冻结上一最佳学生作为弱证据/软监督参考。',
    '旧类图像标签分为确认存在、确认缺失、不确定；不确定项使用软目标，分类器阳性而无稠密支持的冲突项回到0.5。',
    '融合时禁止不受支持的旧类硬标签回填；旧教师背景与参考前景冲突也保留为不确定，并使用允许类别集合内的软分布约束。',
    '保留当前新类CAM/PAR的监督优先级；同步校准PTC、KD及混淆矩阵入口，保留已有语义保护SEP。',
    '旧类图像状态固定，增强图上的稠密参考预测和学生CAM/PAR在线重算。混淆矩阵从warmup重新开始，累计1500次更新。','',
    'GT可信度诊断（448×448网格；GT只在标签/掩码构造后进入统计）：',
    f"旧融合硬标签精度/覆盖率：{p['legacy_fused']['precision']:.3f}% / {p['legacy_fused']['coverage']:.3f}%",
    f"ALD融合硬标签精度/覆盖率：{p['ALD_fused']['precision']:.3f}% / {p['ALD_fused']['coverage']:.3f}%",
    f"不确定区域参考预测精度：{p['reference_unknown']['precision']:.3f}%",
    f"被撤掉的旧融合标签：错误{n['legacy_wrong_removed']}像素、正确{n['legacy_correct_removed']}像素。",
    f"最终旧类分类器精确率/召回率：{c['reference']['old']['precision']:.3f}/{c['reference']['old']['recall']:.3f} → {c['candidate']['old']['precision']:.3f}/{c['candidate']['old']['recall']:.3f}",'',
    '验证与限制：8卡损失/梯度检查、真实短程参数一致性检查已通过；完整终点模型与已评估模型张量逐项一致。',
    '只用8卡机；每卡4张图，全局32张，实际1500次更新、48000次样本曝光；学习率按原样本曝光进度推进，Adam次数不同。',
    '未重训step0或baseline，未筛随机种子。固定种子的自适应开发，不是统计显著性检验，也不是只改变ALD的严格因果对照。',
    '使用了已有当前阶段最佳学生作为固定参考，因此属于回退warmup后的自训练；不能宣传为只依赖上一阶段模型。',
    '混淆矩阵计数是图像曝光次数，重复出现不是独立样本。类别对GT可信度详见analysis.json。','',
    '候选模型：'+a['checkpoint'],'候选SHA256：'+a['checkpoint_sha256'],
    '旧最佳模型：'+str(R/'runs/pair_preserving_kd_v1/formal/10-5/step2/checkpoints/model_final.pth')]
safe_path(O/'result.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
atomic_json(O/'progress.json',{'utc':now(),'phase':'v8_early_ALD_complete','target_exceeded':a['target_exceeded'],
    'reference_miou':r['all_miou'],'candidate_miou':m['all_miou'],'analysis':str(U/'analysis.json'),'native_goal_completion_requires_review':True})
archive=safe_path(O/'implementation_and_records.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in E.glob('*.py'):z.write(safe_path(p),'experiment/'+p.name)
    for p in (E/'src').rglob('*.py'):
        if '__pycache__' not in p.parts:z.write(safe_path(p),'experiment/src/'+str(p.relative_to(E/'src')))
    for p in E.glob('*.json'):z.write(safe_path(p),'experiment/'+p.name)
    for p in U.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):z.write(safe_path(p),'run/'+str(p.relative_to(U)))
    for p in O.glob('*.json'):
        if p.name!='manifest.json':z.write(p,'report/'+p.name)
    z.write(O/'result.txt','report/result.txt')
atomic_json(O/'manifest.json',{'utc':now(),'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in O.iterdir() if p.is_file() and p.name!='manifest.json'}})
print(json.dumps({'output':str(O),'target_exceeded':a['target_exceeded']}))
