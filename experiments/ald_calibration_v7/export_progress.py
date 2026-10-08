from pathlib import Path
import sys,os,json,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/ald_calibration_v7';U=R/'runs/ald_calibration_v7';OUT=R/'runs/ald_redesign_report'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
read=lambda p:json.loads(safe_path(p).read_text())
OUT=safe_path(OUT);OUT.mkdir(exist_ok=True)
labels=['三态门控与可信硬监督','软集合投影与图像冲突校准','刷新CAM后的软标签纠错','可信旧类保护、分类保留与停止继续强化','保持原分类方向的受限修正','受限修正加图像缺席监督','只调整最后一层空间解码特征']
attempts=[]
for i,label in enumerate(labels,1):
    root=safe_path(R/f'runs/ald_calibration_v{i}');src=safe_path(R/f'experiments/ald_calibration_v{i}')
    a=read(root/'analysis.json');study=read(root/'formal/study.json');done=read(root/'formal/training_complete.json')
    assert all(a['verified'].values()) and done['steps']==300
    assert digest(safe_path(a['checkpoint']))==a['checkpoint_sha256']==done['checkpoint_sha256']
    assert all(digest(src/k)==v for k,v in study['sources'].items())
    pre=read(src/'preflight.json');assert pre['passed'] and all(digest(src/k)==v for k,v in pre['sources'].items())
    result=read(root/'formal/evaluations/step2_iter8600/result.json');assert result['images']==1449 and result['checkpoint_sha256']==a['checkpoint_sha256']
    attempts.append({'version':i,'method':label,'metrics':a['candidate'],'delta':a['delta'],'checkpoint':a['checkpoint'],'checkpoint_sha256':a['checkpoint_sha256'],'analysis':str(root/'analysis.json'),'verified':a['verified']})
    atomic_json(OUT/f'v{i}_analysis.json',a)
previous=read(R/'runs/pair_preserving_kd_v1/analysis.json')
baseline={'method':'previous best KD','metrics':previous['candidate'],'checkpoint':previous['checkpoint'],'checkpoint_sha256':previous['checkpoint_sha256']}
assert digest(safe_path(baseline['checkpoint']))==baseline['checkpoint_sha256']
best=max([baseline,*attempts],key=lambda x:x['metrics']['all_miou']);improved=best['metrics']['all_miou']>baseline['metrics']['all_miou']
cal=read(R/'runs/ald_calibration_v2/analysis.json')['classification']
image=read(R/'runs/ald_calibration_v1/diagnostic.json')
gate=read(R/'runs/ald_calibration_v4/diagnostic.json')
train_gt=read(R/'runs/ald_calibration_v4/training_GT_diagnostic.json')
report={'utc':now(),'goal_target_met':improved,'decision':'promote_verified_candidate' if improved else 'retain_previous_best_continue_goal',
    'previous_best':baseline,'best':best,'attempts':attempts,'classification_calibration':cal,'latest_gate_diagnostic':gate,
    'training_GT_diagnostic_only':train_gt,'shared_step0_retrained':False,'baseline_retrained':False,'seed_screening':False,
    'machine':'8-card101.126.31.72:38577 only','write_boundary':'/ML-vePFS/infra_rd/kun/others/wzg',
    'limits':['同一验证集上的固定种子自适应研发；不声称统计显著或同预算优势','像素GT与旧类图像GT只用于诊断/评估，不进入训练监督','在线混淆矩阵旧状态仅保留溯源，本轮没有更新或使用其不可靠的剩余类别对','分类器校准存在召回损失；局部伪标签准确率不代表最终分割收益','本轮使用冻结/分阶段刷新的证据，不是每步在线重算CAM']}
atomic_json(OUT/'progress.json',report);atomic_json(OUT/'best_checkpoint.json',best)
table='\n'.join(f"v{x['version']} {x['method']}：{x['metrics']['all_miou']:.6f}（相对原最佳{x['delta']['all_miou']:+.6f}）" for x in attempts)
text=f'''ALD 重设计：已验证阶段记录

目标是否超过原最佳：{improved}
原最佳 mIoU：{baseline['metrics']['all_miou']:.6f}
当前保留最佳 mIoU：{best['metrics']['all_miou']:.6f}

{table}

已落实的机制
1. 旧分类器预测须经过图像/空间/双视图证据校准，形成存在、不存在、不确定三态。
2. 不确定项不直接作负标签；分类器单独声称存在但缺乏空间支持时，使用降低确定性的监督。
3. 校准分类器用于刷新 CAM/PAR。可信旧类证据能否决新类覆盖；不把被拒绝旧类直接写成背景。
4. 通过门控的冲突只修正候选类别与当前赢家的概率，保留其他类别关系。纠正后停止继续推高，不把剩余困难样本重新放大。
5. 当前新类图像标签中的缺席信息只排除不可能类别，不指定像素必须属于背景或某个旧类。

辅助指标与代价
旧类图像预测 precision：{cal['reference']['old']['precision']:.4f}% -> {cal['candidate']['old']['precision']:.4f}%
旧类图像预测 recall：{cal['reference']['old']['recall']:.4f}% -> {cal['candidate']['old']['recall']:.4f}%
旧类图像误报：{cal['reference']['old']['FP']} -> {cal['candidate']['old']['FP']}
最新候选纠错门控：验证集593/649=91.37%正确；并不保证分割性能提升。
训练集只读GT诊断：sheep候选989/1049=94.28%正确；早期直接修改全局输出权重仍严重损伤sheep/旧类，说明局部准确与全局泛化不可混同。

所有正式候选均完成1449张、原GT分辨率的终点评估，并核对检查点身份；训练使用2145张当前阶段图像。没有使用训练像素GT或旧类图像GT。GT诊断报告不会作为后续训练输入。
复用 step0、已有最佳模型和适用的优化器状态；没有重跑baseline、随机种子筛选或参数网格。只使用8卡机，所有远程写入均在授权目录内。

限制
这是固定种子的自适应研发；不是独立留出测试或同预算比较。分类校准的召回回落已列出，不能只宣传误报减少。旧混淆矩阵/选择器仅保存溯源，本轮不声称更新了它。v5/v6保存了独立残差优化器，折叠模型普通续训会重置新类输出头的优化器状态。

保留检查点：{best['checkpoint']}
SHA256：{best['checkpoint_sha256']}
'''
safe_path(OUT/'result.txt').write_text(text,encoding='utf-8')
with zipfile.ZipFile(safe_path(OUT/'implementation_and_records.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for i in range(1,8):
        src=safe_path(R/f'experiments/ald_calibration_v{i}');root=safe_path(R/f'runs/ald_calibration_v{i}')
        for p in sorted(src.glob('*.py')):z.write(safe_path(p),f'v{i}/src/{p.name}')
        for p in sorted(root.rglob('*.json')):z.write(safe_path(p),f'v{i}/records/{p.relative_to(root)}')
        for p in sorted(root.rglob('training.jsonl')):z.write(safe_path(p),f'v{i}/records/{p.relative_to(root)}')
    z.writestr('result.txt',text)
files=['progress.json','best_checkpoint.json','result.txt','implementation_and_records.zip']+[f'v{i}_analysis.json' for i in range(1,8)]
atomic_json(OUT/'manifest.json',{'utc':now(),'artifacts':{f:{'sha256':digest(OUT/f),'bytes':(OUT/f).stat().st_size} for f in files}})
print(json.dumps({'goal_target_met':improved,'best':best,'report':str(OUT),'artifacts':files},ensure_ascii=False))
