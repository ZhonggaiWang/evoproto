from pathlib import Path
import sys,json,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/hierarchy_kd_v2';U=R/'runs/hierarchy_kd_v2'
sys.path.insert(0,str(E/'src'))
from kd_runtime import safe_path,digest,atomic_json,now
analysis=json.loads(safe_path(U/'analysis.json').read_text())
diagnostic=json.loads(safe_path(U/'endpoint_diagnostic.json').read_text())
assert analysis['status']=='complete_endpoint_analysis' and diagnostic['passed']
assert analysis['final_checkpoint_sha256']==diagnostic['student_sha256']
out=safe_path(U/'export');out.mkdir(parents=True,exist_ok=True)
m=analysis['metrics'];delta=analysis['deltas']['C_new_anchor'];counts=analysis['all_training_updates']
rows=['混淆指导分层 KD + 可信背景保护：完整终点评估',
      '', '评估：VOC 10-5，step2/8000，完整1449张；固定seed0开发实验。',
      '方案                         全部mIoU     旧15类      新5类']
for key,title in [('optimized_KD','原最优KD'),('C_new_anchor','此前最优KD+SEP'),('candidate','本次修正版H2')]:
    x=m[key];rows.append(f"{title:24s} {x['all_miou']:.6f}  {x['old_miou']:.6f}  {x['new_miou']:.6f}")
rows += ['',f"相对此前最优：全部 {delta['all_miou']:+.6f}，旧类 {delta['old_miou']:+.6f}，新类 {delta['new_miou']:+.6f} 个百分点。",
 '结论：本候选未达到提升目标，保留此前最优检查点，不替换。' if delta['all_miou']<=0 else '结论：本候选数值提高；仍需结合误检、召回代价判断，不能宣称统计显著。',
 f"新类精度变化 {delta['new_precision']:+.6f}，召回变化 {delta['new_recall']:+.6f} 个百分点。",
 f"背景误判新类像素 {m['C_new_anchor']['BG_to_new_pixels']} -> {m['candidate']['BG_to_new_pixels']}。",
 f"新类漏到背景像素 {m['C_new_anchor']['new_to_BG_pixels']} -> {m['candidate']['new_to_BG_pixels']}。",
 '', '实际实现：',
 '1. 原有旧类内部条件KD逐字保持；另加旧类组相对背景+新类组的概率质量保留下限。',
 '2. 旧类锚点要求教师、PAR、最强CAM一致；达到教师旧类质量下限后不再施加该项约束。',
 '3. 教师BG+PAR背景一致、无强新类CAM的错误前景，可作背景成对纠错。',
 '4. 图像级标签确认不存在的新类，只抑制该缺席类别；允许其他旧类或新类作为正确答案。',
 '5. 单独在线统计与这些门控相同的可用证据；旧类->背景或新类、背景->前景各选最多3个方向。',
 '6. 混淆率仅用于排序；背景损失按有效纠错像素占ROI比例的平方根缩放，避免稀少像素占满整批损失预算。',
 '',f'6000次新增更新的累计监督记录：{json.dumps(counts,ensure_ascii=False)}',
 '', '已验证：教师无梯度、背景/新类组映射、缺席类别梯度方向、空ROI零梯度、证据状态恢复；',
 '8卡NCCL全局损失、DDP平均梯度及在线矩阵与合并整批计算一致；真实训练短跑通过。',
 '终点模型、优化器、两套在线统计和选择器状态与实际评估的8000步检查点逐项完全相同。',
 '', '本轮失败分支H1：第6000步全量评估66.024103，新类45.536881；同期旧方案新类54.451890。',
 '误检下降伴随明显召回损失，已提前停止并保存检查点。它没有8000步终点，不能作为同终点最终候选。',
 '', '边界：GT只诊断已经固定的门控，没有进入训练损失、在线矩阵或权重。',
 'GT门控诊断是固定128张原生输出网格的开发子集；与1449张原图尺寸评估含义不同。',
 '本结果来自单种子、自适应开发过程，不声称统计显著或独立测试集泛化。',
 '只使用8卡机；复用step0、原最优KD教师和step2/2000模型+优化器。未重训baseline或搜索随机种子。',
 '', '本次候选终点检查点：'+analysis['final_checkpoint'], 'SHA256：'+analysis['final_checkpoint_sha256']]
target=safe_path(out/'kd_hierarchy_result.txt')
with target.open('x',encoding='utf-8') as f:f.write('\n'.join(rows)+'\n')
artifacts={
 'analysis.json':U/'analysis.json','endpoint_diagnostic.json':U/'endpoint_diagnostic.json',
 'readiness.json':R/'runs/hierarchy_kd_v1/readiness.json',
 'readiness_extended.json':R/'runs/hierarchy_kd_v1/readiness_extended.json',
 'reference_production_diagnostic.json':R/'runs/hierarchy_kd_v1/reference_production_diagnostic.json',
 'new_complement_readiness.json':U/'new_complement_readiness.json',
 'H1_early_stop_analysis.json':R/'runs/hierarchy_kd_v1/early_stop_analysis.json',
 'H2_preflight.json':E/'preflight.json','H2_origin.json':E/'origin.json'}
for name,path in artifacts.items():
    with safe_path(out/name).open('xb') as f:f.write(safe_path(path).read_bytes())
archive=safe_path(out/'implementation.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for version in [1,2]:
        exp=R/f'experiments/hierarchy_kd_v{version}'
        for p in sorted((exp/'src').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(safe_path(p),f'H{version}/src/'+p.relative_to(exp/'src').as_posix())
        for p in exp.glob('*.py'):z.write(safe_path(p),f'H{version}/scripts/'+p.name)
        for name in ['origin.json','preflight.json']:z.write(safe_path(exp/name),f'H{version}/'+name)
        run=R/f'runs/hierarchy_kd_v{version}/formal/h{version}'
        for p in run.rglob('*.json'):
            if 'checkpoints' not in p.parts:z.write(safe_path(p),f'H{version}/run/'+p.relative_to(run).as_posix())
        for p in (run/'10-5/step2').glob('*.jsonl'):z.write(safe_path(p),f'H{version}/run/10-5/step2/'+p.name)
    z.write(safe_path(R/'experiments/prototype_sep_v1/run_new_anchor_sep.py'),'support/run_new_anchor_sep.py')
    z.write(safe_path(R/'experiments/kd_pixel_v2/run_kd.py'),'support/run_kd.py')
    for p in out.iterdir():
        if p.is_file() and p!=archive:z.write(safe_path(p),'reports/'+p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=z.namelist();assert len(names)==len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
manifest={'created_utc':now(),'archive_entries':len(names),'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file()}}
atomic_json(out/'manifest.json',manifest)
print(json.dumps({'export':str(out),'archive_entries':len(names),'all_miou':m['candidate']['all_miou'],'delta':delta['all_miou']}))
