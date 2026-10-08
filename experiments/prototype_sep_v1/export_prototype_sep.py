"""Export verified endpoint evidence and source; never bundle large checkpoints."""
from pathlib import Path
import sys,json,zipfile,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/prototype_sep_v1';U=R/'runs/prototype_sep_v1'
S=E/'a_geometry/src';RUN=U/'formal/a_geometry'
sys.path.insert(0,str(S))
from kd_runtime import safe_path,atomic_json,digest,now

def micro(hist,old):
    ids=range(old+1,len(hist));tp=sum(hist[i][i] for i in ids)
    predicted=sum(sum(row[j] for j in ids) for row in hist)
    truth=sum(sum(hist[i]) for i in ids)
    return {'precision_percent':100*tp/predicted,'recall_percent':100*tp/truth,
        'true_positive':tp,'false_positive':predicted-tp,'false_negative':truth-tp,
        'definition':'Micro multiclass precision/recall across current-new foreground classes, not mean class IoU'}

def main():
    analysis=json.loads((U/'result_analysis.json').read_text())
    audit=json.loads((RUN/'completion_audit.json').read_text())
    assert audit['all_training_and_endpoint_requirements_complete']
    assert json.loads((RUN/'status.json').read_text())['status']=='complete'
    stage2=analysis['stages']['2'];recommendation=analysis['recommendation']
    stage1=analysis['stages']['1']
    selected=Path(recommendation['selected_checkpoint']);safe_path(selected)
    decision={**recommendation,'utc':now(),'selected_checkpoint_sha256':digest(selected),
        'selected_source':str(S if recommendation['candidate_exceeds_required_endpoint'] else R/'experiments/kd_parallel_v1/b_relational/src'),
        'goal_result_scope':'Completed one fixedseed0 full two-stage mechanism; empirical development-validation result; no stability/generalization claim',
        'host':'8card','4card':'unused','step0_retrained':False,'baseline_retrained':False,
        'integrity_receipt':str(RUN/'completion_audit.json')}
    atomic_json(U/'final_decision.json',decision)
    metrics=stage2['endpoint']['metrics'];ref=stage2['references']['optimized_KD']['metrics']
    delta=stage2['comparisons']['optimized_KD']['metric_delta_pp']
    key_old='previous_foreground_miou';key_new='current_foreground_miou'
    precision={'candidate':micro(stage2['endpoint']['histogram'],15),
        'optimized_KD':micro(stage2['references']['optimized_KD']['histogram'],15)}
    atomic_json(U/'precision_recall_report.json',precision)
    lines=['混淆指导原型 SEP 与优化 KD：完整终点结果',
        f"最终判断：{'本候选超过现有优化 KD' if recommendation['candidate_exceeds_required_endpoint'] else '本候选尚未超过现有优化 KD'}。",
        'VOC 10-5，固定 seed=0，阶段2/8000步，完整1449张验证图；整体mIoU包含背景，旧15/新5只含前景。',
        '', '方案 | 整体mIoU | 旧15 | 新5',
        f"现有优化 KD | {ref['all_miou']:.6f} | {ref[key_old]:.6f} | {ref[key_new]:.6f}",
        f"混淆指导原型 SEP + 优化 KD | {metrics['all_miou']:.6f} | {metrics[key_old]:.6f} | {metrics[key_new]:.6f}",
        f"差值（百分点） | {delta['all_miou']:+.6f} | {delta[key_old]:+.6f} | {delta[key_new]:+.6f}",
        f"阶段1终点：本候选 {stage1['endpoint']['metrics']['all_miou']:.6f}，现有优化KD {stage1['references']['optimized_KD']['metrics']['all_miou']:.6f}；最终选择按完整阶段2终点。",
        '', '方法：过去在线定向混淆负责挑选涉及新类的类别对；完整 broad EMA 行排序，并要求可信CAM类别对曝光支持。比例不作为精确学习权重。',
        '选中方向去重为原型几何对；旧新对只在SEP项中detach当前学生旧原型，新新对更新双方；背景/旧旧/自配对排除。',
        '固定lambda0.1，继承margin0；平方余弦hinge对全部有效对取均值，达到间隔后该项严格为零。',
        '优化KD代码字节保持一致。原型SEP当前只直接更新原型；通过后续原型分割训练间接影响特征学习。',
        '', '验证：41项CPU行为测试、5张实际训练图像的分量梯度检查、8卡两阶段流程检查通过；7个正式全量评估任务及每个4分片均核对。',
        '同一阶段的最终模型、优化器、在线混淆和选择器状态与被评估8000步检查点逐张量一致。',
        '完整审计结果见completion_audit.json；具体类别、双向GT混淆变化与采样机制统计见result_analysis.json。',
        '', '复用：step0、阶段1模型和优化器的2000步检查点、现有所有基准；阶段1新增6000更新，阶段2完整8000更新。',
        '资源：8卡机8张GPU训练，4个轻量评估分片在同机按队列运行；4卡机未使用，所有远程输出/缓存限定授权工作区。',
        '', '局限：当前候选8x1与历史优化KD的4x2保持global batch8，但采样/增强流重启；差值不能严格归因于SEP单一因素。',
        '这是固定seed0、同一任务的开发验证结果；没有多随机种子稳定性、显著性或独立测试泛化结论。',
        '可信CAM支持记录的是重复训练曝光，不能当成独立样本或GT正确率；单向误判改善不等于整体IoU改善。',
        '', f"保留实现：{decision['selected_source']}",f"保留检查点：{selected}",f"检查点SHA256：{decision['selected_checkpoint_sha256']}",
        '模型检查点保留在远程授权目录，代码包包含实现和证据，不打包大模型权重。']
    report=safe_path(U/'prototype_sep_result.txt');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=safe_path(U/'prototype_sep_implementation.zip');tmp=safe_path(U/'prototype_sep_implementation.tmp.zip')
    assert not archive.exists() and not tmp.exists(),'Refuse accidental duplicate export'
    entries={}
    def add(path,name):
        path=safe_path(path)
        if path.is_file():
            assert name not in entries,name
            entries[name]=path
    for p in S.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:add(p,'candidate/src/'+str(p.relative_to(S)))
    for p in E.glob('*.py'):add(p,'study/'+p.name)
    for p in E.glob('*.json'):add(p,'study/'+p.name)
    for name in ['result_analysis.json','final_decision.json','precision_recall_report.json','preflight.json','gradient_probe.json','prototype_sep_result.txt']:
        add(U/name,'evidence/'+name)
    for p in U.glob('endpoint*diagnostic*.json'):add(p,'evidence/'+p.name)
    for p in U.glob('stage*diagnostic*.json'):add(p,'evidence/'+p.name)
    for p in RUN.rglob('*.json'):
        if 'checkpoints' not in p.parts:add(p,'evidence/formal/'+str(p.relative_to(RUN)))
    for step in [1,2]:
        for name in ['geometry_metrics.jsonl','kd_metrics.jsonl']:
            add(RUN/f'10-5/step{step}'/name,f'evidence/formal/step{step}/'+name)
        baseline=R/f'runs/kd_parallel_v1/formal/b_relational/evaluations/step{step}_iter8000/result.json'
        add(baseline,f'evidence/reference/optimized_KD_step{step}_iter8000.json')
    with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,p in sorted(entries.items()):z.write(p,name)
    os.replace(tmp,archive)
    manifest={'utc':now(),'files':{name:{'sha256':digest(U/name),'size':(U/name).stat().st_size} for name in
        ['prototype_sep_result.txt','prototype_sep_implementation.zip','final_decision.json','result_analysis.json','precision_recall_report.json']},
        'zip_entries':len(entries),'zip_entry_sha256':{name:digest(p) for name,p in sorted(entries.items())},
        'source_sha256':json.loads((RUN/'study.json').read_text())['source_sha256']}
    atomic_json(U/'export_manifest.json',manifest)
    print(json.dumps({'files':list(manifest['files']),'zip_entries':len(entries),'decision':recommendation['status']}))

if __name__=='__main__':main()
