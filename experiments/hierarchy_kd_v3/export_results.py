from pathlib import Path
import sys,json,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/hierarchy_kd_v3';U=R/'runs/hierarchy_kd_v3'
sys.path.insert(0,str(E/'src'))
from kd_runtime import safe_path,digest,atomic_json,now
read=lambda p:json.loads(safe_path(p).read_text())
a=read(U/'analysis.json');g=read(U/'endpoint_diagnostic.json')
assert a['status']=='complete_endpoint_analysis' and g['passed'] and a['final_checkpoint_sha256']==g['student_sha256']
m=a['metrics'];d=a['deltas']['C_new_anchor'];out=safe_path(U/'export');out.mkdir(parents=True,exist_ok=True)
lines=['EvoProto：混淆指导的前景概率转移KD与可信背景保护','',
 'VOC 10-5，step2/8000，完整1449张验证；固定seed0。',
 '方法                         全部mIoU       旧15类         新5类']
for key,title in [('optimized_KD','原最优KD'),('C_new_anchor','此前最优KD+SEP'),('H2','分层KD修正版H2'),('candidate','有向概率转移H3')]:
    x=m[key];lines.append(f"{title:23s} {x['all_miou']:.6f}   {x['old_miou']:.6f}   {x['new_miou']:.6f}")
lines+=['',f"相对此前最优：全部 {d['all_miou']:+.6f}、旧类 {d['old_miou']:+.6f}、新类 {d['new_miou']:+.6f} 个百分点。",
 f"新类精度 {d['new_precision']:+.6f}、召回 {d['new_recall']:+.6f} 个百分点。",
 f"背景误判新类像素 {m['C_new_anchor']['BG_to_new_pixels']} -> {m['candidate']['BG_to_new_pixels']}。",
 f"新类漏到背景像素 {m['C_new_anchor']['new_to_BG_pixels']} -> {m['candidate']['new_to_BG_pixels']}。",
 '候选未超过已有最优，不替换。' if d['all_miou']<=0 else '候选终点数值超过已有最优；精度、召回和误检代价见上表，不能据单次实验声称统计显著。',
 '', '最终采用的KD：',
 '1. 保留原有旧类内部条件KD及原有SEM保护的原型SEP。',
 '2. 在线有向混淆选择新类j -> 旧类i；当前像素还须满足强CAM、PAR都支持j，而且教师预测为i。',
 '3. 将教师识别为旧类i的前景概率，作为学生新类j概率的保留下限；学生j已更强时停止该项。',
 '4. 这会增强新类j并降低包括i和背景在内的其他输出；不把教师背景当作新类正标签。',
 '5. 混淆比例只用于选择方向，不直接作为精确学习权重；当前CAM间隔及教师可靠性用于证据加权。',
 '6. 保留按ROI纠错覆盖率平方根缩放的背景纠错与图像级缺席新类抑制。',
 '7. 旧类组质量约束已停用：终点实际激活区域GT正确率仅56%，追加学生/教师一致仍只有约65%。',
 '', '为什么调整：',
 'H1背景误检下降但新类召回明显受损，在第6000步提前停止。H2完整终点68.636483，仍低于69.277282。',
 '把教师BG解释成可信新类的备选门控，在实际缺额区域GT准确率约32%，因此未投入训练。',
 '有向新类->旧类的概率转移候选区域GT准确率约87%，按原证据权重统计约93%，才进入续训。',
 '', '训练复用与验证：',
 '本候选轨迹为OPT共享warmup第0..2000步 + 已有H2第2001..4000步 + H3第4001..8000步。',
 'H3恢复了模型、优化器和在线状态，只新增4000次正式更新。复用step0和原最优KD教师。',
 '新增监督前100步观察、随后200步渐入；标量权重0.1，未做种子或参数网格搜索。',
 '8卡梯度与全局计算一致、教师冻结、监督方向、背景门控及更强学生免干预检查通过；真实训练短跑通过。',
 '终点全部模型/优化器/在线状态与实际评估的8000步检查点逐项完全一致。',
 '', '累计训练记录（含继承的2000次H2更新，transfer字段仅统计H3阶段）：',
 json.dumps(a['all_training_updates'],ensure_ascii=False),
 '', '终点生产门控的GT诊断：',json.dumps(g['production_transfer_GT'],ensure_ascii=False),
 '', '限制：GT仅用于诊断固定门控，不进入训练。GT门控检查为固定128张原生输出网格的开发子集。',
 '这是单种子自适应开发的分阶段方案，恢复时数据顺序重启，不用于单因素归因或统计显著性声明。',
 '全部新训练仅使用8卡机；4卡机未动。代码、日志与检查点在授权工作区内。',
 '', '候选终点检查点：'+a['final_checkpoint'],'SHA256：'+a['final_checkpoint_sha256']]
with safe_path(out/'kd_transfer_result.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
files={'analysis.json':U/'analysis.json','endpoint_diagnostic.json':U/'endpoint_diagnostic.json',
       'preflight.json':E/'preflight.json','origin.json':E/'origin.json',
       'new_transfer_readiness.json':R/'runs/hierarchy_kd_v2/new_transfer_readiness.json',
       'old_agreement_readiness.json':R/'runs/hierarchy_kd_v2/old_agreement_readiness.json',
       'H2_analysis.json':R/'runs/hierarchy_kd_v2/analysis.json'}
for name,p in files.items():
    with safe_path(out/name).open('xb') as f:f.write(safe_path(p).read_bytes())
archive=safe_path(out/'implementation.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted((E/'src').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:z.write(safe_path(p),'H3/src/'+p.relative_to(E/'src').as_posix())
    for p in E.glob('*.py'):z.write(safe_path(p),'H3/scripts/'+p.name)
    run=U/'formal/h3'
    for p in run.rglob('*.json'):
        if 'checkpoints' not in p.parts:z.write(safe_path(p),'H3/run/'+p.relative_to(run).as_posix())
    for p in (run/'10-5/step2').glob('*.jsonl'):z.write(safe_path(p),'H3/run/10-5/step2/'+p.name)
    z.write(safe_path(R/'runs/hierarchy_kd_v2/export/implementation.zip'),'history/H1_H2_implementation.zip')
    z.write(safe_path(R/'experiments/prototype_sep_v1/run_new_anchor_sep.py'),'support/run_new_anchor_sep.py')
    z.write(safe_path(R/'experiments/kd_pixel_v2/run_kd.py'),'support/run_kd.py')
    for p in out.iterdir():
        if p.is_file() and p!=archive:z.write(safe_path(p),'reports/'+p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=z.namelist();assert len(names)==len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
atomic_json(out/'manifest.json',{'created_utc':now(),'archive_entries':len(names),
    'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in out.iterdir() if p.is_file()}})
print(json.dumps({'export':str(out),'all_miou':m['candidate']['all_miou'],'delta':d['all_miou'],'entries':len(names)}))
