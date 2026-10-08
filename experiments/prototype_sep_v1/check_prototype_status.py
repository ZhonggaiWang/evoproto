from pathlib import Path
import sys,json,re,subprocess,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');U=R/'runs/prototype_sep_v1/formal/a_geometry'
def read(path):return json.loads(path.read_text()) if path.exists() else None
record=read(U/'coordinator.process.json');live=False
if record:
    stat=Path(f'/proc/{record["pid"]}/stat')
    if stat.exists():
        row=stat.read_text().split();live=row[21]==str(record['starttime']) and row[2]!='Z'
result={'status':read(U/'status.json'),'coordinator_live':live,'stages':{},'evaluations':{}}
for step in [1,2]:
    stage=U/f'10-5/step{step}'
    if not stage.exists():continue
    log=stage/'train.log'
    lines=log.read_text(errors='replace').splitlines() if log.exists() else []
    iterations=[line for line in lines if 'Iter: ' in line]
    entry={'latest_training_log':iterations[-1] if iterations else None,'launch':read(stage/'launch.json')}
    entry.pop('launch') if entry['launch'] is None else None
    if entry.get('launch'):entry['launch']={k:v for k,v in entry['launch'].items() if k in ['pid','start_utc','end_utc','returncode']}
    geometry=stage/'geometry_metrics.jsonl'
    if geometry.exists():
        metrics=json.loads(geometry.read_text().splitlines()[-1]);entry['geometry']={k:metrics.get(k) for k in
            ['iteration','active','ramp','prototype_sep','pair_count','active_pair_count','old_new_pair_count','new_new_pair_count']}
    launcher=stage/'launcher.log'
    if not live and launcher.exists():entry['launcher_tail']=launcher.read_text(errors='replace').splitlines()[-8:]
    result['stages'][str(step)]=entry
for p in sorted((U/'evaluations').glob('*/result.json')):
    obj=read(p);result['evaluations'][p.parent.name]={k:obj[k] for k in ['images','all_miou','previous_foreground_miou','current_foreground_miou']}
    reference=R/'runs/kd_parallel_v1/formal/b_relational/evaluations'/p.parent.name/'result.json'
    if reference.exists():
        ref=read(reference)
        result['evaluations'][p.parent.name]['delta_to_archived_optimizedKD_same_iteration']={
            key:obj[key]-ref[key] for key in ['all_miou','previous_foreground_miou','current_foreground_miou']}
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
failure=read(U/'failure.json')
if failure:result['failure']=failure
print(json.dumps(result,ensure_ascii=False))
