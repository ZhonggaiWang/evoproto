from pathlib import Path
import json,sys,re,statistics,datetime
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/confusion_guided_v1';U=R/'runs/confusion_guided_v1'
sys.path.insert(0,str(E/'a_sep/src'));from kd_runtime import atomic_json,now,digest
def historic(arm,step,it):
    p=R/f'runs/fixed_baseline_v1/{arm}/10-5/step{step}/metrics.jsonl'
    row=next(x for x in map(json.loads,p.read_text().splitlines()) if x['iteration']==it)
    values=list(row['class_iou'].values());old=10 if step==1 else 15
    return {'all_miou':sum(values)/len(values),'previous_foreground_miou':sum(values[1:old+1])/old,'current_foreground_miou':sum(values[old+1:])/5}
def last_json(path):
    if not path.exists():return None
    lines=path.read_text().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    return None
def main():
    arms={};speeds=[]
    for arm in ['a_sep','b_pairkd','c_newaware_sep','d_pairkd_only']:
        run=U/'formal'/arm
        if not (run/'study.json').exists():continue
        manifest=json.loads((run/'study.json').read_text());stages=[];processes={}
        for p in run.glob('coordinator_*.process.json'):
            d=json.loads(p.read_text());proc=Path(f'/proc/{d["pid"]}/stat')
            stat=proc.read_text().split() if proc.exists() else None
            processes[p.stem]={'pid':d['pid'],'live':bool(stat and stat[21]==str(d['starttime']) and stat[2]!='Z')}
        for path in sorted(run.glob('10-5/step*/train.log')):
            rows=[]
            for line in path.read_text().splitlines():
                m=re.search(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*?Iter: (\d+);',line)
                if m:rows.append((datetime.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S,%f'),int(m[2])))
            intervals=[(b[0]-a[0]).total_seconds()/(b[1]-a[1]) for a,b in zip(rows[-7:-1],rows[-6:]) if b[1]>a[1]] if len(rows)>=7 else []
            sec=statistics.median(intervals) if intervals else None
            pair=last_json(path.parent/'pair_metrics.jsonl');kd=last_json(path.parent/'kd_metrics.jsonl')
            stages.append({'stage':path.parent.name,'iteration':rows[-1][1] if rows else None,'median_seconds_per_step':sec,
                'remaining_seconds_estimate':(8000-rows[-1][1])*sec if sec else None,'pair':pair,'kd':kd})
        training_live=processes.get('coordinator_train.process',{}).get('live',False)
        if training_live and stages and stages[-1]['median_seconds_per_step']:speeds.append(8/stages[-1]['median_seconds_per_step'])
        comparisons=[]
        for p in sorted(run.glob('evaluations/step*/result.json')):
            d=json.loads(p.read_text());refs={a:historic(a,d['step'],d['iteration']) for a in ['base','kd','sep','kd_sep']}
            old=R/f'runs/kd_parallel_v1/formal/b_relational/evaluations/step{d["step"]}_iter{d["iteration"]}/result.json'
            if old.exists():
                ref=json.loads(old.read_text());refs['optimized_KD']={k:ref[k] for k in ['all_miou','previous_foreground_miou','current_foreground_miou']}
            metrics={k:d[k] for k in ['all_miou','previous_foreground_miou','current_foreground_miou']}
            comparisons.append({'step':d['step'],'iteration':d['iteration'],'candidate':metrics,'references':refs,
                'delta_pp':{a:{k:metrics[k]-v for k,v in ref.items()} for a,ref in refs.items()},'checkpoint_sha256':d['checkpoint_sha256'],
                'dog_to_cat_percent':100*d['histogram'][12][8]/sum(d['histogram'][12]) if len(d['histogram'])>12 else None,
                'sofa_to_chair_percent':100*d['histogram'][18][9]/sum(d['histogram'][18]) if len(d['histogram'])>18 else None,
                'old_to_background_percent':100*d['old_gt_to_background_pixels']/d['old_gt_pixels'],
                'new_to_old_percent':100*d['new_gt_to_old_pixels']/d['new_gt_pixels']})
        source_mismatches=[rel for rel,sha in manifest['source_sha256'].items() if digest(E/arm/'src'/rel)!=sha]
        eval_receipt=run/'evaluation_workers_complete.json'
        expected_codes=[0]* (4 if arm in ['c_newaware_sep','d_pairkd_only'] else 2)
        successful_eval=eval_receipt.exists() and json.loads(eval_receipt.read_text())['returncodes']==expected_codes and all((run/'evaluations'/p.stem/'result.json').exists() for p in (run/'eval_queue').glob('*.json'))
        arms[arm]={'processes':processes,'stages':stages,'source_mismatches':source_mismatches,'comparisons':comparisons,
            'training_complete':(run/'training_complete.json').exists(),'evaluation_complete':successful_eval,'training_failed':(run/'training_failed.json').exists()}
    out={'utc':now(),'arms':arms,'parallel_training_images_per_second':sum(speeds) if len(speeds)==2 else None,
         'scope':'Current traininglog rates and completed evaluations; do not substitute intermediate results for endpoints'}
    atomic_json(U/'report.json',out)
    compact={a:{'processes':v['processes'],'stages':[{'stage':s['stage'],'iteration':s['iteration'],'seconds_per_step':s['median_seconds_per_step'],
        'pair_ramp':s['pair'].get('ramp') if s['pair'] else None,'selected_targets':s['pair'].get('selector',{}).get('targets') if s['pair'] else None,
        'SEP_pixels':s['pair'].get('sep_nonzero_pixels') if s['pair'] else None,'pair_KD_pixels':s['kd'].get('kd_pair_pixels') if s['kd'] else None} for s in v['stages']],
        'source_mismatches':v['source_mismatches'],'latest_evaluation':v['comparisons'][-1] if v['comparisons'] else None,'training_complete':v['training_complete'],'evaluation_complete':v['evaluation_complete']} for a,v in arms.items()}
    print(json.dumps({'utc':out['utc'],'parallel_training_images_per_second':out['parallel_training_images_per_second'],'arms':compact}))
if __name__=='__main__':main()
