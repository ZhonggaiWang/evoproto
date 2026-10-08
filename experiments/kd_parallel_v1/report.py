from pathlib import Path
import json,sys,os,re,statistics,datetime
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/kd_parallel_v1';U=R/'runs/kd_parallel_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2/src'));from kd_runtime import atomic_json,now,digest

def historical(arm,step,iteration):
 p=R/f'runs/fixed_baseline_v1/{arm}/10-5/step{step}/metrics.jsonl'
 row=next(x for x in map(json.loads,p.read_text().splitlines()) if x['iteration']==iteration)
 values=list(row['class_iou'].values());old=10 if step==1 else 15
 return {'all_miou':sum(values)/len(values),'previous_foreground_miou':sum(values[1:old+1])/old,'current_foreground_miou':sum(values[old+1:])/5}

def main():
 os.chdir(R);results={};speed=[]
 for arm in ['a_sigmoid','b_relational']:
  run=U/'formal'/arm
  if not (run/'study.json').exists():continue
  manifest=json.loads((run/'study.json').read_text());processes={}
  for p in run.glob('coordinator_*.process.json'):
   d=json.loads(p.read_text());proc=Path(f'/proc/{d["pid"]}/stat')
   processes[p.stem]={'pid':d['pid'],'live':proc.exists() and proc.read_text().split()[21]==str(d['starttime'])}
  stages=[]
  for p in sorted(run.glob('10-5/step*/train.log')):
   rows=[]
   for line in p.read_text().splitlines():
    match=re.search(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*?Iter: (\d+);',line)
    if match:rows.append((datetime.datetime.strptime(match[1],'%Y-%m-%d %H:%M:%S,%f'),int(match[2])))
   times=[(b[0]-a[0]).total_seconds()/(b[1]-a[1]) for a,b in zip(rows[-7:-1],rows[-6:]) if b[1]>a[1]] if len(rows)>=7 else [(b[0]-a[0]).total_seconds()/(b[1]-a[1]) for a,b in zip(rows,rows[1:])]
   sec=statistics.median(times) if times else None
   stages.append({'stage':p.parent.name,'iteration':rows[-1][1] if rows else None,'median_seconds_per_step':sec,'remaining_stage_seconds':(8000-rows[-1][1])*sec if sec else None})
  if stages and stages[-1]['median_seconds_per_step']:speed.append(8/stages[-1]['median_seconds_per_step'])
  comparisons=[]
  for p in sorted(run.glob('evaluations/step*/result.json')):
   d=json.loads(p.read_text());refs={ref:historical(ref,d['step'],d['iteration']) for ref in ['base','kd']}
   metrics={k:d[k] for k in next(iter(refs.values()))}
   comparisons.append({'step':d['step'],'iteration':d['iteration'],'phase':'pre-KD warmup' if d['iteration']<=2000 else 'stage endpoint' if d['iteration']==8000 else 'intermediate','candidate':metrics,'historical':refs,
      'delta_pp':{ref:{k:metrics[k]-v for k,v in r.items()} for ref,r in refs.items()},
      'checkpoint_sha256':d['checkpoint_sha256'],'dog_iou':d['class_iou'].get('dog'),
      'dog_to_cat_percent':100*d['histogram'][12][8]/sum(d['histogram'][12]),
      'new_to_old_percent':100*d['new_gt_to_old_pixels']/d['new_gt_pixels'],
      'old_to_background_percent':100*d['old_gt_to_background_pixels']/d['old_gt_pixels']})
  mismatches=[name for name,expected in manifest['source_sha256'].items() if digest(E/arm/'src'/name)!=expected]
  results[arm]={'processes':processes,'stages':stages,'source_mismatches':mismatches,'comparisons':comparisons,'training_complete':(run/'training_complete.json').exists()}
 total=sum(speed) if len(speed)==2 else None
 result={'utc':now(),'arms':results,'parallel_images_per_second':total,
  'original_8gpu_single_job_images_per_second':46.79706041147411,
  'throughput_gain_percent':100*(total/46.79706041147411-1) if total else None,
  'caveat':'Rolling training-log throughput, not endpoint accuracy; early intervals and asynchronous evaluation may affect timing',
  'goal_scope':'Efficient meaningful KD optimization; 8card only; archived references and shared step0/common warmup reused'}
 atomic_json(U/'report.json',result)
 compact={'utc':result['utc'],'parallel_images_per_second':total,'throughput_gain_percent':result['throughput_gain_percent'],'arms':{a:{'processes':d['processes'],'stages':d['stages'],'source_mismatches':d['source_mismatches'],'latest_comparison':d['comparisons'][-1] if d['comparisons'] else None,'training_complete':d['training_complete']} for a,d in results.items()}}
 print(json.dumps(compact))

if __name__=='__main__':main()
