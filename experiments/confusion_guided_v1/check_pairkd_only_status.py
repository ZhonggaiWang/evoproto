from pathlib import Path
import json,re,datetime,statistics,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
u=R/'runs/confusion_guided_v1/formal/d_pairkd_only'
out={}
p=u/'10-5/step2/train.log'
if p.exists():
 rows=[]
 for line in p.read_text().splitlines():
  m=re.search(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) .*?Iter: (\d+);',line)
  if m:rows.append((datetime.datetime.strptime(m[1],'%Y-%m-%d %H:%M:%S,%f'),int(m[2])))
 sec=statistics.median([(b[0]-a[0]).total_seconds()/(b[1]-a[1]) for a,b in zip(rows[-7:-1],rows[-6:])]) if len(rows)>6 else None
 out.update(iteration=rows[-1][1] if rows else 2000,seconds_per_step=sec,remaining_min=(8000-rows[-1][1])*sec/60 if sec else None)
for name in ['training_failed.json','training_complete.json','evaluation_workers_complete.json']:
 p=u/name
 if p.exists():out[name]=json.loads(p.read_text())
out['evals']=[{k:d[k] for k in ['iteration','all_miou','previous_foreground_miou','current_foreground_miou']} for d in [json.loads(p.read_text()) for p in sorted(u.glob('evaluations/*/result.json'))]]
p=u/'10-5/step2/pair_metrics.jsonl'
if p.exists():
 d=json.loads(p.read_text().splitlines()[-1]);out['selector']={'iteration':d.get('iteration'),'ramp':d.get('ramp'),'targets':d.get('selector',{}).get('targets')}
p=u/'10-5/step2/kd_metrics.jsonl'
if p.exists():
 d=json.loads(p.read_text().splitlines()[-1]);out['KD']={k:d.get(k) for k in ['iteration','kd_pair_pixels','kd_pair_classes','kd_pair_blend','kd_value','eligible_pixels']}
for p in u.glob('coordinator_*.process.json'):
 d=json.loads(p.read_text());stat=Path(f'/proc/{d["pid"]}/stat');parts=stat.read_text().split() if stat.exists() else []
 out[p.stem]={'pid':d['pid'],'live':bool(parts and parts[21]==str(d['starttime']) and parts[2]!='Z')}
print(json.dumps(out,allow_nan=False))
