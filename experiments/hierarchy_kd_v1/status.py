from pathlib import Path
import json,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/hierarchy_kd_v1')
for kind in ['smoke','formal']:
 p=R/kind/'h1'
 if not p.exists():continue
 out={'kind':kind}
 for filename in ['status.json','failure.json']:
  if (p/filename).is_file():out[filename]=json.loads((p/filename).read_text())
 stage=p/'10-5/step2'
 for filename in ['hierarchy_metrics.jsonl','kd_metrics.jsonl']:
  q=stage/filename
  if q.exists():
   lines=q.read_text().splitlines();out[filename]={'records':len(lines),'latest':json.loads(lines[-1]) if lines else None}
 out['evaluations']=[{k:v for k,v in json.loads(f.read_text()).items() if k in ['iteration','all_miou','previous_foreground_miou','current_foreground_miou','images']} for f in sorted((p/'evaluations').glob('*/result.json'))]
 if out.get('status.json',{}).get('status')=='failed' or 'status.json' not in out:
  for f in [p/'coordinator.log',stage/'launcher.log']:
   if f.exists():out[str(f.name)]=f.read_text()[-4500:]
 print(json.dumps(out))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader'],text=True))
