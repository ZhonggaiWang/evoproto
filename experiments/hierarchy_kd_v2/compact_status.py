from pathlib import Path
import json
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto/runs/hierarchy_kd_v2/formal/h2')
out={}
p=R/'status.json'
if p.exists():out['status']=json.loads(p.read_text())
p=R/'10-5/step2/hierarchy_metrics.jsonl'
if p.exists():
    records=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if records:
        out['latest']={k:v for k,v in records[-1].items() if k not in ['mass_class_counts','rejection_class_counts']}
        out['sampled_records']=len(records)
out['evaluations']=[]
for p in sorted((R/'evaluations').glob('*/result.json')):
    x=json.loads(p.read_text())
    out['evaluations'].append({k:x[k] for k in ['iteration','all_miou','previous_foreground_miou','current_foreground_miou','images']})
if out.get('status',{}).get('status')=='failed' or not out.get('status'):
    for p in [R/'coordinator.log',R/'10-5/step2/launcher.log']:
        if p.exists():out[p.name]=p.read_text()[-5000:]
print(json.dumps(out))
