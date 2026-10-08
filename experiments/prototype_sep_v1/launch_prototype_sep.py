from pathlib import Path
import os,sys,json,subprocess,argparse
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/prototype_sep_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
sys.path.insert(0,str(E/'a_geometry/src'))
from kd_runtime import safe_path,atomic_json,now
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
run=safe_path(R/'runs/prototype_sep_v1'/('smoke' if args.smoke else 'formal')/'a_geometry')
for path in (E/'a_geometry/src').rglob('*'):safe_path(path)
assert not (run/'coordinator.process.json').exists(),'Refuse duplicate launch'
if not args.smoke:
    smoke=json.loads((R/'runs/prototype_sep_v1/smoke/a_geometry/status.json').read_text())
    assert smoke['status']=='complete'
    assert json.loads((R/'runs/prototype_sep_v1/gradient_probe.json').read_text())['passed']
    assert json.loads((R/'runs/prototype_sep_v1/preflight.json').read_text())['passed']
    assert (E/'prototype_geometry.json').is_file()
run.mkdir(parents=True,exist_ok=True)
cmd=[str(R/'.runtime/env/bin/python'),'-B',str(E/'run_prototype_sep.py')]
if args.smoke:cmd.append('--smoke')
with open(safe_path(run/'coordinator.log'),'x') as log:
    child=subprocess.Popen(cmd,cwd=R,env=environment('8card'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
atomic_json(run/'coordinator.process.json',{'pid':child.pid,'starttime':Path(f'/proc/{child.pid}/stat').read_text().split()[21],
    'command':cmd,'utc':now()})
print(json.dumps({'pid':child.pid,'run':str(run),'smoke':args.smoke}))
