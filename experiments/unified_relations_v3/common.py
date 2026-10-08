from pathlib import Path
import sys,os
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E=R/'experiments/unified_relations_v3';U=R/'runs/unified_relations_v3';S=E/'src'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
os.environ.update(environment('8card'))
from kd_runtime import safe_path,atomic_json,digest,now
sys.path.insert(0,str(S))
PARENT=R/'runs/ald_calibration_v9/formal/10-5/step2/checkpoints/model_final.pth'
PARENT_SHA='8330ec6a7ae80aaa224c0c63f282a3e12462a38ac25e6ff448a3416d788822c8'
