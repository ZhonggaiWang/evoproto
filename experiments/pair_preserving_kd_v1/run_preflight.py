from pathlib import Path
import sys,os,subprocess
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/pair_preserving_kd_v1'
sys.path.insert(0,str(R/'experiments/kd_pixel_v2'))
from run_kd import environment
env=environment('8card')
subprocess.run([str(R/'.runtime/env/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(E/'test_pair.py')],cwd=R,env=env,check=True)
