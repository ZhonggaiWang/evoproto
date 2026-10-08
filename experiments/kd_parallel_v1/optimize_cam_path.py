from pathlib import Path
import sys,shutil
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
sys.path.insert(0,str(R/'experiments/kd_pixel_v2/src'));from kd_runtime import safe_path
B=safe_path(R/'experiments/kd_throughput_opt');assert not B.exists()
shutil.copytree(R/'experiments/kd_throughput/src',B/'src',ignore=shutil.ignore_patterns('__pycache__'))
for S in [B/'src',*(p/'src' for p in (R/'experiments/kd_parallel_v1').glob('*') if (p/'src').exists())]:
 p=S/'model/model_seg_neg.py';s=p.read_text();line='        seg, type_seg, prototypes = self.decoder(_x4, cal_sim=cal_sim)\n'
 assert s.count(line)==1;s=s.replace(line,'',1)
 marker='            return cam_aux, cam\n';assert s.count(marker)==1
 s=s.replace(marker,marker+'\n'+line,1);p.write_text(s)
print('Removed unused segmentation decoder only in CAM-only forwards')
