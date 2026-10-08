from pathlib import Path
import sys,shutil,json,os,signal
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
old=R/'experiments/kd_pixel_v1'
sys.path.insert(0,str(old/'src'))
from kd_runtime import safe_path,atomic_json,now
dest=safe_path(R/'experiments/kd_pixel_v2')
assert not dest.exists()
for p in (old/'src').rglob('*'):
    if p.is_file(): safe_path(p)
shutil.copytree(old/'src',dest/'src',ignore=shutil.ignore_patterns('__pycache__'))
for name in ['origin.json','run_kd.py','summarize_kd.py','probe_kd_gradients.py']:
    text=(old/name).read_text().replace('kd_pixel_v1','kd_pixel_v2')
    if name=='run_kd.py':
        text=text.replace('4-card host evaluates it','8-card host also evaluates it').replace("'eval_host_alias':'4card'","'eval_host_alias':'8card'")
        text=text.replace("'new-class-aware class-balanced pixel KD'","'Evidence-gated old-output Bernoulli KD'")
    if name=='probe_kd_gradients.py':
        text=text.replace("ROOT/'runs/kd_pixel_v2/10-5/step1/config.json'", "ROOT/'runs/kd_pixel_v1/10-5/step1/config.json'")
        text=text.replace("ROOT/'runs/kd_pixel_v2/10-5/step1/checkpoints/model_iter_2000.pth'", "ROOT/'runs/kd_pixel_v1/10-5/step1/checkpoints/model_iter_2000.pth'")
        text=text.replace('assert gradients[\'encoder.patch_embed.proj.weight\'][\'kd_gradient_norm\']>0',"assert gradients['decoder.conv8.1.weight']['kd_gradient_norm']==0")
        text=text.replace("assert any(v['kd_gradient_norm']>0 for k,v in gradients.items() if k.startswith('decoder.conv8.'))",'')
    safe_path(dest/name).write_text(text)
for name in ['scripts/dist_train_voc_seg_neg.py','kd_runtime.py']:
    p=dest/'src'/name;p.write_text(p.read_text().replace('kd_pixel_v1','kd_pixel_v2').replace('pixel_kd_v1','pixel_kd_v2'))
atomic_json(dest/'protocol.json',{'created_utc':now(),'parent':'kd_pixel_v1','change':'Old-foreground Bernoulli KL; old CAM support; continuous new CAM veto; square-root support balance', 'unchanged':'shared step0, baseline training objectives, batch8, seed0, 8000x2, KDweight0.1 and temperature2', 'resource':'8card only; 4card untouched', 'motivation':'v1 old/new softmax coupling and equal tiny-class weight; step1 6000 all61.7019 and dog9.351', 'scope':'one mechanism revision, no parameter sweep, no baseline retraining'})
print(dest)
