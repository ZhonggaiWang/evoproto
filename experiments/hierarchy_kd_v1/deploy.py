"""Unpack a new frozen candidate only after checking every destination."""
from pathlib import Path
import sys,json,zipfile
R=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto');E=R/'experiments/hierarchy_kd_v1'
P=R/'experiments/prototype_sep_v1/c_new_anchor/src'
sys.path.insert(0,str(P))
from kd_runtime import safe_path,digest,atomic_json
require=lambda c,m: None if c else (_ for _ in ()).throw(RuntimeError(m))
S=safe_path(E/'src');require(not S.exists(),'Refuse to overwrite frozen candidate')
with zipfile.ZipFile(safe_path(E/'source.zip')) as z:
    for member in z.infolist():
        require(not member.is_dir() and not Path(member.filename).is_absolute(),'Invalid entry')
        out=safe_path(S/member.filename);require(out.resolve().is_relative_to(S),'Zip escape')
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb') as f:f.write(z.read(member))
source={str(p.relative_to(S)):digest(safe_path(p)) for p in sorted(S.rglob('*.py'))}
parent={str(p.relative_to(P)):digest(safe_path(p)) for p in sorted(P.rglob('*.py'))}
changed=sorted(k for k in set(source)|set(parent) if source.get(k)!=parent.get(k))
require(changed==['continual/Trainer.py','model/hierarchical_kd.py','scripts/dist_train_voc_seg_neg.py'],str(changed))
atomic_json(E/'origin.json',{'source_sha256':source,'parent_source_sha256':parent,'changed_files':changed})
print(json.dumps({'deployed':str(S),'changed_files':changed}))
