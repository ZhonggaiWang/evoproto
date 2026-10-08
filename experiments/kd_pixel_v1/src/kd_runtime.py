from pathlib import Path
import hashlib, json, os, socket, datetime

ROOT=Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
ALLOWED=Path('/ML-vePFS/infra_rd/kun/others/wzg')

def safe_path(path):
    path=Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(ALLOWED):
        raise RuntimeError(f'Output escapes allowed workspace: {path}')
    for p in [path, *path.parents]:
        if p == ALLOWED.parent: break
        if p.is_symlink() or (p.is_file() and p.stat().st_nlink != 1):
            raise RuntimeError(f'Unsafe linked output: {p}')
        if p.exists() and p.stat().st_dev != ALLOWED.stat().st_dev:
            raise RuntimeError(f'Nested mount in output path: {p}')
    return path

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        while chunk:=stream.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path=safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=safe_path(path.with_name(path.name+f'.{os.getpid()}.tmp'))
    with open(tmp,'x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,path)

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def publish_evaluation(args, iteration):
    stage=Path(args.work_dir)
    checkpoint=stage/'checkpoints'/f'model_iter_{iteration}.pth'
    if not checkpoint.is_file(): raise RuntimeError(f'Incomplete evaluation checkpoint: {checkpoint}')
    config=vars(args).copy()
    path=stage.parents[1]/'eval_queue'/f'step{args.step}_iter{iteration}.json'
    record={'schema':1,'study':'pixel_kd_v1','checkpoint':str(checkpoint),
            'checkpoint_sha256':digest(checkpoint),'config':config,'step':args.step,
            'iteration':iteration,'published_utc':now(),'publisher_host':socket.gethostname()}
    if path.exists(): raise RuntimeError(f'Refusing duplicate evaluation: {path}')
    atomic_json(path, record)
