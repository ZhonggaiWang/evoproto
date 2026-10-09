from pathlib import Path
import json,os,time,signal,torch
root=Path(__file__).resolve().parents[2]
control=root/'runs/restore_proto_v1/control'
assert root.resolve()==Path('/data/zhonggai/home/python-work-space/WILSS/evoproto')
assert torch.cuda.device_count()==2
for i in range(2):
    with torch.cuda.device(i):
        free,total=torch.cuda.mem_get_info()
        assert free>22*1024**3,(i,free)
base=[torch.empty(64*1024**2,dtype=torch.uint8,device=f'cuda:{i}') for i in range(2)]
held=[];mode=None;running=True
def stop(*args):
    global running
    running=False
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
while running:
    try: requested=json.loads((control/'reservation_request.json').read_text())['mode']
    except FileNotFoundError: requested='reserve'
    if requested=='stop':break
    if requested!=mode:
        held.clear()
        for i in range(2):
            with torch.cuda.device(i):torch.cuda.empty_cache()
        if requested=='reserve':
            held=[torch.empty(20*1024**3,dtype=torch.uint8,device=f'cuda:{i}') for i in range(2)]
        mode=requested
        print('reservation mode',mode,flush=True)
    state={'pid':os.getpid(),'mode':mode,'physical_gpus':[5,6],'time':time.time(),'allocated':[torch.cuda.memory_allocated(i) for i in range(2)]}
    tmp=control/'reservation_state.tmp';tmp.write_text(json.dumps(state));tmp.replace(control/'reservation_state.json')
    time.sleep(2)
print('released',flush=True)
