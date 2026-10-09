from pathlib import Path
import json,os,time,signal,torch
root=Path('/data/zhonggai/home/python-work-space/WILSS/evoproto').resolve()
control=root/'runs/restore_proto_v1/control'
assert torch.cuda.device_count()==2
running=True
held=[];mode=None;fulfilled=False

def stop(*args):
    global running
    running=False
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
base=[torch.empty(64*1024**2,dtype=torch.uint8,device=f'cuda:{i}') for i in range(2)]
while running:
    requested=json.loads((control/'reservation_request.json').read_text())['mode']
    if requested=='stop':break
    if requested!=mode:
        held.clear()
        for i in range(2):
            with torch.cuda.device(i):torch.cuda.empty_cache()
        target=(20 if requested=='reserve' else 8)*1024**3
        fulfilled=True
        for i in range(2):
            with torch.cuda.device(i):
                free,_=torch.cuda.mem_get_info()
                amount=min(target,max(0,free-1024**3))
                if amount<target:fulfilled=False
                try:held.append(torch.empty(amount,dtype=torch.uint8,device=f'cuda:{i}'))
                except torch.cuda.OutOfMemoryError:fulfilled=False
        mode=requested
        print('mode',mode,'target',target,'fulfilled',fulfilled,flush=True)
    state={'pid':os.getpid(),'mode':mode,'physical_gpus':[5,6],'time':time.time(),'allocated':[torch.cuda.memory_allocated(i) for i in range(2)],'target_fulfilled':fulfilled,'version':2}
    name='reservation_state' if (control/'reservation_v2_adopt').exists() else 'reservation_v2_pending'
    tmp=control/(name+'.tmp');tmp.write_text(json.dumps(state));tmp.replace(control/(name+'.json'))
    time.sleep(2)
print('released',flush=True)
