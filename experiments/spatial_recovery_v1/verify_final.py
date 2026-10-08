from explore import *
from predict import configure,load_model,predict_tensor,SHA256
from refine import boundary_margin,test
from cached_eval import true_regions

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args();configure(a.rank)
 m=load_model();original_route=m.decoder.local_sep.routing.clone()
 parent=R/'runs/ald_calibration_v9/formal/10-5/step2/checkpoints/model_final.pth'
 assert digest(parent)=='8330ec6a7ae80aaa224c0c63f282a3e12462a38ac25e6ff448a3416d788822c8'
 state=torch.load(parent,map_location='cpu',weights_only=True,mmap=True)['model_state'];current=m.state_dict()
 assert all(torch.equal(t,current[k].cpu()) for k,t in state.items())
 assert all(k.startswith('decoder.local_sep.') for k in set(current)-set(state))
 # Switching off routing exactly removes the sole adapter; all base weights
 # have been checked against the ALDv9 checkpoint, not approximately matched.
 cfg=json.loads((R/'runs/local_sep_v5/formal/study.json').read_text())['config']
 ds=voc.VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2);ds.label_dir=cfg['val_label_dir']
 loader=DataLoader(Subset(ds,list(range(a.rank,len(ds),8))),batch_size=1,shuffle=False,num_workers=2)
 modes=['v5_engineering','v5_final','aldv9_engineering','aldv9_final']
 hist={k:np.zeros((21,21),np.int64) for k in modes};per_image={};names=[];times={k:0. for k in ['v5_forward','v5_refine','aldv9_forward','aldv9_refine']};start=time.time()
 with torch.inference_mode():
  for nb,image,label,_ in loader:
   name=str(nb[0]);image=image.cuda();outputs={}
   for model_key,route in [('v5',original_route),('aldv9',torch.zeros_like(original_route))]:
    m.decoder.local_sep.routing.copy_(route)
    torch.cuda.synchronize();t=time.time();z=predict_tensor(m,image,refine=False);torch.cuda.synchronize();times[model_key+'_forward']+=time.time()-t
    t=time.time();r=boundary_margin(image,z);torch.cuda.synchronize();times[model_key+'_refine']+=time.time()-t
    # These invariants are checked on every actual validation image.
    assert torch.equal(r[:,1:],z[:,1:])
    margin=z[:,1:].max(1,keepdim=True).values-z[:,:1];anchor=margin.abs()>=math.log(2)
    assert torch.equal(r[:,:1][anchor],z[:,:1][anchor])
    outputs[model_key+'_engineering']=z.argmax(1)[0].cpu().numpy();outputs[model_key+'_final']=r.argmax(1)[0].cpu().numpy()
   gt=label[0].numpy();per_image[name]={}
   for mode,pred in outputs.items():
    h=histogram(gt,pred);hist[mode]+=h;per_image[name][mode]=h.tolist()
   names.append(name)
   if len(names)%25==0:print(a.rank,len(names),round(time.time()-start,1),flush=True)
 atomic_json(U/f'verification/rank{a.rank}.json',dict(images=names,histograms={k:h.tolist() for k,h in hist.items()},per_image=per_image,seconds=times,utc=now(),source_sha256={str(p.name):digest(p) for p in [E/'predict.py',E/'refine.py',Path(__file__)]},base_tensors_exact_ALDv9=True,foreground_logits_exact=True,anchor_logits_exact=True,checkpoint_sha256=SHA256))
 print('DONE',a.rank,flush=True)

if __name__=='__main__':main()
