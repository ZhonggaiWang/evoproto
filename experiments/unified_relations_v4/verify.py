from common import *
import argparse,json
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from datasets.voc import VOC12SegDataset
from data import config
from predict import configure,load_student,predict_tensor
from evaluate import hist

p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args();configure(a.rank);model=load_student();cfg=config()
ds=VOC12SegDataset(root_dir=cfg['data_folder'],name_list_dir=cfg['list_folder'],split='val',stage='val',aug=False,ignore_index=255,num_classes=21,tasks='10-5',step=2);ds.label_dir=cfg['val_label_dir']
loader=DataLoader(Subset(ds,list(range(a.rank,len(ds),8))),batch_size=1,shuffle=False,num_workers=2)
h=np.zeros((21,21),np.int64);names=[]
for nb,x,y,_ in loader:
 pred=predict_tensor(model,x.cuda()).argmax(1)[0].cpu().numpy();h+=hist(y[0].numpy(),pred);names.append(str(nb[0]))
atomic_json(U/f'verification/rank{a.rank}.json',dict(images=names,histogram=h.tolist(),source_sha256=digest(E/'predict.py'),utc=now()))
print('DONE',a.rank)
