"""Plot a saved training confusion graph without using validation labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage-dir',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    stage=Path(args.stage_dir).resolve();out=Path(args.output).resolve()
    assert stage.is_relative_to(ROOT) and out.is_relative_to(ROOT)
    graph=json.loads((stage/'confusion.json').read_text())
    config=json.loads((stage/'config.json').read_text())
    assert graph['mode']=='full', 'This figure describes the measured full-method graph'
    values=np.asarray(graph['ema'])/np.maximum(np.asarray(graph['anchor_mass'])[:,None],1e-6)
    values*=np.asarray(graph['unique_image_support'])>=3
    values[0]=0;values[:,0]=0;np.fill_diagonal(values,0)
    k=len(values);names=NAMES[1:k]
    old=10 if config['step']==1 else 15
    degree=np.clip(values.sum(0)+values.sum(1),0,2)
    fig,(ax,bar)=plt.subplots(1,2,figsize=(13.2,6.8),gridspec_kw={'width_ratios':[1.75,1]},constrained_layout=True)
    fig.suptitle(f'EvoProto | Stage {config["step"]} training confusion at update {graph["updates"]:,}',fontsize=17,fontweight='bold')
    im=ax.imshow(values[1:,1:]*100,cmap='Blues',vmin=0,vmax=max(.01,float(values.max()*100)),interpolation='nearest')
    ax.set_xticks(range(k-1),names,rotation=55,ha='right',fontsize=9)
    ax.set_yticks(range(k-1),names,fontsize=9)
    ax.set_xlabel('Student-predicted rival class');ax.set_ylabel('Trusted anchor class')
    for a in range(1,k):
        for b in np.argsort(values[a])[-2:]:
            if values[a,b]>0:
                ax.add_patch(Rectangle((b-1.45,a-1.45),.9,.9,fill=False,edgecolor='#dd7320',linewidth=1.1))
    ax.axhline(old-.5,color='#263c50',lw=1);ax.axvline(old-.5,color='#263c50',lw=1)
    ax.set_title('EMA confusion rate (%) | orange = SEP rival',fontsize=11)
    fig.colorbar(im,ax=ax,shrink=.7,label='Confused / trusted anchors (%)')
    pos=np.arange(old)
    colors=['#31708c']*old
    bar.barh(pos,100*degree[1:old+1],color=colors)
    bar.set_yticks(pos,names[:old],fontsize=9);bar.invert_yaxis()
    bar.set_xlim(0,max(8,float(degree[1:old+1].max()*110)))
    bar.set_xlabel('Increase over KD class weight 1 (%)')
    bar.set_title('KD weights for old classes only',fontsize=11)
    bar.grid(axis='x',alpha=.2);bar.set_axisbelow(True)
    for spine in ('top','right'):bar.spines[spine].set_visible(False)
    fig.supxlabel('Training weak evidence only; this is not a validation confusion matrix. Lines separate old/new classes; KD excludes current new classes.',fontsize=9)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180)
    fig.savefig(out.with_suffix('.svg'))
    plt.close(fig)
    print(json.dumps({'output':str(out),'classes':k,'updates':graph['updates'],'kd_weight_min':float(1+degree[1:old+1].min()),'kd_weight_max':float(1+degree[1:old+1].max())}))


if __name__=='__main__':main()
