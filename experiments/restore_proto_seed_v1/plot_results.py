"""Plot saved post-warmup validation curves; no model or dataset required."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def main():
    result=json.loads((HERE/'results.json').read_text())
    diagnostic=json.loads((HERE/'final_diagnostics.json').read_text())
    styles={'baseline':('Original EvoProto','#444444'),'correct':('Seed + teacher gate','#d94e45'),'ignore':('Ignore + teacher gate','#9575cd'),'no_graph':('Seed, no extra teacher gate','#008c91')}
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.6))
    for stage,ax in enumerate(axes,1):
        for arm,(label,color) in styles.items():
            rows=diagnostic['baseline_learning_curves'][str(stage)] if arm=='baseline' else result['arms'][arm]['stages'][str(stage)]['learning_curve']
            rows=[x for x in rows if x['iteration']>=4000]
            assert [x['iteration'] for x in rows]==[4000,6000,8000]
            ax.plot([x['iteration'] for x in rows],[x['all_miou'] for x in rows],marker='o',linewidth=2,color=color,label=label)
        ax.set(title=f'Step {stage}: square448 main head',xlabel='Training updates',ylabel='mIoU (%)')
        ax.set_xticks([4000,6000,8000]);ax.grid(alpha=.22);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.015))
    fig.suptitle('Post-warmup validation: compare final 8000-update checkpoints',fontsize=13)
    fig.text(.5,.17,'All variants retain original learned prototypes, student-confusion SEP and KD.',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=[0,.2,1,.95]);fig.savefig(HERE/'training_curves.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
