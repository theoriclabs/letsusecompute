"""Regenerate article figures from the retrieved full-run JSON (matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
results = json.loads((ROOT/'assets/results/full/results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
names = ['base','triage','extract','linear-balanced','linear-skewed','ties']
labels = ['Base','Triage parent','Extraction parent','Balanced linear','Skewed linear','TIES']
colors = {'triage':'#b84d1b','extract':'#176b8b'}
fig, ax = plt.subplots(figsize=(8.2,4.5),layout='constrained')
for j, skill in enumerate(('triage','extract')):
    y = [i+(j-.5)*.33 for i in range(len(names))]
    counts = [results['models'][name]['test']['scores'][skill]['correct'] for name in names]
    total = results['models']['base']['test']['scores'][skill]['total']
    ax.barh(y,counts,height=.29,color=colors[skill],label='Support queue' if skill=='triage' else 'Order extraction')
    for pos,count in zip(y,counts):
        ax.text(count+.65,pos,f'{count}/{total}',va='center',fontsize=9)
ax.set(yticks=range(len(names)),yticklabels=labels,xlim=(0,total+8),xticks=range(0,total+1,12),xlabel='Correct answers on the fixed test set')
ax.invert_yaxis();ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False)
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
fig.savefig(ROOT/'assets/skill-retention.svg',facecolor='white');plt.close(fig)
fig, axes = plt.subplots(1,2,figsize=(8.2,3.4),layout='constrained')
for ax,skill in zip(axes,('triage','extract')):
    losses=results['training'][skill]['losses']
    ax.plot(range(1,len(losses)+1),losses,color=colors[skill],alpha=.25,linewidth=1)
    smooth=[sum(losses[max(0,i-9):i+1])/len(losses[max(0,i-9):i+1]) for i in range(len(losses))]
    ax.plot(range(1,len(losses)+1),smooth,color=colors[skill],linewidth=1.8)
    ax.set(title='Support queue' if skill=='triage' else 'Order extraction',xlabel='Optimizer step',ylabel='Response-token training loss')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.16)
fig.savefig(ROOT/'assets/training-loss.svg',facecolor='white');plt.close(fig)
print('Wrote skill-retention.svg and training-loss.svg')

# Keep generated SVGs clean in Git diffs.
for path in (ROOT/'assets').glob('*.svg'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
