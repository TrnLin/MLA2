from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
root = next(candidate for candidate in p.parents if (candidate / "pyproject.toml").is_file())
folders={'G2':root/'reports/task3/g2_gate_audit_20260905/drive','G-D1':p/'gender'}
rows=[];histories=[]
for name,folder in folders.items():
 for d in folder.glob('t3_*'):
  if not (d/'metrics.json').is_file():continue
  m=json.loads((d/'metrics.json').read_text());f=m['validation_fold']
  if f not in (0,4):continue
  rb=pd.read_csv(d/'robustness.csv').set_index('corruption')
  rows.append(dict(model=name,fold=f,clean=m['macro_f1'],dark=rb.loc['brightness_085','macro_f1'],shift=rb.loc['translation_003','macro_f1']))
  if name=='G-D1':histories.append((f,pd.read_csv(d/'history.csv')))
frame=pd.DataFrame(rows)
fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
for ax,key,title in zip(axes.flat,('clean','dark','shift'),('Clean images','Darker images (brightness ×0.85)','Shifted images (saved translation test)')):
 for i,name in enumerate(('G2','G-D1')):
  vals=frame[frame.model.eq(name)].sort_values('fold')[key].to_numpy()
  bars=ax.bar(np.arange(2)+(i-.5)*.32,vals,width=.32,label=name,color=('#94a3b8','#0f766e')[i])
  ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
 ax.set_xticks(range(2),['Fold 0','Fold 4']);ax.set_ylim(0,1);ax.set_ylabel('Macro-F1');ax.set_title(title);ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(loc='upper left')
ax=axes[1,1]
for f,h in sorted(histories,key=lambda item:item[0]):
 ax.plot(h.epoch,h.train_macro_f1,label=f'Fold {f}: training (augmented)',linestyle='--')
 ax.plot(h.epoch,h.validation_macro_f1,label=f'Fold {f}: validation')
ax.set_title('G-D1 learning curves');ax.set_xlabel('Epoch');ax.set_ylabel('Macro-F1');ax.set_ylim(0,1.02);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.suptitle('G-D1 screen: darker-image gains, mixed clean and shifted results',fontsize=15)
fig.savefig(p/'screen_review.png',dpi=150)
