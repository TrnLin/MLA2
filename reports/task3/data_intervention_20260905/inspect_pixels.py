"""Scientific pixel diagnostics and contact sheets; no training data are created."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageEnhance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fashion.data.task3_clean_slate_eda import foreground_proposal
from fashion.train.task3_clean_slate import fixed_feature_vector
O=Path(__file__).resolve().parent;d=pd.read_csv(O/'development_audit_rows.csv.gz',keep_default_na=False)
# Fixed inspection scope is common training pool for screen folds 0 and 4.
t=d[d.cv_fold.isin([1,2,3])].copy();parts=[]
for target in ['usage','gender']:
 for label,g in t.groupby(target):
  if label=='':continue
  parts.append(g.sort_values('id').drop_duplicates('product_family_group').sample(n=min(35,g.product_family_group.nunique()),random_state=2753))
s=pd.concat(parts).drop_duplicates('id').sort_values('id');s[['id','cv_fold','product_family_group','gender','usage','path']].to_csv(O/'pixel_inspection_scope.csv',index=False)
rows=[]
for r in s.itertuples():
 im=Image.open(r.path).convert('RGB');a=np.asarray(im);p=foreground_proposal(a);m=p.mask;h,w=m.shape
 removed=[]
 for dx in [-2,-1,0,1,2]:
  for dy in [-2,-1,0,1,2]:
   keep=np.zeros_like(m);keep[max(0,-dy):min(h,h-dy),max(0,-dx):min(w,w-dx)]=True
   removed.append(float((m&~keep).sum()/m.sum()))
 crop=np.zeros_like(m);crop[8:-8,6:-6]=True
 erase=np.zeros_like(m);erase[h//2-4:h//2+4,w//2-4:w//2+4]=True
 b=np.asarray(ImageEnhance.Brightness(im).enhance(1.15));dark=np.asarray(ImageEnhance.Brightness(im).enhance(.90))
 rows.append({'id':r.id,'usage':r.usage,'gender':r.gender,'original_mask_fallback':bool(p.fallback_reason),'foreground_fraction':p.raw_fraction,'shift_grid_mean_removed_foreground':np.mean(removed),'shift_grid_worst_removed_foreground':max(removed),'crop10pct_removed_foreground':float((m&~crop).sum()/m.sum()),'center8x8_erased_foreground':float((m&erase).sum()/m.sum()),'bright115_newly_clipped_foreground_channel_fraction':float(((b==255)&(a<255))[m].mean()),'dark090_near_white_fraction':float(np.all(dark>=245,axis=2).mean()),'original_near_white_fraction':float(np.all(a>=245,axis=2).mean())})
pix=pd.DataFrame(rows);pix.to_csv(O/'augmentation_pixel_damage.csv',index=False)
valid=pix[~pix.original_mask_fallback]
summary={'scope':'folds 1,2,3 only, at most 35 families per class per target; not population-weighted','n':len(s),'usable_mask_n':len(valid),'fallback_n':int(pix.original_mask_fallback.sum())}
for c in ['shift_grid_mean_removed_foreground','shift_grid_worst_removed_foreground','crop10pct_removed_foreground','center8x8_erased_foreground','bright115_newly_clipped_foreground_channel_fraction']:
 summary[c]={'median':float(valid[c].median()),'p95':float(valid[c].quantile(.95)),'max':float(valid[c].max()),'above_005_count':int(valid[c].gt(.05).sum())}
summary['dark090_max_near_white_fraction']=float(pix.dark090_near_white_fraction.max())
(O/'pixel_damage_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
# Fixed HOG neighbours: reference images from the SAME permitted training pool.
refs=[]
for label,g in t[t.has_usage_label].groupby('usage'):
 g=g.sort_values('id').drop_duplicates('product_family_group');refs.append(g.sample(n=min(450,len(g)),random_state=2753))
r=pd.concat(refs).drop_duplicates('id').sort_values('id').reset_index(drop=True)
f=np.stack([fixed_feature_vector(Image.open(x),view='full_rgb_hog') for x in r.path]);f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
nb=[]
for i,x in r.iterrows():
 if x.usage not in ['Party','Smart Casual','Travel','NA']:continue
 eligible=r.articleType.eq(x.articleType)&r.product_family_group.ne(x.product_family_group)
 for kind,mask in [('same_label',eligible&r.usage.eq(x.usage)),('different_label',eligible&r.usage.ne(x.usage))]:
  ix=np.flatnonzero(mask.to_numpy())
  if not len(ix):continue
  sims=f[ix]@f[i];j=ix[np.argmax(sims)]
  nb.append({'query_id':x.id,'query_label':x.usage,'articleType':x.articleType,'kind':kind,'reference_id':r.iloc[j].id,'reference_label':r.iloc[j].usage,'cosine_similarity':float(sims.max()),'query_path':x.path,'reference_path':r.iloc[j].path})
pd.DataFrame(nb).to_csv(O/'rare_training_hog_neighbours.csv',index=False)
r[['id','cv_fold','product_family_group','usage','articleType','path']].to_csv(O/'hog_neighbour_reference_scope.csv',index=False)
# Contact sheet all Party images and sample rare classes; explicit catalogue labels.
sel=pd.concat([t[t.usage.eq('Party')],t[t.usage.eq('Smart Casual')].sort_values('id').drop_duplicates('product_family_group').head(8),t[t.usage.eq('Travel')].sort_values('id').head(8),t[t.usage.eq('NA')].sort_values('id').drop_duplicates('product_family_group').head(8)]).drop_duplicates('id')
fig,axes=plt.subplots(int(np.ceil(len(sel)/8)),8,figsize=(14,8),layout='constrained')
for ax in axes.flat:ax.axis('off')
for ax,(_,x) in zip(axes.flat,sel.iterrows()):
 ax.imshow(Image.open(x.path),interpolation='nearest');ax.set_title(f'{x.id} | {x.usage}\n{x.articleType}',fontsize=8);ax.axis('off')
fig.suptitle('Teacher images from training folds 1/2/3: catalogue labels, not human judgements',fontsize=12)
fig.savefig(O/'rare_usage_contact_sheet.png',dpi=150);plt.close(fig)
# Equal-size original vs transform, no rescaling of input beyond display enlargement.
chosen=[]
for label in ['Boys','Girls','Unisex']:
 g=s[s.gender.eq(label)].sort_values('foreground_fraction');chosen.append(g.iloc[len(g)//2])
for label in ['Party','Smart Casual','Travel']:
 g=s[s.usage.eq(label)].sort_values('id');chosen.append(g.iloc[0])
fig,axes=plt.subplots(6,6,figsize=(10.5,11),layout='constrained')
names=['Original','Dark x0.90','Bright x1.15','Shift (+2,+2)','Crop 10% edges','Erase centre 8x8']
for row,x in enumerate(chosen):
 im=Image.open(x.path).convert('RGB');shift=Image.new('RGB',im.size,'white');shift.paste(im,(2,2));erase=im.copy();erase.paste((255,255,255),(26,36,34,44));cropped=im.crop((6,8,54,72)).resize(im.size,Image.Resampling.BILINEAR)
 views=[im,ImageEnhance.Brightness(im).enhance(.90),ImageEnhance.Brightness(im).enhance(1.15),shift,cropped,erase]
 for col,(ax,v) in enumerate(zip(axes[row],views)):
  ax.imshow(v,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
  if row==0:ax.set_title(names[col],fontsize=9)
  if col==0:ax.set_ylabel(f'{x.id}\n{x.gender}\n{x.usage}',fontsize=8)
fig.suptitle('Pixel-only previews at 60x80; no label preservation or model gain is assumed',fontsize=12)
fig.savefig(O/'augmentation_preview.png',dpi=140);plt.close(fig)
# HOG selects shapes; it cannot decide whether an occasion label is wrong.
n=pd.DataFrame(nb);chosen=n[n.kind.eq('different_label')].sort_values('cosine_similarity',ascending=False).drop_duplicates('query_label').head(4)
fig,axes=plt.subplots(len(chosen),2,figsize=(6,9),layout='constrained')
for axrow,(_,x) in zip(axes,chosen.iterrows()):
 for ax,side in zip(axrow,['query','reference']):
  ax.imshow(Image.open(x[side+'_path']),interpolation='nearest');ax.axis('off');ax.set_title(f'{x[side+"_id"]}: {x[side+"_label"]}\n{x.articleType} | HOG cosine {x.cosine_similarity:.3f}',fontsize=10)
fig.suptitle('Similar shape, different catalogue usage\nTraining-only pairs from different saved families',fontsize=12)
fig.savefig(O/'different_usage_neighbours.png',dpi=150);plt.close(fig)
# Compact support and model-error figures.
a=pd.read_csv(O/'class_independent_support.csv',keep_default_na=False);fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for ax,target in zip(axes,['gender','usage']):
 z=a[a.target.eq(target)].sort_values('rows');y=np.arange(len(z));ax.barh(y-.18,z.rows,height=.35,label='Images');ax.barh(y+.18,z.families,height=.35,label='Saved families');ax.set_yticks(y,z['class']);ax.set_xscale('log');ax.set_xlabel('Count (log scale)');ax.set_title(target.title());ax.legend(fontsize=8)
fig.suptitle('Development support: more copies do not add families',fontsize=14);fig.savefig(O/'class_support.png',dpi=160);plt.close(fig)
u=pd.read_csv(O/'usage_error_transitions.csv');u=u[u.slice.isin(['usual','exception'])];fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(3)
for s,offset,color in [('usual',-.18,'#3579aa'),('exception',.18,'#ce7433')]:
 g=u[u.slice.eq(s)];ax.bar(x+offset,100*(g.child_error-g.parent_error),.35,label=s,color=color)
ax.set_xticks(x,['E3 dropout','E8 translation','E9 strong weights']);ax.set_ylabel('Error change vs E2 (percentage points)');ax.axhline(0,color='black',linewidth=.8);ax.legend();ax.set_title('Usage: fixes and new errors must both count');fig.savefig(O/'usage_error_tradeoffs.png',dpi=160);plt.close(fig)
