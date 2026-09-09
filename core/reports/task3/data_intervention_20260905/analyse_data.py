"""Read-only development audit. Writes only within this new report folder. No model fit."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wasserstein_distance
from fashion.data import load_splits
from fashion.data.task3_clean_slate_eda import build_clean_slate_audit_contract,measure_teacher_image
from fashion.train.task3_dataset_v2 import build_visual_component_mapping
ROOT=Path.cwd(); OUT=Path(__file__).resolve().parent
EDA=ROOT/'results/evidence/task3/clean_slate_eda'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
splits=load_splits(ROOT/'data/processed/splits.csv')
dev=splits.loc[splits.partition.eq('development')].copy();dev.cv_fold=dev.cv_fold.astype(int)
print('Rehashing only development teacher images',len(dev),flush=True)
fresh=build_clean_slate_audit_contract(splits,root=ROOT);save('fresh_audit_contract.json',fresh)
old=json.loads((EDA/'audit_contract.json').read_text())
manifest=pd.read_csv(EDA/'artifact_manifest.csv',keep_default_na=False)
checks=[]
for r in manifest.itertuples():
 p=EDA/r.path;checks.append({'path':r.path,'exists':p.is_file(),'hash_matches':p.is_file() and sha(p)==r.sha256})
pd.DataFrame(checks).to_csv(OUT/'old_manifest_check.csv',index=False)
diag=pd.read_csv(EDA/'teacher_image_diagnostics.csv.gz',keep_default_na=False)
contract=json.loads((EDA/'teacher_image_diagnostics.csv.gz.contract.json').read_text())
assert set(diag.id)==set(dev.id)
for c in ['cv_fold','path','product_family_group']:
 a=dev.set_index('id')[c].astype(str).sort_index();b=diag.set_index('id')[c].astype(str).sort_index();assert a.equals(b),c
# Deterministic sample verifies cached numeric measurements against current pixels.
sample=dev.sort_values('id').sample(n=160,random_state=2753)
errs=[]
for r in sample.itertuples():
 m=measure_teacher_image(r.path,root=ROOT);d=diag.set_index('id').loc[r.id]
 errors=[abs(float(v)-float(d[k])) for k,v in m.items() if isinstance(v,(float,int))]
 errs.append({'id':int(r.id),'max_abs_difference':max(errors),'fallback_matches':m['foreground_fallback_reason']==d.foreground_fallback_reason})
pd.DataFrame(errs).to_csv(OUT/'diagnostic_recheck.csv',index=False)
save('provenance_verdict.json',{'rows':len(dev),'fresh_vs_saved_contract_changes':{k:{'saved':old.get(k),'fresh':v} for k,v in fresh.items() if v!=old.get(k)},'old_manifest_pass_count':sum(r['hash_matches'] for r in checks),'old_manifest_count':len(checks),'diagnostic_hash_pass':sha(EDA/'teacher_image_diagnostics.csv.gz')==contract['artifact_sha256']['teacher_image_diagnostics.csv.gz'],'diagnostic_metadata_matches':True,'remeasured_rows':len(errs),'remeasure_max_error':max(r['max_abs_difference'] for r in errs),'completion_hash':json.loads((EDA/'completion.json').read_text())['audit_contract_hash']})
visual,vc=build_visual_component_mapping(splits,candidates_path=ROOT/'data/processed/audit/near_duplicate_candidates.csv.gz');save('visual_component_contract.json',vc)
dev=dev.merge(visual,on='id',validate='one_to_one').merge(diag.drop(columns=['cv_fold','path','product_family_group','width','height']),on='id',validate='one_to_one')
assert dev.groupby('product_family_group').cv_fold.nunique().max()==1
summ=[];folds=[];mixed=[];types=[]
for target in ['gender','usage']:
 rows=dev[dev[f'has_{target}_label']].copy()
 fm=rows.groupby('product_family_group')[target].nunique();rows['mixed_family']=rows.product_family_group.map(fm).gt(1)
 vm=rows.groupby('visual_component_id')[target].nunique();rows['mixed_visual_component']=rows.visual_component_id.map(vm).gt(1)
 for label,g in rows.groupby(target):
  sizes=g.groupby('product_family_group').size();typ=g.articleType.value_counts();p=typ/typ.sum()
  summ.append({'target':target,'class':label,'rows':len(g),'families':len(sizes),'visual_components':g.visual_component_id.nunique(),'sha_unique':g.sha256.nunique(),'largest_family':int(sizes.max()),'family_concentration_ess':float(sizes.sum()**2/(sizes**2).sum()),'mixed_family_rows':int(g.mixed_family.sum()),'mixed_visual_component_rows':int(g.mixed_visual_component.sum()),'article_types':len(typ),'effective_article_types':float(np.exp(-(p*np.log(p)).sum())),'top_type':typ.index[0],'top_type_share':float(typ.iloc[0]/len(g)),'brightness_median':g.brightness.median(),'foreground_fraction_median':g.foreground_fraction.median(),'foreground_lt_015':int(g.foreground_fraction.lt(.15).sum()),'foreground_fallback_rows':int(g.foreground_fallback_reason.ne('').sum())})
  for f in range(5):
   tr=g[g.cv_fold.ne(f)];va=g[g.cv_fold.eq(f)]
   folds.append({'target':target,'class':label,'outer_fold':f,'train_rows':len(tr),'train_families':tr.product_family_group.nunique(),'train_visual_components':tr.visual_component_id.nunique(),'val_rows':len(va),'val_families':va.product_family_group.nunique(),'min_inner_train_families':min(tr[tr.cv_fold.ne(j)].product_family_group.nunique() for j in range(5) if j!=f),'min_inner_calibration_families':min(tr[tr.cv_fold.eq(j)].product_family_group.nunique() for j in range(5) if j!=f)})
  for t,part in g.groupby('articleType'):types.append({'target':target,'class':label,'articleType':t,'rows':len(part),'families':part.product_family_group.nunique()})
 mixed.append({'target':target,'mixed_families':int(fm.gt(1).sum()),'mixed_family_rows':int(rows.mixed_family.sum()),'mixed_visual_components':int(vm.gt(1).sum()),'mixed_visual_component_rows':int(rows.mixed_visual_component.sum())})
 for family,g in rows[rows.mixed_family].groupby('product_family_group'):
  pass
pd.DataFrame(summ).to_csv(OUT/'class_independent_support.csv',index=False)
pd.DataFrame(folds).to_csv(OUT/'fold_class_support.csv',index=False)
pd.DataFrame(types).to_csv(OUT/'class_article_diversity.csv',index=False)
save('mixed_labels.json',mixed)
# Fold-local definitions of article-type exceptions and nuisance quartiles.
parts=[];thresholds=[]
for f in range(5):
 tr=dev[dev.cv_fold.ne(f)]; va=dev[dev.cv_fold.eq(f)].copy()
 usage=tr[tr.has_usage_label]
 counts=usage.groupby(['articleType','usage']).size().reset_index(name='n').sort_values(['articleType','n','usage'],ascending=[True,False,True])
 modes=counts.drop_duplicates('articleType').set_index('articleType').usage
 va['fold_train_usual_usage']=va.articleType.map(modes).fillna('UNSUPPORTED')
 va['usage_type_status']=np.where(va.fold_train_usual_usage.eq('UNSUPPORTED'),'unsupported',np.where(va.usage.eq(va.fold_train_usual_usage),'usual','exception'))
 for col in ['brightness','foreground_fraction','near_white_fraction']:
  q=tr[col].quantile([.25,.5,.75]).to_numpy(); va[col+'_band']=np.searchsorted(q,va[col].to_numpy(),side='right')+1
  thresholds.append({'fold':f,'feature':col,'q25':q[0],'q50':q[1],'q75':q[2]})
 parts.append(va)
dev=pd.concat(parts).sort_values('id')
pd.DataFrame(thresholds).to_csv(OUT/'fold_training_thresholds.csv',index=False)
# Nuisance differences are descriptive, not learned predictors or causal tests.
shifts=[]
for f in range(5):
 for target in ['gender','usage']:
  for label,g in dev[dev[f'has_{target}_label']].groupby(target):
   tr=g[g.cv_fold.ne(f)];va=g[g.cv_fold.eq(f)]
   if not len(tr) or not len(va):continue
   for col in ['brightness','foreground_fraction','near_white_fraction','foreground_center_x','foreground_center_y']:
    sd=tr[col].std()
    shifts.append({'target':target,'class':label,'fold':f,'feature':col,'train_n':len(tr),'val_n':len(va),'train_mean':tr[col].mean(),'val_mean':va[col].mean(),'standardized_mean_difference':(va[col].mean()-tr[col].mean())/sd if sd>0 else None,'wasserstein_distance':wasserstein_distance(tr[col],va[col])})
pd.DataFrame(shifts).to_csv(OUT/'within_class_fold_shifts.csv',index=False)
dev.to_csv(OUT/'development_audit_rows.csv.gz',index=False)
# Inspect accepted near-duplicate label differences. No new relabel or pair acceptance.
pairs=pd.read_csv(ROOT/'data/processed/audit/near_duplicate_candidates.csv.gz',keep_default_na=False)
pairs=pairs[pairs.id_1.isin(dev.id)&pairs.id_2.isin(dev.id)].copy()
pairs=pairs[pairs.accepted_near_duplicate.eq(True)].copy()
lookup=dev.set_index('id')
for c in ['gender','usage','product_family_group','cv_fold','path']:
 for i in [1,2]:pairs[f'{c}_{i}']=pairs[f'id_{i}'].map(lookup[c])
pairs.to_csv(OUT/'accepted_development_pairs.csv',index=False)
print(pd.DataFrame(summ)[['target','class','rows','families','visual_components','effective_article_types','mixed_family_rows']].to_string(index=False))
print(json.dumps(json.loads((OUT/'provenance_verdict.json').read_text()),indent=2))
