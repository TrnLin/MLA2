"""Recompute saved OOF slices; no fit, inference, or dataset mutation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
O=Path(__file__).resolve().parent; E=Path('results/evidence/task3/experiments')
d=pd.read_csv(O/'development_audit_rows.csv.gz',keep_default_na=False)
models={'usage_E2':('usage',E/'t3_usage_e2_class_balanced_ce/usage/aggregate/oof_predictions.csv'),'usage_E3':('usage',E/'t3_usage_e3_classifier_dropout/usage/aggregate/oof_predictions.csv'),'usage_E8':('usage',E/'t3_usage_e8_translation/usage/aggregate/oof_predictions.csv'),'usage_E9':('usage',E/'t3_usage_e9_exception_balance/usage/aggregate/oof_predictions.csv'),'gender_E6':('gender',E/'t3_gender_e6_gem_p3/gender/aggregate/oof_predictions.csv'),'gender_G2_screen':('gender',E/'t3_gender_v2_g2_translation/gender/aggregate/oof_predictions.csv')}
summary=[];slices=[];classrows=[];joined={};coverage=[];standardized=[]
for name,(target,p) in models.items():
 a=pd.read_csv(p,keep_default_na=False)
 assert not a.id.duplicated().any()
 classes=[c.split('_',2)[2] for c in a.columns if c.startswith('probability_')]
 assert set(a.id)<=set(d.id)
 a['true_label']=a.true_index.map(dict(enumerate(classes)));a['predicted_label']=a.predicted_index.map(dict(enumerate(classes)))
 r=d.merge(a[['id','true_label','predicted_label','confidence','cv_fold','product_family_group']],on='id',validate='one_to_one',suffixes=('','_oof'))
 assert r[target].eq(r.true_label).all();assert r.cv_fold.eq(r.cv_fold_oof).all();assert r.product_family_group.eq(r.product_family_group_oof).all()
 r['correct']=r.true_label.eq(r.predicted_label);r['model']=name
 mixed=d.groupby('product_family_group')[target].nunique();r['mixed_family']=r.product_family_group.map(mixed).gt(1)
 coverage.append({'model':name,'rows':len(r),'folds':sorted(r.cv_fold.unique().tolist()),'id_label_fold_family_checks':True,'source':str(p)})
 joined[name]=r
 for f,g in [('all',r)]+list(r.groupby('cv_fold')):
  summary.append({'model':name,'fold':f,'rows':len(g),'error':1-g.correct.mean(),'macro_f1':f1_score(g.true_label,g.predicted_label,labels=classes,average='macro',zero_division=0),'wrong_confidence_mean':g.loc[~g.correct,'confidence'].mean()})
  for label in classes:
   t=g.true_label.eq(label);pred=g.predicted_label.eq(label);tp=int((t&pred).sum());sup=int(t.sum());pc=int(pred.sum())
   classrows.append({'model':name,'fold':f,'class':label,'support':sup,'predicted':pc,'tp':tp,'recall':tp/sup if sup else None,'f1':2*tp/(sup+pc) if sup+pc else 0})
 for feature in ['usage_type_status','mixed_family','brightness_band','foreground_fraction_band','near_white_fraction_band']:
  for (f,value),g in r.groupby(['cv_fold',feature]):
   slices.append({'model':name,'fold':f,'slice':feature,'value':str(value),'rows':len(g),'errors':int((~g.correct).sum()),'error_rate':1-g.correct.mean(),'wrong_confidence':g.loc[~g.correct,'confidence'].mean()})
 # Direct standardization on common class x usual/exception cells. No fitted model.
 if target=='usage':
  common=r[r.usage.isin(['Casual','Ethnic','Formal','Sports']) & r.usage_type_status.ne('unsupported')].copy()
  common['stratum']=common.usage+'|'+common.usage_type_status
  counts=pd.crosstab(common.stratum,common.cv_fold)
  eligible=counts.index[counts.min(axis=1)>=20]
  c=common[common.stratum.isin(eligible)];w=c.stratum.value_counts(normalize=True)
  for f,g in c.groupby('cv_fold'):
   rates=g.groupby('stratum').correct.mean()
   standardized.append({'model':name,'fold':f,'rows':len(g),'coverage_of_fold':len(g)/len(r[r.cv_fold.eq(f)]),'raw_error_matched_scope':1-g.correct.mean(),'standardized_error':1-float((rates*w).sum()),'strata':len(w)})
pd.DataFrame(summary).to_csv(O/'saved_oof_summary.csv',index=False)
pd.DataFrame(slices).to_csv(O/'saved_oof_error_slices.csv',index=False)
pd.DataFrame(classrows).to_csv(O/'saved_oof_class_counts.csv',index=False)
pd.DataFrame(standardized).to_csv(O/'usage_fold_standardization.csv',index=False)
(O/'oof_coverage_checks.json').write_text(json.dumps(coverage,indent=2))
# Matched error transitions support mechanism analysis, not label changes.
trans=[]
for name in ['usage_E3','usage_E8','usage_E9']:
 b=joined['usage_E2'].set_index('id');c=joined[name].set_index('id');assert b.index.equals(c.index)
 for s in ['usual','exception','unsupported']:
  z=b.usage_type_status.eq(s);trans.append({'child':name,'slice':s,'rows':int(z.sum()),'E2_wrong_child_right':int((z&~b.correct&c.correct).sum()),'E2_right_child_wrong':int((z&b.correct&~c.correct).sum()),'parent_error':float((~b.correct[z]).mean()),'child_error':float((~c.correct[z]).mean())})
pd.DataFrame(trans).to_csv(O/'usage_error_transitions.csv',index=False)
# Label-review queue is explicitly diagnostic, with teacher labels, not a blind review form.
r=joined['usage_E2'];r=r[r.cv_fold.isin([1,2,3]) & r.usage_type_status.eq('exception') & ~r.correct].copy()
r.sort_values(['confidence','id'],ascending=[False,True]).drop_duplicates('product_family_group').head(60)[['id','cv_fold','product_family_group','usage','articleType','predicted_label','confidence','path']].to_csv(O/'usage_exception_review_queue.csv',index=False)
print(pd.DataFrame(trans).to_string(index=False));print(pd.DataFrame(standardized).to_string(index=False))
# Recheck latest fit review fold means and G2 source readbacks.
detail=json.load(open('reports/task3/fit_review_20260905/detail.json'))
print([(x['model'],list(x.keys())) for x in detail][-3:])
