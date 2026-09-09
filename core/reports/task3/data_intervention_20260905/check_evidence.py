"""Check saved evidence and decision arithmetic; never instantiate a model."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import f1_score
O=Path(__file__).resolve().parent; E=Path('results/evidence/task3/clean_slate_eda');dev=pd.read_csv(O/'development_audit_rows.csv.gz',keep_default_na=False)
# Existing probe bytes/scopes/score are verifiable even though original contract payload is missing.
summary=pd.read_csv(E/'view_probes/view_probe_summary.csv',keep_default_na=False);out=[]
for r in summary.itertuples():
 p=E/'view_probes'/f'{r.target}_{r.view}_oof.csv';d=pd.read_csv(p,keep_default_na=False)
 if not {'true_label','predicted_label'}<=set(d.columns):print('probe columns',list(d.columns));continue
 joined=d.merge(dev[['id','cv_fold','product_family_group',r.target]],on='id',suffixes=('','_canonical'),validate='one_to_one')
 truth=joined.true_label.replace('', 'NA') if r.target=='usage' else joined.true_label
 pred=joined.predicted_label.replace('', 'NA') if r.target=='usage' else joined.predicted_label
 labels=sorted(dev.loc[dev[f'has_{r.target}_label'],r.target].unique())
 score=f1_score(truth,pred,labels=labels,average='macro',zero_division=0)
 out.append({'target':r.target,'view':r.view,'rows':len(d),'rows_match':len(joined)==len(d),'truth_matches':bool(truth.eq(joined[r.target]).all()),'f1_recomputed':score,'f1_saved':r.pooled_macro_f1,'difference':score-r.pooled_macro_f1})
pd.DataFrame(out).to_csv(O/'old_probe_readback_check.csv',index=False)
# Numeric G2 gates from saved artifacts. Missing bootstrap / full artifact checks are unknown, never pass.
g=json.load(open('reports/task3/fit_review_20260905/g2_readback/aggregate_five_fold/metrics.json'))
e=json.load(open('results/evidence/task3/experiments/t3_gender_e6_gem_p3/gender/aggregate/metrics.json'))
detail=json.load(open('reports/task3/fit_review_20260905/detail.json'));gd=next(x for x in detail if x['model']=='G2-five-fold');ed=next(x for x in detail if x['model']=='g-e6')
f=pd.read_csv(O/'g2_matched_fold_recheck.csv');c=pd.read_csv(O/'g2_matched_class_recheck.csv').set_index('class');gr=gd['robustness']['macro_f1_change'];er=ed['robustness']['macro_f1_change'];r=[]
def gate(name,value,rule,passed):r.append({'gate':name,'value':value,'rule':rule,'status':'pass' if passed else 'fail'})
gate('five_fold_macro_f1',g['macro_f1'],'>=0.7462',g['macro_f1']>=.7462)
gate('improved_folds',int(f.delta.gt(0).sum()),'>=4',f.delta.gt(0).sum()>=4)
gate('worst_fold_delta',f.delta.min(),'>=-0.005',f.delta.min()>=-.005)
gate('fold_sd',g['fold_macro_f1_sample_sd'],'<=0.0209',g['fold_macro_f1_sample_sd']<=.0209)
gap=f.G2_gap.mean();parentgap=f.parent_gap.mean();gate('clean_gap',gap,'<=0.2466',gap<=.2466);gate('gap_reduction',parentgap-gap,'>=0.020',parentgap-gap>=.020);gate('worst_fold_gap_worsening',(f.G2_gap-f.parent_gap).max(),'<=0.005',(f.G2_gap-f.parent_gap).max()<=.005)
minor=c.loc[['Boys','Girls','Unisex'],'G2_f1'].mean();gate('minority_mean',minor,'>=0.6013',minor>=.6013)
for k,floor in [('Men',.9188),('Women',.8950)]:gate(k+'_f1',c.loc[k,'G2_f1'],f'>={floor}',c.loc[k,'G2_f1']>=floor)
for k,cap in [('nll',.4525),('ece_15',.0842)]:gate(k,g[k],f'<={cap}',g[k]<=cap)
for key in gr:
 delta=gr[key]-er[key];is_shift=key=='translation_003';gate(key+'_induced_delta_vs_E6',delta,'>=0.030' if is_shift else '>=-0.020',delta>=(.03 if is_shift else -.02))
for name in ['fresh_fold_family_bootstrap','five_fold_family_bootstrap','complete_registry_checkpoint_oof_integrity']:
 r.append({'gate':name,'value':None,'rule':'full source evidence required','status':'not_recomputed'})
pd.DataFrame(r).to_csv(O/'g2_frozen_gate_readback.csv',index=False)
print(pd.DataFrame(r).to_string(index=False))
# Recheck all fit-table means available in detail, and early curve instability.
l=pd.read_csv('reports/task3/fit_review_20260905/all_experiments.csv');curves=pd.read_csv('reports/task3/fit_review_20260905/mean_curves.csv');re=[]
for x in detail:
 m=x['model'];keys=[m,m.replace('t3_gender_v2_','').replace('t3_usage_v2_','')];matched=l[l.model.isin(keys)]
 if len(matched)!=1 or not x['fold_metrics']:continue
 v=np.mean([f['macro_f1'] for f in x['fold_metrics']]);re.append({'model':m,'validation_mean':v,'ledger_mean':float(matched.iloc[0].val_mean),'difference':v-float(matched.iloc[0].val_mean)})
pd.DataFrame(re).to_csv(O/'fit_ledger_recheck.csv',index=False)
print('fold ledger rows verified',len(re),'max error',max(abs(x['difference']) for x in re))
for m in ['g-e1','g-e2','g-e3']:
 z=curves[curves.model.eq(m)];print(m,'epochs5-20 range',z[z.epoch.between(5,20)].validation_macro_f1.agg(['min','max']).to_dict(),'epochs25-30 range',z[z.epoch.between(25,30)].validation_macro_f1.agg(['min','max']).to_dict())
