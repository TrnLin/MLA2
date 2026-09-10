"""Verify the completed G-D1 screen from saved artifacts, without model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.task3_gender_repair import NAME, load_sources, evaluate_gender_repair
from fashion.train.task3_dataset_v2 import dataset_v2_spec
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_repair_preflight import DIAGNOSTICS, memory_profile_passes, _code_hashes, _data_hashes

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPORT=Path(__file__).resolve().parent
G2=ROOT/'reports/task3/g2_gate_audit_20260905/drive'
E6=ROOT/'results/evidence/task3/experiments/t3_gender_e6_gem_p3/gender'
sources, classes=load_sources(g2_directory=G2,e6_directory=E6,registry_path=REPORT/'runs.csv',root=ROOT)
spec=dataset_v2_spec(NAME,[sources['G2'][f]['run_id'] for f in range(5)])
prereq_path=REPORT/'prerequisites/prerequisites.json'
pre=json.loads(prereq_path.read_text())
assert pre['version']==2 and pre['ready'] is True and pre['optimizer_steps']==0
assert pre['spec']==spec.to_dict()
assert pre['code_sha256']==_code_hashes(ROOT)
assert pre['data_sha256']==_data_hashes(ROOT)
assert pre['memory_profile']['passed'] and memory_profile_passes(pre['memory_profile'])
assert pre['diagnostic_views']==list(DIAGNOSTICS)
assert len(pre['clean_reproduction'])==10 and all(row['passed'] for row in pre['clean_reproduction'])
assert len(pre['diagnostic_sha256'])==301 and len(pre['source_sha256'])==70
runs_by_id={r['run_id']:r for group in sources.values() for r in group.values()}
for remote,digest in pre['source_sha256'].items():
 p=Path(remote); local=Path(runs_by_id[p.parent.name]['directory'])/p.name
 assert compute_sha256(local)==digest,remote
registry=pd.read_csv(REPORT/'runs.csv',keep_default_na=False)
splits=load_splits(ROOT/'data/processed/splits.csv')
child={}
for path in sorted((REPORT/'gender').glob('t3_gender_translation_mild_darkening_*')):
 if not path.is_dir(): continue
 run=inspect_gender_run(path,registry=registry,splits=splits,classes=classes,root=ROOT)
 assert run['fold'] not in child
 assert run['config']['child_experiment']==spec.to_dict()
 assert run['config']['parent_run_id']==sources['G2'][run['fold']]['run_id']
 assert run['metrics']['prerequisite_sha256']==compute_sha256(prereq_path)
 assert run['metrics']['saved_tensors_on_cpu'] is False
 env=json.loads(registry.loc[registry.run_id.eq(run['run_id']),'environment_json'].iloc[0])
 assert {k:env[k] for k in pre['environment']}==pre['environment']
 child[run['fold']]=run
assert set(child)=={0,4}
result=evaluate_gender_repair(child,sources,classes,phase='screen')
saved=json.loads((REPORT/'gender/screen_decision.json').read_text())
def compare(a,b,path="decision"):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a: compare(a[k],b[k],path+"."+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b,strict=True)): compare(x,y,path+f"[{i}]")
 elif isinstance(a,float):
  assert np.isclose(a,b,atol=1e-12,rtol=0),(path,a,b)
  if a!=b: print("Floating-point rounding:",path,abs(a-b))
 else: assert a==b,(path,a,b)
compare(result,saved)
rows=[]
for f in (0,4):
 for name,r in [('G-D1',child[f]),('G2',sources['G2'][f]),('E6',sources['E6'][f])]:
  m=r['metrics']; rb=r['robustness'].set_index('corruption')
  rows.append(dict(fold=f,model=name,clean_f1=m['macro_f1'],train_f1=m['final_train_eval_macro_f1'],gap=m['final_train_validation_macro_f1_gap'],nll=m['nll'],ece_15=m['ece_15'],dark_f1=rb.loc['brightness_085','macro_f1'],dark_change=rb.loc['brightness_085','macro_f1_change'],shift_f1=rb.loc['translation_003','macro_f1'],shift_change=rb.loc['translation_003','macro_f1_change'],seconds=m['train_seconds'],memory_bytes=m['peak_memory_bytes']))
pd.DataFrame(rows).to_csv(REPORT/'fold_comparison.csv',index=False)
pd.DataFrame(result['checks']).to_csv(REPORT/'gates.csv',index=False)
verified_diagnostics=0
for remote,digest in pre['diagnostic_sha256'].items():
 suffix=remote.split('/prerequisites/',1)[1]
 local=REPORT/'prerequisites'/suffix
 if local.is_file():
  assert compute_sha256(local)==digest,remote
  verified_diagnostics+=1
report=dict(result,verified_source_runs=10,verified_child_runs=2,verified_diagnostic_files=verified_diagnostics,declared_diagnostic_files=301,diagnostic_download_complete=verified_diagnostics==301,checkpoint_inference_repeated_locally=False,registry_sha256=compute_sha256(REPORT/'runs.csv'),prerequisite_sha256=compute_sha256(prereq_path))
(REPORT/'verified_decision.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':result['status'],'failed_checks':[r for r in result['checks'] if r['status']!='pass'],'clean':{k:result['evidence']['screen'][k]['macro_f1'] for k in ['candidate','comparison']},'bootstrap':result['evidence']['screen']['bootstrap']},indent=2))
print(pd.DataFrame(rows).to_string(index=False))
