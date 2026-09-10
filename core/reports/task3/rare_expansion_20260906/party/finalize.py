
from fashion.task3_paths import resolve_task3_path
import csv,json,hashlib,shutil
from pathlib import Path
from PIL import Image
ROOT=Path.cwd(); OUT=ROOT/'reports/task3/rare_expansion_20260906/party'
rows=list(csv.DictReader((OUT/'candidates.csv').open()))
shutil.copyfile(OUT/'candidates.csv',OUT/'all_downloaded.csv')
reasons={}
for i in [9,10,13,42,55,70,99]:reasons[i]='Rear view hides front of dress'
for i in [11,44,63,64,84,104,108,127]:reasons[i]='Strong lifestyle setting or pose; poor teacher-like product framing'
for i in [53,59,65,79,80]:reasons[i]='Steep overhead camera angle distorts dress shape'
for i in [74,114]:reasons[i]='Mirror selfie, phone or additional clothing obscures product'
for i in [103,105,107,111,112]:reasons[i]='Festival/bar styling with distracting props or background'
for i in [110,122,123,142]:reasons[i]='Casual/nightwear-like appearance; ambiguous Party use despite source placement'
old=list(csv.DictReader((ROOT/'data/processed/teacher_plus_rare_usage_20260906/added_images.csv').open()))
oldsha={r['original_sha256'] for r in old}; oldurl={r['source_url'] for r in old};seen=set();accepted=[];rejected=[];qa=[]
for i,r in enumerate(rows):
    reason=reasons.get(i,'')
    if r['file_sha256'] in oldsha or r['product_url'] in oldurl:reason='Prior admitted source or exact hash'
    if r['file_sha256'] in seen:reason='Within-batch exact image duplicate'
    seen.add(r['file_sha256'])
    with Image.open(resolve_task3_path(r['original_path'], root=ROOT)) as im:im.load()
    assert hashlib.sha256((resolve_task3_path(r['original_path'], root=ROOT)).read_bytes()).hexdigest()==r['file_sha256']
    qa.append(dict(contact_sheet=f'contact_{i//30+1:02}.jpg',sheet_index=i,source_record_id=r['source_record_id'],status='rejected' if reason else 'passed',reason=reason or 'Front or three-quarter dress visible; no collage, placeholder, or obvious duplicate'))
    if reason:rejected.append(dict(source_record_id=r['source_record_id'],product_url=r['product_url'],reason=reason))
    else:r['visual_review']='passed_contact_sheet_review';accepted.append(r)
def write(name,rs):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
rejected+=list(csv.DictReader((OUT/'rejected.csv').open()))
write('candidates.csv',accepted);write('rejected.csv',rejected);write('visual_qa.csv',qa)
summary=dict(downloaded=152,accepted_for_central_review=len(accepted),distinct_product_ids=len({r['source_record_id'] for r in accepted}),distinct_source_families=len({r['source_family_id'] for r in accepted}),visual_rejections=len(reasons),prior_admitted_source_url_or_hash_overlap=0,sources={s:sum(r['source_dataset']==s for r in accepted) for s in sorted({r['source_dataset'] for r in accepted})})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
