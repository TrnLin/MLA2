import csv
import json
import re
from pathlib import Path

OUT=Path('reports/task3/rare_expansion_20260906/smart_casual')
rows=list(csv.DictReader((OUT/'candidates.csv').open()))
admitted=list(csv.DictReader(open('data/processed/teacher_plus_rare_usage_20260906/added_images.csv')))
old_hashes={r[k] for r in admitted for k in ['sha256','original_sha256']}
old_urls={r['image_url'] for r in admitted}
rejected=list(csv.DictReader((OUT/'rejected.csv').open()))
kept=[]
for row in rows:
    reason=None
    if row['source_record_id']=='ms_P23038702':
        reason='Visual QA: waist detail crop; full trousers not visible'
    elif row['file_sha256'] in old_hashes or row['image_url'] in old_urls:
        reason='Already admitted image hash or source URL'
    if reason:
        rejected.append({'source_record_id':row['source_record_id'],'reason':reason})
        continue
    if row['source_dataset']=='Amazon Berkeley Objects' and row['source_family_id'].endswith(':'):
        title=row['product_name'].split(' | ')[0].lower()
        title=re.sub(r'\b(black|brown|navy|blue|tan|grey|gray|beige|green|red|olive)\b','',title)
        title=re.sub(r'\b\d+(\.\d+)?\b','',title)
        title=re.sub(r'[^a-z]','',title)
        row['source_family_id']='abo:title:'+title
        row['notes']+=' Family grouped conservatively by English title with colors and sizes removed; source model number unavailable.'
    row['notes']+=' Visual QA passed: isolated product, no collage/logo/placeholder. Final central duplicate and label checks pending.'
    kept.append(row)
with (OUT/'candidates.csv').open('w') as f:
    w=csv.DictWriter(f,list(kept[0]));w.writeheader();w.writerows(kept)
with (OUT/'rejected.csv').open('w') as f:
    w=csv.DictWriter(f,['source_record_id','reason']);w.writeheader();w.writerows(rejected)
summary={'downloaded':158,'retained':len(kept),'distinct_source_record_ids':len({r['source_record_id'] for r in kept}),'distinct_sha256':len({r['file_sha256'] for r in kept}),'conservative_family_groups':len({r['source_family_id'] for r in kept}),'rejected':rejected,'reviewed_contact_sheets':['contact_1.jpg','contact_2.jpg','contact_3.jpg','contact_4.jpg'],'teacher_holdout_inspected':False,'admitted_overlap_check':'Compared image URL and SHA-256 against added_images.csv; none among retained.'}
(OUT/'visual_qa.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
