"""Audit intended product audience only; never assign a gender training target."""
from pathlib import Path
import re,json,hashlib
import pandas as pd
O=Path(__file__).parent;R=O.parent
s=pd.read_csv('data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv',dtype=str,keep_default_na=False)
t=s[(s.source_dataset=='teacher')&(s.partition=='development')&s.usage.isin(['Home','Party','Smart Casual','Travel'])].copy()
e=pd.read_csv(O/'external_687_audit.csv',dtype=str,keep_default_na=False)
src=s.set_index('id')
snapshots={};rows=[]
def infer(title,text,url):
 def detect(v):
  v=v.lower().replace('’',"'")
  if re.search(r'\bunisex\b|\bfor (?:men and women|women and men)\b',v):return 'Unisex'
  female=bool(re.search(r'\bwom[ae]n(?:s|\x27s)?\b|\bladies\b|\bfemale\b|\bfor her\b',v))
  male=bool(re.search(r'\bmen(?:s|\x27s)?\b|\bman\b|\bgents?\b|\bmale\b|\bfor him\b',v))
  if female and male:return 'Conflicting/broad audience text'
  if female:return 'Women'
  if male:return 'Men'
  if re.search(r'\bgirls?\b',v):return 'Girls'
  if re.search(r'\bboys?\b',v):return 'Boys'
  return ''
 for basis,txt in [('product_title',title),('product_text_or_recorded_collection_evidence',text),('explicit_product_url_wording',url.replace('-',' ').replace('_',' '))]:
  a=detect(txt)
  if a:
   m=re.search(r'\bunisex\b|\bwom[ae]n\w*|\bmen\w*|\bman\b|\bladies\b|\bgents?\b|\bfor her\b|\bfor him\b|\bgirls?\b|\bboys?\b',txt,re.I)
   pos=m.start() if m else 0
   return a,basis,txt[max(0,pos-70):pos+180]
 return 'Unknown','no_explicit_audience_established','No audience assigned from neutral appearance, bag type, colour, brand name or a model’s perceived gender.'
def kind(v):
 v=v.lower()
 if 'cushion' in v or 'decorative pillow' in v:return 'Cushion Covers'
 if v in ['formal shoes','casual shoes','shoes (formal/casual ambiguous)','boots']:return 'Closed shoes'
 if 'camera' in v:return 'Camera Bags'
 if 'rucksack' in v:return 'Rucksacks'
 if 'backpack' in v:return 'Backpacks'
 if 'duffel' in v:return 'Duffel Bag'
 if v.startswith('handbag'):return 'Handbags'
 if v=='heels':return 'Heeled sandals / heels'
 return v.title()
for _,r in e.iterrows():
 original=src.loc[r.id];text=' '.join(original.get(c,'') for c in ['product_description','label_evidence','source_label'])
 a,b,p=infer(r.productDisplayName,text,r.source_url)
 rows.append(dict(cohort='existing_687',id=r.id,usage=r.usage,audit_type_group=kind(r.audit_product_type),audit_audience=a,audit_audience_basis=b,audit_audience_evidence=p,product_name=r.productDisplayName,source_dataset=r.source_dataset,source_family_id=r.extension_family_group,source_url=r.source_url,source_snapshot='data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv'))
for folder in ['home','party','smart_casual','travel','camera_travel']:
 f=R/folder/'selected_candidates.csv'
 if not f.exists():continue
 raw=f.read_bytes();snapshots[str(f)]={'sha256':hashlib.sha256(raw).hexdigest()}
 x=pd.read_csv(f,dtype=str,keep_default_na=False);snapshots[str(f)]['rows']=len(x)
 for _,r in x.iterrows():
  a,b,p=infer(r.product_name,' '.join(r.get(c,'') for c in ['description','evidence_text']),r.product_url)
  if a=='Unknown' and r.product_type=='Shirts' and 'www.mjbale.com/products/' in r.product_url:
   a='Men';b='retailer_shirt_category_audience_inference';p='Official M.J. Bale /collections/shirts title: Men\'s Classic Shirts | Business & Casual. Category-level inference for its shirts; individual product membership not verified. Cached audit/mjbale_shirts_collection.html.'
  cachematch=re.search(r'Evidence cache: ([^;\s]+\.json)',r.get('notes',''))
  if a=='Unknown' and cachematch and Path(cachematch.group(1)).exists():
   cached=json.loads(Path(cachematch.group(1)).read_text());products=cached.get('products',[]) if isinstance(cached,dict) else []
   record=next((q for q in products if str(q.get('id'))==r.source_record_id.split('_')[-1]),None)
   if record:
    tags=record.get('tags',[]);tags='; '.join(tags) if isinstance(tags,list) else str(tags)
    ta,tb,tp=infer('',tags,'')
    if ta!='Unknown':
     if ta=='Unisex' and re.search(r'\b(?:Women|Ladies|female)\b',tags,re.I) and re.search(r'\b(?:Men|male)\b',tags,re.I):ta='Conflicting/broad audience text'
     a=ta;b='cached_exact_product_audience_tags';p=tags
  if a=='Unknown' and r.product_type=='Shirts' and 'aaiko.com/products/' in r.product_url:
   catalog=json.loads((R/'smart_casual/aaiko_blouses.json').read_text())['products']
   if any(str(q['id'])==r.source_record_id.split('_')[-1] for q in catalog):
    a='Women';b='verified_womens_collection_membership';p='Exact product ID in aaiko_blouses.json; aaiko_blouses_collection.html description explicitly says latest collection women\'s blouses.'
  rows.append(dict(cohort='selected_replacement_snapshot',id=r.source_dataset+':'+r.source_record_id,usage=r.proposed_usage,audit_type_group=kind(r.product_type),audit_audience=a,audit_audience_basis=b,audit_audience_evidence=p,product_name=r.product_name,source_dataset=r.source_dataset,source_family_id=r.source_family_id,source_url=r.product_url,source_snapshot=str(f)))
x=pd.DataFrame(rows);x.to_csv(O/'external_audience_annotations.csv',index=False)
retired=set(pd.read_csv(O/'retirement_shortlist_128.csv',dtype=str,keep_default_na=False).id)
kept=x[(x.cohort=='existing_687')&(~x.id.isin(retired))].copy();kept['cohort']='ADVISORY_retained_after_128_shortlist_not_final_v3'
allx=pd.concat([x,kept],ignore_index=True)
allx.groupby(['cohort','usage','audit_audience','audit_type_group']).size().rename('rows').reset_index().to_csv(O/'audience_coverage_counts.csv',index=False)
t.groupby(['usage','gender','articleType','cv_fold']).size().rename('rows').reset_index().to_csv(O/'teacher_audience_type_fold_counts.csv',index=False)
t.groupby(['usage','gender','baseColour']).size().rename('rows').reset_index().to_csv(O/'teacher_audience_colour_counts.csv',index=False)
gaps=[]
for (u,a,art),grp in t.groupby(['usage','gender','articleType']):
 k=kind(art);warning='';action='Type-level support does not establish audience coverage. Unknown audience is not evidence of exclusion.'
 if '2633' in set(grp.id):k='Closed shoes';warning='Noisy Heels label: ID2633 is named/pictured as mens casual shoe.'
 if '12348' in set(grp.id):k='EXCLUDE NOISY SHIRT';warning='Sole Men Travel row ID12348 is folded shirt with placeholder name; no valid men-specific Travel sourcing gap.';action='Do not source for this anomalous subgroup.'
 if '35828' in set(grp.id) or '39487' in set(grp.id):k='Camera Bags';warning='Camera pouch/bag title and image override broad Handbags/Mobile Pouch for sourcing only.'
 rec=dict(usage=u,teacher_audience=a,teacher_articleType=art,audit_visual_type=k,teacher_rows=len(grp),teacher_ids=';'.join(grp.id),teacher_folds=','.join(sorted(grp.cv_fold.unique())),audit_warning=warning)
 for cohort,pre in [('existing_687','existing'),('ADVISORY_retained_after_128_shortlist_not_final_v3','advisory_retained'),('selected_replacement_snapshot','proposed')]:
  subset=allx[(allx.cohort==cohort)&(allx.usage==u)&(allx.audit_type_group==k)]
  same=subset[subset.audit_audience==a];unknown=subset[subset.audit_audience=='Unknown']
  rec.update({pre+'_same_audience_rows':len(same),pre+'_same_audience_families':same.source_family_id.nunique(),pre+'_same_audience_sources':same.source_dataset.nunique(),pre+'_unknown_audience_rows':len(unknown),pre+'_other_or_conflicting_audience_rows':len(subset)-len(same)-len(unknown)})
 rec['gap_status']='NOT A VALID SOURCING TARGET' if k.startswith('EXCLUDE') else 'NO EXPLICIT SAME-AUDIENCE COVERAGE' if rec['advisory_retained_same_audience_rows']+rec['proposed_same_audience_rows']==0 else 'SOME AUDIENCE SUPPORT; SOURCE AND PHOTO LIMITS REMAIN'
 rec['recommendation']=action;gaps.append(rec)
pd.DataFrame(gaps).to_csv(O/'audience_gap_table.csv',index=False)
(O/'audience_snapshot_inputs.json').write_text(json.dumps(snapshots,indent=2))
print(allx.groupby(['cohort','usage','audit_audience']).size().to_string())
print('\nSmart Women gaps:');print(pd.DataFrame(gaps).query("usage=='Smart Casual' and teacher_audience=='Women'").to_string(index=False))
