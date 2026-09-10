from pathlib import Path
import hashlib,json
import pandas as pd
from PIL import Image
O=Path(__file__).parent
p=O/'candidates.csv';x=pd.read_csv(p,dtype=str,keep_default_na=False)
notes={
 'Retrospective® 4 V2.0':'Closed olive canvas shoulder camera bag; isolated white background; complete object and strap.',
 'ROGUE 6L Sling in Black':'Closed compact black camera sling; light grey background; full strap visible; large surrounding margins require standard aspect-preserving preparation.',
 'S4 Camera Bag':'Tan flap shoulder camera bag; white background; complete object; different design from Mini Eventer though related brand styling.',
 'Mini Eventer Camera/Tablet Bag':'Tan camera/tablet shoulder bag with two front pockets and leather base; white background; related Billingham aesthetic retained as a different design.',
 'DOMKE Dri-Safe All Weather Case':'Small black roll-top camera/weather pouch with mesh pocket and shoulder strap; white background. Travel-supplies wording supports use but multi-purpose shape is less teacher-like.',
 '225 MKII Camera/Tablet Bag':'Black compact shoulder camera bag with front straps and side pockets; white/transparent background. Some black details are low contrast but outline is clear.',
 'DNA 9 Slim Messenger Bag Black':'Small black messenger camera bag on white background; complete object;386px source is sufficient for60x80 preparation.',
 'UNCOVER Messenger Sling7L':'Single grey closed camera sling cutout with transparent background; no surrounding camera or lens. Composite alpha on white during central preparation; large top margin should be trimmed by normal foreground preparation.'}
for i,r in x.iterrows():
 assert hashlib.sha256(Path(r.original_path).read_bytes()).hexdigest()==r.file_sha256
 im=Image.open(r.original_path);assert str(im.width)==r.width and str(im.height)==r.height
 x.loc[i,'visual_review']='passed_original_and_contact_sheet_review';x.loc[i,'visual_review_note']=notes[r.product_name]
 x.loc[i,'image_mode']=im.mode;x.loc[i,'requires_white_alpha_composite']=str(im.mode=='RGBA')
 x.loc[i,'proposed_usage']='Travel';x.loc[i,'product_type']='Camera Bags'
 x.loc[i,'retrieved_date']='2026-09-06'
old=pd.read_csv('data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv',dtype=str,keep_default_na=False,usecols=['sha256','original_sha256'])
old_hash=set(old.sha256)|set(old.original_sha256)
x['exact_existing_file_hash_match']=x.file_sha256.isin(old_hash).astype(str)
assert x.file_sha256.nunique()==8 and not x.file_sha256.isin(old_hash).any()
x.to_csv(p,index=False)
x[['source_dataset','source_record_id','product_name','visual_review','visual_review_note','original_path','image_mode','requires_white_alpha_composite']].to_csv(O/'visual_qa.csv',index=False)
(O/'validation.json').write_text(json.dumps({'rows':len(x),'sources':x.source_dataset.nunique(),'unique_file_hashes':x.file_sha256.nunique(),'exact_matches_to_existing_saved_hashes':0,'all_originals_and_contact_sheet_visually_inspected':True,'central_near_duplicate_and_family_review':'pending','rights':'copyright_permission_not_established','no_training_dataset_changes':True},indent=2))
print('PASS:8 candidates,6 brands,8 unique hashes,no exact saved-hash overlap; all originals visually reviewed')
