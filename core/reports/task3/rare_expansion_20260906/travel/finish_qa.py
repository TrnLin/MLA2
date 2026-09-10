
from fashion.task3_paths import resolve_task3_path
import csv,json,re,collections
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path.cwd();OUT=ROOT/'reports/task3/rare_expansion_20260906/travel'
rows=list(csv.DictReader(open(OUT/'candidates.csv')))
reasons={6:'Product plus phone graphic; composite',13:'Multiple views/packability composite',16:'Passport/document accessory case, not target bag type',17:'Brand logo placeholder',22:'Product plus colour thumbnails; composite',25:'Product plus smaller item; composite',52:'Renewed listing duplicates source model/image of row 56',54:'Yoga-specific bag; sports stronger than travel',57:'Picnic/cooler bag with equipment; not target bag type',60:'Diaper bag; not target bag type',85:'Hydration/cycling bag',87:'Snow/ski-specific bag',89:'Lifestyle cycling photograph',90:'Hydration/cycling bag',107:'Multiple product views in main image'}
rejected=list(csv.DictReader(open(OUT/'rejected.csv')));keep=[]
for i,r in enumerate(rows):
 if i in reasons:
  rejected.append({'source_record_id':r['source_record_id'],'reason':reasons[i]});continue
 title=r['product_name'];brand=r['source_dataset']
 if brand=='dbjourney':
  match=re.search(r'^(.+?\b\d+L)\b',title,re.I)
  fam=match.group(1) if match else re.split(r' 1st Generation| Black| Chrome| Moss',title)[0]
 elif brand=='Amazon Berkeley Objects':
  fam=re.sub(r'\(Renewed\)|\(renewed\)','',title)
  fam=re.sub(r'Amazon Brand - |AmazonBasics |Amazon Brand: ','',fam)
  fam=re.split(r',| - |\(',fam)[0]
  fam=re.sub(r'\b(Black|Blue|Grey|Gray|Purple|Salmon|Red|Yellow)\b','',fam,flags=re.I)
 else:fam=r['source_family_id'].split(':',1)[1]
 r['source_family_id']=brand.lower()+':'+re.sub(r'\s+',' ',fam.lower()).strip()
 r['notes']+='; visual catalogue review passed; any alpha transparency must be composited on white during resize'
 keep.append(r)
for filename,rr,fields in [('candidates.csv',keep,list(rows[0])),('rejected.csv',rejected,['source_record_id','reason'])]:
 with open(OUT/filename,'w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
for page in range((len(keep)+39)//40):
 sheet=Image.new('RGB',(1200,1100),'white');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(keep[page*40:(page+1)*40]):
  im=Image.open(resolve_task3_path(r['original_path'], root=ROOT)).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');im.thumbnail((140,170));x=j%8*150;y=j//8*220;sheet.paste(im,(x+(150-im.width)//2,y));d.text((x+3,y+173),str(page*40+j)+' '+r['source_record_id'][-10:],fill='black');d.text((x+3,y+190),r['product_name'][:22],fill='black')
 sheet.save(OUT/f'qa_contact_sheet_{page+1}.jpg')
summary={'downloaded_files':180,'accepted_candidates':len(keep),'distinct_product_records':len({r['source_record_id'] for r in keep}),'conservative_family_groups':len({r['source_family_id'] for r in keep}),'rejected':len(rejected),'by_source':dict(collections.Counter(r['source_dataset'] for r in keep)),'visual_review':'All five original contact sheets inspected; fifteen unsuitable entries removed. Transparent catalogue PNGs require white alpha compositing. Original files unchanged; extension may be jpg while decoded format is PNG/WebP.','pending':'Root central teacher/earlier-addition exact and near overlap audit. Product text inference is not a source-assigned Usage label. Retail rights remain unverified.'}
(OUT/'visual_qa.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
