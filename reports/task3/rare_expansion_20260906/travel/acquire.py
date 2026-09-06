
from fashion.task3_paths import resolve_task3_path
import csv, gzip, hashlib, html, json, re, shutil, concurrent.futures
from pathlib import Path
import requests
from PIL import Image, ImageDraw

ROOT=Path.cwd(); OUT=ROOT/'reports/task3/rare_expansion_20260906/travel'; DATA=ROOT/'data/external/rare_usage_expansion_20260906/candidates/travel'
RAW=ROOT/'reports/task3/rare_external_intake_20260906/sources/abo/raw'
travel=re.compile(r'\b(travel\w*|hiking|trekking|overnight|weekend\w*|trail\w*|rucksack)\b',re.I)
bag=re.compile(r'backpack|duffel|duffle|rucksack|travel bag|weekender|travel pack',re.I)
reject=re.compile(r'laundry|toiletr|golf|sleeping|chair|dog |pet |packing|shoe bag|cooler|hydration|bike |ski |snow|builder|surf|mini 7l',re.I)
added=list(csv.DictReader(open(ROOT/'data/processed/teacher_plus_rare_usage_20260906/added_images.csv')))
old_ids={r['source_id'] for r in added}; old_urls={r['image_url'] for r in added}; old_hashes={r['original_sha256'] for r in added}
items=[]; rejected=[]
def add(source,sid,title,desc,url,img,family,rights,native):
 if sid in old_ids or img in old_urls:
  rejected.append(dict(source_record_id=sid,reason='Already admitted source or image'));return
 evidence=' | '.join(s.strip() for s in re.split(r'[.!?\n]+',title+'\n'+desc) if travel.search(s))
 if not bag.search(title) or not evidence or reject.search(title):return
 items.append(dict(source_dataset=source,source_record_id=sid,product_name=title,description=desc,proposed_usage='Travel',evidence_text=evidence,evidence_basis='product_text_inference',product_url=url,image_url=img,source_family_id=family,confidence='high' if travel.search(title) else 'medium',notes='Main catalogue photo; infer usage from product text; colour/model variants share family',rights_basis=rights))
 native_records[sid]=native
native_records={}
for f in sorted(OUT.glob('*.json')):
 if f.stem in ['selected_native_records','visual_qa']:continue
 data=json.loads(f.read_text())
 if 'products' not in data:continue
 brand=re.sub(r'\d+$','',f.stem); domain={'dakine':'www.dakine.com','cotopaxi':'www.cotopaxi.com','tombihn':'www.tombihn.com','topodesigns':'topodesigns.com','eaglecreek':'www.eaglecreek.com','dbjourney':'dbjourney.com'}[brand]
 for r in data['products']:
  if not r['images']:continue
  title=html.unescape(r['title']);desc=html.unescape(re.sub('<[^>]+>',' ',r['body_html'] or ''));desc=re.sub(r'\s+',' ',desc)
  fam=re.split(r' - | Black| Blue| Green| Red| Grey| Gray| White| Basil| Affogato| Sand| Chrome| Moss| Fog| Glacier| Midnight| Sunset| Falu| Cloud| Desert| Golden| Raspberry| Sage| Wine| Ice| Forest| Sepia| Deep| Sunrise| Gneiss| Magnesium| Parhelion| Sunbleached',title)[0]
  fam=re.sub(r'\s*\(Discontinued\)| Sale| Final Sale','',fam)
  add(brand,str(r['id']),title,desc,'https://'+domain+'/products/'+r['handle'],r['images'][0]['src'],brand+':'+fam.lower(),'Public brand catalogue; copyright retained by brand; no open training/reuse licence verified',r)
with gzip.open(RAW/'images.csv.gz','rt') as f:index={r['image_id']:r for r in csv.DictReader(f)}
for f in RAW.glob('listings_*.json.gz'):
 for l in gzip.open(f,'rt'):
  r=json.loads(l)
  def vals(k):return [v.get('value','') for v in r.get(k,[]) if v.get('language_tag','en').startswith('en')]
  title=next(iter(vals('item_name')),'');desc=' | '.join(vals('bullet_point')+vals('product_description')+vals('style'))
  if not bag.search(title) or not travel.search(title+' '+desc) or reject.search(title):continue
  ent=index.get(r.get('main_image_id'))
  if not ent:continue
  fam=re.split(r' - |,|\(',title)[0].lower()
  add('Amazon Berkeley Objects',r['item_id'],title,desc,'https://'+r['domain_name']+'/dp/'+r['item_id'],'https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/original/'+ent['path'],'abo:'+fam,'CC BY 4.0; Amazon.com; official ABO release licence and attribution in prior source cache',r)
seen=set(); unique=[]
for r in items:
 if r['source_record_id'] not in seen:unique.append(r);seen.add(r['source_record_id'])
families={}; selected=[]; extras=[]
for r in sorted(unique,key=lambda r:(r['confidence']!='high',r['source_dataset']!='Amazon Berkeley Objects')):
 if r['source_family_id'] in families:extras.append(r)
 else:selected.append(r);families[r['source_family_id']]=1
selected=(selected+extras)[:180]
def download(r):
 try:
  p=DATA/(re.sub(r'\W+','_',r['source_dataset'])+'_'+r['source_record_id']+'.jpg')
  old=ROOT/'data/external/rare_usage_20260906/abo'/ (r['source_record_id']+'.jpg')
  if not p.exists():
   if old.exists():shutil.copyfile(old,p)
   else:
    resp=requests.get(r['image_url'],timeout=50);resp.raise_for_status();p.write_bytes(resp.content)
  with Image.open(p) as im:im.load();w,h=im.size
  sha=hashlib.sha256(p.read_bytes()).hexdigest()
  if sha in old_hashes:raise ValueError('Already admitted exact image hash')
  if min(w,h)<100:raise ValueError('Image too small')
  r.update(original_path=str(p.relative_to(ROOT)),file_sha256=sha,width=w,height=h);return r,None
 except Exception as e:return None,dict(source_record_id=r['source_record_id'],reason=str(e))
rows=[];seenhash=set()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for r,err in pool.map(download,selected):
  if err:rejected.append(err)
  elif r['file_sha256'] in seenhash:rejected.append(dict(source_record_id=r['source_record_id'],reason='Exact duplicate in intake'))
  else:rows.append(r);seenhash.add(r['file_sha256'])
def csvwrite(p,rows,fields=None):
 with open(p,'w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
csvwrite(OUT/'candidates.csv',rows);csvwrite(OUT/'rejected.csv',rejected,['source_record_id','reason'])
(OUT/'selected_native_records.json').write_text(json.dumps({r['source_record_id']:native_records[r['source_record_id']] for r in rows},indent=2))
for page in range((len(rows)+39)//40):
 sheet=Image.new('RGB',(1200,1100),'white');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(rows[page*40:(page+1)*40]):
  im=Image.open(resolve_task3_path(r['original_path'], root=ROOT)).convert('RGB');im.thumbnail((140,170));x=(j%8)*150;y=(j//8)*220;sheet.paste(im,(x+(150-im.width)//2,y));d.text((x+3,y+173),str(page*40+j)+' '+r['source_record_id'][-10:],fill='black');d.text((x+3,y+190),r['product_name'][:22],fill='black')
 sheet.save(OUT/f'contact_sheet_{page+1}.jpg')
print(json.dumps({'downloaded':len(rows),'families':len({r['source_family_id'] for r in rows}),'rejected':len(rejected),'eligible':len(unique)}))
