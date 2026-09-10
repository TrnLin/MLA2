
from fashion.task3_paths import resolve_task3_path
import csv, json, re, html, hashlib, io, concurrent.futures
from pathlib import Path
import requests
from PIL import Image, ImageOps, ImageDraw

ROOT=Path.cwd()
OUT=ROOT/'reports/task3/rare_replacement_20260906/party'
DEST=ROOT/'data/external/rare_usage_replacement_20260906/candidates/party'
rows=[]
def clean(s): return ' '.join(html.unescape(re.sub('<[^>]+>',' ',s)).split())
def add(p,base,kind,ev,basis,family,confidence='high',notes=''):
 rows.append(dict(source_dataset=base.split('//')[1]+' public catalogue',source_record_id=str(p['id']),product_name=p['title'],description=clean(p['body_html']),proposed_usage='Party',evidence_text=ev,evidence_basis=basis,product_url=base+'/products/'+p['handle'],image_url=p['images'][0]['src'],source_family_id=base.split('//')[1]+':'+family,confidence=confidence,notes=notes,rights_status='Retailer/manufacturer copyright; no open image licence established; research review only',original_path='',file_sha256='',width='',height='',product_type=kind))

seen=set()
for p in json.loads((OUT/'olga.json').read_text())['products']:
 tags=p['tags'];family=p['title'].split()[0].lower();ev=[x for x in tags if x in ['Cocktail','Evening','Night Out']]
 if not ev or family in seen or 'clutch' not in p['title'].lower() or any(x in p['title'].lower() for x in ['personal','mrs','straw']):continue
 seen.add(family);add(p,'https://olgaberg.com','Clutches','Retailer product tags: '+', '.join(ev),'explicit_source_label',family)
 if len(seen)>=18:break
for p in json.loads((OUT/'bellaparty.json').read_text())['products']:
 add(p,'https://bellavitaorganic.com','Perfume','Member of manufacturer Party collection: https://bellavitaorganic.com/collections/party','explicit_source_label',p['title'].split(' Perfume')[0].lower(),notes='Party collection membership is explicit; some products also have daily/date/office uses. Not scent inferred from bottle.')
watch_names=['Michael Kors Phoebe MK4923','Michael Kors Lexington MK4842',"U.S. Polo Assn. woman's Felicity Watch - USP8405GR", "U.S. Polo Assn. Azure women's watch - USP5655YG",'Tommy Hilfiger Georgia 1782818','Tommy Hilfiger Chloe 1782860','U.S. Polo Assn. Mirabelle Analog Watch USP8487BL','Michael Kors Slim Runway MK7474','Michael Kors Gramercy MK7529','Tommy Hilfiger Classic Ivy 1782786','Tommy Hilfiger Jade 1782776',"U.S. Polo Assn. Fancy women's watch - EP2064RG"]
for p in json.loads((OUT/'dayt.json').read_text())['products']:
 if p['title'] not in watch_names:continue
 d=clean(p['body_html']);m=re.search(r'evening occasions|special occasions',d,re.I)
 if not m:continue
 add(p,'https://dayt.ie','Watches',d[max(0,m.start()-160):m.end()+80],'product_text_inference',re.sub(r'\b(?:MK|USP|EP|CW)?\d+[A-Z]*\b','',p['title']).strip().lower(),'medium','Retailer explicitly describes evening/special occasions, but also everyday/work use. Party mapping is contextual and not exclusive; parent should review before admission.')

def fetch(r):
 try:
  path=DEST/(r['source_dataset'].split()[0]+'_'+r['source_record_id']+'.jpg')
  if not path.exists():
   u=r['image_url'];response=requests.get(u+('&' if '?' in u else '?')+'width=800',timeout=30);response.raise_for_status();path.write_bytes(response.content)
  with Image.open(path) as im:im.load();r['width'],r['height']=im.size
  r['image_url']+=('&' if '?' in r['image_url'] else '?')+'width=800'
  r['original_path']=str(path.relative_to(ROOT));r['file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();return r,None
 except Exception as e:return None,dict(source_record_id=r['source_record_id'],reason=str(e))
good=[];bad=[]
with concurrent.futures.ThreadPoolExecutor(8) as ex:
 for r,e in ex.map(fetch,rows):
  if r:good.append(r)
  else:bad.append(e)
def write(name,data):
 if not data:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('candidates.csv',good);write('rejected.csv',bad)
for start in range(0,len(good),24):
 sheet=Image.new('RGB',(1200,1080),'white');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(good[start:start+24]):
  x=j%6*200;y=j//6*270
  with Image.open(resolve_task3_path(r['original_path'], root=ROOT)) as im:
   th=ImageOps.contain(im.convert('RGB'),(194,235));sheet.paste(th,(x+(194-th.width)//2,y))
  draw.text((x+2,y+236),f'{start+j:02} '+r['product_name'][:26],fill='black')
 sheet.save(OUT/f'contact_{start//24+1:02}.jpg')
print(json.dumps({'downloaded':len(good),'rejected':bad,'types':{k:sum(r['product_type']==k for r in good) for k in ['Clutches','Perfume','Watches']}}))
