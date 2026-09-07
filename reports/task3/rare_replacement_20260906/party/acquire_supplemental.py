
from fashion.task3_paths import resolve_task3_path
import csv,json,re,html,hashlib,concurrent.futures
from pathlib import Path
import requests
from PIL import Image,ImageOps,ImageDraw
ROOT=Path.cwd();OUT=ROOT/'reports/task3/rare_replacement_20260906/party';DEST=ROOT/'data/external/rare_usage_replacement_20260906/candidates/party'
cols=list(next(csv.reader((OUT/'candidates.csv').open())));rows=[]
def row(**kw):
 r={k:'' for k in cols};r.update(proposed_usage='Party',evidence_basis='explicit_source_label',confidence='high',rights_status='Manufacturer/retailer copyright; no open image licence established; research review only');r.update(kw);rows.append(r)
items=[
 ('68026WM01','Glitch Quartz Analog Rose Gold Dial Metal Strap','fastrack-glitch-quartz-analog-watch-for-girls-with-rose-gold-dial-metal-strap-68026wm01','dwfcae1423'),
 ('6288QM01','Automatics Brown Dial Stainless Steel Strap','fastrack-automatic-brown-dial-watch-for-girls-with-brown-colour-stainless-steel-strap-6288qm01','dwc7adadba'),
 ('6279SM01','Younique Quartz Analog Silver Dial Metal Strap','fastrack-younique-quartz-analog-watch-for-girls-with-silver-dial-metal-strap-6279sm01','dw09773813')]
for code,name,handle,imagehash in items:
 row(source_dataset='Fastrack official Partywear collection',source_record_id=code,product_name='Fastrack '+name+' Watch for Girls',description='Women/girls watch listed in official Partywear Watches collection.',evidence_text='Official Partywear Watches listing https://www.fastrack.in/shop/partywear-watches?lang=en_IN links this exact product. Collection introduction explicitly describes party ensembles and night-out use.',product_url='https://www.fastrack.in/product/'+handle+'.html?catID=partywear-watches&lang=en_IN&plp=true',image_url=f'https://www.fastrack.in/dw/image/v2/BKDD_PRD/on/demandware.static/-/Sites-titan-master-catalog/default/{imagehash}/images/Fastrack/Catalog/{code}_1.jpg?sh=600&sw=600',source_family_id='fastrack:'+code[:5],notes='Collection/product evidence cached in fastrack_web_evidence.json and fastrack_product_web_evidence.json. Public image URL observed via product image link. One colour per model.',product_type='Watches')
for code,imagehash in [('95318WM01','dw1fc1ab95'),('95319WM01','dwda1fc964'),('95320WM02','dwec0b6d32')]:
 row(source_dataset='Titan official Raga Cocktails collection',source_record_id=code,product_name='Titan Raga Cocktails '+code+' Women Watch',description='Titan Raga Cocktails quartz analog stainless steel watch for women.',evidence_text='Product specification: Collection Raga Cocktails. Official https://www.titan.co.in/collection-cocktail.html links this exact product and describes the range as for cocktail parties, date nights, weddings and festive evenings.',product_url='https://www.titan.co.in/product/titan-raga-cocktails-quartz-analog-rose-gold-dial-stainless-steel-strap-watch-for-women-'+code.lower()+'.html',image_url=f'https://www.titan.co.in/dw/image/v2/BKDD_PRD/on/demandware.static/-/Sites-titan-master-catalog/default/{imagehash}/images/Titan/Catalog/{code}_1.jpg?sh=600&sw=600',source_family_id='titan:'+code[:5],notes='Collection/product evidence in titan_cocktail_web_evidence.json and titan_cocktail_product_web_evidence.json. Different design model numbers, no colour variants.',product_type='Watches')
for p in json.loads((OUT/'ohpolly_party.json').read_text())['products']:
 if 'black' not in p['title'].lower() or 'mini' not in p['title'].lower() or any(t in p['title'].lower() for t in ['embellish','sheer','lace','fur','cape']):continue
 if sum(r['product_type']=='Dresses' for r in rows)>=12:break
 row(source_dataset='Oh Polly AU Party Dresses collection',source_record_id=str(p['id']),product_name=p['title'],description=' '.join(html.unescape(re.sub('<[^>]+>',' ',p['body_html'])).split()),evidence_text='Exact product returned by official Party Dresses collection https://au.ohpolly.com/collections/party-dresses/products.json?limit=250',product_url='https://au.ohpolly.com/products/'+p['handle'],image_url=p['images'][0]['src']+'&width=800',source_family_id='ohpolly:'+p['handle'].removesuffix('-black'),notes='Simple black mini dress; party collection membership, not visual guess. On-model framing; visual review required.',product_type='Dresses')
def get(r):
 try:
  path=DEST/('supplemental_'+r['source_record_id']+'.jpg')
  if not path.exists():
   resp=requests.get(r['image_url'],timeout=25);resp.raise_for_status();path.write_bytes(resp.content)
  with Image.open(path) as im:im.load();r['width'],r['height']=im.size
  r['original_path']=str(path.relative_to(ROOT));r['file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();return r,None
 except Exception as e:return None,dict(source_record_id=r['source_record_id'],reason=str(e))
good=[];bad=[]
with concurrent.futures.ThreadPoolExecutor(6) as ex:
 for r,e in ex.map(get,rows):
  if r:good.append(r)
  else:bad.append(e)
with (OUT/'supplemental_candidates.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(good)
sheet=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(good):
 x=i%6*200;y=i//6*330
 with Image.open(resolve_task3_path(r['original_path'], root=ROOT)) as im:
  th=ImageOps.contain(im.convert('RGB'),(194,290));sheet.paste(th,(x+(194-th.width)//2,y))
 draw.text((x+2,y+291),f'{i:02} '+r['product_name'][:26],fill='black')
sheet.save(OUT/'supplemental_contact.jpg')
print({'downloaded':len(good),'failed':bad})
