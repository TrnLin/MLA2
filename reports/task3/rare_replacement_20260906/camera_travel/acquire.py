from pathlib import Path
import json,re,hashlib,io,concurrent.futures
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image,ImageDraw
O=Path(__file__).parent
D=Path('data/external/rare_usage_replacement_20260906/candidates/camera_travel')
jobs=[]
shop=[('www.thinktankphoto.com.json','Think Tank','www.thinktankphoto.com','retrospective-4-v2-0',20),('www.wandrd.com.json','WANDRD','www.wandrd.com','rogue-6l-sling',1),('billingham.co.uk_catalog.json','Billingham','billingham.co.uk','s4-camera-bag',0),('billingham.co.uk_catalog.json','Billingham','billingham.co.uk','mini-eventer-camera-tablet-bag',0),('tiffen.com_catalog.json','Domke','tiffen.com','domke-dri-safe-all-weather-case',0)]
for file,brand,host,handle,i in shop:
 p=next(x for x in json.loads((O/file).read_text())['products'] if x['handle']==handle)
 desc=BeautifulSoup(p['body_html'],'html.parser').get_text(' ',strip=True)
 sentences=re.split(r'(?<=[.!?])\s+',desc)
 evidence=' '.join(q for q in sentences if re.search(r'\btravel\w*|\btrip\b',q,re.I))
 cache=O/(handle+'.json');cache.write_text(json.dumps(p,indent=2))
 jobs.append(dict(brand=brand,source_record_id=str(p['id']),product_name=p['title'],description=desc,evidence_text=evidence,product_url='https://'+host+'/products/'+handle,image_url=p['images'][i]['src'],source_family_id=brand+':'+handle,source_evidence_path=str(cache)))
extra=next(x for x in json.loads((O/'billingham.co.uk_catalog.json').read_text())['products'] if x['handle']=='225-mkii-camera-tablet-bag')
cache=O/'225-mkii-camera-tablet-bag.json';cache.write_text(json.dumps(extra,indent=2));desc=BeautifulSoup(extra['body_html'],'html.parser').get_text(' ',strip=True)
jobs.append(dict(brand='Billingham',source_record_id=str(extra['id']),product_name=extra['title'],description=desc,evidence_text=' '.join(x for x in re.split(r'(?<=[.!?])\s+',desc) if re.search(r'\btravel\w*',x,re.I)),product_url='https://billingham.co.uk/products/'+extra['handle'],image_url=extra['images'][0]['src'],source_family_id='Billingham:'+extra['handle'],source_evidence_path=str(cache)))
htmls=[('Lowepro','LP37248-PWW','Truckee SH120 LX','https://www.lowepro.com/us-en/truckee-sh-120-lx-lp37248-pww/','Camera shoulder bag with everyday style for travel and adventure'),('Lowepro','LP37124-PWW','Nova 180 AW II','https://www.lowepro.com/uk-en/nova-180-aw-ii-mica-and-pixel-camo-lp37124-pww/','Ready to Travel'),('Lowepro','LP36864-0WW','Adventura SH120 II','https://www.lowepro.com/global/adventura-sh-120-ii-lp36864-0ww/','Protective and compact, this camera bag is perfect for a day trip or travel'),('Tenba','638-570','DNA 9 Slim Messenger Bag Black','https://tenba.com/dna-9-messenger-bag-black/','adventure travel'),('Manfrotto','MB-UC-M-7L','UNCOVER Messenger Sling7L','https://www.manfrotto.com/global-uk/uncover-messenger-sling-7l-mb-uc-m-7l/','when commuting or traveling')]
for brand,id,title,url,needle in htmls:
 cache=O/(url.strip('/').split('/')[-1]+'.html');b=BeautifulSoup(cache.read_text(),'html.parser')
 for tag in b(['script','style','nav','header','footer']):tag.decompose()
 text=b.get_text(' ',strip=True);pos=text.lower().find(needle.lower());assert pos>=0,(title,needle)
 desc=text[max(0,pos-600):pos+900];evidence=text[max(0,pos-120):pos+len(needle)+150]
 image=b.select_one('meta[property="og:image"]')['content']
 if brand=='Manfrotto':image='https://www.manfrotto.com/media/catalog/product/b/a/bags-manfrotto-uncover-mb-uc-m-7l-front.png'
 jobs.append(dict(brand=brand,source_record_id=id,product_name=title,description=desc,evidence_text=evidence,product_url=url,image_url=image,source_family_id=brand+':'+id,source_evidence_path=str(cache)))
def get(j):
 try:
  r=requests.get(j['image_url'],timeout=40);r.raise_for_status();im=Image.open(io.BytesIO(r.content));im.load();ext='.png' if im.format=='PNG' else '.jpg';path=D/(hashlib.sha256(j['product_url'].encode()).hexdigest()[:20]+ext);path.write_bytes(r.content)
  j.update(source_dataset=j.pop('brand')+' official catalogue',proposed_usage='Travel',evidence_basis='product_text_inference',confidence='medium',notes='Product-specific manufacturer prose supports travel use; not an official teacher Usage label. One image per design. Pending central duplicate/family checks.',rights_status='copyright_permission_not_established',product_type='Camera Bags',width=im.width,height=im.height,original_path=str(path),file_sha256=hashlib.sha256(r.content).hexdigest(),visual_review='pending')
  return j
 except Exception as ex:return {'error':str(ex),'product_url':j['product_url']}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:results=list(ex.map(get,jobs))
rows=[r for r in results if 'error' not in r];errors=[r for r in results if 'error' in r]
pd.DataFrame(rows).to_csv(O/'candidates.csv',index=False);(O/'download_errors.json').write_text(json.dumps(errors,indent=2))
sheet=Image.new('RGB',(1500,((len(rows)+4)//5)*340),'white');d=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
 im=Image.open(r['original_path']).convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);im=bg.convert('RGB');im.thumbnail((285,270));x=(i%5)*300;y=(i//5)*340;sheet.paste(im,(x+(300-im.width)//2,y));d.text((x+5,y+276),f'{i+1}. '+r['source_dataset'][:33],fill='black');d.text((x+5,y+296),r['product_name'][:39],fill='black')
sheet.save(O/'contact_sheet.jpg');print('Downloaded',len(rows),'errors',errors)
