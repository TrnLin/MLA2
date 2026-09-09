
from fashion.task3_paths import resolve_task3_path
import concurrent.futures as cf
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

ROOT = Path.cwd()
OUT = ROOT / 'reports/task3/rare_expansion_20260906/smart_casual'
IMG = ROOT / 'data/external/rare_usage_expansion_20260906/candidates/smart_casual'
BASE = 'https://www.marksandspencer.com/l/men/mens-smart-casual'
LOG = []

def get_page(n):
    p = OUT / f'ms_page_{n}.html'
    if not p.exists():
        r = requests.get(BASE + (f'?page={n}' if n > 1 else ''), timeout=60)
        r.raise_for_status()
        p.write_text(r.text)
    data = json.loads(BeautifulSoup(p.read_text(), 'html.parser').find(id='__NEXT_DATA__').text)
    results = data['props']['pageProps']['serverSideGqlResponseFed']['productPageData']['search']['results']
    return results['products']

def download(row):
    try:
        p = IMG / (row['source_record_id'] + '.jpg')
        if not p.exists():
            r = requests.get(row['image_url'], timeout=60)
            r.raise_for_status()
            Image.open(io.BytesIO(r.content)).load()
            p.write_bytes(r.content)
        with Image.open(p) as im:
            im.load()
            row.update(width=im.width, height=im.height)
        row.update(original_path=str(p.relative_to(ROOT)), file_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        return row
    except Exception as e:
        LOG.append({'source_record_id':row['source_record_id'], 'reason':str(e)})

def main():
    products = {}
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for batch in ex.map(get_page, range(1,9)):
            for p in batch:
                products[p['productExternalId']] = p
    (OUT / 'ms_native_products.json').write_text(json.dumps(list(products.values()), indent=2))
    rows=[]
    allowed = {'Shirts','Trousers','Shoes','Belts','Knitted Polo Shirts','Polo Shirts','Jumpers','Cardigans','Loafers','Jackets','Blazers','Suit Jacket','Suit Trousers','Coats','Jeans'}
    for sid,p in products.items():
        if p['productDefinition'] not in allowed or 'pack' in p['title'].lower():
            continue
        assets=p['variants'][0]['mediaAssets']
        asset=next((a for a in assets if a['type']=='Cut_Out'),None)
        if asset is None:
            continue
        rows.append(dict(source_dataset='Marks and Spencer public catalogue',source_record_id='ms_'+sid,product_name=p['brand']+' '+p['title'],description=p['description'] or p['title'],proposed_usage='Smart Casual',evidence_text='Product included in retailer collection: Men’s Smart-Casual Clothing. Product type: '+p['productDefinition'],evidence_basis='explicit_source_label',product_url='https://www.marksandspencer.com'+p['seoPath'],image_url='https://assets.digitalcontent.marksandspencer.app/image/upload/w_768,q_auto,f_jpg/'+asset['assetId'],source_family_id='ms:'+p['strokeId'],confidence='medium',notes='Retailer Smart Casual collection membership; native Cut_Out asset. Retailer copyright; no open reuse permission established. Collection URL: '+BASE, rights_status='copyright_permission_not_established'))
    print('M&S products',len(products),'eligible cutouts',len(rows),flush=True)
    # ABO wording is recorded verbatim. Exclude prior exact Smart Casual intake and weak office-only phrases.
    hits=json.loads((OUT/'hits.json').read_text())
    index={r['image_id']:r for r in csv.DictReader(gzip.open(ROOT/'reports/task3/rare_external_intake_20260906/sources/abo/raw/images.csv.gz','rt'))}
    used=set()
    for p in hits:
        ev=' | '.join(p['evidence'])
        iid=p.get('main_image_id')
        if not re.search(r'semi.formal|business.casual',ev,re.I) or iid in used or iid not in index:
            continue
        used.add(iid)
        val=lambda k:' | '.join(v.get('value','') for v in p.get(k,[]))
        rows.append(dict(source_dataset='Amazon Berkeley Objects',source_record_id='abo_'+p['item_id'],product_name=val('item_name'),description=val('bullet_point'),proposed_usage='Smart Casual',evidence_text=ev,evidence_basis='product_text_inference',product_url='https://'+p['domain_name']+'/dp/'+p['item_id'],image_url='https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/original/'+index[iid]['path'],source_family_id='abo:'+val('brand')+':'+val('model_number'),confidence='medium',notes='Source explicitly describes semi-formal or business casual usage; label inferred, not exact teacher label. Amazon Berkeley Objects CC BY 4.0.',rights_status='CC-BY-4.0'))
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        result=[r for r in ex.map(download,rows) if r]
    seen=set(); unique=[]
    for row in result:
        if row['file_sha256'] in seen:
            LOG.append({'source_record_id':row['source_record_id'],'reason':'exact SHA-256 duplicate'})
        else:
            seen.add(row['file_sha256']);unique.append(row)
    fields=list(unique[0])
    with (OUT/'candidates.csv').open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(unique)
    with (OUT/'rejected.csv').open('w') as f:
        w=csv.DictWriter(f,['source_record_id','reason']);w.writeheader();w.writerows(LOG)
    for start in range(0,len(unique),48):
        sheet=Image.new('RGB',(1200,1200),'white');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(unique[start:start+48]):
            im=Image.open(resolve_task3_path(row['original_path'], root=ROOT)).convert('RGB');im.thumbnail((145,170))
            x=(j%8)*150;y=(j//8)*200
            sheet.paste(im,(x+(150-im.width)//2,y));draw.text((x+2,y+172),str(start+j)+' '+row['source_record_id'],fill='black')
        sheet.save(OUT/f'contact_{start//48+1}.jpg')
    print('downloaded',len(result),'unique',len(unique),'failures',len(LOG),flush=True)

if __name__=='__main__':
    main()
