
from fashion.task3_paths import resolve_task3_path
import csv, hashlib, html, io, json, re, concurrent.futures
from pathlib import Path
import requests
from PIL import Image, ImageOps, ImageDraw

ROOT=Path.cwd()
OUT=ROOT/'reports/task3/rare_expansion_20260906/party'
DEST=ROOT/'data/external/rare_usage_expansion_20260906/candidates/party'
SOURCES=[('meshki_metadata.json','https://www.meshki.com.au',60),('www.beginningboutique.com.au.json','https://www.beginningboutique.com.au',85),('www.reddress.com.json','https://www.reddress.com',20)]
PAT=re.compile(r'\b(party|cocktail|evening|after.dark|night out)\b',re.I)

def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

jobs=[];rejected=[]
for filename,base,limit in SOURCES:
    seen=set(); count=0
    for p in json.loads((OUT/filename).read_text())['products']:
        desc=html.unescape(re.sub('<[^>]+>',' ',p['body_html']));desc=' '.join(desc.split())
        hit=PAT.search(desc)
        tags=p['tags'];tags=tags if isinstance(tags,list) else tags.split(', ')
        ev=[t for t in tags if t.lower() in ['cocktail dresses','party dresses','evening-dresses','collection-cocktaildresses','party-dresses']]
        family=next((t for t in tags if t.startswith('colourway-')),p['title'].split(' - ')[0].split()[0].lower())
        if family in seen or 'dress' not in p['title'].lower() or (not hit and not ev):continue
        if count>=limit:continue
        seen.add(family);count+=1
        excerpt=desc[max(0,hit.start()-65):hit.end()+100] if hit else p['title']
        jobs.append(dict(source_dataset=base.split('//')[1]+' public catalogue',source_record_id=str(p['id']),product_name=p['title'],description=excerpt,proposed_usage='Party',evidence_text=('Source collection: Party Dresses; '+ ('product tags: '+', '.join(ev) if ev else 'product description: '+excerpt)),evidence_basis='explicit_source_label' if ev else 'product_text_inference',product_url=base+'/products/'+p['handle'],image_url=p['images'][0]['src']+('&' if '?' in p['images'][0]['src'] else '?')+'width=720',original_path='',file_sha256='',width='',height='',source_family_id=base.split('//')[1]+':'+family,confidence='high' if hit or ev else 'medium',notes='Public retailer catalogue. Product photos remain retailer/photographer copyright; no open photo license established. On-model images may differ from teacher framing.',metadata_path=str((OUT/filename).relative_to(ROOT)),collection_url=base+'/collections/party-dresses',visual_review='pending'))

def acquire(r):
    path=DEST/(r['source_record_id']+'.jpg')
    try:
        if not path.exists():
            response=requests.get(r['image_url'],timeout=(10,30));response.raise_for_status();b=response.content
            if len(b)>8_000_000:raise ValueError('8MB image cap')
            with Image.open(io.BytesIO(b)) as im:im.load()
            path.write_bytes(b)
        b=path.read_bytes()
        with Image.open(path) as im:im.load();r['width'],r['height']=im.size
        r['original_path']=str(path.relative_to(ROOT));r['file_sha256']=hashlib.sha256(b).hexdigest()
        return r,None
    except Exception as e:return None,dict(source_record_id=r['source_record_id'],product_url=r['product_url'],reason=str(e))

rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for r,e in ex.map(acquire,jobs):
        if r:rows.append(r)
        else:rejected.append(e)
write('candidates.csv',rows)
write('rejected.csv',rejected or [dict(source_record_id='',product_url='',reason='No acquisition rejections')])
for start in range(0,len(rows),30):
    sheet=Image.new('RGB',(1200,1300),'white');d=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows[start:start+30]):
        x=(i%6)*200;y=(i//6)*260
        with Image.open(resolve_task3_path(r['original_path'], root=ROOT)) as im:
            thumb=ImageOps.contain(im.convert('RGB'),(190,225));sheet.paste(thumb,(x+(190-thumb.width)//2,y))
        d.text((x+2,y+226),f"{start+i:03} {r['product_name'][:25]}",fill='black')
    sheet.save(OUT/f'contact_{start//30+1:02}.jpg')
print(json.dumps(dict(downloaded=len(rows),distinct_families=len({r['source_family_id'] for r in rows}),rejected=len(rejected))))
