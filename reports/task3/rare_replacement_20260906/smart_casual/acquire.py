
from fashion.task3_paths import resolve_task3_path
import concurrent.futures as cf
import csv, hashlib, io, json, re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

ROOT=Path.cwd()
OUT=ROOT/'reports/task3/rare_replacement_20260906/smart_casual'
IMG=ROOT/'data/external/rare_usage_replacement_20260906/candidates/smart_casual'
ROWS=[]; ERR=[]
PAT=re.compile(r'smart[\s‑–-]?casual|semi[\s-]?formal|business[\s-]?casual',re.I)
def plain(s):return BeautifulSoup(s or '', 'html.parser').get_text(' ',strip=True)
def fetch(url,name):
    path=OUT/name
    if path.exists():return path.read_text()
    r=requests.get(url,timeout=45);r.raise_for_status();path.write_text(r.text);return r.text
def add(p,base,kind,cache,evidence=None,family=None):
    desc=plain(p.get('body_html',''))
    if evidence is None:
        chunks=re.split(r'(?<=[.!?])\s+|\n',desc)
        evidence=' | '.join(s for s in chunks if PAT.search(s))
    if not evidence or not p.get('images'):return
    sid=re.sub(r'[^a-z0-9]+','_',base.split('//')[1].split('/')[0])+'_'+str(p['id'])
    ROWS.append(dict(source_dataset=base.split('//')[1]+' public product catalogue',source_record_id=sid,product_name=p['title'],description=desc,proposed_usage='Smart Casual',evidence_text=evidence,evidence_basis='product_text_inference',product_url=base+'/products/'+p['handle'],image_url=p['images'][0]['src'],source_family_id=family or base.split('//')[1]+':'+p['handle'],confidence='medium',notes='Retailer explicitly mentions smart casual; mapping to teacher usage remains an inference. Evidence cache: '+str((OUT/cache).relative_to(ROOT)),rights_status='copyright_permission_not_established',product_type=kind))
def individual(args):
    base,handle,kind=args
    try:
        cache=re.sub(r'[^a-z0-9]+','_',base.split('//')[1]+'_'+handle)+'.json'
        p=json.loads(fetch(base+'/products/'+handle+'.json',cache))['product'];add(p,base,kind,cache)
    except Exception as e:ERR.append(dict(source_record_id=handle,reason=str(e)))
def download(r):
    try:
        path=IMG/(r['source_record_id']+'.jpg')
        if not path.exists():
            res=requests.get(r['image_url'],timeout=50);res.raise_for_status();im=Image.open(io.BytesIO(res.content));im.load();path.write_bytes(res.content)
        with Image.open(path) as im:r.update(width=im.width,height=im.height)
        r.update(original_path=str(path.relative_to(ROOT)),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest());return r
    except Exception as e:ERR.append(dict(source_record_id=r['source_record_id'],reason=str(e)))
def main():
    # Product-specific evidence is preferable to collection-level descriptions.
    ps=json.loads((OUT/'edmonds.json').read_text())['products'];n=0
    for p in ps:
        if PAT.search(p['body_html']) and 'Gents' in p['title'] and not re.search('Sports|Rubber',p['title']):
            add(p,'https://www.edmondsjewellers.com','Watches','edmonds.json');n+=1
            if n==20:break
    ps=json.loads((OUT/'dayt.json').read_text())['products']
    full=plain((OUT/'dayt_collection.html').read_text())
    evidence='Product belongs to DA:YT dress-watch collection. Collection definition: '+re.search(r'A dress watch is a clean.*?smart-casual wear.*?strap\.',full).group()
    for i in [0,2,6,8,12,13,20,29,33,37,39,70,74,77,78,194,196,201,204,215]:
        add(ps[i],'https://dayt.ie','Watches','dayt.json',evidence=evidence)
        ROWS[-1]['notes']+='; Collection text cache: reports/task3/rare_replacement_20260906/smart_casual/dayt_collection.html; weaker collection-level occasion evidence.'
    ps=json.loads((OUT/'timex.json').read_text())['products']
    for i in [5,6,19,23,28]:
        add(ps[i],'https://shop.timexindia.com','Watches','timex.json',evidence='Marlin collection describes leather-strapped pieces: A leather-strapped piece like the Marlin Blue Round Dial Analog watch reads well with office and smart-casual wear, where the acrylic crystal and vintage dial styling suit a more considered wardrobe than a sports watch would.',family='timex:'+('marlin_chronograph' if i in [19,23,28] else str(ps[i]['id'])))
        ROWS[-1]['notes']+='; Collection evidence: reports/task3/rare_replacement_20260906/smart_casual/timex_collection.html. Weaker collection-level evidence; leather strap must be visually verified.'
    for file in ['rugged.json','rugged2.json']:
        ps=json.loads((OUT/file).read_text())['products']
        used=set()
        for p in ps:
            if PAT.search(p['body_html']) and 'Wallet' in p['title'] and not re.search('No. 8|Black/White',p['title']):add(p,'https://ruggedgentlemenshoppe.com','Wallets',file)
    specs=[
      ('https://www.leatherplus.in','men-wallet-et-2001-black','Wallets'),
      ('https://hedonist-style.com','handmade-mens-wallet-4-card-slots-tan-pull-up','Wallets'),
      ('https://primehideleather.co.uk','premium-oil-pull-up-leather-mens-flap-over-wallet-with-card-slots-and-coin-pocket','Wallets'),
      ('https://www.printvenue.com','kara-wallet-brown','Wallets'),
      ('https://eu.wardow.com','bellroy-l-pocket-wallet-wzla-blk-301','Wallets'),
      ('https://www.mjbale.com','darley-knitted-tie-navy-1','Ties'),
      ('https://www.barkersclothing.com','barkers-silk-knit-tie-bfa70409-olive','Ties'),
      ('https://www.tiesilk.com','teal-mens-skinny-tie','Ties'),
      ('https://roseborn.com','silk-grenadine-tie-black','Ties'),
      ('https://turnbullandasser.com','pink-fine-square-slub-silk-tie','Ties'),
      ('https://www.walkaroo.in','mens-daily-wear-sandals-wgp53452-olive','Sandals'),
      ('https://www.xposedlondon.com','mens-black-grey-handmade-real-leather-open-front-slide-sandals','Sandals'),
      ('https://www.regalshoes.in','regal-black-mens-smart-casual-leather-sandals-40079black','Sandals'),
      ('https://www.junaidjamshed.com','men-sandals-black-leather-sandals-ccmfwsandals05','Sandals')]
    with cf.ThreadPoolExecutor(max_workers=6) as ex:list(ex.map(individual,specs))
    for slug in ['montblanc-sartorial-wallet-6cc-black','montblanc-meisterstuck-wallet-4cc-black','montblanc-mini-sartorial-leather-4cc-wallet-black']:
        try:
            url='https://www.careofcarl.co.uk/en/'+slug;cache='careofcarl_'+slug+'.html';s=BeautifulSoup(fetch(url,cache),'html.parser')
            txt=s.get_text(' ',strip=True); details=txt[txt.find('Product details'):]; assert 'Sartorial Smart Casual' in details
            name=s.select_one('h1').get_text(' ',strip=True);iu=s.select_one('meta[property="og:image"]')['content'];sid='careofcarl_'+re.search(r'/(\d+)\.jpg',iu).group(1)
            ROWS.append(dict(source_dataset='Care of Carl public catalogue',source_record_id=sid,product_name='Montblanc '+name,description=details[:1200],proposed_usage='Smart Casual',evidence_text='Product style tags: Sartorial Smart Casual',evidence_basis='explicit_source_label',product_url=url,image_url=iu,source_family_id='montblanc:'+('sartorial_wallet' if 'sartorial' in slug else 'meisterstuck_wallet'),confidence='medium',notes='Product-specific style label. Source cache: '+str((OUT/cache).relative_to(ROOT)),rights_status='copyright_permission_not_established',product_type='Wallets'))
        except Exception as e:ERR.append(dict(source_record_id=slug,reason=str(e)))
    with cf.ThreadPoolExecutor(max_workers=8) as ex:rows=[r for r in ex.map(download,ROWS) if r]
    rows.sort(key=lambda r:(r['product_type'],r['source_record_id']))
    with (OUT/'candidates.csv').open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'rejected.csv').open('w') as f:w=csv.DictWriter(f,['source_record_id','reason']);w.writeheader();w.writerows(ERR)
    for start in range(0,len(rows),40):
        sheet=Image.new('RGB',(1200,1100),'white');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[start:start+40]):
            im=Image.open(resolve_task3_path(r['original_path'], root=ROOT)).convert('RGB');im.thumbnail((145,185));x=j%8*150;y=j//8*220;sheet.paste(im,(x+(150-im.width)//2,y));d.text((x+3,y+188),str(start+j)+' '+r['product_type'],fill='black');d.text((x+3,y+202),r['source_record_id'][-20:],fill='black')
        sheet.save(OUT/f'contact_{start//40+1}.jpg')
    print('rows',len(rows),'counts',{t:sum(r['product_type']==t for r in rows) for t in set(r['product_type'] for r in rows)},'errors',ERR)
if __name__=='__main__':main()
