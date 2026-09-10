"""Separate women's candidates; does not change the selected 62-row manifest."""
import csv,json,concurrent.futures as cf,re
from PIL import Image,ImageDraw
import acquire as a

ps=json.loads((a.OUT/'edmonds.json').read_text())['products']
for i in [84,85,116]:a.add(ps[i],'https://www.edmondsjewellers.com','Watches','edmonds.json')
ps=json.loads((a.OUT/'dayt.json').read_text())['products']
text=a.plain((a.OUT/'dayt_collection.html').read_text())
ev='Product belongs to DA:YT dress-watch collection. Collection definition: '+re.search(r'A dress watch is a clean.*?smart-casual wear.*?strap\.',text).group()
for i in [14,15,27,36,68]:
    a.add(ps[i],'https://dayt.ie','Watches','dayt.json',evidence=ev)
    a.ROWS[-1]['notes']+='; Independent dress-watch collection evidence in dayt_collection.html, not a reassignment based on any failed Party label. Women-specific title or product metadata.'
ps=json.loads((a.OUT/'aaiko_blouses.json').read_text())['products']
for i in [0,1,2,4,5,7]:
    a.add(ps[i],'https://aaiko.com','Shirts','aaiko_blouses.json',evidence='Official blouse collection page title: Blouses: From Business Wear to Smart Casual – AAIKO',family='aaiko:'+ps[i]['title'].split()[0].lower())
    a.ROWS[-1]['notes']+='; Weaker collection-title evidence (aaiko_blouses_collection.html), not product-specific occasion text. Exact member of https://aaiko.com/collections/blouses/products.json?limit=250 . Women collection; garment is a blouse.'
ps=json.loads((a.OUT/'jerusalem_women.json').read_text())['products']
for i in [1,13]:
    a.add(ps[i],'https://jerusalemsandals.com','Sandals','jerusalem_women.json',evidence="Member of Women's Casual Sandals collection. Collection text: Jerusalem Sandals' casual leather sandals for women work well for business casual meetings, smart casual dinners, weekend brunches, and even travel days.",family='jerusalem:'+ps[i]['title'].split(' - ')[0].lower())
    a.ROWS[-1]['notes']+='; Collection text cache jerusalem_women_collection.html; medium-confidence collection inference, multiple uses stated.'
p=json.loads((a.OUT/'rocia_women.json').read_text())['product']
a.add(p,'https://www.rociashoes.in','Sandals','rocia_women.json',evidence='Product title: Women Tan Smart Casual Flat Sandals. Product description: Stay stylish and comfortable with our Rocia By Regal Women Smart Casual Flats.')
a.ROWS[-1]['evidence_basis']='explicit_source_label'
for r in a.ROWS:r['gender_audit']='Women'
with cf.ThreadPoolExecutor(max_workers=6) as ex:rows=[r for r in ex.map(a.download,a.ROWS) if r]
with (a.OUT/'women_supplemental_candidates.csv').open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
sheet=Image.new('RGB',(1440,1020),'white');d=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
    im=Image.open(a.ROOT/r['original_path']).convert('RGB');im.thumbnail((235,290));x=i%6*240;y=i//6*340;sheet.paste(im,(x+(240-im.width)//2,y));d.text((x+3,y+292),str(i)+' '+r['product_type'],fill='black');d.text((x+3,y+308),r['product_name'][:31],fill='black')
sheet.save(a.OUT/'women_supplemental_contact.jpg');print('downloaded',len(rows),'errors',a.ERR)
