import csv,json,concurrent.futures as cf
from PIL import Image, ImageDraw
import acquire as a

ps=json.loads((a.OUT/'mjbale_shirts.json').read_text())['products']
for i in [13,55,63,69,74,84,96,99,110,112,115,146]:
    p=ps[i]
    assert 'usage:Smart Casual' in p['tags']
    a.add(p,'https://www.mjbale.com','Shirts','mjbale_shirts.json',evidence='Native retailer product tag: usage:Smart Casual',family='mjbale:'+p['title'].lower().replace(' ','_'))
    a.ROWS[-1]['evidence_basis']='explicit_source_label'
    a.ROWS[-1]['notes']='Retailer native product tag explicitly states usage:Smart Casual. Source cache: reports/task3/rare_replacement_20260906/smart_casual/mjbale_shirts.json. One colour per named design.'
with cf.ThreadPoolExecutor(max_workers=6) as ex:rows=[r for r in ex.map(a.download,a.ROWS) if r]
with (a.OUT/'shirt_candidates.csv').open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
sheet=Image.new('RGB',(1200,900),'white');d=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
    im=Image.open(a.ROOT/r['original_path']).convert('RGB');im.thumbnail((195,395));x=i%6*200;y=i//6*450;sheet.paste(im,(x+(200-im.width)//2,y));d.text((x+3,y+397),str(i)+' '+r['product_name'][:24],fill='black');d.text((x+3,y+412),r['source_record_id'][-15:],fill='black')
sheet.save(a.OUT/'shirts_contact.jpg');print(len(rows))
