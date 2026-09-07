"""Use visually inspected native packshots, leaving selected_candidates unchanged."""
import csv,hashlib,json
from PIL import Image,ImageDraw
import acquire as a
rows=list(csv.DictReader((a.OUT/'women_supplemental_candidates.csv').open()))
alts=json.loads((a.OUT/'women_alternatives.json').read_text())
for r in rows:
    if r['product_type']=='Shirts':alt=next(x for x in alts if r['source_record_id'].endswith('_'+x['sid']) and x['j']==3)
    elif r['source_record_id'].startswith('www_rociashoes'):alt=next(x for x in alts if r['source_record_id'].endswith('_'+x['sid']) and x['j']==1)
    else:continue
    r['image_url']=alt['url'];r['original_path']=alt['path'];r['file_sha256']=hashlib.sha256((a.ROOT/alt['path']).read_bytes()).hexdigest()
    with Image.open(a.ROOT/alt['path']) as im:r.update(width=im.width,height=im.height)
    r['notes']+='; Replaced on-model/staged hero with complete native product packshot from same product image list; no image synthesis or background editing.'
with (a.OUT/'women_supplemental_candidates.csv').open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
qa=[];sheet=Image.new('RGB',(1440,1020),'white');d=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
    im=Image.open(a.ROOT/r['original_path']).convert('RGB');im.thumbnail((235,290));x=i%6*240;y=i//6*340;sheet.paste(im,(x+(240-im.width)//2,y));d.text((x+3,y+292),str(i)+' '+r['product_type'],fill='black');d.text((x+3,y+308),r['product_name'][:31],fill='black')
    qa.append(dict(source_record_id=r['source_record_id'],contact_index=i,decision='accept',visual_qa='Complete item on plain light background. Womens item; no folded garment, body occlusion, added accessory or cropped product.',evidence_review='Explicit product title' if r['evidence_basis']=='explicit_source_label' else ('Product-specific smart casual text' if 'edmonds' in r['source_dataset'] else 'Medium-confidence collection inference, documented in manifest'),notes='For central review/replacement only; not appended to selected62'))
sheet.save(a.OUT/'women_supplemental_contact.jpg')
with (a.OUT/'women_supplemental_qa.csv').open('w') as f:w=csv.DictWriter(f,list(qa[0]));w.writeheader();w.writerows(qa)
assert len({r['file_sha256'] for r in rows})==len(rows)
(a.OUT/'WOMEN_SUPPLEMENT.md').write_text('''# Women Smart Casual supplement

17 separate candidates: eight watches, six blouses, three brown/tan sandals. Existing selected_candidates.csv is untouched. All native complete packshots are inspected in women_supplemental_contact.jpg; exploration heroes/alternatives remain for audit.

Evidence is independent of previous Party selection decisions. Three Sekonda watches have product-specific smart-casual descriptions. Five DA:YT watches are members of its dress-watch collection, whose source definition explicitly includes smart-casual wear (medium-confidence collection inference). Six AAIKO blouses are exact members of its blouse collection with the page title "Blouses: From Business Wear to Smart Casual" (weaker title-level collection inference, not product-specific labels). Two brown Jerusalem women's sandals have collection copy explicitly naming business-casual meetings and smart-casual dinners. Rocia tan sandals carry Smart Casual in their product title and description. No women's heels or old teacher Casio watches were added.

Source caches: edmonds.json, dayt.json, dayt_collection.html, aaiko_blouses.json, aaiko_blouses_collection.html, jerusalem_women.json, jerusalem_women_collection.html, rocia_women.json. Selected image URLs match the corresponding product image arrays. Native AAIKO packshots replace on-model heroes; the Rocia native white-background pair replaces a staged hero. No generated/edited imagery.

Run central identity and image duplicate checks before admission. Some DA:YT watches may occur among previously rejected Party candidates: that is not an admitted overlap, and their Smart Casual evidence comes from the separately cached dress-watch collection. Retailer copyright remains; no open reuse permission established.
''')
print('17 independently reviewed women candidates saved; selected62 unchanged')
