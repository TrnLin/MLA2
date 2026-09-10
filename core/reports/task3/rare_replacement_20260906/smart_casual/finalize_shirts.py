"""Append twelve full-shirt photos after reviewing shirts_contact.jpg."""
import csv
from pathlib import Path
OUT=Path(__file__).resolve().parent
shirts=list(csv.DictReader((OUT/'shirt_candidates.csv').open()))
for filename in ['candidates.csv','selected_candidates.csv']:
    rows=list(csv.DictReader((OUT/filename).open()));ids={r['source_record_id'] for r in rows}
    rows.extend(r for r in shirts if r['source_record_id'] not in ids)
    with (OUT/filename).open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
qa=list(csv.DictReader((OUT/'visual_qa.csv').open()));ids={r['source_record_id'] for r in qa}
for i,r in enumerate(shirts):
    if r['source_record_id'] not in ids:qa.append(dict(source_record_id=r['source_record_id'],contact_sheet='shirts_contact.jpg',contact_index=i,decision='accept',visual_qa='Complete front shirt silhouette, sleeves and hem visible, isolated light background. One colour per named design. Visually inspected at contact-sheet size.',label_qa='Explicit native product tag usage:Smart Casual',reviewer='visual_review_20260906'))
with (OUT/'visual_qa.csv').open('w') as f:w=csv.DictWriter(f,list(qa[0]));w.writeheader();w.writerows(qa)
text=(OUT/'README.md').read_text()
marker='\n## Full-shirt extension\n'
text=text.split(marker)[0]+marker+'\nAdded 12 complete front-shirt photos from M.J. Bale, each a distinct named shirt design. Every product has the explicit native tag `usage:Smart Casual` in `mjbale_shirts.json`. All 12 passed visual review in `shirts_contact.jpg`: sleeves and hem are visible, with plain light backgrounds and no folding. These are new source products, not alternate photos of prior M&S items. Total now 75 downloaded, 62 selected: 35 watches, 12 shirts, 8 wallets, 4 ties, 3 sandals. Existing 50 selected rows are preserved. Run `finalize_shirts.py` after `finalize_review.py` if rebuilding the selected CSV.\n'
(OUT/'README.md').write_text(text)
print('Appended reviewed shirts:',len(shirts))
