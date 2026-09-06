"""Apply the explicit visual review of contact sheets 1 and 2."""

from fashion.task3_paths import resolve_task3_path
import csv, collections
from pathlib import Path
ROOT=Path.cwd(); OUT=ROOT/'reports/task3/rare_replacement_20260906/smart_casual'
rows=list(csv.DictReader((OUT/'candidates.csv').open()))
rejected={
 'www_barkersclothing_com_10048539459905':'Tie is cropped; full item outline unavailable in hero.',
 'ruggedgentlemenshoppe_com_4334205010010':'Hero contains several wallet colours rather than one item.',
 'www_printvenue_com_15682067300433':'Hero is a staged multi-view scene with box and several wallet views; poor teacher match.',
 'dayt_ie_14835132989763':'Diver-style watch with broad rotating bezel; weak match to teacher Smart Casual despite collection membership.',
 'shop_timexindia_com_10156315771169':'Metal bracelet; collection smart-casual sentence supports leather-strapped examples only.',
 'shop_timexindia_com_8693378679073':'Same Marlin chronograph design as retained silver model; colour-variant reduction.',
 'shop_timexindia_com_8693378711841':'Same Marlin chronograph design as retained silver model; colour-variant reduction.',
 'www_edmondsjewellers_com_9793971880275':'Same case/dial design as retained silver-dial rose-gold model; colour-variant reduction.',
 'www_edmondsjewellers_com_9794459566419':'Same numeral dial design as retained silver white-dial model; colour-variant reduction.',
 'www_edmondsjewellers_com_9794459631955':'Same numeral dial design as retained silver white-dial model; colour-variant reduction.',
 'www_edmondsjewellers_com_9794459992403':'Same high-contrast dual-time dial family as retained model; design repetition reduction.',
 'www_edmondsjewellers_com_9794460025171':'Same grey multifunction dial as retained grey-strap model; design repetition reduction.',
 'www_edmondsjewellers_com_9794461368659':'Same grey dual-time dial as retained leather model; strap-variant reduction.',
}
groups={
 'sekonda:round_arabic_numerals':['9794459402579','9794459435347','9794459566419','9794459631955'],
 'sekonda:rose_gold_dual_time':['9793971749203','9793971880275'],
 'sekonda:grey_multifunction':['9794459861331','9794460025171'],
 'sekonda:grey_dual_time':['9794461237587','9794461368659'],
 'armani_exchange:cayde':['15144563573059','8862900814147'],
}
qa=[];selected=[]
for i,r in enumerate(rows):
    for family,ids in groups.items():
        if any(r['source_record_id'].endswith('_'+pid) for pid in ids):r['source_family_id']=family
    reject=rejected.get(r['source_record_id'])
    quality='Single isolated product on plain light background; outline and details clear.'
    if r['product_type']=='Ties':quality='Single arranged tie; recognizable shape and fabric visible.'
    if r['source_record_id']=='www_leatherplus_in_8743533052042':quality='Single closed wallet, mild indoor background; clear outline, medium teacher similarity.'
    qa.append(dict(source_record_id=r['source_record_id'],contact_sheet=f'contact_{i//40+1}.jpg',contact_index=i,decision='reject' if reject else 'accept',visual_qa=reject or quality,label_qa='Collection-level inference; not an exact teacher label' if r['source_dataset'].startswith(('dayt.ie','shop.timexindia.com')) else 'Retailer product text/style label explicitly supports smart casual',reviewer='visual_review_20260906'))
    if not reject:selected.append(r)
for name,rs in [('candidates.csv',rows),('selected_candidates.csv',selected),('visual_qa.csv',qa)]:
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
counts=collections.Counter(r['product_type'] for r in selected)
assert all((resolve_task3_path(r['original_path'], root=ROOT)).is_file() for r in rows)
assert len({r['file_sha256'] for r in rows})==len(rows)
(OUT/'README.md').write_text(f'''# Smart Casual replacement candidate review

Downloaded {len(rows)} products. Visually reviewed both contact sheets. Selected {len(selected)} candidates; rejected {len(rejected)}.

Selected counts: {dict(counts)}.

Use `selected_candidates.csv` for the parent duplicate/fold audit. `candidates.csv` keeps all downloads; `visual_qa.csv` records each decision. Source HTML and Shopify product JSON remain beside these CSV files. Source URLs, evidence text, source cache paths, file hashes and image dimensions are retained per row.

Evidence strength: Care of Carl wallets have product-specific Smart Casual style badges. Sekonda, other wallets, ties and sandals have product-specific smart-casual descriptions. DA:YT watches have weaker collection membership evidence: its dress-watch collection is explicitly defined for formal, business and smart-casual wear. Two selected Timex watches use the Marlin collection's leather-strap smart-casual description; only actual leather-strap images pass. These collection inferences remain medium confidence and should be identified in the report, not claimed as original teacher labels.

All photos are retailer catalogue assets. No open reuse rights were established. These are candidate assets for local academic evaluation, not a redistributable open dataset. No teacher/external duplicate audit, fold assignment, dataset integration or training was done here. Related watch designs share conservative family identifiers where identified; the parent should still run image similarity checks across all data.

Rejected paths are kept for audit only. The collection quota was not forced: weak source text or poor images were not relabelled to fill it. Titan blocked direct requests with HTTP 403; failed request HTML is retained. A few initially found product descriptions did not retain the matching text in their live JSON, so they produced no candidate row.
''')
print(len(selected),dict(counts))
