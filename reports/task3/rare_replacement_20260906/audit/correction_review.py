"""Run after build_audit.py; all changes remain audit-only."""
from pathlib import Path
import pandas as pd
O=Path(__file__).parent
s=pd.read_csv('data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv',dtype=str,keep_default_na=False)
t=s[(s.source_dataset=='teacher')&(s.partition=='development')&s.usage.isin(['Home','Party','Smart Casual','Travel'])].copy()
e=pd.read_csv(O/'external_687_audit.csv',dtype=str,keep_default_na=False)
tiny=t.groupby(['usage','articleType']).id.transform('size')<=4
special={
 '2633':('Mens black closed leather shoe','CLEAR ARTICLE TYPE DISAGREEMENT','Name explicitly says Men Black Leather Casual Shoes; image is closed mens shoe. Heels label is noisy.','No new high heels; existing Smart shoe pool covers visual kind.'),
 '12348':('Folded light blue collared shirt','CLEAR IMAGE ARTICLE TYPE DISAGREEMENT','Handbags label conflicts with visible folded collared shirt; name is test dispName.','Exclude from handbag sourcing rationale; do not source shirt carriers or relabel teacher.'),
 '35828':('Small black camera pouch','ARTICLE TYPE TOO BROAD FOR SOURCING','Name explicitly says Digital Camera Pouch; image agrees; Handbags is broad.','Source small travel camera pouches; do not count as ordinary handbag demand.'),
 '39487':('Small grey shoulder camera bag','NAME ARTICLE TYPE DISAGREEMENT','Name explicitly says Digital Series Camera Bag; image has camera-bag form; Mobile Pouch is misleading.','Source small travel camera bags/pouches rather than phone-only pouches.'),
 '45340':('Silver strappy heeled sandal','COMPATIBLE OVERLAPPING TYPES','Name says Sandals; image shows elevated sole/heel; Heels can cover this type.','Party sandals with visible heel remain a valid visual kind; no correction to label.'),
 '40826':('Multicolor printed cushion cover set','CONSISTENT; RESOLUTION LIMIT','Name says Cushion Covers Set of 2; image shows a group/set on light background. Folding/filling cannot be established.','Source real printed cover-set photos; no claim that folded/unfilled form is required.'),
}
visual={'Handbags':'Structured outdoor handbag/tote','Rucksacks':'Large hiking rucksack','Duffel Bag':'Soft carry-handle duffel bag','Casual Shoes':'Mens closed casual leather shoe','Trousers':'Modelled trousers','Clutches':'Metallic rectangular clutch','Sandals':'Brown flat strappy sandal','Wallets':'Rectangular leather wallet','Perfume and Body Mist':'Round perfume bottle','Tops':'Modelled black top','Watches':'Wristwatch','Ties':'Grey necktie'}
rows=[]
for _,r in t[tiny].iterrows():
 v,flag,why,rec=special.get(r.id,(visual.get(r.articleType,r.articleType),'NO CLEAR DISAGREEMENT','Name and class contact-sheet image support broad article type; photo alone cannot prove Usage.','Use supported visual kind with independent source Usage evidence.'))
 rows.append(dict(id=r.id,usage=r.usage,teacher_articleType=r.articleType,productDisplayName=r.productDisplayName,path=r.path,cv_fold=r.cv_fold,tiny_group_size=int(((t.usage==r.usage)&(t.articleType==r.articleType)).sum()),audit_visual_kind=v,audit_disagreement=flag,audit_evidence=why,audit_sourcing_recommendation=rec,teacher_label_action='PRESERVE ALL ORIGINAL LABELS'))
pd.DataFrame(rows).to_csv(O/'teacher_tiny_type_review.csv',index=False)
g=pd.read_csv(O/'rare_type_gap_table.csv',dtype=str,keep_default_na=False)
g['audit_teacher_label_warning']='';g['audit_visual_sourcing_kind']=g.teacher_article_type;g['audit_corrected_sourcing_action']='Use title and image jointly; do not derive external Usage from product type.'
updates={('Smart Casual','Heels'):('ID2633 mens Casual Shoes title/image disagree with Heels','Mens closed leather shoes','DO NOT SOURCE HIGH HEELS; no missing visual-type gap','NO TRUE HEELS GAP; TEACHER TYPE NOISE'),('Travel','Handbags'):('ID12348 is a folded shirt; ID35828 is camera pouch','2 outdoor handbags plus1 camera pouch;1 anomalous shirt excluded','Source outdoor handbags and camera pouches, not shirt-carrier products','GAP REDEFINED USING IMAGES AND NAMES'),('Travel','Mobile Pouch'):('ID39487 named and pictured as camera bag','Small shoulder camera bag','Source travel camera bags/pouches, not phone-only cases','GAP REDEFINED AS CAMERA BAG'),('Party','Heels'):('ID45340 named Sandals; pictured as heeled sandal; compatible overlap','Silver strappy heeled sandals','Heeled sandals supported; title boundary differs','COVERED BROADLY; TITLE BOUNDARY OVERLAP'),('Home','Cushion Covers'):('ID40826 Set of2; folding/filling unknown at60x80','Printed cushion cover set on light background','Source genuine printed cover-set images','PRESENTATION GAP; NOT PROVEN FOLDED-FABRIC GAP')}
for (u,a),(warn,kind,act,status) in updates.items():
 mask=(g.usage==u)&(g.teacher_article_type==a);g.loc[mask,['audit_teacher_label_warning','audit_visual_sourcing_kind','audit_corrected_sourcing_action','gap_status']]=[warn,kind,act,status]
 g.loc[mask,'notes']=warn+'. '+act+'. All teacher labels preserved.'
g.to_csv(O/'rare_type_gap_table.csv',index=False)

chosen=[]
def select(u,ids,reason,priority):
 for id in ids:
  row=e[(e.usage==u)&(e.id==str(id))].iloc[0].to_dict();row.update(shortlist_priority=priority,shortlist_reason=reason);chosen.append(row)
party=e[(e.usage=='Party')&e.audit_product_type.isin(['Bras','Kurtas','Lehenga','Sarees','Caps','Suits','Blazers','Tshirts','Other shoes','Flats'])]
select('Party',party.id,'Teacher-unsupported visual categories; keep old tops/heels and all dress/source variety. Exact retailer Party tag is preserved as evidence, not treated as teacher agreement.',1)
select('Party',['1000000054','1000000055','1000000056','1000000061'],'Remove two singleton shirt families plus one repeated two-image printed-shirt family; retain12 other shirts for boundary diversity.',2)
smart=e[(e.usage=='Smart Casual')&(e.extension_family_group=='external_family_9ad485e18ea86c290d4f')]
select('Smart Casual',smart.id,'Retire whole49-image conservative family: inferred semi-formal shoes and total fold0 concentration. Keep38 other broad shoes from both sources.',1)
select('Smart Casual',e[e.extension_family_group=='external_family_fe279541a59dd78a60c6'].id,'Retire whole four-image knit-polo/Tshirt family absent from teacher types; preserve other polo families.',2)
select('Smart Casual',['1000000249','1000000258','1000000267','1000000281','1000000287','1000000288','1000000289'],'Seven independent outerwear families absent from teacher types; keep remaining six jacket/coat examples and all shirts/trousers.',3)
select('Travel',e[(e.usage=='Travel')&(e.audit_product_type=='Wheeled luggage / hybrid bags')].id,'Wheeled/hybrid luggage differs from all valid teacher Travel forms; retain ordinary backpacks, duffels, outdoor bags and pouches.',1)
select('Travel',e[e.extension_family_group=='external_family_f6c600c27eaaec42b766'].id,'Retire whole five-image Ramverk family to reduce Db repetition while preserving Db source and other backpack families.',2)
select('Travel',['1000000128','1000000152'],'Generic Starter laptop/shoe-pocket or AmazonBasics small backpack titles: travel label depends on broad prose; retain wider backpack source diversity.',3)
home=[1000000322+n-1 for n in [4,12,17,19,25,26,55,88,102,113]]
select('Home',home,'Visual presentation shortlist: pale or narrow single cushions and room scene; whole two-image pale Pinzon family included. Preserve most explicit covers, printed groups and both sources. Not a declaration of bad labels.',2)
z=pd.DataFrame(chosen);z['shortlist_whole_family']=z.extension_family_group.map(z.groupby('extension_family_group').id.size()).astype(int)==z.audit_family_size.astype(int)
assert z.id.nunique()==len(z)==128
assert z.groupby('usage').size().to_dict()=={'Home':10,'Party':38,'Smart Casual':60,'Travel':20}
assert z.shortlist_whole_family.all()
z.to_csv(O/'retirement_shortlist_128.csv',index=False)
for u,a in z.groupby('usage'):a.to_csv(O/('retirement_shortlist_'+u.lower().replace(' ','_')+'.csv'),index=False)
remaining=e[~e.id.isin(z.id)];summary=[]
for u,grp in e.groupby('usage'):
 r=remaining[remaining.usage==u];cut=z[z.usage==u]
 summary.append(dict(usage=u,before=len(grp),proposed_retire=len(cut),remaining=len(r),before_sources=grp.audit_source_normalized.nunique(),remaining_sources=r.audit_source_normalized.nunique(),before_families=grp.extension_family_group.nunique(),remaining_families=r.extension_family_group.nunique(),partial_retired_families=0,remaining_fold_counts=str(r.cv_fold.value_counts().sort_index().to_dict())))
pd.DataFrame(summary).to_csv(O/'retirement_shortlist_summary.csv',index=False)
remaining.groupby(['usage','audit_source_normalized']).size().rename('remaining').reset_index().to_csv(O/'retirement_remaining_source_counts.csv',index=False)
remaining.groupby(['usage','audit_product_type']).size().rename('remaining').reset_index().to_csv(O/'retirement_remaining_type_counts.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False));print('Teacher tiny rows reviewed:',len(rows))
