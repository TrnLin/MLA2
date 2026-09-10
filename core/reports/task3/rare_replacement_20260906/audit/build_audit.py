from pathlib import Path
import re
import pandas as pd
from PIL import Image, ImageDraw

OUT=Path(__file__).parent
s=pd.read_csv('data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv',dtype=str,keep_default_na=False)
t=s[(s.source_dataset=='teacher') & (s.partition=='development')].copy()
e=s[s.source_dataset!='teacher'].copy()
rare=['Home','Party','Smart Casual','Travel']
t[t.usage!=''].groupby(['usage','articleType']).size().rename('count').reset_index().to_csv(OUT/'teacher_type_counts.csv',index=False)
t[t.usage!=''].groupby(['usage','cv_fold']).size().rename('count').reset_index().to_csv(OUT/'teacher_fold_counts.csv',index=False)
t[t.usage!=''].groupby('usage').agg(count=('id','size'),types=('articleType','nunique'),families=('product_family_group','nunique'),folds=('cv_fold','nunique')).to_csv(OUT/'teacher_nine_class_summary.csv')
def typ(r):
 n=r.productDisplayName.lower();u=r.usage
 if u=='Home':return 'Cushion covers (title explicit)' if any(k in n for k in ['cover','case','sham','shell']) else 'Decorative pillows (cover status unspecified in title)'
 if u=='Party':
  for pattern,name in [(r'\bbra\b','Bras'),('dress','Dresses'),('t-shirt','Tshirts'),('shirt','Shirts'),('top','Tops'),('heels|wedges|heel ','Heels'),('flats','Flats'),('kurta','Kurtas'),('lehenga','Lehenga'),('sari','Sarees'),('blazer','Blazers'),('cap','Caps'),('suit','Suits')]:
   if re.search(pattern,n):return name
  return 'Other shoes'
 if u=='Smart Casual':
  for pattern,name in [('belt','Belts'),('boot','Boots'),('mule','Mules'),('loafer|shoe|formal slip','Shoes (formal/casual ambiguous)'),('overshirt','Overshirts'),('polo|t-shirt','Tshirts / polos'),('shirt','Shirts'),('trouser|chino','Trousers'),('jeans','Jeans'),('jumper','Sweaters'),('jacket|blazer|trench| mac ','Jackets / coats')]:
   if re.search(pattern,n):return name
  return 'Unresolved'
 if re.search('rolling|roller|wheeled',n):return 'Wheeled luggage / hybrid bags'
 if re.search('duffel|duffle',n):return 'Duffel Bag'
 if re.search('rucksack|internal frame|inner frame|70l|55l',n):return 'Rucksacks / hiking packs'
 if re.search('shoulder|tote|weekender',n) and 'backpack' not in n:return 'Handbags / totes / weekenders'
 return 'Backpacks / travel packs'
e['audit_product_type']=e.apply(typ,axis=1)
e['audit_annotation_basis']='Title rules plus class contact-sheet review; broad groups only, not target labels'
e['audit_family_size']=e.groupby('extension_family_group').id.transform('size')
e['audit_source_normalized']=e.source_dataset.str.lower().replace({'amazon berkeley objects':'amazon_berkeley_objects'})
e['audit_evidence_level']=e.label_strength
e['audit_photo_style']='Isolated catalogue product; inspected class contact sheets'
e.loc[(e.usage=='Party')&(e.source_dataset!='flipkart_party'),'audit_photo_style']='Modelled dress; studio/lifestyle backgrounds; class sheet inspected'
e.loc[e.source_dataset=='flipkart_party','audit_photo_style']='Mixed product-only and modelled catalogue; old-intake sheets inspected'
e['audit_retirement_score']=0;e['audit_retirement_reason']='No specific retirement priority'
for i,r in e.iterrows():
 score=0;reasons=[]
 if r.usage=='Party' and r.audit_product_type in ['Bras','Kurtas','Lehenga','Sarees','Caps','Suits','Blazers','Tshirts','Other shoes','Flats','Shirts']:
  score+=70;reasons.append('Product type absent from teacher Party development; source tag may not match teacher taxonomy')
 if r.usage=='Smart Casual' and r.audit_product_type in ['Belts','Boots','Mules','Overshirts','Tshirts / polos','Jeans','Sweaters','Jackets / coats']:
  score+=45;reasons.append('Product type absent from teacher Smart Casual development')
 if r.usage=='Smart Casual' and r.label_strength=='product_text_inference':
  score+=25;reasons.append('Formal/semi-formal product text does not establish exact Smart Casual label')
 if r.usage=='Travel' and r.audit_product_type=='Wheeled luggage / hybrid bags':
  score+=40;reasons.append('Wheeled/hybrid shape absent from teacher Travel development')
 if r.audit_family_size>=5:
  score+=min(35,r.audit_family_size);reasons.append('Large conservative family: reduce repeated views/variants without splitting retained family')
 if r.usage=='Party' and r.source_dataset!='flipkart_party':
  score+=15;reasons.append('New intake is all modelled dresses; reduce style/source concentration only after replacement evidence passes')
 if r.usage=='Home' and r.audit_product_type=='Decorative pillows (cover status unspecified in title)':
  score+=20;reasons.append('Teacher Home is one cushion cover; title does not establish cover-only product')
 e.loc[i,'audit_retirement_score']=score;e.loc[i,'audit_retirement_reason']='; '.join(reasons) or 'No specific retirement priority'
cols=['id','usage','cv_fold','path','source_dataset','source_id','external_id','productDisplayName','source_url','label_evidence','label_strength','extension_family_group']+[c for c in e if c.startswith('audit_')]
e[cols].to_csv(OUT/'external_687_audit.csv',index=False)
e[cols].sort_values(['usage','audit_retirement_score','audit_family_size','id'],ascending=[True,False,False,True]).to_csv(OUT/'retirement_ranked.csv',index=False)
for u in rare:
 rank=e[e.usage==u][cols].sort_values(['audit_retirement_score','audit_family_size','id'],ascending=[False,False,True]).reset_index(drop=True)
 rank.insert(0,'retirement_review_rank',range(1,len(rank)+1))
 rank.to_csv(OUT/('retirement_'+u.lower().replace(' ','_')+'.csv'),index=False)
mapping={'Cushion Covers':['Cushion covers (title explicit)'],'Dresses':['Dresses'],'Clutches':[],'Perfume and Body Mist':[],'Tops':['Tops'],'Heels':['Heels'],'Watches':[],'Shirts':['Shirts'],'Formal Shoes':[],'Casual Shoes':[],'Wallets':[],'Trousers':['Trousers'],'Sandals':[],'Ties':[],'Backpacks':['Backpacks / travel packs'],'Rucksacks':['Rucksacks / hiking packs'],'Duffel Bag':['Duffel Bag'],'Handbags':['Handbags / totes / weekenders'],'Mobile Pouch':[]}
gaps=[]
for (u,art),tg in t[t.usage.isin(rare)].groupby(['usage','articleType']):
 eg=e[(e.usage==u)&(e.audit_product_type.isin(mapping[art]))];amb=e.iloc[0:0];note='External counts are title-inferred audit groups, not training articleType labels.'
 if art in ['Formal Shoes','Casual Shoes']:
  amb=e[(e.usage==u)&(e.audit_product_type=='Shoes (formal/casual ambiguous)')];note='Same 87 shoes appear as possible matches for BOTH shoe labels; cannot allocate exact teacher type from title/style alone. Do not sum ambiguous counts.'
 if art=='Cushion Covers':amb=e[(e.usage==u)&(e.audit_product_type.str.startswith('Decorative pillows'))];note='30 explicit cover titles;106 pillow titles need description-level cover/filling check. Almost all are photographed inflated.'
 if art=='Handbags':note='Seven broad shoulder/tote/weekender title matches; not seven exact teacher-style handbags. Teacher label includes small shoulder and shirt-carrier forms.'
 if art=='Heels' and u=='Party':note='22 heel/wedge titles, some visually low/flat sandals; source article-type vocabulary is not teacher ground truth.'
 if art=='Heels' and u=='Smart Casual':note='Teacher ID2633 labelled Heels visually resembles closed black shoe; retain teacher label. No external exact Heels annotation.'
 gaps.append(dict(usage=u,teacher_article_type=art,teacher_rows=len(tg),teacher_families=tg.product_family_group.nunique(),teacher_folds=','.join(sorted(tg.cv_fold.unique())),external_rows_title_group=len(eg),external_families=eg.extension_family_group.nunique(),external_source_count=eg.audit_source_normalized.nunique(),external_sources='; '.join(sorted(eg.audit_source_normalized.unique())),external_folds=','.join(sorted(eg.cv_fold.unique())),external_ambiguous_possible_rows=len(amb),external_ambiguous_families=amb.extension_family_group.nunique(),external_ambiguous_sources='; '.join(sorted(amb.audit_source_normalized.unique())),gap_status='MISSING' if len(eg)==0 and len(amb)==0 else 'UNCERTAIN TYPE ALLOCATION' if len(eg)==0 else 'COVERED IN BROAD TITLE GROUP; PHOTO/EVIDENCE LIMITS REMAIN',notes=note))
pd.DataFrame(gaps).to_csv(OUT/'rare_type_gap_table.csv',index=False)
for group,name in [(['usage','audit_product_type'],'external_type_counts'),(['usage','audit_source_normalized'],'external_source_counts'),(['usage','cv_fold'],'external_fold_counts'),(['usage','extension_family_group','cv_fold'],'external_family_counts'),(['usage','audit_product_type','cv_fold'],'external_type_fold_counts')]:
 e.groupby(group).size().rename('count').reset_index().to_csv(OUT/(name+'.csv'),index=False)
for u in rare:
 x=t[t.usage==u];w=900;cellw=150;cellh=160;im=Image.new('RGB',(w,40+((len(x)+5)//6)*cellh),'white');d=ImageDraw.Draw(im);d.text((10,10),f'Teacher DEVELOPMENT only: {u}',fill='black')
 for k,(_,r) in enumerate(x.iterrows()):
  pic=Image.open(r.path).convert('RGB');pic.thumbnail((105,120));xx=(k%6)*cellw;yy=40+(k//6)*cellh;im.paste(pic,(xx+(cellw-pic.width)//2,yy));d.text((xx+4,yy+121),f'{r.id} fold{r.cv_fold}',fill='black');d.text((xx+4,yy+136),r.articleType[:23],fill='black')
 im.save(OUT/(u.lower().replace(' ','_')+'_teacher_development.png'))
t[t.usage!=''].groupby(['articleType','usage']).size().rename('count').reset_index().to_csv(OUT/'teacher_cross_class_type_counts.csv',index=False)
x=e[e.id.astype(int)<1000000120];im=Image.new('RGB',(1000,40+((len(x)+9)//10)*125),'white');d=ImageDraw.Draw(im)
for k,(_,r) in enumerate(x.iterrows()):
 pic=Image.open(r.path).convert('RGB');pic.thumbnail((75,85));xx=(k%10)*100;yy=40+(k//10)*125;im.paste(pic,(xx,yy));d.text((xx,yy+86),r.id[-4:]+' '+r.usage[:5],fill='black');d.text((xx,yy+101),r.audit_product_type[:16],fill='black')
im.save(OUT/'original_120_contact_sheet.png')
print(e.groupby(['usage','audit_product_type']).size().to_string())
