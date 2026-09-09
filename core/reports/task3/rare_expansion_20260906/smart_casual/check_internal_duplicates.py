"""Check every unordered new-collection pair, without source-family shortcuts."""

from fashion.task3_paths import resolve_task3_path
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from fashion.data.external_usage_audit import _cropped_path, _normalized_path
from fashion.data.perceptual import compute_pair_pixel_metrics, meets_near_duplicate_rule

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = ROOT / 'reports/task3/rare_expansion_20260906'
CACHE = OUT / 'smart_casual/internal_duplicate_views'
PREFIXES = ('', 'view_', 'crop_')


def main():
    frame = pd.concat([pd.read_csv(OUT / name / 'audited.csv', dtype=str, keep_default_na=False)
                       for name in ('home', 'party', 'smart_casual', 'travel')], ignore_index=True)
    rows = frame.to_dict('records')
    assert len({r['external_id'] for r in rows}) == len(rows)
    jobs=[]
    for i,a in enumerate(rows):
        for j in range(i+1,len(rows)):
            b=rows[j]
            exact=[]
            if a['file_sha256']==b['file_sha256']:
                exact.append('same_file_sha256')
            for ak in ('pixel_sha256','view_pixel_sha256','crop_pixel_sha256'):
                for bk in ('pixel_sha256','view_pixel_sha256','crop_pixel_sha256'):
                    if a[ak] and a[ak]==b[bk]:
                        exact.append(ak+'='+bk)
            signals=[]
            if not exact:
                for ap in PREFIXES:
                    for bp in PREFIXES:
                        dh=(int(a[ap+'dhash_hex'],16)^int(b[bp+'dhash_hex'],16)).bit_count()
                        ah=(int(a[ap+'ahash_hex'],16)^int(b[bp+'ahash_hex'],16)).bit_count()
                        if dh<=2 and ah<=1:
                            signals.append((ap,bp,dh,ah))
            if exact or signals:
                jobs.append((i,j,exact,signals))
    paths={}
    required={(idx,prefix) for i,j,_,signals in jobs for ap,bp,_,_ in signals for idx,prefix in ((i,ap),(j,bp))}
    for idx,prefix in sorted(required):
        path=resolve_task3_path(rows[idx]['original_path'], root=ROOT)
        paths[(idx,prefix)]=(_normalized_path(path,CACHE) if prefix=='view_' else
                            _cropped_path(path,CACHE) if prefix=='crop_' else path)
    def check(job):
        i,j,exact,signals=job
        base={'first_id':rows[i]['external_id'],'second_id':rows[j]['external_id'],
              'first_usage':rows[i]['proposed_usage'],'second_usage':rows[j]['proposed_usage']}
        if exact:
            return {**base,'reason':'exact:'+','.join(exact)}
        for ap,bp,dh,ah in signals:
            metrics=compute_pair_pixel_metrics(paths[(i,ap)],paths[(j,bp)])
            if meets_near_duplicate_rule(SimpleNamespace(dhash_distance=dh,ahash_distance=ah,**metrics)):
                return {**base,'reason':'frozen_near_duplicate_rule',
                        'first_view':ap or 'original','second_view':bp or 'original',
                        'dhash_distance':dh,'ahash_distance':ah,**metrics}
        return None
    print(f'{len(rows)} images; {len(rows)*(len(rows)-1)//2} unique pairs; {len(jobs)} metric/exact candidates',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted=[r for r in pool.map(check,jobs) if r]
    pd.DataFrame(accepted,columns=['first_id','second_id','reason','first_usage','second_usage','first_view','second_view','dhash_distance','ahash_distance','mse','mae','max_difference','crop_mse','crop_mae','foreground_fraction_1','foreground_fraction_2','foreground_ratio']).to_csv(OUT/'internal_near_duplicate_matches.csv',index=False)
    summary={'images':len(rows),'unordered_pairs_checked':len(rows)*(len(rows)-1)//2,'prefiltered_pairs':len(jobs),'accepted_edges':len(accepted),'cross_class_edges':sum(r['first_usage']!=r['second_usage'] for r in accepted),'method':'Exact decoded/file hashes across all original/view/foreground combinations; dHash<=2 and aHash<=1 prefilter followed by frozen meets_near_duplicate_rule. Every source family checked; no family-equality skip. One row per accepted unordered pair.'}
    (OUT/'smart_casual/internal_duplicate_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
