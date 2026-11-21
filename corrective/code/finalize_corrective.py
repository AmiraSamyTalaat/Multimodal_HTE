from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from corrective_common import COR,REV,DATA,now,sha,write

def read(p):return pd.read_csv(p) if p.exists() else pd.DataFrame()
def main():
    sources=[('clinical_adjustment','results/clinical_adjustment/per_seed_results.csv','results/clinical_adjustment/summary_results.csv'),
             ('nonoracle_tuning','results/nonoracle_tuning/per_seed_results.csv','results/nonoracle_tuning/summary_results.csv'),
             ('fixed_dgp_splits','results/fixed_dgp_splits/dgp_cluster_results.csv','results/fixed_dgp_splits/summary_results.csv'),
             ('fair_neural','results/fair_neural/per_seed_results.csv','results/fair_neural/summary_results.csv')]
    per=[];summary=[]
    for analysis,a,b in sources:
        d=read(COR/a);s=read(COR/b)
        if not d.empty:
            d['analysis']=analysis;d['source_file']=a;d['source_file_sha256']=sha(COR/a);per.append(d)
        if not s.empty:
            if analysis=='fair_neural' and 'mean' in s:s=s.rename(columns={'mean':'mean_root_pehe'})
            s['analysis']=analysis;s['source_file']=b;s['source_file_sha256']=sha(COR/b);summary.append(s)
    if per:pd.concat(per,ignore_index=True,sort=False).to_csv(COR/'per_seed_results.csv',index=False)
    if summary:pd.concat(summary,ignore_index=True,sort=False).to_csv(COR/'summary_results.csv',index=False)
    manifests=[]
    for folder in [COR/'results/clinical_adjustment',COR/'results/nonoracle_tuning',
                   COR/'results/fixed_dgp_splits',COR/'kaggle_neural_downloads']:
        if not folder.exists():continue
        for p in folder.rglob('COMPLETE.json'):
            if 'smoke' in p.parts:continue
            try:obj=json.loads(p.read_text(encoding='utf-8'))
            except Exception:continue
            code_hash=obj.get('code_sha256')
            if not code_hash and 'kaggle_neural_downloads' in p.parts:
                batch=next((x for x in p.parts if x.startswith('batch_')),None)
                if batch:
                    _,start,end=batch.split('_')
                    bm=COR/'kaggle_bundle'/f'b01-p1-corrective-neural-{start}-{end}'/'batch_manifest.json'
                    if bm.exists():code_hash=json.loads(bm.read_text(encoding='utf-8')).get('script_sha256')
            analysis_name='fair_neural' if 'kaggle_neural_downloads' in p.parts else p.relative_to(COR).parts[1]
            manifests.append(dict(analysis=analysis_name,manifest=str(p.relative_to(COR)).replace('\\','/'),
              manifest_sha256=sha(p),dgp_seed=obj.get('dgp_seed'),effect_regime=obj.get('effect_regime'),
              split_seed=obj.get('split_seed'),finished_utc=obj.get('finished_utc'),
              source_data_sha256=obj.get('data_sha256'),source_code_sha256=code_hash))
    pd.DataFrame(manifests).to_csv(COR/'experiment_manifest.csv',index=False)
    checks=json.loads((COR/'reporting/reporting_checks.json').read_text(encoding='utf-8'))
    status={}
    for analysis,_,_ in sources:
        p=COR/'results'/analysis/'RUN_STATUS.json'
        status[analysis]=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'status':'NOT_STARTED'}
    complete=(checks['status']=='PASS' and all(v.get('status')=='COMPLETE' for v in status.values()))
    write(COR/'RUN_STATUS.json',dict(status='COMPLETE' if complete else 'PARTIAL',updated_utc=now(),
      reporting_checks=checks['status'],experiments=status,source_data_sha256=sha(DATA),
      original_results_unchanged=True,zenodo_published=False,
      independence='DGP seed; fixed-DGP resplits and learner seeds are repeated measurements'))
    print('COMPLETE' if complete else 'PARTIAL',len(manifests))
if __name__=='__main__':main()
