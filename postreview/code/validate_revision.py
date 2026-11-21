from __future__ import annotations
import hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd

REV=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(condition,message):
    if not condition:raise AssertionError(message)
def main():
    source=pd.read_csv(REV/'source_provenance.csv')
    check(len(source)==309,'Source snapshot count changed')
    for r in source.itertuples():
        check(sha(r.original_full_path)==r.original_sha256,'Original source changed: '+r.relative_path)
        check(sha(r.snapshot_full_path)==r.copy_sha256,'Local snapshot changed: '+r.relative_path)
    data_hash=source[source.relative_path.str.replace('\\','/',regex=False).str.endswith(
        'ispy2_core_multimodal.npz')].original_sha256.iloc[0]
    rep=REV/'results/replication';markers=list((rep/'runs').glob('*/COMPLETE.json'))
    check(len(markers)==270,'Replication marker count is not 270')
    units=[]
    for p in markers:
        d=json.loads(p.read_text(encoding='utf-8'))
        check(d.get('status')=='COMPLETE','Incomplete marker '+str(p))
        check(d.get('data_sha256')==data_hash,'Replication data hash mismatch '+str(p))
        check((d.get('n_train'),d.get('n_dev'),d.get('n_test'))==(441,147,148),
              'Wrong split sizes '+str(p))
        m=pd.read_csv(p.parent/'metrics.csv')
        check(len(m)==8,'Wrong method count '+str(p))
        check(m.method.nunique()==8,'Duplicate method '+str(p))
        check(np.isfinite(m[['root_pehe','policy_regret']].to_numpy()).all(),
              'Nonfinite replication metric '+str(p))
        oracle=m.set_index('method').loc['oracle_test_mean_constant','root_pehe']
        zero=m.set_index('method').loc['zero_effect','root_pehe']
        check(oracle<=zero+1e-8,'Oracle constant worse than zero '+str(p))
        grid=pd.read_csv(p.parent/'ridge_dev_grid.csv')
        chosen=float(d['ridge_alpha_selected_on_DEV'])
        best=grid.sort_values(['dev_root_pehe','alpha']).iloc[0]
        check(np.isclose(chosen,float(best.alpha)),'Ridge DEV choice mismatch '+str(p))
        units.append((d['dgp_seed'],d['patient_split_seed'],d['effect_regime']))
    idx=pd.MultiIndex.from_tuples(units)
    check(idx.is_unique,'Duplicate DGP/split/regime unit')
    check(len(idx.get_level_values(0).unique())==30,'Not 30 DGP seeds')
    check(len(idx.get_level_values(1).unique())==3,'Not 3 patient splits')
    check(len(idx.get_level_values(2).unique())==3,'Not 3 effect regimes')
    check(len(pd.read_csv(rep/'per_seed_results.csv'))==2160,'Wrong replication row count')
    ci=pd.read_csv(rep/'uncertainty_intervals.csv')
    check(len(ci)==24 and ci.inference_ready.all(),'Replication intervals not ready')
    check((ci.ci95_lower<=ci.mean_root_pehe).all() and
          (ci.mean_root_pehe<=ci.ci95_upper).all(),'Invalid intervals')
    overlap=pd.read_csv(REV/'results/additional/clean_overlap_only_perturbation.csv')
    paired=overlap.pivot(index=['dgp_seed','split'],columns='condition',values='true_cate_sha256')
    check((paired.iloc[:,0]==paired.iloc[:,1]).all(),'Clean overlap changed CATE')
    fairness=pd.read_csv(REV/'results/fairness/per_run_results.csv')
    check(len(fairness[fairness.method=='bagged_early_fusion'])==15,'Bagged run count mismatch')
    ridge=pd.read_csv(REV/'results/ridge_extension/per_run_results.csv')
    check(len(ridge)==30,'Extended Ridge run count mismatch')
    packages={}
    for p in sorted((REV/'packages').glob('B01_P1_*POSTREVIEW.zip'))+[
        REV/'packages/B01_P1_REVISION_RESULTS_ALL.zip']:
        with zipfile.ZipFile(p) as z:
            check(z.testzip() is None,'Corrupt ZIP '+str(p))
            names=z.namelist()
            check(not any(n.endswith(('.npz','.npy','.pkl','.joblib')) or
                          '/02_DATA/' in n or '/predictions/' in n or
                          'patient_audit' in n for n in names),'Patient-level file in ZIP '+str(p))
            packages[p.name]={'sha256':sha(p),'entries':len(names),'bytes':p.stat().st_size}
    report={'status':'PASS','finished_utc':datetime.now(timezone.utc).isoformat(),
      'source_files_sha256_verified':len(source),'replication_units_verified':len(units),
      'replication_unique_dgp':30,'replication_patient_splits':3,'replication_effect_regimes':3,
      'replication_methods_per_unit':8,'replication_metric_rows':2160,
      'fairness_bagged_runs':15,'ridge_extension_rows':30,
      'clean_overlap_cate_identity':True,'packages':packages}
    (REV/'reports/validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
