from __future__ import annotations
import argparse, json
from pathlib import Path

REV=Path(__file__).resolve().parents[1]
TEMPLATE=REV/'postreview_code/kaggle_bagged_fairness_template.py'
def prepare(name,dgp_seeds,net_seeds,mode,max_epochs,patience):
    folder=REV/'kaggle_bundle'/name
    folder.mkdir(parents=True,exist_ok=True)
    config={'batch_name':name,'mode':mode,'dgp_seeds':dgp_seeds,'net_seeds':net_seeds,
            'max_epochs':max_epochs,'patience':patience}
    script=TEMPLATE.read_text(encoding='utf-8').replace('__BATCH_CONFIG__',repr(config))
    (folder/'b01_p1_bagged_fairness.py').write_text(script,encoding='utf-8')
    meta={'id':f'prinprin/{name}','title':name.replace('-',' '),
          'code_file':'b01_p1_bagged_fairness.py','language':'python','kernel_type':'script',
          'is_private':True,'enable_gpu':True,'enable_internet':False,
          'dataset_sources':['prinprin/b01-causaltx-mm-core-runtime'],
          'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return folder
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--dgp-seeds',nargs='+',type=int,required=True)
    ap.add_argument('--net-seeds',nargs='+',type=int,required=True)
    ap.add_argument('--mode',choices=['smoke','full'],required=True)
    a=ap.parse_args()
    folder=prepare(a.name,a.dgp_seeds,a.net_seeds,a.mode,2 if a.mode=='smoke' else 250,1 if a.mode=='smoke' else 30)
    print(folder)
if __name__=='__main__':main()
