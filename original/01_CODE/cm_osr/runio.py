from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
from .utils import write_json

def init_run(root,run_id,config):
    p=Path(root)/'04_RUNS'/run_id
    for d in ['logs','checkpoints','predictions','metrics','environment']: (p/d).mkdir(parents=True,exist_ok=True)
    manifest={'run_id':run_id,'created_unix':time.time(),'status':'RUNNING','config':config,
              'python':sys.version,'platform':platform.platform()}
    write_json(p/'RUN_MANIFEST.json',manifest); return p,manifest

def finish_run(run_dir,manifest,status='COMPLETE'):
    manifest['status']=status; manifest['finished_unix']=time.time(); write_json(Path(run_dir)/'RUN_MANIFEST.json',manifest)
