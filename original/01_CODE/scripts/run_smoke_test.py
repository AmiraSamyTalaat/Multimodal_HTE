import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cm_osr.smoke import run
if __name__=='__main__':
    out=run(); print(json.dumps(out,indent=2));
    assert out['weights_sum_max_error'] < 1e-5
