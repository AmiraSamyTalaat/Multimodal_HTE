"""Verify keyed comparator ignores row order but detects a metric change."""
import hashlib
import importlib.util
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
COMPARATOR = BASE/'COMPARE_INDEPENDENT_RERUN.py'
ARCHIVE = BASE/'B01_Paper1_Complete_Reviewer_Code_Updated.zip'
TABLES = ('STAGE6A/ALL_STAGE6A_DEV_METRICS.csv',
          'STAGE6B/ALL_STAGE6B_TEST_METRICS.csv',
          'STAGE6C/ALL_STAGE6C_MECHANISM_DEV_METRICS.csv')


def main():
    spec = importlib.util.spec_from_file_location('b01_compare', COMPARATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []
    with zipfile.ZipFile(ARCHIVE) as original:
        for name in TABLES:
            rows = module.read_table(module.read_arc(original,name))
            order_pass, _ = module.compare_rows(rows,list(reversed(rows)),1e-3,1e-6)
            modified = [r.copy() for r in reversed(rows)]
            modified[0]['root_pehe'] = str(float(modified[0]['root_pehe'])+0.1)
            changed_pass, changed_info = module.compare_rows(rows,modified,1e-3,1e-6)
            extra_metric_pass = True
            if 'mean_support' in rows[0]:
                changed_extra = [r.copy() for r in reversed(rows)]
                target = next((r for r in changed_extra if module.isfloat(r['mean_support'])),None)
                if target is not None:
                    target['mean_support'] = str(float(target['mean_support'])+0.1)
                    extra_ok, extra_info = module.compare_rows(rows,changed_extra,1e-3,1e-6)
                    extra_metric_pass = (not extra_ok and 'error' not in extra_info and
                                         any(x['column']=='mean_support' for x in extra_info['mismatches']))
            checks.append({'table':name,'rows':len(rows),'reversed_rows_pass':order_pass,
                           'modified_metric_fails':not changed_pass and 'error' not in changed_info,
                           'blank_capable_numeric_metric_compared':extra_metric_pass,
                           'pass':order_pass and not changed_pass and 'error' not in changed_info and extra_metric_pass})
    report = {'utc':datetime.now(timezone.utc).isoformat(),
              'comparator_sha256':hashlib.sha256(COMPARATOR.read_bytes()).hexdigest(),
              'rtol':1e-3,'atol':1e-6,'checks':checks,
              'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL'}
    (BASE/'COMPARATOR_VALIDATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report['status'],len(checks))
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    main()
