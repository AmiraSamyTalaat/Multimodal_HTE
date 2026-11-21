from pathlib import Path
import json, hashlib, compileall, sys
ROOT=Path(__file__).resolve().parents[1]
MAN=ROOT/'REPRODUCE/PACKAGE_MANIFEST_SHA256.json'
m=json.loads(MAN.read_text())
missing=[]; bad=[]
for rel,h in m.items():
 p=ROOT/rel
 if not p.exists(): missing.append(rel); continue
 hh=hashlib.sha256(p.read_bytes()).hexdigest()
 if hh!=h: bad.append(rel)
print('manifest entries',len(m),'missing',len(missing),'hash mismatches',len(bad))
if missing: print('MISSING',missing[:20])
if bad: print('HASH MISMATCH',bad[:20])
ok=compileall.compile_dir(str(ROOT/'01_CODE'),quiet=1)
print('python compile', 'PASS' if ok else 'FAIL')
required=['06_RESULTS/METHOD_SUMMARY_STAGE4.csv','06_RESULTS/STAGE5A/METHOD_STRESS_SUMMARY.csv','06_RESULTS/STAGE6B/STAGE6B_CONFIRMATORY_DECISION.json','06_RESULTS/STAGE6C/STAGE6C_MECHANISM_DECISION.json','06_RESULTS/STAGE7B/STAGE7B_DEVELOPMENT_SELECTION.json']
req=all((ROOT/x).exists() for x in required); print('required results','PASS' if req else 'FAIL')
if missing or bad or not ok or not req: raise SystemExit(2)
print('FINAL: PASS')
