from pathlib import Path
import csv, hashlib,sys
root=Path(__file__).resolve().parent
with (root/'MANIFEST_SHA256.csv').open(encoding='utf-8',newline='') as f:
    rows=list(csv.DictReader(f))
failed=[]
for rec in rows:
    p=root/rec['relative_path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256'] or p.stat().st_size!=int(rec['bytes']):
        failed.append(rec['relative_path'])
print('SHA-256 verified:',len(rows)-len(failed),'/',len(rows))
if failed:
    print('Mismatch:',*failed,sep='\n')
    sys.exit(1)
