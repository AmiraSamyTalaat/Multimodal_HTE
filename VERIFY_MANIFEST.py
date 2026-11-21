#!/usr/bin/env python3
"""Check SHA-256 digests of every repository file listed in MANIFEST_SHA256.csv."""
from pathlib import Path
import csv,hashlib,sys
base=Path(__file__).resolve().parent
bad=[]
with (base/'MANIFEST_SHA256.csv').open(newline='',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        p=base/row['relative_path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            bad.append(row['relative_path'])
if bad:
    print('FAIL',*bad,sep='\n');sys.exit(1)
print('PASS: all manifest-listed files matched SHA-256')
