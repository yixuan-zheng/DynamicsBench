"""Verify packet files; accept only declared UTF-8 CRLF-to-LF equivalence."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'CANDIDATE.json').read_text());equivalent=[]
for relative,expected in m['files'].items():
    path=root/relative
    if not path.is_file():raise SystemExit(f'Missing candidate file: {relative}')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()==expected:continue
    canonical=m.get('canonical_files',{}).get(relative,{})
    if canonical.get('canonicalization')=='utf8-crlf-to-lf':
        try:data.decode('utf-8')
        except UnicodeDecodeError:raise SystemExit(f'Content changed: {relative}')
        if hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==canonical['canonical_sha256']:
            equivalent.append(relative);continue
    raise SystemExit(f'Content changed: {relative}')
print(json.dumps({'passed':True,'files':len(m['files']),'line_ending_only':equivalent}))
