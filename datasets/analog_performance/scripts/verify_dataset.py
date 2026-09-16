# -*- coding: utf-8 -*-
"""Verify all three serializations against every value extracted by MATLAB."""
import argparse
import collections
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from scipy.io import loadmat

COLS=['gen','index','pm','gbw','gain','noise','CMRR','PSRR','SR','ivdd_27','vos','tc','d_settle','settlingTime','FOMS','FOML','chip_area','fitness','FOM_AW']

def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();out=a.output
    manifest=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
    exports=json.loads((out/'_work/export_manifest.json').read_text(encoding='utf-8'))
    bypath={e['source_relative_path']:e for e in exports}
    expected={}; expected_ids=set(); expected_flags=collections.Counter(); flag_rows=0
    for src in manifest['sources']:
        sid=src['source_id']; e=bypath[src['relative_path']]
        assert hash_file(Path(manifest['source_root'])/src['relative_path'])==src['source_sha256']
        x=loadmat(out/'_work'/e['intermediate'])['values']; expected[sid]=(x,src)
        assert np.isfinite(x).all(), 'This verification profile requires finite numeric inputs.'
        assert len(np.unique(x,axis=0))==len(x), 'Unexpected duplicates in this dataset.'
        assert len(np.unique(x[:,:2],axis=0))==len(x), 'Unexpected key conflict.'
        for i,row in enumerate(x):
            expected_ids.add(f'{sid}_r{i+1:06d}')
            checks={'sr_negative':row[8]<0,'sr_minus_one':row[8]==-1,'foml_negative':row[15]<0,'settling_fields_equal':row[12]==row[13],'nonpositive_basic_metric':np.any(row[[3,5,9,16]]<=0),'phase_outside_0_180':row[2]<0 or row[2]>180}
            expected_flags.update(k for k,v in checks.items() if v); flag_rows+=any(checks.values())
    total=sum(len(x) for x,s in expected.values())
    print('Source hashes and native arrays checked:',total,flush=True)
    db=sqlite3.connect((out/'data/performance.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert db.execute('PRAGMA foreign_key_check').fetchall()==[]
    for sid,(x,s) in expected.items():
        cols=','.join('"'+c+'"' for c in COLS)
        rows=db.execute(f'SELECT {cols} FROM performance WHERE source_id=? ORDER BY source_row',(sid,)).fetchall()
        assert np.array_equal(np.array(rows),x),sid
    assert db.execute('SELECT count(*) FROM performance').fetchone()[0]==total
    assert db.execute('SELECT count(*) FROM performance_unflagged').fetchone()[0]==total-flag_rows
    db.close(); print('SQLite all numeric values checked.',flush=True)
    csv_ids=set()
    with (out/'data/performance.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            x,s=expected[row['source_id']]; i=int(row['source_row'])-1
            assert np.array_equal(np.array([float(row[c]) for c in COLS]),x[i])
            assert row['topology']==s['topology']
            assert float(row['vdd_label'])==s['folder_labels']['vdd']
            assert float(row['vcm_label'])==s['folder_labels']['vcm']
            assert float(row['cl_label'])==s['folder_labels']['cl']
            assert int(row['tech_node_label'])==s['folder_labels']['tech_node']
            assert row['run_label']==s['folder_labels']['run']
            assert row['record_id']==f"{row['source_id']}_r{i+1:06d}"
            assert row['record_id'] not in csv_ids; csv_ids.add(row['record_id'])
    assert csv_ids==expected_ids;print('CSV all numeric values and IDs checked.',flush=True)
    json_ids=set(); json_flags=collections.Counter()
    for path in sorted((out/'data/by_topology').glob('*.jsonl')):
        with path.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
                sid=r['source']['source_id']; i=r['source']['row_1based']-1;x,s=expected[sid]
                assert set(r['performance'])==set(COLS[2:])
                assert np.array_equal(np.array([r['generation'],r['individual_index']]+[r['performance'][c] for c in COLS[2:]]),x[i])
                assert r['topology']==path.stem==s['topology'] and r['folder_labels']==s['folder_labels']
                assert r['record_id']==f'{sid}_r{i+1:06d}' and r['record_id'] not in json_ids
                json_ids.add(r['record_id']);json_flags.update(r['quality_flags'])
    assert json_ids==expected_ids and json_flags==expected_flags
    q=json.loads((out/'audit/quality_report.json').read_text(encoding='utf-8'))
    assert q['original_rows']==q['exported_rows']==total
    assert q['rows_flagged']==flag_rows and q['flag_counts']==dict(expected_flags)
    index=json.loads((out/'topology_index.json').read_text(encoding='utf-8'))
    assert sum(t['rows'] for t in index)==total and sum(t['unflagged_rows'] for t in index)==total-flag_rows
    result={'passed':True,'native_matlab_source_files':len(expected),'rows_per_format':total,'numeric_values_compared_per_format':total*len(COLS),'formats':['CSV','JSONL','SQLite'],'comparison':'Exact equality to double values exported by native MATLAB, no tolerance or rounding. All records and all 19 source fields checked.','source_sha256_all_match':True,'unique_record_ids':len(expected_ids),'json_strict_parsing':True,'sqlite_integrity':'ok','foreign_key_violations':0,'independent_quality_counts_match':True,'summary_counts_match':True}
    (out/'audit/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
