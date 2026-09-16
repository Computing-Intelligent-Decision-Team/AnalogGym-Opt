# -*- coding: utf-8 -*-
"""Convert native MATLAB numeric exports into a traceable LLM-readable dataset."""
import argparse
import collections
import csv
import hashlib
import json
import math
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from dataset_docs import field_dictionary, write_docs, write_topology_summary

COLS = ['gen','index','pm','gbw','gain','noise','CMRR','PSRR','SR','ivdd_27','vos','tc','d_settle','settlingTime','FOMS','FOML','chip_area','fitness','FOM_AW']
META = ['record_id','source_id','source_row','topology','tech_node_label','vdd_label','vcm_label','cl_label','run_label']
TAIL = ['quality_flag_count','quality_flags']

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def flags_for(x, conflicts):
    result=[]
    for i,row in enumerate(x):
        flag=[]
        if not np.isfinite(row).all():flag.append('nonfinite')
        if row[8]<0:flag.append('sr_negative')
        if row[8]==-1:flag.append('sr_minus_one')
        if row[15]<0:flag.append('foml_negative')
        if row[12]==row[13]:flag.append('settling_fields_equal')
        if any(row[j]<=0 for j in [3,5,9,16]):flag.append('nonpositive_basic_metric')
        if row[2]<0 or row[2]>180:flag.append('phase_outside_0_180')
        if tuple(row[:2]) in conflicts:flag.append('key_conflict')
        result.append(flag)
    return result

def summary(x):
    ret={}
    for j,k in enumerate(COLS[2:],2):
        y=x[:,j]; y=y[np.isfinite(y)]
        ret[k]=dict(zip(['min','p05','median','p95','max'],map(float,np.quantile(y,[0,.05,.5,.95,1])))) if len(y) else None
    return ret

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1])
    args=ap.parse_args(); out=args.output; source=args.source
    for folder in ['data/by_topology','summaries','audit','examples','evidence']:(out/folder).mkdir(parents=True,exist_ok=True)
    entries=json.loads((out/'_work/export_manifest.json').read_text(encoding='utf-8'))
    dbfile=out/'data/performance.sqlite'
    if dbfile.exists():raise FileExistsError('Refusing to overwrite an existing database; use a fresh output folder.')
    db=sqlite3.connect(dbfile)
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('CREATE TABLE sources(source_id TEXT PRIMARY KEY, relative_path TEXT, source_sha256 TEXT, row_count INTEGER, metadata_json TEXT)')
    spec=['record_id TEXT PRIMARY KEY','source_id TEXT REFERENCES sources(source_id)','source_row INTEGER','topology TEXT','tech_node_label INTEGER','vdd_label REAL','vcm_label REAL','cl_label REAL','run_label TEXT']
    spec += ['"'+c+'" '+('INTEGER' if c in ['gen','index'] else 'REAL') for c in COLS]
    spec += ['quality_flag_count INTEGER','quality_flags TEXT']
    db.execute('CREATE TABLE performance('+','.join(spec)+')')
    db.execute('CREATE VIEW performance_unflagged AS SELECT * FROM performance WHERE quality_flag_count=0')
    all_flags=collections.Counter(); topo_data=collections.defaultdict(list); topo_good=collections.defaultdict(list)
    sources=[]; topic_handles={}; total=0; removed=0; unflagged=0; nf_count=0; samples=[]
    with (out/'data/performance.csv').open('w',encoding='utf-8',newline='') as cf, (out/'audit/duplicates.jsonl').open('w',encoding='utf-8') as dupf, (out/'audit/nonfinite_values.jsonl').open('w',encoding='utf-8') as nff:
        cw=csv.writer(cf); cw.writerow(META+COLS+TAIL)
        for n,e in enumerate(entries,1):
            assert e['columns']==COLS
            rel=e['source_relative_path']; parts=rel.split('/'); assert len(parts)==8
            topology,tech,vdd,vcm,cl,run=parts[:6]
            sid='src_'+hashlib.sha256(rel.encode()).hexdigest()[:16]
            original=source/rel; x=loadmat(out/'_work'/e['intermediate'])['values']
            assert x.shape==(e['rows'],len(COLS)); assert np.isfinite(x[:,:2]).all() and (x[:,:2]==np.floor(x[:,:2])).all()
            # Deduplicate only identical full rows in the SAME source; do not collapse performance-only matches.
            unique, first=np.unique(x,axis=0,return_index=True); keep=sorted(first.tolist())
            keycount=collections.Counter(map(tuple,unique[:,:2])); conflicts={k for k,c in keycount.items() if c>1}
            flags=flags_for(x,conflicts)
            removed_here=len(x)-len(keep)
            if removed_here:
                canonical={tuple(x[i]):i+1 for i in reversed(keep)}
                for i in sorted(set(range(len(x)))-set(keep)):
                    dupf.write(json.dumps({'source_id':sid,'removed_row':i+1,'kept_row':canonical[tuple(x[i])]},allow_nan=False)+'\n')
            removed+=removed_here
            srcmeta={'source_id':sid,'relative_path':rel,'source_sha256':sha(original),'rows_original':len(x),'rows_exported':len(keep),'exact_duplicates_removed':removed_here,'conflicting_keys':len(conflicts),'matlab_variable':e['variable'],'matlab_units':e['units'],'matlab_descriptions':e['descriptions'],'topology':topology,'folder_labels':{'tech_node':int(tech),'vdd':float(vdd),'vcm':float(vcm),'cl':float(cl),'run':run},'related_files':{k:rel.replace('Perf_pop_data.mat',k+'_pop_data.mat') for k in ['TD','TBM']},'quality_flags':dict(collections.Counter(f for i in keep for f in flags[i])),'unflagged_rows':sum(not flags[i] for i in keep)}
            sources.append(srcmeta)
            db.execute('INSERT INTO sources VALUES(?,?,?,?,?)',(sid,rel,srcmeta['source_sha256'],len(keep),json.dumps(srcmeta,ensure_ascii=False,allow_nan=False)))
            if topology not in topic_handles:topic_handles[topology]=(out/'data/by_topology'/f'{topology}.jsonl').open('w',encoding='utf-8')
            inserts=[]
            for i in keep:
                row=x[i]; vals=[int(v) if j<2 else float(v) if math.isfinite(v) else None for j,v in enumerate(row)]
                if any(v is None for v in vals):
                    nonfinite={COLS[j]:('NaN' if np.isnan(v) else '+Inf' if v>0 else '-Inf') for j,v in enumerate(row) if not np.isfinite(v)}
                    nff.write(json.dumps({'source_id':sid,'source_row':i+1,'original_nonfinite_values':nonfinite})+'\n'); nf_count+=len(nonfinite)
                rid=f'{sid}_r{i+1:06d}'
                meta=[rid,sid,i+1,topology,int(tech),float(vdd),float(vcm),float(cl),run]
                flat=meta+vals+[len(flags[i]),'|'.join(flags[i])]
                cw.writerow(flat); inserts.append(flat)
                rec={'record_id':rid,'topology':topology,'folder_labels':srcmeta['folder_labels'],'generation':vals[0],'individual_index':vals[1],'performance':dict(zip(COLS[2:],vals[2:])),'quality_flags':flags[i],'source':{'source_id':sid,'row_1based':i+1}}
                topic_handles[topology].write(json.dumps(rec,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
                if i in {0,len(x)//2,len(x)-1}:samples.append(rec)
                all_flags.update(flags[i]); unflagged+=not flags[i]; total+=1
            db.executemany('INSERT INTO performance VALUES('+','.join('?' for _ in META+COLS+TAIL)+')',inserts)
            db.commit()
            topo_data[topology].append(x[keep]); goodidx=[i for i in keep if not flags[i]]
            if goodidx:topo_good[topology].append(x[goodidx])
            print(f'{n}/{len(entries)} {topology}: {len(keep)} rows',flush=True)
    for h in topic_handles.values():h.close()
    db.execute('CREATE INDEX idx_context ON performance(topology,tech_node_label,vdd_label,vcm_label,cl_label)')
    db.execute('CREATE INDEX idx_source_key ON performance(source_id,gen,"index")')
    db.execute('CREATE INDEX idx_flags ON performance(quality_flag_count)')
    db.commit()
    integrity=db.execute('PRAGMA integrity_check').fetchone()[0]; assert integrity=='ok'
    assert db.execute('SELECT COUNT(*) FROM performance').fetchone()[0]==total
    db.close()
    metadata=field_dictionary()
    dictionary=metadata['fields']
    dump(out/'field_dictionary.json',metadata)
    source_root=str(source.resolve())
    dump(out/'source_manifest.json',{'source_root':source_root,'sources':sources})
    topo_summary=[]
    for topology in sorted(topo_data):
        x=np.concatenate(topo_data[topology]); sx=[s for s in sources if s['topology']==topology]
        good=np.concatenate(topo_good[topology]) if topo_good[topology] else np.empty((0,len(COLS)))
        desc={'topology':topology,'rows':len(x),'unflagged_rows':len(good),'runs':len(sx),'tech_node_labels':sorted({s['folder_labels']['tech_node'] for s in sx}),'jsonl':f'data/by_topology/{topology}.jsonl','metrics_all_records':summary(x),'metrics_unflagged':summary(good) if len(good) else {}}
        topo_summary.append(desc)
        write_topology_summary(out,desc,sx)
    dump(out/'topology_index.json',topo_summary)
    with (out/'examples/sample_records.jsonl').open('w',encoding='utf-8') as f:
        for s in samples:f.write(json.dumps(s,ensure_ascii=False,allow_nan=False)+'\n')
    quality={'created_at_utc':datetime.now(timezone.utc).isoformat(),'input_files':len(entries),'original_rows':sum(e['rows'] for e in entries),'exported_rows':total,'topologies':len(topo_data),'exact_duplicates_removed':removed,'nonfinite_cells':nf_count,'rows_unflagged':unflagged,'rows_flagged':total-unflagged,'flag_counts':dict(all_flags),'sqlite_integrity':integrity,'numeric_policy':'All finite numbers kept unchanged. No rounding, clipping, imputation, absolute value, unit conversion, or performance-only deduplication.','scope':'73 Perf_pop_data.mat tables only; paired TD/TBM tables are linked through manifest, not expanded.'}
    dump(out/'audit/quality_report.json',quality)
    write_docs(out,quality,topo_summary,dictionary)
    # Bundle reference snippets with English annotations; do not execute them.
    evidence=Path(__file__).resolve().parents[1]/'evidence'
    for name in ['extract_pop_data.m','filterByFOM.m','computeFOM.m','database_select_range_of_data.m']:
        reference=evidence/name; target=out/'evidence'/name
        if reference.resolve()!=target.resolve():shutil.copyfile(reference,target)
    print(json.dumps(quality,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
