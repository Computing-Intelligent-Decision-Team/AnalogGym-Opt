# -*- coding: utf-8 -*-
"""Add an optional exact-profile view and produce a shareable archive."""
import argparse
import json
import sqlite3
import zipfile
from pathlib import Path

from dataset_docs import profile_notes, write_llm_guide

COLS=['pm','gbw','gain','noise','CMRR','PSRR','SR','ivdd_27','vos','tc','d_settle','settlingTime','FOMS','FOML','chip_area','fitness','FOM_AW']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();out=a.output
    db=sqlite3.connect(out/'data/performance.sqlite')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_source_row ON performance(source_id,source_row)')
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='profile_groups'").fetchone():
        group=','.join('"'+c+'"' for c in COLS)
        db.execute('CREATE TABLE profile_groups(source_id TEXT, representative_source_row INTEGER, observation_count INTEGER)')
        db.execute('INSERT INTO profile_groups SELECT source_id, MIN(source_row), COUNT(*) FROM performance GROUP BY source_id,'+group)
        db.execute('CREATE UNIQUE INDEX idx_profile_group ON profile_groups(source_id,representative_source_row)')
        db.execute('CREATE VIEW performance_unique_profiles AS SELECT p.*, g.observation_count FROM profile_groups g JOIN performance p ON p.source_id=g.source_id AND p.source_row=g.representative_source_row')
        db.commit()
    unique,total=db.execute('SELECT COUNT(*),SUM(observation_count) FROM profile_groups').fetchone()
    assert total==db.execute('SELECT COUNT(*) FROM performance').fetchone()[0]
    assert unique==db.execute('SELECT COUNT(*) FROM performance_unique_profiles').fetchone()[0]
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';db.close()
    from scipy.io import loadmat
    import numpy as np
    exports=json.loads((out/'_work/export_manifest.json').read_text(encoding='utf-8'))
    independent_count=sum(len(np.unique(loadmat(out/'_work'/e['intermediate'])['values'][:,2:],axis=0)) for e in exports)
    assert independent_count==unique
    verification_path=out/'audit/verification.json'
    verification=json.loads(verification_path.read_text(encoding='utf-8'))
    verification['profile_groups_independently_checked']=unique
    verification['profile_observation_counts_reconcile']=True
    verification_path.write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    profile={'unique_profiles_within_source':unique,'original_observations':total,'repeated_observations':total-unique,'grouping':'Same source_id and exact equality of all 17 performance fields; exclude gen/index only. This does not identify unique transistor designs.','representative':'Smallest original MATLAB source row; all original observations remain in performance and main CSV/JSONL.'}
    (out/'audit/profile_summary.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    readme=out/'README.md'
    text=readme.read_text(encoding='utf-8').split('\n## Optional performance groups')[0]
    readme.write_text(text.rstrip()+'\n'+profile_notes(profile),encoding='utf-8')
    write_llm_guide(out,profile)
    schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Analog performance JSONL record','type':'object','additionalProperties':False,'required':['record_id','topology','folder_labels','generation','individual_index','performance','quality_flags','source'],'properties':{'record_id':{'type':'string'},'topology':{'type':'string'},'folder_labels':{'type':'object','additionalProperties':False,'required':['tech_node','vdd','vcm','cl','run'],'properties':{'tech_node':{'type':'integer'},'vdd':{'type':'number'},'vcm':{'type':'number'},'cl':{'type':'number'},'run':{'type':'string'}}},'generation':{'type':'integer','minimum':0},'individual_index':{'type':'integer','minimum':0},'performance':{'type':'object','additionalProperties':False,'required':COLS,'properties':{c:{'type':['number','null']} for c in COLS}},'quality_flags':{'type':'array','items':{'type':'string'},'uniqueItems':True},'source':{'type':'object','additionalProperties':False,'required':['source_id','row_1based'],'properties':{'source_id':{'type':'string'},'row_1based':{'type':'integer','minimum':1}}}}}
    (out/'record_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and not any(t in {'_work','__pycache__'} for t in f.relative_to(out).parts):
                z.write(f,Path(out.name)/f.relative_to(out))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'profile_summary':profile,'archive':str(archive),'archive_MB':round(archive.stat().st_size/1024**2,2)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
