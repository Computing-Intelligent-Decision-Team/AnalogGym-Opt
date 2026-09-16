"""Read-only, bounded SQLite query; Python standard library only."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,default=Path(__file__).resolve().parents[1]/'data/performance.sqlite')
    p.add_argument('--sql',required=True,help='One SELECT or WITH ... SELECT query')
    p.add_argument('--limit',type=int,default=20)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not 1<=a.limit<=1000:p.error('--limit must be between 1 and 1000')
    sql=a.sql.strip().rstrip(';')
    if not sql or sql.split(None,1)[0].upper() not in {'SELECT','WITH'}:p.error('Only SELECT/WITH queries are accepted')
    if not a.db.is_file():p.error('Database not found. Run scripts/unpack.py --format sqlite from the dataset directory first.')
    db=sqlite3.connect(a.db.resolve().as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA query_only=ON'); db.row_factory=sqlite3.Row
    rows=db.execute('SELECT * FROM ('+sql+') LIMIT ?', (a.limit,)).fetchall()
    stream=a.output.open('w',encoding='utf-8') if a.output else sys.stdout
    try:
        for row in rows:stream.write(json.dumps(dict(row),ensure_ascii=False,allow_nan=False)+'\n')
    finally:
        if a.output:stream.close()
        db.close()

if __name__=='__main__':main()
