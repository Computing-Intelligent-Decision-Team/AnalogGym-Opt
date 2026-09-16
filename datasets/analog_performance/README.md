# Historical analog circuit performance dataset

[中文说明](README_zh.md) · [Single-file LLM guide (Chinese)](给大模型的数据库说明.md)

This collection contains **358,234 observations from 73 MATLAB performance tables,
covering 18 amplifier topologies**. It provides a documented, queryable copy of
historical circuit performance results, with stable record IDs and source hashes.

The original directory labels include **22 and 180 nm** technologies. These are
historical results with their original measurement conventions, rather than outputs
of the repository's current ngspice/sky130 simulations. The directory labels alone
do not establish the actual PVT conditions of each measurement.

## Read or query the data

The data are stored in gzip files to keep individual Git objects small. All formats
contain the full collection. Python's standard library is sufficient to unpack and
query them; MATLAB is only needed to reproduce extraction from the original tables.

From the repository root:

```bash
# Verify and unpack the SQLite database.
python datasets/analog_performance/scripts/unpack.py --format sqlite

# Retrieve a bounded set of records. Thresholds here are illustrative, not a
# definition of feasible designs or a recommendation across operating conditions.
python datasets/analog_performance/scripts/query.py --sql "SELECT record_id, topology, gain, gbw, pm, ivdd_27 FROM performance_unflagged WHERE tech_node_label=180 AND gain>=80 AND pm>=60 ORDER BY ivdd_27 ASC" --limit 10

# Optional: unpack CSV, JSONL, or all formats.
python datasets/analog_performance/scripts/unpack.py --format all

# Verify every compressed asset without creating uncompressed files.
python datasets/analog_performance/scripts/unpack.py --format all --verify-only
```

JSONL can also be read directly without unpacking:

```python
import gzip
import json

path = "datasets/analog_performance/data/by_topology/Fan_SMC_Pin_3.jsonl.gz"
with gzip.open(path, "rt", encoding="utf-8") as stream:
    record = json.loads(next(stream))
print(record["performance"], record["quality_flags"])
```

Read [FIELD_GUIDE.md](FIELD_GUIDE.md) before interpreting values. For LLM workflows,
start with [LLM_START_HERE.md](LLM_START_HERE.md) and retrieve only the relevant rows.
Use SQL for numerical filtering and aggregation, rather than sending the full
collection in a prompt. SQL queries are read-only and return at most 1,000 rows.

## Contents

| Path | Purpose |
|---|---|
| `data/performance.sqlite.gz` | Indexed observations, sources, and optional views |
| `data/performance.csv.gz` | Flat table with all 19 original columns and provenance |
| `data/by_topology/*.jsonl.gz` | Complete records split into 18 topology files |
| `assets.json` | Compressed/uncompressed sizes and SHA-256 checksums |
| `source_manifest.json` | Original relative paths, MATLAB row counts, source hashes |
| `field_dictionary.json`, `record_schema.json` | Field meanings and JSONL schema |
| `topology_index.json`, `summaries/` | Topology counts, condition labels, and distributions |
| `examples/sample_records.jsonl` | First/middle/last rows of each source, 219 examples |
| `audit/` | Cleaning rules, reconciliation, and full-value verification |
| `evidence/` | Original extraction/selection code used to interpret fields |
| `scripts/` | Unpacking, bounded queries, and extraction/rebuild utilities |

The original MAT tables and paired TD/TBM parameter tables are not bundled.
Their relative paths and source keys are retained for tracing results in an
original copy of the collection. Machine-specific absolute source paths have
been removed from the published manifest.

## Data quality and interpretation

All finite values are preserved without rounding, filling, clipping, or inferred
unit conversions. Full-value checks compared **6,806,446 original values in each
format** against the numeric arrays exported by native MATLAB. No fully identical
19-column rows were found within a source table, and no non-finite values were found.

**125,974 records have at least one review flag; 232,260 trigger none of the defined
checks.** A record without flags is not a certified feasible or successful design.
The main table and the CSV/JSONL files retain both groups.

| Check | Observations |
|---|---:|
| Negative `SR` | 40,444 |
| `SR` exactly -1, a suspected sentinel | 39,684 |
| Negative `FOML` | 40,444 |
| `d_settle == settlingTime`, excluded by a source selection script | 86,290 |

Checks overlap. Negative values and suspect observations remain in the data with
explicit flags. The source extraction code already removed negative phase-margin
observations, so this is not a complete simulation trajectory or a basis for
estimating total simulation success rates.

The MATLAB tables have no unit metadata. Suggested units in the field dictionary
are explicitly marked as inferred where appropriate. In particular:

- The source extraction code transforms `chip_area` with `sqrt(abs(...))`; it must
  not be interpreted directly as physical chip area.
- Noise, temperature-coefficient, and settling-related definitions/units are not
  fully established; retain the source conventions until verified.
- The current source formula for `FOM_AW` uses a fixed `1500e-12`, not each run's
  directory load-capacitance label. It has not been recomputed.
- Exact PVT conditions and aggregation conventions cannot be recovered from these
  performance columns alone.

SQLite offers `performance_unflagged` for optional flag-based selection and
`performance_unique_profiles` for **317,619 exact performance groups**. The latter
groups identical values of all 17 performance fields within the same source,
keeps the first original row as representative, and exposes `observation_count`.
It does not prove that the transistor designs are identical. All original
observations remain in `performance`.

For model training, define the target and split by run, design, or topology as
appropriate. A random row split may leak repeated designs or optimization history.

## Reproduce extraction

With the original `Topo-Tech_nodes-VDD-VCM-CL-RUN` collection and adjacent
`used_code/` directory available, use MATLAB plus Python with `numpy` and `scipy`.
The published record IDs derive from relative source paths and MATLAB row numbers.

On Windows:

```powershell
./datasets/analog_performance/scripts/rebuild.ps1 -Source /path/to/Topo-Tech_nodes-VDD-VCM-CL-RUN -Destination /path/to/new-output -Matlab matlab -Python python
```

On other platforms, set `ANALOG_SOURCE` and `ANALOG_OUTPUT`, create the output
`_work/` directory, run `scripts/export_mat_tables.m` in MATLAB, and invoke
`scripts/build_dataset.py --source ... --output ...`, followed by
`scripts/verify_dataset.py --output ...`. Those verification scripts require the
original MAT collection; ordinary use of the published data does not.
