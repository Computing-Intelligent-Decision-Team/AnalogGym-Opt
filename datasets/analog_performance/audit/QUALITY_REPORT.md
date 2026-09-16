# Data quality report

The export covers **358,234 observations across 18 topologies**, from 73 MATLAB performance tables containing 358,234 source rows.

## Cleaning results

| Measure | Count |
|---|---:|
| Identical full rows removed within a source | 0 |
| Non-finite numeric cells | 0 |
| Observations with at least one review flag | 125,974 |
| Observations with no review flags | 232,260 |

| Review flag | Observations |
|---|---:|
| `sr_negative` | 40,444 |
| `sr_minus_one` | 39,684 |
| `foml_negative` | 40,444 |
| `settling_fields_equal` | 86,290 |

Flags overlap, so their counts must not be added to obtain a row count. Unflagged observations are not certified feasible designs.

## Numeric and duplicate policy

All finite double values are preserved without rounding, imputation, clipping, absolute-value transforms, or unit conversion. Full-row deduplication requires exact equality of all 19 source fields, including `gen` and `index`, within one source file. Removed rows would be recorded in `duplicates.jsonl`; none were found in this collection.

Observations with equal performance values remain in the main dataset. The optional `performance_unique_profiles` view groups them within a source while preserving their observation counts. Equal performance values do not identify the same physical design.

## Interpretation limits

- The source extraction code already filtered `pm < 0` and removed duplicates. This collection is not a complete simulation trajectory and cannot establish the overall simulation success rate.
- The source script applies `sqrt(abs(chip_area))`. Historical generating versions have not been verified for every file; do not treat this metric as physical area.
- Equality of `d_settle` and `settlingTime` is a condition used by a source selection script. It does not prove that every flagged observation failed simulation.
- The current `FOM_AW` source formula uses a fixed `1500e-12`, not the directory CL label. No figure of merit has been recomputed.
- Only performance tables are exported. Paired TD/TBM parameter tables remain in the original collection. Verify key uniqueness before joining them through `source_id`, `gen`, and `index`.
- Topology summaries pool multiple condition groups and serve as navigation aids, not evidence of an optimal design.
- For training, split by run, design, or topology as appropriate and check for repeated designs. Random row splits can leak repeated designs or optimization history.

## Verification

See [verification.json](verification.json) for the full-value comparison against native MATLAB exports, source hashes, row counts, quality flags, and SQLite integrity. Verification compares all 6,806,446 source numeric values in each of CSV, JSONL, and SQLite. See [profile_summary.json](profile_summary.json) for optional group counts.

The reference MATLAB snippets in `../evidence/` retain their calculations and selection logic. Comments and the extraction completion message are translated into English; these annotated copies are not byte-identical archives of the original scripts.
