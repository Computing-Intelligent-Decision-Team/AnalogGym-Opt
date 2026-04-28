# Circuit Contract

Use this note when adding a new circuit or debugging why a run fails before training starts.

## Required YAML Fields

The circuit config loader expects these top-level fields:

- `base_dir`
- `file_paths`
- `device`
- `performance`

The optimizer runtime also depends on:

- `action_dim`
- `ckt_hierarchy`
- `graph`

## Required `file_paths` Keys

The current environment and parser expect these entries:

- `ACDC_cir_path`
- `Tran_cir_path`
- `netlist_path`
- `vars_path`
- `dev_params_path`
- `op_mean_std_path`
- `dc_results_path`
- `ac_results_path`
- `op_results_path`
- `GBW_PM_path`
- `tran_results_path`
- `tran_dat_path`

All of them are resolved relative to `base_dir`.

## Device Block

Each item in `device` defines a design-variable group. The optimizer derives the action space from the keys inside `range`.

Example shape:

```yaml
M0:
  range: {W: [0.5, 10], L: [0.5, 5], M: [1, 50]}
  step: {W: 0.1, L: 0.1, M: 1}
  init: {W: 1, L: 1, M: 4}
  num: 8
```

`action_dim` must equal the total number of scalar range entries across all device blocks.

## Graph Block

The current RGCN policy needs:

- `num_relations`
- `num_nodes`
- `num_node_features`
- `edge_index`
- `edge_type`
- `observation_matrix`

Checks worth doing:

- `len(edge_index) == len(edge_type)`
- every edge is a two-item list
- every observation row has `num_node_features` entries
- `len(observation_matrix) == num_nodes`

## Performance Block

This block drives reward parsing and reporting. Keep the metric names aligned with the existing environment and reporting helpers. For the paper code, the commonly reported metrics include:

- `phase_margin`
- `dcgain`
- `PSRR`
- `cmrrdc`
- `settlingTime`
- `FOML`
- `FOMS`
- `Active_Area`
- `Power`
- `GBW`
- `sr`

## Output Locations

A successful training run writes under:

- `training_saves/<run>/top_designs_tt`
- `training_saves/<run>/recommended_candidates_tt`
- `training_saves/<run>/training_history_GRPO_*.pkl`

When `tt-proxy` mode is enabled, it may also write:

- `training_saves/<run>/top_designs_verified_pvt`
- `training_saves/<run>/recommended_candidates_verified_pvt`

## Important Boundary

Structural validation is necessary but not sufficient. A config can be valid while `ngspice` still fails because a deck includes a missing PDK model file or a broken relative include path. If training reports missing `.op` or parser output files, inspect the generated `ACDC.log` and `Tran.log` first.
