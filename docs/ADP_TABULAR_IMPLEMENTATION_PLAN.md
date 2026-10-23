# Implement the canonical ADP models for tabular MLPs

## Summary

Build a new, modular PyTorch package in the empty `simple-adp` checkout. Treat `~/Documents/ADP_algorithms.docx` as the sole algorithm specification. Preserve its distinct controllers and training rules while making task-specific inputs, losses, validation scores, and predictions configurable through adapters.

Expose one shared estimator and a typed model registry. Keep fixed-grid coordinates caller-supplied because the guide does not specify them.

## Implementation

- Register the guide’s 14 configurations:
  1. `LGND-W-Greedy`
  2. `ADP-W-RR`
  3. `ADP-W-All`
  4. `ADP-WD-RR-D1`
  5. `ADP-WD-All-Dcur`
  6. `ADP-WD-RR-Dcur`
  7. `ADP-WD-All-D1`
  8. `ADP-DW-D1-RR`
  9. `ADP-DW-Dcur-All`
  10. `ADP-DW-Dcur-RR`
  11. `ADP-DW-D1-All`
  12. `ADP-D-D1`
  13. `ADP-D-Dcur`
  14. `MLP-Fixed`

- Separate the package into modules for the MLP and architecture mutations; dedicated DEN/LG-ND controllers; generic candidate search; task adapters; tabular preprocessing; training stages; checkpoint/resume; configuration and registry; and tests. Share mechanisms where the guide says they are common, while keeping distinct search policies explicit.

- Implement the guide’s mechanics: hidden-width-vector architecture state; overlap-copy warm starts with fresh initialization for new parameters; all-layer and round-robin widening; progressive and matched-width depth growth; patience, delta, capacity checks, nested outer-width/inner-depth search (including its configured new-layer seed width and width-growth strategy), and global-best restoration. Keep current forward state separate from global best. Do not claim function preservation.

- Preserve the dedicated trainer’s documented defaults: MSE with Adam and cosine warm restarts; reset optimizer/scheduler at architecture boundaries; resume optimizer, scheduler, RNG, and stage position within an interrupted stage. Implement generic `adp_search` as a separate path with its AdamW and optional auxiliary-loss behavior, exposing its prediction/equality/inequality loss weights through the search settings and task adapter. When both auxiliary weights are zero, use prediction loss alone as the guide specifies.

- Represent the guide’s documented implementation differences explicitly. Keep a faithful generic-search compatibility profile for reproducing its described behavior and a separately identified corrected profile; do not silently change the canonical configuration identities. The task adapter defines output semantics.

- Provide `TabularADP.fit/predict` with explicit validation data, model selection by registry ID, and checkpoint save/load. Use a `TaskAdapter` interface for output specification, loss, validation score, and prediction conversion. Include classification and regression adapters and allow custom adapters.

- Accept pandas DataFrames and numeric arrays. Fit preprocessing only on training data; use median imputation and standardization for numeric columns, one-hot encoding with an unknown category for categorical columns, and preserve feature order/schema for inference. Arrays default to numeric unless a feature schema is supplied.

- Require callers of `MLP-Fixed` to supply the architecture grid; reject an omitted grid instead of inventing default coordinates.

## Public Interfaces

- `ModelConfig`: canonical ID, controller family, growth order and strategy, depth seed, optimizer profile, patience, margins, and capacity limits.
- `TaskAdapter`: task output definition, loss computation, validation score, and prediction conversion.
- `TabularPreprocessor`: fit on training features; transform validation and inference features with the fitted schema.
- `TabularADP`: shared estimator that selects a registered configuration and exposes `fit`, `predict`, task-appropriate prediction methods, `save`, and `load`.
- `CheckpointStore`: persist configuration identity, preprocessing/task metadata, current and best architectures, growth cursor, counters, stage state, optimizer/scheduler, RNG, and data position needed for documented resume behavior.

## Tests and Acceptance

- Verify all 14 IDs are registered and map to their documented mechanisms; fixed MLP requires an explicit grid.
- Use deterministic toy tabular classification and regression data to exercise the adapters, preprocessing, fit/predict, and custom-task extension.
- Cover each growth mutation, round-robin cursor restoration, progressive and matched-width depth behavior, capacity limits, patience/delta acceptance, forward traversal after failure, final best restoration, and LG-ND’s strict first-failure stop.
- Verify stage-boundary optimizer reset and same-stage checkpoint resume; reject incompatible checkpoint/configuration signatures.
- Test generic search’s faithful and corrected profiles separately, including the documented failure-counter, global-best, and `max_neurons` behavior.
- Confirm preprocessing learns only from training rows, handles missing and unseen categories, and rejects inference schema mismatches.
- Run the full unit and integration suite before calling the implementation complete.

## Assumptions

- PyTorch is the model/training framework, consistent with the guide’s MLP and trainer mechanisms.
- The regression adapter reproduces the documented MSE path; other tasks supply their own loss and validation score without changing architecture-search rules.
- The 14 configuration count remains fixed; compatibility behavior profiles do not inflate that canonical model count.
