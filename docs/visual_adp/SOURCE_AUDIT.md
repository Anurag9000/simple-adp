# Live source audit: visual ADP
Baseline main: a2e6b6325a80ca6da41b1898ebf71c6ec8829a62 (2026-10-29).
This is an incremental evidence ledger, not a claim that the full vision suite is finished.

## Existing algorithm and API anchors
- src/simple_adp/models.py: TabularMLP, snapshot/from_snapshot, rebuild_mlp. Hidden layer stack is nn.Linear + optional BatchNorm + ReLU, followed by fixed-dimensional output. rebuild_mlp copies overlapping state tensor coordinates.
- src/simple_adp/growth.py: RR, synchronized all-layer width, DEN-specific RR, width-one and matched-width depth, can_* capacity predicates. These accept TabularMLP rather than arbitrary CNN/ViT composite models.
- src/simple_adp/search.py: SearchController._arrays converts feature rows to float tensors; _initial_model creates TabularMLP; run dispatches the 14 configurations and validates TabularMLP-specific resume payloads.
- src/simple_adp/registry.py: exactly 14 registered configurations. They must not be redefined by the vision extension.
- src/simple_adp/tasks.py: TaskAdapter classification/regression objective, target encoding, validation selection, and reporting.
- src/simple_adp/estimator.py: TabularADP preprocessing, native search controller, checkpoints, fit/predict/save/load. The V01 frozen-feature route delegates to this API without changing its semantics.
- src/simple_adp/runner/models.py: ModelRegistry plus SimpleADPModelAdapter enumerate_routes/fit_predict; assumes TabularTaskData and TabularADP.
- src/simple_adp/runner/datasets.py: DatasetProvider, DatasetRegistry, PreparedTaskData. Vision provider is not yet registered.
- src/simple_adp/runner/plan.py: SuitePlan and RunIdentity with provider/suite/task/split/route signatures.
- src/simple_adp/config.py, generic_search.py, dedicated.py: native ADP method settings, generic/dedicated controllers and variance-stagnation criteria.

## Current data flow
Raw image [N,C,H,W] -> user-provided frozen classifier-free encoder (no-grad, eval)
-> pooled [N,D] feature embeddings -> original TabularADP TabularPreprocessor
-> original SearchController -> original TabularMLP growth and TaskAdapter validation.
Inference repeats the same frozen embedding path. Cache equivalence depends on
identical encoder and preprocessing; cache contents must be identified independently.

## Unresolved integration
For V02-V11, inserting a local MLP inside a composite requires passing the whole
image model through candidate train/evaluation, with fixed downstream differentiable
layers kept in autograd. SearchController, StageTrainer, model snapshots, candidate
mutation, and runner remain TabularMLP-specific. Replacing a module in place without
those integrations is **not** a complete ADP route. Runner vision job registration,
raw-image provider, stable split manifests, real CIFAR training, whole-repo regression
execution, and full serial-depth parity are not yet implemented in this increment.

## Additional opt-in image preparation surface
src/simple_adp/vision_data.py adds torchvision dataset loading and deterministic
classwise source-train/validation partitioning, preserving official test
isolation and recording raw checksums and original indices. This is an
installed package API, not yet a DatasetRegistry implementation or a runner
job. The optional torchvision extra does not affect existing dependencies.
