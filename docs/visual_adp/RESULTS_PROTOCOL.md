# Results and comparison protocol (future vision jobs)

Use dataset-indexed disjoint train / architecture-validation / test splits.
Record dataset source/revision/license, split indices/checksums, pretraining
dataset and checkpoint digest, normalization, augmentations, seeds, model/site,
method/profile, growth criterion, optimizer budget and training hardware.

Pair each ADP job against same-initialization MLP-Fixed with identical training
data, initialization, encoder and compute budgets. Log candidate sequence,
global-best selection, top-1 and balanced accuracy, NLL/calibration when valid,
frozen/trainable/new parameter counts, FLOPs including shared spatial repetition,
peak VRAM, training steps, train time and candidate rejection reasons.
Never choose architecture on held-out test metrics. Report actual measured
outcomes separately from pipeline conformance. 

## Machine-readable parameter accounting (2026-11-02 integration)

New coordinated visual worker results include `parameter_accounting`, constructed
by `src/simple_adp/visual_accounting.py::visual_parameter_accounting` from
the **selected model instance**, not guessed from the route name:

- `total_parameters`: unique parameter tensors across the composite model,
  including frozen encoder/backbone, any classifier and eligible MLP sites.
- `frozen_parameters` / `trainable_parameters`: mutually exclusive counts
  by current `requires_grad` state; their sum equals total.
- `eligible_mlp_parameters`: sum over distinct actual `TabularMLP`
  submodules advertised by `growth_targets`. In V08's scheduled multi-site
  model, this includes previously committed currently frozen MLPs.
- `currently_trainable_eligible_mlp_parameters`: eligible-site subset whose
  parameter tensors are **currently** trainable. Other trainable parameters
  such as an allowed fixed-dimension supervised source task head are counted
  separately as `other_trainable_parameters`.
- `eligible_growth_sites[site_id]`: fixed local interfaces, current hidden
  widths, MLP parameter counts, active-trainable flag, and Linear matrix MACs
  per **one actual MLP invocation**.

MAC counts are not full-image FLOPs and do **not** account for repeated
token/spatial application, interpolation, attention, convolution, FFT,
normalization or entire-backbone inference. They must not be advertised as
whole-model compute. Neither these selected-state counts nor real
frozen-source micro-smokes establish fairness across **search training
budgets** or estimate newly trained FLOPs. Future scientific result exports
must include measured per-candidate training steps, exact geometry-specific
forward compute and hardware peak memory in separately identified fields.

For the frozen V01 head, a temporary composite ModuleDict includes both
the original frozen encoder and the selected native ADP head **only for
read-only accounting**, without changing either serialized model.
