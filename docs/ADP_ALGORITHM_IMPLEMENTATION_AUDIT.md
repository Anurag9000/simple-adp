# ADP algorithm implementation and verification audit

This audit maps the canonical requirements in [`ADP_algorithms.docx`](ADP_algorithms.docx)
and [`ADP_TABULAR_IMPLEMENTATION_PLAN.md`](ADP_TABULAR_IMPLEMENTATION_PLAN.md)
to the current implementation and executable contracts. The DOCX is the
algorithm reference. The copied OPF runner ledger is a separate specification
for orchestration and does not define model algorithms.

The current `docs/ADP_algorithms.docx` is byte-identical to the user's
canonical `/home/anurag-basistha/Documents/ADP_algorithms.docx` (SHA-256
`5a949387d9709e3d09001893e413a22d0c4c57f16a71470786a2cd5bb7861948`, 68,642
bytes; verified 2026-10-22).

## Algorithm-to-code map

| Guide mechanism | Implementation | Current evidence |
|---|---|---|
| Fourteen experiment identities, including the two 2×2 mixed-order blocks | `registry.py`, with explicit strategy, order, and depth seed per ID | `test_registry_has_all_documented_ids_and_custom_fixed_grid`; `test_all_eight_mixed_order_variants_execute_their_declared_mutations` |
| Registry patience, delta, and capacity values act as defaults when callers omit runtime settings; explicit per-run `SearchSettings` overrides them | `config.py:ModelConfig`; `estimator.py:TabularADP.__init__` | `test_estimator_uses_model_config_search_defaults_and_explicit_settings_override` injects non-default registry values and verifies all seven controls reach runtime settings, then verifies explicit settings take precedence. |
| Ordered hidden-width vector and architecture snapshots | `models.py:TabularMLP`, `snapshot`, `from_snapshot` | `test_growth_mutations_cursor_depth_and_overlap_copy`; estimator checkpoint contracts |
| Overlap-copy warm start; new tensor cells retain fresh initialization; no function-preservation guarantee | `models.py:rebuild_mlp`, `_copy_overlap` | `test_growth_mutations_cursor_depth_and_overlap_copy`; `test_architecture_growth_keeps_new_linear_cells_at_fresh_initialization` compares added hidden/output weight and bias cells with an independently initialized target under the same seed; shape-changing model rebuilds in search contracts. Controller materialization of global-best snapshots now preserves RNG, so bookkeeping constructors do not shift initialization of later candidate growth; `test_global_best_snapshot_restoration_does_not_shift_growth_rng` pins this for dedicated and generic search. |
| Generic candidate rebuild copies matching BatchNorm affine parameters and running buffers, overlapping prefixes on width changes | `models.py:TabularMLP`, `rebuild_mlp` | `test_generic_candidate_rebuild_copies_batchnorm_overlap_and_snapshot_buffers` verifies matching parameter/buffer overlap, fresh new BatchNorm entries, copied scalar batch counter, output-layer overlap, and BatchNorm/cursor snapshot restoration. |
| Unit budget counts hidden units plus output units; it is not a parameter budget | `growth.py:total_neurons`, `can_grow_*`, `can_add_depth` | `test_width_capacity_guards_exhaust_small_architecture_state_space` checks all-layer and RR budget predicates across depth 1–3, widths 1–4, increments 1–3, width caps 1–5, and every neuron-budget value through each transition boundary; `test_depth_mutations_exhaust_small_architecture_state_space` checks depth 1–3, widths 1–3, depth caps 1–4, neuron caps 1–12, both seed widths, progressive insertion, matched-width insertion, and non-uniform-stack eligibility |
| All-layer widening by `max(1, ex_k_width)` in the generic helper | `growth.py:expand_width_all_layers`; `dedicated.py:_width_search` | width mutation tests, forward-traversal test, exhaustive small-state transition coverage in `test_width_mutations_exhaust_small_architecture_state_space`, and direct zero/negative clamp contracts in `test_generic_growth_primitives_clamp_nonpositive_size_arguments` |
| Growth-strategy width summary (all-layer maximum; dedicated RR minimum), cursor-ordered eligible layer, exact `ex_k_width` increment, hard per-layer/unit caps, saved cursor | `growth.py:scalar_width`, `expand_width_den_round_robin`, `can_grow_width_round_robin`; `dedicated.py:_width_search` | `test_scalar_width_summary_depends_on_growth_strategy` pins max-versus-min scalar semantics on a nonuniform stack; RR proposal and cursor restore assertions in `test_growth_mutations_cursor_depth_and_overlap_copy`; exhaustive small-state coverage across nonuniform stacks, cursors, increments, width ceilings, and unit budgets in `test_width_mutations_exhaust_small_architecture_state_space` and `test_width_capacity_guards_exhaust_small_architecture_state_space`; cap boundary distinguishes dedicated exact-`+k` behavior from generic clamping in `test_dedicated_round_robin_does_not_clip_configured_increment_at_width_cap`; complete controller trajectory and valley crossing in `test_round_robin_den_traverses_failed_shape_from_cursor`; interrupted cursor resume in `test_round_robin_growth_cursor_survives_estimator_resume` |
| Generic round-robin candidate helper, its different cursor selection rule, and `max(1, ex_k)` increment | `growth.py:expand_width_round_robin`; `generic_search.py` | Exhaustive small-state transition coverage distinguishes generic from dedicated RR across nonuniform stacks, cursors, increments, and ceilings in `test_width_mutations_exhaust_small_architecture_state_space`; direct zero/negative clamp contracts in `test_generic_growth_primitives_clamp_nonpositive_size_arguments`; mixed-order execution test |
| Patience DEN accepts only `candidate < global_best - delta`, retains failed candidate as forward state, and restores global best at termination | `dedicated.py:_width_search`, `_depth_search` | `test_patience_delta_failure_traversal_and_final_global_best`; exact equality at the margin is pinned by `test_architecture_improvement_equal_to_delta_is_a_failure`; `test_dedicated_patience_margin_and_capacity_finite_matrix` sweeps all-layer width DEN and matched-width depth DEN patience 0–3, delta 0/0.1/0.2, and limits against an independent trajectory oracle. `test_round_robin_den_controller_matches_margin_patience_and_cursor_oracle` adds a 12-case dedicated RR controller cross-product over patience 0–3 and delta 0/0.05/0.1, independently checking proposal cursor, forward state, acceptance, and global-best restoration. |
| Depth acceptance uses the same strict margin rule for both width-1 progressive and matched-width seeds | `dedicated.py:_depth_search` | `test_depth_candidate_equal_to_delta_is_failure_and_restores_global_best` runs both `ADP-D-D1` and `ADP-D-Dcur`, including progressive stepping stones, and proves a candidate exactly at `best - delta_depth` fails and the original global best is restored; the finite controller matrix also sweeps `ADP-D-Dcur` patience 0–3, delta 0/0.1/0.2, and depth caps 2–4 |
| LG-ND strict `candidate < best`, ignores DEN delta and patience, stops on the first failure, restores its best, and retains the attempted shape as forward state | `dedicated.py:_width_search(greedy=True)` | `test_lgnd_stops_at_first_failure_and_advances_forward` pins positive delta being ignored and equality/flat score as the first failure; `test_lgnd_resumed_failed_expansion_stops_before_next_growth` pins immediate first-failure termination after an interrupted greedy candidate is resumed; `test_lgnd_matches_bounded_independent_trajectory_oracle` cross-checks 2,592 trajectories over initial width/depth, width increment, width and unit-budget headroom, patience 0/2, delta 0/0.5/2, strict improvement/equality/regression score sequences, capacity stop, best restoration, and failed-candidate forward-state retention. |
| LG-ND's manually selected RR mode retains the all-layer capacity precheck quirk | `dedicated.py:_width_search(greedy=True)` | `test_lgnd_round_robin_override_keeps_all_layer_capacity_precheck` |
| Progressive depth inserts width 1, trains stepping stones, catches up only the new terminal layer, and compares only the completed-width candidate | `dedicated.py:_depth_search`; `growth.py:expand_depth` | `test_progressive_and_matched_width_depth_proposals`; `test_progressive_depth_stepping_scores_do_not_replace_global_best` proves intermediate losses cannot replace the global best; `test_progressive_depth_capacity_stop_retains_last_forward_stepping_architecture` pins forward-state retention when the neuron cap prevents catch-up; `test_progressive_depth_growth_resumes_at_every_stage_boundary` interrupts at each of the 6 reachable loss-call boundaries for initial width 2 and each of the 9 boundaries for initial width 3, spanning the initial stage, every progressive catch-up stage, and the completed-width candidate, then compares history and predictions with uninterrupted training; same-stage progressive-resume contracts |
| Matched-width depth appends the previous final width directly | `growth.py:expand_depth_match_width`; `dedicated.py:_depth_search` | `test_progressive_and_matched_width_depth_proposals` |
| Depth DEN retains a failed completed architecture as forward state and can accept a later depth proposal | `dedicated.py:_depth_search` | `test_depth_den_crosses_failed_completed_candidate` |
| Dedicated progressive and matched-width depth controller trajectories across growth, unit budgets, patience, and margins | `dedicated.py:_depth_search` | `test_dedicated_depth_controllers_match_bounded_trajectory_oracle` checks 864 combinations across both canonical depth IDs, initial widths 1–3, depth ceilings 2–4, four unit-budget headrooms, patience 0–3, and deltas 0/0.05/0.15; it independently checks every training proposal, progressive catch-up step, score/history entry, selected global best, retained forward state, and failure counter |
| Dedicated nested outer-width/inner-depth sweep, both patience counters, width-1 or current-width depth seed, inclusive outer-width `delta_width` boundary, and signature-checked resume | `dedicated.py:run_nested_width_depth`; `search.py:run_nested_width_depth` | `test_nested_controller_exhaustive_small_domain_matches_independent_oracle` enumerates 3,456 cases across initial stack shape, width increment/cap, depth cap, neuron budget, seed mode, both patience values 0–2, and both deltas 0/0.07; `test_nested_outer_width_patience_resets_when_gain_equals_delta` pins the guide's “at least δW” equality behavior and proves it admits another outer round; `test_nested_controller_resume_matches_uninterrupted_at_each_stage_position` verifies base-stage, first inner-depth candidate, and later inner-depth candidate resume equivalence; `test_nested_controller_resume_matches_each_outer_and_inner_stage` additionally interrupts at every reachable scored stage in a bounded two-round current-width-seeded trajectory, including the later outer-width round; `test_nested_controller_resumes_interrupted_inner_depth_candidate` additionally rejects changed settings or seed mode |
| Nested DEN outer-width strategy follows the guide's reported `width_growth_strategy`, including dedicated RR cursor and capacity semantics; selected strategy and cursor are checkpoint identity/state | `dedicated.py:run_nested_width_depth`; `search.py:run_nested_width_depth` | `test_nested_controller_honors_round_robin_outer_width_strategy` proves an incomplete RR round remains nonuniform and blocks inner depth until the stack synchronizes; `test_nested_resume_rejects_changed_width_growth_strategy` rejects resume under a changed strategy; `test_nested_round_robin_resume_restores_growth_cursor` compares interrupted and uninterrupted RR trajectories from a nonzero cursor. The default remains all-layer growth. |
| Generic `width_only`, `depth_only`, `width_to_depth`, and `depth_to_width` phase mappings; narrow depth insertion without catch-up; nonuniform-stack depth stop | `generic_search.py:run`; growth helpers | standalone width/depth trajectory tests plus all-eight mixed-order trajectory test; width and depth phase matrices independently verify patience 0–3 and margins 0/0.1/0.2 for all eight canonical mixed-order IDs under both profiles, including the width-1 nonuniform-stack stop; `test_generic_mixed_order_phase_transition_preserves_forward_candidate` exercises both phases for all eight IDs under both profiles, asserting retained forward architecture, second-phase proposal, counters/global selection, and shared candidate ceiling. Its Depth→Width cases distinguish generic RR's first-below-cap cursor rule from dedicated RR's minimum-width target rule. |
| Generic source-compatibility defects: attempted-candidate failure count, local-predecessor candidate selection, and omitted `max_neurons` enforcement | `generic_search.py` faithful profile | `test_generic_faithful_and_corrected_global_best_and_capacity`; width/depth phase matrices compare independent trajectory oracles across all eight mixed-order IDs, patience 0–3, and margins 0/0.1/0.2; `test_generic_neuron_capacity_profile_matrix_across_mixed_order_models` pins the faithful cap omission in every mixed-order identity; the new two-phase transition matrix verifies faithful counters and local-predecessor selection across phase boundaries. |
| Separately named corrected generic behavior: margin-aware phase failures, retained global best, and unit cap | `generic_search.py` corrected profile | width/depth phase matrices check acceptance, counters, forward state, and best restoration against independent oracles across all eight mixed-order IDs; the mixed-order capacity matrix verifies candidate rejection in each route's first phase |
| Dedicated MSE with fresh Adam and cosine warm restarts at every architecture boundary | `training.py:StageTrainer`; `search.py:_stage` | `test_controller_resets_adam_and_cosine_scheduler_at_architecture_boundary` runs two real architecture stages and proves Adam update count and scheduler position restart; `test_stage_optimizer_profiles_keep_documented_parameters_and_scheduler` checks configured learning rate and exact `T_0`, `T_mult`, and `eta_min`; `test_stage_optimizer_boundary_reset_and_same_stage_resume` additionally checks all 16 same-stage resume positions; `test_gradient_clipping_is_generic_adamw_only` confirms dedicated Adam stages do not apply the generic clipping option; task regression contracts |
| Generic AdamW, optional gradient clipping, and optional auxiliary loss with prediction/equality/inequality weights | `training.py:StageTrainer`; `config.py:SearchSettings`; `tasks.py:TaskAdapter.adp_search_loss`, `auxiliary_loss_components` | `test_stage_optimizer_profiles_keep_documented_parameters_and_scheduler` checks AdamW, configured learning rate/weight decay, and absent scheduler; `test_gradient_clipping_is_generic_adamw_only` checks configured clipping is applied for AdamW and omitted for dedicated Adam; `test_generic_search_penalty_weights_follow_documented_gate_and_components` verifies the zero-penalty bypass and weighted composition; generic fit invokes the task penalty hook when enabled |
| Generic `max_candidates` is one mutation-count ceiling shared across ordered phases; zero means no explicit candidate-count ceiling | `config.py:SearchSettings`; `generic_search.py:GenericCandidateSearch.run` | `test_generic_max_candidates_is_a_shared_cross_phase_ceiling` checks both profiles: a cap of one allows the depth candidate and blocks the later width candidate, while zero allows one candidate in each phase until each phase's patience stops it; `test_depth_then_width_resume_preserves_global_candidate_ceiling` verifies an interrupted depth candidate resumes and does not exceed the shared cap or enter the later width phase; `test_generic_width_to_depth_resume_preserves_phase_transition` interrupts the first depth candidate after a completed failing width phase and verifies phase/candidate state plus resumed equivalence for both profiles; `test_generic_mixed_order_routes_resume_candidate_stage_exactly` adds interrupted candidate-stage resume equivalence for all eight mixed-order configurations under both faithful and corrected profiles |
| Generic `max(1, min_new_layer_width)`, caller-supplied widening/deepening functions, and stable mutation identity in checkpoint signatures | `config.py:SearchSettings`; `growth.py:expand_depth`; `generic_search.py:GenericCandidateSearch`; `search.py:SearchController`; `estimator.py:TabularADP` | `test_generic_mutation_hooks_and_minimum_depth_seed_are_configurable`; `test_generic_growth_primitives_clamp_nonpositive_size_arguments`; `test_estimator_custom_generic_mutation_is_checkpoint_identity` |
| `ex_k_depth` is retained as a configured/reportable alias while dedicated depth mutation appends one layer per call | `config.py:SearchSettings`; `growth.py:expand_depth` | `test_growth_mutations_cursor_depth_and_overlap_copy` asserts setting the alias does not change the one-layer proposal |
| Epoch-level early stop at `score < best - 1e-12`; stage-best restore; `max_epochs=0` means no explicit epoch cap | `training.py:StageTrainer.train_stage` | `test_epoch_early_stop_uses_strict_one_e_minus_twelve_margin`; `test_zero_epoch_limit_means_train_until_epoch_patience_stops` proves a zero epoch limit still trains until epoch patience expires; stage training and deterministic resume contracts |
| Completed search checkpoint short-circuits controller execution and restores the recorded global-best model/state | `search.py:SearchController.run` | `test_completed_search_checkpoint_restores_global_best_without_training` verifies architecture, cursor, weights, score, history, state, and that no stage starts |
| Same-stage resume of weights, optimizer, scheduler, RNG, generator, permutation, epoch, and batch cursor; checkpoint identity includes the mechanism behind a canonical ID | `training.py`, `checkpoint.py`, `estimator.py`, `config.py:ModelConfig.signature_dict` | `test_stage_optimizer_boundary_reset_and_same_stage_resume` interrupts and compares against uninterrupted training at all 16 reachable batch/epoch boundaries in a three-epoch, four-batch-per-epoch stage; `test_depth_candidate_resume_matches_uninterrupted_search` interrupts completed depth-candidate stages for both `ADP-D-D1` and `ADP-D-Dcur` and compares history, final forward architecture, and predictions with uninterrupted runs; progressive-depth catch-up, round-robin width, and nested-search resume tests cover their controller positions; checkpoint signature and malformed RNG contracts reject incompatible state. `test_estimator_resumes_interrupted_stage_exactly` additionally changes the registered width strategy while keeping the ID fixed and verifies exact-resume rejection. |
| Saved estimator artifacts retain and validate the registered algorithm mechanism as well as the model ID | `estimator.py:TabularADP.save`, `TabularADP.load` | Artifacts use format version 2 and persist the mechanism fields. `test_saved_estimator_rejects_same_id_with_changed_registered_mechanism` verifies loading rejects an artifact when its ID is unchanged but its registered growth strategy differs; `test_classification_regression_custom_task_and_roundtrip` checks prediction-preserving save/load. Older artifact formats are rejected rather than loaded without mechanism identity. |
| Fixed MLP as a supplied architecture grid, without invented grid coordinates | `dedicated.py:_fixed_search`, `estimator.py:fit` | fixed-grid rejection and supplied-grid fit in `test_registry_has_all_documented_ids_and_custom_fixed_grid`; candidate traversal order, strict best replacement on ties, and final global-best restoration in `test_fixed_grid_preserves_candidate_order_ties_and_global_best` |

## Canonical route identity

The registry contains exactly these identities and mappings:

| IDs | Mechanism |
|---|---|
| `LGND-W-Greedy` | All-layer width proposals; strict first-failure stop |
| `ADP-W-RR`, `ADP-W-All` | Dedicated patience width search with round-robin or all-layer mutation |
| `ADP-WD-RR-D1`, `ADP-WD-RR-Dcur`, `ADP-WD-All-D1`, `ADP-WD-All-Dcur` | Generic sequential width-then-depth candidate search; the suffix selects mutation and depth seed |
| `ADP-DW-D1-RR`, `ADP-DW-D1-All`, `ADP-DW-Dcur-RR`, `ADP-DW-Dcur-All` | Generic sequential depth-then-width candidate search; the suffix selects depth seed and mutation |
| `ADP-D-D1`, `ADP-D-Dcur` | Dedicated progressive width-1 catch-up or matched-width depth search |
| `MLP-Fixed` | Caller-supplied fixed architecture grid |

The nested DEN in the guide is a distinct controller mechanism, not an extra
canonical ID. It is exposed as `SearchController.run_nested_width_depth` and
does not change the 14-ID registry or the coordinated runner's route count.

## Fidelity boundaries

- The faithful generic profile intentionally preserves the guide's recorded
  `adp_search` defects. The corrected profile is separate and opt-in; it does
  not alter canonical IDs.
- The guide's §§4.5, 5.7, and 5.8 document behavior differences in duplicated
  output-head source files. This package intentionally has no head-count
  variants, as directed. It implements the guide's canonical single-output
  controller mechanics; head-copy-only manual RR increment/precheck quirks and
  intermediate-depth-best behavior are not separate routes. The guide reports
  the standard duplicated-model runners use all-layer widening, so those
  manual-override differences do not alter the canonical suite routes.
- Task adapters supply output interpretation, loss, validation score, and
  predictions. The default regression task uses MSE. Classification and custom
  tasks intentionally use their own task-appropriate loss and score.
- The package is a new domain-independent implementation of the described
  mechanisms. It does not claim line-for-line source identity with OPF_ADP or
  function-preserving architecture growth.
- Tests use deterministic synthetic stages and small CPU datasets. `tests/test_smoke_suite.py`
  executes each registered model on classification and regression with one
  epoch and a batch equal to the training partition; both faithful and corrected
  profiles are exercised for all eight mixed-order IDs. The fixed baseline uses
  its supplied 50-architecture grid, and both generic profiles also run through
  a custom task adapter. These contracts verify end-to-end wiring, not that a
  particular model improves accuracy on every benchmark dataset. On 2026-10-22,
  the current checkout passed `.venv/bin/python -m pytest -q
  tests/test_smoke_suite.py tests/test_runner_worker.py
  tests/test_runner_end_to_end.py` (**48 passed in 8.40 sec**). The worker
  smoke path overrides only epoch count to 1 and batch size to the number of
  training indices; validation and test rows remain outside that batch.

## Line-by-line reconciliation checkpoints

These bounded checkpoints record guide passages compared directly with the
implementation. A completed row does not certify the whole guide or every
possible numeric/custom configuration.

| Guide section | Reconciliation result | Evidence |
|---|---|---|
| §§8.1–8.10, generic candidate search | Compared the default mutation primitives, four ordered modes, candidate training profile, forward-candidate transitions, narrow depth eligibility, width-to-depth nonuniform termination, shared candidate ceiling, and three documented compatibility defects against `generic_search.py`, `growth.py`, and `training.py`. The `faithful` profile retains the update-before-compare failure counter, predecessor-relative selected-candidate bookkeeping, and omission of `max_neurons`; the separately named `corrected` profile uses consecutive phase failures, global-best margin selection, and the aggregate unit cap. These profiles do not add canonical model IDs. Both use AdamW; the generic training adapter enables the auxiliary-loss path and configured clipping independently of dedicated Adam/cosine stages. No discrepancy was found in this bounded comparison. The source's durable OPF task-directory layout is represented by this package's signed checkpoint and runner state surfaces rather than copied directory names. | `test_generic_faithful_and_corrected_global_best_and_capacity`, `test_generic_max_candidates_is_a_shared_cross_phase_ceiling`, the 16-case phase-transition matrix across all mixed-order IDs/profiles, the width and depth patience/margin matrices, the generic neuron-capacity profile matrix, generic optimizer/penalty/clipping contracts, and both-profile task smoke routes. |
| §§2.1–2.8 and 3.1–3.7, bounded reread | Compared the guide's ordered hidden-width representation, fresh-layer overlap copying, output projection resizing, unit-count capacity proxy, dedicated stage optimizer lifecycle, strict architecture margin, forward-state valley crossing, and final global-best restoration against `models.py`, `growth.py`, `training.py`, and `dedicated.py`. The implementation keeps architecture state separate from the returned best snapshot, initializes new tensor regions freshly, resets optimizer/scheduler state between architecture stages, and restores optimizer/scheduler/RNG only for same-stage resume. Dedicated validation uses the task adapter's score/loss surface so tabular classification and custom tasks remain supported; regression preserves the guide's MSE path. No discrepancy was found in this bounded comparison. This does not complete line-by-line reconciliation of the full guide. | `test_growth_mutations_cursor_depth_and_overlap_copy`, `test_architecture_growth_keeps_new_linear_cells_at_fresh_initialization`, `test_patience_delta_failure_traversal_and_final_global_best`, `test_architecture_improvement_equal_to_delta_is_a_failure`, `test_stage_optimizer_boundary_reset_and_same_stage_resume`, `test_controller_resets_adam_and_cosine_scheduler_at_architecture_boundary`, and `test_stage_optimizer_profiles_keep_documented_parameters_and_scheduler`. |
| §5.1–5.8, canonical single-output depth path | Compared depth eligibility, seed-width capacity, narrow insertion, terminal-only +1 catch-up, optimization-only stepping stones, completed-candidate margin acceptance, forward-state retention after failure/capacity stop, and the matched-width alternative against `growth.py` and `DedicatedController._depth_search`. The canonical paths match the guide. The alternate duplicated-output-controller catch-up quirks in §§5.7–5.8 remain outside this package's head-agnostic scope, as directed; they do not create additional canonical model IDs. | `test_progressive_and_matched_width_depth_proposals`, `test_progressive_depth_stepping_scores_do_not_replace_global_best`, `test_progressive_depth_capacity_stop_retains_last_forward_stepping_architecture`, the independent 864-case progressive/matched-depth controller oracle, interrupted-candidate resume equivalence for both dedicated depth variants, and `test_progressive_depth_growth_resumes_at_every_stage_boundary`. |
| §§4.1–4.4, canonical round-robin width DEN | Compared minimum-width scalar reporting, cursor-ordered first eligible layer, `+ex_k_width` proposal, exact increment rejection at the per-layer cap, `+k` unit-budget accounting, cursor advancement, patience/delta acceptance, failed-shape forward traversal, and final global-best restoration against the dedicated RR helper/controller. The canonical path matches the guide. The duplicated-controller-only `+1` override and all-layer capacity-precheck quirks in §4.5 are not added as head variants, consistent with the user's scope. | `test_growth_mutations_cursor_depth_and_overlap_copy`, `test_dedicated_round_robin_does_not_clip_configured_increment_at_width_cap`, `test_round_robin_den_traverses_failed_shape_from_cursor`, `test_round_robin_den_controller_matches_margin_patience_and_cursor_oracle`, and `test_round_robin_growth_cursor_survives_estimator_resume`. |
| §6, nested width/depth DEN | Compared with `DedicatedController.run_nested_width_depth`: outer width rounds contain inner depth sweeps; epoch, inner-depth, and outer-width patience are independent; nested depth insertion is width 1 unless current-width seeding is selected; the depth and width margins are applied at their separate levels; current forward state is separate from global best; and completion returns the global best. No controller-mechanics discrepancy was identified. OPF-specific stage-directory naming is not reproduced as an OPF telemetry layout because this package uses its own stage/checkpoint format. | `.venv/bin/pytest -q tests/test_search_training.py -k nested`: 11 passed. Includes the independent 3,456-case trajectory oracle, inclusive outer-width margin equality, both depth seeds, round-robin growth, and resume equivalence/mismatch checks. |
| §7, LG-ND | Compared with `DedicatedController._width_search(greedy=True)`: the capacity precheck assumes all-layer growth even under a round-robin override; strict score improvement ignores `delta_width`; the first non-improvement ends search; the failed candidate remains the forward state while the returned model is global best; and a failed resumed expansion stops before another proposal. No controller-mechanics discrepancy was identified. | `test_lgnd_stops_at_first_failure_and_advances_forward`, `test_lgnd_round_robin_override_keeps_all_layer_capacity_precheck`, and `test_lgnd_resumed_failed_expansion_stops_before_next_growth`. |
| §9, generic architecture mutations | Compared the generic and dedicated round-robin helpers, all-layer widening, progressive and matched-width depth insertion, cursor updates, width/depth limits, and overlap-copy rebuilding. New parameter cells retain fresh initialization; same-shaped and overlapping old tensors are retained. | `.venv/bin/pytest -q tests/test_core.py -k 'growth or checkpoint'`: 11 passed. Relevant contracts include `test_growth_mutations_cursor_depth_and_overlap_copy`, exhaustive small-domain width/depth and capacity matrices, BatchNorm/buffer overlap copies, and `test_architecture_growth_keeps_new_linear_cells_at_fresh_initialization`. |
| §10, checkpointing and search continuity | Re-read the guide's stage recovery sequence against `StageTrainer`, `SearchController`, `DedicatedController`, `GenericCandidateSearch`, `TabularADP.persist`, and `CheckpointStore`. A per-batch signed payload preserves current weights plus stage-best weights, optimizer/scheduler, Python/NumPy/Torch RNG, sampler generator/permutation, epoch/batch cursor, patience, and architecture cursor; controller context separately preserves current forward architecture and global-best snapshot. Completed top-level search restores global best without retraining; interrupted active stage rebuilds recorded widths and resumes; new architecture stages create fresh optimizer/scheduler instances. Data, preprocessing, task, complete model mechanism, training settings, profile, and fixed grid participate in the resume signature, and stage/current-architecture mismatch fails closed. The generic persistence format uses the signed estimator checkpoint as exact-resume authority and mirrors per-batch last plus stage-boundary best-validation snapshots into telemetry and coordinator job checkpoint paths. At a fresh stage boundary it initializes best-batch state from the initial architecture snapshot with infinite loss, matching the specified initial checkpoint. The guide and copied snapshot still define no later best-batch selection/update rule, so no post-initial update is invented. | `test_checkpoint_signature_rejection`, `test_estimator_custom_generic_mutation_is_checkpoint_identity`, invalid-checkpoint contracts, `test_completed_search_checkpoint_restores_global_best_without_training`, same-stage training/search resume contracts, `test_checkpoint_store_fsyncs_file_and_directory`, and the interrupted telemetry integration asserting local/coordinator last, best, and initial best-batch mirrors. |
| §§1, 11–19, appendices A–C | Checked taxonomy/count, algorithm comparison matrix, worked width/depth/LG-ND/generic traces, design-choice versus defect distinctions, reproducibility knobs, source-independent taxonomy, symmetry boundaries, 14 canonical definitions, pseudocode, parameter semantics, and scope notes against the registry/config/controllers. The 14 identities and 13-adaptive-plus-fixed count match. The code keeps sequential D→W separate from nested W→D, does not add the guide's explicitly absent symmetric nested D[W] controller, and keeps the documented generic defects isolated in the faithful profile. No registry/taxonomy discrepancy was identified. | `test_registry_has_all_documented_ids_and_custom_fixed_grid`, generic faithful/corrected profile trajectory and defect tests, nested independent-oracle tests, fixed-grid traversal, optimizer-profile assertions, checkpoint identity tests, and the canonical route table above. |

## Remaining verification boundary

The guide's 458 nonempty body blocks have been reread in document order and
compared by the implementation-bearing section ranges recorded below. The
generated [block crosswalk](ADP_ALGORITHM_BLOCK_CROSSWALK.md) provides a stable
entry for each paragraph/table row, its review range, source text, and
implementation modules; regenerate/check it with
`python3 scripts/generate_adp_block_crosswalk.py [--check]`. The separate
`GenericSearchArtifactStore` persistence contract writes candidate state,
best/last/context checkpoints, method/task state, replay metadata, and a signed
task manifest; recovery rejects incompatible signatures and confines artifact
paths. Artifact creation, completed-task recovery, mismatch rejection, and
resume at a durable candidate boundary are covered by
`tests/test_generic_artifacts.py`.

Verification covers every canonical identity's registered mechanism,
principal mutation, acceptance rule, capacity path, and resume path. Independent
oracles cover the explicitly enumerated small domains and trajectory matrices
documented in this audit, including the dedicated/generic RR cap boundary, the
3,456-case nested controller, the 864-case progressive/matched-depth controller,
and the 16-case mixed-order profile/resume matrix. Custom width/depth mutations
and custom-task resume are covered for both generic profiles. This is bounded
evidence, not an exhaustive proof over unbounded numeric settings, arbitrary
callbacks, every optimizer combination, every task adapter, every interruption
position, or live accelerator/runtime states. No OpenML training campaign or
live hardware pressure run is part of this verification, consistent with the
instruction not to launch the suite. The audit remains active for any remaining
bounded source/behavior gaps and should not be read as universal runtime
certification.

### Verification refresh: generic candidate artifacts (2026-10-22)

Rechecked guide §8.11 against `GenericSearchArtifactStore` and
`GenericCandidateSearch`: completed candidates write candidate state, best and
last snapshots, and search context; method/task/replay records are updated;
artifact identity is signed and mismatches fail; a candidate artifact can
resume construction of later candidates when the primary estimator checkpoint
is absent. Focused artifact contracts passed (**3 passed in 1.74 sec**), and
the full Linux suite passed (**18,430 passed in 100.44 sec**). This verifies
this persistence slice in the current checkout; it does not close the remaining
line-by-line guide reconciliation or establish exact replay across every
custom mutation and interruption point.

### Recovery validation follow-up (2026-10-22)

The §8.11 artifact reader now confines task and candidate checkpoint paths to
their owning artifact directories, including symlink resolution; it verifies
candidate directory/index/context identity and rejects artifact schema
mismatches. The artifact schema is now version 2 so the added context identity
cannot be mistaken for the older format. Focused ADP/core verification passed
(**548 passed in 9.69 sec**), and the full Linux suite passed (**18,433 passed
in 99.78 sec**, exit code 0). The runner function inventory also passed. This
closes the newly found artifact-path/schema gap; full line-by-line algorithm
reconciliation and all custom-resume combinations remain open.

### Bounded canonical taxonomy and search-order recheck (2026-10-22)

Re-extracted the complete canonical DOCX body, including tables (458 nonempty
paragraph/table-row blocks), and compared its 14-name table, mixed-order
factorization, sequential-versus-nested distinction, and stated symmetry gaps
with `registry.py`, `generic_search.py`, and the dedicated nested controller.
The registry preserves the 14 exact IDs and parameterized distinctions; the
Depth→Width registry routes execute the sequential generic phase order, while
the distinct outer-width/inner-depth method remains the dedicated nested
controller. No unlisted symmetric outer-depth/inner-global-width controller or
depth-greedy LG-ND route was added. Focused contracts passed (**4 passed in
3.15 sec**): the canonical registry/fixed-grid test, all-eight mixed-order
mutation test, 3,456-case nested-controller oracle, and faithful/corrected
generic defect-profile test. This verifies only the bounded taxonomy/order
slice; the guide's full paragraph-by-paragraph reconciliation and custom
settings/interruption space remain open.

### Custom generic width and depth mutation resume across both profiles (2026-10-22)

Compared the guide's requirement that custom `widen_fn` and `deepen_fn` behavior
and checkpointed forward state determine the next proposal with
`GenericCandidateSearch.run` and `TabularADP.fit`. Added
`test_custom_generic_mutation_resumes_candidate_without_reapplying_mutation`
for both callbacks under faithful and corrected profiles. It interrupts inside
the callback-produced candidate, resumes from saved candidate weights and
optimizer state without invoking the callback again for that active candidate,
and compares mutation-call order, history, search state, and predictions with
an uninterrupted run. Added
`test_custom_task_adapter_resumes_generic_candidate_stage` to cover a custom
MAE task adapter restoring the active candidate under each profile. The
combined focused tests passed (**6 passed**), and the full Linux suite passed
(**18,447 passed in 101.79 sec**). These verify bounded custom width/depth
mutation and custom-task resume paths; arbitrary callback behavior, all
interruption positions, and arbitrary optimizer combinations remain open, as
does the full paragraph-by-paragraph ADP guide reconciliation.

### Shared architecture and growth mechanics recheck (2026-10-22)

Compared the guide's §§2.1–2.4, 3.1–3.7, 4.1–4.4, 5.1–5.6, 5.8, and 9.1–9.4
with `models.py`, `growth.py`, `dedicated.py`, and `generic_search.py`. The
ordered hidden-width vector and architecture snapshots preserve the RR cursor;
resizing copies the origin-aligned overlap for each same-rank state tensor,
including the output projection and generic BatchNorm state, while added cells
retain fresh initialization. Dedicated capacity counts hidden plus output
units. Dedicated all-layer widening proposes `+ex_k_width` on every layer and
rejects over-cap proposals; dedicated RR reports the minimum width, selects
the first cursor-ordered layer below `minimum + k`, proposes exact `+k`, and
rejects an over-cap increment. Generic RR instead selects the first
cursor-ordered layer below `max_width` and clamps the increment to its ceiling;
generic all-layer growth also clamps. Depth DEN eligibility requires a
nonempty uniform stack and available depth/unit budget. Canonical progressive
depth trains narrow terminal-layer stepping stones without allowing their
scores to replace the global best, then compares the completed-width candidate;
the current-width variant appends the previous terminal width directly. The
duplicated-head all-layer catch-up divergence in §5.8 is excluded with the
head-specific implementation differences, as directed; it is not claimed to be
implemented by the generic nested controller. No head-count routes were
introduced. No discrepancy was found in the canonical, head-agnostic mechanics
within this bounded comparison.

Focused Linux verification after this source review:

```text
.venv/bin/python -m pytest -q tests/test_core.py tests/test_search_training.py \
  -k 'growth or depth or lgnd or patience or capacity'
469 passed, 73 deselected in 6.74 sec
```

This confirms the selected growth, capacity, depth, LG-ND, and patience
contracts in the current checkout. It does not close the remaining guide
paragraph reconciliation, all optimizer/task/custom-callback combinations,
or the runner-ledger audit.

### Generic candidate search and artifact recovery recheck (2026-10-22)

Compared the guide's §§8.1–8.11 and the generic mutation details in §§9.1–9.4
with `generic_search.py`, `growth.py`, `generic_artifacts.py`, and the estimator
checkpoint path. The phase table is width, depth, width→depth, or depth→width;
each transition starts from the most recently trained forward candidate. The
faithful profile preserves the documented update-before-compare failure
counter, predecessor-relative candidate selection, and missing `max_neurons`
guard. The corrected profile is separate: phase failures use a margin against
the current candidate score, selection compares against the retained global
best, and the unit cap is enforced. `max_candidates` is one mutation-count
ceiling shared by both phases; zero means no explicit ceiling. Default depth
insertion requires a uniform stack and does not catch up the new layer, so a
width-greater-than-one stack normally cannot take a second narrow-depth step;
default width→depth can likewise stop at depth when the width phase leaves a
nonuniform stack. Built-in and custom mutation identity participate in
checkpoint identity.

The artifact store writes a signed manifest, per-candidate state plus best,
last, and search-context checkpoints, method/task/replay views, and a terminal
task-best snapshot. Resume first honors a valid exact estimator checkpoint;
otherwise the artifact reader can load a completed task or reconstruct the
forward candidate and controller context from the latest completed candidate.
Signature, schema, candidate directory/index, context identity, and checkpoint
path containment are validated before loading. This matches the guide's
coarse-grained candidate recovery contract while keeping exact in-stage resume
under `CheckpointStore`.

Focused Linux verification:

```text
.venv/bin/python -m pytest -q tests/test_search_training.py \
  tests/test_generic_artifacts.py tests/test_runner_contracts.py \
  -k 'generic or candidate_artifact'
454 passed, 17655 deselected in 5.00 sec
```

This verifies the selected generic profiles, phase transitions, candidate
artifacts, mutation identity, and bounded resume paths. It does not establish
every possible custom callback, task adapter, optimizer, or interruption
position, and the complete guide-to-code reconciliation remains open.

### Stage training and exact same-stage resume recheck (2026-10-22)

Compared guide §§2.5–2.7 and 10.1–10.3 with `StageTrainer`,
`CheckpointStore`, `SearchController`, and the estimator checkpoint bridge.
Each fresh architecture stage constructs a new Adam plus cosine-warm-restart
scheduler for dedicated DEN/LG-ND; generic candidates use AdamW without that
scheduler and can enable the adapter penalty and configured gradient clipping.
The trainer uses the task loss/validation adapter, with regression retaining
MSE. Dedicated stage early stopping uses the strict `score < best - 1e-12`
rule, snapshots the stage-best weights, and restores those weights when the
stage finishes. Within a stage, checkpoint state includes current and best
weights, optimizer/scheduler state, patience, epoch and batch cursor, shuffled
permutation, loader generator, and Python/NumPy/Torch/CUDA RNG state. Resume
checks the exact architecture and fails closed on missing or incompatible
state; the checkpoint envelope also binds the full estimator/data signature.
This preserves the guide's boundary: optimizer state resets for a new
architecture, while an interrupted architecture stage restores its optimizer,
scheduler, and random/data position. The separate child telemetry bridge
records stage identity and mirrors checkpoint state; its existing source audit
and tests remain the evidence for runner-side sidecar reconciliation.

Focused Linux verification:

```text
.venv/bin/python -m pytest -q tests/test_search_training.py tests/test_core.py \
  tests/test_runner_child_telemetry.py \
  -k 'optimizer or resume or checkpoint or rng or sidecar or stage_boundary or scheduler_profile'
59 passed, 511 deselected in 2.75 sec
```

This verifies the selected optimizer lifecycle, exact stage resume, signature,
RNG, and sidecar contracts. It does not prove restore behavior across every
device/runtime or arbitrary user-defined task/optimizer combination; the
remaining guide and runner-ledger audits stay open.

### Paragraph-level reread: shared mechanics and all-layer/RR width DEN (2026-10-22)

Re-read canonical DOCX blocks 067–144 in document order, including the
architecture and resizing equations, capacity definitions, dedicated stage
training pseudocode, forward-state trace, round-robin table, and the three
duplicated-controller divergence notes. Compared the implementation-bearing
claims directly with `models.py`, `growth.py`, `training.py`, and
`DedicatedController._width_search`. The ordered architecture vector and
cursor are snapshotted; linear and BatchNorm tensors copy same-rank overlap
while new cells keep fresh initialization; unit capacity includes output units;
dedicated all-layer growth rejects over-cap `+k` proposals; dedicated RR uses
the minimum-width target and cursor-ordered exact `+k` proposal; both retain a
failed candidate as forward state and restore global best only on return. Each
dedicated architecture stage creates fresh Adam/scheduler state, uses the
adapter's task objective (MSE for regression), and same-stage resume restores
its optimizer/scheduler and data/RNG cursor. LG-ND's duplicated-source-only
quirks remain excluded from this canonical single-output implementation, as
recorded in the preceding section. No mismatch was found in these blocks.

Focused current-checkout verification: `.venv/bin/python -m pytest -q
tests/test_core.py -k 'growth or stage or optimizer or patience or depth'` —
**5 passed, 20 deselected in 1.36 sec**. The broader independent trajectory,
mutation, stage-boundary, and resume contracts cited in the table above remain
the evidence for their respective finite domains. The subsequent paragraph-level
reviews below cover blocks 145–457; the old note above is superseded. Those
reviews complete the line-ordered guide reconciliation, while the remaining
limits concern unbounded numeric, arbitrary custom-task/callback, and live
runtime combinations rather than unread guide blocks.

### Paragraph-level reread: depth, nested DEN, and LG-ND (2026-10-22)

Re-read DOCX blocks 145–207 in order, covering canonical progressive versus
matched-width depth, the duplicated depth-controller divergences, nested
outer-width/inner-depth ordering and its distinct width-1 depth insertion, and
LG-ND's strict first-failure semantics and all-layer capacity precheck. Compared
these claims with `DedicatedController._depth_search`,
`DedicatedController.run_nested_width_depth`, `DedicatedController._width_search`,
and the growth/capacity helpers. The pure progressive controller trains
intermediate narrow-layer stepping stones without comparing them to global
best, then trains and judges the completed width; the nested controller
inserts width 1 without progressive catch-up; matched-width insertion uses the
current terminal width. Nested depth and width failure counters remain
separate, outer width acceptance uses the inclusive `>= delta_width` boundary,
and its resume signature binds data/settings/seed mode/width strategy. LG-ND
uses strict score improvement, ignores the width margin for acceptance, checks
all-layer capacity even for its supported RR override, and stops after the
first failed candidate, including after a failed resumed candidate. Duplicated
output-layout-only divergences remain excluded as specified by the user.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_search_training.py -k 'nested or depth or lgnd'` — **220 passed,
297 deselected in 7.62 sec**. This confirms the enumerated depth, nested, and
LG-ND controller tests in the current checkout. It does not close generic
search or the remaining guide blocks; the paragraph-level review continues.

### Paragraph-level reread: generic candidate search and mutation primitives (2026-10-22)

Re-read DOCX blocks 208–262 in order: candidate artifact lifecycle, four
phase orders, AdamW/penalty training distinction, forward-candidate behavior,
the documented failure-counter/global-best/capacity defects, and exact generic
mutation pseudocode. Compared these against `generic_search.py`, `growth.py`,
`training.py`, `tasks.py`, and `artifacts.py`. The default generic RR cursor
chooses the first below-cap layer from the cursor and clamps `+max(1, ex_k)`;
all-layer growth clamps each layer; narrow depth requires uniformity but has no
catch-up; D→W remains sequential and differs from the dedicated nested
controller. Faithful mode keeps the guide's update-before-compare failure
counter, predecessor-relative selection, and missing aggregate-unit-cap
check; corrected mode is explicit and separate. Both paths preserve
forward-candidate transitions and candidate-boundary artifacts, while custom
mutation hooks are part of checkpoint identity. No discrepancy was found in
these blocks. Task adapters generalize the guide's MSE-plus-application-penalty
surface without changing the search phases.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_search_training.py -k 'generic or candidate'` — **428 passed,
89 deselected in 2.71 sec**. This covers the selected generic profiles,
mutation, phase, artifact, and bounded resume contracts. It does not close the
checkpoint-guide blocks or remaining explanatory/reproducibility blocks.

### Paragraph-level reread: checkpoints, taxonomy, traces, and parameter semantics (2026-10-22)

Re-read DOCX blocks 263–457 in order, including stage-recovery pseudocode,
comparison and worked-trace tables, issue classifications, search-graph
interpretation, every reproducibility knob, source/runner mapping, symmetry
gaps, the canonical 14-name table, Appendix A pseudocode, Appendix B parameter
semantics, and Appendix C scope/authority notes. Compared executable claims
with `CheckpointStore`, `TabularADP`, `StageTrainer`, `DedicatedController`,
`GenericCandidateSearch`, `MODEL_REGISTRY`, and fixed-grid/generic artifact
contracts. Resume identity binds model mechanism, settings, data/split and
task/preprocessing state; active stages restore current and best state plus
optimizer/scheduler and cursor, while completed searches restore global best.
The 14-name equation and route matrix match the registry; sequential D→W is
distinct from nested W→D; no symmetric nested depth/width or depth-greedy
counterpart is introduced. The fixed route requires caller-provided grid
coordinates. Parameter semantics distinguish dedicated `ex_k_depth` (recorded
knob, one inserted layer per call) from generic mutation controls, and generic
`max_candidates=0` means no explicit ceiling. Worked numerical traces are
consistent with the tested controller rules. Interpretive prose and the
source-lock notes are treated as specification context, not additional runtime
mechanisms. No mismatch was found in this block range.

Focused Linux verification: `.venv/bin/python -m pytest -q
tests/test_search_training.py tests/test_core.py tests/test_generic_artifacts.py
-k 'checkpoint or resume or fixed_grid or registry'` — **40 passed, 508
deselected in 2.54 sec**. Combined with the paragraph-level reviews of blocks
000–262, all 458 extracted nonempty guide blocks have been reread and assigned
to the section-level review. The generated block crosswalk supplies the
previously missing index; it does not extend finite tests to arbitrary settings.

### Paragraph-level reread: taxonomy, scope, and notation (2026-10-22)

Re-read DOCX blocks 000–066, including title/source-lock context, the complete
14-row taxonomy, mixed-order factorization, head-abstraction rule, dedicated
versus candidate-search distinction, normalized controller table, source-file
roles, and notation. Compared the experiment identities to `registry.py` and
the registry contract; checked that width/depth order, RR/all-layer strategy,
depth seed, and fixed baseline remain distinct configuration fields. The
equation `3 + 4 + 4 + 2 + 1 = 14` agrees with the registry (13 adaptive IDs
plus fixed MLP). OPF file names and output-head copies are source-location
context only; their algorithmic divergence is recorded separately and no head
variants are registered. The definitions of `A`, `D`, strategy-dependent `W`,
`kW`, `kD`, validation score, margins, both patience levels, and unit-count
`Nmax` map to model/search settings and the implementation notes above. No
taxonomy or notation mismatch was found.

Focused Linux verification: `.venv/bin/python -m pytest -q tests/test_core.py
-k 'registry or fixed_grid'` — **1 passed, 24 deselected**. All 458 extracted
nonempty blocks have now been re-read and assigned to the section-level review
above. A generated one-entry-per-block index is maintained in
[`ADP_ALGORITHM_BLOCK_CROSSWALK.md`](ADP_ALGORITHM_BLOCK_CROSSWALK.md), with a
freshness test. The complete Linux suite has since passed (**18,449 passed in
101.25 sec**, 2026-10-22). Remaining acceptance boundaries are the arbitrary
numeric/custom/runtime spaces listed above, not an unindexed source block.

The latest full Linux suite also passed after a runner recovery-default
regression test and generated function-map refresh: **18,450 passed in 106.01
sec** (2026-10-22). The ADP algorithm source and tests were unchanged by that
runner-only update; the arbitrary numeric/custom/runtime acceptance boundaries
above remain.

The subsequent full Linux rerun after extending the runner recovery persistence
contract passed **18,450 tests in 105.99 sec** (2026-10-22). No ADP algorithm
source changed in this runner-only audit slice.

The 2026-10-23 full Linux suite passed **18,451 tests in 112.16 sec** after
adding a runner source-differential test; no ADP algorithm code changed in that
audit slice.

The subsequent 2026-10-23 full Linux suite passed **18,451 tests in 105.69
sec** after a runner selector fidelity correction; no ADP algorithm code
changed in that slice.

The subsequent full Linux suite passed **18,451 tests in 104.60 sec** after
extending the runner source-differential contract to selector and capacity
behavior. ADP implementation files were unchanged.

The 2026-10-23 full Linux suite subsequently passed **18,451 tests in 106.89
sec** after expanding the runner source-differential test to pressure-resume
readiness. No ADP algorithm code changed.

The latest full Linux suite passed **18,451 tests in 108.96 sec** after adding
runner gate-transition differential coverage; the ADP implementation remained
unchanged.

The 2026-10-23 runner audit added a direct differential for pressure-victim
filtering and ordering: **360** cases compared the copied source's executable
candidate-selection statements with the generic coordinator across GPU,
host, and combined pressure and all candidate input-order permutations. The
focused reference snapshot and differential tests passed, and the full Linux
suite passed **18,451 tests in 110.16 sec**. This runner-only audit did not
change ADP model code.

### Canonical estimator smoke refresh (2026-10-23)

Re-ran `tests/test_smoke_suite.py` together with the core growth and search
controller contracts: **588 passed in 11.23 sec**. The smoke file alone passes
**46 cases in 2.80 sec**. Its matrix traverses every registered model, both
faithful and corrected profiles where applicable, classification and
regression, one epoch with a full training batch, and save/load prediction
round trips on the representative `ADP-W-All` route. Two custom-task profile
cases also run. This is end-to-end route wiring evidence; the algorithm
fidelity claims remain the bounded controller, growth, optimizer, and resume
comparisons above, not an exhaustive proof over arbitrary settings/tasks.


### Current audit reconciliation status (2026-10-23)

Re-read both user-supplied runner attachments before continuing. The ADP audit
contains paragraph-level reviews covering guide blocks 000–066, 067–144,
145–207, 208–262, and 263–457; together these cover all 458 extracted
nonempty guide blocks in order. Verified the generated block crosswalk with
`python scripts/generate_adp_block_crosswalk.py --check`. The source-to-code
reconciliation is complete for the guide text and its stated mechanisms;
verification remains finite for numeric trajectories and does not prove every
arbitrary task adapter, custom callback/optimizer, interruption point, or live
accelerator/runtime state. No ADP implementation files changed in this turn.

### Current focused verification refresh (2026-10-23)

The current checkout's ADP smoke, core mutation, controller, checkpoint, and
generic-artifact tests passed together (**594 passed in 10.27 sec**). The full
Linux suite, including these ADP contracts, passed **18,455 tests in 107.35 sec**.
The generated guide crosswalk is current. No ADP code changed during this
runner-launch audit; these results refresh integration evidence and do not
extend the bounded-domain claims above.


The complete Linux suite passed **18,456 tests in 108.43 sec** after the
runner source-executing retry comparison. ADP route smoke and algorithm tests
remain included in that suite; no algorithm code changed in this runner audit
slice.


The refreshed complete Linux suite passed **18,457 tests in 109.93 sec** after
adding the source-executing host-pressure pause/resume comparison. Generated
runner and ADP maps are current.

The full Linux suite passed **18,460 tests in 111.96 sec** after adding a real
Linux process-group stop/continue case for the standalone pressure wrapper.
The current checkout's acceptance-focused tests also passed: **46 model/task
smoke tests in 2.55 sec** and **542 core, growth, controller, and checkpoint
tests in 10.18 sec**. These reruns confirm the existing ADP contracts remain
green; ADP model and algorithm files were not changed, so they add no new
ADP-specific fidelity cases.

### Full-suite verification refresh (2026-10-23)

After removing an exploratory real-child pressure comparison whose generic
child completed during the synthetic pressure window, the complete Linux
suite passed **18,460 tests in 112.39 sec**. The source fidelity module passed
**7 tests in 11.00 sec** independently. No ADP implementation or test contract
changed in this refresh; it updates execution evidence only. The finite-domain
and custom-runtime limits above remain.

### Full Linux verification refresh (2026-10-23)

After adding the real-child source/generic central pressure restart comparison,
the full Linux suite passed **18,461 tests in 114.64 sec**. The source-fidelity
module passed **8 tests in 13.11 sec**. No ADP implementation changed in this
runner verification step; algorithm-domain claims and their finite/custom-task
limits are unchanged.

### Full Linux verification refresh (2026-10-23)

After additional coordinated-runner source comparisons, the full Linux suite
passed **18,463 tests in 116.46 sec**. The source-fidelity module passed **10
tests in 15.71 sec**. ADP implementation files were unchanged in this runner
step; model algorithm verification remains as bounded in the guide audit.

### Linux suite refresh after coordinated-runner artifact crosswalk (2026-10-23)

The full Linux suite passed **18,464 tests in 117.60 sec** after adding a
source-executing report-field mapping contract for the coordinated runner.
The ADP algorithm code and ADP-specific contracts were unchanged in this
runner-only slice. This refresh confirms the existing bounded ADP fidelity
contracts remain green; it does not expand their numeric, custom-task,
callback, optimizer, interruption, or hardware coverage.

### Fidelity audit refresh (2026-10-23)

Rechecked the audit boundary against the canonical guide and its generated
458-block crosswalk. The implementation-bearing guide sections are mapped to
the registry, dedicated and generic controllers, architecture mutations,
optimizers, checkpoints, and task adapters. The full Linux suite passed
**18,472 tests in 126.13 sec**; both generated inventories passed their
`--check` commands. This confirms current regression evidence for the registered
ADP algorithms and their integration with the runner. It does not convert the
bounded trajectory matrices into a proof for every numeric setting, custom
callback/optimizer, task, interruption timing, or live accelerator state.

The coordinated-runner smoke audit additionally exercises each of the 50
individual fixed-MLP routes through `SimpleADPModelAdapter` for both built-in
task kinds, with one epoch and a full training-partition batch. The 100-case
matrix passed in 2.56 sec; the complete Linux suite passed **18,572 tests in
127.62 sec** after this addition. This supplements the canonical ADP registry
route smoke tests and does not substitute for live OpenML campaigns.

The runner adapter also now smoke-fits all 13 non-fixed canonical ADP
identities, with both generic profiles where applicable, on classification and
regression fixtures (**42 cases**). The focused model/smoke/end-to-end group
passed **194 tests in 10.17 sec**, and the complete Linux suite passed **18,614
tests in 125.82 sec**. This adds task and split adapter coverage around the
already-audited algorithm mechanisms; it does not claim live dataset coverage.

The coordinator integration fixture now also registers a custom model adapter
through the package entry-point mechanism and executes it in a real worker
subprocess alongside built-in routes. The plugin route completes and persists
its result through the same coordinator path. Focused plugin/end-to-end checks
passed (**2 tests in 9.38 sec**) and the full Linux suite passed (**18,614 tests
in 130.84 sec**). This validates the extension seam and process handoff, not the
algorithmic correctness of arbitrary third-party adapters.

The coordinator's aggregate `MLP-Fixed` route now also has runner-adapter smoke
coverage on both classification and regression fixtures. Each one-epoch,
full-training-batch run supplies all 50 grid architectures and asserts 50
candidate history entries. The aggregate route plus individual vanilla-route
selection passed **102 tests in 2.97 sec**. The complete Linux suite then
passed **18,616 tests in 129.85 sec**. Generated runner and ADP inventories
passed their `--check` commands, and source/test compilation completed.

### ADP regression refresh (2026-10-23)

The combined ADP algorithm, model-route, end-to-end, standalone pressure, and
source-fidelity group passed (**754 passed in 55.30 sec**). The complete Linux
suite passed (**18,645 passed in 147.05 sec**); the generated guide crosswalk,
runner function inventory, compilation, and immutable reference-snapshot
integrity test passed. No ADP implementation changed in this audit
continuation, so this refresh confirms existing registered-model contracts
remain green without extending the finite-domain, custom-runtime, optimizer,
or interruption-position claims documented above.

### Worker-boundary route smoke matrix (2026-10-23)

Added and passed **292 worker route/task smoke cases** (**21.78 sec**) using
serialized route specifications, the real built-in model adapter, synthetic
classification/regression data, one epoch, and a full training-partition
batch. The routes include the 14 canonical identities with both generic
profiles where applicable, aggregate and 50 individual fixed architectures,
and diagnostic screen architectures/seeds. The complete Linux suite passed
**18,937 tests in 166.73 sec**. This strengthens worker handoff coverage; it
does not extend algorithm-oracle domains or certify live OpenML tasks, custom
task adapters at scale, or accelerator execution.

### Combined algorithm and Linux source-differential audit refresh (2026-10-23)

Rechecked the canonical DOCX identity against the user's Documents copy; both
files have SHA-256
`5a949387d9709e3d09001893e413a22d0c4c57f16a71470786a2cd5bb7861948`.
The generated 458-block guide crosswalk and 272-definition runner function
inventory both pass their `--check` commands. The focused algorithm,
checkpoint/artifact, crosswalk, and copied-runner source-differential group
passed **594 tests in 54.82 sec**. That group exercises the registered ADP
mechanisms, bounded independent trajectory oracles, exact-resume contracts,
and the source/generic Linux scheduler and process traces. No mismatch was
found in this pass, and no implementation code changed. This refresh does not
prove equivalence for arbitrary controller interleavings, unbounded numeric
settings, every third-party callback/optimizer, or live GPU/host pressure.

### Linux regression refresh after the OOM/pressure trajectory (2026-10-23)

Reran the focused Linux group covering copied-runner differential behavior,
model routes, and the generated ADP guide crosswalk: **196 passed in 53.71 sec**.
The complete Linux suite then passed **18,957 tests in 179.49 sec**. The runner
function map and ADP block crosswalk both pass freshness checks;
`git diff --check` and the immutable runner snapshot test also passed. No ADP
algorithm implementation changed in this refresh. These results confirm
regression status for the checked suite; the bounded numeric/custom-runtime
and live hardware limits above remain.

The full Linux suite was rerun after adding the runner reference-independence
guard and passed **18,958 tests in 182.85 sec**. The guard is structural runner
coverage; no ADP implementation changed in that run.

The subsequent Linux suite, including the real-worker pressure-pause and
estimator-checkpoint resume integration, passed **18,959 tests in 184.67 sec**.
The strengthened integration now pauses while the worker is held immediately
after a committed positive batch cursor, reads that authoritative checkpoint
at the pressure decision, and asserts that the relaunched process consumes the
same cursor. Its focused test passed (**1 passed in 9.95 sec**), and the full
Linux suite passed again (**18,959 tests in 191.48 sec**). This adds a
process-level handoff regression contract; it does not expand the bounded ADP
algorithm oracle domains or add physical-pressure evidence.

### Combined ADP and Linux runner fidelity refresh (2026-10-23)

Reran the focused algorithm and source-differential group:
`.venv/bin/python -m pytest -q tests/test_core.py tests/test_search_training.py tests/test_runner_source_fidelity.py tests/test_runner_reference_snapshot.py tests/test_adp_block_crosswalk.py`
passed **591 tests in 56.21 sec**. This refreshes registered ADP mechanism,
controller, checkpoint, copied-runner differential, immutable-reference, and
generated-guide-crosswalk contracts together. It does not close the documented
bounded numeric/custom-callback/interruption limits or certify arbitrary
pressure-controller timelines or physical host/GPU pressure.


### Full Linux suite refresh after ledger reread (2026-10-23)

The complete current Linux suite passed **18,961 tests in 190.17 sec**. The
immutable reference snapshot and generated 458-block ADP crosswalk passed
their focused contracts (**3 tests**); both generated inventories are current
and `git diff --check` is clean. No ADP algorithm implementation changed in
this refresh. The bounded oracle, custom callback/task, arbitrary interruption,
and live accelerator limits remain open.


### Linux runner regression refresh after nested watchdog correction (2026-10-23)

After correcting the outer wrapper's recursive TERM/KILL order for nested
process trees, the complete Linux suite passed **18,961 tests in 187.94 sec**.
The change does not modify ADP model algorithms. The bounded ADP algorithm
coverage and its stated custom-runtime limits are unchanged.


### Full Linux regression refresh after watchdog source differential (2026-10-23)

The current full Linux suite passed **18,962 tests in 190.43 sec** after adding
an executing source differential for the Linux outer watchdog. No ADP model
implementation changed; the bounded algorithm-oracle and live accelerator
limits remain unchanged.

### Full Linux regression refresh (2026-10-23)

The full current suite passed **18,962 tests in 189.17 sec**. Generated ADP
block crosswalk and runner function map checks are current. This adds no claim
of universal behavioral proof: algorithm fidelity remains supported by the
458-block guide crosswalk, independent trajectory oracles, registry and
one-epoch wiring tests, and checkpoint/resume contracts; arbitrary custom
runtime behavior and live accelerator execution are outside this evidence.

### Runner source-differential addition (2026-10-23)

The source-fidelity tests now also execute both central controller loops for an
active runtime with a missing process handle. Both drop the active record while
preserving unknown exit state and zero completion/retry/failure counts. This is
runner-only evidence and does not change the ADP algorithm audit boundary.

The full Linux suite passed **18,963 tests in 191.68 sec** after that runner
contract was added. No ADP implementation was changed by the runner-only test.

### Central process-tree termination source differential (2026-10-23)

An executing helper-level differential now compares copied-source and generic
central process-tree termination on the same nested tree and survivor sequence.
Descendant TERM order, wait arguments/timeouts, KILL order, and final root wait
match. Focused source contracts for termination plus the missing-handle
controller path passed (**2 passed**); this is a bounded deterministic trace.

The full Linux suite passed **18,964 tests in 191.12 sec** after adding the
central termination source differential. ADP source and model implementation
were unchanged in this runner-only addition.

A second central termination differential now covers the source's early-return
and exception boundaries: exited root, root PID lookup failure, and unexpected
descendant enumeration failure. The copied source and generic helper produce
identical side effects and propagation. Together with the normal escalation
trace, the focused termination contracts passed (**2 passed**).

The full Linux suite passed **18,965 tests in 193.77 sec** after the added
termination failure-boundary source contract. This update changes no ADP model
implementation.

An additional runner-only source differential verifies TERM/KILL exceptions
are swallowed and cleanup proceeds through both waits and the final root wait;
the focused termination contracts now pass (**3 passed**). The full Linux
suite passed **18,966 tests in 199.96 sec**; generated inventory checks and
`git diff --check` also passed.

The runner audit found and corrected a standalone-wrapper cleanup mismatch:
generic shutdown now performs the source's unconditional second process wait
and configured-timeout root wait, through a dedicated helper distinct from the
central scheduler terminator. Source differential plus pressure/process tests
passed (**47 passed**). The full Linux suite passed **18,968 tests in
201.14 sec**; immutable snapshot verification, generated inventories, and
`git diff --check` passed.

An additional full-loop startup differential compares the copied runner and
generic coordinator for a persisted ordinary failure at `max_retries` and a
terminal `inexact_checkpoint_resume` failure. Both relaunch only the ordinary
route; it completes without incrementing the retry count. This supplements
the existing 66-state reconciliation matrix and generic queue contract. The
full Linux suite passed **18,969 tests in 204.59 sec**; immutable snapshot
verification, both generated inventories, and `git diff --check` passed.

The controller source-fidelity pass added a full-loop success-at-hard-pressure
comparison: both the copied runner and generic coordinator reap a successful
child before victim selection on the same hard-pressure sample. It passed with
the focused termination comparisons (**3 passed**). The full Linux suite
passed **18,967 tests in 201.21 sec**; generated inventories and
`git diff --check` also passed.

### Full fidelity regression rerun (2026-10-23)

Following the request to proceed with the combined ADP-model and Linux
pressure-aware-runner fidelity audit, the complete suite passed **18,969 tests
in 202.25 sec**. The two generated inventories pass their freshness checks, and
the immutable copied-runner snapshot test passed (**2 tests**). This includes
the route-grid source differential (seven configurations spanning surface and
task filtering, ADP/vanilla repeat asymmetry, unknown tasks, backlog, include
flags, deterministic source ordering, fixed-MLP grid routes, and post-generation
slicing), plus the ADP mechanism, worker, process, state/retry, pressure, and
checkpoint contracts. This remains regression evidence, not a proof across all
possible schedules, numeric inputs, custom adapters, or physical pressure
conditions. No live dataset campaign or live GPU/host-pressure run was launched.

### Linux runtime-tuning source differential refresh (2026-10-23)

Added a source-executing comparison of the copied and generic child CPU
environment builders across 2,250 combinations of concurrency hints, shared
mode, inherited affinity values, explicit slots, and stable job keys. All
compared thread, worker, affinity, and shared-mode settings matched. The full
Linux suite passed **18,970 tests in 204.91 sec** after the new differential;
the generated runner inventory, ADP guide crosswalk, and `git diff --check`
were current. This closes the helper-level child CPU environment comparison
for that finite matrix, but does not certify best-effort OS tuning or physical
GPU behavior.

### Full Linux regression after runtime-tuning differentials (2026-10-23)

The child CPU environment differential (2,250 cases) and bootstrap differential
(five configurations) both pass against the copied Linux source. The complete
suite passed **18,971 tests in 205.49 sec** with those contracts included.
The runner function inventory and ADP guide crosswalk were current, and
`git diff --check` passed. The comparisons mock OS mutation hooks; no physical
pressure or GPU campaign was run.

### Combined focused fidelity refresh (2026-10-23)

Reran the focused copied-source runner differentials, ADP growth/search
contracts, and one-epoch model/task smoke routes with
`.venv/bin/python -m pytest -q tests/test_runner_source_fidelity.py
tests/test_core.py tests/test_search_training.py tests/test_smoke_suite.py`:
**644 passed in 73.99 sec**. The runner function inventory, 458-block guide
crosswalk, and `git diff --check` all pass. This refresh confirms those bounded
contracts on the current checkout; it does not close the documented partial
runner integration rows or certify arbitrary numeric/task configurations or
live host/GPU pressure behavior.
