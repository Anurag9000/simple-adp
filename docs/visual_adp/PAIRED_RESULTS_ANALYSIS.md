# Descriptive paired fixed-versus-growth visual metrics

This is an **analysis of signed worker summaries**, not a training pipeline.
It reads existing original `SuitePlan`/`run_spec` results and never
trains a candidate, changes an ADP controller, or opens a held-out dataset.

## Why pairing is necessary

Each native route is evaluated twice on the same task/split and derived
runner seed: once with the source `MLP-Fixed` configuration and once
with `ADP-W-RR`. Comparing their raw metrics across unrelated splits,
class vocabularies, seeds or growth sites would be invalid. The auditor
therefore requires a *complete* fixed/RR pair for all 19 declared
architecture/site variants within each model seed, a single signed
dataset fingerprint, and one signed split checksum across seeds.

The analysis also requires globally distinct worker job IDs and all
jobs completed successfully. It refuses missing, nonfinite and
out-of-range metrics. It does not claim to cryptographically authenticate
an arbitrary supplied JSON file: authenticity still depends on the
original signed worker specifications, result artifacts and CI records.

## Running on an existing matrix

```bash
python scripts/visual_paired_metrics_analysis.py \
  --matrix /path/to/official_fashion_vision_matrix_2seeds.json
```

This writes
`official_fashion_vision_matrix_2seeds_paired_metrics.json`
alongside the input, leaving the signed source summary unchanged.
`--output /path/to/report.json` selects a separate destination.

The same CLI also accepts a complete single-seed Fashion/CIFAR matrix
or the one-seed step-balanced matrix. For a complete N-seed matrix,
the CLI insists on `N * 19 * 2` completed original runner jobs,
N distinct *derived* model seeds, and a declared nonempty
`scientific_scope`. The built-in list of 19 adapter names comes
from the original official-image driver, not caller-supplied labels.

## Direction of improvement

Each metric is computed *within the same seed and site*. Higher is
better for accuracy, balanced accuracy and macro-F1. Lower is better
for log-loss, multiclass Brier and ten-bin top-label ECE.

```text
For an increasing metric: improvement = ADP - Fixed
For a decreasing metric: improvement = Fixed - ADP
```

Positive values are favorable to ADP and negative values are favorable
to Fixed. The output preserves both raw pair values for each seed.
It reports the arithmetic mean and median improvement, sample
standard deviation only when at least two seeds exist, and counts
of positive, zero and negative paired differences. **All are
descriptive statistics**, conditional on the *particular*
dataset, held-out split and random seeds.

Separate `validation_metrics` and `test_metrics` trees are
maintained; no validation-based architecture is reselected based
on test metrics. The analysis never pools effects across the
19 architectures or treats architectures as additional independent
random seeds.

## Work parity and scientific limitations

The preexisting budget auditor also checks actual optimizer-step
equality within each pair. The paired result labels a whole adapter
`all-equal-optimizer-steps` only when every supplied seed matches.
It labels incomplete original V01 budgets `unknown-optimizer-steps`
and unequal native jobs `unequal-optimizer-steps`.

A *six-step fixed job versus six-step growth job* is **not**
an equal-training-FLOP experiment: per-step computation changes with
the model. Different architectural growth paths may have different
numbers of frozen and trainable parameters, RAM footprints, input
tokens and attention kernel costs.

For these reasons, the result always includes:

```json
{
  "full_training_flops_matched": null,
  "confidence_intervals_computed": false,
  "multiple_comparisons_adjusted": false,
  "scientific_performance_conclusion": null
}
```

Likewise each metric has `confidence_interval=null` and
`p_value=null` — not an interval of zero and not a computed
nonsignificance result. Even five or more seeds would not alone
establish full-data generalization, paired FLOPs, valid multiple-
comparison correction, or a causal architecture improvement.

Future scientific milestones: full official dataset partitions;
real source-faithful V05–V11; more pre-registered independent
seeds and split repetitions; complete backward/optimizer FLOP
accounting; physical CUDA allocator and process-memory evidence;
and paired confidence intervals/statistical tests under an
explicit, defensible experimental design.

## Automated evidence

`.github/workflows/visual-official-multiseed.yml` runs the existing
76-worker original Fashion matrix, then invokes the analysis
on its completed signed JSON. Both the original matrix and its
separate paired-report file are retained together in Actions artifacts.

Focused CI:
`tests/test_visual_paired_metrics.py` checks directions, raw
seed-level differences, cross-seed architecture consistency,
missing/duplicate job IDs and invalid metrics.
`tests/test_visual_paired_metrics_cli.py` executes the real CLI
against a synthetic complete 19-site matrix and confirms that
malformed or partial inputs produce **no report file**.

A passing test or micro-smoke is not a scientific model ranking.
