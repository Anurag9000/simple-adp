# Real-data Iris four native architecture-growth families — bounded proof

**Status:** `BOUNDED_REAL_DATA_FAMILY_PASSED` — source/variance pairs
successfully executed on the genuine scikit-learn Iris classification data
for four **representative** source models, one per major architecture family.
This is neither all 13 original adaptive architectures nor a full
scientific study.

## Actual execution evidence

- **GitHub Actions workflow:** https://github.com/Anurag9000/simple-adp/actions/runs/41841932711
- **Source commit:** `01323511dea54075ed06f46f36d3c076c21c7961`
- Four original-plus-variance jobs: **4 successful paired jobs / 8 trained
  model arms**.
- Downstream `verify-family-matrix` CI **succeeded** after checking all
  four downloaded archive receipts and inference-model SHA-256s.
- **Verified downstream audit artifact ID:** `13330051450`.
- **Audit JSON SHA-256:**
  `21f791b8558a69da86fb5818f73e574d6563aadf231494e2d254b759c8cb4be5`.
- **Real dataset hash:** `d51e090047c0dc3825bc350fe3c9282962c8d44dbf0fafba565b0328c461872b`.
- Same stratified Iris partition, train/validation/test sizes **90/30/30**;
  same model seed **73** and same source/optimizer/architecture maximum budget.
- Each original-vs-variance pair differs intentionally only in
  `expansion_criterion`. Checkpoints and inference bundles were downloaded,
  independently SHA-256 checked, and both checkpoints safely parsed.

## Held-out outcomes and executed controller histories

| Family | Representative method | Original selected widths | Variance selected widths | Original acc. | Variance acc. | Observed proposal/retention behavior |
|---|---|---|---|---:|---:|---|
| Width only | `ADP-W-All` | [6] | [3] | 0.6667 | 0.3333 | Original accepted successive [4], [5], [6] widths; variance retained initial [3] |
| Depth only | `ADP-D-Dcur` | [3] | [3] | 0.3333 | 0.3333 | Original proposed [3,3] depth and rejected it; variance retained initial [3] |
| Width then depth | `ADP-WD-All-Dcur` | [5] | [3] | 0.7000 | 0.3667 | Original accepted [4], [5] width proposals but did not reach an accepted depth stage; variance retained initial [3] |
| Depth then width | `ADP-DW-Dcur-All` | [3] | [3] | 0.3667 | 0.3667 | Original proposed and rejected [3,3] depth, then [4,4] width; variance retained initial [3] |

These detailed proposal histories come from inspecting actual persisted
`search_history` in each stage checkpoint, not from method-name inference.
An executed search controller, a proposed architecture, and a retained
architectural change are distinct evidence levels. Specifically, successful
execution of the depth-only route is **not** proof of accepted depth growth,
and these runs do not prove a variance-triggered depth proposal.

## Unexecuted research requirements

- Nine other adaptive configurations in the original
  13-method taxonomy are not individually exercised in this four-method
  experiment (although other source tests exist).
- No variance-triggered growth was observed in these four bounded jobs;
  the threshold/persistence settings remain a scientific hypothesis.
- No full visual V01–V11 real-source Bayesian pairing campaign, no GPU
  deterministic comparisons, and no coordinated planner-to-worker real-data
  execution receipts.
- Only four-epoch CPU budgets; no capacity-/compute-matched controls,
  no Monte Carlo gradient-pass overhead accounting and no multiseed or
  cross-dataset inference over the four growth mechanisms.

The four-family result is additional mechanism-route evidence, not a
multiplicative independent sample of four datasets or multiple seeds.
