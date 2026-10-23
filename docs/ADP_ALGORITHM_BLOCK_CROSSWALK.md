# ADP guide block crosswalk

This generated index preserves the source document's body order: each
nonempty paragraph or table row is assigned a stable zero-based block ID.
It indexes the paragraph-level review recorded in
[the ADP implementation audit](ADP_ALGORITHM_IMPLEMENTATION_AUDIT.md);
it is not a claim of testing every possible numeric setting, custom
callback, optimizer, runtime, or interruption point.

| Block IDs | Review area | Implementation sources | Audit record |
|---:|---|---|---|
| 000–066 | Taxonomy, notation, and controller definitions | `registry.py; config.py; dedicated.py; generic_search.py` | taxonomy and notation |
| 067–144 | Shared architecture and all-layer/round-robin width DEN | `models.py; growth.py; training.py; dedicated.py` | shared mechanics and width DEN |
| 145–207 | Depth growth, nested DEN, and LG-ND | `growth.py; dedicated.py; search.py` | depth, nested, and LG-ND |
| 208–262 | Generic candidate search and mutation primitives | `generic_search.py; growth.py; training.py; artifacts.py` | generic search and mutations |
| 263–457 | Checkpoint recovery, traces, reproducibility, taxonomy, and appendices | `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py` | checkpoint and taxonomy |

## Indexed source blocks

### Block 000

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
DYNAMIC MLP ARCHITECTURE ADAPTATION
</pre>
</details>

### Block 001

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
ADP &amp; LG-ND
Algorithms and Exact Implementation Guide
</pre>
</details>

### Block 002

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Domain-independent architecture-search core reconstructed from DNN/VANILLA/Dyn_DNN4OPF/models
</pre>
</details>

### Block 003

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Repository: Anurag9000/OPF_ADP
Analyzed branch: main
Analyzed commit: 6461b8ee8d378ef6e5c7eacee9bbc6abde63f20c
Scope: adaptive architecture algorithms and their implementation mechanics; output-head decomposition intentionally abstracted away.
Authoritative taxonomy: 14 experiment-level model configurations.
</pre>
</details>

### Block 004

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Interpretation rule used throughout this document: 1-head, 2-head, and 4-head files are collapsed into one algorithm family unless the duplicated copies change the architecture-growth controller itself. When they do, the divergence is documented explicitly because it can alter experimental results.
</pre>
</details>

### Block 005

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Technical implementation report
</pre>
</details>

### Block 006

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
1. Scope and algorithm taxonomy
</pre>
</details>

### Block 007

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Authoritative experiment-level count: 14 models. This count treats configuration-level width/depth order, width-growth policy, and depth-seeding policy as separate models even when they share implementation code.
</pre>
</details>

### Block 008

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Complete 14-model taxonomy
</pre>
</details>

### Block 009

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Family | # | Short name | Exact model definition
</pre>
</details>

### Block 010

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Width-only | 1 | LG-ND width | Loss-guided neural densification; stop on first non-improvement
</pre>
</details>

### Block 011

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Width-only | 2 | Patience width — RR | Patience-based round-robin layer-wise widening
</pre>
</details>

### Block 012

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Width-only | 3 | Patience width — all-layer | Patience-based simultaneous widening of every hidden layer
</pre>
</details>

### Block 013

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Width→Depth | 4 | RR → depth@1 | Round-robin width search/growth, then add depth seeded at width 1
</pre>
</details>

### Block 014

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Width→Depth | 5 | All → depth@current | All-layer width search/growth, then add depth seeded at current width
</pre>
</details>

### Block 015

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Width→Depth | 6 | RR → depth@current | Round-robin width search/growth, then add depth seeded at current width
</pre>
</details>

### Block 016

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Width→Depth | 7 | All → depth@1 | All-layer width search/growth, then add depth seeded at width 1
</pre>
</details>

### Block 017

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Depth→Width | 8 | depth@1 → RR | Add depth seeded at width 1, then perform round-robin widening
</pre>
</details>

### Block 018

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Depth→Width | 9 | depth@current → All | Add depth seeded at current width, then perform all-layer widening
</pre>
</details>

### Block 019

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Depth→Width | 10 | depth@current → RR | Add depth seeded at current width, then perform round-robin widening
</pre>
</details>

### Block 020

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Depth→Width | 11 | depth@1 → All | Add depth seeded at width 1, then perform all-layer widening
</pre>
</details>

### Block 021

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Depth-only | 12 | Depth stage @1 | Stagewise depth growth; each new depth stage starts at width 1
</pre>
</details>

### Block 022

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Depth-only | 13 | Depth stage @current | Stagewise depth growth; each new depth stage starts at current width
</pre>
</details>

### Block 023

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Fixed | 14 | Vanilla fixed MLP | One-shot fixed-architecture MLP grid/baseline
</pre>
</details>

### Block 024

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Factorization: the mixed-order families each contain 2 width-growth policies (round-robin vs all-layer) × 2 depth-seeding policies (width 1 vs current width) = 4 models. Because search order itself can be Width→Depth or Depth→Width, the mixed-order block contains 8 models total.
</pre>
</details>

### Block 025

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
This document reconstructs the exact architecture-search behavior implemented in the referenced dynamic-MLP model directory, but expresses the algorithms independently of any particular application domain, dataset, network case, or output semantics. It does not treat output-head count as an algorithmic variant. Instead it isolates the architectural controller: how a network is widened or deepened, how inherited parameters are copied, how each architecture is trained, how validation improvements are judged, how failure patience is updated, what capacity limits are enforced, how checkpoints/resume work, and when the search rolls back to the best architecture.
</pre>
</details>

### Block 026

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Domain-independence convention: x and y are generic input/target tensors; validation loss is a generic model-selection signal; width/depth mutation, patience, rollback, checkpointing, and warm-start inheritance do not depend on a specific power-system case, dataset identity, physical constraint, or output-head interpretation.
</pre>
</details>

### Block 027

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Naming clarification for models 8–11: because these are Depth→Width variants, the symmetric wording is &#x27;add depth at width 1/current width, then widen by round-robin/all-layer&#x27;. This is the order-reversed counterpart of models 4–7.
</pre>
</details>

### Block 028

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
The repository contains two distinct implementation styles. The dedicated DEN/LG-ND classes mutate a live model in place and maintain explicit architecture snapshots. The generic adp_search.py path constructs a sequence of MLP candidates using mutation helpers from mlp.py. Although they pursue similar width/depth adaptation, their training objective, optimizer, state bookkeeping, and several stopping rules differ.
</pre>
</details>

### Block 029

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Normalized algorithm | Core proposal | Outer search rule | Final selection
</pre>
</details>

### Block 030

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Fixed Vanilla MLP | No architecture mutation | None | Best epoch of fixed architecture
</pre>
</details>

### Block 031

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
DEN width — all layers | Add k units to every hidden layer | Forward-only; tolerate width failures | Restore global best snapshot after termination
</pre>
</details>

### Block 032

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
DEN width — round robin | Add k units to one hidden layer at a time | Forward-only; tolerate width failures | Restore global best snapshot after termination
</pre>
</details>

### Block 033

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
DEN depth — progressive | Append width-1 layer; progressively densify it | Forward-only; tolerate depth failures | Restore global best snapshot after termination
</pre>
</details>

### Block 034

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
DEN depth — matched-width seed | Append full-width layer directly | Forward-only; tolerate depth failures | Restore global best snapshot after termination
</pre>
</details>

### Block 035

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Nested width/depth DEN | Widen; then run an inner depth sweep | Width patience outside depth patience | Restore best architecture found
</pre>
</details>

### Block 036

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
LG-ND | Width densification inherited from DEN | Stop on first non-improving expansion | Restore best pre-failure snapshot
</pre>
</details>

### Block 037

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Generic ADP search | Build/train separate MLP candidates | Mode-dependent width/depth phases | Intended best-candidate selection; current bookkeeping has defects
</pre>
</details>

### Block 038

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Central distinction: DEN is a non-monotonic, patience-based forward exploration policy; LG-ND is a greedy monotonic-improvement policy. Both warm-start larger architectures by copying overlapping weights, but neither implements a strictly function-preserving network morphism.
</pre>
</details>

### Block 039

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
1.1 Source files in scope
</pre>
</details>

### Block 040

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Source | Role
</pre>
</details>

### Block 041

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
mlp.py | Base MLP and generic width/depth mutation primitives
</pre>
</details>

### Block 042

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_den_width_only.py | Canonical single-output width DEN; also hosts shared LG-ND controller
</pre>
</details>

### Block 043

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_den_depth_only.py | Canonical depth DEN and nested depth-to-width routine
</pre>
</details>

### Block 044

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_den_2_head_width_only.py | Duplicated width DEN; core divergences noted where relevant
</pre>
</details>

### Block 045

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_den_2head_depth_only.py | Duplicated depth DEN; core divergences noted where relevant
</pre>
</details>

### Block 046

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_den_4_head_width_only.py | Duplicated width DEN; core divergences noted where relevant
</pre>
</details>

### Block 047

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_den_4head_depth_only.py | Duplicated depth DEN; core divergences noted where relevant
</pre>
</details>

### Block 048

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_lg_nd_width_only.py | LG-ND wrapper around width DEN
</pre>
</details>

### Block 049

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_lg_nd_2_head_width_only.py | LG-ND wrapper; same active shared controller
</pre>
</details>

### Block 050

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_lg_nd_4_head_width_only.py | LG-ND wrapper; same active shared controller
</pre>
</details>

### Block 051

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
adp_search.py | Generic candidate-based ADP search
</pre>
</details>

### Block 052

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
checkpoint_utils.py | Architecture-aware checkpoint restoration helpers
</pre>
</details>

### Block 053

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
utils/optim_sched.py | Adam + CosineAnnealingWarmRestarts factory used by DEN/LG-ND
</pre>
</details>

### Block 054

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
utils/adp_external_adapter.py | AdamW + optional application-specific auxiliary penalty loss used by adp_search.py
</pre>
</details>

### Block 055

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
1.2 Notation used in this guide
</pre>
</details>

### Block 056

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Symbol | Meaning
</pre>
</details>

### Block 057

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
A = [h1, …, hD] | Current hidden-layer architecture; hi is the output width of hidden layer i
</pre>
</details>

### Block 058

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
D | Current number of hidden layers (depth)
</pre>
</details>

### Block 059

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
W | A scalar width summary used by the controller; exact definition depends on growth strategy
</pre>
</details>

### Block 060

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
kW = ex_k_width | Width increment per proposal
</pre>
</details>

### Block 061

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
kD = ex_k_depth | Configured depth increment; current dedicated depth expansion effectively appends one layer per call
</pre>
</details>

### Block 062

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Lval(A) | Best validation loss obtained while training architecture A
</pre>
</details>

### Block 063

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
δW, δD | Minimum architecture-level validation improvement required by DEN
</pre>
</details>

### Block 064

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
pES | Epoch-level early-stopping patience within one fixed architecture
</pre>
</details>

### Block 065

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
pW, pD | Architecture-level failure patience for width/depth search
</pre>
</details>

### Block 066

**Review area:** Taxonomy, notation, and controller definitions  
**Code crosswalk:** `registry.py; config.py; dedicated.py; generic_search.py`  
**Audit section:** taxonomy and notation

<details>
<summary>Source text</summary>

<pre>
Nmax | max_neurons capacity proxy; counts units, not trainable parameters
</pre>
</details>

### Block 067

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2. Shared mechanics underlying every adaptive variant
</pre>
</details>

### Block 068

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2.1 Architecture state representation
</pre>
</details>

### Block 069

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The dedicated DEN classes store hidden layers explicitly in an nn.ModuleList. The architecture is therefore represented by the ordered vector A = [h1, h2, …, hD], where hi equals hidden_layers[i].out_features. The output projection is resized whenever the last hidden width changes.
</pre>
</details>

### Block 070

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
snapshot_arch_and_state() records both architecture metadata and a deep-copied state_dict. Width models additionally persist a width-growth cursor so that a round-robin sequence can be resumed at the same layer. restore_arch_and_state() reconstructs the required layer shapes first and then partially restores compatible tensors.
</pre>
</details>

### Block 071

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2.2 Warm-start resizing rule
</pre>
</details>

### Block 072

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
When a dense layer is resized, a fresh nn.Linear object with the new shape is created. The overlapping top-left block of the old weight matrix is copied into the new matrix; the corresponding prefix of the bias is copied. Newly created rows/columns retain PyTorch&#x27;s fresh initialization.
</pre>
</details>

### Block 073

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Given old Linear: W_old ∈ R^(o_old × i_old), b_old ∈ R^(o_old)
Requested shape: o_new × i_new

rows = min(o_old, o_new)
cols = min(i_old, i_new)

W_new[0:rows, 0:cols] ← W_old[0:rows, 0:cols]
b_new[0:rows]         ← b_old[0:rows]

All other entries in W_new / b_new remain newly initialized.
</pre>
</details>

### Block 074

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Consequently expansion is a partial-parameter warm start, not a strict network morphism. In general the function immediately after expansion changes: fnew(x) is not guaranteed to equal fold(x). Old parameters remain trainable; no freezing mask is applied.
</pre>
</details>

### Block 075

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2.3 Rebuilding a whole MLP candidate
</pre>
</details>

### Block 076

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
mlp.py uses _rebuild_mlp() plus _copy_matching_state(). A fresh MLP with the requested hidden_widths is instantiated, then every state tensor with a matching key is copied. If shapes differ, only the common prefix along every tensor dimension is copied.
</pre>
</details>

### Block 077

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
This generic candidate path also copies BatchNorm state where matching keys exist, because the base MLP can include BatchNorm. The dedicated DEN classes do not construct BatchNorm layers.
</pre>
</details>

### Block 078

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2.4 Capacity measures and hard limits
</pre>
</details>

### Block 079

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The dedicated classes define total_neurons() as the sum of hidden-layer unit counts plus output units. This is a unit-count proxy, not a parameter count. For hidden widths [h1,…,hD], the true dense-parameter cost grows roughly with products hi-1×hi, while total_neurons grows only linearly with hi.
</pre>
</details>

### Block 080

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Dedicated DEN capacity checks typically enforce:
    each hidden width ≤ max_width
    depth ≤ max_depth                     (depth algorithms)
    total_neurons() + proposed_units ≤ max_neurons

Important:
    max_neurons is NOT equivalent to parameter count, FLOPs, VRAM, or latency.
</pre>
</details>

### Block 081

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
A width doubling can increase dense matrix parameter count quadratically inside uniform hidden stacks even though the max_neurons accounting only increases linearly.
</pre>
</details>

### Block 082

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2.5 Inner training algorithm used by dedicated DEN/LG-ND stages
</pre>
</details>

### Block 083

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Every architecture proposal is trained as a complete stage. The dedicated classes instantiate a fresh optimizer/scheduler for that stage using get_optimizer_scheduler(): Adam at the configured learning rate plus CosineAnnealingWarmRestarts.
</pre>
</details>

### Block 084

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
TRAIN_STAGE(model A):
    create Adam(model.parameters(), lr)
    create CosineAnnealingWarmRestarts(Adam, configured scheduler parameters)

    best_val  ← +∞
    best_snap ← snapshot(A)
    patience  ← pES

    while patience &gt; 0 and epoch budget not exhausted:
        for minibatch (x, y):
            y_hat ← model(x)
            loss  ← MSE(y_hat, y)
            zero gradients
            loss.backward()
            optimizer.step()
            checkpoint latest batch state

        scheduler.step()

        val ← mean validation MSE

        if val &lt; best_val - 1e-12:
            best_val  ← val
            best_snap ← snapshot(A)
            patience  ← pES
            checkpoint best epoch
        else:
            patience ← patience - 1

        checkpoint latest epoch state

    restore_arch_and_state(best_snap)
    return best_val, best_snap
</pre>
</details>

### Block 085

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Two levels of early stopping therefore coexist. pES decides how long to optimize weights for one architecture. pW/pD decide how long to continue modifying architecture after candidate architectures fail to improve the global architecture-level best.
</pre>
</details>

### Block 086

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2.6 Optimizer-state inheritance
</pre>
</details>

### Block 087

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Architecture expansion preserves overlapping model weights, but a new call to train_with_early_stopping() creates a new optimizer and scheduler. Therefore Adam first/second moment estimates are not transferred from the previous architecture stage. They are only restored when resuming an interrupted checkpoint belonging to the same stage.
</pre>
</details>

### Block 088

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Across architecture boundary:
    θ_new(0) ← resized/copy-inherited θ_old*
    Adam m    ← freshly initialized
    Adam v    ← freshly initialized
    scheduler ← freshly initialized

Within interrupted same stage:
    model_state, optimizer_state, scheduler_state, patience, epoch/batch position
    are restored when a compatible checkpoint is available.
</pre>
</details>

### Block 089

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2.7 Objective function in the dedicated vanilla classes
</pre>
</details>

### Block 090

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
For the dedicated DEN and LG-ND classes, the backpropagated objective is plain mean-squared error. The surrounding application may compute additional domain-specific diagnostics or constraint metrics for telemetry, but those diagnostics are not added to loss.backward() in these class-local training loops. Therefore the architecture controller described here is independent of any particular domain-specific constraint definition.
</pre>
</details>

### Block 091

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Ltrain = MSE(y_hat, y)
</pre>
</details>

### Block 092

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
This point becomes important later: adp_search.py uses a different external training adapter and explicitly activates its penalty-capable loss path.
</pre>
</details>

### Block 093

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3. ADP/DEN width growth — all hidden layers together
</pre>
</details>

### Block 094

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3.1 Purpose and state
</pre>
</details>

### Block 095

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
This is the standard patience-based width-growth experiment exposed by the normal width-only runners. Depth remains fixed. A proposal increases every hidden layer width by ex_k_width, preserving as much of the existing parameter tensors as possible.
</pre>
</details>

### Block 096

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
A_t = [h1, h2, …, hD]
k = ex_k_width

PROPOSE_ALL_LAYER_WIDTH:
    A_(t+1) = [h1+k, h2+k, …, hD+k]
</pre>
</details>

### Block 097

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3.2 Width semantics
</pre>
</details>

### Block 098

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
In all-layer mode, width is reported as the maximum hidden width. For the usual uniform architecture this equals every hidden width. The controller refuses a proposal if any resulting hidden width would exceed max_width or if the aggregate unit-count increment D×k would exceed max_neurons.
</pre>
</details>

### Block 099

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3.3 Exact outer controller
</pre>
</details>

### Block 100

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
INPUT:
    initial model A0
    pW = patience_width_exp
    δW = delta_width
    k  = ex_k_width

1. Train A0 with TRAIN_STAGE.
2. best_val   ← Lval(A0)
   best_snap  ← snapshot(A0)
   best_width ← width(A0)
   width_fail ← 0

3. While:
       width_fail &lt; pW
       and width(Acurrent) &lt; max_width
       and proposed growth satisfies max_width/max_neurons:

       Acurrent ← widen EVERY hidden layer by k
       val ← TRAIN_STAGE(Acurrent)

       if val &lt; best_val - δW:
           best_val   ← val
           best_snap  ← snapshot(Acurrent)
           best_width ← width(Acurrent)
           width_fail ← 0
       else:
           width_fail ← width_fail + 1
           KEEP Acurrent AS THE FORWARD SEARCH STATE
           DO NOT ROLLBACK YET

4. restore_arch_and_state(best_snap)
5. return best_val
</pre>
</details>

### Block 101

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3.4 Forward-only search and valley crossing
</pre>
</details>

### Block 102

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The non-improving expanded architecture is intentionally retained as the current search state. The next expansion is applied to that larger architecture rather than to the best historical architecture. This makes the architecture walk capable of crossing temporary validation-loss valleys.
</pre>
</details>

### Block 103

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Example with pW ≥ 2:

[16,16]  L=0.100  ← best
[17,17]  L=0.090  ← new best
[18,18]  L=0.094  ← failure #1, but keep [18,18]
[19,19]  L=0.087  ← recovery; new best

A greedy rollback-on-failure method would never reach [19,19].
</pre>
</details>

### Block 104

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Only after the stopping condition is met does the algorithm reconstruct the recorded best architecture and reload its recorded best state.
</pre>
</details>

### Block 105

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3.5 Exact improvement test
</pre>
</details>

### Block 106

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Architecture-level improvement uses a configurable margin: candidate loss must be less than best_val - delta_width. A tiny improvement smaller than delta_width is treated as an architectural failure even if it is numerically lower.
</pre>
</details>

### Block 107

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
accept_as_new_global_best ⇔ Lcandidate &lt; Lbest - δW
</pre>
</details>

### Block 108

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3.6 What is and is not inherited
</pre>
</details>

### Block 109

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Inherited: overlapping weights and biases of every resized hidden layer.
</pre>
</details>

### Block 110

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Inherited: overlapping input columns of the output projection after the last hidden layer is widened.
</pre>
</details>

### Block 111

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Not inherited across stages: Adam momentum/variance state.
</pre>
</details>

### Block 112

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Not preserved exactly: network function; new weights are randomly initialized.
</pre>
</details>

### Block 113

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Not frozen: older units continue receiving gradients after widening.
</pre>
</details>

### Block 114

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Not present: pruning, learned gates, neuron importance scores, or sparsity-driven selection.
</pre>
</details>

### Block 115

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3.7 Stopping conditions
</pre>
</details>

### Block 116

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
width_fail reaches patience_width_exp.
</pre>
</details>

### Block 117

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Reported width reaches max_width.
</pre>
</details>

### Block 118

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The next width proposal would violate max_width.
</pre>
</details>

### Block 119

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The next proposal would violate max_neurons.
</pre>
</details>

### Block 120

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
A resumed completed stage can short-circuit the entire task by restoring its recorded best checkpoint.
</pre>
</details>

### Block 121

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
4. ADP/DEN width growth — round-robin layer-wise densification
</pre>
</details>

### Block 122

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The round-robin variant changes the architecture proposal granularity while retaining the same forward-only, patience-based global search philosophy. Instead of increasing all hidden layers simultaneously, it advances one eligible hidden layer at a time and rotates a persistent cursor.
</pre>
</details>

### Block 123

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
4.1 Mutation rule
</pre>
</details>

### Block 124

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
A = [h1, h2, …, hD]
cursor = c
target_width = current_scalar_width + k

Search indices:
    c, c+1, …, D-1, 0, …, c-1

Select first layer i with hi &lt; target_width.
If budget permits:
    hi ← hi + k
    rebuild hidden stack with copied overlap
    cursor ← (i + 1) mod D
</pre>
</details>

### Block 125

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
4.2 Why scalar width becomes the minimum
</pre>
</details>

### Block 126

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
During a partially completed round, hidden widths are non-uniform. The dedicated width classes therefore report width as the minimum hidden width in round-robin mode. This prevents the controller from claiming that a new global width level has been reached merely because one layer has already advanced.
</pre>
</details>

### Block 127

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
[16,16,16] → [17,16,16]
reported scalar width = min(...) = 16

then:
[17,16,16] → [17,17,16]
reported scalar width = 16

then:
[17,17,16] → [17,17,17]
reported scalar width = 17
</pre>
</details>

### Block 128

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
4.3 Search trajectory versus all-layer growth
</pre>
</details>

### Block 129

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Step | All-layer k=1 | Round-robin k=1
</pre>
</details>

### Block 130

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
0 | [16,16,16] | [16,16,16]
</pre>
</details>

### Block 131

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
1 | [17,17,17] | [17,16,16]
</pre>
</details>

### Block 132

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
2 | [18,18,18] | [17,17,16]
</pre>
</details>

### Block 133

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
3 | [19,19,19] | [17,17,17]
</pre>
</details>

### Block 134

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
4 | [20,20,20] | [18,17,17]
</pre>
</details>

### Block 135

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Round-robin therefore samples a substantially denser set of architecture shapes and can reveal whether local capacity in one layer is useful before paying for a full-stack widening.
</pre>
</details>

### Block 136

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
4.4 Outer acceptance and rollback
</pre>
</details>

### Block 137

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The architecture-level logic is otherwise DEN-like: each mutated architecture is trained to its stage-level best, compared against the global architecture best using delta_width, failure patience is updated, the widened current architecture is retained after a failed comparison, and the global best snapshot is restored only after the search terminates.
</pre>
</details>

### Block 138

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
4.5 Current implementation-copy divergence
</pre>
</details>

### Block 139

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Although head count is intentionally ignored as a scientific variable here, the duplicated source files do not execute perfectly identical round-robin logic. This matters because it changes the core architecture algorithm.
</pre>
</details>

### Block 140

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The canonical single-output width class uses ex_k_width consistently when its generic expand_width() dispatches to round-robin growth.
</pre>
</details>

### Block 141

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
One duplicated width controller has an explicit round-robin outer branch that advances target_width by 1 and calls grow_width_round_robin(..., 1), effectively ignoring ex_k_width when ex_k_width &gt; 1.
</pre>
</details>

### Block 142

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
Another duplicated width controller exposes round-robin mutation internally but its outer train_task loop checks capacity as if all hidden layers were widened together. This can conservatively reject a legal one-layer proposal.
</pre>
</details>

### Block 143

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
The official standard runners use all-layer widening for the corresponding duplicated models, so these discrepancies primarily matter when manually overriding width_growth_strategy.
</pre>
</details>

### Block 144

**Review area:** Shared architecture and all-layer/round-robin width DEN  
**Code crosswalk:** `models.py; growth.py; training.py; dedicated.py`  
**Audit section:** shared mechanics and width DEN

<details>
<summary>Source text</summary>

<pre>
For a controlled architecture-study, factor the width-growth controller into one shared implementation before comparing output-head layouts. Otherwise &#x27;head count&#x27; can be confounded with different architecture-search trajectories.
</pre>
</details>

### Block 145

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5. ADP/DEN depth growth — progressive narrow-layer insertion
</pre>
</details>

### Block 146

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
The depth controller is more distinctive than the width controller. Under the default runner configuration, adding depth does not immediately append a full-width hidden layer. A new layer is inserted at width 1 and then progressively densified toward the preceding network width.
</pre>
</details>

### Block 147

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5.1 Preconditions for adding depth
</pre>
</details>

### Block 148

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
can_add_depth_stage() requires:
    hidden stack is non-empty
    all existing hidden widths are equal
    D + 1 ≤ max_depth

If seed_new_depth_to_current_width is False:
    proposed new layer contributes 1 unit to max_neurons

If True:
    proposed new layer contributes current width W
</pre>
</details>

### Block 149

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
The uniform-width requirement is algorithmically important. A partially round-robin-widened stack such as [17,16,16] is not eligible for a new depth stage.
</pre>
</details>

### Block 150

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5.2 Default insertion rule
</pre>
</details>

### Block 151

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Current architecture:
    A = [W, W, …, W]

Default depth proposal:
    A&#x27; = [W, W, …, W, 1]

New layer:
    Linear(W → 1)

Output projection:
    resized from input width W to input width 1,
    copying only the overlapping prefix.
</pre>
</details>

### Block 152

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
This introduces an intentional but severe representational bottleneck. The transformation is not an identity insertion. Immediately after the new layer appears, a W-dimensional hidden representation is compressed to one scalar before reaching the output projection.
</pre>
</details>

### Block 153

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5.3 Progressive terminal-layer densification in the canonical controller
</pre>
</details>

### Block 154

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
When seed_new_depth_to_current_width is False, the canonical single-output depth controller remembers the pre-insertion width Wstart, appends the width-1 layer, then repeatedly trains and widens only the terminal new layer by one unit until it reaches Wstart.
</pre>
</details>

### Block 155

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Example: A = [8, 8]

append depth:
    [8, 8, 1]

repeat:
    train [8, 8, 1]
    terminal width 1 → 2

    train [8, 8, 2]
    terminal width 2 → 3

    ...
    train [8, 8, 7]
    terminal width 7 → 8

final stage candidate:
    train [8, 8, 8]
    compare stage loss against global best using δD
</pre>
</details>

### Block 156

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5.4 Canonical global-best semantics
</pre>
</details>

### Block 157

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
In the canonical single-output depth controller, the intermediate [W,…,j] states primarily act as optimization stepping stones. Their validation values are not used to replace the global architecture-level best during the catch-up loop. The meaningful global comparison is performed after the terminal layer has reached the stage-start width and the completed deeper architecture is trained again.
</pre>
</details>

### Block 158

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Canonical progressive depth stage:

append width-1 layer
while terminal_width &lt; Wstart:
    TRAIN_STAGE(current partial layer)
    grow terminal layer by +1

TRAIN_STAGE(full-width deeper model)

if Lfull &lt; Lbest - δD:
    record deeper model as best
    depth_fail = 0
else:
    depth_fail += 1
    keep deeper model as current forward-search state
</pre>
</details>

### Block 159

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5.5 Depth failure patience and forward-only behavior
</pre>
</details>

### Block 160

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
A failed completed depth stage does not immediately roll back. The deeper architecture remains the current search state. If another depth stage is still legal and depth_fail is below patience_depth_exp, search continues forward. The best snapshot is restored only after depth patience or a hard depth/capacity limit terminates the outer loop.
</pre>
</details>

### Block 161

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5.6 Alternative: seed the new layer at current width
</pre>
</details>

### Block 162

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Setting seed_new_depth_to_current_width=True removes the width-1 bottleneck. The new hidden layer is immediately created with the same width as the previous last hidden layer.
</pre>
</details>

### Block 163

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Progressive insertion:
    [W,W] → [W,W,1] → ... → [W,W,W]

Matched-width insertion:
    [W,W] → [W,W,W]
</pre>
</details>

### Block 164

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
This latent variant changes both optimization difficulty and the architecture path. It evaluates depth itself without the intermediate low-rank bottleneck trajectory.
</pre>
</details>

### Block 165

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5.7 Core divergence present in duplicated depth implementations
</pre>
</details>

### Block 166

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
The duplicated depth controllers used for alternative output layouts do not exactly preserve the canonical single-output semantics. Their catch-up loops can compare intermediate partially densified states against the global best and can save those states as best_snap.
</pre>
</details>

### Block 167

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Alternative duplicated behavior can effectively allow:

[W,W,1]  train → compare against global best
[W,W,2]  train → compare against global best
...
[W,W,W]  train → compare against global best

Therefore a non-uniform intermediate architecture can become the final restored best.
</pre>
</details>

### Block 168

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
This is a genuine architecture-search difference, not a head-layout difference. If experiments are intended to compare heads only, this behavior should be unified.
</pre>
</details>

### Block 169

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
5.8 All-layer catch-up edge case in duplicated depth controllers
</pre>
</details>

### Block 170

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
When width_growth_strategy is manually set to an all-layer mode inside some duplicated depth controllers, the catch-up logic can widen the old hidden layers at the same time as the new width-1 layer grows. The target is set near the pre-growth width, but grow_width_all_layers(1) increments every hidden layer.
</pre>
</details>

### Block 171

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Starting from [10,10]:

append width-1 depth:
    [10,10,1]

all-layer catch-up:
    [11,11,2]
    [12,12,3]
    [13,13,4]
    ...
</pre>
</details>

### Block 172

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
This no longer means &#x27;densify only the newly added layer&#x27;. It creates a combined width+depth expansion trajectory and can leave the stack non-uniform, which then blocks another depth insertion because can_add_depth_stage() requires uniform hidden widths. The default round-robin-oriented depth runners avoid this pathology.
</pre>
</details>

### Block 173

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
6. Nested width/depth DEN controller
</pre>
</details>

### Block 174

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Terminology correction: despite the routine name depth_to_width, the implemented nested control flow is outer WIDTH → inner DEPTH. A fully symmetric outer DEPTH → inner GLOBAL WIDTH controller is not present; see Section 18.
</pre>
</details>

### Block 175

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
The canonical single-output and one duplicated depth implementation expose an alternate loop_order that dispatches to train_task_depth_to_width()/fit_task_depth_to_width(). Despite the function name, the active control flow is best understood as width growth in the outer loop with a depth search performed after each width proposal.
</pre>
</details>

### Block 176

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
6.1 Hierarchical structure
</pre>
</details>

### Block 177

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
INITIAL:
    train current architecture
    best_val, best_snap, best_width, best_depth

OUTER WIDTH LOOP:
    while width_fail &lt; pW and width proposal is legal:
        widen current hidden architecture

        reset depth_fail = 0

        INNER DEPTH LOOP:
            while depth_fail &lt; pD and another depth is legal:
                append a depth stage
                TRAIN_STAGE
                if loss improves global best by δD:
                    update best snapshot/depth
                    depth_fail = 0
                else:
                    depth_fail += 1

        compare best_val after inner depth sweep to value before this width round

        if global best improved by at least δW:
            width_fail = 0
            best_width = current width
        else:
            width_fail += 1

FINAL:
    restore global best snapshot
</pre>
</details>

### Block 178

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
6.2 Three independent patience scales
</pre>
</details>

### Block 179

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Counter | Scope | Meaning
</pre>
</details>

### Block 180

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
patience_es | Epochs within one architecture | How long ordinary optimization may stall
</pre>
</details>

### Block 181

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
patience_depth_exp | Depth proposals inside a width round | How many depth failures may be traversed
</pre>
</details>

### Block 182

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
patience_width_exp | Outer width rounds | How many width-level rounds may fail to improve
</pre>
</details>

### Block 183

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
This is the most exploratory dedicated controller in the directory because it can traverse multiple widths and, at each width, multiple deeper architectures.
</pre>
</details>

### Block 184

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
6.3 Important implementation nuance
</pre>
</details>

### Block 185

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
The helper _depth_stage() computes its stage directory/name before repeatedly appending deeper layers, so multiple depth candidates in that inner sweep can share the same stage-name/directory context. Architecturally the loop still changes depth repeatedly, but telemetry/checkpoint naming is less granular than the architecture trajectory itself.
</pre>
</details>

### Block 186

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Also note that this nested routine&#x27;s depth proposals call expand_depth(1) directly; when seed_new_depth_to_current_width=False, that means width-1 insertion without the same explicit terminal-width catch-up loop used by the canonical pure-depth train_task(). Thus the nested depth sweep is not identical to the standalone progressive-depth algorithm.
</pre>
</details>

### Block 187

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
This is a major distinction: &#x27;pure depth mode&#x27; and the inner depth sweep of the nested controller do not execute the same insertion-and-densification procedure.
</pre>
</details>

### Block 188

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
7. LG-ND — Loss-Guided Neural Densification
</pre>
</details>

### Block 189

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
LG-ND subclasses the corresponding width-only DEN model and reuses its architecture mutation and stage-training machinery. The active runtime controller is centralized in _train_lg_nd_width_task() inside adp_den_width_only.py. The wrapper files immediately return from this shared helper; the older duplicated implementation code that appears below that return is unreachable.
</pre>
</details>

### Block 190

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
7.1 Core search rule
</pre>
</details>

### Block 191

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
1. Train initial architecture A0.
2. best_val  ← Lval(A0)
   best_snap ← snapshot(A0)

3. Repeatedly:
       verify next width/capacity is legal
       widen model by ex_k_width
       val ← TRAIN_STAGE(expanded model)

       if val &lt; best_val:
           best_val  ← val
           best_snap ← snapshot(expanded model)
           continue
       else:
           STOP IMMEDIATELY

4. restore best_snap
5. return best_val
</pre>
</details>

### Block 192

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
7.2 Exact difference from DEN
</pre>
</details>

### Block 193

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Property | Patience DEN | LG-ND
</pre>
</details>

### Block 194

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Failure tolerance | Several failures up to pW | Exactly first non-improvement stops
</pre>
</details>

### Block 195

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Improvement test | Lnew &lt; Lbest - δW | Lnew &lt; Lbest
</pre>
</details>

### Block 196

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
After a failed expansion | Keep larger current model and continue if patience remains | Terminate and restore best
</pre>
</details>

### Block 197

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Valley crossing | Possible | Not possible after first upward/flat step
</pre>
</details>

### Block 198

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Search bias | More exploratory | Strongly favors earliest locally sufficient width
</pre>
</details>

### Block 199

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
7.3 Why delta is effectively not used for LG-ND acceptance
</pre>
</details>

### Block 200

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
The shared helper may copy the supplied delta argument into model.delta_width for API compatibility, but the active architecture comparison is a strict val &lt; best_val test. Therefore any positive numerical improvement is accepted, even if it is smaller than the DEN architecture margin.
</pre>
</details>

### Block 201

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
7.4 Capacity pre-check
</pre>
</details>

### Block 202

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Before an LG-ND expansion, the helper predicts next_width = width + ex_k_width and next_total_neurons = current_total_neurons + depth × ex_k_width. This matches simultaneous all-layer width growth.
</pre>
</details>

### Block 203

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Because LG-ND inherits round-robin mutation support from the base class, manually selecting round_robin creates a semantic mismatch: the pre-check assumes D×k new units even though a round-robin mutation adds only one layer&#x27;s increment. The check is therefore conservative and may stop earlier than necessary.
</pre>
</details>

### Block 204

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
7.5 Search landscape interpretation
</pre>
</details>

### Block 205

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
Validation trajectory:
    width 16 → 0.100
    width 17 → 0.090   accept
    width 18 → 0.091   first non-improvement → STOP
    width 19 → 0.080   never evaluated

LG-ND returns width 17.
DEN with sufficient pW may traverse width 18 and discover width 19.
</pre>
</details>

### Block 206

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
7.6 Resume semantics
</pre>
</details>

### Block 207

**Review area:** Depth growth, nested DEN, and LG-ND  
**Code crosswalk:** `growth.py; dedicated.py; search.py`  
**Audit section:** depth, nested, and LG-ND

<details>
<summary>Source text</summary>

<pre>
LG-ND persists whether a stage is active, the current hidden widths, the stage-local checkpoint, the global best checkpoint, best width, best total-neuron count, and a stop_reason. On resume it can restore an interrupted expanded architecture, finish its stage training, evaluate it under the same greedy rule, and immediately terminate if the resumed expansion fails to improve.
</pre>
</details>

### Block 208

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8. Generic candidate-based ADP search (adp_search.py)
</pre>
</details>

### Block 209

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
adp_search.py is architecturally separate from the dedicated DEN/LG-ND classes. It receives a base MLP, constructs one candidate directory per architecture, trains the candidate, writes candidate/method/task state, generates the next architecture using a mutation function, and advances through an ordered list of search phases.
</pre>
</details>

### Block 210

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.1 Candidate mutation primitives
</pre>
</details>

### Block 211

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Primitive | Default implementation | Architecture effect
</pre>
</details>

### Block 212

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Width | mlp.expand_width() | Round-robin widening of one hidden layer by ex_k_width
</pre>
</details>

### Block 213

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
All-layer width (available helper) | mlp.expand_width_all_layers() | Every hidden layer widened simultaneously
</pre>
</details>

### Block 214

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Depth | mlp.expand_depth() | Append min_new_layer_width (default 1), only when existing stack is uniform
</pre>
</details>

### Block 215

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Matched-width depth (available helper) | mlp.expand_depth_match_width() | Append one layer matching last hidden width
</pre>
</details>

### Block 216

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.2 Mode-to-phase mapping
</pre>
</details>

### Block 217

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
adp_mode | Phase order
</pre>
</details>

### Block 218

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
width_only | width
</pre>
</details>

### Block 219

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
depth_only | depth
</pre>
</details>

### Block 220

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
width_to_depth (default/fallback) | width → depth
</pre>
</details>

### Block 221

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
depth_to_width / alt_depth | depth → width
</pre>
</details>

### Block 222

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.3 Candidate training protocol differs from dedicated DEN
</pre>
</details>

### Block 223

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Each candidate is trained through utils/adp_external_adapter.train_with_early_stopping(), not the class-local DEN trainer. That adapter uses AdamW with configured weight decay and optional gradient clipping. adp_search.py explicitly calls it with is_penalty=True, so the generic-search training path can include application-specific auxiliary penalties in addition to the prediction loss.
</pre>
</details>

### Block 224

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
adp_search candidate loss path:

Penalty-capable adapter:
    prediction = MSE(y_hat, y)

    if lambda_aux_1 == 0 and lambda_aux_2 == 0:
        total = prediction
    else:
        total =
            lambda_prediction * prediction
          + lambda_aux_1      * auxiliary_constraint_term_1
          + lambda_aux_2      * auxiliary_constraint_term_2

_optimizer = AdamW(...)

The auxiliary terms are application-specific; they are not part of the architecture-search definition.
</pre>
</details>

### Block 225

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Therefore adp_search.py is not guaranteed to be a pure prediction-loss-only experiment even though it resides under the VANILLA directory. It becomes MSE-only when the auxiliary penalty weights are zero. The architecture-search logic itself can be understood independently of what those auxiliary application-specific penalties represent.
</pre>
</details>

### Block 226

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.4 Generic width_only mode
</pre>
</details>

### Block 227

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
candidate 0: train A0

repeat:
    A_next ← expand_width(A_current)
             # default: one-layer round-robin mutation
    if no legal width mutation:
        stop

    train A_next as a separate candidate
    update method/task state
    update width failure counter
    continue until trial limit / max_candidates / width limit
</pre>
</details>

### Block 228

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
The search moves forward from the most recently trained candidate rather than reconstructing the historical best before proposing the next model. It therefore shares DEN&#x27;s ability, in principle, to traverse a bad intermediate architecture.
</pre>
</details>

### Block 229

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.5 Generic depth_only mode
</pre>
</details>

### Block 230

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
The default depth mutation appends a narrow layer of min_new_layer_width (default 1), but unlike the dedicated pure-depth DEN controller, adp_search.py does not contain a terminal-width catch-up loop. The next call to expand_depth() also requires the hidden stack to be uniform.
</pre>
</details>

### Block 231

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
If current A = [16,16]:

first depth proposal:
    [16,16] → [16,16,1]

next depth proposal:
    rejected because len(set([16,16,1])) != 1

Therefore, with width &gt; 1 and default deepen_fn,
generic depth_only typically performs at most one narrow depth insertion.
</pre>
</details>

### Block 232

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
If all existing widths are already 1, repeated depth additions remain possible because the stack stays uniform. A caller can also override deepen_fn, for example using a matched-width depth mutation, to obtain different behavior.
</pre>
</details>

### Block 233

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.6 Generic width_to_depth mode
</pre>
</details>

### Block 234

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
The width phase runs first. Once width mutation becomes unavailable or its trial logic advances phases, the search switches to depth. Because default width mutation is round-robin, the final width-phase architecture may be non-uniform; in that case the default depth mutation can immediately return None because it requires equal hidden widths.
</pre>
</details>

### Block 235

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Thus width_to_depth is only guaranteed to reach a meaningful default depth phase when the width phase ends on a synchronized hidden stack (or when the user supplies a custom deepen_fn that accepts non-uniform widths).
</pre>
</details>

### Block 236

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.7 Generic depth_to_width mode
</pre>
</details>

### Block 237

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
This phase order first applies the default narrow depth mutation and then switches to width. Because width mutation can operate on non-uniform stacks, it can subsequently densify the architecture one layer at a time. This is not the same controller as the dedicated train_task_depth_to_width(): the generic path uses independent MLP candidates and phase switching, whereas the dedicated routine nests a depth sweep inside width rounds.
</pre>
</details>

### Block 238

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.8 Current failure-counter defect
</pre>
</details>

### Block 239

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
The current implementation updates stage_best_val using min(stage_best_val, best_val) before testing whether best_val is smaller than stage_best_val - delta. With positive delta, a value cannot be less than a minimum that already includes itself. As a result the success branch is normally unreachable at that point.
</pre>
</details>

### Block 240

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Current order:

stage_best_val = min(stage_best_val, best_val)

then:
    if best_val &lt; stage_best_val - delta:
        failure = 0
    else:
        failure += 1

But after the min:
    stage_best_val ≤ best_val

So for delta &gt; 0:
    best_val &lt; stage_best_val - delta
is impossible.
</pre>
</details>

### Block 241

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Practically, trials_width/trials_depth can behave like counts of attempted candidates rather than true counts of consecutive non-improvements.
</pre>
</details>

### Block 242

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.9 Current global-best bookkeeping defect
</pre>
</details>

### Block 243

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
The variable best_val is overwritten with the current candidate&#x27;s loss. best_candidate_dir is then conditionally updated relative to the previous method_state best_val, which itself stores the previous current candidate rather than an immutable global minimum. A later candidate that merely improves over a bad immediate predecessor can therefore overwrite the recorded best directory even when it is worse than an older historical optimum.
</pre>
</details>

### Block 244

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Example:

candidate A: 0.80  → recorded best = A
candidate B: 0.90  → best directory remains A, but method_state.best_val becomes 0.90
candidate C: 0.85  → C improves over 0.90, so it may overwrite best directory

Yet:
    0.85 &gt; 0.80
so C is not the true global best.
</pre>
</details>

### Block 245

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.10 max_neurons is recorded but not enforced by the generic search controller
</pre>
</details>

### Block 246

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
adp_search.py stores max_neurons in configuration signatures/manifests, but its default _next_model_for_stage() selects width/depth mutations using max_width and max_depth only. The mlp.py mutation helpers invoked there also do not consult max_neurons. Therefore the generic candidate search does not presently enforce the same aggregate neuron cap as the dedicated DEN classes.
</pre>
</details>

### Block 247

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
8.11 What adp_search does robustly
</pre>
</details>

### Block 248

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Creates one durable directory per candidate and records candidate_state.json plus best/last checkpoints.
</pre>
</details>

### Block 249

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Creates method-level and task-level state files after candidates, enabling coarse-grained replay/resume.
</pre>
</details>

### Block 250

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Hashes the run manifest and refuses to overwrite a task directory whose configuration signature differs.
</pre>
</details>

### Block 251

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Can load a completed candidate from disk and continue constructing subsequent candidates.
</pre>
</details>

### Block 252

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
Allows caller-provided widen_fn and deepen_fn, making the candidate search more extensible than the dedicated hard-coded controllers.
</pre>
</details>

### Block 253

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
9. Generic MLP mutation primitives in exact detail
</pre>
</details>

### Block 254

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
9.1 expand_width(model, ex_k, max_width)
</pre>
</details>

### Block 255

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
This helper performs round-robin one-layer widening. It keeps a private _opf_width_cursor, scans layers cyclically from that cursor, chooses the first layer whose width is below the limit, increments it by max(1, ex_k), rebuilds the MLP, copies all overlapping state, and moves the cursor to the following layer.
</pre>
</details>

### Block 256

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
target = hidden_widths.copy()
cursor = _opf_width_cursor mod number_of_hidden_layers

for step in 0..D-1:
    i = (cursor + step) mod D
    if target[i] &lt; max_width:
        target[i] = min(max_width, target[i] + max(1, ex_k))
        new_model = rebuild(target)
        new_model._opf_width_cursor = (i + 1) mod D
        return new_model

return None
</pre>
</details>

### Block 257

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
9.2 expand_width_all_layers(model, ex_k, max_width)
</pre>
</details>

### Block 258

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
delta = max(1, ex_k)
target[i] = min(max_width, hidden_widths[i] + delta) for every hidden layer

if target == hidden_widths:
    return None

return rebuilt/copy-inherited MLP(target)
</pre>
</details>

### Block 259

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
9.3 expand_depth(model, max_depth, min_new_layer_width=1)
</pre>
</details>

### Block 260

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
if current_depth ≥ max_depth:
    return None

if hidden stack exists and widths are not all equal:
    return None

target = hidden_widths + [max(1, min_new_layer_width)]
return rebuilt/copy-inherited MLP(target)
</pre>
</details>

### Block 261

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
9.4 expand_depth_match_width(model, max_depth)
</pre>
</details>

### Block 262

**Review area:** Generic candidate search and mutation primitives  
**Code crosswalk:** `generic_search.py; growth.py; training.py; artifacts.py`  
**Audit section:** generic search and mutations

<details>
<summary>Source text</summary>

<pre>
if current_depth ≥ max_depth or no hidden layer exists:
    return None

target = hidden_widths + [hidden_widths[-1]]
return rebuilt/copy-inherited MLP(target)
</pre>
</details>

### Block 263

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
10. Checkpointing, restoration, and search continuity
</pre>
</details>

### Block 264

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Checkpointing is unusually detailed in the dedicated classes. There are stage-level last checkpoints, best-epoch checkpoints, and best-batch checkpoints. Telemetry also records stage state so an interrupted architectural proposal can be reconstructed before weight restoration.
</pre>
</details>

### Block 265

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
10.1 Stage-level recovery sequence
</pre>
</details>

### Block 266

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
If task stage_state says COMPLETED:
    restore recorded best stage checkpoint
    return recorded best_val

Else if stage_state describes an ACTIVE stage:
    rebuild the exact recorded current_hidden_widths
    restore stage-local last checkpoint
    continue TRAIN_STAGE from recorded epoch/batch state
    evaluate the completed stage under the normal architecture rule

Else if a non-final completed stage exists:
    restore its current architecture and stage-local latest weights
    continue proposing the next architecture
</pre>
</details>

### Block 267

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
10.2 Separation of global best and current forward state
</pre>
</details>

### Block 268

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The design deliberately keeps two conceptual states: (1) a global-best snapshot/checkpoint, and (2) the current, potentially worse, forward-search architecture. This separation is essential to the DEN valley-crossing behavior and is preserved in stage-state metadata.
</pre>
</details>

### Block 269

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
10.3 Stale checkpoint defense
</pre>
</details>

### Block 270

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Before restoring a stage checkpoint, the dedicated training code compares checkpoint architecture metadata with the current grown architecture. A checkpoint belonging to a different layer shape is skipped rather than blindly loaded. Partial-state restoration helpers are used where architecture reconstruction changes tensor shapes.
</pre>
</details>

### Block 271

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
11. Exact algorithmic comparison matrix
</pre>
</details>

### Block 272

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Algorithm | Mutation | Failure policy | Acceptance | Crosses bad step? | Rollback | Stage trainer
</pre>
</details>

### Block 273

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Fixed MLP | None | None | N/A | No | N/A | MSE / baseline trainer
</pre>
</details>

### Block 274

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
DEN width all | All layers +k | pW failures | δW margin | Yes | End of search | MSE; Adam + cosine restarts
</pre>
</details>

### Block 275

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
DEN width RR | One layer +k | pW failures | δW margin | Yes | End of search | MSE; Adam + cosine restarts
</pre>
</details>

### Block 276

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
DEN depth progressive | Append width 1; catch up | pD failures | δD margin | Yes | End of search | MSE; Adam + cosine restarts
</pre>
</details>

### Block 277

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
DEN depth matched | Append width W | pD failures | δD margin | Yes | End of search | MSE; Adam + cosine restarts
</pre>
</details>

### Block 278

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Nested DEN | Width outer + depth inner | pW and pD | δW/δD | Yes | End of nested search | MSE; Adam + cosine restarts
</pre>
</details>

### Block 279

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
LG-ND | Width expansion | 0 tolerated failures | Strict Lnew &lt; Lbest | No | Immediately after first failure | MSE; Adam + cosine restarts
</pre>
</details>

### Block 280

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
adp_search width | Generic RR width candidate | Intended trials_width | Current logic defective | Intended | Candidate state | Penalty-capable; AdamW
</pre>
</details>

### Block 281

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
adp_search depth | Append narrow layer | Intended trials_depth | Current logic defective | Limited | Candidate state | Penalty-capable; AdamW
</pre>
</details>

### Block 282

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
adp_search W→D | Width phase then depth | Per phase | Current logic defective | Limited | Candidate state | Penalty-capable; AdamW
</pre>
</details>

### Block 283

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
adp_search D→W | Depth phase then width | Per phase | Current logic defective | Limited | Candidate state | Penalty-capable; AdamW
</pre>
</details>

### Block 284

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
12. Worked architecture traces
</pre>
</details>

### Block 285

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
12.1 All-layer DEN trace
</pre>
</details>

### Block 286

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Configuration:
    A0 = [4,4]
    ex_k_width = 1
    patience_width_exp = 2
    delta_width = 0.001

Stage 0: [4,4] → val 0.100
    best = 0.100, failures = 0

Stage 1 proposal: [5,5] → val 0.090
    0.090 &lt; 0.100 - 0.001
    best = 0.090, failures = 0

Stage 2 proposal: [6,6] → val 0.091
    not &lt; 0.089
    failures = 1
    CURRENT SEARCH MODEL REMAINS [6,6]

Stage 3 proposal: [7,7] → val 0.087
    0.087 &lt; 0.090 - 0.001
    best = 0.087, failures = 0

...
On termination: restore best snapshot, not necessarily last architecture.
</pre>
</details>

### Block 287

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
12.2 Round-robin DEN trace
</pre>
</details>

### Block 288

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Configuration:
    A0 = [4,4,4]
    cursor = 0
    ex_k_width = 1

Proposal sequence:
    [4,4,4]
    [5,4,4]   cursor→1
    [5,5,4]   cursor→2
    [5,5,5]   cursor→0
    [6,5,5]
    [6,6,5]
    [6,6,6]
    ...

Every listed architecture is independently trained and judged by the outer
patience logic in the canonical round-robin controller.
</pre>
</details>

### Block 289

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
12.3 Progressive depth trace
</pre>
</details>

### Block 290

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Configuration:
    A0 = [4,4]
    seed_new_depth_to_current_width = False

Initial:
    train [4,4]

Depth proposal:
    [4,4,1]
    train
    grow terminal width → [4,4,2]
    train
    grow terminal width → [4,4,3]
    train
    grow terminal width → [4,4,4]
    train completed deeper candidate
    compare completed candidate against global best using δD

If accepted:
    best snapshot becomes depth 3.

If rejected:
    depth_fail += 1 but the deeper architecture remains the forward state
    until search termination.
</pre>
</details>

### Block 291

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
12.4 LG-ND trace
</pre>
</details>

### Block 292

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
A0 [4,4]  : 0.100
A1 [5,5]  : 0.090  → improve, continue
A2 [6,6]  : 0.089  → improve, continue
A3 [7,7]  : 0.0895 → NOT strictly lower than 0.089

Stop immediately.
Restore A2.
Never evaluate [8,8].
</pre>
</details>

### Block 293

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
12.5 Generic adp_search depth limitation trace
</pre>
</details>

### Block 294

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Base candidate:
    [4,4]

default expand_depth:
    [4,4,1]

next expand_depth check:
    len(set([4,4,1])) = 2
    → reject

No automatic [4,4,2] → [4,4,4] catch-up exists in adp_search.py.
This differs fundamentally from dedicated pure-depth DEN.
</pre>
</details>

### Block 295

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
13. Implementation issues and experimental confounds
</pre>
</details>

### Block 296

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Issue | Exact mechanism | Experimental consequence
</pre>
</details>

### Block 297

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Cross-copy controller drift | Nominal head-only copies change RR increment/budget checks or depth candidate scoring. | Head comparisons can unintentionally compare different architecture searches.
</pre>
</details>

### Block 298

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
LG-ND unreachable legacy code | Wrapper returns through shared helper before duplicated local body. | Read shared helper as authoritative runtime behavior.
</pre>
</details>

### Block 299

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
LG-ND RR capacity estimate | Pre-check assumes all-layer D×k growth. | Round-robin LG-ND can stop conservatively.
</pre>
</details>

### Block 300

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Generic search penalty path | adp_search passes is_penalty=True. | Not automatically comparable to vanilla MSE-only DEN.
</pre>
</details>

### Block 301

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Generic search optimizer mismatch | AdamW vs dedicated Adam + cosine restarts. | Training protocol confounds architecture-policy comparisons.
</pre>
</details>

### Block 302

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Generic search failure counter | stage_best updated before improvement test. | Failure reset condition is effectively unreachable for δ&gt;0.
</pre>
</details>

### Block 303

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Generic search global best | best_val is current candidate, not durable global min. | best_candidate_dir can be overwritten by a suboptimal rebound.
</pre>
</details>

### Block 304

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Generic search max_neurons | Stored in config but not applied to default next-model generation. | Capacity budget differs from dedicated classes.
</pre>
</details>

### Block 305

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Generic depth semantics | No terminal catch-up after narrow depth insertion. | Depth-only generic search is not the same algorithm as depth DEN.
</pre>
</details>

### Block 306

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Unit budget vs parameter budget | max_neurons counts units. | Compute/memory scaling can be much larger than budget number suggests.
</pre>
</details>

### Block 307

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Non-function-preserving expansion | Fresh new rows/columns and new depth layer alter function. | Immediate post-growth loss may reflect perturbation as well as useful capacity.
</pre>
</details>

### Block 308

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
13.1 Which issues are design choices versus probable defects?
</pre>
</details>

### Block 309

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Category | Items
</pre>
</details>

### Block 310

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Intentional design choices | Forward-only DEN valley crossing; width-1 depth insertion; architecture-level patience; strict first-failure LG-ND; warm-start by overlap copy; hard capacity ceilings.
</pre>
</details>

### Block 311

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Configurable design alternatives | All-layer vs round-robin width; narrow vs matched-width depth seed; loop order; custom widen_fn/deepen_fn in generic search.
</pre>
</details>

### Block 312

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Implementation divergences to normalize | Round-robin increment/budget differences across duplicated files; intermediate depth candidate scoring differences; missing nested mode in one duplicated controller.
</pre>
</details>

### Block 313

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Likely correctness issues | adp_search failure-counter order; global-best bookkeeping; max_neurons non-enforcement; VANILLA search invoking penalty-enabled adapter by default.
</pre>
</details>

### Block 314

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
14. Research-level interpretation of the search policies
</pre>
</details>

### Block 315

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
14.1 Search graph view
</pre>
</details>

### Block 316

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Let every architecture A=[h1,…,hD] be a node in a discrete graph. Width and depth mutations define directed edges. The algorithms differ primarily in which edge is chosen and whether a non-improving node is allowed to remain on the forward trajectory.
</pre>
</details>

### Block 317

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
All-layer width edge:
    TWall(A) = [h1+k, ..., hD+k]

Round-robin width edge at layer i:
    TW,i(A) = [h1, ..., hi+k, ..., hD]

Narrow depth edge:
    TD,1(A) = [h1, ..., hD, 1]

Matched-width depth edge:
    TD,W(A) = [h1, ..., hD, hD]
</pre>
</details>

### Block 318

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
14.2 DEN as bounded non-monotonic hill traversal
</pre>
</details>

### Block 319

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
DEN is not greedy hill climbing because it permits several steps whose validation scores fail the acceptance test. However it is not a general global search either: proposals are monotonic in capacity and follow one deterministic growth trajectory. It is best viewed as a bounded non-monotonic traversal along a predefined architecture path.
</pre>
</details>

### Block 320

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
14.3 LG-ND as first-descent termination
</pre>
</details>

### Block 321

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
LG-ND assumes that once marginal capacity fails to reduce validation loss, subsequent larger models along the same path are not worth exploring. This is computationally economical but imposes a monotonicity assumption on useful capacity.
</pre>
</details>

### Block 322

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
14.4 Progressive depth as architecture continuation
</pre>
</details>

### Block 323

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The narrow-layer depth algorithm treats a new layer as something that is gradually &#x27;grown into&#x27; the network rather than inserted at final size. That creates a continuation-like path through architectures [W,…,W,1], [W,…,W,2], …, [W,…,W,W]. Because the transformation is not function preserving, the trajectory simultaneously changes capacity and perturbs the function.
</pre>
</details>

### Block 324

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
14.5 Search complexity intuition
</pre>
</details>

### Block 325

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
For a fixed maximum width, all-layer search evaluates roughly one candidate per global width level. Round-robin search can evaluate roughly D times as many width mutations for a D-layer stack. Progressive depth insertion may train up to W intermediate terminal widths per added layer. Nested width/depth search multiplies these effects by running a depth sweep after width proposals.
</pre>
</details>

### Block 326

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Very rough candidate-count intuition:

All-layer width:
    O(number_of_width_levels)

Round-robin width:
    O(D × number_of_width_levels)

Progressive depth:
    O(sum over depth stages of target_width)

Nested width/depth:
    O(width rounds × depth trials)   plus inner epoch training cost
</pre>
</details>

### Block 327

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
15. Reproducibility specification for each normalized method
</pre>
</details>

### Block 328

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
To reproduce the algorithms independently of output heads, the following settings must be reported because each changes the actual architecture trajectory rather than merely tuning optimization.
</pre>
</details>

### Block 329

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Initial hidden_widths (or init_width and init_depth).
</pre>
</details>

### Block 330

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
width_growth_strategy: all_layers versus round_robin.
</pre>
</details>

### Block 331

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ex_k_width and, where relevant, ex_k_depth.
</pre>
</details>

### Block 332

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
seed_new_depth_to_current_width.
</pre>
</details>

### Block 333

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
loop_order for controllers that expose it.
</pre>
</details>

### Block 334

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
patience_es, patience_width_exp, patience_depth_exp.
</pre>
</details>

### Block 335

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
delta_width and delta_depth.
</pre>
</details>

### Block 336

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
max_width, max_depth, max_neurons.
</pre>
</details>

### Block 337

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Learning rate and scheduler parameters for dedicated DEN/LG-ND.
</pre>
</details>

### Block 338

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
For adp_search: lambda_loss/lambda_eq/lambda_ineq, weight_decay, grad_clip, max_candidates, and any custom widen_fn/deepen_fn.
</pre>
</details>

### Block 339

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Checkpoint/resume state, because resumed architecture cursors and current forward states affect the subsequent proposal sequence.
</pre>
</details>

### Block 340

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
15.1 Minimal normalized method names
</pre>
</details>

### Block 341

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Suggested name | Meaning
</pre>
</details>

### Block 342

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-W-All-Patience | All-layer width expansion + architecture failure patience
</pre>
</details>

### Block 343

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-W-RR-Patience | Round-robin layer-wise width expansion + architecture failure patience
</pre>
</details>

### Block 344

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-D-Progressive-Patience | Width-1 depth insertion + progressive terminal densification + depth patience
</pre>
</details>

### Block 345

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-D-Matched-Patience | Full-width depth insertion + depth patience
</pre>
</details>

### Block 346

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-WD-Nested-Patience | Outer width growth with inner depth sweep
</pre>
</details>

### Block 347

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
LG-ND-W-Greedy | Width densification; stop on first non-improvement
</pre>
</details>

### Block 348

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-Search-W | Generic candidate search, width_only mode
</pre>
</details>

### Block 349

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-Search-D | Generic candidate search, depth_only mode
</pre>
</details>

### Block 350

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-Search-WD | Generic candidate search, width_to_depth mode
</pre>
</details>

### Block 351

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-Search-DW | Generic candidate search, depth_to_width mode
</pre>
</details>

### Block 352

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
16. Concrete source/runners mapping
</pre>
</details>

### Block 353

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The table below maps the normalized algorithms to the concrete repository entry points. Output-head-specific files are listed only to show where the same intended controller is replicated; the algorithm descriptions above remain head-independent.
</pre>
</details>

### Block 354

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Normalized method | Core source | Runner/configuration
</pre>
</details>

### Block 355

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-W-All-Patience | models/adp_den_width_only.py | examples/run_adp_den_width_only.py
</pre>
</details>

### Block 356

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-W-RR-Patience | models/adp_den_width_only.py | examples/run_adp_den_width_only_round_robin.py
</pre>
</details>

### Block 357

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-D-Progressive-Patience | models/adp_den_depth_only.py | examples/run_adp_den_depth_only.py
</pre>
</details>

### Block 358

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-D-Matched-Patience | models/adp_den_depth_only.py | No separate standard runner; enable seed_new_depth_to_current_width
</pre>
</details>

### Block 359

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-WD-Nested-Patience | models/adp_den_depth_only.py (and duplicated 2-output copy) | Enable loop_order=depth_to_width
</pre>
</details>

### Block 360

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
LG-ND-W-Greedy | models/adp_lg_nd_width_only.py + shared helper in adp_den_width_only.py | examples/run_adp_lg_nd_width_only.py
</pre>
</details>

### Block 361

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ADP-Search-* | models/adp_search.py + models/mlp.py | Programmatic adp_search(...) caller / mode configuration
</pre>
</details>

### Block 362

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
16.1 Repository links
</pre>
</details>

### Block 363

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Models directory: https://github.com/Anurag9000/OPF_ADP/tree/main/DNN/VANILLA/Dyn_DNN4OPF/models
</pre>
</details>

### Block 364

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
mlp.py: https://github.com/Anurag9000/OPF_ADP/blob/main/DNN/VANILLA/Dyn_DNN4OPF/models/mlp.py
</pre>
</details>

### Block 365

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
adp_den_width_only.py: https://github.com/Anurag9000/OPF_ADP/blob/main/DNN/VANILLA/Dyn_DNN4OPF/models/adp_den_width_only.py
</pre>
</details>

### Block 366

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
adp_den_depth_only.py: https://github.com/Anurag9000/OPF_ADP/blob/main/DNN/VANILLA/Dyn_DNN4OPF/models/adp_den_depth_only.py
</pre>
</details>

### Block 367

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
adp_search.py: https://github.com/Anurag9000/OPF_ADP/blob/main/DNN/VANILLA/Dyn_DNN4OPF/models/adp_search.py
</pre>
</details>

### Block 368

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
LG-ND wrapper: https://github.com/Anurag9000/OPF_ADP/blob/main/DNN/VANILLA/Dyn_DNN4OPF/models/adp_lg_nd_width_only.py
</pre>
</details>

### Block 369

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
16.2 Authoritative model-count equation
</pre>
</details>

### Block 370

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
3 + (2×2) + (2×2) + 2 + 1 = 14 total models = 13 adaptive + 1 fixed baseline.
</pre>
</details>

### Block 371

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
17. One-page conceptual reference
</pre>
</details>

### Block 372

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Question | DEN width | DEN depth | Nested DEN | LG-ND | adp_search
</pre>
</details>

### Block 373

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
What changes? | Width | Depth + optional catch-up width | Width + depth | Width | Mode-dependent
</pre>
</details>

### Block 374

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Proposal granularity | All-layer or one-layer | New layer, often width 1 | Hierarchical | Inherited width growth | Generic candidate
</pre>
</details>

### Block 375

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Keeps failed architecture as forward state? | Yes | Yes | Yes | No | Generally yes
</pre>
</details>

### Block 376

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Failure tolerance | pW | pD | pW outside pD | 0 | Intended trials counters
</pre>
</details>

### Block 377

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Global rollback | At end | At end | At end | At first failure/end | Candidate bookkeeping
</pre>
</details>

### Block 378

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Stage optimizer | Adam + cosine | Adam + cosine | Adam + cosine | Adam + cosine | AdamW
</pre>
</details>

### Block 379

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Vanilla MSE guaranteed? | Yes | Yes | Yes | Yes | No; penalty path enabled
</pre>
</details>

### Block 380

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Bottom line: the repository does not contain one ADP algorithm with cosmetic variants. It contains multiple distinct architecture-search policies. The scientifically meaningful axes are proposal dimension (width/depth), proposal granularity (all-layer/layer-wise/progressive), failure tolerance (patience/first-failure), and search ordering (single-axis/nested/phased). Output-head count should be treated separately from these controller differences.
</pre>
</details>

### Block 381

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
18. Symmetry audit: what exists, what is only analogous, and what is missing
</pre>
</details>

### Block 382

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The architecture family is not fully symmetric. Three composition patterns must be separated: (a) single-axis growth, (b) sequential phase ordering, and (c) genuinely nested outer/inner search. The repository implements different subsets of these patterns.
</pre>
</details>

### Block 383

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Design question | Width→Depth side | Depth→Width side | Symmetric? | Exact status
</pre>
</details>

### Block 384

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Single-axis patience search | Width DEN exists | Depth DEN exists | Broadly yes | Both axes have dedicated patience controllers, but mutation mechanics are intentionally different.
</pre>
</details>

### Block 385

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Sequential phase order | adp_search width→depth | adp_search depth→width | Yes | Both orders exist; these are sequential phases, not nested searches.
</pre>
</details>

### Block 386

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Nested search | Outer WIDTH with inner DEPTH sweep exists | No free outer DEPTH with inner GLOBAL WIDTH sweep | No | This is the main missing symmetry.
</pre>
</details>

### Block 387

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Constrained nested-like growth | — | Depth stage contains terminal-width catch-up | Only analogous | Inner width is predetermined catch-up of the new layer, not an independent width search.
</pre>
</details>

### Block 388

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Greedy first-failure search | LG-ND width exists | No LG-ND depth counterpart | No | Second major symmetry gap.
</pre>
</details>

### Block 389

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Mutation granularity | All-layer and round-robin widening | Width-1 or matched-width layer insertion | Not a mirror pair | Different axes expose different growth primitives.
</pre>
</details>

### Block 390

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
18.1 Existing true nested controller: outer width, inner depth
</pre>
</details>

### Block 391

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The dedicated alternate routine named train_task_depth_to_width()/fit_task_depth_to_width() is, by executed control flow, an OUTER WIDTH search containing an INNER DEPTH search. After each width proposal, the helper repeatedly proposes deeper architectures until depth patience is exhausted; the outer controller then decides whether that entire width round improved enough to reset width patience.
</pre>
</details>

### Block 392

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
OUTER WIDTH LOOP:
    propose wider architecture

    INNER DEPTH LOOP:
        while depth_fail &lt; pD:
            append depth
            train
            compare using δD
            update depth_fail

    compare round-level best improvement using δW
    update width_fail
</pre>
</details>

### Block 393

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
18.2 Existing outer-depth / inner-width-like mechanism: terminal catch-up only
</pre>
</details>

### Block 394

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The default progressive depth controller does have an outer-depth / inner-width-looking structure, but it is NOT the symmetric inverse of the nested controller above. For each new depth stage it appends a narrow layer (normally width 1) and then grows that terminal layer toward a predetermined target width. The inner loop does not independently search global network width, does not use its own width-failure patience, and does not explore arbitrary width trajectories.
</pre>
</details>

### Block 395

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
OUTER DEPTH STAGE:
    append new layer at width 1

    INNER TERMINAL-WIDTH CATCH-UP:
        while new_layer_width &lt; previous_stage_width:
            train
            new_layer_width += 1

    evaluate completed deeper architecture using δD
</pre>
</details>

### Block 396

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
18.3 Fully symmetric nested controller that is NOT implemented
</pre>
</details>

### Block 397

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
At the model-grid level, the Depth→Width block supplies the four order-reversed counterparts obtained by crossing the two depth seeds (width 1/current width) with the two widening rules (round-robin/all-layer). At the source-code level, however, not every member is necessarily exposed as a separately named runner or a completely independent controller body; several are configuration-level realizations of shared growth primitives. This distinction is why counting only explicit classes/runners undercounts the intended 14-model experiment family.
</pre>
</details>

### Block 398

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
CONCEPTUAL SYMMETRIC ALGORITHM — NOT PRESENT IN CURRENT CODE:

OUTER DEPTH LOOP:
    propose a deeper architecture
    width_fail = 0

    INNER GLOBAL WIDTH LOOP:
        while width_fail &lt; pW and width growth is legal:
            widen architecture (all-layer or round-robin)
            train
            if loss &lt; inner_best - δW:
                record inner best
                width_fail = 0
            else:
                width_fail += 1

    compare depth-round improvement against pre-depth best using δD
    update depth_fail

FINAL:
    restore global best architecture
</pre>
</details>

### Block 399

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Therefore symmetry should be assessed at two levels: the experiment/model grid is symmetric across the 2×2 mixed-order combinations, while the current implementation may realize those combinations through shared or asymmetrically structured controller code.
</pre>
</details>

### Block 400

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
18.4 Sequential D→W is not nested D[W]
</pre>
</details>

### Block 401

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The generic adp_search depth_to_width mode runs one depth phase and then one width phase. Width is not restarted as an inner search for each new outer depth. Therefore it is symmetric to width_to_depth only at the sequential-phase level, not at the nested-search level.
</pre>
</details>

### Block 402

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
18.5 LG-ND symmetry gap
</pre>
</details>

### Block 403

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
LG-ND remains a width-only greedy member of the 14-model grid. It is intentionally one of the three width-only methods rather than part of the 2×2 mixed-order block. The 14-model count does not require a depth-greedy LG-ND counterpart.
</pre>
</details>

### Block 404

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
18.6 Symmetry-aware interpretation of the 14-model grid
</pre>
</details>

### Block 405

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The authoritative experiment-level taxonomy contains 14 total models: 13 adaptive configurations plus the fixed MLP baseline. The previous smaller count collapsed parameterized mixed-order configurations into shared controller families. For model accounting, width-growth strategy, depth-seeding strategy, and search order must remain explicit because they generate different architecture trajectories.
</pre>
</details>

### Block 406

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
19. Compact canonical names for all 14 models
</pre>
</details>

### Block 407

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
# | Canonical name | Definition
</pre>
</details>

### Block 408

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
1 | LGND-W-Greedy | Width only; first non-improvement stops
</pre>
</details>

### Block 409

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
2 | ADP-W-RR | Width only; round-robin + patience
</pre>
</details>

### Block 410

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
3 | ADP-W-All | Width only; all-layer + patience
</pre>
</details>

### Block 411

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
4 | ADP-WD-RR-D1 | Width→Depth; RR width; depth seed 1
</pre>
</details>

### Block 412

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
5 | ADP-WD-All-Dcur | Width→Depth; all-layer width; depth seed current
</pre>
</details>

### Block 413

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
6 | ADP-WD-RR-Dcur | Width→Depth; RR width; depth seed current
</pre>
</details>

### Block 414

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
7 | ADP-WD-All-D1 | Width→Depth; all-layer width; depth seed 1
</pre>
</details>

### Block 415

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
8 | ADP-DW-D1-RR | Depth→Width; depth seed 1; RR width
</pre>
</details>

### Block 416

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
9 | ADP-DW-Dcur-All | Depth→Width; depth seed current; all-layer width
</pre>
</details>

### Block 417

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
10 | ADP-DW-Dcur-RR | Depth→Width; depth seed current; RR width
</pre>
</details>

### Block 418

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
11 | ADP-DW-D1-All | Depth→Width; depth seed 1; all-layer width
</pre>
</details>

### Block 419

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
12 | ADP-D-D1 | Depth only; stage starts at width 1
</pre>
</details>

### Block 420

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
13 | ADP-D-Dcur | Depth only; stage starts at current width
</pre>
</details>

### Block 421

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
14 | MLP-Fixed | Fixed one-shot architecture grid/baseline
</pre>
</details>

### Block 422

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Appendix A. Canonical pseudocode library
</pre>
</details>

### Block 423

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
A.1 Patience-based all-layer width DEN
</pre>
</details>

### Block 424

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
function DEN_WIDTH_ALL(model):
    best_val, _ = TRAIN_STAGE(model)
    best_snap = snapshot(model)
    fail = 0

    while fail &lt; pW and can_grow_all_layers(kW):
        grow_width_all_layers(model, kW)
        val, _ = TRAIN_STAGE(model)

        if val &lt; best_val - δW:
            best_val = val
            best_snap = snapshot(model)
            fail = 0
        else:
            fail += 1

    restore(model, best_snap)
    return best_val
</pre>
</details>

### Block 425

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
A.2 Patience-based round-robin width DEN
</pre>
</details>

### Block 426

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
function DEN_WIDTH_RR(model):
    best_val, _ = TRAIN_STAGE(model)
    best_snap = snapshot(model)
    fail = 0

    while fail &lt; pW and can_grow_one_layer(kW):
        target = scalar_width(model) + kW
        grow_first_eligible_layer_from_cursor(model, target, kW)
        val, _ = TRAIN_STAGE(model)

        if val &lt; best_val - δW:
            best_val = val
            best_snap = snapshot(model)
            fail = 0
        else:
            fail += 1

    restore(model, best_snap)
    return best_val
</pre>
</details>

### Block 427

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
A.3 Canonical progressive depth DEN
</pre>
</details>

### Block 428

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
function DEN_DEPTH_PROGRESSIVE(model):
    best_val, _ = TRAIN_STAGE(model)
    best_snap = snapshot(model)
    depth_fail = 0

    while depth_fail &lt; pD and can_add_depth():
        Wstart = current_width(model)
        append_new_layer(width = 1)

        while terminal_width(model) &lt; Wstart and depth_fail &lt; pD:
            TRAIN_STAGE(model)               # optimization stepping stone
            grow_terminal_width(+1)

        if depth_fail &gt;= pD:
            break

        val, _ = TRAIN_STAGE(model)           # completed deeper candidate

        if val &lt; best_val - δD:
            best_val = val
            best_snap = snapshot(model)
            depth_fail = 0
        else:
            depth_fail += 1

    restore(model, best_snap)
    return best_val
</pre>
</details>

### Block 429

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
A.4 Matched-width depth DEN
</pre>
</details>

### Block 430

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
function DEN_DEPTH_MATCHED(model):
    best_val, _ = TRAIN_STAGE(model)
    best_snap = snapshot(model)
    depth_fail = 0

    while depth_fail &lt; pD and can_add_depth():
        append_new_layer(width = current_width(model))
        val, _ = TRAIN_STAGE(model)

        if val &lt; best_val - δD:
            best_val = val
            best_snap = snapshot(model)
            depth_fail = 0
        else:
            depth_fail += 1

    restore(model, best_snap)
    return best_val
</pre>
</details>

### Block 431

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
A.5 LG-ND
</pre>
</details>

### Block 432

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
function LG_ND(model):
    best_val, _ = TRAIN_STAGE(model)
    best_snap = snapshot(model)

    while capacity_allows_next_width():
        expand_width(model, kW)
        val, _ = TRAIN_STAGE(model)

        if val &lt; best_val:
            best_val = val
            best_snap = snapshot(model)
        else:
            break

    restore(model, best_snap)
    return best_val
</pre>
</details>

### Block 433

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
A.6 Generic phase-based ADP search
</pre>
</details>

### Block 434

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
function ADP_SEARCH(model, mode):
    phases = {
        width_only:      [WIDTH],
        depth_only:      [DEPTH],
        width_to_depth:  [WIDTH, DEPTH],
        depth_to_width:  [DEPTH, WIDTH]
    }[mode]

    current = model

    for phase in phases:
        while phase-specific termination has not fired:
            train current candidate
            persist candidate/method/task state

            next = mutate(current, phase)
            if next is None:
                break

            current = next
            update phase failure counters

    mark task complete
    return task state
</pre>
</details>

### Block 435

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Appendix B. Parameter semantics
</pre>
</details>

### Block 436

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Parameter | Exact algorithmic role
</pre>
</details>

### Block 437

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
init_width | Initial width used by runners to construct hidden stack.
</pre>
</details>

### Block 438

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
init_depth | Initial hidden-layer count.
</pre>
</details>

### Block 439

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ex_k / ex_k_width | Number of units added by a width proposal; some duplicated RR code hardcodes +1.
</pre>
</details>

### Block 440

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
ex_k_depth | Configured depth increment alias; dedicated expand_depth currently appends one layer per invocation.
</pre>
</details>

### Block 441

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
patience / patience_es | Epoch-level early stopping inside one architecture.
</pre>
</details>

### Block 442

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
trials_width / patience_width_exp | Number of architecture-level width failures tolerated by DEN.
</pre>
</details>

### Block 443

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
trials_depth / patience_depth_exp | Number of architecture-level depth failures tolerated by DEN.
</pre>
</details>

### Block 444

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
delta / delta_width | Required validation margin for width DEN acceptance.
</pre>
</details>

### Block 445

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
delta / delta_depth | Required validation margin for depth DEN acceptance.
</pre>
</details>

### Block 446

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
max_width | Per-layer width ceiling / scalar search limit.
</pre>
</details>

### Block 447

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
max_depth | Hidden-layer count ceiling.
</pre>
</details>

### Block 448

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
max_neurons | Aggregate unit-count ceiling in dedicated classes; not parameter count; not enforced by generic search next-model logic.
</pre>
</details>

### Block 449

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
width_growth_strategy | all_layers/uniform/simultaneous versus round_robin/rr/per_layer.
</pre>
</details>

### Block 450

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
seed_new_depth_to_current_width | False: append width 1. True: append current width.
</pre>
</details>

### Block 451

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
loop_order | Dedicated depth controller dispatch; depth_to_width/d2w/inverse selects nested alternate routine.
</pre>
</details>

### Block 452

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
min_new_layer_width | Generic mlp.expand_depth width for newly appended layer; default 1.
</pre>
</details>

### Block 453

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
max_candidates | Hard candidate-count ceiling in generic adp_search; 0 means no explicit ceiling.
</pre>
</details>

### Block 454

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Appendix C. Source-lock and interpretation notes
</pre>
</details>

### Block 455

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
This report is source-locked to the repository state identified on the title page, but the algorithm descriptions deliberately abstract away the application domain. Repository-specific names are retained only as source-location evidence. Executed control flow, rather than comments alone, is treated as authoritative.
</pre>
</details>

### Block 456

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
Where the same intended controller exists in multiple output-head files, the common behavior is described once. Differences are included only if they alter architecture proposals, candidate scoring, patience, budget checks, or search order.
</pre>
</details>

### Block 457

**Review area:** Checkpoint recovery, traces, reproducibility, taxonomy, and appendices  
**Code crosswalk:** `checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py`  
**Audit section:** checkpoint and taxonomy

<details>
<summary>Source text</summary>

<pre>
The term &#x27;DEN&#x27; in this document refers only to the repository&#x27;s implemented progressive dense-network expansion controller. The description is domain-independent and should not be assumed to reproduce every mechanism of any external Dynamically Expandable Network paper (for example neuron splitting, sparsity regularization, or selective freezing) unless those mechanisms are explicitly present in this code.
</pre>
</details>
