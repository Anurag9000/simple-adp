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
