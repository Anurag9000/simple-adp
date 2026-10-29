# Visual ADP implementation ledger

States distinguish source/static implementation from executed tests and science.

| Item | State | Evidence / missing |
|---|---|---|
| Legacy 14 model identities | preserved | registry.py unchanged |
| V01 frozen image feature -> native ADP MLP head | runtime-tested (bounded CPU) | src/simple_adp/visual.py |
| V01 actual candidate growth | runtime-tested (bounded CPU) | delegates to native TabularADP; tests/test_visual_v01.py |
| V01 fixed non-growing control | runtime-tested (bounded CPU) | MLP-Fixed native route, tested in new test |
| V01 concrete torchvision ResNet18/ConvNeXt-tiny/ViT-B16 frozen encoder construction | runtime-tested ResNet18; other two static-complete | vision_encoders.py; local-only optional weights, trained origins not asserted |
| V01 packaged estimator export | static-complete | src/simple_adp/__init__.py |
| CIFAR-10/CIFAR-100/Fashion-MNIST optional loader | runtime-tested API on offline synthetic arrays (not real datasets) | vision_data.py, torchvision optional; no real dataset execution | 
| V01 opt-in coordinated runner and signed split jobs | runtime-tested (toy CPU) | runner/vision.py, plan.py, plugins.py, tests/test_runner_vision.py; toy and offline image provider only |
| V02 residual CNN adapter local MLP stage + native RR/all/D1/Dcur mutation | runtime-tested (bounded CPU); canonical controller and runner missing | visual_inner.py; direct StageTrainer with whole-image gradient, site-specific growth, frozen-state snapshots |
| V03/V04 | missing | requires composite controller and source FFN integration |
| V05/V06a/V06b/V07 | missing | operator/model integration, mutation and resume |
| V08/V09 | missing | multistage/bank routing, isolation and candidate scheduling |
| V10/V11 | missing | ResMLP/GFNet operators and full integration |
| Real image scientific mini-runs and multi-seed benchmarks | missing | do not infer from synthetic CI |
| CPU/CUDA complete regression | partial | 231 selected CPU tests passed including V01 worker and torchvision ResNet18, plus variance contract CI; CUDA/full suite not executed |

Next: implement native inner-MLP adapter
and snapshot codec; connect StageTrainer to composite image models; implement
V02-V04 first. Preserve the 14 canonical methods and old runner job identities. Add loss vs variance
criterion pairs only where the corresponding native controller really implements
the criterion.

**Next:** integrate V02 native mutations into all canonical search-controller lifecycles and the coordinated runner; retain source RR/D1/Dcur semantics before extending V03/V04.
