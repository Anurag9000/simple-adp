# Cross-repository certificate for OPF_ADP V01 optimizer checkpoint journal

This test fixture is a verbatim source snapshot of
`Anurag9000/OPF_ADP/vision_adp/v01_resume.py` from commit
`7d5fa0e0f74b9c6659363c5526359e28de83ea53` (file blob SHA
`cd323ad87a97b32b7d045f31a52af539c939af7e`) and its standalone journal test suite from
`Anurag9000/OPF_ADP/tests/test_vision_v01_resume_store.py`
(blob SHA `099f59f5d43644295679699ce72c137b953bc971`).

Purpose: the OPF_ADP repository's GitHub Actions currently fail
without any recorded job steps. This isolated source snapshot
can be executed on the functioning `simple-adp` CI runner to
verify the journal's exact Python code, including real SIGKILL-like
`os._exit` subprocess tests, without claiming that the full
OPF model's runtime was exercised.

This fixture is **not** imported by the simple-adp training
package and has no influence on scientific jobs. Its test
file name is deliberately not `test_*.py`, so the normal
`python -m pytest tests` suite does not duplicate these
external certificate tests. Only the dedicated workflow sets
`PYTHONPATH` to the isolated snapshot namespace.

The source should be refreshed only after verifying its
blob SHA in the OPF repository. A green certificate validates
this exact pinned snapshot, not future OPF changes or GPU
numerical replay.

## Exact training-stage source certificate

The certificate now also pins exact byte slices of the native V01
`evaluate()` and `train_stage()` functions from
`Anurag9000/OPF_ADP/vision_adp/v01.py` (source blob
`b33a51e22a50df24877ee894c8a85a7e1d8b84a1`),
wrapped only with the imports required for independent execution.
The two source functions are not algorithmically modified. The exact
`Dyn_DNN4OPF/utils/shadow_bgd_variance.py` implementation is copied
without modifications (both source and snapshot Git blobs
`0cab481c9be1228d94c231a6abb14fb3bac7be4c`).

`stage_certificate_contracts.py` exercises that training code
against a small deterministic PyTorch model implementing only the
`snapshot_arch_and_state`/`restore_arch_and_state` protocol. It
checks uninterrupted-versus-interrupted matching for original AdamW
and shadow-BGD criteria at both a mid-epoch optimizer step and an
epoch boundary, plus fallback to the previous intact checkpoint after
corrupting the newest optimizer step. It requires exact trained-head
tensors and (where applicable) shadow-BGD sigma and independent
MC generator state.

**Boundary:** this is not a test of ADP_DEN's actual width/depth
mutations, ResNet18/CIFAR inputs, V01's outer growth controller,
or whole-repository OPF execution. Those remain gated by the native
OPF workflow; the source-snapshot certificate proves only the
unchanged native stage loop and its dependencies on synthetic
CPU inputs. The [journal-only CI run
41575834184](https://github.com/Anurag9000/simple-adp/actions/runs/41575834184)
has executed with **six tests passed**. The expanded training-stage
certificate requires its own completed CI result before it may
be described as verified.
