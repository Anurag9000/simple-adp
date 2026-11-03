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
