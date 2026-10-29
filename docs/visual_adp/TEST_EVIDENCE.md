# Test evidence — vision ADP
Baseline GitHub main inspected: a2e6b6325a80ca6da41b1898ebf71c6ec8829a62.

Before publishing this increment, source/contract audit was performed against
live source. Locally, an isolated substitute-head V01 shape/freeze/manifest
smoke passed 2 tests on CPU (torch 2.10.0), but it is **not** evidence that
the actual repository's TabularADP search or runner has passed.

Authoritative repository tests are in tests/test_visual_v01.py.
The .github/workflows/visual-adp-v01.yml workflow requests these tests and
the existing core regression tests on a fresh Python installation.
Do not mark as executed until a matching GitHub Actions run is inspected.
No CIFAR-10/100, Fashion-MNIST, CUDA, pretrained-backbone or multi-seed
scientific result has been executed in this increment.
