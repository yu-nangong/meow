# Legacy MEOW workshop baseline (reference only)

Ridge regression + 6 hand-crafted features (`feat.py`). Typical test Pearson ≈ **0.022**.

**CORAL agents should not iterate this code.** Use it only to see how the original workshop loaded HDF5 files.

Run from repo root (with `data/` symlink):

```bash
cd reference/legacy_baseline
PYTHONPATH=. python meow.py
```
