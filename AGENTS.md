# MEOW CORAL task

## Goal

Beat the **workshop template** (Ridge + 6 features, test Pearson ≈ **0.022**).

## Starting template (current default)

- `feat.py` + `mdl.py` + `solution.py` — same logic as MEOW workshop
- `python run_benchmark.py` should be near **0.022**

## You may edit

- `feat.py`, `mdl.py`, `solution.py`
- `models/` — optional deep models (DeepLOB, TLOB, …); see `REFERENCES.md`

## Data

Physical files: `/data/moew/data/`. Worktree `data/` is symlinked at setup.

## Eval

```bash
python run_benchmark.py
uv run coral eval -m "message"
```

## Do not change

`run_benchmark.py`, train/test dates in `data_io.py`.
