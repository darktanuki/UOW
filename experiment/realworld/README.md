# Real-world experiments

This suite runs 81 experiments: 3 tasks × 9 methods × seeds 0, 1, and 2.
Every training run is immediately followed by frozen-feature evaluation.
The launcher prints the exact command with completed/remaining counters for
each run, then prints `OK` only after evaluation is stored in the CSV.

All shared hyperparameters, tasks, and methods are listed in `config.py`.
Training uses 8 data-loader workers; frozen evaluation uses 0 to avoid Linux
file-descriptor exhaustion during feature extraction.
Gaussian OW/UOW use `sigreginduced`; spherical OW/UOW use the standard `mmd`.
OW, UOW, LeJEPA, and EnD share `lambda_reg = 0.05`; losses such as CE and
FSCL do not consume this coefficient.

From the repository root:

```bash
uv run python experiment/realworld/run_all.py --device 0
```

To resume without repeating finished work (completed rows, saved evaluation
metrics, or completed training checkpoints):

```bash
uv run python experiment/realworld/run_all.py --device 0 --skip-existing
```

Useful checks:

```bash
# Print all 81 train/eval command pairs without executing them.
uv run python experiment/realworld/run_all.py --dry-run

# Run one small slice of the matrix.
uv run python experiment/realworld/run_all.py --tasks waterbirds --methods ce --seeds 0

# Rebuild the LaTeX table from the current CSV.
uv run python experiment/realworld/make_table.py
```

Outputs:

- `results.csv`: one LP result per task, method, and seed (metrics in percent)
- `table.tex`: mean ± sample standard deviation over available seeds
- `artifacts/checkpoints/`: checkpoints
- `artifacts/logs/`: Lightning training logs
- `artifacts/evaluations/`: detailed LP and k-NN CSV files
- `artifacts/features/`: cached frozen features
- `artifacts/benchmark.csv`: detailed evaluation history

If a run fails, the process stops and no summary row is added for that run.
Fix the issue and restart with `--skip-existing`.
Rerunning without `--skip-existing` replaces the matching CSV row instead of
creating a duplicate.
