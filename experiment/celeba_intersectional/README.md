# Intersectional CelebA

Target: Attractive. Observed biases: Male and Young. The suite compares OW
against UOW conditioning on Male only, Male and Young separately, or their
four-valued joint variable. Every method runs on seeds 0, 1, and 2. All runs
use `lambda_reg=0.05`; the three UOW variants use `alpha=0.5` (OW has no
conditional/global mixing parameter).
Training uses 8 loader workers and frozen evaluation uses 0. With
`--skip-existing`, the runner reuses saved metrics or a completed checkpoint;
it resumes a partial checkpoint from its last epoch.

```bash
# Inspect all 12 train/eval command pairs.
uv run python experiment/celeba_intersectional/run_all.py --device 0 --dry-run

# Test one configuration.
uv run python experiment/celeba_intersectional/run_all.py --device 0 --methods uow_joint --seeds 0

# Run or resume everything.
uv run python experiment/celeba_intersectional/run_all.py --device 0 --skip-existing

# Rebuild table.tex.
uv run python experiment/celeba_intersectional/run_all.py --only-table
```

All outputs are stored here and under `artifacts/`.
