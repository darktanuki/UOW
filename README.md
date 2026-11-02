# Unbiased Open World (UOW)

Official implementation of [Unbiased Open World Regularization for Fair
Self-Supervised Learning](https://arxiv.org/abs/2607.22149) by Léo Nicollier,
Marc Pic, Pablo Musé, Enric Meinhardt-Llopis, and Gabriele Facciolo.

This repository contains training and evaluation code and experiment scripts for
Colored MNIST, CelebA, Waterbirds, FairFace, and synthetic engraving.

## Installation

The project requires Python 3.10 or newer and uses [uv](https://docs.astral.sh/uv/)
to manage dependencies. From the repository root:

```bash
uv sync
```

The configured PyTorch package index targets CUDA 12.6; choose suitable PyTorch
and torchvision builds for your platform if necessary. Dataset downloads and
pretrained model weights may require network access and substantial storage.

## Experiments

Run commands from the repository root. The experiment READMEs document the
settings, datasets, output locations, and commands for individual runs and
complete suites. The `--dry-run` option prints commands without training.

| Experiment | Instructions |
| --- | --- |
| Colored MNIST | [experiment/colormnist](experiment/colormnist/README.md) |
| Waterbirds, FairFace and CelebA | [experiment/realworld](experiment/realworld/README.md) |
| CelebA lambda and alpha ablation | [experiment/celeba_ablation](experiment/celeba_ablation/README.md) |
| CelebA training dynamics | [experiment/celeba_dynamics](experiment/celeba_dynamics/README.md) |
| Intersectional CelebA | [experiment/celeba_intersectional](experiment/celeba_intersectional/README.md) |
| Synthetic engraving | [experiment/synthetic_engraving](experiment/synthetic_engraving/README.md) |

For example, inspect the Colored MNIST suite, then run or resume it on GPU 0:

```bash
uv run python experiment/colormnist/run_all.py --device 0 --dry-run
uv run python experiment/colormnist/run_all.py --device 0 --skip-existing
```

Check each experiment's dataset requirements and available options before
launching a full suite; these runs can be computationally expensive. The main
entry points are `run_pretraining.py` and `run_eval.py`; synthetic engraving
uses its own trainer in `experiment/synthetic_engraving/`.

## Project structure

- `uow.py` and `intersectional_uow.py`: UOW losses.
- `src/`: regularizers, kernels, models, and baseline losses.
- `dataloaders/`: dataset loading and preprocessing.
- `experiment/`: experiment configurations, runners, and table or figure scripts.
- `lightning_module.py`: training and evaluation modules.

## Paper and citation

If you use this code, please cite the paper:

```bibtex
@article{nicollier2026unbiased,
  title         = {Unbiased Open World Regularization for Fair Self-Supervised Learning},
  author        = {Nicollier, L{\'e}o and Pic, Marc and Mus{\'e}, Pablo and Meinhardt-Llopis, Enric and Facciolo, Gabriele},
  journal       = {arXiv preprint arXiv:2607.22149},
  year          = {2026},
  doi           = {10.48550/arXiv.2607.22149},
  url           = {https://arxiv.org/abs/2607.22149}
}
```
