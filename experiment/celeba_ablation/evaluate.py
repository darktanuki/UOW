#!/usr/bin/env python3
"""CelebA ablation entry point for frozen-feature evaluation."""

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from run_eval import main


if __name__ == "__main__":
    main()
