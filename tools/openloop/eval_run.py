"""Run Gate 1 for one finalized checkpoint selected by GCS run URI and step.

This file is the public operator entry point. It adds the repository's ``src`` directory
to the import path so callers do not need ``PYTHONPATH`` or an installed editable package.
"""

# ruff: noqa: E402, I001 -- the public file must add src before importing the package.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastwam_jax.openloop.cli import entrypoint


if __name__ == "__main__":
    entrypoint()
