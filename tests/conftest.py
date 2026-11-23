"""Run unit tests on CPU with isolated copies of the example recipes."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_CONFIG_WORKSPACE: tempfile.TemporaryDirectory[str] | None = None


def pytest_configure() -> None:
    """Materialize safe example recipes before test modules are imported."""
    global _CONFIG_WORKSPACE
    _CONFIG_WORKSPACE = tempfile.TemporaryDirectory(prefix="fastwam-pytest-configs-")
    workspace = Path(_CONFIG_WORKSPACE.name)
    configs = workspace / "configs"
    shutil.copytree(ROOT / "configs", configs)
    shutil.copytree(ROOT / "src" / "fastwam_jax" / "specs", workspace / "src/fastwam_jax/specs")
    task_source = ROOT / "eval/scripts/fastwam_eval/tasks/t10.txt"
    task_target = workspace / "eval/scripts/fastwam_eval/tasks/t10.txt"
    task_target.parent.mkdir(parents=True)
    shutil.copy2(task_source, task_target)

    for recipe in configs.rglob("*.example.yaml"):
        recipe.write_text(
            recipe.read_text(encoding="utf-8").replace("your-fastwam-bucket", "example-bucket"),
            encoding="utf-8",
        )

    from fastwam_jax import config as config_module

    config_module.default_configs_dir = lambda start=None: configs


def pytest_unconfigure() -> None:
    global _CONFIG_WORKSPACE
    if _CONFIG_WORKSPACE is not None:
        _CONFIG_WORKSPACE.cleanup()
        _CONFIG_WORKSPACE = None
