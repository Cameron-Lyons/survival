import importlib
import sys
from pathlib import Path
from types import ModuleType


def setup_survival_import() -> ModuleType:
    python_root = Path(__file__).resolve().parents[1]

    try:
        return importlib.import_module("survival")
    except ImportError:
        sys.modules.pop("survival", None)
        sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != python_root]

    try:
        return importlib.import_module("survival")
    except ImportError as exc:
        raise ImportError(
            "Could not import `survival`. Build the extension first, for example:\n"
            "  maturin develop --release\n"
            "Or install the built wheel, for example:\n"
            "  pip install target/wheels/survival-*.whl\n"
            "Then run:\n"
            "  pytest python/tests -v"
        ) from exc
