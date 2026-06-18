import importlib
import sys
from pathlib import Path
from types import ModuleType


def _path_matches(path: str, root: Path) -> bool:
    try:
        return Path(path or ".").resolve() == root
    except (OSError, RuntimeError):
        return False


def _survival_loaded_from(root: Path) -> bool:
    module = sys.modules.get("survival")
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False

    try:
        Path(module_file).resolve().relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _traceback_loaded_from(exc: ImportError, root: Path) -> bool:
    traceback = exc.__traceback__
    while traceback is not None:
        try:
            Path(traceback.tb_frame.f_code.co_filename).resolve().relative_to(root)
        except (OSError, RuntimeError, ValueError):
            pass
        else:
            return True
        traceback = traceback.tb_next
    return False


def _source_tree_binding_error(exc: ImportError, root: Path) -> bool:
    message = str(exc)
    binding_error = (
        "cannot import name '_survival'" in message
        or "No module named 'survival._survival'" in message
        or "Rust extension is missing declared binding symbol(s)" in message
    )
    return binding_error and (_survival_loaded_from(root) or _traceback_loaded_from(exc, root))


def _remove_source_tree_import(root: Path) -> None:
    for name in tuple(sys.modules):
        if name == "survival" or name.startswith("survival."):
            sys.modules.pop(name, None)
    sys.path[:] = [path for path in sys.path if not _path_matches(path, root)]


def setup_survival_import() -> ModuleType:
    python_root = Path(__file__).resolve().parents[1]

    try:
        return importlib.import_module("survival")
    except ModuleNotFoundError as exc:
        if exc.name != "survival":
            raise ImportError(
                "Could not import `survival`. Build the extension first, for example:\n"
                "  maturin develop --release\n"
                "Or install the built wheel, for example:\n"
                "  pip install target/wheels/survival-*.whl\n"
                "Then run:\n"
                "  pytest python/tests -v"
            ) from exc
        _remove_source_tree_import(python_root)
    except ImportError as exc:
        if not _source_tree_binding_error(exc, python_root):
            raise ImportError(
                "Could not import `survival`. Build the extension first, for example:\n"
                "  maturin develop --release\n"
                "Or install the built wheel, for example:\n"
                "  pip install target/wheels/survival-*.whl\n"
                "Then run:\n"
                "  pytest python/tests -v"
            ) from exc
        _remove_source_tree_import(python_root)

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
