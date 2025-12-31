import glob
import os
import sys
from typing import Any


def setup_survival_import() -> Any:
    try:
        import survival

        return survival
    except ImportError:
        pass

    test_dir: str = os.path.dirname(__file__)
    wheels_path: str = os.path.join(test_dir, "..", "target", "wheels")
    if os.path.exists(wheels_path):
        wheel_files: list[str] = glob.glob(os.path.join(wheels_path, "survival-*.whl"))
        if wheel_files:
            wheel_file: str = max(wheel_files, key=os.path.getmtime)
            try:
                import zipfile

                extract_dir: str = os.path.join(test_dir, "..", ".test_wheel_extract")
                if not os.path.exists(extract_dir):
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(wheel_file, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)

                if extract_dir not in sys.path:
                    sys.path.insert(0, extract_dir)

                try:
                    import survival

                    return survival
                except ImportError:
                    raise
            except Exception as e:
                if isinstance(e, ImportError):
                    raise
                pass

    try:
        import ctypes

        debug_path: str = os.path.join(test_dir, "..", "target", "debug")
        so_file: str = os.path.join(debug_path, "libsurvival.so")
        if os.path.exists(so_file):
            ctypes.CDLL(so_file)
            import importlib.util

            spec = importlib.util.spec_from_file_location("survival", so_file)
            if spec and spec.loader:
                module: Any = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    except Exception:
        pass

    raise ImportError(
        "Could not import survival module. "
        "Please build the project first:\n"
        "  maturin build\n"
        "Then install the wheel:\n"
        "  pip install target/wheels/survival-*.whl\n"
        "or for development:\n"
        "  maturin develop"
    )
