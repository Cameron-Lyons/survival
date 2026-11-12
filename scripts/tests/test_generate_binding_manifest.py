import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "generate_binding_manifest.py"
SPEC = importlib.util.spec_from_file_location("generate_binding_manifest", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT}")
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


class RustExportTests(unittest.TestCase):
    def test_renamed_wrappers_keep_export_names_with_explicit_lifetimes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/api/python"
            source.mkdir(parents=True)
            (source / "bindings.rs").write_text(
                """#[pyfunction(name = "curves")]
#[pyo3(signature = (x))]
pub(crate) fn curves_py<'py>(py: Python<'py>, x: FloatVec) {}
#[pyfunction(name = "fit")]
pub(crate) fn fit_py(x: FloatVec) {}
fn register() {
    wrap_pyfunction!(curves_py, m);
    wrap_pyfunction!(fit_py, m);
    wrap_pyfunction!(plain_function, m);
}
"""
            )
            self.assertEqual(  # noqa: PT009
                manifest.extract_rust_registrations(root), ("curves", "fit", "plain_function")
            )
