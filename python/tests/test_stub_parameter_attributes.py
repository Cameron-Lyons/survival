"""PyO3 conversion hooks retain constructor parameter names and types in stubs."""

import runpy
from pathlib import Path

import pytest

GENERATOR = Path(__file__).resolve().parents[2] / "scripts/generate_stubs.py"


@pytest.mark.parametrize(
    "attribute",
    [
        "",
        "#[pyo3(from_py_with = covariate_values)]",
        "#[pyo3(from_py_with = Bound::<'_, PyAny>::extract)]",
    ],
)
def test_constructor_parameter_conversion_attributes_keep_types(attribute):
    parse_fn = runpy.run_path(str(GENERATOR))["_parse_fn"]
    method, _ = parse_fn(
        f"pub fn new({attribute} values: Vec<f64>, n_obs: usize) -> PyResult<Self> {{}}", 0
    )
    assert method is not None
    assert method.params == [("values", "Vec<f64>"), ("n_obs", "usize")]
