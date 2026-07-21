import ast
import dataclasses
import importlib
import importlib.util
import inspect
import math
import re
import runpy
import sys
import tomllib
from pathlib import Path

import pytest

from .helpers import setup_survival_import

ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_SCRIPT = ROOT / "scripts/generate_binding_manifest.py"
_MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "generate_binding_manifest",
    _MANIFEST_SCRIPT,
)
if _MANIFEST_SPEC is None or _MANIFEST_SPEC.loader is None:
    raise RuntimeError(f"Could not load {_MANIFEST_SCRIPT}")
_MANIFEST_MODULE = importlib.util.module_from_spec(_MANIFEST_SPEC)
_MANIFEST_SPEC.loader.exec_module(_MANIFEST_MODULE)

extract_feature_registrations = _MANIFEST_MODULE.extract_feature_registrations
extract_python_module_bindings = _MANIFEST_MODULE.extract_python_module_bindings
extract_rust_registrations = _MANIFEST_MODULE.extract_rust_registrations

PACKAGE_ROOT = ROOT / "python/survival"
README = ROOT / "README.md"
MODULE_METADATA_NAMES = {
    "__all__",
    "__deprecated_root_export_reason__",
    "__deprecated_root_exports__",
    "__legacy_root_exports__",
    "__preferred__",
    "__version__",
}


def _manifest_bindings() -> tuple[str, ...]:
    manifest = runpy.run_path(str(PACKAGE_ROOT / "_binding_manifest.py"))
    return tuple(manifest["BINDINGS"])


def _feature_bindings() -> dict[str, tuple[str, ...]]:
    manifest = runpy.run_path(str(PACKAGE_ROOT / "_binding_manifest.py"))
    return manifest["FEATURE_BINDINGS"]


def _module_bindings() -> dict[str, tuple[str, ...]]:
    manifest = runpy.run_path(str(PACKAGE_ROOT / "_binding_manifest.py"))
    return manifest["MODULE_BINDINGS"]


def _has_ml_bindings(core) -> bool:
    sentinels = ("DeepSurv", "survival_forest", "GradientBoostSurvival")
    return any(hasattr(core, name) for name in sentinels)


def _literal_string_list(node: ast.AST) -> list[str]:
    if not isinstance(node, ast.List | ast.Tuple):
        return []
    return [
        element.value
        for element in node.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]


def _binding_names_by_module() -> dict[str, set[str]]:
    modules: dict[str, set[str]] = {}
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        if path.name.startswith("_") or path.name in {"__init__.py", "sklearn_compat.py"}:
            continue

        tree = ast.parse(path.read_text(), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "bind_names"
                and len(node.args) >= 2
            ):
                names.update(_literal_string_list(node.args[1]))

        if names:
            modules[path.stem] = names
    return modules


def _module_export_names_by_module() -> dict[str, tuple[str, ...]]:
    modules: dict[str, tuple[str, ...]] = {}
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        if path.name.startswith("_") or path.name in {"__init__.py", "sklearn_compat.py"}:
            continue

        tree = ast.parse(path.read_text(), filename=str(path))
        names: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "bind_names"
                and len(node.args) >= 2
            ):
                for name in _literal_string_list(node.args[1]):
                    if name not in names:
                        names.append(name)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "__all__"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value not in names
            ):
                names.append(node.args[0].value)

        if names:
            modules[path.stem] = tuple(names)
    return modules


def _top_level_names() -> set[str]:
    tree = ast.parse((PACKAGE_ROOT / "__init__.py").read_text())
    public_modules: set[str] = set()
    sklearn_exports: set[str] = set()
    r_exports: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "_PUBLIC_MODULES"
                and isinstance(node.value, ast.Dict)
            ):
                public_modules.update(
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
            if isinstance(target, ast.Name) and target.id == "_SKLEARN_EXPORTS":
                sklearn_exports.update(_literal_string_list(node.value))
            if isinstance(target, ast.Name) and target.id == "_R_EXPORTS":
                r_exports.update(_literal_string_list(node.value))

    domain_exports = set().union(*_module_export_names_by_module().values())
    return public_modules | r_exports | sklearn_exports | domain_exports | MODULE_METADATA_NAMES


def _sklearn_names() -> set[str]:
    tree = ast.parse((PACKAGE_ROOT / "sklearn_compat.py").read_text())
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and not node.name.startswith("_")
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_PUBLIC_EXPORTS":
                names.update(_literal_string_list(node.value))
            if isinstance(target, ast.Name) and target.id == "__all__":
                names.update(_literal_string_list(node.value))
    return names


def _remove_survival_modules() -> None:
    for name in tuple(sys.modules):
        if name == "survival" or name.startswith("survival."):
            sys.modules.pop(name, None)


def _pyi_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _pyi_function_arg_names(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = [arg.arg for arg in node.args.posonlyargs]
            args.extend(arg.arg for arg in node.args.args)
            args.extend(arg.arg for arg in node.args.kwonlyargs)
            return args
    raise AssertionError(f"{name} not found in {path}")


def _pyi_function_kwarg_name(path: Path, name: str) -> str | None:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node.args.kwarg.arg if node.args.kwarg is not None else None
    raise AssertionError(f"{name} not found in {path}")


def _pyi_function_return(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            if node.returns is None:
                return ""
            return ast.unparse(node.returns)
    raise AssertionError(f"{name} not found in {path}")


def _pyi_class_node(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _pyi_class_method_arg_names(path: Path, class_name: str, method_name: str) -> list[str]:
    class_node = _pyi_class_node(path, class_name)
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            args = [arg.arg for arg in node.args.posonlyargs]
            args.extend(arg.arg for arg in node.args.args)
            args.extend(arg.arg for arg in node.args.kwonlyargs)
            return args
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _pyi_class_method_return(path: Path, class_name: str, method_name: str) -> str:
    class_node = _pyi_class_node(path, class_name)
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            if node.returns is None:
                return ""
            return ast.unparse(node.returns)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _pyi_class_property_names(path: Path, class_name: str) -> set[str]:
    class_node = _pyi_class_node(path, class_name)
    return {
        node.name
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Name) and decorator.id == "property"
            for decorator in node.decorator_list
        )
    }


def _pyi_class_annotation_names(path: Path, class_name: str) -> set[str]:
    class_node = _pyi_class_node(path, class_name)
    return {
        node.target.id
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _readme_python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)


def _format_missing(missing: list[tuple[str, str]]) -> str:
    return "\n".join(f"{source}: {name}" for source, name in missing)


def test_generated_manifest_matches_rust_registration():
    assert _manifest_bindings() == extract_rust_registrations(ROOT)


def test_generated_feature_manifest_matches_rust_registration():
    assert _feature_bindings() == extract_feature_registrations(ROOT)


def test_generated_module_manifest_matches_python_modules():
    assert _module_bindings() == extract_python_module_bindings(ROOT)


def test_python_modules_only_declare_registered_bindings():
    manifest = set(_manifest_bindings())
    missing = [
        (f"python/survival/{module}.py", name)
        for module, names in _binding_names_by_module().items()
        for name in sorted(names - manifest)
    ]

    assert missing == [], _format_missing(missing)


def test_survival_stub_only_declares_registered_bindings():
    manifest = set(_manifest_bindings())
    stub_names = _pyi_top_level_names(PACKAGE_ROOT / "_survival.pyi")
    missing = [("python/survival/_survival.pyi", name) for name in sorted(stub_names - manifest)]

    assert missing == [], _format_missing(missing)


def test_survreg_residual_low_level_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "SurvivalData",
        "Weights",
        "CountingProcessData",
        "CoxMartInput",
        "AndersenGillInput",
        "SurvregResiduals",
        "coxmart",
        "agmart",
        "residuals_survreg",
        "survreg_residual_matrix",
        "survreg_influence_residuals",
        "survreg_dfbeta_residuals",
        "dfbeta_survreg",
    } <= stub_names
    assert "time2" in inspect.signature(core.residuals_survreg).parameters
    assert _pyi_function_arg_names(stub_path, "residuals_survreg") == [
        "time",
        "status",
        "linear_pred",
        "scale",
        "distribution",
        "residual_type",
        "time2",
        "distribution_parameter",
    ]
    assert _pyi_function_arg_names(stub_path, "survreg_residual_matrix") == [
        "time",
        "status",
        "linear_pred",
        "scale",
        "distribution",
        "time2",
        "distribution_parameter",
    ]
    assert _pyi_function_arg_names(stub_path, "survreg_influence_residuals") == [
        "derivative_matrix",
        "covariates",
        "scales",
        "strata",
        "var_matrix",
        "residual_type",
        "rsigma",
    ]
    assert _pyi_function_arg_names(stub_path, "survreg_dfbeta_residuals") == [
        "derivative_matrix",
        "covariates",
        "scales",
        "strata",
        "var_matrix",
        "rsigma",
        "standardized",
    ]
    assert _pyi_function_arg_names(stub_path, "dfbeta_survreg") == [
        "time",
        "status",
        "covariates",
        "linear_pred",
        "scale",
        "var_matrix",
        "distribution",
        "time2",
        "distribution_parameter",
    ]
    assert list(inspect.signature(core.coxmart).parameters) == ["input", "method"]
    assert _pyi_function_arg_names(stub_path, "coxmart") == ["input", "method"]
    assert list(inspect.signature(core.agmart).parameters) == ["input", "method"]
    assert _pyi_function_arg_names(stub_path, "agmart") == ["input", "method"]

    class_init_args = {
        "SurvivalData": ["self", "time", "status"],
        "Weights": ["self", "values"],
        "CountingProcessData": ["self", "start", "stop", "event"],
        "CoxMartInput": ["self", "survival", "score", "weights", "strata"],
        "AndersenGillInput": ["self", "counting", "score", "weights", "strata"],
    }
    for class_name, expected_args in class_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert list(inspect.signature(core.Weights.unit).parameters) == ["n_obs"]
    assert _pyi_class_method_arg_names(stub_path, "Weights", "unit") == ["n_obs"]
    assert _pyi_class_annotation_names(stub_path, "SurvivalData") == {"time", "status"}
    assert _pyi_class_annotation_names(stub_path, "Weights") == {"values"}
    assert _pyi_class_annotation_names(stub_path, "CountingProcessData") == {
        "start",
        "stop",
        "event",
    }
    assert _pyi_class_property_names(stub_path, "CoxMartInput") == {"n_obs"}
    assert _pyi_class_property_names(stub_path, "AndersenGillInput") == {"n_obs"}
    assert _pyi_class_property_names(stub_path, "SurvregResiduals") == {
        "residuals",
        "residual_type",
        "n",
    }

    survival_data = core.SurvivalData([1.0, 2.0, 3.0], [1, 0, 1])
    weights = core.Weights([1.0, 2.0, 1.0])
    cox_input = core.CoxMartInput(survival_data, [0.2, 0.4, 0.8], weights, [0, 0, 1])
    counting = core.CountingProcessData([0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [1, 0, 1])
    ag_input = core.AndersenGillInput(counting, [1.0, 1.0, 1.0], core.Weights.unit(3), [1, 0, 0])
    survreg_residuals = core.residuals_survreg(
        [1.0, 2.0, 3.0],
        [1, 0, 1],
        [0.0, 0.5, 1.0],
        1.0,
        "weibull",
        "working",
        None,
    )
    survreg_matrix = core.survreg_residual_matrix(
        [1.0, 2.0, 3.0],
        [1, 0, 1],
        [0.0, 0.5, 1.0],
        1.0,
        "weibull",
        None,
    )
    survreg_influence = core.survreg_influence_residuals(
        [[0.0, 2.0, 3.0, 5.0, 7.0, 11.0]],
        [[1.0, 4.0]],
        [1.5],
        [0],
        [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]],
        "ldcase",
        True,
    )
    survreg_dfbeta = core.survreg_dfbeta_residuals(
        [[0.0, 2.0, 3.0, 5.0, 7.0, 11.0]],
        [[1.0, 4.0]],
        [1.5],
        [0],
        [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]],
        True,
        False,
    )

    assert survival_data.time == pytest.approx([1.0, 2.0, 3.0])
    assert survival_data.status == [1, 0, 1]
    assert survival_data.is_empty() is False
    assert weights.values == pytest.approx([1.0, 2.0, 1.0])
    assert core.Weights.unit(3).values == pytest.approx([1.0, 1.0, 1.0])
    assert counting.start == pytest.approx([0.0, 0.0, 1.0])
    assert counting.stop == pytest.approx([1.0, 2.0, 3.0])
    assert counting.event == [1, 0, 1]
    assert cox_input.n_obs == 3
    assert ag_input.n_obs == 3
    assert core.coxmart(cox_input, 0) == pytest.approx(
        [0.8888888888888888, -0.22222222222222224, -0.44444444444444464]
    )
    assert core.agmart(ag_input, 0) == pytest.approx([0.0, 0.0, 0.0])
    assert type(survreg_residuals).__name__ == "SurvregResiduals"
    assert survreg_residuals.residual_type == "working"
    assert survreg_residuals.n == 3
    assert len(survreg_matrix) == 3
    assert all(len(row) == 6 for row in survreg_matrix)
    assert survreg_influence == pytest.approx([238.2])
    assert len(survreg_dfbeta) == 1
    assert survreg_dfbeta[0] == pytest.approx([3.8, 17.7, 17.8])


def test_survreg_prediction_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "DistributionType",
        "SurvivalFit",
        "SurvregConfig",
        "SurvregPrediction",
        "SurvregQuantilePrediction",
        "survreg",
        "survreg_distribution",
        "predict_survreg",
        "predict_survreg_quantile",
        "survreg_quantile_prediction_se_matrix",
    } <= stub_names

    expected_args = {
        "survreg": [
            "time",
            "status",
            "covariates",
            "weights",
            "offsets",
            "initial_beta",
            "strata",
            "distribution",
            "max_iter",
            "eps",
            "tol_chol",
            "time2",
            "fixed_scale",
            "distribution_parameter",
        ],
        "predict_survreg": [
            "covariates",
            "coefficients",
            "scale",
            "distribution",
            "predict_type",
            "offset",
            "var_matrix",
            "se_fit",
        ],
        "predict_survreg_quantile": [
            "covariates",
            "coefficients",
            "scale",
            "distribution",
            "quantiles",
            "offset",
        ],
        "survreg_distribution": [
            "values",
            "mean",
            "scale",
            "distribution",
            "kind",
        ],
        "survreg_quantile_prediction_se_matrix": [
            "rows",
            "scales",
            "strata",
            "variance",
            "quantile_scores",
            "predictions",
            "transform_se",
        ],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    expected_method_args = {
        "predict": ["self", "covariates", "predict_type", "offset", "se_fit"],
        "predict_quantile": ["self", "covariates", "quantiles", "offset"],
        "residuals": ["self", "residual_type"],
        "dfbeta": ["self"],
    }
    for method_name, args in expected_method_args.items():
        assert list(inspect.signature(getattr(core.SurvivalFit, method_name)).parameters) == args
        assert _pyi_class_method_arg_names(stub_path, "SurvivalFit", method_name) == args

    assert _pyi_class_method_arg_names(stub_path, "SurvregConfig", "__init__") == [
        "self",
        "distribution",
        "max_iter",
        "eps",
        "tol_chol",
    ]
    assert list(inspect.signature(core.SurvregConfig).parameters) == [
        "distribution",
        "max_iter",
        "eps",
        "tol_chol",
    ]
    assert _pyi_class_annotation_names(stub_path, "DistributionType") == {
        "extreme_value",
        "logistic",
        "gaussian",
        "weibull",
        "lognormal",
        "loglogistic",
    }
    assert _pyi_class_annotation_names(stub_path, "SurvregConfig") == {
        "max_iter",
        "eps",
        "tol_chol",
        "distribution",
    }
    assert _pyi_class_annotation_names(stub_path, "SurvivalFit") == {
        "coefficients",
        "location_coefficients",
        "scale",
        "scales",
        "distribution",
        "distribution_parameters",
        "n_covariates",
        "n_strata",
        "linear_predictors",
        "time",
        "time2",
        "status",
        "covariates",
        "strata",
        "weights",
        "iterations",
        "variance_matrix",
        "log_likelihood",
        "convergence_flag",
        "score_vector",
    }
    assert _pyi_class_property_names(stub_path, "SurvregPrediction") == {
        "predictions",
        "se",
        "prediction_type",
        "n",
    }
    assert _pyi_class_property_names(stub_path, "SurvregQuantilePrediction") == {
        "quantiles",
        "predictions",
        "n",
    }

    assert hasattr(core.DistributionType, "weibull")
    assert not hasattr(core.DistributionType, "Weibull")
    config = core.SurvregConfig(core.DistributionType.weibull, 7, 1e-5, 1e-8)
    assert config.distribution == core.DistributionType.weibull
    assert config.max_iter == 7
    assert config.eps == pytest.approx(1e-5)
    assert config.tol_chol == pytest.approx(1e-8)

    prediction = core.predict_survreg(
        [[1.0, 2.0], [2.0, 3.0]],
        [0.1, 0.2],
        1.0,
        "weibull",
        "lp",
        None,
        [[1.0, 0.0], [0.0, 1.0]],
        True,
    )
    quantiles = core.predict_survreg_quantile(
        [[1.0, 2.0], [2.0, 3.0]],
        [0.1, 0.2],
        1.0,
        "weibull",
        [0.25, 0.5],
        None,
    )

    assert type(prediction).__name__ == "SurvregPrediction"
    assert prediction.predictions == pytest.approx([0.5, 0.8])
    assert prediction.se == pytest.approx([5**0.5, 13**0.5])
    assert prediction.prediction_type == "lp"
    assert prediction.n == 2
    assert type(quantiles).__name__ == "SurvregQuantilePrediction"
    assert quantiles.quantiles == pytest.approx([0.25, 0.5])
    assert len(quantiles.predictions) == 2
    assert all(len(row) == 2 for row in quantiles.predictions)
    quantile_se = core.survreg_quantile_prediction_se_matrix(
        [[1.0, 2.0], [3.0, 4.0]],
        [2.0, 3.0],
        [0, 1],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [0.5, 1.0],
        [[10.0, 20.0], [30.0, 40.0]],
        False,
    )
    assert quantile_se[0] == pytest.approx([math.sqrt(6.0), 3.0])
    assert quantile_se[1] == pytest.approx([math.sqrt(27.25), math.sqrt(34.0)])

    covariates = [
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
        [2.5, 3.5],
        [3.0, 4.0],
        [3.5, 4.5],
        [4.0, 5.0],
        [4.5, 5.5],
    ]
    fit = core.survreg(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        covariates,
        None,
        None,
        None,
        None,
        "extreme_value",
        20,
        1e-5,
        1e-9,
        None,
        None,
    )
    fit_prediction = fit.predict(covariates[:2], "lp")
    fit_quantiles = fit.predict_quantile(covariates[:2], [0.25, 0.5])

    assert type(fit).__name__ == "SurvivalFit"
    assert fit.n_covariates == 2
    assert fit.distribution == "extreme_value"
    assert fit.weights == pytest.approx([1.0] * 8)
    assert type(fit_prediction).__name__ == "SurvregPrediction"
    assert fit_prediction.n == 2
    assert type(fit_quantiles).__name__ == "SurvregQuantilePrediction"
    assert fit_quantiles.n == 2

    rayleigh_covariates = [
        [1.0, 0.2],
        [1.0, 0.4],
        [1.0, 0.1],
        [1.0, 0.8],
        [1.0, 1.0],
        [1.0, 1.2],
        [1.0, 0.6],
        [1.0, 1.4],
    ]
    rayleigh_fit = core.survreg(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        rayleigh_covariates,
        None,
        None,
        None,
        None,
        "rayleigh",
        200,
        1e-8,
        1e-9,
        None,
        None,
    )
    weibull_half_fit = core.survreg(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        rayleigh_covariates,
        None,
        None,
        None,
        None,
        "weibull",
        200,
        1e-8,
        1e-9,
        None,
        0.5,
    )

    assert rayleigh_fit.distribution == "rayleigh"
    assert rayleigh_fit.scale == pytest.approx(0.5)
    assert rayleigh_fit.coefficients == pytest.approx(weibull_half_fit.coefficients)
    assert rayleigh_fit.predict(rayleigh_covariates[:2], "response").predictions == pytest.approx(
        weibull_half_fit.predict(rayleigh_covariates[:2], "response").predictions
    )

    loggaussian_fit = core.survreg(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        rayleigh_covariates,
        None,
        None,
        None,
        None,
        "loggaussian",
        20,
        1e-5,
        1e-9,
        None,
        None,
    )

    assert loggaussian_fit.distribution == "lognormal"


def test_coxph_fit_detail_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "CoxPHFit",
        "CoxphDetail",
        "CoxphDetailRow",
        "coxph_fit",
        "coxph_detail",
    } <= stub_names

    expected_args = {
        "coxph_fit": [
            "time",
            "status",
            "covariates",
            "strata",
            "weights",
            "offset",
            "initial_beta",
            "max_iter",
            "eps",
            "toler",
            "method",
            "entry_times",
            "nocenter",
        ],
        "coxph_detail": [
            "time",
            "status",
            "covariates",
            "coefficients",
            "weights",
            "entry_times",
            "strata",
            "offset",
            "method",
            "center",
        ],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    coxph_fit_methods = {
        "predict": ["self", "covariates"],
        "hazard_ratios": ["self"],
        "basehaz": ["self", "centered"],
        "basehaz_with_strata": ["self", "centered"],
        "survival_curve": ["self", "covariates", "centered"],
        "survival_curve_with_strata": ["self", "covariates", "strata", "centered"],
        "expected_events": ["self"],
        "martingale_residuals": ["self"],
        "deviance_residuals": ["self"],
        "score_residuals": ["self"],
        "dfbeta": ["self"],
        "dfbetas": ["self"],
        "schoenfeld_residuals": ["self"],
        "scaled_schoenfeld_residuals": ["self"],
        "partial_residuals": ["self"],
    }
    for method_name, args in coxph_fit_methods.items():
        runtime_args = list(inspect.signature(getattr(core.CoxPHFit, method_name)).parameters)
        assert runtime_args == args
        assert _pyi_class_method_arg_names(stub_path, "CoxPHFit", method_name) == args

    detail_methods = {
        "times",
        "hazards",
        "cumulative_hazards",
        "n_risk_at_times",
        "scores",
        "means",
        "information_matrices",
        "variance_hazards",
        "weighted_risk",
        "schoenfeld_residuals",
    }
    for method_name in detail_methods:
        assert list(inspect.signature(getattr(core.CoxphDetail, method_name)).parameters) == [
            "self"
        ]
        assert _pyi_class_method_arg_names(stub_path, "CoxphDetail", method_name) == ["self"]

    assert _pyi_class_property_names(stub_path, "CoxPHFit") == {
        "coefficients",
        "means",
        "score_vector",
        "information_matrix",
        "log_likelihood",
        "score_test",
        "convergence_flag",
        "iterations",
        "risk_scores",
        "event_times",
        "status",
        "linear_predictors",
        "entry_times",
        "weights",
        "covariates",
        "strata",
        "method",
        "nocenter",
    }
    assert _pyi_class_property_names(stub_path, "CoxphDetailRow") == {
        "stratum",
        "time",
        "n_risk",
        "n_event",
        "n_censor",
        "hazard",
        "cumhaz",
        "varhaz",
        "wtrisk",
        "n_event_weight",
        "score",
        "schoenfeld",
        "means",
        "imat",
    }
    assert _pyi_class_property_names(stub_path, "CoxphDetail") == {
        "rows",
        "n_events",
        "n_observations",
        "n_covariates",
    }
    assert (
        _pyi_class_method_return(stub_path, "CoxPHFit", "basehaz")
        == "tuple[list[float], list[float]]"
    )
    assert (
        _pyi_class_method_return(stub_path, "CoxPHFit", "basehaz_with_strata")
        == "tuple[list[float], list[float], list[int]]"
    )
    assert (
        _pyi_class_method_return(stub_path, "CoxPHFit", "survival_curve")
        == "tuple[list[float], list[list[float]]]"
    )

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 0, 1, 1, 0, 1, 0, 1]
    covariates = [[0.0], [1.0], [0.5], [1.5], [0.2], [1.2], [0.8], [1.8]]
    fit = core.coxph_fit(
        time,
        status,
        covariates,
        None,
        None,
        None,
        None,
        20,
        1e-8,
        1e-9,
        "breslow",
        None,
    )

    assert type(fit).__name__ == "CoxPHFit"
    assert len(fit.coefficients[-1]) == 1
    assert fit.means == pytest.approx([0.875])
    assert len(fit.score_vector) == 1
    assert len(fit.information_matrix) == 1
    assert len(fit.log_likelihood) == 2
    assert fit.iterations > 0
    assert len(fit.risk_scores) == len(time)
    assert fit.event_times == pytest.approx(time)
    assert fit.status == status
    assert len(fit.linear_predictors) == len(time)
    assert fit.entry_times is None
    assert fit.weights == pytest.approx([1.0] * len(time))
    assert fit.covariates == covariates
    assert fit.strata == [0] * len(time)
    assert fit.method == "breslow"

    predictions = fit.predict(covariates[:2])
    hazard_ratios = fit.hazard_ratios()
    base_times, base_hazards = fit.basehaz()
    strata_times, strata_hazards, baseline_strata = fit.basehaz_with_strata()
    curve_times, survival_curves = fit.survival_curve(covariates[:2])

    assert len(predictions) == 2
    assert len(hazard_ratios) == 1
    assert len(base_times) == len(base_hazards) == 5
    assert strata_times == base_times
    assert strata_hazards == base_hazards
    assert baseline_strata == [0] * len(base_times)
    assert curve_times == base_times
    assert len(survival_curves) == 2
    assert all(len(curve) == len(base_times) for curve in survival_curves)
    assert len(fit.expected_events()) == len(time)
    assert len(fit.martingale_residuals()) == len(time)
    assert len(fit.deviance_residuals()) == len(time)
    assert len(fit.score_residuals()) == len(time)
    assert len(fit.dfbeta()) == len(time)
    assert len(fit.dfbetas()) == len(time)
    assert len(fit.schoenfeld_residuals()) == sum(status)
    assert len(fit.scaled_schoenfeld_residuals()) == sum(status)
    assert len(fit.partial_residuals()) == len(time)

    detail = core.coxph_detail(
        time,
        status,
        covariates,
        fit.coefficients[-1],
        None,
        None,
        None,
        None,
        "breslow",
        0.0,
    )

    assert type(detail).__name__ == "CoxphDetail"
    assert detail.n_events == sum(status)
    assert detail.n_observations == len(time)
    assert detail.n_covariates == 1
    assert len(detail.rows) == sum(status)
    assert detail.times() == pytest.approx([1.0, 3.0, 4.0, 6.0, 8.0])
    assert len(detail.hazards()) == sum(status)
    assert len(detail.cumulative_hazards()) == sum(status)
    assert detail.n_risk_at_times() == [8, 6, 5, 3, 1]
    assert len(detail.scores()) == sum(status)
    assert len(detail.means()) == sum(status)
    assert len(detail.information_matrices()) == sum(status)
    assert len(detail.variance_hazards()) == sum(status)
    assert len(detail.weighted_risk()) == sum(status)
    assert len(detail.schoenfeld_residuals()) == sum(status)

    first_row = detail.rows[0]
    assert type(first_row).__name__ == "CoxphDetailRow"
    assert first_row.stratum == 0
    assert first_row.time == pytest.approx(1.0)
    assert first_row.n_risk == 8
    assert first_row.n_event == 1
    assert first_row.n_censor == 0
    assert first_row.n_event_weight == pytest.approx(1.0)
    assert len(first_row.score) == 1
    assert len(first_row.schoenfeld) == 1
    assert len(first_row.means) == 1
    assert len(first_row.imat) == 1


def test_cure_model_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "BoundedCumulativeHazardConfig",
        "BoundedCumulativeHazardResult",
        "CureDistribution",
        "CureModelComparisonResult",
        "LinkFunction",
        "MixtureCureConfig",
        "MixtureCureResult",
        "NonMixtureCureConfig",
        "NonMixtureCureResult",
        "NonMixtureType",
        "PromotionTimeCureResult",
        "bounded_cumulative_hazard_model",
        "compare_cure_models",
        "mixture_cure_model",
        "non_mixture_cure_model",
        "predict_bounded_cumulative_hazard",
        "predict_non_mixture_survival",
        "promotion_time_cure_model",
    } <= stub_names

    expected_args = {
        "bounded_cumulative_hazard_model": ["time", "status", "covariates", "config"],
        "compare_cure_models": ["time", "status", "covariates", "distributions"],
        "mixture_cure_model": ["time", "status", "x_cure", "x_surv", "config"],
        "non_mixture_cure_model": ["time", "status", "covariates", "config"],
        "predict_bounded_cumulative_hazard": [
            "result",
            "time_points",
            "covariates",
            "n_subjects",
            "distribution",
        ],
        "predict_non_mixture_survival": [
            "result",
            "time_points",
            "covariates",
            "n_subjects",
            "model_type",
            "distribution",
        ],
        "promotion_time_cure_model": ["time", "status", "x", "distribution", "max_iter", "tol"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    expected_init_args = {
        "BoundedCumulativeHazardConfig": ["self", "distribution", "max_iter", "tol", "alpha"],
        "CureDistribution": ["self", "name"],
        "LinkFunction": ["self", "name"],
        "MixtureCureConfig": [
            "self",
            "distribution",
            "link",
            "max_iter",
            "tol",
            "em_max_iter",
        ],
        "NonMixtureCureConfig": [
            "self",
            "model_type",
            "distribution",
            "max_iter",
            "tol",
            "dispersion",
        ],
        "NonMixtureType": ["self", "name"],
    }
    for class_name, args in expected_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    for method_name, args in {
        "link": ["self", "p"],
        "inv_link": ["self", "eta"],
        "deriv": ["self", "eta"],
    }.items():
        assert list(inspect.signature(getattr(core.LinkFunction, method_name)).parameters) == args
        assert _pyi_class_method_arg_names(stub_path, "LinkFunction", method_name) == args

    assert _pyi_class_annotation_names(stub_path, "CureDistribution") == {
        "Weibull",
        "LogNormal",
        "LogLogistic",
        "Exponential",
        "Gamma",
    }
    assert _pyi_class_annotation_names(stub_path, "LinkFunction") == {
        "Logit",
        "Probit",
        "CLogLog",
        "Identity",
    }
    assert _pyi_class_annotation_names(stub_path, "NonMixtureType") == {
        "GeometricGeneralized",
        "NegativeBinomial",
        "Poisson",
        "Destructive",
    }
    assert _pyi_class_annotation_names(stub_path, "MixtureCureConfig") == {
        "distribution",
        "link",
        "max_iter",
        "tol",
        "em_max_iter",
    }
    assert _pyi_class_annotation_names(stub_path, "BoundedCumulativeHazardConfig") == {
        "distribution",
        "max_iter",
        "tol",
        "alpha",
    }
    assert _pyi_class_annotation_names(stub_path, "NonMixtureCureConfig") == {
        "model_type",
        "distribution",
        "max_iter",
        "tol",
        "dispersion",
    }

    assert _pyi_class_property_names(stub_path, "MixtureCureResult") == {
        "cure_coef",
        "survival_coef",
        "scale",
        "shape",
        "cure_fraction",
        "log_likelihood",
        "aic",
        "bic",
        "n_iter",
        "converged",
        "cure_prob",
    }
    assert _pyi_class_property_names(stub_path, "BoundedCumulativeHazardResult") == {
        "coef",
        "scale",
        "shape",
        "alpha",
        "cure_fraction",
        "log_likelihood",
        "aic",
        "bic",
        "n_iter",
        "converged",
        "cumulative_hazard_bound",
        "std_errors",
    }
    assert _pyi_class_property_names(stub_path, "NonMixtureCureResult") == {
        "coef",
        "theta",
        "scale",
        "shape",
        "dispersion",
        "cure_fraction",
        "log_likelihood",
        "aic",
        "bic",
        "n_iter",
        "converged",
        "std_errors",
        "survival_probs",
    }
    assert _pyi_class_property_names(stub_path, "PromotionTimeCureResult") == {
        "theta",
        "coef",
        "scale",
        "shape",
        "cure_fraction",
        "log_likelihood",
        "aic",
        "bic",
        "n_iter",
        "converged",
    }
    assert _pyi_class_property_names(stub_path, "CureModelComparisonResult") == {
        "model_names",
        "log_likelihoods",
        "aic_values",
        "bic_values",
        "cure_fractions",
        "best_model_aic",
        "best_model_bic",
    }

    assert repr(core.CureDistribution("weibull")) == "CureDistribution.Weibull"
    assert repr(core.LinkFunction("logit")) == "LinkFunction.Logit"
    assert core.LinkFunction.Logit.inv_link(0.0) == pytest.approx(0.5)
    assert repr(core.NonMixtureType("poisson")) == "NonMixtureType.Poisson"

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 0, 1, 1, 0, 1, 0, 0]
    covariates = [0.0, 1.0, 0.5, 1.5, 0.2, 1.2, 0.8, 1.8]

    mixture_config = core.MixtureCureConfig(
        core.CureDistribution.Weibull,
        core.LinkFunction.Logit,
        10,
        1e-5,
        20,
    )
    assert mixture_config.distribution == core.CureDistribution.Weibull
    assert mixture_config.link == core.LinkFunction.Logit
    assert mixture_config.max_iter == 10
    assert mixture_config.tol == pytest.approx(1e-5)
    assert mixture_config.em_max_iter == 20

    mixture = core.mixture_cure_model(time, status, covariates, [], mixture_config)
    assert type(mixture).__name__ == "MixtureCureResult"
    assert len(mixture.cure_coef) == 1
    assert len(mixture.survival_coef) == 1
    assert mixture.scale > 0.0
    assert mixture.shape > 0.0
    assert 0.0 <= mixture.cure_fraction <= 1.0
    assert mixture.n_iter > 0
    assert len(mixture.cure_prob) == len(time)

    bounded_config = core.BoundedCumulativeHazardConfig(
        core.CureDistribution.Weibull,
        20,
        1e-5,
        1.0,
    )
    bounded = core.bounded_cumulative_hazard_model(time, status, covariates, bounded_config)
    bounded_prediction = core.predict_bounded_cumulative_hazard(
        bounded,
        [1.0, 3.0],
        covariates[:2],
        2,
        core.CureDistribution.Weibull,
    )
    assert type(bounded).__name__ == "BoundedCumulativeHazardResult"
    assert len(bounded.coef) == 1
    assert bounded.scale > 0.0
    assert bounded.shape > 0.0
    assert bounded.alpha > 0.0
    assert 0.0 <= bounded.cure_fraction <= 1.0
    assert bounded.n_iter > 0
    assert len(bounded.std_errors) == 1
    assert len(bounded_prediction) == 2
    assert all(len(row) == 2 for row in bounded_prediction)

    non_mixture_config = core.NonMixtureCureConfig(
        core.NonMixtureType.GeometricGeneralized,
        core.CureDistribution.Weibull,
        20,
        1e-5,
        1.0,
    )
    non_mixture = core.non_mixture_cure_model(time, status, covariates, non_mixture_config)
    non_mixture_prediction = core.predict_non_mixture_survival(
        non_mixture,
        [1.0, 3.0],
        covariates[:2],
        2,
        core.NonMixtureType.GeometricGeneralized,
        core.CureDistribution.Weibull,
    )
    assert type(non_mixture).__name__ == "NonMixtureCureResult"
    assert len(non_mixture.coef) == 1
    assert non_mixture.theta > 0.0
    assert non_mixture.scale > 0.0
    assert non_mixture.shape > 0.0
    assert non_mixture.dispersion == pytest.approx(1.0)
    assert 0.0 <= non_mixture.cure_fraction <= 1.0
    assert non_mixture.n_iter > 0
    assert len(non_mixture.std_errors) == 1
    assert len(non_mixture.survival_probs) == len(time)
    assert len(non_mixture_prediction) == 2
    assert all(len(row) == 2 for row in non_mixture_prediction)

    promotion = core.promotion_time_cure_model(
        time,
        status,
        covariates,
        core.CureDistribution.Weibull,
        20,
        1e-5,
    )
    assert type(promotion).__name__ == "PromotionTimeCureResult"
    assert promotion.theta > 0.0
    assert len(promotion.coef) == 1
    assert promotion.scale > 0.0
    assert promotion.shape > 0.0
    assert 0.0 <= promotion.cure_fraction <= 1.0
    assert promotion.n_iter > 0

    comparison = core.compare_cure_models(time, status, covariates, ["weibull"])
    assert type(comparison).__name__ == "CureModelComparisonResult"
    assert comparison.model_names == [
        "Mixture-weibull",
        "BCH-weibull",
        "NonMixture-weibull",
    ]
    assert len(comparison.log_likelihoods) == 3
    assert len(comparison.aic_values) == 3
    assert len(comparison.bic_values) == 3
    assert len(comparison.cure_fractions) == 3
    assert comparison.best_model_aic in comparison.model_names
    assert comparison.best_model_bic in comparison.model_names

    aliased_comparison = core.compare_cure_models(time, status, covariates, ["log-logistic"])
    assert aliased_comparison.model_names == [
        "Mixture-loglogistic",
        "BCH-loglogistic",
        "NonMixture-loglogistic",
    ]
    with pytest.raises(ValueError, match="distribution must be one of"):
        core.compare_cure_models(time, status, covariates, ["mystery"])
    with pytest.raises(ValueError, match="distributions must not be empty"):
        core.compare_cure_models(time, status, covariates, [])


def test_cause_specific_cox_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "CensoringType",
        "CauseSpecificCoxConfig",
        "CauseSpecificCoxResult",
        "cause_specific_cox",
        "cause_specific_cox_all",
    } <= stub_names

    expected_args = {
        "cause_specific_cox": ["x", "n_obs", "n_vars", "time", "cause", "config", "weights"],
        "cause_specific_cox_all": [
            "x",
            "n_obs",
            "n_vars",
            "time",
            "cause",
            "max_cause",
            "weights",
            "max_iter",
            "tol",
        ],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    expected_init_args = {
        "CensoringType": ["self", "name"],
        "CauseSpecificCoxConfig": [
            "self",
            "cause_of_interest",
            "treat_other_causes_as",
            "max_iter",
            "tol",
            "ties",
        ],
    }
    for class_name, args in expected_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    for method_name in ["predict_cumulative_hazard", "predict_survival", "predict_cif"]:
        assert list(
            inspect.signature(getattr(core.CauseSpecificCoxResult, method_name)).parameters
        ) == [
            "self",
            "x",
            "n_obs",
        ]
        assert _pyi_class_method_arg_names(stub_path, "CauseSpecificCoxResult", method_name) == [
            "self",
            "x",
            "n_obs",
        ]

    assert _pyi_class_annotation_names(stub_path, "CensoringType") == {
        "Censored",
        "Competing",
    }
    assert _pyi_class_annotation_names(stub_path, "CauseSpecificCoxConfig") == {
        "cause_of_interest",
        "treat_other_causes_as",
        "max_iter",
        "tol",
        "ties",
    }
    assert _pyi_class_property_names(stub_path, "CauseSpecificCoxResult") == {
        "coefficients",
        "std_errors",
        "hazard_ratios",
        "hr_ci_lower",
        "hr_ci_upper",
        "log_likelihood",
        "n_events",
        "n_at_risk",
        "n_competing",
        "n_censored",
        "n_iter",
        "converged",
        "cause_of_interest",
        "baseline_hazard_times",
        "baseline_hazard",
        "cumulative_baseline_hazard",
    }

    assert repr(core.CensoringType.Censored) == "CensoringType.Censored"
    assert repr(core.CensoringType("competing")) == "CensoringType.Competing"

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    cause = [1, 0, 2, 1, 0, 1, 2, 1]
    x = [0.0] * len(time)
    config = core.CauseSpecificCoxConfig(1, core.CensoringType.Censored, 20, 1e-6, "breslow")

    assert config.cause_of_interest == 1
    assert config.treat_other_causes_as == core.CensoringType.Censored
    assert config.max_iter == 20
    assert config.tol == pytest.approx(1e-6)
    assert config.ties == "breslow"

    result = core.cause_specific_cox(x, len(time), 1, time, cause, config)
    assert type(result).__name__ == "CauseSpecificCoxResult"
    assert result.cause_of_interest == 1
    assert result.n_events == 4
    assert result.n_competing == 2
    assert result.n_censored == 2
    assert result.n_at_risk == len(time)
    assert len(result.coefficients) == 1
    assert len(result.std_errors) == 1
    assert len(result.hazard_ratios) == 1
    assert len(result.hr_ci_lower) == 1
    assert len(result.hr_ci_upper) == 1
    assert len(result.baseline_hazard_times) == result.n_events
    assert len(result.baseline_hazard) == result.n_events
    assert len(result.cumulative_baseline_hazard) == result.n_events
    assert result.n_iter <= 20

    cumulative_hazard = result.predict_cumulative_hazard([0.0], 1)
    survival = result.predict_survival([0.0], 1)
    cif = result.predict_cif([0.0], 1)

    assert len(cumulative_hazard) == 1
    assert len(survival) == 1
    assert len(cif) == 1
    assert len(cumulative_hazard[0]) == result.n_events
    assert len(survival[0]) == result.n_events
    assert len(cif[0]) == result.n_events
    assert all(math.isfinite(value) for row in cumulative_hazard for value in row)
    assert all(math.isfinite(value) for row in survival for value in row)
    assert all(math.isfinite(value) for row in cif for value in row)
    assert survival[0] == sorted(survival[0], reverse=True)

    all_results = core.cause_specific_cox_all(x, len(time), 1, time, cause, 2, None, 20, 1e-6)
    assert [item.cause_of_interest for item in all_results] == [1, 2]
    assert [item.n_events for item in all_results] == [4, 2]


def test_joint_competing_risk_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "CorrelationType",
        "JointCompetingRisksConfig",
        "CauseResult",
        "JointCompetingRisksResult",
        "joint_competing_risks",
    } <= stub_names

    assert list(inspect.signature(core.joint_competing_risks).parameters) == [
        "x",
        "n_obs",
        "n_vars",
        "time",
        "cause",
        "config",
        "weights",
    ]
    assert _pyi_function_arg_names(stub_path, "joint_competing_risks") == [
        "x",
        "n_obs",
        "n_vars",
        "time",
        "cause",
        "config",
        "weights",
    ]

    expected_init_args = {
        "CorrelationType": ["self", "name"],
        "JointCompetingRisksConfig": [
            "self",
            "num_causes",
            "correlation_structure",
            "frailty_variance",
            "max_iter",
            "tol",
            "estimate_correlation",
        ],
    }
    for class_name, args in expected_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    assert _pyi_class_annotation_names(stub_path, "CorrelationType") == {
        "Independent",
        "SharedFrailty",
        "CopulaBased",
    }
    assert _pyi_class_annotation_names(stub_path, "JointCompetingRisksConfig") == {
        "num_causes",
        "correlation_structure",
        "frailty_variance",
        "max_iter",
        "tol",
        "estimate_correlation",
    }
    assert _pyi_class_property_names(stub_path, "CauseResult") == {
        "cause",
        "coefficients",
        "std_errors",
        "hazard_ratios",
        "baseline_hazard_times",
        "baseline_hazard",
        "cumulative_baseline_hazard",
    }
    assert _pyi_class_property_names(stub_path, "JointCompetingRisksResult") == {
        "cause_specific_results",
        "subdistribution_results",
        "correlation_matrix",
        "frailty_variance",
        "log_likelihood",
        "aic",
        "bic",
        "n_events_by_cause",
        "n_obs",
        "n_iter",
        "converged",
    }

    for method_name, expected_args in {
        "predict_cif": ["self", "x", "n_obs", "cause_idx"],
        "predict_overall_survival": ["self", "x", "n_obs"],
    }.items():
        assert (
            list(inspect.signature(getattr(core.JointCompetingRisksResult, method_name)).parameters)
            == expected_args
        )
        assert (
            _pyi_class_method_arg_names(stub_path, "JointCompetingRisksResult", method_name)
            == expected_args
        )

    assert repr(core.CorrelationType("shared_frailty")) == "CorrelationType.SharedFrailty"
    assert repr(core.CorrelationType("copula")) == "CorrelationType.CopulaBased"

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    cause = [1, 0, 2, 1, 0, 1, 2, 1]
    x = [0.0] * len(time)
    config = core.JointCompetingRisksConfig(
        2,
        core.CorrelationType.SharedFrailty,
        1.5,
        20,
        1e-6,
        True,
    )

    assert config.num_causes == 2
    assert config.correlation_structure == core.CorrelationType.SharedFrailty
    assert config.frailty_variance == pytest.approx(1.5)
    assert config.max_iter == 20
    assert config.tol == pytest.approx(1e-6)
    assert config.estimate_correlation is True

    result = core.joint_competing_risks(x, len(time), 1, time, cause, config)
    assert type(result).__name__ == "JointCompetingRisksResult"
    assert result.n_obs == len(time)
    assert result.n_events_by_cause == [4, 2]
    assert result.frailty_variance == pytest.approx(1.5)
    assert result.correlation_matrix == [[1.0, 0.0], [0.0, 1.0]]
    assert len(result.cause_specific_results) == 2
    assert len(result.subdistribution_results) == 2
    assert result.n_iter <= 20

    first_cause = result.cause_specific_results[0]
    assert type(first_cause).__name__ == "CauseResult"
    assert first_cause.cause == 1
    assert len(first_cause.coefficients) == 1
    assert len(first_cause.std_errors) == 1
    assert len(first_cause.hazard_ratios) == 1
    assert len(first_cause.baseline_hazard_times) == result.n_events_by_cause[0]
    assert len(first_cause.baseline_hazard) == result.n_events_by_cause[0]
    assert len(first_cause.cumulative_baseline_hazard) == result.n_events_by_cause[0]

    cif = result.predict_cif([0.0], 1, 0)
    overall_survival = result.predict_overall_survival([0.0], 1)
    assert len(cif) == 1
    assert len(cif[0]) == result.n_events_by_cause[0]
    assert all(math.isfinite(value) for row in cif for value in row)
    assert all(0.0 <= value <= 1.0 for row in cif for value in row)
    assert len(overall_survival) == 1
    assert len(overall_survival[0]) == result.n_events_by_cause[0]
    assert all(math.isfinite(value) for row in overall_survival for value in row)
    assert all(0.0 <= value <= 1.0 for row in overall_survival for value in row)


def test_longitudinal_survival_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "JointModelConfig",
        "JointLongSurvResult",
        "LandmarkAnalysisResult",
        "LongDynamicPredResult",
        "TimeVaryingCoxResult",
        "joint_longitudinal_model",
        "landmark_cox_analysis",
        "longitudinal_dynamic_pred",
        "time_varying_cox",
    } <= stub_names

    expected_args = {
        "joint_longitudinal_model": [
            "subject_id",
            "longitudinal_times",
            "longitudinal_values",
            "survival_time",
            "survival_event",
            "longitudinal_covariates",
            "survival_covariates",
            "config",
        ],
        "landmark_cox_analysis": ["time", "event", "covariates", "landmark_time", "horizon"],
        "longitudinal_dynamic_pred": [
            "subject_id",
            "measurement_times",
            "measurement_values",
            "prediction_time",
            "horizon",
            "model_coefficients",
        ],
        "time_varying_cox": ["start_time", "stop_time", "event", "covariates", "n_time_points"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    config_args = [
        "self",
        "n_quadrature_points",
        "max_iter",
        "tolerance",
        "association_type",
        "random_effects_structure",
    ]
    assert list(inspect.signature(core.JointModelConfig).parameters) == config_args[1:]
    assert _pyi_class_method_arg_names(stub_path, "JointModelConfig", "__init__") == config_args
    assert _pyi_class_annotation_names(stub_path, "JointModelConfig") == {
        "n_quadrature_points",
        "max_iter",
        "tolerance",
        "association_type",
        "random_effects_structure",
    }

    assert _pyi_class_property_names(stub_path, "JointLongSurvResult") == {
        "longitudinal_fixed_effects",
        "survival_coefficients",
        "association_parameter",
        "random_effects_variance",
        "residual_variance",
        "log_likelihood",
        "aic",
        "bic",
        "convergence_iterations",
    }
    assert _pyi_class_property_names(stub_path, "LandmarkAnalysisResult") == {
        "landmark_time",
        "coefficients",
        "standard_errors",
        "n_at_risk",
        "n_events",
        "prediction_times",
        "survival_probabilities",
    }
    assert _pyi_class_property_names(stub_path, "LongDynamicPredResult") == {
        "prediction_time",
        "horizon",
        "survival_probabilities",
        "confidence_lower",
        "confidence_upper",
        "risk_scores",
    }
    assert _pyi_class_property_names(stub_path, "TimeVaryingCoxResult") == {
        "coefficients",
        "coefficient_times",
        "standard_errors",
        "log_likelihood",
        "n_events",
    }

    for class_name, method_name, args in [
        ("JointLongSurvResult", "predict_longitudinal", ["self", "time", "covariates"]),
        ("JointLongSurvResult", "predict_survival", ["self", "time", "covariates"]),
        ("TimeVaryingCoxResult", "coefficients_at_time", ["self", "t"]),
    ]:
        assert (
            list(inspect.signature(getattr(getattr(core, class_name), method_name)).parameters)
            == args
        )
        assert _pyi_class_method_arg_names(stub_path, class_name, method_name) == args

    subject_id = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    longitudinal_times = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    longitudinal_values = [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 1.5, 2.0, 2.5]
    survival_time = [3.0, 2.5, 4.0]
    survival_event = [1, 1, 0]
    config = core.JointModelConfig(5, 10, 1e-4, "value", "intercept")
    assert config.n_quadrature_points == 5
    assert config.max_iter == 10
    assert config.tolerance == pytest.approx(1e-4)
    assert config.association_type == "value"
    assert config.random_effects_structure == "intercept"

    joint = core.joint_longitudinal_model(
        subject_id,
        longitudinal_times,
        longitudinal_values,
        survival_time,
        survival_event,
        [],
        [],
        config,
    )
    assert type(joint).__name__ == "JointLongSurvResult"
    assert len(joint.longitudinal_fixed_effects) == 2
    assert joint.longitudinal_fixed_effects == pytest.approx([1.5, 0.5])
    assert joint.survival_coefficients == []
    assert len(joint.random_effects_variance) == 2
    assert joint.residual_variance >= 0.0
    assert joint.convergence_iterations <= 10
    predicted_longitudinal = joint.predict_longitudinal([0.0, 1.0], [[], []])
    predicted_survival = joint.predict_survival([1.0, 2.0], [[], []])
    assert predicted_longitudinal == pytest.approx([1.5, 2.0])
    assert len(predicted_survival) == 2
    assert all(0.0 <= value <= 1.0 for value in predicted_survival)

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    event = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    covariates = [
        [0.1, 0.2],
        [0.2, 0.3],
        [0.3, 0.4],
        [0.4, 0.5],
        [0.5, 0.6],
        [0.6, 0.7],
        [0.7, 0.8],
        [0.8, 0.9],
        [0.9, 1.0],
        [1.0, 1.1],
    ]
    landmark = core.landmark_cox_analysis(time, event, covariates, 2.0, 5.0)
    assert type(landmark).__name__ == "LandmarkAnalysisResult"
    assert landmark.landmark_time == pytest.approx(2.0)
    assert landmark.n_at_risk == 9
    assert landmark.n_events == 3
    assert len(landmark.coefficients) == 2
    assert len(landmark.standard_errors) == 2
    assert len(landmark.prediction_times) == 11
    assert len(landmark.survival_probabilities) == 11
    assert landmark.survival_probabilities == sorted(landmark.survival_probabilities, reverse=True)

    dynamic = core.longitudinal_dynamic_pred(
        [0, 0, 0, 1, 1, 1],
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        [1.0, 1.5, 2.0, 2.0, 2.2, 2.5],
        2.0,
        3.0,
        [0.1, 0.05],
    )
    assert type(dynamic).__name__ == "LongDynamicPredResult"
    assert dynamic.prediction_time == pytest.approx(2.0)
    assert dynamic.horizon == pytest.approx(3.0)
    assert len(dynamic.survival_probabilities) == 2
    assert len(dynamic.confidence_lower) == 2
    assert len(dynamic.confidence_upper) == 2
    assert len(dynamic.risk_scores) == 2
    assert all(0.0 <= value <= 1.0 for value in dynamic.survival_probabilities)

    time_varying = core.time_varying_cox(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        [0, 1, 0, 1, 0, 1],
        [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7]],
        5,
    )
    assert type(time_varying).__name__ == "TimeVaryingCoxResult"
    assert len(time_varying.coefficient_times) == 5
    assert len(time_varying.coefficients) == 10
    assert len(time_varying.standard_errors) == 5
    assert all(len(row) == 2 for row in time_varying.standard_errors)
    assert time_varying.n_events == 3
    assert time_varying.coefficients_at_time(2.5) == pytest.approx([0.0, 0.0])


def test_functional_and_spline_regression_bindings_are_typed_to_registered_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "BasisType",
        "FunctionalSurvivalConfig",
        "FunctionalPCAResult",
        "FunctionalSurvivalResult",
        "SplineConfig",
        "FlexibleParametricResult",
        "RestrictedCubicSplineResult",
        "HazardSplineResult",
        "functional_cox",
        "fpca_survival",
        "flexible_parametric_model",
        "restricted_cubic_spline",
        "predict_hazard_spline",
    } <= stub_names

    function_args = {
        "functional_cox": [
            "functional_covariates",
            "curve_times",
            "time",
            "event",
            "scalar_covariates",
            "config",
        ],
        "fpca_survival": ["curves", "n_components"],
        "flexible_parametric_model": ["time", "event", "covariates", "config"],
        "restricted_cubic_spline": ["x", "n_knots", "knots"],
        "predict_hazard_spline": ["model_result", "eval_times", "covariate_values"],
    }
    for name, args in function_args.items():
        assert _pyi_function_arg_names(stub_path, name) == args

    config_init_args = {
        "FunctionalSurvivalConfig": [
            "self",
            "basis_type",
            "n_basis",
            "n_pca_components",
            "regularization",
            "max_iter",
            "tol",
        ],
        "SplineConfig": ["self", "n_knots", "degree", "knot_placement", "boundary_knots"],
    }
    for class_name, args in config_init_args.items():
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    result_init_args = {
        "FlexibleParametricResult": [
            "self",
            "coefficients",
            "spline_coefficients",
            "std_errors",
            "knots",
            "log_likelihood",
            "aic",
            "bic",
            "n_iterations",
            "converged",
        ],
        "RestrictedCubicSplineResult": [
            "self",
            "knots",
            "basis_matrix",
            "coefficients",
            "std_errors",
        ],
        "HazardSplineResult": [
            "self",
            "time_points",
            "hazard",
            "cumulative_hazard",
            "survival",
            "lower_ci",
            "upper_ci",
        ],
    }
    for class_name, args in result_init_args.items():
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    assert _pyi_class_annotation_names(stub_path, "BasisType") == {
        "BSpline",
        "Fourier",
        "Wavelet",
        "FunctionalPCA",
    }
    assert _pyi_class_annotation_names(stub_path, "FunctionalSurvivalConfig") == {
        "basis_type",
        "n_basis",
        "n_pca_components",
        "regularization",
        "max_iter",
        "tol",
    }
    assert _pyi_class_annotation_names(stub_path, "SplineConfig") == {
        "n_knots",
        "degree",
        "knot_placement",
        "boundary_knots",
    }

    assert _pyi_class_property_names(stub_path, "FunctionalPCAResult") == {
        "eigenvalues",
        "explained_variance_ratio",
        "cumulative_variance",
        "mean_function",
        "principal_components",
        "scores",
    }
    assert _pyi_class_property_names(stub_path, "FunctionalSurvivalResult") == {
        "coefficients",
        "coefficient_se",
        "coefficient_function",
        "coefficient_times",
        "hazard_ratio",
        "ci_lower",
        "ci_upper",
        "p_values",
        "log_likelihood",
        "aic",
        "bic",
        "functional_pca",
        "basis_coefficients",
    }
    assert _pyi_class_method_arg_names(
        stub_path, "FunctionalSurvivalResult", "predict_coefficient"
    ) == ["self", "t"]
    assert _pyi_class_property_names(stub_path, "FlexibleParametricResult") == {
        "coefficients",
        "spline_coefficients",
        "std_errors",
        "knots",
        "log_likelihood",
        "aic",
        "bic",
        "n_iterations",
        "converged",
    }
    assert _pyi_class_property_names(stub_path, "RestrictedCubicSplineResult") == {
        "knots",
        "basis_matrix",
        "coefficients",
        "std_errors",
    }
    assert _pyi_class_property_names(stub_path, "HazardSplineResult") == {
        "time_points",
        "hazard",
        "cumulative_hazard",
        "survival",
        "lower_ci",
        "upper_ci",
    }

    if not _has_ml_bindings(core):
        return

    for name, args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
    for class_name, args in config_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == args[1:]

    curves = [
        [1.0, 1.2, 1.4, 1.5, 1.7],
        [0.8, 1.0, 1.1, 1.3, 1.4],
        [1.5, 1.6, 1.8, 1.9, 2.0],
        [1.1, 1.2, 1.3, 1.5, 1.6],
    ]
    fpca = core.fpca_survival(curves, 2)
    assert type(fpca).__name__ == "FunctionalPCAResult"
    assert len(fpca.eigenvalues) == 2
    assert len(fpca.scores) == 4

    functional_config = core.FunctionalSurvivalConfig(
        core.BasisType.BSpline,
        5,
        2,
        0.01,
        5,
        1e-6,
    )
    functional_result = core.functional_cox(
        curves,
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [1, 0, 1, 0],
        None,
        functional_config,
    )
    assert type(functional_result).__name__ == "FunctionalSurvivalResult"
    assert functional_result.functional_pca is not None
    assert len(functional_result.coefficient_times) == 5
    assert math.isfinite(functional_result.predict_coefficient(2.0))

    spline_config = core.SplineConfig(3, 3, "quantile", None)
    time = [float(value) for value in range(1, 21)]
    event = [1 if idx % 3 == 0 else 0 for idx in range(20)]
    covariates = [[idx * 0.1] for idx in range(20)]
    flexible = core.flexible_parametric_model(time, event, covariates, spline_config)
    assert type(flexible).__name__ == "FlexibleParametricResult"
    assert flexible.knots
    assert math.isfinite(flexible.log_likelihood)

    rcs = core.restricted_cubic_spline([float(value) for value in range(1, 51)], 4, None)
    assert type(rcs).__name__ == "RestrictedCubicSplineResult"
    assert len(rcs.knots) == 4
    assert len(rcs.basis_matrix) == 50

    hazard = core.predict_hazard_spline(
        flexible,
        [float(value) for value in range(1, 11)],
        [0.5],
    )
    assert type(hazard).__name__ == "HazardSplineResult"
    assert len(hazard.time_points) == 10
    assert len(hazard.hazard) == 10
    assert all(0.0 <= value <= 1.0 for value in hazard.survival)


def test_bayesian_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "PriorType",
        "BayesianCoxConfig",
        "BayesianCoxResult",
        "BayesianDistribution",
        "BayesianParametricConfig",
        "BayesianParametricResult",
        "DirichletProcessConfig",
        "DirichletProcessResult",
        "BayesianModelAveragingConfig",
        "BayesianModelAveragingResult",
        "SpikeSlabConfig",
        "SpikeSlabResult",
        "HorseshoeConfig",
        "HorseshoeResult",
        "bayesian_cox",
        "bayesian_cox_predict_survival",
        "bayesian_parametric",
        "bayesian_parametric_predict",
        "dirichlet_process_survival",
        "bayesian_model_averaging_cox",
        "spike_slab_cox",
        "horseshoe_cox",
    } <= stub_names

    function_args = {
        "bayesian_cox": ["x", "n_obs", "n_vars", "time", "status", "config"],
        "bayesian_cox_predict_survival": [
            "result",
            "x_new",
            "n_new",
            "n_vars",
            "baseline_hazard",
            "time_points",
        ],
        "bayesian_parametric": ["time", "status", "x", "n_obs", "n_vars", "config"],
        "bayesian_parametric_predict": [
            "result",
            "x_new",
            "n_new",
            "n_vars",
            "time_points",
            "distribution",
        ],
        "dirichlet_process_survival": ["time", "event", "covariates", "n_covariates", "config"],
        "bayesian_model_averaging_cox": [
            "time",
            "event",
            "covariates",
            "n_covariates",
            "config",
        ],
        "spike_slab_cox": ["time", "event", "covariates", "n_covariates", "config"],
        "horseshoe_cox": ["time", "event", "covariates", "n_covariates", "config"],
    }
    for name, args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    config_init_args = {
        "PriorType": ["self", "name"],
        "BayesianCoxConfig": [
            "self",
            "prior_type",
            "prior_scale",
            "n_samples",
            "n_warmup",
            "n_chains",
            "target_accept",
            "seed",
        ],
        "BayesianDistribution": ["self", "name"],
        "BayesianParametricConfig": [
            "self",
            "distribution",
            "beta_prior_scale",
            "shape_prior_mean",
            "shape_prior_sd",
            "n_samples",
            "n_warmup",
            "n_chains",
            "seed",
        ],
        "DirichletProcessConfig": [
            "self",
            "concentration",
            "n_components",
            "n_iter",
            "burnin",
            "seed",
        ],
        "BayesianModelAveragingConfig": [
            "self",
            "n_iter",
            "burnin",
            "prior_inclusion_prob",
            "seed",
        ],
        "SpikeSlabConfig": [
            "self",
            "spike_var",
            "slab_var",
            "prior_inclusion",
            "n_iter",
            "burnin",
            "seed",
        ],
        "HorseshoeConfig": ["self", "tau_global", "n_iter", "burnin", "seed"],
    }
    for class_name, args in config_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    assert _pyi_class_annotation_names(stub_path, "PriorType") == {
        "Normal",
        "Laplace",
        "Cauchy",
        "Horseshoe",
        "Flat",
    }
    assert _pyi_class_annotation_names(stub_path, "BayesianDistribution") == {
        "Weibull",
        "LogNormal",
        "LogLogistic",
        "Exponential",
    }
    assert _pyi_class_annotation_names(stub_path, "BayesianCoxConfig") == {
        "prior_type",
        "prior_scale",
        "n_samples",
        "n_warmup",
        "n_chains",
        "target_accept",
        "seed",
    }
    assert _pyi_class_annotation_names(stub_path, "BayesianParametricConfig") == {
        "distribution",
        "beta_prior_scale",
        "shape_prior_mean",
        "shape_prior_sd",
        "n_samples",
        "n_warmup",
        "n_chains",
        "seed",
    }
    assert _pyi_class_annotation_names(stub_path, "DirichletProcessConfig") == {
        "concentration",
        "n_components",
        "n_iter",
        "burnin",
        "seed",
    }
    assert _pyi_class_annotation_names(stub_path, "BayesianModelAveragingConfig") == {
        "n_iter",
        "burnin",
        "prior_inclusion_prob",
        "seed",
    }
    assert _pyi_class_annotation_names(stub_path, "SpikeSlabConfig") == {
        "spike_var",
        "slab_var",
        "prior_inclusion",
        "n_iter",
        "burnin",
        "seed",
    }
    assert _pyi_class_annotation_names(stub_path, "HorseshoeConfig") == {
        "tau_global",
        "n_iter",
        "burnin",
        "seed",
    }

    assert _pyi_class_property_names(stub_path, "BayesianCoxResult") == {
        "posterior_mean",
        "posterior_sd",
        "credible_lower",
        "credible_upper",
        "hazard_ratio_mean",
        "hazard_ratio_lower",
        "hazard_ratio_upper",
        "samples",
        "log_posterior",
        "waic",
        "loo",
        "rhat",
        "n_eff",
    }
    assert _pyi_class_property_names(stub_path, "BayesianParametricResult") == {
        "beta_mean",
        "beta_sd",
        "beta_lower",
        "beta_upper",
        "shape_mean",
        "shape_sd",
        "shape_lower",
        "shape_upper",
        "acceleration_factor_mean",
        "acceleration_factor_lower",
        "acceleration_factor_upper",
        "beta_samples",
        "shape_samples",
        "log_posterior",
        "dic",
        "waic",
    }
    assert _pyi_class_property_names(stub_path, "DirichletProcessResult") == {
        "cluster_assignments",
        "cluster_sizes",
        "cluster_survival",
        "eval_times",
        "posterior_mean_survival",
        "posterior_lower",
        "posterior_upper",
        "n_clusters",
        "concentration_posterior",
    }
    assert _pyi_class_property_names(stub_path, "BayesianModelAveragingResult") == {
        "posterior_inclusion_prob",
        "posterior_mean_coef",
        "posterior_sd_coef",
        "model_posterior_probs",
        "best_model_indices",
        "bayes_factor_vs_null",
        "n_models_visited",
        "n_vars",
    }
    assert _pyi_class_property_names(stub_path, "SpikeSlabResult") == {
        "posterior_inclusion_prob",
        "posterior_mean",
        "posterior_sd",
        "credible_lower",
        "credible_upper",
        "selected_variables",
        "n_selected",
        "log_marginal_likelihood",
    }
    assert _pyi_class_property_names(stub_path, "HorseshoeResult") == {
        "posterior_mean",
        "posterior_sd",
        "credible_lower",
        "credible_upper",
        "shrinkage_factors",
        "local_scales",
        "global_scale",
        "effective_df",
    }

    assert repr(core.PriorType("normal")) == "PriorType.Normal"
    assert repr(core.BayesianDistribution("weibull")) == "BayesianDistribution.Weibull"

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    event = [1, 0, 1, 1, 0, 1]
    covariates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    cox_config = core.BayesianCoxConfig(core.PriorType.Normal, 1.0, 20, 10, 1, 0.8, 42)
    assert cox_config.prior_type == core.PriorType.Normal
    assert cox_config.prior_scale == pytest.approx(1.0)
    cox_result = core.bayesian_cox(covariates, 6, 1, time, event, cox_config)
    assert type(cox_result).__name__ == "BayesianCoxResult"
    assert len(cox_result.posterior_mean) == 1
    assert len(cox_result.samples) == 20
    cox_prediction = core.bayesian_cox_predict_survival(
        cox_result,
        [0.2, 0.4],
        2,
        1,
        [0.1, 0.2, 0.3],
        [1.0, 2.0, 3.0],
    )
    assert len(cox_prediction) == 3
    assert all(len(matrix) == 2 for matrix in cox_prediction)
    assert all(0.0 <= value <= 1.0 for matrix in cox_prediction for row in matrix for value in row)

    parametric_config = core.BayesianParametricConfig(
        core.BayesianDistribution.Weibull,
        2.5,
        1.0,
        1.0,
        20,
        10,
        1,
        42,
    )
    parametric_result = core.bayesian_parametric(time, event, covariates, 6, 1, parametric_config)
    assert type(parametric_result).__name__ == "BayesianParametricResult"
    assert len(parametric_result.beta_mean) == 1
    assert len(parametric_result.beta_samples) == 20
    parametric_prediction = core.bayesian_parametric_predict(
        parametric_result,
        [0.2, 0.4],
        2,
        1,
        [1.0, 2.0, 3.0],
        core.BayesianDistribution.Weibull,
    )
    assert len(parametric_prediction) == 3
    assert all(len(matrix) == 2 for matrix in parametric_prediction)
    assert all(
        0.0 <= value <= 1.0 for matrix in parametric_prediction for row in matrix for value in row
    )

    dp_result = core.dirichlet_process_survival(
        time,
        event,
        covariates,
        1,
        core.DirichletProcessConfig(1.0, 3, 20, 10, 42),
    )
    assert type(dp_result).__name__ == "DirichletProcessResult"
    assert len(dp_result.cluster_assignments) == 6
    assert len(dp_result.eval_times) == 50
    assert len(dp_result.posterior_mean_survival) == 50

    bma_result = core.bayesian_model_averaging_cox(
        time,
        event,
        covariates,
        1,
        core.BayesianModelAveragingConfig(20, 10, 0.5, 42),
    )
    assert type(bma_result).__name__ == "BayesianModelAveragingResult"
    assert len(bma_result.posterior_inclusion_prob) == 1
    assert bma_result.n_vars == 1

    spike_slab_result = core.spike_slab_cox(
        time,
        event,
        covariates,
        1,
        core.SpikeSlabConfig(0.001, 10.0, 0.5, 20, 10, 42),
    )
    assert type(spike_slab_result).__name__ == "SpikeSlabResult"
    assert len(spike_slab_result.posterior_mean) == 1
    assert spike_slab_result.n_selected == len(spike_slab_result.selected_variables)

    horseshoe_result = core.horseshoe_cox(
        time,
        event,
        covariates,
        1,
        core.HorseshoeConfig(1.0, 20, 10, 42),
    )
    assert type(horseshoe_result).__name__ == "HorseshoeResult"
    assert len(horseshoe_result.posterior_mean) == 1
    assert len(horseshoe_result.shrinkage_factors) == 1
    assert math.isfinite(horseshoe_result.effective_df)


def test_penalized_and_fast_cox_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "PenaltyType",
        "ElasticNetConfig",
        "ElasticNetPathConfig",
        "ElasticNetCVConfig",
        "ElasticNetCoxResult",
        "ElasticNetCoxPath",
        "ScreeningRule",
        "FastCoxSolverConfig",
        "FastCoxConfig",
        "FastCoxPathConfig",
        "FastCoxCVConfig",
        "FastCoxResult",
        "FastCoxPath",
        "elastic_net_cox",
        "elastic_net_cox_path",
        "elastic_net_cox_cv",
        "fast_cox",
        "fast_cox_numpy",
        "fast_cox_path",
        "fast_cox_cv",
    } <= stub_names

    expected_args = {
        "elastic_net_cox": ["input", "config"],
        "elastic_net_cox_path": ["input", "config"],
        "elastic_net_cox_cv": ["input", "config"],
        "fast_cox": ["input", "config"],
        "fast_cox_numpy": ["x", "time", "status", "config", "weights", "offset"],
        "fast_cox_path": ["input", "config"],
        "fast_cox_cv": ["input", "config"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    expected_init_args = {
        "PenaltyType": ["self", "name"],
        "ElasticNetConfig": [
            "self",
            "alpha",
            "l1_ratio",
            "max_iter",
            "tol",
            "standardize",
            "warm_start",
        ],
        "ElasticNetPathConfig": [
            "self",
            "l1_ratio",
            "n_lambda",
            "lambda_min_ratio",
            "max_iter",
            "tol",
        ],
        "ElasticNetCVConfig": ["self", "l1_ratio", "n_lambda", "n_folds"],
        "ScreeningRule": ["self", "name"],
        "FastCoxSolverConfig": [
            "self",
            "max_iter",
            "tol",
            "screening",
            "working_set_size",
            "active_set_update_freq",
        ],
        "FastCoxConfig": [
            "self",
            "lambda_",
            "l1_ratio",
            "max_iter",
            "tol",
            "screening",
            "working_set_size",
            "active_set_update_freq",
            "standardize",
            "use_simd",
        ],
        "FastCoxPathConfig": [
            "self",
            "l1_ratio",
            "n_lambda",
            "lambda_min_ratio",
            "max_iter",
            "tol",
            "screening",
        ],
        "FastCoxCVConfig": ["self", "l1_ratio", "n_lambda", "n_folds", "screening", "seed"],
    }
    for class_name, args in expected_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    assert list(inspect.signature(core.ElasticNetConfig.lasso).parameters) == ["alpha"]
    assert _pyi_class_method_arg_names(stub_path, "ElasticNetConfig", "lasso") == ["alpha"]
    assert list(inspect.signature(core.ElasticNetConfig.ridge).parameters) == ["alpha"]
    assert _pyi_class_method_arg_names(stub_path, "ElasticNetConfig", "ridge") == ["alpha"]

    assert _pyi_class_annotation_names(stub_path, "ElasticNetConfig") == {
        "alpha",
        "l1_ratio",
        "max_iter",
        "tol",
        "standardize",
        "warm_start",
    }
    assert _pyi_class_annotation_names(stub_path, "ElasticNetPathConfig") == {
        "l1_ratio",
        "n_lambda",
        "lambda_min_ratio",
        "max_iter",
        "tol",
    }
    assert _pyi_class_annotation_names(stub_path, "ElasticNetCVConfig") == {
        "l1_ratio",
        "n_lambda",
        "n_folds",
    }
    assert _pyi_class_annotation_names(stub_path, "FastCoxSolverConfig") == {
        "max_iter",
        "tol",
        "screening",
        "working_set_size",
        "active_set_update_freq",
    }
    assert _pyi_class_annotation_names(stub_path, "FastCoxConfig") == {
        "lambda_",
        "l1_ratio",
        "max_iter",
        "tol",
        "screening",
        "working_set_size",
        "active_set_update_freq",
        "standardize",
        "use_simd",
    }
    assert _pyi_class_annotation_names(stub_path, "FastCoxPathConfig") == {
        "l1_ratio",
        "n_lambda",
        "lambda_min_ratio",
        "max_iter",
        "tol",
        "screening",
    }
    assert _pyi_class_annotation_names(stub_path, "FastCoxCVConfig") == {
        "l1_ratio",
        "n_lambda",
        "n_folds",
        "screening",
        "seed",
    }

    assert _pyi_class_property_names(stub_path, "ElasticNetCoxResult") == {
        "coefficients",
        "nonzero_indices",
        "lambda_used",
        "l1_ratio",
        "n_iter",
        "converged",
        "deviance",
        "df",
        "scale_factors",
        "intercept",
    }
    assert _pyi_class_property_names(stub_path, "ElasticNetCoxPath") == {
        "lambdas",
        "coefficients",
        "deviances",
        "df",
        "n_iters",
    }
    assert _pyi_class_property_names(stub_path, "FastCoxResult") == {
        "coefficients",
        "nonzero_indices",
        "lambda_used",
        "l1_ratio",
        "n_iter",
        "converged",
        "deviance",
        "df",
        "scale_factors",
        "center_values",
        "screened_out",
        "active_set_size",
    }
    assert _pyi_class_property_names(stub_path, "FastCoxPath") == {
        "lambdas",
        "coefficients",
        "deviances",
        "df",
        "n_iters",
        "converged",
    }

    x_values = [0.0, 1.0, 0.5, 1.5, 0.2, 1.2, 0.8, 1.8]
    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    status = [1, 0, 1, 1, 0, 1, 0, 1]
    input_data = core.CoxRegressionInput(
        core.CovariateMatrix(x_values, 8, 1),
        core.SurvivalData(time, status),
    )

    elastic_config = core.ElasticNetConfig(0.1, 0.5, 50, 1e-7, True, False)
    lasso_config = core.ElasticNetConfig.lasso(0.1)
    ridge_config = core.ElasticNetConfig.ridge(0.1)

    assert elastic_config.alpha == pytest.approx(0.1)
    assert elastic_config.l1_ratio == pytest.approx(0.5)
    assert elastic_config.max_iter == 50
    assert elastic_config.tol == pytest.approx(1e-7)
    assert elastic_config.standardize is True
    assert elastic_config.warm_start is False
    assert lasso_config.l1_ratio == pytest.approx(1.0)
    assert ridge_config.l1_ratio == pytest.approx(0.0)

    elastic_result = core.elastic_net_cox(input_data, elastic_config)
    elastic_path = core.elastic_net_cox_path(
        input_data,
        core.ElasticNetPathConfig(0.5, 4, 0.1, 30, 1e-7),
    )
    lambda_min, lambda_1se, mean_deviance, se_deviance = core.elastic_net_cox_cv(
        input_data,
        core.ElasticNetCVConfig(0.5, 4, 2),
    )

    assert type(elastic_result).__name__ == "ElasticNetCoxResult"
    assert len(elastic_result.coefficients) == 1
    assert elastic_result.lambda_used == pytest.approx(0.1)
    assert elastic_result.l1_ratio == pytest.approx(0.5)
    assert elastic_result.n_iter > 0
    assert elastic_result.converged is True
    assert elastic_result.df >= 0.0
    assert len(elastic_result.scale_factors) == 1
    assert type(elastic_path).__name__ == "ElasticNetCoxPath"
    assert len(elastic_path.lambdas) == 4
    assert len(elastic_path.coefficients) == 4
    assert all(len(row) == 1 for row in elastic_path.coefficients)
    assert len(elastic_path.deviances) == 4
    assert len(elastic_path.df) == 4
    assert len(elastic_path.n_iters) == 4
    assert lambda_min > 0.0
    assert lambda_1se > 0.0
    assert len(mean_deviance) == 4
    assert len(se_deviance) == 4

    fast_solver = core.FastCoxSolverConfig(30, 1e-7, core.ScreeningRule.Strong, None, 10)
    fast_config = core.FastCoxConfig(
        lambda_=0.1,
        l1_ratio=1.0,
        max_iter=30,
        tol=1e-7,
        screening=core.ScreeningRule.Strong,
        working_set_size=None,
        active_set_update_freq=10,
        standardize=True,
        use_simd=True,
    )
    assert fast_solver.max_iter == 30
    assert fast_solver.screening == core.ScreeningRule.Strong
    assert fast_config.lambda_ == pytest.approx(0.1)
    fast_config.lambda_ = 0.2
    assert fast_config.lambda_ == pytest.approx(0.2)
    fast_config.lambda_ = 0.1
    assert fast_config.l1_ratio == pytest.approx(1.0)
    assert fast_config.max_iter == 30
    assert fast_config.tol == pytest.approx(1e-7)
    assert fast_config.screening == core.ScreeningRule.Strong
    assert fast_config.working_set_size is None
    assert fast_config.active_set_update_freq == 10
    assert fast_config.standardize is True
    assert fast_config.use_simd is True

    fast_result = core.fast_cox(input_data, fast_config)
    fast_path = core.fast_cox_path(
        input_data,
        core.FastCoxPathConfig(1.0, 4, 0.1, 30, 1e-7, core.ScreeningRule.Strong),
    )
    fast_lambda_min, fast_lambda_1se, fast_mean_deviance, fast_se_deviance = core.fast_cox_cv(
        input_data,
        core.FastCoxCVConfig(1.0, 4, 2, core.ScreeningRule.Strong, 1),
    )
    np = pytest.importorskip("numpy")
    numpy_result = core.fast_cox_numpy(
        np.array(x_values, dtype=float).reshape(8, 1),
        np.array(time, dtype=float),
        np.array(status, dtype=np.int32),
        fast_config,
    )

    assert type(fast_result).__name__ == "FastCoxResult"
    assert len(fast_result.coefficients) == 1
    assert fast_result.lambda_used == pytest.approx(0.1)
    assert fast_result.l1_ratio == pytest.approx(1.0)
    assert fast_result.n_iter > 0
    assert fast_result.converged is True
    assert fast_result.df >= 0.0
    assert len(fast_result.scale_factors) == 1
    assert len(fast_result.center_values) == 1
    assert fast_result.screened_out >= 0
    assert fast_result.active_set_size >= len(fast_result.nonzero_indices)
    assert type(numpy_result).__name__ == "FastCoxResult"
    assert numpy_result.coefficients == pytest.approx(fast_result.coefficients)
    assert type(fast_path).__name__ == "FastCoxPath"
    assert len(fast_path.lambdas) == 4
    assert len(fast_path.coefficients) == 4
    assert all(len(row) == 1 for row in fast_path.coefficients)
    assert len(fast_path.deviances) == 4
    assert len(fast_path.df) == 4
    assert len(fast_path.n_iters) == 4
    assert len(fast_path.converged) == 4
    assert fast_lambda_min > 0.0
    assert fast_lambda_1se > 0.0
    assert len(fast_mean_deviance) == 4
    assert len(fast_se_deviance) == 4


def test_high_dimensional_cox_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "GroupLassoConfig",
        "GroupLassoResult",
        "SparseBoostingConfig",
        "SparseBoostingResult",
        "SISConfig",
        "SISResult",
        "StabilitySelectionConfig",
        "StabilitySelectionResult",
        "group_lasso_cox",
        "sparse_boosting_cox",
        "sis_cox",
        "stability_selection_cox",
    } <= stub_names

    expected_args = {
        "group_lasso_cox": ["time", "event", "covariates", "groups", "config"],
        "sparse_boosting_cox": ["time", "event", "covariates", "config"],
        "sis_cox": ["time", "event", "covariates", "config"],
        "stability_selection_cox": ["time", "event", "covariates", "config"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    expected_init_args = {
        "GroupLassoConfig": ["self", "lambda_", "max_iter", "tol", "standardize", "group_weights"],
        "SparseBoostingConfig": [
            "self",
            "n_iterations",
            "learning_rate",
            "subsample_ratio",
            "early_stopping_rounds",
            "l1_penalty",
            "seed",
        ],
        "SISConfig": ["self", "n_select", "iterative", "max_iter", "threshold"],
        "StabilitySelectionConfig": [
            "self",
            "n_bootstrap",
            "subsample_ratio",
            "lambda_range",
            "threshold",
            "seed",
        ],
    }
    for class_name, args in expected_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    assert _pyi_class_annotation_names(stub_path, "GroupLassoConfig") == {
        "lambda_",
        "max_iter",
        "tol",
        "standardize",
        "group_weights",
    }
    assert _pyi_class_annotation_names(stub_path, "SparseBoostingConfig") == {
        "n_iterations",
        "learning_rate",
        "subsample_ratio",
        "early_stopping_rounds",
        "l1_penalty",
        "seed",
    }
    assert _pyi_class_annotation_names(stub_path, "SISConfig") == {
        "n_select",
        "iterative",
        "max_iter",
        "threshold",
    }
    assert _pyi_class_annotation_names(stub_path, "StabilitySelectionConfig") == {
        "n_bootstrap",
        "subsample_ratio",
        "lambda_range",
        "threshold",
        "seed",
    }

    assert _pyi_class_property_names(stub_path, "GroupLassoResult") == {
        "coefficients",
        "selected_groups",
        "group_norms",
        "log_likelihood",
        "n_iter",
        "converged",
        "lambda_",
        "n_groups",
        "df",
    }
    assert _pyi_class_property_names(stub_path, "SparseBoostingResult") == {
        "coefficients",
        "selected_features",
        "feature_importance",
        "iteration_scores",
        "best_iteration",
        "n_selected",
    }
    assert _pyi_class_property_names(stub_path, "SISResult") == {
        "selected_features",
        "marginal_scores",
        "ranking",
        "n_selected",
        "iteration_selections",
    }
    assert _pyi_class_property_names(stub_path, "StabilitySelectionResult") == {
        "selected_features",
        "selection_probabilities",
        "stable_features",
        "per_lambda_selections",
        "n_selected",
    }

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    event = [1, 0, 1, 1, 0, 1, 0, 1]
    covariates = [
        0.0,
        1.0,
        1.0,
        0.5,
        0.5,
        1.5,
        1.5,
        0.2,
        0.2,
        1.2,
        1.2,
        0.8,
        0.8,
        1.8,
        1.8,
        0.1,
    ]

    group_config = core.GroupLassoConfig(
        lambda_=0.01,
        max_iter=20,
        tol=1e-6,
        standardize=True,
        group_weights=None,
    )
    assert group_config.lambda_ == pytest.approx(0.01)
    group_config.lambda_ = 0.02
    assert group_config.lambda_ == pytest.approx(0.02)
    group_config.lambda_ = 0.01
    assert group_config.max_iter == 20
    assert group_config.standardize is True
    assert group_config.group_weights is None

    group_result = core.group_lasso_cox(time, event, covariates, [0, 1], group_config)
    assert type(group_result).__name__ == "GroupLassoResult"
    assert len(group_result.coefficients) == 2
    assert len(group_result.group_norms) == 2
    assert group_result.lambda_ == pytest.approx(0.01)
    assert group_result.n_groups == 2
    assert group_result.df <= 2
    assert group_result.n_iter <= 20

    sparse_config = core.SparseBoostingConfig(10, 0.05, 0.8, 3, 0.0, 1)
    assert sparse_config.n_iterations == 10
    assert sparse_config.learning_rate == pytest.approx(0.05)
    assert sparse_config.subsample_ratio == pytest.approx(0.8)
    assert sparse_config.early_stopping_rounds == 3
    assert sparse_config.l1_penalty == pytest.approx(0.0)
    assert sparse_config.seed == 1

    sparse_result = core.sparse_boosting_cox(time, event, covariates, sparse_config)
    assert type(sparse_result).__name__ == "SparseBoostingResult"
    assert len(sparse_result.coefficients) == 2
    assert len(sparse_result.feature_importance) == 2
    assert len(sparse_result.iteration_scores) <= 10
    assert sparse_result.best_iteration < len(sparse_result.iteration_scores)
    assert sparse_result.n_selected == len(sparse_result.selected_features)

    sis_config = core.SISConfig(1, False, 3, 0.0)
    assert sis_config.n_select == 1
    assert sis_config.iterative is False
    assert sis_config.max_iter == 3
    assert sis_config.threshold == pytest.approx(0.0)

    sis_result = core.sis_cox(time, event, covariates, sis_config)
    assert type(sis_result).__name__ == "SISResult"
    assert sis_result.n_selected == 1
    assert len(sis_result.selected_features) == 1
    assert len(sis_result.marginal_scores) == 2
    assert sorted(sis_result.ranking) == [0, 1]
    assert sis_result.iteration_selections == []

    stability_config = core.StabilitySelectionConfig(5, 0.75, [0.01, 0.05], 0.2, 1)
    assert stability_config.n_bootstrap == 5
    assert stability_config.subsample_ratio == pytest.approx(0.75)
    assert stability_config.lambda_range == [0.01, 0.05]
    assert stability_config.threshold == pytest.approx(0.2)
    assert stability_config.seed == 1

    stability_result = core.stability_selection_cox(time, event, covariates, stability_config)
    assert type(stability_result).__name__ == "StabilitySelectionResult"
    assert len(stability_result.selection_probabilities) == 2
    assert len(stability_result.per_lambda_selections) == 2
    assert all(len(row) == 2 for row in stability_result.per_lambda_selections)
    assert stability_result.n_selected == len(stability_result.selected_features)
    assert set(stability_result.stable_features) <= set(stability_result.selected_features)


def test_cox_diagnostic_low_level_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    expected_names = {
        "SurvfitResiduals",
        "DfbetaResult",
        "LeverageResult",
        "SchoenfeldSmoothResult",
        "OutlierDetectionResult",
        "ModelInfluenceResult",
        "GofTestResult",
        "residuals_survfit",
        "clustered_crossprod",
        "clustered_sandwich_variance",
        "dfbeta_cox",
        "cox_dfbeta_from_score_residuals",
        "cox_event_indices",
        "cox_interval_cumulative_hazard_se",
        "cox_zph_group_variance",
        "cox_zph_term_matrix",
        "leverage_cox",
        "prediction_se_from_variance",
        "scale_schoenfeld_residuals",
        "term_prediction_se_from_variance",
        "smooth_schoenfeld",
        "outlier_detection_cox",
        "model_influence_cox",
        "goodness_of_fit_cox",
    }
    assert expected_names <= stub_names

    expected_args = {
        "residuals_survfit": ["time", "status", "surv_time", "surv", "residual_type"],
        "clustered_crossprod": ["rows", "weights", "cluster", "width"],
        "clustered_sandwich_variance": ["rows", "weights", "cluster", "variance"],
        "dfbeta_cox": [
            "time",
            "event",
            "covariates",
            "n_covariates",
            "coefficients",
            "threshold",
        ],
        "cox_event_indices": ["time", "status", "strata"],
        "cox_interval_cumulative_hazard_se": [
            "centered_rows",
            "start_hazard",
            "start_varhaz",
            "start_xbar",
            "stop_hazard",
            "stop_varhaz",
            "stop_xbar",
            "risk",
            "variance",
        ],
        "scale_schoenfeld_residuals": ["raw", "beta", "information_matrix"],
        "cox_dfbeta_from_score_residuals": ["score", "information_matrix", "scaled"],
        "cox_zph_term_matrix": ["scaled", "groups", "beta"],
        "cox_zph_group_variance": ["information_matrix", "groups", "beta"],
        "prediction_se_from_variance": ["rows", "variance"],
        "term_prediction_se_from_variance": ["rows", "variance", "groups"],
        "leverage_cox": [
            "time",
            "event",
            "covariates",
            "n_covariates",
            "coefficients",
            "threshold_multiplier",
        ],
        "smooth_schoenfeld": [
            "event_times",
            "schoenfeld_residuals",
            "n_covariates",
            "coefficients",
            "bandwidth",
            "transform",
        ],
        "outlier_detection_cox": [
            "time",
            "event",
            "covariates",
            "n_covariates",
            "coefficients",
            "outlier_threshold",
        ],
        "model_influence_cox": ["time", "event", "covariates", "n_covariates", "coefficients"],
        "goodness_of_fit_cox": ["time", "event", "covariates", "n_covariates", "coefficients"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    assert _pyi_class_property_names(stub_path, "DfbetaResult") == {
        "dfbeta",
        "dfbetas",
        "max_dfbeta",
        "influential_obs",
        "n_obs",
        "n_vars",
    }
    assert _pyi_class_property_names(stub_path, "SchoenfeldSmoothResult") == {
        "times",
        "smoothed_residuals",
        "coefficient_path",
        "slope_test_stats",
        "slope_p_values",
        "non_proportional_vars",
        "n_events",
        "n_vars",
    }

    assert core.cox_event_indices([2.0, 1.0, 2.0, 3.0], [1, 0, 1, 1], [1, 0, 0, 1]) == [
        2,
        0,
        3,
    ]
    scaled = core.scale_schoenfeld_residuals(
        [[1.0, 2.0], [3.0, 4.0]],
        [0.5, -0.5],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    assert scaled[0] == pytest.approx([1.9, 1.5])
    assert scaled[1] == pytest.approx([3.5, 3.9])
    assert core.cox_dfbeta_from_score_residuals(
        [[1.0, 2.0]],
        [[0.1, 0.2], [0.3, 0.4]],
        False,
    )[0] == pytest.approx([0.5, 1.1])
    assert core.cox_zph_term_matrix([[1.0, 2.0, 3.0]], [[0, 1], [2]], [0.5, 2.0, 7.0])[
        0
    ] == pytest.approx([4.5, 3.0])
    group_variance = core.cox_zph_group_variance(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0, 1], [2]],
        [0.5, 2.0, 7.0],
    )
    assert group_variance[0] == pytest.approx([4.25, 0.0])
    assert group_variance[1] == pytest.approx([0.0, 1.0])
    crossprod = core.clustered_crossprod(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [1.0, 0.5, 2.0],
        [0, 0, 1],
        2,
    )
    assert crossprod[0] == pytest.approx([106.25, 130.0])
    assert crossprod[1] == pytest.approx([130.0, 160.0])
    sandwich = core.clustered_sandwich_variance(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [1.0, 0.5, 2.0],
        [0, 0, 1],
        [[2.0, 0.5], [0.5, 1.0]],
    )
    assert sandwich[0] == pytest.approx([725.0, 478.75])
    assert sandwich[1] == pytest.approx([478.75, 316.5625])
    prediction_se = core.prediction_se_from_variance(
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 0.5], [0.5, 1.0]],
    )
    assert prediction_se == pytest.approx([math.sqrt(8.0), math.sqrt(46.0)])
    term_prediction_se = core.term_prediction_se_from_variance(
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 0.5], [0.5, 1.0]],
        [[0], [1], [0, 1]],
    )
    assert term_prediction_se[0] == pytest.approx([math.sqrt(2.0), 2.0, math.sqrt(8.0)])
    assert term_prediction_se[1] == pytest.approx([math.sqrt(18.0), 4.0, math.sqrt(46.0)])
    interval_se = core.cox_interval_cumulative_hazard_se(
        [[1.0, 2.0], [0.0, 0.0]],
        [0.25, 0.0],
        [0.04, 0.50],
        [[0.1, 0.2], [0.0, 0.0]],
        [1.0, 0.0],
        [0.25, 0.25],
        [[0.4, 0.8], [0.0, 0.0]],
        [3.0, 2.0],
        [[2.0, 0.5], [0.5, 1.0]],
    )
    assert interval_se == pytest.approx([3.0 * math.sqrt(1.83), 0.0])


def test_data_prep_low_level_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    expected_names = {
        "SplitResult",
        "ClusterResult",
        "StrataResult",
        "AeqSurvResult",
        "NearDateResult",
        "RttrightResult",
        "Surv2DataResult",
        "CondenseResult",
        "TcutResult",
        "TimelineResult",
        "IntervalResult",
        "survsplit",
        "tmerge",
        "tmerge2",
        "tmerge3",
        "collapse",
        "cluster",
        "cluster_str",
        "strata",
        "strata_str",
        "aeq_surv",
        "neardate",
        "neardate_str",
        "rttright",
        "rttright_stratified",
        "surv2data",
        "survcondense",
        "tcut",
        "tcut_expand",
        "to_timeline",
        "from_timeline",
    }
    assert expected_names <= stub_names

    expected_args = {
        "survsplit": ["tstart", "tstop", "cut"],
        "tmerge": ["id", "time1", "newx", "nid", "ntime", "x"],
        "tmerge2": ["id", "time1", "nid", "ntime"],
        "tmerge3": ["id", "miss"],
        "collapse": ["y", "x", "istate", "id", "wt", "order"],
        "cluster": ["id"],
        "cluster_str": ["id"],
        "strata": ["variables"],
        "strata_str": ["variables"],
        "aeq_surv": ["time", "tolerance"],
        "neardate": ["id1", "date1", "id2", "date2", "best", "nomatch"],
        "neardate_str": ["id1", "date1", "id2", "date2", "best", "nomatch"],
        "rttright": ["time", "status", "weights", "timefix", "renorm"],
        "rttright_stratified": ["time", "status", "strata", "weights", "timefix", "renorm"],
        "surv2data": ["id", "time", "event_time", "event_status"],
        "survcondense": ["id", "time1", "time2", "status"],
        "tcut": ["value", "breaks", "labels"],
        "tcut_expand": ["start", "stop", "cuts"],
        "to_timeline": ["id", "time1", "time2", "status", "time_points"],
        "from_timeline": ["id", "states", "time_points"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    assert _pyi_class_property_names(stub_path, "SplitResult") == {
        "row",
        "interval",
        "start",
        "end",
        "censor",
    }
    assert _pyi_class_property_names(stub_path, "ClusterResult") == {
        "cluster_ids",
        "n_clusters",
        "cluster_sizes",
        "levels",
    }
    assert _pyi_class_property_names(stub_path, "StrataResult") == {
        "strata",
        "levels",
        "counts",
        "n_strata",
    }
    expected_properties = {
        "AeqSurvResult": {"time", "adjusted_count", "adjusted_indices"},
        "NearDateResult": {"indices", "distances", "n_matched"},
        "RttrightResult": {"weights", "time", "status", "order"},
        "Surv2DataResult": {"id", "time1", "time2", "status", "row_index"},
        "CondenseResult": {"id", "time1", "time2", "status", "row_map"},
        "TcutResult": {"codes", "levels", "breaks", "counts"},
        "TimelineResult": {"id", "states", "time_points"},
        "IntervalResult": {"id", "time1", "time2", "status"},
    }
    for class_name, properties in expected_properties.items():
        assert _pyi_class_property_names(stub_path, class_name) == properties

    collapsed = core.collapse(
        [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 1.0, 0.0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 2, 2],
        [1.0, 1.0, 1.0, 1.0],
        [0, 1, 2, 3],
    )
    assert collapsed == {
        "matrix": [[1, 1], [2, 2], [3, 3], [4, 4]],
        "dimnames": ["start", "end"],
    }


def test_cox_counting_low_level_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {"CoxCountOutput", "coxcount1", "coxcount2", "norisk"} <= stub_names

    expected_args = {
        "coxcount1": ["time", "status", "strata"],
        "coxcount2": ["time1", "time2", "status", "sort1", "sort2", "strata"],
        "norisk": ["time1", "time2", "status", "sort1", "sort2", "strata"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    assert _pyi_class_property_names(stub_path, "CoxCountOutput") == {
        "time",
        "nrisk",
        "index",
        "status",
    }

    right_censored = core.coxcount1([1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [1, 0, 0])
    assert type(right_censored).__name__ == "CoxCountOutput"
    assert right_censored.time == [1.0, 3.0]
    assert all(index >= 1 for index in right_censored.index)

    counting = core.coxcount2(
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0],
        [0, 1, 2],
        [0, 1, 2],
        [1, 0, 0],
    )
    assert type(counting).__name__ == "CoxCountOutput"
    assert len(counting.nrisk) == len(counting.time)

    not_used = core.norisk([0.0, 1.0], [1.0, 2.0], [1, 0], [0, 1], [0, 1], [])
    assert len(not_used) == 2


def test_survdiff_low_level_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "SurvDiffResult",
        "compute_counting_logrank_components",
        "compute_logrank_components",
        "stratified_counting_logrank_components",
        "stratified_logrank_components",
        "survdiff2",
    } <= stub_names

    expected_args = ["time", "status", "group", "strata", "rho", "timefix"]
    for name in ("compute_logrank_components", "stratified_logrank_components", "survdiff2"):
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    counting_expected_args = [
        "time",
        "status",
        "group",
        "entry_times",
        "strata",
        "rho",
        "timefix",
    ]
    for name in (
        "compute_counting_logrank_components",
        "stratified_counting_logrank_components",
    ):
        assert list(inspect.signature(getattr(core, name)).parameters) == counting_expected_args
        assert _pyi_function_arg_names(stub_path, name) == counting_expected_args

    assert _pyi_class_property_names(stub_path, "SurvDiffResult") == {
        "observed",
        "expected",
        "variance",
        "chi_squared",
        "degrees_of_freedom",
    }

    result = core.compute_logrank_components(
        [1.0, 2.0, 3.0, 4.0],
        [1, 1, 0, 1],
        [1, 2, 1, 2],
        None,
        0.0,
        False,
    )
    alias_result = core.survdiff2(
        [1.0, 2.0, 3.0, 4.0],
        [1, 1, 0, 1],
        [1, 2, 1, 2],
        None,
        0.0,
        None,
    )
    counting_result = core.compute_counting_logrank_components(
        [1.0, 2.0, 3.0, 4.0],
        [1, 1, 0, 1],
        [1, 2, 1, 2],
        [0.0, 0.0, 1.0, 1.0],
        None,
        0.0,
        True,
    )
    strata_time = [2.0, 1.0, 3.0, 1.5, 2.5, 3.5]
    strata_status = [0, 1, 1, 1, 1, 0]
    strata_group = [2, 1, 2, 1, 2, 1]
    strata_codes = [1, 0, 0, 1, 0, 1]
    marker_order = sorted(range(len(strata_time)), key=lambda idx: (strata_codes[idx], idx))
    marker_strata = [
        int(
            idx + 1 == len(marker_order)
            or strata_codes[marker_order[idx + 1]] != strata_codes[marker_order[idx]]
        )
        for idx in range(len(marker_order))
    ]
    marker_result = core.compute_logrank_components(
        [strata_time[idx] for idx in marker_order],
        [strata_status[idx] for idx in marker_order],
        [strata_group[idx] for idx in marker_order],
        marker_strata,
        0.0,
        True,
    )
    stratified_result = core.stratified_logrank_components(
        strata_time,
        strata_status,
        strata_group,
        strata_codes,
        0.0,
        True,
    )
    stratified_counting_result = core.stratified_counting_logrank_components(
        strata_time,
        strata_status,
        strata_group,
        [0.0, 0.0, 1.0, 0.5, 1.5, 2.5],
        strata_codes,
        0.0,
        True,
    )
    marker_counting_result = core.compute_counting_logrank_components(
        [strata_time[idx] for idx in marker_order],
        [strata_status[idx] for idx in marker_order],
        [strata_group[idx] for idx in marker_order],
        [[0.0, 0.0, 1.0, 0.5, 1.5, 2.5][idx] for idx in marker_order],
        marker_strata,
        0.0,
        True,
    )
    fixed_result = core.compute_logrank_components(
        [1.0, 1.0 + 5e-10, 2.0, 3.0],
        [1, 1, 0, 1],
        [1, 2, 1, 2],
        None,
        0.0,
        True,
    )
    exact_result = core.compute_logrank_components(
        [1.0, 1.0 + 5e-10, 2.0, 3.0],
        [1, 1, 0, 1],
        [1, 2, 1, 2],
        None,
        0.0,
        False,
    )

    assert type(result).__name__ == "SurvDiffResult"
    assert len(result.observed) == 2
    assert len(result.expected) == 2
    assert len(result.variance) == 2
    assert type(counting_result).__name__ == "SurvDiffResult"
    assert len(counting_result.observed) == 2
    assert len(counting_result.variance) == 2
    assert fixed_result.chi_squared == 0.0
    assert exact_result.chi_squared > fixed_result.chi_squared
    assert alias_result.observed == result.observed
    assert alias_result.expected == result.expected
    assert alias_result.chi_squared == result.chi_squared
    assert stratified_result.observed == pytest.approx(marker_result.observed)
    assert stratified_result.expected == pytest.approx(marker_result.expected)
    for actual_row, expected_row in zip(
        stratified_result.variance,
        marker_result.variance,
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)
    assert stratified_counting_result.observed == pytest.approx(marker_counting_result.observed)
    assert stratified_counting_result.expected == pytest.approx(marker_counting_result.expected)
    for actual_row, expected_row in zip(
        stratified_counting_result.variance,
        marker_counting_result.variance,
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row)


def test_survobrien_and_trend_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {"SurvObrienResult", "TrendTestResult", "survobrien", "logrank_trend"} <= stub_names

    assert list(inspect.signature(core.survobrien).parameters) == [
        "time",
        "status",
        "covariate",
        "strata",
    ]
    assert list(inspect.signature(core.logrank_trend).parameters) == [
        "time",
        "status",
        "group",
        "scores",
    ]
    assert _pyi_function_arg_names(stub_path, "survobrien") == [
        "time",
        "status",
        "covariate",
        "strata",
    ]
    assert _pyi_function_arg_names(stub_path, "logrank_trend") == [
        "time",
        "status",
        "group",
        "scores",
    ]
    assert _pyi_class_property_names(stub_path, "SurvObrienResult") == {
        "statistic",
        "p_value",
        "df",
        "scores",
        "score_sum",
        "expected",
        "variance",
    }
    assert _pyi_class_property_names(stub_path, "TrendTestResult") == {
        "statistic",
        "p_value",
        "trend_direction",
    }

    obrien = core.survobrien(
        [1.0, 2.0, 3.0],
        [1, 0, 1],
        [0.2, 0.5, 0.9],
        None,
    )
    trend = core.logrank_trend(
        [1.0, 2.0, 3.0, 4.0],
        [1, 0, 1, 1],
        [0, 1, 2, 2],
        None,
    )

    assert type(obrien).__name__ == "SurvObrienResult"
    assert obrien.df == 1
    assert len(obrien.scores) == 3
    assert 0.0 <= obrien.p_value <= 1.0
    assert type(trend).__name__ == "TrendTestResult"
    assert trend.trend_direction in {"increasing", "decreasing", "none"}
    assert 0.0 <= trend.p_value <= 1.0


def test_turnbull_estimator_weights_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "TurnbullResult",
        "IntervalDistribution",
        "IntervalCensoredResult",
        "turnbull_estimator",
        "npmle_interval",
        "interval_censored_regression",
    } <= stub_names
    assert "weights" in inspect.signature(core.turnbull_estimator).parameters
    assert _pyi_function_arg_names(stub_path, "turnbull_estimator") == [
        "left",
        "right",
        "max_iter",
        "tol",
        "weights",
    ]
    assert list(inspect.signature(core.interval_censored_regression).parameters) == [
        "left",
        "right",
        "censor_type",
        "x",
        "n_obs",
        "n_vars",
        "distribution",
        "max_iter",
        "tol",
    ]
    assert _pyi_function_arg_names(stub_path, "interval_censored_regression") == [
        "left",
        "right",
        "censor_type",
        "x",
        "n_obs",
        "n_vars",
        "distribution",
        "max_iter",
        "tol",
    ]
    assert list(inspect.signature(core.IntervalDistribution).parameters) == ["name"]
    assert _pyi_class_method_arg_names(stub_path, "IntervalDistribution", "__init__") == [
        "self",
        "name",
    ]
    assert _pyi_class_property_names(stub_path, "IntervalCensoredResult") == {
        "coefficients",
        "std_errors",
        "scale",
        "shape",
        "log_likelihood",
        "aic",
        "bic",
        "n_iter",
        "converged",
        "survival_prob",
    }

    result = core.interval_censored_regression(
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 5.0, 6.0],
        [3, 3, 3, 3],
        [1.0, 0.5, 0.0, 1.0],
        4,
        1,
        core.IntervalDistribution("weibull"),
        100,
        1e-4,
    )

    assert type(result).__name__ == "IntervalCensoredResult"
    assert len(result.coefficients) == 1
    assert len(result.std_errors) == 1
    assert len(result.survival_prob) == 4
    assert result.scale > 0.0
    assert result.shape > 0.0


def test_baseline_survival_step_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "compute_baseline_survival_steps",
        "agsurv4",
        "compute_tied_baseline_summaries",
        "agsurv5",
        "cox_expected_baseline_by_stratum",
    } <= stub_names

    step_args = ["ndeath", "risk", "wt", "sn", "denom"]
    tied_args = ["n", "nvar", "dd", "x1", "x2", "xsum", "xsum2"]
    expected_baseline_args = [
        "time",
        "status",
        "covariates",
        "beta",
        "weights",
        "strata",
        "offset",
        "means",
        "entry_times",
        "method",
    ]
    assert list(inspect.signature(core.compute_baseline_survival_steps).parameters) == step_args
    assert list(inspect.signature(core.agsurv4).parameters) == step_args
    assert list(inspect.signature(core.compute_tied_baseline_summaries).parameters) == tied_args
    assert list(inspect.signature(core.agsurv5).parameters) == tied_args
    assert (
        list(inspect.signature(core.cox_expected_baseline_by_stratum).parameters)
        == expected_baseline_args
    )
    assert _pyi_function_arg_names(stub_path, "compute_baseline_survival_steps") == step_args
    assert _pyi_function_arg_names(stub_path, "agsurv4") == step_args
    assert _pyi_function_arg_names(stub_path, "compute_tied_baseline_summaries") == tied_args
    assert _pyi_function_arg_names(stub_path, "agsurv5") == tied_args
    assert (
        _pyi_function_arg_names(stub_path, "cox_expected_baseline_by_stratum")
        == expected_baseline_args
    )

    ndeath = [1, 2, 0]
    risk = [1.0, 1.0, 1.0]
    wt = [0.2, 0.3, 0.4]
    denom = [5.0, 4.0, 3.0]
    direct_steps = core.compute_baseline_survival_steps(ndeath, risk, wt, 3, denom)
    alias_steps = core.agsurv4(ndeath, risk, wt, 3, denom)
    assert alias_steps == pytest.approx(direct_steps)

    tied_args_values = (
        2,
        1,
        [1, 2],
        [10.0, 9.0],
        [0.0, 1.0],
        [10.0, 9.0],
        [0.0, 0.5],
    )
    direct_tied = core.compute_tied_baseline_summaries(*tied_args_values)
    alias_tied = core.agsurv5(*tied_args_values)
    assert set(alias_tied) == {"sum1", "sum2", "xbar"}
    assert alias_tied == direct_tied

    baseline_args = (
        [1.0, 2.0, 2.0, 3.0],
        [1, 1, 1, 0],
        [[0.0], [1.0], [2.0], [3.0]],
        [0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0, 0, 0, 0],
        [0.0, 0.0, 0.0, 0.0],
        [1.5],
        None,
    )
    strata, times, cumhaz, varhaz, xbar = core.cox_expected_baseline_by_stratum(
        *baseline_args,
        "breslow",
    )
    efron = core.cox_expected_baseline_by_stratum(*baseline_args, "efron")

    assert strata == [0]
    assert times[0] == pytest.approx([1.0, 2.0])
    assert cumhaz[0] == pytest.approx([0.25, 11.0 / 12.0])
    assert varhaz[0] == pytest.approx([0.0625, 41.0 / 144.0])
    assert xbar[0][0] == pytest.approx([0.0])
    assert xbar[0][1] == pytest.approx([1.0 / 3.0])
    assert efron[2][0] == pytest.approx([0.25, 13.0 / 12.0])
    assert efron[3][0] == pytest.approx([0.0625, 61.0 / 144.0])
    assert efron[4][0][1] == pytest.approx([13.0 / 24.0])


def test_survfitkm_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "SurvFitKMOutput",
        "SurvFitKMInfluenceOutput",
        "CountingSurvfitTables",
        "SurvfitCurveResult",
        "KaplanMeierConfig",
        "SurvfitKMOptions",
        "survfitkm",
        "survfitkm_influence",
        "survfitkm_counting_influence",
        "robust_survfitkm",
        "robust_counting_survfit_variance",
        "survfitkm_with_options",
        "counting_survfit_tables",
        "survfit_curve_from_tables",
    } <= stub_names

    assert list(inspect.signature(core.survfitkm).parameters) == [
        "time",
        "status",
        "weights",
        "entry_times",
        "position",
        "reverse",
        "computation_type",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "survfitkm") == [
        "time",
        "status",
        "weights",
        "entry_times",
        "position",
        "reverse",
        "computation_type",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert list(inspect.signature(core.robust_survfitkm).parameters) == [
        "time",
        "status",
        "cluster",
        "weights",
        "reverse",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "robust_survfitkm") == [
        "time",
        "status",
        "cluster",
        "weights",
        "reverse",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert list(inspect.signature(core.survfitkm_influence).parameters) == [
        "time",
        "status",
        "cluster",
        "weights",
        "reverse",
        "stype",
        "ctype",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "survfitkm_influence") == [
        "time",
        "status",
        "cluster",
        "weights",
        "reverse",
        "stype",
        "ctype",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert list(inspect.signature(core.survfitkm_counting_influence).parameters) == [
        "start",
        "stop",
        "status",
        "curve_time",
        "curve_estimate",
        "cluster",
        "weights",
        "reverse",
        "stype",
        "ctype",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "survfitkm_counting_influence") == [
        "start",
        "stop",
        "status",
        "curve_time",
        "curve_estimate",
        "cluster",
        "weights",
        "reverse",
        "stype",
        "ctype",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert list(inspect.signature(core.robust_counting_survfit_variance).parameters) == [
        "start",
        "stop",
        "status",
        "curve_time",
        "curve_estimate",
        "cluster",
        "weights",
        "reverse",
        "conf_level",
        "conf_type",
        "timefix",
        "stype",
        "ctype",
    ]
    assert _pyi_function_arg_names(stub_path, "robust_counting_survfit_variance") == [
        "start",
        "stop",
        "status",
        "curve_time",
        "curve_estimate",
        "cluster",
        "weights",
        "reverse",
        "conf_level",
        "conf_type",
        "timefix",
        "stype",
        "ctype",
    ]
    assert list(inspect.signature(core.survfitkm_with_options).parameters) == [
        "time",
        "status",
        "options",
    ]
    assert _pyi_function_arg_names(stub_path, "survfitkm_with_options") == [
        "time",
        "status",
        "options",
    ]
    assert list(inspect.signature(core.counting_survfit_tables).parameters) == [
        "start",
        "stop",
        "status",
        "id",
        "weights",
        "include_entry",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "counting_survfit_tables") == [
        "start",
        "stop",
        "status",
        "id",
        "weights",
        "include_entry",
        "timefix",
    ]
    assert list(inspect.signature(core.survfit_curve_from_tables).parameters) == [
        "time",
        "n_risk",
        "n_event",
        "n_event_count",
        "n_censor",
        "n_censor_count",
        "n_enter",
        "reverse",
        "stype",
        "ctype",
        "conf_level",
        "conf_type",
    ]
    assert _pyi_function_arg_names(stub_path, "survfit_curve_from_tables") == [
        "time",
        "n_risk",
        "n_event",
        "n_event_count",
        "n_censor",
        "n_censor_count",
        "n_enter",
        "reverse",
        "stype",
        "ctype",
        "conf_level",
        "conf_type",
    ]

    assert list(inspect.signature(core.KaplanMeierConfig).parameters) == [
        "reverse",
        "computation_type",
        "conf_level",
        "conf_type",
    ]
    assert _pyi_class_method_arg_names(stub_path, "KaplanMeierConfig", "__init__") == [
        "self",
        "reverse",
        "computation_type",
        "conf_level",
        "conf_type",
    ]
    assert _pyi_class_annotation_names(stub_path, "KaplanMeierConfig") == {
        "reverse",
        "computation_type",
        "conf_level",
        "conf_type",
    }
    assert list(inspect.signature(core.SurvfitKMOptions).parameters) == [
        "weights",
        "entry_times",
        "position",
        "reverse",
        "computation_type",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert _pyi_class_method_arg_names(stub_path, "SurvfitKMOptions", "__init__") == [
        "self",
        "weights",
        "entry_times",
        "position",
        "reverse",
        "computation_type",
        "conf_level",
        "conf_type",
        "timefix",
    ]
    assert _pyi_class_annotation_names(stub_path, "SurvfitKMOptions") == {
        "weights",
        "entry_times",
        "position",
        "reverse",
        "computation_type",
        "conf_level",
        "conf_type",
        "timefix",
    }
    for method_name, expected_arg in {
        "with_weights": "weights",
        "with_entry_times": "entry_times",
        "with_position": "position",
        "with_reverse": "reverse",
        "with_computation_type": "computation_type",
        "with_conf_level": "conf_level",
        "with_conf_type": "conf_type",
        "with_timefix": "timefix",
    }.items():
        assert list(inspect.signature(getattr(core.SurvfitKMOptions, method_name)).parameters) == [
            "self",
            expected_arg,
        ]
        assert _pyi_class_method_arg_names(stub_path, "SurvfitKMOptions", method_name) == [
            "self",
            expected_arg,
        ]

    assert _pyi_class_property_names(stub_path, "SurvFitKMOutput") == {
        "time",
        "n_risk",
        "n_risk_count",
        "n_event",
        "n_event_count",
        "n_censor",
        "n_censor_count",
        "estimate",
        "std_err",
        "cumhaz",
        "std_chaz",
        "cumulative_hazard",
        "cumulative_hazard_std_err",
        "conf_lower",
        "conf_upper",
    }
    assert _pyi_class_property_names(stub_path, "SurvFitKMInfluenceOutput") == {
        "time",
        "influence_surv",
        "influence_chaz",
    }
    assert _pyi_class_property_names(stub_path, "CountingSurvfitTables") == {
        "time",
        "n_risk",
        "n_risk_count",
        "n_event",
        "n_event_count",
        "n_censor",
        "n_censor_count",
        "n_enter",
        "n_enter_count",
    }
    assert _pyi_class_property_names(stub_path, "SurvfitCurveResult") == {
        "time",
        "n_risk",
        "n_event",
        "n_censor",
        "estimate",
        "std_err",
        "conf_lower",
        "conf_upper",
        "cumhaz",
        "std_chaz",
        "n_enter",
    }

    result = core.survfitkm([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 1.0])
    influence = core.survfitkm_influence(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 0.0, 1.0, 0.0],
        [0, 1, 2, 3],
    )
    counting_influence = core.survfitkm_counting_influence(
        [0.0, 10.0, 25.0, 0.0, 5.0],
        [10.0, 20.0, 30.0, 15.0, 25.0],
        [0, 0, 1, 1, 0],
        [10.0, 15.0, 20.0, 25.0, 30.0],
        [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0],
        [0, 0, 0, 1, 2],
    )
    assert counting_influence.time == pytest.approx([10.0, 15.0, 20.0, 25.0, 30.0])
    robust = core.robust_survfitkm(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0, 0, 1, 1, 2, 2],
    )
    robust_counting = core.robust_counting_survfit_variance(
        [0.0, 2.0, 0.0, 3.0, 0.0, 4.0],
        [2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
        [0, 1, 1, 0, 0, 1],
        [2.0, 3.0, 5.0, 6.0, 7.0],
        [1.0, 2.0 / 3.0, 4.0 / 9.0, 4.0 / 9.0, 0.0],
        [0, 0, 1, 1, 2, 2],
    )
    options = core.SurvfitKMOptions(conf_level=0.9).with_conf_type("plain")
    with_options = core.survfitkm_with_options(
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 1.0, 0.0, 1.0],
        options,
    )
    config = core.KaplanMeierConfig(None, None, 0.9, "plain")
    exact = core.survfitkm(
        [1.0, 1.0 + 5e-10, 2.0],
        [1.0, 1.0, 0.0],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    exact_with_options = core.survfitkm_with_options(
        [1.0, 1.0 + 5e-10, 2.0],
        [1.0, 1.0, 0.0],
        core.SurvfitKMOptions().with_timefix(False),
    )
    counting_tables = core.counting_survfit_tables(
        [0.0, 10.0, 25.0, 0.0, 5.0],
        [10.0, 20.0, 30.0, 15.0, 25.0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 2],
        None,
        True,
        True,
    )
    exact_counting_tables = core.counting_survfit_tables(
        [0.0, 0.0, 1.0, 1.0 + 5e-10],
        [1.0, 1.0 + 5e-10, 2.0, 2.0],
        [1, 1, 0, 0],
        [0, 1, 2, 3],
        None,
        False,
        False,
    )
    styled_curve = core.survfit_curve_from_tables(
        [1.0, 2.0],
        [4.0, 1.0],
        [3.0, 1.0],
        [2.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        None,
        False,
        2,
        2,
        0.95,
        "log",
    )

    assert type(result).__name__ == "SurvFitKMOutput"
    assert result.time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert result.n_risk == pytest.approx([4.0, 3.0, 2.0, 1.0])
    assert result.n_risk_count == pytest.approx([4.0, 3.0, 2.0, 1.0])
    assert result.n_event == pytest.approx([1.0, 1.0, 0.0, 1.0])
    assert result.n_event_count == pytest.approx([1.0, 1.0, 0.0, 1.0])
    assert result.n_censor == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert result.n_censor_count == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert result.estimate == pytest.approx([0.75, 0.5, 0.5, 0.0])
    assert result.cumulative_hazard == pytest.approx(result.cumhaz)
    assert result.cumulative_hazard_std_err == pytest.approx(result.std_chaz)
    assert type(influence).__name__ == "SurvFitKMInfluenceOutput"
    assert influence.time == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert influence.influence_chaz[0] == pytest.approx([0.1875, 0.1875, 0.1875, 0.1875])
    assert influence.influence_surv[3] == pytest.approx([0.0625, 0.0625, 0.21875, 0.21875])
    assert robust.std_err == pytest.approx(
        [0.1360828, 0.2721655, 0.2721655, 0.2771598, 0.2771598, 0.0]
    )
    assert robust_counting[0] == pytest.approx([0.0, 0.2721655, 0.1814437, 0.1814437, 0.0])
    assert robust_counting[1] == pytest.approx([0.0, 0.2721655, 0.2721655, 0.2721655, 0.2721655])
    assert with_options.conf_lower != pytest.approx(result.conf_lower)
    assert options.conf_level == pytest.approx(0.9)
    assert options.conf_type == "plain"
    assert config.reverse is False
    assert config.computation_type == 0
    assert config.conf_level == pytest.approx(0.9)
    assert config.conf_type == "plain"
    assert exact.time == pytest.approx([1.0, 1.0 + 5e-10, 2.0])
    assert exact.n_risk_count == pytest.approx([3.0, 2.0, 1.0])
    assert exact.n_event == pytest.approx([1.0, 1.0, 0.0])
    assert exact.n_event_count == pytest.approx([1.0, 1.0, 0.0])
    assert exact_with_options.time == pytest.approx(exact.time)
    assert type(counting_tables).__name__ == "CountingSurvfitTables"
    assert counting_tables.time == pytest.approx([0.0, 5.0, 15.0, 20.0, 25.0, 30.0])
    assert counting_tables.n_risk == pytest.approx([0.0, 2.0, 3.0, 2.0, 1.0, 1.0])
    assert counting_tables.n_risk_count == pytest.approx([0.0, 2.0, 3.0, 2.0, 1.0, 1.0])
    assert counting_tables.n_event == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    assert counting_tables.n_censor == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    assert counting_tables.n_enter == pytest.approx([2.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert counting_tables.n_enter_count == pytest.approx([2.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    assert exact_counting_tables.time == pytest.approx([1.0, 1.0 + 5e-10, 2.0])
    assert exact_counting_tables.n_event == pytest.approx([1.0, 1.0, 0.0])
    assert exact_with_options.estimate == pytest.approx(exact.estimate)
    assert type(styled_curve).__name__ == "SurvfitCurveResult"
    assert styled_curve.n_event == pytest.approx([3.0, 1.0])
    assert styled_curve.n_censor == pytest.approx([0.0, 0.0])
    assert styled_curve.cumhaz[0] == pytest.approx(3.0 / (2.0 * 4.0) + 3.0 / (2.0 * 2.5))
    assert styled_curve.n_enter is None


def test_survfit_matrix_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "SurvfitMatrixResult",
        "survfit_from_hazard",
        "survfit_from_cumhaz",
        "survfit_from_matrix",
        "survfit_multistate",
        "step_values_at",
        "condition_cox_survfit_curves",
        "cox_survfit_from_baseline",
    } <= stub_names
    assert list(inspect.signature(core.survfit_from_hazard).parameters) == [
        "time",
        "hazard",
        "n_risk",
        "n_event",
    ]
    assert list(inspect.signature(core.cox_survfit_from_baseline).parameters) == [
        "base_times",
        "base_hazards",
        "linear_predictors",
        "center",
        "base_strata",
        "curve_strata",
        "requested_times",
    ]
    assert _pyi_class_method_arg_names(stub_path, "SurvfitMatrixResult", "__init__") == [
        "self",
        "time",
        "surv",
        "cumhaz",
        "std_err",
        "n_risk",
        "n_event",
        "n_states",
    ]
    assert _pyi_function_arg_names(stub_path, "survfit_from_hazard") == [
        "time",
        "hazard",
        "n_risk",
        "n_event",
    ]
    assert _pyi_function_arg_names(stub_path, "survfit_from_cumhaz") == [
        "time",
        "cumhaz",
        "n_risk",
        "n_event",
    ]
    assert _pyi_function_arg_names(stub_path, "survfit_from_matrix") == [
        "time",
        "hazard_matrix",
    ]
    assert _pyi_function_arg_names(stub_path, "survfit_multistate") == [
        "time",
        "transition_hazards",
        "initial_state",
    ]
    assert list(inspect.signature(core.step_values_at).parameters) == [
        "times",
        "values",
        "requested_times",
        "initial",
    ]
    assert _pyi_function_arg_names(stub_path, "step_values_at") == [
        "times",
        "values",
        "requested_times",
        "initial",
    ]
    assert list(inspect.signature(core.condition_cox_survfit_curves).parameters) == [
        "times",
        "cumhaz",
        "t0",
        "include_time0",
        "filter_start_time",
        "time_epsilon",
    ]
    assert _pyi_function_arg_names(stub_path, "condition_cox_survfit_curves") == [
        "times",
        "cumhaz",
        "t0",
        "include_time0",
        "filter_start_time",
        "time_epsilon",
    ]
    assert _pyi_function_arg_names(stub_path, "cox_survfit_from_baseline") == [
        "base_times",
        "base_hazards",
        "linear_predictors",
        "center",
        "base_strata",
        "curve_strata",
        "requested_times",
    ]

    result = core.survfit_from_hazard([1.0, 2.0], [0.1, 0.2])
    curve_times, curves, cumhaz = core.cox_survfit_from_baseline(
        [1.0, 3.0, 2.0, 4.0],
        [0.2, 0.5, 0.1, 0.4],
        [0.0, math.log(2.0)],
        0.0,
        [0, 0, 1, 1],
        [1, 0],
        None,
    )
    assert result.n_states == 1
    assert len(result.get_surv_at_state(0)) == 2
    assert curve_times == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert cumhaz[0] == pytest.approx([0.0, 0.1, 0.1, 0.4])
    assert cumhaz[1] == pytest.approx([0.4, 0.4, 1.0, 1.0])
    assert curves[0] == pytest.approx([math.exp(-value) for value in cumhaz[0]])
    assert core.step_values_at(
        [1.0, 3.0, 5.0],
        [10.0, 30.0, 50.0],
        [0.5, 1.0, 4.0, 6.0],
        0.0,
    ) == pytest.approx([0.0, 10.0, 30.0, 50.0])
    conditioned_time, conditioned_surv, conditioned_cumhaz = core.condition_cox_survfit_curves(
        [1.0, 3.0, 5.0],
        [[0.2, 0.6, 1.1]],
        2.5,
        True,
        True,
        1e-9,
    )
    assert conditioned_time == pytest.approx([2.5, 3.0, 5.0])
    assert conditioned_cumhaz[0] == pytest.approx([0.0, 0.4, 0.9])
    assert conditioned_surv[0] == pytest.approx([1.0, math.exp(-0.4), math.exp(-0.9)])


def test_survfitaj_binding_is_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    arg_names = [
        "y",
        "sort1",
        "sort2",
        "utime",
        "cstate",
        "wt",
        "grp",
        "ngrp",
        "p0",
        "i0",
        "sefit",
        "entry",
        "position",
        "hindx",
        "trmat",
        "t0",
    ]

    assert {"SurvFitAJ", "survfitaj"} <= stub_names
    assert list(inspect.signature(core.survfitaj).parameters) == arg_names
    assert _pyi_function_arg_names(stub_path, "survfitaj") == arg_names
    assert _pyi_class_property_names(stub_path, "SurvFitAJ") == {
        "n_risk",
        "n_event",
        "n_censor",
        "pstate",
        "cumhaz",
        "std_err",
        "std_chaz",
        "std_auc",
        "influence",
        "n_enter",
        "n_transition",
    }

    result = core.survfitaj(
        [0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0],
        [0, 1, 2],
        [0, 1, 2],
        [1.0, 2.0],
        [0, 0, 0],
        [1.0, 1.0, 1.0],
        [0, 0, 0],
        1,
        [1.0, 0.0],
        [0.0, 0.0],
        0,
        False,
        [2, 2, 2],
        [[0]],
        [[0, 1]],
        0.0,
    )

    assert type(result).__name__ == "SurvFitAJ"
    assert len(result.pstate) == 2
    assert result.std_err is None
    assert result.n_transition[1] == pytest.approx([1.0, 1.0])


def test_survfitaj_extended_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "VarianceEstimator",
        "TransitionType",
        "AalenJohansenExtendedConfig",
        "TransitionMatrix",
        "AalenJohansenExtendedResult",
        "survfitaj_extended",
    } <= stub_names

    assert list(inspect.signature(core.VarianceEstimator).parameters) == ["name"]
    assert list(inspect.signature(core.TransitionType).parameters) == ["name"]
    assert list(inspect.signature(core.AalenJohansenExtendedConfig).parameters) == [
        "variance_estimator",
        "transition_type",
        "n_bootstrap",
        "confidence_level",
        "compute_sojourn",
        "seed",
    ]
    assert list(inspect.signature(core.survfitaj_extended).parameters) == [
        "from_state",
        "to_state",
        "time",
        "config",
        "weights",
    ]
    assert _pyi_class_annotation_names(stub_path, "VarianceEstimator") == {
        "Greenwood",
        "Aalen",
        "Bootstrap",
    }
    assert _pyi_class_annotation_names(stub_path, "TransitionType") == {
        "Standard",
        "MarkovIllnessDeath",
        "Progressive",
        "Custom",
    }
    assert _pyi_class_annotation_names(stub_path, "AalenJohansenExtendedConfig") == {
        "variance_estimator",
        "transition_type",
        "n_bootstrap",
        "confidence_level",
        "compute_sojourn",
        "seed",
    }
    assert _pyi_class_method_arg_names(
        stub_path,
        "AalenJohansenExtendedConfig",
        "__init__",
    ) == [
        "self",
        "variance_estimator",
        "transition_type",
        "n_bootstrap",
        "confidence_level",
        "compute_sojourn",
        "seed",
    ]
    assert _pyi_class_property_names(stub_path, "TransitionMatrix") == {
        "time",
        "matrix",
        "n_at_risk",
        "n_transitions",
    }
    assert _pyi_class_property_names(stub_path, "AalenJohansenExtendedResult") == {
        "time",
        "state_probs",
        "variance",
        "ci_lower",
        "ci_upper",
        "transition_matrices",
        "cumulative_incidence",
        "expected_sojourn",
        "n_states",
        "n_obs",
        "n_events",
    }
    assert _pyi_class_method_arg_names(
        stub_path,
        "AalenJohansenExtendedResult",
        "get_cif",
    ) == ["self", "to_state"]
    assert _pyi_class_method_arg_names(
        stub_path,
        "AalenJohansenExtendedResult",
        "get_state_prob",
    ) == ["self", "from_state", "to_state"]
    assert _pyi_class_method_arg_names(
        stub_path,
        "AalenJohansenExtendedResult",
        "interpolate_at",
    ) == ["self", "query_time"]
    assert _pyi_function_arg_names(stub_path, "survfitaj_extended") == [
        "from_state",
        "to_state",
        "time",
        "config",
        "weights",
    ]

    config = core.AalenJohansenExtendedConfig()
    config.variance_estimator = core.VarianceEstimator("aalen")
    config.transition_type = core.TransitionType("progressive")
    result = core.survfitaj_extended(
        [0, 0, 0],
        [1, 2, 0],
        [1.0, 1.0 + 5e-10, 2.0],
        config,
        None,
    )

    assert type(result).__name__ == "AalenJohansenExtendedResult"
    assert result.n_states == 3
    assert result.n_obs == 3
    assert result.n_events == 2
    assert result.time == pytest.approx([1.0, 2.0])
    assert result.get_cif(1) == pytest.approx([1 / 3, 1 / 3])
    assert result.get_state_prob(0, 2) == pytest.approx([1 / 3, 1 / 3])
    assert result.interpolate_at(1.0)[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert result.expected_sojourn == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert repr(result) == "AalenJohansenExtendedResult(n_states=3, n_times=2, n_obs=3)"

    transition = result.transition_matrices[0]
    assert type(transition).__name__ == "TransitionMatrix"
    assert transition.time == pytest.approx(1.0)
    assert transition.n_at_risk == [3, 0, 0]
    assert transition.n_transitions[0][1] == 1
    assert transition.n_transitions[0][2] == 1
    assert transition.matrix[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_pseudo_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "PseudoResult",
        "GEEConfig",
        "GEEResult",
        "pseudo",
        "pseudo_fast",
        "pseudo_gee_regression",
    } <= stub_names

    pseudo_args = ["time", "status", "eval_times", "type_"]
    gee_args = ["pseudo_values", "covariates", "cluster_id", "config"]
    assert list(inspect.signature(core.pseudo).parameters) == pseudo_args
    assert list(inspect.signature(core.pseudo_fast).parameters) == pseudo_args
    assert list(inspect.signature(core.pseudo_gee_regression).parameters) == gee_args
    assert _pyi_function_arg_names(stub_path, "pseudo") == pseudo_args
    assert _pyi_function_arg_names(stub_path, "pseudo_fast") == pseudo_args
    assert _pyi_function_arg_names(stub_path, "pseudo_gee_regression") == gee_args
    assert _pyi_class_method_arg_names(stub_path, "GEEConfig", "__init__") == [
        "self",
        "correlation_structure",
        "link_function",
        "max_iter",
        "tol",
    ]
    assert _pyi_class_property_names(stub_path, "PseudoResult") == {
        "pseudo",
        "time",
        "type_",
        "n",
    }
    assert _pyi_class_property_names(stub_path, "GEEResult") == {
        "coefficients",
        "std_errors",
        "z_values",
        "p_values",
        "confidence_intervals",
        "qic",
        "n_iterations",
        "converged",
    }

    pseudo_result = core.pseudo([1.0, 2.0, 3.0], [1, 0, 1], None, "survival")
    gee_result = core.pseudo_gee_regression(
        [[0.8], [0.7], [0.6]],
        [[1.0, 0.5], [1.0, 1.0], [1.0, 1.5]],
        None,
        core.GEEConfig(),
    )

    assert type(pseudo_result).__name__ == "PseudoResult"
    assert pseudo_result.n == 3
    assert pseudo_result.type_ == "survival"
    assert type(gee_result).__name__ == "GEEResult"
    assert len(gee_result.coefficients) == 2
    assert len(gee_result.confidence_intervals) == 2


def test_ridge_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "RidgePenalty",
        "RidgeResult",
        "ridge_fit",
        "ridge_cv",
    } <= stub_names

    fit_args = ["x", "n_obs", "n_vars", "time", "status", "penalty", "weights"]
    cv_args = ["x", "n_obs", "n_vars", "time", "status", "theta_grid", "n_folds"]
    assert list(inspect.signature(core.RidgePenalty).parameters) == ["theta", "scale"]
    assert list(inspect.signature(core.RidgePenalty.from_df).parameters) == [
        "df",
        "n_vars",
        "scale",
    ]
    assert list(inspect.signature(core.ridge_fit).parameters) == fit_args
    assert list(inspect.signature(core.ridge_cv).parameters) == cv_args
    assert _pyi_class_annotation_names(stub_path, "RidgePenalty") == {"theta", "scale"}
    assert _pyi_class_property_names(stub_path, "RidgePenalty") == {"df"}
    assert _pyi_class_method_arg_names(stub_path, "RidgePenalty", "__init__") == [
        "self",
        "theta",
        "scale",
    ]
    assert _pyi_class_method_arg_names(stub_path, "RidgePenalty", "from_df") == [
        "df",
        "n_vars",
        "scale",
    ]
    assert _pyi_class_method_arg_names(stub_path, "RidgePenalty", "penalty_value") == [
        "self",
        "beta",
    ]
    assert _pyi_class_method_arg_names(
        stub_path,
        "RidgePenalty",
        "penalty_gradient",
    ) == ["self", "beta"]
    assert _pyi_class_method_arg_names(
        stub_path,
        "RidgePenalty",
        "apply_to_information",
    ) == ["self", "info_diag"]
    assert _pyi_class_property_names(stub_path, "RidgeResult") == {
        "coefficients",
        "std_err",
        "df",
        "gcv",
        "theta",
        "scale_factors",
    }
    assert _pyi_function_arg_names(stub_path, "ridge_fit") == fit_args
    assert _pyi_function_arg_names(stub_path, "ridge_cv") == cv_args

    penalty = core.RidgePenalty(0.1, False)
    from_df = core.RidgePenalty.from_df(1.0, 2)
    fit = core.ridge_fit(
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        3,
        2,
        [1.0, 2.0, 3.0],
        [1, 1, 1],
        penalty,
        None,
    )
    best_theta, cv_scores = core.ridge_cv(
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        3,
        2,
        [1.0, 2.0, 3.0],
        [1, 1, 1],
        [0.01, 0.1],
        2,
    )

    assert penalty.penalty_value([1.0, 2.0]) == pytest.approx(0.25)
    assert penalty.penalty_gradient([1.0, 2.0]) == pytest.approx([0.1, 0.2])
    assert penalty.apply_to_information([1.0, 2.0]) == pytest.approx([1.1, 2.1])
    assert from_df.df == pytest.approx(1.0)
    assert type(fit).__name__ == "RidgeResult"
    assert len(fit.coefficients) == len(fit.std_err) == 2
    assert fit.theta == pytest.approx(0.1)
    assert best_theta in [0.01, 0.1]
    assert len(cv_scores) == 2


def test_aggregate_survfit_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "AggregateSurvfitResult",
        "aggregate_survfit",
        "aggregate_survfit_by_group",
    } <= stub_names

    assert list(inspect.signature(core.aggregate_survfit).parameters) == [
        "times",
        "survs",
        "std_errs",
        "weights",
        "conf_level",
    ]
    assert list(inspect.signature(core.aggregate_survfit_by_group).parameters) == [
        "times",
        "survs",
        "groups",
        "weights",
    ]
    assert _pyi_function_arg_names(stub_path, "aggregate_survfit") == [
        "times",
        "survs",
        "std_errs",
        "weights",
        "conf_level",
    ]
    assert _pyi_function_arg_names(stub_path, "aggregate_survfit_by_group") == [
        "times",
        "survs",
        "groups",
        "weights",
    ]
    assert _pyi_class_property_names(stub_path, "AggregateSurvfitResult") == {
        "time",
        "surv",
        "std_err",
        "lower",
        "upper",
        "n_curves",
        "weights",
    }

    result = core.aggregate_survfit(
        [[1.0, 2.0], [1.0, 2.0]],
        [[0.9, 0.8], [0.8, 0.6]],
        None,
        [3.0, 1.0],
        None,
    )
    assert type(result).__name__ == "AggregateSurvfitResult"
    assert result.n_curves == 2
    assert result.weights == [0.75, 0.25]
    assert result.surv == pytest.approx([0.875, 0.75])
    assert len(result.lower) == len(result.time) == len(result.upper)

    grouped = core.aggregate_survfit_by_group(
        [[1.0], [1.0], [2.0]],
        [[0.9], [0.8], [0.7]],
        [2, 1, 2],
        None,
    )
    assert [item.n_curves for item in grouped] == [1, 2]
    assert all(type(item).__name__ == "AggregateSurvfitResult" for item in grouped)


def test_survcheck_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "SurvCheckResult",
        "survcheck",
        "survcheck_simple",
    } <= stub_names

    check_args = ["id", "time1", "time2", "status", "istate"]
    simple_args = ["time", "status"]
    assert list(inspect.signature(core.survcheck).parameters) == check_args
    assert list(inspect.signature(core.survcheck_simple).parameters) == simple_args
    assert _pyi_function_arg_names(stub_path, "survcheck") == check_args
    assert _pyi_function_arg_names(stub_path, "survcheck_simple") == simple_args
    assert _pyi_class_property_names(stub_path, "SurvCheckResult") == {
        "n_subjects",
        "n_transitions",
        "n_problems",
        "overlap_ids",
        "gap_ids",
        "teleport_ids",
        "invalid_ids",
        "transitions",
        "flags",
        "is_valid",
        "messages",
    }

    result = core.survcheck(
        [1, 1, 2],
        [0.0, 2.0, 0.0],
        [2.0, 4.0, 1.0],
        [1, 0, 1],
        [0, 1, 0],
    )
    simple = core.survcheck_simple([1.0, -1.0], [1, 0])

    assert type(result).__name__ == "SurvCheckResult"
    assert result.n_subjects == 2
    assert result.transitions == {"0 -> 1": 2, "1 -> 0": 1}
    assert result.flags == [0, 0, 0]
    assert type(simple).__name__ == "SurvCheckResult"
    assert simple.is_valid is False
    assert simple.invalid_ids == [1]


def test_nelson_aalen_and_stratified_km_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "NelsonAalenResult",
        "StratifiedKMResult",
        "nelson_aalen_estimator",
        "stratified_kaplan_meier",
    } <= stub_names

    assert list(inspect.signature(core.nelson_aalen_estimator).parameters) == [
        "time",
        "status",
        "weights",
        "confidence_level",
    ]
    assert list(inspect.signature(core.stratified_kaplan_meier).parameters) == [
        "time",
        "status",
        "strata",
        "confidence_level",
    ]
    assert _pyi_function_arg_names(stub_path, "nelson_aalen_estimator") == [
        "time",
        "status",
        "weights",
        "confidence_level",
    ]
    assert _pyi_function_arg_names(stub_path, "stratified_kaplan_meier") == [
        "time",
        "status",
        "strata",
        "confidence_level",
    ]
    assert _pyi_class_property_names(stub_path, "NelsonAalenResult") == {
        "time",
        "cumulative_hazard",
        "variance",
        "ci_lower",
        "ci_upper",
        "n_risk",
        "n_events",
    }
    assert _pyi_class_method_arg_names(stub_path, "NelsonAalenResult", "survival") == ["self"]
    assert _pyi_class_property_names(stub_path, "StratifiedKMResult") == {
        "strata",
        "times",
        "survival",
        "ci_lower",
        "ci_upper",
        "n_risk",
        "n_events",
    }

    hazard = core.nelson_aalen_estimator(
        [1.0, 2.0, 3.0],
        [1, 0, 1],
        None,
        None,
    )
    assert type(hazard).__name__ == "NelsonAalenResult"
    assert hazard.time == [1.0, 3.0]
    assert hazard.n_events == [1, 1]
    assert len(hazard.survival()) == len(hazard.cumulative_hazard)

    stratified = core.stratified_kaplan_meier(
        [1.0, 2.0, 1.5, 2.5],
        [1, 0, 1, 0],
        [2, 2, 1, 1],
        None,
    )
    assert type(stratified).__name__ == "StratifiedKMResult"
    assert stratified.strata == [1, 2]
    assert len(stratified.times) == len(stratified.survival) == 2
    assert stratified.n_events == [[1], [1]]


def test_power_and_accrual_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "SampleSizeResult",
        "AccrualResult",
        "sample_size_survival",
        "sample_size_survival_freedman",
        "power_survival",
        "expected_events",
    } <= stub_names

    function_args = {
        "sample_size_survival": [
            "hazard_ratio",
            "power",
            "alpha",
            "allocation_ratio",
            "sided",
        ],
        "sample_size_survival_freedman": [
            "hazard_ratio",
            "prob_event",
            "power",
            "alpha",
            "allocation_ratio",
            "sided",
        ],
        "power_survival": [
            "n_events",
            "hazard_ratio",
            "alpha",
            "allocation_ratio",
            "sided",
        ],
        "expected_events": [
            "n_total",
            "hazard_control",
            "hazard_ratio",
            "accrual_time",
            "followup_time",
            "allocation_ratio",
            "dropout_rate",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "SampleSizeResult": [
            "self",
            "n_total",
            "n_events",
            "n_per_group",
            "power",
            "alpha",
            "hazard_ratio",
            "method",
        ],
        "AccrualResult": [
            "self",
            "n_total",
            "accrual_time",
            "followup_time",
            "study_duration",
            "expected_events",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert _pyi_class_property_names(stub_path, "SampleSizeResult") == {
        "n_total",
        "n_events",
        "n_per_group",
        "power",
        "alpha",
        "hazard_ratio",
        "method",
    }
    assert _pyi_class_property_names(stub_path, "AccrualResult") == {
        "n_total",
        "accrual_time",
        "followup_time",
        "study_duration",
        "expected_events",
    }

    schoenfeld = core.sample_size_survival(0.7, None, None, None, None)
    freedman = core.sample_size_survival_freedman(0.7, 0.6, None, None, None, None)
    power = core.power_survival(64, 0.7, None, None, None)
    accrual = core.expected_events(100, 0.1, 0.7, 12.0, 6.0, None, None)

    assert type(schoenfeld).__name__ == "SampleSizeResult"
    assert schoenfeld.method == "Schoenfeld"
    assert schoenfeld.n_total == sum(schoenfeld.n_per_group)
    assert type(freedman).__name__ == "SampleSizeResult"
    assert freedman.method == "Freedman"
    assert 0.0 < power < 1.0
    assert type(accrual).__name__ == "AccrualResult"
    assert accrual.study_duration == pytest.approx(18.0)
    assert 0.0 < accrual.expected_events < accrual.n_total


def test_frailty_cox_and_link_function_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {"LinkFunctionParams", "perform_cox_regression_frailty"} <= stub_names

    function_args = [
        "time",
        "event",
        "covariates",
        "offset",
        "weights",
        "strata",
        "frail",
        "max_iter",
        "eps",
    ]
    signature = inspect.signature(core.perform_cox_regression_frailty)
    assert list(signature.parameters) == function_args
    assert all(
        signature.parameters[name].default is None
        for name in ["offset", "weights", "strata", "frail", "max_iter", "eps"]
    )
    assert _pyi_function_arg_names(stub_path, "perform_cox_regression_frailty") == function_args

    assert list(inspect.signature(core.LinkFunctionParams).parameters) == ["edge"]
    assert _pyi_class_method_arg_names(stub_path, "LinkFunctionParams", "__init__") == [
        "self",
        "edge",
    ]
    for method_name in ["blogit", "bprobit", "bcloglog", "blog"]:
        assert list(
            inspect.signature(getattr(core.LinkFunctionParams, method_name)).parameters
        ) == [
            "self",
            "input",
        ]
        assert _pyi_class_method_arg_names(stub_path, "LinkFunctionParams", method_name) == [
            "self",
            "input",
        ]

    link = core.LinkFunctionParams(0.001)
    assert link.blogit(0.5) == pytest.approx(0.0)
    assert link.bprobit(0.5) == pytest.approx(0.0)
    assert link.bcloglog(0.5) == pytest.approx(math.log(-math.log(0.5)))
    assert link.blog(0.5) == pytest.approx(math.log(0.5))
    assert math.isfinite(link.blogit(0.0))
    assert math.isfinite(link.blogit(1.0))

    bounded = core.LinkFunctionParams(0.05)
    assert bounded.bprobit(0.0) == pytest.approx(-1.6448536)
    assert bounded.bcloglog(0.0) == pytest.approx(-2.9701952)
    large_edge = core.LinkFunctionParams(0.6)
    assert large_edge.blogit(0.75) == pytest.approx(0.4054651)
    assert large_edge.bprobit(0.75) == pytest.approx(0.2533471)
    assert large_edge.bcloglog(0.75) == pytest.approx(-0.08742157)

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    event = [1, 0, 1, 0, 1, 0]
    column_major = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    row_major = [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]
    expected_keys = {
        "coefficients",
        "standard_errors",
        "p_values",
        "confidence_intervals",
        "log_likelihood",
        "score",
        "wald_test",
        "iterations",
        "converged",
        "variance_matrix",
    }

    column_result = core.perform_cox_regression_frailty(time, event, column_major, max_iter=5)
    row_result = core.perform_cox_regression_frailty(time, event, row_major, max_iter=5)
    assert set(column_result) == expected_keys
    assert set(row_result) == expected_keys
    assert len(column_result["coefficients"]) == 1
    assert row_result["coefficients"] == pytest.approx(column_result["coefficients"])
    assert all(math.isfinite(value) for value in row_result["coefficients"])
    assert len(row_result["variance_matrix"]) == 1
    assert len(row_result["variance_matrix"][0]) == 1

    frailty_result = core.perform_cox_regression_frailty(
        time,
        event,
        row_major,
        frail=[0, 1, 0, 1, 0, 1],
        max_iter=5,
    )
    assert set(frailty_result) == expected_keys
    assert len(frailty_result["coefficients"]) == 2
    assert len(frailty_result["standard_errors"]) == 2
    assert len(frailty_result["variance_matrix"]) == 2
    assert all(math.isfinite(value) for value in frailty_result["coefficients"])

    with pytest.raises(RuntimeError, match="Offset vector length"):
        core.perform_cox_regression_frailty(time, event, row_major, offset=[0.0])


def test_classical_utility_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "cipoisson_exact",
        "cipoisson_anscombe",
        "cipoisson",
        "agexact",
    } <= stub_names

    function_args = {
        "cipoisson_exact": ["k", "time", "p"],
        "cipoisson_anscombe": ["k", "time", "p"],
        "cipoisson": ["k", "time", "p", "method"],
        "agexact": [
            "maxiter",
            "nused",
            "nvar",
            "start",
            "stop",
            "event",
            "covar",
            "offset",
            "strata",
            "means",
            "beta",
            "u",
            "imat",
            "loglik",
            "work",
            "work2",
            "eps",
            "tol_chol",
            "nocenter",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    exact = core.cipoisson_exact(5, 10.0, 0.95)
    anscombe = core.cipoisson_anscombe(5, 10.0, 0.95)
    dispatched = core.cipoisson(5, 10.0, 0.95, "exact")
    assert isinstance(exact, tuple)
    assert isinstance(anscombe, tuple)
    assert exact[0] < exact[1]
    assert len(anscombe) == 2
    assert dispatched == pytest.approx(exact)

    agexact = core.agexact(
        0,
        1,
        1,
        [0.0],
        [1.0],
        [1],
        [1.0],
        [0.0],
        [0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0, 0.0],
        [0.0] * 5,
        [0, 0],
        1e-9,
        1e-9,
        [1],
    )

    assert sorted(agexact) == [
        "beta",
        "covar",
        "flag",
        "imat",
        "loglik",
        "maxiter",
        "means",
        "sctest",
        "u",
    ]
    assert agexact["beta"] == pytest.approx([0.0])
    assert agexact["loglik"] == pytest.approx([0.0, 0.0])
    assert agexact["flag"] == 0


def test_low_level_bridge_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "perform_concordance1_calculation",
        "perform_concordance3_calculation",
        "perform_concordance_calculation",
        "perform_score_calculation",
        "perform_agscore3_calculation",
        "perform_pystep_simple_calculation",
        "perform_pystep_calculation",
        "perform_pyears_calculation",
        "cox_callback",
    } <= stub_names

    function_args = {
        "perform_concordance1_calculation": [
            "time_data",
            "weights",
            "indices",
            "ntree",
        ],
        "perform_concordance3_calculation": [
            "time_data",
            "indices",
            "weights",
            "time_weights",
            "sort_stop",
            "do_residuals",
        ],
        "perform_concordance_calculation": [
            "time_data",
            "predictor_values",
            "weights",
            "time_weights",
            "sort_stop",
            "sort_start",
            "do_residuals",
        ],
        "perform_score_calculation": [
            "time_data",
            "covariates",
            "strata",
            "score",
            "weights",
            "method",
        ],
        "perform_agscore3_calculation": [
            "time_data",
            "covariates",
            "strata",
            "score",
            "weights",
            "method",
            "sort1",
        ],
        "perform_pystep_simple_calculation": [
            "odim",
            "data",
            "ofac",
            "odims",
            "ocut",
            "timeleft",
        ],
        "perform_pystep_calculation": [
            "edim",
            "data",
            "efac",
            "edims",
            "ecut",
            "tmax",
        ],
        "perform_pyears_calculation": [
            "time_data",
            "weights",
            "expected_dim",
            "expected_factors",
            "expected_dims",
            "expected_cuts",
            "expected_rates",
            "expected_data",
            "observed_dim",
            "observed_factors",
            "observed_dims",
            "observed_cuts",
            "method",
            "observed_data",
            "do_event",
            "ny",
        ],
        "cox_callback": [
            "which",
            "coef",
            "first",
            "second",
            "penalty",
            "flag",
            "fexpr",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    time_data = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0]
    covariates = [0.5, 1.0, 1.5]
    strata = [0, 0, 0]
    scores = [1.0, 1.0, 1.0]
    weights = [1.0, 1.0, 1.0]

    score = core.perform_score_calculation(time_data, covariates, strata, scores, weights, 0)
    agscore3 = core.perform_agscore3_calculation(
        time_data,
        covariates,
        strata,
        scores,
        weights,
        0,
        [1, 2, 3],
    )
    assert score["method"] == agscore3["method"] == "breslow"
    assert score["n_observations"] == agscore3["n_observations"] == 3
    assert len(score["residuals"]) == len(agscore3["residuals"]) == 3

    concordance1 = core.perform_concordance1_calculation(
        [1.0, 2.0, 3.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0, 1, 2],
        4,
    )
    concordance3 = core.perform_concordance3_calculation(
        [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0, 1, 2, 3, 4],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0, 1, 2, 3, 4],
        False,
    )
    concordance5 = core.perform_concordance_calculation(
        [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        [0, 1, 2, 3, 4],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0, 1, 2, 3, 4],
    )
    assert "counts" in concordance1
    assert "information_matrix" in concordance3
    assert "information_matrix" in concordance5
    assert "concordance_index" in concordance1
    assert "concordance_index" in concordance3
    assert "concordance_index" in concordance5

    simple_step = core.perform_pystep_simple_calculation(
        1,
        [0.5],
        [0],
        [2],
        [[0.0, 1.0, 2.0]],
        10.0,
    )
    step = core.perform_pystep_calculation(
        1,
        [0.25],
        [0],
        [2],
        [[0.0, 1.0]],
        1.0,
    )
    assert simple_step["time_step"] == pytest.approx(0.5)
    assert simple_step["index"] == 0
    assert step["updated_data"] == pytest.approx([1.0])

    pyears = core.perform_pyears_calculation(
        [2.0, 1.0],
        [1.0],
        1,
        [0],
        [1],
        [0.0],
        [0.5],
        [0.0],
        1,
        [0],
        [1],
        [0.0, 5.0],
        1,
        [0.0],
        1,
        2,
    )
    assert sorted(pyears) == ["offtable", "pcount", "pexpect", "pn", "pyears"]
    assert pyears["pyears"] == pytest.approx([2.0])

    def callback(coef, *, which):
        return {
            "coef": [value + which for value in coef],
            "first": [1.0, 2.0],
            "second": [3.0, 4.0],
            "penalty": [5.0, 6.0],
            "flag": [True, False],
        }

    callback_result = core.cox_callback(
        2,
        [1.0, 2.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0, 0],
        callback,
    )
    assert callback_result == ([3.0, 4.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1, 0])


def test_reliability_tool_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "ReliabilityResult",
        "WarrantyConfig",
        "WarrantyResult",
        "RenewalResult",
        "ReliabilityGrowthResult",
        "reliability",
        "reliability_inverse",
        "hazard_to_reliability",
        "failure_probability",
        "conditional_reliability",
        "mean_residual_life",
        "warranty_analysis",
        "renewal_analysis",
        "reliability_growth",
    } <= stub_names

    function_args = {
        "reliability": ["time", "surv", "std_err", "conf_level", "scale"],
        "reliability_inverse": ["estimate", "scale"],
        "hazard_to_reliability": ["time", "hazard"],
        "failure_probability": ["surv"],
        "conditional_reliability": ["time", "surv", "t0"],
        "mean_residual_life": ["time", "surv", "at_time"],
        "warranty_analysis": ["time", "event", "n_units", "config"],
        "renewal_analysis": ["failure_times", "event", "time_horizon", "repair_time"],
        "reliability_growth": ["failure_times", "cumulative_time"],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "ReliabilityResult": ["self", "time", "estimate", "std_err", "lower", "upper", "scale"],
        "WarrantyConfig": [
            "self",
            "warranty_period",
            "cost_per_failure",
            "cost_per_repair",
            "discount_rate",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert list(inspect.signature(core.ReliabilityScale.from_str).parameters) == ["s"]
    assert _pyi_class_method_arg_names(stub_path, "ReliabilityScale", "from_str") == ["s"]
    assert _pyi_class_property_names(stub_path, "ReliabilityResult") == {
        "time",
        "estimate",
        "std_err",
        "lower",
        "upper",
        "scale",
    }
    assert _pyi_class_property_names(stub_path, "WarrantyConfig") == {
        "warranty_period",
        "cost_per_failure",
        "cost_per_repair",
        "discount_rate",
    }
    assert _pyi_class_property_names(stub_path, "WarrantyResult") == {
        "expected_failures",
        "expected_cost",
        "cost_per_unit",
        "failure_probability",
        "time_points",
        "cumulative_failures",
        "cumulative_cost",
    }
    assert _pyi_class_property_names(stub_path, "RenewalResult") == {
        "expected_renewals",
        "renewal_variance",
        "mtbf",
        "availability",
        "time_points",
        "renewal_function",
    }
    assert _pyi_class_property_names(stub_path, "ReliabilityGrowthResult") == {
        "initial_mtbf",
        "final_mtbf",
        "growth_rate",
        "time_points",
        "mtbf_trajectory",
    }

    direct = core.ReliabilityResult([1.0], [0.9], None, None, None, "surv")
    transformed = core.reliability(
        [1.0, 2.0],
        [0.9, 0.8],
        [0.01, 0.02],
        0.95,
        "cumhaz",
    )
    inverse = core.reliability_inverse(transformed.estimate, "cumhaz")
    hazard = core.hazard_to_reliability([1.0, 2.0], [0.1, 0.2])
    conditional = core.conditional_reliability([1.0, 2.0], [0.9, 0.8], 1.0)

    assert type(direct).__name__ == "ReliabilityResult"
    assert direct.estimate == pytest.approx([0.9])
    assert transformed.scale == "cumhaz"
    assert transformed.estimate == pytest.approx([0.10536051565782628, 0.2231435513142097])
    assert transformed.lower is not None
    assert transformed.upper is not None
    assert inverse == pytest.approx([0.9, 0.8])
    assert hazard.scale == "surv"
    assert hazard.estimate == pytest.approx([0.9048374180, 0.7408182207])
    assert core.failure_probability([0.9, 0.8]) == pytest.approx([0.1, 0.2])
    assert conditional.scale == "conditional_surv"
    assert conditional.time == pytest.approx([0.0, 1.0])
    assert core.mean_residual_life([1.0, 2.0, 3.0], [0.9, 0.8, 0.5], 1.0) == pytest.approx(
        5.0 / 3.0
    )

    warranty_config = core.WarrantyConfig(5.0, 100.0, None, 0.0)
    assert warranty_config.cost_per_repair == pytest.approx(50.0)
    warranty_config.discount_rate = 0.01
    assert warranty_config.discount_rate == pytest.approx(0.01)

    warranty = core.warranty_analysis([1.0, 2.0, 3.0], [1, 0, 1], 100, warranty_config)
    renewal = core.renewal_analysis([1.0, 2.0, 3.0], [1, 1, 1], 10.0, None)
    growth = core.reliability_growth([1.0, 2.0, 3.0], [1.0, 3.0, 6.0])

    assert type(warranty).__name__ == "WarrantyResult"
    assert warranty.expected_failures >= 0.0
    assert len(warranty.time_points) == 101
    assert type(renewal).__name__ == "RenewalResult"
    assert renewal.mtbf == pytest.approx(2.0)
    assert len(renewal.renewal_function) == 101
    assert type(growth).__name__ == "ReliabilityGrowthResult"
    assert growth.initial_mtbf == pytest.approx(1.0)
    assert growth.final_mtbf == pytest.approx(2.0)
    assert len(growth.time_points) == len(growth.mtbf_trajectory) == 100


def test_monitoring_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "DriftConfig",
        "DriftReport",
        "FeatureDriftResult",
        "PerformanceDriftResult",
        "detect_drift",
        "monitor_performance",
        "FairnessAuditResult",
        "ModelCard",
        "ModelPerformanceMetrics",
        "SubgroupPerformance",
        "create_model_card",
        "fairness_audit",
    } <= stub_names

    function_args = {
        "detect_drift": [
            "reference_features",
            "current_features",
            "feature_names",
            "reference_predictions",
            "current_predictions",
            "config",
        ],
        "monitor_performance": [
            "predictions",
            "time",
            "event",
            "period_labels",
            "c_index_threshold",
        ],
        "create_model_card": [
            "model_name",
            "model_type",
            "version",
            "description",
            "intended_use",
            "training_data_description",
            "n_training_samples",
            "n_events",
            "feature_names",
            "overall_performance",
            "subgroup_performance",
            "limitations",
            "ethical_considerations",
            "caveats",
        ],
        "fairness_audit": [
            "predictions",
            "time",
            "event",
            "protected_attribute",
            "group_labels",
            "disparity_threshold",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "DriftConfig": ["self", "window_size", "threshold_psi", "threshold_ks", "n_bins"],
        "ModelPerformanceMetrics": [
            "self",
            "c_index",
            "brier_score",
            "integrated_brier",
            "calibration_slope",
            "calibration_intercept",
            "ci_lower_c_index",
            "ci_upper_c_index",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert _pyi_class_annotation_names(stub_path, "DriftConfig") == {
        "window_size",
        "threshold_psi",
        "threshold_ks",
        "n_bins",
    }
    assert _pyi_class_annotation_names(stub_path, "ModelPerformanceMetrics") == {
        "c_index",
        "brier_score",
        "integrated_brier",
        "calibration_slope",
        "calibration_intercept",
        "ci_lower_c_index",
        "ci_upper_c_index",
    }

    assert _pyi_class_property_names(stub_path, "FeatureDriftResult") == {
        "feature_name",
        "psi",
        "ks_statistic",
        "ks_pvalue",
        "has_drift",
        "drift_severity",
    }
    assert _pyi_class_property_names(stub_path, "DriftReport") == {
        "feature_results",
        "overall_drift_detected",
        "n_features_drifted",
        "prediction_drift_psi",
        "prediction_drift_detected",
    }
    assert _pyi_class_property_names(stub_path, "PerformanceDriftResult") == {
        "time_periods",
        "c_indices",
        "calibration_slopes",
        "drift_detected",
        "c_index_change",
        "recommendation",
    }
    assert _pyi_class_property_names(stub_path, "SubgroupPerformance") == {
        "subgroup_name",
        "n_samples",
        "c_index",
        "event_rate",
    }
    assert _pyi_class_property_names(stub_path, "ModelCard") == {
        "model_name",
        "model_type",
        "version",
        "description",
        "intended_use",
        "limitations",
        "training_data_description",
        "n_training_samples",
        "n_events",
        "feature_names",
        "overall_performance",
        "subgroup_performance",
        "ethical_considerations",
        "caveats",
    }
    assert _pyi_class_property_names(stub_path, "FairnessAuditResult") == {
        "protected_attribute",
        "group_names",
        "group_c_indices",
        "group_sizes",
        "max_disparity",
        "passes_threshold",
    }

    assert list(inspect.signature(core.DriftReport.to_summary).parameters) == ["self"]
    assert _pyi_class_method_arg_names(stub_path, "DriftReport", "to_summary") == ["self"]
    assert list(inspect.signature(core.ModelCard.to_markdown).parameters) == ["self"]
    assert _pyi_class_method_arg_names(stub_path, "ModelCard", "to_markdown") == ["self"]
    assert list(inspect.signature(core.ModelCard.to_json).parameters) == ["self"]
    assert _pyi_class_method_arg_names(stub_path, "ModelCard", "to_json") == ["self"]

    config = core.DriftConfig(100, 0.05, 0.05, 5)
    assert config.window_size == 100
    assert config.threshold_psi == pytest.approx(0.05)
    config.n_bins = 4
    assert config.n_bins == 4

    drift = core.detect_drift(
        [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3]],
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]],
        ["x1", "x2"],
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        config,
    )
    assert type(drift).__name__ == "DriftReport"
    assert len(drift.feature_results) == 2
    assert type(drift.feature_results[0]).__name__ == "FeatureDriftResult"
    assert drift.feature_results[0].feature_name == "x1"
    assert math.isfinite(drift.feature_results[0].psi)
    assert "Drift Detection Report" in drift.to_summary()

    predictions = [float(idx % 10) / 10.0 for idx in range(40)]
    time = [float((idx % 8) + 1) for idx in range(40)]
    event = [1 if idx % 3 else 0 for idx in range(40)]
    periods = ["early"] * 20 + ["late"] * 20
    performance = core.monitor_performance(predictions, time, event, periods, 0.05)
    assert type(performance).__name__ == "PerformanceDriftResult"
    assert performance.time_periods == ["early", "late"]
    assert len(performance.c_indices) == len(performance.calibration_slopes) == 2
    assert all(math.isfinite(value) for value in performance.c_indices)

    groups = ["A"] * 20 + ["B"] * 20
    fairness = core.fairness_audit(predictions, time, event, "arm", groups, 0.1)
    assert type(fairness).__name__ == "FairnessAuditResult"
    assert fairness.protected_attribute == "arm"
    assert fairness.group_names == ["A", "B"]
    assert fairness.group_sizes == [20, 20]
    assert math.isfinite(fairness.max_disparity)

    metrics = core.ModelPerformanceMetrics(0.75, 0.15, None, None, None, None, None)
    assert metrics.c_index == pytest.approx(0.75)
    assert metrics.brier_score == pytest.approx(0.15)
    card = core.create_model_card(
        "cox-demo",
        "Cox",
        "1.0",
        "Demo survival model",
        "Risk ranking",
        "Toy data",
        40,
        25,
        ["x1", "x2"],
        metrics,
        None,
        ["Small sample"],
        ["Audit subgroups before deployment"],
        ["Retrain periodically"],
    )
    assert type(card).__name__ == "ModelCard"
    assert card.model_name == "cox-demo"
    assert card.overall_performance.c_index == pytest.approx(0.75)
    assert card.limitations == ["Small sample"]
    assert "cox-demo" in card.to_markdown()
    assert '"model_name":"cox-demo"' in card.to_json()


def test_multistate_and_semimarkov_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "MultiStateConfig",
        "TransitionIntensityResult",
        "MultiStateResult",
        "MarkovMSMResult",
        "IllnessDeathType",
        "IllnessDeathConfig",
        "TransitionHazard",
        "IllnessDeathResult",
        "IllnessDeathPrediction",
        "SojournDistribution",
        "SemiMarkovConfig",
        "SojournTimeParams",
        "SemiMarkovResult",
        "SemiMarkovPrediction",
        "estimate_transition_intensities",
        "fit_multi_state_model",
        "fit_markov_msm",
        "fit_illness_death",
        "predict_illness_death",
        "fit_semi_markov",
        "predict_semi_markov",
    } <= stub_names

    function_args = {
        "estimate_transition_intensities": [
            "entry_time",
            "exit_time",
            "from_state",
            "to_state",
            "event",
            "config",
        ],
        "fit_multi_state_model": [
            "entry_time",
            "exit_time",
            "from_state",
            "to_state",
            "event",
            "eval_times",
            "config",
        ],
        "fit_markov_msm": [
            "entry_time",
            "exit_time",
            "from_state",
            "to_state",
            "event",
            "eval_times",
            "config",
        ],
        "fit_illness_death": [
            "entry_time",
            "transition_time",
            "exit_time",
            "from_state",
            "to_state",
            "covariates",
            "config",
        ],
        "predict_illness_death": [
            "model",
            "current_state",
            "time_in_state",
            "prediction_times",
            "covariates",
        ],
        "fit_semi_markov": ["entry_times", "exit_times", "from_states", "to_states", "config"],
        "predict_semi_markov": ["model", "current_state", "time_in_state", "prediction_times"],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "MultiStateConfig": [
            "self",
            "n_states",
            "state_names",
            "transition_matrix",
            "absorbing_states",
        ],
        "TransitionIntensityResult": [
            "self",
            "intensities",
            "cumulative_intensities",
            "time_points",
            "variance",
            "n_at_risk",
            "n_transitions",
        ],
        "MultiStateResult": [
            "self",
            "state_probabilities",
            "time_points",
            "transition_intensities",
            "restricted_mean_times",
            "sojourn_times",
            "state_occupancy",
        ],
        "MarkovMSMResult": [
            "self",
            "transition_matrix",
            "generator_matrix",
            "stationary_distribution",
            "time_points",
            "state_probabilities",
            "log_likelihood",
        ],
        "IllnessDeathConfig": [
            "self",
            "model_type",
            "state_names",
            "clock_type",
            "max_iter",
            "tol",
            "n_bootstrap",
        ],
        "SemiMarkovConfig": [
            "self",
            "n_states",
            "state_names",
            "sojourn_distributions",
            "absorbing_states",
            "max_iter",
            "tol",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert _pyi_class_annotation_names(stub_path, "IllnessDeathType") == {
        "Progressive",
        "Reversible",
        "MarkovProgressive",
        "SemiMarkovProgressive",
    }
    assert _pyi_class_annotation_names(stub_path, "SojournDistribution") == {
        "Exponential",
        "Weibull",
        "LogNormal",
        "Gamma",
        "GeneralizedGamma",
    }

    assert _pyi_class_property_names(stub_path, "MultiStateConfig") == {
        "n_states",
        "state_names",
        "transition_matrix",
        "absorbing_states",
    }
    assert _pyi_class_property_names(stub_path, "TransitionIntensityResult") == {
        "intensities",
        "cumulative_intensities",
        "time_points",
        "variance",
        "n_at_risk",
        "n_transitions",
    }
    assert _pyi_class_property_names(stub_path, "MultiStateResult") == {
        "state_probabilities",
        "time_points",
        "transition_intensities",
        "restricted_mean_times",
        "sojourn_times",
        "state_occupancy",
    }
    assert _pyi_class_property_names(stub_path, "MarkovMSMResult") == {
        "transition_matrix",
        "generator_matrix",
        "stationary_distribution",
        "time_points",
        "state_probabilities",
        "log_likelihood",
    }
    assert _pyi_class_property_names(stub_path, "IllnessDeathConfig") == {
        "model_type",
        "state_names",
        "clock_type",
        "max_iter",
        "tol",
        "n_bootstrap",
    }
    assert _pyi_class_property_names(stub_path, "TransitionHazard") == {
        "from_state",
        "to_state",
        "coefficient",
        "se",
        "hazard_ratio",
        "ci_lower",
        "ci_upper",
        "p_value",
        "baseline_hazard",
        "baseline_times",
    }
    assert _pyi_class_property_names(stub_path, "IllnessDeathResult") == {
        "transition_hazards",
        "state_occupation_probs",
        "time_points",
        "cumulative_incidence",
        "sojourn_times",
        "log_likelihood",
        "aic",
        "bic",
        "n_transitions",
        "model_type",
    }
    assert _pyi_class_property_names(stub_path, "IllnessDeathPrediction") == {
        "state_probs",
        "time_points",
        "survival_prob",
        "illness_free_survival",
        "death_prob",
    }
    assert _pyi_class_property_names(stub_path, "SemiMarkovConfig") == {
        "n_states",
        "state_names",
        "sojourn_distributions",
        "absorbing_states",
        "max_iter",
        "tol",
    }
    assert _pyi_class_property_names(stub_path, "SojournTimeParams") == {
        "distribution",
        "shape",
        "scale",
        "location",
        "mean",
        "variance",
        "median",
    }
    assert _pyi_class_property_names(stub_path, "SemiMarkovResult") == {
        "transition_probs",
        "sojourn_params",
        "state_occupation_probs",
        "time_points",
        "mean_sojourn_times",
        "n_transitions",
        "log_likelihood",
        "aic",
        "bic",
    }
    assert _pyi_class_property_names(stub_path, "SemiMarkovPrediction") == {
        "state_probs",
        "time_points",
        "expected_sojourn",
        "transition_hazards",
    }

    class_method_args = {
        "IllnessDeathResult": {
            "get_survival_probability": ["self", "time"],
            "get_illness_probability": ["self", "time"],
            "get_death_probability": ["self", "time"],
        },
        "SemiMarkovResult": {
            "get_transition_prob": ["self", "from_state", "to_state"],
            "predict_state_at_time": ["self", "time"],
        },
    }
    for class_name, methods in class_method_args.items():
        runtime_class = getattr(core, class_name)
        for method_name, expected_args in methods.items():
            assert (
                list(inspect.signature(getattr(runtime_class, method_name)).parameters)
                == expected_args
            )
            assert _pyi_class_method_arg_names(stub_path, class_name, method_name) == expected_args

    entry_time = [0.0, 0.0, 0.0, 0.0]
    exit_time = [1.0, 2.0, 1.5, 2.5]
    from_state = [0, 0, 0, 0]
    to_state = [1, 1, 1, 1]
    event = [1, 1, 1, 0]
    eval_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    config = core.MultiStateConfig(2)

    intensities = core.estimate_transition_intensities(
        entry_time,
        exit_time,
        from_state,
        to_state,
        event,
        config,
    )
    multistate = core.fit_multi_state_model(
        entry_time,
        exit_time,
        from_state,
        to_state,
        event,
        eval_times,
        config,
    )
    msm = core.fit_markov_msm(
        entry_time, exit_time, from_state, to_state, event, eval_times, config
    )

    assert type(intensities).__name__ == "TransitionIntensityResult"
    assert "0->1" in intensities.intensities
    assert type(multistate).__name__ == "MultiStateResult"
    assert len(multistate.state_probabilities) == len(eval_times)
    assert multistate.state_probabilities[0][0] == pytest.approx(1.0)
    assert type(msm).__name__ == "MarkovMSMResult"
    assert len(msm.generator_matrix) == 2
    assert len(msm.state_probabilities) == len(eval_times)

    illness_model = core.fit_illness_death(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0 + 5e-10],
        [2.0, 2.0, 2.0],
        [0, 0, 0],
        [1, 2, 1],
        [[10.0], [20.0], [30.0]],
        None,
    )
    illness_prediction = core.predict_illness_death(illness_model, 0, 0.0, [0.5, 1.0], [0.1])

    assert type(illness_model).__name__ == "IllnessDeathResult"
    assert len(illness_model.transition_hazards) == 3
    assert illness_model.transition_hazards[0].coefficient == pytest.approx(20.0)
    assert illness_model.get_survival_probability(0.5) > 0.0
    assert type(illness_prediction).__name__ == "IllnessDeathPrediction"
    assert illness_prediction.time_points == pytest.approx([0.5, 1.0])

    semi_config = core.SemiMarkovConfig(3)
    semi_model = core.fit_semi_markov(
        [0.0, 1.0, 2.0, 0.5],
        [1.0, 2.0, 3.0, 1.5],
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        semi_config,
    )
    semi_prediction = core.predict_semi_markov(semi_model, 0, 0.5, [0.5, 1.0])

    assert type(semi_model).__name__ == "SemiMarkovResult"
    assert len(semi_model.sojourn_params) == 3
    assert semi_model.get_transition_prob(0, 1) == pytest.approx(1.0)
    assert len(semi_model.predict_state_at_time(1.0)) == 3
    assert type(semi_prediction).__name__ == "SemiMarkovPrediction"
    assert semi_prediction.time_points == pytest.approx([0.5, 1.0])


def test_statefig_calibration_and_imputation_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "StateFigData",
        "MultiTimeCalibrationResult",
        "ImputationMethod",
        "MultipleImputationResult",
        "statefig",
        "statefig_matplotlib_code",
        "statefig_transition_matrix",
        "statefig_validate",
        "multi_time_calibration",
        "analyze_missing_pattern",
        "multiple_imputation_survival",
        "PatternMixtureResult",
        "SensitivityAnalysisType",
        "pattern_mixture_model",
        "sensitivity_analysis",
        "tipping_point_analysis",
    } <= stub_names

    function_args = {
        "statefig": ["states", "transitions", "layout"],
        "statefig_matplotlib_code": ["data"],
        "statefig_transition_matrix": ["data"],
        "statefig_validate": ["data", "allowed_transitions"],
        "multi_time_calibration": [
            "time",
            "status",
            "survival_predictions",
            "prediction_times",
            "n_groups",
        ],
        "analyze_missing_pattern": [
            "time",
            "status",
            "x",
            "n_obs",
            "n_vars",
            "missing_indicators",
            "n_imputations",
        ],
        "multiple_imputation_survival": [
            "time",
            "status",
            "x",
            "n_obs",
            "n_vars",
            "missing_indicators",
            "n_imputations",
            "method",
            "max_iter",
            "seed",
        ],
        "pattern_mixture_model": [
            "time",
            "status",
            "x",
            "n_obs",
            "n_vars",
            "dropout_pattern",
            "dropout_time",
        ],
        "sensitivity_analysis": [
            "time",
            "status",
            "x",
            "n_obs",
            "n_vars",
            "dropout_pattern",
            "delta_values",
            "analysis_type",
        ],
        "tipping_point_analysis": [
            "time",
            "status",
            "x",
            "n_obs",
            "n_vars",
            "dropout_pattern",
            "coef_index",
            "target_value",
            "delta_range",
            "n_steps",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "MultiTimeCalibrationResult": [
            "self",
            "time_points",
            "brier_scores",
            "integrated_brier",
            "calibration_slopes",
            "calibration_intercepts",
            "ici_values",
            "mean_ici",
            "mean_slope",
        ],
        "ImputationMethod": ["self", "name"],
        "SensitivityAnalysisType": ["self", "name"],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert _pyi_class_annotation_names(stub_path, "ImputationMethod") == {
        "PMM",
        "Regression",
        "MICE",
        "KNN",
    }
    assert _pyi_class_annotation_names(stub_path, "SensitivityAnalysisType") == {
        "TiltingModel",
        "SelectionModel",
        "DeltaAdjustment",
    }
    assert _pyi_class_property_names(stub_path, "StateFigData") == {
        "states",
        "positions",
        "edges",
        "box_sizes",
        "layout",
    }
    assert _pyi_class_property_names(stub_path, "MultiTimeCalibrationResult") == {
        "time_points",
        "brier_scores",
        "integrated_brier",
        "calibration_slopes",
        "calibration_intercepts",
        "ici_values",
        "mean_ici",
        "mean_slope",
    }
    assert _pyi_class_property_names(stub_path, "MultipleImputationResult") == {
        "pooled_coefficients",
        "pooled_se",
        "pooled_ci_lower",
        "pooled_ci_upper",
        "within_variance",
        "between_variance",
        "total_variance",
        "fraction_missing_info",
        "relative_efficiency",
        "n_imputations",
    }
    assert _pyi_class_property_names(stub_path, "PatternMixtureResult") == {
        "pattern_coefficients",
        "pattern_se",
        "pattern_weights",
        "averaged_coefficients",
        "averaged_se",
        "averaged_ci_lower",
        "averaged_ci_upper",
        "n_patterns",
        "pattern_sizes",
    }

    state_data = core.statefig(
        ["Healthy", "Illness", "Death"],
        {
            ("Healthy", "Illness"): 2,
            ("Illness", "Death"): 1,
            ("Healthy", "Death"): 1,
        },
        [2, 1],
    )
    matrix = core.statefig_transition_matrix(state_data)
    issues = core.statefig_validate(
        state_data,
        {
            ("Healthy", "Illness"): True,
            ("Illness", "Death"): True,
            ("Healthy", "Death"): False,
        },
    )

    assert type(state_data).__name__ == "StateFigData"
    assert state_data.states == ["Healthy", "Illness", "Death"]
    assert state_data.layout == [2, 1]
    assert matrix == [[0, 2, 1], [0, 0, 1], [0, 0, 0]]
    assert issues == ["Invalid transition: Healthy -> Death (1 occurrences)"]
    assert "matplotlib.pyplot" in core.statefig_matplotlib_code(state_data)

    calibration = core.multi_time_calibration(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1, 1, 0, 1, 0, 1],
        [
            [0.95, 0.9],
            [0.9, 0.8],
            [0.98, 0.95],
            [0.85, 0.7],
            [0.99, 0.97],
            [0.8, 0.6],
        ],
        [2.0, 4.0],
        2,
    )

    assert type(calibration).__name__ == "MultiTimeCalibrationResult"
    assert calibration.time_points == pytest.approx([2.0, 4.0])
    assert len(calibration.brier_scores) == 2
    assert calibration.integrated_brier > 0.0
    assert calibration.mean_ici > 0.0

    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    status = [1, 0, 1, 0, 1]
    x = [1.0, 0.5, 0.0, 1.2, 1.0, 0.0, 0.0, 0.8, 1.0, 1.5]
    missing = [False, False, False, True, False, False, False, False, False, True]

    missing_pct, patterns, monotone = core.analyze_missing_pattern(
        time,
        status,
        x,
        5,
        2,
        missing,
        3,
    )
    imputed = core.multiple_imputation_survival(
        time,
        status,
        x,
        5,
        2,
        missing,
        3,
        core.ImputationMethod.PMM,
        10,
        42,
    )

    assert missing_pct == pytest.approx([0.0, 40.0])
    assert patterns == ["00", "01"]
    assert monotone is True
    assert type(imputed).__name__ == "MultipleImputationResult"
    assert imputed.n_imputations == 3
    assert len(imputed.pooled_coefficients) == 2
    assert len(imputed.pooled_se) == 2

    pattern_time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    pattern_status = [1, 0, 1, 0, 1, 0, 1, 0]
    pattern_x = [
        1.0,
        0.5,
        0.0,
        1.2,
        1.0,
        0.0,
        0.0,
        0.8,
        1.0,
        1.5,
        0.0,
        0.3,
        1.0,
        0.9,
        0.0,
        1.1,
    ]
    dropout = [0, 0, 0, 1, 1, 1, 2, 2]
    pattern_mixture = core.pattern_mixture_model(
        pattern_time,
        pattern_status,
        pattern_x,
        8,
        2,
        dropout,
        None,
    )

    sensitivity_time = [1.0, 2.0, 3.0, 4.0, 5.0]
    sensitivity_status = [1, 0, 1, 0, 1]
    sensitivity_x = [1.0, 0.0, 1.0, 0.0, 1.0]
    sensitivity_dropout = [0, 0, 1, 1, 0]
    delta_values = [-0.5, 0.0, 0.5]
    sensitivity = core.sensitivity_analysis(
        sensitivity_time,
        sensitivity_status,
        sensitivity_x,
        5,
        1,
        sensitivity_dropout,
        delta_values,
        core.SensitivityAnalysisType.DeltaAdjustment,
    )
    tipping_point = core.tipping_point_analysis(
        sensitivity_time,
        sensitivity_status,
        sensitivity_x,
        5,
        1,
        sensitivity_dropout,
        0,
        0.0,
        (-1.0, 1.0),
        10,
    )

    assert str(core.SensitivityAnalysisType("delta")) == "SensitivityAnalysisType.DeltaAdjustment"
    assert type(pattern_mixture).__name__ == "PatternMixtureResult"
    assert pattern_mixture.n_patterns == 3
    assert pattern_mixture.pattern_sizes == [3, 3, 2]
    assert pattern_mixture.pattern_weights == pytest.approx([0.375, 0.375, 0.25])
    assert pattern_mixture.averaged_coefficients == pytest.approx(
        [14.231618646664103, 11.442494681988526]
    )
    assert pattern_mixture.averaged_se == [float("inf"), float("inf")]
    assert pattern_mixture.averaged_ci_lower == [float("-inf"), float("-inf")]
    assert pattern_mixture.averaged_ci_upper == [float("inf"), float("inf")]
    assert [delta for delta, _, _ in sensitivity] == delta_values
    assert [coefficients[0] for _, coefficients, _ in sensitivity] == pytest.approx(
        [1.4104925426524506, 23.76952743812799, 2.055807228850012]
    )
    assert tipping_point is None


def test_d_calibration_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "DCalibrationResult",
        "OneCalibrationResult",
        "CalibrationPlotData",
        "BrierCalibrationResult",
        "SmoothedCalibrationCurve",
        "d_calibration",
        "one_calibration",
        "calibration_plot",
        "brier_calibration",
        "smoothed_calibration",
    } <= stub_names

    function_args = {
        "d_calibration": ["survival_probs", "status", "n_bins"],
        "one_calibration": [
            "time",
            "status",
            "predicted_survival_at_t",
            "time_point",
            "n_groups",
        ],
        "calibration_plot": [
            "time",
            "status",
            "predicted_survival_at_t",
            "time_point",
            "n_groups",
        ],
        "brier_calibration": [
            "time",
            "status",
            "predicted_survival_at_t",
            "time_point",
            "n_groups",
        ],
        "smoothed_calibration": [
            "time",
            "status",
            "predicted_survival_at_t",
            "time_point",
            "n_grid_points",
            "bandwidth",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "DCalibrationResult": [
            "self",
            "statistic",
            "p_value",
            "degrees_of_freedom",
            "n_bins",
            "observed_counts",
            "expected_counts",
            "bin_edges",
            "n_events",
            "is_calibrated",
        ],
        "OneCalibrationResult": [
            "self",
            "time_point",
            "statistic",
            "p_value",
            "degrees_of_freedom",
            "n_groups",
            "predicted_survival",
            "observed_survival",
            "n_per_group",
            "n_events_per_group",
            "is_calibrated",
        ],
        "CalibrationPlotData": [
            "self",
            "predicted",
            "observed",
            "n_per_group",
            "ci_lower",
            "ci_upper",
            "ici",
            "e50",
            "e90",
            "emax",
        ],
        "BrierCalibrationResult": [
            "self",
            "time_point",
            "brier_score",
            "calibration_slope",
            "calibration_intercept",
            "ici",
            "e50",
            "e90",
            "emax",
            "predicted",
            "observed",
            "ci_lower",
            "ci_upper",
            "n_per_group",
        ],
        "SmoothedCalibrationCurve": [
            "self",
            "predicted_grid",
            "smoothed_observed",
            "ci_lower",
            "ci_upper",
            "bandwidth",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert _pyi_class_property_names(stub_path, "DCalibrationResult") == {
        "statistic",
        "p_value",
        "degrees_of_freedom",
        "n_bins",
        "observed_counts",
        "expected_counts",
        "bin_edges",
        "n_events",
        "is_calibrated",
    }
    assert _pyi_class_property_names(stub_path, "OneCalibrationResult") == {
        "time_point",
        "statistic",
        "p_value",
        "degrees_of_freedom",
        "n_groups",
        "predicted_survival",
        "observed_survival",
        "n_per_group",
        "n_events_per_group",
        "is_calibrated",
    }
    assert _pyi_class_property_names(stub_path, "CalibrationPlotData") == {
        "predicted",
        "observed",
        "n_per_group",
        "ci_lower",
        "ci_upper",
        "ici",
        "e50",
        "e90",
        "emax",
    }
    assert _pyi_class_property_names(stub_path, "BrierCalibrationResult") == {
        "time_point",
        "brier_score",
        "calibration_slope",
        "calibration_intercept",
        "ici",
        "e50",
        "e90",
        "emax",
        "predicted",
        "observed",
        "ci_lower",
        "ci_upper",
        "n_per_group",
    }
    assert _pyi_class_property_names(stub_path, "SmoothedCalibrationCurve") == {
        "predicted_grid",
        "smoothed_observed",
        "ci_lower",
        "ci_upper",
        "bandwidth",
    }

    time = list(range(1, 21))
    status = [1] * 20
    predicted = [0.99 - i * 0.03 for i in range(20)]

    d_cal = core.d_calibration(
        survival_probs=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] * 2,
        status=[1] * 20,
        n_bins=5,
    )
    one = core.one_calibration(time, status, predicted, 10.0, 4)
    plot = core.calibration_plot(time, status, predicted, 10.0, 4)
    brier = core.brier_calibration(time, status, predicted, 10.0, 4)
    smooth = core.smoothed_calibration(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1, 1, 0, 1, 0, 1],
        [0.95, 0.9, 0.98, 0.85, 0.99, 0.8],
        4.0,
        10,
        0.1,
    )

    assert type(d_cal).__name__ == "DCalibrationResult"
    assert d_cal.n_bins == 5
    assert d_cal.observed_counts == [4, 4, 4, 4, 4]
    assert d_cal.is_calibrated is True
    assert type(one).__name__ == "OneCalibrationResult"
    assert one.time_point == pytest.approx(10.0)
    assert one.n_per_group == [5, 5, 5, 5]
    assert one.is_calibrated is False
    assert type(plot).__name__ == "CalibrationPlotData"
    assert plot.predicted == pytest.approx([0.4800000000000001, 0.63, 0.78, 0.93])
    assert plot.ici == pytest.approx(0.65)
    assert type(brier).__name__ == "BrierCalibrationResult"
    assert brier.brier_score == pytest.approx(0.47195)
    assert brier.calibration_slope == pytest.approx(-2.666666666666667)
    assert type(smooth).__name__ == "SmoothedCalibrationCurve"
    assert len(smooth.predicted_grid) == 10
    assert smooth.bandwidth == pytest.approx(0.1)


def test_validation_reporting_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "KaplanMeierPlotData",
        "ForestPlotData",
        "CalibrationCurveData",
        "SurvivalReport",
        "ROCPlotData",
        "km_plot_data",
        "forest_plot_data",
        "calibration_plot_data",
        "generate_survival_report",
        "roc_plot_data",
    } <= stub_names

    function_args = {
        "km_plot_data": ["time", "event", "confidence_level", "group_name"],
        "forest_plot_data": [
            "variable_names",
            "coefficients",
            "standard_errors",
            "confidence_level",
        ],
        "calibration_plot_data": ["predicted", "observed", "n_bins"],
        "generate_survival_report": ["title", "time", "event", "landmark_times"],
        "roc_plot_data": ["scores", "labels"],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    assert _pyi_class_property_names(stub_path, "KaplanMeierPlotData") == {
        "time_points",
        "survival_prob",
        "lower_ci",
        "upper_ci",
        "at_risk",
        "n_events",
        "n_censored",
        "group_name",
    }
    assert _pyi_class_property_names(stub_path, "ForestPlotData") == {
        "variable_names",
        "hazard_ratios",
        "lower_ci",
        "upper_ci",
        "p_values",
        "weights",
    }
    assert _pyi_class_property_names(stub_path, "CalibrationCurveData") == {
        "predicted_prob",
        "observed_prob",
        "n_per_bin",
        "bin_boundaries",
        "hosmer_lemeshow_stat",
        "hosmer_lemeshow_p",
    }
    assert _pyi_class_property_names(stub_path, "SurvivalReport") == {
        "title",
        "n_subjects",
        "n_events",
        "median_survival",
        "median_ci",
        "survival_rates",
        "rmst",
        "hazard_ratio",
        "hazard_ratio_ci",
        "logrank_p",
    }
    assert _pyi_class_property_names(stub_path, "ROCPlotData") == {
        "fpr",
        "tpr",
        "thresholds",
        "auc",
    }

    class_method_args = {
        "KaplanMeierPlotData": {"to_step_data": ["self"]},
        "ForestPlotData": {"significant_at": ["self", "alpha"]},
        "SurvivalReport": {"to_markdown": ["self"], "to_latex": ["self"]},
        "ROCPlotData": {"optimal_threshold": ["self", "method"]},
    }
    for class_name, methods in class_method_args.items():
        runtime_class = getattr(core, class_name)
        for method_name, expected_args in methods.items():
            assert (
                list(inspect.signature(getattr(runtime_class, method_name)).parameters)
                == expected_args
            )
            assert _pyi_class_method_arg_names(stub_path, class_name, method_name) == expected_args

    km = core.km_plot_data([1.0, 2.0, 3.0, 4.0, 5.0], [1, 0, 1, 0, 1], 0.95, "Test")
    forest = core.forest_plot_data(["age", "trt"], [0.2, -0.3], [0.1, 0.15], 0.95)
    calibration_curve = core.calibration_plot_data(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0, 0, 0, 1, 0, 1, 1, 1],
        4,
    )
    report = core.generate_survival_report(
        "Test Report",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1, 0, 1, 0, 1],
        [2.0, 4.0],
    )
    roc = core.roc_plot_data([0.9, 0.8, 0.7, 0.6, 0.3, 0.2], [1, 1, 1, 0, 0, 0])

    assert type(km).__name__ == "KaplanMeierPlotData"
    assert km.group_name == "Test"
    assert km.time_points == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert km.at_risk == [5, 5, 4, 3, 2, 1]
    step_x, step_y = km.to_step_data()
    assert len(step_x) == len(step_y)
    assert step_y[0] == pytest.approx(1.0)

    assert type(forest).__name__ == "ForestPlotData"
    assert forest.hazard_ratios == pytest.approx([1.2214027581601699, 0.7408182206817179])
    assert forest.significant_at(0.2) == [True, True]
    assert all(value > 0.0 for value in forest.lower_ci)

    assert type(calibration_curve).__name__ == "CalibrationCurveData"
    assert calibration_curve.predicted_prob == pytest.approx([0.15, 0.35, 0.55, 0.75])
    assert calibration_curve.observed_prob == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert calibration_curve.n_per_bin == [2, 2, 2, 2]
    assert len(calibration_curve.bin_boundaries) == 5

    assert type(report).__name__ == "SurvivalReport"
    assert report.title == "Test Report"
    assert report.n_subjects == 5
    assert report.n_events == 3
    assert report.median_survival == pytest.approx(5.0)
    assert report.rmst == pytest.approx(3.666666666666667)
    assert "# Test Report" in report.to_markdown()
    assert "\\section{Test Report}" in report.to_latex()

    assert type(roc).__name__ == "ROCPlotData"
    assert roc.auc == pytest.approx(1.0)
    assert roc.fpr[0] == pytest.approx(0.0)
    assert roc.tpr[-1] == pytest.approx(1.0)
    assert roc.optimal_threshold("youden") == pytest.approx(0.7)


def test_decision_curve_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "DecisionCurveResult",
        "ClinicalUtilityResult",
        "ModelComparisonResult",
        "decision_curve_analysis",
        "clinical_utility_at_threshold",
        "compare_decision_curves",
    } <= stub_names

    function_args = {
        "decision_curve_analysis": [
            "predicted_risk",
            "time",
            "event",
            "time_horizon",
            "thresholds",
        ],
        "clinical_utility_at_threshold": [
            "predicted_risk",
            "time",
            "event",
            "time_horizon",
            "threshold",
        ],
        "compare_decision_curves": [
            "model_predictions",
            "model_names",
            "time",
            "event",
            "time_horizon",
            "thresholds",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    assert _pyi_class_property_names(stub_path, "DecisionCurveResult") == {
        "thresholds",
        "net_benefit",
        "net_benefit_all",
        "net_benefit_none",
        "interventions_avoided",
    }
    assert _pyi_class_property_names(stub_path, "ClinicalUtilityResult") == {
        "threshold",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "nnt",
        "net_benefit",
    }
    assert _pyi_class_property_names(stub_path, "ModelComparisonResult") == {
        "model_names",
        "net_benefit_difference",
        "thresholds",
        "best_model_per_threshold",
    }

    class_method_args = {
        "DecisionCurveResult": {
            "optimal_threshold": ["self"],
            "area_under_curve": ["self"],
        },
    }
    for class_name, methods in class_method_args.items():
        runtime_class = getattr(core, class_name)
        for method_name, expected_args in methods.items():
            assert (
                list(inspect.signature(getattr(runtime_class, method_name)).parameters)
                == expected_args
            )
            assert _pyi_class_method_arg_names(stub_path, class_name, method_name) == expected_args

    result = core.decision_curve_analysis(
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0, 1, 0, 1, 1],
        3.0,
        [0.25, 0.5, 0.75],
    )
    clinical = core.clinical_utility_at_threshold(
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0, 1, 0, 1, 1],
        3.0,
        0.5,
    )
    comparison = core.compare_decision_curves(
        [
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [0.2, 0.2, 0.4, 0.6, 0.8],
        ],
        ["m1", "m2"],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0, 1, 0, 1, 1],
        3.0,
        [0.25, 0.5, 0.75],
    )

    assert type(result).__name__ == "DecisionCurveResult"
    assert result.thresholds == pytest.approx([0.25, 0.5, 0.75])
    assert result.net_benefit_none == pytest.approx([0.0, 0.0, 0.0])
    assert result.optimal_threshold() == pytest.approx(0.25)
    assert result.area_under_curve() == pytest.approx(0.0)
    assert type(clinical).__name__ == "ClinicalUtilityResult"
    assert clinical.threshold == pytest.approx(0.5)
    assert clinical.specificity == pytest.approx(0.25)
    assert clinical.net_benefit == pytest.approx(-0.6)
    assert type(comparison).__name__ == "ModelComparisonResult"
    assert comparison.model_names == ["m1", "m2"]
    assert comparison.best_model_per_threshold == ["m1", "m2", "m2"]
    assert len(comparison.net_benefit_difference) == 2
    assert len(comparison.net_benefit_difference[0]) == 2


def test_validation_summary_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "RiskStratificationResult",
        "MedianSurvivalResult",
        "CumulativeIncidenceResult",
        "NNTResult",
        "risk_stratification",
        "survival_quantile",
        "cumulative_incidence",
        "number_needed_to_treat",
    } <= stub_names

    function_args = {
        "risk_stratification": ["risk_scores", "events", "n_groups"],
        "survival_quantile": ["time", "status", "quantile", "confidence_level"],
        "cumulative_incidence": ["time", "status"],
        "number_needed_to_treat": [
            "time",
            "status",
            "group",
            "time_horizon",
            "confidence_level",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "RiskStratificationResult": [
            "self",
            "risk_groups",
            "cutpoints",
            "group_sizes",
            "group_event_rates",
            "group_median_risk",
        ],
        "MedianSurvivalResult": ["self", "median", "ci_lower", "ci_upper", "quantile"],
        "CumulativeIncidenceResult": ["self", "time", "cif", "variance", "event_types", "n_risk"],
        "NNTResult": [
            "self",
            "nnt",
            "nnt_ci_lower",
            "nnt_ci_upper",
            "absolute_risk_reduction",
            "arr_ci_lower",
            "arr_ci_upper",
            "time_horizon",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert _pyi_class_property_names(stub_path, "RiskStratificationResult") == {
        "risk_groups",
        "cutpoints",
        "group_sizes",
        "group_event_rates",
        "group_median_risk",
    }
    assert _pyi_class_property_names(stub_path, "MedianSurvivalResult") == {
        "median",
        "ci_lower",
        "ci_upper",
        "quantile",
    }
    assert _pyi_class_property_names(stub_path, "CumulativeIncidenceResult") == {
        "time",
        "cif",
        "variance",
        "event_types",
        "n_risk",
    }
    assert _pyi_class_property_names(stub_path, "NNTResult") == {
        "nnt",
        "nnt_ci_lower",
        "nnt_ci_upper",
        "absolute_risk_reduction",
        "arr_ci_lower",
        "arr_ci_upper",
        "time_horizon",
    }

    stratification = core.risk_stratification(
        [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95],
        [0, 0, 0, 1, 0, 1, 1, 1],
        3,
    )
    quantile = core.survival_quantile(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1, 1, 1, 1, 1, 1],
        None,
        None,
    )
    incidence = core.cumulative_incidence(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1, 2, 1, 0, 2],
    )
    nnt = core.number_needed_to_treat(
        [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
        [1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        5.0,
        None,
    )

    assert type(stratification).__name__ == "RiskStratificationResult"
    assert stratification.cutpoints == pytest.approx([0.3, 0.8])
    assert stratification.group_sizes == [2, 3, 3]
    assert stratification.group_event_rates == pytest.approx([0.0, 1.0 / 3.0, 1.0])
    assert stratification.group_median_risk == pytest.approx([0.2, 0.4, 0.9])
    assert stratification.risk_groups == [0, 0, 1, 1, 1, 2, 2, 2]
    assert type(quantile).__name__ == "MedianSurvivalResult"
    assert quantile.median == pytest.approx(3.0)
    assert quantile.ci_lower == pytest.approx(5.0)
    assert quantile.ci_upper == pytest.approx(2.0)
    assert quantile.quantile == pytest.approx(0.5)
    assert type(incidence).__name__ == "CumulativeIncidenceResult"
    assert incidence.time == pytest.approx([1.0, 2.0, 3.0, 5.0])
    assert incidence.event_types == [1, 2]
    assert incidence.n_risk == [5, 4, 3, 1]
    assert incidence.cif[0] == pytest.approx([0.2, 0.2, 0.4, 0.4])
    assert incidence.cif[1] == pytest.approx([0.0, 0.2, 0.2, 0.6000000000000001])
    assert type(nnt).__name__ == "NNTResult"
    assert nnt.nnt == pytest.approx(-4.0)
    assert nnt.absolute_risk_reduction == pytest.approx(-0.25)
    assert nnt.time_horizon == pytest.approx(5.0)


def test_recurrent_event_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "GapTimeResult",
        "PWPTimescale",
        "PWPConfig",
        "PWPResult",
        "AndersonGillResult",
        "WLWConfig",
        "WLWResult",
        "NegativeBinomialFrailtyConfig",
        "NegativeBinomialFrailtyResult",
        "MarginalMethod",
        "MarginalModelResult",
        "FrailtyDistribution",
        "JointFrailtyResult",
        "gap_time_model",
        "pwp_gap_time",
        "pwp_model",
        "anderson_gill_model",
        "wlw_model",
        "negative_binomial_frailty",
        "marginal_recurrent_model",
        "andersen_gill",
        "wei_lin_weissfeld",
        "joint_frailty_model",
    } <= stub_names

    function_args = {
        "gap_time_model": [
            "subject_id",
            "start_time",
            "stop_time",
            "event_status",
            "x",
            "n_obs",
            "n_vars",
            "max_iter",
            "tol",
        ],
        "pwp_gap_time": [
            "subject_id",
            "event_time",
            "event_status",
            "x",
            "n_obs",
            "n_vars",
            "stratify_by_event_number",
        ],
        "pwp_model": ["id", "start", "stop", "event", "event_number", "covariates", "config"],
        "anderson_gill_model": ["id", "start", "stop", "event", "covariates", "max_iter", "tol"],
        "wlw_model": ["id", "time", "event", "stratum", "covariates", "config"],
        "negative_binomial_frailty": ["id", "time", "event", "covariates", "offset", "config"],
        "marginal_recurrent_model": [
            "subject_id",
            "start_time",
            "stop_time",
            "event_status",
            "x",
            "n_obs",
            "n_vars",
            "method",
            "max_iter",
            "tol",
        ],
        "andersen_gill": [
            "subject_id",
            "start_time",
            "stop_time",
            "event_status",
            "x",
            "n_obs",
            "n_vars",
        ],
        "wei_lin_weissfeld": ["subject_id", "event_time", "event_status", "x", "n_obs", "n_vars"],
        "joint_frailty_model": [
            "subject_id",
            "rec_start",
            "rec_stop",
            "rec_status",
            "x_recurrent",
            "n_rec_obs",
            "n_rec_vars",
            "term_time",
            "term_status",
            "x_terminal",
            "n_subjects",
            "n_term_vars",
            "frailty_dist",
            "max_iter",
            "tol",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "PWPTimescale": ["self", "name"],
        "PWPConfig": [
            "self",
            "timescale",
            "max_iter",
            "tol",
            "stratify_by_event",
            "robust_variance",
        ],
        "WLWConfig": ["self", "max_iter", "tol", "robust_variance", "common_baseline"],
        "NegativeBinomialFrailtyConfig": ["self", "max_iter", "tol", "em_max_iter"],
        "MarginalMethod": ["self", "name"],
        "FrailtyDistribution": ["self", "name"],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert _pyi_class_property_names(stub_path, "GapTimeResult") == {
        "coefficients",
        "std_errors",
        "hazard_ratios",
        "hr_ci_lower",
        "hr_ci_upper",
        "log_likelihood",
        "aic",
        "bic",
        "n_events",
        "n_subjects",
        "baseline_hazard",
        "baseline_times",
    }
    assert _pyi_class_property_names(stub_path, "PWPConfig") == {
        "timescale",
        "max_iter",
        "tol",
        "stratify_by_event",
        "robust_variance",
    }
    assert _pyi_class_property_names(stub_path, "PWPResult") == {
        "coef",
        "std_errors",
        "robust_std_errors",
        "z_scores",
        "p_values",
        "hazard_ratios",
        "hr_lower",
        "hr_upper",
        "log_likelihood",
        "n_events",
        "n_subjects",
        "n_iter",
        "converged",
        "event_specific_coef",
        "baseline_cumhaz",
    }
    assert _pyi_class_property_names(stub_path, "AndersonGillResult") == {
        "coef",
        "std_errors",
        "robust_std_errors",
        "z_scores",
        "p_values",
        "hazard_ratios",
        "hr_lower",
        "hr_upper",
        "log_likelihood",
        "n_events",
        "n_subjects",
        "n_iter",
        "converged",
        "mean_event_rate",
    }
    assert _pyi_class_annotation_names(stub_path, "WLWConfig") == {
        "max_iter",
        "tol",
        "robust_variance",
        "common_baseline",
    }
    assert _pyi_class_property_names(stub_path, "WLWResult") == {
        "coef",
        "std_errors",
        "robust_std_errors",
        "z_scores",
        "p_values",
        "hazard_ratios",
        "hr_lower",
        "hr_upper",
        "log_likelihood",
        "n_events",
        "n_subjects",
        "n_strata",
        "n_iter",
        "converged",
        "stratum_coef",
        "global_test_stat",
        "global_test_pvalue",
    }
    assert _pyi_class_annotation_names(stub_path, "NegativeBinomialFrailtyConfig") == {
        "max_iter",
        "tol",
        "em_max_iter",
    }
    assert _pyi_class_property_names(stub_path, "NegativeBinomialFrailtyResult") == {
        "coef",
        "std_errors",
        "z_scores",
        "p_values",
        "rate_ratios",
        "rr_lower",
        "rr_upper",
        "theta",
        "theta_se",
        "frailty_variance",
        "log_likelihood",
        "aic",
        "bic",
        "n_events",
        "n_subjects",
        "n_iter",
        "converged",
        "frailty_estimates",
    }
    assert _pyi_class_property_names(stub_path, "MarginalModelResult") == {
        "coefficients",
        "robust_se",
        "naive_se",
        "hazard_ratios",
        "hr_ci_lower",
        "hr_ci_upper",
        "log_likelihood",
        "score_test",
        "wald_test",
        "n_events",
        "n_subjects",
        "mean_events_per_subject",
    }
    assert _pyi_class_property_names(stub_path, "JointFrailtyResult") == {
        "recurrent_coef",
        "recurrent_se",
        "terminal_coef",
        "terminal_se",
        "frailty_variance",
        "alpha",
        "frailty_values",
        "log_likelihood",
        "aic",
        "bic",
        "n_iter",
        "converged",
        "n_recurrent_events",
        "n_terminal_events",
        "n_subjects",
    }

    subject_id = [0, 0, 1, 1, 2]
    start = [0.0, 5.0, 0.0, 3.0, 0.0]
    stop = [5.0, 10.0, 3.0, 8.0, 7.0]
    event = [1, 1, 1, 0, 1]
    covariates = [1.0, 0.5, 1.0, 0.3, 0.0]

    gap = core.gap_time_model(subject_id, start, stop, event, covariates, 5, 1, 50, 1e-4)
    pwp_gap = core.pwp_gap_time(subject_id, stop, event, covariates, 5, 1, False)
    pwp_config = core.PWPConfig(core.PWPTimescale("gap"), 50, 1e-4, True, True)
    pwp_config.max_iter = 40
    pwp = core.pwp_model(
        subject_id,
        start,
        stop,
        event,
        [1, 2, 1, 2, 1],
        covariates,
        pwp_config,
    )
    anderson_gill = core.anderson_gill_model(subject_id, start, stop, event, covariates, 50, 1e-4)
    wlw_config = core.WLWConfig(50, 1e-4, True, False)
    wlw = core.wlw_model(
        [1, 1, 2, 2, 3, 3],
        [10.0, 20.0, 5.0, 15.0, 8.0, 25.0],
        [1, 0, 1, 1, 0, 0],
        [1, 2, 1, 2, 1, 2],
        [],
        wlw_config,
    )
    nb_config = core.NegativeBinomialFrailtyConfig(50, 1e-4, 20)
    negative_binomial = core.negative_binomial_frailty(
        [1, 1, 2, 2, 2, 3],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1, 0, 1, 1, 0, 0],
        [],
        None,
        nb_config,
    )
    marginal = core.marginal_recurrent_model(
        subject_id,
        start,
        stop,
        event,
        covariates,
        5,
        1,
        core.MarginalMethod("andersen_gill"),
        50,
        1e-4,
    )
    ag_wrapper = core.andersen_gill(subject_id, start, stop, event, covariates, 5, 1)
    wlw_wrapper = core.wei_lin_weissfeld(subject_id, stop, event, covariates, 5, 1)
    joint_frailty = core.joint_frailty_model(
        subject_id,
        start,
        stop,
        event,
        covariates,
        5,
        1,
        [12.0, 10.0, 8.0],
        [1, 0, 1],
        [1.0, 1.0, 0.0],
        3,
        1,
        core.FrailtyDistribution("gamma"),
        100,
        1e-4,
    )

    assert type(gap).__name__ == "GapTimeResult"
    assert gap.n_events == 4
    assert gap.n_subjects == 3
    assert len(gap.coefficients) == 1
    assert type(pwp_gap).__name__ == "GapTimeResult"
    assert pwp_gap.baseline_times == pytest.approx(gap.baseline_times)
    assert pwp_config.max_iter == 40
    assert type(pwp).__name__ == "PWPResult"
    assert pwp.n_events == 4
    assert len(pwp.coef) == 1
    assert len(pwp.event_specific_coef) == 2
    assert type(anderson_gill).__name__ == "AndersonGillResult"
    assert anderson_gill.n_events == 4
    assert anderson_gill.mean_event_rate == pytest.approx(0.16)
    assert wlw_config.max_iter == 50
    assert wlw_config.tol == pytest.approx(1e-4)
    assert wlw_config.robust_variance is True
    assert wlw_config.common_baseline is False
    assert type(wlw).__name__ == "WLWResult"
    assert wlw.n_subjects == 3
    assert wlw.n_strata == 2
    assert wlw.n_events == 3
    assert len(wlw.coef) == 1
    assert len(wlw.std_errors) == 1
    assert len(wlw.robust_std_errors) == 1
    assert len(wlw.hazard_ratios) == 1
    assert len(wlw.stratum_coef) == 2
    assert wlw.global_test_stat >= 0.0
    assert 0.0 <= wlw.global_test_pvalue <= 1.0
    assert nb_config.max_iter == 50
    assert nb_config.tol == pytest.approx(1e-4)
    assert nb_config.em_max_iter == 20
    assert type(negative_binomial).__name__ == "NegativeBinomialFrailtyResult"
    assert negative_binomial.n_subjects == 3
    assert negative_binomial.n_events == 3
    assert len(negative_binomial.coef) == 1
    assert len(negative_binomial.std_errors) == 1
    assert len(negative_binomial.rate_ratios) == 1
    assert negative_binomial.theta > 0.0
    assert negative_binomial.frailty_variance == pytest.approx(negative_binomial.theta)
    assert len(negative_binomial.frailty_estimates) == 3
    assert type(marginal).__name__ == "MarginalModelResult"
    assert marginal.n_subjects == 3
    assert marginal.mean_events_per_subject == pytest.approx(4.0 / 3.0)
    assert type(ag_wrapper).__name__ == "MarginalModelResult"
    assert ag_wrapper.n_events == 4
    assert type(wlw_wrapper).__name__ == "MarginalModelResult"
    assert wlw_wrapper.n_events == 4
    assert type(joint_frailty).__name__ == "JointFrailtyResult"
    assert joint_frailty.n_subjects == 3
    assert joint_frailty.n_recurrent_events == 4
    assert joint_frailty.n_terminal_events == 2
    assert len(joint_frailty.recurrent_coef) == 1
    assert len(joint_frailty.terminal_coef) == 1
    assert len(joint_frailty.frailty_values) == 3


def test_royston_and_yates_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "RoystonResult",
        "YatesResult",
        "YatesPairwiseResult",
        "royston",
        "royston_from_model",
        "yates",
        "yates_contrast",
        "yates_pairwise",
    } <= stub_names

    function_args = {
        "royston": ["linear_predictor", "time", "status"],
        "royston_from_model": ["x", "coef", "n_obs", "time", "status"],
        "yates": ["predictions", "factor", "weights", "conf_level"],
        "yates_contrast": [
            "x",
            "coef",
            "n_obs",
            "n_vars",
            "factor_col",
            "factor_levels",
            "predict_type",
        ],
        "yates_pairwise": ["yates_result"],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    assert _pyi_class_property_names(stub_path, "RoystonResult") == {
        "d",
        "se",
        "r_squared_d",
        "r_squared_ko",
        "z",
        "p_value",
        "n_events",
    }
    assert _pyi_class_property_names(stub_path, "YatesResult") == {
        "levels",
        "means",
        "se",
        "lower",
        "upper",
        "n",
        "predict_type",
    }
    assert _pyi_class_property_names(stub_path, "YatesPairwiseResult") == {
        "level1",
        "level2",
        "difference",
        "se",
        "z",
        "p_value",
    }

    royston = core.royston(
        [0.5, -0.3, 0.8, -0.1, 0.2, -0.5, 0.9, -0.2],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1, 1, 1, 0, 1, 1, 1, 0],
    )
    from_model = core.royston_from_model(
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.5, 1.0],
        3,
        [1.0, 2.0, 3.0],
        [1, 1, 1],
    )
    yates = core.yates(
        [1.0, 2.0, 2.0, 4.0],
        ["A", "A", "B", "B"],
        [1.0, 2.0, 1.0, 1.0],
        0.95,
    )
    contrast = core.yates_contrast(
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.5, 1.0],
        3,
        2,
        0,
        [0.0, 1.0],
        "risk",
    )
    pairwise = core.yates_pairwise(yates)

    assert type(royston).__name__ == "RoystonResult"
    assert royston.d > 0.0
    assert royston.se > 0.0
    assert 0.0 <= royston.r_squared_d <= 1.0
    assert 0.0 <= royston.r_squared_ko <= 1.0
    assert royston.n_events == 6
    assert type(from_model).__name__ == "RoystonResult"
    assert from_model.n_events == 3
    assert type(yates).__name__ == "YatesResult"
    assert yates.levels == ["A", "B"]
    assert yates.means == pytest.approx([5.0 / 3.0, 3.0])
    assert yates.predict_type == "linear"
    assert type(contrast).__name__ == "YatesResult"
    assert contrast.levels == ["0", "1"]
    assert contrast.predict_type == "risk"
    assert contrast.means[1] > contrast.means[0]
    assert type(pairwise).__name__ == "YatesPairwiseResult"
    assert pairwise.level1 == ["A"]
    assert pairwise.level2 == ["B"]
    assert pairwise.difference == pytest.approx([-4.0 / 3.0])


def test_population_survival_and_pyears_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "RateDimension",
        "RateTable",
        "RatetableDateResult",
        "SurvExpResult",
        "PyearsSummary",
        "PyearsCell",
        "ExpectedSurvivalResult",
        "create_simple_ratetable",
        "is_ratetable",
        "ratetable_date",
        "days_to_date",
        "survexp",
        "survexp_individual",
        "summary_pyears",
        "pyears_by_cell",
        "pyears_ci",
        "survexp_us",
        "survexp_mn",
        "survexp_usr",
        "compute_expected_survival",
    } <= stub_names

    class_args = {
        "RateDimension": ["name", "dim_type", "cutpoints", "levels"],
        "RateTable": ["dimensions", "rates", "summary"],
    }
    for class_name, expected_args in class_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == expected_args
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == [
            "self",
            *expected_args,
        ]

    function_args = {
        "create_simple_ratetable": [
            "age_breaks",
            "year_breaks",
            "rates_male",
            "rates_female",
        ],
        "is_ratetable": ["ndim", "has_rates", "has_dims"],
        "ratetable_date": ["year", "month", "day", "origin_year"],
        "days_to_date": ["days", "origin_year"],
        "survexp": ["time", "age", "year", "ratetable", "sex", "times", "method"],
        "survexp_individual": ["time", "age", "year", "ratetable", "sex"],
        "summary_pyears": ["pyears", "pn", "pcount", "pexpect", "offtable"],
        "pyears_by_cell": ["pyears", "pn", "pcount", "pexpect"],
        "pyears_ci": ["observed", "expected", "conf_level"],
        "survexp_us": [],
        "survexp_mn": [],
        "survexp_usr": [],
        "compute_expected_survival": ["age", "sex", "year", "times", "ratetable"],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    rate_table_methods = {
        "ndim": ["self"],
        "dim_names": ["self"],
        "lookup": ["self", "coords"],
        "lookup_interpolate": ["self", "coords"],
        "cumulative_hazard": ["self", "age_start", "age_end", "year_start", "sex"],
        "expected_survival": ["self", "age_start", "age_end", "year_start", "sex"],
    }
    for method_name, expected_args in rate_table_methods.items():
        assert list(inspect.signature(getattr(core.RateTable, method_name)).parameters) == (
            expected_args
        )
        assert _pyi_class_method_arg_names(stub_path, "RateTable", method_name) == expected_args
    assert _pyi_class_method_arg_names(stub_path, "PyearsSummary", "to_table") == ["self"]

    assert _pyi_class_property_names(stub_path, "RateDimension") == {
        "name",
        "dim_type",
        "levels",
        "cutpoints",
    }
    assert _pyi_class_property_names(stub_path, "RateTable") == {"summary"}
    assert _pyi_class_property_names(stub_path, "RatetableDateResult") == {
        "days",
        "years",
        "origin_year",
    }
    assert _pyi_class_property_names(stub_path, "SurvExpResult") == {
        "time",
        "surv",
        "n_risk",
        "cumhaz",
        "method",
        "n",
    }
    assert _pyi_class_property_names(stub_path, "PyearsSummary") == {
        "total_person_years",
        "total_events",
        "total_expected",
        "n_observations",
        "offtable",
        "observed_rate",
        "expected_rate",
        "smr",
        "sir",
    }
    assert _pyi_class_property_names(stub_path, "PyearsCell") == {
        "person_years",
        "n",
        "events",
        "expected",
        "rate",
        "smr",
    }
    assert _pyi_class_property_names(stub_path, "ExpectedSurvivalResult") == {
        "expected_survival",
        "time",
        "n",
    }

    dimension = core.RateDimension("age", core.DimType.Age, [0.0, 36500.0], None)
    custom_table = core.RateTable([dimension], [0.001], "single-age table")
    assert custom_table.ndim() == 1
    assert custom_table.dim_names() == ["age"]
    assert custom_table.summary == "single-age table"
    assert custom_table.lookup({"age": 1000.0}) == pytest.approx(0.001)

    ratetable = core.create_simple_ratetable(
        [0.0, 36500.0, 73000.0],
        [1990.0, 2020.0],
        [0.00001, 0.00005],
        [0.000008, 0.00004],
    )
    assert type(ratetable).__name__ == "RateTable"
    assert core.is_ratetable(ratetable.ndim(), True, True) is True
    assert ratetable.lookup({"age": 1000.0, "year": 2000.0, "sex": 0.0}) == pytest.approx(
        0.00001,
    )

    date = core.ratetable_date(2000, 2, 29, 1960)
    assert type(date).__name__ == "RatetableDateResult"
    assert date.days == pytest.approx(14669.0)
    assert core.days_to_date(date.days, date.origin_year) == (2000, 2, 29)

    survexp = core.survexp(
        [365.0, 730.0],
        [18250.0, 21900.0],
        [2000.0, 2000.0],
        ratetable,
        [0, 1],
        None,
        "conditional",
    )
    assert type(survexp).__name__ == "SurvExpResult"
    assert survexp.method == "conditional"
    assert survexp.n == 2
    assert survexp.n_risk == [2.0, 1.0]
    individual = core.survexp_individual(
        [365.0, 730.0],
        [18250.0, 21900.0],
        [2000.0, 2000.0],
        ratetable,
        [0, 1],
    )
    assert len(individual) == 2
    assert all(0.0 < value <= 1.0 for value in individual)

    summary = core.summary_pyears([2.0, 3.0], [1.0, 1.0], [1.0, 2.0], [0.5, 1.5], 0.25)
    cells = core.pyears_by_cell([2.0, 3.0], [1.0, 1.0], [1.0, 2.0], [0.5, 1.5])
    smr, lower, upper = core.pyears_ci(5.0, 2.5, 0.95)
    assert type(summary).__name__ == "PyearsSummary"
    assert summary.total_person_years == pytest.approx(5.0)
    assert "SMR" in summary.to_table()
    assert [type(cell).__name__ for cell in cells] == ["PyearsCell", "PyearsCell"]
    assert cells[0].smr == pytest.approx(2.0)
    assert lower < smr < upper

    us = core.survexp_us()
    mn = core.survexp_mn()
    usr = core.survexp_usr()
    expected = core.compute_expected_survival(
        [365.25 * 50.0, 365.25 * 60.0],
        [0, 1],
        [2000.0, 2005.0],
        [365.25, 365.25 * 5.0],
        None,
    )
    assert [type(table).__name__ for table in (us, mn, usr)] == [
        "RateTable",
        "RateTable",
        "RateTable",
    ]
    assert type(expected).__name__ == "ExpectedSurvivalResult"
    assert expected.n == 2
    assert expected.time == pytest.approx([365.25, 365.25 * 5.0])
    assert expected.expected_survival[0] >= expected.expected_survival[1]


def test_relative_survival_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "NetSurvivalMethod",
        "NetSurvivalResult",
        "RelativeSurvivalResult",
        "ExcessHazardModelResult",
        "net_survival",
        "crude_probability_of_death",
        "relative_survival",
        "excess_hazard_regression",
    } <= stub_names

    assert list(inspect.signature(core.NetSurvivalMethod).parameters) == ["name"]
    assert _pyi_class_method_arg_names(stub_path, "NetSurvivalMethod", "__init__") == [
        "self",
        "name",
    ]

    function_args = {
        "net_survival": ["time", "status", "expected_survival", "method", "weights"],
        "crude_probability_of_death": [
            "time",
            "status",
            "_expected_survival",
            "cause",
            "time_points",
        ],
        "relative_survival": [
            "time",
            "status",
            "expected_hazard",
            "age_at_diagnosis",
            "follow_up_years",
        ],
        "excess_hazard_regression": [
            "time",
            "status",
            "x",
            "n_obs",
            "n_vars",
            "expected_hazard",
            "max_iter",
            "tol",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    assert _pyi_class_property_names(stub_path, "NetSurvivalResult") == {
        "time_points",
        "net_survival",
        "net_survival_se",
        "net_survival_lower",
        "net_survival_upper",
        "cumulative_excess_hazard",
        "n_at_risk",
        "n_events",
        "method",
    }
    assert _pyi_class_property_names(stub_path, "RelativeSurvivalResult") == {
        "time_points",
        "observed_survival",
        "expected_survival",
        "relative_survival",
        "relative_survival_se",
        "cumulative_excess_hazard",
        "excess_mortality_rate",
        "n_at_risk",
        "n_events",
    }
    assert _pyi_class_property_names(stub_path, "ExcessHazardModelResult") == {
        "coefficients",
        "std_errors",
        "excess_hazard_ratio",
        "ehr_ci_lower",
        "ehr_ci_upper",
        "baseline_excess_hazard",
        "log_likelihood",
        "aic",
        "n_iter",
        "converged",
    }

    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    status = [1, 0, 1, 0, 1]
    expected_survival = [0.98, 0.96, 0.94, 0.92, 0.90]
    expected_hazard = [0.01, 0.012, 0.014, 0.016, 0.018]
    age = [50.0, 60.0, 55.0, 65.0, 70.0]

    method = core.NetSurvivalMethod("pohar_perme")
    net = core.net_survival(time, status, expected_survival, method, None)
    crude = core.crude_probability_of_death(
        time,
        status,
        expected_survival,
        [1, 0, 2, 0, 1],
        [2.0, 5.0],
    )
    relative = core.relative_survival(time, status, expected_hazard, age, None)
    model = core.excess_hazard_regression(
        time,
        status,
        [0.0, 1.0, 0.5, 1.5, 2.0],
        5,
        1,
        expected_hazard,
        10,
        1e-6,
    )

    assert type(method).__name__ == "NetSurvivalMethod"
    assert type(net).__name__ == "NetSurvivalResult"
    assert net.time_points == pytest.approx(time)
    assert net.method == "Pohar-Perme"
    assert len(net.net_survival) == len(time)
    assert all(0.0 <= value <= 1.0 for value in net.net_survival)
    assert crude[0] == pytest.approx([2.0, 5.0])
    assert crude[1] == pytest.approx([0.2, 0.4])
    assert crude[2] == pytest.approx([0.0, 0.2])
    assert type(relative).__name__ == "RelativeSurvivalResult"
    assert relative.time_points == pytest.approx(time)
    assert relative.n_events == [1, 0, 1, 0, 1]
    assert len(relative.relative_survival) == len(time)
    assert type(model).__name__ == "ExcessHazardModelResult"
    assert len(model.coefficients) == 1
    assert len(model.excess_hazard_ratio) == 1
    assert len(model.baseline_excess_hazard) == len(time)
    assert model.n_iter <= 10


def test_spatial_survival_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "SpatialCorrelationStructure",
        "SpatialFrailtyResult",
        "CentralityType",
        "NetworkSurvivalConfig",
        "NetworkSurvivalResult",
        "DiffusionSurvivalConfig",
        "DiffusionSurvivalResult",
        "NetworkHeterogeneityResult",
        "spatial_frailty_model",
        "compute_spatial_smoothed_rates",
        "moran_i_test",
        "network_survival_model",
        "diffusion_survival_model",
        "network_heterogeneity_survival",
    } <= stub_names

    class_init_args = {
        "SpatialCorrelationStructure": ["self", "name"],
        "CentralityType": ["self", "name"],
        "NetworkSurvivalConfig": [
            "self",
            "include_peer_effects",
            "include_centrality",
            "centrality_type",
            "peer_lag",
            "max_iter",
            "tol",
        ],
        "DiffusionSurvivalConfig": [
            "self",
            "diffusion_rate",
            "recovery_rate",
            "susceptibility_covariate",
            "max_iter",
            "tol",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    function_args = {
        "spatial_frailty_model": [
            "time",
            "status",
            "x",
            "n_obs",
            "n_vars",
            "region_id",
            "adjacency_matrix",
            "n_regions",
            "correlation_structure",
            "max_iter",
            "tol",
        ],
        "compute_spatial_smoothed_rates": [
            "observed_events",
            "expected_events",
            "adjacency_matrix",
            "n_regions",
            "smoothing_param",
        ],
        "moran_i_test": ["values", "adjacency_matrix", "n_regions"],
        "network_survival_model": [
            "time",
            "event",
            "covariates",
            "n_covariates",
            "adjacency_matrix",
            "n_nodes",
            "config",
        ],
        "diffusion_survival_model": [
            "infection_time",
            "infected",
            "covariates",
            "n_covariates",
            "adjacency_matrix",
            "n_nodes",
            "config",
        ],
        "network_heterogeneity_survival": [
            "time",
            "event",
            "adjacency_matrix",
            "n_nodes",
            "n_communities",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    assert _pyi_class_annotation_names(stub_path, "SpatialCorrelationStructure") == {
        "CAR",
        "SAR",
        "Exponential",
        "Matern",
    }
    assert _pyi_class_annotation_names(stub_path, "CentralityType") == {
        "Degree",
        "Betweenness",
        "Closeness",
        "Eigenvector",
        "PageRank",
    }
    assert _pyi_class_annotation_names(stub_path, "NetworkSurvivalConfig") == {
        "include_peer_effects",
        "include_centrality",
        "centrality_type",
        "peer_lag",
        "max_iter",
        "tol",
    }
    assert _pyi_class_annotation_names(stub_path, "DiffusionSurvivalConfig") == {
        "diffusion_rate",
        "recovery_rate",
        "susceptibility_covariate",
        "max_iter",
        "tol",
    }

    assert _pyi_class_property_names(stub_path, "SpatialFrailtyResult") == {
        "coefficients",
        "std_errors",
        "hazard_ratios",
        "hr_ci_lower",
        "hr_ci_upper",
        "spatial_frailties",
        "frailty_variance",
        "spatial_correlation",
        "log_likelihood",
        "dic",
        "n_regions",
        "converged",
    }
    assert _pyi_class_property_names(stub_path, "NetworkSurvivalResult") == {
        "coefficients",
        "std_errors",
        "hazard_ratios",
        "hr_ci_lower",
        "hr_ci_upper",
        "peer_effect",
        "peer_effect_se",
        "centrality_effect",
        "centrality_effect_se",
        "centrality_values",
        "log_likelihood",
        "aic",
        "bic",
        "n_nodes",
        "n_edges",
        "converged",
    }
    assert _pyi_class_property_names(stub_path, "DiffusionSurvivalResult") == {
        "diffusion_rate",
        "diffusion_rate_se",
        "recovery_rate",
        "recovery_rate_se",
        "susceptibility_coef",
        "susceptibility_se",
        "infection_probabilities",
        "expected_infection_times",
        "log_likelihood",
        "r0",
        "converged",
    }
    assert _pyi_class_property_names(stub_path, "NetworkHeterogeneityResult") == {
        "community_hazard_ratios",
        "within_community_correlation",
        "between_community_effect",
        "modularity",
        "community_assignments",
        "log_likelihood",
    }

    assert repr(core.SpatialCorrelationStructure("car")) == "SpatialCorrelationStructure.CAR"
    assert repr(core.CentralityType("degree")) == "CentralityType.Degree"

    time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    event = [1, 0, 1, 0, 1, 0]
    covariates = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    adjacency = [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    spatial = core.spatial_frailty_model(
        time,
        event,
        covariates,
        6,
        1,
        [0, 0, 1, 1, 2, 2],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        3,
        core.SpatialCorrelationStructure.CAR,
        5,
        1e-4,
    )
    assert type(spatial).__name__ == "SpatialFrailtyResult"
    assert spatial.n_regions == 3
    assert len(spatial.coefficients) == 1
    assert len(spatial.spatial_frailties) == 3
    assert math.isfinite(spatial.log_likelihood)

    raw_rates, smoothed_rates = core.compute_spatial_smoothed_rates(
        [2.0, 4.0, 6.0],
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        3,
        0.5,
    )
    assert raw_rates == pytest.approx([2.0, 2.0, 2.0])
    assert smoothed_rates == pytest.approx([2.0, 2.0, 2.0])

    moran_i, z_score, p_value = core.moran_i_test(
        [1.0, 2.0, 1.5, 2.5],
        [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        4,
    )
    assert -1.0 <= moran_i <= 1.0
    assert math.isfinite(z_score)
    assert 0.0 <= p_value <= 1.0

    network_config = core.NetworkSurvivalConfig(True, True, core.CentralityType.Degree, 1, 5, 1e-4)
    network = core.network_survival_model(
        time,
        event,
        covariates,
        1,
        adjacency,
        6,
        network_config,
    )
    assert type(network).__name__ == "NetworkSurvivalResult"
    assert network.n_nodes == 6
    assert network.n_edges == 6
    assert len(network.coefficients) == 1
    assert len(network.centrality_values) == 6
    assert all(math.isfinite(value) for value in network.centrality_values)

    diffusion_config = core.DiffusionSurvivalConfig(0.2, 0.1, True, 5, 1e-4)
    diffusion = core.diffusion_survival_model(
        [0.0, 1.0, 2.0, 3.0, 10.0, 10.0],
        [1, 1, 1, 1, 0, 0],
        covariates,
        1,
        adjacency,
        6,
        diffusion_config,
    )
    assert type(diffusion).__name__ == "DiffusionSurvivalResult"
    assert math.isfinite(diffusion.diffusion_rate)
    assert math.isfinite(diffusion.r0)
    assert len(diffusion.infection_probabilities) == 6
    assert all(0.0 <= value <= 1.0 for value in diffusion.infection_probabilities)

    heterogeneity = core.network_heterogeneity_survival(time, event, adjacency, 6, 2)
    assert type(heterogeneity).__name__ == "NetworkHeterogeneityResult"
    assert len(heterogeneity.community_hazard_ratios) == 2
    assert len(heterogeneity.community_assignments) == 6
    assert math.isfinite(heterogeneity.modularity)


def test_joint_dynamic_prediction_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "AssociationStructure",
        "JointSurvivalModelConfig",
        "JointModelResult",
        "DynamicPredictionResult",
        "TimeVaryingAUCResult",
        "DynamicCIndexResult",
        "IPCWAUCResult",
        "SuperLandmarkResult",
        "TimeDependentROCResult",
        "joint_model",
        "dynamic_prediction",
        "dynamic_auc",
        "dynamic_brier_score",
        "landmarking_analysis",
        "time_varying_auc",
        "dynamic_c_index",
        "ipcw_auc",
        "super_landmark_model",
        "time_dependent_roc",
    } <= stub_names

    class_init_args = {
        "AssociationStructure": ["self", "name"],
        "JointSurvivalModelConfig": [
            "self",
            "association",
            "n_quadrature",
            "max_iter",
            "tol",
            "baseline_hazard_knots",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    function_args = {
        "joint_model": [
            "y_longitudinal",
            "times_longitudinal",
            "x_longitudinal",
            "n_long_obs",
            "n_long_vars",
            "subject_ids_long",
            "event_time",
            "event_status",
            "x_survival",
            "n_subjects",
            "n_surv_vars",
            "config",
        ],
        "dynamic_prediction": [
            "beta_long",
            "gamma_surv",
            "alpha",
            "random_effects",
            "baseline_hazard",
            "baseline_times",
            "y_history",
            "times_history",
            "x_long_fixed",
            "n_history",
            "n_long_vars",
            "x_surv",
            "n_surv_vars",
            "landmark_time",
            "prediction_times",
            "n_monte_carlo",
        ],
        "dynamic_auc": [
            "beta_long",
            "gamma_surv",
            "alpha",
            "baseline_hazard",
            "baseline_times",
            "y_observed",
            "times_observed",
            "x_long_fixed",
            "n_obs",
            "n_long_vars",
            "x_surv",
            "n_surv_vars",
            "event_time",
            "event_status",
            "horizon",
        ],
        "dynamic_brier_score": [
            "survival_predictions",
            "event_time",
            "event_status",
            "prediction_times",
        ],
        "landmarking_analysis": [
            "event_time",
            "event_status",
            "covariates",
            "n_subjects",
            "n_vars",
            "landmark_times",
            "horizon",
        ],
        "time_varying_auc": [
            "risk_scores",
            "event_time",
            "event_status",
            "eval_times",
            "prediction_window",
            "method",
        ],
        "dynamic_c_index": [
            "risk_scores",
            "event_time",
            "event_status",
            "landmark_time",
            "horizon",
            "eval_times",
        ],
        "ipcw_auc": ["risk_scores", "event_time", "event_status", "eval_times"],
        "super_landmark_model": [
            "event_time",
            "event_status",
            "covariates",
            "n_vars",
            "landmark_times",
            "horizon",
            "max_iter",
        ],
        "time_dependent_roc": [
            "risk_scores",
            "event_time",
            "event_status",
            "eval_times",
            "n_thresholds",
        ],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    assert _pyi_class_annotation_names(stub_path, "AssociationStructure") == {
        "Value",
        "Slope",
        "ValueSlope",
        "Area",
        "SharedRandomEffects",
    }
    assert _pyi_class_annotation_names(stub_path, "JointSurvivalModelConfig") == {
        "association",
        "n_quadrature",
        "max_iter",
        "tol",
        "baseline_hazard_knots",
    }

    assert _pyi_class_property_names(stub_path, "JointModelResult") == {
        "longitudinal_fixed",
        "longitudinal_fixed_se",
        "survival_fixed",
        "survival_fixed_se",
        "association_param",
        "association_se",
        "random_effects_var",
        "residual_var",
        "baseline_hazard",
        "baseline_hazard_times",
        "log_likelihood",
        "aic",
        "bic",
        "n_iter",
        "converged",
        "random_effects",
    }
    assert _pyi_class_property_names(stub_path, "DynamicPredictionResult") == {
        "time_points",
        "survival_mean",
        "survival_lower",
        "survival_upper",
        "cumulative_risk",
        "conditional_survival",
        "auc",
        "brier_score",
    }
    assert _pyi_class_property_names(stub_path, "TimeVaryingAUCResult") == {
        "times",
        "auc_values",
        "auc_lower",
        "auc_upper",
        "integrated_auc",
        "n_cases",
        "n_controls",
    }
    assert _pyi_class_property_names(stub_path, "DynamicCIndexResult") == {
        "c_index",
        "se",
        "lower",
        "upper",
        "n_concordant",
        "n_discordant",
        "n_tied",
        "n_pairs",
        "time_dependent_c",
        "eval_times",
    }
    assert _pyi_class_property_names(stub_path, "IPCWAUCResult") == {
        "times",
        "auc_values",
        "auc_se",
        "integrated_auc",
        "ipcw_weights",
    }
    assert _pyi_class_property_names(stub_path, "SuperLandmarkResult") == {
        "landmark_times",
        "coefficients",
        "std_errors",
        "c_indices",
        "brier_scores",
        "n_at_risk",
        "n_events",
        "pooled_coef",
        "pooled_se",
    }
    assert _pyi_class_property_names(stub_path, "TimeDependentROCResult") == {
        "times",
        "sensitivity",
        "specificity",
        "thresholds",
        "auc",
        "optimal_threshold",
    }

    assert repr(core.AssociationStructure("value")) == "AssociationStructure.Value"
    assert repr(core.AssociationStructure("shared")) == "AssociationStructure.SharedRandomEffects"

    config = core.JointSurvivalModelConfig(core.AssociationStructure.Value, 5, 5, 1e-4, 3)
    assert config.n_quadrature == 5
    assert config.association == core.AssociationStructure.Value
    config.tol = 1e-5
    assert config.tol == pytest.approx(1e-5)

    joint = core.joint_model(
        [1.0, 2.0, 1.5, 2.5],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.5, 1.0, 0.6, 1.0, 0.7, 1.0, 0.8],
        4,
        2,
        [0, 0, 1, 1],
        [2.0, 3.0],
        [1, 0],
        [0.5, 0.6],
        2,
        1,
        config,
    )
    assert type(joint).__name__ == "JointModelResult"
    assert len(joint.longitudinal_fixed) == 2
    assert len(joint.survival_fixed) == 1
    assert len(joint.random_effects) == 2
    assert joint.n_iter <= 5

    with pytest.raises(ValueError, match="x_longitudinal length"):
        core.joint_model(
            [1.0, 2.0, 1.5, 2.5],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.5, 1.0, 0.6],
            4,
            2,
            [0, 0, 1, 1],
            [2.0, 3.0],
            [1, 0],
            [0.5, 0.6],
            2,
            1,
            config,
        )

    prediction = core.dynamic_prediction(
        [0.5, 0.3],
        [0.2],
        0.1,
        [0.0, 0.0],
        [0.01, 0.02, 0.03],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 0.5, 0.3],
        [1.0, 0.5, 1.0, 0.3, 1.0, 0.7],
        3,
        2,
        [0.5],
        1,
        2.0,
        [3.0, 4.0, 5.0],
        20,
    )
    assert type(prediction).__name__ == "DynamicPredictionResult"
    assert prediction.time_points == pytest.approx([3.0, 4.0, 5.0])
    assert len(prediction.survival_mean) == 3
    assert all(0.0 <= value <= 1.0 for value in prediction.survival_mean)

    risk_scores = [0.8, 0.6, 0.4, 0.2, 0.9, 0.3]
    event_time = [1.0, 2.0, 3.0, 4.0, 1.5, 5.0]
    event_status = [1, 1, 0, 1, 1, 0]
    eval_times = [2.0, 3.0, 4.0]

    auc = core.dynamic_auc(
        [0.1],
        [0.2],
        0.1,
        [0.01, 0.02],
        [1.0, 2.0],
        [],
        [],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        6,
        1,
        [0.5, 0.6, 0.4, 0.2, 0.7, 0.3],
        1,
        event_time,
        event_status,
        4.0,
    )
    brier = core.dynamic_brier_score(
        [[0.9, 0.8, 0.7] for _ in event_time],
        event_time,
        event_status,
        eval_times,
    )
    landmark = core.landmarking_analysis(
        [float(idx + 5) for idx in range(12)],
        [1 if idx % 2 == 0 else 0 for idx in range(12)],
        [float(idx) / 10.0 for idx in range(12)],
        12,
        1,
        [0.0, 2.0],
        10.0,
    )
    tv_auc = core.time_varying_auc(
        risk_scores,
        event_time,
        event_status,
        [1.5, 2.5, 3.5],
        1.0,
        "cumulative/dynamic",
    )
    c_index = core.dynamic_c_index(risk_scores, event_time, event_status, 0.0, 6.0, None)
    ipcw = core.ipcw_auc(risk_scores, event_time, event_status, eval_times)
    super_landmark = core.super_landmark_model(
        [float(idx + 1) for idx in range(24)],
        [1 if idx % 3 else 0 for idx in range(24)],
        [float(idx) / 20.0 for idx in range(24)],
        1,
        [0.0, 5.0],
        20.0,
        5,
    )
    roc = core.time_dependent_roc(risk_scores, event_time, event_status, [2.0, 3.5], 10)

    assert 0.0 <= auc <= 1.0
    assert len(brier) == len(eval_times)
    assert all(value >= 0.0 for value in brier)
    assert len(landmark) == 2
    assert all(len(row[1]) == 1 for row in landmark)
    assert type(tv_auc).__name__ == "TimeVaryingAUCResult"
    assert tv_auc.times == pytest.approx([1.5, 2.5, 3.5])
    assert len(tv_auc.auc_values) == 3
    assert 0.0 <= tv_auc.integrated_auc <= 1.0
    assert type(c_index).__name__ == "DynamicCIndexResult"
    assert 0.0 <= c_index.c_index <= 1.0
    assert c_index.n_pairs > 0
    assert len(c_index.time_dependent_c) == len(c_index.eval_times)
    assert type(ipcw).__name__ == "IPCWAUCResult"
    assert ipcw.times == pytest.approx(eval_times)
    assert len(ipcw.ipcw_weights) == len(event_time)
    assert 0.0 <= ipcw.integrated_auc <= 1.0
    assert type(super_landmark).__name__ == "SuperLandmarkResult"
    assert super_landmark.landmark_times == pytest.approx([0.0, 5.0])
    assert len(super_landmark.coefficients) == 2
    assert len(super_landmark.pooled_coef) == 1
    assert type(roc).__name__ == "TimeDependentROCResult"
    assert roc.times == pytest.approx([2.0, 3.5])
    assert len(roc.thresholds) == 10
    assert len(roc.sensitivity) == len(roc.specificity) == 2


def test_rmst_and_landmark_summary_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "RMSTResult",
        "RMSTComparisonResult",
        "ChangepointInfo",
        "RMSTOptimalThresholdResult",
        "ConditionalSurvivalResult",
        "HazardRatioResult",
        "SurvivalAtTimeResult",
        "LifeTableResult",
        "rmst",
        "rmst_comparison",
        "rmst_optimal_threshold",
        "conditional_survival",
        "hazard_ratio",
        "survival_at_times",
        "life_table",
    } <= stub_names

    function_args = {
        "rmst": ["time", "status", "tau", "confidence_level"],
        "rmst_comparison": ["time", "status", "group", "tau", "confidence_level"],
        "rmst_optimal_threshold": [
            "time",
            "status",
            "alpha",
            "min_events_per_interval",
            "confidence_level",
        ],
        "conditional_survival": [
            "time",
            "status",
            "given_time",
            "target_time",
            "confidence_level",
        ],
        "hazard_ratio": ["time", "status", "group", "confidence_level"],
        "survival_at_times": ["time", "status", "eval_times", "confidence_level"],
        "life_table": ["time", "status", "breaks"],
    }
    for name, expected_args in function_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == expected_args
        assert _pyi_function_arg_names(stub_path, name) == expected_args

    class_init_args = {
        "RMSTResult": ["self", "rmst", "variance", "se", "ci_lower", "ci_upper", "tau"],
        "RMSTComparisonResult": [
            "self",
            "rmst_diff",
            "rmst_ratio",
            "diff_se",
            "diff_ci_lower",
            "diff_ci_upper",
            "ratio_ci_lower",
            "ratio_ci_upper",
            "p_value",
            "rmst_group1",
            "rmst_group2",
        ],
        "ChangepointInfo": [
            "self",
            "time",
            "hazard_before",
            "hazard_after",
            "likelihood_ratio",
            "p_value",
        ],
        "RMSTOptimalThresholdResult": [
            "self",
            "optimal_tau",
            "max_followup",
            "changepoints",
            "n_changepoints",
            "rmst_at_optimal",
        ],
        "ConditionalSurvivalResult": [
            "self",
            "given_time",
            "target_time",
            "conditional_survival",
            "ci_lower",
            "ci_upper",
            "n_at_risk",
        ],
        "HazardRatioResult": [
            "self",
            "hazard_ratio",
            "ci_lower",
            "ci_upper",
            "se_log_hr",
            "z_statistic",
            "p_value",
        ],
        "SurvivalAtTimeResult": [
            "self",
            "time",
            "survival",
            "ci_lower",
            "ci_upper",
            "n_at_risk",
            "n_events",
        ],
        "LifeTableResult": [
            "self",
            "interval_start",
            "interval_end",
            "n_at_risk",
            "n_deaths",
            "n_censored",
            "n_effective",
            "hazard",
            "survival",
            "se_survival",
        ],
    }
    for class_name, expected_args in class_init_args.items():
        runtime_args = list(inspect.signature(getattr(core, class_name)).parameters)
        assert runtime_args == expected_args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == expected_args

    assert _pyi_class_property_names(stub_path, "RMSTResult") == {
        "rmst",
        "variance",
        "se",
        "ci_lower",
        "ci_upper",
        "tau",
    }
    assert _pyi_class_property_names(stub_path, "RMSTComparisonResult") == {
        "rmst_diff",
        "rmst_ratio",
        "diff_se",
        "diff_ci_lower",
        "diff_ci_upper",
        "ratio_ci_lower",
        "ratio_ci_upper",
        "p_value",
        "rmst_group1",
        "rmst_group2",
    }
    assert _pyi_class_property_names(stub_path, "ChangepointInfo") == {
        "time",
        "hazard_before",
        "hazard_after",
        "likelihood_ratio",
        "p_value",
    }
    assert _pyi_class_property_names(stub_path, "RMSTOptimalThresholdResult") == {
        "optimal_tau",
        "max_followup",
        "changepoints",
        "n_changepoints",
        "rmst_at_optimal",
    }
    assert _pyi_class_property_names(stub_path, "ConditionalSurvivalResult") == {
        "given_time",
        "target_time",
        "conditional_survival",
        "ci_lower",
        "ci_upper",
        "n_at_risk",
    }
    assert _pyi_class_property_names(stub_path, "HazardRatioResult") == {
        "hazard_ratio",
        "ci_lower",
        "ci_upper",
        "se_log_hr",
        "z_statistic",
        "p_value",
    }
    assert _pyi_class_property_names(stub_path, "SurvivalAtTimeResult") == {
        "time",
        "survival",
        "ci_lower",
        "ci_upper",
        "n_at_risk",
        "n_events",
    }
    assert _pyi_class_property_names(stub_path, "LifeTableResult") == {
        "interval_start",
        "interval_end",
        "n_at_risk",
        "n_deaths",
        "n_censored",
        "n_effective",
        "hazard",
        "survival",
        "se_survival",
    }

    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    status = [1, 0, 1, 0, 1]
    group = [0, 0, 1, 1, 1]

    rmst = core.rmst(time, status, 4.0, None)
    comparison = core.rmst_comparison(time, status, group, 4.0, None)
    threshold = core.rmst_optimal_threshold(time, status, None, None, None)
    conditional = core.conditional_survival(time, status, 1.0, 4.0, None)
    hazard = core.hazard_ratio(time, status, group, None)
    survival_times = core.survival_at_times(time, status, [1.5, 3.5], None)
    table = core.life_table(time, status, [0.0, 2.0, 4.0, 6.0])

    assert type(rmst).__name__ == "RMSTResult"
    assert rmst.rmst > 0.0
    assert 0.0 <= rmst.ci_lower <= rmst.ci_upper
    assert type(comparison).__name__ == "RMSTComparisonResult"
    assert type(comparison.rmst_group1).__name__ == "RMSTResult"
    assert type(threshold).__name__ == "RMSTOptimalThresholdResult"
    assert type(threshold.rmst_at_optimal).__name__ == "RMSTResult"
    assert type(conditional).__name__ == "ConditionalSurvivalResult"
    assert conditional.given_time == pytest.approx(1.0)
    assert type(hazard).__name__ == "HazardRatioResult"
    assert hazard.hazard_ratio > 0.0
    assert [type(item).__name__ for item in survival_times] == [
        "SurvivalAtTimeResult",
        "SurvivalAtTimeResult",
    ]
    assert survival_times[0].time == pytest.approx(1.5)
    assert type(table).__name__ == "LifeTableResult"
    assert table.interval_start == pytest.approx([0.0, 2.0, 4.0])
    assert len(table.survival) == 3


def test_counting_concordance_binding_is_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "concordance_index",
        "concordance_summary",
        "stratified_concordance_summary",
        "concordance_rank_rows",
        "stratified_concordance_rank_rows",
        "concordance_influence_rows",
        "stratified_concordance_influence_rows",
        "counting_concordance_index",
        "counting_concordance_summary",
        "stratified_counting_concordance_summary",
        "counting_concordance_rank_rows",
        "stratified_counting_concordance_rank_rows",
        "counting_concordance_influence_rows",
        "stratified_counting_concordance_influence_rows",
    } <= stub_names
    assert "time" in inspect.signature(core.concordance_index).parameters
    assert "time" in inspect.signature(core.concordance_summary).parameters
    assert "time" in inspect.signature(core.stratified_concordance_summary).parameters
    assert "time" in inspect.signature(core.concordance_rank_rows).parameters
    assert "time" in inspect.signature(core.stratified_concordance_rank_rows).parameters
    assert "time" in inspect.signature(core.concordance_influence_rows).parameters
    assert "time" in inspect.signature(core.stratified_concordance_influence_rows).parameters
    assert "start" in inspect.signature(core.counting_concordance_index).parameters
    assert "start" in inspect.signature(core.counting_concordance_summary).parameters
    assert "start" in inspect.signature(core.stratified_counting_concordance_summary).parameters
    assert "start" in inspect.signature(core.counting_concordance_rank_rows).parameters
    assert "start" in inspect.signature(core.stratified_counting_concordance_rank_rows).parameters
    assert "start" in inspect.signature(core.counting_concordance_influence_rows).parameters
    assert (
        "start" in inspect.signature(core.stratified_counting_concordance_influence_rows).parameters
    )
    assert "timewt" in inspect.signature(core.concordance_index).parameters
    assert "timewt" in inspect.signature(core.concordance_summary).parameters
    assert "timewt" in inspect.signature(core.stratified_concordance_summary).parameters
    assert "timewt" in inspect.signature(core.concordance_rank_rows).parameters
    assert "timewt" in inspect.signature(core.stratified_concordance_rank_rows).parameters
    assert "timewt" in inspect.signature(core.concordance_influence_rows).parameters
    assert "timewt" in inspect.signature(core.stratified_concordance_influence_rows).parameters
    assert "timewt" in inspect.signature(core.counting_concordance_index).parameters
    assert "timewt" in inspect.signature(core.counting_concordance_summary).parameters
    assert "timewt" in inspect.signature(core.stratified_counting_concordance_summary).parameters
    assert "timewt" in inspect.signature(core.counting_concordance_rank_rows).parameters
    assert "timewt" in inspect.signature(core.stratified_counting_concordance_rank_rows).parameters
    assert "timewt" in inspect.signature(core.counting_concordance_influence_rows).parameters
    assert (
        "timewt"
        in inspect.signature(core.stratified_counting_concordance_influence_rows).parameters
    )
    assert _pyi_function_arg_names(stub_path, "concordance_index") == [
        "time",
        "status",
        "risk_scores",
        "weights",
        "timewt",
    ]
    assert _pyi_function_arg_names(stub_path, "concordance_summary") == [
        "time",
        "status",
        "risk_scores",
        "weights",
        "timewt",
    ]
    assert _pyi_function_arg_names(stub_path, "stratified_concordance_summary") == [
        "time",
        "status",
        "risk_scores",
        "strata",
        "weights",
        "timewt",
    ]
    assert _pyi_function_arg_names(stub_path, "concordance_rank_rows") == [
        "time",
        "status",
        "risk_scores",
        "weights",
        "timewt",
    ]
    assert _pyi_function_arg_names(stub_path, "stratified_concordance_rank_rows") == [
        "time",
        "status",
        "risk_scores",
        "strata",
        "weights",
        "timewt",
    ]
    assert _pyi_function_arg_names(stub_path, "concordance_influence_rows") == [
        "time",
        "status",
        "risk_scores",
        "weights",
        "timewt",
    ]
    assert _pyi_function_arg_names(stub_path, "stratified_concordance_influence_rows") == [
        "time",
        "status",
        "risk_scores",
        "strata",
        "weights",
        "timewt",
    ]
    assert _pyi_function_arg_names(stub_path, "counting_concordance_index") == [
        "start",
        "stop",
        "status",
        "risk_scores",
        "weights",
        "timewt",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "counting_concordance_summary") == [
        "start",
        "stop",
        "status",
        "risk_scores",
        "weights",
        "timewt",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "stratified_counting_concordance_summary") == [
        "start",
        "stop",
        "status",
        "risk_scores",
        "strata",
        "weights",
        "timewt",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "counting_concordance_rank_rows") == [
        "start",
        "stop",
        "status",
        "risk_scores",
        "weights",
        "timewt",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "stratified_counting_concordance_rank_rows") == [
        "start",
        "stop",
        "status",
        "risk_scores",
        "strata",
        "weights",
        "timewt",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "counting_concordance_influence_rows") == [
        "start",
        "stop",
        "status",
        "risk_scores",
        "weights",
        "timewt",
        "timefix",
    ]
    assert _pyi_function_arg_names(stub_path, "stratified_counting_concordance_influence_rows") == [
        "start",
        "stop",
        "status",
        "risk_scores",
        "strata",
        "weights",
        "timewt",
        "timefix",
    ]
    assert "timefix" in inspect.signature(core.counting_concordance_index).parameters
    assert "timefix" in inspect.signature(core.counting_concordance_summary).parameters
    assert "timefix" in inspect.signature(core.stratified_counting_concordance_summary).parameters
    assert "timefix" in inspect.signature(core.counting_concordance_rank_rows).parameters
    assert "timefix" in inspect.signature(core.stratified_counting_concordance_rank_rows).parameters
    assert "timefix" in inspect.signature(core.counting_concordance_influence_rows).parameters
    assert (
        "timefix"
        in inspect.signature(core.stratified_counting_concordance_influence_rows).parameters
    )
    with pytest.raises(ValueError, match="start must be less than stop"):
        core.counting_concordance_summary(
            [1.0],
            [1.0 + 5e-10],
            [1],
            [0.4],
        )
    exact = core.counting_concordance_summary(
        [0.0, 0.0, 0.0],
        [1.0, 1.0 + 5e-10, 2.0],
        [1, 0, 0],
        [0.5, 0.9, 0.1],
        None,
        "n",
        False,
    )
    right_ranks = core.concordance_rank_rows(
        [1.0, 2.0, 3.0, 4.0],
        [1, 1, 1, 0],
        [0.9, 0.6, 0.4, 0.1],
        [2.0, 1.0, 3.0, 1.0],
    )
    stratified_right = core.stratified_concordance_summary(
        [1.0, 2.0, 1.0, 2.0],
        [1, 0, 1, 0],
        [0.9, 0.1, 0.2, 0.8],
        [0, 0, 1, 1],
    )
    stratified_right_ranks = core.stratified_concordance_rank_rows(
        [1.0, 2.0, 1.0, 2.0],
        [1, 0, 1, 0],
        [0.9, 0.1, 0.2, 0.8],
        [0, 0, 1, 1],
    )
    stratified_right_influence, stratified_right_dfbeta, stratified_right_variance = (
        core.stratified_concordance_influence_rows(
            [1.0, 2.0, 1.0, 2.0],
            [1, 0, 1, 0],
            [0.9, 0.1, 0.2, 0.8],
            [0, 0, 1, 1],
        )
    )
    right_influence, right_dfbeta, right_variance = core.concordance_influence_rows(
        [1.0, 2.0, 3.0, 4.0],
        [1, 1, 1, 0],
        [0.9, 0.1, 0.4, 0.2],
        [2.0, 1.0, 3.0, 1.0],
    )
    counting_ranks = core.counting_concordance_rank_rows(
        [0.0, 0.0, 0.5, 1.5],
        [1.0, 2.0, 3.0, 4.0],
        [1, 1, 1, 0],
        [0.9, 0.7, 0.4, 0.1],
    )
    stratified_counting = core.stratified_counting_concordance_summary(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, 2.0],
        [1, 0, 1, 0],
        [0.9, 0.1, 0.2, 0.8],
        [0, 0, 1, 1],
    )
    stratified_counting_ranks = core.stratified_counting_concordance_rank_rows(
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, 2.0],
        [1, 0, 1, 0],
        [0.9, 0.1, 0.2, 0.8],
        [0, 0, 1, 1],
    )
    stratified_counting_influence, stratified_counting_dfbeta, stratified_counting_variance = (
        core.stratified_counting_concordance_influence_rows(
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 2.0],
            [1, 0, 1, 0],
            [0.9, 0.1, 0.2, 0.8],
            [0, 0, 1, 1],
        )
    )
    counting_influence, counting_dfbeta, counting_variance = (
        core.counting_concordance_influence_rows(
            [0.0, 0.0, 0.5, 1.5],
            [1.0, 2.0, 3.0, 4.0],
            [1, 1, 1, 0],
            [0.9, 0.1, 0.4, 0.2],
        )
    )
    fixed = core.counting_concordance_summary(
        [0.0, 0.0, 0.0],
        [1.0, 1.0 + 5e-10, 2.0],
        [1, 0, 0],
        [0.5, 0.9, 0.1],
        None,
        "n",
        True,
    )
    assert exact["concordance"] == pytest.approx(0.5)
    assert fixed["comparable"] == pytest.approx(1.0)
    assert stratified_right["concordance"] == pytest.approx(0.5)
    assert stratified_right["n_event"] == pytest.approx(2.0)
    assert stratified_right_ranks == pytest.approx([(1.0, 0.5, 2.0, 1.0), (1.0, -0.5, 2.0, 1.0)])
    assert len(stratified_right_influence) == 4
    assert stratified_right_variance == pytest.approx(
        sum(value * value for value in stratified_right_dfbeta)
    )
    assert stratified_counting["concordance"] == pytest.approx(0.5)
    assert stratified_counting["n_event"] == pytest.approx(2.0)
    assert stratified_counting_ranks == pytest.approx([(1.0, 0.5, 2.0, 1.0), (1.0, -0.5, 2.0, 1.0)])
    assert len(stratified_counting_influence) == 4
    assert stratified_counting_variance == pytest.approx(
        sum(value * value for value in stratified_counting_dfbeta)
    )
    assert right_ranks[0] == pytest.approx((1.0, 5.0 / 7.0, 7.0, 2.0))
    assert counting_ranks[0] == pytest.approx((1.0, 2.0 / 3.0, 3.0, 1.0))
    assert len(right_influence) == 4
    assert len(counting_influence) == 4
    assert right_variance == pytest.approx(sum(value * value for value in right_dfbeta))
    assert counting_variance == pytest.approx(sum(value * value for value in counting_dfbeta))


def test_core_utility_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "CovariateMatrix",
        "CoxRegressionInput",
        "NaturalSplineKnot",
        "SplineBasisResult",
        "PSpline",
        "nsk",
        "cox_score_residuals",
        "schoenfeld_residuals",
        "concordance",
    } <= stub_names

    assert list(inspect.signature(core.CovariateMatrix).parameters) == [
        "values",
        "n_obs",
        "n_vars",
    ]
    assert list(inspect.signature(core.CoxRegressionInput).parameters) == [
        "covariates",
        "survival",
        "weights",
        "offset",
    ]
    assert list(inspect.signature(core.NaturalSplineKnot).parameters) == [
        "knots",
        "boundary_knots",
        "df",
        "intercept",
    ]
    assert list(inspect.signature(core.SplineBasisResult).parameters) == []
    assert list(inspect.signature(core.PSpline).parameters) == [
        "x",
        "df",
        "theta",
        "eps",
        "method",
        "boundary_knots",
        "intercept",
        "penalty",
    ]

    expected_args = {
        "nsk": ["x", "df", "knots", "boundary_knots"],
        "cox_score_residuals": ["y", "strata", "covar", "score", "weights", "nvar", "method"],
        "schoenfeld_residuals": ["y", "score", "strata", "covar", "nvar", "method"],
        "concordance": ["y", "x", "wt", "timewt", "sortstart", "sortstop"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    assert _pyi_class_annotation_names(stub_path, "CovariateMatrix") == {
        "values",
        "n_obs",
        "n_vars",
    }
    assert _pyi_class_property_names(stub_path, "CoxRegressionInput") == {"n_obs", "n_vars"}
    assert _pyi_class_property_names(stub_path, "NaturalSplineKnot") == {
        "knots",
        "boundary_knots",
        "intercept",
        "df",
    }
    assert _pyi_class_property_names(stub_path, "SplineBasisResult") == {
        "basis",
        "n_rows",
        "n_cols",
        "knots",
        "boundary_knots",
    }
    assert _pyi_class_property_names(stub_path, "PSpline") == {
        "coefficients",
        "fitted",
        "df",
        "eps",
    }

    assert _pyi_class_method_arg_names(stub_path, "CovariateMatrix", "__init__") == [
        "self",
        "values",
        "n_obs",
        "n_vars",
    ]
    assert _pyi_class_method_arg_names(stub_path, "CoxRegressionInput", "__init__") == [
        "self",
        "covariates",
        "survival",
        "weights",
        "offset",
    ]
    assert _pyi_class_method_arg_names(stub_path, "NaturalSplineKnot", "__init__") == [
        "self",
        "knots",
        "boundary_knots",
        "df",
        "intercept",
    ]
    assert _pyi_class_method_arg_names(stub_path, "NaturalSplineKnot", "basis") == ["self", "x"]
    assert _pyi_class_method_arg_names(stub_path, "NaturalSplineKnot", "predict") == [
        "self",
        "x",
        "coef",
    ]
    assert _pyi_class_method_arg_names(stub_path, "PSpline", "__init__") == [
        "self",
        "x",
        "df",
        "theta",
        "eps",
        "method",
        "boundary_knots",
        "intercept",
        "penalty",
    ]
    assert _pyi_class_method_arg_names(stub_path, "PSpline", "fit") == ["self"]
    assert _pyi_class_method_arg_names(stub_path, "PSpline", "predict") == ["self", "new_x"]

    covariates = core.CovariateMatrix([1.0, 0.0, 0.0, 1.0], 2, 2)
    survival_data = core.SurvivalData([1.0, 2.0], [1, 0])
    cox_input = core.CoxRegressionInput(covariates, survival_data)
    default_basis = core.nsk([1.0, 2.0, 3.0, 4.0, 5.0], None, None, None)
    basis = core.nsk([1.0, 2.0, 3.0, 4.0, 5.0], 3, None, None)
    explicit_basis = core.nsk([1.0, 2.0, 3.0, 4.0], None, [2.0, 3.0], (1.0, 4.0))
    spline = core.NaturalSplineKnot([2.0, 3.0], (1.0, 4.0), None, False)
    spline_basis = spline.basis([1.0, 2.0, 3.0])
    pspline = core.PSpline([1.0, 2.0, 3.0, 4.0], 3, 0.1, 1e-6, "GCV", (0.0, 5.0), False, False)

    assert len(covariates) == 2
    assert covariates.shape() == (2, 2)
    assert cox_input.n_obs == 2
    assert cox_input.n_vars == 2
    assert type(basis).__name__ == "SplineBasisResult"
    assert default_basis.n_cols == 1
    assert default_basis.basis == pytest.approx(
        [
            -0.05555555555555554,
            0.22222222222222215,
            0.5,
            0.7777777777777779,
            1.0555555555555556,
        ]
    )
    assert basis.n_rows == 5
    assert basis.n_cols == 3
    assert basis.knots == pytest.approx([2.6666666666666665, 3.333333333333333])
    assert basis.boundary_knots == pytest.approx((1.2, 4.8))
    assert basis.basis[:3] == pytest.approx(
        [-0.30663390663390683, 0.12972972972972977, -0.007507507507507517]
    )
    expected_identity = [
        1.0 if row > 0 and row - 1 == col else 0.0 for row in range(4) for col in range(3)
    ]
    assert explicit_basis.basis == pytest.approx(expected_identity)
    assert spline.df == 3
    assert spline.intercept is False
    assert spline_basis.n_rows == 3
    assert spline_basis.n_cols == 3
    assert spline.predict([1.0, 2.0, 3.0, 4.0], [20.0, 30.0, 40.0]) == pytest.approx(
        [0.0, 20.0, 30.0, 40.0]
    )
    assert pspline.df == 3
    assert pspline.eps == pytest.approx(1e-6)
    assert pspline.fitted is False
    assert pspline.coefficients is None

    y_score = [1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 0.0]
    score = [1.0, 1.0, 1.0, 1.0]
    strata = [0, 0, 0, 0]
    covar = [1.0, 2.0, 3.0, 4.0]
    weights = [1.0, 1.0, 1.0, 1.0]
    y_schoenfeld = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 0.0, 1.0]
    concordance = core.concordance(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1, 2, 1, 2, 1],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        None,
        [0, 1, 2, 3, 4],
    )

    assert core.cox_score_residuals(y_score, strata, covar, score, weights, 1, 0) == pytest.approx(
        [-0.75, -0.25, -0.25, -0.75]
    )
    assert core.schoenfeld_residuals(y_schoenfeld, score, strata, covar, 1, 0) == pytest.approx(
        [-1.5, -1.0, 3.0, 0.0]
    )
    assert concordance["count"] == pytest.approx([0.0, 0.0, 5.0, 0.0, 0.0])


def test_core_nsk_rejects_malformed_inputs():
    setup_survival_import()
    core = importlib.import_module("survival._survival")

    with pytest.raises(ValueError, match="x contains non-finite"):
        core.nsk([1.0, float("nan")], 3, None, None)
    with pytest.raises(ValueError, match="non-zero finite range"):
        core.nsk([1.0, 1.0], 3, None, None)
    with pytest.raises(ValueError, match="df must be at least 1"):
        core.nsk([1.0, 2.0], 0, None, None)
    tied = core.nsk([0.0, 1.0, 1.0, 1.0, 2.0], 4, None, None)
    assert tied.n_cols == 2
    assert tied.knots == pytest.approx([1.0])
    with pytest.raises(ValueError, match="boundary_knots must be finite"):
        core.NaturalSplineKnot(None, (1.0, 1.0), 3, False)
    with pytest.raises(ValueError, match="knots contains non-finite"):
        core.NaturalSplineKnot([float("inf")], (0.0, 3.0), None, False)

    spline = core.NaturalSplineKnot(None, (0.0, 2.0), 1, False)
    with pytest.raises(ValueError, match="coef contains non-finite"):
        spline.predict([0.0, 1.0], [1.0, float("inf")])


def test_qol_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "QALYResult",
        "QTWISTResult",
        "qaly_calculation",
        "qaly_comparison",
        "incremental_cost_effectiveness",
        "qtwist_analysis",
        "qtwist_comparison",
        "qtwist_sensitivity",
    } <= stub_names

    expected_args = {
        "qaly_calculation": [
            "time",
            "status",
            "utility_values",
            "utility_times",
            "discount_rate",
            "horizon",
        ],
        "qaly_comparison": [
            "time_treated",
            "status_treated",
            "utility_treated",
            "time_control",
            "status_control",
            "utility_control",
            "utility_times",
            "discount_rate",
            "horizon",
            "n_bootstrap",
        ],
        "incremental_cost_effectiveness": [
            "qaly_treated",
            "qaly_control",
            "cost_treated",
            "cost_control",
            "wtp_threshold",
        ],
        "qtwist_analysis": [
            "time",
            "status",
            "toxicity_start",
            "toxicity_end",
            "relapse_time",
            "utility_tox",
            "utility_rel",
            "tau",
        ],
        "qtwist_comparison": [
            "time_treated",
            "status_treated",
            "tox_start_treated",
            "tox_end_treated",
            "relapse_treated",
            "time_control",
            "status_control",
            "tox_start_control",
            "tox_end_control",
            "relapse_control",
            "utility_tox",
            "utility_rel",
            "tau",
            "n_bootstrap",
        ],
        "qtwist_sensitivity": [
            "time",
            "status",
            "toxicity_start",
            "toxicity_end",
            "relapse_time",
            "utility_tox_range",
            "utility_rel_range",
            "tau",
        ],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    assert _pyi_class_property_names(stub_path, "QALYResult") == {
        "qaly",
        "life_years",
        "mean_utility",
        "qaly_by_period",
        "discounted_qaly",
        "qaly_se",
        "qaly_ci_lower",
        "qaly_ci_upper",
    }
    assert _pyi_class_property_names(stub_path, "QTWISTResult") == {
        "qtwist",
        "tox",
        "twistt",
        "rel",
        "total_time",
        "utility_tox",
        "utility_rel",
        "qtwist_difference",
        "ci_lower",
        "ci_upper",
    }

    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 0, 1, 0]
    utility_times = [0.0, 2.0]
    treated_utility = [1.0, 0.8]
    control_time = [1.5, 2.5, 3.5, 4.5]
    control_status = [1, 1, 0, 0]
    control_utility = [0.9, 0.7]

    qaly = core.qaly_calculation(time, status, treated_utility, utility_times, 0.03, None)
    qaly_comparison = core.qaly_comparison(
        time,
        status,
        treated_utility,
        control_time,
        control_status,
        control_utility,
        utility_times,
        0.03,
        None,
        10,
    )
    icer = core.incremental_cost_effectiveness(3.0, 2.5, 50000.0, 30000.0, 50000.0)

    toxicity_start = [0.2, None, 0.5, 1.0]
    toxicity_end = [0.6, None, 1.0, 1.5]
    relapse_time = [None, None, 2.5, None]
    control_tox_start = [None, 0.5, None, 1.0]
    control_tox_end = [None, 0.8, None, 1.4]
    control_relapse = [None, 2.0, None, None]

    qtwist = core.qtwist_analysis(
        time,
        status,
        toxicity_start,
        toxicity_end,
        relapse_time,
        0.5,
        0.25,
        None,
    )
    qtwist_comparison = core.qtwist_comparison(
        time,
        status,
        toxicity_start,
        toxicity_end,
        relapse_time,
        control_time,
        control_status,
        control_tox_start,
        control_tox_end,
        control_relapse,
        0.5,
        0.25,
        None,
        10,
    )
    sensitivity = core.qtwist_sensitivity(
        time,
        status,
        toxicity_start,
        toxicity_end,
        relapse_time,
        [0.0, 0.5],
        [0.0, 0.5],
        None,
    )

    assert type(qaly).__name__ == "QALYResult"
    assert qaly.qaly == pytest.approx(2.65)
    assert qaly.life_years == pytest.approx(2.875)
    assert qaly.mean_utility == pytest.approx(0.9217391304347826)
    assert qaly.qaly_by_period == pytest.approx([1.75, 0.9])
    assert qaly.discounted_qaly == pytest.approx(2.522656619631436)
    assert qaly.qaly_ci_lower <= qaly.qaly <= qaly.qaly_ci_upper
    assert [type(item).__name__ for item in qaly_comparison[:2]] == ["QALYResult", "QALYResult"]
    assert qaly_comparison[2] == pytest.approx(qaly_comparison[0].qaly - qaly_comparison[1].qaly)
    assert qaly_comparison[3] <= qaly_comparison[4]
    assert icer == pytest.approx((40000.0, 5000.0, True))

    assert type(qtwist).__name__ == "QTWISTResult"
    assert qtwist.qtwist == pytest.approx(2.23125)
    assert qtwist.tox == pytest.approx(0.35)
    assert qtwist.twistt == pytest.approx(2.025)
    assert qtwist.rel == pytest.approx(0.125)
    assert qtwist.total_time == pytest.approx(2.5)
    assert qtwist.utility_tox == pytest.approx(0.5)
    assert qtwist.utility_rel == pytest.approx(0.25)
    assert qtwist.qtwist_difference is None
    assert qtwist.ci_lower is None
    assert qtwist.ci_upper is None
    assert [type(item).__name__ for item in qtwist_comparison[:2]] == [
        "QTWISTResult",
        "QTWISTResult",
    ]
    assert qtwist_comparison[2] == pytest.approx(
        qtwist_comparison[0].qtwist - qtwist_comparison[1].qtwist
    )
    assert qtwist_comparison[3] <= qtwist_comparison[4]
    assert [(u_tox, u_rel) for u_tox, u_rel, _ in sensitivity] == [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.0),
        (0.5, 0.5),
    ]
    assert [qtwist_value for _, _, qtwist_value in sensitivity] == pytest.approx(
        [2.025, 2.0875, 2.2, 2.2625]
    )


def test_survival_dataset_loaders_are_typed_and_well_formed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    expected_shapes = {
        "load_lung": (228, 10),
        "load_aml": (23, 3),
        "load_veteran": (137, 8),
        "load_ovarian": (26, 6),
        "load_colon": (1858, 16),
        "load_pbc": (418, 20),
        "load_cgd": (203, 16),
        "load_bladder": (340, 7),
        "load_heart": (172, 8),
        "load_kidney": (76, 7),
        "load_rats": (32, 3),
        "load_stanford2": (184, 5),
        "load_udca": (170, 15),
        "load_myeloid": (646, 9),
        "load_flchain": (7874, 11),
        "load_transplant": (815, 6),
        "load_mgus": (241, 12),
        "load_mgus2": (1384, 11),
        "load_diabetic": (394, 8),
        "load_retinopathy": (394, 9),
        "load_gbsg": (686, 11),
        "load_rotterdam": (2982, 15),
        "load_logan": (838, 4),
        "load_nwtco": (4028, 9),
        "load_solder": (900, 6),
        "load_tobin": (20, 3),
        "load_rats2": (253, 6),
        "load_nafld": (17549, 9),
        "load_cgd0": (128, 20),
        "load_pbcseq": (1945, 19),
        "load_hoel": (43, 3),
        "load_myeloma": (61, 8),
        "load_rhdnase": (40, 8),
    }

    assert set(expected_shapes) <= stub_names

    for name, (n_rows, n_cols) in expected_shapes.items():
        loader = getattr(core, name)
        assert list(inspect.signature(loader).parameters) == []
        assert _pyi_function_arg_names(stub_path, name) == []

        data = loader()
        columns = [key for key in data if not key.startswith("_")]

        assert data["_nrow"] == n_rows
        assert data["_ncol"] == n_cols
        assert len(columns) == n_cols
        assert {len(data[column]) for column in columns} == {n_rows}


def test_interpretability_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    interpretability = importlib.import_module("survival.interpretability")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert set(interpretability.__all__) <= stub_names

    for name in interpretability.__all__:
        runtime = getattr(core, name)
        if inspect.isclass(runtime):
            runtime_init_args = list(inspect.signature(runtime).parameters)
            if runtime_init_args:
                assert _pyi_class_method_arg_names(stub_path, name, "__init__") == [
                    "self",
                    *runtime_init_args,
                ]

            runtime_attrs = {
                attr
                for attr, value in runtime.__dict__.items()
                if type(value).__name__ == "getset_descriptor" or type(value) is runtime
            }
            runtime_attrs -= {"__dict__"}
            assert _pyi_class_annotation_names(stub_path, name) == runtime_attrs

            for method_name, method in runtime.__dict__.items():
                if method_name.startswith("_") or type(method).__name__ != "method_descriptor":
                    continue
                runtime_method_args = list(inspect.signature(method).parameters)
                assert (
                    _pyi_class_method_arg_names(stub_path, name, method_name) == runtime_method_args
                )
        else:
            runtime_args = list(inspect.signature(runtime).parameters)
            assert _pyi_function_arg_names(stub_path, name) == runtime_args

    assert interpretability.SurvShapConfig is core.SurvShapConfig
    assert repr(core.AggregationMethod("mean")) == "AggregationMethod.Mean"
    assert repr(core.TimeVaryingTestType("slope")) == "TimeVaryingTestType.SlopeTest"
    assert repr(core.ChangepointMethod("pelt")) == "ChangepointMethod.PELT"
    assert repr(core.CostFunction("l2")) == "CostFunction.L2"
    assert repr(core.GroupingMethod("automatic")) == "GroupingMethod.Automatic"
    assert repr(core.LinkageType("average")) == "LinkageType.Average"
    assert repr(core.ViewRecommendation("global")) == "UseGlobal"

    shap_values = [
        [[0.1, 0.2, 0.3], [0.05, 0.1, 0.15]],
        [[0.2, 0.1, 0.0], [0.1, 0.2, 0.3]],
        [[0.0, 0.1, 0.2], [0.2, 0.1, 0.0]],
        [[0.3, 0.2, 0.1], [0.15, 0.2, 0.25]],
    ]
    covariates = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 1.5]]
    predictions = [0.1, 0.4, 0.7, 1.0]
    survival_predictions = [
        [0.9, 0.8, 0.7],
        [0.8, 0.7, 0.6],
        [0.7, 0.6, 0.5],
        [0.6, 0.5, 0.4],
    ]
    time_points = [1.0, 2.0, 3.0]

    shap_config = core.SurvShapConfig(4, 2, 3, False)
    shap_config.parallel = True
    assert shap_config.parallel is True
    shap_config.parallel = False

    shap = core.survshap(
        [0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.8, 0.7, 0.6],
        [0.9, 0.8, 0.7, 0.7, 0.6, 0.5],
        time_points,
        1,
        2,
        2,
        shap_config,
        core.AggregationMethod.Mean,
    )
    assert type(shap).__name__ == "SurvShapResult"
    assert len(shap.get_sample_shap(0)) == 2
    assert len(shap.get_feature_shap(0)) == 1
    assert len(shap.get_shap_at_time(0)) == 1
    assert len(shap.feature_ranking(core.AggregationMethod.Mean, 1)) == 1
    assert len(shap.mean_absolute_shap()) == 2
    assert shap.check_additivity([0.8, 0.7, 0.6], 10.0) == [True, True, True]
    assert core.aggregate_survshap(shap.shap_values, time_points, core.AggregationMethod.Mean) == [
        0.0,
        0.0,
    ]

    def predict_fn(_x_values, n_rows):
        return [0.5 - 0.01 * index for index in range(n_rows * len(time_points))]

    model_shap = core.survshap_from_model(
        [0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        time_points,
        1,
        2,
        2,
        predict_fn,
        shap_config,
        None,
    )
    assert type(model_shap).__name__ == "SurvShapResult"

    bootstrap = core.survshap_bootstrap(
        [0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.8, 0.7, 0.6],
        [0.9, 0.8, 0.7, 0.7, 0.6, 0.5],
        time_points,
        1,
        2,
        2,
        2,
        0.8,
        shap_config,
    )
    assert type(bootstrap).__name__ == "BootstrapSurvShapResult"
    assert bootstrap.n_bootstrap == 2

    permutation = core.permutation_importance(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4],
        time_points,
        [1.0, 2.0, 3.0, 4.0],
        [1, 1, 0, 1],
        4,
        2,
        2,
        4,
        False,
    )
    assert type(permutation).__name__ == "PermutationImportanceResult"
    assert len(permutation.feature_ranking(None)) == 2

    interactions = core.compute_shap_interactions(
        shap_values,
        time_points,
        2,
        core.AggregationMethod.Mean,
    )
    assert type(interactions).__name__ == "ShapInteractionResult"
    assert len(interactions.get_interaction(0, 1)) == 3
    assert len(interactions.top_interactions(1)) == 1

    time_config = core.TimeVaryingTestConfig(core.TimeVaryingTestType.SlopeTest, 2, 1, 0.05, 10)
    time_varying = core.detect_time_varying_features(shap_values, time_points, 4, 2, time_config)
    assert type(time_varying).__name__ == "TimeVaryingAnalysis"
    assert len(time_varying.results) == 2
    assert time_varying.get_feature_result(0) is not None

    change_config = core.ChangepointConfig(
        core.ChangepointMethod.PELT,
        core.CostFunction.L2,
        0.1,
        1,
        1,
    )
    changepoints = core.detect_changepoints(shap_values, time_points, 4, 2, change_config)
    assert type(changepoints).__name__ == "AllChangepointsResult"
    single_changepoint = core.detect_changepoints_single_series(
        [1.0, 1.1, 3.0],
        time_points,
        change_config,
    )
    assert type(single_changepoint).__name__ == "ChangepointResult"
    assert single_changepoint.get_segment_at(0) == 0

    grouping_config = core.VariableGroupingConfig(
        core.GroupingMethod.Automatic,
        2,
        0.1,
        core.LinkageType.Average,
        10,
        3,
    )
    grouping = core.group_variables(shap_values, 4, 2, 3, grouping_config)
    assert type(grouping).__name__ == "VariableGroupingResult"
    assert grouping.n_features == 2
    assert grouping.get_group(0) is not None
    assert grouping.get_feature_group(0) is not None
    assert grouping.get_group_by_feature(0) is not None

    local_config = core.LocalGlobalConfig(0.3, 0.8, 0.2, 5, 0.9)
    local_global = core.analyze_local_global(
        shap_values,
        [value for row in covariates for value in row],
        4,
        2,
        3,
        local_config,
        1,
    )
    assert type(local_global).__name__ == "LocalGlobalResult"
    assert local_global.summary_statistics.n_features == 2
    assert local_global.get_feature_analysis(0) is not None
    assert isinstance(
        local_global.features_by_recommendation(core.ViewRecommendation.UseGlobal),
        list,
    )

    ale = core.compute_ale(covariates, predictions, 0, 2)
    assert type(ale).__name__ == "ALEResult"
    assert ale.feature_index == 0
    ale_2d = core.compute_ale_2d(covariates, predictions, 0, 1, 2)
    assert type(ale_2d).__name__ == "ALE2DResult"
    assert ale_2d.feature1_index == 0
    time_varying_ale = core.compute_time_varying_ale(
        covariates,
        survival_predictions,
        time_points,
        0,
        2,
    )
    assert len(time_varying_ale) == 3

    friedman = core.compute_friedman_h(covariates, predictions, 0, 1, 3)
    assert type(friedman).__name__ == "FriedmanHResult"
    assert len(core.compute_all_pairwise_interactions(covariates, predictions, None, 3)) == 1
    assert len(core.compute_feature_importance_decomposition(covariates, predictions, 3)) == 2

    ice = core.compute_ice(covariates, predictions, 0, 4, True, None)
    assert type(ice).__name__ == "ICEResult"
    assert len(ice.get_curve(0)) == 4
    dice = core.compute_dice(covariates, predictions, 0, 4, None)
    assert type(dice).__name__ == "DICEResult"
    survival_ice = core.compute_survival_ice(
        covariates,
        survival_predictions,
        time_points,
        0,
        4,
        None,
    )
    assert len(survival_ice) == 3
    assert isinstance(core.detect_heterogeneity(ice, 0.1), list)
    assert len(core.cluster_ice_curves(ice, 2)) == 4


def test_causal_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    causal = importlib.import_module("survival.causal")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert set(causal.__all__) <= stub_names

    for name in causal.__all__:
        runtime = getattr(core, name)
        if inspect.isclass(runtime):
            runtime_init_args = list(inspect.signature(runtime).parameters)
            if runtime_init_args:
                assert _pyi_class_method_arg_names(stub_path, name, "__init__") == [
                    "self",
                    *runtime_init_args,
                ]

            runtime_attrs = {
                attr
                for attr, value in runtime.__dict__.items()
                if type(value).__name__ == "getset_descriptor" or type(value) is runtime
            }
            runtime_attrs -= {"__dict__"}
            assert _pyi_class_annotation_names(stub_path, name) == runtime_attrs

            for method_name, method in runtime.__dict__.items():
                if method_name.startswith("_") or type(method).__name__ != "method_descriptor":
                    continue
                runtime_method_args = list(inspect.signature(method).parameters)
                assert (
                    _pyi_class_method_arg_names(stub_path, name, method_name) == runtime_method_args
                )
        else:
            runtime_args = list(inspect.signature(runtime).parameters)
            assert _pyi_function_arg_names(stub_path, name) == runtime_args

    assert causal.TrialEmulationConfig is core.TrialEmulationConfig
    assert repr(core.CopulaType("independent")) == "CopulaType.Independent"

    n_obs = 12
    covariates = [[1.0, idx / 10.0] for idx in range(n_obs)]
    flat_covariates = [value for row in covariates for value in row]
    treatment = [idx % 2 for idx in range(n_obs)]
    time = [float(idx + 1) for idx in range(n_obs)]
    event = [0 if idx % 4 == 0 else 1 for idx in range(n_obs)]
    outcome = [0.5 + 0.2 * treatment[idx] + 0.01 * idx for idx in range(n_obs)]
    time_points = [3.0, 6.0, 9.0]

    g_comp = core.g_computation(time, event, treatment, flat_covariates, n_obs, 2, 10.0, 5)
    assert type(g_comp).__name__ == "GComputationResult"
    assert len(g_comp.survival_treated) == len(g_comp.time_points)
    g_curves = core.g_computation_survival_curves(
        time,
        event,
        treatment,
        flat_covariates,
        n_obs,
        2,
        time_points,
    )
    assert len(g_curves) == 3
    assert g_curves[0] == pytest.approx(time_points)

    ipcw = core.compute_ipcw_weights(time, event, flat_covariates, n_obs, 2, True, 0.01)
    assert type(ipcw).__name__ == "IPCWResult"
    assert len(ipcw.weights) == n_obs
    ipcw_km = core.ipcw_kaplan_meier(time, event, flat_covariates, n_obs, 2, time_points)
    assert ipcw_km[0] == pytest.approx(time_points)
    ipcw_effect = core.ipcw_treatment_effect(
        time,
        event,
        treatment,
        outcome,
        flat_covariates,
        n_obs,
        2,
        10.0,
    )
    assert type(ipcw_effect).__name__ == "IPCWResult"

    msm = core.marginal_structural_model(
        time,
        event,
        treatment,
        flat_covariates,
        flat_covariates,
        n_obs,
        2,
        2,
        True,
        0.01,
    )
    assert type(msm).__name__ == "MSMResult"
    assert len(msm.coefficients) == 3
    longitudinal_weights = core.compute_longitudinal_iptw(
        [0, 1, 1, 0, 1, 0, 0, 1],
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.2,
            0.5,
            0.4,
            0.6,
            0.5,
            0.7,
            0.6,
            0.8,
            0.7,
            0.9,
            0.8,
            1.0,
        ],
        4,
        2,
        2,
        True,
        0.01,
    )
    assert len(longitudinal_weights) == 4

    trial_config = core.TrialEmulationConfig(1.0, 8.0, True, True, 0.01, 5)
    treatment_time = [0.5 if idx % 2 == 0 else None for idx in range(n_obs)]
    target_trial = core.target_trial_emulation(
        time,
        event,
        treatment_time,
        flat_covariates,
        flat_covariates,
        n_obs,
        2,
        2,
        trial_config,
    )
    assert type(target_trial).__name__ == "TargetTrialResult"
    assert target_trial.n_eligible == n_obs
    sequential_trials = core.sequential_trial_emulation(
        [0.0] * n_obs,
        treatment_time,
        time,
        event,
        flat_covariates,
        n_obs,
        2,
        [0.0],
    )
    assert len(sequential_trials) == 1

    forest, forest_result = core.causal_forest_survival(
        covariates,
        treatment,
        time,
        event,
        8.0,
        core.CausalForestConfig(3, 3, 2, 4, None, False, 0.5, 1),
    )
    assert type(forest).__name__ == "CausalForestSurvival"
    assert type(forest_result).__name__ == "CausalForestResult"
    assert len(forest.predict_cate(covariates[:2])) == 2
    assert len(forest.predict_variance(covariates[:2])) == 2
    assert len(forest.feature_importance()) == 2

    counterfactual = core.estimate_counterfactual_survival(
        covariates,
        treatment,
        time,
        event,
        time_points,
        core.CounterfactualSurvivalConfig(4, [4], 1.0, 0.01, 2, 4, 0.1, 1),
    )
    assert type(counterfactual).__name__ == "CounterfactualSurvivalResult"
    assert len(counterfactual.ite) == n_obs

    sequence_covariates = [
        [[float(idx) / 10.0, float(step)] for step in range(3)] for idx in range(n_obs)
    ]
    sequence_treatment = [[int((idx + step) % 2 == 0) for step in range(3)] for idx in range(n_obs)]
    tv_result = core.estimate_tv_survcaus(
        sequence_covariates,
        sequence_treatment,
        time,
        event,
        time_points,
        core.TVSurvCausConfig(4, 1, 1.0, 0.01, 2, 0.1),
    )
    assert type(tv_result).__name__ == "TVSurvCausResult"
    assert len(tv_result.time_points) == 3

    tmle_config = core.TMLEConfig(2, 0.01, 20, 1e-6, 1)
    tmle = core.tmle_ate(covariates, treatment, outcome, tmle_config)
    assert type(tmle).__name__ == "TMLEResult"
    assert isinstance(tmle.is_significant(0.05), bool)
    tmle_survival = core.tmle_survival(covariates, treatment, time, event, time_points, tmle_config)
    assert type(tmle_survival).__name__ == "TMLESurvivalResult"
    assert tmle_survival.time_points == pytest.approx(time_points)

    copula = core.copula_censoring_model(
        time,
        event,
        [1 - status for status in event],
        flat_covariates,
        core.CopulaCensoringConfig(core.CopulaType.Independent, None, 5, 1e-4, 5),
    )
    assert type(copula).__name__ == "CopulaCensoringResult"
    mnar = core.mnar_sensitivity_survival(
        time,
        event,
        flat_covariates,
        core.MNARSurvivalConfig([-0.5, 0.0, 0.5], "tilt"),
    )
    assert type(mnar).__name__ == "MNARSurvivalResult"
    bounds = core.sensitivity_bounds_survival(
        time,
        event,
        treatment,
        flat_covariates,
        8.0,
        core.SensitivityBoundsConfig([1.0, 1.5], 10, "rosenbaum"),
    )
    assert type(bounds).__name__ == "SensitivityBoundsResult"

    double_ml = core.double_ml_survival(
        covariates,
        treatment,
        outcome,
        time,
        event,
        core.DoubleMLConfig(2, 1, None, 0.01, 1),
    )
    assert type(double_ml).__name__ == "DoubleMLResult"
    assert isinstance(double_ml.is_significant(0.05), bool)
    cate = core.double_ml_cate(
        covariates,
        treatment,
        outcome,
        time,
        event,
        [0 if idx < 6 else 1 for idx in range(n_obs)],
        core.DoubleMLConfig(2, 1, None, 0.01, 1),
    )
    assert type(cate).__name__ == "CATEResult"
    assert len(cate.cate_estimates) == len(cate.group_labels)
    assert len(cate.cate_se) == len(cate.group_sizes)

    g_estimation = core.g_estimation_aft(
        time,
        event,
        [float(value) for value in treatment],
        flat_covariates,
        core.GEstimationConfig(10, 1e-4, "aft"),
    )
    assert type(g_estimation).__name__ == "GEstimationResult"
    iv = core.iv_cox(
        time,
        event,
        [float(value) + 0.1 for value in treatment],
        [float(idx % 3) for idx in range(n_obs)],
        flat_covariates,
        core.IVCoxConfig(10, 1e-4, True, True),
    )
    assert type(iv).__name__ == "IVCoxResult"
    mediation = core.mediation_survival(
        time,
        event,
        [float(value) for value in treatment],
        [0.2 + 0.1 * idx for idx in range(n_obs)],
        flat_covariates,
        core.MediationSurvivalConfig(10, 1e-4, 5, 1),
    )
    assert type(mediation).__name__ == "MediationSurvivalResult"

    rd_n = 24
    rd_time = [float(idx % 8 + 1) for idx in range(rd_n)]
    rd_event = [1 if idx % 5 else 0 for idx in range(rd_n)]
    running_var = [-1.2 + 0.1 * idx for idx in range(rd_n)]
    rd_treatment = [1.0 if value >= 0.0 else 0.0 for value in running_var]
    rd = core.rd_survival(
        rd_time,
        rd_event,
        running_var,
        0.0,
        rd_treatment,
        [],
        core.RDSurvivalConfig(1.2, "triangular", 1, False),
    )
    assert type(rd).__name__ == "RDSurvivalResult"


def test_validation_statistical_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "AnovaCoxphResult",
        "AnovaRow",
        "BootstrapResult",
        "CVResult",
        "LogRankResult",
        "ProportionalityTest",
        "RCLLResult",
        "TestResult",
        "anova_coxph",
        "anova_coxph_single",
        "bootstrap_cox_ci",
        "cv_cox_concordance",
        "fleming_harrington_test",
        "logrank_test",
        "lrt_test",
        "ph_test",
        "rcll",
        "score_test_py",
        "wald_test_py",
    } <= stub_names

    expected_args = {
        "anova_coxph": ["logliks", "dfs", "model_names", "test"],
        "anova_coxph_single": ["loglik_null", "loglik_full", "df_null", "df_full"],
        "bootstrap_cox_ci": [
            "time",
            "status",
            "covariates",
            "weights",
            "n_bootstrap",
            "confidence_level",
            "seed",
        ],
        "cv_cox_concordance": [
            "time",
            "status",
            "covariates",
            "weights",
            "n_folds",
            "shuffle",
            "seed",
        ],
        "fleming_harrington_test": [
            "time",
            "status",
            "group",
            "rho",
            "gamma",
            "entry_times",
        ],
        "logrank_test": ["time", "status", "group", "weight_type", "entry_times"],
        "lrt_test": ["loglik_full", "loglik_reduced", "df"],
        "ph_test": ["schoenfeld_residuals", "event_times", "weights"],
        "rcll": ["survival_predictions", "prediction_times", "event_times", "status", "weights"],
        "score_test_py": ["score_vector", "information_matrix"],
        "wald_test_py": ["coefficients", "std_errors"],
    }
    for name, args in expected_args.items():
        assert list(inspect.signature(getattr(core, name)).parameters) == args
        assert _pyi_function_arg_names(stub_path, name) == args

    expected_properties = {
        "AnovaRow": {"model_name", "loglik", "df", "chisq", "p_value"},
        "AnovaCoxphResult": {"rows", "test_type"},
        "BootstrapResult": {
            "coefficients",
            "std_errors",
            "ci_lower",
            "ci_upper",
            "bootstrap_samples",
        },
        "CVResult": {"fold_scores", "mean_score", "std_score", "fold_coefficients"},
        "LogRankResult": {
            "statistic",
            "p_value",
            "df",
            "observed",
            "expected",
            "variance",
            "weight_type",
        },
        "ProportionalityTest": {
            "variable_names",
            "chi2_values",
            "p_values",
            "global_chi2",
            "global_df",
            "global_p_value",
        },
        "RCLLResult": {
            "rcll",
            "mean_rcll",
            "n_events",
            "n_censored",
            "event_contribution",
            "censored_contribution",
        },
        "TestResult": {"statistic", "p_value", "df", "test_name"},
    }
    for class_name, properties in expected_properties.items():
        assert _pyi_class_property_names(stub_path, class_name) == properties

    expected_init_args = {
        "BootstrapResult": [
            "self",
            "coefficients",
            "std_errors",
            "ci_lower",
            "ci_upper",
            "bootstrap_samples",
        ],
        "CVResult": ["self", "fold_scores", "mean_score", "std_score", "fold_coefficients"],
        "LogRankResult": [
            "self",
            "statistic",
            "p_value",
            "df",
            "observed",
            "expected",
            "variance",
            "weight_type",
        ],
        "ProportionalityTest": [
            "self",
            "variable_names",
            "chi2_values",
            "p_values",
            "global_chi2",
            "global_df",
            "global_p_value",
        ],
        "RCLLResult": [
            "self",
            "rcll",
            "mean_rcll",
            "n_events",
            "n_censored",
            "event_contribution",
            "censored_contribution",
        ],
        "TestResult": ["self", "statistic", "df", "p_value", "test_name"],
    }
    for class_name, args in expected_init_args.items():
        assert list(inspect.signature(getattr(core, class_name)).parameters) == args[1:]
        assert _pyi_class_method_arg_names(stub_path, class_name, "__init__") == args

    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 1, 0, 1]
    group = [0, 0, 1, 1]
    logrank = core.logrank_test(time, status, group, None, None)
    fleming = core.fleming_harrington_test(time, status, group, 1.0, 0.0, None)

    assert type(logrank).__name__ == "LogRankResult"
    assert logrank.statistic == pytest.approx(2.8823529411764715)
    assert logrank.p_value == pytest.approx(0.08955507441493349)
    assert logrank.df == 1
    assert logrank.observed == pytest.approx([2.0, 1.0])
    assert logrank.expected == pytest.approx([0.8333333333333333, 2.1666666666666665])
    assert logrank.variance == pytest.approx(0.4722222222222222)
    assert logrank.weight_type == "LogRank"
    assert fleming.weight_type == "FlemingHarrington(p=1, q=0)"
    assert fleming.statistic == pytest.approx(2.6666666666666665)

    anova = core.anova_coxph([-10.0, -8.0, -7.5], [1, 2, 3], ["null", "m1", "m2"], "LRT")
    single = core.anova_coxph_single(-10.0, -8.0, 1, 2)

    assert type(anova).__name__ == "AnovaCoxphResult"
    assert anova.test_type == "LRT"
    assert [row.model_name for row in anova.rows] == ["null", "m1", "m2"]
    assert anova.rows[1].chisq == pytest.approx(4.0)
    assert anova.rows[1].p_value == pytest.approx(0.04550026389168993)
    assert "Analysis of Deviance Table" in anova.to_table()
    assert [row.model_name for row in single.rows] == ["Null", "Full"]
    assert single.rows[1].chisq == pytest.approx(4.0)

    lrt = core.lrt_test(-10.0, -12.0, 1)
    wald = core.wald_test_py([1.0, 2.0], [1.0, 2.0])
    score = core.score_test_py([1.0], [[2.0]])
    ph = core.ph_test(
        [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]],
        [1.0, 2.0, 3.0, 4.0],
        None,
    )

    assert (lrt.test_name, lrt.statistic, lrt.df) == ("LikelihoodRatioTest", pytest.approx(4.0), 1)
    assert (wald.test_name, wald.statistic, wald.df) == ("WaldTest", pytest.approx(2.0), 2)
    assert (score.test_name, score.statistic, score.df) == ("ScoreTest", pytest.approx(0.5), 1)
    assert ph.variable_names == ["var0", "var1"]
    assert ph.chi2_values == pytest.approx([2.0, 2.0])
    assert ph.global_chi2 == pytest.approx(4.0)
    assert ph.global_df == 2

    survival_predictions = [
        [0.95, 0.85, 0.70],
        [0.90, 0.75, 0.55],
        [0.98, 0.92, 0.80],
    ]
    rcll = core.rcll(
        survival_predictions,
        [1.0, 2.0, 3.0],
        [2.5, 1.5, 3.0],
        [1, 1, 0],
        [1.0, 2.0, 1.0],
    )

    assert type(rcll).__name__ == "RCLLResult"
    assert rcll.rcll == pytest.approx(5.914503505971853)
    assert rcll.mean_rcll == pytest.approx(1.4786258764929632)
    assert rcll.n_events == 2
    assert rcll.n_censored == 1
    assert rcll.event_contribution > rcll.censored_contribution

    bootstrap = core.bootstrap_cox_ci(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1, 1, 0, 1, 0, 1],
        [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]],
        None,
        4,
        0.9,
        123,
    )
    cv = core.cv_cox_concordance(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1, 1, 0, 1, 0, 1],
        [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]],
        None,
        2,
        False,
        123,
    )

    assert type(bootstrap).__name__ == "BootstrapResult"
    assert len(bootstrap.coefficients) == 1
    assert len(bootstrap.std_errors) == 1
    assert len(bootstrap.ci_lower) == 1
    assert len(bootstrap.ci_upper) == 1
    assert len(bootstrap.bootstrap_samples) == 4
    assert all(len(sample) == 1 for sample in bootstrap.bootstrap_samples)
    assert type(cv).__name__ == "CVResult"
    assert len(cv.fold_scores) == 2
    assert cv.mean_score == pytest.approx(0.0)
    assert cv.std_score == pytest.approx(0.0)
    assert len(cv.fold_coefficients) == 2
    assert all(len(coefficients) == 1 for coefficients in cv.fold_coefficients)


def test_concordance_metric_bindings_are_typed():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "UnoCIndexResult",
        "ConcordanceComparisonResult",
        "CIndexDecompositionResult",
        "GonenHellerResult",
        "TimeDepAUCResult",
        "CumulativeDynamicAUCResult",
        "uno_c_index",
        "compare_uno_c_indices",
        "c_index_decomposition",
        "gonen_heller_concordance",
        "time_dependent_auc",
        "cumulative_dynamic_auc",
    } <= stub_names

    assert list(inspect.signature(core.uno_c_index).parameters) == [
        "time",
        "status",
        "risk_score",
        "tau",
    ]
    assert list(inspect.signature(core.compare_uno_c_indices).parameters) == [
        "time",
        "status",
        "risk_score_1",
        "risk_score_2",
        "tau",
    ]
    assert list(inspect.signature(core.c_index_decomposition).parameters) == [
        "time",
        "status",
        "risk_score",
        "tau",
    ]
    assert list(inspect.signature(core.gonen_heller_concordance).parameters) == ["linear_predictor"]
    assert list(inspect.signature(core.time_dependent_auc).parameters) == [
        "time",
        "status",
        "marker",
        "t",
    ]
    assert list(inspect.signature(core.cumulative_dynamic_auc).parameters) == [
        "time",
        "status",
        "marker",
        "times",
    ]
    assert _pyi_function_arg_names(stub_path, "uno_c_index") == [
        "time",
        "status",
        "risk_score",
        "tau",
    ]
    assert _pyi_function_arg_names(stub_path, "compare_uno_c_indices") == [
        "time",
        "status",
        "risk_score_1",
        "risk_score_2",
        "tau",
    ]
    assert _pyi_function_arg_names(stub_path, "c_index_decomposition") == [
        "time",
        "status",
        "risk_score",
        "tau",
    ]
    assert _pyi_function_arg_names(stub_path, "gonen_heller_concordance") == ["linear_predictor"]
    assert _pyi_function_arg_names(stub_path, "time_dependent_auc") == [
        "time",
        "status",
        "marker",
        "t",
    ]
    assert _pyi_function_arg_names(stub_path, "cumulative_dynamic_auc") == [
        "time",
        "status",
        "marker",
        "times",
    ]
    assert _pyi_class_property_names(stub_path, "UnoCIndexResult") == {
        "c_index",
        "concordant",
        "discordant",
        "tied_risk",
        "comparable_pairs",
        "variance",
        "std_error",
        "ci_lower",
        "ci_upper",
        "tau",
    }
    assert _pyi_class_property_names(stub_path, "ConcordanceComparisonResult") == {
        "c_index_1",
        "c_index_2",
        "difference",
        "variance_diff",
        "std_error_diff",
        "z_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
    }
    assert _pyi_class_property_names(stub_path, "CIndexDecompositionResult") == {
        "c_index",
        "c_index_ee",
        "c_index_ec",
        "alpha",
        "n_event_event_pairs",
        "n_event_censored_pairs",
        "concordant_ee",
        "concordant_ec",
        "discordant_ee",
        "discordant_ec",
        "tied_ee",
        "tied_ec",
    }
    assert _pyi_class_property_names(stub_path, "GonenHellerResult") == {
        "cpe",
        "n_pairs",
        "n_ties",
        "variance",
        "std_error",
        "ci_lower",
        "ci_upper",
    }
    assert _pyi_class_property_names(stub_path, "TimeDepAUCResult") == {
        "auc",
        "time",
        "n_cases",
        "n_controls",
        "std_error",
        "ci_lower",
        "ci_upper",
    }
    assert _pyi_class_property_names(stub_path, "CumulativeDynamicAUCResult") == {
        "times",
        "auc",
        "mean_auc",
        "integrated_auc",
        "n_cases",
        "n_controls",
    }

    time = [1.0, 2.0, 3.0, 4.0]
    status = [1, 1, 0, 1]
    risk = [0.9, 0.7, 0.4, 0.2]
    reverse_risk = list(reversed(risk))

    uno = core.uno_c_index(time, status, risk, None)
    comparison = core.compare_uno_c_indices(time, status, risk, reverse_risk, None)
    decomposition = core.c_index_decomposition(time, status, risk, None)
    gonen = core.gonen_heller_concordance(risk)
    auc = core.time_dependent_auc(time, status, risk, 2.5)
    cumulative_auc = core.cumulative_dynamic_auc(time, status, risk, [1.5, 2.5, 3.5])

    assert type(uno).__name__ == "UnoCIndexResult"
    assert uno.c_index == pytest.approx(1.0)
    assert uno.comparable_pairs > 0.0
    assert type(comparison).__name__ == "ConcordanceComparisonResult"
    assert comparison.c_index_1 > comparison.c_index_2
    assert type(decomposition).__name__ == "CIndexDecompositionResult"
    assert decomposition.n_event_event_pairs > 0
    assert type(gonen).__name__ == "GonenHellerResult"
    assert 0.0 <= gonen.cpe <= 1.0
    assert type(auc).__name__ == "TimeDepAUCResult"
    assert auc.time == pytest.approx(2.5)
    assert type(cumulative_auc).__name__ == "CumulativeDynamicAUCResult"
    assert cumulative_auc.times == pytest.approx([1.5, 2.5, 3.5])
    assert len(cumulative_auc.auc) == 3


def test_aareg_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "AaregOptions",
        "AaregResult",
        "AaregConfidenceInterval",
        "AaregFitDetails",
        "AaregDiagnostics",
        "aareg",
    } <= stub_names
    assert list(inspect.signature(core.AaregOptions).parameters) == [
        "formula",
        "data",
        "variable_names",
        "max_iter",
    ]
    assert list(inspect.signature(core.aareg).parameters) == ["options"]
    assert _pyi_class_method_arg_names(stub_path, "AaregOptions", "__init__") == [
        "self",
        "formula",
        "data",
        "variable_names",
        "max_iter",
    ]
    assert _pyi_function_arg_names(stub_path, "aareg") == ["options"]
    assert _pyi_class_annotation_names(stub_path, "AaregResult") == {
        "coefficients",
        "standard_errors",
        "confidence_intervals",
        "p_values",
        "goodness_of_fit",
        "fit_details",
        "residuals",
        "diagnostics",
    }
    assert _pyi_class_annotation_names(stub_path, "AaregConfidenceInterval") == {
        "lower_bound",
        "upper_bound",
    }
    assert _pyi_class_annotation_names(stub_path, "AaregFitDetails") == {
        "iterations",
        "converged",
        "final_objective_value",
        "convergence_threshold",
        "change_in_objective",
        "max_iterations",
        "optimization_method",
        "warnings",
    }
    assert _pyi_class_annotation_names(stub_path, "AaregDiagnostics") == {
        "dfbetas",
        "cooks_distance",
        "leverage",
        "deviance_residuals",
        "martingale_residuals",
        "schoenfeld_residuals",
        "score_residuals",
        "additional_measures",
    }

    options = core.AaregOptions(
        "time ~ x",
        [[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 0.0]],
        ["time", "x"],
        10,
    )
    result = core.aareg(options)

    assert type(result).__name__ == "AaregResult"
    assert len(result.coefficients) == 2
    assert type(result.confidence_intervals[0]).__name__ == "AaregConfidenceInterval"
    assert type(result.fit_details).__name__ == "AaregFitDetails"
    assert type(result.diagnostics).__name__ == "AaregDiagnostics"


def test_finegray_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {
        "FineGrayOutput",
        "FineGrayResult",
        "CompetingRisksCIF",
        "finegray",
        "finegray_regression",
        "competing_risks_cif",
    } <= stub_names
    assert list(inspect.signature(core.finegray).parameters) == [
        "tstart",
        "tstop",
        "ctime",
        "cprob",
        "extend",
        "keep",
    ]
    assert list(inspect.signature(core.finegray_regression).parameters) == [
        "time",
        "status",
        "covariates",
        "event_type",
        "max_iter",
        "eps",
    ]
    assert list(inspect.signature(core.competing_risks_cif).parameters) == [
        "time",
        "status",
        "event_type",
        "confidence_level",
    ]
    assert _pyi_function_arg_names(stub_path, "finegray") == [
        "tstart",
        "tstop",
        "ctime",
        "cprob",
        "extend",
        "keep",
    ]
    assert _pyi_function_arg_names(stub_path, "finegray_regression") == [
        "time",
        "status",
        "covariates",
        "event_type",
        "max_iter",
        "eps",
    ]
    assert _pyi_function_arg_names(stub_path, "competing_risks_cif") == [
        "time",
        "status",
        "event_type",
        "confidence_level",
    ]
    assert _pyi_class_property_names(stub_path, "FineGrayOutput") == {
        "row",
        "start",
        "end",
        "wt",
        "add",
    }
    assert _pyi_class_property_names(stub_path, "FineGrayResult") == {
        "coefficients",
        "std_errors",
        "z_scores",
        "p_values",
        "ci_lower",
        "ci_upper",
        "variance_matrix",
        "log_likelihood",
        "log_likelihood_null",
        "n_events",
        "n_competing",
        "n_censored",
        "n_observations",
        "event_type",
        "convergence",
        "iterations",
    }
    assert _pyi_class_property_names(stub_path, "CompetingRisksCIF") == {
        "times",
        "cif",
        "variance",
        "ci_lower",
        "ci_upper",
        "n_risk",
        "n_events",
        "event_type",
    }

    expanded = core.finegray(
        [0.0, 0.0],
        [1.0, 2.0],
        [0.5, 1.5],
        [0.9, 0.8],
        [True, False],
        [True, True],
    )
    assert type(expanded).__name__ == "FineGrayOutput"
    assert len(expanded.row) >= 2

    model = core.finegray_regression(
        [1.0, 2.0, 3.0, 4.0],
        [1, 2, 0, 1],
        [[0.0], [1.0], [0.5], [1.5]],
        1,
        max_iter=5,
        eps=1e-8,
    )
    assert type(model).__name__ == "FineGrayResult"
    assert len(model.hazard_ratio()) == 1

    cif = core.competing_risks_cif([1.0, 2.0, 3.0, 4.0], [1, 2, 0, 1], 1)
    assert type(cif).__name__ == "CompetingRisksCIF"
    assert len(cif.times) == len(cif.cif)


def test_legacy_cox_model_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"

    assert list(inspect.signature(core.Subject).parameters) == [
        "id",
        "covariates",
        "is_case",
        "is_subcohort",
        "stratum",
    ]
    assert _pyi_class_method_arg_names(stub_path, "Subject", "__init__") == [
        "self",
        "id",
        "covariates",
        "is_case",
        "is_subcohort",
        "stratum",
    ]

    method_args = {
        "new_with_data": ["covariates", "event_times", "censoring"],
        "predict_survival": ["self", "time"],
        "survival_curve": ["self", "covariates", "time_points"],
        "cumulative_hazard": ["self", "covariates"],
        "predicted_survival_time": ["self", "covariates", "percentile"],
        "restricted_mean_survival_time": ["self", "covariates", "tau"],
        "brier_score": ["self", "time"],
        "std_errors": ["self"],
        "vcov": ["self"],
        "log_likelihood": ["self"],
        "n_observations": ["self"],
        "n_events": ["self"],
    }
    for method_name, expected_args in method_args.items():
        runtime_args = list(inspect.signature(getattr(core.CoxPHModel, method_name)).parameters)
        assert runtime_args == expected_args
        assert _pyi_class_method_arg_names(stub_path, "CoxPHModel", method_name) == expected_args

    assert (
        _pyi_class_method_return(stub_path, "CoxPHModel", "cumulative_hazard")
        == "tuple[list[float], list[list[float]]]"
    )
    assert (
        _pyi_class_method_return(stub_path, "CoxPHModel", "hazard_ratios_with_ci")
        == "tuple[list[float], list[float], list[float]]"
    )
    assert (
        _pyi_class_method_return(stub_path, "CoxPHModel", "predicted_survival_time")
        == "list[float | None]"
    )

    subject = core.Subject(
        id=1,
        covariates=[1.0, 2.0],
        is_case=True,
        is_subcohort=False,
        stratum=0,
    )
    assert subject.id == 1
    assert subject.covariates == [1.0, 2.0]
    subject.stratum = 2
    assert subject.stratum == 2

    model = core.CoxPHModel.new_with_data(
        [[0.0, 1.0], [1.0, 0.0], [1.5, 1.0], [2.0, 0.5]],
        [1.0, 2.0, 3.0, 4.0],
        [1, 0, 1, 1],
    )
    model.fit(n_iters=3)

    assert len(model.coefficients) == 1
    assert len(model.coefficients[0]) == 2
    assert isinstance(model.predict_survival(2.0), float)
    times, curves = model.survival_curve([[0.0, 1.0]], None)
    assert len(times) == 4
    assert len(curves) == 1
    hazard_times, hazards = model.cumulative_hazard([[0.0, 1.0]])
    assert hazard_times == times
    assert len(hazards) == 1
    hazard_ratios, ci_lower, ci_upper = model.hazard_ratios_with_ci()
    assert len(hazard_ratios) == len(ci_lower) == len(ci_upper) == 2
    assert len(model.restricted_mean_survival_time([[0.0, 1.0]], 3.0)) == 1
    assert isinstance(model.brier_score(), float)
    assert isinstance(model.brier_score(2.0), float)
    assert len(model.std_errors()) == 2
    assert len(model.vcov()) == 2
    assert isinstance(model.log_likelihood(), float)
    assert model.n_observations() == 4
    assert model.n_events() == 3


def test_case_cohort_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {"CchMethod", "CohortData"} <= stub_names
    assert _pyi_class_method_arg_names(stub_path, "CohortData", "add_subject") == [
        "self",
        "subject",
    ]
    assert _pyi_class_method_arg_names(stub_path, "CohortData", "get_subject") == [
        "self",
        "index",
    ]
    assert _pyi_class_method_arg_names(stub_path, "CohortData", "fit") == [
        "self",
        "method",
        "max_iter",
    ]

    cohort = core.CohortData.new()
    assert len(cohort) == 0
    assert cohort.is_empty() is True
    subject = core.Subject(1, [0.1], True, True, 0)
    cohort.add_subject(subject)
    assert len(cohort) == 1
    assert cohort.get_subject(0).id == 1


def test_clogit_bindings_are_typed_to_runtime_surface():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    stub_path = PACKAGE_ROOT / "_survival.pyi"
    stub_names = _pyi_top_level_names(stub_path)

    assert {"ClogitDataSet", "ConditionalLogisticRegression"} <= stub_names
    assert list(inspect.signature(core.ClogitDataSet.add_observation).parameters) == [
        "self",
        "case_control_status",
        "stratum",
        "covariates",
    ]
    assert _pyi_class_method_arg_names(stub_path, "ClogitDataSet", "add_observation") == [
        "self",
        "case_control_status",
        "stratum",
        "covariates",
    ]
    assert _pyi_class_method_arg_names(stub_path, "ClogitDataSet", "__len__") == ["self"]
    assert _pyi_class_method_arg_names(stub_path, "ClogitDataSet", "is_empty") == ["self"]
    assert _pyi_class_method_arg_names(
        stub_path,
        "ConditionalLogisticRegression",
        "__init__",
    ) == ["self", "data", "max_iter", "tol"]
    assert _pyi_class_method_arg_names(
        stub_path,
        "ConditionalLogisticRegression",
        "predict",
    ) == ["self", "covariates"]
    assert _pyi_class_property_names(stub_path, "ConditionalLogisticRegression") == {
        "coefficients",
        "max_iter",
        "tol",
        "iterations",
        "converged",
    }

    dataset = core.ClogitDataSet()
    assert len(dataset) == 0
    assert dataset.is_empty() is True
    dataset.add_observation(1, 0, [2.0])
    dataset.add_observation(0, 0, [1.0])
    assert len(dataset) == 2

    model = core.ConditionalLogisticRegression(dataset, max_iter=3, tol=1e-6)
    model.fit()
    assert len(model.coefficients) == 1
    assert isinstance(model.predict([1.5]), float)
    assert len(model.odds_ratios()) == 1


def test_init_stub_references_existing_python_symbols():
    top_level_names = _top_level_names()
    sklearn_names = _sklearn_names()
    tree = ast.parse((PACKAGE_ROOT / "__init__.pyi").read_text())
    missing: list[tuple[str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        imported = {alias.name for alias in node.names if alias.name != "*"}

        if node.module == "_survival":
            continue
        if node.module == "sklearn_compat":
            missing.extend(
                ("python/survival/__init__.pyi", name) for name in sorted(imported - sklearn_names)
            )
        elif node.level == 1 and node.module is None:
            missing.extend(
                ("python/survival/__init__.pyi", name)
                for name in sorted(imported - top_level_names)
            )

    assert missing == [], _format_missing(missing)


def test_sklearn_compat_stub_tracks_public_exports():
    setup_survival_import()
    sklearn_compat = importlib.import_module("survival.sklearn_compat")
    stub_path = PACKAGE_ROOT / "sklearn_compat.pyi"

    assert _pyi_top_level_names(stub_path) == set(sklearn_compat.__all__)
    assert _pyi_function_arg_names(stub_path, "iter_chunks") == ["X", "batch_size"]
    assert _pyi_function_arg_names(stub_path, "predict_large_dataset") == [
        "estimator",
        "X",
        "batch_size",
        "output_file",
        "verbose",
    ]
    assert _pyi_function_arg_names(stub_path, "survival_curves_to_disk") == [
        "estimator",
        "X",
        "output_file",
        "batch_size",
        "verbose",
    ]
    assert (
        _pyi_function_return(stub_path, "survival_curves_to_disk")
        == "tuple[NDArray[np.float64], np.memmap]"
    )
    for method_name, expected in {
        "predict_batched": ["self", "X", "batch_size"],
        "predict_survival_batched": ["self", "X", "batch_size"],
        "predict_to_array": ["self", "X", "batch_size", "out"],
    }.items():
        assert _pyi_class_method_arg_names(stub_path, "StreamingMixin", method_name) == expected


def test_readme_python_examples_reference_public_symbols():
    manifest = set(_manifest_bindings())
    module_exports = _binding_names_by_module()
    top_level_names = _top_level_names()
    sklearn_names = _sklearn_names()
    module_names = set(module_exports) | {"sklearn_compat", "_survival"}
    missing: list[tuple[str, str]] = []

    for index, block in enumerate(_readme_python_blocks(), start=1):
        tree = ast.parse(block, filename=f"README.md python block {index}")
        aliases: dict[str, set[str]] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "survival":
                    for alias in node.names:
                        name = alias.name
                        if name not in top_level_names:
                            missing.append((f"README.md python block {index}", name))
                            continue
                        if name in module_exports:
                            aliases[alias.asname or name] = (
                                module_exports[name] | MODULE_METADATA_NAMES
                            )
                elif module.startswith("survival."):
                    submodule = module.removeprefix("survival.")
                    if submodule not in module_names:
                        missing.append((f"README.md python block {index}", module))
                        continue
                    surface = (
                        manifest
                        if submodule == "_survival"
                        else sklearn_names
                        if submodule == "sklearn_compat"
                        else module_exports[submodule]
                    )
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        if alias.name not in surface:
                            missing.append(
                                (f"README.md python block {index}", f"{module}.{alias.name}")
                            )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name == "survival":
                        aliases[alias.asname or "survival"] = top_level_names
                    elif name.startswith("survival."):
                        submodule = name.removeprefix("survival.")
                        if submodule not in module_names:
                            missing.append((f"README.md python block {index}", name))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            surface = aliases.get(node.value.id)
            if surface is not None and node.attr not in surface:
                missing.append((f"README.md python block {index}", f"{node.value.id}.{node.attr}"))

    assert missing == [], _format_missing(missing)


def test_runtime_extension_exports_manifest_symbols():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    expected = set(_manifest_bindings())
    if not _has_ml_bindings(core):
        expected -= set(_feature_bindings()["ml"])

    missing = [name for name in sorted(expected) if not hasattr(core, name)]

    assert missing == []


def test_feature_gated_module_exports_follow_runtime_features():
    setup_survival_import()
    core = importlib.import_module("survival._survival")
    survival = importlib.import_module("survival")

    if _has_ml_bindings(core):
        assert "deep_surv" in survival.ml.__all__
        return

    assert "deep_surv" not in survival.ml.__all__
    try:
        _ = survival.ml.deep_surv
    except AttributeError as exc:
        assert "built without the 'ml' feature" in str(exc)
    else:
        raise AssertionError("deep_surv should be unavailable without the ml feature")


def test_package_root_marks_curated_and_legacy_exports():
    setup_survival_import()
    survival = importlib.import_module("survival")

    assert "regression" in survival.__all__
    assert "StrataFactor" in survival.__all__
    assert "Surv" in survival.__all__
    assert "FineGrayOutput" in survival.__all__
    assert "RateTable" in survival.__all__
    assert "PyearsResult" in survival.__all__
    assert "SurvObrienResult" in survival.__all__
    assert "SurvExpResult" in survival.__all__
    assert "aic" in survival.__all__
    assert "aeqSurv" in survival.__all__
    assert "anova" in survival.__all__
    assert "as_data_frame" in survival.__all__
    assert "basehaz" in survival.__all__
    assert "bcloglog" in survival.__all__
    assert "cipoisson" in survival.__all__
    assert "blog" in survival.__all__
    assert "blogit" in survival.__all__
    assert "bprobit" in survival.__all__
    assert "coef" in survival.__all__
    assert "coef_names" in survival.__all__
    assert "clogit" in survival.__all__
    assert "confint" in survival.__all__
    assert "cox_zph" in survival.__all__
    assert "coxph_detail" in survival.__all__
    assert "df_residual" in survival.__all__
    assert "degrees_freedom" in survival.__all__
    assert "dsurvreg" in survival.__all__
    assert "bic" in survival.__all__
    assert "extract_aic" in survival.__all__
    assert "fitted" in survival.__all__
    assert "finegray" in survival.__all__
    assert "format_surv" in survival.__all__
    assert "is_na_surv" in survival.__all__
    assert "is_ratetable" in survival.__all__
    assert "is_surv" in survival.__all__
    assert "loglik" in survival.__all__
    assert "model_formula" in survival.__all__
    assert "model_frame" in survival.__all__
    assert "model_matrix" in survival.__all__
    assert "model_weights" in survival.__all__
    assert "model_summary" in survival.__all__
    assert "nobs" in survival.__all__
    assert "pyears" in survival.__all__
    assert "pseudo" in survival.__all__
    assert "psurvreg" in survival.__all__
    assert "qsurvreg" in survival.__all__
    assert "ratetableDate" in survival.__all__
    assert "rsurvreg" in survival.__all__
    assert "rttright" in survival.__all__
    assert "strata" in survival.__all__
    assert "survcheck" in survival.__all__
    assert "survConcordance" in survival.__all__
    assert "survConcordance_fit" in survival.__all__
    assert "survcondense" in survival.__all__
    assert "survexp" in survival.__all__
    assert "survexp_individual" in survival.__all__
    assert "survexp_mn" in survival.__all__
    assert "survobrien" in survival.__all__
    assert "survexp_us" in survival.__all__
    assert "survexp_usr" in survival.__all__
    assert "survfit0" in survival.__all__
    assert "survSplit" in survival.__all__
    assert "survreg" in survival.__all__
    assert "vcov" in survival.__all__
    assert "validation" in survival.__all__
    assert "ridge_fit" not in survival.__all__
    assert "ridge_fit" in survival.__deprecated_root_exports__
    assert "ridge_fit" not in vars(survival)
    assert survival.StrataFactor is survival.r_api.StrataFactor
    assert survival.Surv is survival.r_api.Surv
    assert survival.FineGrayOutput is survival.r_api.FineGrayOutput
    assert survival.RateTable is survival.r_api.RateTable
    assert survival.PyearsResult is survival.r_api.PyearsResult
    assert survival.SurvObrienResult is survival.r_api.SurvObrienResult
    assert survival.SurvExpResult is survival.r_api.SurvExpResult
    assert survival.aic is survival.r_api.aic
    assert survival.aeqSurv is survival.r_api.aeqSurv
    assert survival.anova is survival.r_api.anova
    assert survival.as_data_frame is survival.r_api.as_data_frame
    assert survival.basehaz is survival.r_api.basehaz
    assert survival.bcloglog is survival.r_api.bcloglog
    assert survival.cipoisson is survival.r_api.cipoisson
    assert survival.blog is survival.r_api.blog
    assert survival.blogit is survival.r_api.blogit
    assert survival.bprobit is survival.r_api.bprobit
    assert survival.coef is survival.r_api.coef
    assert survival.survfit0 is survival.r_api.survfit0
    assert survival.coef_names is survival.r_api.coef_names
    assert survival.clogit is survival.r_api.clogit
    assert survival.confint is survival.r_api.confint
    assert survival.cox_zph is survival.r_api.cox_zph
    assert survival.coxph_detail is survival.r_api.coxph_detail
    assert survival.df_residual is survival.r_api.df_residual
    assert survival.degrees_freedom is survival.r_api.degrees_freedom
    assert survival.dsurvreg is survival.r_api.dsurvreg
    assert survival.bic is survival.r_api.bic
    assert survival.extract_aic is survival.r_api.extract_aic
    assert survival.fitted is survival.r_api.fitted
    assert survival.finegray is survival.r_api.finegray
    assert survival.format_surv is survival.r_api.format_surv
    assert survival.is_na_surv is survival.r_api.is_na_surv
    assert survival.is_ratetable is survival.r_api.is_ratetable
    assert survival.is_surv is survival.r_api.is_surv
    assert survival.is_surv(survival.Surv([1.0])) is True
    assert survival.is_surv([1.0]) is False
    assert survival.loglik is survival.r_api.loglik
    assert survival.model_formula is survival.r_api.model_formula
    assert survival.model_frame is survival.r_api.model_frame
    assert survival.model_matrix is survival.r_api.model_matrix
    assert survival.model_weights is survival.r_api.model_weights
    assert survival.model_summary is survival.r_api.model_summary
    assert survival.neardate is survival.r_api.neardate
    assert survival.neardate is not survival.data_prep.neardate
    assert survival.nobs is survival.r_api.nobs
    assert survival.pyears is survival.r_api.pyears
    assert survival.pseudo is survival.r_api.pseudo
    assert survival.psurvreg is survival.r_api.psurvreg
    assert survival.qsurvreg is survival.r_api.qsurvreg
    assert survival.ratetableDate is survival.r_api.ratetableDate
    assert survival.rsurvreg is survival.r_api.rsurvreg
    assert survival.rttright is survival.r_api.rttright
    assert survival.strata is survival.r_api.strata
    assert survival.survcheck is survival.r_api.survcheck
    assert survival.survConcordance is survival.r_api.survConcordance
    assert survival.survConcordance_fit is survival.r_api.survConcordance_fit
    assert survival.survcondense is survival.r_api.survcondense
    assert survival.survexp is survival.r_api.survexp
    assert survival.survexp_individual is survival.r_api.survexp_individual
    assert survival.survexp_mn is survival.r_api.survexp_mn
    assert survival.survobrien is survival.r_api.survobrien
    assert survival.survexp_us is survival.r_api.survexp_us
    assert survival.survexp_usr is survival.r_api.survexp_usr
    assert survival.survSplit is survival.r_api.survSplit
    assert survival.survreg is survival.r_api.survreg
    assert survival.survreg is not survival.regression.survreg
    assert survival.vcov is survival.r_api.vcov
    assert survival.ridge_fit is survival.regression.ridge_fit
    assert "ridge_fit" not in vars(survival)


def test_package_root_lazy_loads_domain_modules():
    _remove_survival_modules()
    survival = setup_survival_import()

    assert "survival._survival" not in sys.modules
    assert "survival._binding_utils" not in sys.modules
    assert "survival.regression" not in sys.modules
    assert "regression" not in vars(survival)
    assert "ridge_fit" in survival.__deprecated_root_exports__

    regression = survival.regression

    assert sys.modules["survival.regression"] is regression
    assert survival.ridge_fit is regression.ridge_fit
    assert "ridge_fit" not in vars(survival)

    assert survival._survival is sys.modules["survival._survival"]
    assert "_survival" in vars(survival)


def test_package_root_lazy_loads_r_api_exports():
    _remove_survival_modules()
    survival = setup_survival_import()

    assert "survival.r_api" not in sys.modules
    assert "r_api" not in vars(survival)
    assert "Surv" not in vars(survival)

    assert survival.Surv is survival.r_api.Surv
    assert "survival.r_api" in sys.modules
    assert "r_api" in vars(survival)
    assert "Surv" in vars(survival)


def test_r_api_stub_tracks_surv_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "self",
        "type",
        "origin",
        "time",
        "time1",
        "time2",
        "event",
        "status",
        "start",
        "stop",
    ]
    runtime_params = inspect.signature(survival.r_api.Surv.__init__).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_POSITIONAL
    ] == expected
    assert runtime_params["args"].kind is inspect.Parameter.VAR_POSITIONAL
    assert _pyi_class_method_arg_names(stub_path, "Surv", "__init__") == expected


def test_r_api_stub_tracks_dataclass_public_fields():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    for class_name in sorted(_pyi_top_level_names(stub_path)):
        runtime_class = getattr(survival.r_api, class_name, None)
        if not isinstance(runtime_class, type) or not dataclasses.is_dataclass(runtime_class):
            continue
        runtime_fields = set(runtime_class.__dataclass_fields__)
        assert _pyi_class_annotation_names(stub_path, class_name) == runtime_fields


def test_r_api_stub_tracks_concordance_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "response",
        "data",
        "scores",
        "risk_scores",
        "weights",
        "subset",
        "na_action",
        "cluster",
        "ymin",
        "ymax",
        "timewt",
        "influence",
        "ranks",
        "reverse",
        "timefix",
        "keepstrata",
    ]
    runtime_params = inspect.signature(survival.r_api.concordance).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "concordance") == expected
    assert _pyi_function_kwarg_name(stub_path, "concordance") == "kwargs"


def test_r_api_stub_tracks_model_generic_public_signatures():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected_by_name = {
        "coef": ["fit"],
        "coef_names": ["fit", "complete"],
        "confint": ["fit", "parm", "level"],
        "vcov": ["fit", "complete"],
        "loglik": ["fit"],
        "model_formula": ["fit"],
        "model_summary": ["fit"],
        "model_weights": ["fit"],
        "nobs": ["fit"],
        "degrees_freedom": ["fit"],
        "df_residual": ["fit"],
        "aic": ["fit", "k"],
        "bic": ["fit"],
        "brier": ["fit", "times", "newdata", "ties", "detail", "timefix", "efron"],
        "royston": ["fit", "newdata", "ties", "adjust"],
        "extract_aic": ["fit", "scale", "k"],
        "model_frame": ["fit"],
        "model_matrix": ["fit"],
        "as_data_frame": ["result"],
        "fitted": [
            "fit",
            "type",
            "centered",
            "terms",
            "collapse",
            "reference",
            "se_fit",
            "times",
            "p",
            "quantiles",
        ],
    }
    for name, expected in expected_by_name.items():
        runtime_params = inspect.signature(getattr(survival.r_api, name)).parameters
        assert [
            param_name
            for param_name, parameter in runtime_params.items()
            if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        ] == expected
        assert _pyi_function_arg_names(stub_path, name) == expected
        if name == "fitted":
            assert _pyi_function_kwarg_name(stub_path, name) == "kwargs"


def test_r_api_stub_tracks_survfit_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "response",
        "data",
        "group",
        "newdata",
        "weights",
        "subset",
        "na_action",
        "conf_level",
        "conf_int",
        "conf_type",
        "se_fit",
        "start_time",
        "time0",
        "reverse",
        "censor",
        "type",
        "stype",
        "ctype",
        "id",
        "cluster",
        "robust",
        "istate",
        "etype",
        "model",
        "error",
        "entry",
        "timefix",
    ]
    runtime_params = inspect.signature(survival.r_api.survfit).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "survfit") == expected
    tree = ast.parse(stub_path.read_text(), filename=str(stub_path))
    survfit_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "survfit"
    )
    assert survfit_node.args.kwarg is not None
    assert survfit_node.args.kwarg.arg == "kwargs"

    survfit0_params = inspect.signature(survival.r_api.survfit0).parameters
    assert list(survfit0_params) == ["x", "args", "kwargs"]
    assert survfit0_params["args"].kind is inspect.Parameter.VAR_POSITIONAL
    assert survfit0_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "survfit0") == ["x"]
    tree = ast.parse(stub_path.read_text(), filename=str(stub_path))
    survfit0_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "survfit0"
    )
    assert survfit0_node.args.vararg is not None
    assert survfit0_node.args.vararg.arg == "args"
    assert survfit0_node.args.kwarg is not None
    assert survfit0_node.args.kwarg.arg == "kwargs"


def test_r_api_stub_tracks_predict_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "fit",
        "newdata",
        "type",
        "centered",
        "terms",
        "collapse",
        "reference",
        "se_fit",
        "times",
        "p",
        "quantiles",
    ]
    runtime_params = inspect.signature(survival.r_api.predict).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "predict") == expected
    tree = ast.parse(stub_path.read_text(), filename=str(stub_path))
    predict_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict"
    )
    assert predict_node.args.kwarg is not None
    assert predict_node.args.kwarg.arg == "kwargs"


def test_r_api_stub_tracks_residuals_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "fit",
        "type",
        "terms",
        "collapse",
        "weighted",
        "rsigma",
    ]
    runtime_params = inspect.signature(survival.r_api.residuals).parameters
    assert list(runtime_params) == expected
    assert _pyi_function_arg_names(stub_path, "residuals") == expected


def test_r_api_stub_tracks_aeqsurv_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = ["x", "tolerance"]
    assert list(inspect.signature(survival.r_api.aeqSurv).parameters) == expected
    assert _pyi_function_arg_names(stub_path, "aeqSurv") == expected


def test_r_api_stub_tracks_surv_utility_public_signatures():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected_by_name = {
        "is_na_surv": ["x"],
        "format_surv": ["x"],
        "is_ratetable": ["x", "has_rates", "has_dims", "verbose"],
        "ratetableDate": ["x", "month", "day", "origin_year"],
        "survexp_us": [],
        "survexp_mn": [],
        "survexp_usr": [],
    }
    for name, expected in expected_by_name.items():
        assert list(inspect.signature(getattr(survival.r_api, name)).parameters) == expected
        assert _pyi_function_arg_names(stub_path, name) == expected
    assert (
        inspect.signature(survival.r_api.ratetableDate).parameters["origin_year"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_r_api_stub_tracks_survexp_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "time",
        "age",
        "year",
        "ratetable",
        "sex",
        "times",
        "method",
        "cohort",
        "conditional",
        "scale",
        "se_fit",
    ]
    runtime_params = inspect.signature(survival.r_api.survexp).parameters
    assert list(runtime_params) == expected
    assert runtime_params["cohort"].kind is inspect.Parameter.KEYWORD_ONLY
    assert runtime_params["conditional"].kind is inspect.Parameter.KEYWORD_ONLY
    assert runtime_params["scale"].kind is inspect.Parameter.KEYWORD_ONLY
    assert runtime_params["se_fit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert _pyi_function_arg_names(stub_path, "survexp") == expected

    expected_individual = ["time", "age", "year", "ratetable", "sex"]
    assert (
        list(inspect.signature(survival.r_api.survexp_individual).parameters) == expected_individual
    )
    assert _pyi_function_arg_names(stub_path, "survexp_individual") == expected_individual


def test_r_api_stub_tracks_cipoisson_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = ["k", "time", "p", "method"]
    assert list(inspect.signature(survival.r_api.cipoisson).parameters) == expected
    assert _pyi_function_arg_names(stub_path, "cipoisson") == expected


def test_r_api_stub_tracks_bounded_link_public_signatures():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = ["x", "edge"]
    for name in ["blogit", "bprobit", "bcloglog", "blog"]:
        assert list(inspect.signature(getattr(survival.r_api, name)).parameters) == expected
        assert _pyi_function_arg_names(stub_path, name) == expected


def test_r_api_stub_tracks_pyears_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "response",
        "data",
        "time",
        "start",
        "stop",
        "event",
        "group",
        "weights",
        "subset",
        "na_action",
        "scale",
        "data_frame",
    ]
    runtime_params = inspect.signature(survival.r_api.pyears).parameters
    assert list(runtime_params) == expected
    for name in expected[2:]:
        assert runtime_params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert _pyi_function_arg_names(stub_path, "pyears") == expected


def test_r_api_stub_tracks_finegray_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = ["tstart", "tstop", "ctime", "cprob", "extend", "keep"]
    assert list(inspect.signature(survival.r_api.finegray).parameters) == expected
    assert _pyi_function_arg_names(stub_path, "finegray") == expected


def test_r_api_stub_tracks_survobrien_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = ["time", "status", "covariate", "strata", "data", "subset", "na_action", "transform"]
    runtime_params = inspect.signature(survival.r_api.survobrien).parameters
    assert list(runtime_params) == expected
    for name in expected[4:]:
        assert runtime_params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert _pyi_function_arg_names(stub_path, "survobrien") == expected
    assert _pyi_class_annotation_names(stub_path, "SurvObrienResult") == {
        "statistic",
        "p_value",
        "df",
        "scores",
        "score_sum",
        "expected",
        "variance",
    }


def test_r_api_stub_tracks_survsplit_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = ["response", "data", "cut", "start", "end", "event", "episode", "id", "zero"]
    runtime_params = inspect.signature(survival.r_api.survSplit).parameters
    assert list(runtime_params) == expected
    assert runtime_params["cut"].kind is inspect.Parameter.KEYWORD_ONLY
    assert _pyi_function_arg_names(stub_path, "survSplit") == expected


def test_r_api_stub_tracks_survcondense_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "formula",
        "data",
        "subset",
        "weights",
        "na_action",
        "id",
        "start",
        "end",
        "event",
    ]
    runtime_params = inspect.signature(survival.r_api.survcondense).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "survcondense") == expected
    assert _pyi_function_kwarg_name(stub_path, "survcondense") == "kwargs"


def test_r_api_stub_tracks_survconcordance_public_signatures():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = ["formula", "data", "weights", "subset", "na_action"]
    runtime_params = inspect.signature(survival.r_api.survConcordance).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "survConcordance") == expected
    assert _pyi_function_kwarg_name(stub_path, "survConcordance") == "kwargs"

    fit_expected = ["y", "x", "strata", "weight"]
    fit_params = inspect.signature(survival.r_api.survConcordance_fit).parameters
    assert [
        name
        for name, parameter in fit_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == fit_expected
    assert fit_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "survConcordance_fit") == fit_expected
    assert _pyi_function_kwarg_name(stub_path, "survConcordance_fit") == "kwargs"


def test_r_api_stub_tracks_survcheck_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "response",
        "data",
        "subset",
        "na_action",
        "id",
        "istate",
        "istate0",
        "timefix",
        "time1",
        "time2",
        "status",
    ]
    runtime_params = inspect.signature(survival.r_api.survcheck).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "survcheck") == expected
    assert _pyi_function_kwarg_name(stub_path, "survcheck") == "kwargs"


def test_r_api_stub_tracks_rttright_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "response",
        "status",
        "weights",
        "data",
        "subset",
        "na_action",
        "times",
        "id",
        "timefix",
        "renorm",
    ]
    runtime_params = inspect.signature(survival.r_api.rttright).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["data"].kind is inspect.Parameter.KEYWORD_ONLY
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "rttright") == expected
    assert _pyi_function_kwarg_name(stub_path, "rttright") == "kwargs"


def test_r_api_stub_tracks_pseudo_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "fit",
        "status",
        "eval_times",
        "type_",
        "times",
        "type",
        "collapse",
        "data_frame",
        "time",
    ]
    runtime_params = inspect.signature(survival.r_api.pseudo).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["times"].kind is inspect.Parameter.KEYWORD_ONLY
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "pseudo") == expected
    assert _pyi_function_kwarg_name(stub_path, "pseudo") == "kwargs"


def test_r_api_stub_tracks_strata_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"
    tree = ast.parse(stub_path.read_text(), filename=str(stub_path))
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "strata"
    )

    runtime_params = inspect.signature(survival.r_api.strata).parameters
    assert runtime_params["variables"].kind is inspect.Parameter.VAR_POSITIONAL
    assert list(_pyi_function_arg_names(stub_path, "strata")) == [
        "na_group",
        "shortlabel",
        "sep",
        "labels",
    ]
    assert node.args.vararg is not None
    assert node.args.vararg.arg == "variables"


def test_r_api_stub_tracks_survreg_distribution_helper_signatures():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected_by_name = {
        "dsurvreg": ["x", "mean", "scale", "distribution", "parms"],
        "psurvreg": ["q", "mean", "scale", "distribution", "parms"],
        "qsurvreg": ["p", "mean", "scale", "distribution", "parms"],
        "rsurvreg": ["n", "mean", "scale", "distribution", "parms"],
    }
    for name, expected in expected_by_name.items():
        assert list(inspect.signature(getattr(survival.r_api, name)).parameters) == expected
        assert _pyi_function_arg_names(stub_path, name) == expected


def test_r_api_stub_tracks_survdiff_public_signature():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected = [
        "response",
        "data",
        "group",
        "subset",
        "na_action",
        "rho",
        "timefix",
    ]
    runtime_params = inspect.signature(survival.r_api.survdiff).parameters
    assert [
        name
        for name, parameter in runtime_params.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    ] == expected
    assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert _pyi_function_arg_names(stub_path, "survdiff") == expected
    assert _pyi_function_kwarg_name(stub_path, "survdiff") == "kwargs"


def test_r_api_stub_tracks_fit_control_public_signatures():
    setup_survival_import()
    survival = importlib.import_module("survival")
    stub_path = PACKAGE_ROOT / "r_api.pyi"

    expected_by_name = {
        "clogit": [
            "formula",
            "data",
            "weights",
            "subset",
            "na_action",
            "method",
        ],
        "coxph": [
            "response",
            "data",
            "x",
            "weights",
            "offset",
            "strata",
            "cluster",
            "subset",
            "na_action",
            "init",
            "initial_beta",
            "max_iter",
            "eps",
            "toler",
            "method",
            "ties",
            "robust",
            "model",
            "y",
            "tt",
            "id",
            "istate",
            "statedata",
            "singular_ok",
            "nocenter",
            "control",
        ],
        "survreg": [
            "response",
            "data",
            "x",
            "time",
            "time2",
            "status",
            "covariates",
            "weights",
            "offset",
            "offsets",
            "init",
            "initial",
            "initial_beta",
            "strata",
            "subset",
            "na_action",
            "dist",
            "distribution",
            "scale",
            "parms",
            "model",
            "y",
            "robust",
            "cluster",
            "score",
            "max_iter",
            "eps",
            "tol_chol",
            "control",
        ],
    }
    for name, expected in expected_by_name.items():
        runtime_params = inspect.signature(getattr(survival.r_api, name)).parameters
        assert [
            param_name
            for param_name, parameter in runtime_params.items()
            if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        ] == expected
        assert runtime_params["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
        assert _pyi_function_arg_names(stub_path, name) == expected
        assert _pyi_function_kwarg_name(stub_path, name) == "kwargs"


def test_package_root_version_matches_project_metadata():
    setup_survival_import()
    survival = importlib.import_module("survival")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    cargo_version_match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"',
        (ROOT / "Cargo.toml").read_text(),
    )

    assert cargo_version_match is not None
    assert survival.__version__ == pyproject["project"]["version"]
    assert survival.__version__ == cargo_version_match.group(1)
    assert "__version__" in dir(survival)


def test_package_root_lazy_loads_sklearn_exports():
    _remove_survival_modules()
    survival = setup_survival_import()

    assert "CoxPHEstimator" in survival.__all__
    assert "CoxPHEstimator" not in vars(survival)

    sklearn_compat = importlib.import_module("survival.sklearn_compat")

    assert "survival._sklearn_cox" not in sys.modules
    assert "survival._sklearn_aft" not in sys.modules
    assert "survival._sklearn_deep" not in sys.modules
    assert "survival._sklearn_ensemble" not in sys.modules

    assert survival.CoxPHEstimator is sklearn_compat.CoxPHEstimator
    assert survival.CoxPHEstimator is sklearn_compat.CoxPHEstimator
    assert "survival._sklearn_cox" in sys.modules
    assert "survival._sklearn_aft" not in sys.modules
    assert "survival._sklearn_deep" not in sys.modules
    assert "survival._sklearn_ensemble" not in sys.modules


def test_package_root_lazy_loads_streaming_sklearn_exports():
    _remove_survival_modules()
    survival = setup_survival_import()

    assert "StreamingMixin" in survival.__all__
    assert "StreamingMixin" not in vars(survival)

    assert (
        survival.StreamingMixin is importlib.import_module("survival.sklearn_compat").StreamingMixin
    )
    assert "survival._sklearn_streaming" in sys.modules
    assert "survival._sklearn_cox" not in sys.modules
    assert "survival._sklearn_aft" not in sys.modules
    assert "survival._sklearn_deep" not in sys.modules
    assert "survival._sklearn_ensemble" not in sys.modules

    streaming_cox = survival.StreamingCoxPHEstimator

    assert "survival._sklearn_streaming_cox" in sys.modules
    assert "survival._sklearn_cox" in sys.modules
    assert "survival._sklearn_aft" not in sys.modules
    assert "survival._sklearn_deep" not in sys.modules
    assert "survival._sklearn_ensemble" not in sys.modules
    assert issubclass(
        streaming_cox,
        importlib.import_module("survival._sklearn_cox").CoxPHEstimator,
    )
