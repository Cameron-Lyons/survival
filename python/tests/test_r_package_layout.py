from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R_PACKAGE = ROOT / "r" / "survivalr"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "rust.yml"


def _namespace_exports() -> set[str]:
    exports: set[str] = set()
    for line in (R_PACKAGE / "NAMESPACE").read_text().splitlines():
        line = line.strip()
        if line.startswith("export(") and line.endswith(")"):
            exports.add(line.removeprefix("export(").removesuffix(")"))
    return exports


def _namespace_s3_methods() -> set[tuple[str, str]]:
    methods: set[tuple[str, str]] = set()
    for line in (R_PACKAGE / "NAMESPACE").read_text().splitlines():
        line = line.strip()
        if line.startswith("S3method(") and line.endswith(")"):
            generic, cls = line.removeprefix("S3method(").removesuffix(")").split(", ")
            methods.add((generic, cls))
    return methods


def test_r_bridge_package_metadata_is_present():
    description = (R_PACKAGE / "DESCRIPTION").read_text()

    assert "Package: survivalr" in description
    assert "Author: Cameron Lyons" in description
    assert "Maintainer: Cameron Lyons" in description
    assert "reticulate" in description
    assert "    survival," in description
    assert "SystemRequirements: Python" in description
    assert (R_PACKAGE / "LICENSE").is_file()
    assert (R_PACKAGE / "R" / "bridge.R").is_file()
    assert (R_PACKAGE / "man" / "survivalr.Rd").is_file()
    assert (R_PACKAGE / "tests" / "testthat.R").is_file()


def test_r_bridge_exports_core_survival_entry_points():
    exports = _namespace_exports()
    manual = (R_PACKAGE / "man" / "survivalr.Rd").read_text()

    assert {
        "Surv",
        "is.Surv",
        "survfit",
        "survdiff",
        "coxph",
        "coxph.control",
        "survreg",
        "survreg.control",
        "basehaz",
        "concordance",
        "cox.zph",
        "coxph.detail",
    } <= exports
    for name in exports:
        assert f"\\alias{{{name}}}" in manual


def test_r_bridge_defines_exported_functions():
    bridge = (R_PACKAGE / "R" / "bridge.R").read_text()

    assert 'reticulate::import("survival.r_api", convert = TRUE)' in bridge
    assert 'inherits(x, "python.builtin.object")' in bridge
    assert '.call_r_api("coef_names", object' in bridge
    assert '.call_r_api("confint", object' in bridge
    assert '.call_r_api("extract_aic", fit' in bridge
    assert '.call_r_api("model_formula", x)' in bridge
    assert '.call_r_api("model_weights", object)' in bridge
    assert '.call_r_api("model_matrix", object' in bridge
    assert '.call_r_api("model_frame", formula' in bridge
    assert '"fitted",\n    object' in bridge
    assert '.call_r_api("as_data_frame", x)' in bridge
    assert '.call_r_api("model_summary", object' in bridge
    assert 'UseMethod("survfit")' in bridge
    assert "coxph.control <- function" in bridge
    assert "survreg.control <- function" in bridge
    assert "response = formula" in bridge
    assert "NextMethod()" in bridge
    for name in _namespace_exports() - {"anova"}:
        assert f"{name} <- function" in bridge


def test_r_bridge_registers_model_s3_methods():
    bridge = (R_PACKAGE / "R" / "bridge.R").read_text()
    manual = (R_PACKAGE / "man" / "survivalr.Rd").read_text()
    methods = _namespace_s3_methods()

    assert {
        ("coef", "survival_py_model"),
        ("confint", "survival_py_model"),
        ("vcov", "survival_py_model"),
        ("logLik", "survival_py_model"),
        ("nobs", "survival_py_model"),
        ("df.residual", "survival_py_survreg"),
        ("extractAIC", "survival_py_model"),
        ("formula", "survival_py_model"),
        ("terms", "survival_py_model"),
        ("weights", "survival_py_model"),
        ("model.matrix", "survival_py_model"),
        ("model.frame", "survival_py_model"),
        ("fitted", "survival_py_model"),
        ("summary", "survival_py_model"),
        ("summary", "survival_py_survfit"),
        ("summary", "survival_py_basehaz"),
        ("summary", "survival_py_survdiff"),
        ("summary", "survival_py_concordance"),
        ("summary", "survival_py_cox_zph"),
        ("summary", "survival_py_coxph_detail"),
        ("summary", "survival_py_anova"),
        ("survfit", "character"),
        ("survfit", "formula"),
        ("survfit", "survival_py_coxph"),
        ("print", "survival_py_model"),
        ("print", "summary.survival_py_model"),
        ("print", "survival_py_surv"),
        ("print", "survival_py_survfit"),
        ("print", "survival_py_basehaz"),
        ("print", "survival_py_survdiff"),
        ("print", "survival_py_concordance"),
        ("print", "survival_py_cox_zph"),
        ("print", "survival_py_coxph_detail"),
        ("print", "survival_py_anova"),
        ("predict", "survival_py_model"),
        ("residuals", "survival_py_model"),
        ("anova", "survival_py_model"),
        ("as.data.frame", "survival_py_survfit"),
        ("as.data.frame", "survival_py_surv"),
        ("as.data.frame", "survival_py_basehaz"),
        ("as.data.frame", "survival_py_survdiff"),
        ("as.data.frame", "survival_py_concordance"),
        ("as.data.frame", "survival_py_cox_zph"),
        ("as.data.frame", "survival_py_coxph_detail"),
        ("as.data.frame", "survival_py_anova"),
    } <= methods
    for generic, cls in methods:
        method = f"{generic}.{cls}"
        assert f"{method} <- function" in bridge
        assert f"\\alias{{{method}}}" in manual


def test_r_bridge_has_ci_check_with_python_extension():
    workflow = CI_WORKFLOW.read_text()

    assert "R Bridge Package" in workflow
    assert "maturin develop --release --features extension-module,ml" in workflow
    assert "R CMD check --no-manual --no-build-vignettes r/survivalr" in workflow
    assert "RETICULATE_PYTHON" in workflow
