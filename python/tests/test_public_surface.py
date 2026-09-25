"""The public surface of ``survival.r``.

``survival.r`` ships no stub: type checkers read the inline annotations of its modules
(``py.typed``), so every exported function must be annotated and the classes the functions
return must be exported.  ``typing_smoke.py`` checks the same surface with mypy.  The R
reference values were computed with R 4.5.3 and survival 3.8-12.
"""

import inspect
import math
import typing
from pathlib import Path

import pytest

from .helpers import setup_survival_import
from .r_fixture_support import RFactor

survival = setup_survival_import()
r = survival.r
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "survival"

# R's formals(concordancefit), with std.err spelled std_err
R_CONCORDANCEFIT_FORMALS = [
    "y",
    "x",
    "strata",
    "weights",
    "ymin",
    "ymax",
    "timewt",
    "cluster",
    "influence",
    "ranks",
    "reverse",
    "timefix",
    "keepstrata",
    "std_err",
]


def approx(values, rel=1e-10):
    return pytest.approx(values, rel=rel, abs=1e-12)


def counts(rows):
    return [[row[name] for name in ("concordant", "discordant", "tied.x")] for row in rows]


@pytest.fixture(scope="module")
def ovarian():
    return {k: v for k, v in survival.datasets.load_ovarian().items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def cox(ovarian):
    return r.coxph("Surv(futime, fustat) ~ age", ovarian)


@pytest.fixture(scope="module")
def weibull(ovarian):
    return r.survreg("Surv(futime, fustat) ~ age", ovarian)


def _public_functions():
    for name in r.__all__:
        value = getattr(r, name)
        if callable(value) and not isinstance(value, type):
            yield name, value


def test_every_exported_name_resolves_and_r_api_reexports_it():
    assert len(set(r.__all__)) == len(r.__all__)
    for name in r.__all__:
        assert getattr(r, name) is getattr(survival.r_api, name), name
    assert survival.r_api.__all__ == r.__all__


def test_inline_annotations_are_the_typed_surface():
    assert (PACKAGE_ROOT / "py.typed").is_file()
    assert not (PACKAGE_ROOT / "r_api.pyi").exists()
    assert not (PACKAGE_ROOT / "r" / "__init__.pyi").exists()


@pytest.mark.parametrize("name", [name for name, _ in _public_functions()])
def test_every_public_function_declares_its_return_type(name):
    function = getattr(r, name)
    assert inspect.signature(function).return_annotation is not inspect.Signature.empty
    # the annotations name real objects (a typo would silently become Any for mypy)
    assert "return" in typing.get_type_hints(function)


def test_model_functions_return_the_exported_classes(ovarian, cox, weibull):
    mstate = {
        "time": [1, 2, 3, 4, 5, 6],
        "ev": RFactor(["a", "b", "censor", "a", "censor", "b"], ["censor", "a", "b"]),
    }
    km = r.survfit("Surv(futime, fustat) ~ rx", ovarian)
    overlap = r.survcheck(
        "Surv(start, stop, status) ~ 1",
        {"id": ["s1", "s1", "s2"], "start": [0, 0.5, 0], "stop": [1, 2, 2], "status": [0, 1, 1]},
        id="id",
    )
    results = [
        (cox, r.CoxphModel),
        (r.clogit("fustat ~ age + strata(rx)", ovarian), r.ClogitModel),
        (weibull, r.SurvregModelResult),
        (r.anova(cox), r.AnovaCoxphResult),
        (r.anova(weibull), r.SurvregAnovaResult),
        (r.brier(cox), r.BrierResult),
        (r.concordance(cox), r.ConcordanceResult),
        (overlap, r.SurvCheckResult),
        (overlap.overlap, r.SurvCheckProblem),
        (km, r.SurvfitResult),
        (r.survfit0(km), r.SurvfitResult),
        (r.survfit(cox), r.CoxSurvfitResult),
        (r.survfit("Surv(time, ev) ~ 1", mstate), r.SurvfitMultiStateResult),
        (r.pspline(ovarian["age"]), r.PsplineResult),
        (r.statefig([1, 1], {"a": [0, 1], "b": [0, 0]}), r.StateFigResult),
        (r.nsk(ovarian["age"], df=3), survival._survival.SplineBasisResult),
    ]
    for value, cls in results:
        assert type(value) is cls


def test_concordancefit_takes_r_arguments_in_r_order(ovarian):
    parameters = inspect.signature(r.concordancefit).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == R_CONCORDANCEFIT_FORMALS
    # the extras are keyword-only: names stands in for colnames(x), the rest is private
    extras = {name for name in parameters if name not in R_CONCORDANCEFIT_FORMALS}
    assert extras == {"names", "_strata_levels", "_formula", "kwargs"}

    # R: concordancefit(Surv(ovarian$futime, ovarian$fustat), ovarian$age, ovarian$rx)
    y = r.Surv(ovarian["futime"], ovarian["fustat"])
    fit = r.concordancefit(y, ovarian["age"], ovarian["rx"])
    assert fit.concordance == approx(0.19047619047619)
    assert counts(fit.count) == [[12, 49, 0], [8, 36, 0]]
    assert fit.var == approx(0.00322663910613376)
    assert fit.cvar == approx(0.00906273620559335)
    # std.err = FALSE given positionally, after R's defaults for weights ... keepstrata
    r_defaults = [None, None, None, "n", None, 0, False, False, True, 10]
    quick = r.concordancefit(y, ovarian["age"], ovarian["rx"], *r_defaults, False)
    assert quick.var is None
    assert quick.concordance == approx(0.19047619047619)
    with pytest.raises(TypeError, match="unexpected argument"):
        r.concordancefit(y, ovarian["age"], strata_levels=["1", "2"])


def test_cluster_is_the_identity():
    ids = [3, 1, 2, 1]
    assert r.cluster(ids) is ids


def test_survreg_distribution_helpers_match_r():
    # the names of R's survreg.distributions, in R's order
    assert list(r.survreg_distributions) == [
        "extreme",
        "logistic",
        "gaussian",
        "weibull",
        "exponential",
        "rayleigh",
        "loggaussian",
        "lognormal",
        "loglogistic",
        "t",
    ]
    # the exported registry is the one survreg(dist=) looks names up in
    assert r.survreg_distributions is survival.r._survreg.survreg_distributions
    assert r.survregDtest(r.survreg_distributions["weibull"]) is True
    # R: survreg.control() has iter.max 30, rel.tolerance 1e-9, toler.chol 1e-10
    control = r.survreg_control()
    assert control.iter_max == 30
    assert math.isclose(control.rel_tolerance, 1e-9)
    assert math.isclose(control.toler_chol, 1e-10)
