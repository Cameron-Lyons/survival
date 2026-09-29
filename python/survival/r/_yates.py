"""Population marginal means and SAS type III tests (R survival yates)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

from .. import _survival as _core
from ._coerce import (
    _integer_scalar,
    _match_string_arg,
    _materialize_1d,
)
from ._coxph import _coxph_model_frame, survfit_coxph
from ._coxphms import CoxphmsModel
from ._fit import _formula_design_for_fit
from ._formula import (
    _column,
    _covariate_term_columns,
    _covariate_term_name,
    _data_column_names,
    _design_rows_from_spec,
    _design_term_output_names,
    _split_terms,
)
from ._misc import _coxph_engine, _unique_in_order
from ._models import _plain_model_frame, coef, vcov
from ._types import (
    CoxSurvfitResult,
    YatesResult,
    _CategoricalDesignTerm,
    _FormulaDesign,
    _InteractionDesignTerm,
    _InteractionTerm,
)
from ._yates_model import YatesModel


@dataclass(frozen=True)
class _YatesTerm:
    """Tested variables and their Cartesian product of requested levels."""

    columns: list[str]
    names: list[str]
    levels: dict[str, list[Any]]
    estimate_names: list[str]


def _design_factors(design: _FormulaDesign) -> list[Any]:
    """Every single (non-interaction) design term, interaction factors included."""

    factors: list[Any] = []
    for spec in design.covariates:
        factors.extend(spec.factors if isinstance(spec, _InteractionDesignTerm) else [spec])
    return factors


def _yates_term(design: _FormulaDesign, term: Any, levels: Any | None) -> _YatesTerm:
    """R's ``cmatrix`` variable selection and ``expand.grid`` level ordering."""

    if isinstance(term, str):
        parsed = _split_terms(term.strip().removeprefix("~").strip())
        if parsed.strata or parsed.offsets or parsed.clusters:
            raise ValueError("the term must select variables from the fitted formula")
        parts = [
            part
            for item in parsed.covariates
            for part in (item.factors if isinstance(item, _InteractionTerm) else [item])
        ]
    else:
        values = [term] if isinstance(term, Real) else _materialize_1d(term, "term")
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
            raise TypeError("the term must be a character string or integer term numbers")
        assignments = design.term_assignments or tuple(range(1, len(design.covariates) + 1))
        selected = [_integer_scalar(value, "term") for value in values]
        if any(value not in assignments for value in selected):
            raise ValueError("numeric term must select a fitted covariate term (1-based)")
        parts = [
            part.term
            for value in selected
            for assignment, item in zip(assignments, design.covariates, strict=True)
            if assignment == value
            for part in (item.factors if isinstance(item, _InteractionDesignTerm) else [item])
        ]
    columns = _unique_in_order(
        [column for part in parts for column in _covariate_term_columns(part)]
    )
    if not columns:
        raise ValueError("the term must select variables from the fitted formula")
    factors = {}
    for spec in _design_factors(design):
        factors.setdefault(spec.term.column, spec)
    missing = [column for column in columns if column not in factors]
    if missing:
        raise ValueError(f"variable {' '.join(missing)} not found in the formula")
    level_names = _data_column_names(levels)
    if levels is not None and level_names is None and len(columns) > 1:
        raise ValueError("levels should be a data frame or mapping for multiple variables")
    selected = {}
    for column in columns:
        spec = factors[column]
        categorical = isinstance(spec, _CategoricalDesignTerm)
        if levels is None or (level_names is not None and column not in level_names):
            if not categorical:
                raise ValueError(
                    "continuous variables require the levels argument"
                    if levels is None
                    else f"levels information not found for: {column}"
                )
            values = list(spec.levels)
        elif level_names is not None:
            values = _column(levels, column)
            if len(_unique_in_order(values)) != len(values):
                raise ValueError("levels data frame has duplicates")
        else:
            values = _unique_in_order(_materialize_1d(levels, "levels"))
        if categorical and any(value not in spec.levels for value in values):
            raise ValueError(f"invalid level for term {column}")
        if not values:
            raise ValueError(f"levels for {column} must not be empty")
        selected[column] = values
    names = [_covariate_term_name(factors[column].term) for column in columns]
    return _YatesTerm(
        columns,
        names,
        _factorial_population({}, selected),
        names if levels is None else columns,
    )


def _factorial_population(
    template: Mapping[str, list[Any]], categorical: dict[str, list[Any]]
) -> dict[str, list[Any]]:
    """R's ``yates_factorial_pop``: every combination of the adjusters' levels, the first
    adjuster varying fastest, with the other columns copied from the first data row."""

    n = math.prod(len(levels) for levels in categorical.values())
    pdata = {name: [values[0]] * n for name, values in template.items()}
    n1 = 1
    for name, levels in categorical.items():
        pdata[name] = [levels[(idx // n1) % len(levels)] for idx in range(n)]
        n1 *= len(levels)
    return pdata


def _yates_model_frame(fit: Any) -> dict[str, list[Any]]:
    """``mframe <- fit$model; if (is.null(mframe)) mframe <- model.frame(fit)``."""

    return _plain_model_frame(fit.model if isinstance(fit, YatesModel) else _coxph_model_frame(fit))


def _yates_population(
    mframe: dict[str, list[Any]],
    design: _FormulaDesign,
    term: _YatesTerm,
    population: str,
) -> dict[str, list[Any]]:
    """R's ``yates_xmat`` population rows over the adjusting variables."""

    adjusters = [spec for spec in _design_factors(design) if spec.term.column not in term.columns]
    categorical = {
        (_covariate_term_name(spec.term) if spec.term.strata else spec.term.column): list(
            spec.levels
        )
        for spec in adjusters
        if isinstance(spec, _CategoricalDesignTerm)
    }
    continuous = [
        spec.term.column for spec in adjusters if not isinstance(spec, _CategoricalDesignTerm)
    ]
    if population == "data" or (population == "sas" and not categorical):
        return mframe
    if population == "factorial" and continuous:
        raise ValueError(
            "population=factorial only applies if all the adjusting terms are categorical"
        )
    pdata = _factorial_population(mframe, categorical)
    if not continuous:
        return pdata
    # sas with a mixed population: each factorial row crossed with every data row
    n_data = len(next(iter(mframe.values())))
    n_pop = len(next(iter(pdata.values())))
    out = {
        name: [values[idx // n_data] for idx in range(n_pop * n_data)]
        for name, values in pdata.items()
    }
    for name in continuous:
        out[name] = [mframe[name][idx % n_data] for idx in range(n_pop * n_data)]
    return out


def _yates_weights(mframe: Mapping[str, list[Any]], population: str) -> list[float] | None:
    """Case weights of the ``data`` population: the model weights, else equal weight per id."""

    if population != "data":
        return None
    if "(weights)" in mframe:
        return [float(value) for value in mframe["(weights)"]]
    if "(id)" in mframe:
        counts: dict[Any, int] = {}
        for value in mframe["(id)"]:
            counts[value] = counts.get(value, 0) + 1
        return [1.0 / counts[value] for value in mframe["(id)"]]
    return None


def _yates_design_names(design: _FormulaDesign) -> list[str]:
    names = ["(Intercept)"] if design.intercept else []
    for spec in design.covariates:
        names.extend(_design_term_output_names(spec))
    return names


def _columns(rows: Sequence[Sequence[float]], keep: Sequence[int]) -> list[list[float]]:
    return [[row[idx] for idx in keep] for row in rows]


def _yates_sgtt(
    fit: Any,
    design: _FormulaDesign,
    term: _YatesTerm,
    kept: list[int],
    beta: list[float],
    vmat: list[list[float]],
) -> tuple[Any, list[str]]:
    """Build R's full indicator design and delegate the type III tests to Rust."""

    external = isinstance(fit, YatesModel)
    full_intercept = design.intercept or not external
    factors = {}
    for spec in _design_factors(design):
        factors.setdefault(_covariate_term_name(spec.term), spec)
    factor_order = [
        name for name, spec in factors.items() if isinstance(spec, _CategoricalDesignTerm)
    ]

    def expanded(spec):
        if isinstance(spec, _InteractionDesignTerm):
            return replace(spec, factors=tuple(expanded(factor) for factor in spec.factors))
        if isinstance(spec, _CategoricalDesignTerm):
            levels = spec.levels
            # R's model.matrix uses natural indicators for a promoted full
            # factor, otherwise the supplied contrasts with baseline last.
            if not spec.full and (
                full_intercept or factor_order.index(_covariate_term_name(spec.term)) > 0
            ):
                levels = (*levels[1:], levels[0])
            return replace(spec, levels=levels, full=True)
        return spec

    full = replace(
        design,
        covariates=tuple(expanded(spec) for spec in design.covariates),
        intercept=full_intercept,
    )
    assignments = design.term_assignments or tuple(range(1, len(design.covariates) + 1))
    nterms = max(assignments, default=0)
    variables = [set() for _ in range(nterms)]
    categorical = [False] * nterms
    assign = [0] if full_intercept else []
    original_assign = [0] if design.intercept else []
    term_codes = {}
    for code, original, spec in zip(assignments, design.covariates, full.covariates, strict=True):
        components = spec.factors if isinstance(spec, _InteractionDesignTerm) else (spec,)
        variables[code - 1] = {_covariate_term_name(factor.term) for factor in components}
        categorical[code - 1] = all(
            isinstance(factor, _CategoricalDesignTerm) for factor in components
        )
        assign.extend([code] * len(_design_term_output_names(spec)))
        original_assign.extend([code] * len(_design_term_output_names(original)))
        if len(components) == 1:
            term_codes[components[0].term.column] = code
    if any(column not in term_codes for column in term.columns):
        raise ValueError("each tested variable must have a main-effect term for sgtt")
    test_terms = [
        (term_codes[column], name) for column, name in zip(term.columns, term.names, strict=True)
    ]
    adjusters = [
        [
            j + 1
            for j in range(i + 1, nterms)
            if categorical[i] and categorical[j] and variables[i] & variables[j]
        ]
        for i in range(nterms)
    ]
    start = int(design.intercept and not external)
    coefficient_assign = [original_assign[start + index] for index in kept]
    frame = _yates_model_frame(fit)
    rows = len(next(iter(frame.values())))
    result = _core.yates_sgtt(
        _design_rows_from_spec(frame, full, rows),
        assign,
        adjusters,
        beta,
        vmat,
        coefficient_assign,
        test_terms,
        sigma2=fit.sigma2 if external else None,
        include_intercept=external and design.intercept,
    )
    names = _yates_design_names(full)
    return result, [names[index] for index in result.columns]


@dataclass(frozen=True)
class _YatesSetup:
    """What R's ``yates_setup`` gives ``yates``: the prediction (``linear``, ``risk`` or
    ``survival``), the seed of its simulation and, for ``survival``, the baseline curve
    and the restricted-mean horizon ``rmean``."""

    predict: str
    seed: int = 0
    baseline: CoxSurvfitResult | None = None
    rmean: float = math.inf


_COXPH_PREDICT = ["lp", "risk", "expected", "terms", "survival", "linear"]


def _yates_setup(fit: Any, predict: Any, options: Any | None) -> _YatesSetup:
    """R's ``yates_setup``: ``yates_setup.coxph`` for a Cox model; ``yates_setup.default``
    for a ``YatesModel``, which gives the linear predictor whatever ``predict`` is (``yates``
    passes it as ``predict=``, which that method's ``type`` argument never receives, so R
    neither checks nor warns).

    ``options`` holds R's ``rmean`` for ``predict="survival"`` and the ``seed`` of R's
    generator (``set.seed``) for the simulated predictions.
    """

    if callable(predict) or isinstance(predict, Mapping):
        raise ValueError("user written prediction functions are not yet supported")
    if isinstance(fit, YatesModel):
        return _YatesSetup("linear")
    kind = _match_string_arg(
        # match.arg(NULL) is the first choice
        "lp" if predict is None else predict,
        "predict",
        _COXPH_PREDICT,
        "'predict' should be one of " + ", ".join(f'"{name}"' for name in _COXPH_PREDICT),
    )
    if kind in ("lp", "linear"):
        return _YatesSetup("linear")
    if kind in ("expected", "terms"):
        raise ValueError(f"type {kind} is not supported")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    settings = dict(options or {})
    unknown = set(settings) - ({"seed"} if kind == "risk" else {"seed", "rmean"})
    if unknown:
        raise TypeError(f"unrecognized {kind} options: {', '.join(sorted(map(str, unknown)))}")
    seed = _integer_scalar(settings.get("seed", 0), "seed")
    if kind == "risk":
        return _YatesSetup("risk", seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline = survfit_coxph(fit, censor=False)
    rmean = settings.get("rmean")
    try:
        rmean = max(baseline.time) if rmean is None else float(rmean)
    except (TypeError, ValueError) as exc:
        raise TypeError("rmean must be numeric") from exc
    if baseline.strata is not None:
        raise ValueError("stratified models not yet supported")
    return _YatesSetup("survival", seed, baseline, rmean)


def _yates_estimable(fit: Any, design: _FormulaDesign, xmatlist: list[Any]) -> list[bool]:
    """R's estimability check for a fit with aliased coefficients, against the unique rows
    of ``model.matrix(fit)`` (a Cox model's with a leading column of ones)."""

    if isinstance(fit, YatesModel):
        n = len(next(iter(fit.model.values())))
        return _core.yates_estimable(xmatlist, _design_rows_from_spec(fit.model, design, n))
    return _core.yates_estimable(xmatlist, fit.x, intercept=design.intercept)


def _yates_survival_summary(
    baseline: CoxSurvfitResult, curves: _core.YatesCurves
) -> CoxSurvfitResult:
    """R's ``summary`` function of ``yates_setup.coxph``: the baseline curve carrying each
    level's simulated mean survival, one column per level."""

    return replace(
        baseline,
        surv=curves.surv,
        cumhaz=curves.cumhaz,
        std_err=curves.std_err,
        lower=curves.lower,
        upper=curves.upper,
    )


def yates(
    fit: Any,
    term: Any,
    population: Any = None,
    levels: Any | None = None,
    test: Any = "global",
    predict: Any = "linear",
    options: Any | None = None,
    nsim: Any = 200,
    method: Any = "direct",
) -> YatesResult:
    """Population marginal means of a term of a Cox model and their tests (R's ``yates``).

    ``predict`` is the linear predictor (``"linear"``/``"lp"``), ``"risk"`` or
    ``"survival"`` (the mean survival restricted to ``options={"rmean": ...}``, by default
    the last time of the baseline curve, with the simulated curves in ``summary``); the
    latter two average ``nsim`` coefficient draws, and ``options={"seed": 0}`` seeds
    their R-compatible random stream.  ``population`` is ``"data"``, ``"factorial"``,
    ``"sas"`` or a data frame.  With aliased coefficients, a level the fit cannot
    estimate has an NA mean and the tests that use it are NA.  ``YatesModel`` adapts
    externally fitted linear models without refitting them.
    ``term`` can select several variables, e.g. ``"a + b"`` or ``"a:b"``,
    or use one-based fitted term numbers such as ``[1, 2]``.
    A ``levels`` mapping supplies per-variable values; omitted categorical
    variables use their fitted levels. The first variable varies fastest.
    ``method="sgtt"`` computes a SAS-style type III test for each selected
    main-effect variable, using the SAS population and linear predictions.
    The estimable hypothesis matrix is returned in ``sas``.
    """

    if isinstance(fit, CoxphmsModel):
        raise ValueError("multi-state coxph not yet supported")
    external = isinstance(fit, YatesModel)
    engine = None if external else _coxph_engine(fit, "the fit does not have a terms structure")
    design = _formula_design_for_fit(fit)
    if design is None:
        raise TypeError("the fit does not have a terms structure")
    setup = _yates_setup(fit, predict, options)
    method_value = _match_string_arg(
        method.lower() if isinstance(method, str) else method,
        "method",
        ["direct", "sgtt"],
        "invalid method",
    )
    if population is None:
        population = "sas" if method_value == "sgtt" else "data"
    if isinstance(population, str):
        population = _match_string_arg(
            population.lower(),
            "population",
            ["data", "factorial", "sas", "empirical", "yates"],
            "unknown population",
        )
        population = {"empirical": "data", "yates": "factorial"}.get(population, population)
    elif not isinstance(population, Mapping):
        raise TypeError("the population argument must be a data frame or character")
    if method_value == "sgtt" and (population != "sas" or setup.predict != "linear"):
        raise ValueError("sgtt method only applies if population = sas and predict = linear")
    test_value = _match_string_arg(test, "test", ["global", "trend", "pairwise"], "invalid test")

    beta = fit.coefficients if external else coef(fit)
    kept = [idx for idx, value in enumerate(beta) if not math.isnan(value)]
    vmat = fit.variance if external else vcov(fit, complete=False)
    if len(vmat) > len(kept):
        vmat = _columns([vmat[idx] for idx in kept], kept)
    yates_term = _yates_term(design, term, levels)
    if isinstance(population, Mapping):
        pdata = {str(name): list(values) for name, values in population.items()}
        weights = None
    else:
        mframe = _yates_model_frame(fit)
        pdata = _yates_population(mframe, design, yates_term, population)
        weights = _yates_weights(mframe, population)
    n_pop = len(next(iter(pdata.values())))
    factor_values = (
        {
            spec.term: pdata[_covariate_term_name(spec.term)]
            for spec in _design_factors(design)
            if spec.term.strata and _covariate_term_name(spec.term) in pdata
        }
        if isinstance(population, str) and population != "data"
        else None
    )
    xmatlist = [
        _design_rows_from_spec(
            {
                **pdata,
                **{
                    column: [value] * n_pop
                    for column, value in zip(yates_term.columns, combination, strict=True)
                },
            },
            design,
            n_pop,
            factor_values=factor_values,
        )
        for combination in zip(*yates_term.levels.values(), strict=True)
    ]
    estimable = _yates_estimable(fit, design, xmatlist) if len(kept) < len(beta) else None
    # the coefficient columns: a Cox model's baseline absorbs the intercept
    first = 1 if design.intercept and not external else 0
    columns = [first + idx for idx in kept]
    design_names = _yates_design_names(design)
    names = [design_names[idx] for idx in columns]
    beta = [beta[idx] for idx in kept]
    means = [0.0] * len(beta) if external else [engine.means[idx] for idx in kept]
    summary = None
    sas = None
    sas_names = []
    if setup.predict == "linear":
        result = _core.yates(
            _columns(_core.yates_population_means(xmatlist, weights), columns),
            beta,
            vmat,
            offset=-sum(mean * value for mean, value in zip(means, beta, strict=True)),
            sigma2=fit.sigma2 if external else None,
            estimable=estimable,
            test=test_value,
        )
        if not result.cmat:
            names = []
        if method_value == "sgtt":
            sas, sas_names = _yates_sgtt(fit, design, yates_term, kept, beta, vmat)
    else:
        simulation = {
            "estimable": estimable,
            "nsim": _integer_scalar(nsim, "nsim"),
            "seed": setup.seed,
            "test": test_value,
            "term": yates_term.names[0] if len(yates_term.names) == 1 else "global",
        }
        xmatlist = [_columns(rows, columns) for rows in xmatlist]
        if setup.baseline is None:
            result = _core.yates_risk(xmatlist, beta, vmat, means, **simulation)
        else:
            baseline = setup.baseline
            result = _core.yates_survival(
                xmatlist,
                beta,
                vmat,
                means,
                baseline.time,
                baseline.cumhaz,
                setup.rmean,
                conf_int=baseline.conf_int,
                **simulation,
            )
            if result.summary is not None:
                summary = _yates_survival_summary(baseline, result.summary)
        names = []
    return YatesResult(
        estimate={
            **dict(zip(yates_term.estimate_names, yates_term.levels.values(), strict=True)),
            "pmm": [row.pmm for row in result.estimate],
            "std": [row.std for row in result.estimate],
        },
        test=result.test if sas is None else sas.test,
        mvar=result.mvar,
        cmat=result.cmat,
        cmat_names=names,
        summary=summary,
        sas=None if sas is None else sas.sas,
        sas_names=sas_names,
        sas_row_names=[] if sas is None else [f"L{index + 1}" for index in sas.columns],
    )
