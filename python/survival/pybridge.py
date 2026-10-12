"""Internal hooks for R interoperability and native-result reconstruction."""

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ._binding_utils import bind_names

__all__ = bind_names(
    globals(),
    [
        "CoxPenaltyTerms",
        "_survpenal_fit_from_state",
        "_unpickle",
        "cox_callback",
    ],
)


def _serialize_r_object(value: Any) -> dict[str, Any]:
    """Capture current model state when R serializes its lazy state handle."""
    import io
    import pickle
    from types import FunctionType

    callbacks: list[Any] = []
    indices: dict[Callable[..., Any], int] = {}

    class RPickler(pickle.Pickler):
        def persistent_id(self, obj: Any) -> Any:
            if (
                isinstance(obj, FunctionType)
                and obj.__module__ == "rpytools.call"
                and obj.__qualname__ == "make_python_function.<locals>.python_function"
                and obj.__code__.co_freevars == ("f",)
                and obj.__closure__ is not None
                and type(obj.__closure__[0].cell_contents).__name__ == "PyCapsule"
            ):
                if obj not in indices:
                    indices[obj] = len(callbacks)
                    callbacks.append(obj.__closure__[0].cell_contents)
                return ("r_callback", indices[obj])
            return None

    stream = io.BytesIO()
    RPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return {"version": 1, "pickle": bytearray(stream.getvalue()), "callbacks": callbacks}


def _unserialize_r_object(state: Mapping[str, Any]) -> Any:
    """Restore a model from the embedded pickle in a trusted R model file."""
    import io
    import pickle

    if not isinstance(state, Mapping) or state.get("version") != 1:
        raise ValueError("unsupported survival model serialization version")
    payload = state.get("pickle")
    callbacks = state.get("callbacks")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("survival model serialization requires a byte payload")
    if not isinstance(callbacks, (list, tuple)) or not all(callable(f) for f in callbacks):
        raise ValueError("survival model serialization requires R callbacks")
    callback_table = tuple(callbacks)

    class RUnpickler(pickle.Unpickler):
        def persistent_load(self, key: Any) -> Any:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or key[0] != "r_callback"
                or type(key[1]) is not int
                or not 0 <= key[1] < len(callback_table)
            ):
                raise pickle.UnpicklingError("invalid R callback reference")
            return callback_table[key[1]]

    return RUnpickler(io.BytesIO(payload)).load()


def _call_fit_with_warnings(
    function: Callable[..., Any],
    arguments: Mapping[str, Any],
    *,
    positional: Sequence[Any] = (),
    user_warnings: bool = False,
    capture_error: bool = False,
) -> dict[str, Any]:
    """Return a model call and its warnings for R's condition system."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        if user_warnings:
            warnings.simplefilter("always", UserWarning)
        try:
            if "fit" in arguments:
                keywords = dict(arguments)
                fit = keywords.pop("fit")
                result = function(fit, *positional, **keywords)
            else:
                result = function(*positional, **arguments)
        except Exception as error:
            if not capture_error:
                raise
            return {
                "result": None,
                "error": error,
                "warnings": [str(issue.message) for issue in recorded],
            }
    return {"result": result, "warnings": [str(issue.message) for issue in recorded]}


def _call_dpqr_with_warnings(
    function: Callable[..., Any],
    keywords: Mapping[str, Any],
    *,
    positional: Sequence[Any] | None = None,
    warning: Callable[[str], Any],
) -> Any:
    """Forward distribution arithmetic warnings when they occur, including R warn=2."""
    with warnings.catch_warnings():
        warnings.simplefilter("always", RuntimeWarning)
        warnings.simplefilter("always", UserWarning)
        previous = warnings.showwarning

        def emit(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            if issubclass(category, (RuntimeWarning, UserWarning)):
                warning(str(message))
            else:
                previous(message, category, filename, lineno, file, line)

        warnings.showwarning = emit
        return function(*(positional or ()), **keywords)


def _raise_captured_error(error: Exception) -> None:
    """R re-signals a captured Python error after emitting its preceding warnings."""
    raise error


def _r_time_transform(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Pass an R callback its input factor levels without per-row bridge calls."""
    from .r._coerce import _categories

    def transform(values: Any, time: Any, riskset: Any, weights: Any, *, status: Any = None) -> Any:
        return callback(list(values), time, riskset, weights, _categories(values), status)

    transform._survival_tt_r_callback = True
    return transform


def _surv_columns(response: Any) -> dict[str, Any]:
    """Bulk normalized response columns for R's native model-frame adapter."""
    import numpy as np

    from .r._surv import Surv

    if not isinstance(response, Surv):
        raise TypeError("argument is not a Surv object")
    return {
        **{
            name: None
            if (values := getattr(response, name)) is None
            else np.asarray(values, dtype=float)
            for name in ("time", "event", "start", "time2")
        },
        "type": response.type,
        "states": list(response.states),
        "clabel": response.clabel,
    }


def _r_subset(rows: list[int]) -> Any:
    """Preserve R's missing selected rows until the shared na.action step."""
    from .r._coerce import _RSubset

    return _RSubset(rows)


def _r_logical(values: Any) -> Any:
    """Retain R's logical declaration even when every value is NA or absent."""
    from .r._expression import _ExpressionVector

    return _ExpressionVector(values, "logical")


def _r_data_frame(columns: dict[str, Any], n: int, row_names: Any = None) -> Any:
    """Carry R row names separately from formula variables and retain empty frame sizes."""
    from .r._formula import _FormulaRows

    labels = None if row_names is None else tuple(row_names)
    if labels is not None and len(labels) != n:
        raise ValueError("row names must have one label per data row")
    return _FormulaRows(columns, n, labels)


def _r_model_frame(
    columns: dict[str, Any],
    n: int,
    row_names: Any = None,
    column_metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Carry evaluated R columns, including matrix widths and factor contrasts."""
    from .r._formula import _EvaluatedModelFrame

    labels = None if row_names is None else tuple(row_names)
    if labels is not None and len(labels) != n:
        raise ValueError("row names must have one label per data row")
    return _EvaluatedModelFrame(columns, n, labels, column_metadata)


def _yates_model_metadata(fit: Any) -> dict[str, Any]:
    """Factor levels needed to rebuild a fitted Python model's R model frame."""
    from .r._fit import _formula_design_for_fit
    from .r._formula import _covariate_term_name
    from .r._types import _CategoricalDesignTerm
    from .r._yates import _design_factors

    design = _formula_design_for_fit(fit)
    if design is None:
        raise TypeError("Yates requires fitted formula metadata")
    factors = [part for part in _design_factors(design) if isinstance(part, _CategoricalDesignTerm)]
    return {
        "xlevels": {_covariate_term_name(part.term): list(part.levels) for part in factors},
        "raw_levels": {
            part.term.column: list(part.levels) for part in factors if not part.term.strata
        },
    }


def _concordance_lm_data(
    data: list[dict[str, Any]], names: list[str], options: dict[str, Any], newdata: bool = False
) -> Any:
    """R evaluates external model frames and predictions; Rust scores them."""
    from .r._concordance import _concordance_from_data, _FitData
    from .r._surv import Surv

    prepared = [
        _FitData(
            y=Surv(value["y"]),
            x=list(value["x"]),
            strata=None,
            strata_levels=(),
            weights=None if value.get("weights") is None else list(value["weights"]),
            cluster=value.get("cluster"),
            timefix=None if newdata else False,
        )
        for value in data
    ]
    return _concordance_from_data(prepared, options=options, names=names)


def _concordance_survival_models(
    fits: list[Any],
    names: list[str],
    options: dict[str, Any],
    newdata: Any | None,
    clusters: list[Any],
) -> Any:
    """Keep fitted rows in Python; R supplies cluster codes in its sort order."""
    from .r._concordance import _concordance_from_data, _fit_data

    need_weights = any(fit.weights is not None for fit in fits)
    prepared = [
        _fit_data(fit, newdata, need_weights, cluster)
        for fit, cluster in zip(fits, clusters, strict=True)
    ]
    return _concordance_from_data(prepared, options=options, names=names)


def _survexp_cox_fit(
    fit: Any, data: Any, group: Any, weights: Any, y: Any, times: Any, method: str
) -> Any:
    """Prepared R population rows applied to a Python-backed Cox rate model."""
    import numpy as np

    from .r._coxph import _check_interaction_margins, _survfit_newdata, predict_coxph

    if method.startswith("individual"):
        hazard = np.asarray(predict_coxph(fit, data, type="expected", na_action="na.fail"))
        return hazard if method == "individual.h" else np.exp(-hazard)
    _check_interaction_margins(fit)
    new, _, _ = _survfit_newdata(fit, data, individual=False, id=None, na_action="na.fail")
    engine = fit.penalized if fit.penalized is not None else fit.fit
    return engine.expected_survival(
        new.x,
        group,
        weights,
        new_strata=new.strata,
        new_offset=new.offset,
        y=y,
        times=times,
        method=method,
    ).to_arrays()


_R_BRIDGE_HELPERS = {
    "_r_factor": ("_coerce", "_r_factor"),
    "_r_term_subscript": ("_coerce", "_r_term_subscript"),
    "_cox_fit_diagnostic_messages": ("_coxph", "_cox_fit_diagnostic_messages"),
    "_frailty_encoding": ("_misc", "_frailty_encoding"),
    "_survcheck_codes": ("_misc", "_survcheck_codes"),
    "_prediction_term_names": ("_models", "_prediction_term_names"),
    "_subset_coxms_curves": ("_models", "_subset_coxms_curves"),
    "_subset_survfit_multistate": ("_models", "_subset_survfit_multistate"),
    "_survfit_multistate_structure": ("_models", "_survfit_multistate_structure"),
    "_survfit_strata_curves": ("_models", "_survfit_strata_curves"),
    "_penalty_control": ("_penalties", "_penalty_control"),
    "_subset_cox_survfit": ("_survfit", "_subset_cox_survfit"),
    "_survreg_vcov_names": ("_survreg", "survreg_vcov_names"),
}


def __getattr__(name: str) -> Any:
    """The ``survival.r`` internals the R bridge calls, imported on first use."""

    if name in _R_BRIDGE_HELPERS:
        from importlib import import_module

        module, attribute = _R_BRIDGE_HELPERS[name]
        return getattr(import_module(f"survival.r.{module}"), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
