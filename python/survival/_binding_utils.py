from . import _survival as _core
from ._binding_manifest import BINDING_NAMES, ML_BINDING_NAMES

_ML_SENTINELS = ("DeepSurv", "survival_forest", "GradientBoostSurvival")
_ML_AVAILABLE = any(hasattr(_core, name) for name in _ML_SENTINELS)
_OPTIONAL_MISSING_BINDING_FEATURES = {} if _ML_AVAILABLE else dict.fromkeys(ML_BINDING_NAMES, "ml")


def _install_optional_binding_getattr(module_globals, missing_features):
    def _optional_binding_getattr(name):
        feature = missing_features.get(name)
        if feature is not None:
            raise AttributeError(
                f"Rust binding {name!r} is unavailable because this extension was "
                f"built without the {feature!r} feature"
            )
        raise AttributeError(
            f"module {module_globals.get('__name__', '<module>')!r} has no attribute {name!r}"
        )

    module_globals["__getattr__"] = _optional_binding_getattr


def bind_names(module_globals, names):
    requested = list(names)
    missing_manifest = [name for name in requested if name not in BINDING_NAMES]
    if missing_manifest:
        formatted = ", ".join(missing_manifest)
        raise ImportError(f"binding manifest is missing declared symbol(s): {formatted}")

    bound = {}
    missing_runtime = []
    for name in requested:
        try:
            bound[name] = getattr(_core, name)
        except AttributeError:
            missing_runtime.append(name)

    if missing_runtime:
        missing_features = {
            name: _OPTIONAL_MISSING_BINDING_FEATURES[name]
            for name in missing_runtime
            if name in _OPTIONAL_MISSING_BINDING_FEATURES
        }
        unexpected_missing = [name for name in missing_runtime if name not in missing_features]
        if unexpected_missing:
            formatted = ", ".join(unexpected_missing)
            raise ImportError(f"Rust extension is missing declared binding symbol(s): {formatted}")
        _install_optional_binding_getattr(module_globals, missing_features)

    module_globals.update(bound)
    return [name for name in requested if name in bound]
