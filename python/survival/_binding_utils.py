from . import _survival as _core
from ._binding_manifest import BINDING_NAMES


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
        formatted = ", ".join(missing_runtime)
        raise ImportError(f"Rust extension is missing declared binding symbol(s): {formatted}")

    module_globals.update(bound)
    return requested
