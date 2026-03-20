from . import _survival as _core


def bind_names(module_globals, names):
    available = [name for name in names if hasattr(_core, name)]
    module_globals.update({name: getattr(_core, name) for name in available})
    return available
