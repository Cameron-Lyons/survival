from types import ModuleType


def setup_survival_import() -> ModuleType:
    try:
        import survival
    except ImportError as exc:
        raise ImportError(
            "Could not import `survival`. Build the extension first, for example:\n"
            "  maturin develop --release\n"
            "Then run:\n"
            "  pytest python/tests -v"
        ) from exc

    return survival
