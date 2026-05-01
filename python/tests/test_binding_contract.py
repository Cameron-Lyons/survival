import ast
import importlib
import re
import runpy
from pathlib import Path

from scripts.generate_binding_manifest import extract_rust_registrations

from .helpers import setup_survival_import

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "python/survival"
README = ROOT / "README.md"
MODULE_METADATA_NAMES = {"__all__"}


def _manifest_bindings() -> tuple[str, ...]:
    manifest = runpy.run_path(str(PACKAGE_ROOT / "_binding_manifest.py"))
    return tuple(manifest["BINDINGS"])


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


def _top_level_names() -> set[str]:
    tree = ast.parse((PACKAGE_ROOT / "__init__.py").read_text())
    public_modules: set[str] = set()
    sklearn_exports: set[str] = set()

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

    domain_exports = set().union(*_binding_names_by_module().values())
    return public_modules | sklearn_exports | domain_exports


def _sklearn_names() -> set[str]:
    tree = ast.parse((PACKAGE_ROOT / "sklearn_compat.py").read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and not node.name.startswith("_")
    }


def _pyi_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _readme_python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)


def _format_missing(missing: list[tuple[str, str]]) -> str:
    return "\n".join(f"{source}: {name}" for source, name in missing)


def test_generated_manifest_matches_rust_registration():
    assert _manifest_bindings() == extract_rust_registrations(ROOT)


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
    missing = [name for name in _manifest_bindings() if not hasattr(core, name)]

    assert missing == []
