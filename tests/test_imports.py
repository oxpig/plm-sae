"""Every Python file compiles, and every library module imports.

Modules are imported in one process. A module counts as a script (compiled, not imported) when it runs work at import
time: a top-level statement other than imports, definitions, assignments, a docstring or an `if __name__ == "__main__"`
block. Modules that need optional third-party code that is not installed (ProGen2 code, FlashTAP, AbLang-2) are
skipped and listed."""
import ast
import importlib
import py_compile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("plm_sae", "paper", "third_party/dictionary_learning")
OPTIONAL = ("models", "flash_abb", "ablang2", "anarci", "iglm", "sparsify", "wandb")


def py_files():
    for top in PACKAGES + ("tests",):
        for f in sorted((REPO / top).rglob("*.py")):
            if "__pycache__" not in f.parts and "fixtures" not in f.parts:
                yield f


def _is_main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__")


def is_script(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if _is_main_guard(node):
            continue
        if isinstance(node, ast.Try) and all(isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign)) for n in node.body):
            continue
        return True
    return False


def module_name(path):
    rel = path.relative_to(REPO).with_suffix("")
    if rel.parts[0] == "third_party":
        rel = rel.relative_to("third_party")
    parts = list(rel.parts)
    if parts[-1] == "__init__": parts = parts[:-1]
    return ".".join(parts)


def test_compile_all(tmp_path):
    bad = []
    for i, f in enumerate(py_files()):
        try:
            py_compile.compile(str(f), cfile=str(tmp_path / f"{i}.pyc"), doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(f"{f}: {e.msg}")
    assert not bad, "\n".join(bad)


def test_import_library_modules():
    failed, skipped, scripts, ok = [], [], [], 0
    for top in PACKAGES:
        for f in sorted((REPO / top).rglob("*.py")):
            if "__pycache__" in f.parts: continue
            name = module_name(f)
            if is_script(f): scripts.append(name); continue
            try:
                importlib.import_module(name); ok += 1
            except ImportError as e:
                root = (getattr(e, "name", None) or "").split(".")[0]
                if root in OPTIONAL: skipped.append(f"{name} ({root})")
                else: failed.append(f"{name}: {type(e).__name__}: {e}")
            except Exception as e:  # noqa: BLE001
                failed.append(f"{name}: {type(e).__name__}: {e}")
    print(f"imported {ok}; scripts (compiled only) {len(scripts)}; skipped for optional packages {len(skipped)}: {skipped}")
    assert not failed, "\n".join(failed)
