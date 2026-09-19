"""The package version's public contract: ``tors.__version__`` is the
installed distribution's version — read from the distribution metadata,
not a second literal in the source that release tooling cannot bump —
and it is declared on the typed surface (the stub), so a type check on
``tors.__version__`` resolves. The metadata-off fallback (``"unknown"``)
is import-time and environment-shaped (a source tree imported off-path);
it is deliberately not reloaded-and-simulated here: a module reload of
the Rust-binding package is the flake-shaped kind of test that pins
implementation machinery, not behavior."""

import ast
import importlib.metadata
from pathlib import Path

import tors


def test_version_is_the_installed_distribution_version():
    """The attribute reads the same source ``pip``/``uv`` report, so they
    cannot disagree."""
    assert isinstance(tors.__version__, str)
    assert tors.__version__ == importlib.metadata.version("tors")


def test_version_is_declared_on_the_typed_surface():
    """The stub declares the attribute with the same shape the runtime
    gives it: a plain ``str`` annotation, no default (the value is
    import-computed, not a constant the stub could lie about)."""
    pyi = Path(tors.__file__).with_name("__init__.pyi")
    declarations = [
        node
        for node in ast.parse(pyi.read_text(encoding="utf-8")).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "__version__"
    ]
    assert len(declarations) == 1
    assert ast.unparse(declarations[0].annotation) == "str"
