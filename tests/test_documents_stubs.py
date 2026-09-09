"""The documents stubs' drift guard — the tors.documents layer's
counterpart of test_pyi_drift.py (which holds the BASE wheel's stub to
``tors.__all__`` and the live pyo3 functions). Two layers of stub, two
pins:

- The PAYLOAD stubs (``tors_documents/__init__.pyi`` and ``aio.pyi``) are
  held to the live typed surface: every ``__all__`` name appears in the
  stub (nothing ships untyped, nothing stale ships), the enum classes
  carry the live members, and every stub ``def`` — package and aio alike
  — is diffed against ``inspect.signature`` of the live function:
  parameter names in order, kinds, defaults, and ANNOTATIONS. The
  annotations are comparable here (unlike the base wheel's pyo3 surface,
  the documents functions are typed Python wrappers, so inspect sees the
  full contract — the aio wrappers are ``wraps`` of the typed sync
  functions, and inspect follows the chain). Default semantics follow pyi
  convention: ``= ...`` means "optional, value deliberately not pinned"
  (only the default's EXISTENCE is compared — the payload spells enum and
  None-machinery defaults the stub cannot), while a spelled literal
  (``= None``) is compared by value.
- The SHIM stubs (``python/tors/documents/*.pyi``) are the payload stubs'
  standalone twins — base-only type checks must resolve without the
  payload wheel — so their bodies are compared node-for-node, modulo the
  deltas the mirror doctrine documents: each file's own header docstring;
  the ``__version__`` blocks (same declaration on both sides,
  deliberately different nuance docstrings); and, on the aio pair, the
  import line (the shim imports the names from ``tors.documents``, the
  payload from ``tors_documents`` — the one line that IS the two-wheel
  split). Everything else — classes, enum members, signatures, every
  inner docstring — must be identical.
- The sync-only rule, stub side: ``sniff`` present in both package stubs,
  absent from both aio stubs (the runtime side of the same rule is pinned
  by the engines suite's aio gate).

Skipped loudly when the payload wheel is not built: this guard compares
the payload's stubs against the payload's live surface, so without the
payload there is nothing to hold (the base wheel's own stub guard,
test_pyi_drift.py, stays payload-free by design)."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

try:
    import tors_documents
    from tors_documents import aio as payload_aio
except ImportError:  # pragma: no cover - the source-checkout alternative
    pytest.skip(
        "documents payload not built: the stub guard compares its stubs to its "
        "live surface (uv sync --reinstall-package tors-documents, or maturin "
        "develop in tors-documents/) — there is nothing to validate",
        allow_module_level=True,
    )

_ROOT = Path(__file__).resolve().parent.parent
_PAYLOAD_STUB = _ROOT / "tors-documents" / "python" / "tors_documents" / "__init__.pyi"
_PAYLOAD_AIO_STUB = _ROOT / "tors-documents" / "python" / "tors_documents" / "aio.pyi"
_SHIM_STUB = _ROOT / "python" / "tors" / "documents" / "__init__.pyi"
_SHIM_AIO_STUB = _ROOT / "python" / "tors" / "documents" / "aio.pyi"

# "No default at all", distinct from every legal default VALUE (``None`` is
# a real default; Ellipsis is the pyi spelling of "optional, value not
# pinned"), and from inspect's own ``Parameter.empty`` sentinel — unified
# on both sides of the comparison.
_NO_DEFAULT = object()

# The two spellings of "import the documents surface's types": the payload
# stubs' module name and the shim stubs' (the aio pair's one intentional
# difference, normalized away by _normalized_nodes).
_DOCUMENTS_SURFACE_IMPORTS = {"tors_documents", "tors.documents"}


def _stub_functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """The pyi's top-level function defs (sync and async alike), by name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert defs, f"{path}: no top-level defs parsed — the guard is broken"
    return defs


def _annotation(annotation: ast.expr | None) -> str | None:
    """The stub annotation's source text (``None`` = unannotated — a
    typed-surface defect the signature test fails on)."""
    return None if annotation is None else ast.unparse(annotation)


def _stub_params(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, str, Any, str | None]]:
    """The stub function's parameters as ``(name, kind, default,
    annotation)`` — kind an inspect ``_ParameterKind`` spelling, default
    ``_NO_DEFAULT`` when absent / Ellipsis when the pyi spells ``= ...`` /
    the ast constant's value otherwise, annotation per :func:`_annotation`.
    Any non-constant default raises: these stubs spell literals and
    ``...`` only, and the guard refuses to guess at anything richer."""
    a = fn.args
    out: list[tuple[str, str, Any, str | None]] = []

    def literal(expr: ast.expr | None, param: str) -> Any:
        if expr is None:
            return _NO_DEFAULT
        assert isinstance(expr, ast.Constant), (
            f"{fn.name}: a non-literal default for {param} ({ast.unparse(expr)}): "
            "the guard compares literal values and `...` only; teach it the shape first"
        )
        return expr.value

    positional = [*a.posonlyargs, *a.args]
    first_defaulted = len(positional) - len(a.defaults)
    for index, arg in enumerate(positional):
        kind = "positional-only" if arg in a.posonlyargs else "positional-or-keyword"
        default = (
            literal(a.defaults[index - first_defaulted], arg.arg)
            if index >= first_defaulted
            else _NO_DEFAULT
        )
        out.append((arg.arg, kind, default, _annotation(arg.annotation)))
    if a.vararg is not None:
        out.append((a.vararg.arg, "var-positional", _NO_DEFAULT, _annotation(a.vararg.annotation)))
    for arg, expr in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        out.append((arg.arg, "keyword-only", literal(expr, arg.arg), _annotation(arg.annotation)))
    if a.kwarg is not None:
        out.append((a.kwarg.arg, "var-keyword", _NO_DEFAULT, _annotation(a.kwarg.annotation)))
    return out


_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: "positional-only",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional-or-keyword",
    inspect.Parameter.VAR_POSITIONAL: "var-positional",
    inspect.Parameter.KEYWORD_ONLY: "keyword-only",
    inspect.Parameter.VAR_KEYWORD: "var-keyword",
}


def _live_signature(
    fn: Callable[..., Any],
) -> tuple[list[tuple[str, str, Any, str | None]], str | None]:
    """The live function's parameters and return in the stub's comparison
    shape, via ``inspect.signature`` (``Parameter.empty`` normalized to
    ``_NO_DEFAULT``/``None``; the annotations arrive as source text because
    the payload's wrappers evaluate them lazily — and the aio wrappers'
    ``wraps`` chain resolves to the typed sync signatures)."""
    sig = inspect.signature(fn)
    params: list[tuple[str, str, Any, str | None]] = []
    for param in sig.parameters.values():
        default = _NO_DEFAULT if param.default is inspect.Parameter.empty else param.default
        annotation = None if param.annotation is inspect.Parameter.empty else param.annotation
        params.append((param.name, _KINDS[param.kind], default, annotation))
    returns = None if sig.return_annotation is inspect.Signature.empty else sig.return_annotation
    return params, returns


def _default_agrees(stub_default: Any, live_default: Any) -> bool:
    """The pyi default semantics: absent must match absent; ``...`` asserts
    only that a default EXISTS (the value is the payload's own machinery,
    deliberately unpinned by the stub); a spelled literal is pinned by
    value."""
    if stub_default is _NO_DEFAULT or live_default is _NO_DEFAULT:
        return stub_default is live_default
    if stub_default is ...:
        return True
    return stub_default == live_default


def _normalized_nodes(path: Path) -> list[str]:
    """The stub's top-level nodes as comparable dumps, MODULO the deltas the
    mirror doctrine documents (see the module docstring): the module
    docstring and the ``from __future__`` import drop entirely; an import
    of the documents surface is canonicalized to the imported names alone;
    the ``__version__`` declaration keeps its target and annotation (both
    sides declare the same shape) but drops its docstring (the two sides
    carry deliberately different nuance text). Everything else dumps
    verbatim — inner docstrings included — so wording drift anywhere else
    in the mirror fails."""
    out: list[str] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue  # the module docstring: each stub's own header
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in _DOCUMENTS_SURFACE_IMPORTS
            and node.level == 0
        ):
            out.append(f"documents-surface-import:{sorted(alias.name for alias in node.names)}")
            continue
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
        ):
            out.append(f"__version__:{ast.unparse(node.annotation)}")
            continue
        out.append(ast.dump(node))
    return out


# --- the payload stubs vs the live surface ---------------------------------------


@pytest.mark.parametrize(
    ("stub", "exported"),
    [
        (_PAYLOAD_STUB, tors_documents.__all__),
        (_PAYLOAD_AIO_STUB, payload_aio.__all__),
    ],
    ids=["package", "aio"],
)
def test_the_payload_stubs_name_exactly_the_exported_surface(
    stub: Path, exported: list[str]
) -> None:
    """The runtime ``__all__`` is the truth; the stub must carry exactly
    that set as top-level defs, classes, and declarations: no missing
    entries (a new export that ships untyped), no stale ones (a stub for a
    name that no longer exists) — the name-set pin the base wheel's guard
    runs, on both of this layer's stubs."""
    stubbed: set[str] = set()
    for node in ast.parse(stub.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stubbed.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            stubbed.add(node.target.id)
    assert stubbed == set(exported), (
        f"{stub} drifted from the exported surface: missing from the stub: "
        f"{sorted(set(exported) - stubbed)}; stale in the stub: "
        f"{sorted(stubbed - set(exported))}"
    )


def test_the_payload_stub_enums_carry_the_live_members() -> None:
    """The str enums are the accepted-string vocabulary: every stub enum's
    members (names and VALUES, in declaration order) must be the live
    enum's — a member drift here is wrong types at every call site that
    spells the string."""
    tree = ast.parse(_PAYLOAD_STUB.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        live = getattr(tors_documents, node.name, None)
        if not (isinstance(live, type) and issubclass(live, Enum)):
            continue  # PdfClassification/NeedsOcrError: not enums (the name-set pin covers them)
        stub_members = [
            (stmt.targets[0].id, stmt.value.value)
            for stmt in node.body
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
        ]
        assert stub_members == [(member.name, member.value) for member in live], (
            f"{node.name}: the stub's enum members drifted from the live enum"
        )


@pytest.mark.parametrize(
    ("stub", "module"),
    [(_PAYLOAD_STUB, tors_documents), (_PAYLOAD_AIO_STUB, payload_aio)],
    ids=["package", "aio"],
)
def test_every_payload_stub_signature_matches_the_live_typed_function(
    stub: Path, module: ModuleType
) -> None:
    """The full-signature pin: for every function in the payload stub, the
    stub's parameter names IN ORDER and kinds, its defaults (by existence
    where the pyi spells ``...``, by value where it spells a literal), and
    its parameter and return ANNOTATIONS must equal the live function's
    ``inspect.signature`` — the aio lane included (the live aio wrappers
    are ``wraps`` of the typed sync functions, so their signatures ARE the
    typed surface's). A drift on any axis fails naming the function and
    the axis."""
    for name, stub_fn in _stub_functions(stub).items():
        live_fn = getattr(module, name, None)
        assert live_fn is not None, f"{name}: stubbed but absent from the live {module.__name__}"
        stub_params = _stub_params(stub_fn)
        live_params, live_return = _live_signature(live_fn)
        assert [p[:2] for p in stub_params] == [p[:2] for p in live_params], (
            f"{name}: the stub's parameters (names in order, kinds) drifted from "
            f"the live function: stub {[p[:2] for p in stub_params]} vs live "
            f"{[p[:2] for p in live_params]}"
        )
        for stub_p, live_p in zip(stub_params, live_params, strict=True):
            assert _default_agrees(stub_p[2], live_p[2]), (
                f"{name}: default for {stub_p[0]} drifted: stub {stub_p[2]!r} vs live {live_p[2]!r}"
            )
            assert stub_p[3] == live_p[3], (
                f"{name}: annotation for {stub_p[0]} drifted: stub {stub_p[3]!r} "
                f"vs live {live_p[3]!r}"
            )
        assert _annotation(stub_fn.returns) == live_return, (
            f"{name}: return annotation drifted: stub {_annotation(stub_fn.returns)!r} "
            f"vs live {live_return!r}"
        )


# --- the shim stubs vs the payload stubs (the mirror) ----------------------------


@pytest.mark.parametrize(
    ("payload_stub", "shim_stub"),
    [(_PAYLOAD_STUB, _SHIM_STUB), (_PAYLOAD_AIO_STUB, _SHIM_AIO_STUB)],
    ids=["package", "aio"],
)
def test_the_shim_stubs_mirror_the_payload_stubs_modulo_the_documented_deltas(
    payload_stub: Path, shim_stub: Path
) -> None:
    """The standalone-mirror doctrine (the shim stub's own header): the
    shim stub is the payload stub's twin, compared node-for-node —
    signature, enum, class, and docstring drift between the two fails
    loudly instead of shipping a base-only stub that documents a different
    API than the payload's. A NEW delta must be taught to
    :func:`_normalized_nodes` deliberately, never absorbed silently."""
    payload_nodes = _normalized_nodes(payload_stub)
    shim_nodes = _normalized_nodes(shim_stub)
    assert len(payload_nodes) == len(shim_nodes), (
        f"{shim_stub}: the stub pair carries different top-level node counts "
        f"({len(payload_nodes)} vs {len(shim_nodes)})"
    )
    for index, (payload_node, shim_node) in enumerate(zip(payload_nodes, shim_nodes, strict=True)):
        assert payload_node == shim_node, (
            f"{shim_stub}: node {index} diverged from the payload stub "
            f"(payload {payload_node[:200]} vs shim {shim_node[:200]})"
        )


def test_sniff_is_package_only_in_the_stubs_as_it_is_at_runtime() -> None:
    """The sync-only rule, stub side: sniff's marker scan is
    microsecond-scale, so neither aio stub carries it — and both package
    stubs must (it is the surface's cheapest entry point and the bytes
    pipeline's first step). The runtime side of the same rule is pinned by
    the engines suite's aio gate (``"sniff" not in aio.__all__`` on both
    modules)."""
    for stub in (_PAYLOAD_STUB, _SHIM_STUB):
        assert "sniff" in _stub_functions(stub), f"{stub}: sniff missing from the package stub"
    for stub in (_PAYLOAD_AIO_STUB, _SHIM_AIO_STUB):
        assert "sniff" not in _stub_functions(stub), f"{stub}: sniff must stay sync-only"
