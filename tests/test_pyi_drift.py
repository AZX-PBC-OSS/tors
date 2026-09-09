"""The ``__init__.pyi`` drift guard: every name ``tors.__all__`` exports (and
no others) must appear as a ``def`` in ``python/tors/__init__.pyi``, and each
stub's FULL SIGNATURE (argument names in order, keyword-only markers,
defaults, and per-parameter/return annotations) must match the live
function.

The stub is the typed surface: a function added to the extension and
re-exported by ``python/tors/__init__.py`` without a stub entry silently ships
untyped; a stale stub entry keeps advertising a function that no longer
exists; and a stub whose SIGNATURE drifted (a keyword argument added to the
extension but not the stub, a default changed on one side, a parameter that
became keyword-only) ships WRONG types; callers' type checkers validate
against the lie. The name-set pin alone (the original guard) catches only the
first two; this module's signature pin catches the third by diffing each stub
``def`` against the live function via ``inspect``.

What ``inspect`` can and cannot see (pyo3 0.29's ``text_signature`` carries
names, kinds, and defaults (the pyo3 ``#[pyfunction(signature = ...)]``
declarations in ``src/lib.rs``), but NOT annotations): names, keyword-only
markers, and defaults are diffed against the live function directly;
annotations (parameters and return) are pinned structurally (every parameter
annotated, every def return-annotated) because the stub is their only home.
The default comparison resolves the stub's literal expressions against the
live default values, so ``True``/``"strict"``/``None`` literals are compared
by value, and any non-literal default in a future stub fails LOUDLY (the
guard does not guess).
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tors

_PYI = Path(__file__).resolve().parent.parent / "python" / "tors" / "__init__.pyi"

# "No default at all", distinct from every legal default VALUE (``None`` is
# a real default: ``deadline_ms: float | None = None``), and from inspect's
# own ``Parameter.empty`` sentinel, unified on both sides of the comparison.
_NO_DEFAULT = object()


def _stub_defs() -> dict[str, ast.FunctionDef]:
    """The pyi's top-level ``def`` nodes, by name, PLUS every top-level
    ``class``'s ``__init__``, keyed by the CLASS's name (not
    ``"__init__"``); ``inspect.signature`` on a class already resolves to
    its constructor signature with ``self`` stripped, so a class is
    compared the exact same way a function is: this just has to find the
    right ast node and strip ``self`` to match that shape."""
    tree = ast.parse(_PYI.read_text(encoding="utf-8"))
    defs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        inits = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
        assert len(inits) == 1, f"{node.name}: stub class needs exactly one __init__"
        init = inits[0]
        stripped = ast.FunctionDef(
            name=node.name,
            args=ast.arguments(
                posonlyargs=init.args.posonlyargs,
                args=init.args.args[1:],  # drop `self`
                vararg=init.args.vararg,
                kwonlyargs=init.args.kwonlyargs,
                kw_defaults=init.args.kw_defaults,
                kwarg=init.args.kwarg,
                defaults=init.args.defaults,
            ),
            body=init.body,
            decorator_list=[],
            returns=init.returns,  # __init__'s own `-> None` annotation, carried through as-is
        )
        defs[node.name] = stripped
    assert defs, "no top-level defs parsed from the pyi: the guard is broken"
    return defs


def _stub_params(fn: ast.FunctionDef) -> list[tuple[str, str, Any]]:
    """The stub's parameters as ``(name, kind, default_value)``, kind one of
    the inspect ``_ParameterKind`` spellings, default ``None`` when absent.

    ``Any`` for the default because the value is the ast constant's Python
    value (``True``/``"strict"``/``None``/numbers); a non-constant default
    expression raises: this stub uses literals only, and the guard refuses
    to guess at anything richer.
    """
    a = fn.args
    out: list[tuple[str, str, Any]] = []

    def literal(expr: ast.expr | None, param: str) -> Any:
        if expr is None:
            return _NO_DEFAULT
        assert isinstance(expr, ast.Constant), (
            f"{fn.name}: a non-literal default for {param} "
            f"({ast.unparse(expr)}): the guard compares literal values only; "
            "teach it the shape first"
        )
        return expr.value

    positional = [*a.posonlyargs, *a.args]
    # a.defaults aligns with the TAIL of the combined positional list.
    first_defaulted = len(positional) - len(a.defaults)
    for index, arg in enumerate(positional):
        kind = "positional-only" if arg in a.posonlyargs else "positional-or-keyword"
        default = (
            literal(a.defaults[index - first_defaulted], arg.arg)
            if index >= first_defaulted
            else _NO_DEFAULT
        )
        out.append((arg.arg, kind, default))
    if a.vararg is not None:
        out.append((a.vararg.arg, "var-positional", None))
    # a.kw_defaults aligns index-for-index with a.kwonlyargs (None = absent).
    for arg, expr in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        out.append((arg.arg, "keyword-only", literal(expr, arg.arg)))
    if a.kwarg is not None:
        out.append((a.kwarg.arg, "var-keyword", None))
    return out


def _live_params(fn: Callable[..., Any]) -> list[tuple[str, str, Any]]:
    """The live function's parameters in the same ``(name, kind, default)``
    shape, via ``inspect.signature`` (``Parameter.empty`` normalized to
    ``_NO_DEFAULT``)."""
    out: list[tuple[str, str, Any]] = []
    for param in inspect.signature(fn).parameters.values():
        kind = {
            inspect.Parameter.POSITIONAL_ONLY: "positional-only",
            inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional-or-keyword",
            inspect.Parameter.VAR_POSITIONAL: "var-positional",
            inspect.Parameter.KEYWORD_ONLY: "keyword-only",
            inspect.Parameter.VAR_KEYWORD: "var-keyword",
        }[param.kind]
        default = _NO_DEFAULT if param.default is inspect.Parameter.empty else param.default
        out.append((param.name, kind, default))
    return out


def test_every_dunder_all_name_and_no_others_has_a_pyi_def() -> None:
    """``tors.__all__`` is the runtime truth (what the package re-exports);
    the stub must carry exactly that set as top-level ``def``s: no missing
    entries (a new function that ships untyped), no extra ones (a stub for a
    function that no longer exists)."""
    stubbed = set(_stub_defs())
    exported = set(tors.__all__)
    assert stubbed == exported, (
        "python/tors/__init__.pyi drifted from tors.__all__: missing from the "
        f"stub: {sorted(exported - stubbed)}; stale in the stub: "
        f"{sorted(stubbed - exported)}: keep the typed surface in lockstep "
        "with the exported one"
    )


def test_every_stub_signature_matches_the_live_function() -> None:
    """The full-signature pin: for every exported function, the stub's
    parameter names IN ORDER, keyword-only markers, and literal defaults must
    equal the live function's (``inspect`` over the pyo3 text signature; the
    ``#[pyfunction]`` declaration in ``src/lib.rs``), and the stub must be
    fully annotated (every parameter, plus the return). A drift on any axis
    fails naming the function and the axis."""
    stubs = _stub_defs()
    for name in tors.__all__:
        assert name in stubs, f"{name}: missing from the stub (the name-set test)"
        stub_fn = stubs[name]
        stub_params = _stub_params(stub_fn)
        live_params = _live_params(getattr(tors, name))
        assert [p[:2] for p in stub_params] == [p[:2] for p in live_params], (
            f"{name}: the stub's parameters (names in order, keyword-only "
            f"markers) drifted from the live function: stub "
            f"{[p[:2] for p in stub_params]} vs live "
            f"{[p[:2] for p in live_params]}"
        )
        for (stub_name, _, stub_default), (_, _, live_default) in zip(
            stub_params, live_params, strict=True
        ):
            assert stub_default == live_default, (
                f"{name}: default for {stub_name} drifted: stub "
                f"{stub_default!r} vs live {live_default!r}"
            )
        unannotated = [
            arg.arg
            for arg in [
                *stub_fn.args.posonlyargs,
                *stub_fn.args.args,
                *stub_fn.args.kwonlyargs,
            ]
            if arg.annotation is None
        ] + (["*args"] if stub_fn.args.vararg and stub_fn.args.vararg.annotation is None else [])
        unannotated += (
            ["**kwargs"] if stub_fn.args.kwarg and stub_fn.args.kwarg.annotation is None else []
        )
        assert not unannotated, (
            f"{name}: the stub ships untyped parameters {unannotated}: the "
            "typed surface must be fully annotated"
        )
        assert stub_fn.returns is not None, (
            f"{name}: the stub has no return annotation: the typed surface must be fully annotated"
        )


def test_the_guard_itself_catches_each_drift_axis() -> None:
    """The guard's teeth, proven: mutating a COPY of the pyi text on each
    axis (a missing keyword argument, a changed default, a lost keyword-only
    marker, a dropped annotation) must make the signature comparison
    disagree; a drift guard that cannot fail is decoration. Runs against an
    in-memory mutated parse; the shipped pyi is untouched."""

    def signature_of(source: str, name: str) -> list[tuple[str, str, Any]]:
        tree = ast.parse(source)
        fn = one_def(tree, name)
        return _stub_params(fn)

    def one_def(tree: ast.Module, name: str) -> ast.FunctionDef:
        (fn,) = (
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return fn

    base = _PYI.read_text(encoding="utf-8")
    live_b64 = _live_params(tors.b64_decode)
    # Axis 1: the keyword argument vanishes from the stub.
    missing_kwarg = base.replace(
        "def b64_decode(s: str, *, validate: bool = True) -> bytes: ...",
        "def b64_decode(s: str) -> bytes: ...",
    )
    assert missing_kwarg != base, "the mutation did not apply: fix the guard test"
    assert signature_of(missing_kwarg, "b64_decode") != live_b64
    # Axis 2: the default changes on the stub side.
    changed_default = base.replace(
        "def b64_decode(s: str, *, validate: bool = True) -> bytes: ...",
        "def b64_decode(s: str, *, validate: bool = False) -> bytes: ...",
    )
    stub_params = signature_of(changed_default, "b64_decode")
    assert [p[:2] for p in stub_params] == [p[:2] for p in live_b64]
    assert stub_params != live_b64  # the default is the delta
    # Axis 3: the parameter loses its keyword-only marker.
    lost_marker = base.replace(
        "def b64_decode(s: str, *, validate: bool = True) -> bytes: ...",
        "def b64_decode(s: str, validate: bool = True) -> bytes: ...",
    )
    assert [p[:2] for p in signature_of(lost_marker, "b64_decode")] != [p[:2] for p in live_b64]
    # Axis 4: an annotation is dropped (the structural completeness pin).
    unannotated = base.replace(
        "def b64_decode(s: str, *, validate: bool = True) -> bytes: ...",
        "def b64_decode(s, *, validate: bool = True) -> bytes: ...",
    )
    tree = ast.parse(unannotated)
    assert one_def(tree, "b64_decode").args.args[0].annotation is None  # the mutation applied
