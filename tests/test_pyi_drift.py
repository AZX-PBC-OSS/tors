"""The ``__init__.pyi`` drift guard: every name ``tors.__all__`` exports (and
no others) must appear in ``python/tors/__init__.pyi`` — functions and
classes as a ``def``, published constants as an annotated constant entry
(``CHARSET_B62: str`` and kin), never both under one name — and each stub
``def``'s full signature (argument names in order, keyword-only markers,
defaults, and per-parameter/return annotations) must match the live
function.

The stub is the typed surface: a function added to the extension and
re-exported by ``python/tors/__init__.py`` without a stub entry silently ships
untyped; a stale stub entry keeps advertising a function that no longer
exists; and a stub whose signature drifted (a keyword argument added to the
extension but not the stub, a default changed on one side, a parameter that
became keyword-only) ships wrong types; callers' type checkers validate
against the lie. The name-set pin alone (the original guard) catches only the
first two; this module's signature pin catches the third by diffing each stub
``def`` against the live function via ``inspect``.

What ``inspect`` can and cannot see (pyo3 0.29's ``text_signature`` carries
names, kinds, and defaults (the pyo3 ``#[pyfunction(signature = ...)]``
declarations in ``src/lib.rs``), but not annotations): names, keyword-only
markers, and defaults are diffed against the live function directly;
annotations (parameters and return) are pinned structurally (every parameter
annotated, every def return-annotated) because the stub is their only home.
Constants have no signature at all, so their pin is structural the same way:
present in the stub, annotated exactly ``str`` (lexical data, a live
``str``) or ``tuple[str, ...]`` (a closed name tuple like KEY_FAMILIES,
a live tuple of ``str``) — their content is contract (byte-exact, in
tests/test_first_invalid_charset.py and tests/test_scrub_pii.py), not
this guard's job, and the stub deliberately does not re-spell it. The
default comparison resolves the stub's literal expressions against the
live default values, so ``True``/``"strict"``/``None`` literals are compared
by value, and any non-literal default in a future stub fails loudly (the
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

# "No default at all", distinct from every legal default value (``None`` is
# a real default: ``deadline_ms: float | None = None``), and from inspect's
# own ``Parameter.empty`` sentinel, unified on both sides of the comparison.
_NO_DEFAULT = object()


def _stub_defs(source: str | None = None) -> dict[str, ast.FunctionDef]:
    """The pyi's top-level ``def`` nodes, by name, plus every top-level
    ``class``'s ``__init__``, keyed by the class's name (not
    ``"__init__"``); ``inspect.signature`` on a class already resolves to
    its constructor signature with ``self`` stripped, so a class is
    compared the exact same way a function is: this just has to find the
    right ast node and strip ``self`` to match that shape. ``source``
    defaults to the shipped pyi; the guard test passes mutated copies of
    the same text."""
    tree = ast.parse(source if source is not None else _PYI.read_text(encoding="utf-8"))
    defs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    typed_dicts = _typed_dict_names(tree)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        # TypedDict classes are type-only declarations (Span, ScrubPiiReport,
        # and anything inheriting one): `import tors` exposes no such class,
        # so the __init__ requirement — about runtime-constructible classes
        # — does not apply to them.
        if node.name in typed_dicts:
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


def _typed_dict_names(tree: ast.Module) -> set[str]:
    """The tree's TypedDict class names: every class with a base spelling
    ``TypedDict`` (name or attribute form), plus their transitive
    subclasses, whatever the declaration order. A class with no TypedDict
    base — direct or inherited — is never in the set."""
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    def _base_name(base: ast.expr) -> str:
        return getattr(base, "id", getattr(base, "attr", ""))

    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in classes:
            if node.name in names:
                continue
            bases = {_base_name(base) for base in node.bases}
            if "TypedDict" in bases or bases & names:
                names.add(node.name)
                changed = True
    return names


def _stub_constants() -> dict[str, ast.AnnAssign]:
    """The pyi's top-level annotated assignments — the published module
    constants (``CHARSET_B62: str`` and kin) — by name. Value-less by
    assertion: a stub constant carries its type only, never its content
    (the live module is the single spelling of a 62-character alphabet; a
    stub value would be a second place to typo it, and the byte-exact
    pins live in tests/test_first_invalid_charset.py)."""
    tree = ast.parse(_PYI.read_text(encoding="utf-8"))
    constants: dict[str, ast.AnnAssign] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        assert isinstance(node.target, ast.Name), (
            f"{ast.unparse(node.target)}: a stub constant must be a plain "
            "name (the guard does not guess at richer targets)"
        )
        assert node.value is None, (
            f"{node.target.id}: the stub spells a constant's type, not its "
            "content; the live module is the single spelling"
        )
        constants[node.target.id] = node
    return constants


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
    # a.defaults aligns with the tail of the combined positional list.
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


def test_every_dunder_all_name_and_no_others_has_a_pyi_entry() -> None:
    """``tors.__all__`` is the runtime truth (what the package re-exports);
    the stub must carry exactly that set — functions and classes as
    top-level ``def``s, published constants as annotated constant entries:
    no missing entries (a new function or constant that ships untyped), no
    extra ones (a stub entry for a name that no longer exists)."""
    stubbed = set(_stub_defs()) | set(_stub_constants())
    exported = set(tors.__all__)
    assert stubbed == exported, (
        "python/tors/__init__.pyi drifted from tors.__all__: missing from the "
        f"stub: {sorted(exported - stubbed)}; stale in the stub: "
        f"{sorted(stubbed - exported)}: keep the typed surface in lockstep "
        "with the exported one"
    )


def test_no_stub_name_is_both_a_def_and_a_constant() -> None:
    """The partition pin: a stub name is a ``def`` (function or class) or
    an annotated constant, never both. A name spelled both ways is the
    shadowing defect, and it slips past both other pins: the name-set
    test unions defs ∪ constants (the name appears either way, so the
    duplicate is invisible), and the signature test's constants
    continue-path exempts the same-named ``def`` from signature checking
    — a drifted shadowed ``def`` would ship untyped-pinned and
    undetected. The stub's grammar has no dual form; neither may the
    guard tolerate one."""
    shadowed = set(_stub_defs()) & set(_stub_constants())
    assert not shadowed, (
        "python/tors/__init__.pyi name shadowing: "
        f"{sorted(shadowed)} is spelled as both a def and an annotated "
        "constant; the def would escape the signature pin via the "
        "constants path — a published name is one or the other, never "
        "both: delete one of the two entries"
    )


def _constant_shape(annotation: ast.expr, name: str) -> str:
    """The stub's constant type spelling: ``str`` (lexical data) or
    ``tuple[str, ...]`` (a closed name tuple like KEY_FAMILIES) — anything
    else fails loudly; teach the guard the shape first, the same doctrine
    as the non-literal-default refusal."""
    if isinstance(annotation, ast.Name) and annotation.id == "str":
        return "str"
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "tuple"
        and isinstance(annotation.slice, ast.Tuple)
        and len(annotation.slice.elts) == 2
        and isinstance(annotation.slice.elts[0], ast.Name)
        and annotation.slice.elts[0].id == "str"
        and isinstance(annotation.slice.elts[1], ast.Constant)
        and annotation.slice.elts[1].value is Ellipsis
    ):
        return "tuple[str, ...]"
    raise AssertionError(
        f"{name}: the stub's constant annotation must be exactly str or "
        f"tuple[str, ...] (got {ast.unparse(annotation)}); teach the guard "
        "the shape before publishing other constant types"
    )


def _check_constant(name: str, annotation: ast.expr, live: Any) -> None:
    """The constants pin in one place (the signature test below and axis 7
    of the teeth test share it): the annotation's shape must match the live
    value — ``str`` is lexical data, ``tuple[str, ...]`` a tuple of ``str``."""
    shape = _constant_shape(annotation, name)
    if shape == "str":
        assert isinstance(live, str), (
            f"{name}: a published constant is lexical data: the live "
            f"value must be str, not {type(live).__name__}"
        )
    else:
        assert isinstance(live, tuple) and all(isinstance(v, str) for v in live), (
            f"{name}: a published name tuple must be a tuple of str, not "
            f"{type(live).__name__}"
        )


def test_every_stub_signature_matches_the_live_function() -> None:
    """The full-signature pin: for every exported function, the stub's
    parameter names in order, keyword-only markers, and literal defaults must
    equal the live function's (``inspect`` over the pyo3 text signature; the
    ``#[pyfunction]`` declaration in ``src/lib.rs``), and the stub must be
    fully annotated (every parameter, plus the return). A drift on any axis
    fails naming the function and the axis. Published constants have no
    signature: their pin is presence (the name-set test) plus the
    shape-matched live value (``str`` lexical data, ``tuple[str, ...]``
    name tuples) — a future constant of another type must teach this
    guard its shape first, the same doctrine as the non-literal-default
    refusal."""
    stubs = _stub_defs()
    constants = _stub_constants()
    for name in tors.__all__:
        assert name in stubs or name in constants, (
            f"{name}: missing from the stub (the name-set test)"
        )
        if name in constants:
            _check_constant(name, constants[name].annotation, getattr(tors, name))
            continue
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
    """The guard's teeth, proven: mutating a copy of the pyi text on each
    axis (a missing keyword argument, a changed default, a lost keyword-only
    marker, a dropped annotation, a vanished constant entry, a def shadowing
    a constant entry) must make the relevant comparison disagree; a drift
    guard that cannot fail is decoration. Runs against an in-memory mutated
    parse; the shipped pyi is untouched."""

    def signature_of(source: str, name: str) -> list[tuple[str, str, Any]]:
        tree = ast.parse(source)
        fn = one_def(tree, name)
        return _stub_params(fn)

    def one_def(tree: ast.Module, name: str) -> ast.FunctionDef:
        (fn,) = (
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return fn

    def constant_names(source: str) -> set[str]:
        return {
            node.target.id
            for node in ast.parse(source).body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

    def def_names(source: str) -> set[str]:
        return {
            node.name
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }

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
    # Axis 5: a published constant's stub entry vanishes (the name-set pin
    # over the constants half of the guard).
    missing_constant = base.replace("CHARSET_B62: str\n", "", 1)
    assert missing_constant != base, "the mutation did not apply: fix the guard test"
    assert "CHARSET_B62" not in constant_names(missing_constant)
    assert "CHARSET_B62" in constant_names(base)
    # Axis 6: a def shadows a published constant's entry (the partition
    # pin): the name-set union is blind to the duplicate, and the
    # signature test's constants continue-path would exempt the def from
    # signature checking.
    shadowed_constant = base + "\ndef CHARSET_B62(x: int) -> int: ...\n"
    assert shadowed_constant != base, "the mutation did not apply: fix the guard test"
    assert def_names(shadowed_constant) & constant_names(shadowed_constant) == {"CHARSET_B62"}
    assert not (def_names(base) & constant_names(base))
    # Axis 7: a name tuple's annotation mistyped as str (the shape pin):
    # the guard must reject the mismatch rather than treating every
    # constant as lexical data.
    mistyped_tuple = base.replace("KEY_FAMILIES: tuple[str, ...]\n", "KEY_FAMILIES: str\n", 1)
    assert mistyped_tuple != base, "the mutation did not apply: fix the guard test"
    (mistyped_node,) = [
        node
        for node in ast.parse(mistyped_tuple).body
        if isinstance(node, ast.AnnAssign) and node.target.id == "KEY_FAMILIES"
    ]
    assert _constant_shape(mistyped_node.annotation, "KEY_FAMILIES") == "str"
    try:
        _check_constant("KEY_FAMILIES", mistyped_node.annotation, tors.KEY_FAMILIES)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "the mistyped tuple constant did not fail the pin: the guard is decoration"
        )


def test_inherited_typeddict_is_exempt_from_the_init_requirement() -> None:
    """The guard's TypedDict exemption, by behavior on the real pyi text:
    appending a TypedDict that INHERITS another TypedDict must parse
    without the ``__init__`` demand (it is a type-only declaration like
    its base, and the class name is never in the runtime's value surface
    the defs dict feeds), while appending a plain value class without
    ``__init__`` must still fail."""
    base = _PYI.read_text(encoding="utf-8")

    inherited = base + (
        "\nclass ScrubPiiReportExtended(ScrubPiiReport):\n    extra: int\n"
    )
    defs = _stub_defs(inherited)
    assert "ScrubPiiReportExtended" not in defs
    assert "ScrubPiiReport" not in defs

    value_class = base + "\nclass NotATypedDict:\n    pass\n"
    try:
        _stub_defs(value_class)
    except AssertionError as e:
        assert "NotATypedDict" in str(e)
    else:
        raise AssertionError("a plain value class without __init__ was accepted")
