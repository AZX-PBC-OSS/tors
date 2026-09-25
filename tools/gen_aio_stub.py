"""Regenerate ``python/tors/aio.pyi`` from ``python/tors/__init__.pyi``.

The async facade's signatures are the sync signatures with ``async def``
in place of ``def`` and the same return annotation (an ``async def``
function's annotation is what awaiting it resolves to, not
``Awaitable[T]``; that describes the coroutine object itself, not its
result); deriving the stub from the sync stub keeps the two from
drifting apart, and ``tests/test_aio.py`` fails loudly if the committed
artifact is stale (regenerate with this script as part of the same
change that touches the sync stub).

Only the curated large-input functions named in ``tors.aio._WRAPPED``
are translated (see ``tors/aio.py``'s module docstring for why the rest
of ``tors`` intentionally has no async twin: most functions are cheap
enough that a thread hop would cost more than the call itself). Only the
signatures travel: the extraction is each function's ``def`` lines
(``lines[node.lineno - 1 : node.end_lineno]``), so the sync stub's
comments stay in the sync stub and the generated ``aio.pyi`` carries no
comments at all.

The translated text is piped through ``ruff format`` (the repo's
formatter, a dev-group dependency, invoked as ``python -m ruff``)
before it is returned or written: the emitted artifact is
ruff-format-clean by construction, so the committed stub, the
generator, and the formatter can never disagree about style: the
freshness gate in ``tests/test_aio.py`` compares this function's output
against the committed file, and both sides are the formatted spelling.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SYNC_STUB = Path(__file__).resolve().parent.parent / "python" / "tors" / "__init__.pyi"
ASYNC_STUB = Path(__file__).resolve().parent.parent / "python" / "tors" / "aio.pyi"


def _ruff_format(text: str) -> str:
    """Run ``ruff format`` over ``text`` as if it were ``aio.pyi``.

    ``--stdin-filename`` makes ruff apply the ``.pyi`` spacing rules (zero
    blank lines between top-level defs) and the project's ``pyproject.toml``
    config, so the output is exactly what formatting the committed file in
    place would produce. Raises ``RuntimeError`` rather than emitting an
    unformatted stub if ruff is missing or refuses the input: a stale or
    malformed artifact must never be written silently.
    """
    # Fixed argv, no shell, the repo's own formatter — then the repo's
    # own check --fix (the import-sort fixer merges the multi-line
    # `from tors import` header the type-layer import emits): the
    # emitted artifact must be CHECK-clean, not merely format-clean —
    # the freshness pin compares against exactly this pipeline.
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(ASYNC_STUB), "-"],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ruff format failed (rc={proc.returncode}) while generating {ASYNC_STUB}: "
            f"{proc.stderr.strip()}"
        )
    fixed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--fix",
            "--stdin-filename",
            str(ASYNC_STUB),
            "-",
        ],
        input=proc.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    if fixed.returncode != 0:
        raise RuntimeError(
            f"ruff check --fix failed (rc={fixed.returncode}) while generating "
            f"{ASYNC_STUB}: {fixed.stderr.strip()}"
        )
    return fixed.stdout


def _spells_name(node: ast.FunctionDef, name: str) -> bool:
    """Whether one sync stub signature's annotations spell a given bare
    name (``Any``, ``Hashable``).

    Walks the annotation subtrees (parameters and return) only: a comment
    or docstring elsewhere in the source may name the type in prose, and
    the header must import a name exactly when a signature needs it (an
    unused import in the stub is as stale as a missing one).
    """
    args = node.args
    annotations = [
        *(arg.annotation for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]),
        args.vararg.annotation if args.vararg else None,
        args.kwarg.annotation if args.kwarg else None,
        node.returns,
    ]
    for annotation in annotations:
        if annotation is None:
            continue
        for sub in ast.walk(annotation):
            if (isinstance(sub, ast.Name) and sub.id == name) or (
                isinstance(sub, ast.Attribute) and sub.attr == name
            ):
                return True
    return False


def _spells_any(node: ast.FunctionDef) -> bool:
    return _spells_name(node, "Any")


def _translate(source: str, wrapped: frozenset[str]) -> tuple[str, int]:
    tree = ast.parse(source)
    lines = source.split("\n")
    # The body is collected first so the import header can react to it:
    # `Any` travels only when a translated signature actually spells it
    # (the repair family's `dict[str, Any] | bool | type[Any]` schema
    # annotations); an unused import in the stub is as stale as a missing
    # one, and ruff's F401/F821 gates read this file.
    body: list[str] = []
    translated = 0
    needs_any = False
    needs_hashable = False
    used_type_names: set[str] = set()
    import re

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in wrapped:
            continue
        chunk = "\n".join(lines[node.lineno - 1 : node.end_lineno]).rstrip()
        chunk = chunk.replace(f"def {node.name}(", f"async def {node.name}(", 1)
        body.append(chunk)
        body.append("")
        body.append("")
        translated += 1
        needs_any = needs_any or _spells_any(node)
        needs_hashable = needs_hashable or _spells_name(node, "Hashable")
        # The sync stub's type-layer names (the structural TypedDicts and
        # the recursive alias, all module-level assigns in
        # ``__init__.pyi``): a translated signature that spells one must
        # import it — the generated stub is a standalone module, and
        # ruff's F821 gate reads it. Word-boundary match: a name must
        # appear as ITSELF, not as a substring of another identifier.

        for type_name in (
            "JSONValue",
            "Span",
            "ScrubPiiReport",
            "RepairAction",
            "DedupResult",
            "GroundingResult",
            "SentenceGrounding",
        ):
            if re.search(rf"\b{type_name}\b", chunk):
                used_type_names.add(type_name)
    # The collections.abc names travel only when a translated signature
    # spells them (the same freshness rule as Any below: an unused
    # import in the stub is as stale as a missing one); the scan runs
    # inline in the header build below.
    header = [
        '"""The awaitable spellings of tors\'s large-input functions (see',
        "``tors/aio.py`` for which functions and why only these). Signatures",
        "mirror ``__init__.pyi`` exactly, ``async def`` in place of ``def`` and",
        "the bare resolved type in place of the return annotation (an",
        "``async def`` function's annotation is what awaiting it resolves to,",
        "not ``Awaitable[T]``, which describes the coroutine object itself,",
        "not its result). This stub is generated by ``tools/gen_aio_stub.py``",
        "and checked against the sync stub by ``tests/test_aio.py``.",
        '"""',
        "",
        "from collections.abc import "
        + ", ".join(
            sorted(
                {
                    "Sequence",
                    *(["Hashable"] if needs_hashable else []),
                    *{
                        name
                        for name in ("Callable", "Iterator", "Sequence")
                        if re.search(rf"\b{name}\b", "\n".join(body))
                    },
                }
            )
        ),
        f"from typing import {('Any, ' if needs_any else '')}Literal",
        "",
        "from tors import CompiledLemmaDict, StemmerLanguage",
    ]
    if used_type_names:
        # Sorted after the two always-imported names: the import is
        # emitted only when a translated signature spells one, so the
        # stub carries no unused type import (F401's gate reads it).
        header.append(
            f"from tors import {', '.join(sorted(used_type_names))}"
        )
    header.append("")
    out: list[str] = header
    out.extend(body)
    raw = "\n".join(out).rstrip("\n") + "\n"
    return _ruff_format(raw), translated


def main() -> None:
    from tors.aio import _WRAPPED

    source = SYNC_STUB.read_text(encoding="utf-8")
    text, count = _translate(source, frozenset(_WRAPPED))
    ASYNC_STUB.write_text(text, encoding="utf-8")
    print(f"written {ASYNC_STUB}: {count} async signatures")


if __name__ == "__main__":
    main()
