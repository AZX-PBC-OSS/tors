"""Honest measurement: tors's native JSON repair against the pure-Python
json_repair library it ports (upstream mangiucugna/json_repair, MIT, pinned
json-repair==0.63.4), on the SAME input strings with the same call shape:
tors.repair_json and tors.repair_json_loads vs json_repair.repair_json
(str-out, default args; the schema cells pass the same fixed schema through).

Four corpus shapes, each a five-document corpus built deterministically at
three document sizes (1 KiB, 64 KiB, 1 MiB):

- valid: canonical escaped-string-heavy documents (prose values with embedded
  quotes, backslashes, newlines and tabs, nested objects and arrays) --- the
  strict fast path both sides try first.
- malformed: the SAME documents with a fixed defect rotation across the five
  docs (dropped closing braces, unquoted keys, single-quoted strings, dropped
  commas, 7/8 truncation) --- the repair heuristics.
- fenced: the valid documents each wrapped in one ```json fence --- the
  CommonMark fence pre-pass plus fast path on the tors side, the repair
  parser's garbage skip on the library side.
- schema: malformed-ish payloads (single-quoted strings, one dropped comma)
  repaired against a fixed JSON Schema exercising coercion ("1" -> integer,
  "yes" -> boolean, "2.25" -> number), a missing optional property filled
  from its default, and a nested array of objects --- the alignment layer
  (the library side runs it on jsonschema, which must be installed).

Every cell is parity-checked before it is timed: tors vs library output on
each document, str and loads lanes. A cell whose outputs differ is reported
as divergent with its first differing input and is NOT timed --- the
tors-native schema behaviors (key-remap ladder, comma-split, date
normalization, enum suggestion) are the intentional divergences that would
surface that way. --check runs only that verification pass.

Timing is time.perf_counter min-of-N over full-corpus passes, N sized so a
cell accumulates at least 50 ms (at least 2 passes, GC disabled while
measuring), reported as per-call microseconds (one call = one document).
Run with `uv run --no-sync python tools/bench_json_repair.py`.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    import json_repair
except ImportError:
    json_repair = None  # reported loudly in main(), not at import time

import tors

KIB = 1024
SIZES = ((1 * KIB, "1 KiB"), (64 * KIB, "64 KiB"), (1024 * KIB, "1 MiB"))
SHAPES = ("valid", "malformed", "fenced", "schema")
DOCS_PER_CELL = 5
DEFECTS = (
    "dropped closing braces",
    "unquoted keys",
    "single-quoted strings",
    "dropped commas",
    "7/8 truncation",
)
MIN_CELL_SECONDS = 0.050
MAX_PASSES = 4000

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "count": {"type": "integer"},
        "active": {"type": "boolean"},
        "note": {"type": "string", "default": "n/a"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "ok": {"type": "boolean"},
                    "score": {"type": "number"},
                },
                "required": ["id", "name"],
            },
        },
    },
    "required": ["title", "items"],
}

_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu"
).split()

# Raw prose carrying exactly the characters that make JSON escaping earn its
# keep: quotes, backslashes, newlines, tabs, and commas inside string values.
_ESCAPED = (
    'He said "let\'s go" and left.',
    "Path C:\\Users\\rich\\notes.txt was moved.",
    "Line one\nline two\ttabbed.",
    "Cost: $12,50 (discounted)",
    'Quote "x" then \\backslash\\ done.',
)


def _prose(seed: int, sentences: int) -> str:
    out = []
    for j in range(sentences):
        if (seed + j) % 2:
            out.append(_ESCAPED[(seed * 3 + j) % len(_ESCAPED)])
        else:
            words = [_WORDS[(seed * 13 + j * 5 + k * 7) % len(_WORDS)] for k in range(8)]
            out.append(words[0].capitalize() + " " + " ".join(words[1:]) + ".")
    return " ".join(out)


def _record(i: int) -> dict[str, Any]:
    return {
        "id": i,
        "title": _prose(i, 1),
        "body": _prose(i + 1, 3),
        "tags": [_WORDS[(i * 3 + k) % len(_WORDS)] for k in range(3)],
        "flags": {
            "verified": i % 2 == 0,
            "priority": i % 5,
            "score": (i % 9) / 4,
            "note": _prose(i + 2, 1),
        },
    }


def _entry(group: int, idx: int) -> dict[str, Any]:
    return {
        "key": _WORDS[(group * 7 + idx * 3) % len(_WORDS)],
        "value": idx * 3,
        "meta": {
            "seen": idx % 3 == 0,
            "ref": f"r{group}-{idx}",
            "memo": _prose(group * 31 + idx, 1),
        },
    }


def _group(g: int) -> dict[str, Any]:
    return {
        "name": f"g{g}",
        "owner": _WORDS[g % len(_WORDS)],
        "entries": [_entry(g, 0), _entry(g, 1)],
    }


_DOC_CACHE: dict[tuple[int, int], str] = {}


def _build_doc(target: int, variant: int) -> str:
    """One canonical JSON document of ~target chars, deterministic in both args.

    Assembled by hand (json.dumps per record, exact default-argument separators
    everywhere else) so the text is byte-identical to json.dumps of the whole
    object --- asserted at the end --- while the size can be tracked without
    re-dumping the document on every growth step. Coarse growth adds records
    (plus a two-entry group on every third record); the final < 900-char gap
    is closed one tags-word at a time, so the overshoot stays under ~15 chars.
    """
    key = (target, variant)
    if key in _DOC_CACHE:
        return _DOC_CACHE[key]
    prefix = '{"id": "d0", "summary": ' + json.dumps(_prose(variant, 2)) + ', "records": ['
    mid = '], "groups": ['
    suffix = "]}"
    records: list[dict[str, Any]] = []
    rec_texts: list[str] = []
    grp_texts: list[str] = []

    def current() -> int:
        rec = sum(map(len, rec_texts)) + 2 * max(0, len(rec_texts) - 1)
        grp = sum(map(len, grp_texts)) + 2 * max(0, len(grp_texts) - 1)
        return len(prefix) + rec + len(mid) + grp + len(suffix)

    while not records or current() < target - 900:
        records.append(_record(len(records)))
        rec_texts.append(json.dumps(records[-1]))
        if len(records) % 3 == 1:
            grp_texts.append(json.dumps(_group(len(grp_texts))))
    while current() < target:
        last = records[-1]
        last["tags"].append(_WORDS[(len(records) * 11 + len(last["tags"]) * 5) % len(_WORDS)])
        rec_texts[-1] = json.dumps(last)
    doc = prefix + ", ".join(rec_texts) + mid + ", ".join(grp_texts) + suffix
    assert doc == json.dumps(json.loads(doc))  # canonical-by-construction self-check
    _DOC_CACHE[key] = doc
    return doc


def _dump_sq(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\t", "\\t")
    return "'" + escaped + "'"


def _dump(obj: Any, *, quote: str = '"', unquoted_keys: bool = False, comma: str = ", ") -> str:
    """Serialize with deliberate JSON defects: quote character, bare keys, missing commas."""
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if obj is None:
        return "null"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, str):
        return _dump_sq(obj) if quote == "'" else json.dumps(obj)
    if isinstance(obj, list):
        sep = comma.join(
            _dump(x, quote=quote, unquoted_keys=unquoted_keys, comma=comma) for x in obj
        )
        return "[" + sep + "]"
    items = []
    for key, value in obj.items():
        if unquoted_keys:
            assert key.isidentifier()
        key_text = key if unquoted_keys else (_dump_sq(key) if quote == "'" else json.dumps(key))
        value_text = _dump(value, quote=quote, unquoted_keys=unquoted_keys, comma=comma)
        items.append(key_text + ": " + value_text)
    return "{" + comma.join(items) + "}"


def _defect_doc(doc: str, defect: int) -> str:
    """Apply rotation defect ``defect`` to the canonical ``doc`` text."""
    if defect == 0:
        assert doc.endswith("]}")
        return doc[:-2]  # drop the last array's and the object's closing brackets
    if defect == 4:
        return doc[: len(doc) * 7 // 8]
    obj = json.loads(doc)
    if defect == 1:
        return _dump(obj, unquoted_keys=True)
    if defect == 2:
        return _dump(obj, quote="'")
    return _dump(obj, comma=" ")  # defect == 3


def _schema_item(j: int) -> dict[str, Any]:
    return {
        "id": str(j + 1),  # string "1" -> integer coercion
        "name": _prose(500 + j, 1),
        "ok": "yes" if j % 2 else "no",  # -> boolean coercion
        "score": str((j % 9) / 4),  # -> number coercion
    }


def _schema_payload(target: int, variant: int) -> str:
    """A malformed-ish single-quoted payload for SCHEMA, sized to ~target chars.

    One dropped comma (after 'count') plus single quotes everywhere keeps the
    repair parser in play before the alignment layer runs; the omitted 'note'
    is the missing-optional-with-default case, and 'items' is the nested array
    of coerced objects.
    """
    head = (
        "{'title': "
        + _dump_sq(_prose(100 + variant, 2))
        + ", 'count': '"
        + str(variant + 1)
        + "' 'active': 'yes', 'items': ["  # the comma after 'count' is dropped
    )
    items: list[dict[str, Any]] = []
    item_texts: list[str] = []

    def current() -> int:
        body = sum(map(len, item_texts)) + 2 * max(0, len(item_texts) - 1)
        return len(head) + body + len("]}")

    while not items or current() < target - 300:
        items.append(_schema_item(len(items)))
        item_texts.append(_dump(items[-1], quote="'"))
    while current() < target:
        items[-1]["name"] += " " + _WORDS[(len(items) * 17 + len(items[-1]["name"])) % len(_WORDS)]
        item_texts[-1] = _dump(items[-1], quote="'")
    return head + ", ".join(item_texts) + "]}"


def build_cell(shape: str, target: int) -> list[str]:
    docs = [_build_doc(target, variant) for variant in range(DOCS_PER_CELL)]
    if shape == "valid":
        return docs
    if shape == "malformed":
        return [_defect_doc(doc, i % len(DEFECTS)) for i, doc in enumerate(docs)]
    if shape == "fenced":
        return ["```json\n" + doc + "\n```" for doc in docs]
    return [_schema_payload(target, variant) for variant in range(DOCS_PER_CELL)]


def _cell_schema(shape: str) -> dict[str, Any] | None:
    return SCHEMA if shape == "schema" else None


def _lanes(schema: dict[str, Any] | None) -> list[Callable[..., Any]]:
    """Timed callables, fixed order: tors str, tors loads, json_repair str."""
    return [
        partial(tors.repair_json, schema=schema),
        partial(tors.repair_json_loads, schema=schema),
        partial(json_repair.repair_json, schema=schema),
    ]


def _excerpt(text: str, limit: int = 280) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text):,} chars total]"


def _diff_report(
    shape: str, label: str, lane: str, index: int, doc: str, got: Any, want: Any
) -> str:
    got_s = got if isinstance(got, str) else repr(got)
    want_s = want if isinstance(want, str) else repr(want)
    k = min(len(got_s), len(want_s))
    cut = next((j for j in range(k) if got_s[j] != want_s[j]), k)
    return (
        f"{shape} / {label}: DIVERGENT ({lane} lane, doc {index + 1} of {DOCS_PER_CELL},"
        f" outputs differ at char {cut})\n"
        f"    input:       {_excerpt(doc)}\n"
        f"    tors:        {_excerpt(got_s)}\n"
        f"    json_repair: {_excerpt(want_s)}"
    )


def verify_cell(shape: str, label: str, docs: list[str]) -> str | None:
    """tors vs library output equality per doc (str and loads lanes); None = equal."""
    schema = _cell_schema(shape)
    pairs = (
        (
            "str",
            partial(tors.repair_json, schema=schema),
            partial(json_repair.repair_json, schema=schema),
        ),
        (
            "loads",
            partial(tors.repair_json_loads, schema=schema),
            partial(json_repair.loads, schema=schema),
        ),
    )
    for lane, tors_fn, lib_fn in pairs:
        for i, doc in enumerate(docs):
            got = tors_fn(doc)
            want = lib_fn(doc)
            if got != want:
                return _diff_report(shape, label, lane, i, doc, got, want)
    return None


def time_cell(docs: list[str], fn: Callable[..., Any]) -> float:
    """Per-call microseconds: min-of-N full-corpus passes, N sized for >= 50 ms."""
    fn(docs[0])  # warmup: extension init, jsonschema validator cache
    gc_on = gc.isenabled()
    gc.disable()
    try:
        best = math.inf
        total = 0.0
        passes = 0
        while True:
            start = time.perf_counter()
            for doc in docs:
                fn(doc)
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
            total += elapsed
            passes += 1
            if (total >= MIN_CELL_SECONDS and passes >= 2) or passes >= MAX_PASSES:
                break
    finally:
        if gc_on:
            gc.enable()
    return best / len(docs) * 1e6


def _fmt_us(value: float) -> str:
    if value >= 10_000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fmt_ratio(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


_TABLE_HEADER = (
    f"{'shape':<9} {'size':>7} {'doc chars':>9} {'tors str':>10} "
    f"{'tors loads':>11} {'json_repair':>11} {'speedup':>8}  check"
)


def _row(shape: str, label: str, chars: int, times: list[float] | None) -> str:
    if times is None:
        return (
            f"{shape:<9} {label:>7} {chars:>9,} {'-':>10} {'-':>11} {'-':>11} {'-':>8}  divergent"
        )
    tors_str, tors_loads, lib = times
    ratio = _fmt_ratio(lib / tors_str)
    return (
        f"{shape:<9} {label:>7} {chars:>9,} {_fmt_us(tors_str):>10} "
        f"{_fmt_us(tors_loads):>11} {_fmt_us(lib):>11} {ratio:>8}  ok"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head benchmark: tors JSON repair vs the json_repair library."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify tors-vs-library output parity per cell (no timing)",
    )
    args = parser.parse_args()
    if json_repair is None:
        sys.exit("json-repair (pinned 0.63.4) is required; expected in the uv venv")
    shapes = list(SHAPES)
    if importlib.util.find_spec("jsonschema") is None:
        print("note: jsonschema not installed; skipping schema cells (the library side needs it)")
        shapes.remove("schema")
    try:
        lib_version = version("json-repair")
    except PackageNotFoundError:
        lib_version = "unknown"
    context = (
        f"tors vs json_repair {lib_version} | python {sys.version.split()[0]} | "
        f"{DOCS_PER_CELL} deterministic docs per cell | per-call us, min-of-N passes"
    )
    print(context)
    if args.check:
        print("parity check: tors vs json_repair outputs on every corpus cell (str + loads)")
        divergent = 0
        for shape in shapes:
            for target, label in SIZES:
                report = verify_cell(shape, label, build_cell(shape, target))
                if report is None:
                    print(f"{shape} / {label}: ok ({DOCS_PER_CELL} docs, str + loads)")
                else:
                    divergent += 1
                    print(report)
        print(f"summary: {len(shapes) * len(SIZES)} cells, {divergent} divergent")
        return
    print(f"malformed defect rotation: {', '.join(DEFECTS)}")
    print(_TABLE_HEADER)
    print("-" * len(_TABLE_HEADER))
    for shape in shapes:
        for target, label in SIZES:
            docs = build_cell(shape, target)
            report = verify_cell(shape, label, docs)
            chars = sum(len(doc) for doc in docs) // len(docs)
            if report is not None:
                print(_row(shape, label, chars, None))
                print(report)
                continue
            times = [time_cell(docs, fn) for fn in _lanes(_cell_schema(shape))]
            print(_row(shape, label, chars, times))
    print(
        "speedup = json_repair str us/call / tors str us/call; check = outputs equal on every doc"
    )


if __name__ == "__main__":
    main()
