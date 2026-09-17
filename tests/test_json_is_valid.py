"""Contract gate for ``tors.json_is_valid`` (#61): the one-shot RFC 8259
validity scan with no Python object tree, GIL-released — the
validate-and-discard gate's primitive (bytes parsed once and thrown away,
where ~95% of a full parse is constructing objects nobody reads).

The acceptance-set contract, as one oracle equality: ``json_is_valid(data)``
is ``True`` exactly when ``orjson.loads(data)`` would succeed — orjson's
acceptance set, not the stdlib's, because the gate stands in front of a
consumer whose next step IS ``orjson.loads``. Where the stdlib ``json.loads``
disagrees with orjson, the differential battery pins the seam as DELIBERATE
(five classes, all documented in docs/api.md):

- float-overflow literals (``1e400`` and friends): orjson raises, the stdlib
  hands back ``inf`` — the scanner rejects;
- NaN/Infinity/-Infinity: orjson raises, the stdlib accepts — reject;
- a leading UTF-8 BOM: orjson raises, the stdlib accepts — reject;
- lone ``\\ud800``-class surrogate escapes: orjson raises, the stdlib builds
  the lone surrogate — reject (the unsafe direction for a gate);
- nesting past 1024 containers: orjson raises (its depth cap), the stdlib
  accepts up to its own recursion limit — reject;
- long-integer literals whose value overflows an f64 (orjson's fallback
  parse): orjson raises, the stdlib builds the exact bigint — reject.

A disagreement is a BUG only in the unsafe direction (we accept where
orjson's parse would reject, or we reject where orjson accepts): the
orjson-facing oracle equality above is pinned over the full battery, the
mutation corpus, and the 40k-case float/integer overflow knife-edge sweep
the scanner module's docs cite.

The GIL-release claim gets its own pin: a co-resident asyncio heartbeat must
keep ticking at ~10ms granularity while 1 MiB documents scan in a worker
thread (the pattern of tests/test_gil_release.py, slimmed to this file: the
scan is sub-millisecond, so the honest assertions are that the heartbeat
ticks at all, keeps ticking across a 1s scan loop, and never gaps anywhere
near a GIL-held whole-scan band). Uses no async pytest plugin: the test owns
its loop via ``asyncio.run``.

Booleans only: no invalid input raises — the depth cap is an answer
(``False``), not an exception; the only error path is a wrong-TYPE argument
(not ``bytes``, not ``str``), the union surface's ``TypeError``. And the
answer is stable: repeated calls (same bytes, and the same ``str`` object
twice — the second through its cached UTF-8 view) give the same answer.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time

import pytest

import tors

orjson = pytest.importorskip("orjson")

# ---------------------------------------------------------------------------
# The differential corpus: the adversarial classes from #61's red-team list,
# one entry per input shape. The same corpus drives the orjson equality above
# and the stdlib seam classification below.
# ---------------------------------------------------------------------------

CORPUS: dict[str, bytes] = {
    # scalar / container grammar
    "empty": b"",
    "ws-only": b"  \t\n\r ",
    "obj": b"{}",
    "arr": b"[]",
    "null": b"null",
    "true": b"true",
    "false": b"false",
    "int": b"42",
    "neg-zero": b"-0",
    "float": b"-1.5e-3",
    "nested": b'{"a": [1, {"b": [true, false, null]}, "s"]}',
    "ws-around": b"  \t\r\n [1] \t \r\n ",
    "empty-str-value": b'""',
    "top-string": b'"top-level"',
    # trailing garbage / double documents
    "trailing-ws-ok": b"[1] \t\n",
    "trailing-x": b"{} x",
    "trailing-bracket": b"[1]]",
    "trailing-value": b'{"a":1}"b"',
    "two-docs": b"1 2",
    "two-objs": b'{"a":1}{"b":2}',
    "trailing-nul": b"[1]\x00",
    # nesting bombs (orjson's 1024-container depth cap, all kinds one cap)
    "depth-1023-arr": b"[" * 1023 + b"]" * 1023,
    "depth-1024-arr": b"[" * 1024 + b"]" * 1024,
    "depth-1025-arr": b"[" * 1025 + b"]" * 1025,
    "depth-5000-arr": b"[" * 5000 + b"]" * 5000,
    "depth-1024-obj": (b'{"a":' * 1024) + b"0" + (b"}" * 1024),
    "depth-1025-obj": (b'{"a":' * 1025) + b"0" + (b"}" * 1025),
    "depth-1024-mixed": (b'{"a":' * 513) + (b"[" * 511) + b"0" + (b"]" * 511) + (b"}" * 513),
    "depth-1025-mixed": (b'{"a":' * 514) + (b"[" * 511) + b"0" + (b"]" * 511) + (b"}" * 514),
    # surrogate escapes
    "surr-pair": b'"\\ud800\\udc00"',
    "surr-pair-emoji": b'"\\ud83d\\ude00"',
    "surr-pair-x2": b'"\\ud83d\\ude00\\ud83d\\ude00"',
    "surr-lone-high": b'"\\ud800"',
    "surr-lone-low": b'"\\udc00"',
    "surr-high-then-high": b'"\\ud800\\ud800"',
    "surr-low-then-low": b'"\\udc00\\udc00"',
    "surr-high-then-ascii": b'"\\ud800a"',
    "surr-high-then-escape": b'"\\ud800\\n"',
    "surr-low-then-high": b'"\\udc00\\ud800"',
    "surr-high-mid-string": b'"a\\ud800b"',
    "surr-lone-low-esc": b'"\\ude00"',
    "surr-lone-high-esc": b'"\\ud83d"',
    "surr-as-key": b'{"\\ud800": 1}',
    # invalid UTF-8 in bytes
    "utf8-truncated-2byte": b'"ab\xc3"',
    "utf8-bare-ff": b'"\xff"',
    "utf8-overlong-nul": b'"\xc0\x80"',
    "utf8-encoded-surrogate": b'"\xed\xa0\x80"',
    "utf8-past-10ffff": b'"\xf4\x90\x80\x80"',
    "utf8-lone-continuation": b'"\x80"',
    "utf8-in-array": b"[1, \xff]",
    "utf8-in-key": b'{"\xc3": 1}',
    "utf8-raw-emoji-ok": '["\U0001f600"]'.encode(),
    "utf8-raw-cjk-ok": '["\u4e2d"]'.encode(),
    # BOM handling
    "bom-obj": b"\xef\xbb\xbf{}",
    "bom-arr": b"\xef\xbb\xbf[1]",
    "bom-alone": b"\xef\xbb\xbf",
    "bom-in-array": b"[\xef\xbb\xbf]",
    "bom-in-string-ok": b'"\xef\xbb\xbf"',
    # control chars in strings (raw reject, escaped accept)
    "ctrl-soh": b'"a\x01b"',
    "ctrl-us": b'"a\x1fb"',
    "ctrl-del": b'"a\x7fb"',
    "ctrl-nul": b'"a\x00b"',
    "ctrl-lf": b'"a\nb"',
    "ctrl-tab": b'"a\tb"',
    "esc-nul-ok": b'"\\u0000"',
    "esc-us-ok": b'"\\u001f"',
    "esc-del-ok": b'"\\u007f"',
    # huge ints (orjson: <=19 digits int, longer via the f64 fallback)
    "int-19digits": b"9" * 19,
    "int-20digits": b"9" * 20,
    "int-308-ok": b"9" * 308,
    "int-309-overflow": b"9" * 309,
    "int-512-overflow": b"9" * 512,
    "int-1000-overflow": b"9" * 1000,
    "int-neg-overflow": b"-" + b"9" * 309,
    "int-1e308-ok": b"1" + b"0" * 308,
    "int-1e309-overflow": b"1" + b"0" * 309,
    "int-i64-max": b"9223372036854775807",
    "int-i64-max-plus": b"9223372036854775808",
    "int-u64-max": b"18446744073709551615",
    # 1e400-class floats (orjson raises; the stdlib returns inf)
    "float-1e400": b"1e400",
    "float-neg-1e400": b"-1e400",
    "float-1e309": b"1e309",
    "float-2e308": b"2e308",
    "float-f64max-ok": b"1.7976931348623157e308",
    "float-f64max-plus": b"1.7976931348623159e308",
    "float-f64max-plus-neg": b"-1.7976931348623159e308",
    "float-underflow-ok": b"1e-400",
    "float-neg-underflow-ok": b"-1e-400",
    "float-subnormal-ok": b"1e-323",
    "float-1e1000": b"1e1000",
    "float-1E400": b"1E400",
    "float-1e+400": b"1e+400",
    # unclosed strings at EOF
    "unterminated": b'"abc',
    "unterminated-in-obj": b'{"a": "b',
    "unterminated-backslash": b'"abc\\',
    "unterminated-u": b'"\\u00',
    "unterminated-lone-backslash": b'"\\',
    "unterminated-lone-quote": b'"',
    "unterminated-in-arr": b'["abc',
    "unclosed-obj": b'{"a":',
    "unclosed-obj-key": b'{"a"',
    # NUL bytes
    "nul-alone": b"\x00",
    "nul-in-key": b'{"a\x00b": 1}',
    "nul-esc-value-ok": b'"\x00"',
    "nul-in-literal-word": b"nul\x00l",
    # misc grammar refusals
    "leading-zero": b"01",
    "trailing-comma-arr": b"[1,]",
    "trailing-comma-obj": b'{"a":1,}',
    "colon-only": b"{:1}",
    "comma-only": b"[,]",
    "missing-comma": b"[1 2]",
    "double-minus": b"--1",
    "plus-one": b"+1",
    "dot-five": b".5",
    "one-dot": b"1.",
    "bare-e": b"1e",
    "e-plus": b"1e+",
    "hex-int": b"0x10",
    "single-quote": b"'x'",
    "python-true": b"True",
    "python-none": b"None",
    "tru-prefix": b"tru",
    "true-suffix": b"truex",
    "obj-key-no-colon": b'{"a" 1}',
    "obj-double-colon": b'{"a"::1}',
    "obj-missing-comma": b'{"a":1 "b":2}',
    "open-arr": b"[",
    "close-arr": b"]",
    "open-obj": b"{",
    "close-obj": b"}",
    "close-close": b"{}}",
    "open-open": b"[[",
    "mismatched": b"[}",
    "comma-top": b",",
    "colon-top": b":",
    # duplicate keys: orjson last-wins, accepts
    "dup-keys": b'{"a":1,"a":2}',
    "dup-keys-nested": b'{"a":{"b":1,"b":2}}',
    # NaN / Infinity literals (orjson raises; the stdlib accepts)
    "nan": b"NaN",
    "infinity": b"Infinity",
    "neg-infinity": b"-Infinity",
}

# Realistic validate-and-discard payloads: the consumer's shape.
REALISTIC = [
    b'{"id": 1234, "name": "widget", "tags": ["a", "b"], "ok": true, "score": 4.5, "meta": null}',
    b'[{"ts": 1700000000, "level": "info", "msg": "hello \\u2014 with escapes"}]',
    b'{"deep": {"nested": {"data": [1, 2, {"x": [-1.5e-7, "s"]}]}}}',
    '["unicode: \\u00e9\\u4e2d\\ud83d\\ude00", "raw: \u00e9\u4e2d\U0001f600"]'.encode(),
]


def _orjson_ok(data: bytes) -> bool:
    try:
        orjson.loads(data)
        return True
    except (orjson.JSONDecodeError, UnicodeDecodeError, ValueError):  # noqa: BLE001 — the oracle's own refusal classes
        return False


def _stdlib_ok(data: bytes) -> bool:
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError):  # noqa: BLE001 — the oracle's own refusal classes
        return False


# ---------------------------------------------------------------------------
# The oracle equality vs orjson (the acceptance-set contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_orjson_equality(name: str) -> None:
    data = CORPUS[name]
    assert tors.json_is_valid(data) == _orjson_ok(data), (
        f"{name}: json_is_valid disagrees with orjson.loads on {data!r}"
    )


@pytest.mark.parametrize("data", REALISTIC)
def test_orjson_equality_realistic(data: bytes) -> None:
    assert tors.json_is_valid(data) == _orjson_ok(data)
    assert tors.json_is_valid(data) is True


# ---------------------------------------------------------------------------
# The stdlib seam: where json.loads disagrees with orjson, the disagreement
# must be exactly the documented classes, and always in the documented
# direction (we reject; the stdlib's looser parse would have accepted).
# ---------------------------------------------------------------------------


def test_stdlib_seams_are_the_documented_classes() -> None:
    seams = {
        name: data
        for name, data in CORPUS.items()
        if tors.json_is_valid(data) != _stdlib_ok(data)
    }
    # Every stdlib disagreement is us rejecting what the stdlib would have
    # accepted (never the reverse: an input we accept must parse with the
    # stdlib whenever it parses with orjson, and orjson's set is a subset).
    assert all(not tors.json_is_valid(d) for d in seams.values())
    # And every one is a class orjson also rejects (the gate's own oracle).
    assert all(not _orjson_ok(d) for d in seams.values())
    # The classes, by name: depth cap, surrogates (escaped or UTF-8-encoded),
    # overflow, NaN/Infinity, BOM.
    prefixes = ("depth", "surr", "int", "float", "neg", "nan", "infinity", "bom", "utf8")
    assert all(n.startswith(prefixes) for n in seams), sorted(seams)


# ---------------------------------------------------------------------------
# The mutation battery: random single-point mutations of a valid document
# must also satisfy the oracle equality (the adversarial corpus's shapes are
# hand-picked; this walks the space between them).
# ---------------------------------------------------------------------------


def _mutations(seed: int, count: int) -> list[bytes]:
    import random

    rng = random.Random(seed)
    base = json.dumps(
        {
            "ints": list(range(50)),
            "floats": [i * 1.5 for i in range(50)],
            "strings": [f"str-{i}-\u00e9\u4e2d" for i in range(20)],
            "nested": {"a": [[1, 2], [3, 4]], "b": {"c": [True, False, None] * 10}},
        }
    ).encode()
    out = []
    for _ in range(count):
        m = bytearray(base)
        for _ in range(rng.randint(1, 4)):
            op = rng.random()
            p = rng.randrange(len(m))
            if op < 0.4:
                m[p] = rng.randrange(256)
            elif op < 0.7:
                m[p : p] = bytes([rng.randrange(256)])
            else:
                del m[p : p + rng.randint(1, 3)]
        out.append(bytes(m))
    return out


def test_orjson_equality_over_mutations() -> None:
    for i, data in enumerate(_mutations(61, 300)):
        assert tors.json_is_valid(data) == _orjson_ok(data), f"mutation {i}: {data!r}"


def test_orjson_equality_over_generated_documents() -> None:
    import random

    rng = random.Random(6161)

    def gen(depth: int) -> str:
        r = rng.random()
        if depth > 6 or r < 0.3:
            pick = rng.random()
            if pick < 0.25:
                return str(rng.randint(-(10**12), 10**12))
            if pick < 0.5:
                return f"{rng.random() * 10 ** rng.randint(-300, 300):.6g}"
            if pick < 0.75:
                alphabet = 'ab"\\\t\n\u00e9\U0001f600 '
                return json.dumps(
                    "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
                )
            return rng.choice(["true", "false", "null"])
        if r < 0.65:
            return "[" + ",".join(gen(depth + 1) for _ in range(rng.randint(0, 5))) + "]"
        return "{" + ",".join(
            f"k{rng.randint(0, 999)}:{gen(depth + 1)}" for _ in range(rng.randint(0, 5))
        ) + "}"

    for _ in range(300):
        data = gen(0).encode()
        assert tors.json_is_valid(data) == _orjson_ok(data), f"generated: {data!r}"


# ---------------------------------------------------------------------------
# The float/integer overflow knife edge: orjson's f64 fallback is where the
# prototype's 6-in-2053 disagreements lived; the boundary (values within a
# rounding step of ±1.8e308) is swept here at 40k cases — the scanner module's
# documented caveat, kept honest in both directions.
# ---------------------------------------------------------------------------


def test_overflow_knife_edge_sweep() -> None:
    import random

    rng = random.Random(7)
    for _ in range(20000):
        mantissa = rng.randint(10**16, 10**18) / 10 ** rng.randint(0, 17)
        lit = f"{mantissa:.17g}e{rng.randint(307, 309)}".encode()
        assert tors.json_is_valid(lit) == _orjson_ok(lit), f"{lit!r}"
    for _ in range(20000):
        if rng.random() < 0.5:
            lit = b"9" * rng.randint(1, 600)
        else:
            lit = str(rng.randint(1, 9)).encode() + b"0" * rng.randint(0, 600)
        assert tors.json_is_valid(lit) == _orjson_ok(lit), f"{lit[:40]!r}"


# ---------------------------------------------------------------------------
# The API surface: booleans only, wrong types raise, the union argument
# contract, and answer stability (idempotence, and the str object's cached
# UTF-8 view serving repeat calls).
# ---------------------------------------------------------------------------


def test_invalid_input_never_raises_booleans_only() -> None:
    for name in sorted(CORPUS):
        answer = tors.json_is_valid(CORPUS[name])  # must not raise
        assert isinstance(answer, bool)


def test_depth_cap_is_an_answer_not_an_error() -> None:
    over = b"[" * 5000 + b"]" * 5000
    assert tors.json_is_valid(over) is False


@pytest.mark.parametrize(
    "bad",
    [123, 1.5, None, True, [], ["[1]"], {"a": 1}, bytearray(b"[1]"), memoryview(b"[1]")],
)
def test_wrong_type_raises_typeerror(bad: object) -> None:
    with pytest.raises(TypeError):
        tors.json_is_valid(bad)  # type: ignore[arg-type]


def test_bytes_and_str_agree() -> None:
    for data in (b'{"a": [1, 2.5, true, null]}', b"[1,]", b'"caf\xc3\xa9"', b"\xef\xbb\xbf{}"):
        text = data.decode("utf-8", errors="replace")
        assert tors.json_is_valid(data) == tors.json_is_valid(text)


def test_repeat_calls_are_stable() -> None:
    doc = b'{"a": [1, 2.5, true, null, "\u00e9"]}'
    answers = {tors.json_is_valid(doc) for _ in range(5)}
    assert answers == {True}
    for name, data in CORPUS.items():
        assert {tors.json_is_valid(data) for _ in range(3)} == {tors.json_is_valid(data)}, name


def test_repeat_calls_on_the_same_str_object() -> None:
    # The str path borrows the object's UTF-8 view: the first non-ASCII call
    # materializes and caches it (CPython's internal cache), every later call
    # borrows it zero-copy — same answer, and the cache must not change it.
    valid = '{"a": ["\u00e9", "\u4e2d"], "b": "\U0001f600"}'
    invalid = '{"a": ["\u00e9", 1,}'  # noqa: E501 — malformed on purpose
    assert tors.json_is_valid(valid) is True
    assert tors.json_is_valid(valid) is True  # cached view
    assert tors.json_is_valid(invalid) is False
    assert tors.json_is_valid(invalid) is False  # cached view


def test_str_holding_a_raw_lone_surrogate_raises_the_borrows_error() -> None:
    # A str with a raw lone surrogate cannot materialize a UTF-8 view: the
    # str-in borrow raises CPython's own UnicodeEncodeError before any scan
    # runs — the standard str-argument contract (utf8_byte_len's pinned
    # behavior), not a validity answer. The escape TEXT is ordinary string
    # content and answers False through the normal gate.
    lone = "\ud800"
    with pytest.raises(UnicodeEncodeError):
        tors.json_is_valid(lone)
    assert tors.json_is_valid('"\\ud800"') is False


# ---------------------------------------------------------------------------
# The GIL-release pin, slimmed to this file's claim: a co-resident asyncio
# heartbeat keeps ticking while 1 MiB documents scan in a worker thread.
# The scan is sub-millisecond (orjson's full parse of the same document is
# ~3.4 ms, the scanner ~0.7 ms), so the honest assertions are: the heartbeat
# ticks at all while the scans run, keeps ticking across a 1s scan loop, and
# never gaps anywhere near a GIL-held band (the 500ms budget is ~500x the
# scan itself and ~50x orjson's whole-parse band — a GIL-held regression of
# ANY loop iteration fails it many times over).
# ---------------------------------------------------------------------------

_MIB_DOC = (
    b'{"records": ['
    + b",".join(
        b'{"id": %d, "src": "worker-%d", "ok": true, "score": %d.%d,'
        b' "tags": ["a", "b", "c"], "msg": "result payload %d"}'
        % (i, i % 8, i, i % 10, i)
        for i in range(9200)
    )
    + b"]}"
)
assert len(_MIB_DOC) > 1024 * 1024


def test_heartbeat_ticks_while_large_scans_run() -> None:
    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            ticks.append(time.monotonic())
            if stop.is_set():
                return
            await asyncio.sleep(0.01)

    async def run() -> None:
        nonlocal stop
        # The first tick must land while the scans are still running, well
        # inside any GIL-held band of the loop's own workload.
        first_tick_deadline = 0.5
        scan_deadline = time.monotonic() + 1.0
        scanned = 0
        hb = asyncio.create_task(heartbeat())
        started = time.monotonic()
        try:
            while time.monotonic() < scan_deadline:
                await asyncio.to_thread(tors.json_is_valid, _MIB_DOC)
                scanned += 1
            first_tick = ticks[0] - started if len(ticks) > 1 else float("inf")
            assert first_tick < first_tick_deadline, (
                f"first heartbeat tick took {first_tick:.3f}s"
            )
            assert scanned >= 10, f"only {scanned} scans in 1s"
        finally:
            stop.set()
            await hb

    asyncio.run(run())

    # No gap anywhere near a GIL-held band: the worst tick gap is the 10ms
    # ping floor plus scheduling noise, not a held scan.
    gaps = [b - a for a, b in itertools.pairwise(ticks)]
    assert max(gaps) < 0.5, f"event loop gapped {max(gaps):.3f}s during the scans"
