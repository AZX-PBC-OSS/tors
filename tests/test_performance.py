"""Wall-time proof: tors's native passes versus the expressions they replace,
measured in the same process so both sides pay identical machine conditions.
This is load-fair by construction: a shared-runner slowdown inflates both
sides together, which is why the threshold can stay tight.

``tors.finalize`` vs the pure-Python reference finalize, measured on the dev
box this test was written on (WSL2, 28 logical cores, ambient load ~3.5, CPython
3.12.7), min-of-5 per side after one warm-up call:

    corpus      size   tors        reference   tors/ref
    prose       1 KiB  0.007ms     0.008ms     0.80  (wins, but thin: µs-scale call
                                                      overhead dominates; recorded here,
                                                      not CI-asserted)
    prose       1 MiB  8.61ms      12.82ms     0.67
    prose       12 MiB 105.8ms     163.0ms     0.65
    decomposed  1 MiB  9.06ms      16.94ms     0.53
    decomposed  12 MiB 109.8ms     246.3ms     0.45
    crlf        1 MiB  8.85ms      13.58ms     0.65
    crlf        12 MiB 107.3ms     162.8ms     0.66

The CI assertion covers 1 MiB and 12 MiB over all three corpora with a tolerant
``tors < 0.9 x reference`` margin: the measured ratios sit at 0.45-0.67, so the margin
absorbs a loaded 2-vCPU CI runner. The one observed flake class (a 1 MiB
decomposed leg at ratio 1.02 under concurrent compile load) is asymmetric
contention, not symmetric noise: a ~10ms native sample can have one preempted
run set its min-of-3, while the reference's 13-17ms samples amortize the same
preemption: "both sides inflate together" only holds when neither side's
samples are short enough for a single scheduler hit to dominate their minimum.
The fast cells therefore draw min-of-7 (``_FAST_CELL_SAMPLES``), giving the
native side enough draws to find an uncontended window; the 12 MiB cells keep
min-of-3 (a 100ms+ sample amortizes preemptions the way the reference's do).
If tors ever fails to beat the reference at an asserted size, that is a design failure
of the native pass, not something to threshold away; see the numbers above for what
healthy looks like.

The bytes-in functions use the same methodology, with corpora from
``reference.corpus_utf8`` (the same recipes as UTF-8 bytes; ambient load 5.4;
min-of-5 after warmup):

    ``tors.b64_encode_bytes`` vs ``base64.b64encode(raw).decode("ascii")``: a win at
    every cell, asserted with the same 0.9 margin:

        corpus      size   tors        stdlib     tors/stdlib
        prose       1 MiB  1.07ms      1.32ms     0.81
        decomposed  1 MiB  0.30ms      0.87ms     0.34
        crlf        1 MiB  0.30ms      0.86ms     0.34
        prose       12 MiB 14.60ms     17.46ms    0.84  (a contended outlier: the
                                                         same cell measured 4.2ms vs
                                                         10.9ms, ratio 0.38, at load
                                                         ~5.6 in an adjacent pass;
                                                         recorded as the worst observed)
        decomposed  12 MiB 4.76ms      11.28ms    0.42
        crlf        12 MiB 5.60ms      18.15ms    0.31

    ``tors.finalize_utf8`` vs the decode-plus-reference tail it replaces
    (``raw.decode`` + the pure-Python reference finalize): a win at every cell,
    asserted with the same margin:

        corpus      size   tors        decode+ref  tors/ref
        prose       1 MiB  10.0ms      13.4ms      0.75
        decomposed  1 MiB  9.6ms       20.1ms      0.48
        crlf        1 MiB  9.3ms       14.3ms      0.65
        prose       12 MiB 113.1ms     166.3ms     0.68
        decomposed  12 MiB 121.7ms     263.4ms     0.46
        crlf        12 MiB 120.2ms     170.5ms     0.70

    ``tors.decode_utf8`` vs ``raw.decode("utf-8")``: not asserted, and reported as a
    measured loss, ratios 1.07-1.60 across the six cells (e.g. prose 12 MiB 0.61ms vs
    0.45ms; decomposed 12 MiB 3.82ms vs 2.81ms). CPython's UTF-8 decode is a C
    fast-path with no Python-level overhead, and tors marshals its result through the
    same str construction, so there is no wall-time case for ``decode_utf8`` standalone.
    Its value is the GIL release (pinned in tests/test_gil_release.py), CPython-parity
    guarantees (tests/test_decode_utf8.py), and its role inside ``finalize_utf8``.
    Deliberately recorded, not thresholded away.

``tors.minhash_signature`` vs the pure-Python oracle
(``reference.reference_minhash_signature``, the transcribed MinHash: same tokens
via ``word_bounds``, same shingles, XXH64 via the pinned ``xxhash`` package, the
same SplitMix64-to-affine arithmetic in Python bigints), at the default
``num_perm=128``, prose corpus, measured on the dev box (macOS/arm64, tors
min-of-7 after warmup, oracle single sample -- its wall is seconds-scale and
deterministic work, so one sample is the conservative denominator; the
difflib-race precedent):

    size    tors        oracle      tors/oracle
    1 KiB   0.02ms      3.4ms       0.006  (~170x)
    100KiB  1.6ms       280.9ms     0.006  (~175x)
    1 MiB   17.7ms      2886.5ms    0.006  (~163x)

The margin is 0.5, not the near-parity 0.9 elsewhere in this file: the oracle is
slow Python (an O(shingles x num_perm) bigint inner loop), so the criterion the
cell pins is ``the native pass keeps its advantage`` (a regression to within 2x
of pure Python fails it), not a close race; the measured ratios leave ~80x
headroom, so load asymmetry cannot flake it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import re
import secrets
import string
import time
import uuid as stdlib_uuid
from collections.abc import Callable

import pytest

import tors
from reference import (
    content_object,
    corpus_b64,
    corpus_utf8,
    crlf,
    decomposed,
    entities,
    prose,
    reference_finalize,
    reference_minhash_signature,
    reference_scrub_log_text,
    scrub_corpus,
)
from tors import (
    chunk_by_lines,
    chunk_by_paragraphs,
    chunk_by_sentences,
    chunk_by_words,
    chunk_hierarchical,
)

# The timing lane: every test in this module is a measurement cell (wall-time
# bands and races over multi-MiB corpora), slow and load-sensitive, so CI's
# matrix legs deselect it (`-m "not timing"`) and one dedicated step on the
# 3.12 leg runs it (`-m timing`). The wall contracts still gate every push,
# once, instead of being paid on all five legs. Locally a bare ``pytest``
# (and ``make test``) run everything, marker or not.
pytestmark = pytest.mark.timing

_MIB = 1024 * 1024
_MARGIN = 0.9
_SAMPLES = 3
# The fast-cell sample count (see the module docstring's flake-class note):
# a 1 MiB cell's ~10ms native samples are short enough that one preempted
# run can set min-of-3 while the pure-Python side's longer samples amortize
# the same preemption; min-of-7 gives the native side enough draws to find
# an uncontended window under concurrent load.
_FAST_CELL_SAMPLES = 7
# Cells at or above this many bytes draw _SAMPLES; smaller cells draw
# _FAST_CELL_SAMPLES. 4 MiB: the 1 MiB cells flaked at 3, the 12 MiB cells
# never have (100ms+ samples amortize scheduler hits symmetrically).
_FAST_CELL_MAX = 4 * _MIB


def _min_wall_ms(
    op: Callable[[str | bytes], object],
    corpus: str | bytes,
    warmup: int = 1,
    samples: int = _SAMPLES,
) -> float:
    """Min-of-``samples`` wall after warmup. The corpus parameter is ``str |
    bytes`` because the suite measures both str-in functions (``finalize`` over
    the reference corpora) and bytes-in functions (the surface over
    ``corpus_utf8``). ``samples`` is the fast-cell knob: see
    ``_FAST_CELL_SAMPLES`` for why the 1 MiB native-vs-pure-Python cells draw
    more samples than the rest."""
    for _ in range(warmup):
        op(corpus)
    best = float("inf")
    for _ in range(samples):
        started = time.monotonic()
        op(corpus)
        best = min(best, time.monotonic() - started)
    return best * 1000.0


def _samples_for(size_bytes: int) -> int:
    """The sample count for a cell of this size: ``_FAST_CELL_SAMPLES`` below
    ``_FAST_CELL_MAX``, ``_SAMPLES`` at or above."""
    return _FAST_CELL_SAMPLES if size_bytes < _FAST_CELL_MAX else _SAMPLES


def _stdlib_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _stdlib_b64_decode(s: str) -> bytes:
    """The stdlib expression ``tors.b64_decode`` replaces."""
    return base64.b64decode(s)


def _stdlib_html_unescape(text: str) -> str:
    """The stdlib expression ``tors.html_unescape`` replaces."""
    return html.unescape(text)


def _stdlib_decode_finalize(raw: bytes) -> tuple[str, str]:
    """The replaced tail today: GIL-held decode plus the pure-Python reference
    finalize (normalize + sha256), what ``tors.finalize_utf8`` collapses into one
    detached call."""
    return reference_finalize(raw.decode("utf-8"))


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed", "crlf"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_finalize_beats_the_reference_pipeline_on_the_same_corpus(
    corpus_kind: str, size_bytes: int
) -> None:
    corpus = {"prose": prose, "decomposed": decomposed, "crlf": crlf}[corpus_kind](size_bytes)
    samples = _samples_for(size_bytes)
    tors_ms = _min_wall_ms(tors.finalize, corpus, samples=samples)
    ref_ms = _min_wall_ms(reference_finalize, corpus, samples=samples)
    assert tors_ms < _MARGIN * ref_ms, (
        f"{corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.1f}ms vs "
        f"reference {ref_ms:.1f}ms (ratio {tors_ms / ref_ms:.2f}): the native pass lost "
        "more than the tolerance margin to the pure-Python pipeline"
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed", "crlf"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_b64_encode_bytes_beats_the_stdlib_expression_on_the_same_corpus(
    corpus_kind: str, size_bytes: int
) -> None:
    """The content-addressing swap's wall case: the stdlib expression holds the GIL for
    ``b64encode`` and pays a second 4/3-sized str construction in ``decode("ascii")``;
    tors encodes detached and marshals one output. Measured ratios 0.31-0.84 (the
    worst a contended prose 12 MiB cell; see the module docstring)."""
    corpus = corpus_utf8(corpus_kind, size_bytes)
    samples = _samples_for(size_bytes)
    tors_ms = _min_wall_ms(tors.b64_encode_bytes, corpus, samples=samples)
    std_ms = _min_wall_ms(_stdlib_b64, corpus, samples=samples)
    assert tors_ms < _MARGIN * std_ms, (
        f"{corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.1f}ms vs "
        f"stdlib {std_ms:.1f}ms (ratio {tors_ms / std_ms:.2f}): the native b64 pass "
        "lost more than the tolerance margin to the stdlib expression"
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed", "crlf"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_finalize_utf8_beats_the_decode_plus_reference_pipeline_on_the_same_corpus(
    corpus_kind: str, size_bytes: int
) -> None:
    """The one-call shape's wall case: decode, normalize, and hash in a single detached
    pass vs the tail it replaces (a GIL-held decode plus the pure-Python
    reference finalize). Measured ratios 0.46-0.75."""
    corpus = corpus_utf8(corpus_kind, size_bytes)
    samples = _samples_for(size_bytes)
    tors_ms = _min_wall_ms(tors.finalize_utf8, corpus, samples=samples)
    ref_ms = _min_wall_ms(_stdlib_decode_finalize, corpus, samples=samples)
    assert tors_ms < _MARGIN * ref_ms, (
        f"{corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.1f}ms vs "
        f"decode+reference {ref_ms:.1f}ms (ratio {tors_ms / ref_ms:.2f}): the "
        "one-call native pass lost more than the tolerance margin to the "
        "decode-then-finalize pipeline"
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed", "crlf"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_b64_decode_wall_time_vs_the_stdlib_is_measured_not_asserted(
    corpus_kind: str, size_bytes: int
) -> None:
    """``tors.b64_decode`` vs ``base64.b64decode`` (the b64-encoded corpora from
    ``reference.corpus_b64``), measured and not asserted for the
    same reason as the ``decode_utf8`` case: CPython's ``a2b_base64`` is
    a C fast-path with no Python-level overhead, and tors marshals its result
    through the same ``PyBytes`` construction, so there is no wall-time case
    for the standalone decode. Measured on the dev box (ambient load 6.0,
    min-of-3 after warmup):

        corpus      size   tors        stdlib     tors/stdlib
        prose       1 MiB  0.99ms      0.96ms     1.03
        decomposed  1 MiB  1.18ms      1.21ms     0.97
        crlf        1 MiB  1.04ms      1.02ms     1.03
        prose       12 MiB 13.04ms     15.05ms    0.87
        decomposed  12 MiB 13.32ms     13.96ms    0.95
        crlf        12 MiB 13.19ms     13.98ms    0.94

    Even-parity at 1 MiB, a thin win at 12 MiB (0.87-0.95, under the 0.9
    assertion margin on two of three corpora, which is why nothing is
    asserted). The value is the GIL release (pinned in tests/test_gil_release.py:
    stdlib ratio 0.98-0.99 held in every sample vs tors's 0.50-0.78 band at
    12 MiB), the error-path parity contract (tests/test_b64_decode.py), and
    symmetry with ``b64_encode_bytes``. Recorded, not thresholded away."""
    corpus = corpus_b64(corpus_kind, size_bytes)
    tors_ms = _min_wall_ms(tors.b64_decode, corpus)
    std_ms = _min_wall_ms(_stdlib_b64_decode, corpus)
    print(
        f"b64_decode {corpus_kind} {size_bytes // _MIB}MiB: tors {tors_ms:.2f}ms "
        f"stdlib {std_ms:.2f}ms ratio {tors_ms / std_ms:.2f}"
    )


@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_html_unescape_beats_the_stdlib_on_entity_bearing_prose(
    size_bytes: int,
) -> None:
    """The wall headline: ``tors.html_unescape`` vs ``html.unescape`` on
    the entities corpus (~7% entity density). The stdlib's cost is structural:
    a ``re.sub`` whose every match invokes the Python callback
    ``_replace_charref``, so the native scan wins by ~3x regardless of load
    (the callback count, not the machine, dominates). Measured on the dev box
    (ambient load 10.0, min-of-3 after warmup):

        size    tors        stdlib      tors/stdlib
        1 MiB   6.3ms       18.1ms      0.346
        12 MiB  82.5ms      257.9ms     0.320

    Asserted with the same 0.9 margin as the other wall cells. The measured
    ratios (0.32-0.35, under the heaviest load window in these tables) leave
    ~2.8x of headroom."""
    corpus = entities(size_bytes)
    tors_ms = _min_wall_ms(tors.html_unescape, corpus)
    std_ms = _min_wall_ms(_stdlib_html_unescape, corpus)
    assert tors_ms < _MARGIN * std_ms, (
        f"entities {size_bytes // _MIB}MiB: tors {tors_ms:.1f}ms vs "
        f"stdlib {std_ms:.1f}ms (ratio {tors_ms / std_ms:.2f}): the native "
        "entity scan lost more than the tolerance margin to the regex+callback"
    )


def test_html_unescape_no_ampersand_path_is_measured_not_asserted() -> None:
    """The degenerate path, measured and not asserted: on text
    with no ``&`` at all, ``html.unescape`` is a single C ``in`` check
    (~0.10ms at 12 MiB) while tors pays the same class of scan through a
    bytewise ``contains`` (~0.58ms at 12 MiB, min-of-3 at load 10.0), a
    measured loss by ~5x on a path that is microseconds-vs-milliseconds in
    absolute terms either way. The zero-copy identity return (the wrapper
    returns the input object, exactly as CPython's ``return s`` does;
    measured: ``tors.html_unescape(s) is s``) removes the O(n) marshalling
    copy this path would otherwise pay (1.17ms before it), leaving only the
    scan itself. Recorded, not thresholded away: no caller routes
    megabytes of ampersand-free text through an entity decoder for its speed,
    and the wall cells that matter are the entity-bearing ones above."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(tors.html_unescape, corpus)
    std_ms = _min_wall_ms(_stdlib_html_unescape, corpus)
    print(
        f"html_unescape no-& prose 12MiB: tors {tors_ms:.2f}ms "
        f"stdlib {std_ms:.2f}ms ratio {tors_ms / std_ms:.2f}"
    )


# --- The one-shot hashing surface -------------------------------------------------
#
# The honest hashlib comparison, measured on the dev box (Apple Silicon,
# ambient load 7.8-9.7): hashlib's digest engines are OpenSSL-backed with
# hardware SHA extensions, and at throughput sizes they win or tie —
# sha256 ~1.1-1.2x (a real stdlib win, recorded below and asserted
# nowhere; absolute figures move with box and load, the band is the
# statement), sha1 1.06-1.11, md5 0.88-1.03 and sha512 ~0.98 (dead heats).
# tors's genuine wall wins are the sizes this surface exists for, where
# per-call overhead dominates the engine: hashing a SHORT ASCII STR (the
# cache-key/ETag spelling, where hashlib makes the caller encode first;
# ASCII is the zero-copy borrow lane, non-ASCII pays the one-time O(input)
# UTF-8 materialization, so the win narrows there — the asserted cell pins
# the ASCII band and a recorded micro-cell covers "é"*512 alongside it)
# at 0.40-0.54 of the stdlib expression, and HMAC at request-signature
# sizes at ~0.31-0.36 of even the stdlib's fastest one-shot spelling
# (``hmac.digest(...).hex()``, measured against explicitly so the
# asserted cell does not race a slow opponent).
# Micro-scale cells draw more samples: a sub-µs sample is one scheduler
# hit away from its minimum, so min-of-15 gives the short side enough
# draws to find an uncontended window (the _FAST_CELL_SAMPLES
# derivation, sized for this surface's µs-scale cells).
_HASH_MICRO_SAMPLES = 15


@pytest.mark.parametrize(
    "tors_fn_name", ["md5_hex", "sha1_hex", "sha256_hex", "sha512_hex"]
)
@pytest.mark.parametrize("size_bytes", [1024, 12 * _MIB], ids=["1KiB", "12MiB"])
def test_digest_wall_time_vs_hashlib_is_measured_not_asserted(
    tors_fn_name: str, size_bytes: int
) -> None:
    """The digest engines head-to-head at bytes-throughput sizes, measured
    and not asserted, the decode_utf8/b64_decode precedent: hashlib's
    OpenSSL engines (hardware SHA extensions) win or tie at every size
    where the engine dominates the call. Measured (min-of-3 after warmup,
    prose corpus bytes; absolute figures move with box and ambient load,
    the band — sha256 ~1.1-1.2x, sha1 ~1.06-1.11, md5/sha512 dead heats —
    is the load-stable statement, not any single pair):

        algorithm   size    tors        hashlib     tors/hashlib
        md5         1 KiB   ~0.001ms    ~0.001ms    0.88
        sha1        1 KiB   ~0.001ms    ~0.001ms    0.69
        sha256      1 KiB   ~0.001ms    ~0.001ms    0.83
        sha512      1 KiB   ~0.001ms    ~0.001ms    0.74
        md5         12 MiB  ~15.0ms     ~14.5ms     ~1.03
        sha1        12 MiB  ~4.4ms      ~4.2ms      ~1.06
        sha256      12 MiB  ~4-5ms      ~4ms        ~1.1-1.2
        sha512      12 MiB  ~7.2ms      ~7.4ms      ~0.98

    The sha256/sha1 losses are real and recorded, not thresholded away:
    the surface's value at these sizes is the parity digest (pinned
    differentially in tests/test_hash.py), the str convenience, and the
    GIL story told in tests/test_gil_release.py (hashlib releases the GIL
    for 2048+-byte updates, so the honest claim there is uniformity, not a
    latency win). The wall wins this surface can assert are the
    short-str and hmac cells below, the request-signing sizes where the
    per-call overhead is the cost."""
    raw = corpus_utf8("prose", size_bytes)
    tors_fn = getattr(tors, tors_fn_name)
    stdlib_fn = getattr(hashlib, tors_fn_name.removesuffix("_hex"))
    tors_ms = _min_wall_ms(tors_fn, raw)
    std_ms = _min_wall_ms(lambda r: stdlib_fn(r).hexdigest(), raw)
    print(
        f"{tors_fn_name} {size_bytes // 1024}KiB: tors {tors_ms:.3f}ms "
        f"hashlib {std_ms:.3f}ms ratio {tors_ms / std_ms:.2f}"
    )


@pytest.mark.parametrize("size_bytes", [128, 512], ids=["128B", "512B"])
def test_sha256_hex_beats_encode_plus_hashlib_on_short_strings(size_bytes: int) -> None:
    """The short-str wall win, asserted on ASCII str: hashing a str directly vs the
    expression a hashlib caller must write
    (``hashlib.sha256(s.encode("utf-8")).hexdigest()``), the cache-key /
    ETag / request-ID spelling. tors pays one pyo3 call and the borrowed
    UTF-8 (zero-copy on ASCII/cached inputs); the stdlib expression pays
    ``str.encode`` (a fresh bytes object), the hash-object constructor,
    and the ``hexdigest`` call.
    Measured 0.17µs vs 0.33µs at 128B and 0.28µs vs 0.52µs at 512B
    (ratios 0.40-0.54, min-of-many at load ~10); asserted with the 0.9
    margin, ~1.7-2.2x of headroom. ASCII-scoped on purpose: non-ASCII str
    pays the one-time O(input) UTF-8 materialization (see the recorded
    micro-cell below), so the win narrows there."""
    text = prose(4096)[:size_bytes]
    tors_ms = _min_wall_ms(tors.sha256_hex, text, samples=_HASH_MICRO_SAMPLES)
    std_ms = _min_wall_ms(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest(),
        text,
        samples=_HASH_MICRO_SAMPLES,
    )
    assert tors_ms < _MARGIN * std_ms, (
        f"sha256_hex str {size_bytes}B: tors {tors_ms:.4f}ms vs "
        f"encode+hashlib {std_ms:.4f}ms (ratio {tors_ms / std_ms:.2f}): the "
        "one-call str spelling lost more than the tolerance margin to the "
        "encode-then-hash expression"
    )


def test_sha256_hex_non_ascii_short_str_is_measured_not_asserted() -> None:
    """The non-ASCII companion to the asserted ASCII cell above, recorded
    not asserted: ``"é" * 512`` (512 chars, 1024 UTF-8 bytes) pays the
    one-time O(input) UTF-8 materialization through pyo3's ``to_str``
    borrow on top of the digest, where the ASCII lane above is zero-copy
    — so the ~2x win narrows and no threshold is pinned here. Kept
    record-only by design: no regression band or ceiling is asserted on
    this shape. What IS
    pinned is value parity (the digest equals the UTF-8-bytes spelling
    on both sides); the walls are printed so the run's log carries the
    recorded shape."""
    text = "é" * 512
    assert tors.sha256_hex(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    tors_ms = _min_wall_ms(tors.sha256_hex, text, samples=_HASH_MICRO_SAMPLES)
    std_ms = _min_wall_ms(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest(),
        text,
        samples=_HASH_MICRO_SAMPLES,
    )
    print(
        f"sha256_hex non-ascii 512ch: tors {tors_ms:.4f}ms vs "
        f"encode+hashlib {std_ms:.4f}ms ratio {tors_ms / std_ms:.2f}"
    )


@pytest.mark.parametrize(
    ("key_len", "data_len"), [(32, 256), (131, 200)], ids=["request", "long-key"]
)
def test_hmac_sha256_hex_beats_the_fastest_stdlib_hmac_spelling(
    key_len: int, data_len: int
) -> None:
    """The request-signing wall win, asserted against the stdlib's
    FASTEST spelling, not the common slow one: ``hmac.digest(key, data,
    "sha256").hex()`` is CPython's optimized one-shot C path (the one the
    docs point performance-sensitive callers at), measured 0.92µs where
    the idiomatic ``hmac.new(...).hexdigest()`` costs 1.12µs. tors's one
    call (0.29µs) beats even the fast spelling by ~3x (ratio ~0.31; the
    long-key/short-data RFC 4231 case-6 shape ~0.36), because the stdlib
    spelling still pays two CPython calls (``hmac.digest`` plus ``.hex()``)
    against tors's single pyo3 call. The webhook-verification loop is
    exactly this shape: one HMAC per request, overhead-dominated."""
    key = b"k" * key_len
    data = b"d" * data_len
    tors_ms = _min_wall_ms(
        lambda _: tors.hmac_sha256_hex(key, data), None, samples=_HASH_MICRO_SAMPLES
    )
    std_ms = _min_wall_ms(
        lambda _: hmac.digest(key, data, "sha256").hex(), None, samples=_HASH_MICRO_SAMPLES
    )
    assert tors_ms < _MARGIN * std_ms, (
        f"hmac_sha256_hex k={key_len} d={data_len}: tors {tors_ms:.4f}ms vs "
        f"hmac.digest+hex {std_ms:.4f}ms (ratio {tors_ms / std_ms:.2f}): the "
        "one-call native HMAC lost more than the tolerance margin to the "
        "stdlib's fastest one-shot spelling"
    )


@pytest.mark.parametrize("size_bytes", [1024, 100 * 1024], ids=["1KiB", "100KiB"])
def test_scrub_log_text_beats_the_regex_chain_on_exception_text(size_bytes: int) -> None:
    """The scrub wall cells, at the two sizes the consumer's error path
    actually pays: a single failed job scrubs a message and a traceback at
    up to ~100 KB scale, and the issue's own cost profile for the chain
    (~20-35µs/KiB, ~2-3.4ms at 100 KB) is what these cells measure against
    the corpus that fires every rule once per unit (the DETAIL line, both
    credential shapes on the DSN, the repr()-flattened run).

    The comparator is the pinned regex chain itself (the four TaskQ
    patterns as compiled in ``tests/reference.py``, the same spellings the
    differential suite races tors against), so the wall race and the parity
    harness cross-reference on one oracle. Measured on the dev box (min-of-7
    at 1 KiB, min-of-3 at 100 KiB, after warmup):

        size    tors        chain      tors/chain
        1 KiB   0.001ms     0.045ms    0.03
        100 KiB 0.064ms     2.735ms    0.02

    A ~30-40x win, asserted with the shared 0.9 margin: the chain is four
    whole-text ``re.sub`` passes while tors is four linear memchr/memmem
    scans + splice under one ``py.detach``. The 1 KiB cell is fast-cell
    territory (µs-scale samples) and draws ``_FAST_CELL_SAMPLES``
    accordingly; even at that scale the margin
    absorbs a loaded runner many times over. The GIL-release side of the
    same surface is pinned in tests/test_gil_release.py (the 96 MiB
    heartbeat cell; the chain holds the loop for ~2.5s of a ~2.76s wall at
    that size, the red side this port exists for)."""
    corpus = scrub_corpus(size_bytes)
    tors_ms = _min_wall_ms(tors.scrub_log_text, corpus, samples=_samples_for(size_bytes))
    chain_ms = _min_wall_ms(
        reference_scrub_log_text, corpus, samples=_samples_for(size_bytes)
    )
    assert tors_ms < _MARGIN * chain_ms, (
        f"scrub_log_text {size_bytes}B: tors {tors_ms:.3f}ms vs chain "
        f"{chain_ms:.3f}ms (ratio {tors_ms / chain_ms:.3f}): the hand-rolled "
        "scan+splice lost the wall race it exists to win"
    )


@pytest.mark.parametrize("corpus_kind", ["prose", "decomposed"])
@pytest.mark.parametrize("size_bytes", [1 * _MIB, 12 * _MIB], ids=["1MiB", "12MiB"])
def test_grapheme_count_absolute_band_holds(corpus_kind: str, size_bytes: int) -> None:
    """``tors.grapheme_count`` has no stdlib comparator (the gap is the
    feature), so its wall cell is an absolute band, a regression ceiling
    with generous margin, not a race. Measured on the dev box (ambient load
    8.3, min-of-3 after warmup):

        corpus      size    time        throughput
        prose       1 MiB   ~10ms       ~100 MiB/s
        decomposed  1 MiB   9.7ms       ~105 MiB/s
        prose       12 MiB  122.8ms     ~102 MiB/s
        decomposed  12 MiB  117.6ms     ~104 MiB/s

    The band: 12 MiB must complete under 400ms (~3.3x above the measured
    118-123ms; the scan is a linear walk, stable under load, so a regression to
    a quadratic or per-cluster-allocation path blows straight through it).
    The GIL-release side of this function is pinned separately in
    tests/test_gil_release.py (worst gaps 10.4-12.8ms at 12 MiB, a single
    int return with no marshalling class at all)."""
    corpus = {"prose": prose, "decomposed": decomposed}[corpus_kind](size_bytes)
    took_ms = _min_wall_ms(tors.grapheme_count, corpus)
    assert took_ms < 400.0, (
        f"grapheme_count {corpus_kind} {size_bytes // _MIB}MiB took {took_ms:.0f}ms, "
        "outside the absolute band (measured ~118-123ms at 12 MiB, ceiling 400ms "
        "with 3.3x margin); the cluster scan regressed"
    )


# ---------------------------------------------------------------------------
# The utf8_byte_len wall cells (#52): tors.utf8_byte_len(s) vs
# len(s.encode("utf-8")), the expression it replaces. The full measured
# lane table (ambient load ~10-18 on the calibration box, macOS, 16 cores,
# min-of-7 after warmup unless noted):
#
#     ASCII (prose), the TaskQ serialized-JSON case (ensure_ascii=True
#     output is pure ASCII): tors is FLAT ~0.1µs at every size (the
#     zero-copy alias: compact ASCII data is its own UTF-8, nothing to
#     build), while the expression pays alloc+memcpy every call:
#
#         size    tors        encode     ratio
#         1 KiB   0.08-0.13µs 0.13µs     0.7-1.0  (a dead heat: both sides are
#                                                   pure call overhead; recorded,
#                                                   not asserted)
#         64 KiB  0.13µs      0.9µs      0.14   (the TaskQ terminal size: ~0.9µs
#                                                 of pure alloc+memcpy per
#                                                 success — the figure every
#                                                 doc site cites for the
#                                                 terminal case)
#         1 MiB   0.13µs      14.3µs     0.009
#         12 MiB  0.13µs      184µs      0.0007
#
#     non-ASCII (decomposed), the cache lanes (the borrow's UTF-8 view is
#     materialized once per OBJECT and cached by CPython; the sharing with
#     encode is one-directional — the str-in borrow fills the cache and
#     encode reads it but never fills it, observed on CPython 3.12 here
#     and expected from the sources on 3.10-3.14 — see docs/cache-proof.md
#     for the per-version Objects/unicodeobject.c links and the
#     ripgrep recipe (`unicode_fill_utf8`, the only writer, reachable
#     solely from PyUnicode_AsUTF8AndSize, while unicode_encode_utf8
#     returns a copy of a filled cache and writes nothing on a miss).
#     Semantic pins are the contract; timing is not):
#
#         64 KiB:  cold-encode 21.8µs | first-call 26.7µs | warm-encode 1.8µs
#                  | cached-tors 0.08µs
#         1 MiB:   cold-encode 355.8µs | first-call 409.7µs | warm-encode 14.8µs
#                  | cached-tors 0.13µs
#         12 MiB:  cold-encode 4.9ms | first-call 4.7ms | warm-encode 196µs
#                  | cached-tors 0.13µs
#
#     The honest reading: on a FRESH non-ASCII object the first call is
#     encode-parity (the materialization IS an encode - ucs2lib encoder
#     pass plus a malloc plus a second full memcpy into the permanent
#     cache, measured within ~10-20% of a cold encode), so the win there
#     is only the absence of a Python-visible bytes object; the win is on
#     REPEAT calls on the same object (the warm-encode lane itself is 20-
#     1600x the cached call), and unconditionally on ASCII.
#
#     Both sharing directions, measured (a red-team pass reported the
#     reverse of the recorded one; re-measured to adjudicate, min-of-7
#     FRESH 12 MiB objects per lane, this box at ambient load ~6-7):
#     tors-primed encode 177µs (the consult: one memcpy out of the filled
#     cache - the recorded 184µs lane, reproduced); encode-primed FIRST
#     utf8_byte_len 3.85-4.79ms, indistinguishable from the cold
#     first-call lane (3.91-4.22ms) - a prior encode does NOT warm the
#     tors lane. The reported 0.12-0.21µs "first call after encode" is
#     this table's own cached-tors band (0.13µs), i.e. a warm object: the
#     measurement to make that number is a cached call, not a first call.
#     The same re-measurement pass re-checked the disputed 64 KiB figure
#     (P2-2): the ASCII expression cell landed 0.75µs here (min-of-7,
#     same load) against the recorded 0.9µs - the same class, and the
#     lane table's 0.9µs stands as the ONE figure cited everywhere; the
#     ~1.5µs the binding/test docstrings had carried matches no recorded
#     lane (the nearest class is the non-ASCII warm encode's single
#     1.8µs memcpy, which is not the terminal case - the terminal's
#     serialized result is ASCII).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes"),
    [
        ("ascii", 64 * 1024),
        ("ascii", 1 * _MIB),
        ("nonascii-cached", 64 * 1024),
        ("nonascii-cached", 1 * _MIB),
    ],
    ids=["ascii-64KiB", "ascii-1MiB", "nonascii-cached-64KiB", "nonascii-cached-1MiB"],
)
def test_utf8_byte_len_beats_the_encode_expression_on_both_winnable_lanes(
    corpus_kind: str, size_bytes: int
) -> None:
    """The race, asserted only where it is honestly winnable. Two lanes:
    ``ascii`` (prose, the TaskQ serialized case — compact ASCII is its own
    UTF-8, so the borrow is a zero-copy alias and the call is O(1) with no
    allocation, while the expression pays alloc+memcpy every call) and
    ``nonascii-cached`` (decomposed, the methodology's warmup having primed
    the object's UTF-8 cache, so the race is the repeat-call semantics:
    the cached O(1) borrow against the expression's warm one-memcpy copy
    out of the same cache). Measured ratios 0.14/0.009 (ASCII 64 KiB/1 MiB)
    and 0.05/0.008 (non-ASCII cached 64 KiB/1 MiB) against the shared 0.9
    margin: the fresh-object non-ASCII lane, where the first call is
    encode-parity by construction, is measured and recorded in the cell
    below, never asserted."""
    corpus = prose(size_bytes) if corpus_kind == "ascii" else decomposed(size_bytes)
    samples = _samples_for(size_bytes)
    tors_ms = _min_wall_ms(tors.utf8_byte_len, corpus, samples=samples)
    enc_ms = _min_wall_ms(lambda s: len(s.encode("utf-8")), corpus, samples=samples)
    assert tors_ms < _MARGIN * enc_ms, (
        f"utf8_byte_len {corpus_kind} {size_bytes // 1024}KiB: tors {tors_ms * 1000:.2f}µs vs "
        f"encode {enc_ms * 1000:.2f}µs (ratio {tors_ms / enc_ms:.3f}): the count lost more "
        "than the tolerance margin to the copy it exists to avoid"
    )


def test_utf8_byte_len_ascii_calls_stay_o1_at_12mib() -> None:
    """The O(1) pin the ratio race cannot make by itself: at 12 MiB the
    ASCII call must stay in the call-overhead band (measured ~0.13µs flat
    from 1 KiB to 12 MiB), under a 5µs ceiling (~40x margin; µs-scale
    samples draw the fast-cell sample count, min-of-7, since one preempted
    run can set a min-of-3). A per-call O(n) regression - a validation
    pass over the borrowed bytes, a lost zero-copy alias in a pyo3 upgrade
    - lands at the encode class (~180µs at 12 MiB) and blows through by
    ~36x. The same regression class on the non-ASCII path is caught by the
    nonascii-cached race leg (the cached lane would regress to the
    materialization class)."""
    corpus = prose(12 * _MIB)
    tors_us = _min_wall_ms(tors.utf8_byte_len, corpus, samples=_FAST_CELL_SAMPLES) * 1000
    assert tors_us < 5.0, (
        f"utf8_byte_len ASCII 12MiB took {tors_us:.2f}µs, outside the O(1) call band "
        "(measured ~0.13µs flat across sizes, ceiling 5µs); the borrow stopped being "
        "a zero-copy alias or gained a per-call scan"
    )


def test_utf8_byte_len_fresh_object_lanes_are_measured_not_asserted() -> None:
    """The lanes where there is no win to assert, recorded instead (the
    decode_utf8/b64_decode precedent): the 1 KiB ASCII race is a dead heat
    (both sides ~0.1µs of pure call overhead - the 1 KiB memcpy is
    invisible at that size), and a non-ASCII FRESH object's first call is
    encode-parity by construction (the materialization is an encode: the
    same ucs-to-UTF-8 pass plus a malloc plus a second memcpy into the
    permanent cache; measured within ~10-20% of a cold encode at 1 MiB).
    The parity is the honest cost of the route the implementation chose
    (borrow, not hand-rolled arithmetic), and the reason the asserted
    cells above carry only the lanes that are structurally winnable."""
    one_kib = prose(1024)
    tors_us = _min_wall_ms(tors.utf8_byte_len, one_kib, samples=_FAST_CELL_SAMPLES) * 1000
    enc_us = (
        _min_wall_ms(lambda s: len(s.encode("utf-8")), one_kib, samples=_FAST_CELL_SAMPLES) * 1000
    )
    print(f"utf8_byte_len ASCII 1KiB: tors {tors_us:.2f}µs encode {enc_us:.2f}µs (dead heat)")
    # The cold lanes need a FRESH object per timed call, so the copies are
    # built up front (a generator would build each corpus inside the timed
    # lambda) and the helper runs with warmup=0 — every sample is a first
    # contact with its own object.
    size = 1 * _MIB
    cold_copies = iter([decomposed(size) for _ in range(_FAST_CELL_SAMPLES)])
    cold_us = (
        _min_wall_ms(
            lambda _: len(next(cold_copies).encode("utf-8")),
            "",
            warmup=0,
            samples=_FAST_CELL_SAMPLES,
        )
        * 1000
    )
    fresh_copies = iter([decomposed(size) for _ in range(_FAST_CELL_SAMPLES)])
    first_us = (
        _min_wall_ms(
            lambda _: tors.utf8_byte_len(next(fresh_copies)),
            "",
            warmup=0,
            samples=_FAST_CELL_SAMPLES,
        )
        * 1000
    )
    print(
        f"utf8_byte_len non-ASCII 1MiB fresh-object: cold-encode {cold_us:.1f}µs "
        f"first-call {first_us:.1f}µs (parity, no assert)"
    )


# ---------------------------------------------------------------------------
# The utf16_byte_len wall cells (#52, the interop twin):
# tors.utf16_byte_len(s) vs len(s.encode("utf-16-le")), the expression it
# replaces. CROSS-ARCH lane table (the calibration box, arm64 NEON, and
# the CI 3.12 leg, x86-64 SSE2-baseline — the chunk loop auto-vectorizes
# on the first and not on the second, which is why the ASCII fast path
# in src/scan_impl.rs exists: is_ascii is the one portably-SIMD part,
# so the ASCII lane is target-independent and the non-ASCII lane is
# target-dependent, both stated):
#
#     the warm lanes (repeat calls on one object):
#
#         ASCII (the fast path — 2*len after a std-SIMD is_ascii scan):
#         wins outright on EVERY target (arm64 ~0.1x of the expression,
#         x86-baseline comparable; the expression pays its 2n alloc +
#         widen pass everywhere) — asserted at the shared 0.9 margin.
#
#         non-ASCII (the chunk loop): arm64 0.21-0.25 of the expression
#         at 64 KiB / 1 MiB (the ~30 GB/s NEON scan); x86-64 SSE2-baseline
#         (the CI runners) 2.2-2.5x SLOWER — the loop does not
#         auto-vectorize there and the expression is memcpy-class C. The
#         value on that target is the zero-allocation and the GIL release,
#         not the wall win — asserted at a 4.0 cross-arch bound (no
#         catastrophe; the win itself is recorded, per-lane, not
#         asserted cross-arch).
#
#         12 MiB ASCII: the fast-path pin — ~0.4ms arm64, ~1.5-2.5ms
#         x86-baseline, ceiling 4.0ms. The regression the ceiling is
#         sized for is the loss of the fast path on the non-vectorizing
#         target (the chunk loop measured ~10ms there, 10-25x the band)
#         and any superlinear blowup anywhere.
#
#     The honest lane, recorded not asserted: a FRESH non-ASCII object's
#     first call pays the UTF-8-cache materialization (the utf8 twin's
#     cold class, GIL-held) before the scan — measured 376µs at 1 MiB
#     against the expression's own cold 146µs. The utf-16 expression
#     never materializes UTF-8 at all, so the cold first call is the one
#     lane the expression wins; every call after it is the warm lanes
#     above, and no call ever allocates the 2n bytes object.
# ---------------------------------------------------------------------------

_UTF16_CROSS_ARCH_MARGIN = 4.0


@pytest.mark.parametrize(
    ("corpus_kind", "size_bytes"),
    [
        ("ascii", 64 * 1024),
        ("ascii", 1 * _MIB),
        ("nonascii-cached", 64 * 1024),
        ("nonascii-cached", 1 * _MIB),
    ],
    ids=["ascii-64KiB", "ascii-1MiB", "nonascii-cached-64KiB", "nonascii-cached-1MiB"],
)
def test_utf16_byte_len_beats_the_encode_expression_on_the_warm_lanes(
    corpus_kind: str, size_bytes: int
) -> None:
    """The race, asserted per lane where it is honestly winnable: the
    ASCII lanes win outright on EVERY target (the is_ascii fast path is
    std-SIMD everywhere — 2*len after the scan, against the expression's
    2n alloc + widen), so they keep the shared 0.9 margin; the non-ASCII
    lanes run the chunk loop, which auto-vectorizes on NEON (measured
    0.21-0.25 of the expression) but NOT on SSE2-baseline x86-64 (the CI
    runners measured 2.2-2.5x slower there — the expression is
    memcpy-class C), so the cross-arch assertion is the 4.0 no-catastrophe
    bound with both lanes' numbers recorded in the module comment above.
    The one lane the expression always wins — a FRESH non-ASCII object's
    first call, where the borrow materializes the UTF-8 cache — is
    measured and recorded in the cell below, never asserted."""
    corpus = prose(size_bytes) if corpus_kind == "ascii" else decomposed(size_bytes)
    samples = _samples_for(size_bytes)
    tors_ms = _min_wall_ms(tors.utf16_byte_len, corpus, samples=samples)
    enc_ms = _min_wall_ms(lambda s: len(s.encode("utf-16-le")), corpus, samples=samples)
    margin = _MARGIN if corpus_kind == "ascii" else _UTF16_CROSS_ARCH_MARGIN
    assert tors_ms < margin * enc_ms, (
        f"utf16_byte_len {corpus_kind} {size_bytes // 1024}KiB: tors {tors_ms * 1000:.2f}µs vs "
        f"encode {enc_ms * 1000:.2f}µs (ratio {tors_ms / enc_ms:.3f}): outside the "
        f"{corpus_kind} lane's {margin}x cross-arch bound (the lane table in this "
        "module's comment records both targets' measured numbers)"
    )


def test_utf16_byte_len_warm_ascii_calls_stay_in_the_fast_path_band_at_12mib() -> None:
    """The fast-path pin (the utf8 twin's O(1) pin, translated to this
    core's ASCII lane): at 12 MiB the warm ASCII call is the is_ascii
    scan + a multiply — measured ~0.4ms on arm64, ~1.5-2.5ms on the
    SSE2-baseline CI runners — under a 4.0ms cross-arch ceiling. The
    regression the ceiling is sized for is the loss of the ASCII fast
    path on the non-vectorizing target (the chunk loop measured ~10ms
    there, 10-25x over) and any superlinear blowup anywhere; a
    constant-factor scan change is bench-visible, not cell-caught, and
    this cell does not pretend otherwise."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(tors.utf16_byte_len, corpus, samples=_SAMPLES)
    assert tors_ms < 4.0, (
        f"utf16_byte_len ASCII 12MiB took {tors_ms * 1000:.0f}µs, outside the fast-path "
        "band (measured ~0.4ms arm64 / ~1.5-2.5ms x86-baseline, ceiling 4.0ms); the "
        "ASCII fast path was lost or the scan went superlinear "
        "(the chunk loop measured ~10ms on the non-vectorizing target)"
    )


def test_utf16_byte_len_fresh_object_first_call_is_measured_not_asserted() -> None:
    """The lane the expression wins, recorded (the decode_utf8/b64_decode/
    utf8-twin precedent): a fresh non-ASCII object's first call pays the
    borrow's UTF-8-cache materialization (GIL-held, the utf8 twin's cold
    class, encode-utf-8-parity in cost) before the scan — and the utf-16
    expression never materializes UTF-8 at all, so on a cold object the
    expression is the cheaper call (measured 146µs against 376µs at
    1 MiB on this box). The trade buys every subsequent call (the warm
    lanes above, 4-5x) and the absence of a 2n bytes object per call;
    the ASCII lane has no cold case at all (compact ASCII is its own
    UTF-8)."""
    size = 1 * _MIB
    cold_copies = iter([decomposed(size) for _ in range(_FAST_CELL_SAMPLES)])
    cold_us = (
        _min_wall_ms(
            lambda _: len(next(cold_copies).encode("utf-16-le")),
            "",
            warmup=0,
            samples=_FAST_CELL_SAMPLES,
        )
        * 1000
    )
    fresh_copies = iter([decomposed(size) for _ in range(_FAST_CELL_SAMPLES)])
    first_us = (
        _min_wall_ms(
            lambda _: tors.utf16_byte_len(next(fresh_copies)),
            "",
            warmup=0,
            samples=_FAST_CELL_SAMPLES,
        )
        * 1000
    )
    print(
        f"utf16_byte_len non-ASCII 1MiB fresh-object: cold-encode {cold_us:.1f}µs "
        f"first-call {first_us:.1f}µs (the borrow's materialization lane, no assert)"
    )


# ---------------------------------------------------------------------------
# The chunking family's document-scale cost shape (#22, then #30): the
# per-call cost must be the segmentation walks the function is for, not
# per-codepoint structures built unconditionally, and, since #30's lazy
# levels, only the walks a call actually consults. Load-fair ratios (both
# sides measured in the same process, min-of-3 after warmup), so a
# shared-runner slowdown inflates both sides together.
#
# Measured on the dev box (Linux, CPython 3.12, min-of-3 after warmup),
# before the fix -> after:
#
#     chunk_hierarchical, 12 MiB degenerate single-char run, custom
#     never-matching separators, whole-document budget: ~1048ms -> ~2.6ms
#     after #22 (the unconditional grapheme HashSet + Vec<char> collect;
#     ~400x); after #30's lazy levels the same call pays only the
#     codepoint count: the literal scan itself is skipped, since no
#     window ever opens a level
#
#     chunk_hierarchical, 12 MiB prose, default hierarchy: ~2996ms ->
#     ~351ms after #22; -> ~1.2ms at a 2000-char budget after #30's lazy
#     levels (every window is served by the paragraph level alone on this
#     corpus, so the sentence/word walks are never built), ~0.1ms under a
#     whole-document budget (no level consulted at all; ~340ms before,
#     ~176ms at 6 MiB), and ~396ms at a 100-char budget that descends
#     to the word level: the word (~132ms) plus sentence
#     (~188ms) walks it actually uses, the accurate UAX #29 hierarchy
#     the function exists to provide
#
#     chunk_by_words, 12 MiB prose, 200 words/chunk: ~1923ms -> ~165ms
#     chunk_by_sentences, 12 MiB prose, 10 sentences/chunk:
#     ~1328ms -> ~200ms
#
# Follow-up pass (lazy level builds, byte-level line/paragraph scans,
# char_count ASCII fast path): default hierarchy @2000 12 MiB prose
# ~351ms -> ~1.2ms (a paragraph-scale budget consults only the
# paragraph level; the sentence/word walks never run), whole-document
# budgets ~340ms -> ~0.1ms (one codepoint count, no level consulted,
# and the count itself is the ASCII fast path), by-lines @50 12 MiB
# ~8ms -> ~0.5ms and by-paras on a log-density corpus ~7.5ms -> ~1.4ms
# (density-guarded byte scan with a sliding ASCII certificate;
# non-ASCII segments keep the byte path per segment). The
# dedup/laziness contract cells (min-of-3 after warmup): default
# hierarchy @2000 over 12 MiB ~1.5ms vs ~1.5ms for a ["\n\n"]-only
# custom hierarchy (ratio ~1.0; the eager pre-#30 spelling measured
# ~351ms on the same corpus); and [None]*8 / [" "]*8 at a descending
# 8-codepoint budget over 6 MiB each ~1.0x their lone-entry spellings
# with the slot-construction dedups in the built extension (the
# duplicated [" "] measured ~1.7x pre-dedup -- the literal spelling of
# the [None]*8 walk re-payment the dedup closes). The docstrings below
# carry the detail.
# ---------------------------------------------------------------------------


def test_chunk_hierarchical_custom_no_match_is_scan_cost_not_per_char_structures() -> None:
    """The whole-document-budget custom-hierarchy cell from #22: a
    never-matching separator list under a budget that swallows the whole
    text must cost one codepoint count and nothing else. Under the
    lazy-level pass the level is never consulted (the first window fits
    the entire document), so it is never built and never scanned -- the
    literal's scan does not run at all, where the pre-lazy spelling paid
    one count pass plus one scan pass. The reference stays CPython's own
    ``in``, a C-speed scan of the same text, because the class of
    regression this cell guards (an unconditional per-codepoint
    structure or scan built before the budget is even examined) shows up
    as wall time against exactly that reference. With the char_count
    ASCII fast path the count lane itself drops to ~0.1ms at 12 MiB, so
    the measured ratio against ``in`` is ~0.05 (count pass only; ~2
    after #22, when the scan still ran on top). The pre-#22 spelling
    measured ~700x.

    The absolute ceiling (0.8ms) is the fast path's own pin, which the
    8x scan race above cannot provide: a char_count reverted to the
    predicate-only spelling (no ``is_ascii`` gate) measures ~2.5ms on
    this corpus (red-proofed: that revert fails this cell), which still
    passes 8x an ~8ms scan, so the race is blind to exactly the
    fast-path loss. 0.8ms sits ~3-5x above the measured band
    (0.15-0.25ms, min-of-3 after warmup on the box this ceiling was
    calibrated on) and ~3x below the predicate-only spelling, so the
    gate's loss fails this cell while CI load does not."""
    q = "q" * (12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 12 * _MIB, ["xyz"]), q)
    scan_ms = _min_wall_ms(lambda s: "xyz" in s, q)
    assert tors_ms < 8.0 * scan_ms, (
        f"chunk_hierarchical no-match 12MiB took {tors_ms:.1f}ms against an "
        f"{scan_ms:.1f}ms bare scan ({tors_ms / scan_ms:.0f}x); the lazily-built "
        "grapheme machinery regressed to an unconditional structure"
    )
    assert tors_ms < 0.8, (
        f"chunk_hierarchical no-match 12MiB took {tors_ms:.2f}ms, over the "
        "whole-budget absolute ceiling (measured ~0.15-0.25ms with the char_count "
        "ASCII fast path, ceiling 0.8ms; the predicate-only spelling measures "
        "~2.5ms and must fail this cell); the char_count ASCII fast path regressed"
    )


def test_chunk_hierarchical_custom_no_match_skips_the_grapheme_walk_on_non_ascii() -> None:
    """The laziness contract, on the input where it is load-bearing: for
    non-ASCII text a whole-text level build is a full segmentation walk
    (the grapheme index ~150ms at 12 MiB), so a call that never consults
    a level (never-matching separators, whole-document budget: no cuts
    to filter, no raw-cut window, no overlap snap) must build nothing.
    Under the lazy-level pass the literal is never even scanned -- the
    whole call is one codepoint count (an O(n) walk on non-ASCII text)
    plus marshalling, where the pre-lazy spelling paid the scan pass too.
    The reference is the same ``in`` scan; measured ~2.6ms vs ~1.4ms on
    decomposed prose (ratio ~1.9) on this box (Linux, CPython 3.13,
    ambient load 42-85, min-of-3 after warmup), and both builds sit in
    the same 2.4-3.6ms noise-bound band -- no speedup claimed, the
    cheaper structure is the point. ~75x if the index were built
    unconditionally -- the ASCII fast path masks that regression on
    single-byte corpora, which is why this cell runs decomposed text."""
    corpus = decomposed(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, len(s), ["xyz"]), corpus)
    scan_ms = _min_wall_ms(lambda s: "xyz" in s, corpus)
    assert tors_ms < 8.0 * scan_ms, (
        f"chunk_hierarchical no-match non-ASCII 12MiB took {tors_ms:.1f}ms against "
        f"an {scan_ms:.1f}ms bare scan ({tors_ms / scan_ms:.0f}x); the grapheme "
        "index is being built on a call that cannot use it"
    )


def test_chunk_hierarchical_whole_document_budget_pays_no_level_walks() -> None:
    """The #30 lazy-level headline cell: a whole-document budget consults
    no level at all -- the loop's first iteration takes its own
    ``remaining <= max_chars`` exit before any level is realized -- so
    the default hierarchy must cost the same nothing the never-matching
    custom one does: one codepoint count, one chunk out. Ratioed against
    CPython's own ``in`` (a C-speed scan of the same text): measured
    ~0.1ms vs ~2ms (ratio ~0.05) after the lazy levels; the eager
    spelling measured ~340ms here (~170x, and ~176ms at 6 MiB) -- the
    three default walks paid for levels that supplied zero cuts.

    The absolute ceiling (0.8ms) is the char_count ASCII fast path's pin
    on this lane, the no-match cell's twin rationale: the 8x scan race
    cannot see the fast path's loss (a predicate-only count ~2.5ms at
    12 MiB still passes 8x an ~8ms scan), while 0.8ms sits ~2.4-5x above
    the measured band (0.16-0.33ms, min-of-3 after warmup on the box
    this ceiling was calibrated on) and ~3x below the predicate-only
    spelling (red-proofed: that revert fails this cell), so the
    ``is_ascii`` gate's loss fails this cell."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, len(s)), corpus)
    scan_ms = _min_wall_ms(lambda s: "xyz" in s, corpus)
    assert tors_ms < 8.0 * scan_ms, (
        f"chunk_hierarchical whole-document default 12MiB took {tors_ms:.1f}ms "
        f"against an {scan_ms:.1f}ms bare scan ({tors_ms / scan_ms:.0f}x); "
        "levels are being built on a call that consults none of them"
    )
    assert tors_ms < 0.8, (
        f"chunk_hierarchical whole-document default 12MiB took {tors_ms:.2f}ms, "
        "over the whole-budget absolute ceiling (measured ~0.16-0.33ms with the "
        "char_count ASCII fast path, ceiling 0.8ms; the predicate-only spelling "
        "measures ~2.5ms and must fail this cell); the char_count ASCII fast "
        "path regressed"
    )


def test_chunk_hierarchical_default_hierarchy_is_its_own_segmentation_walks() -> None:
    """The default hierarchy's per-call cost contract: no more than its
    own UAX #29 walks (word_count + sentence_count on the same corpus,
    measured in-process), at a budget that descends to the
    word level so every walk is consulted and the contract has teeth.
    The hierarchy is those walks; everything around them -- level
    realization, the cut filter, the chunk loop, marshalling -- must be
    marginal. Measured ~396ms against a ~310ms reference sum (ratio
    ~1.3) at a 100-codepoint budget (at a 2000-codepoint budget on this
    corpus every window is served by the paragraph level alone, so the
    same call drops to ~1.2ms -- the sentence/word walks are never
    built; the dedicated ["\\n\\n"]-ratio cell below pins that property
    against a tighter reference); the pre-#22 spelling measured ~9.4x
    (the grapheme hash set dominated the segmentation it was
    filtering). The assertion is the standing ceiling in the other
    direction: whatever levels a budget does consult, the machinery
    around the walks must not dominate them."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 100), corpus)
    walks_ms = _min_wall_ms(tors.word_count, corpus) + _min_wall_ms(tors.sentence_count, corpus)
    assert tors_ms < 2.0 * walks_ms, (
        f"chunk_hierarchical default 12MiB took {tors_ms:.0f}ms against "
        f"{walks_ms:.0f}ms of its own segmentation walks "
        f"({tors_ms / walks_ms:.1f}x); per-call machinery is dominating the walks"
    )


def test_chunk_hierarchical_default_hierarchy_builds_only_the_levels_the_budget_consults() -> None:
    """The lazy-level contract at the budget where it pays: at a
    paragraph-scale budget (2000 codepoints over 12 MiB of prose) every
    window is answered by the paragraph level, so the sentence and word
    walks -- the expensive lower levels of the default hierarchy -- must
    never run at all. The load-fair reference is ``["\\n\\n"]``, a custom
    hierarchy whose single literal supplies an equivalent top level
    without the default hierarchy's lower levels: both sides pay the
    same paragraph scan plus the same windowing, so the default spelling
    may only add marginal spec-construction cost, never two more
    whole-text walks. Measured (min-of-3 after warmup): ~1.5ms default
    vs ~1.5ms literal (ratio ~1.0). The eager pre-#30 spelling measured
    ~351ms on the same corpus (the sentence walk ~188ms and the word
    walk ~132ms ran on a call whose budget never consulted them, ~200x
    this reference) -- the exact regression class the 3.0x ceiling is
    positioned to catch."""
    corpus = prose(12 * _MIB)
    default_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000), corpus)
    literal_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000, ["\n\n"]), corpus)
    assert default_ms < 3.0 * literal_ms, (
        f"chunk_hierarchical default @2000 12MiB took {default_ms:.1f}ms against "
        f"{literal_ms:.1f}ms for the paragraph-literal-only hierarchy "
        f"({default_ms / literal_ms:.1f}x); a budget the lower levels never "
        "answer is building them anyway"
    )


def test_chunk_hierarchical_duplicate_separator_entries_are_deduped_not_rebuilt() -> None:
    """The dedup contract at slot construction, load-fair ratios (both
    sides in the same process, min-of-3 after warmup): duplicate entries
    in a custom separator list are collapsed when the slot list is
    built -- a second ``None`` never re-splices the default hierarchy, a
    repeated literal never re-pays its scan -- so ``[None] * 8`` must
    cost one spliced hierarchy and ``[" "] * 8`` one literal level, not
    eight of either. The budget is descending (8 codepoints: windows
    inside long words exhaust the ``" "`` level down to the raw cut),
    which is where a lazy spelling without the dedup would still re-pay
    a duplicate -- the find_map only reaches one after its original
    returned None for that window, so a budget whose every window is
    answered by the first copy can never see the duplicates. Measured
    (min-of-3 after warmup; 6 MiB prose): ``[None]`` ~350ms vs ``[None] * 8`` ~355ms
    (ratio ~1.0), ``[" "]`` ~170ms vs ``[" "] * 8`` ~170ms (ratio
    1.0) with the dedup in the built extension -- the same ``[" "]``
    pair measures ~1.7x (~150ms vs ~258ms) without the dedup, the
    seven extra duplicate scans and cut vectors the descent re-paid.
    The eager spelling measured ~8x here outright (the ``[None]`` pair
    1347.1ms vs 169.2ms on the pre-dedup build), and the
    ``[" "] * 100`` spelling was a 1.69 GiB peak OOM shape
    (``[None] * 100`` its splice twin) without them. The
    structural teeth wall time cannot see -- that a duplicate after a
    consulted level adds no builds at all -- are the Rust seam pins
    (the build_seam LEVELS_BUILT counts in
    src/chunk_hierarchical_impl.rs)."""
    corpus = prose(6 * _MIB)
    lone_none_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [None]), corpus)
    eight_none_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [None] * 8), corpus)
    assert eight_none_ms < 1.5 * lone_none_ms, (
        f"chunk_hierarchical [None]*8 @8 6MiB took {eight_none_ms:.1f}ms against "
        f"{lone_none_ms:.1f}ms for [None] ({eight_none_ms / lone_none_ms:.1f}x); "
        "duplicate separator entries are being rebuilt per slot instead of deduped"
    )
    lone_space_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [" "]), corpus)
    eight_space_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [" "] * 8), corpus)
    assert eight_space_ms < 1.5 * lone_space_ms, (
        f"chunk_hierarchical [' ']*8 @8 6MiB took {eight_space_ms:.1f}ms against "
        f"{lone_space_ms:.1f}ms for [' '] ({eight_space_ms / lone_space_ms:.1f}x); "
        "duplicate separator entries are being rebuilt per slot instead of deduped"
    )


def test_chunk_hierarchical_none_splice_duplicates_are_inert_at_documented_budgets() -> None:
    """The ``[None] * 100`` dedup contract at the budget the docs disclose
    (the README's and api.md's "~0.5 ms at a 2000-codepoint budget over
    6 MiB of prose, the paragraph walk alone" parenthetical): at a
    paragraph-scale budget every window is answered by the first paragraph
    slot, so the 99 duplicate splices are never consulted at all (the
    find_map dominance argument: the first slot supplying a cut wins) and
    the ``*100`` spelling must cost exactly the lone spelling: one spliced
    hierarchy, one paragraph walk. Measured (min-of-3 after warmup, 6 MiB
    prose, the box this cell was written on: Linux, 32 logical cores,
    ambient load ~5): ~0.50 ms for both spellings, ratio 1.00-1.01.

    That same dominance argument is the @2000 leg's limit:
    a lost dedup is invisible at this budget, because the duplicate slots
    a lost dedup would leave in the list are never consulted; measured
    directly, with the slot-construction dedup disabled in a scratch
    build, the @2000 pair still measures ratio ~1.0 (green). The teeth
    therefore live in the second leg, the descending budget the sibling
    cell above established (@8, ``[None] * 8``): at 8 codepoints over
    1 MiB of prose, windows inside long words exhaust every spliced level
    down to the raw cut, so the find_map walks past every slot and each
    duplicate splice re-pays the three walks; with the dedup disabled,
    that scratch build measured the ``*100`` spelling at ~3.6 s against
    ~48 ms (ratio ~75x) at this exact leg, where the pristine tree
    measures ~48 ms for both spellings (ratio ~1.0). The ``*100`` scale
    (not the sibling's ``*8``) is the docs' own OOM shape: the pre-dedup
    eager spelling measured 17.2 s and +3,120 MiB of peak RSS at 6 MiB.
    The 1 MiB corpus (not the sibling's 6) keeps the leg at ~0.2 s of
    measurement while the disabled-dedup red side still lands at ~75x.

    Budgets: the @2000 leg takes the docs-claim ratio (1.3x, ~1.3x above
    the worst measured ratio) plus an absolute 2.5 ms ceiling (~5x above
    the measured 0.50 ms, CI-load headroom; the regression this leg is
    positioned to catch (levels built eagerly again, every duplicate
    paying its walks up front, the 17.2 s class) blows through it by
    ~7000x). The descending leg takes the sibling cell's 1.5x margin.
    The @2000 wall is the paragraph walk, so a paragraph-scanner retune
    moves the ceiling's headroom, not the ratio legs (both sides pay the
    same walk)."""
    corpus = prose(6 * _MIB)
    lone_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000, [None]), corpus)
    hundred_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 2000, [None] * 100), corpus)
    assert hundred_ms < 1.3 * lone_ms, (
        f"chunk_hierarchical [None]*100 @2000 6MiB took {hundred_ms:.2f}ms against "
        f"{lone_ms:.2f}ms for [None] ({hundred_ms / lone_ms:.2f}x); duplicate None "
        "splices are costing more than one spliced hierarchy at a budget that "
        "never consults them"
    )
    assert hundred_ms < 2.5, (
        f"chunk_hierarchical [None]*100 @2000 6MiB took {hundred_ms:.2f}ms, over the "
        "absolute ceiling (measured ~0.50ms, ceiling 2.5ms with ~5x load headroom); "
        "duplicates are being built eagerly instead of deduped at slot construction"
    )
    small = prose(1 * _MIB)
    lone8_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [None]), small)
    hundred8_ms = _min_wall_ms(lambda s: chunk_hierarchical(s, 8, [None] * 100), small)
    assert hundred8_ms < 1.5 * lone8_ms, (
        f"chunk_hierarchical [None]*100 @8 1MiB took {hundred8_ms:.1f}ms against "
        f"{lone8_ms:.1f}ms for [None] ({hundred8_ms / lone8_ms:.1f}x); duplicate "
        "None splices are being rebuilt per slot instead of deduped on the "
        "descending budget that consults them"
    )


def test_chunk_by_words_is_its_own_word_walk() -> None:
    """chunk_by_words' per-call cost contract: the word walk it is for,
    plus only marginal machinery (the grapheme index build, the merge, the
    streaming token filter). Measured ~165ms against ~132ms of word_count
    (ratio ~1.25) after the fix; the former spelling measured ~14.6x."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_by_words(s, 200), corpus)
    walk_ms = _min_wall_ms(tors.word_count, corpus)
    assert tors_ms < 2.5 * walk_ms, (
        f"chunk_by_words 12MiB took {tors_ms:.0f}ms against a {walk_ms:.0f}ms "
        f"word walk ({tors_ms / walk_ms:.1f}x); the merge/filter machinery is "
        "dominating the segmentation it windows"
    )


def test_chunk_by_sentences_is_its_own_sentence_walk() -> None:
    """chunk_by_sentences' cost contract, same shape as chunk_by_words':
    measured ~200ms against ~188ms of sentence_count (ratio ~1.07) after
    the fix; the former spelling measured ~7.1x."""
    corpus = prose(12 * _MIB)
    tors_ms = _min_wall_ms(lambda s: chunk_by_sentences(s, 10), corpus)
    walk_ms = _min_wall_ms(tors.sentence_count, corpus)
    assert tors_ms < 2.5 * walk_ms, (
        f"chunk_by_sentences 12MiB took {tors_ms:.0f}ms against a "
        f"{walk_ms:.0f}ms sentence walk ({tors_ms / walk_ms:.1f}x); the merge "
        "machinery is dominating the segmentation it windows"
    )


def test_chunk_by_paragraphs_absolute_band_holds() -> None:
    """``chunk_by_paragraphs`` has no stdlib comparator (a paragraph here
    is tors's own documented 2+-newline heuristic), so like
    ``grapheme_count`` its wall cell is an absolute regression ceiling
    with generous margin, not a race. Measured (min-of-3 after warmup,
    the helper's own sample count): ~0.5ms at 12 MiB of prose against
    the current byte-level scanner (the pre-fast-path per-char spelling
    measured ~10.4ms at the same corpus on the box that calibrated this
    cell, and 12.8ms on a CRLF-dense log there). The ceiling is 10ms
    (~20x margin over the current measurement), positioned so that a
    regression back to the per-char decode loop fails this cell: the old
    spelling's ~10.4ms blows straight through 10ms, where the former
    30ms ceiling would have absorbed it silently. This cell exists
    because the #28 codegen change moved this function's wall by +23-28%
    (8.55ms -> 10.96ms on the same corpus, measured at the criterion
    level) without any existing cell noticing -- the chunking-family
    section had no paragraph row at all; a future change of that class
    should at least have a ceiling to argue against."""
    corpus = prose(12 * _MIB)
    took_ms = _min_wall_ms(lambda s: chunk_by_paragraphs(s, 200), corpus)
    assert took_ms < 10.0, (
        f"chunk_by_paragraphs 12MiB took {took_ms:.1f}ms, outside the absolute "
        "band (measured ~0.5ms at 12 MiB, ceiling 10ms; the pre-fast-path "
        "per-char spelling measured ~10.4ms and must fail this "
        "cell); the paragraph scan regressed"
    )


def test_chunk_by_lines_absolute_band_holds() -> None:
    """``chunk_by_lines``' cell, same absolute-band shape as
    ``chunk_by_paragraphs``' (no stdlib comparator, one fused linear
    scan): measured ~0.6ms at 12 MiB of prose (min-of-3 after warmup,
    the helper's own sample count) against the current byte-level
    scanner; the pre-fast-path per-char spelling measured ~13.6ms at
    the same corpus on the box that calibrated this cell (a CRLF-dense
    log ran 12.8ms there). Ceiling 10ms (~16x margin over the current
    measurement), positioned so that a regression back to the per-char
    decode loop fails it: the old spelling's ~13.6ms blows straight
    through 10ms,
    where the former 40ms ceiling would have absorbed it silently. The
    scan and the real-line filter are one pass with O(1) memory beyond
    the output, so only a class regression (a per-line allocation, a
    lost fusion, a lost fast path) reaches the ceiling."""
    corpus = prose(12 * _MIB)
    took_ms = _min_wall_ms(lambda s: chunk_by_lines(s, 200), corpus)
    assert took_ms < 10.0, (
        f"chunk_by_lines 12MiB took {took_ms:.1f}ms, outside the absolute band "
        "(measured ~0.6ms at 12 MiB, ceiling 10ms; the pre-fast-path per-char "
        "spelling measured ~13.6ms and must fail this cell); the "
        "line scan regressed"
    )


# --- The random-generation family ---------------------------------------------
#
# Microsecond-scale cells: unlike the module's multi-ms corpora cells, both
# sides here are syscall-plus-format calls measured in single-digit
# microseconds, so the draws are min-of-25 (both sides get plenty of windows
# to find an uncontended run) and the margins are regression nets over a
# measured win-or-parity, not close races. Measured on the dev box (Apple
# Silicon, quiet, min-of-25 after warm-up):
#
#     n         tors.random_hex   secrets.token_hex   ratio
#     64 B      1.5us             1.5us               1.00 (parity: the call
#                                                          overhead is the
#                                                          whole cost)
#     1 KiB     4.4us             4.8us               0.92
#     64 KiB    266us             296us               0.90
#
#     n         tors.random_b64url secrets.token_urlsafe
#     64 B      1.2us             1.3us               0.92
#     1 KiB     4.6us             5.9us               0.78
#     64 KiB    278us             334us               0.83
#
#     tors.uuid4 1.1us   uuid.uuid4 1.3us             0.85
#     tors.uuid7 0.9us   (no CI-safe comparator; absolute band below)
#
# The margin story, honestly: tors never loses a leg (the fused
# one-syscall-plus-format pass beats the stdlib's urandom-object-then-format
# chain at every size, parity at the overhead floor), but the wins are
# modest at these sizes because both sides are one OS syscall plus SIMD-ish
# C formatting -- the value proposition measured elsewhere in this module is
# the GIL release (tests/test_gil_release.py) and the seeded determinism
# (tests/test_random.py), not a wall blowout. The 1.5x margins therefore
# catch class regressions (an extra syscall per call would put ~+1.2us on a
# ~1.5us floor, ~2x, and a per-char draw regression on the sampler would
# add tens of microseconds at the 64 KiB legs), not fine tunings.


def _min_wall_us_fn(op: Callable[[], object], samples: int = 25, warmup: int = 3) -> float:
    """Min-of-``samples`` wall in MICROSECONDS for a no-argument call: the
    random family's calls take no corpus argument (they generate their own
    bytes), so this is the family's local spelling of the module's
    ``_min_wall_ms`` shape (the ``test_grounded_performance.py``
    microsecond-cell precedent). 25 samples: a microsecond-scale sample is
    one scheduler hit away from its worst run, so both sides of a race need
    many more draws than the millisecond cells' 3-7 to find their floor."""
    for _ in range(warmup):
        op()
    best = float("inf")
    for _ in range(samples):
        started = time.monotonic()
        op()
        best = min(best, time.monotonic() - started)
    return best * 1e6


@pytest.mark.parametrize("n_bytes", [64, 1024, 64 * 1024], ids=["64B", "1KiB", "64KiB"])
def test_random_hex_keeps_parity_or_better_with_secrets_token_hex(n_bytes: int) -> None:
    """``secrets.token_hex(n)`` is the exact expression ``random_hex``
    replaces (same 2n lowercase-hex format, same OS entropy source); the
    cell is the module's shared race shape at microsecond scale: 1.5x over
    a measured 0.90-1.00 (the table in the family's section header)."""
    tors_us = _min_wall_us_fn(lambda: tors.random_hex(n_bytes))
    stdlib_us = _min_wall_us_fn(lambda: secrets.token_hex(n_bytes))
    assert tors_us < 1.5 * stdlib_us, (
        f"random_hex({n_bytes}): tors {tors_us:.1f}us vs secrets.token_hex "
        f"{stdlib_us:.1f}us ({tors_us / stdlib_us:.2f}x): the native pass lost "
        "the syscall-plus-format race by more than the regression margin"
    )


@pytest.mark.parametrize("n_bytes", [64, 1024, 64 * 1024], ids=["64B", "1KiB", "64KiB"])
def test_random_b64url_keeps_parity_or_better_with_secrets_token_urlsafe(n_bytes: int) -> None:
    """``secrets.token_urlsafe(n)`` is the unpadded urlsafe expression
    ``random_b64url(n)`` replaces; measured 0.78-0.92 across the ladder
    (the family section's table), same 1.5x regression margin."""
    tors_us = _min_wall_us_fn(lambda: tors.random_b64url(n_bytes))
    stdlib_us = _min_wall_us_fn(lambda: secrets.token_urlsafe(n_bytes))
    assert tors_us < 1.5 * stdlib_us, (
        f"random_b64url({n_bytes}): tors {tors_us:.1f}us vs "
        f"secrets.token_urlsafe {stdlib_us:.1f}us "
        f"({tors_us / stdlib_us:.2f}x): the native pass lost the "
        "syscall-plus-format race by more than the regression margin"
    )


def test_uuid4_keeps_parity_or_better_with_the_stdlib_constructor() -> None:
    """``uuid.uuid4()`` is the spelling ``tors.uuid4()`` replaces; measured
    1.1us vs 1.3us (the stdlib builds a Python object and formats it through
    the uuid module's own machinery). 2.0x margin: the stdlib floor wobbles
    more than the byte codecs' (object construction), and the cell's teeth
    are class regressions (a per-call engine rebuild, an extra syscall)."""
    tors_us = _min_wall_us_fn(tors.uuid4)
    stdlib_us = _min_wall_us_fn(stdlib_uuid.uuid4)
    assert tors_us < 2.0 * stdlib_us, (
        f"tors.uuid4: {tors_us:.1f}us vs uuid.uuid4 {stdlib_us:.1f}us "
        f"({tors_us / stdlib_us:.2f}x): the native pass lost more than the "
        "regression margin to the stdlib constructor"
    )


def test_the_no_stdlib_comparator_generators_absolute_bands_hold() -> None:
    """``random_b62``/``random_string`` (no stdlib spelling exists for
    base62/alphabetic sampling) and ``uuid7`` (no CI-safe comparator: the
    stdlib has no v7, and uuid_utils is not a dependency this suite may
    assume) take absolute ceilings over measured floors, the
    ``chunk_by_paragraphs`` no-comparator shape at microsecond scale:

    - ``random_b62(22)`` measured ~4.2us (one 1024-byte block fill for the
      whole id, 22 Lemire draws). Ceiling 25us (~6x): the positioned
      regression is a lost block buffer -- a per-draw syscall spelling
      costs ~22 x 1.2us = ~26us of syscalls alone and trips it.
    - ``random_string(22, "ab")`` measured ~4.1us, same engine, same
      ceiling class (30us, a hair wider: the multibyte-capable push path).
    - ``uuid7()`` measured ~0.9us. Ceiling 25us (~28x): catches a per-call
      engine-class regression (a rebuilt ChaCha, a second syscall), not
      tunings."""
    b62_us = _min_wall_us_fn(lambda: tors.random_b62(22))
    assert b62_us < 25.0, (
        f"random_b62(22) took {b62_us:.1f}us, outside the absolute band "
        "(measured ~4.2us, ceiling 25us; a per-draw-syscall spelling measures "
        "~26us+ and must fail this cell); the block-buffered sampler regressed"
    )
    string_us = _min_wall_us_fn(lambda: tors.random_string(22, "ab"))
    assert string_us < 30.0, (
        f"random_string(22, 'ab') took {string_us:.1f}us, outside the absolute "
        "band (measured ~4.1us, ceiling 30us); the block-buffered sampler "
        "regressed"
    )
    uuid7_us = _min_wall_us_fn(tors.uuid7)
    assert uuid7_us < 25.0, (
        f"uuid7() took {uuid7_us:.1f}us, outside the absolute band (measured "
        "~0.9us, ceiling 25us); the one-syscall-plus-format pass regressed"
    )


# The minhash race's tolerance margin: measured tors/oracle ratios
# 0.005-0.007 across the ladder (the module docstring's table), so 0.5
# leaves ~80x headroom while still failing a lost-native-advantage
# regression (the native pass within 2x of the pure-Python bigint loop).
# The _GCM_WALL_MARGIN precedent: the assertion pins the relationship,
# not a close race.
_MINHASH_WALL_MARGIN = 0.5


@pytest.mark.parametrize(
    "size_bytes", [1 * 1024, 100 * 1024, 1 * _MIB], ids=["1KiB", "100KiB", "1MiB"]
)
def test_minhash_signature_beats_the_pure_python_oracle_on_the_doc_ladder(
    size_bytes: int,
) -> None:
    """``minhash_signature`` vs its transcribed pure-Python oracle over a
    document-size ladder at the default ``num_perm=128`` (prose corpus).
    The oracle is slow by construction -- one Python-level bigint affine
    per (shingle, permutation) pair, O(shingles x 128) interpreted
    iterations -- which is the point: it is the expression a caller
    without tors would run, and the measured ~160-175x gap is the native
    pass's wall-time case. The oracle draws a single sample (its wall is
    deterministic work at 3ms-2.8s across the ladder, and noise only
    ever adds time, making the single sample conservative for the
    denominator); tors draws the size-appropriate min-of-N (7 under
    4 MiB). The 0.5 margin is the criterion, not a calibration: it fails
    a regression that brings the native pass within 2x of pure Python
    while no realistic load asymmetry can flake it at ~37x headroom."""
    corpus = prose(size_bytes)
    tors_ms = _min_wall_ms(tors.minhash_signature, corpus, samples=_samples_for(size_bytes))
    started = time.perf_counter()
    reference_minhash_signature(corpus)
    oracle_ms = (time.perf_counter() - started) * 1000.0
    assert tors_ms < _MINHASH_WALL_MARGIN * oracle_ms, (
        f"prose {size_bytes // 1024}KiB: tors {tors_ms:.2f}ms vs oracle "
        f"{oracle_ms:.1f}ms (ratio {tors_ms / oracle_ms:.4f}): the native "
        "tokenize+shingle+hash+sweep pass lost more than the margin to the "
        "pure-Python MinHash loop"
    )


def _stdlib_content_hash(obj: object) -> str:
    """The stdlib expression ``tors.content_hash`` replaces: the exact
    canonical-form spelling the contract defines, hashed with hashlib."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    "size_bytes", [64 * 1024, 1 * _MIB, 12 * _MIB], ids=["64KiB", "1MiB", "12MiB"]
)
def test_content_hash_wall_time_vs_the_stdlib_is_measured_not_asserted(
    size_bytes: int,
) -> None:
    """``tors.content_hash`` vs the full stdlib spelling over the records
    corpus (``reference.content_object``), measured and deliberately not
    asserted: a structural dead heat, because the two sides do equivalent
    work. CPython's C encoder builds the whole canonical string in one
    GIL-held pass (fast: no Python-level per-value calls for str/int, the
    same storage reads tors's walk makes), then pays ``str.encode`` (a
    second full-size GIL-held copy) and a released-GIL ``sha256``; tors
    pays the GIL-held walk (borrow+copy per str into the owned tree, i64
    reads, one ``repr`` call per float) and a detached emit+hash. Measured
    on the dev box (min-of-7 below 4 MiB, min-of-3 above, after warmup):

        size     tors        stdlib     tors/stdlib
        64 KiB   0.20ms      0.21ms     0.97
        1 MiB    3.38ms      3.37ms     1.00
        12 MiB   41.87ms     41.79ms    1.00

    Every cell a dead heat (0.97-1.00): no wall win to assert, and none
    pretended at -- the value is the GIL release (the stdlib holds the
    loop for ``json.dumps`` + ``str.encode``, ~the whole wall, inline
    ratio 1.00-1.02 vs tors's 0.45-0.60, pinned in tests/test_gil_release.
    py), the byte-exact parity contract (tests/test_content_hash.py, whose
    differential this cell re-asserts at each measured size), and the
    detached half of the call. The same dead-heat precedent as
    ``decode_utf8`` and ``b64_decode``: recorded, not thresholded away.
    """
    obj = content_object(size_bytes)
    assert tors.content_hash(obj) == _stdlib_content_hash(obj)  # parity at the measured size
    samples = _samples_for(size_bytes)
    tors_ms = _min_wall_ms(tors.content_hash, obj, samples=samples)
    std_ms = _min_wall_ms(_stdlib_content_hash, obj, samples=samples)
    print(
        f"content_hash {size_bytes // 1024}KiB: tors {tors_ms:.2f}ms "
        f"stdlib {std_ms:.2f}ms ratio {tors_ms / std_ms:.2f}"
    )


# --- The batch-charset validator wall race ---------------------------------------
#
# The batch-only design's premise, raced against the expression it replaces:
# the per-item anchored-regex loop an enqueue path spells around identifier
# validators (TaskQ's _IDENT_RE shape).

# The identifier rule's two halves (letters and underscore at position 0,
# digits joining after) and the equivalent anchored regex, rebuilt from the
# same halves.
_IDENT_FIRST = string.ascii_letters + "_"
_IDENT_REST = string.ascii_letters + string.digits + "_"
_IDENT_RE = re.compile(rf"\A[{_IDENT_FIRST}][{_IDENT_REST}]*\Z")


def _ident_items(count: int) -> list[str]:
    """``count`` deterministic identifier-shaped items (the job/queue/worker/
    tag spellings an enqueue path validates): the all-valid batch, the shape
    where both sides of the race do full work with no short-circuit."""
    shapes = ("job_{n}", "queue_eu_{n}", "worker_{n}", "tag_{n}")
    return [shapes[n % 4].format(n=n) for n in range(count)]


@pytest.mark.parametrize("count", [100, 1000], ids=["100-items", "1000-items"])
def test_first_invalid_charset_beats_the_per_item_regex_loop_on_the_same_batch(
    count: int,
) -> None:
    """The whole-batch tors call vs ``all(_IDENT_RE.match(i) for i in
    items)`` over the same all-valid identifier batch, min-of-7 per side
    after warmup (the fast-cell discipline: the native samples are
    µs-scale) with the suite's 0.9 margin.

    Measured on the dev box (ambient load 5.5, min-of-7): 100 items tors
    2.0 µs vs the regex loop's 9.7 µs (ratio 0.20); 1000 items 14.0 µs
    vs 96.7 µs (0.14) — ~5-7x inside the margin. The honest other half,
    recorded in docs/api.md's section and deliberately not asserted here
    because it is a LOSS: a per-item tors call (one item per call)
    measures ~0.25 µs against ~0.08 µs for one regex match — the detach
    round trip paid per call — which is exactly why the API is
    batch-only: the win exists only when one call covers the batch, and
    the crossover is already at single-digit item counts (measured ~0.5
    at a 10-item batch)."""
    items = _ident_items(count)
    tors_ms = _min_wall_ms(
        lambda its: tors.first_invalid_charset(its, first=_IDENT_FIRST, rest=_IDENT_REST),
        items,
        samples=_FAST_CELL_SAMPLES,
    )
    re_ms = _min_wall_ms(
        lambda its: all(_IDENT_RE.match(item) for item in its), items, samples=_FAST_CELL_SAMPLES
    )
    assert tors_ms < _MARGIN * re_ms, (
        f"{count}-item identifier batch: tors {tors_ms * 1000:.1f}µs vs the "
        f"per-item regex loop {re_ms * 1000:.1f}µs (ratio {tors_ms / re_ms:.2f}): the "
        "batch call lost more than the tolerance margin to the regex loop it replaces"
    )
