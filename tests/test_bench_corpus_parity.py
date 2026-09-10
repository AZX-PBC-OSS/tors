"""The bench↔test corpus identity, enforced. The performance tables in docs/performance.md (GIL
release, wall time vs the reference/stdlib, criterion throughput) cross-reference
one another on the claim that the criterion benches' Rust-built corpora and the
Python suites' ``tests/reference.py``-built corpora are the same bytes, and
nothing else pinned that, so a sentence, unit-assembly, or quantization change on
either side would leave every table internally consistent while the cross-table
comparisons quietly turned apples-to-oranges. These tests parse the bench sources
themselves (no copied constants), decode their Rust string literals, and assert
byte-equality of both the sentence constants and the built corpora against
``reference``, which stays the single source.

The shared recipes (``prose`` / ``decomposed`` / ``crlf`` and their sentence
constants) live once in ``benches/common/mod.rs`` (criterion's shared-module
pattern: every bench file declares ``mod common;``), so the cross-check parses
that one source for them, and a separate wiring pin below requires every bench to
build its corpora from the common module (a bench silently re-growing its own
copy cannot drift unnoticed). The text bench's own recipes (``compat`` /
``entities``, used by no other bench) stay local to ``benches/text.rs`` and are
pinned there. The bytes bench additionally has its UTF-8 rendering pinned:
its measured corpora are the reference corpora encoded to UTF-8, exactly
``reference.corpus_utf8``, the same bytes the Python-side GIL/wall cells use. The
text bench's b64-decode corpus rendering is pinned: the prose corpus through
the same standard engine ``b64_impl::encode`` uses, the same bytes as
``reference.corpus_b64``.

The diff bench's pair corpora are built (mutated/shuffled), not repeated,
so they cannot be rebuilt-in-Python from parsed constants the way the
``repeat_to`` recipes can; the parity mechanism there is three-layered: the
shared constants (word swap, edit-position fractions, the LCG, the numbering
width) are parsed out of ``benches/diff.rs`` and cross-checked against
``tests/reference.py``'s mirror constants, the load-bearing statements (the
``split('\\n')`` line semantics (Rust ``str::lines()`` would drop the trailing
empty lines and diverge from the Python twin's ``split("\\n")``); the
Fisher-Yates loop; the numbering format; the near-identical edit branches) are
pinned textually, and the wiring pin requires the bench to build from the
common prose recipe. The Rust builders were verified byte-identical to the
reference builders by SHA-256 over both shapes at 256 KiB / 1 MiB / 12 MiB
when introduced; the pins exist so that verified state cannot silently drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reference import (  # noqa: I001 -- the shared oracle module (tests/reference.py)
    _COMPAT_SENTENCE,
    _DECOMPOSED_SENTENCE,
    _DIFF_DELETE_FRACTION,
    _DIFF_INSERT_FRACTION,
    _DIFF_LCG_INC,
    _DIFF_LCG_MUL,
    _DIFF_LCG_SEED,
    _DIFF_LINE_NUMBER_WIDTH,
    _DIFF_REPLACE_FRACTIONS,
    _DIFF_WORD_SWAP,
    _ENTITY_SENTENCE,
    _PROSE_SENTENCE,
    SEARCH_DENSE_PATTERNS,
    SEARCH_SPARSE_PATTERNS,
    corpus_utf8,
)

_MIB = 1024 * 1024
_BENCHES_DIR = Path(__file__).resolve().parent.parent / "benches"
_BENCH_SOURCES: dict[str, str] = {
    # The one Rust-side source for the shared corpus recipes (every bench does
    # `mod common;`), plus the text bench, the only one with recipes of its own.
    "common/mod.rs": (_BENCHES_DIR / "common" / "mod.rs").read_text(encoding="utf-8"),
    "text.rs": (_BENCHES_DIR / "text.rs").read_text(encoding="utf-8"),
    # The diff bench: pair corpora built from the shared prose recipe (see the
    # module docstring for the built-corpus parity mechanism).
    "diff.rs": (_BENCHES_DIR / "diff.rs").read_text(encoding="utf-8"),
}

# Each pinned source's expected sentence constants and corpus kinds: the shared
# three in the common module; the text-bench-only two (the compat corpus that
# still pays the K-forms' full pass under the quick-check fast paths, and
# the entity-bearing prose corpus) in text.rs.
_EXPECTED_SENTENCES: dict[str, dict[str, str]] = {
    "common/mod.rs": {
        "PROSE_SENTENCE": _PROSE_SENTENCE,
        "DECOMPOSED_SENTENCE": _DECOMPOSED_SENTENCE,
    },
    "text.rs": {
        "COMPAT_SENTENCE": _COMPAT_SENTENCE,
        "ENTITY_SENTENCE": _ENTITY_SENTENCE,
    },
}
_EXPECTED_KINDS: dict[str, list[str]] = {
    "common/mod.rs": ["prose", "decomposed", "crlf"],
    "text.rs": ["compat", "entities"],
}

# The bench's quantization, pinned textually: byte-length division (Rust ``str::len()`` is
# the UTF-8 byte length) with a 1-unit floor. Any change to this shape must be mirrored in
# reference.py's ``_repeat_to``; the corpus test below verifies that it was.
_REPEAT_TO = re.compile(
    r"fn repeat_to\(target_bytes: usize, unit: &str\) -> String \{\s*"
    r"unit\.repeat\(\(target_bytes / unit\.len\(\)\)\.max\(1\)\)\s*\}"
)
_SENTENCE = re.compile(r'const (?P<name>\w+): &str = "(?P<literal>[^"]*)";')
# A corpus builder: repeat_to(target, &format!("{}{}", sentence.repeat(n), "suffix")).
_BUILDER = re.compile(
    r"fn (?P<kind>\w+)\(target_bytes: usize\) -> String \{\s*"
    r'repeat_to\(\s*target_bytes,\s*&format!\("\{\}\{\}",\s*(?P<unit>\w+)\.repeat\((?P<count>\d+)\),'
    r'\s*"(?P<suffix>[^"]*)"\),?\s*\)\s*\}'
)
# A bench file wired to the shared module (the wiring pin: no bench may build
# the shared corpora from its own copy).
_USES_COMMON = re.compile(r"^mod common;$", re.MULTILINE)
# The bytes bench's rendering, pinned textually: it measures the UTF-8 bytes of the
# built str corpus (the same rendering as reference.corpus_utf8), not some other
# byte view of it.
_BYTES_RENDER = re.compile(r"let bytes = corpus\.as_bytes\(\);")
# The text bench's b64-decode corpus rendering, pinned textually: the prose
# corpus encoded through the same standard engine b64_impl::encode uses,
# exactly reference.corpus_b64's bytes.
_B64_RENDER = re.compile(
    r"let b64 = base64::engine::general_purpose::STANDARD\.encode\(corpus\.as_bytes\(\)\);"
)


def _decode_rust_literal(literal: str) -> str:
    """Decode the Rust escape subset these literals use: ``\\u{…}`` plus ``\\t``, ``\\r``,
    ``\\n``."""
    decoded = re.sub(r"\\u\{([0-9a-fA-F]+)\}", lambda m: chr(int(m.group(1), 16)), literal)
    return decoded.replace("\\t", "\t").replace("\\r", "\r").replace("\\n", "\n")


def _bench_facts(source: str) -> tuple[dict[str, str], dict[str, str]]:
    """A source's sentence constants (decoded) and, per corpus kind, the unit it
    repeats: the sentence ``.repeat(n)`` plus the builder's literal suffix."""
    sentences = {m["name"]: _decode_rust_literal(m["literal"]) for m in _SENTENCE.finditer(source)}
    units: dict[str, str] = {}
    for m in _BUILDER.finditer(source):
        assert m["unit"] in sentences, f"bench builder {m['kind']!r} lost its sentence const"
        units[m["kind"]] = sentences[m["unit"]] * int(m["count"]) + _decode_rust_literal(
            m["suffix"]
        )
    return sentences, units


def _rust_repeat_to(target_bytes: int, unit: str) -> str:
    """The bench's quantization as Python: UTF-8 byte-length floor division with a 1-unit
    floor, the exact shape ``_REPEAT_TO`` pins in the bench source."""
    return unit * max(1, target_bytes // len(unit.encode("utf-8")))


@pytest.mark.parametrize("source", ["common/mod.rs", "text.rs"])
def test_bench_sentences_are_byte_identical_to_the_reference_sentences(
    source: str,
) -> None:
    sentences, _ = _bench_facts(_BENCH_SOURCES[source])
    expected = _EXPECTED_SENTENCES[source]
    assert {name: text.encode("utf-8") for name, text in sentences.items()} == {
        name: text.encode("utf-8") for name, text in expected.items()
    }, (
        f"benches/{source}'s and reference.py's sentence constants drifted; every "
        "performance table cross-references on these being the same bytes"
    )


@pytest.mark.parametrize("source", ["common/mod.rs", "text.rs"])
def test_bench_quantization_semantics_are_pinned(source: str) -> None:
    assert _REPEAT_TO.search(_BENCH_SOURCES[source]), (
        f"benches/{source}'s repeat_to no longer matches the pinned quantization "
        "(usize byte-length division with a 1-unit floor); mirror any change in "
        "tests/reference.py's _repeat_to"
    )


def test_every_bench_builds_its_corpora_from_the_common_module() -> None:
    """The wiring pin: each bench file must declare the shared module, so the
    triplicated corpus recipes cannot quietly re-grow inside a bench file;
    the corpus identity above is only meaningful while the benches actually
    build from ``benches/common/mod.rs``."""
    for name in ("normalize.rs", "bytes.rs", "text.rs", "utf8.rs", "diff.rs", "search.rs"):
        source = (_BENCHES_DIR / name).read_text(encoding="utf-8")
        assert _USES_COMMON.search(source), (
            f"benches/{name} no longer declares `mod common;`: its corpora are "
            "no longer the shared, corpus-test-pinned recipes"
        )


def test_bytes_bench_measures_the_utf8_rendering_of_the_built_corpus() -> None:
    """The bytes bench's corpora must be the UTF-8 bytes of the (pinned-identical) str
    corpora (``corpus.as_bytes()``), so its numbers are comparable with the
    Python-side bytes cells (``reference.corpus_utf8``) and the normalize bench."""
    source = (_BENCHES_DIR / "bytes.rs").read_text(encoding="utf-8")
    assert _BYTES_RENDER.search(source), (
        "benches/bytes.rs no longer renders its corpora via corpus.as_bytes(): the "
        "bytes-bench numbers would no longer be the UTF-8 corpora the tables "
        "cross-reference"
    )


def test_text_bench_measures_the_b64_rendering_of_the_prose_corpus() -> None:
    """The text bench's b64-decode corpus must be the (pinned-identical) prose
    corpus rendered through the same standard engine ``b64_impl::encode``
    uses, the exact bytes of ``reference.corpus_b64('prose', ...)``, which
    the Python-side b64_decode GIL/wall cells measure, so the bench numbers
    stay comparable with them."""
    assert _B64_RENDER.search(_BENCH_SOURCES["text.rs"]), (
        "benches/text.rs no longer renders its b64 corpus via the STANDARD "
        "engine over corpus.as_bytes(): the b64-decode bench numbers would "
        "no longer be the corpora the Python-side cells measure"
    )


@pytest.mark.parametrize("target_bytes", [1, 1024, 12 * _MIB], ids=["1B-floor", "1KiB", "12MiB"])
@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("common/mod.rs", "prose"),
        ("common/mod.rs", "decomposed"),
        ("common/mod.rs", "crlf"),
        ("text.rs", "compat"),
        ("text.rs", "entities"),
    ],
    ids=[
        "common-prose",
        "common-decomposed",
        "common-crlf",
        "text-compat",
        "text-entities",
    ],
)
def test_bench_corpus_is_byte_identical_to_the_reference_corpus(
    source: str, kind: str, target_bytes: int
) -> None:
    """The 1B cell exercises the ``.max(1)`` floor (a target smaller than one unit); 1 KiB
    and 12 MiB are the sizes the performance and criterion tables quote (docs/performance.md). The
    shared kinds are asserted against ``benches/common/mod.rs`` (the one source every
    bench builds them from); the text-bench-only kinds against ``benches/text.rs``
    (where those recipes live)."""
    _, units = _bench_facts(_BENCH_SOURCES[source])
    assert kind in units, f"benches/{source} lost its {kind!r} builder"
    bench_corpus = _rust_repeat_to(target_bytes, units[kind])
    assert bench_corpus.encode("utf-8") == corpus_utf8(kind, target_bytes), (
        f"{source} {kind} at a {target_bytes}-byte target: the bench's and "
        "reference.py's corpora are not the same bytes"
    )


# --- The diff bench's built pair corpora --------------------------------------
#
# The diff corpora are built (line mutations, a shuffle), not repeated, so the
# repeat_to rebuilding mechanism above cannot reach them. The mechanism here:
# parse benches/diff.rs's shared constants and cross-check them against
# reference.py's mirror constants, plus textual pins of the load-bearing
# statements (see the module docstring for the layering and the SHA-256
# verification provenance).

_DIFF_STRING_CONST = re.compile(r'const (?P<name>\w+): &str = "(?P<literal>[^"]*)";')
_DIFF_FRACTION_ARRAY = re.compile(
    r"const (?P<name>\w+): \[\(usize, usize\); \d+\] = (?P<pairs>\[[^;]*\]);"
)
_DIFF_FRACTION_PAIR = re.compile(
    r"const (?P<name>\w+): \(usize, usize\) = \((?P<num>\d+), (?P<den>\d+)\);"
)
_DIFF_LCG_CONST = re.compile(r"const (?P<name>\w+): u64 = (?P<value>0x[0-9A-Fa-f_]+|\d[\d_]*);")
_DIFF_WIDTH_CONST = re.compile(r"const LINE_NUMBER_WIDTH: usize = (?P<width>\d+);")
_DIFF_FRACTION_PAIRS = re.compile(r"\((\d+), (\d+)\)")


def test_diff_bench_word_swap_and_numbering_constants_match_reference() -> None:
    """The word-swap pair (what the near-identical pair's replacement lines and
    inserted line carry) and the numbering width (what makes the shuffled
    pair's lines distinct) must be the same values on both sides."""
    source = _BENCH_SOURCES["diff.rs"]
    strings = {m["name"]: m["literal"] for m in _DIFF_STRING_CONST.finditer(source)}
    assert {
        "WORD_SWAP_OLD": strings["WORD_SWAP_OLD"],
        "WORD_SWAP_NEW": strings["WORD_SWAP_NEW"],
    } == {
        "WORD_SWAP_OLD": _DIFF_WORD_SWAP[0],
        "WORD_SWAP_NEW": _DIFF_WORD_SWAP[1],
    }, "benches/diff.rs's word-swap pair drifted from reference.py's _DIFF_WORD_SWAP"
    width = _DIFF_WIDTH_CONST.search(source)
    assert width is not None, "benches/diff.rs lost its LINE_NUMBER_WIDTH constant"
    assert int(width["width"]) == _DIFF_LINE_NUMBER_WIDTH, (
        "benches/diff.rs's line-number width drifted from reference.py's "
        "_DIFF_LINE_NUMBER_WIDTH: the shuffled corpora are no longer the same bytes"
    )


def test_diff_bench_edit_position_fractions_match_reference() -> None:
    """The six edit positions (four replacements, one delete, one insert) as
    (numerator, denominator) line fractions; integer math both sides compute
    identically only while the fractions themselves agree."""
    source = _BENCH_SOURCES["diff.rs"]
    arrays = {
        m["name"]: [tuple(map(int, p)) for p in _DIFF_FRACTION_PAIRS.findall(m["pairs"])]
        for m in _DIFF_FRACTION_ARRAY.finditer(source)
    }
    pairs = {
        m["name"]: (int(m["num"]), int(m["den"])) for m in _DIFF_FRACTION_PAIR.finditer(source)
    }
    assert arrays["REPLACE_FRACTIONS"] == list(_DIFF_REPLACE_FRACTIONS), (
        "benches/diff.rs's replacement fractions drifted from reference.py's "
        "_DIFF_REPLACE_FRACTIONS"
    )
    assert pairs["DELETE_FRACTION"] == _DIFF_DELETE_FRACTION, (
        "benches/diff.rs's delete fraction drifted from reference.py's _DIFF_DELETE_FRACTION"
    )
    assert pairs["INSERT_FRACTION"] == _DIFF_INSERT_FRACTION, (
        "benches/diff.rs's insert fraction drifted from reference.py's _DIFF_INSERT_FRACTION"
    )


def test_diff_bench_lcg_constants_match_reference() -> None:
    """The shuffle's LCG seed/multiplier/increment: the permutation is a
    function of all three plus the line count, so any drift re-shuffles the
    corpus entirely."""
    source = _BENCH_SOURCES["diff.rs"]

    def parse(value: str) -> int:
        digits = value.removeprefix("0x").replace("_", "")
        return int(digits, 16 if value.startswith("0x") else 10)

    consts = {m["name"]: parse(m["value"]) for m in _DIFF_LCG_CONST.finditer(source)}
    assert consts["LCG_SEED"] == _DIFF_LCG_SEED, "LCG seed drifted between bench and reference"
    assert consts["LCG_MUL"] == _DIFF_LCG_MUL, "LCG multiplier drifted between bench and reference"
    assert consts["LCG_INC"] == _DIFF_LCG_INC, "LCG increment drifted between bench and reference"


def test_diff_bench_builds_lines_with_python_split_semantics() -> None:
    """Rust ``str::lines()`` drops the final trailing-empty line elements that
    Python ``split("\\n")`` keeps, a one-line count difference that re-runs
    the size-dependent shuffle into a different corpus entirely. Both pair
    builders must split exactly the way the Python twin does."""
    source = _BENCH_SOURCES["diff.rs"]
    assert source.count("split('\\n')") >= 2, (
        "benches/diff.rs no longer builds its lines via split('\\n') in both "
        "builders: the line count (and with it the whole shuffle) would diverge "
        'from reference.py\'s split("\\n") corpora'
    )
    assert ".lines()" not in source, (
        "benches/diff.rs uses str::lines(): its trailing-empty-line semantics "
        'diverge from the Python twin\'s split("\\n")'
    )


def test_diff_bench_shuffle_loop_and_edit_branches_are_pinned() -> None:
    """The load-bearing statements, pinned textually (the _B64_RENDER
    mechanism): the Fisher-Yates loop (the exact u64 wrapping arithmetic and
    the modulo draw), the numbered-line format, and the near-identical pair's
    edit branches (replace wins over a colliding delete; the inserted line
    lands after the kept line), the semantics the constant cross-checks
    above cannot see on their own."""
    source = _BENCH_SOURCES["diff.rs"]
    for pin in (
        "for i in (1..lines.len()).rev() {",
        "state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);",
        "let j = (state % (i as u64 + 1)) as usize;",
        "lines.swap(i, j);",
        'format!("{idx:0w$} {line}", w = LINE_NUMBER_WIDTH)',
        "if replace_at.contains(&idx) {",
        "} else if idx == delete_at {",
        "if idx == insert_at {",
        "out.push(line.replace(WORD_SWAP_OLD, WORD_SWAP_NEW));",
        "out.push(PROSE_SENTENCE.replace(WORD_SWAP_OLD, WORD_SWAP_NEW));",
    ):
        assert pin in source, (
            f"benches/diff.rs no longer contains the pinned statement {pin!r}: "
            "its pair corpora are no longer built by the recipe the Python-side "
            "cells measure"
        )


# --- The search bench's pattern sets --------------------------------------------
#
# The search bench's corpora are the shared prose recipe (the wiring pin
# above), so what can drift is the pattern sets: the two arrays the cells
# scan with. They are parsed out of benches/search.rs and cross-checked
# against reference.py's SEARCH_SPARSE_PATTERNS / SEARCH_DENSE_PATTERNS
# mirrors, so the bench numbers and the Python-side GIL/wall cell numbers
# (which drive find_patterns with the reference tuples) cross-reference on
# the same patterns.

_SEARCH_PATTERN_ARRAY = re.compile(r"const (?P<name>\w+): \[&str; \d+\] = \[(?P<items>[^;]*)\];")
_SEARCH_PATTERN_ITEM = re.compile(r'"(?P<word>[^"]*)"')


def test_search_bench_pattern_sets_match_reference() -> None:
    """Both pattern arrays (the sparse terminology-scan set and the dense
    prose-word set) must be the same words, in the same order, on both
    sides; a drifted word (or count) would silently point the bench at a
    different workload than every Python-side cell measures."""
    source = (_BENCHES_DIR / "search.rs").read_text(encoding="utf-8")
    arrays = {
        m["name"]: [item["word"] for item in _SEARCH_PATTERN_ITEM.finditer(m["items"])]
        for m in _SEARCH_PATTERN_ARRAY.finditer(source)
    }
    assert arrays["SPARSE_PATTERNS"] == list(SEARCH_SPARSE_PATTERNS), (
        "benches/search.rs's sparse pattern set drifted from reference.py's SEARCH_SPARSE_PATTERNS"
    )
    assert arrays["DENSE_PATTERNS"] == list(SEARCH_DENSE_PATTERNS), (
        "benches/search.rs's dense pattern set drifted from reference.py's SEARCH_DENSE_PATTERNS"
    )
