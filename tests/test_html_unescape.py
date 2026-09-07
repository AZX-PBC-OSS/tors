"""Contract gate for ``tors.html_unescape``: indistinguishable from
``html.unescape`` over the FULL HTML5 named-entity table (all 2231 entries of
``html.entities.html5``, with- and without-semicolon spellings), plus every
numeric-reference class (decimal/hex, semicolon or not, the Windows-1252
remap, the invalid-codepoint-to-empty quirk, out-of-range to U+FFFD) and the
longest-prefix fallback with verbatim remainder.

Why full-table parity is the product: a caller swapping ``html.unescape`` for
tors must get identical text out for identical text in, on every entity the
HTML5 table defines, not just the common ones. The audit that gated this
function (spec's build gate) found exactly one real ``html.unescape``
call site across the originating ingestion service (unreleased) and its
sibling: unescaping VTT speaker-name tags
(tiny strings, correctness-critical: the unescaped name keys a
speaker-email lookup, so a miss is data corruption), and none anywhere else.
So this function ships as core-library surface with an explicit value
statement: its measured value is the GIL release (``html.unescape`` is a
regex-sub over the whole string with a PYTHON callback per entity, a
GIL-held whole-text pass) and throughput on large escaped documents, not a
currently-measured pain site in those codebases.

Crate decision, with evidence (the parity test as arbiter, pre-implementation,
over this 4505-case oracle (every html5 key, a 52-case tricky battery, and a
~2200-case numeric sweep): ``htmlescape`` 0.3.6 fails 3609/4505 (incomplete
entity table (``UnknownEntity`` on ``&Abreve;``) and hard ERRORS on
without-semicolon refs, where Python decodes); ``html_escape`` 0.2 fails
1495/4505 (leaves legacy without-semicolon refs like ``&amp`` verbatim;
truncates multi-char values (``&acE;`` decodes to ``∾`` losing the combining
U+0333); follows WHATWG numeric semantics where Python's differ, e.g.
``&#1;`` → ``\\x01`` where Python's ``_invalid_codepoints`` maps it to the
EMPTY string). Neither achieves parity, so tors implements CPython's exact
algorithm (the ``_charref`` regex + ``_replace_charref`` classification,
Lib/html/__init__.py) over tables GENERATED from ``html.entities.html5`` /
``html._invalid_charrefs`` / ``html._invalid_codepoints``, pinned per CI leg
by the full-table iteration and the numeric-set sweeps below, which re-verify
every entry of all three tables against the RUNNING interpreter's own data.

Measured CPython 3.12.7 behaviors the port preserves, each pinned by name
below: ``&#10FFFF;`` → ``"\\nFFFF;"`` (the decimal class stops at the first
non-digit, so only ``10`` is consumed); ``&notit;`` → ``"¬it;"`` (longest
matching prefix ``not`` + verbatim remainder); ``&#38;amp;`` → ``"&amp;"``
(single-pass sub: replacements are never re-scanned); the 34-entry
Windows-1252 remap (``&#128;`` → ``€``); 126 invalid codepoints to the EMPTY
string (``&#1;``, ``&#xFFFE;``); surrogates and > U+10FFFF to U+FFFD;
``&#;`` / ``&#x;`` / ``&;`` / bare ``&`` left verbatim.
"""

from __future__ import annotations

import html
import html.entities
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from reference import pathological_text  # noqa: I001 -- the shared oracle module
from tors import html_unescape

# The tricky battery: measured on CPython 3.12.7 pre-implementation, and
# re-verified live against the running interpreter on every leg. Names follow
# the CPython behavior each row pins.
_TRICKY: list[tuple[str, str]] = [
    # named refs, with and without semicolon (the legacy spellings)
    ("&amp;", "&"),
    ("&amp", "&"),
    ("&AMP;", "&"),
    ("&AMP", "&"),  # the all-caps legacy keys exist too
    ("&Amp;", "&Amp;"),  # case-sensitive: no such key
    ("&lt;tag&gt;", "<tag>"),
    ("&fjlig;", "fj"),  # a compat-mapping entity (multi-char value)
    ("&acE;", "\u223e\u0333"),  # a TWO-char value: the truncation trap
    ("&CounterClockwiseContourIntegral;", "\u2233"),  # the longest key, 31 chars
    # longest-prefix fallback with verbatim remainder
    ("&notit;", "\u00acit;"),
    ("&notit", "\u00acit"),
    ("&ampx;", "&x;"),
    ("&amper;&amper;", "&er;&er;"),
    ("&am&amp", "&am&"),  # '&' ends the name chars: only 'am' is the group
    ("&there4;", "\u2234"),
    ("&there4", "&there4"),  # 'there4;' exists, 'there4' does not, no prefix hits
    # unknown names left verbatim, '&' and all
    ("&nope;", "&nope;"),
    ("&" + "a" * 40, "&" + "a" * 40),  # the 32-char name cap
    ("&" + "a" * 32 + ";", "&" + "a" * 32 + ";"),
    # numeric refs: decimal, hex, capital X, with and without semicolon
    ("&#65;", "A"),
    ("&#x41;", "A"),
    ("&#X41;", "A"),
    ("&#41", ")"),
    ("&#00000065;", "A"),  # leading zeros
    ("&#x00000041;", "A"),
    # the greedy-digit quirk: the decimal class stops at the first non-digit
    ("&#10FFFF;", "\nFFFF;"),
    ("&#0x41;", "\ufffdx41;"),  # parses '#0' -> NUL -> FFFD, remainder verbatim
    # classification: Windows-1252 remap, invalid codepoints, range guard
    ("&#128;", "\u20ac"),
    ("&#0;", "\ufffd"),
    ("&#x0;", "\ufffd"),
    ("&#1;", ""),  # _invalid_codepoints -> EMPTY string (126 members)
    ("&#xFFFE;", ""),
    ("&#13;", "\r"),  # _invalid_charrefs[0x0D]
    ("&#xD;", "\r"),
    ("&#xD800;", "\ufffd"),  # surrogates -> FFFD
    ("&#x110000;", "\ufffd"),  # > U+10FFFF -> FFFD
    ("&#10FFFF;", "\nFFFF;"),  # (as above; the decimal quirk, hex would differ)
    ("&#x10FFFF;", ""),  # the hex spelling DOES reach the codepoint, and it
    # is a noncharacter (last two of the plane), so _invalid_codepoints maps
    # it to the EMPTY string, not the max scalar
    # degenerate refs left verbatim
    ("&#;", "&#;"),
    ("&#x;", "&#x;"),
    ("&;", "&;"),
    ("&", "&"),
    ("&# 65;", "&# 65;"),  # space breaks the numeric classes
    ("& #65;", "& #65;"),
    # adjacent entities and the single-pass (no re-scan) guarantee
    ("&&amp;;", "&&;"),
    ("&amp&amp", "&&"),
    ("&#65&#66;", "AB"),
    ("a&amp;b", "a&b"),
    ("&#38;amp;Dangerous", "&amp;Dangerous"),  # the classic double-unescape guard
    (" &amp; ", " & "),  # whitespace preserved: no strip stage, unlike normalize
    # no '&' at all: the fast path
    ("plain text", "plain text"),
    ("", ""),
]


class TestFullHtml5Table:
    def test_every_html5_entity_ref_decodes_to_its_table_value(self) -> None:
        """THE headline pin: for every one of the 2231 keys of the RUNNING
        interpreter's ``html.entities.html5`` (2125 with-semicolon + 106
        without), ``tors.html_unescape('&' + key)`` must equal the table's own
        value AND the interpreter's ``html.unescape`` of the same string, so
        tors's generated table is verified entry-by-entry against each leg's
        own table (a Python version that grows or changes the table fails here
        loudly instead of silently diverging)."""
        html5 = html.entities.html5
        assert len(html5) > 2200  # the sweep must actually sweep (3.12: 2231)
        with_semi = sum(1 for k in html5 if k.endswith(";"))
        assert with_semi > 2100 and len(html5) - with_semi > 100  # 2125 + 106
        for key, value in html5.items():
            ref = "&" + key
            assert html_unescape(ref) == value, f"ref {ref!r}"
            assert html_unescape(ref) == html.unescape(ref), f"ref {ref!r}"

    def test_entities_embedded_in_plain_text_decode_in_place(self) -> None:
        """The same table, embedded: every entity ref surrounded by prose must
        decode exactly where it sits, with the surrounding text untouched."""
        for key in ("amp;", "amp", "lt;", "not", "acE;", "fjlig;"):
            ref = "&" + key
            embedded = "pre " + ref + " post"
            assert html_unescape(embedded) == html.unescape(embedded), embedded


class TestNumericSets:
    """The two NUMERIC classification sets, swept per CI leg exactly like the
    named-entity table above: the html_table.rs header used to overclaim
    per-leg re-verification (only the named-entity sweep ran), so a
    regenerated table with stale numeric data would have passed every test;
    these sweeps close that gap. Every member of both sets, in
    BOTH ``&#N;`` and ``&#xN;`` spellings, must agree with the RUNNING
    interpreter's ``html.unescape``, including the classification-order
    subtlety that a codepoint in BOTH sets (the 0x80-0x9F C1 range is) takes
    the ``_invalid_charrefs`` remap, which comparing against the interpreter's
    own output settles without re-deriving. The two attributes are CPython
    implementation details (private but stable since 3.8, the exact data
    ``html.unescape`` itself consults); if a future Python removes them the
    sweep skips LOUDLY and specifically instead of silently shrinking to
    nothing."""

    _SKIP_CHARREFS = (
        "html._invalid_charrefs has vanished from this interpreter: the "
        "numeric-set sweep cannot run, and the generated INVALID_CHARREFS table "
        "in src/html_table.rs is now unverified against the running interpreter "
        "(tests/test_html_unescape.py::TestNumericSets)"
    )
    _SKIP_CODEPOINTS = (
        "html._invalid_codepoints has vanished from this interpreter: the "
        "numeric-set sweep cannot run, and the generated INVALID_CODEPOINTS table "
        "in src/html_table.rs is now unverified against the running interpreter "
        "(tests/test_html_unescape.py::TestNumericSets)"
    )

    def test_every_invalid_charref_is_remapped_identically_in_both_spellings(self) -> None:
        charrefs = getattr(html, "_invalid_charrefs", None)
        if charrefs is None:
            pytest.skip(self._SKIP_CHARREFS)
        assert len(charrefs) > 30  # the sweep must actually sweep (3.12: 34)
        for cp in charrefs:
            for ref in (f"&#{cp};", f"&#x{cp:x};"):
                assert html_unescape(ref) == html.unescape(ref), f"ref {ref!r}"

    def test_every_invalid_codepoint_maps_identically_in_both_spellings(self) -> None:
        codepoints = getattr(html, "_invalid_codepoints", None)
        if codepoints is None:
            pytest.skip(self._SKIP_CODEPOINTS)
        assert len(codepoints) > 120  # the sweep must actually sweep (3.12: 126)
        for cp in codepoints:
            for ref in (f"&#{cp};", f"&#x{cp:x};"):
                assert html_unescape(ref) == html.unescape(ref), f"ref {ref!r}"


@pytest.mark.skipif(
    not hasattr(sys, "get_int_max_str_digits"),
    reason=(
        "CPython 3.10 has no integer string conversion limit at all "
        "(sys.get_int_max_str_digits arrived in 3.11): its html.unescape "
        "classifies numeric references of ANY length, tors matches that "
        "(no limit is read, nothing raises), so the limit-boundary pins "
        "cannot run on this interpreter; the difference is documented in "
        "the pyi and the README"
    ),
)
class TestIntegerParseLimit:
    """CPython's integer string conversion limit
    (``sys.get_int_max_str_digits()``, default 4300, present on every 3.11+
    AND on 3.10.7+ via the CVE-2020-10735 backport; only 3.10.0–3.10.6 lack
    it) applies to ``html.unescape``'s DECIMAL numeric refs:
    ``_replace_charref``'s ``int()`` raises ``ValueError("Exceeds the limit
    (…) for integer string conversion: value has 4301 digits; ...")`` BEFORE
    any classification, where tors previously classified the run to U+FFFD.
    The wording is VERSION-DEPENDENT ("the limit (4300 digits)" on
    3.12+/late-3.11, "the limit (4300)" on 3.10.7–3.11.x), so tors raises
    the running interpreter's OWN message (the error path replays ``int()``
    on the offending digit run) and these pins assert the DIFFERENTIAL
    (tors's exception equals the running stdlib's exception) with only the
    version-stable fragments spelled literally. Measured on 3.10.21,
    3.12.7 and 3.13.14: the boundary is the digit run's full length
    INCLUDING leading zeros; ``n == limit`` passes, ``n == limit + 1``
    raises; HEX refs are EXEMPT (base 16 is a power of two, and the limit
    applies only to non-power-of-two bases, so arbitrarily long hex refs still
    classify, out-of-range to U+FFFD, never raising); and the raise happens
    before the Windows-1252 remap or the out-of-range guard could rescue the
    value. tors reads the limit from the RUNNING interpreter once per call,
    so a caller's ``sys.set_int_max_str_digits()`` change is honored on the
    very next call (pinned here at a lowered limit)."""

    def test_decimal_refs_at_the_boundary_pass_and_just_over_raises(self) -> None:
        """The boundary pins, differential against the running interpreter's
        own ``html.unescape`` at BOTH edges: exactly-at-limit classifies
        normally (out of range → U+FFFD), limit+1 raises the stdlib's own
        ValueError with the stdlib's own message, both with and without the
        trailing semicolon (the digit run is the same)."""
        limit = sys.get_int_max_str_digits()
        if limit == 0:
            pytest.skip(
                "the integer-parse limit is disabled on this interpreter "
                "(sys.get_int_max_str_digits() == 0): any-length refs "
                "classify, so the boundary pins cannot run"
            )
        for suffix in (";", ""):
            at_limit = f"&#{'9' * limit}{suffix}"
            over = f"&#{'9' * (limit + 1)}{suffix}"
            assert html_unescape(at_limit) == html.unescape(at_limit) == "\ufffd"
            stdlib_exc, tors_exc = None, None
            try:
                html.unescape(over)
            except ValueError as exc:
                stdlib_exc = exc
            try:
                html_unescape(over)
            except ValueError as exc:
                tors_exc = exc
            assert stdlib_exc is not None, "the running stdlib did not raise; re-pin"
            assert tors_exc is not None, "tors classified an over-limit decimal ref"
            assert type(tors_exc) is ValueError
            # THE pin: tors raises the running interpreter's OWN ValueError,
            # obtained by replaying int() on the offending run, so the message
            # matches even where CPython's wording is version-dependent
            # ("limit (4300 digits)" on 3.12+/late-3.11, "limit (4300)" on
            # 3.10.7–3.11.x; the limit number itself is the stable part).
            assert str(tors_exc) == str(stdlib_exc)
            assert f"Exceeds the limit ({limit}" in str(tors_exc)
            assert f"value has {limit + 1} digits" in str(tors_exc)

    def test_leading_zeros_count_toward_the_limit(self) -> None:
        """CPython's count is the digit RUN's full length (measured: a run of
        4800 zeros raises "value has 4800 digits"), so a 4301-zero run whose
        VALUE is 0 still raises; tors must count the run, not the parsed
        value."""
        limit = sys.get_int_max_str_digits()
        if limit == 0:
            pytest.skip("limit disabled on this interpreter (see the boundary test)")
        zeros = f"&#{'0' * (limit + 1)};"
        with pytest.raises(ValueError, match=f"value has {limit + 1} digits"):
            html_unescape(zeros)

    def test_hex_refs_are_exempt_no_matter_their_length(self) -> None:
        """The power-of-two-base exemption, pinned: ``int(s, 16)`` is not
        subject to the limit (measured on 3.12.7 and 3.13.14), so an
        arbitrarily long hex ref still CLASSIFIES (out of range → U+FFFD)
        and never raises, at the default limit or a lowered one."""
        limit = sys.get_int_max_str_digits()
        over_hex = f"&#x{'f' * (max(limit, 1) + 500)};"
        assert html_unescape(over_hex) == html.unescape(over_hex) == "\ufffd"

    def test_the_raise_precedes_every_classification_rescue(self) -> None:
        """CPython raises inside ``int()`` BEFORE ``_replace_charref``
        classifies, so no classification can rescue an over-limit run: a ref
        whose VALUE would Windows-1252-remap (128 padded past the limit with
        leading zeros) and one far out of range both raise the limit error,
        never €, never U+FFFD."""
        limit = sys.get_int_max_str_digits()
        if limit == 0:
            pytest.skip("limit disabled on this interpreter (see the boundary test)")
        would_remap = f"&#{'0' * limit}128;"  # 4300+3 digits, value 128 → remap class
        would_saturate = f"&#{'9' * (limit + 500)};"
        for ref in (would_remap, would_saturate):
            with pytest.raises(ValueError, match="Exceeds the limit"):
                html_unescape(ref)

    def test_the_limit_is_read_from_the_running_interpreter_per_call(self) -> None:
        """The per-call read: with the limit lowered via
        ``sys.set_int_max_str_digits`` (restored in ``finally``), the NEXT
        call honors it: exactly-at passes, one over raises with the LOWERED
        limit interpolated, proving tors consults the running interpreter
        rather than a cached or compiled-in number."""
        original = sys.get_int_max_str_digits()
        if original == 0:
            pytest.skip("limit disabled on this interpreter (see the boundary test)")
        try:
            sys.set_int_max_str_digits(700)
            at_limit = f"&#{'9' * 700};"
            over = f"&#{'9' * 701};"
            assert html_unescape(at_limit) == html.unescape(at_limit) == "\ufffd"
            # "limit (700" matches both CPython wordings (with and without
            # the trailing "digits"; see the boundary test's note).
            with pytest.raises(ValueError, match=r"limit \(700") as excinfo:
                html_unescape(over)
            assert str(excinfo.value) == str(_stdlib_raise(over))
        finally:
            sys.set_int_max_str_digits(original)


def _stdlib_raise(ref: str) -> BaseException:
    """The stdlib's own exception object for ``ref`` (it raises; the callers
    assert that as a precondition)."""
    try:
        html.unescape(ref)
    except ValueError as exc:
        return exc
    raise AssertionError(f"the running stdlib did not raise on {ref!r}")


class TestTrickyBattery:
    @pytest.mark.parametrize(
        ("ref", "expected"),
        _TRICKY,
        ids=[s[:24] for s, _ in _TRICKY],
    )
    def test_matches_the_recorded_and_running_interpreter_outcome(
        self, ref: str, expected: str
    ) -> None:
        """Every battery row must equal BOTH the recorded literal (measured on
        3.12.7 pre-implementation) and the RUNNING interpreter's
        ``html.unescape``, result-identical on the tricky classes a
        longest-prefix + numeric-classification implementation gets wrong."""
        stdlib = html.unescape(ref)
        assert stdlib == expected, f"the literal drifted from this leg's stdlib: {ref!r}"
        assert html_unescape(ref) == expected, f"tors diverged on {ref!r}"


@st.composite
def entity_text(draw: DrawFn) -> str:
    """Entity-dense text: plain-unicode fragments, named refs from the RUNNING
    interpreter's table (both spellings), numeric refs in every spelling
    spanning the classification boundaries (valid, remapped, invalid-codepoint,
    surrogate, out-of-range, non-ASCII-max), and degenerate '&' shapes,
    joined with '&' so adjacent-ref cases occur naturally. The input class
    this function exists for, biased toward exactly the boundary behaviors
    rather than uniform random unicode."""
    numeric = st.sampled_from(
        [
            "&#{cp};",
            "&#{cp}",
            "&#x{cp:x};",
            "&#X{cp:X};",
            "&#{cp}zz;",
            "&#x{cp}g;",
            "&#;",
            "&;",
            "&&",
        ]
    )
    fragment = st.one_of(
        st.text(alphabet=st.characters(exclude_categories=("Cn", "Cs")), max_size=10),
        st.builds(lambda key: "&" + key, st.sampled_from(list(html.entities.html5))),
        st.builds(
            lambda tmpl, cp: tmpl.format(cp=cp),
            numeric,
            st.sampled_from(
                [0, 1, 9, 13, 65, 128, 0xFFFD, 0xD800, 0xFFFE, 0x10FFFF, 0x110000, 0xAC00, 0x1CCD6]
            ),
        ),
    )
    parts = draw(st.lists(fragment, min_size=0, max_size=15))
    return "&".join(parts)


class TestHypothesisParity:
    @given(entity_text())
    @settings(max_examples=500)
    def test_entity_bearing_text_decodes_identically(self, text: str) -> None:
        assert html_unescape(text) == html.unescape(text)

    @given(st.text(max_size=64))
    @settings(max_examples=300)
    def test_arbitrary_text_decodes_identically(self, text: str) -> None:
        """Uniform random unicode, '&' or not, guarding the no-'&' fast path and
        the scan's handling of multibyte characters around any '&'."""
        assert html_unescape(text) == html.unescape(text)


class TestArgumentContract:
    @pytest.mark.parametrize(
        "not_str",
        [b"a&amp;b", bytearray(b"&amp;"), memoryview(b"&amp;"), 123, None],
        ids=["bytes", "bytearray", "memoryview", "int", "none"],
    )
    def test_non_str_arguments_raise_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            html_unescape(not_str)  # type: ignore[arg-type]

    def test_lone_surrogates_are_refused_at_the_argument_boundary(self) -> None:
        """Same pyo3 ``&str`` boundary as every str-in function (pinned for
        ``finalize`` in tests/test_finalize.py): a str holding a lone
        surrogate cannot be UTF-8-borrowed, so ``UnicodeEncodeError`` before any
        Rust code runs."""
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            html_unescape("ok\ud800ok")  # type: ignore[arg-type]


class TestIdentityReturnContract:
    """The generalization of the no-``&`` fast path: ``html_unescape``
    returns the ORIGINAL object whenever the unescape changes nothing, not
    only when there is no ``&`` at all (CPython's own ``if '&' not in s:
    return s``), but also when ampersands are present and nothing decodes
    (failed refs, bare ``&``, degenerate shapes): the scan's output equals the
    input byte-for-byte, so the input comes back instead of a marshalled
    copy."""

    def test_ampersand_free_text_returns_the_same_object(self) -> None:
        text = "plain text, no entities here"
        assert html_unescape(text) is text

    @pytest.mark.parametrize(
        "verbatim",
        [
            "&x;",
            "&;",
            "&",
            "& #65;",
            "&# 65;",
            "&zzz;",  # unknown name with no prefix hit
            "&xyzzy",  # unknown name, no semicolon
            "a & b & c",
            "&there4",  # exists with ';' only; the without-';' spelling fails
        ],
        ids=[
            "unknown-ref",
            "empty-name",
            "bare-amp",
            "space-inside",
            "space-after-hash",
            "unknown-with-semi",
            "unknown-no-semi",
            "mid-text-amps",
            "needs-semicolon",
        ],
    )
    def test_ampersand_bearing_text_where_nothing_decodes_returns_the_same_object(
        self, verbatim: str
    ) -> None:
        assert html_unescape(verbatim) == verbatim  # CPython parity, checked
        assert html_unescape(verbatim) is verbatim

    def test_decoding_text_returns_a_new_object(self) -> None:
        text = "Tom &amp; Jerry &#233;"
        result = html_unescape(text)
        assert result == "Tom & Jerry \u00e9"
        assert result is not text

    @given(pathological_text())
    @settings(max_examples=500)
    def test_value_identity_implies_object_identity(self, text: str) -> None:
        result = html_unescape(text)
        if result == text:
            assert result is text
