"""Parity pins: tors's own Unicode tables against the running interpreter's.

tors ships its own tables (the ``unicode-normalization`` crate) precisely so its output is
identical on every supported interpreter (3.10-3.14), the property that makes its hashes
safe for hash-gated dedupe, and one CPython's per-version ``unicodedata`` cannot promise.
These tests pin the other direction: parity with *each interpreter's own* tables, by
exhausting every canonically-decomposable codepoint, every Hangul syllable, and every
character of Python's ``str.isspace()`` set. Running on every CI matrix leg, each Python
version pins its own version's parity: zero divergence on every codepoint that leg's ucd
assigns, with the residual (mappings from newer UCDs than that leg's) confined to
codepoints the leg leaves unassigned (measured per leg in the confinement test below).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

import tors

# U+2000 EN QUAD and U+2001 EM QUAD are the only codepoints whose canonical decomposition
# is entirely Python-whitespace (U+2000 -> U+2002, U+2001 -> U+2003, measured on UCD
# 15.0/16.0; every other decomposable codepoint's NFD contains a non-whitespace
# character). For their NFD forms the interpreter's raw NFC is whitespace-only, so the
# pipeline's final ``strip()`` legitimately empties it while raw NFC keeps the space: the
# naive ``tors.normalize(nfd) == unicodedata.normalize("NFC", nfd)`` is confounded by the
# strip stage, not by a table divergence; tors's composition agrees with the
# interpreter's here (the pure-Python oracle ``reference_normalize`` strips identically,
# and the strip-set test below proves both agree on every whitespace char). The sweep
# compares against the interpreter's NFC with that one documented adjustment, and asserts
# the confounded set is exactly these two codepoints: a future Unicode or crate version
# that grows it fails here loudly instead of being silently masked.
_WHITESPACE_DECOMPOSING_CODEPOINTS = frozenset({0x2000, 0x2001})

# Built from codepoints (pure-ASCII source): U+200B zero width space, U+2060 word joiner,
# U+FEFF BYTE ORDER MARK, all invisible, none Python-whitespace.
_NON_WS_LOOKALIKES = [chr(0x200B), chr(0x2060), chr(0xFEFF)]

# U+FB01 latin small ligature fi and kin, compatibility (not canonical) composites.
_COMPAT_LIGATURES = [chr(0xFB01), chr(0xFB02), chr(0xFB00)]

# tors's tables are Unicode 16.0 (the ``unicode-normalization`` crate, 0.1.25);
# update the major with a crate bump. The residual bound in the confinement
# test below scales with the ucd gap between tors's tables and the running
# interpreter, so every matrix leg gets a bound calibrated for its own ucd
# rather than the newest one.
_TORS_UCD_MAJOR = 16


def test_exhaustive_nfc_sweep_over_every_canonically_decomposable_codepoint() -> None:
    """For every codepoint with a canonical decomposition, tors's normalize of its NFD
    form must equal the running interpreter's NFC of the same, i.e. tors's composition
    tables agree with this Python's ``unicodedata`` everywhere, not just on sampled
    characters."""
    confounded: set[int] = set()
    checked = 0
    for cp in range(0x110000):
        char = chr(cp)
        decomposition = unicodedata.decomposition(char)
        if not decomposition or decomposition.startswith("<"):
            continue  # compat-only decompositions are out of scope (see the NFKC pin)
        checked += 1
        nfd = unicodedata.normalize("NFD", char)
        expected = unicodedata.normalize("NFC", nfd)
        if all(map(str.isspace, expected)):
            confounded.add(cp)
            expected = ""  # the pipeline strips whitespace-only results; see module docs
        assert tors.normalize(nfd) == expected, (
            f"U+{cp:04X}: tors={tors.normalize(nfd)!r} py={expected!r}"
        )
    assert checked > 2000  # the sweep must actually sweep (ucd 15.0 has 2061)
    assert confounded == _WHITESPACE_DECOMPOSING_CODEPOINTS, (
        "codepoints whose canonical decomposition is entirely Python-whitespace changed: "
        f"{', '.join(f'U+{cp:04X}' for cp in sorted(confounded))}: a Unicode-version "
        "change on this interpreter or a crate table change; inspect before touching the "
        "carve-out"
    )


def test_every_hangul_syllable_composes_like_the_interpreter() -> None:
    """Hangul composition is algorithmic (L+V+T jamo -> syllable), not table-driven, so
    it needs its own sweep: every one of the 11172 precomposed syllables, through its
    three-char NFD form, must come back as exactly the syllable the interpreter's NFC
    produces."""
    for cp in range(0xAC00, 0xD7A4):
        syllable = chr(cp)
        nfd = unicodedata.normalize("NFD", syllable)
        expected = unicodedata.normalize("NFC", nfd)
        assert tors.normalize(nfd) == expected, f"U+{cp:04X}"


def test_every_python_whitespace_char_strips_from_both_ends() -> None:
    """For every character of the running interpreter's ``str.isspace()`` set (computed,
    not hardcoded, so a new Python version's set is swept automatically),
    ``normalize(ws + "x")`` and ``normalize("x" + ws)`` must both be ``"x"``.

    Reasoned per character class, then verified exhaustively: space/tab are also dropped
    mid-string only before a newline but always stripped at the ends; \\r folds to \\n
    first and is then stripped as leading/trailing whitespace; \\v, \\f, 0x1c-0x1f, NEL,
    and the Unicode space block members are not ``[ \\t]`` and not line endings, so only
    the end-strip removes them, which is exactly what the assertion needs at the ends."""
    whitespace = [chr(cp) for cp in range(0x110000) if chr(cp).isspace()]
    assert len(whitespace) > 25  # the sweep must actually sweep (ucd 15.0 has 29)
    for ws in whitespace:
        assert tors.normalize(ws + "x") == "x", f"U+{ord(ws):04X} prefix"
        assert tors.normalize("x" + ws) == "x", f"U+{ord(ws):04X} suffix"


def test_non_whitespace_lookalikes_are_preserved() -> None:
    """U+200B zero width space, U+2060 WORD JOINER and U+FEFF BYTE ORDER MARK look like
    whitespace to humans and to some pipelines, but Python's ``str.strip()`` does not
    remove them, so tors must not either (a silent drop would change hashes and diverge
    from the reference pipeline)."""
    for char in _NON_WS_LOOKALIKES:
        assert tors.normalize(char + "x") == char + "x", f"U+{ord(char):04X} prefix"
        assert tors.normalize("x" + char) == "x" + char, f"U+{ord(char):04X} suffix"


def test_compat_ligatures_do_not_compose() -> None:
    """tors implements NFC, not NFKC: compatibility decompositions (``decomposition()``
    starting with ``<``) must not fire: U+FB01 "ﬁ" stays one codepoint rather than
    becoming "fi", and likewise the other Latin ligatures. A hash pipeline that quietly
    applied NFKC would produce different digests than every consumer that expects NFC."""
    for ligature in _COMPAT_LIGATURES:
        assert tors.normalize(ligature) == ligature, f"U+{ord(ligature):04X}"
        assert unicodedata.decomposition(ligature).startswith("<")  # it really is compat


# --- the standalone normalization forms (nfc / nfd / nfkc / nfkd) --------------

# Every decomposable codepoint, both decomposition classes, every form: the
# sweep discipline extended per the spec ("every canonically-
# decomposable codepoint for nfc/nfd; every compatibility-decomposable
# codepoint for nfkc/nfkd") and strengthened one step, because the K-forms
# also transform canonically-decomposable codepoints (NFKD decomposes
# canonically and compatibly) and the C-forms must leave compat codepoints
# alone: so every decomposable codepoint is checked under all four forms, on
# both the raw character and its NFD rendering (the composed-input and
# decomposed-input shapes).


def _forms() -> dict[str, Callable[[str], str]]:
    """The four standalone forms, resolved at call time so every test in this
    module is independently red until the surface exists (no shared module-level
    state, no test-ordering dependency)."""
    return {"NFC": tors.nfc, "NFD": tors.nfd, "NFKC": tors.nfkc, "NFKD": tors.nfkd}


def test_standalone_forms_sweep_over_every_decomposable_codepoint() -> None:
    """For every codepoint with any decomposition mapping (canonical or
    compatibility), each standalone form must agree with the running
    interpreter's ``unicodedata.normalize`` on both the raw character and its
    NFD rendering: tors's tables and its NFKC/NFKD compat mappings match this
    Python's everywhere, not just on sampled characters. Raw equality, no
    adjustment: the standalone forms have no strip stage (the pipeline sweep's
    confound carve-out does not apply; see the next test)."""
    forms = _forms()
    canonical = 0
    compat = 0
    for cp in range(0x110000):
        char = chr(cp)
        decomposition = unicodedata.decomposition(char)
        if not decomposition:
            continue
        if decomposition.startswith("<"):
            compat += 1
        else:
            canonical += 1
        nfd = unicodedata.normalize("NFD", char)
        for form, tors_form in forms.items():
            for label, text in (("raw", char), ("nfd", nfd)):
                expected = unicodedata.normalize(form, text)
                assert tors_form(text) == expected, (
                    f"U+{cp:04X} {form}({label}): tors={tors_form(text)!r} py={expected!r}"
                )
    # The sweep must actually sweep (ucd 15.0: 2061 canonical, 3796 compat).
    assert canonical > 2000
    assert compat > 3000


def test_every_hangul_syllable_matches_the_interpreter_in_every_form() -> None:
    """Hangul composition/decomposition is algorithmic (L+V+T jamo <-> syllable),
    not table-driven, so it needs its own sweep per form: every one of the 11172
    precomposed syllables must come back as exactly what the interpreter's form
    produces: NFC composes the jamo sequence back, NFD/NFKD decompose to it,
    NFKC leaves the syllable alone (Hangul has no compatibility decomposition)."""
    for cp in range(0xAC00, 0xD7A4):
        syllable = chr(cp)
        nfd = unicodedata.normalize("NFD", syllable)
        for form, tors_form in _forms().items():
            for text in (syllable, nfd):
                expected = unicodedata.normalize(form, text)
                assert tors_form(text) == expected, f"U+{cp:04X} {form}"


def test_the_pipeline_strip_confound_does_not_apply_to_the_standalone_forms() -> None:
    """The NFC sweep's {U+2000, U+2001} carve-out exists because
    ``tors.normalize`` strips whitespace-only results: for those two codepoints
    the interpreter's raw NFC of their NFD forms is whitespace-only (U+2002 /
    U+2003), so the pipeline legitimately empties it and the naive raw-equality
    assertion is confounded by the strip stage. The standalone forms have no
    strip stage, so the confound vanishes: raw parity must hold for exactly
    these codepoints, with the whitespace result preserved, pinned explicitly
    so nobody carries the pipeline sweep's adjustment over to the forms, and so
    a future change that grows the confounded set stays caught by the sweep
    above."""
    for cp in _WHITESPACE_DECOMPOSING_CODEPOINTS:
        nfd = unicodedata.normalize("NFD", chr(cp))
        for form, tors_form in _forms().items():
            expected = unicodedata.normalize(form, nfd)
            assert expected  # whitespace-only, not empty; the strip is what emptied it
            assert tors_form(nfd) == expected, f"U+{cp:04X} {form}"


def test_form_divergences_are_confined_to_codepoints_the_interpreter_does_not_know() -> None:
    """The exhaustive divergence-confinement pin, over every non-surrogate
    codepoint (surrogates are refused at the ``&str`` argument boundary by
    contract; tests/test_forms.py's surrogate class): wherever a standalone
    form's output differs from the running interpreter's, the input codepoint
    must be one that interpreter leaves unassigned (``unicodedata.name`` is
    None). Zero divergences on assigned codepoints; that is the load-bearing
    per-leg parity guarantee, true on every matrix leg regardless of ucd
    version, and it fails loudly on a crate table defect or a Unicode change
    that makes tors disagree with an interpreter about a character that
    interpreter knows.

    Why divergences on unassigned codepoints exist at all (measured, not
    theoretical): tors's tables (the ``unicode-normalization`` crate, 0.1.25)
    are Unicode 16.0.0 (the same ucd CPython 3.14 ships), while older legs
    run older UCDs, so the residual is the newer-ucd mappings on codepoints
    the leg's own ucd predates. Measured per leg with the real wheel
    (residual by form, {NFC, NFD, NFKC, NFKD}):

    ========= ======= ==================================
    leg       ucd     residual
    ========= ======= ==================================
    3.10.21   13.0    356  {0, 20, 158, 178}
    3.11.16   14.0    238  {0, 20, 99, 119}
    3.12.7    15.0    114  {0, 20, 37, 57}
    3.13.14   15.1    114  {0, 20, 37, 57}
    3.14.0    16.0      2  {0, 0, 1, 1}
    ========= ======= ==================================

    The 3.12-class residual (also 3.10/3.11, proportionally more) is all
    new-in-16.0 mappings on codepoints 15.0 leaves unassigned: 20 canonical
    decompositions (Todhri U+105C9/U+105E4, compositions of a new base with
    the ancient U+0307, plus Tulu-Tigalari U+11383..U+113C8, Gurung Khema
    U+16121..U+16128, Kirat Rai U+16D68..U+16D6A) and 37 compatibility
    mappings (the Outlined Latin capital letters and digits U+1CCD6-U+1CCF9,
    ``<font>``-decomposing to ASCII, plus U+A7F1). The 3.14 residual is the
    one surprise: even on the matched-UCD leg, U+A7F1 (which CPython 3.14's
    own 16.0 tables leave unassigned with no decomposition) carries an
    NFKC/NFKD mapping to ``"S"`` in the crate's tables, so the crate's 16.0
    snapshot is not byte-identical to CPython's; still unassigned on the
    running interpreter, so the confinement guarantee holds. This also means
    the interpreter-assigned confinement is the correct per-leg statement of
    "parity with the running interpreter" for the forms; total raw equality
    is provably unattainable on legs whose ucd lags tors's tables, without
    giving up tors's own cross-interpreter determinism guarantee (same input
    -> same output on every supported Python, the reason tors ships its own
    tables at all).

    The bound below therefore scales with the ucd gap between tors's tables
    and the running interpreter (``60 + 130 * gap``), not a flat number
    calibrated on one leg: per leg it leaves 3.10 -> 450 vs 356 (~1.26x),
    3.11 -> 320 vs 238 (~1.34x), 3.12/3.13 -> 190 vs 114 (~1.67x), 3.14 ->
    60 vs 2; a residual beyond the leg's own gap-scaled bound is a crate
    table defect or a ucd jump, not an older interpreter."""
    forms = _forms()
    divergent_on_assigned: list[tuple[str, int]] = []
    residual_counts: dict[str, int] = {form: 0 for form in forms}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue  # refused at the &str boundary by contract
        char = chr(cp)
        for form, tors_form in forms.items():
            if tors_form(char) != unicodedata.normalize(form, char):
                residual_counts[form] += 1
                if unicodedata.name(char, None) is not None:
                    divergent_on_assigned.append((form, cp))
    assert not divergent_on_assigned, (
        "tors disagrees with the running interpreter about an ASSIGNED codepoint: "
        + ", ".join(f"{form} U+{cp:04X}" for form, cp in divergent_on_assigned[:10])
    )
    # The residual must be exactly the newer-ucd mappings, not noise: empty for
    # NFC (no new canonical compositions fire on single characters; the
    # additions are decompositions of newly assigned codepoints), and bounded
    # by the leg's own ucd-gap-scaled allowance (measured residuals and
    # headrooms in the docstring above).
    ucd_major = int(unicodedata.unidata_version.split(".")[0])
    ucd_gap = max(_TORS_UCD_MAJOR - ucd_major, 0)
    assert residual_counts["NFC"] == 0, "NFC diverged on single characters"
    assert sum(residual_counts.values()) < 60 + 130 * ucd_gap, (
        f"divergence residual implausibly large for UCD {unicodedata.unidata_version} "
        f"(gap {ucd_gap} to tors's {_TORS_UCD_MAJOR}.0 tables): {residual_counts}: "
        "a crate table defect or a UCD jump; inspect before adjusting anything"
    )
