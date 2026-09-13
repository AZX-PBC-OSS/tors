"""Differential parity for the tors scrub_pii port: tors against the
quoted-pin oracle, and (opt-in) against the live source module.

Provenance: the behavior oracle is a private consumer's telemetry-safety
module, transcribed into ``tests/reference.py`` as
``reference_scrub_pii`` — its two pattern shapes and both token fields
quoted as literals, with the digest salt parameterized (the source chain
digests unsalted, so ``salt=""`` is its byte-exact token spelling). The
pin is deliberate: a change to the source module's grammar is a parity
re-sync request, not a drive-by fix. Mapping:
``tors.scrub_pii(text, rules, salt=salt) ==
reference_scrub_pii(text, rules, salt=salt)`` for every ``rules``/``salt``
lane below, and, on the live lane,
``tors.scrub_pii(text, salt="") == <the source module's free-text scrub
entry point>(text)``.

tors's phone rule carries one deliberate extension past that contract:
the domestic NANP matcher (un-plussed shapes the source leaves
untouched). The lanes are split accordingly: every QUOTED-PIN lane below
guards on ``has_domestic_shape(text)`` — parity is asserted exactly where
the source's own semantics apply — and ``TestDomesticExtension`` pins the
other side in both directions (the oracle must leave the shape, tors must
scrub exactly the span), so a regression on either side of the extension
fails loudly instead of surfacing as a parity mystery.

The live re-sync lane is env-gated and NEVER runs in CI:
``TORS_SCRUB_PII_ORACLE`` carries the full locator —
``path/to/module.py:entry_point``, a module file path and the scrub
callable's name, split on the last colon — so nothing about the
source's spelling lives in this repo. Unset, every live-lane test
skips and the quoted pin above remains the CI oracle. The one pinned
divergence is
deliberate and documented in both directions: a str holding lone
surrogates is refused by tors with ``UnicodeEncodeError`` (the crate-wide
str contract) while the source chain — none of whose classes can match a
surrogate — returns it unchanged; the live lane asserts that divergence
instead of skipping it, so a re-sync can never mistake it for a port bug.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import unicodedata
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tors
from reference import (
    SCRUB_PII_EMAILS,
    SCRUB_PII_PHONES,
    SCRUB_PII_SEPARATORS,
    reference_scrub_pii,
)

# The grammar-edge fragments the corpus composes with the zoo pieces: the
# backtracking corners (a domain split that leaves a trailing digit, the
# one-letter-TLD non-match, the adjacent-match re-fire pair), the phone
# anchoring edges, and a Unicode-digit spelling (codepoint-built, the
# reference.py pure-ASCII-source idiom).
_ARABIC_TWELVE = "+" + "".join(chr(0x0660 + i % 10) for i in range(12))
_EDGE_FRAGMENTS = [
    "a@b.co9",
    "a@b.co.",
    "a@b.c.d",
    "a@b.co9@x.yz",
    "a@b@c.co",
    "a!b@x.co",
    "https://user@x.io/path",
    "+1234567",
    "+12345678",
    "+1+4155552671",
    "+ (415) 555-2671",
    "+4712345678 1234567890",
    _ARABIC_TWELVE,
    "read 4096 bytes in 1200 ms",
    "ticket 4096 closed",
    "no contacts here at all",
    "",
]

CORPUS: list[str] = (
    _EDGE_FRAGMENTS
    + [
        f"{email}{sep}{phone}"
        for email in SCRUB_PII_EMAILS
        for phone in SCRUB_PII_PHONES
        for sep in SCRUB_PII_SEPARATORS
    ]
    + [
        f"{phone}{sep}{email}"
        for email in SCRUB_PII_EMAILS
        for phone in SCRUB_PII_PHONES
        for sep in SCRUB_PII_SEPARATORS
    ]
    + [
        # the re-fire corner and its email-only/phone-only reductions, the
        # canonical rejection excerpt, and a nested phone-inside-email
        f"{frag} {frag}"
        for frag in ("a@b.co9@x.yz", "+1 415 557 8901ada@x.co", "+1 415 555 2671")
    ]
    + [
        "unknown candidate fungai.chetima@example.com called from +14155552671 twice",
        "user+14155552671@example.com",
        # The composition falsifier hypothesis found under the domestic
        # extension: adjacent addresses whose email tokens' digest tails
        # plus the following numbers would spell ten-digit runs — the
        # clean-boundary rule keeps the phone pass off them, and the
        # engines agree (a regression here is a token-digest mangle).
        "fungai.chetima@example.comfungai.chetima@example.com"
        "fungai.chetima@example.comread 4096 bytes in 1200 ms",
        "a@b.co 4096-1200-4096 x",
    ]
)

_RULES_LANES: list[list[str] | None] = [
    None,
    ["contact_email", "contact_phone"],
    ["contact_phone", "contact_email"],
    ["contact_email"],
    ["contact_phone"],
    [],
]

_SALT_LANES: list[str | None] = [None, "", "site-secret"]

# The domestic guard: whether tors's domestic matcher (the extension
# past the source's + anchored grammar) would fire anywhere in `text`.
# A faithful test-side mirror of the landed grammar — a FULL un-plussed
# class run of exactly ten Nd digits, or eleven with an ASCII leading
# '1', carrying at least one separator inside the match span — used only
# to route inputs between the parity lanes (conservative by design: an
# input this flags whose domestic shape the email pass then consumes
# still agrees between the engines, and is simply not asserted here).
_DOMESTIC_SEPARATORS = "-. ()"


def _is_nd(c: str) -> bool:
    return unicodedata.category(c) == "Nd"


def has_domestic_shape(text: str) -> bool:
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if _is_nd(c) or c in _DOMESTIC_SEPARATORS:
            run_start = i
            j = i
            digits = 0
            first_digit: str | None = None
            last_digit = -1
            while j < n and (_is_nd(text[j]) or text[j] in _DOMESTIC_SEPARATORS):
                if _is_nd(text[j]):
                    digits += 1
                    if first_digit is None:
                        first_digit = text[j]
                    last_digit = j
                j += 1
            plussed = run_start > 0 and text[run_start - 1] == "+"
            if not plussed and (
                digits == 10 or (digits == 11 and first_digit == "1")
            ):
                k = run_start
                while text[k] == " ":
                    k += 1
                if any(ch in _DOMESTIC_SEPARATORS for ch in text[k:last_digit]):
                    return True
            i = j
        else:
            i += 1
    return False


@pytest.mark.parametrize("salt", _SALT_LANES, ids=["default-salt", "unsalted", "custom-salt"])
@pytest.mark.parametrize("text", CORPUS, ids=[f"corpus-{i}" for i in range(len(CORPUS))])
def test_quoted_pin_parity(text: str, salt: str | None) -> None:
    """The CI oracle lane, both rules (the canonical order): every corpus
    shape at every salt spelling — tors's documented default, the
    unsalted source-parity spelling, and a caller secret. Domestic-bearing
    inputs route to the extension lane (the source has no domestic
    grammar to be parity-checked against)."""
    if has_domestic_shape(text):
        pytest.skip("domestic shape: pinned in TestDomesticExtension, not the parity lane")
    assert tors.scrub_pii(text, salt=salt) == reference_scrub_pii(text, salt=salt)


@pytest.mark.parametrize("rules", _RULES_LANES, ids=lambda r: f"rules-{r}")
@pytest.mark.parametrize("text", CORPUS, ids=[f"corpus-{i}" for i in range(len(CORPUS))])
def test_quoted_pin_rules_parity(text: str, rules: list[str] | None) -> None:
    """The rules lanes over the unsalted spelling (the source chain's own
    digest): both canonical orders, each subset, and the identity — the
    last one pinning that the oracle and tors agree on doing nothing."""
    if has_domestic_shape(text):
        pytest.skip("domestic shape: pinned in TestDomesticExtension, not the parity lane")
    assert tors.scrub_pii(text, rules, salt="") == reference_scrub_pii(text, rules, salt="")


# Structured-random compositions from the zoo pieces: contact material,
# grammar edges, and inert filler interleaved, the shape real error
# excerpts have (contacts embedded in prose with stray digits and
# separators around them). Non-ASCII pieces built from codepoints (the
# reference.py pure-ASCII-source idiom): inert prose accents, and Nd
# digits from three scripts — Arabic-Indic, fullwidth, Devanagari.
_ELLIPSIS = chr(0x2026)
_E_ACUTE = chr(0x00E9)
_BOLD_DIGITS = "".join(chr(0x1D7CE + i) for i in range(3))
_SEGMENTED_DIGIT = chr(0x1FBF7)  # legacy-computing segmented seven: Nd, not No
_INERT = ["ok", "ticket", "4096", "1200 ms", "x", _ELLIPSIS, "caf" + _E_ACUTE]
_ND_DIGITS = [chr(0x0661), chr(0xFF19), chr(0x0968)]
_PIECES = (
    list(SCRUB_PII_EMAILS)
    + list(SCRUB_PII_PHONES)
    + list(SCRUB_PII_SEPARATORS)
    + _EDGE_FRAGMENTS
    + _INERT
    + _ND_DIGITS
    + [_BOLD_DIGITS, _SEGMENTED_DIGIT]
    + ["@", ".", "+", "~", "9", "-"]
)

_composed_text = st.lists(st.sampled_from(_PIECES), min_size=0, max_size=25).map("".join)


class TestHypothesisDifferential:
    @pytest.mark.parametrize("salt", _SALT_LANES, ids=["default-salt", "unsalted", "custom-salt"])
    @given(text=_composed_text)
    @settings(max_examples=300, deadline=None)
    def test_both_rules_match_the_quoted_pin(self, text: str, salt: str | None) -> None:
        assume(not has_domestic_shape(text))
        assert tors.scrub_pii(text, salt=salt) == reference_scrub_pii(text, salt=salt)

    @pytest.mark.parametrize(
        "rules", [["contact_email"], ["contact_phone"], []], ids=["email", "phone", "identity"]
    )
    @given(text=_composed_text)
    @settings(max_examples=150, deadline=None)
    def test_rule_subsets_match_the_quoted_pin(self, text: str, rules: list[str]) -> None:
        assume(not has_domestic_shape(text))
        assert tors.scrub_pii(text, rules, salt="") == reference_scrub_pii(text, rules, salt="")

    @given(text=_composed_text)
    @settings(max_examples=200, deadline=None)
    def test_scrubbing_twice_converges(self, text: str) -> None:
        twice = tors.scrub_pii(tors.scrub_pii(text, salt=""), salt="")
        assert tors.scrub_pii(twice, salt="") == twice


# --- The extension lane: the domestic matcher past the source -----------------
#
# The source's phone grammar is + anchored, so it leaves every un-plussed
# digit run untouched. tors scrubs the NANP shapes (maintainer-directed),
# with the discipline rules tests/test_scrub_pii.py pins in full. This lane
# pins the DELIBERATE divergence in both directions — the oracle must
# leave the shape, tors must scrub exactly the span — so a regression on
# either side of the extension fails with its own name on it.


_DOMESTIC_CASES: list[tuple[str, str]] = [
    # (input, the span tors scrubs)
    ("(415) 555-2671", "(415) 555-2671"),
    ("415-555-2671", "415-555-2671"),
    ("415.555.2671", "415.555.2671"),
    ("415 555 2671", "415 555 2671"),
    ("1-415-555-2671", "1-415-555-2671"),
    ("1 (415) 555-2671", "1 (415) 555-2671"),
    ("1 415 555 2671", "1 415 555 2671"),
    ("1415 555 2671", "1415 555 2671"),
    ("call 415-555-2671 ok", "415-555-2671"),
    ("415-555-2671 x1234", "415-555-2671"),
]

_DOMESTIC_NON_MATCHES: list[str] = [
    # The discipline cuts, shared by BOTH engines (parity on the
    # non-matches too): bare runs, wrong widths, wrong leading digit.
    "4155552671",
    "14155552671",
    "order 1234567890 closed",
    "415-555-267",
    "415-555-267123",
    "415-555-2671-555-123-4567",
    "915-555-26712",
]


class TestDomesticExtension:
    @pytest.mark.parametrize(
        ("text", "matched"), _DOMESTIC_CASES, ids=[c[1] for c in _DOMESTIC_CASES]
    )
    def test_the_quoted_oracle_leaves_the_shape(self, text: str, matched: str) -> None:
        assert reference_scrub_pii(text, None, salt="") == text

    @pytest.mark.parametrize(
        ("text", "matched"), _DOMESTIC_CASES, ids=[c[1] for c in _DOMESTIC_CASES]
    )
    def test_tors_scrubs_exactly_the_span(self, text: str, matched: str) -> None:
        token = f"{matched[:3]}~{hashlib.sha256(matched.encode('utf-8')).hexdigest()[:12]}"
        assert text.count(matched) == 1
        assert tors.scrub_pii(text, salt="") == text.replace(matched, token, 1)

    @pytest.mark.parametrize(
        "text", _DOMESTIC_NON_MATCHES, ids=[f"shared-nonmatch-{i}" for i in range(7)]
    )
    def test_the_discipline_cuts_are_shared_non_matches(self, text: str) -> None:
        assert reference_scrub_pii(text, None, salt="") == text
        assert tors.scrub_pii(text, salt="") == text


# --- The opt-in live re-sync lane ---------------------------------------------------
#
# TORS_SCRUB_PII_ORACLE carries the full locator —
# "path/to/module.py:entry_point" (a private machine's path and the
# callable's name, never committed): the telemetry-safety module whose
# free-text scrub entry point the port pinned. Unset, every test below
# skips — the quoted pin above is the CI oracle, so no CI lane ever
# references the source, and no spelling of the source's names lives in
# this repo.


def _live_oracle() -> Any:
    locator = os.environ.get("TORS_SCRUB_PII_ORACLE")
    if not locator:
        pytest.skip(
            "TORS_SCRUB_PII_ORACLE unset: the live re-sync lane is opt-in "
            "(set it to 'path/to/module.py:entry_point' on a machine that "
            "has the telemetry-safety module); the quoted pin in "
            "tests/reference.py is the CI oracle"
        )
    path, sep, name = locator.rpartition(":")
    if not sep or not path or not name.isidentifier():
        pytest.fail(
            "TORS_SCRUB_PII_ORACLE must be 'path/to/module.py:entry_point' "
            "(a module file path, one colon, the callable's name): got "
            f"{locator!r}"
        )
    spec = importlib.util.spec_from_file_location("tors_scrub_pii_live_oracle", path)
    if spec is None or spec.loader is None:
        pytest.fail(f"TORS_SCRUB_PII_ORACLE names an unloadable module file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, name, None)
    if not callable(entry):
        pytest.fail(f"{path} exposes no callable {name}(text) entry point")
    return entry


class TestLiveResyncLane:
    def test_the_corpus_matches_the_live_module(self) -> None:
        live = _live_oracle()
        for text in CORPUS:
            if has_domestic_shape(text):
                continue  # the extension lane's territory; the live module has no domestic grammar
            assert tors.scrub_pii(text, salt="") == live(text), text

    @given(text=_composed_text)
    @settings(max_examples=300, deadline=None)
    def test_compositions_match_the_live_module(self, text: str) -> None:
        assume(not has_domestic_shape(text))
        live = _live_oracle()
        assert tors.scrub_pii(text, salt="") == live(text)

    def test_the_surrogate_divergence_is_pinned_not_skipped(self) -> None:
        """The one documented divergence, asserted on both sides: tors
        refuses a surrogate-bearing str at the argument boundary
        (UnicodeEncodeError, the crate-wide str contract) while the source
        chain keeps going — none of its classes can match a surrogate, so
        no digest ever sees one, but the NON-surrogate matches around it
        still scrub. A re-sync that changes either side shows up here
        instead of masquerading as parity."""
        live = _live_oracle()
        text = "a\ud800b@x.co"
        with pytest.raises(UnicodeEncodeError):
            tors.scrub_pii(text, salt="")
        # The live side: the surrogate survives (no class contains it),
        # the "b@x.co" around it still scrubs, unsalted digest and all.
        unsalted = hashlib.sha256(b"b@x.co").hexdigest()[:12]
        assert live(text) == f"a\ud800@x.co~{unsalted}"
