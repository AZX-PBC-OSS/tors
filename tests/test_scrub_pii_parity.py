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
untouched), and the api_keys rule carries a second (the credential
families). The lanes are split accordingly: every QUOTED-PIN lane below
routes on the domestic guard — the input guard for single-rule lanes,
the email-pass-output guard for BOTH-rules lanes (the matcher runs on
the email pass's result, and the email local removal can trim a
too-long digit run into a phone shape) — and on the keys guard for
every lane where api_keys is active (the input guard suffices there:
the keys pass runs BEFORE the email pass, so a key-free input leaves it
the identity and the email-pass output the domestic guard needs is the
same both engines produce) — parity is asserted exactly
where the source's own semantics apply — and ``TestDomesticExtension``
/ ``TestApiKeyExtension``
pin the other side in both directions (the oracle must leave the shape, tors must
scrub exactly the span), so a regression on either side of an extension
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
from functools import cache
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
    ["api_keys"],
    [],
]

_SALT_LANES: list[str | None] = [None, "", "site-secret"]

# The domestic guard: whether tors's domestic matcher (the extension
# past the source's + anchored grammar) would fire anywhere in `text`.
# A faithful test-side mirror of the landed grammar — a FULL un-plussed
# class run of exactly ten Nd digits, or eleven with an ASCII leading
# '1', carrying at least one separator inside the match span, starting
# at a CLEAN boundary (the first char glued to neither `~` nor a lowercase
# a-f, the token-interior alphabet, unless a token span ends exactly
# there) — used only to route inputs between the parity lanes
# (conservative by design: an input this flags whose domestic shape the
# email pass then consumes still agrees between the engines, and is
# simply not asserted here). The clean-boundary clause is load-bearing:
# without it hex-glued runs (`job255128-4096`,
# `value~255-123-4567`) route to the extension lane even though both
# engines leave them untouched, hiding parity agreement. Token spans
# (`~` + 12 digest hex) are breakers, mirroring the phone scan: a run
# starting inside a digest resumes after the token, and the byte after a
# token is clean even when the digest ends hex-dirty — so an adjacent
# number routes to the extension lane instead of hiding as parity.
_TOKEN_INTERIOR = frozenset("~abcdef")
_TOKEN_HEX_LEN = 12
_DOMESTIC_SEPARATORS = "-. ()"


@cache
def _tors_is_nd_codepoint(cp: int) -> bool:
    """Whether tors's phone grammar treats `chr(cp)` as a digit,
    observed through tors itself (`+`-anchored probe) rather than the
    running interpreter's `unicodedata`: the crate pins Unicode 16.0.0
    while the interpreter may run an older UCD (80 codepoints of skew on
    3.12/13, 110 on 3.10), and the guard must follow the crate, not the
    interpreter. ASCII answers directly (the crate's fast path)."""
    if cp < 128:
        return chr(cp).isdigit()
    probe = "+" + chr(cp) * 8
    try:
        return tors.scrub_pii(probe, ["contact_phone"], salt="") != probe
    except UnicodeEncodeError:
        return False  # lone surrogates: no class contains them (pinned)


def _is_nd(c: str) -> bool:
    return _tors_is_nd_codepoint(ord(c))


def _is_token_hex(ch: str) -> bool:
    return "0" <= ch <= "9" or "a" <= ch <= "f"


def _token_span_end_at(text: str, tilde: int) -> int | None:
    """End offset of the token span opening at `tilde`, or None."""
    if tilde >= len(text) or text[tilde] != "~":
        return None
    end = tilde + 1 + _TOKEN_HEX_LEN
    if end > len(text):
        return None
    if all(_is_token_hex(ch) for ch in text[tilde + 1 : end]):
        return end
    return None


def _token_span_containing(text: str, off: int) -> int | None:
    """End offset of the token span strictly covering `off`, or None."""
    for tilde in range(max(0, off - _TOKEN_HEX_LEN), min(off, len(text) - 1) + 1):
        if text[tilde] != "~":
            continue
        end = _token_span_end_at(text, tilde)
        if end is not None and tilde <= off < end:
            return end
    return None


def _token_ends_at(text: str, pos: int) -> bool:
    """Whether a token span ends exactly at `pos` (a clean boundary)."""
    return (
        pos >= 1 + _TOKEN_HEX_LEN
        and text[pos - 1 - _TOKEN_HEX_LEN] == "~"
        and all(_is_token_hex(ch) for ch in text[pos - _TOKEN_HEX_LEN : pos])
    )


def has_domestic_shape(text: str) -> bool:
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "~":
            end = _token_span_end_at(text, i)
            if end is not None:
                i = end
                continue
        c = text[i]
        if _is_nd(c) or c in _DOMESTIC_SEPARATORS:
            run_start = i
            covering = _token_span_containing(text, run_start)
            if covering is not None:
                i = covering
                continue
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
                while k < n and text[k] == " ":
                    k += 1
                if k < n and any(ch in _DOMESTIC_SEPARATORS for ch in text[k:last_digit]):
                    clean = (
                        k == 0
                        or text[k - 1] not in _TOKEN_INTERIOR
                        or _token_ends_at(text, k)
                    )
                    if clean:
                        return True
            i = j
        else:
            i += 1
    return False


def _both_lane_has_domestic_shape(text: str, salt: str | None) -> bool:
    """The BOTH-lane router: the matcher runs on the EMAIL-PASS OUTPUT,
    not the input — the email local removal can trim a too-long digit run
    into exactly 10 (or 11-with-1), e.g. `1415 555 2671 12345a@b.co`
    (16 digits in, a phone shape out). Guarding the input alone routes
    those to the parity lane where tors scrubs and the oracle leaves.
    The email-pass spelling here is the reference transcription (the
    email grammar is parity-pinned), at the lane's own salt (digests
    shape the composition). Conservative on both sides: input-flagged
    shapes the email pass consumes still skip."""
    if has_domestic_shape(text):
        return True
    email_pass = reference_scrub_pii(text, ["contact_email"], salt=salt)
    return has_domestic_shape(email_pass)


# The api-key guard: whether tors's keys rule (the credential extension
# past the source's two-rule contact contract) would fire anywhere in
# `text`. A faithful test-side mirror of the landed grammar — the family
# table longest-prefix-first with fall-through, the maximal tail run in
# the family's own charset (the shared [A-Za-z0-9_-] for most, [0-9A-Z]
# for the AWS pair, [A-Za-z0-9+/=] for the Azure marker), the
# prefix-boundary rule (a prefix glued to a preceding key-charset char is
# mid-token, the `xak-` cut — the PEM carve-out aside: a `-----BEGIN ` head
# directly after a dash run IS a clean boundary, the previous block's
# `-----END …-----` close being armor, not a word), the JWT marker
# scoping, and the PEM span (both markers, same words) —
# used only to route inputs between the lanes, the same posture as
# `has_domestic_shape`. The scanner's first-byte dispatch is a pure
# optimization and is deliberately NOT mirrored: every family prefix is
# tried at every clean position, which is behaviorally identical and one
# less thing to drift.
_KEY_FAMILIES: tuple[tuple[str, int], ...] = (
    ("github_pat_", 22),
    ("sk-svcacct-", 20),
    ("AccountKey=", 40),
    ("sk-proj-", 20),
    ("sk-ant-", 20),
    ("azxdev_", 20),
    ("glpat-", 20),
    ("glagent-", 20),
    ("glsoat-", 20),
    ("glrtr-", 20),
    ("glcbt-", 20),
    ("glptt-", 20),
    ("glimt-", 20),
    ("gloas-", 20),
    ("glft-", 20),
    ("gldt-", 20),
    ("glrt-", 20),
    ("ya29.", 20),
    ("ghp_", 36),
    ("AIza", 35),
    ("xai-", 20),
    ("AKIA", 16),
    ("ASIA", 16),
    ("fw-", 20),
    ("fw_", 20),
    ("ak-", 20),
    ("wk-", 20),
    ("wd-", 43),
    ("cn-", 20),
    ("sk-", 20),
    ("w-", 43),
)
_KEY_TAIL = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)
# The two per-family tail alphabets past the shared charset: the AWS
# access-key ID alphabet (uppercase + digits — no lowercase anywhere in
# it) and the Azure connection-string secret alphabet (base64 plus the
# padding/trailing `=`). Every other prefix family — xai-, ya29., the
# JWT segments included — runs on the shared charset above.
_KEY_TAIL_AWS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_KEY_TAIL_AZURE = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="
)
_PEM_WORD_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
_BEARER_MARKER = "Bearer eyJ"


def _key_charset_for(prefix: str) -> frozenset[str]:
    """The tail class one table prefix scans: the AWS/Azure alphabets for
    their own markers, the shared charset for everything else — the one
    tail-class branch a 14th family with a new alphabet extends (a
    shared-charset family touches the table only)."""
    if prefix in ("AKIA", "ASIA"):
        return _KEY_TAIL_AWS
    if prefix == "AccountKey=":
        return _KEY_TAIL_AZURE
    return _KEY_TAIL


def _jwt_span_at(text: str, start: int) -> int | None:
    """The JWT family at one position: the `Bearer eyJ` marker plus three
    maximal `[A-Za-z0-9_-]+` segments, single-dot separated (the marker
    consumed the first segment's `eyJ` head, so the grammar needs at
    least one more charset char before the first dot — the degenerate
    `Bearer eyJ.a.b` is a non-match). The end offset, or None."""
    if not text.startswith(_BEARER_MARKER, start):
        return None
    i = start + len(_BEARER_MARKER)
    for seg in range(3):
        run_start = i
        while i < len(text) and text[i] in _KEY_TAIL:
            i += 1
        if i == run_start:
            return None
        if seg < 2:
            if i >= len(text) or text[i] != ".":
                return None
            i += 1
    return i


_PEM_BEGIN = "-----BEGIN "
_PEM_CLOSE = " PRIVATE KEY-----"


def _pem_span_at(text: str, start: int) -> int | None:
    """The PEM family at one position: `-----BEGIN <words> PRIVATE
    KEY-----`, any bytes (newlines included), then `-----END <the same
    words> PRIVATE KEY-----` — `<words>` one-or-more `[A-Za-z0-9]+`
    runs, single-space separated (an empty, doubled-space, or non-alnum
    spelling never opens a block). Both markers required: an
    unterminated BEGIN or mismatched END words is a non-match, and the
    whole block is the one span. The end offset, or None."""
    if not text.startswith(_PEM_BEGIN, start):
        return None
    words_at = start + len(_PEM_BEGIN)
    close = text.find(_PEM_CLOSE, words_at)
    if close < 0:
        return None
    words = text[words_at:close]
    parts = words.split(" ")
    if any(part == "" or any(c not in _PEM_WORD_CHARS for c in part) for part in parts):
        return None
    marker = "-----END " + words + _PEM_CLOSE
    end = text.find(marker, close + len(_PEM_CLOSE))
    if end < 0:
        return None
    return end + len(marker)


def has_api_key_shape(text: str) -> bool:
    n = len(text)
    for i in range(n):
        if i > 0 and text[i - 1] in _KEY_TAIL and not (
            # a `-----BEGIN ` head directly after a dash run is a clean
            # boundary: the previous block's `-----END …-----` close is
            # armor, not a word (the twin of pem_head_after_dash_run)
            text[i - 1] == "-" and text.startswith("-----BEGIN ", i)
        ):
            continue  # a mid-token prefix: the boundary rule
        for prefix, min_tail in _KEY_FAMILIES:
            if not text.startswith(prefix, i):
                continue
            charset = _key_charset_for(prefix)
            j = i + len(prefix)
            while j < n and text[j] in charset:
                j += 1
            if j - (i + len(prefix)) >= min_tail:
                return True
            # a too-short tail falls through to the shorter prefixes
        if _jwt_span_at(text, i) is not None:
            return True
        if _pem_span_at(text, i) is not None:
            return True
    return False


def _both_lane_diverges(text: str, salt: str | None) -> bool:
    """The BOTH-rules router, both extensions at once: key shapes guard
    on the INPUT (the keys pass runs BEFORE the email pass, so a key-free
    input leaves it the identity and the email-pass output the domestic
    guard inspects is the same both engines produce), and the domestic
    matcher still guards on the EMAIL-PASS OUTPUT per the existing
    design. Key-bearing rows pin in TestApiKeyExtension instead."""
    if has_api_key_shape(text):
        return True
    return _both_lane_has_domestic_shape(text, salt)


@pytest.mark.parametrize("salt", _SALT_LANES, ids=["default-salt", "unsalted", "custom-salt"])
@pytest.mark.parametrize("text", CORPUS, ids=[f"corpus-{i}" for i in range(len(CORPUS))])
def test_quoted_pin_parity(text: str, salt: str | None) -> None:
    """The CI oracle lane, every rule at the canonical order: every corpus
    shape at every salt spelling — tors's documented defaults (the
    per-rule contact/keys split), the
    unsalted source-parity spelling, and a caller secret. Domestic-bearing
    and key-bearing inputs route to their extension lanes (the source has
    no domestic or key grammar to be parity-checked against)."""
    if _both_lane_diverges(text, salt):
        pytest.skip("domestic or api-key shape: pinned in the extension lanes, not the parity lane")
    assert tors.scrub_pii(text, salt=salt) == reference_scrub_pii(text, salt=salt)


def _rules_lane_diverges(text: str, rules: list[str] | None) -> bool:
    """The rules-lane router: lanes where api_keys is active (None and
    any set naming it) route key shapes on the input — the keys pass runs
    before the email pass, so input-guarding suffices; BOTH-rules lanes
    (None and both contact orders) still route the domestic matcher on
    the email-pass output, the existing design; single-rule lanes keep
    the input guard (the identity lane skips nothing extra: the existing
    conservative posture)."""
    wanted = None if rules is None else set(rules)
    if (wanted is None or "api_keys" in wanted) and has_api_key_shape(text):
        return True
    if wanted is None or wanted == {"contact_email", "contact_phone"}:
        return _both_lane_has_domestic_shape(text, "")
    return has_domestic_shape(text)


@pytest.mark.parametrize("rules", _RULES_LANES, ids=lambda r: f"rules-{r}")
@pytest.mark.parametrize("text", CORPUS, ids=[f"corpus-{i}" for i in range(len(CORPUS))])
def test_quoted_pin_rules_parity(text: str, rules: list[str] | None) -> None:
    """The rules lanes over the unsalted spelling (the source chain's own
    digest): the canonical None order, both contact orders, each subset,
    and the identity — the last one pinning that the oracle and tors
    agree on doing nothing."""
    if _rules_lane_diverges(text, rules):
        pytest.skip("domestic or api-key shape: pinned in the extension lanes, not the parity lane")
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
        assume(not _both_lane_diverges(text, salt))
        assert tors.scrub_pii(text, salt=salt) == reference_scrub_pii(text, salt=salt)

    @pytest.mark.parametrize(
        "rules",
        [["contact_email"], ["contact_phone"], ["api_keys"], []],
        ids=["email", "phone", "keys", "identity"],
    )
    @given(text=_composed_text)
    @settings(max_examples=150, deadline=None)
    def test_rule_subsets_match_the_quoted_pin(self, text: str, rules: list[str]) -> None:
        assume(not _rules_lane_diverges(text, rules))
        assert tors.scrub_pii(text, rules, salt="") == reference_scrub_pii(text, rules, salt="")

    @given(text=_composed_text)
    @settings(max_examples=200, deadline=None)
    def test_scrubbing_twice_converges(self, text: str) -> None:
        twice = tors.scrub_pii(tors.scrub_pii(text, salt=""), salt="")
        assert tors.scrub_pii(twice, salt="") == twice


# --- Fuzz-invariant ports (fuzz/fuzz_targets/pii.rs, hypothesis spelling) ------
#
# The fuzz target asserts two structural invariants the differential lanes
# above do not: (a) no match of the INPUT survives verbatim into the
# converged (twice-scrubbed) output, and (b) the converged output is a
# fixed point (a third pass is the identity object). The tests below are
# the Python port of exactly those two asserts, over the same composed
# alphabet plus a domestic-seeded strategy (below) so the domestic
# extension is covered too.


def _input_contacts_present_in(text: str) -> list[str]:
    """Contact pieces of the zoo present verbatim in `text`."""
    return [p for p in list(SCRUB_PII_EMAILS) + list(SCRUB_PII_PHONES) if p in text]


class TestFuzzInvariantPorts:
    @given(text=_composed_text)
    @settings(max_examples=200, deadline=None)
    def test_no_input_match_survives_convergence(self, text: str) -> None:
        once = tors.scrub_pii(text, salt="")
        twice = tors.scrub_pii(once, salt="")
        # The converged output is a fixed point (fuzz: third pass Borrowed).
        assert tors.scrub_pii(twice, salt="") == twice
        # Any contact piece the first pass removed must stay gone
        # (fuzz: any_survivor over the converged output).
        for contact in _input_contacts_present_in(text):
            if contact not in once:
                assert contact not in twice, contact

    @given(text=_composed_text)
    @settings(max_examples=150, deadline=None)
    def test_phone_only_converges_without_survivors(self, text: str) -> None:
        once = tors.scrub_pii(text, ["contact_phone"], salt="")
        assert tors.scrub_pii(once, ["contact_phone"], salt="") == once
        for contact in [p for p in SCRUB_PII_PHONES if p in text]:
            if contact not in once:
                assert contact not in tors.scrub_pii(once, ["contact_phone"], salt="")


# Domestic-seeded convergence is defined after the extension lane
# (it seeds from _DOMESTIC_CASES); see TestDomesticSeededConvergence.


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
        # The token is the digest alone: a domestic match's head digits
        # are the area code, so the token keeps no prefix (the oracle's
        # `_scrub_phone_token` rule: the prefix only on a `+`-led
        # match).
        token = f"~{hashlib.sha256(matched.encode('utf-8')).hexdigest()[:12]}"
        assert text.count(matched) == 1
        assert tors.scrub_pii(text, salt="") == text.replace(matched, token, 1)

    @pytest.mark.parametrize(
        "text",
        _DOMESTIC_NON_MATCHES,
        ids=[f"shared-nonmatch-{i}" for i in range(len(_DOMESTIC_NON_MATCHES))],
    )
    def test_the_discipline_cuts_are_shared_non_matches(self, text: str) -> None:
        assert reference_scrub_pii(text, None, salt="") == text
        assert tors.scrub_pii(text, salt="") == text

    @pytest.mark.parametrize(
        "text",
        [
            "job255128-4096",
            "value~255-123-4567",
            "ref c415-555-2671",
            "ref a415-555-2671",
            "ref f415-555-2671",
            "x~e292cb255128 4096",
        ],
        ids=[
            "hex-glued-job",
            "tilde-glued-value",
            "hex-c-glued",
            "hex-a-glued",
            "hex-f-glued",
            "digest-tail-composition",
        ],
    )
    def test_hex_or_tilde_glued_runs_are_shared_non_matches(self, text: str) -> None:
        # The clean-boundary rule (H2): a domestic match's first char may
        # not be glued to neither `~` nor a lowercase a-f. Both engines leave these
        # untouched, and the guard must NOT route them to the extension
        # lane (it mirrors clean, so parity is asserted here, not skipped).
        assert not has_domestic_shape(text)
        assert reference_scrub_pii(text, None, salt="") == text
        assert tors.scrub_pii(text, salt="") == text

    @pytest.mark.parametrize(
        ("text", "matched"),
        [
            ("jobg415-555-2671", "415-555-2671"),
            ("jobz415-555-2671", "415-555-2671"),
            ("jobG415-555-2671", "415-555-2671"),
            ("jobA415-555-2671", "415-555-2671"),
            ("jobF415-555-2671", "415-555-2671"),
        ],
        ids=["g-clean", "z-clean", "G-clean", "A-clean", "F-clean"],
    )
    def test_non_hex_letters_are_clean_boundaries(self, text: str, matched: str) -> None:
        # H3: clean means exactly `~` + lowercase a-f are dirty. g/z and
        # uppercase A-F are clean, so the domestic shape still fires.
        assert has_domestic_shape(text)
        # Digest alone: a domestic token keeps no prefix (the head
        # digits are the area code).
        token = f"~{hashlib.sha256(matched.encode('utf-8')).hexdigest()[:12]}"
        assert tors.scrub_pii(text, salt="") == text.replace(matched, token, 1)


_ARABIC_ONE = chr(0x0661)


def _unsalted_token(prefix: str, matched: str) -> str:
    return f"{prefix}~{hashlib.sha256(matched.encode('utf-8')).hexdigest()[:12]}"


class TestEmailTokenAdjacentNumbers:
    """P0: the email pass emits `@domain~<12hex>`, and the digest tail
    must not swallow an adjacent domestic number (silent
    under-redaction). The token is a breaker — the number after it
    scrubs exactly — pinned in both directions (the oracle, having no
    domestic grammar, scrubs the email but leaves the number)."""

    _CASES: list[tuple[str, str, str]] = [
        # (input, the email address, the span tors scrubs)
        (
            "candidate ada+tag@azx.io 415-555-2671 no answer",
            "ada+tag@azx.io",
            "415-555-2671",
        ),
        (
            "fungai.chetima@example.com-415-555-2671",
            "fungai.chetima@example.com",
            "-415-555-2671",
        ),
    ]

    @pytest.mark.parametrize(
        ("text", "email", "matched"), _CASES, ids=["space-separated", "dash-glued"]
    )
    def test_the_quoted_oracle_leaves_the_number(self, text: str, email: str, matched: str) -> None:
        out = reference_scrub_pii(text, None, salt="")
        assert email not in out  # the email still scrubs
        assert matched in out  # but the number survives the oracle whole

    @pytest.mark.parametrize(
        ("text", "email", "matched"), _CASES, ids=["space-separated", "dash-glued"]
    )
    def test_tors_scrubs_the_number_exactly(self, text: str, email: str, matched: str) -> None:
        domain = email.split("@")[1]
        email_token = _unsalted_token(f"@{domain}", email)
        phone_token = _unsalted_token("", matched)
        assert tors.scrub_pii(text, salt="") == text.replace(email, email_token).replace(
            matched, phone_token
        )
        twice = tors.scrub_pii(tors.scrub_pii(text, salt=""), salt="")
        assert tors.scrub_pii(twice, salt="") == twice

    def test_unicode_nd_digits_after_a_token_scrub_exactly(self) -> None:
        # The non-ASCII spelling of the same corner: an email token whose
        # digest ends hex-dirty, glued to a dot-led ten-digit Nd run
        # (mathematical bold digits + one Arabic-Indic one).
        digits = "".join(chr(0x1D7D8 + i % 3) for i in range(9)) + _ARABIC_ONE
        text = f"a@b.co.{digits}"
        matched = f".{digits}"
        out = reference_scrub_pii(text, None, salt="")
        assert "a@b.co" not in out
        assert matched in out
        email_token = _unsalted_token("@b.co", "a@b.co")
        phone_token = _unsalted_token("", matched)
        assert tors.scrub_pii(text, salt="") == email_token + phone_token


class TestBothLaneTrimmedRuns:
    """P1: the email local removal trims a too-long digit run into exactly
    10 (or 11-with-1) — the matcher runs on the EMAIL-PASS OUTPUT, so
    the BOTH lanes route there. Pinned in both directions."""

    _CASES = ["1415 555 2671 12345a@b.co", "415 555 2671 123456a@b.co"]

    @pytest.mark.parametrize("text", _CASES)
    def test_the_input_guard_misses_but_the_email_pass_output_hits(self, text: str) -> None:
        assert not has_domestic_shape(text)
        assert _both_lane_has_domestic_shape(text, "")
        assert _both_lane_has_domestic_shape(text, None)

    @pytest.mark.parametrize("text", _CASES)
    def test_the_oracle_leaves_the_trimmed_shape_tors_scrubs_it(self, text: str) -> None:
        trimmed = text.split(" 123")[0]
        oracle = reference_scrub_pii(text, None, salt="")
        assert trimmed in oracle  # the oracle has no domestic grammar
        got = tors.scrub_pii(text, salt="")
        assert trimmed not in got  # tors scrubs the trimmed phone shape


class TestZeroLedShapes:
    """P2c: ten-digit runs carry no leading-digit check — a `0`-led
    ten-digit shape scrubs like any other; only the eleven-digit
    non-`1`-led spelling is excluded. Pinned in both directions."""

    def test_ten_digit_zero_led_scrubs(self) -> None:
        matched = "020-794-6095"
        assert reference_scrub_pii(matched, None, salt="") == matched
        assert tors.scrub_pii(matched, salt="") == _unsalted_token("", matched)

    def test_eleven_digit_zero_led_is_a_shared_non_match(self) -> None:
        text = "0-415-555-2671"
        assert reference_scrub_pii(text, None, salt="") == text
        assert tors.scrub_pii(text, salt="") == text


# Domestic-seeded convergence: compositions built from the extension
# lane's own cases (plus separators and re-fire glue), so hypothesis
# covers the domestic matcher structurally, not just via the generic
# alphabet above.
_DOMESTIC_SEEDS = [c[0] for c in _DOMESTIC_CASES] + [
    "415-555-2671, 415-555-2672",
    "(415) 555-2671 415-555-2672 +14155552673",
    "job255128-4096",
    "value~255-123-4567",
    "~",
    " ",
    ", ",
    # H3: the disputable lanes — plus-led (international territory, never
    # a domestic fallback), an email piece (the email pass eats `+`-led
    # locals first), a non-ASCII Nd domestic spelling (Arabic-Indic ten),
    # width-negative lanes (twelve- and nine-digit runs are ids, not
    # phones), and a leading-space spelling (word separation, skipped).
    "+14155552671",
    "+ (415) 555-2671",
    "user+14155552671@example.com",
    "call 415-555-2671 ok",  # leading-space-adjacent word separation
    " 415-555-2671",
    "\u0664\u0661\u0665 \u0665\u0665\u0665 \u0662\u0666\u0667\u0661",
    "415-555-267123",
    "415-555-267",
]
_domestic_composed = st.lists(
    st.sampled_from(_DOMESTIC_SEEDS), min_size=0, max_size=12
).map("".join)


class TestDomesticSeededConvergence:
    @given(text=_domestic_composed)
    @settings(max_examples=200, deadline=None)
    def test_domestic_compositions_converge(self, text: str) -> None:
        twice = tors.scrub_pii(tors.scrub_pii(text, salt=""), salt="")
        assert tors.scrub_pii(twice, salt="") == twice

    @given(text=_domestic_composed)
    @settings(max_examples=150, deadline=None)
    def test_domestic_phone_only_is_idempotent(self, text: str) -> None:
        once = tors.scrub_pii(text, ["contact_phone"], salt="")
        assert tors.scrub_pii(once, ["contact_phone"], salt="") == once


# --- Oracle freshness: the transcription pin, not a live contract ------------
#
# The CI oracle is a TRANSCRIPTION (tests/reference.py), and the live lane
# never runs in CI — so the transcription itself must carry its provenance
# (source revision/hash + transcription date + UCD) and CI must fail when
# it goes stale (N-day freshness) or when the interpreter's UCD moves past
# the pinned tables.


# --- The opt-in live re-sync lane ---------------------------------------------------
#
# Re-sync cadence / owner: the quoted pin in tests/reference.py is the CI
# oracle; the live module is re-checked manually on a machine that holds
# it (a) whenever the source telemetry-safety module changes grammar or
# token shape, and (b) at least once per Unicode/dependency bump that
# could move the Nd table (the UCD CPython's `re` matches on). Owner: the
# scrub_pii maintainer for this repo. A grammar change upstream is a
# parity re-sync request (update reference.py + the parity corpus), never
# a drive-by grammar widening here — the `salt=""` byte-identical
# contract forbids silent widening.
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
            if _both_lane_diverges(text, ""):
                continue  # the extension lanes' territory; the live module has neither grammar
            assert tors.scrub_pii(text, salt="") == live(text), text

    @given(text=_composed_text)
    @settings(max_examples=300, deadline=None)
    def test_compositions_match_the_live_module(self, text: str) -> None:
        assume(not _both_lane_diverges(text, ""))
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


class TestOracleFreshness:
    """H1: the CI oracle is a transcription, and the live lane never runs
    in CI — so the transcription carries its provenance and CI fails when
    it goes stale. Two independent trip-wires: (a) the transcription date
    is at most _ORACLE_FRESH_DAYS old (a re-sync clock, not a grammar
    check), and (b) the running interpreter's UCD still equals the UCD
    the Rust Nd tables pin (a Unicode/dependency bump that could move
    the `re` digit class re-opens the re-sync). The UCD pin is strict
    only where it can fire: interpreters at or past the pinned UCD must
    match it exactly (a newer UCD is a re-sync request), while older legs
    skip explicitly (their UCD predates the tables by construction, and
    the behavioral lanes — the Nd-exhaustive and parity suites — run
    unskipped everywhere)."""

    def test_provenance_constants_exist(self) -> None:
        import reference

        assert isinstance(reference.SCRUB_PII_ORACLE_REVISION, str)
        assert reference.SCRUB_PII_ORACLE_REVISION
        assert isinstance(reference.SCRUB_PII_ORACLE_DATE, str)
        assert isinstance(reference.SCRUB_PII_ORACLE_UCD, str)

    def test_transcription_is_fresh(self) -> None:
        from datetime import date

        import reference

        today = date.today()
        age = today - date.fromisoformat(reference.SCRUB_PII_ORACLE_DATE)
        assert age.days <= reference.SCRUB_PII_ORACLE_FRESH_DAYS, (
            f"scrub_pii oracle transcription is {age.days} days old "
            f"(limit {reference.SCRUB_PII_ORACLE_FRESH_DAYS}): re-sync against "
            "the live telemetry-safety module and bump SCRUB_PII_ORACLE_DATE"
        )

    def test_interpreter_ucd_matches_pinned_tables(self) -> None:
        import reference

        pinned = reference.SCRUB_PII_ORACLE_UCD
        running = unicodedata.unidata_version
        running_t = tuple(int(p) for p in running.split("."))
        pinned_t = tuple(int(p) for p in pinned.split("."))
        if running_t < pinned_t:
            pytest.skip(
                f"interpreter UCD {running} < pinned {pinned}: older leg "
                "predates the tables by construction; the behavioral lanes "
                "(Nd-exhaustive, parity) run unskipped everywhere"
            )
        assert running == pinned, (
            f"interpreter UCD {running} != pinned "
            f"{pinned}: re-sync the scrub_pii oracle "
            "(the Nd table the `re` digit class matches on may have moved)"
        )


def _fuzz_domestic_match_present(text: str) -> bool:
    """H2 second mirror: an independent transcription of the fuzz target's
    `phone_matches_of` domestic branch (fuzz/fuzz_targets/pii.rs) — char
    space, per-position, greedy runs spent whole, leading spaces skipped,
    separator required, never behind `+`, clean boundary `~`/a-f (or a
    token span ending exactly there), token spans skipped whole. The
    parity guard `has_domestic_shape` must agree with it exactly; a drift
    on either side fails loudly instead of hiding parity agreement. The
    digit predicate is the same tors-observed `_is_nd` the guard uses."""
    seps = set("-. ()")
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "~":
            end = _token_span_end_at(text, i)
            if end is not None:
                i = end
                continue
        c = text[i]
        if _is_nd(c) or c in seps:
            run_start = i
            covering = _token_span_containing(text, run_start)
            if covering is not None:
                i = covering
                continue
            j = i
            digits = 0
            first_digit: str | None = None
            last_digit = -1
            while j < n and (_is_nd(text[j]) or text[j] in seps):
                if _is_nd(text[j]):
                    digits += 1
                    if first_digit is None:
                        first_digit = text[j]
                    last_digit = j
                j += 1
            plussed = run_start > 0 and text[run_start - 1] == "+"
            if not plussed and (digits == 10 or (digits == 11 and first_digit == "1")):
                k = run_start
                while k < n and text[k] == " ":
                    k += 1
                if k < n and any(ch in seps for ch in text[k : last_digit + 1]):
                    clean = (
                        k == 0
                        or text[k - 1] not in _TOKEN_INTERIOR
                        or _token_ends_at(text, k)
                    )
                    if clean:
                        return True
            i = j
        else:
            i += 1
    return False


class TestDomesticGuardMirror:
    """H2: the parity guard is itself pinned — `has_domestic_shape` must
    agree with the fuzz target's independent domestic matcher over every
    corpus input plus the domestic seeds. Two mirrors, one grammar."""

    @pytest.mark.parametrize("text", CORPUS, ids=[f"corpus-{i}" for i in range(len(CORPUS))])
    def test_guard_agrees_with_fuzz_mirror_over_corpus(self, text: str) -> None:
        assert has_domestic_shape(text) == _fuzz_domestic_match_present(text), text

    @given(text=_domestic_composed)
    @settings(max_examples=150, deadline=None)
    def test_guard_agrees_with_fuzz_mirror_over_seeds(self, text: str) -> None:
        assert has_domestic_shape(text) == _fuzz_domestic_match_present(text), text

    def test_guard_nd_follows_tors_across_scripts(self) -> None:
        """P2d: the guard's digit predicate is tors-observed, not the
        interpreter's `unicodedata` (the crate pins Unicode 16.0.0 while
        older interpreters run older UCDs) — spot-pinned per script,
        with the non-Nd numerics and the surrogate boundary."""
        for ch in ["0", "9", chr(0x0661), chr(0xFF19), chr(0x0968), chr(0x1D7CE), chr(0x1FBF7)]:
            assert _is_nd(ch), f"U+{ord(ch):04X}"
        for ch in ["a", "~", "+", " ", chr(0x00B2), chr(0x2169), chr(0xFF0B)]:
            assert not _is_nd(ch), f"U+{ord(ch):04X}"
        assert not _is_nd("\ud800")


class TestDomesticSeedCoverage:
    """H3: the domestic seeds must span the disputable lanes — plus-led,
    @-bearing, non-ASCII Nd, width-negative, and leading-space — so the
    seeded convergence lanes cannot pass while blind to them."""

    def test_seeds_cover_the_disputable_lanes(self) -> None:
        assert any("+" in s for s in _DOMESTIC_SEEDS), "no plus-led seed"
        assert any("@" in s for s in _DOMESTIC_SEEDS), "no email/@ seed"
        assert any(
            any(unicodedata.category(c) == "Nd" and not c.isascii() for c in s)
            for s in _DOMESTIC_SEEDS
        ), "no non-ASCII Nd seed"
        assert any(
            sum(unicodedata.category(c) == "Nd" for c in s) == 12 for s in _DOMESTIC_SEEDS
        ), "no 12-digit width-negative seed"
        assert any(s.startswith(" ") or s.startswith("  ") for s in _DOMESTIC_SEEDS), (
            "no leading-space seed"
        )


# --- The api-keys extension lane: the keys rule past the source -----------------
#
# The source's contract is two contact rules; tors's api_keys rule is the
# credential extension, pinned the way the domestic matcher is: the
# quoted oracle leaves every key shape whole (it has no key grammar), and
# tors scrubs exactly the span — both directions named, so a regression
# on either side fails as itself instead of surfacing as a parity
# mystery. The key material here is independently spelled from
# tests/test_scrub_pii.py's zoo (a rot-13 offset into the same 62-char
# alphabet: different bytes, the same charset class — and the AWS/Azure
# tails a rot-13 offset into their own alphabets, the same posture), the
# file's independent-transcription posture. The five newer families (the
# AWS pair, xai-, ya29., the PEM span, the Azure marker) ride the same
# table, and `TestApiKeyFamilyRouting` below pins the `families=` mask
# over the closed set (selected redacts, unselected preserves whole).

_KEYS_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _key_tail(n: int) -> str:
    return "".join(_KEYS_ALPHABET[(i + 13) % 62] for i in range(n))


# The AWS access-key tail: uppercase letters and digits only ([0-9A-Z] —
# the access-key ID alphabet, no lowercase anywhere in it), rot-13-offset
# into its own 36-char alphabet (different bytes from the battery's
# offset-zero spelling, same class).
_KEY_AWS_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _key_aws_tail(n: int) -> str:
    return "".join(_KEY_AWS_ALPHABET[(i + 13) % 36] for i in range(n))


# The Azure storage-key tail: the connection-string secret alphabet
# ([A-Za-z0-9+/=] — base64 plus the padding/trailing `=`), the same
# rot-13-offset posture into its 65-char alphabet.
_KEY_AZURE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="
)


def _key_azure_tail(n: int) -> str:
    return "".join(_KEY_AZURE_ALPHABET[(i + 13) % 65] for i in range(n))


# PEM material: a multi-line SPAN family, raised by its own block builder
# (body lines independently spelled from the battery's, same base64-line
# shape). The body is base64 lines (no separators, no `@`), so an
# unterminated block stays identity under the contact passes too — the
# near-miss pins below demand it.
_PEM_BODY = (
    "MIIBoQIBAAJBAKzv",
    "tCx2DeFgHiJkLmNoP",
    "qRsTuVwXyZ012345",
)


def _pem_block(words: str, body: tuple[str, ...] = _PEM_BODY) -> str:
    lines = [f"-----BEGIN {words} PRIVATE KEY-----", *body, f"-----END {words} PRIVATE KEY-----"]
    return "\n".join(lines)


_PEM_EC = _pem_block("EC")

# The tail builder per tail class: the one branch a 14th family with a
# new alphabet extends, next to the guard's `_key_charset_for`.
_KEY_TAIL_BUILDERS = {"std": _key_tail, "aws": _key_aws_tail, "azure": _key_azure_tail}


def _key_case(shape: str, prefix: str, n: int, kind: str) -> tuple[str, str, str]:
    key = shape + _KEY_TAIL_BUILDERS[kind](n)
    return (key, key, prefix)


_KEYS_JWT = (
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)

# (the key's literal shape, the token's family prefix, the tail length, the
# tail class): every prefix family at its own floor or a realistic
# multiple — the AWS pair at their 16 floor (uppercase-only tails), xai-
# and ya29. at their 20 floor, the Azure marker at its 40 floor — so the
# minimums pin exactly, not loosely.
_KEYS_TABLE: tuple[tuple[str, str, int, str], ...] = (
    ("sk-", "sk-", 48, "std"),
    ("sk-proj-", "sk-proj-", 48, "std"),
    ("sk-svcacct-", "sk-svcacct-", 48, "std"),
    ("sk-ant-api03-", "sk-ant-", 95, "std"),
    ("AIza", "AIza", 35, "std"),
    ("fw-", "fw-", 48, "std"),
    ("fw_", "fw_", 48, "std"),
    ("ak-", "ak-", 48, "std"),
    ("wk-", "wk-", 48, "std"),
    ("ghp_", "ghp_", 36, "std"),
    ("github_pat_", "github_pat_", 22, "std"),
    ("azxdev_", "azxdev_", 20, "std"),
    ("wd-", "wd-", 43, "std"),
    ("w-", "w-", 43, "std"),
    ("cn-", "cn-", 20, "std"),
    ("AKIA", "AKIA", 16, "aws"),
    ("ASIA", "ASIA", 16, "aws"),
    ("xai-", "xai-", 20, "std"),
    ("ya29.", "ya29.", 20, "std"),
    ("AccountKey=", "AccountKey=", 40, "azure"),
)

# (input, the span tors scrubs, the token's family prefix): every family
# at its own shape, the JWT, the PEM span block, the keys-before-phone
# order (the dash-separated ten-digit run inside the tail — the phone
# pass must see only the token), and a key embedded in error prose.
_KEYS_CASES: list[tuple[str, str, str]] = [
    _key_case(shape, prefix, n, kind) for shape, prefix, n, kind in _KEYS_TABLE
] + [
    (_KEYS_JWT, _KEYS_JWT, "Bearer"),
    (
        f"leaked sk-proj-415-555-2671{_key_tail(20)} in an error",
        f"sk-proj-415-555-2671{_key_tail(20)}",
        "sk-proj-",
    ),
    (_PEM_EC, _PEM_EC, "PEM"),
]

_KEYS_NON_MATCHES: list[str] = [
    # The grammar cuts, shared by BOTH engines: one-under tails at every
    # distinct minimum, the uppercase spelling, the bare prefix, the
    # mid-token prefix, and the unmarked/degenerate JWT spellings — plus
    # the five newer families' own cuts: one-under AWS/xai/gcp/Azure
    # tails, the lowercase (or uppercase, where the family is lowercase)
    # spellings, the mid-token markers, and the unterminated, mismatched,
    # empty-words, lowercase, and glued PEM blocks.
    "sk-" + _key_tail(19),
    "sk-",
    "SKI-" + _key_tail(48),
    "AIza" + _key_tail(34),
    "ghp_" + _key_tail(35),
    "github_pat_" + _key_tail(21),
    "azxdev_" + _key_tail(19),
    "wd-" + _key_tail(42),
    "w-" + _key_tail(42),
    "cn-" + _key_tail(19),
    "fw-" + _key_tail(19),
    "ak-" + _key_tail(19),
    "xak-" + _key_tail(48),
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKx",
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.c2ln",
    "Bearer eyJ.a.b.c",
    "AKIA" + _key_aws_tail(15),
    "akia" + _key_aws_tail(16),
    "ASIA" + _key_aws_tail(15),
    "x" + "AKIA" + _key_aws_tail(16),
    "xai-" + _key_tail(19),
    "XAI-" + _key_tail(20),
    "ya29." + _key_tail(19),
    "YA29." + _key_tail(20),
    "xya29." + _key_tail(20),
    "\n".join(["-----BEGIN EC PRIVATE KEY-----", *_PEM_BODY]),
    _pem_block("EC").replace(
        "-----END EC PRIVATE KEY-----", "-----END RSA PRIVATE KEY-----"
    ),
    "-----BEGIN PRIVATE KEY-----\n" + "\n".join(_PEM_BODY) + "\n-----END PRIVATE KEY-----",
    "-----begin ec private key-----\n" + "\n".join(_PEM_BODY) + "\n-----end ec private key-----",
    "abc" + _PEM_EC,
    "AccountKey=" + _key_azure_tail(39),
    "Accountkey=" + _key_azure_tail(44),
    "xAccountKey=" + _key_azure_tail(40),
]


class TestApiKeyExtension:
    @pytest.mark.parametrize(
        ("text", "matched", "prefix"),
        _KEYS_CASES,
        ids=[f"{c[2]}-{i}" for i, c in enumerate(_KEYS_CASES)],
    )
    def test_the_quoted_oracle_leaves_the_shape(self, text: str, matched: str, prefix: str) -> None:
        # The oracle is the two-rule contact contract: no key grammar, no
        # domestic grammar — every key shape (the digit-bearing tail
        # included) survives it whole.
        assert reference_scrub_pii(text, None, salt="") == text

    @pytest.mark.parametrize(
        ("text", "matched", "prefix"),
        _KEYS_CASES,
        ids=[f"{c[2]}-{i}" for i, c in enumerate(_KEYS_CASES)],
    )
    def test_tors_scrubs_exactly_the_span(self, text: str, matched: str, prefix: str) -> None:
        # The unsalted digest, derived independently (hashlib): the family
        # prefix verbatim over the digest of the FULL matched key. The
        # phone-bearing key pins the pass order — the whole key is one
        # span, the phone pass never sees the digit run — and every
        # output converges.
        token = f"{prefix}~{hashlib.sha256(matched.encode('utf-8')).hexdigest()[:12]}"
        assert text.count(matched) == 1
        once = tors.scrub_pii(text, salt="")
        assert once == text.replace(matched, token, 1)
        twice = tors.scrub_pii(once, salt="")
        assert tors.scrub_pii(twice, salt="") == twice

    @pytest.mark.parametrize(
        "text",
        _KEYS_NON_MATCHES,
        ids=[f"key-nonmatch-{i}" for i in range(len(_KEYS_NON_MATCHES))],
    )
    def test_the_grammar_cuts_are_shared_non_matches(self, text: str) -> None:
        # Both engines leave these untouched, and the guard must NOT
        # route them to the extension lane (it mirrors the boundary rule
        # and the minimums exactly), so parity is asserted here, never
        # skipped.
        assert not has_api_key_shape(text)
        assert reference_scrub_pii(text, None, salt="") == text
        assert tors.scrub_pii(text, salt="") == text

    def test_the_guard_routes_every_key_case(self) -> None:
        # The mirror must agree with the landed scanner's routing: every
        # extension case (the embedded-in-prose and phone-bearing
        # spellings included) flags, so the parity lanes skip exactly
        # what this lane pins.
        for text, _matched, _prefix in _KEYS_CASES:
            assert has_api_key_shape(text), text

    def test_the_guard_agrees_with_the_scanner_over_cases_and_cuts(self) -> None:
        # The shared-miss killer: for every extension case and every
        # grammar cut, the guard's verdict must equal the scanner's own
        # (scrubbed vs whole) under families=None — if both miss a
        # shape, one side still disagrees with the scrub outcome here.
        for text, _matched, _prefix in _KEYS_CASES:
            assert has_api_key_shape(text), text
            assert tors.scrub_pii(text, ["api_keys"], salt="") != text, text
        for text in _KEYS_NON_MATCHES:
            assert not has_api_key_shape(text), text
            assert tors.scrub_pii(text, ["api_keys"], salt="") == text, text

    def test_the_corpus_is_key_free_so_the_parity_lanes_assert_it(self) -> None:
        # The existing quoted-pin lanes stay meaningful only while the
        # corpus (and the composed hypothesis alphabet's pieces) carry no
        # key shape: a key-bearing row would silently SKIP instead of
        # asserting parity. Pinned so a future corpus edit that adds one
        # fails loudly here instead of quietly thinning the differential.
        # (`+`, hex, and the zoo's letters cannot compose a family prefix
        # — no piece ends in a family head's lead chars — and the
        # hypothesis lanes route composed exceptions out with the guard.)
        for row in CORPUS:
            assert not has_api_key_shape(row), row
        for piece in _PIECES:
            assert not has_api_key_shape(piece), piece


# --- The family-selection routing: the `families=` mask over the closed set --
#
# `families=None` is all thirteen families; a recipe names its subset.
# The scanner still walks longest-prefix-first and the FIRST family whose
# grammar HOLDS wins the span: selected, it redacts; unselected, the span
# is spent whole and preserved verbatim (counted as skipped, no redaction
# inside it); a family whose grammar FAILS (a too-short tail) falls
# through to shorter prefixes as today. So with `families=[one]`, inputs
# bearing only other families are parity-assertable directly — tors and
# the oracle both leave them whole — and the selected family's own shapes
# pin in the extension lane. The routing tables below derive from the one
# case table above, so the 15th family joins them by joining it.
_ALL_KEY_FAMILIES: tuple[str, ...] = (
    "openai",
    "anthropic",
    "google",
    "fireworks",
    "modal",
    "github",
    "minted",
    "jwt",
    "aws",
    "xai",
    "gcp_oauth",
    "pem",
    "azure",
    "gitlab",
)

# Token prefix -> recipe family: the new families' markers are
# unambiguous (AKIA/ASIA, xai-, ya29., the PEM block, AccountKey=) and
# the JWT is its own recipe; the old prefixes' grouping follows the
# recipe's names as read. Only the new-family selections below are
# exercised, so an old case is "other" under every one of them no matter
# which old name it carries — a regroup of the old names cannot redden a
# routing pin, only the registry pin above names the set.
_KEY_TOKEN_FAMILY: dict[str, str] = {
    "sk-": "openai",
    "sk-proj-": "openai",
    "sk-svcacct-": "openai",
    "sk-ant-": "anthropic",
    "AIza": "google",
    "fw-": "fireworks",
    "fw_": "fireworks",
    "ak-": "modal",
    "wk-": "modal",
    "ghp_": "github",
    "gho_": "github",
    "ghu_": "github",
    "ghs_": "github",
    "ghr_": "github",
    "glpat-": "gitlab",
    "github_pat_": "github",
    "azxdev_": "minted",
    "wd-": "minted",
    "w-": "minted",
    "cn-": "minted",
    "Bearer": "jwt",
    "AKIA": "aws",
    "ASIA": "aws",
    "xai-": "xai",
    "ya29.": "gcp_oauth",
    "PEM": "pem",
    "AccountKey=": "azure",
}

# The selections this lane exercises: each new family alone (its own
# shapes pin, everything else passes through) and the all-but-jwt recipe
# (the JWT passes through).
_NEW_FAMILIES: tuple[str, ...] = ("aws", "xai", "gcp_oauth", "pem", "azure")
_ALL_BUT_JWT: list[str] = [f for f in _ALL_KEY_FAMILIES if f != "jwt"]

# (selecting family, input, span, token prefix): every case whose family
# is NOT the selection passes through whole ...
_ROUTING_UNSELECTED: list[tuple[str, str, str, str]] = [
    (family, text, matched, prefix)
    for family in _NEW_FAMILIES
    for text, matched, prefix in _KEYS_CASES
    if _KEY_TOKEN_FAMILY[prefix] != family
]
# ... and every case whose family IS the selection scrubs exactly.
_ROUTING_SELECTED: list[tuple[str, str, str, str]] = [
    (family, text, matched, prefix)
    for family in _NEW_FAMILIES
    for text, matched, prefix in _KEYS_CASES
    if _KEY_TOKEN_FAMILY[prefix] == family
]


class TestApiKeyFamilyRouting:
    def test_the_registry_lists_the_closed_set(self) -> None:
        # getattr, not a top-level import: before the core lands there is
        # no such name, and the empty default fails this equality — the
        # correct red, in one test, not at collection.
        assert set(getattr(tors, "KEY_FAMILIES", ())) == set(_ALL_KEY_FAMILIES)

    @pytest.mark.parametrize(
        ("family", "text", "matched", "prefix"),
        _ROUTING_UNSELECTED,
        ids=[f"{f}-leaves-{p}-{k}" for k, (f, _t, _m, p) in enumerate(_ROUTING_UNSELECTED)],
    )
    def test_unselected_families_leave_the_shape_whole(
        self, family: str, text: str, matched: str, prefix: str
    ) -> None:
        # With `families=[one new family]`, every other family's shape is
        # parity-assertable directly: tors spends the span whole and
        # preserves it verbatim (the keys-only lane, so the contact passes
        # cannot touch the phone-bearing key either), and the oracle — no
        # key grammar under any lane — leaves it whole. Asserted equal,
        # never skipped. (Red until the core lands: `families=` is still
        # a TypeError there.)
        assert has_api_key_shape(text)  # non-vacuous: the all-family guard flags it
        assert reference_scrub_pii(text, ["api_keys"], salt="") == text
        got = tors.scrub_pii(text, ["api_keys"], salt="", families=[family])
        assert got == text
        assert got == reference_scrub_pii(text, ["api_keys"], salt="")

    @pytest.mark.parametrize(
        ("family", "text", "matched", "prefix"),
        _ROUTING_SELECTED,
        ids=[f"{f}-scrubs-{p}" for f, _t, _m, p in _ROUTING_SELECTED],
    )
    def test_the_selected_family_scrubs_exactly_the_span(
        self, family: str, text: str, matched: str, prefix: str
    ) -> None:
        # The other side of the mask: the selecting recipe redacts exactly
        # the span — the family prefix verbatim over the digest of the
        # FULL match, PEM block newlines included. (Red until the core
        # lands, same TypeError.)
        token = _unsalted_token(prefix, matched)
        assert text.count(matched) == 1
        assert tors.scrub_pii(text, ["api_keys"], salt="", families=[family]) == text.replace(
            matched, token, 1
        )

    def test_all_but_jwt_leaves_a_jwt_whole(self) -> None:
        # The all-but-jwt recipe's routing: jwt unselected, its span spent
        # whole and preserved verbatim — parity-assertable directly, both
        # engines leaving it whole. (Red until the core lands.)
        assert has_api_key_shape(_KEYS_JWT)
        assert reference_scrub_pii(_KEYS_JWT, ["api_keys"], salt="") == _KEYS_JWT
        got = tors.scrub_pii(_KEYS_JWT, ["api_keys"], salt="", families=_ALL_BUT_JWT)
        assert got == _KEYS_JWT
        assert got == reference_scrub_pii(_KEYS_JWT, ["api_keys"], salt="")
