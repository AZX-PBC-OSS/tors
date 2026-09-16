"""The domestic phone token's prefix contract: a correlation token keeps
only the NON-IDENTIFYING dialling prefix, never digits of the number.

The scrub's token shape is ``prefix~<12 hex>`` where the prefix is the
match's dialling prefix (the first three code points of a ``+``-led
canonical E.164: the country code, ``"+47"`` compact, ``"+1 "`` for the
international spelling of a NANP number, where the third code point is
the space), and any other spelling gets the digest alone. That rule is
the ported source's own, verbatim: ``reference._scrub_phone_token``
emits the prefix only when the match starts with ``+``. The domestic
NANP extension (an un-plussed run of exactly ten digits, or eleven with
an ASCII leading ``1``, carrying a separator) widens the MATCHER, not
the token rule: a domestic match has no dialling prefix to keep (its
head digits are the AREA CODE, the NPA, the identifying half), so its
token is the digest alone.

The failure this file pins: the extension kept the ``+``-led spelling's
"first three code points" rule for domestic matches, so
``scrub_pii("call 415-555-2671", ["contact_phone"])`` answered
``"call 415~<hex>"``: three of the ten digits (the area code) survive
in the clear, though the docs promise the token keeps only the
non-identifying prefix and the stub still describes the phone rule as
``+``-led only. Three claims must agree, and each is pinned here:

- the behavior: a domestic token is ``~<digest>``, the reference
  oracle's own token for a non-``+``-led match, byte-exact at the
  default salt (``_scrub_phone_token``), with the digest taken over the
  exact matched span (leading separators absorbed, leading spaces
  skipped, extensions left past the match);
- the invariant behind it: no digit of the number survives before the
  token's ``~`` (``\\d+~`` never matches the scrubbed text);
- the docs: the stub's phone-rule description must name the domestic
  matcher instead of advertising ``+``-led-only matching (the stale
  restrictive phrase gone), and ``docs/api.md``'s token definition must
  state the digest-alone rule for non-``+``-led matches.

The ``+``-led international behavior is pinned non-regression: it never
had the bug (its first three code points ARE the dialling prefix) and
must not change when the domestic prefix is fixed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import tors
from reference import (
    SCRUB_PII_DEFAULT_SALT,
    _scrub_phone_token,
    reference_scrub_pii,
)

REPO = Path(__file__).resolve().parents[1]
API_MD = REPO / "docs" / "api.md"
PYI = REPO / "python" / "tors" / "__init__.pyi"

# The domestic NANP shapes the extension matches, each spelled as the
# exact matched span (verified by digest arithmetic: the token's digest
# is taken over precisely this text). Leading spaces are skipped (word
# separation, not part of the match); a leading structural separator is
# ABSORBED into the match; the trunk-prefix spellings (eleven digits
# with the ASCII leading 1) match too.
DOMESTIC_MATCHES = (
    "415-555-2671",
    "(415) 555-2671",
    "415.555.2671",
    "415 555 2671",
    "1-415-555-2671",
    "1 (415) 555-2671",
)

INTERNATIONAL = (
    "+14155552671",
    "+1 (415) 555-2671",
    "+1 415 555 2671",
    "+4712345678",
    "+44 20 7946 0958",
)


def _scrubbed_line(match: str) -> str:
    return tors.scrub_pii(f"call {match} now", ["contact_phone"])


# --- The behavior: a domestic token is the digest alone --------------------


@pytest.mark.parametrize("match", DOMESTIC_MATCHES)
def test_a_domestic_token_is_the_digest_alone(match: str) -> None:
    """The token for a domestic match is exactly the reference oracle's
    token for that match: the digest alone (no prefix: the match's head
    digits are the area code, the identifying half). Byte-exact at the
    default salt, digest over the exact matched span."""
    out = _scrubbed_line(match)
    expected = f"call {_scrub_phone_token(match, SCRUB_PII_DEFAULT_SALT)} now"
    assert out == expected, f"scrubbing {match!r}: {out!r} != {expected!r}"


@pytest.mark.parametrize("match", DOMESTIC_MATCHES)
def test_no_digit_of_the_number_survives_before_the_token_separator(
    match: str,
) -> None:
    """The invariant behind the token shape: scrubbed text never carries
    ``<digits>~`` (a prefix of digits ahead of a token separator is a
    piece of the number, the area code at the match's head, surviving
    in the clear). The digest is hex and follows its own separator, so
    any ``\\d+~`` hit is leaked number material by construction."""
    out = _scrubbed_line(match)
    assert not re.search(r"\d+~", out), (
        f"scrubbing {match!r} leaked digit(s) into the token prefix: {out!r}"
    )


def test_leading_spaces_are_skipped_and_extensions_survive_the_match() -> None:
    """The domestic matcher's span discipline is unchanged by the prefix
    fix: leading spaces are word separation (not match material), and an
    extension past the last digit survives untouched, so the digest
    input, and therefore the token, is the bare number's."""
    out = tors.scrub_pii("call   415-555-2671 x1234", ["contact_phone"])
    expected = (
        f"call   {_scrub_phone_token('415-555-2671', SCRUB_PII_DEFAULT_SALT)} x1234"
    )
    assert out == expected


def test_an_absorbed_leading_separator_never_leaks_a_digit() -> None:
    """A leading structural separator is absorbed INTO the match (the
    documented span rule), so the domestic token keeps none of it: the
    digest alone, over the match that begins at the separator."""
    out = tors.scrub_pii("call (415) 555-2671", ["contact_phone"])
    assert out == f"call {_scrub_phone_token('(415) 555-2671', SCRUB_PII_DEFAULT_SALT)}"
    assert not re.search(r"\d+~", out)


# --- The invariant's scope: `+`-led matches keep their dialling prefix -----


@pytest.mark.parametrize("number", INTERNATIONAL)
def test_international_tokens_keep_their_dialling_prefix(number: str) -> None:
    """Non-regression: a ``+``-led match's first three code points are
    the country code (coarse, operational, non-identifying) and stay in
    the token, the reference oracle's rule for ``+``-led matches, which
    the domestic fix must not touch. Oracle parity, both spellings of
    the NANP international form and two other country codes."""
    out = tors.scrub_pii(f"ring {number} twice", ["contact_phone"])
    assert out == reference_scrub_pii(f"ring {number} twice", ["contact_phone"])
    prefix = number[:3]
    assert out.startswith(f"ring {prefix}~"), out


def test_the_dialling_prefix_is_never_a_digit_run() -> None:
    """The international prefix the token keeps is dialling code
    material (``+`` + digits + separator), never a bare digit run: any
    digit run ahead of a token's ``~`` is introduced by the ``+`` (the
    compact spellings' own country code, ``+14~``), so the leak
    invariant (no BARE digit run before the separator, the domestic
    leak shape) holds internationally too."""
    for number in INTERNATIONAL:
        out = tors.scrub_pii(f"ring {number} twice", ["contact_phone"])
        assert not re.search(r"(?<![\d+])\d+~", out), (number, out)


# --- Token discipline is unchanged ------------------------------------------


def test_phone_only_stays_idempotent_with_domestic_tokens() -> None:
    """Scrub-twice convergence survives the prefix fix: a ``~<digest>``
    token (with or without a ``+``-led prefix) is a fixed point of the
    phone pass (the token span is a breaker, its digest can never
    compose a fresh match)."""
    for match in DOMESTIC_MATCHES:
        once = tors.scrub_pii(f"call {match} now", ["contact_phone"])
        twice = tors.scrub_pii(once, ["contact_phone"])
        assert twice == once, (match, once, twice)


def test_the_report_spelling_and_the_text_spelling_agree() -> None:
    """The report twin scrubs the same text byte-exactly for the same
    arguments, the domestic shapes included."""
    for match in DOMESTIC_MATCHES:
        text = f"call {match} now"
        report = tors.scrub_pii_report(text, ["contact_phone"])
        assert report["text"] == tors.scrub_pii(text, ["contact_phone"])


def test_unicode_nd_digits_get_the_same_prefixless_token() -> None:
    """The digit class is Unicode Nd (every decimal-digit script): a
    domestic number spelled in Arabic-Indic digits scrubs to the digest
    alone too; no Nd digit survives in the prefix."""
    arabic = "\u0664\u0661\u0665-\u0665\u0665\u0665-\u0662\u0666\u0667\u0661"
    out = tors.scrub_pii(f"call {arabic} now", ["contact_phone"])
    assert not re.search(r"\d+~", out), out
    assert out.startswith("call ~"), out


# --- The docs carry the contract ---------------------------------------------


def _pyi_scrub_pii_block() -> str:
    """The scrub_pii doc block from the stub: the comment from its
    opening sentence to the ``def scrub_pii`` signature."""
    source = PYI.read_text(encoding="utf-8")
    start = source.index("Replace contact material")
    end = source.index("def scrub_pii(")
    return source[start:end]


def test_the_stub_no_longer_says_only_plus_led_numbers_match() -> None:
    """The stub's stale restrictive phrase (the phone rule described as
    ``+``-led only) is gone: the domestic NANP extension is part of the
    matcher this version ships, and a stub that hides it tells callers
    the identifying head digits of an un-plussed number are safe to
    echo."""
    block = _pyi_scrub_pii_block()
    assert "`+`-led phone numbers" not in block, (
        "python/tors/__init__.pyi still describes the phone rule as "
        "`+`-led only: the domestic NANP matcher (an un-plussed ten/"
        "eleven-digit run with a separator) is documented, pinned "
        "behavior, and its tokens keep no digits; the stub must say so"
    )


def test_the_stub_names_the_domestic_matcher_where_the_phone_rule_is_described() -> None:
    """The positive half: the stub's phone-rule description (the digit
    class sentence) names the domestic matcher and its no-digit token,
    next to the bare-run rule it qualifies."""
    block = _pyi_scrub_pii_block()
    marker = "The phone rule's digit class"
    tail = block[block.index(marker):]
    assert "domestic" in tail, (
        "python/tors/__init__.pyi's phone-rule description must name the "
        "domestic NANP matcher (and that its token keeps no digits), not "
        f"just the bare-run refusal:\n{tail}"
    )


def test_api_md_token_definition_states_the_digest_alone_rule() -> None:
    """docs/api.md's token definition must carry the ``+``-led qualifier:
    the prefix is the dialling prefix of a ``+``-led canonical E.164, and
    any other spelling gets the digest alone (the reference oracle's own
    wording); the sentence as written ("the match's first three code
    points", unqualified) is exactly the rule that leaked the area
    code."""
    source = " ".join(API_MD.read_text(encoding="utf-8").split())
    start = source.index("`prefix~<digest>` for a phone number")
    paragraph = source[start : source.index("for an API key", start)]
    assert "digest alone" in paragraph, (
        "docs/api.md's phone-token definition must state that a non-"
        "`+`-led (domestic) match keeps no prefix, the digest alone:\n"
        f"{paragraph}"
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
