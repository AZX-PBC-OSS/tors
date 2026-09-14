"""Contract gate for ``tors.scrub_pii``: replace contact material (email
addresses, ``+``-led phone numbers) and credential material (the
evidence-backed api-key families) inside free text with correlation
tokens, the scrub an error excerpt or rejection message needs before it
reaches telemetry — the one store a data purge cannot reach.

What this gate pins (the contract is a port of a private consumer's
telemetry-safety module, pinned byte-identical to it at ``salt=""``):

- the email rule's grammar: the deliberately permissive local part, the
  greedy-backtracking domain split (``a@b.co9`` scrubs ``a@b.co`` and
  leaves the ``9``; ``a@b.c.d`` never matches), domains kept verbatim
  including their leading/doubled dots, and the fact that URLs, mailto
  links, and code spans get NO special treatment (whatever sits inside
  them matches).
- the phone rule's grammar, two matchers: INTERNATIONAL, anchored on a
  literal ``+`` (bare digit runs are order numbers and byte counts, and
  must survive), a digit directly after the ``+``, six or more middle
  class characters, and a final digit — the middle counts separators,
  so a long spelling matches on seven digits while ``+1234567`` never
  does — with the digit class in the Python-regex sense (every Unicode
  Nd decimal digit, not ASCII-only) and the token prefix being the
  match's first three CODE POINTS (``"+1 "`` for a domestic spelling,
  ``"+47"`` for a compact one); and DOMESTIC (the extension past the
  source), un-plussed NANP shapes — a full run of exactly ten digits,
  or eleven with an ASCII leading ``1``, in any ``[-. ()]`` spelling,
  carrying at least one separator (a bare digit run is an id even at
  ten digits, and the requirement is what keeps a token's own digest
  hex unmatchable, so phone-only stays strictly idempotent), no partial
  match inside a longer run, and never firing behind a ``+``
  (international territory: match or the source's non-match).
- token shapes: ``@domain~<12 hex>`` for email, ``prefix~<12 hex>`` for
  phone, ``<family prefix>~<12 hex>`` for a key (the prefix verbatim —
  the non-secret half that tells the operator WHICH credential to
  rotate); every digest is ``sha256(salt + match)`` truncated to 12 hex
  chars, so ``salt=""`` is the source chain's unsalted digest exactly —
  and ``salt=None`` salts the contact rules with ``tors/scrub_pii/v1``
  and the keys rule with its own ``tors/scrub_keys/v1`` tag, so a key
  digest can never alias a contact digest.
- the api_keys rule's grammar: the evidence-backed closed family set —
  OpenAI ``sk-``/``sk-proj-``/``sk-svcacct-``, Anthropic ``sk-ant-``,
  Google ``AIza``, Fireworks ``fw-``/``fw_``, Modal ``ak-``/``wk-``,
  GitHub ``ghp_``/``github_pat_``, the minted ``azxdev_``/``wd-``/``w-``/
  ``cn-`` shapes, and marker-scoped ``Bearer`` JWTs (a bare ``eyJ`` never
  matches: one consumer's API legitimately carries eyJ-shaped non-secret
  cursors) — each a literal prefix plus a minimal ``[A-Za-z0-9_-]`` tail
  consumed maximally, tried longest-prefix-first with fall-through, and
  never firing mid-token (a prefix glued to a preceding key-charset char
  is that token's fragment, the ``xak-...`` cut).
- the rules contract: ``None`` = every rule in the canonical order
  (api_keys FIRST — a key's tail can spell a dash-separated domestic
  phone run and its local part an email, so the key pass must eat the
  whole credential before the contact passes scan — then email, then
  phone over its result); ``[]`` = the identity (the original object);
  duplicates deduped and order irrelevant; an unknown name is a
  ``ValueError`` naming the accepted set.
- the salt contract: ``None`` = tors's documented default constant (a
  known, non-secret tag), ``""`` = unsalted (source parity), and the
  digest is the only field a salt touches.
- identity: no rule firing returns the original object.
- idempotence, pinned as it true is: the output is a fixed point
  UNLESS an email token is immediately followed by @-shaped text (two
  reachable shapes: two adjacent email matches, and a match whose end
  is followed by an unmatched `@`-run whose own local part the match
  consumed) — a token's digest hex is local-part material, so a second
  pass fires once more on that boundary; scrubbing twice always
  converges (the second output is a fixed point), and phone-only is
  strictly idempotent.
- the argument boundary: lone surrogates are refused with
  ``UnicodeEncodeError`` (the crate-wide str contract). The source chain
  diverges there — its classes never match a surrogate, so it returns
  such text unchanged — and this gate pins tors's divergence from it
  deliberately.
"""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reference import SCRUB_PII_EMAILS as EMAILS
from reference import SCRUB_PII_PHONES as PHONES
from reference import SCRUB_PII_SEPARATORS as SEPARATORS
from tors import KEY_FAMILIES, scrub_pii, scrub_pii_report


# The unsalted digest (salt=""), spelled directly against hashlib so the
# expected tokens below are derived independently of both tors and the
# reference oracle: a third transcription of the construction.
def _hex12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _email_token(match: str) -> str:
    local, sep, domain = match.rpartition("@")
    assert sep and local and domain  # a regex match is never degenerate
    return f"@{domain}~{_hex12(match)}"


def _phone_token(match: str) -> str:
    return f"{match[:3]}~{_hex12(match)}"


# Non-ASCII zoo pieces built from codepoints (pure-ASCII source, the
# reference.py idiom): Arabic-Indic and fullwidth digit runs, and the
# o-umlaut domain char.
_ARABIC_TWELVE = "".join(chr(0x0660 + i % 10) for i in range(12))
_FULLWIDTH_EIGHT = "".join(chr(0xFF10 + i) for i in range(8))
_ARABIC_FIVE = "".join(chr(0x0665 + i) for i in range(5))
_O_UMLAUT = chr(0x00F6)

# API-key material: real-shaped synthetic keys for every family of the
# evidence-backed closed set. The tails are deterministic 62-char-alphabet
# cycles, letters-only below length 53, so no battery vector accidentally
# carries a phone or email shape (a bare in-run digit block is inert to
# both contact matchers); the interaction pins below build the composed
# shapes deliberately.
_KEY_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _key_tail(n: int) -> str:
    return "".join(_KEY_ALPHABET[i % len(_KEY_ALPHABET)] for i in range(n))


def _key_token(prefix: str, match: str, salt: str = "") -> str:
    return f"{prefix}~{_hex12(salt + match)}"


# The AWS access-key tail: uppercase letters and digits only ([0-9A-Z] —
# the access-key ID alphabet, no lowercase anywhere in it).
_AWS_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _aws_tail(n: int) -> str:
    return "".join(_AWS_ALPHABET[i % len(_AWS_ALPHABET)] for i in range(n))


# The documented example shape: AKIA + a 16-char access-key ID.
_AKIA = "AKIAIOSFODNN7EXAMPLE"

# The Azure storage-key tail: the connection-string secret alphabet
# ([A-Za-z0-9+/=] — base64 plus the padding/trailing `=`).
_AZURE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="
)


def _azure_tail(n: int) -> str:
    return "".join(_AZURE_ALPHABET[i % len(_AZURE_ALPHABET)] for i in range(n))


# PEM material: a multi-line SPAN family. The body is base64 lines (no
# separators, no `@`), so an unterminated block stays identity under the
# contact passes too — the near-miss pins below demand it.
_PEM_BODY = (
    "MIIEpAIBAAKCAQEA7b",
    "qY4sLk2MnOpQrStUvW",
    "xYz0123456789ABCD",
)


def _pem_block(words: str, body: tuple[str, ...] = _PEM_BODY) -> str:
    lines = [f"-----BEGIN {words} PRIVATE KEY-----", *body, f"-----END {words} PRIVATE KEY-----"]
    return "\n".join(lines)


_RSA_PEM = _pem_block("RSA")

# The token-prefix-to-family map for the battery vectors (one prefix
# per vector; the multi-prefix families list each of theirs).
_PREFIX_FAMILY = {
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


# A JWT exactly the shape the family grammar states: the `Bearer ` marker,
# a first segment beginning `eyJ`, and two more base64url segments
# (middle and signature, any length >= 1 each).
_JWT = (
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
    "dozjgNryP4J3jVmNHc0FKW3YtV9zZ2YwXqR8uT1aB5cDe"
)

# The family table as the contract states it: (whole synthetic key, the
# token's verbatim family prefix). Lengths at each family's own minimum
# or a realistic multiple: ghp_ at exactly 36, AIza at exactly 35, the
# minted shapes at their exact tails (azxdev_ 20, wd- 43, w- 43, cn- 20),
# github_pat_ at its 22 floor, the sk- families at 20-plus.
_OPENAI = "sk-" + _key_tail(48)
_KEY_VECTORS: tuple[tuple[str, str], ...] = (
    (_OPENAI, "sk-"),
    ("sk-proj-" + _key_tail(48), "sk-proj-"),
    ("sk-svcacct-" + _key_tail(48), "sk-svcacct-"),
    ("sk-ant-api03-" + _key_tail(95), "sk-ant-"),
    ("AIza" + _key_tail(35), "AIza"),
    ("fw-" + _key_tail(48), "fw-"),
    ("fw_" + _key_tail(48), "fw_"),
    ("ak-" + _key_tail(48), "ak-"),
    ("wk-" + _key_tail(48), "wk-"),
    ("ghp_" + _key_tail(36), "ghp_"),
    ("github_pat_" + _key_tail(22), "github_pat_"),
    ("azxdev_" + _key_tail(20), "azxdev_"),
    ("wd-" + _key_tail(43), "wd-"),
    ("w-" + _key_tail(43), "w-"),
    ("cn-" + _key_tail(20), "cn-"),
    (_JWT, "Bearer"),
    (_AKIA, "AKIA"),
    ("ASIA" + _aws_tail(16), "ASIA"),
    ("xai-" + _key_tail(20), "xai-"),
    ("ya29." + _key_tail(20), "ya29."),
    (_RSA_PEM, "PEM"),
    ("AccountKey=" + _azure_tail(44), "AccountKey="),
)

# The documented non-matches: one-under tails at every distinct minimum,
# the uppercase spelling, the bare prefix, the mid-token prefix (the
# boundary rule), the unmarked/lowercase/degenerate JWT spellings.
_KEY_NON_MATCHES: tuple[tuple[str, str], ...] = (
    ("sk-one-under", "sk-" + _key_tail(19)),
    ("sk-bare", "sk-"),
    ("ski-uppercase", "SKI-" + _key_tail(48)),
    ("aiza-one-under", "AIza" + _key_tail(34)),
    ("ghp-one-under", "ghp_" + _key_tail(35)),
    ("github-pat-one-under", "github_pat_" + _key_tail(21)),
    ("azxdev-one-under", "azxdev_" + _key_tail(19)),
    ("wd-one-under", "wd-" + _key_tail(42)),
    ("w-one-under", "w-" + _key_tail(42)),
    ("cn-one-under", "cn-" + _key_tail(19)),
    ("fw-one-under", "fw-" + _key_tail(19)),
    ("ak-one-under", "ak-" + _key_tail(19)),
    ("wk-one-under", "wk-" + _key_tail(19)),
    ("aws-one-under", "AKIA" + _aws_tail(15)),
    ("aws-lowercase", "akia" + _aws_tail(16)),
    ("asia-one-under", "ASIA" + _aws_tail(15)),
    ("aws-midtoken", "x" + _AKIA),
    ("xai-one-under", "xai-" + _key_tail(19)),
    ("xai-uppercase", "XAI-" + _key_tail(20)),
    ("gcp-one-under", "ya29." + _key_tail(19)),
    ("gcp-uppercase", "YA29." + _key_tail(20)),
    ("gcp-midtoken", "xya29." + _key_tail(20)),
    ("pem-unterminated", _pem_block("RSA")[: -len("-----END RSA PRIVATE KEY-----")]),
    (
        "pem-mismatched-words",
        _pem_block("RSA").replace("-----END RSA PRIVATE KEY-----", "-----END EC PRIVATE KEY-----"),
    ),
    (
        "pem-empty-words",
        "-----BEGIN PRIVATE KEY-----\n" + "\n".join(_PEM_BODY) + "\n-----END PRIVATE KEY-----",
    ),
    (
        "pem-lowercase",
        "-----begin rsa private key-----\n"
        + "\n".join(_PEM_BODY)
        + "\n-----end rsa private key-----",
    ),
    ("pem-glued", "abc" + _RSA_PEM),
    (
        "pem-doubled-space",
        "-----BEGIN RSA  PRIVATE KEY-----\nMIIE\n-----END RSA  PRIVATE KEY-----",
    ),
    (
        "pem-hyphen-word",
        "-----BEGIN RSA-EC PRIVATE KEY-----\nMIIE\n-----END RSA-EC PRIVATE KEY-----",
    ),
    (
        "pem-underscore-word",
        "-----BEGIN RSA_EC PRIVATE KEY-----\nMIIE\n-----END RSA_EC PRIVATE KEY-----",
    ),
    # Split across the concatenation: the joined shape trips push
    # protection (a synthetic vector, not a secret).
    ("slack-xox-excluded", "xoxb-" + "123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUv"),
    ("stripe-test-excluded", "sk_test_51MZABCDefghijklmnOP0123456789abcdefghiJ"),
    ("bare-b64-secret-excluded", "MIIEpAIBAAKCAQEA7bqY4sLk2MnOpQrStUvWxYz0123456789ABCD"),
    ("azure-without-marker-excluded", _azure_tail(44)),
    ("azure-one-under", "AccountKey=" + _azure_tail(39)),
    ("azure-lowercase-k", "Accountkey=" + _azure_tail(44)),
    ("azure-midtoken", "xAccountKey=" + _azure_tail(40)),
    ("midtoken-prefix", "xak-" + _key_tail(48)),
    ("bearer-lowercase", "bearer " + _JWT[len("Bearer ") :]),
    ("bare-eyJ-cursor", _JWT[len("Bearer ") :]),
    ("jwt-two-segments", "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.c2ln"),
    ("jwt-degenerate-first-segment", "Bearer eyJ.a.b.c"),
    ("jwt-empty-middle-segment", "Bearer eyJhbGciOiJIUzI1Ni..dozjgNryP4J3"),
)


class TestEmailZoo:
    def test_the_canonical_rejection_excerpt(self) -> None:
        """The shape the function exists for: an upstream rejection echoing
        a candidate's address and number into telemetry retention."""
        text = "unknown candidate fungai.chetima@example.com called from +14155552671 twice"
        assert scrub_pii(text, salt="") == (
            "unknown candidate "
            f"{_email_token('fungai.chetima@example.com')} "
            f"called from {_phone_token('+14155552671')} twice"
        )

    def test_plus_addressing_is_local_part_material(self) -> None:
        # "+" is local-part class material: every plus-addressed spelling
        # matches whole, the tag included — multiple tags, a trailing "+",
        # and the bare-"+" local edge.
        assert scrub_pii("ada+tag@azx.io", salt="") == _email_token("ada+tag@azx.io")
        assert scrub_pii("a+b+c@x.co", salt="") == _email_token("a+b+c@x.co")
        assert scrub_pii("foo+@x.co", salt="") == _email_token("foo+@x.co")
        assert scrub_pii("+@x.co", salt="") == _email_token("+@x.co")

    def test_trailing_punctuation_is_not_consumed(self) -> None:
        # The domain split backtracks to the largest dot with a two-letter
        # tail, so a trailing period/digit survives the match.
        assert scrub_pii("a@b.co.", salt="") == f"{_email_token('a@b.co')}."
        assert scrub_pii("a@b.co9", salt="") == f"{_email_token('a@b.co')}9"
        assert scrub_pii("a@b.co.uk", salt="") == _email_token("a@b.co.uk")

    def test_a_one_letter_tld_never_matches(self) -> None:
        # "d" is one letter short of [A-Za-z]{2,}: no split exists, and the
        # whole address survives (the documented non-match, pinned).
        assert scrub_pii("a@b.c.d", salt="") == "a@b.c.d"

    def test_degenerate_domain_dots_are_kept_verbatim(self) -> None:
        # The domain run class includes '.', so ".b.co" and "b..co" are
        # domains in their own right, echoed in the token as matched.
        assert scrub_pii("a@.b.co", salt="") == _email_token("a@.b.co")
        assert scrub_pii("a@b..co", salt="") == _email_token("a@b..co")

    def test_uppercase_is_matched_and_preserved(self) -> None:
        assert scrub_pii("A@B.CO", salt="") == _email_token("A@B.CO")

    def test_local_part_class_edges(self) -> None:
        # % _ . + - are all local-part material; the whole address is eaten.
        assert scrub_pii("a%b@x.co", salt="") == _email_token("a%b@x.co")
        assert scrub_pii("a_b@x.co", salt="") == _email_token("a_b@x.co")
        assert scrub_pii("xxa@b.co", salt="") == _email_token("xxa@b.co")

    def test_a_non_local_char_splits_the_match(self) -> None:
        # "!" is not in the local class: the match starts at "b", and the
        # "a!" prefix survives — the source's documented permissiveness
        # trade-off (over-matching costs a token; under-matching leaks).
        assert scrub_pii("a!b@x.co", salt="") == f"a!{_email_token('b@x.co')}"

    def test_only_the_second_at_can_match(self) -> None:
        # The first "@" has no dot in its domain run ("b" stops at "@"), so
        # the match starts at "b" and "a@" survives.
        assert scrub_pii("a@b@c.co", salt="") == f"a@{_email_token('b@c.co')}"

    def test_a_resume_digit_becomes_the_next_local_part(self) -> None:
        # The first match ends before "9"; scanning resumes there and the
        # digit is a local part for the second match.
        assert scrub_pii("a@b.co9@x.yz", salt="") == (
            f"{_email_token('a@b.co')}{_email_token('9@x.yz')}"
        )

    def test_urls_mailto_and_code_spans_get_no_special_treatment(self) -> None:
        assert scrub_pii("https://user@x.io/path", salt="") == (
            f"https://{_email_token('user@x.io')}/path"
        )
        assert scrub_pii("mailto:a@b.co", salt="") == f"mailto:{_email_token('a@b.co')}"
        assert scrub_pii("see `a@b.co` in code", salt="") == (
            f"see `{_email_token('a@b.co')}` in code"
        )

    def test_non_ascii_domains_never_match(self) -> None:
        # Every email class is ASCII: a non-ASCII domain char breaks the
        # run and the address survives (the documented non-match).
        assert scrub_pii("a@b.c" + _O_UMLAUT, salt="") == "a@b.c" + _O_UMLAUT
        assert scrub_pii("a@" + _O_UMLAUT + ".co", salt="") == "a@" + _O_UMLAUT + ".co"

    def test_rfc_quoted_locals_leak_whole(self) -> None:
        # RFC quoted-string locals ("..."@domain, with @ inside the quotes)
        # are NOT fragment-matches: `"` is outside the local class, and the
        # quote before the real `@` blocks the match, so the whole address
        # survives. No grammar widening (parity): canonicalize (strip RFC
        # quotes / split display-names) before scrubbing if quoted locals
        # are in threat.
        for text in ('"user@name"@example.com', '"a@b"@x.co'):
            assert scrub_pii(text, salt="") is text

    def test_ip_literal_and_dotted_quad_domains_leak_whole(self) -> None:
        # `user@[192.168.1.1]` (the `[` breaks the domain run) and
        # `user@192.168.1.1` (the trailing quad has no two-letter tail)
        # never match, so the whole address survives. Canonicalize
        # (idna-to-punycode / IP-literal normalization) before scrubbing
        # if these spellings are in threat; no grammar widening (parity).
        for text in ("user@[192.168.1.1]", "user@192.168.1.1"):
            assert scrub_pii(text, salt="") is text

    def test_long_fields_are_matched_whole(self) -> None:
        # No length cap anywhere in the grammar: a maximal local part and
        # a multi-label domain are one match, and the token's domain field
        # carries the whole domain as matched.
        match = f"{'ada.' * 32}@example.co.uk"
        assert scrub_pii(match, salt="") == _email_token(match)

    def test_email_shaped_text_inside_a_phone_match_is_eaten_by_the_email_pass(self) -> None:
        # The email substitution runs first: "8901ada" is a local part, so
        # the phone match ends at "557" and the two tokens stay separated
        # by the surviving space (the phone match ends at a digit, never
        # at the separator or the token that follows).
        assert scrub_pii("+1 415 557 8901ada@x.co", salt="") == (
            f"{_phone_token('+1 415 557')} {_email_token('8901ada@x.co')}"
        )


class TestPhoneZoo:
    def test_compact_e164(self) -> None:
        # The prefix is the first three code points: "+14" here.
        assert scrub_pii("+14155552671", salt="") == _phone_token("+14155552671")

    def test_domestic_spellings_and_the_space_in_the_prefix(self) -> None:
        # "+1 (415) 555-2671" and "+1 415 555 2671" both match; their
        # prefixes are "+1 " — the third code point is the space.
        assert scrub_pii("+1 (415) 555-2671", salt="") == _phone_token("+1 (415) 555-2671")
        assert scrub_pii("+1 415 555 2671", salt="") == _phone_token("+1 415 555 2671")
        assert scrub_pii("+1 415 555 2671x", salt="") == f"{_phone_token('+1 415 555 2671')}x"

    def test_the_plus_anchoring_spares_bare_digit_runs(self) -> None:
        text = "read 4096 bytes across 3 pages in 1200 ms"
        assert scrub_pii(text, salt="") is text

    def test_a_separator_right_after_the_plus_never_matches(self) -> None:
        # \d must follow the "+": "+ (415)..." and "+1-555-" both fail.
        assert scrub_pii("+ (415) 555-2671", salt="") == "+ (415) 555-2671"
        assert scrub_pii("+1-555-", salt="") == "+1-555-"

    def test_the_minimum_is_eight_digits(self) -> None:
        assert scrub_pii("+1234567", salt="") == "+1234567"
        assert scrub_pii("+12345678", salt="") == _phone_token("+12345678")
        # A leading zero is a \d like any other.
        assert scrub_pii("+004155552671", salt="") == _phone_token("+004155552671")

    def test_a_second_plus_starts_the_real_match(self) -> None:
        # "+" is not in the separator class: the first attempt dies at the
        # class run, the second "+4155552671" is the match, and "+1"
        # survives.
        assert scrub_pii("+1+4155552671", salt="") == f"+1{_phone_token('+4155552671')}"

    def test_two_numbers_split_by_one_space_are_one_match(self) -> None:
        # The space is a class char: the run spans both numbers, and one
        # token covers the pair (the digest is of the whole run).
        match = "+4712345678 1234567890"
        assert scrub_pii(match, salt="") == _phone_token(match)

    def test_a_tilde_breaks_the_run(self) -> None:
        # "~" is not a class char (and not a \d): the match ends at "8",
        # and the literal digit run after the tilde survives.
        assert scrub_pii("+12345678~1234567890", salt="") == (
            f"{_phone_token('+12345678')}~1234567890"
        )

    def test_the_final_digit_backtrack_leaves_trailing_separators(self) -> None:
        # The match ends at the LAST digit in the run: the trailing space
        # and the comma survive around it.
        assert scrub_pii("call +1 (415) 555-2671 , ok", salt="") == (
            f"call {_phone_token('+1 (415) 555-2671')} , ok"
        )

    def test_two_numbers_separated_by_a_comma_both_match(self) -> None:
        text = "+1 415 555 2671, +1 415 555 2672"
        assert scrub_pii(text, salt="") == (
            f"{_phone_token('+1 415 555 2671')}, {_phone_token('+1 415 555 2672')}"
        )

    def test_unicode_nd_digits_match_and_set_a_multibyte_prefix(self) -> None:
        # The digit class is the Python-regex sense: every Unicode Nd
        # decimal digit. The prefix is the first three CODE POINTS, so a
        # non-ASCII digit run carries its own spelling into the token.
        arabic = "+" + _ARABIC_TWELVE
        assert scrub_pii(arabic, salt="") == f"+{chr(0x0660)}{chr(0x0661)}~{_hex12(arabic)}"
        fullwidth = "+" + _FULLWIDTH_EIGHT
        assert scrub_pii(fullwidth, salt="") == (f"+{chr(0xFF10)}{chr(0xFF11)}~{_hex12(fullwidth)}")
        mixed = "+1234" + _ARABIC_FIVE  # ASCII and Arabic-Indic in one run
        assert scrub_pii(mixed, salt="") == f"+12~{_hex12(mixed)}"

    def test_non_nd_numerics_do_not_match(self) -> None:
        # Superscript two is No (not Nd): the class run dies at it, and a
        # fullwidth "+" is not the literal "+" either.
        assert scrub_pii("+12" + chr(0x00B2) + "45678", salt="") == "+12" + chr(0x00B2) + "45678"
        assert scrub_pii(chr(0xFF0B) + "12345678", salt="") == chr(0xFF0B) + "12345678"

    def test_a_phone_shaped_local_part_is_eaten_by_the_email_pass(self) -> None:
        # The email rule runs first and its local class contains "+":
        # "user+14155552671" is one local part, nothing remains for the
        # phone rule to fire on.
        assert scrub_pii("user+14155552671@example.com", salt="") == (
            _email_token("user+14155552671@example.com")
        )


class TestRulesContract:
    def test_none_is_every_rule_in_the_canonical_order(self) -> None:
        # keys FIRST (a key's tail can spell a domestic phone run, its
        # local part an email), then email, then phone over its result.
        text = "a@b.co +14155552671"
        assert scrub_pii(text) == scrub_pii(text, ["api_keys", "contact_email", "contact_phone"])
        # ...and the order the caller lists them in is irrelevant.
        assert scrub_pii(text, ["contact_phone", "contact_email", "api_keys"]) == scrub_pii(text)

    def test_empty_rules_is_the_identity_object(self) -> None:
        text = "a@b.co +14155552671"
        assert scrub_pii(text, []) is text

    def test_a_tuple_is_accepted_like_a_list(self) -> None:
        assert scrub_pii("a@b.co", ("contact_email",), salt="") == _email_token("a@b.co")

    def test_duplicates_dedupe(self) -> None:
        assert scrub_pii("a@b.co", ["contact_email", "contact_email"], salt="") == (
            _email_token("a@b.co")
        )

    def test_unknown_names_name_the_accepted_set(self) -> None:
        # The value's quoting is Rust's Debug rendering, the same spelling
        # the errors=/byteorder= closed-set messages ship.
        with pytest.raises(ValueError) as exc:
            scrub_pii("a@b.co", ["emails"])
        assert str(exc.value) == (
            "rules must be one of ('contact_email', 'contact_phone', 'api_keys'), not \"emails\""
        )

    def test_a_valid_name_plus_an_unknown_one_still_raises(self) -> None:
        with pytest.raises(ValueError, match="contact_phone"):
            scrub_pii("a@b.co", ["api_keys", "ssn"])

    def test_a_bare_string_is_not_a_rules_sequence(self) -> None:
        # pyo3's sequence extraction refuses a str (it would iterate
        # characters), so the Sequence[...] spelling means list or tuple
        # and a bare string is a TypeError — the boundary on the accepted
        # side of the set, pinned.
        with pytest.raises(TypeError):
            scrub_pii("a@b.co", "contact_email")

    def test_non_sequence_rules_arguments_are_type_errors(self) -> None:
        with pytest.raises(TypeError):
            scrub_pii("a@b.co", 42)
        with pytest.raises(TypeError):
            scrub_pii("a@b.co", {"contact_email"})

    def test_phone_only_leaves_email_text_alone(self) -> None:
        assert scrub_pii("+1 415 557 8901ada@x.co", ["contact_phone"], salt="") == (
            f"{_phone_token('+1 415 557 8901')}ada@x.co"
        )

    def test_email_only_leaves_phone_text_alone(self) -> None:
        assert scrub_pii("+1 415 557 8901ada@x.co", ["contact_email"], salt="") == (
            f"+1 415 557 {_email_token('8901ada@x.co')}"
        )


class TestSaltContract:
    def test_default_salt_is_the_documented_constant(self) -> None:
        # The literal is pinned here in full: the default is a fixed,
        # non-secret domain-separation tag, frozen because changing it
        # would silently change every deployment's token values.
        token = scrub_pii("a@b.co +14155552671")
        assert token == (
            f"@b.co~{_hex12('tors/scrub_pii/v1' + 'a@b.co')} "
            f"+14~{_hex12('tors/scrub_pii/v1' + '+14155552671')}"
        )

    def test_empty_salt_is_the_unsalted_digest(self) -> None:
        # salt="" is the migration lane: byte-identical token values with
        # an unsalted upstream scrubber (the source chain's own digest).
        token = scrub_pii("a@b.co +14155552671", salt="")
        assert token == f"@b.co~{_hex12('a@b.co')} +14~{_hex12('+14155552671')}"

    def test_same_input_and_salt_agree_across_calls(self) -> None:
        assert scrub_pii("a@b.co", salt="secret") == scrub_pii("a@b.co", salt="secret")

    def test_different_salts_change_only_the_digest_field(self) -> None:
        one = scrub_pii("a@b.co +1 (415) 555-2671", salt="one")
        two = scrub_pii("a@b.co +1 (415) 555-2671", salt="two")
        assert one != two

        # The kept-context fields are salt-independent: drop the 12 digest
        # chars after every "~" (each tilde introduces exactly one digest)
        # and the two outputs agree exactly — same domain, same dialling
        # prefix, the space inside "+1 " included.
        def strip_digests(out: str) -> str:
            stripped = ""
            while (cut := out.find("~")) >= 0:
                stripped += out[: cut + 1]
                out = out[cut + 13 :]
            return stripped + out

        assert strip_digests(one) == strip_digests(two) == "@b.co~ +1 ~"

    def test_a_salt_is_utf8_material_like_the_text(self) -> None:
        salt = "üñí/v1"
        assert scrub_pii("a@b.co", salt=salt) == f"@b.co~{_hex12(salt + 'a@b.co')}"

    def test_the_salt_applies_to_both_rules(self) -> None:
        salt = "call-site"
        assert scrub_pii("a@b.co +14155552671", salt=salt) == (
            f"@b.co~{_hex12(salt + 'a@b.co')} +14~{_hex12(salt + '+14155552671')}"
        )


class TestIdentityContract:
    def test_no_match_returns_the_original_object(self) -> None:
        text = "read 4096 bytes across 3 pages in 1200 ms"
        assert scrub_pii(text) is text

    def test_empty_text_is_the_identity_object(self) -> None:
        empty = ""
        assert scrub_pii(empty) is empty

    def test_the_documented_non_matches_are_all_identity(self) -> None:
        for text in (
            "a@b.c.d",
            "a@" + _O_UMLAUT + ".co",
            "+ (415) 555-2671",
            "+1234567",
            "not-an-address",
        ):
            assert scrub_pii(text, salt="") is text

    def test_identity_holds_for_subset_rules_too(self) -> None:
        text = "+14155552671"
        assert scrub_pii(text, ["contact_email"], salt="") is text


class TestTokenShape:
    @pytest.mark.parametrize(
        "text", ["a@b.co", "ada+tag@azx.io", "+14155552671", "+1 (415) 555-2671"]
    )
    def test_every_tilde_introduces_twelve_lowercase_hex(self, text: str) -> None:
        out = scrub_pii(text, salt="")
        fields = out.split("~")
        assert len(fields) == 2
        digest = fields[1]
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)

    def test_email_tokens_lead_with_the_domain_at(self) -> None:
        assert scrub_pii("a@b.co", salt="").startswith("@b.co~")

    def test_phone_tokens_lead_with_the_dialling_prefix(self) -> None:
        assert scrub_pii("+4712345678", salt="").startswith("+47~")
        assert scrub_pii("+1 (415) 555-2671", salt="").startswith("+1 ~")

    def test_the_original_match_does_not_survive_its_token(self) -> None:
        for text, match in (
            ("a@b.co", "a@b.co"),
            ("+14155552671", "+14155552671"),
            ("+1 (415) 555-2671", "4155552671"),
        ):
            assert match not in scrub_pii(text, salt="")


class TestIdempotence:
    def test_zoo_outputs_are_fixed_points(self) -> None:
        # An output re-fires only where an email token is immediately
        # followed by @-shaped text; the zoo's compositions (contacts
        # joined by separators, in both orders) never place an `@` right
        # after a match, so every one of these outputs is a fixed point.
        zoo = [
            f"{email}{sep}{phone}" for email in EMAILS for phone in PHONES for sep in SEPARATORS
        ] + [f"{phone}{sep}{email}" for email in EMAILS for phone in PHONES for sep in SEPARATORS]
        for text in zoo:
            once = scrub_pii(text, salt="")
            assert scrub_pii(once, salt="") == once, text

    def test_adjacent_email_matches_re_fire_once_then_converge(self) -> None:
        # The re-fire corner, first shape, pinned literally: adjacent
        # email matches ("a@b.co" ends where "9@x.yz" begins) produce
        # back-to-back tokens, and the second pass re-fires on the
        # boundary — the first token's digest hex is a local part for a
        # fresh match of the second token's domain. Pass three is a
        # fixed point.
        text = "a@b.co9@x.yz"
        once = scrub_pii(text, salt="")
        assert once == f"{_email_token('a@b.co')}{_email_token('9@x.yz')}"
        twice = scrub_pii(once, salt="")
        assert twice == (f"@b.co~@x.yz~{_hex12(_hex12('a@b.co') + '@x.yz')}~{_hex12('9@x.yz')}")
        assert scrub_pii(twice, salt="") is twice

    def test_a_match_followed_by_an_unmatched_at_run_re_fires_once(self) -> None:
        # The corner's second shape: "x@b.co" consumed the local-part
        # material before the second "@", so "@w.vu" never matched on
        # its own — but the token's digest hex is local-part material,
        # and pass two fires on [hex + "@w.vu"] once, then holds.
        text = "x@b.co@w.vu"
        once = scrub_pii(text, salt="")
        assert once == f"{_email_token('x@b.co')}@w.vu"
        hex1 = _hex12("x@b.co")
        twice = scrub_pii(once, salt="")
        assert twice == f"@b.co~@w.vu~{_hex12(hex1 + '@w.vu')}"
        assert scrub_pii(twice, salt="") is twice

    def test_an_invalid_at_run_after_a_match_is_not_the_corner(self) -> None:
        # "d4.e5" has a one-letter TLD: the second `@`-run cannot match
        # in pass two either, so the output is a fixed point.
        text = "x1@b2.co3@d4.e5"
        once = scrub_pii(text, salt="")
        assert once == _email_token("x1@b2.co") + "3@d4.e5"
        assert scrub_pii(once, salt="") is once

    def test_tokens_are_individually_fixed_points(self) -> None:
        for token in (
            _email_token("a@b.co"),
            _phone_token("+14155552671"),
            _phone_token("+1 (415) 555-2671"),
            _phone_token("+4712345678"),
        ):
            assert scrub_pii(token, salt="") is token

    def test_phone_only_is_strictly_idempotent(self) -> None:
        # Phone tokens can never re-fire the phone rule (the "~" breaks
        # every digit run three code points in), and no email pass runs
        # to create the re-fire corner.
        for text in (
            "+14155552671 +14155552672",
            "+1 415 555 2671, +44 20 7946 0958",
            "+4712345678 1234567890",
            "a@b.co9@x.yz +14155552671",
            "x@b.co@w.vu +14155552671",
        ):
            once = scrub_pii(text, ["contact_phone"], salt="")
            assert scrub_pii(once, ["contact_phone"], salt="") is once

    @given(
        pieces=st.lists(
            st.sampled_from(
                list(EMAILS)
                + list(PHONES)
                + ["9", "a", "@", ".", "~", "+", " ", "(", ")", "-", ","]
            ),
            max_size=25,
        )
    )
    @settings(max_examples=300)
    def test_scrubbing_twice_always_converges(self, pieces: list[str]) -> None:
        text = "".join(pieces)
        once = scrub_pii(text, salt="")
        twice = scrub_pii(once, salt="")
        assert scrub_pii(twice, salt="") == twice


class TestArgumentBoundary:
    def test_lone_surrogates_are_refused(self) -> None:
        # The crate-wide str contract: the UTF-8 borrow refuses a str
        # holding lone surrogates with UnicodeEncodeError before any Rust
        # code runs. The source chain diverges here (its classes never
        # contain a surrogate, so it returns such text unchanged); tors
        # pins the crate contract, divergence and all.
        with pytest.raises(UnicodeEncodeError):
            scrub_pii("a\ud800b")

    def test_surrogates_are_refused_even_when_no_rule_could_match(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            scrub_pii("a\ud800b@x.co")

    def test_surrogates_are_refused_for_subset_rules_too(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            scrub_pii("a\ud800b", ["contact_phone"])

    def test_a_surrogate_in_the_salt_is_refused(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            scrub_pii("a@b.co", salt="\udcff")

    def test_non_str_arguments_are_type_errors(self) -> None:
        # The str-in/str-out boundary: bytes text and a non-str salt are
        # TypeErrors, the same shape every str-in function here documents.
        with pytest.raises(TypeError):
            scrub_pii(b"a@b.co")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            scrub_pii("a@b.co", salt=42)  # type: ignore[arg-type]


class TestDomesticPhoneZoo:
    """The domestic matcher: the extension past the source's ``+``-anchored
    grammar (the source leaves every un-plussed digit run untouched; tors
    scrubs the NANP shapes, maintainer-directed). The discipline rules that
    keep it safe are pinned here with the same literal thoroughness as the
    source port: the separator requirement (a bare digit run is an id even
    at exactly ten digits, and the requirement is what keeps a token's own
    digest hex unmatchable, so phone-only stays strictly idempotent), the
    full-run width rule (no partial match inside a longer run), and the
    ``+``-territory rule (a run behind a ``+`` is international-only, never
    a domestic fallback)."""

    def test_every_separator_spelling_matches(self) -> None:
        for matched in [
            "(415) 555-2671",
            "415-555-2671",
            "415.555.2671",
            "415 555 2671",
            "1-415-555-2671",
            "1 (415) 555-2671",
            "1 415 555 2671",
            "1415 555 2671",
        ]:
            assert scrub_pii(matched, ["contact_phone"], salt="") == _phone_token(matched)

    def test_leading_spaces_are_skipped_and_trailing_separators_spared(self) -> None:
        assert (
            scrub_pii("call 415-555-2671 ok", ["contact_phone"], salt="")
            == f"call {_phone_token('415-555-2671')} ok"
        )
        # A leading structural separator is part of the spelling.
        assert (
            scrub_pii("x -415-555-2671 , ok", ["contact_phone"], salt="")
            == f"x {_phone_token('-415-555-2671')} , ok"
        )
        # Extensions survive past the last digit, same as international.
        assert (
            scrub_pii("415-555-2671 x1234", ["contact_phone"], salt="")
            == f"{_phone_token('415-555-2671')} x1234"
        )

    def test_bare_digit_runs_never_match_even_at_ten_or_eleven(self) -> None:
        # The documented cut: an unseparated digit run is an order number
        # or id; the + form is the scrubbed spelling of the same digits.
        for text in [
            "4155552671",
            "14155552671",
            "ref 4155552671 x",
            "id 4155552671 ",
            "order 1234567890 closed",
        ]:
            assert scrub_pii(text, ["contact_phone"], salt="") == text

    def test_the_width_rule_rejects_short_long_and_wrong_leading_digits(self) -> None:
        for text in [
            "415-555-267",  # nine
            "415-555-267123",  # twelve: an id
            "415-555-2671-555-123-4567",  # twenty in ONE run: an id
            "915-555-26712",  # eleven, not led by ASCII '1'
        ]:
            assert scrub_pii(text, ["contact_phone"], salt="") == text
        # The NANP trunk prefix is an ASCII artifact: an eleven-digit Nd
        # run led by the Arabic-Indic ONE is not the shape.
        arabic_eleven = (
            "\U00000661\U00000661\U00000665 \U00000665\U00000665\U00000665"
            " \U00000662\U00000666\U00000667\U00000661\U00000662"
        )
        assert scrub_pii(arabic_eleven, ["contact_phone"], salt="") == arabic_eleven

    def test_unicode_nd_digits_spell_domestic_numbers(self) -> None:
        arabic = (
            "\U00000664\U00000661\U00000665 \U00000665\U00000665\U00000665"
            " \U00000662\U00000666\U00000667\U00000661"
        )
        assert scrub_pii(arabic, ["contact_phone"], salt="") == _phone_token(arabic)

    def test_a_plus_before_a_run_is_international_territory(self) -> None:
        # Ten digits behind a + is the INTERNATIONAL match; a separator
        # right after the +, or a run too short for the {6,} middle, is
        # the source's own documented non-match — never a domestic
        # fallback. The middle counts separators, so the long
        # seven-digit spelling matches while the short one never does.
        assert (
            scrub_pii("+415-555-2671", ["contact_phone"], salt="")
            == _phone_token("+415-555-2671")
        )
        assert (
            scrub_pii("ring +1 415 555 now", ["contact_phone"], salt="")
            == f"ring {_phone_token('+1 415 555')} now"
        )
        assert scrub_pii("+1415555", ["contact_phone"], salt="") == "+1415555"
        assert scrub_pii("+ (415) 555-2671", ["contact_phone"], salt="") == "+ (415) 555-2671"
        assert scrub_pii("+415-555", ["contact_phone"], salt="") == "+415-555"

    def test_two_domestic_matches_with_a_break_both_scrub(self) -> None:
        assert (
            scrub_pii("415-555-2671, 415-555-2672", ["contact_phone"], salt="")
            == f"{_phone_token('415-555-2671')}, {_phone_token('415-555-2672')}"
        )

    def test_an_email_token_domain_that_spells_a_number_is_re_tokenized(self) -> None:
        # The documented over-redaction corner: the email pass tokens the
        # whole address, then the phone pass re-tokens the domestic shape
        # the DOMAIN spelled (the dot between the digit groups is the
        # separator that makes it matchable). Safe direction; converges.
        matched = "user@555.1234567.co"
        once = scrub_pii(matched, salt="")
        assert once == (
            f"@555~{_hex12('555.1234567')}.co~{_hex12(matched)}"
        )
        twice = scrub_pii(once, salt="")
        assert scrub_pii(twice, salt="") == twice

    def test_phone_only_stays_strictly_idempotent_with_domestic_matches(self) -> None:
        once = scrub_pii(
            "(415) 555-2671 415-555-2672 +14155552673",
            ["contact_phone"],
            salt="",
        )
        assert scrub_pii(once, ["contact_phone"], salt="") == once

    def test_hex_or_tilde_glued_runs_are_identifier_fragments(self) -> None:
        # H2/H3: the clean-boundary rule pins `~` + lowercase a-f as the
        # exact dirty set (the token-interior alphabet), not "letters".
        for text in (
            "job255128-4096",
            "value~255-123-4567",
            "ref c415-555-2671",
            "ref a415-555-2671",
            "ref f415-555-2671",
        ):
            assert scrub_pii(text, ["contact_phone"], salt="") == text, text

    def test_non_hex_letters_are_clean_boundaries(self) -> None:
        # g/z and uppercase A-F are clean: the domestic shape still fires.
        for text, matched in (
            ("jobg415-555-2671", "415-555-2671"),
            ("jobz415-555-2671", "415-555-2671"),
            ("jobG415-555-2671", "415-555-2671"),
            ("jobA415-555-2671", "415-555-2671"),
            ("jobF415-555-2671", "415-555-2671"),
        ):
            out = scrub_pii(text, ["contact_phone"], salt="")
            assert out != text, text
            assert matched not in out, text


class TestApiKeyZoo:
    """The api_keys rule: the credential scrub, an extension past the
    source's two-rule contact contract the same way the domestic phone
    matcher is. The leak vector is the error text itself: provider and
    platform error strings can quote the credential back (five private
    consumers evidenced; the strongest, a platform whose own code
    comments that a vendor auth failure "can quote the key" and keeps the
    full text in an admin-served ledger), and telemetry is the one store
    a purge cannot reach. The family set is the evidence-backed closed
    set — Slack xox, Stripe, and AWS AKIA shapes are deliberately absent
    (zero evidence) — each family a literal prefix plus a minimal
    ``[A-Za-z0-9_-]`` tail, longest-prefix-first with fall-through, never
    mid-token. The token keeps the family prefix verbatim (WHICH
    credential to rotate) over the digest of the full match; ``salt=None``
    salts key digests with tors's own ``tors/scrub_keys/v1`` tag so a key
    digest can never alias a contact digest."""

    @pytest.mark.parametrize(
        ("text", "prefix"), _KEY_VECTORS, ids=[v[1] for v in _KEY_VECTORS]
    )
    def test_every_family_scrubs_with_its_verbatim_prefix(self, text: str, prefix: str) -> None:
        # The token is <family prefix>~<first 12 hex of sha256(salt +
        # FULL match)>: the prefix verbatim (the non-secret half), the
        # digest over prefix + tail, derived here independently of both
        # tors and the parity oracle (hashlib, the third-transcription
        # discipline).
        assert scrub_pii(text, ["api_keys"], salt="") == _key_token(prefix, text)
        # The default-rules call composes identically: keys first, and
        # every battery vector is contact-inert by construction.
        assert scrub_pii(text, salt="") == _key_token(prefix, text)

    @pytest.mark.parametrize(
        ("label", "text"), _KEY_NON_MATCHES, ids=[n for n, _ in _KEY_NON_MATCHES]
    )
    def test_the_documented_non_matches_are_identity(self, label: str, text: str) -> None:
        # One under every distinct minimum, the uppercase spelling, the
        # bare prefix, the mid-token prefix, and the JWT spellings that
        # are not three maximal base64url segments behind the marker.
        # Identity at default rules: no key family fires and no contact
        # shape rides inside (the tails are contact-inert by design).
        assert scrub_pii(text, salt="") is text

    def test_jwt_segments_beyond_the_third_survive_verbatim(self) -> None:
        # The grammar is exactly three maximal segments (the residual-risk
        # red-team pin, in the contract gate): a 4-segment spell keeps
        # `.d`, a 5-part JWE keeps `.d.e`.
        assert scrub_pii("Bearer eyJa.b.c.d", ["api_keys"], salt="") == (
            f"Bearer~{_hex12('Bearer eyJa.b.c')}.d"
        )
        assert scrub_pii("Bearer eyJa.b.c.d.e", ["api_keys"], salt="") == (
            f"Bearer~{_hex12('Bearer eyJa.b.c')}.d.e"
        )

    def test_an_empty_body_pem_is_still_both_markers(self) -> None:
        # Adjacent markers with nothing between them: both markers are
        # present, so the degenerate block matches whole.
        text = "-----BEGIN RSA PRIVATE KEY----------END RSA PRIVATE KEY-----"
        assert scrub_pii(text, ["api_keys"], salt="") == _key_token("PEM", text)

    def test_a_mismatched_end_inside_the_body_does_not_terminate(self) -> None:
        # The first END whose words verify wins: the stranger's END line
        # is body material, and the block runs whole to its own END.
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA7b\n"
            "-----END EC PRIVATE KEY-----\n"
            "-----END RSA PRIVATE KEY-----"
        )
        assert scrub_pii(text, ["api_keys"], salt="") == _key_token("PEM", text)

    def test_an_azure_key_inside_a_connection_string(self) -> None:
        # The realistic context: the match is the marker plus its maximal
        # tail only, and the `;`-separated fields around it survive.
        secret = _azure_tail(44)
        matched = "AccountKey=" + secret
        text = f"DefaultEndpointsProtocol=https;AccountName=acct;{matched};EndpointSuffix=core"
        assert scrub_pii(text, ["api_keys"], salt="") == (
            "DefaultEndpointsProtocol=https;AccountName=acct;"
            f"{_key_token('AccountKey=', matched)};EndpointSuffix=core"
        )

    def test_a_dot_after_a_ya29_key_is_not_the_tail(self) -> None:
        # The `ya29.` dot is prefix, not tail: the tail stops at the next
        # dot, and `.x.co` survives the match.
        key = "ya29." + _key_tail(20)
        assert scrub_pii(key + ".x.co", ["api_keys"], salt="") == (
            f"{_key_token('ya29.', key)}.x.co"
        )

    def test_keys_at_string_start_and_end(self) -> None:
        # The boundary rule's clean edge cases: position 0 is clean, a
        # key running to the string's end matches, and the first
        # non-charset byte (punctuation, whitespace) ends the tail.
        key = "cn-" + _key_tail(20)
        assert scrub_pii(key, salt="") == _key_token("cn-", key)
        assert scrub_pii(key + " rotated", salt="") == _key_token("cn-", key) + " rotated"
        assert scrub_pii("rotate " + key, salt="") == "rotate " + _key_token("cn-", key)
        assert scrub_pii(key + "!", salt="") == _key_token("cn-", key) + "!"
        # A dot is not tail charset for any family (only the JWT grammar
        # carries dots, inside its own marker-scoped shape).
        assert scrub_pii(key + ".x.co", salt="") == _key_token("cn-", key) + ".x.co"

    def test_the_longest_prefix_wins_and_falls_through(self) -> None:
        # `sk-ant-` outranks bare `sk-` when its own grammar holds...
        key = "sk-ant-" + _key_tail(40)
        assert scrub_pii(key, ["api_keys"], salt="") == _key_token("sk-ant-", key)
        # ...and when it does not (one under the Anthropic minimum), the
        # longest-first discipline falls through to the bare `sk-`
        # family, whose tail swallows the `ant-` spelling: still
        # scrubbed, with the generic prefix. `wd-` shows the other side:
        # no fall-through to `w-` (the prefixes disagree at their second
        # char), so one under stays one under.
        short = "sk-ant-" + _key_tail(19)
        assert scrub_pii(short, ["api_keys"], salt="") == _key_token("sk-", short)

    def test_a_prefix_glued_to_a_charset_char_is_mid_token(self) -> None:
        # The boundary rule: `x` is tail-charset material, so the `ak-`
        # inside `xak-...` is that token's fragment, never a family head
        # — the same reasoning as the phone rule's clean-boundary cut.
        text = "xak-" + _key_tail(48)
        assert scrub_pii(text, ["api_keys"], salt="") == text
        # A second key glued directly after a key token's digest hex is
        # mid-token the same way (the hex is charset material),
        # conservative and documented; word-separated keys both scrub.
        first = scrub_pii("sk-" + _key_tail(48), ["api_keys"], salt="")
        second = "ghp_" + _key_tail(36)
        glued = first + second
        assert scrub_pii(glued, ["api_keys"], salt="") == glued
        assert scrub_pii(first + " " + second, ["api_keys"], salt="") == (
            first + " " + _key_token("ghp_", second)
        )

    def test_a_key_glued_to_contact_material_is_mid_token(self) -> None:
        # The boundary rule's reach past words (red-team pin, landed
        # behavior): a key glued with ZERO separator to a phone number's
        # last digit, an email domain's last letter, or a prior token's
        # digest hex is mid-token exactly the way `xak-` is — the keys
        # pass leaves it WHOLE, and no later pass can recover it (the
        # contact passes still scrub their own matches). Pinned so any
        # future widening of the boundary rule is a deliberate grammar
        # change with this corner named, never a drift.
        key = "sk-" + _key_tail(48)
        for text in (
            "+14155552671" + key,
            "415-555-2671" + key,
            "a@b.co" + key,
            "+14~abc123def456" + key,
        ):
            assert scrub_pii(text, ["api_keys"], salt="") == text
        # The composed pipeline scrubs the contact half; the glued key
        # survives whole — the documented mid-token cut applied to
        # contact glue.
        assert scrub_pii("+14155552671" + key, salt="") == (
            f"+14~{_hex12('+14155552671')}{key}"
        )

    def test_glued_keys_merge_into_one_maximal_tail(self) -> None:
        # The tail run is maximal: a second key's whole spelling is
        # charset material for the first family's tail, so the pair is
        # ONE match (over-redaction in the safe direction).
        a = "sk-" + _key_tail(48)
        b = "ghp_" + _key_tail(36)
        assert scrub_pii(a + b, ["api_keys"], salt="") == _key_token("sk-", a + b)

    def test_the_jwt_family_is_marker_scoped(self) -> None:
        # A bare eyJ-shaped triple is never touched: one consumer's API
        # legitimately carries eyJ-shaped non-secret cursors, so the JWT
        # family fires only behind the literal `Bearer ` marker.
        cursor = _JWT[len("Bearer ") :]
        assert scrub_pii(cursor, salt="") is cursor
        # Segments are [A-Za-z0-9_-]+ of any length >= 1: a 4-char middle
        # segment is a fine JWT shape, three segments being the grammar.
        jwt = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.c2ln.dozjgNryP4J3jVmNHc0FKW3YtV9"
        assert scrub_pii(jwt, ["api_keys"], salt="") == _key_token("Bearer", jwt)

    def test_none_uses_the_keys_default_tag_not_the_contact_tag(self) -> None:
        # salt=None resolves PER RULE: the contact rules keep
        # "tors/scrub_pii/v1" and the keys rule salts with its own
        # "tors/scrub_keys/v1" tag — a key digest can never alias a
        # contact digest at the default settings.
        key = "sk-" + _key_tail(48)
        assert scrub_pii(key) == f"sk-~{_hex12('tors/scrub_keys/v1' + key)}"
        assert scrub_pii(key) != f"sk-~{_hex12('tors/scrub_pii/v1' + key)}"

    def test_the_default_salt_splits_per_rule(self) -> None:
        key = "sk-" + _key_tail(48)
        text = "a@b.co " + key
        assert scrub_pii(text) == (
            f"@b.co~{_hex12('tors/scrub_pii/v1' + 'a@b.co')} "
            f"sk-~{_hex12('tors/scrub_keys/v1' + key)}"
        )

    def test_an_explicit_salt_applies_to_every_rule(self) -> None:
        key = "sk-" + _key_tail(48)
        text = "a@b.co +14155552671 " + key
        assert scrub_pii(text, salt="site") == (
            f"@b.co~{_hex12('site' + 'a@b.co')} "
            f"+14~{_hex12('site' + '+14155552671')} "
            f"sk-~{_hex12('site' + key)}"
        )

    def test_empty_salt_is_the_unsalted_digest_for_every_rule(self) -> None:
        key = "sk-" + _key_tail(48)
        text = "a@b.co +14155552671 " + key
        assert scrub_pii(text, salt="") == (
            f"@b.co~{_hex12('a@b.co')} "
            f"+14~{_hex12('+14155552671')} "
            f"sk-~{_hex12(key)}"
        )

    def test_keys_only_leaves_contact_material_alone(self) -> None:
        text = "a@b.co +14155552671 " + _OPENAI
        assert scrub_pii(text, ["api_keys"], salt="") == (
            f"a@b.co +14155552671 {_key_token('sk-', _OPENAI)}"
        )

    def test_none_composes_every_rule_keys_first(self) -> None:
        key = "sk-" + _key_tail(48)
        text = "a@b.co +14155552671 " + key
        expected = (
            f"@b.co~{_hex12('a@b.co')} "
            f"+14~{_hex12('+14155552671')} "
            f"{_key_token('sk-', key)}"
        )
        assert scrub_pii(text, salt="") == expected
        # ...and caller order is irrelevant, duplicates dedupe.
        assert scrub_pii(text, ["contact_phone", "api_keys", "contact_email"], salt="") == expected
        assert scrub_pii(key, ["api_keys", "api_keys"], salt="") == _key_token("sk-", key)

    def test_a_key_shaped_email_local_part_is_eaten_by_the_keys_pass(self) -> None:
        # keys BEFORE email, both orders of the composition pinned: the
        # keys pass tokenizes the key-shaped local part first, and the
        # email pass never sees the address whole — it fires on the key
        # token's digest hex as a fresh local part (hex is local-class
        # material), so the domain tokens too and the token prefix
        # survives. The email-only call eats the whole address as ONE
        # match (every key char is local-class), the divergence the pass
        # order exists to prevent. The composed output converges.
        key = "sk-" + _key_tail(48)
        text = key + "@x.co"
        keys_first = scrub_pii(text, salt="")
        hex12 = _hex12(key)
        assert keys_first == f"sk-~@x.co~{_hex12(hex12 + '@x.co')}"
        assert scrub_pii(text, ["contact_email"], salt="") == _email_token(text)
        assert scrub_pii(keys_first, salt="") is keys_first

    def test_a_key_bearing_a_phone_shape_is_eaten_whole(self) -> None:
        # keys BEFORE phone: a dash-separated ten-digit run inside a key
        # tail would be a domestic match if the phone pass ran first; the
        # canonical order runs keys first, so the phone pass sees only
        # the token (its `~` + 12 hex is a breaker) and the digits never
        # surface. The phone-only contrast shows what the order prevents.
        key = "sk-proj-415-555-2671" + _key_tail(20)
        text = "leaked " + key + " in an error"
        out = scrub_pii(text, salt="")
        assert out == f"leaked {_key_token('sk-proj-', key)} in an error"
        assert "415~" not in out
        phone_only = scrub_pii(text, ["contact_phone"], salt="")
        assert "-41~" in phone_only  # the domestic matcher ate the run

    def test_a_number_after_a_key_token_keeps_its_clean_run(self) -> None:
        # A key token's digest (`~` + 12 hex) is a token span for the
        # phone pass's existing generic breaker, and the byte after it is
        # a clean boundary: the number after a scrubbed key scrubs
        # exactly, never composed with the digest tail.
        key = "fw-" + _key_tail(48)
        text = key + " 415-555-2671"
        assert scrub_pii(text, salt="") == (
            f"{_key_token('fw-', key)} {_phone_token('415-555-2671')}"
        )

    @pytest.mark.parametrize(
        ("text", "prefix"), _KEY_VECTORS, ids=[v[1] for v in _KEY_VECTORS]
    )
    def test_key_tokens_are_fixed_points(self, text: str, prefix: str) -> None:
        # Idempotence by construction, verified per family: the token's
        # prefix ends in `-`/`_` (or is `AIza`/`Bearer`), the byte after
        # it is `~` (never tail charset), and no family prefix can be
        # spelled inside 12 lowercase digest hex — so the second scrub
        # never fires a rule and returns the original object.
        once = scrub_pii(text, salt="")
        assert once == _key_token(prefix, text)
        assert scrub_pii(once, salt="") is once


class TestKeyFamiliesContract:
    """The per-family selection: ``families=`` scopes the api_keys rule to
    a closed set of family names — the library's own closed-set-of-strings
    convention (the ``rules=`` / ``parse_errors_mode`` shape, not bit
    flags: a frozen ``(1<<N)-1`` sentinel would silently exclude the next
    family, and a second paradigm in a strings API is overengineering).
    ``None`` is every family this version knows (the set grows on new
    families — callers needing stability list names explicitly); a list
    selects exactly those families (duplicates dedupe, order is
    irrelevant); ``[]`` and unknown names are ``ValueError``s; and the
    knob is ignored when api_keys is not in the active rules."""

    def test_key_families_is_the_pinned_canonical_tuple(self) -> None:
        # The canonical family-name tuple, in the scanner table's order:
        # the base for "all but X" comprehensions, and the set the
        # unknown-name error names.
        assert KEY_FAMILIES == (
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
        )
        assert isinstance(KEY_FAMILIES, tuple)

    def test_a_single_family_selects_only_that_family(self) -> None:
        # families=["jwt"] scrubs only JWTs: the OpenAI key is preserved
        # whole (its grammar held, but its family is unselected — the
        # scanner spends the match without redacting it).
        text = f"{_JWT} {_OPENAI}"
        assert scrub_pii(text, ["api_keys"], families=["jwt"], salt="") == (
            f"{_key_token('Bearer', _JWT)} {_OPENAI}"
        )

    def test_all_but_jwt_leaves_jwts_verbatim_while_scrubbing_everything_else(self) -> None:
        # The maintainer's pinned use case: every family but "jwt" — the
        # JWT survives verbatim (preserved for separate logging) while an
        # OpenAI and an Anthropic key in the same excerpt still scrub.
        anthropic = "sk-ant-api03-" + _key_tail(95)
        text = f"leaked {_OPENAI} and {_JWT} and {anthropic}"
        families = [f for f in KEY_FAMILIES if f != "jwt"]
        assert scrub_pii(text, families=families, salt="") == (
            f"leaked {_key_token('sk-', _OPENAI)} and {_JWT} and {_key_token('sk-ant-', anthropic)}"
        )

    def test_all_but_one_family_is_still_exact(self) -> None:
        # The same recipe for an AWS key beside an OpenAI key: excluding
        # "aws" preserves the AKIA shape whole while the sk- key scrubs.
        text = f"{_AKIA} {_OPENAI}"
        families = [f for f in KEY_FAMILIES if f != "aws"]
        assert scrub_pii(text, families=families, salt="") == (
            f"{_AKIA} {_key_token('sk-', _OPENAI)}"
        )

    def test_duplicates_dedupe_and_order_is_irrelevant(self) -> None:
        text = f"{_JWT} {_AKIA}"
        assert scrub_pii(text, ["api_keys"], families=["jwt", "jwt"], salt="") == scrub_pii(
            text, ["api_keys"], families=["jwt"], salt=""
        )
        assert scrub_pii(text, ["api_keys"], families=["jwt", "aws"], salt="") == scrub_pii(
            text, ["api_keys"], families=["aws", "jwt"], salt=""
        )

    def test_empty_families_is_a_value_error_not_a_silent_no_op(self) -> None:
        # A silent no-op would be a misconfiguration trap: selecting
        # nothing is refused, with the recipe for what to write instead.
        with pytest.raises(ValueError) as exc:
            scrub_pii("a@b.co", families=[])
        assert str(exc.value) == (
            "families selects no key families; use None for all or list names"
        )

    def test_unknown_names_name_the_accepted_set(self) -> None:
        # The exact parse_errors_mode shape: the closed set as a tuple,
        # the offender in Rust's Debug quoting. The set is derived from
        # the tuple (not re-spelled) so only the tuple pin reddens when
        # the 14th family lands.
        with pytest.raises(ValueError) as exc:
            scrub_pii("a@b.co", families=["ssn"])
        assert str(exc.value) == (
            f"families must be one of ({', '.join(repr(f) for f in KEY_FAMILIES)}), "
            'not "ssn"'
        )
        # ...and the derived spelling equals the landed literal.
        assert str(exc.value) == (
            "families must be one of ('openai', 'anthropic', 'google', "
            "'fireworks', 'modal', 'github', 'minted', 'jwt', 'aws', "
            "'xai', 'gcp_oauth', 'pem', 'azure'), not \"ssn\""
        )

    def test_a_valid_name_plus_an_unknown_one_still_raises(self) -> None:
        with pytest.raises(ValueError, match="gcp_oauth"):
            scrub_pii("a@b.co", families=["jwt", "nope"])

    def test_families_is_harmless_when_api_keys_is_inactive(self) -> None:
        # An irrelevant knob never changes the scrub: a valid selection
        # over a contact-only call is simply ignored, documented.
        assert scrub_pii("a@b.co", ["contact_email"], families=["jwt"], salt="") == (
            _email_token("a@b.co")
        )
        assert scrub_pii("a@b.co", ["contact_email"], families=["jwt"]) == scrub_pii(
            "a@b.co", ["contact_email"]
        )

    def test_invalid_families_still_raise_when_api_keys_is_inactive(self) -> None:
        # The validation is at the argument boundary, never late: an
        # unknown name or an empty selection raises even though the knob
        # would have been ignored — the closed-set discipline holds
        # wherever the parameter is spelled.
        with pytest.raises(ValueError, match="must be one of"):
            scrub_pii("a@b.co", ["contact_email"], families=["nope"])
        with pytest.raises(ValueError, match="selects no key families"):
            scrub_pii("a@b.co", ["contact_email"], families=[])

    def test_a_bare_string_is_not_a_families_sequence(self) -> None:
        # The same sequence boundary as rules=: a bare string is a
        # TypeError, never an iterated character list.
        with pytest.raises(TypeError):
            scrub_pii("a@b.co", families="jwt")  # type: ignore[arg-type]

    def test_a_tuple_is_accepted_like_a_list(self) -> None:
        assert scrub_pii(_JWT, ["api_keys"], families=("jwt",), salt="") == (
            _key_token("Bearer", _JWT)
        )

    def test_a_set_is_not_a_families_sequence(self) -> None:
        # The same sequence boundary as rules=: a set is a TypeError
        # (symmetric with the rules= pin).
        with pytest.raises(TypeError):
            scrub_pii("a@b.co", families={"jwt"})  # type: ignore[arg-type]

    def test_the_mask_meets_the_fall_through(self) -> None:
        # The longest-first discipline crossed with the selection: the
        # mask is consulted AFTER the winning family is determined —
        # a one-under sk-ant- tail falls through to sk- FIRST, then the
        # mask decides.
        short = "sk-ant-" + _key_tail(19)
        # Anthropic selected, openai not: the fall-through winner is
        # sk- (unselected) — detected, preserved whole, counted.
        rep = scrub_pii_report(short, ["api_keys"], families=["anthropic"], salt="")
        assert rep["text"] == short
        assert rep["redacted"] == {}
        assert rep["skipped"] == {"openai": 1}
        # Openai selected: the same short key scrubs as generic sk-.
        assert scrub_pii(short, ["api_keys"], families=["openai"], salt="") == (
            _key_token("sk-", short)
        )
        # Both selected: still the generic prefix (the Anthropic grammar
        # never held).
        assert scrub_pii(short, ["api_keys"], families=["openai", "anthropic"], salt="") == (
            _key_token("sk-", short)
        )
        # The other side: wd- has no fall-through to w- (the prefixes
        # disagree at the second char), so one under stays one under —
        # no detection, no skip, the identity object.
        wd_short = "wd-" + _key_tail(42)
        assert scrub_pii(wd_short, ["api_keys"], families=["minted"], salt="") is wd_short
        rep = scrub_pii_report(wd_short, ["api_keys"], families=["minted"], salt="")
        assert rep == {"text": wd_short, "redacted": {}, "skipped": {}, "spans": []}

    def test_a_preserved_span_is_the_identity_object(self) -> None:
        # The zero-alloc discipline extends to the skipped path: a
        # detected-but-unselected match passes its bytes through
        # untouched, so the call returns the original object.
        assert scrub_pii(_JWT, ["api_keys"], families=["openai"], salt="") is _JWT
        assert scrub_pii(_OPENAI, ["api_keys"], families=["jwt"], salt="") is _OPENAI

    @pytest.mark.parametrize("family", [f for f in KEY_FAMILIES])
    def test_every_single_family_selection_round(self, family: str) -> None:
        # The mask-bit wiring, all thirteen: selecting exactly one
        # family scrubs every vector of that family and preserves every
        # other family's vectors whole (a bit swap between two families
        # reddens here, not as a parity mystery).
        own = [key for key, prefix in _KEY_VECTORS if _PREFIX_FAMILY[prefix] == family]
        assert own, family
        for key in own:
            assert scrub_pii(key, ["api_keys"], families=[family], salt="") == (
                _key_token(_family_token_prefix(family, key), key)
            ), (family, key)
        for key, prefix in _KEY_VECTORS:
            if _PREFIX_FAMILY[prefix] != family:
                assert scrub_pii(key, ["api_keys"], families=[family], salt="") == key, (
                    family,
                    key,
                )

    def test_the_new_families_scrub_verbatim_prefixes(self) -> None:
        # The five new families, end to end through the families= lens:
        # each selected alone scrubs with its verbatim prefix, and all
        # five fire under the default selection.
        pem = _pem_block("EC")
        azure = "AccountKey=" + _azure_tail(44)
        xai = "xai-" + _key_tail(20)
        gcp = "ya29." + _key_tail(20)
        for key, prefix, family in (
            (_AKIA, "AKIA", "aws"),
            (xai, "xai-", "xai"),
            (gcp, "ya29.", "gcp_oauth"),
            (pem, "PEM", "pem"),
            (azure, "AccountKey=", "azure"),
        ):
            assert scrub_pii(key, ["api_keys"], families=[family], salt="") == (
                _key_token(prefix, key)
            ), family
            assert scrub_pii(key, salt="") == _key_token(prefix, key), family


_REPORT_CONTACT_SALT = "tors/scrub_pii/v1"
_REPORT_KEYS_SALT = "tors/scrub_keys/v1"


def _family_token_prefix(family: str, matched: str) -> str:
    """The token's verbatim family prefix for one matched key: the
    matched head for the longest-first table families, the constant for
    the marker/span families. The test-side transcription of the
    scanner's own prefix table (the file's third-transcription
    discipline: hashlib digests, hand-spelled prefixes)."""
    table = {
        "anthropic": ("sk-ant-",),
        "google": ("AIza",),
        "fireworks": ("fw-", "fw_"),
        "modal": ("ak-", "wk-"),
        "github": ("github_pat_", "ghp_"),
        "minted": ("azxdev_", "wd-", "w-", "cn-"),
        "jwt": ("Bearer",),
        "aws": (),
        "xai": ("xai-",),
        "gcp_oauth": ("ya29.",),
        "pem": (),
        "azure": ("AccountKey=",),
    }
    if family == "openai":
        for prefix in ("sk-svcacct-", "sk-proj-", "sk-"):
            if matched.startswith(prefix):
                return prefix
        raise AssertionError(f"no openai head in {matched!r}")
    if family == "aws":
        # Non-tautological: the head is asserted from the input, never
        # mirrored from the implementation (a wrong-head token must fail
        # here, not agree).
        assert matched.startswith("AKIA") != matched.startswith("ASIA"), matched
        return "AKIA" if matched.startswith("AKIA") else "ASIA"
    if family == "pem":
        return "PEM"
    for prefix in table[family]:
        if matched.startswith(prefix):
            return prefix
    raise AssertionError(f"no {family} head in {matched!r}")


def _reconstruct(
    text: str, spans: list[dict[str, object]], contact_salt: str, keys_salt: str
) -> str:
    """Rebuild the scrubbed output from the input, the report's spans,
    and the digest construction — the strong invariant, pinned: the
    walk below must reproduce ``report["text"]`` byte-exactly. Spans are
    consumed in start order; three shapes need more than a blind splice,
    each the offset map's exact contract:

    - a ``contact_email`` span that starts exactly where the preceding
      ``api_keys`` span ended consumed that key token's digest hex as its
      local part (the keys-before-email corner): the digest is over
      ``hex12 + input[span]``, the hex re-derived from the key match.
    - an ``api_keys`` span overlapped by a preceding email span (the
      email's domain run flowed into the key token's verbatim head): the
      key contributes its token's REMNANT — the head the email ate is
      gone, only ``token[offset:]`` survives.
    - a ``contact_phone`` span inside a preceding email span (the domain
      spelled a number): the phone token replaces the affine substring
      of the staged email token, not a frontier slice.
    """
    # The discipline first: spans live inside the input, ordered by
    # start — a map_back bug emitting out-of-range or unordered spans
    # must fail here, never slice-and-agree below.
    assert all(
        isinstance(s["start"], int)
        and isinstance(s["end"], int)
        and 0 <= s["start"] <= s["end"] <= len(text)
        for s in spans
    ), spans
    assert [s["start"] for s in spans] == sorted(s["start"] for s in spans), spans
    # Pieces in emission order: passthrough slices and tokens (the
    # tokens mutable in place, so the affine-nesting corner's replace
    # lands in the final join). email_pieces tracks every staged email
    # span for the remnant branch (a nested phone may come between the
    # overlapping email and the key, so `prev` is not reliable there).
    pieces: list[list[object]] = []
    email_pieces: list[tuple[int, int]] = []
    frontier = 0
    prev: dict[str, object] | None = None
    for span in spans:
        typ = span["type"]
        assert isinstance(typ, str)
        start = span["start"]
        end = span["end"]
        assert isinstance(start, int) and isinstance(end, int)
        assert start >= 0 and end >= start
        if start > frontier:
            pieces.append(["text", text[frontier:start]])
            frontier = start
        if typ == "contact_email":
            # Email spans never overlap (stage-2 matches are disjoint
            # and the map is monotone); the corners touch at most — an
            # overlap here is a span bug, fail loud.
            assert start >= frontier, (span, prev)
        if (
            typ == "contact_email"
            and prev is not None
            and isinstance(prev["type"], str)
            and prev["type"].startswith("api_keys:")
            and prev["end"] == start
        ):
            # The hex-consumption corner: the match was
            # hex12 + the input suffix; the hex is the preceding key
            # token's digest half, re-derived here — and the staged key
            # token loses its eaten hex half (only its head survives).
            assert isinstance(prev["start"], int)
            key_match = text[prev["start"] : prev["end"]]
            matched = _hex12(keys_salt + key_match) + text[start:end]
            domain = matched.split("@", 1)[1]
            token = f"@{domain}~{_hex12(contact_salt + matched)}"
            assert pieces and pieces[-1][0] == "token"
            staged_token = pieces[-1][4]
            assert isinstance(staged_token, str)
            pieces[-1][4] = staged_token[: staged_token.index("~") + 1]
        elif typ == "contact_email":
            matched = text[start:end]
            domain = matched.split("@", 1)[1]
            token = f"@{domain}~{_hex12(contact_salt + matched)}"
        elif typ == "contact_phone" and start < frontier:
            # The affine-nesting corner: the number lived inside the
            # preceding email token's verbatim domain half — replace the
            # corresponding substring of the staged token in place. The
            # offset is in ORIGINAL token coordinates; the piece's shift
            # carries the length change of earlier nested replaces (two
            # numbers can share one domain).
            assert pieces and pieces[-1][0] == "token" and pieces[-1][1] == "contact_email"
            assert len(pieces[-1]) == 6
            _, _, sstart, send, stoken, shift = pieces[-1]
            assert isinstance(sstart, int) and isinstance(send, int)
            assert isinstance(stoken, str) and isinstance(shift, int)
            at = text[sstart:send].index("@") + sstart
            assert at < start
            off = start - at
            matched = text[start:end]
            token = f"{matched[:3]}~{_hex12(contact_salt + matched)}"
            assert stoken[off + shift : off + shift + (end - start)] == matched
            pieces[-1][4] = (
                stoken[: off + shift] + token + stoken[off + shift + (end - start) :]
            )
            pieces[-1][5] = shift + len(token) - (end - start)
            prev = span
            continue
        elif typ == "contact_phone":
            matched = text[start:end]
            token = f"{matched[:3]}~{_hex12(contact_salt + matched)}"
        else:
            assert typ.startswith("api_keys:")
            family = typ[len("api_keys:") :]
            matched = text[start:end]
            prefix = _family_token_prefix(family, matched)
            token = f"{prefix}~{_hex12(keys_salt + matched)}"
            if start < frontier:
                # The email-ate-the-head corner is the ONLY overlap an
                # api_keys span may arrive in — anything else is a span
                # bug, fail loud. The overlapping email is found by
                # coordinates, not by `prev` (a nested phone replace may
                # sit between them); exactly one email span can overlap
                # (email images are disjoint-or-touching).
                overlapped = [e for e in email_pieces if e[0] <= start < e[1]]
                assert len(overlapped) == 1, (span, email_pieces)
                email_end = overlapped[0][1]
                # The email-ate-the-head corner: only the token's
                # remnant survives past the email splice.
                offset = email_end - start if email_end < end else len(prefix)
                token = token[offset:]
                pieces.append(["token", typ, start, end, token, 0])
                frontier = end
                prev = span
                continue
        pieces.append(["token", typ, start, end, token, 0])
        if typ == "contact_email":
            email_pieces.append((start, end))
        frontier = end
        prev = span
    pieces.append(["text", text[frontier:]])
    return "".join(p[1] if p[0] == "text" else p[4] for p in pieces)


class TestScrubPiiReport:
    """``tors.scrub_pii_report``: the same scrub under the same single
    detach, plus the accounting — per-rule and per-family redacted
    counts, the skipped (detected-but-preserved) families, and the
    input-coordinate spans. The ``"text"`` field is the scrubbed string
    itself (``report["text"] == scrub_pii(...)`` for the same arguments,
    the consistency the fuzz target asserts byte-exact in Rust)."""

    def test_the_dict_shape_on_a_composed_excerpt(self) -> None:
        key2 = "sk-proj-" + _key_tail(48)
        text = f"a@b.co {_OPENAI} {key2} {_JWT}"
        rep = scrub_pii_report(text, salt="")
        assert rep["text"] == scrub_pii(text, salt="")
        assert rep["redacted"] == {"contact_email": 1, "api_keys": 3, "openai": 2, "jwt": 1}
        assert rep["skipped"] == {}
        email_end = len("a@b.co")
        key1_start = email_end + 1
        key1_end = key1_start + len(_OPENAI)
        key2_start = key1_end + 1
        key2_end = key2_start + len(key2)
        jwt_start = key2_end + 1
        jwt_end = jwt_start + len(_JWT)
        assert rep["spans"] == [
            {"type": "contact_email", "start": 0, "end": email_end},
            {"type": "api_keys:openai", "start": key1_start, "end": key1_end},
            {"type": "api_keys:openai", "start": key2_start, "end": key2_end},
            {"type": "api_keys:jwt", "start": jwt_start, "end": jwt_end},
        ]
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_empty_input_is_the_empty_report(self) -> None:
        assert scrub_pii_report("") == {
            "text": "",
            "redacted": {},
            "skipped": {},
            "spans": [],
        }

    def test_no_match_is_an_empty_accounting(self) -> None:
        text = "read 4096 bytes across 3 pages in 1200 ms"
        rep = scrub_pii_report(text, salt="")
        assert rep == {"text": text, "redacted": {}, "skipped": {}, "spans": []}

    def test_skipped_names_the_preserved_families(self) -> None:
        # The "we preserved a JWT, log it separately" signal: families
        # NOT in the active selection that would have matched anyway —
        # detection runs, redaction does not.
        text = f"{_JWT} {_OPENAI}"
        families = [f for f in KEY_FAMILIES if f != "jwt"]
        rep = scrub_pii_report(text, families=families, salt="")
        assert rep["text"] == f"{_JWT} {_key_token('sk-', _OPENAI)}"
        assert rep["redacted"] == {"api_keys": 1, "openai": 1}
        assert rep["skipped"] == {"jwt": 1}
        assert rep["spans"] == [
            {
                "type": "api_keys:openai",
                "start": len(_JWT) + 1,
                "end": len(_JWT) + 1 + len(_OPENAI),
            }
        ]
        # ...and skipped is always {} when families=None, even with keys
        # present (nothing was preserved on purpose).
        full = scrub_pii_report(text, salt="")
        assert full["skipped"] == {}
        assert full["redacted"] == {"api_keys": 2, "openai": 1, "jwt": 1}

    def test_a_single_family_reports_its_own_counts(self) -> None:
        text = f"{_JWT} {_OPENAI}"
        rep = scrub_pii_report(text, ["api_keys"], families=["jwt"], salt="")
        assert rep["text"] == f"{_key_token('Bearer', _JWT)} {_OPENAI}"
        assert rep["redacted"] == {"api_keys": 1, "jwt": 1}
        assert rep["skipped"] == {"openai": 1}
        assert rep["spans"] == [{"type": "api_keys:jwt", "start": 0, "end": len(_JWT)}]

    def test_spans_are_codepoint_indices_into_the_input(self) -> None:
        # A two-byte é and a four-byte emoji precede the key: byte
        # offsets would overshoot, codepoint indices slice the str
        # exactly (the reconstruction below indexes the str, so it pins
        # the units).
        text = "caf\u00e9 \U0001f600 " + _OPENAI
        rep = scrub_pii_report(text, salt="")
        assert len(rep["spans"]) == 1
        span = rep["spans"][0]
        assert span["type"] == "api_keys:openai"
        assert text[span["start"] : span["end"]] == _OPENAI
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_codepoint_indices_cover_email_and_phone_spans(self) -> None:
        # The byte-to-codepoint walk is per-pass, not per-rule: an email
        # after non-ASCII and a phone after an emoji must slice exactly
        # too.
        for text, match in (
            ("caf\u00e9 a@b.co", "a@b.co"),
            ("\U0001f600 415-555-2671", "415-555-2671"),
        ):
            rep = scrub_pii_report(text, salt="")
            assert len(rep["spans"]) == 1
            span = rep["spans"][0]
            assert text[span["start"] : span["end"]] == match
            assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_an_email_eating_a_key_head_with_a_number_in_its_domain(self) -> None:
        # The three-deep corner hypothesis found: the email's domain run
        # flows into a key token's verbatim head (overlapping the key
        # span) while the domain ALSO spells a number (a nested phone
        # span sorts between them) — the remnant branch must find the
        # overlapping email by coordinates, not by recency.
        key = "sk-" + _key_tail(48)
        text = "user@555.1234567.co." + key
        rep = scrub_pii_report(text, salt="")
        assert rep["spans"] == [
            {"type": "contact_email", "start": 0, "end": 22},
            {"type": "contact_phone", "start": 5, "end": 16},
            {"type": "api_keys:openai", "start": 20, "end": len(text)},
        ]
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_two_numbers_in_one_domain_fire_the_shift_path(self) -> None:
        # Two domestic runs inside one email token's domain: the second
        # affine replace lands on the shifted staged token (the helper's
        # shift path — and the report's — fires only here).
        text = "user@555.1234567g890-123-4567.co"
        rep = scrub_pii_report(text, salt="")
        assert rep["spans"] == [
            {"type": "contact_email", "start": 0, "end": len(text)},
            {"type": "contact_phone", "start": 5, "end": 16},
            {"type": "contact_phone", "start": 17, "end": 29},
        ]
        assert text[5:16] == "555.1234567"
        assert text[17:29] == "890-123-4567"
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_the_offset_map_survives_all_three_passes(self) -> None:
        # The later passes run on later stages: a phone number AFTER a
        # scrubbed key shifts in stage coordinates, and an email after a
        # key token sits in stage material — both spans must name the
        # INPUT positions, pinned literally.
        key = "fw-" + _key_tail(48)
        phone = "415-555-2671"
        email = "a@b.co"
        text = f"{key} {phone} {email}"
        rep = scrub_pii_report(text, salt="")
        key_end = len(key)
        phone_start = key_end + 1
        phone_end = phone_start + len(phone)
        email_start = phone_end + 1
        assert rep["spans"] == [
            {"type": "api_keys:fireworks", "start": 0, "end": key_end},
            {"type": "contact_phone", "start": phone_start, "end": phone_end},
            {"type": "contact_email", "start": email_start, "end": len(text)},
        ]
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_a_key_shaped_local_part_pins_the_hex_consumption_corner(self) -> None:
        # The keys-before-email corner: the keys pass eats the key-shaped
        # local part, and the email pass fires on hex12@domain — a match
        # that BEGAN inside the key token's digest. The offset map
        # records it from the token's input end (the first position whose
        # material is real input), and the reconstruction re-derives the
        # consumed hex from the preceding key span.
        text = _OPENAI + "@x.co"
        rep = scrub_pii_report(text, salt="")
        assert rep["redacted"] == {"contact_email": 1, "api_keys": 1, "openai": 1}
        assert rep["skipped"] == {}
        assert rep["spans"] == [
            {"type": "api_keys:openai", "start": 0, "end": len(_OPENAI)},
            {"type": "contact_email", "start": len(_OPENAI), "end": len(text)},
        ]
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_an_email_domain_that_spells_a_number_pins_the_affine_corner(self) -> None:
        # The phone pass re-tokens the domestic shape the email token's
        # verbatim DOMAIN spelled: the phone span maps back through the
        # token's verbatim head to the input's own domain digits.
        text = "user@555.1234567.co"
        rep = scrub_pii_report(text, salt="")
        assert rep["spans"] == [
            {"type": "contact_email", "start": 0, "end": len(text)},
            {"type": "contact_phone", "start": 5, "end": 16},
        ]
        assert text[5:16] == "555.1234567"
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_report_text_matches_scrub_pii_for_every_rule_lane(self) -> None:
        # The consistency contract, per lane: the report's text is the
        # scrub's output for the same arguments (the fuzz target asserts
        # the same in Rust, byte-exact, over arbitrary input).
        key = "sk-proj-415-555-2671" + _key_tail(20)
        text = f"a@b.co +14155552671 {key} {_JWT}"
        for rules in (None, ["api_keys"], ["contact_email"], ["contact_phone"], []):
            for families in (None, ["jwt"], [f for f in KEY_FAMILIES if f != "jwt"]):
                rep = scrub_pii_report(text, rules, salt="", families=families)
                assert rep["text"] == scrub_pii(text, rules, salt="", families=families)
                assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    def test_salt_semantics_flow_through_the_report(self) -> None:
        # An explicit salt salts every rule alike; None resolves per
        # rule (the contact tag and the keys tag) — pinned through the
        # report's tokens, with the reconstruction carrying each salt.
        text = f"a@b.co {_OPENAI}"
        site = scrub_pii_report(text, salt="site")
        assert site["text"] == (
            f"@b.co~{_hex12('site' + 'a@b.co')} {_key_token('sk-', _OPENAI, salt='site')}"
        )
        assert _reconstruct(text, site["spans"], "site", "site") == site["text"]
        defaulted = scrub_pii_report(text)
        assert defaulted["text"] == (
            f"@b.co~{_hex12(_REPORT_CONTACT_SALT + 'a@b.co')} "
            f"sk-~{_hex12(_REPORT_KEYS_SALT + _OPENAI)}"
        )
        assert (
            _reconstruct(text, defaulted["spans"], _REPORT_CONTACT_SALT, _REPORT_KEYS_SALT)
            == defaulted["text"]
        )

    def test_rules_subsets_scope_the_accounting(self) -> None:
        text = f"a@b.co +14155552671 {_OPENAI}"
        keys_only = scrub_pii_report(text, ["api_keys"], salt="")
        assert keys_only["redacted"] == {"api_keys": 1, "openai": 1}
        assert keys_only["skipped"] == {}
        assert [s["type"] for s in keys_only["spans"]] == ["api_keys:openai"]
        email_only = scrub_pii_report(text, ["contact_email"], salt="", families=["jwt"])
        assert email_only["redacted"] == {"contact_email": 1}
        assert email_only["skipped"] == {}
        assert [s["type"] for s in email_only["spans"]] == ["contact_email"]
        phone_only = scrub_pii_report(text, ["contact_phone"], salt="")
        assert phone_only["redacted"] == {"contact_phone": 1}
        assert phone_only["skipped"] == {}
        assert [s["type"] for s in phone_only["spans"]] == ["contact_phone"]
        assert _reconstruct(text, phone_only["spans"], "", "") == phone_only["text"]

    def test_two_skips_of_one_family_count_together(self) -> None:
        anthropic = "sk-ant-api03-" + _key_tail(95)
        text = f"{_JWT} {_JWT} {anthropic}"
        rep = scrub_pii_report(text, ["api_keys"], families=["anthropic"], salt="")
        assert rep["text"] == f"{_JWT} {_JWT} {_key_token('sk-ant-', anthropic)}"
        assert rep["redacted"] == {"api_keys": 1, "anthropic": 1}
        assert rep["skipped"] == {"jwt": 2}

    def test_a_mutated_span_breaks_the_reconstruction(self) -> None:
        # The oracle's teeth (the pyi-guard pattern): shifting a span's
        # start by one must NOT reconstruct — a helper that agrees with
        # wrong spans is decoration.
        text = f"a@b.co {_OPENAI}"
        rep = scrub_pii_report(text, salt="")
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]
        mutated = [dict(s) for s in rep["spans"]]
        mutated[0] = {**mutated[0], "start": mutated[0]["start"] + 1}
        assert _reconstruct(text, mutated, "", "") != rep["text"]
        mutated = [dict(s) for s in rep["spans"]]
        mutated[1] = {**mutated[1], "end": mutated[1]["end"] - 1}
        assert _reconstruct(text, mutated, "", "") != rep["text"]

    def test_reconstruction_holds_over_composed_excerpts(self) -> None:
        # The strong invariant over realistic compositions: keys (every
        # family), contacts, and separators in both orders — the splices
        # plus the digest construction reproduce the report's text.
        pieces = [v[0] for v in _KEY_VECTORS]
        contacts = ["a@b.co", "user@555.1234567.co", "+14155552671", "415-555-2671"]
        seps = [" ", "\n", ", ", " | ", "rotated ", "leaked ", "! "]
        texts = [
            f"{a}{s}{b}" for a in pieces for b in contacts for s in seps[:3]
        ] + [f"{c}{s}{k}" for c in contacts for k in pieces for s in seps[:3]]
        for text in texts:
            rep = scrub_pii_report(text, salt="")
            assert _reconstruct(text, rep["spans"], "", "") == rep["text"], text
            assert rep["text"] == scrub_pii(text, salt=""), text

    @given(
        pieces=st.lists(
            st.sampled_from(
                [v[0] for v in _KEY_VECTORS]
                + [
                    "a@b.co",
                    "user@555.1234567.co",
                    "user@555.1234567g890-123-4567.co",
                    "+14155552671",
                    "415-555-2671",
                    " ",
                    "\n",
                    ",",
                    ".",
                    "!",
                    "-",
                    "~",
                    "x",
                    "9",
                    "@",
                    "Bearer ",
                    "-----BEGIN ",
                    "glued",
                ]
            ),
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_reconstruction_holds_over_glued_compositions(self, pieces: list[str]) -> None:
        # The adversarial spelling of the invariant: keys glued to
        # contacts and to each other with zero or one separator — the
        # zero-gap hex corner, mid-token cuts, and maximal tails all
        # compose here, and the reconstruction must still reproduce the
        # report's text exactly.
        text = "".join(pieces)
        rep = scrub_pii_report(text, salt="")
        assert rep["text"] == scrub_pii(text, salt="")
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]

    @given(
        pieces=st.lists(
            st.sampled_from(
                [v[0] for v in _KEY_VECTORS]
                + [
                    "a@b.co",
                    "+14155552671",
                    "415-555-2671",
                    " ",
                    "\n",
                    ",",
                    ".",
                    "~",
                    "-",
                    "@",
                ]
            ),
            max_size=8,
        ),
        families=st.sampled_from(
            [
                None,
                ["jwt"],
                ["aws"],
                ["openai", "aws"],
                [f for f in KEY_FAMILIES if f != "jwt"],
            ]
        ),
    )
    @settings(max_examples=100)
    def test_reconstruction_holds_under_family_selection(
        self, pieces: list[str], families: list[str] | None
    ) -> None:
        # The selection matrix, structured-random: unselected families
        # spend whole and count skipped (never re-scanned inside), and
        # the reconstruction still reproduces the report's text — with
        # the per-rule salts the selection does not disturb.
        text = "".join(pieces)
        rep = scrub_pii_report(text, salt="", families=families)
        assert rep["text"] == scrub_pii(text, salt="", families=families)
        assert _reconstruct(text, rep["spans"], "", "") == rep["text"]
        if families is None:
            assert rep["skipped"] == {}
        selected = set(KEY_FAMILIES if families is None else families)
        for span in rep["spans"]:
            assert isinstance(span["type"], str)
            if span["type"].startswith("api_keys:"):
                assert span["type"][len("api_keys:") :] in selected, span
        for skipped_family in rep["skipped"]:
            assert skipped_family not in selected


class TestNdExhaustive:
    """H1: the Nd table is pinned exhaustively per interpreter, not by
    representatives. Every codepoint the running interpreter calls Nd
    must behave as a digit in the phone grammar (observable through
    ``+``-anchored matches); No/Nl numerics must never do so. Nd
    assignments are append-only across Unicode versions, so the Rust
    table (Unicode 16.0.0) covers every older interpreter's Nd set:
    the assertion is one-directional (interpreter-Nd ⇒ tors-digit)."""

    def test_every_interpreter_nd_digit_matches(self) -> None:
        import unicodedata

        nds = [chr(i) for i in range(0x110000) if unicodedata.category(chr(i)) == "Nd"]
        assert len(nds) > 600  # sanity: the table is not empty/trivial
        for c in nds:
            text = "+" + c * 8
            out = scrub_pii(text, ["contact_phone"], salt="")
            assert out != text, f"U+{ord(c):04X} is Nd but did not match"

    def test_no_nl_numerics_never_match(self) -> None:
        import unicodedata

        # Representative No/Nl pins (fast) plus an exhaustive sweep that
        # every No/Nl codepoint breaks the digit run ("+12<c>45678" must
        # stay untouched because the run dies at <c>).
        assert scrub_pii("+12\u00b2345678", ["contact_phone"], salt="") == "+12\u00b2345678"
        assert scrub_pii("+12\u216945678", ["contact_phone"], salt="") == "+12\u216945678"
        assert scrub_pii("\uFF0B12345678", ["contact_phone"], salt="") == "\uFF0B12345678"
        for i in range(0x110000):
            c = chr(i)
            if unicodedata.category(c) in ("No", "Nl"):
                text = "+12" + c + "45678"
                assert scrub_pii(text, ["contact_phone"], salt="") == text, f"U+{i:04X}"
