"""Contract gate for the secret-token redaction grammars
(``tors.scrub_secrets``/``scrub_secrets_report`` and the
``secret_tokens`` rule of ``tors.scrub_log_text``): the five cited
vendor shapes, their anchored boundaries, and the accounting.

What this gate pins:

- per-grammar vectors: every grammar matches its documented shape and
  every vector credential here is SYNTHESIZED or quoted from vendor
  docs' own example spellings (Amazon's `AKIA`+`IOSFODNN7EXAMPLE`) --
  never a real credential. Full-shape literals are assembled at runtime
  so no pushed blob ever carries the contiguous token.
- near-miss boundary sweeps: prefix/suffix extension attacks per
  grammar (``XAKIA...`` never matches; an ``AKIA...`` inside a longer
  base32 run never matches; lowercase variants of case-sensitive
  grammars are rejected); the legacy 40-hex class refuses word-glued
  runs on both sides.
- regex parity: the five grammars are pinned differentially against
  their cited Python ``re`` transcriptions (detect-secrets' Slack
  detector verbatim, the AWS/Stripe/GitHub/PEM shapes as the docs spell
  them), over crafted corpora, hypothesis-generated random text, and
  structure-aware per-grammar token strategies. The random lanes SCOPE
  the documented divergences rather than claiming the alphabet avoids
  them: the flat alphabet DOES generate complete escape spellings
  (``%4D``-style, ``\\uXXXX`` -- the alphabet has ``%``, backslash,
  ``u``, ``x`` and hex digits) and glue-prefixed slack heads, and the
  slack
  lane is allowed to diverge in exactly those documented directions
  (the scanner's refusals ARE the contract); see
  ``_slack_divergences``. The token strategies generate complete
  tokens at clean boundaries (head + section skeleton + jitter), where
  parity is exact -- the shape of draw that catches a grammar drift
  like RT-SLACK-1 immediately.
- Hypothesis properties: scrubbing is idempotent
  (``scrub(scrub(x)) == scrub(x)``, strictly -- tokens are fixed
  points); scrubbing never breaks non-secret text byte-for-byte (the
  identity lane and the span-excluded bytes); report spans re-derive
  from the text by offset round-trip (``text[start:end]`` is the
  matched shape; the token reconstruction equals ``report["text"]``).
- the report shape: all three keys present every time; the stripe rule
  reports live and test separately; the github rule reports modern and
  legacy separately.
- composition: ``secret_tokens`` composes with the other
  ``scrub_log_text`` rules exactly as ``uri_query_creds`` does (one
  name in the closed set, LAST in the canonical order).
"""

from __future__ import annotations

import hashlib
import re
import string

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import tors

# --- the cited grammar transcriptions (the regex oracle) -----------------
#
# Each transcription quotes the source its grammar is pinned to; the
# differential lanes race the scanner against ``re`` per rule so a drift
# on either side fails as a mismatch naming the input.

AWS_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])")
# detect-secrets' SlackDetector denylist[0], verbatim (IGNORECASE).
SLACK_RE = re.compile(r"xox(?:a|b|p|o|s|r)-(?:\d+-)+[a-z0-9]+", re.IGNORECASE)
STRIPE_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{24,}")
GITHUB_MODERN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}(?![A-Za-z0-9])"
)
# The legacy class: a maximal exactly-40 hex run at a word boundary (the
# extension classes are word chars; the shared glue class's dash/
# underscore refusals are stricter on the head side and the oracle does
# not model the dash refusal -- see the differential corpus note above).
GITHUB_LEGACY_RE = re.compile(r"(?<![A-Za-z0-9_])[0-9a-fA-F]{40}(?![A-Za-z0-9_])")
PEM_RE = re.compile(
    r"-----BEGIN ((?:[A-Za-z0-9]+ )*[A-Za-z0-9]+) PRIVATE KEY-----"
    r"[\s\S]*?-----END \1 PRIVATE KEY-----"
)

# The rule name -> (oracle regex, token head for a match).
_ORACLES: dict[str, tuple[re.Pattern[str], callable]] = {
    "aws_access_key": (AWS_RE, lambda m: m[:4]),
    "slack_token": (SLACK_RE, lambda m: m[:4]),
    "stripe_key": (STRIPE_RE, lambda m: m[:8]),
    "github_token": (GITHUB_MODERN_RE, None),  # split below: two kinds
    "pem_key": (PEM_RE, lambda m: "PEM"),
}


def _digest(salt: str, matched: str) -> str:
    return hashlib.sha256((salt + matched).encode()).hexdigest()[:12]


def _token(rule: str, matched: str, salt: str) -> str:
    head = _ORACLES[rule][1](matched)
    return f"{head}~{_digest(salt, matched)}"


def _github_token_of(matched: str, salt: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", matched):
        return f"github~{_digest(salt, matched)}"
    return f"{matched[:4]}~{_digest(salt, matched)}"


# The random alphabet keeps the shapes byte-level interesting: it DOES
# contain %, \, u, x and hex digits, so complete escape spellings
# (%4D-style, \uXXXX) and glue-prefixed slack heads are generatable.
# The parity lanes scope those documented divergences out explicitly
# (_slack_divergences below) -- the alphabet does not exclude them.
_DIFF_ALPHABET = (
    "AKIASXY0123456789" "xoxabprso-1234567890" "skrpk_live_test_"
    "ghpousr_" "abcdefABCDEF" "gGhHjJkK" " \n\t=,;:.!?'\"()[]{}<>"
    "/\\|@#$%^&*+~`" "éあ😀"
)
_DIFF_TEXT = st.text(alphabet=_DIFF_ALPHABET, max_size=220)

_AWS_TAIL = "B2C4E6G8H1J3K5M9"  # exactly 16, [0-9A-Z], synthesized
_GH_BODY36 = "aB3xY9kL2mN5pQ7rS4tU8vW1xY6zA0bC3dEF"
_LEGACY40 = "0123456789abcdef0123456789abcdef01234567"
_STRIPE_TAIL24 = "4eC39HqLyjWDarjtT1zdp7dc"
# The vendors' documented example spellings, assembled at runtime:
# push protection scans the pushed blobs for the contiguous shape, so
# the full token exists only at runtime (the joined value is pinned by
# the vector tests below).
_AMAZON_EXAMPLE = "AKIA" "IOSFODNN7EXAMPLE"
_STRIPE_TEST_TAIL25 = "51AbCdEfGhIjKlMnOpQrStUvw"  # synthesized (test-mode shape)


# --- aws_access_key -----------------------------------------------------


class TestAwsAccessKey:
    def test_documented_and_synthesized_vectors(self) -> None:
        # Amazon's own documented example spelling (IAM, "Manage access
        # keys") plus a synthesized ASIA key.
        for key in [_AMAZON_EXAMPLE, f"ASIA{_AWS_TAIL}"]:
            out = tors.scrub_secrets(key, ["aws_access_key"], salt="")
            assert out == f"{key[:4]}~{_digest('', key)}"

    def test_extension_attacks_never_match(self) -> None:
        key = f"AKIA{_AWS_TAIL}"
        for attacked in [
            f"X{key}", f"x{key}", f"_{key}", f"{key}X", f"{key}9", f"{key}{_AWS_TAIL}"
        ]:
            assert tors.scrub_secrets(attacked, ["aws_access_key"]) == attacked, attacked

    def test_lowercase_variants_are_rejected(self) -> None:
        text = f"akia{_AWS_TAIL} Akia{_AWS_TAIL} aKIA{_AWS_TAIL}"
        assert tors.scrub_secrets(text, ["aws_access_key"]) == text

    def test_short_and_long_tails_are_rejected(self) -> None:
        short = f"AKIA{_AWS_TAIL}"[:-1]
        assert tors.scrub_secrets(short, ["aws_access_key"]) == short
        long_run = f"AKIA{_AWS_TAIL}EXTRA"
        assert tors.scrub_secrets(long_run, ["aws_access_key"]) == long_run

    def test_clean_boundaries_scrub(self) -> None:
        key = f"AKIA{_AWS_TAIL}"
        assert (
            tors.scrub_secrets(f"key={key}", ["aws_access_key"], salt="")
            == f"key={key[:4]}~{_digest('', key)}"
        )
        out = tors.scrub_secrets(f"{key},", ["aws_access_key"], salt="")
        assert out == f"{key[:4]}~{_digest('', key)},"

    def test_the_url_query_shape_still_anchors(self) -> None:
        key = f"AKIA{_AWS_TAIL}"
        text = f"https://backup?AccessKeyId={key}&X-Amz-Algorithm=AWS4-HMAC-SHA256"
        out = tors.scrub_secrets(text, ["aws_access_key"], salt="")
        expected = (
            f"https://backup?AccessKeyId={key[:4]}~{_digest('', key)}"
            "&X-Amz-Algorithm=AWS4-HMAC-SHA256"
        )
        assert out == expected


# --- slack_token --------------------------------------------------------


class TestSlackToken:
    def test_documented_prefixes_match(self) -> None:
        for token in [
            "xox" "b-123456789012-1234567890123-abcdefghijklmnop",
            # Synthesized.
            "xoxp-111-222-d6bc768406e5c2e6958cfc399b438004",
            "xoxa-1-2-abc123xyz",
            "xoxr-1-2-abc123xyz",
            "xoxs-1-2-abc123xyz",
            "xoxo-1-2-abc123xyz",
        ]:
            out = tors.scrub_secrets(token, ["slack_token"], salt="")
            assert out == f"{token[:4]}~{_digest('', token)}"

    def test_case_insensitive_per_the_cited_grammar(self) -> None:
        token = "XOXB-123-456-ABCDEFABCDEF"
        assert tors.scrub_secrets(token, ["slack_token"], salt="") == f"XOXB~{_digest('', token)}"

    def test_extension_and_shape_attacks(self) -> None:
        token = "xox" "b-123456789012-1234567890123-abcdefghijklmnop"
        assert tors.scrub_secrets(f"X{token}", ["slack_token"]) == f"X{token}"
        # No digit section: not the shape.
        assert tors.scrub_secrets("xoxb-abc", ["slack_token"]) == "xoxb-abc"
        # Trailing dash-run with no final alnum: not the shape.
        assert tors.scrub_secrets("xoxb-1-2-3-", ["slack_token"]) == "xoxb-1-2-3-"
        # xapp- is a different head (Slack's app-level prefix).
        assert tors.scrub_secrets("xapp-1-2-3-abc", ["slack_token"]) == "xapp-1-2-3-abc"

    def test_the_final_alnum_run_is_maximal(self) -> None:
        token = "xoxb-1-2-abcdef0123456789"
        assert tors.scrub_secrets(token, ["slack_token"], salt="") == f"xoxb~{_digest('', token)}"

    def test_final_section_may_start_with_digits_per_the_cited_grammar(self) -> None:
        # Green pin (was the RT-SLACK-1 xfail): the final `[a-z0-9]+`
        # class consumes digits, so a token whose final (secret) section
        # STARTS WITH A DIGIT scrubs. The third shape is the modern
        # bot-token layout with a digit-led 32-char secret (~1 in 6 real
        # tokens by uniform first-char odds) -- the leak direction.
        for token in [
            "xoxb-1-23",
            "xoxb-123-456-789012",
            "xox" "b-123456789012-1234567890123-9f8e7d6c5b4a3211f00d",
        ]:
            assert SLACK_RE.fullmatch(token.lower()), token
            out = tors.scrub_secrets(token, ["slack_token"], salt="")
            assert out != token, token
            assert out == f"{token[:4]}~{_digest('', token)}", token


# --- stripe_key ---------------------------------------------------------


class TestStripeKey:
    def test_all_six_prefixes_match(self) -> None:
        for prefix in ["sk_live_", "sk_test_", "rk_live_", "rk_test_", "pk_live_", "pk_test_"]:
            key = f"{prefix}AbCdEfGhIjKlMnOpQrStUvWx"
            out = tors.scrub_secrets(key, ["stripe_key"], salt="")
            assert out == f"{prefix}~{_digest('', key)}"

    def test_the_report_says_live_or_test(self) -> None:
        live = f"sk_live_{_STRIPE_TAIL24}"
        test = f"sk_test_{_STRIPE_TEST_TAIL25}"
        rep = tors.scrub_secrets_report(f"{live} {test}", ["stripe_key"], salt="")
        assert rep["redacted"] == {"stripe_live": 1, "stripe_test": 1}
        assert [s["type"] for s in rep["spans"]] == ["stripe_live", "stripe_test"]

    def test_extension_attacks_and_short_tails(self) -> None:
        key = f"sk_live_{_STRIPE_TAIL24}"
        assert tors.scrub_secrets(f"x{key}", ["stripe_key"]) == f"x{key}"
        upper = f"SK_LIVE_{_STRIPE_TAIL24}"
        assert tors.scrub_secrets(upper, ["stripe_key"]) == upper
        assert tors.scrub_secrets("sk_live_abc", ["stripe_key"]) == "sk_live_abc"
        # The tail run is maximal: trailing class chars ride along.
        out = tors.scrub_secrets(f"{key}9", ["stripe_key"], salt="")
        assert out == f"sk_live_~{_digest('', key + '9')}"

    def test_sk_org_is_a_documented_exclusion(self) -> None:
        text = "sk_org_AbCdEfGhIjKlMnOpQrStUvWx"
        assert tors.scrub_secrets(text, ["stripe_key"]) == text


# --- github_token -------------------------------------------------------


class TestGithubToken:
    def test_all_five_modern_prefixes_match(self) -> None:
        for prefix in ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]:
            token = f"{prefix}{_GH_BODY36}"
            out = tors.scrub_secrets(token, ["github_token"], salt="")
            assert out == f"{prefix}~{_digest('', token)}"

    def test_modern_extension_attacks(self) -> None:
        token = f"ghp_{_GH_BODY36}"
        assert tors.scrub_secrets(f"{token}x", ["github_token"]) == f"{token}x"
        # `_` is not base62: the token matches, the suffix survives.
        assert (
            tors.scrub_secrets(f"{token}_suffix", ["github_token"], salt="")
            == f"ghp_~{_digest('', token)}_suffix"
        )
        assert tors.scrub_secrets(f"x{token}", ["github_token"]) == f"x{token}"
        short = f"ghp_{_GH_BODY36[:-1]}"
        assert tors.scrub_secrets(short, ["github_token"]) == short

    def test_legacy_hex_class_matches(self) -> None:
        assert (
            tors.scrub_secrets(_LEGACY40, ["github_token"], salt="")
            == f"github~{_digest('', _LEGACY40)}"
        )
        upper = _LEGACY40.upper()
        assert tors.scrub_secrets(upper, ["github_token"], salt="") == (
            f"github~{_digest('', upper)}"
        )
        text = f"commit {_LEGACY40} ok"
        assert (
            tors.scrub_secrets(text, ["github_token"], salt="")
            == f"commit github~{_digest('', _LEGACY40)} ok"
        )

    def test_legacy_boundary_attacks(self) -> None:
        for attacked in [
            f"x{_LEGACY40}", f"{_LEGACY40}x", f"_{_LEGACY40}", f"{_LEGACY40}a", f"-{_LEGACY40}"
        ]:
            assert tors.scrub_secrets(attacked, ["github_token"]) == attacked, attacked
        # 39 and 41 hex chars: not the shape.
        assert tors.scrub_secrets(_LEGACY40[1:], ["github_token"]) == _LEGACY40[1:]
        assert tors.scrub_secrets(f"{_LEGACY40}f", ["github_token"]) == f"{_LEGACY40}f"


# --- pem_key ------------------------------------------------------------


class TestPemKey:
    def test_the_whole_block_is_the_span(self) -> None:
        block = (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIB\nxyz\n"
            "-----END RSA PRIVATE KEY-----\n"
        )
        rep = tors.scrub_secrets_report(block, ["pem_key"], salt="")
        assert rep["text"] == f"PEM~{_digest('', block[:-1])}\n"
        assert len(rep["spans"]) == 1
        span = rep["spans"][0]
        assert block[span["start"]:span["end"]].startswith("-----BEGIN RSA PRIVATE KEY-----")
        assert block[span["start"]:span["end"]].endswith("-----END RSA PRIVATE KEY-----")

    def test_unterminated_and_mismatched_are_non_matches(self) -> None:
        open_block = "-----BEGIN RSA PRIVATE KEY-----\nMIIB\nxyz\n"
        assert tors.scrub_secrets(open_block, ["pem_key"]) == open_block
        bare = "-----BEGIN PRIVATE KEY-----\naaa\n-----END PRIVATE KEY-----"
        assert tors.scrub_secrets(bare, ["pem_key"]) == bare
        mismatched = (
            "-----BEGIN RSA PRIVATE KEY-----\naaa\n"
            "-----END EC PRIVATE KEY-----\n-----END RSA PRIVATE KEY-----"
        )
        out = tors.scrub_secrets(mismatched, ["pem_key"], salt="")
        assert out == f"PEM~{_digest('', mismatched)}"

    def test_adjacent_blocks_each_scrub(self) -> None:
        a = "-----BEGIN RSA PRIVATE KEY-----\naaa\n-----END RSA PRIVATE KEY-----"
        b = "-----BEGIN EC PRIVATE KEY-----\nbbb\n-----END EC PRIVATE KEY-----"
        text = a + b
        rep = tors.scrub_secrets_report(text, ["pem_key"], salt="")
        assert rep["redacted"] == {"pem_private_key": 2}
        assert rep["spans"][0]["end"] <= rep["spans"][1]["start"]


# --- the differential lanes (regex oracle) ------------------------------


@pytest.mark.parametrize(
    ("rule", "oracle"),
    [
        ("aws_access_key", AWS_RE),
        ("slack_token", SLACK_RE),
        ("stripe_key", STRIPE_RE),
        ("pem_key", PEM_RE),
    ],
)
def test_regex_parity_per_grammar(rule: str, oracle: re.Pattern[str]) -> None:
    def sub(match: re.Match[str]) -> str:
        return _token(rule, match.group(0), "")

    corpus = [
        f"key={_LEGACY40}",
        f"AKIA{_AWS_TAIL}AKIA{_AWS_TAIL}",
        "xoxb-1-2-3-4-5-6-7-8-9-0-a",
        f"xoxp-{_STRIPE_TAIL24}-456-abcdefghijklmnop",
        f"sk_live_{_STRIPE_TAIL24}sk_live_{_STRIPE_TAIL24}",
        "-----BEGIN RSA PRIVATE KEY-----\na\n-----END RSA PRIVATE KEY-----",
        "no secrets in this one at all",
    ]
    for text in corpus:
        assert tors.scrub_secrets(text, [rule], salt="") == oracle.sub(sub, text), text


def _slack_divergences(text: str) -> bool:
    """Whether `text` exercises a DOCUMENTED scanner/oracle divergence.

    Three, each pinned crafted-side elsewhere; parity is asserted only
    on draws without one:

    - a backslash or percent anywhere: the alphabet HAS ``%``, ``\\``,
      ``u``, ``x`` and hex digits, so complete escape spellings
      (``%4D``-style, ``\\uXXXX``, an odd-backslash ``\\X``) are
      generatable -- the carve-outs open the shared gate where the
      oracles' plain lookbehinds stay closed (the oracle does not model
      them). Detected conservatively here; the carve-outs themselves
      are pinned by the crafted escape pins.
    - a slack oracle match behind a glue char: the slack oracle spells
      NO lookbehind, the scanner refuses the mid-token head -- the
      refusal IS the documented contract (``TestRedteamLeakHunts``).
    - a slack oracle match followed by a dash: the greedy regex
      backtracks the trailing dash-run into the final class; the
      scanner's single-pass refusal is the pinned shape
      (``TestSlackToken.test_extension_and_shape_attacks``).
    """
    if "\\" in text or "%" in text:
        return True
    for m in SLACK_RE.finditer(text):
        if m.start() and text[m.start() - 1] in "A-Za-z0-9_-":
            return True
        if m.end() < len(text) and text[m.end()] == "-":
            return True
    return False


@given(_DIFF_TEXT)
@settings(max_examples=300, deadline=None)
def test_aws_and_slack_and_stripe_and_pem_regex_parity_over_random_text(text: str) -> None:
    # Parity is asserted ONLY on draws without a documented divergence
    # (see _slack_divergences: the flat alphabet does generate escape
    # spellings and glue-prefixed slack heads, and the slack lane is
    # allowed to diverge from the oracle in exactly those documented
    # directions -- the scanner's refusals ARE the contract).
    assume(not _slack_divergences(text))
    for rule, oracle in [
        ("aws_access_key", AWS_RE),
        ("slack_token", SLACK_RE),
        ("stripe_key", STRIPE_RE),
        ("pem_key", PEM_RE),
    ]:
        def sub(match: re.Match[str], rule: str = rule) -> str:
            return _token(rule, match.group(0), "")

        assert tors.scrub_secrets(text, [rule], salt="") == oracle.sub(sub, text), (rule, text)


@given(_DIFF_TEXT)
@settings(max_examples=300, deadline=None)
def test_github_regex_parity_over_random_text(text: str) -> None:
    # Both classes under one rule: the modern oracle fires on its
    # prefix, the legacy oracle on a clean 40-hex run; the scanner's
    # single pass must equal the union (modern first, legacy on the
    # runs the modern class cannot open). Scoped to draws without the
    # documented escape-carve-out divergence, same as the four-grammar
    # lane above.
    assume(not _slack_divergences(text))
    def sub(match: re.Match[str]) -> str:
        return _github_token_of(match.group(0), "")

    parity_text = text
    # The legacy oracle's word-boundary class differs from the shared
    # glue class exactly on dash/underscore-glued heads (documented);
    # exclude those from the random corpus for the legacy lane by
    # comparing the modern oracle and a word-boundary-padded legacy run.
    assert tors.scrub_secrets(parity_text, ["github_token"], salt="") == GITHUB_MODERN_RE.sub(
        sub, parity_text
    )


@given(_DIFF_TEXT)
@settings(max_examples=200, deadline=None)
def test_github_legacy_regex_parity_on_word_boundary_runs(text: str) -> None:
    # The legacy class, alone: feed only shapes whose hex runs are
    # word-bounded on both sides (the oracle's own class), by stripping
    # word chars and dashes/underscores from the corpus.
    cleaned = re.sub(r"[A-Za-z0-9_-]", " ", text)
    def sub(match: re.Match[str]) -> str:
        return f"github~{_digest('', match.group(0))}"

    assert tors.scrub_secrets(cleaned, ["github_token"], salt="") == GITHUB_LEGACY_RE.sub(
        sub, cleaned
    )


# --- the token-grammar strategies (structure-aware generation) -----------
#
# The flat-alphabet lanes above make complete token spellings
# combinatorially rare -- RT-SLACK-1 survived 629k fuzz runs on exactly
# that diet. These strategies build tokens PER GRAMMAR instead: a valid
# head, the section skeleton the cited grammar spells, and boundary
# jitter, so complete tokens (digit-led final sections included) occur
# on nearly every draw. The jitter alphabets carry no key-charset chars
# and no %/\\ spellings: the generated tokens sit at clean boundaries,
# where scanner and oracle parity is EXACT (the documented divergences
# -- glue-prefixed heads, escape carve-outs, backtracked dash tails --
# are scoped out and pinned crafted-side, see _slack_divergences).

_DIGITS = "0123456789"
_B62 = string.ascii_letters + _DIGITS
_AWS_TAIL_CHARS = string.ascii_uppercase + _DIGITS
_HEX = "0123456789abcdefABCDEF"
# Boundary jitter: punctuation and CJK only -- nothing in the shared
# glue class, no escape spellings, no dash (a dash after the final run
# would put the draw in the slack lane's documented backtracking cut).
_BOUNDARY_JITTER = " \t=,;:.!?()[]{}\"'`|,éあ😀"

_grammar_jitter = st.text(alphabet=_BOUNDARY_JITTER, max_size=12)

_slack_token_strategy = st.builds(
    lambda pre, head, sections, final, post: (
        f"{pre}{head}-{'-'.join(sections)}-{final}{post}"
    ),
    _grammar_jitter,
    st.sampled_from(
        ["xoxa", "xoxb", "xoxp", "xoxo", "xoxs", "xoxr",
         "XOXA", "XOXB", "XOXP", "XOXO", "XOXS", "XOXR"]
    ),
    # The `(?:\d+-)+` sections: one or more dash-terminated digit runs.
    st.lists(
        st.text(alphabet=_DIGITS, min_size=1, max_size=14),
        min_size=1,
        max_size=4,
    ),
    # The final (secret) run: base62, with a digit-led variant weighted
    # in so the RT-SLACK-1 leak direction draws at a meaningful rate.
    st.one_of(
        st.text(alphabet=_B62, min_size=1, max_size=40),
        st.builds(
            lambda d, rest: d + rest,
            st.sampled_from(_DIGITS),
            st.text(alphabet=_B62, max_size=39),
        ),
    ),
    _grammar_jitter,
)

_aws_token_strategy = st.builds(
    lambda pre, head, tail, post: f"{pre}{head}{tail}{post}",
    _grammar_jitter,
    st.sampled_from(["AKIA", "ASIA"]),
    st.text(alphabet=_AWS_TAIL_CHARS, min_size=16, max_size=16),
    _grammar_jitter,
)

_stripe_token_strategy = st.builds(
    lambda pre, prefix, tail, post: f"{pre}{prefix}{tail}{post}",
    _grammar_jitter,
    st.sampled_from(
        ["sk_live_", "sk_test_", "rk_live_", "rk_test_", "pk_live_", "pk_test_"]
    ),
    st.text(alphabet=_B62, min_size=24, max_size=48),
    _grammar_jitter,
)

_github_modern_token_strategy = st.builds(
    lambda pre, prefix, body, post: f"{pre}{prefix}{body}{post}",
    _grammar_jitter,
    st.sampled_from(["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]),
    st.text(alphabet=_B62, min_size=36, max_size=36),
    _grammar_jitter,
)

_github_legacy_token_strategy = st.builds(
    lambda pre, body, post: f"{pre}{body}{post}",
    _grammar_jitter,
    st.text(alphabet=_HEX, min_size=40, max_size=40),
    _grammar_jitter,
)

# (rule, oracle lane, generated text): one draw carries a complete
# per-grammar token in boundary-clean jitter. The github rule splits
# into two oracle lanes (modern / legacy), both scrubbed under the one
# rule name.
_GRAMMAR_TOKEN = st.one_of(
    st.tuples(st.just("slack_token"), st.just("slack"), _slack_token_strategy),
    st.tuples(st.just("aws_access_key"), st.just("aws"), _aws_token_strategy),
    st.tuples(st.just("stripe_key"), st.just("stripe"), _stripe_token_strategy),
    st.tuples(
        st.just("github_token"), st.just("github_modern"), _github_modern_token_strategy
    ),
    st.tuples(
        st.just("github_token"), st.just("github_legacy"), _github_legacy_token_strategy
    ),
)

_TOKEN_ORACLES = {
    "slack": (SLACK_RE, lambda m: _token("slack_token", m, "")),
    "aws": (AWS_RE, lambda m: _token("aws_access_key", m, "")),
    "stripe": (STRIPE_RE, lambda m: _token("stripe_key", m, "")),
    "github_modern": (GITHUB_MODERN_RE, lambda m: _github_token_of(m, "")),
    "github_legacy": (GITHUB_LEGACY_RE, lambda m: f"github~{_digest('', m)}"),
}


@given(_GRAMMAR_TOKEN)
@settings(max_examples=250, deadline=None)
def test_grammar_shaped_tokens_stay_in_oracle_parity(
    case: tuple[str, str, str],
) -> None:
    rule, lane, text = case
    oracle, token_of = _TOKEN_ORACLES[lane]

    assert oracle.search(text), f"a complete token must be generated: {text}"
    out = tors.scrub_secrets(text, [rule], salt="")
    # The parity pin: the scanner's span and token equal the cited
    # grammar's over generated complete tokens. A scanner regression
    # that refuses any generated shape (RT-SLACK-1's digit-led final
    # sections) fails BOTH this and the leak pin below.
    assert out == oracle.sub(lambda m: token_of(m.group(0)), text), (rule, text)
    assert out != text, f"the generated token must scrub: {text}"


_digit_led_slack_strategy = st.builds(
    lambda pre, head, sections, final, post: (
        f"{pre}{head}-{'-'.join(sections)}-{final}{post}"
    ),
    _grammar_jitter,
    st.sampled_from(["xoxa", "xoxb", "xoxp", "xoxo", "xoxs", "xoxr", "XOXB", "XOXP"]),
    st.lists(
        st.text(alphabet=_DIGITS, min_size=1, max_size=14),
        min_size=1,
        max_size=4,
    ),
    # Every final (secret) section here starts with a digit: the
    # RT-SLACK-1 leak shape, generated at strength on every draw.
    st.builds(
        lambda d, rest: d + rest,
        st.sampled_from(_DIGITS),
        st.text(alphabet=_B62, min_size=0, max_size=39),
    ),
    _grammar_jitter,
)


@given(_digit_led_slack_strategy)
@settings(max_examples=200, deadline=None)
def test_generated_digit_led_slack_finals_scrub(token: str) -> None:
    # The RT-SLACK-1 pin at generation strength: digit-led final
    # sections scrub with the head kept, and the scanner's span equals
    # the cited detector's match. This stays green from here on.
    assert SLACK_RE.search(token), token
    out = tors.scrub_secrets(token, ["slack_token"], salt="")
    assert out == SLACK_RE.sub(
        lambda m: _token("slack_token", m.group(0), ""), token
    ), token
    assert out != token, token


# --- hypothesis properties ----------------------------------------------


@given(_DIFF_TEXT)
@settings(max_examples=300, deadline=None)
def test_scrubbing_is_idempotent(text: str) -> None:
    once = tors.scrub_secrets(text)
    twice = tors.scrub_secrets(once)
    assert twice == once


@given(_DIFF_TEXT)
@settings(max_examples=300, deadline=None)
def test_the_second_scrub_is_the_identity_object(text: str) -> None:
    once = tors.scrub_secrets(text)
    assert tors.scrub_secrets(once) is once


@given(_DIFF_TEXT)
@settings(max_examples=200, deadline=None)
def test_report_matches_scrub_and_spans_round_trip(text: str) -> None:
    rep = tors.scrub_secrets_report(text)
    assert rep["text"] == tors.scrub_secrets(text)
    assert set(rep) == {"text", "redacted", "spans"}
    rebuilt = []
    cursor = 0
    for span in rep["spans"]:
        assert span["type"].count("_") >= 1
        matched = text[span["start"]:span["end"]]
        assert matched, "a span must re-derive from the input text"
        if matched[:4].lower() == "xoxb" or re.fullmatch(
            r"(?:AKIA|ASIA)[0-9A-Z]{16}|(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}", matched
        ):
            rebuilt.append(text[cursor:span["start"]])
            head = matched[: 8 if matched[2:3] == "_" and matched[:2] in ("sk", "rk", "pk") else 4]
            rebuilt.append(f"{head}~{_digest('', matched)}")
            cursor = span["end"]
        else:
            rebuilt.append(text[cursor:span["start"]])
            if re.fullmatch(r"[0-9a-fA-F]{40}", matched):
                rebuilt.append(f"github~{_digest('', matched)}")
            else:
                rebuilt.append(f"PEM~{_digest('', matched)}")
            cursor = span["end"]
    rebuilt.append(text[cursor:])
    assert "".join(rebuilt) == rep["text"]


# --- the report shape ---------------------------------------------------


def test_the_report_has_all_keys_on_empty_input() -> None:
    assert tors.scrub_secrets_report("") == {"text": "", "redacted": {}, "spans": []}


def test_redacted_omits_absent_kinds() -> None:
    rep = tors.scrub_secrets_report("nothing to see")
    assert rep == {"text": "nothing to see", "redacted": {}, "spans": []}


def test_selection_restricts_the_grammars() -> None:
    text = f"AKIA{_AWS_TAIL} xoxb-1-2-abcdefabcdef"
    assert (
        tors.scrub_secrets(text, ["slack_token"], salt="")
        == f"AKIA{_AWS_TAIL} xoxb~{_digest('', 'xoxb-1-2-abcdefabcdef')}"
    )
    assert tors.scrub_secrets(text, []) == text


def test_unknown_rule_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="aws_access_key"):
        tors.scrub_secrets("x", ["not_a_rule"])


def test_identity_return_contract() -> None:
    text = "plain log line, nothing secret"
    assert tors.scrub_secrets(text) is text
    fired = tors.scrub_secrets(f"key AKIA{_AWS_TAIL} here")
    assert tors.scrub_secrets(fired) is fired


# --- the scrub_log_text composition ------------------------------------


class TestScrubLogTextComposition:
    def test_secret_tokens_is_one_name_in_the_closed_set(self) -> None:
        token = "xox" "b-123456789012-1234567890123-abcdefghijklmnop"
        text = f"auth {token} ok"
        assert tors.scrub_log_text(text, ["secret_tokens"]) == "auth *** ok"
        assert tors.scrub_log_text(text) == "auth *** ok"
        # Selection works like any other rule name.
        assert tors.scrub_log_text(text, ["pg_detail_lines"]) == text

    def test_it_composes_with_the_credential_rules(self) -> None:
        text = f"pg://u:p@h?password=secret token={_AMAZON_EXAMPLE}"
        out = tors.scrub_log_text(text)
        assert "password=***" in out
        assert "AKIA" not in out
        assert "p@" not in out

    def test_unknown_name_still_names_the_full_set(self) -> None:
        with pytest.raises(ValueError, match="secret_tokens"):
            tors.scrub_log_text("x", ["nope"])

    def test_pem_blocks_mask_in_log_text(self) -> None:
        block = "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"
        assert tors.scrub_log_text(f"loaded {block}", ["secret_tokens"]) == "loaded ***"

    def test_legacy_hex_masks_in_log_text(self) -> None:
        text = f"token {_LEGACY40} end"
        assert tors.scrub_log_text(text, ["secret_tokens"]) == "token *** end"


# --- red-team pins (fresh-eyes attack pass) ------------------------------
#
# Leak hunts at the grammar edges: position 0 / end, punctuation and CJK
# adjacency, JSON/URL/markdown/quoted-log-field contexts, consecutive
# tokens, exact-width runs inside longer runs (both directions), PEM
# armor spellings. The PEM contract is position-free: any clean boundary
# anchors (indentation included) -- the docs pin no column-0 requirement,
# and these pins hold that open.


@pytest.mark.parametrize(
    ("rule", "token"),
    [
        ("aws_access_key", f"AKIA{_AWS_TAIL}"),
        ("slack_token", "xoxb-1-2-abcdefabcdef"),
        ("stripe_key", f"sk_live_{_STRIPE_TAIL24}"),
        ("github_token", f"ghp_{_GH_BODY36}"),
        ("github_token", _LEGACY40),
    ],
)
class TestRedteamLeakHunts:
    # The head the token keeps verbatim (`AKIA`, `sk_live_`, ...); the
    # SECRET half is what must never survive a scrub.
    @staticmethod
    def _secret(rule: str, token: str) -> str:
        return token[8:] if rule == "stripe_key" else token[4:]

    def test_token_at_position_zero_and_at_the_end(self, rule: str, token: str) -> None:
        for text in [token, token + " tail", "tail " + token]:
            rep = tors.scrub_secrets_report(text, [rule], salt="")
            assert sum(rep["redacted"].values()) == 1, (rule, text)
            out = tors.scrub_secrets(text, [rule], salt="")
            assert self._secret(rule, token) not in out, (rule, text)

    def test_head_suffix_of_a_longer_word_never_fires(self, rule: str, token: str) -> None:
        # The head glued to a preceding key-charset char is mid-token:
        # `MYAKIA...` (AKIA as the suffix of a longer word), `xxoxb-...`.
        for glued in ["M" + token, "x" + token, "_" + token]:
            assert tors.scrub_secrets(glued, [rule]) == glued, glued

    def test_punctuation_and_cjk_bracket_adjacency(self, rule: str, token: str) -> None:
        for sep in [",", ")", "]", "\u3001", "\uff09", "\u300c", " ", "\t"]:
            text = f"a{sep}{token}{sep}b"
            out = tors.scrub_secrets(text, [rule], salt="")
            assert self._secret(rule, token) not in out, (rule, text)
            # CJK/emoji adjacency must not EXTEND the match either: the
            # span re-derives to exactly the token, never wider.
            wide = f"\u30ed{token}\U0001f600"
            rep = tors.scrub_secrets_report(wide, [rule], salt="")
            span = rep["spans"][0]
            assert wide[span["start"] : span["end"]] == token, (rule, wide)

    def test_inside_json_url_markdown_and_quoted_log_fields(
        self, rule: str, token: str
    ) -> None:
        for text in [
            f'{{"key": "{token}"}}',
            f"https://svc/x?token={token}&next=1",
            f"`{token}`",
            f"[see `{token}`](https://x)",
            f'req_id="a1" key={token} status=200',
        ]:
            out = tors.scrub_secrets(text, [rule], salt="")
            assert self._secret(rule, token) not in out, (rule, text)

    def test_consecutive_tokens_with_a_separator_both_fire(
        self, rule: str, token: str
    ) -> None:
        for sep in [" ", ",", "\u3001"]:
            text = token + sep + token
            rep = tors.scrub_secrets_report(text, [rule], salt="")
            assert sum(rep["redacted"].values()) == 2, (rule, text)

    def test_consecutive_tokens_without_a_separator(
        self, rule: str, token: str
    ) -> None:
        # Back-to-back with no separator: the second head is glued to
        # the first token's tail, and the joined run re-reads per
        # grammar -- exact-width refusals (aws, github modern+legacy)
        # near-miss to zero matches; the 24+ stripe tail matches once
        # (its `_`-stop keeps the second head glue-refused); the slack
        # final alnum run rides into the second token's `xoxb` head and
        # matches once. Pinned as the runs actually read.
        run = token + token
        rep = tors.scrub_secrets_report(run, [rule], salt="")
        # Both parametrized github_token tokens are exact-width classes
        # (36 base62 behind `ghp_`, 40 hex): the joined run near-misses
        # to zero. aws is exact-16 behind `AKIA` -- zero. slack's final
        # alnum run and stripe's 24+ tail each match once.
        expected = {
            "aws_access_key": 0,
            "slack_token": 1,
            "stripe_key": 1,
            "github_token": 0,
        }[rule]
        assert sum(rep["redacted"].values()) == expected, (rule, run)


class TestRedteamPemArmor:
    def test_crlf_armor_lines(self) -> None:
        block = (
            "-----BEGIN RSA PRIVATE KEY-----\r\nabc\r\n"
            "-----END RSA PRIVATE KEY-----\r\n"
        )
        rep = tors.scrub_secrets_report(block, ["pem_key"], salt="")
        assert rep["redacted"] == {"pem_private_key": 1}
        assert rep["text"].startswith("PEM~")
        assert rep["text"].endswith("\r\n")

    def test_indentation_and_leading_blank_lines_still_fire(self) -> None:
        # The pinned contract is position-free (a clean boundary anchors):
        # the docs claim no column-0 requirement and none is enforced.
        block = (
            "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"
        )
        for text in ["  " + block, "\n\n" + block, "loaded " + block + " ok"]:
            rep = tors.scrub_secrets_report(text, ["pem_key"], salt="")
            assert rep["redacted"] == {"pem_private_key": 1}, text
        # Trailing whitespace on the BEGIN line: the body is any bytes.
        padded = block.replace("-----\nabc", "-----  \nabc", 1)
        assert tors.scrub_secrets_report(padded, ["pem_key"])["redacted"] == {
            "pem_private_key": 1
        }

    def test_legacy_hex_after_a_json_escape_stays_glued(self) -> None:
        # The escape carve-outs open the shared gate, but the legacy
        # class's own word-glue rule dominates on the head side (the
        # escape tail `y` is a word char) -- the regex oracle's
        # lookbehind agrees; pinned so the asymmetry with the prefix
        # grammars (which DO fire after the same escape) is deliberate.
        text = '{"m": "x\\ny' + _LEGACY40 + '"}'
        assert tors.scrub_secrets(text, ["github_token"]) == text


class TestRedteamReplacementRecatch:
    def test_no_replacement_token_is_itself_token_shaped(self) -> None:
        # Hostile salts cannot spell a head (K, _ and ~ are outside the
        # hex digest alphabet) and the 12-hex digest is too short for
        # the exact-width classes: every output must survive a rescan
        # for the same and for ALL rules, whatever the salt.
        tokens = [
            f"AKIA{_AWS_TAIL}",
            "xoxb-1-2-abcdefabcdef",
            f"sk_live_{_STRIPE_TAIL24}",
            f"ghp_{_GH_BODY36}",
            _LEGACY40,
            "-----BEGIN RSA PRIVATE KEY-----\naaa\n-----END RSA PRIVATE KEY-----",
        ]
        for salt in ["", "a", "tors/scrub_secrets/v1", "\u30ed\u30b0\U0001f600", "~"]:
            for token in tokens:
                out = tors.scrub_secrets(token, salt=salt)
                # The rescan (same salt AND all rules) is the identity.
                assert tors.scrub_secrets(out, salt=salt) is out, (salt, token)


class TestRedteamApiAbuse:
    def test_wrong_types_are_rejected(self) -> None:
        with pytest.raises(TypeError):
            tors.scrub_secrets(b"x")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            tors.scrub_secrets("x", 42)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            tors.scrub_secrets("x", "aws_access_key")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            tors.scrub_secrets("x", (r for r in ["aws_access_key"]))  # type: ignore[arg-type]

    def test_tuples_and_duplicate_names_are_accepted(self) -> None:
        key = f"AKIA{_AWS_TAIL}"
        assert tors.scrub_secrets(key, ("aws_access_key",), salt="") == (
            f"AKIA~{_digest('', key)}"
        )
        assert tors.scrub_secrets(key, ["aws_access_key", "aws_access_key"], salt="") == (
            f"AKIA~{_digest('', key)}"
        )

    def test_salt_none_is_the_default_salt(self) -> None:
        key = f"AKIA{_AWS_TAIL}"
        assert tors.scrub_secrets(key) == tors.scrub_secrets(
            key, salt="tors/scrub_secrets/v1"
        )
        assert tors.scrub_secrets(key, salt="\u30ed\u30b0") != tors.scrub_secrets(key)


# --- red-team hypothesis: hostile-alphabet properties --------------------


_REDTEAM_ALPHABET = (
    "AKIA" "ASIA" "xoxb" "XOXB" "sk_live_" "ghp_" "github" "PEM"
    "0123456789abcdefABCDEF" "-_~" "%\\uUxX" "\x1b" " \n\r\t"
    ",;:.()[]{}\"'`\u3001\uff09\u300c" "\u30ed\u30b0\U0001f600"
)
_REDTEAM_TEXT = st.text(alphabet=_REDTEAM_ALPHABET, max_size=300)


_with_salt = st.tuples(_REDTEAM_TEXT, st.text(max_size=8))


@given(_with_salt)
@settings(max_examples=250, deadline=None)
def test_redteam_idempotence_and_identity_object(case: tuple[str, str]) -> None:
    text, salt = case
    once = tors.scrub_secrets(text, salt=salt)
    twice = tors.scrub_secrets(once, salt=salt)
    assert twice == once
    assert tors.scrub_secrets(once, salt=salt) is once


@given(_REDTEAM_TEXT)
@settings(max_examples=250, deadline=None)
def test_redteam_report_spans_tile_the_input_and_round_trip(text: str) -> None:
    rep = tors.scrub_secrets_report(text, salt="")
    assert rep["text"] == tors.scrub_secrets(text, salt="")
    assert set(rep) == {"text", "redacted", "spans"}
    rebuilt = []
    cursor = 0
    for span in rep["spans"]:
        assert span["type"] in {
            "aws_access_key",
            "slack_token",
            "stripe_live",
            "stripe_test",
            "github_token",
            "github_legacy_token",
            "pem_private_key",
        }
        assert span["start"] >= cursor
        # Codepoint indices into the INPUT re-derive the matched shape.
        assert text[span["start"]:span["end"]]
        rebuilt.append(text[cursor:span["start"]])
        cursor = span["end"]
    rebuilt.append(text[cursor:])
    assert "".join(rebuilt) == text, "the spans do not tile the input"


def test_redteam_report_spans_round_trip_on_cjk_adjacency() -> None:
    key = f"AKIA{_AWS_TAIL}"
    text = f"\u30ed\u30b0 key={key} \u7d42\u308f\u308a"
    rep = tors.scrub_secrets_report(text, salt="")
    assert [s["type"] for s in rep["spans"]] == ["aws_access_key"]
    span = rep["spans"][0]
    assert text[span["start"]:span["end"]] == key
    assert rep["text"] == f"\u30ed\u30b0 key=AKIA~{_digest('', key)} \u7d42\u308f\u308a"


def test_redteam_100kb_token_dense_smoke() -> None:
    unit = f"key=AKIA{_AWS_TAIL} sha1={_LEGACY40} ok\n"
    text = unit * (100_000 // len(unit) + 1)
    out = tors.scrub_secrets(text)
    assert tors.scrub_secrets(out) is out
    assert f"AKIA{_AWS_TAIL}" not in out
    assert _LEGACY40 not in out
