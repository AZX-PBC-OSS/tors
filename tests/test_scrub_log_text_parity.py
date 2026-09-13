"""Differential parity for the ``tors.scrub_log_text`` port of TaskQ's
exception-text scrub chain.

Provenance, the pin this file exists to enforce: the behavior oracle is the
consumer's own chain, ``src/taskq/obs/_redact_exc.py`` in the TaskQ repo
(pinned to the sibling checkout this suite runs against on the dev box; the
locator below also honors ``TORS_TASKQ_REPO``). The four compiled regexes,
quoted verbatim from that source, are:

.. code-block:: python

    _PG_DETAIL_RE = re.compile(r"^[ \\t]*DETAIL:.*$", re.MULTILINE)
    _PG_DETAIL_ESCAPED_RE = re.compile(
        r"(?:\\\\r)?\\\\n[ \\t]*DETAIL:.*?(?=(?:\\\\r)?\\\\n|['\\"]\\)?\\s*$)",
        re.MULTILINE,
    )
    _URI_CRED_RE = re.compile(r"(\\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\\s:/@]*):([^\\s@]+)@")
    _URI_PARAM_CRED_RE = re.compile(r"([?&](?:password|passphrase|passwd|pwd)=)([^\\s&@]+)")

(the ``\\``-doubling is this docstring's; ``tests/reference.py`` carries the
machine-checked single-escaped spellings, and ``TestQuotedPin`` below fails
if they ever differ from what this header means). The chain order is
``_scrub_text``'s own with the redaction flag on: DETAIL lines (real
newlines), then DETAIL runs (repr-flattened ``\\n`` separators), then the
userinfo mask, then the password-family query-param mask — each a whole pass
over the current text, which is exactly the canonical order
``tors.scrub_log_text`` applies per rule selection. A TaskQ change to any
pattern or to the order is a visible re-sync request (the live-oracle class
below fails loudly when the checkout is present), never a silent tors
behavior change.

Why the failure mode needs this file: silent under-redaction. A port bug
that leaves a row value or a password in the output crashes nothing and
fails no structural check; only equality against the exact chain the
consumer runs can catch it, so that equality is asserted over a hand corpus
(the rule cruxes and their adversarial combinations), generated grids, the
suite's scrub corpus at three sizes, seed-free hypothesis lanes over four
alphabets, and an exhaustive short-string sweep over the param-rule alphabet
(the ``sweep`` lane, json_repair's precedent).

The live-oracle lane is the ``test_json_repair_parity.py`` importorskip
pattern adapted for a LOCAL repo rather than a pip package: TaskQ is not
installable into this suite's environment (its package ``__init__`` pulls
the worker's dependency tree — opentelemetry, structlog, asyncpg — none of
which tors's dev environment carries), so ``import taskq`` cannot succeed
here and a sys.path route would importorskip-skip even on the box where the
oracle exists. The scrub module itself is stdlib-only (``re``/``sys``/
``traceback``), so the lane file-loads ``_redact_exc.py`` directly via
``importlib`` — the live source, not a copy — and skips (the importorskip
semantics) when no checkout is found.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import re
import subprocess
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from reference import (
    SCRUB_RULES,
    _PG_DETAIL_ESCAPED_RE,
    _PG_DETAIL_RE,
    _URI_CRED_RE,
    _URI_PARAM_CRED_RE,
    reference_scrub_log_text,
    scrub_corpus,
)

# The quoted pin, mechanically enforced against reference.py's compiled
# patterns (TestQuotedPin): these are the TaskQ source's exact pattern
# strings, byte for byte, and both copies must stay that way.
QUOTED_PATTERNS: dict[str, str] = {
    "_PG_DETAIL_RE": r"^[ \t]*DETAIL:.*$",
    "_PG_DETAIL_ESCAPED_RE": (
        r"(?:\\r)?\\n[ \t]*DETAIL:.*?(?=(?:\\r)?\\n|['\"]\)?\s*$)"
    ),
    "_URI_CRED_RE": r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@]*):([^\s@]+)@",
    "_URI_PARAM_CRED_RE": r"([?&](?:password|passphrase|passwd|pwd)=)([^\s&@]+)",
}
_COMPILED_PATTERNS: dict[str, re.Pattern[str]] = {
    "_PG_DETAIL_RE": _PG_DETAIL_RE,
    "_PG_DETAIL_ESCAPED_RE": _PG_DETAIL_ESCAPED_RE,
    "_URI_CRED_RE": _URI_CRED_RE,
    "_URI_PARAM_CRED_RE": _URI_PARAM_CRED_RE,
}

# The rule lanes every corpus case runs under: the full chain, each rule
# alone, and the pg+userinfo composition (the pair whose interaction is the
# canonical-order contract — a DETAIL deletion can eat the `@` the userinfo
# mask anchors on).
RULE_LANES: list[tuple[str, list[str] | None]] = [
    ("full", None),
    ("pg_detail_lines", ["pg_detail_lines"]),
    ("uri_userinfo", ["uri_userinfo"]),
    ("uri_query_creds", ["uri_query_creds"]),
    ("pg+userinfo", ["pg_detail_lines", "uri_userinfo"]),
]


class TestQuotedPin:
    @pytest.mark.parametrize("name", sorted(QUOTED_PATTERNS))
    def test_reference_patterns_are_the_quoted_pin(self, name: str) -> None:
        assert _COMPILED_PATTERNS[name].pattern == QUOTED_PATTERNS[name], (
            f"reference.py's {name} drifted from the TaskQ-quoted pin in this "
            "file's header: a re-sync (deliberate, against the live module) "
            "or a bug; either way the two spellings must not diverge silently"
        )

    def test_the_canonical_rule_order_is_pinned(self) -> None:
        # The chain order _scrub_text applies with the flag on: DETAIL's two
        # segmenters, userinfo, then query params. reference.py must carry
        # the same tuple tors spells.
        assert SCRUB_RULES == ("pg_detail_lines", "uri_userinfo", "uri_query_creds")


# --- the hand corpus: every rule crux plus its adversarial combinations ---------------

_CORPUS: list[str] = [
    # DETAIL, real newlines: blank line left behind, CRLF's \r consumed with
    # the line, leading [ \t] part of the line, HINT/CONTEXT kept, near-miss
    # names not matched, end-of-text line, whole-line deletion.
    "duplicate key\nDETAIL:  Key (id)=(9) exists.",
    "duplicate key\r\nDETAIL:  Key (id)=(9) exists.\r\nHINT: x",
    "\t DETAIL: v\nnext",
    "DETAIL: x\ny",
    "x\nDETAIL: tail",
    "pre\nDETAIL: v and everything to EOL\nHINT: h",
    "boom\nDETAIL: one\nHINT: keep\nCONTEXT: keep2\nDETAIL: two\nend",
    "  DETAILX: v\ndetail: v\nDETAIL v\nnext",
    "a\rDETAIL: cr-only line endings\nb",
    # DETAIL, repr-flattened: closing quote kept, escaped CRLF, consecutive
    # runs, interior quotes, real tab prefix vs literal \t, the three pinned
    # non-matches (no terminator; real-newline termination; quote not at
    # EOL), trailing-ws and real-newline terminations, repr inside a
    # rendered traceback.
    "PostgresError('msg\\nDETAIL:  Key (id)=(9) exists.')",
    'PostgresError("msg\\nDETAIL:  Key (id)=(9) exists.")',
    "PostgresError('msg\\r\\nDETAIL: secret')",
    "E('a\\nDETAIL: one\\nDETAIL: two')",
    "E('a\\nDETAIL: Key (x)=('val') exists.')",
    "E('a\\n\tDETAIL: v')",
    "E('a\\n\\tDETAIL: v')",
    "E('a\\nDETAIL: leaks",
    "E('a\\nDETAIL: leaks\nnext line",
    "E('a\\nDETAIL: v')  tail",
    "E('a\\nDETAIL: v')  \nnext",
    "E('a\\nDETAIL: v')\nnext",
    'E("a\\nDETAIL: v") and "tail"',
    "Traceback (most recent call last):\nJobError('dup\\nDETAIL: K=(v)')\nafter",
    "x\\nDETAIL: y\\nDETAIL: z') tail\\nDETAIL: w')",
    "a\\nDETAIL: b\\r\\nDETAIL: c')",
    # userinfo: the mask itself.
    "postgresql://worker:hunter2@db/prod",
    "postgresql://:SECRET@host/db",
    "postgresql://user:@host/db",
    "a://u:p@ss@h",
    "a://u:p:q@h",
    "DB+sql-x.9://u:pw@h",
    "1st://u:pw@h",
    "-st://u:pw@h",
    ".st://u:pw@h",
    "+st://u:pw@h",
    "0https://u:pw@h",
    "éhttps://u:pw@h",
    "_https://u:pw@h",
    "\u093ehttps://u:pw@h",
    "\u24b6https://u:pw@h",
    "\u4e94a://u:pw@h",
    "\u2167a://u:pw@h",
    "五a://u:pw@h",
    "a://u/v:pw@h",
    "a://u@v:pw@h",
    "a://u\u00a0v:pw@h",
    "a://u\x1cv:pw@h",
    "a://u:p\x1cq@h",
    "a://u:p\u00a0q@h",
    "a://u:p\u2028q@h",
    "a://u:p://q@h",
    "a://u:pw host",
    "a://u:pw@",
    "://u:pw@h",
    "a://:pw@h",
    "HTTP://U:PW@H",
    "a+b-c.d1://u:pw@h",
    "mailto:user:pass@host",
    "scheme://user:pa?password=zz@host",
    # query params: the four names, near misses, value terminators.
    "?password=x&passphrase=y&passwd=z&pwd=w",
    "?Password=x&PASSWORD=y&passwords=z&passw=w&pwdx=1&p=2",
    "?password=a b?pwd=c@d&passwd=e",
    "?password=a=b?c",
    "?password=&x=1",
    "?pwd=",
    "host/db?password=x",
    "?password=a\x1cb",
    "?password=a\u00a0b",
    "a?password=1?pwd=2",
    "&password=1&pwd=2&password=3",
    "x?passphrase=a?b&passwd=",
    # chain shapes: both credential shapes on one DSN, embedded param inside
    # a userinfo password, the DETAIL-eats-the-@ order interaction, an
    # escaped DETAIL inside a would-be password, mixed real text.
    "postgresql://worker:S3cr3t@db/prod?password=fallback",
    "pg://u:p\\nDETAIL:x@h')",
    "a://u:p\\nDETAIL:x@h",
    "a://u:p@h more text?password=zz",
    "postgres://w:pw@a.b/c?passphrase=sec&other=1",
    "JobError('pg://u:pw@h/db')\nDETAIL:  Key (k)=('v') s.",
    "DETAIL: a pg://u:pw@h and ?password=x\nnext",
    "",
    "no rules fire here at all",
    "x",
    "\n",
    "\\n",
    "\\nDETAIL:",
]

# --- generated grids: the combinatorial neighborhoods of each crux -------------------

_USERINFO_GRID_PREFIX = ["", " ", "-", "0", "é", "\u093e", "_", ".", "\u4e94", "(", "'"]
_USERINFO_GRID_SCHEME = ["a", "https", "HTTPS", "a+b", "a-b.c", "a1", "1a", "-a", "a-"]
_USERINFO_GRID_USER = ["u", "", "u.v", "u+v", "ü"]


def _userinfo_grid() -> list[str]:
    out = []
    for prefix, scheme, user in itertools.product(
        _USERINFO_GRID_PREFIX, _USERINFO_GRID_SCHEME, _USERINFO_GRID_USER
    ):
        out.append(f"{prefix}{scheme}://{user}:pw@h")
    return out


_ESCAPED_GRID_OPENER = ["", "E('", 'E("', "x ", "\n", "''"]
_ESCAPED_GRID_ESC_NL = ["\\n", "\\r\\n"]
_ESCAPED_GRID_WS = ["", " ", "\t", " \t "]
_ESCAPED_GRID_PAYLOAD = ["v", "K=('q')", "a\\nb", "v'", "v')", "secret@host", "a b"]
_ESCAPED_GRID_TERMINATOR = [
    "",
    "\\n",
    "\\r\\n",
    "'",
    "')",
    "'  ",
    "')  ",
    "'')",
    "' x",
    "\n",
    "\\n\\n",
    "x'",
]


def _escaped_grid() -> list[str]:
    out = []
    for opener, esc_nl, ws, payload, term in itertools.product(
        _ESCAPED_GRID_OPENER,
        _ESCAPED_GRID_ESC_NL,
        _ESCAPED_GRID_WS,
        _ESCAPED_GRID_PAYLOAD,
        _ESCAPED_GRID_TERMINATOR,
    ):
        out.append(f"{opener}{esc_nl}{ws}DETAIL:{payload}{term}")
    return out


_PARAM_GRID_DELIM = ["?", "&"]
_PARAM_GRID_NAME = [
    "password",
    "passphrase",
    "passwd",
    "pwd",
    "Password",
    "passwords",
    "passwo",
    "pwdx",
    "pass",
    "p",
]
_PARAM_GRID_VALUE = ["x", "", " ", "a b", "a&b", "a@b", "a=b", "***", "a\\nb", "x://y:z"]
_PARAM_GRID_TAIL = ["", "&n=1", " x", "@h", "\n", "?password=2"]


def _param_grid() -> list[str]:
    out = []
    for delim, name, value, tail in itertools.product(
        _PARAM_GRID_DELIM, _PARAM_GRID_NAME, _PARAM_GRID_VALUE, _PARAM_GRID_TAIL
    ):
        out.append(f"pre {delim}{name}={value}{tail}")
    return out


_CORPUS_CASES: list[str] = (
    _CORPUS
    + _userinfo_grid()
    + _escaped_grid()
    + _param_grid()
    + [scrub_corpus(1024), scrub_corpus(100 * 1024), scrub_corpus(1024 * 1024)]
)


class TestCorpusDifferential:
    @pytest.mark.parametrize(
        ("lane", "rules"), RULE_LANES, ids=[lane for lane, _ in RULE_LANES]
    )
    @pytest.mark.parametrize(
        "text", _CORPUS_CASES, ids=[f"case-{i}" for i in range(len(_CORPUS_CASES))]
    )
    def test_tors_matches_the_chain(self, text: str, lane: str, rules: list[str] | None) -> None:
        assert tors.scrub_log_text(text, rules) == reference_scrub_log_text(text, rules)

    def test_empty_rules_lane_is_the_identity(self) -> None:
        for text in _CORPUS:
            assert tors.scrub_log_text(text, []) == text


# --- hypothesis lanes: four alphabets, each hunting its own divergence class ----------

_ASCII_ALPHABET = st.text(
    alphabet="ab:?&@/=.-+'\"x \t\r\n\\",
    max_size=48,
)
_UNICODE_EDGE_ALPHABET = st.text(
    alphabet=(
        "a:?&@/=up5_-.\\n'"
        "\u093e\u00e9\u4e94\u2167\u24b6\u00a0\x1c\x1d\u2028"
    ),
    max_size=48,
)
_DETAIL_PIECES = st.lists(
    st.sampled_from(
        [
            "\\n",
            "\\r\\n",
            "DETAIL:",
            "DETAIL",
            " ",
            "\t",
            "'",
            "')",
            "x",
            "\n",
            "HINT:",
            '"',
            ")",
            "a://u:p@h",
            "?password=v",
            "E('",
        ]
    ),
    min_size=0,
    max_size=24,
)
_FULL_UNICODE = st.text(
    alphabet=st.characters(max_codepoint=0x10FFFF, exclude_categories=("Cs",)),
    max_size=48,
)


class TestHypothesisDifferential:
    @given(_ASCII_ALPHABET)
    @settings(max_examples=250, deadline=None)
    def test_ascii_structural_alphabet_every_lane(self, text: str) -> None:
        assert tors.scrub_log_text(text) == reference_scrub_log_text(text)
        for rules in (["pg_detail_lines"], ["uri_userinfo"], ["uri_query_creds"]):
            assert tors.scrub_log_text(text, rules) == reference_scrub_log_text(text, rules)

    @given(_UNICODE_EDGE_ALPHABET)
    @settings(max_examples=250, deadline=None)
    def test_unicode_edge_alphabet_every_lane(self, text: str) -> None:
        # The alphabet carries the two classification seams where Rust std
        # and CPython re disagree: Other_Alphabetic marks (U+093E, U+24B6)
        # that Python's `\w` rejects (the `\b` before a scheme), and the
        # U+001C..U+001F file separators that Python's `\s` accepts (the
        # username/password/value classes) — plus NBSP, U+2028, CJK
        # numerals (Numeric_Type letters: word chars under both), and Nl.
        assert tors.scrub_log_text(text) == reference_scrub_log_text(text)
        for rules in (["pg_detail_lines"], ["uri_userinfo"], ["uri_query_creds"]):
            assert tors.scrub_log_text(text, rules) == reference_scrub_log_text(text, rules)

    @given(_DETAIL_PIECES)
    @settings(max_examples=250, deadline=None)
    def test_detail_piece_composite_alphabet(self, pieces: list[str]) -> None:
        # Pieces, not chars: random chars almost never spell `DETAIL:`, so
        # the escaped-pass's lazy-scan/lookahead machinery needs its inputs
        # assembled from the real shapes (escaped separators, quote
        # closers, real newlines, HINT lines, DSNs).
        text = "".join(pieces)
        assert tors.scrub_log_text(text) == reference_scrub_log_text(text)
        assert tors.scrub_log_text(text, ["pg_detail_lines"]) == (
            reference_scrub_log_text(text, ["pg_detail_lines"])
        )

    @given(_FULL_UNICODE)
    @settings(max_examples=200, deadline=None)
    def test_full_unicode_alphabet(self, text: str) -> None:
        assert tors.scrub_log_text(text) == reference_scrub_log_text(text)


# --- the exhaustive sweep: the param-rule alphabet ------------------------------------
#
# Every string up to length 6 over [? & p w d = @] that contains a delimiter
# and a `p` (the shortest firing shape, `?pwd=x` aside, is 6 chars; without
# a `p` no name can match): 68,200 cells, the json_repair sweep's scale. The
# alphabet drops the value filler `x` on purpose — `=` and `@` as value
# chars exercise the value-class boundaries (a `?`/`=` inside a value, the
# empty-value non-match) more densely than another letter would.
_SWEEP_ALPHABET = ["?", "&", "p", "w", "d", "=", "@"]
_SWEEP_MAXLEN = 6


def _sweep_texts() -> list[str]:
    texts: list[str] = []
    for length in range(1, _SWEEP_MAXLEN + 1):
        for tup in itertools.product(_SWEEP_ALPHABET, repeat=length):
            s = "".join(tup)
            if ("?" in s or "&" in s) and "p" in s:
                texts.append(s)
    return texts


@pytest.mark.sweep
class TestExhaustiveSweep:
    @pytest.mark.parametrize("text", _sweep_texts())
    def test_param_alphabet_parity(self, text: str) -> None:
        got = tors.scrub_log_text(text)
        want = reference_scrub_log_text(text)
        assert got == want


# --- the live-oracle re-sync lane (TaskQ checkout gated) ------------------------------


def _load_taskq_module() -> object | None:
    """File-load the live ``_redact_exc.py`` if a TaskQ checkout is findable.

    Locator: ``TORS_TASKQ_REPO`` wins if set; otherwise the sibling of this
    repo's MAIN checkout (derived via ``git rev-parse --git-common-dir``, so
    the route works from any worktree). Returns ``None`` (the caller skips,
    importorskip semantics) when neither holds a checkout.
    """
    candidates: list[Path] = []
    env_repo = os.environ.get("TORS_TASKQ_REPO")
    if env_repo:
        candidates.append(Path(env_repo))
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
        main_checkout = (Path(__file__).resolve().parent.parent / common_dir).resolve().parent
        candidates.append(main_checkout.parent / "TaskQ")
    except (OSError, subprocess.SubprocessError):
        pass
    for candidate in candidates:
        module_path = candidate / "src" / "taskq" / "obs" / "_redact_exc.py"
        if not module_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("taskq_obs_redact_exc", module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


_TASKQ = _load_taskq_module()
_requires_taskq = pytest.mark.skipif(
    _TASKQ is None,
    reason=(
        "no TaskQ checkout found (sibling of the main checkout, or "
        "$TORS_TASKQ_REPO): the live-oracle re-sync lane is skipped; the "
        "quoted-pattern differential above still runs"
    ),
)


@_requires_taskq
class TestLiveOracleResync:
    def test_the_live_patterns_are_the_quoted_pin(self) -> None:
        for name, quoted in QUOTED_PATTERNS.items():
            live = getattr(_TASKQ, name)
            assert isinstance(live, re.Pattern[str])
            assert live.pattern == quoted, (
                f"TaskQ's {name} changed (or this pin is stale): {live.pattern!r} "
                f"vs the quoted {quoted!r} — re-sync the pin in tests/reference.py "
                "and this header TOGETHER, as one deliberate change"
            )

    def test_the_live_chain_is_the_local_chain_over_the_corpus(self) -> None:
        module = _TASKQ
        assert module is not None
        flag_before = module._redaction_enabled  # noqa: SLF001
        try:
            module._redaction_enabled = True  # noqa: SLF001
            for text in _CORPUS + [scrub_corpus(1024), scrub_corpus(100 * 1024)]:
                assert module._scrub_text(text) == reference_scrub_log_text(text)  # noqa: SLF001
        finally:
            module._redaction_enabled = flag_before  # noqa: SLF001

    def test_tors_matches_the_live_chain_over_the_corpus(self) -> None:
        module = _TASKQ
        assert module is not None
        flag_before = module._redaction_enabled  # noqa: SLF001
        try:
            module._redaction_enabled = True  # noqa: SLF001
            for text in _CORPUS + [scrub_corpus(1024), scrub_corpus(100 * 1024)]:
                assert tors.scrub_log_text(text) == module._scrub_text(text)  # noqa: SLF001
        finally:
            module._redaction_enabled = flag_before  # noqa: SLF001
