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
consumer runs can catch it, so that equality is asserted over per-PR lanes
(hand corpus, generated grids, needle-chain + userinfo-fail-chain parity
shapes, the suite's scrub corpus at three sizes, seed-free hypothesis lanes
over four alphabets) and nightly lanes (the exhaustive short-string sweep
over the param-rule alphabet — 67,200 cells, the ``sweep`` lane,
json_repair's precedent — and the ``timing`` wall/scaling cells). Lane
counts are reported per lane, never as one combined headline presented as
the per-PR gate: per-PR is the default selection (``-m "not timing and not
sweep"``); sweep + timing run once on the 3.12 leg.

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
import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import tors
from reference import (
    _PG_DETAIL_ESCAPED_RE,
    _PG_DETAIL_RE,
    _URI_CRED_RE,
    _URI_PARAM_CRED_RE,
    SCRUB_RULES,
    reference_scrub_log_text,
    scrub_corpus,
)

# The quoted pin, mechanically enforced against reference.py's compiled
# patterns (TestQuotedPin): these are the TaskQ source's exact pattern
# strings, byte for byte, and both copies must stay that way.
QUOTED_PATTERNS: dict[str, str] = {
    "_PG_DETAIL_RE": r"^[ \t]*DETAIL:.*$",
    "_PG_DETAIL_ESCAPED_RE": (r"(?:\\r)?\\n[ \t]*DETAIL:.*?(?=(?:\\r)?\\n|['\"]\)?\s*$)"),
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


# --- classification pins: the two CPython-re-vs-Rust-std seams ----------------------
#
# Lane split (M2): the per-PR differential corpus above runs by default
# (`-m "not timing and not sweep"`); the 67,200-cell exhaustive sweep lane
# (`-m sweep`, 3.12 leg only) and the timing lane (`-m timing`, 3.12 leg
# only) are nightly/once-per-push, not per-PR. Headline cell counts must
# name their lane (e.g. "67k sweep (nightly)" vs "per-PR corpus"), never a
# combined total presented as the per-PR gate.
PINNED_RUSTC = "1.98.1"
PINNED_UNIDATA = "16.0.0"


class TestClassificationPins:
    def test_word_demote_toolchain_pins(self) -> None:
        """H1 tripwire: WORD_DEMOTE_RANGES was generated against rustc
        1.98.1 + UCD 16.0.0 (CPython 3.14). A toolchain/UCD jump that moves
        the recomputed ranges is a deliberate re-sync — rerun
        tools/enum.rs + tools/gen_word_demote_table.py and update the pins
        here, the script, and the table together — never a silent edit."""
        import unicodedata

        running = unicodedata.unidata_version

        def _version_tuple(v: str) -> tuple[int, ...]:
            return tuple(int(p) for p in v.split("."))

        if _version_tuple(running) < _version_tuple(PINNED_UNIDATA):
            pytest.skip(f"toolchain pin leg requires UCD {PINNED_UNIDATA}+; running {running}")

        assert unicodedata.unidata_version == PINNED_UNIDATA, (
            f"UCD drift: running {unicodedata.unidata_version} vs pinned "
            f"{PINNED_UNIDATA} — rerun tools/gen_word_demote_table.py; if the "
            "ranges moved, re-sync the table and the pins together"
        )
        try:
            out = subprocess.run(
                ["rustc", "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            # The pin leg guarantees rustc (the regen recipe needs it), so a
            # missing toolchain is a failure, not a skip: skipping would
            # silently waive the tripwire this test exists to enforce.
            pytest.fail(f"rustc not available on the pin leg: {exc}")
            return
        assert PINNED_RUSTC in out, (
            f"rustc drift: {out!r} vs pinned {PINNED_RUSTC} — rerun "
            "tools/enum.rs + tools/gen_word_demote_table.py; if the ranges "
            "moved, re-sync the table and the pins together"
        )

    def test_space_table_exhaustive(self) -> None:
        """H2: pin the whole \\s seam, not just the spot checks — over every
        codepoint, CPython ``re`` ``\\s`` agrees with ``str.isspace`` (which
        is what the hypothesis ``_legal_credential_payload`` helper and the
        Rust ``is_python_space`` close over: White_Space + U+001C..U+001F).
        Any UCD change that moves this equivalence is a re-sync, not a
        silent pass."""
        pat = re.compile(r"\s")
        bad: list[int] = []
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            ch = chr(cp)
            if (pat.match(ch) is not None) != ch.isspace():
                bad.append(cp)
                if len(bad) >= 8:
                    break
        assert not bad, (
            f"\\s vs isspace diverged at {[hex(c) for c in bad]} "
            f"(UCD {__import__('unicodedata').unidata_version}): the "
            "is_python_space seam moved; re-sync src/scrub_impl.rs"
        )

    def test_file_separators_are_python_space(self) -> None:
        """The documented seam inside the exhaustive pin: U+001C..U+001F are
        ``\\s`` for the chain (and ``str.isspace``) — the chars Rust's
        ``is_whitespace`` misses and ``is_python_space`` adds back."""
        pat = re.compile(r"\s")
        for cp in range(0x1C, 0x20):
            ch = chr(cp)
            assert pat.match(ch) is not None and ch.isspace(), hex(cp)


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
    "http://u:p@192.168.1.1:8080/x",
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


# --- the needle-chain lane: K escaped DETAIL needles sharing one line ------------------
#
# The escaped pass's line state (line end, line start, trailing-ws start) is
# per-line, but its needles are not: a repr-flattened line can carry thousands
# of ``\\nDETAIL:`` needles, every one of which used to recompute that state
# from scratch — the superlinear scan the pass's line-state memo exists to
# prevent (the timing cell below pins the time side; these cells pin that the
# memo never moves a byte). The shapes, each hunting its own seam:
#
# * the K-chain: K needles on ONE line at growing K (every needle after the
#   first is a memo REUSE; the last needle is left alone as the pinned
#   no-terminator non-match, so the oracle's own lazy scan and the memoized
#   scan must agree on exactly which K-1 runs die);
# * the ws-run shape: the K-chain plus a 200k trailing-whitespace run, the
#   quote-candidate/trailing-ws interplay at scale (the run's start is the
#   one value the ws memo exists for — recomputed per needle before the fix,
#   carried once per line after);
# * the mixed chain: real-newline lines each carrying 16 escaped needles and
#   a ``')`` quote terminator, so the memo is reused within a line and
#   rebuilt across lines in the same scan;
# * the failed-needle interleave: needles that fail the DETAIL check (their
#   space run ends at the next needle's backslash) between successful ones,
#   all on one line — a failed needle must not touch the memo it sits inside.
_NEEDLE_CHAIN_UNIT = "\\nDETAIL:'x"


def _needle_chain_cases() -> list[str]:
    out = [_NEEDLE_CHAIN_UNIT * k for k in (1, 2, 17, 257, 4096)]
    out.append(_NEEDLE_CHAIN_UNIT * 3_200 + " " * 200_000)
    out.append(("E('a" + _NEEDLE_CHAIN_UNIT * 16 + "')\n") * 64)
    out.append("\\n  \\nDETAIL:x" * 512)
    return out


# --- the userinfo fail-chain lane: K anchors sharing one tail -----------------------
#
# C1: mask_uri_userinfo's cursor advances only on match, so K failed anchors
# each re-walk the same tail (the password class [^\s@]+ permits :/?=, so
# "a://u:"*K + "p"*M has K anchors whose password scans all run to the end:
# O(K*M), ~486ms pre-fix at K=2000/M=200k where a linear scan is <1ms).
# The chain's own re is quadratic here too (~4s); availability wins over
# matching its complexity class. Small-K shapes below join the per-PR
# differential corpus (parity); the full-K wall + scaling cells are
# timing-lane (nightly/3.12 leg), not per-PR.
_USERINFO_FAIL_UNIT = "a://u:"


def _userinfo_fail_chain(k: int, m: int) -> str:
    return _USERINFO_FAIL_UNIT * k + "p" * m


def _userinfo_fail_cases() -> list[str]:
    return [
        _userinfo_fail_chain(8, 64),
        _userinfo_fail_chain(32, 256),
        # No-@ with whitespace terminator (fail at ws, not at end).
        _userinfo_fail_chain(16, 128) + " ",
        # Fail with a trailing @ that still cannot match (empty password
        # after the last colon is not a mask; earlier anchors see ws/end).
        _userinfo_fail_chain(16, 128) + "@",
    ]


_CORPUS_CASES: list[str] = (
    _CORPUS
    + _userinfo_grid()
    + _escaped_grid()
    + _param_grid()
    + _needle_chain_cases()
    + _userinfo_fail_cases()
    + [scrub_corpus(1024), scrub_corpus(100 * 1024), scrub_corpus(1024 * 1024)]
)


class TestCorpusDifferential:
    @pytest.mark.parametrize(("lane", "rules"), RULE_LANES, ids=[lane for lane, _ in RULE_LANES])
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
    alphabet=("a:?&@/=up5_-.\\n'\u093e\u00e9\u4e94\u2167\u24b6\u00a0\x1c\x1d\u2028"),
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
# a `p` no name can match): 67,200 cells, the json_repair sweep's scale. The
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


# --- the timing-lane wall cell: the needle-chain linearity contract ---------------------


def _min_wall_ms(op, samples: int = 7, warmup: int = 1) -> float:
    # Min-of-7 like _FAST_CELL_SAMPLES in tests/test_performance.py: the
    # 60ms ceilings below sit ~60-140x above the measured linear band, but a
    # min-of-3 can still flake under scheduler load on short samples, and a
    # debug (maturin-develop) build runs an order of magnitude slower than
    # release — run timing cells in release, compare min-of-7 to the ceiling.
    for _ in range(warmup):
        op()
    best = float("inf")
    for _ in range(samples):
        started = time.perf_counter()
        op()
        best = min(best, time.perf_counter() - started)
    return best * 1000.0


@pytest.mark.timing
def test_scrub_escaped_needle_chain_wall_stays_linear_on_the_ws_run_shape() -> None:
    """The escaped pass's linearity contract as a wall band: the 235KB
    worst shape of the needle-chain lane above (3,200 ``\\nDETAIL:`` needles
    sharing one line, then a 200k trailing-whitespace run) scrubs inside
    60ms. Derivation: the memoized pass is linear and measures ~0.4ms on
    this box (min-of-7, release), so 60ms is a ~140x sanity margin that only a
    complexity regression can reach — the shape is the red-team P1 repro
    whose PRE-fix cost was ~350ms here (per-needle line-state
    recomputation, O(K*M) on exactly this input), so a return to the
    superlinear scan overshoots the ceiling ~6x while any linear
    implementation — even one two orders of magnitude slower per byte —
    stays under. A timing-lane cell: the load-sensitive wall measurement
    CI's matrix legs deselect (``-m "not timing and not sweep"``), one 3.12
    leg running it: the marker-split contract every lane cell carries."""
    text = _NEEDLE_CHAIN_UNIT * 3_200 + " " * 200_000
    assert _min_wall_ms(lambda: tors.scrub_log_text(text)) < 60.0


@pytest.mark.timing
def test_scrub_userinfo_fail_chain_wall_stays_linear() -> None:
    """C1 linearity contract as a wall band: K=2000 failed userinfo anchors
    sharing one 200k tail (``"a://u:"*K + "p"*M``, no ``@`` anywhere) scrubs
    inside 60ms. PRE-fix this shape cost ~486ms here (each of the K anchors
    re-walks the tail: O(K*M)); the memoized/fail-skip pass is linear and
    measures <1ms, so 60ms is a ~60x+ margin only a complexity regression
    can reach. Timing-lane (``-m timing``): the 3.12 leg runs it, per-PR
    legs deselect it."""
    text = _userinfo_fail_chain(2000, 200_000)
    # Parity first: the shape is a no-match (no @), so tors must agree with
    # the chain exactly (identity), even though the chain itself is
    # quadratic here (~4s) — availability wins, output stays identical.
    assert tors.scrub_log_text(text, ["uri_userinfo"]) == reference_scrub_log_text(
        text, ["uri_userinfo"]
    )
    assert _min_wall_ms(lambda: tors.scrub_log_text(text, ["uri_userinfo"])) < 60.0


@pytest.mark.timing
def test_scrub_userinfo_fail_chain_scales_linearly() -> None:
    """C1 scaling contract: doubling the fail-chain input less than triples
    the wall (linear scaling; a quadratic regression ~4x). Ratio-of-mins,
    same process, back-to-back, so load noise divides out. Timing-lane."""
    small = _userinfo_fail_chain(1000, 100_000)
    large = _userinfo_fail_chain(2000, 200_000)
    t_small = _min_wall_ms(lambda: tors.scrub_log_text(small, ["uri_userinfo"]))
    t_large = _min_wall_ms(lambda: tors.scrub_log_text(large, ["uri_userinfo"]))
    assert t_large < 3.0 * max(t_small, 0.05), (
        f"userinfo fail-chain scaling regressed: 106KB {t_small:.2f}ms vs "
        f"212KB {t_large:.2f}ms (ratio {t_large / max(t_small, 1e-9):.2f}x, "
        "linear must stay <3x)"
    )


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
        try:
            spec.loader.exec_module(module)
        except ImportError:
            # The checkout exists but cannot import HERE: the live module's
            # own dependencies (opentelemetry, since TaskQ's 2026-09-18
            # teardown/trace split) are not this venv's, and tors'
            # deliberately does not carry them (taskq is an optional
            # sibling, not a locked dependency). That is the same
            # "no usable TaskQ" case the docstring's None means: the live
            # lane skips, the quoted-pattern differential above still
            # runs. An ImportError escaping this loader was a COLLECTION
            # error that took the whole file (both halves) down with it —
            # observed 2026-09-18 when the sibling tree grew the import.
            return None
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
            assert isinstance(live, re.Pattern)
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
            for text in (
                _CORPUS
                + _needle_chain_cases()
                + [
                    scrub_corpus(1024),
                    scrub_corpus(100 * 1024),
                ]
            ):
                assert module._scrub_text(text) == reference_scrub_log_text(text)  # noqa: SLF001
        finally:
            module._redaction_enabled = flag_before  # noqa: SLF001

    def test_tors_matches_the_live_chain_over_the_corpus(self) -> None:
        module = _TASKQ
        assert module is not None
        flag_before = module._redaction_enabled  # noqa: SLF001
        try:
            module._redaction_enabled = True  # noqa: SLF001
            for text in (
                _CORPUS
                + _needle_chain_cases()
                + [
                    scrub_corpus(1024),
                    scrub_corpus(100 * 1024),
                ]
            ):
                assert tors.scrub_log_text(text) == module._scrub_text(text)  # noqa: SLF001
        finally:
            module._redaction_enabled = flag_before  # noqa: SLF001
