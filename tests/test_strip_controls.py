"""Contract gate for ``tors.strip_controls``: every maximal run of C0
controls (U+0000-U+001F, tabs and newlines included) and DEL (U+007F)
becomes a single ASCII space — the scrub model-authored display text
needs before it is stored or served.

What this gate pins:

- REGEX PARITY: byte-identical to
  ``re.compile(r"[\\x00-\\x1f\\x7f]+").sub(" ", text)`` over a hypothesis
  corpus spanning C0, C1, DEL, multibyte text, and emoji — the three
  ta_worker call sites (fit/enrich/derive prompts) this replaces, so
  adoption changes no stored value.
- RUN COLLAPSE: one control or a hundred adjacent ones yield exactly one
  space, including the full C0 range plus DEL in a single run.
- SCOPE CUTS: C1 controls (U+0080-U+009F) pass through untouched
  (deliberate: the adopted regexes do not cover them either); tab,
  newline, and CR ARE scrubbed (they are C0) — pinned literally, since
  that surprises callers reaching for this on multi-line prose.
- NO EDGE STRIP: edge runs become edge spaces; stripping is the caller's.
- IDENTITY: no C0/DEL anywhere returns the original object (``is``).
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tors import strip_controls

_ORACLE = re.compile(r"[\x00-\x1f\x7f]+")

_TEXT = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs", "P", "Mn", "Mc", "Cf", "Cc"),
        max_codepoint=0x2FFFF,
    ),
    max_size=300,
)


class TestRegexParity:
    @given(text=_TEXT)
    @settings(max_examples=500)
    def test_matches_the_adopted_regex_exactly(self, text: str) -> None:
        assert strip_controls(text) == _ORACLE.sub(" ", text)

    @given(text=_TEXT)
    @settings(max_examples=200)
    def test_identity_when_nothing_to_scrub(self, text: str) -> None:
        if _ORACLE.search(text) is None:
            assert strip_controls(text) is text


class TestRunCollapse:
    def test_single_controls_become_single_spaces(self) -> None:
        assert strip_controls("a\x00b") == "a b"
        assert strip_controls("a\x1fb") == "a b"
        assert strip_controls("a\x7fb") == "a b"

    def test_whole_c0_range_plus_del_collapses_to_one_space(self) -> None:
        run = "a" + "".join(chr(cp) for cp in range(0x00, 0x20)) + "\x7f" + "b"
        assert strip_controls(run) == "a b"

    def test_adjacent_runs_stay_separate(self) -> None:
        assert strip_controls("a\x00b\x00c") == "a b c"

    def test_empty_text_is_identity(self) -> None:
        assert strip_controls("") == ""


class TestScopeCuts:
    def test_c1_controls_pass_through(self) -> None:
        assert strip_controls("\u0080") == "\u0080"
        assert strip_controls("a\u0085b") == "a\u0085b"

    def test_tab_newline_cr_are_scrubbed(self) -> None:
        assert strip_controls("a\tb") == "a b"
        assert strip_controls("a\nb") == "a b"
        assert strip_controls("a\rb") == "a b"
        assert strip_controls("a\r\nb") == "a b"

    def test_multibyte_text_around_controls_survives(self) -> None:
        assert strip_controls("caf\u00e9\x00\U0001f600") == "caf\u00e9 \U0001f600"


class TestEdges:
    def test_leading_run_becomes_leading_space(self) -> None:
        assert strip_controls("\x00abc") == " abc"

    def test_trailing_run_becomes_trailing_space(self) -> None:
        assert strip_controls("abc\x00") == "abc "

    def test_only_controls_becomes_one_space(self) -> None:
        assert strip_controls("\x00") == " "


class TestArgumentBoundary:
    def test_lone_surrogate_is_refused(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            strip_controls("\ud800abc")
