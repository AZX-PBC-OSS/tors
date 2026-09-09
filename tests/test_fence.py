"""Contract gate for the code-fence surface: ``tors.extract_code_blocks``,
``tors.strip_code_fences``, and ``tors.dedent``.

``dedent`` has a stdlib twin (``textwrap.dedent``) and must be byte-exact with it:
the same parity discipline as ``decode_utf8``/``b64_decode``/``html_unescape``,
gated by a hypothesis differential against the RUNNING interpreter. The fence
functions have no stdlib equivalent (the gap is the point); their contract is
CommonMark §4.5's fenced-code-block grammar, hand-pinned by a rule-cited battery
in the same spirit as ``tests/test_sentence_bounds.py``'s UAX #29 table.
"""

from __future__ import annotations

import textwrap

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tors import dedent, extract_code_blocks, strip_code_fences


class TestExtractCodeBlocks:
    def test_basic_single_block_with_language(self) -> None:
        text = "before\n```python\nprint(1)\n```\nafter"
        got = extract_code_blocks(text)
        assert got == [("python", "print(1)\n", 7, 30)]
        start, end = got[0][2], got[0][3]
        # end runs through the closing fence line's OWN terminator.
        assert text[start:end] == "```python\nprint(1)\n```\n"

    def test_no_info_string_is_none_language(self) -> None:
        assert extract_code_blocks("```\nplain\n```") == [(None, "plain\n", 0, 13)]

    def test_tilde_fence_allows_backtick_in_info_string(self) -> None:
        got = extract_code_blocks("~~~text with ` backtick\ncontent\n~~~")
        assert got[0][0] == "text"
        assert got[0][1] == "content\n"

    def test_backtick_fence_with_backtick_in_info_is_not_a_fence(self) -> None:
        # The info string rule: a backtick fence's info string may not itself
        # contain a backtick. The line is read as ordinary text instead.
        text = "```has ` backtick\nbody\nmore body"
        assert extract_code_blocks(text) == []

    def test_unterminated_fence_runs_to_end_of_input(self) -> None:
        text = "```rust\nfn main() {}\n"
        got = extract_code_blocks(text)
        assert got == [("rust", "fn main() {}\n", 0, len(text))]

    def test_closing_fence_must_match_char_and_at_least_the_opening_length(self) -> None:
        # A 2-backtick line never closes a 3-backtick opener; it's content.
        text = "```\ncode\n``\nmore\n```"
        got = extract_code_blocks(text)
        assert len(got) == 1
        assert got[0][1] == "code\n``\nmore\n"
        # A longer closer (4 backticks) legally closes a 3-backtick opener.
        text2 = "```\ncode\n````"
        got2 = extract_code_blocks(text2)
        assert got2[0][1] == "code\n"
        assert got2[0][3] == len(text2)

    def test_content_indentation_is_stripped_up_to_the_fence_indent(self) -> None:
        text = "  ```\n  aligned\n    extra\naligned less\n  ```"
        got = extract_code_blocks(text)
        assert got[0][1] == "aligned\n  extra\naligned less\n"

    def test_multiple_blocks_in_document_order(self) -> None:
        text = "```a\n1\n```\ntext\n```b\n2\n```"
        got = extract_code_blocks(text)
        assert [(lang, code) for lang, code, _, _ in got] == [("a", "1\n"), ("b", "2\n")]

    def test_lang_filter_is_exact_and_case_sensitive(self) -> None:
        text = "```python\na = 1\n```\n```Python\nb = 2\n```\n```rust\nfn f() {}\n```"
        got = extract_code_blocks(text, lang="python")
        assert [code for _, code, _, _ in got] == ["a = 1\n"]
        assert extract_code_blocks(text, lang="go") == []

    def test_no_fences_answers_empty(self) -> None:
        assert extract_code_blocks("just some prose, no fences at all") == []
        assert extract_code_blocks("") == []

    def test_empty_block_between_fences(self) -> None:
        assert extract_code_blocks("```\n```") == [(None, "", 0, 7)]

    # Fence-forming characters plus a real slice of Unicode outside ASCII:
    # combining marks, astral-plane codepoints (emoji), RTL script, and
    # zero-width/BOM characters, not just the ASCII skeleton needed to
    # form fences at all.
    _WIDE_ALPHABET = st.one_of(
        st.sampled_from("`~abc \n\t123"),
        st.characters(min_codepoint=0x300, max_codepoint=0x36F),  # combining marks
        st.characters(min_codepoint=0x1F300, max_codepoint=0x1FAFF),  # astral emoji/symbols
        st.characters(min_codepoint=0x0600, max_codepoint=0x06FF),  # Arabic (RTL)
        st.sampled_from(["​", "‌", "‍", "﻿"]),  # zero-width + BOM
    )

    @given(st.text(alphabet=_WIDE_ALPHABET, max_size=60))
    @settings(max_examples=500)
    def test_never_panics_and_spans_are_always_in_bounds(self, text: str) -> None:
        for _lang, code, start, end in extract_code_blocks(text):
            assert 0 <= start <= end <= len(text)
            # The reported span must actually be sliceable and its content
            # must roundtrip through the same dedent the block itself
            # already applied, i.e. code is always a substring of the
            # raw span, not a garbled offset.
            assert code in text[start:end] or text[start:end] == ""

    def test_closing_fence_indented_four_or_more_spaces_does_not_close(self) -> None:
        text = "```\ncode\n    ```\nstill code\n```"
        got = extract_code_blocks(text)
        assert len(got) == 1
        assert got[0][1] == "code\n    ```\nstill code\n"

    def test_closing_fence_rejects_non_space_tab_trailing_whitespace(self) -> None:
        # Form feed / vertical tab / NBSP are Unicode whitespace but are NOT
        # "spaces or tabs" per CommonMark §4.5 -- must not close.
        for trailing in ("", "", " "):
            text = f"```\ncode\n```{trailing}\nmore\n```"
            got = extract_code_blocks(text)
            assert len(got) == 1
            assert "more" in got[0][1]

    def test_crlf_line_endings_are_preserved_as_content(self) -> None:
        text = "```python\r\nprint(1)\r\n```\r\nafter"
        got = extract_code_blocks(text)
        assert got == [("python", "print(1)\r\n", 0, len(text) - len("after"))]

    def test_eof_right_after_unterminated_opening_fence_no_newline(self) -> None:
        assert extract_code_blocks("```python") == [("python", "", 0, 9)]

    def test_extremely_long_fence_run_does_not_hang(self) -> None:
        fence = "`" * 10_000
        text = f"{fence}rs\ncode\n{fence}"
        got = extract_code_blocks(text)
        assert len(got) == 1
        assert got[0][0] == "rs"
        assert got[0][1] == "code\n"

    def test_control_characters_in_info_string_do_not_raise(self) -> None:
        got = extract_code_blocks("```py\x00\x07weird\ncode\n```")
        assert len(got) == 1
        assert got[0][1] == "code\n"


class TestStripCodeFences:
    def test_unwraps_a_whole_response_wrapped_in_one_fence(self) -> None:
        assert strip_code_fences("```python\nprint(1)\n```") == "print(1)\n"

    def test_unwraps_with_surrounding_whitespace(self) -> None:
        assert strip_code_fences("  \n```python\nprint(1)\n```\n  ") == "print(1)\n"

    def test_unwraps_an_unterminated_whole_response_fence(self) -> None:
        assert strip_code_fences("```\nprint(1)") == "print(1)"

    def test_is_a_no_op_when_not_exactly_one_wrapping_block(self) -> None:
        for text in [
            "no fences here",
            "prose\n```py\ncode\n```\nmore prose",
            "```a\n```\n```b\n```",
            "",
            "leading text\n```py\ncode\n```",
        ]:
            assert strip_code_fences(text) == text

    def test_no_op_does_not_even_trim_whitespace(self) -> None:
        text = "  no fences here  "
        assert strip_code_fences(text) == text

    def test_trailing_bom_prevents_the_single_block_unwrap(self) -> None:
        # A cleanly closed fence (its OWN line is bare "```", nothing
        # trailing) followed by a BOM on the line after. U+FEFF is NOT
        # Unicode whitespace (matches Python's own str.strip() semantics),
        # so text.trim() does not strip it -- the block no longer spans the
        # ENTIRE trimmed input, and the no-op path correctly takes over.
        text = "```py\nprint(1)\n```\n﻿"
        assert strip_code_fences(text) == text

    @given(st.text(alphabet=TestExtractCodeBlocks._WIDE_ALPHABET, max_size=60))
    @settings(max_examples=500)
    def test_never_panics_and_the_identity_contract_holds(self, text: str) -> None:
        """``strip_code_fences`` never panics over the same wide alphabet as
        its sibling ``extract_code_blocks`` (backticks/tildes, combining
        marks, astral emoji, RTL script, zero-width/BOM), under the
        documented no-op contract (src/py/fence.rs, python/tors/__init__.pyi)
        documented contract the result is the input object itself (``is``, not just ``==``)
        whenever it isn't the single-wrapping-block unwrap case."""
        result = strip_code_fences(text)
        assert isinstance(result, str)
        if result == text:
            # The identity idiom this crate uses throughout: an unchanged
            # result must be the SAME object, not a coincidentally-equal copy.
            assert result is text

    def test_identity_is_returned_object_for_a_concrete_no_op(self) -> None:
        text = "prose\n```py\ncode\n```\nmore prose"
        assert strip_code_fences(text) is text


class TestFenceArgumentContract:
    """The str-in argument contract (non-str -> TypeError, lone surrogates ->
    UnicodeEncodeError) is pinned for every other str-in function in this
    crate (e.g. tests/test_html_unescape.py, tests/test_grounded.py) but was
    never exercised for the fence surface, a real, precisely-named gap."""

    @pytest.mark.parametrize(
        "not_str", [b"abc", bytearray(b"abc"), 123, None], ids=["bytes", "bytearray", "int", "none"]
    )
    def test_extract_code_blocks_non_str_text_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            extract_code_blocks(not_str)  # type: ignore[arg-type]

    def test_extract_code_blocks_non_str_lang_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            extract_code_blocks("```py\ncode\n```", lang=123)  # type: ignore[arg-type]

    def test_extract_code_blocks_lone_surrogate_raises_unicode_encode_error(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            extract_code_blocks("```py\ncode\ud800\n```")

    @pytest.mark.parametrize(
        "not_str", [b"abc", bytearray(b"abc"), 123, None], ids=["bytes", "bytearray", "int", "none"]
    )
    def test_strip_code_fences_non_str_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            strip_code_fences(not_str)  # type: ignore[arg-type]

    def test_strip_code_fences_lone_surrogate_raises_unicode_encode_error(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            strip_code_fences("```py\ncode\ud800\n```")

    @pytest.mark.parametrize(
        "not_str", [b"abc", bytearray(b"abc"), 123, None], ids=["bytes", "bytearray", "int", "none"]
    )
    def test_dedent_non_str_raises_type_error(self, not_str: object) -> None:
        with pytest.raises(TypeError):
            dedent(not_str)  # type: ignore[arg-type]

    def test_dedent_lone_surrogate_raises_unicode_encode_error(self) -> None:
        with pytest.raises(UnicodeEncodeError):
            dedent("abc\ud800\n  def\n")


class TestDedentParityWithStdlib:
    """``tors.dedent`` must be byte-exact with ``textwrap.dedent``, the
    same closed-set differential discipline as the other stdlib-twin
    functions, over a whitespace/text-heavy alphabet biased toward the
    margin edge cases (mixed tabs/spaces, blank lines, no common margin).

    One cross-version divergence, probed and scoped (the b64 gate's
    discipline): CPython 3.14 changed ``textwrap.dedent`` to normalize
    whitespace-only lines and compute margins with ``str.strip``'s
    whitespace set (which includes the four control separators
    ``\\x1c``-``\\x1f`` that Unicode's White_Space property excludes);
    every earlier release used ``[ \\t]``-only margins and left separator
    characters ordinary. tors ships ONE machine: the 3.14 rule, on every
    interpreter (the same one-behavior discipline as the b64 core). On
    interpreters with the OLD stdlib the differential below therefore
    excludes text containing any whitespace character of the changed
    class, and tors's own behavior is pinned on both stdlib generations
    by the rows in ``TestDedentPins``."""

    # The whitespace class whose handling differs between stdlib
    # generations: Python's isspace set beyond ASCII space/tab/newline
    # (the Unicode White_Space characters plus the four controls). The
    # probe char is a VT, in both sets' changed region.
    _CHANGED_WHITESPACE = (
        "\x0b\x0c\r\x1c\x1d\x1e\x1f\x85\u00a0\u1680"
        "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
        "\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
    )

    @staticmethod
    def _stdlib_is_old() -> bool:
        # Behavioral probe, not a version tuple: patch-level behavior is
        # exactly where version gates lie. The 3.14 stdlib strips a
        # whitespace-only single-line text of its margin; the old one
        # leaves it verbatim.
        return textwrap.dedent("\x0b") == "\x0b"

    @given(
        st.text(
            alphabet=st.sampled_from(" \tabc\n"),
            max_size=200,
        )
    )
    @settings(max_examples=1000)
    def test_matches_stdlib_dedent(self, text: str) -> None:
        assert dedent(text) == textwrap.dedent(text)

    @given(st.text(max_size=100))
    @settings(max_examples=300)
    def test_matches_stdlib_dedent_over_arbitrary_text(self, text: str) -> None:
        if self._stdlib_is_old() and any(ws in text for ws in self._CHANGED_WHITESPACE):
            assume(False)
        assert dedent(text) == textwrap.dedent(text)

    def test_basic_common_margin_is_stripped(self) -> None:
        assert dedent("  a\n  b\n") == "a\nb\n"
        assert dedent("    a\n      b\n") == "a\n  b\n"

    def test_mixed_tabs_and_spaces_share_no_margin(self) -> None:
        text = "  a\n\tb\n"
        assert dedent(text) == textwrap.dedent(text) == text

    def test_whitespace_only_lines_normalize_but_dont_gate_the_margin(self) -> None:
        text = "  a\n   \n  b\n"
        assert dedent(text) == textwrap.dedent(text) == "a\n\nb\n"

    def test_no_common_margin_is_unchanged(self) -> None:
        text = "a\n  b\n"
        assert dedent(text) == textwrap.dedent(text) == text


class TestDedentPins:
    """tors's dedent machine pinned directly, on every interpreter
    regardless of the running stdlib's generation: the CPython 3.14
    ``textwrap.dedent`` rule (whitespace-only lines normalize, margins
    use ``str.strip``'s whitespace set including the ``\\x1c``-``\\x1f``
    controls). See TestDedentParityWithStdlib's docstring for the stdlib
    generations and the scoping decision."""

    def test_whitespace_only_single_line_normalizes_to_empty(self) -> None:
        # Every Python-isspace character alone: the 3.14 stdlib's own
        # answers, pinned on both generations.
        for ws in TestDedentParityWithStdlib._CHANGED_WHITESPACE:
            assert dedent(ws) == "", f"dedent({ws!r})"

    def test_controls_join_the_margin_per_pythons_strip_set(self) -> None:
        # The line-normalization set is Python's isspace (includes the
        # \x1c-\x1f controls); the MARGIN set is exactly [ \t] (measured
        # on the running 3.14 stdlib: every other whitespace char leaves a
        # leading run untouched). Pin both sides of that line.
        assert dedent("\x1f") == ""
        assert dedent("\x1f0") == "\x1f0"
        assert dedent("\x1f  a") == "\x1f  a"
        assert dedent("\x1f  a\n\x1f  b") == "\x1f  a\n\x1f  b"
        assert dedent("\x0b  a") == "\x0b  a"

    def test_separator_lines_normalize_and_the_margin_strips(self) -> None:
        # The \x0b line is whitespace-only: normalizes to empty; the
        # remaining lines share the "  " margin.
        assert dedent("  a\n\x0b\n  b") == "a\n\nb"

    def test_mixed_zero_indent_and_pure_whitespace_lines_of_varying_composition(
        self,
    ) -> None:
        # A zero-indent line forces an empty margin; whitespace-only lines
        # (regardless of their own tab/space mix) must neither gate that
        # margin nor survive as anything but "".
        text = "a\n  \t\nb\n\t \n"
        assert dedent(text) == textwrap.dedent(text) == "a\n\nb\n\n"

    def test_empty_input(self) -> None:
        assert dedent("") == textwrap.dedent("") == ""

    def test_docstring_example_from_textwrap(self) -> None:
        text = "    hello\n      world\n    "
        assert dedent(text) == textwrap.dedent(text)

    def test_no_common_margin_returns_the_identical_object(self) -> None:
        """The ``Cow`` identity idiom (pinned at the Rust level,
        src/fence_impl.rs `dedent_no_common_margin_is_identity_shaped`, via
        ``Cow::Borrowed``) is never asserted at the Python boundary that
        actually matters to callers: ``tors.dedent(s) is s`` exactly when
        it's a no-op, the same contract already tested for
        ``truncate_to_bounds`` (tests/test_truncate.py)."""
        text = "a\n  b\n"
        assert dedent(text) is text
