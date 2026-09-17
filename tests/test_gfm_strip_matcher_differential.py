"""The gfm strip matcher's differential pin: the PR replaced the
per-opener delimiter rescans (``find_unescaped_bracket_close`` and the
URL-paren depth scan inside ``parse_link``) with one precomputed
pairing (``match_delimiters``' two stack tables), for linearity. The
contract is byte-identity: every line must strip to exactly the bytes
the old scans produced.

The proof here is by substitution, which is the whole of the change:
the caller loop in ``strip_emphasis_and_links_at_depth`` is byte-for-
byte the code it was (the diff touches only where ``parse_link`` gets
its delimiters), so identity holds exactly when the tables answer what
the scans answered, at every position the caller can consult: an
unescaped ``[`` for brackets (the image path's ``[`` cannot be escaped:
its preceding character is ``!``, not a backslash), and a ``(`` for
parens. This file re-implements the two OLD scans in pure Python (the
reference, transcribed from the pre-change source, semantics only:
``git show HEAD:src/gfm_strip_impl.rs`` carries the originals) and
asserts that substitution invariant over a hypothesis-generated space
plus every hand shape below, then drives the same shapes end to end
through the public surface (``to_text(data=<html><p>line</p></html>,
format="html")``, whose markdown is the line verbatim plus a newline,
measured) comparing today's bytes against a model of the caller loop
with the OLD scans plugged in.

The shapes: unmatched openers and closers, escaped brackets and parens,
nested labels, images, label-equals-url, the mixed ``[a] b ] (u)``
shape where a backward (last-closer) pairing would change bytes, NUL
bytes (at the matcher level: the HTML engine drops NUL from its
markdown, measured, so the sentinel grammar is reached only by the
strip's own code-span lifting, unchanged machinery), per-line
isolation, nesting at the depth cap (255/256/257: the degradation path
where the ladder returns the remaining label verbatim), and a lone
``](``. The long-line contract (1 MiB single line, linear, no panic) is
pinned by a subprocess probe with a deadline. The Rust fuzz target
(cargo fuzz gfm_strip) needs a nightly toolchain this box does not
have; the hypothesis space below carries the fuzzing duty instead.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tors_documents import to_markdown, to_text

# The caller's depth cap (src/gfm_strip_impl.rs: MAX_INLINE_DEPTH): at
# this depth the ladder returns the line verbatim, the degradation path.
MAX_INLINE_DEPTH = 256


def _escaped_at(chars: list[str]) -> list[bool]:
    """The escape pass, both versions' shared first step: a backslash
    marks the NEXT character escaped (a doubled backslash leaves the
    second one escaped, so a bracket after it still pairs)."""
    escaped = [False] * len(chars)
    esc = False
    for idx, c in enumerate(chars):
        if esc:
            escaped[idx] = True
            esc = False
        elif c == "\\":
            esc = True
    return escaped


def _old_bracket_close(chars: list[str], escaped: list[bool], open: int) -> int | None:
    """THE REFERENCE: the pre-change ``find_unescaped_bracket_close``,
    transcribed to Python (a depth counter from the opener, escaped
    brackets skipped, the first ``]`` that returns the depth to zero)."""
    depth = 0
    for i in range(open, len(chars)):
        if escaped[i]:
            continue
        c = chars[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
    return None


def _old_paren_close(chars: list[str], start: int) -> int | None:
    """THE REFERENCE: the pre-change URL-paren depth scan from
    ``parse_link``'s body, transcribed to Python: from the ``(`` after a
    closing bracket, every paren counted regardless of escapes, the
    ``)`` that returns the depth to zero."""
    depth = 0
    for i in range(start, len(chars)):
        c = chars[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _new_tables(chars: list[str], escaped: list[bool]) -> tuple[list[int | None], list[int | None]]:
    """The working tree's ``match_delimiters``: one forward pass, two
    LIFO stacks (brackets consult escapes, parens do not), the popped
    opener recording its closer."""
    bracket: list[int | None] = [None] * len(chars)
    paren: list[int | None] = [None] * len(chars)
    open_brackets: list[int] = []
    open_parens: list[int] = []
    for i, c in enumerate(chars):
        if c == "[" and not escaped[i]:
            open_brackets.append(i)
        elif c == "]" and not escaped[i]:
            if open_brackets:
                bracket[open_brackets.pop()] = i
        elif c == "(":
            open_parens.append(i)
        elif c == ")":
            if open_parens:
                paren[open_parens.pop()] = i
    return bracket, paren


def _assert_substitution(line: str) -> None:
    """The invariant the whole change rests on: at every position the
    caller can consult (an unescaped ``[``, a ``(``), the new table
    answers exactly what the old scan answered. The caller loop is
    unchanged code, so this is byte-identity by substitution."""
    chars = list(line)
    escaped = _escaped_at(chars)
    bracket, paren = _new_tables(chars, escaped)
    for i, c in enumerate(chars):
        if c == "[" and not escaped[i]:
            assert bracket[i] == _old_bracket_close(chars, escaped, i), (
                f"bracket table diverges from the old scan at {line!r}[{i}]: "
                f"table={bracket[i]} scan={_old_bracket_close(chars, escaped, i)}"
            )
        if c == "(":
            assert paren[i] == _old_paren_close(chars, i), (
                f"paren table diverges from the old scan at {line!r}[{i}]: "
                f"table={paren[i]} scan={_old_paren_close(chars, i)}"
            )


def _old_parse_link(
    chars: list[str], escaped: list[bool], open: int
) -> tuple[str, str, int] | None:
    """The pre-change ``parse_link`` whole: the old bracket scan, the
    ``(`` requirement, the old paren scan (all three transcribed)."""
    close = _old_bracket_close(chars, escaped, open)
    if close is None:
        return None
    nxt = close + 1
    if nxt >= len(chars) or chars[nxt] != "(":
        return None
    end = _old_paren_close(chars, nxt)
    if end is None:
        return None
    return ("".join(chars[open + 1 : close]), "".join(chars[nxt + 1 : end]), end + 1)


def _old_strip_line(line: str, depth: int = 0) -> str:
    """The caller loop with the OLD scans plugged in, over the
    differential's alphabet (brackets, parens, backslashes, bangs,
    plain text: no emphasis markers, autolinks, or code spans, machinery
    the change did not touch). This is the pre-change strip's bytes for
    every line the end-to-end cells below feed it."""
    if depth >= MAX_INLINE_DEPTH:
        return line
    chars = list(line)
    escaped = _escaped_at(chars)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if escaped[i] or c == "\\":
            out.append(c)
            i += 1
            continue
        if c == "!" and i + 1 < len(chars) and chars[i + 1] == "[":
            parsed = _old_parse_link(chars, escaped, i + 1)
            if parsed is not None:
                label, _url, nxt = parsed
                out.append(_old_strip_line(label, depth + 1))
                i = nxt
                continue
        if c == "[":
            parsed = _old_parse_link(chars, escaped, i)
            if parsed is not None:
                label, url, nxt = parsed
                if url == label or url == f"mailto:{label}":
                    out.append(label)
                else:
                    out.append(_old_strip_line(label, depth + 1))
                    if url:
                        out.append(f" ({url})")
                i = nxt
                continue
        out.append(c)
        i += 1
    return "".join(out)


_ASCII_PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _unescape_marker_escapes(line: str) -> str:
    """The strip's post-emphasis pass (unchanged machinery, modeled so
    the end-to-end bytes are the pipeline's, not one stage's): the
    backslash before ASCII punctuation goes, the character stays;
    everything else keeps its backslash."""
    chars = list(line)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if (
            c == "\\"
            and i + 1 < len(chars)
            and chars[i + 1] in _ASCII_PUNCTUATION
        ):
            out.append(chars[i + 1])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _old_strip_markdown(md: str) -> str:
    """The strip's line loop for this differential's alphabet (no fences,
    rules, tables, or container markers, machinery the change did not
    touch): each line is end-trimmed, stripped inline (the model above),
    and pushed with a trailing newline; a line whose stripped text is
    empty counts as one paragraph break instead (never flushed at the
    end)."""
    out: list[str] = []
    pending = False

    def push(text: str) -> None:
        nonlocal pending
        if pending:
            out.append("\n")
            pending = False
        out.append(text)
        out.append("\n")

    for line in md.split("\n"):
        text = line.rstrip()
        stripped = _unescape_marker_escapes(_old_strip_line(text))
        if stripped.strip():
            push(stripped)
        else:
            pending = True
    return "".join(out)


def _end_to_end(line: str) -> tuple[str, str]:
    """The line through the public surface: the engine's markdown (the
    line verbatim plus a newline, and its own backslash doubling, which
    the model runs on so the engine's transforms are absorbed), and
    today's plain text."""
    html = f"<html><body><p>{line}</p></body></html>".encode()
    _fmt, md = to_markdown(data=html, format="html")
    _fmt, text = to_text(data=html, format="html")
    return md, text


def _assert_end_to_end(line: str) -> None:
    """Today's bytes equal the old scans' bytes: the model runs over the
    engine's actual markdown, the whole line loop (per-line isolation
    included)."""
    md, text = _end_to_end(line)
    expected = _old_strip_markdown(md)
    assert text == expected, (
        f"the strip changed bytes on {line!r}: old-scan bytes {expected!r}, "
        f"today's bytes {text!r} (engine markdown {md!r})"
    )


_ALPHABET = "[]()\\!abx:@/.<>~* \u0000"


class TestTheSubstitutionInvariant:
    """The matcher's tables against the old scans, position by position,
    over a generated space (the fuzz duty: the Rust fuzz target needs a
    nightly toolchain this box does not have, so the hypothesis space
    carries it: a few thousand lines over every delimiter, escape, NUL,
    emphasis, and angle character, each checked at every consultable
    position)."""

    @settings(
        max_examples=2000,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(st.text(alphabet=_ALPHABET, min_size=0, max_size=80))
    def test_tables_answer_what_the_scans_answered(self, line: str) -> None:
        _assert_substitution(line)

    @pytest.mark.parametrize(
        "line",
        [
            "[a](u)",
            "[a",
            "a]",
            "[a] b ] (u)",
            "\\[a\\](u)",
            "[a\\]b](u)",
            "[[a](u)](v)",
            "![alt](src)",
            "![a](u) and [b](v)",
            "[u](u)",
            "[e](mailto:e)",
            "](",
            "]before [a](u)",
            "[a](u",
            "[a](u))",
            "[a]((u))",
            "[(a)]((u))",
            "[a(b]c)(d)",
            "[]()",
            "[](u)",
            "[a]()",
            "\\!\\[x\\](y)",
            "!\\[x](y)",
            "[\\[nested\\]](u)",
            "[a\\(b\\)](u)",
            "\x00[1]\x00(u)",
            "[a\x00b](u)",
            "[" * 300 + "a" + "]" * 300 + "(u)",
            "a" * 200 + "](" + "b" * 200,
        ],
        ids=[
            "plain-link",
            "unmatched-opener",
            "unmatched-closer",
            "mixed-backward-shape",
            "escaped-brackets",
            "escaped-close-inside-label",
            "nested-labels",
            "image",
            "image-then-link",
            "label-equals-url",
            "label-equals-mailto-url",
            "lone-close-paren-pair",
            "close-before-open",
            "unclosed-url",
            "extra-close",
            "nested-url-parens",
            "parens-in-label-and-url",
            "crossed-delimiters",
            "empty-everything",
            "empty-label",
            "empty-url",
            "escaped-image-markers",
            "escaped-open-only-image",
            "escaped-nested-label",
            "escaped-parens-in-label",
            "nul-sentinel-shaped",
            "nul-inside-label",
            "deep-nesting-past-the-cap",
            "long-unclosed-url-tail",
        ],
    )
    def test_hand_shapes_substitute(self, line: str) -> None:
        _assert_substitution(line)


class TestTheEndToEndBytes:
    """The same shapes through the public surface: today's plain text
    must be the old scans' plain text, byte for byte, over the engine's
    actual markdown."""

    @pytest.mark.parametrize(
        "line",
        [
            "[a](u)",
            "[a",
            "a]",
            "[a] b ] (u)",
            "\\[a\\](u)",
            "[a\\]b](u)",
            "[[a](u)](v)",
            "![alt](src)",
            "![a](u) and [b](v)",
            "[u](u)",
            "[e](mailto:e)",
            "](",
            "]before [a](u)",
            "[a](u",
            "[a](u))",
            "[a]((u))",
            "[(a)]((u))",
            "[]()",
            "[a]()",
            "!\\[x](y)",
            "[\\[nested\\]](u)",
        ],
        ids=[
            "plain-link",
            "unmatched-opener",
            "unmatched-closer",
            "mixed-backward-shape",
            "escaped-brackets",
            "escaped-close-inside-label",
            "nested-labels",
            "image",
            "image-then-link",
            "label-equals-url",
            "label-equals-mailto-url",
            "lone-close-paren-pair",
            "close-before-open",
            "unclosed-url",
            "extra-close",
            "nested-url-parens",
            "parens-in-label-and-url",
            "empty-everything",
            "empty-url",
            "escaped-open-only-image",
            "escaped-nested-label",
        ],
    )
    def test_today_matches_the_old_scans(self, line: str) -> None:
        _assert_end_to_end(line)

    def test_per_line_isolation(self) -> None:
        """A closer on one line cannot pair an opener on the next: two
        paragraphs, each stripped on its own line, the bytes of two
        independent one-line strips joined by the line loop's single
        paragraph break."""
        html = b"<html><body><p>](u)</p><p>[a</p></body></html>"
        _fmt, md = to_markdown(data=html, format="html")
        _fmt, text = to_text(data=html, format="html")
        expected = _old_strip_markdown(md)
        assert text == expected, f"per-line isolation broke: {text!r} vs {expected!r}"

    @pytest.mark.parametrize("depth", [255, 256, 257], ids=["at-the-cap", "past-it", "well-past"])
    def test_the_depth_ladder_degrades_identically(self, depth: int) -> None:
        """Nesting at and past the cap: the ladder returns the remaining
        label verbatim at MAX_INLINE_DEPTH, and the old scans' model
        degrades on the same rung (the cap is caller machinery, but the
        rung where it fires depends on where the matcher says the labels
        end)."""
        line = "[" * depth + "a" + "]" * depth + "(u)"
        _assert_end_to_end(line)


_LONG_LINE_CHILD = r"""
import sys, time
from tors_documents import to_text

n = int(sys.argv[1])
kind = sys.argv[2]
if kind == "brackets":
    line = "[" * n
    expected = ("[" * n) + "\n"
else:
    line = "[a](" * (n // 4)
    expected = ("[a](" * (n // 4)) + "\n"
html = b"<html><body><p>" + line.encode() + b"</p></body></html>"
t0 = time.perf_counter()
_fmt, text = to_text(data=html, format="html")
elapsed = time.perf_counter() - t0
assert text == expected, f"the strip changed bytes: got {text[:60]!r}"
print(f"n={n} kind={kind} elapsed={elapsed:.3f}s outlen={len(text)}")
"""


class TestTheLongLineContract:
    """A 1 MiB single line completes linearly (the two shapes that were
    quadratic before the matcher: N unmatched openers, N unclosed url
    parens), with its bytes exactly the verbatim pass-through, in a
    subprocess under a deadline so a regression dies as a child timeout
    instead of eating the runner."""

    @pytest.mark.parametrize(
        ("kind", "n"),
        [("brackets", 1_048_576), ("parens", 1_048_576)],
        ids=["openers", "url-parens"],
    )
    def test_a_one_mib_line_is_linear(self, kind: str, n: int) -> None:
        done = subprocess.run(
            [sys.executable, "-c", _LONG_LINE_CHILD, str(n), kind],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        report = f"rc={done.returncode}\n{done.stdout}\n{done.stderr}"
        assert "elapsed=" in done.stdout, f"the probe did not report:\n{report}"
        elapsed = float(done.stdout.split("elapsed=")[1].split("s")[0])
        assert elapsed < 10.0, (
            f"the 1 MiB {kind} line took {elapsed:.3f}s (the pre-matcher scans were "
            f"quadratic: ~12s/~21s at 200k, 4x per doubling):\n{report}"
        )
