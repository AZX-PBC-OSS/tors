//! Fenced code block extraction and dedenting: the pure-Rust cores of
//! `tors.extract_code_blocks`, `tors.strip_code_fences`, and `tors.dedent`.
//!
//! `extract_code_blocks` hand-rolls just CommonMark §4.5's fenced-code-block
//! grammar directly (not a dependency pull of a full CommonMark/Markdown
//! parser): a well-specified, self-contained sub-grammar, matching how
//! `html_impl`/`normalize_impl` are surgical ports of a single spec rather
//! than wrappers around a general-purpose engine. The scope is
//! narrower than full CommonMark in two ways, both documented on the
//! function: no indented-code-block recognition (only *fenced* blocks: the
//! model-output shape this exists for), and no tab-expansion (a fence or a
//! content line's indent is counted in literal ASCII space characters only;
//! a line indented with a tab is not treated as fence-relevant indentation).
//!
//! `dedent` is a port of CPython 3.14's rewritten `textwrap.dedent`
//! (Lib/textwrap.py, gh-131792): normalize whitespace-only lines to empty,
//! then compute the longest common leading-whitespace-run string (not
//! count: `"  "` and `"\t"` share no common prefix), differential-tested
//! against the running stdlib in tests/test_fence.py. tors ships the
//! CPython 3.14 behavior on every supported Python version; see the
//! version note on `fence_impl::dedent` for the specific pre-3.14
//! divergence (Unicode-whitespace-only lines like a lone `\v` or `\f`) this
//! choice documents rather than replicates, the same convention
//! `src/b64_impl.rs` uses for `b64_decode`.

use std::borrow::Cow;

use memchr::memchr;

use crate::normalize_impl::is_py_whitespace;

/// One fenced code block: `language` is the info string's first
/// whitespace-delimited word (`None` if the info string is empty or
/// absent); `code` is the dedented content between the fences; `start`/`end`
/// are Python str index (codepoint) offsets of the block's raw span in the
/// original text: the opening fence line's first character through the end
/// of the closing fence line's line terminator (or end of input, for an
/// unterminated fence); `code_start`/`code_end` are the same-unit offsets of
/// the raw (undedented, terminator-preserving) content between the fences:
/// the span `code` was dedented from, exposed for callers that need the
/// bytes verbatim (`json_repair_impl`'s fence pre-pass must not let the
/// dedent rewrite JSON string content).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeBlock {
    pub language: Option<String>,
    pub code: String,
    pub start: usize,
    pub end: usize,
    pub code_start: usize,
    pub code_end: usize,
}

/// A physical line of the input, as both byte and char spans: `content` is
/// `text[byte_start..byte_content_end]` (no line terminator); `char_start`/
/// `char_content_end` are the same range in codepoint units; `char_line_end`
/// is one past the terminator (equal to `char_content_end` for the final
/// line when the input has no trailing newline); `has_newline` records
/// whether a `\n` terminator was actually present (the unterminated-fence
/// content-join needs to know this for the true last line of the input).
struct Line<'a> {
    content: &'a str,
    char_start: usize,
    char_line_end: usize,
    has_newline: bool,
}

/// Split `text` into physical lines (split on `\n`, `\r` is ordinary
/// content: the same naive split `textwrap.dedent`'s `text.split('\n')`
/// performs), one forward pass computing char offsets alongside the byte
/// spans used to slice `text` zero-copy.
fn lines(text: &str) -> Vec<Line<'_>> {
    let mut out = Vec::new();
    let mut byte_start = 0usize;
    let mut char_start = 0usize;
    let bytes = text.as_bytes();
    loop {
        // memchr for the line-terminator scan (SIMD) rather than
        // `str::find(char)`'s byte-at-a-time search: this loop is the
        // whole-document cost of line-splitting, the same discipline
        // html_impl/url_impl already apply to their byte scans.
        let rel_newline = memchr(b'\n', &bytes[byte_start..]);
        let (byte_content_end, byte_line_end, has_newline) = match rel_newline {
            Some(rel) => (byte_start + rel, byte_start + rel + 1, true),
            None => (bytes.len(), bytes.len(), false),
        };
        let content = &text[byte_start..byte_content_end];
        let char_content_end = char_start + content.chars().count();
        let char_line_end = if has_newline {
            char_content_end + 1
        } else {
            char_content_end
        };
        out.push(Line {
            content,
            char_start,
            char_line_end,
            has_newline,
        });
        if !has_newline {
            break;
        }
        byte_start = byte_line_end;
        char_start = char_line_end;
    }
    out
}

/// A recognized opening fence: `fence_char` (`` ` `` or `~`), `fence_len`
/// (3+), `indent` (0-3 leading ASCII spaces, stripped from content lines),
/// and `language`.
struct OpenFence {
    fence_char: char,
    fence_len: usize,
    indent: usize,
    language: Option<String>,
}

/// Does `line` open a fence? CommonMark §4.5: up to 3 leading spaces, then
/// 3+ of the same `` ` `` or `~`, then an info string. A backtick fence's
/// info string must not itself contain a backtick (a tilde fence's may
/// contain anything): a line that looks like a fence but violates this is
/// ordinary content, not an opener.
fn match_open_fence(line: &str) -> Option<OpenFence> {
    let indent = line.chars().take_while(|&c| c == ' ').count().min(3);
    if line.chars().take(indent).any(|c| c != ' ') {
        return None;
    }
    let rest = &line[indent..];
    let fence_char = rest.chars().next()?;
    if fence_char != '`' && fence_char != '~' {
        return None;
    }
    let fence_len = rest.chars().take_while(|&c| c == fence_char).count();
    if fence_len < 3 {
        return None;
    }
    let info = rest[fence_len..].trim();
    if fence_char == '`' && info.contains('`') {
        return None;
    }
    let language = info.split_whitespace().next().map(|s| s.to_string());
    Some(OpenFence {
        fence_char,
        fence_len,
        indent,
        language,
    })
}

/// Does `line` close the fence described by `open`? Up to 3 leading spaces,
/// then a run of `open.fence_char` of length >= `open.fence_len`, then only
/// trailing whitespace: nothing else on the line.
///
/// CommonMark §4.5 permits only spaces/tabs after the fence run: not the
/// full Unicode `White_Space` set `char::is_whitespace` covers, which also
/// matches form feed, vertical tab, NBSP, and line/paragraph separators.
/// `\r` is the one addition beyond the spec text: `lines()` never
/// strips it (a CRLF document's `\r` stays attached to the line as ordinary
/// content: see the module docs), so a CRLF-terminated closing fence line
/// like `` "```\r" `` needs `\r` treated as trailing blank here or every
/// CRLF document would fail to close its fences at all.
fn is_closing_fence(line: &str, open: &OpenFence) -> bool {
    let indent = line.chars().take_while(|&c| c == ' ').count().min(3);
    if line.chars().take(indent).any(|c| c != ' ') {
        return false;
    }
    let rest = &line[indent..];
    let run_len = rest.chars().take_while(|&c| c == open.fence_char).count();
    if run_len < open.fence_len {
        return false;
    }
    rest[run_len..]
        .chars()
        .all(|c| c == ' ' || c == '\t' || c == '\r')
}

/// Strip up to `indent` leading ASCII spaces from `line`: "as far as it
/// goes" when the line has fewer than `indent` leading spaces (CommonMark
/// §4.5's under-indented-content-line rule).
fn strip_indent(line: &str, indent: usize) -> &str {
    let strip = line.chars().take(indent).take_while(|&c| c == ' ').count();
    let byte_offset: usize = line.chars().take(strip).map(char::len_utf8).sum();
    &line[byte_offset..]
}

/// The fenced-code-block scan behind [`extract_code_blocks`]: collects
/// every block (unfiltered); the `lang` filter is applied by the caller so
/// this core has one job.
fn scan(text: &str) -> Vec<CodeBlock> {
    let physical = lines(text);
    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < physical.len() {
        let Some(open) = match_open_fence(physical[i].content) else {
            i += 1;
            continue;
        };
        let start = physical[i].char_start;
        let mut j = i + 1;
        let mut closing = None;
        while j < physical.len() {
            if is_closing_fence(physical[j].content, &open) {
                closing = Some(j);
                break;
            }
            j += 1;
        }
        let mut code = String::new();
        let content_lines = &physical[i + 1..j.min(physical.len())];
        for (idx, line) in content_lines.iter().enumerate() {
            code.push_str(strip_indent(line.content, open.indent));
            if line.has_newline || idx + 1 < content_lines.len() {
                code.push('\n');
            }
        }
        let end = match closing {
            Some(close_idx) => physical[close_idx].char_line_end,
            None => text.chars().count(),
        };
        // The raw content span: from the first byte after the opening fence
        // line's terminator (or the degenerate no-content-lines position,
        // which `physical.get(i + 1)` handles by falling back to the opener
        // line's own end) to the start of the closing fence line, or to the
        // end of input for an unterminated fence.
        let code_start = physical
            .get(i + 1)
            .map_or(physical[i].char_line_end, |line| line.char_start);
        let code_end = match closing {
            Some(close_idx) => physical[close_idx].char_start,
            None => text.chars().count(),
        };
        blocks.push(CodeBlock {
            language: open.language,
            code,
            start,
            end,
            code_start,
            code_end,
        });
        i = match closing {
            Some(close_idx) => close_idx + 1,
            None => physical.len(),
        };
    }
    blocks
}

/// `tors.extract_code_blocks`'s core: every fenced code block in `text`, in
/// document order, optionally filtered to those whose language matches
/// `lang` exactly (case-sensitive: model output overwhelmingly emits
/// lowercase info strings, and a silent case-fold would hide a caller's own
/// typo). See the module docs for the scope this hand-rolled sub-grammar
/// does not cover (indented code blocks, tab-expanded indents).
pub fn extract_code_blocks(text: &str, lang: Option<&str>) -> Vec<CodeBlock> {
    let blocks = scan(text);
    match lang {
        None => blocks,
        Some(want) => blocks
            .into_iter()
            .filter(|b| b.language.as_deref() == Some(want))
            .collect(),
    }
}

/// `tors.strip_code_fences`'s core: if `text`, trimmed of leading/trailing
/// whitespace, is exactly one fenced code block (nothing before the opening
/// fence, nothing after the closing fence or EOF), return its dedented code
/// content; otherwise return `text` unchanged, not even whitespace-trimmed,
/// so this is a safe no-op on anything but the single-block-wraps-the-
/// whole-response case it exists for. An unterminated fence counts too (the
/// model forgot to close it): the block still runs to the trimmed text's end.
pub fn strip_code_fences(text: &str) -> Cow<'_, str> {
    let trimmed = text.trim();
    let mut blocks = scan(trimmed);
    if let [block] = blocks.as_mut_slice()
        && block.start == 0
        && block.end == trimmed.chars().count()
    {
        return Cow::Owned(std::mem::take(&mut block.code));
    }
    Cow::Borrowed(text)
}

/// `tors.repair_json`'s fence pre-pass core: the single-fence unwrap in its
/// raw form. Same single-block-spans-the-whole-trimmed-input gate as
/// [`strip_code_fences`], but the content comes back verbatim: undedented,
/// CRLF-preserving, because the JSON repair that consumes it must see the
/// bytes the model actually emitted: the fence dedent strips up to three
/// leading spaces per line, and inside a JSON *string value* those spaces
/// are payload, not indentation. `None` for every non-single-block input
/// (prose around the fence, multiple blocks, no fence). The CommonMark
/// grammar (backtick or tilde fences, 3+ long, longer closers, tolerated
/// fence indent) is the same scan `extract_code_blocks` runs: json_repair
/// itself reaches the embedded JSON by skipping the wrapper characters, which
/// lands on the same result for container payloads; tors additionally runs
/// its strict fast path over the unwrapped text and recovers fenced
/// top-level scalars, both documented on the repair surface.
pub fn unwrap_code_fence(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    let blocks = scan(trimmed);
    let total_chars = trimmed.chars().count();
    if let [block] = &blocks[..]
        && block.start == 0
        && block.end == total_chars
    {
        let to_byte = |char_idx: usize| -> usize {
            trimmed
                .char_indices()
                .nth(char_idx)
                .map_or(trimmed.len(), |(byte, _)| byte)
        };
        let start = to_byte(block.code_start);
        let end = to_byte(block.code_end);
        return trimmed.get(start..end);
    }
    None
}

/// `tors.dedent`'s core: `textwrap.dedent`'s CPython 3.14+ algorithm (see
/// the module docs): whitespace-only lines normalize to empty, then the
/// longest common leading-whitespace-run string among the remaining
/// non-empty lines is computed and stripped from every line that starts
/// with it. Returns `Cow::Borrowed(text)` only when the transform is a true
/// no-op (no whitespace-only line to normalize and nothing to strip), the
/// crate's `Cow` identity convention, checked by a final `out == text`
/// comparison rather than short-circuited, so it is exact by construction
/// rather than by a separate no-op detector that could drift from the
/// transform itself.
///
/// **Version note**: `textwrap.dedent` was rewritten in CPython 3.14
/// (gh-131792), and the rewrite changed observable behavior, not just
/// performance: the old implementation tested "is this line whitespace-
/// only" with the regex `^[ \t]+$`, so a line made of some other Unicode
/// whitespace character (`\v`, `\f`, a non-breaking space, ...) was never
/// recognized as blank and its (zero-length, since `[ \t]` doesn't match
/// it) leading run collapsed the common margin to nothing. 3.14 tests
/// blankness with `str.strip()`, which is Unicode-whitespace-aware, and
/// computes the margin from a plain string common-prefix rather than a
/// `[ \t]*`-anchored regex, so the leading run it credits toward the
/// margin is any Unicode whitespace, not just space/tab. Following this
/// crate's `b64_decode`-style convention for cross-version stdlib parity
/// (see `src/b64_impl.rs`'s module doc), tors ships the fixed, 3.14
/// behavior unconditionally on every Python version it supports (3.10+);
/// callers on an older interpreter will see `tors.dedent` diverge from
/// their own `textwrap.dedent` on inputs containing non-space/tab
/// whitespace-only lines. Ordinary indentation (space/tab, by far the
/// common case) is unaffected either way.
pub fn dedent(text: &str) -> Cow<'_, str> {
    let raw_lines: Vec<&str> = text.split('\n').collect();
    // Step 1: whitespace-only lines (>=1 char, all Unicode-whitespace)
    // normalize to "": matches `str.strip()`'s truthiness test, not the
    // old `[ \t]+` regex (see the version note above).
    let normalized: Vec<&str> = raw_lines
        .iter()
        .map(|&line| {
            if !line.is_empty() && line.chars().all(is_py_whitespace) {
                ""
            } else {
                line
            }
        })
        .collect();
    // Step 2: the leading [ \t] run of every line that has a non-whitespace
    // char (i.e. every non-empty line post-normalization) contributes to
    // the margin: exactly space and tab, the CPython 3.14 stdlib's own
    // margin set (measured: every other whitespace char, VT through
    // ideographic space, leaves a leading run untouched on the running
    // stdlib; only the step-1 line normalization uses the full isspace
    // set).
    let mut margin: Option<&str> = None;
    for &line in &normalized {
        if line.is_empty() {
            continue;
        }
        let lead_len: usize = line
            .chars()
            .take_while(|&c| matches!(c, ' ' | '\t'))
            .count();
        let lead_byte: usize = line.chars().take(lead_len).map(char::len_utf8).sum();
        let indent = &line[..lead_byte];
        margin = Some(match margin {
            None => indent,
            Some(m) if indent.starts_with(m) => m,
            Some(m) if m.starts_with(indent) => indent,
            Some(m) => {
                let common = m
                    .chars()
                    .zip(indent.chars())
                    .take_while(|(x, y)| x == y)
                    .count();
                let common_bytes: usize = m.chars().take(common).map(char::len_utf8).sum();
                &m[..common_bytes]
            }
        });
    }
    let margin = margin.unwrap_or("");
    let out = if margin.is_empty() {
        normalized.join("\n")
    } else {
        normalized
            .iter()
            .map(|&line| line.strip_prefix(margin).unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
    };
    if out == text {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_single_block() {
        let text = "before\n```python\nprint(1)\n```\nafter";
        let got = extract_code_blocks(text, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].language.as_deref(), Some("python"));
        assert_eq!(got[0].code, "print(1)\n");
        assert_eq!(text, "before\n```python\nprint(1)\n```\nafter");
    }

    #[test]
    fn no_language() {
        let text = "```\nplain\n```";
        let got = extract_code_blocks(text, None);
        assert_eq!(got[0].language, None);
        assert_eq!(got[0].code, "plain\n");
    }

    #[test]
    fn tilde_fence_allows_backtick_in_info() {
        let text = "~~~text with ` backtick\ncontent\n~~~";
        let got = extract_code_blocks(text, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].language.as_deref(), Some("text"));
    }

    #[test]
    fn backtick_info_with_backtick_is_not_a_fence() {
        let text = "```has ` backtick\nnot a real fence body\n```";
        let got = extract_code_blocks(text, None);
        // The first line fails as an opener (info string carries a
        // backtick); the trailing "```" line is then read as an unterminated
        // opener of its own, with no content lines after it.
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].code, "");
        assert_eq!(got[0].start, text.rfind("```").unwrap());
    }

    #[test]
    fn unterminated_fence_runs_to_eof() {
        let text = "```rust\nfn main() {}\n";
        let got = extract_code_blocks(text, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].code, "fn main() {}\n");
        assert_eq!(got[0].end, text.chars().count());
    }

    #[test]
    fn closing_fence_needs_matching_length_and_char() {
        // A 2-backtick line never closes anything (not a fence at all).
        let text = "```\ncode\n``\nmore\n```";
        let got = extract_code_blocks(text, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].code, "code\n``\nmore\n");
        // A longer closer (4 backticks) closes a 3-backtick opener.
        let text2 = "```\ncode\n````";
        let got2 = extract_code_blocks(text2, None);
        assert_eq!(got2[0].code, "code\n");
        assert_eq!(got2[0].end, text2.chars().count());
    }

    #[test]
    fn indentation_is_stripped_up_to_fence_indent() {
        let text = "  ```\n  aligned\n    extra indent\naligned less\n  ```";
        let got = extract_code_blocks(text, None);
        assert_eq!(got[0].code, "aligned\n  extra indent\naligned less\n");
    }

    #[test]
    fn multiple_blocks_and_lang_filter() {
        let text = "```python\na = 1\n```\ntext\n```rust\nfn f() {}\n```\n```python\nb = 2\n```";
        let all = extract_code_blocks(text, None);
        assert_eq!(all.len(), 3);
        let py = extract_code_blocks(text, Some("python"));
        assert_eq!(py.len(), 2);
        assert_eq!(py[0].code, "a = 1\n");
        assert_eq!(py[1].code, "b = 2\n");
        let none = extract_code_blocks(text, Some("go"));
        assert!(none.is_empty());
    }

    #[test]
    fn strip_whole_response_wrapped_in_one_fence() {
        assert_eq!(strip_code_fences("```python\nprint(1)\n```"), "print(1)\n");
        assert_eq!(
            strip_code_fences("  \n```python\nprint(1)\n```\n  "),
            "print(1)\n"
        );
        // Unterminated whole-response wrap.
        assert_eq!(strip_code_fences("```\nprint(1)"), "print(1)");
    }

    #[test]
    fn strip_is_a_no_op_outside_the_single_block_case() {
        for text in [
            "no fences here",
            "prose\n```py\ncode\n```\nmore prose",
            "```a\n```\n```b\n```",
            "",
        ] {
            assert_eq!(&*strip_code_fences(text), text);
        }
    }

    #[test]
    fn strip_a_bare_empty_fence_is_the_degenerate_single_block_case() {
        // "```" alone is a single unterminated fence spanning the whole
        // (trimmed) input, with no content lines: the rule applies
        // consistently even at this degenerate edge, matching
        // unterminated_fence_runs_to_eof's semantics.
        assert_eq!(strip_code_fences("```"), "");
    }

    #[test]
    fn dedent_basic() {
        assert_eq!(&*dedent("  a\n  b\n"), "a\nb\n");
        assert_eq!(&*dedent("    a\n      b\n"), "a\n  b\n");
    }

    #[test]
    fn dedent_mixed_tabs_and_spaces_share_no_margin() {
        assert_eq!(&*dedent("  a\n\tb\n"), "  a\n\tb\n");
    }

    #[test]
    fn dedent_whitespace_only_lines_normalize_but_dont_gate_margin() {
        assert_eq!(&*dedent("  a\n   \n  b\n"), "a\n\nb\n");
    }

    #[test]
    fn dedent_form_feed_and_vertical_tab_only_lines_normalize_too() {
        // CPython 3.14 (gh-131792) tests blankness with `str.strip()`
        // (Unicode-whitespace-aware), not the pre-3.14 `^[ \t]+$` regex, so
        // a line made only of `\v`/`\f` now normalizes and no longer
        // collapses the common margin to nothing. See the version note on
        // `dedent`.
        assert_eq!(&*dedent("  a\n\x0b\n  b\n"), "a\n\nb\n");
        assert_eq!(&*dedent("  a\n\x0c\n  b\n"), "a\n\nb\n");
    }

    #[test]
    fn dedent_no_common_margin_is_identity_shaped() {
        assert!(matches!(dedent("a\n  b\n"), Cow::Borrowed(_)));
    }

    #[test]
    fn dedent_empty_input() {
        assert_eq!(&*dedent(""), "");
    }

    #[test]
    fn closing_fence_indented_four_or_more_spaces_does_not_close() {
        // 4+ leading spaces is content, not a (tolerated up-to-3) closer:
        // the block runs to the next valid closer instead.
        let text = "```\ncode\n    ```\nstill code\n```";
        let got = extract_code_blocks(text, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].code, "code\n    ```\nstill code\n");
        assert_eq!(got[0].end, text.chars().count());
    }

    #[test]
    fn closing_fence_tolerates_only_space_tab_cr_not_other_unicode_whitespace() {
        // Form feed, vertical tab, and NBSP are Unicode `White_Space` but are
        // not "spaces or tabs" per CommonMark §4.5: a line with one of
        // these trailing the fence run must not close.
        for trailing in ['\u{000B}', '\u{000C}', '\u{00A0}'] {
            let text = format!("```\ncode\n```{trailing}\nmore\n```");
            let got = extract_code_blocks(&text, None);
            assert_eq!(got.len(), 1, "trailing {trailing:?} incorrectly closed");
            assert!(got[0].code.contains("more"));
        }
        // Space and tab close the fence.
        for trailing in [' ', '\t'] {
            let text = format!("```\ncode\n```{trailing}\nmore\n```");
            let got = extract_code_blocks(&text, None);
            assert_eq!(got.len(), 2, "trailing {trailing:?} should still close");
        }
    }

    #[test]
    fn crlf_line_endings_are_preserved_as_ordinary_content() {
        // \r is never a line terminator here (only \n is: see `lines()`'s
        // docs); a CRLF document's \r rides along as the last byte of each
        // line's content. The fence/indent/info-string machinery still
        // works (is_closing_fence tolerates a trailing \r, see above), but
        // the extracted code retains every \r verbatim: pinning this as
        // the real, documented current behavior, not silently assuming
        // CRLF gets normalized away.
        let text = "```python\r\nprint(1)\r\n```\r\nafter";
        let got = extract_code_blocks(text, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].language.as_deref(), Some("python"));
        assert_eq!(got[0].code, "print(1)\r\n");
        assert_eq!(
            &text[got[0].start..got[0].end],
            "```python\r\nprint(1)\r\n```\r\n"
        );
    }

    #[test]
    fn eof_immediately_after_unterminated_opening_fence_line_with_no_newline() {
        // The whole input is the opening fence line, with no trailing \n at
        // all: zero content lines, not a panic on an empty line slice.
        let text = "```python";
        let got = extract_code_blocks(text, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].language.as_deref(), Some("python"));
        assert_eq!(got[0].code, "");
        assert_eq!(got[0].end, text.chars().count());
    }

    #[test]
    fn extremely_long_fence_run_does_not_panic_and_stays_linear() {
        let fence = "`".repeat(10_000);
        let text = format!("{fence}rs\ncode\n{fence}");
        let got = extract_code_blocks(&text, None);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].language.as_deref(), Some("rs"));
        assert_eq!(got[0].code, "code\n");
    }

    #[test]
    fn control_characters_in_info_string_do_not_panic() {
        let text = "```py\u{0}\u{7}weird\ncode\n```";
        let got = extract_code_blocks(text, None);
        assert_eq!(got.len(), 1);
        // Not asserting a specific language spelling here: only that
        // control bytes in the info string are handled without panicking
        // and without corrupting the surrounding scan.
        assert!(got[0].language.is_some());
        assert_eq!(got[0].code, "code\n");
    }

    #[test]
    fn unwrap_returns_raw_undedented_content() {
        // unwrap_code_fence trims the whole input first, so the single
        // block's opening fence line always sits at indent 0 and the raw
        // content keeps every content line's own leading whitespace
        // verbatim (which the JSON consumer tolerates as inter-token ws).
        // The dedent-vs-raw distinction is exercised mid-document below.
        let text = "  ```json\n  {\n    \"a\": 1\n  }\n  ```";
        let raw = unwrap_code_fence(text).unwrap();
        assert_eq!(raw, "  {\n    \"a\": 1\n  }\n");
        // Mid-document (untrimmed): the fence carries a 2-space indent that
        // `code` strips (up to the fence indent, per CommonMark) while the
        // raw span keeps it.
        let doc = "x\n  ```json\n  {\n  ```";
        let blocks = extract_code_blocks(doc, None);
        assert_eq!(blocks[0].code, "{\n");
        let to_byte = |ci: usize| doc.char_indices().nth(ci).map_or(doc.len(), |(b, _)| b);
        assert_eq!(
            &doc[to_byte(blocks[0].code_start)..to_byte(blocks[0].code_end)],
            "  {\n"
        );
    }

    #[test]
    fn unwrap_handles_tilde_and_longer_closer_fences() {
        assert_eq!(unwrap_code_fence("~~~json\n[1]\n~~~"), Some("[1]\n"));
        // A 4-tick opener is not closed by a 3-tick line: that line is
        // content, and the block runs to EOF (which still spans the whole
        // input, so the unwrap applies with the stray fence verbatim:
        // here with no trailing newline, since the input has none).
        assert_eq!(
            unwrap_code_fence("````json\n{\"k\": \"v\"}\n```"),
            Some("{\"k\": \"v\"}\n```")
        );
    }

    #[test]
    fn unwrap_preserves_crlf() {
        assert_eq!(
            unwrap_code_fence("```json\r\n{\"a\": 1}\r\n```\r\n"),
            Some("{\"a\": 1}\r\n")
        );
    }

    #[test]
    fn unwrap_is_none_outside_the_single_block_case() {
        assert_eq!(unwrap_code_fence("no fences"), None);
        assert_eq!(unwrap_code_fence("prose\n```json\n{}\n```\ntail"), None);
        assert_eq!(
            unwrap_code_fence("```json\n{}\n```\n```json\n[]\n```"),
            None
        );
        assert_eq!(unwrap_code_fence("```"), Some(""));
    }

    #[test]
    fn code_span_is_consistent_with_the_block_span() {
        // Invariant pins for the new raw-span fields: content span inside
        // block span, and the raw slice round-trips through the same bytes
        // the dedent started from.
        for text in [
            "```json\n{\"a\": 1}\n```",
            "```\nplain\n```",
            "```rust\nfn main() {}\n",
            "  ~~~\n  x\n  ~~~",
            "```",
        ] {
            let trimmed = text.trim();
            let blocks = scan(trimmed);
            for block in &blocks {
                assert!(block.code_start <= block.code_end);
                assert!(block.start <= block.code_start);
                assert!(block.code_end <= block.end);
            }
        }
    }
}
