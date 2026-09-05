use unicode_normalization::UnicodeNormalization;

/// Matches CPython's `str.isspace()` character set (Unicode `White_Space` plus the four
/// ASCII separator controls 0x1c-0x1f that Python treats as space but Unicode's formal
/// `White_Space` property excludes), so `.strip()` here drops exactly what Python's would.
fn is_py_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{09}'..='\u{0d}'
            | '\u{1c}'..='\u{1f}'
            | ' '
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

fn py_strip(s: &str) -> &str {
    let start = s.char_indices().find(|&(_, c)| !is_py_whitespace(c)).map(|(i, _)| i);
    let Some(start) = start else {
        return "";
    };
    let end = s
        .char_indices()
        .rev()
        .find(|&(_, c)| !is_py_whitespace(c))
        .map(|(i, c)| i + c.len_utf8())
        .unwrap();
    &s[start..end]
}

fn flush(out: &mut String, nl_run: &mut usize, pending_ws: &mut String) {
    if *nl_run > 0 {
        out.push_str(if *nl_run >= 3 { "\n\n" } else { "\n" });
        if *nl_run == 2 {
            out.push('\n');
        }
        *nl_run = 0;
    }
    if !pending_ws.is_empty() {
        out.push_str(pending_ws);
        pending_ws.clear();
    }
}

/// Single native-Rust pass over the NFC-normalized text doing what cennan's
/// `normalize_text` does with two whole-string-scale `re.sub` calls (chunked there only to
/// bound GIL-hold time under `re`, which never releases the GIL regardless of input size):
/// fold CRLF/CR to LF, drop trailing `[ \t]+` runs immediately before a newline, and
/// collapse runs of 3+ newlines to exactly two. Folding, dropping, and collapsing are all
/// interleaved in one left-to-right scan because each only needs to know, at any position,
/// the run of pending newlines and the run of pending space/tab since the last newline.
pub fn normalize(text: &str) -> String {
    let nfc: String = text.nfc().collect();

    let mut out = String::with_capacity(nfc.len());
    let mut pending_ws = String::new();
    let mut nl_run: usize = 0;

    let mut chars = nfc.chars().peekable();
    while let Some(mut c) = chars.next() {
        if c == '\r' {
            if let Some('\n') = chars.peek() {
                chars.next();
            }
            c = '\n';
        }
        match c {
            ' ' | '\t' => pending_ws.push(c),
            '\n' => {
                pending_ws.clear();
                nl_run += 1;
            }
            _ => {
                flush(&mut out, &mut nl_run, &mut pending_ws);
                out.push(c);
            }
        }
    }
    flush(&mut out, &mut nl_run, &mut pending_ws);

    py_strip(&out).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string() {
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn all_whitespace_collapses_to_empty() {
        assert_eq!(normalize("   \t\n\n\n   "), "");
    }

    #[test]
    fn crlf_and_lone_cr_fold_to_lf() {
        assert_eq!(normalize("a\r\nb\rc\nd"), "a\nb\nc\nd");
    }

    #[test]
    fn trailing_tab_before_newline_is_trimmed() {
        assert_eq!(normalize("a\t\nb\n\n\nc"), "a\nb\n\nc");
    }

    #[test]
    fn blank_run_of_three_collapses_to_two() {
        assert_eq!(normalize("a\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn blank_run_of_two_is_untouched() {
        assert_eq!(normalize("a\n\nb"), "a\n\nb");
    }

    #[test]
    fn nfc_composes_decomposed_sequences() {
        assert_eq!(normalize("cafe\u{0301} \n\n\n\nwater"), "café\n\nwater");
    }

    #[test]
    fn trailing_whitespace_with_no_final_newline_is_stripped() {
        assert_eq!(normalize("hello   "), "hello");
    }

    #[test]
    fn leading_whitespace_is_stripped() {
        assert_eq!(normalize("   hello"), "hello");
    }

    #[test]
    fn interleaved_whitespace_and_newlines_collapse_correctly() {
        // " \t" before the first "\n", then a bare "\n", then " " before the last "\n":
        // every space/tab run immediately preceding a newline is dropped, and the three
        // resulting newlines collapse to two.
        assert_eq!(normalize("a \t\n\n \nb"), "a\n\nb");
    }

    #[test]
    fn whitespace_not_before_a_newline_is_preserved() {
        assert_eq!(normalize("a\n\n\n  b"), "a\n\n  b");
    }

    #[test]
    fn mid_line_spaces_are_never_touched() {
        assert_eq!(normalize("a  b   c"), "a  b   c");
    }

    #[test]
    fn matches_a_pure_python_reference_pipeline() {
        // Mirrors cennan's normalize_text, unchunked, as an independent oracle.
        fn reference(text: &str) -> String {
            let nfc: String = text.nfc().collect();
            let folded = nfc.replace("\r\n", "\n").replace('\r', "\n");
            let mut trimmed = String::new();
            let mut ws = String::new();
            for c in folded.chars() {
                match c {
                    ' ' | '\t' => ws.push(c),
                    '\n' => {
                        ws.clear();
                        trimmed.push('\n');
                    }
                    _ => {
                        trimmed.push_str(&ws);
                        ws.clear();
                        trimmed.push(c);
                    }
                }
            }
            trimmed.push_str(&ws);
            let mut collapsed = String::new();
            let mut run = 0usize;
            for c in trimmed.chars() {
                if c == '\n' {
                    run += 1;
                } else {
                    if run > 0 {
                        let piece = if run >= 3 { "\n\n".to_string() } else { "\n".repeat(run) };
                        collapsed.push_str(&piece);
                        run = 0;
                    }
                    collapsed.push(c);
                }
            }
            if run > 0 {
                let piece = if run >= 3 { "\n\n".to_string() } else { "\n".repeat(run) };
                collapsed.push_str(&piece);
            }
            py_strip(&collapsed).to_string()
        }

        let cases = [
            "",
            "plain text",
            "a\r\nb\rc\n\n\n\nd   \n",
            "  \t\n\n\n leading and trailing \t\n\n\n\n  ",
            "line1\nline2\n\n\n\n\nline3\t \t\n",
            "café\u{0301}\n\n\nnaïve",
        ];
        for case in cases {
            assert_eq!(normalize(case), reference(case), "mismatch for {case:?}");
        }
    }
}
