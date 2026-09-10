//! C0/DEL control-run scrub, the pure-Rust core of `tors.strip_controls`.
//!
//! Replaces every maximal run of C0 controls (`U+0000`–`U+001F`) and DEL
//! (`U+007F`) with a single ASCII space: the shape model-authored display
//! text needs before it is stored or served (a chatty or injected model
//! cannot plant terminal control sequences in a row the UI renders).
//! Three `ta_worker` prompt modules (`fit_prompts`, `enrich_prompts`,
//! `derive_prompts`) each carried their own copy of this exact regex
//! (`re.compile(r"[\x00-\x1f\x7f]+").sub(" ", ...)`) and this is that
//! pattern as one GIL-free primitive, pinned byte-identical to it.
//!
//! Two deliberate scope cuts, both documented rather than hidden:
//!
//! * C1 controls (`U+0080`–`U+009F`) are not touched. The three call-site
//!   regexes do not cover them either (despite docstrings claiming "C0/C1"),
//!   so covering them here would silently change adopted behavior: a value
//!   the old regex passed through would come back scrubbed. If C1 scrubbing
//!   is wanted it is a follow-up with its own contract, not a silent
//!   extension of this one.
//! * `\t`, `\n`, `\r` are C0 (`U+0009`, `U+000A`, `U+000D`) and are
//!   therefore scrubbed like any other control run: newlines included.
//!   That matches the regex exactly, and matches the call sites (each
//!   applies the scrub to single-line display text, then `.strip()`s): do
//!   not reach for this on multi-line prose you want to keep line-shaped.
//!
//! No new dependency: one forward pass over `chars`, O(n), allocating only
//! when the input actually contains a control character (the crate's `Cow`
//! identity convention: `tors.strip_controls(s) is s` exactly when `s`
//! holds no C0/DEL character).

use std::borrow::Cow;

/// Whether `c` is in this module's scrub set: C0 (`U+0000`–`U+001F`) or DEL
/// (`U+007F`). C1 (`U+0080`–`U+009F`) is deliberately excluded; see the
/// module docs.
fn is_scrubbed(c: char) -> bool {
    matches!(c, '\0'..='\x1f' | '\x7f')
}

/// Replace every maximal C0/DEL run in `text` with a single ASCII space.
///
/// * No C0/DEL character anywhere: `text` comes back unchanged
///   (`Cow::Borrowed`), the identity path.
/// * Otherwise each maximal run (one control or a hundred adjacent ones)
///   becomes exactly one `" "`, including runs at either edge (no
///   strip: `"a\x00"` becomes `"a "`, and edge whitespace is the caller's
///   `.strip()` to own, the way the three adopted call sites already do).
pub fn strip_controls(text: &str) -> Cow<'_, str> {
    let first = match text.char_indices().find(|&(_, c)| is_scrubbed(c)) {
        None => return Cow::Borrowed(text),
        Some((byte_idx, _)) => byte_idx,
    };
    // `first` is a `char_indices` offset: a char boundary by construction.
    let mut out = String::with_capacity(text.len());
    out.push_str(&text[..first]);
    let mut in_run = false;
    for c in text[first..].chars() {
        if is_scrubbed(c) {
            if !in_run {
                out.push(' ');
                in_run = true;
            }
        } else {
            out.push(c);
            in_run = false;
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(text: &str) -> String {
        strip_controls(text).into_owned()
    }

    #[test]
    fn clean_input_is_identity() {
        let text = "plain text, café, emoji \u{1f600}, C1 \u{0085} untouched";
        assert!(matches!(strip_controls(text), Cow::Borrowed(_)));
    }

    #[test]
    fn empty_input_is_identity() {
        assert!(matches!(strip_controls(""), Cow::Borrowed(_)));
    }

    #[test]
    fn single_controls_become_single_spaces() {
        assert_eq!(owned("a\x00b"), "a b");
        assert_eq!(owned("a\x7fb"), "a b");
        assert_eq!(owned("a\x1fb"), "a b");
    }

    #[test]
    fn runs_collapse_to_one_space() {
        // The whole C0 range plus DEL in one run: one space, not 33.
        let mut run = String::from("a");
        for cp in 0x00..=0x1fu32 {
            run.push(char::from_u32(cp).unwrap());
        }
        run.push('\x7f');
        run.push('b');
        assert_eq!(owned(&run), "a b");
    }

    #[test]
    fn whitespace_controls_are_scrubbed_too() {
        // \t \n \r are C0: the regex this replaces eats them, so this does.
        assert_eq!(owned("a\tb"), "a b");
        assert_eq!(owned("a\nb"), "a b");
        assert_eq!(owned("a\rb"), "a b");
        assert_eq!(owned("a\r\nb"), "a b");
    }

    #[test]
    fn edge_runs_become_edge_spaces_without_stripping() {
        assert_eq!(owned("\x00abc"), " abc");
        assert_eq!(owned("abc\x00"), "abc ");
        assert_eq!(owned("\x00"), " ");
    }

    #[test]
    fn c1_controls_pass_through_untouched() {
        // U+0080–U+009F are not in the scrub set (the adopted regexes do
        // not cover them either): NEL, and the range endpoints, survive.
        assert!(matches!(strip_controls("\u{0080}"), Cow::Borrowed(_)));
        assert!(matches!(strip_controls("\u{0085}"), Cow::Borrowed(_)));
        assert!(matches!(strip_controls("\u{009f}"), Cow::Borrowed(_)));
        assert_eq!(owned("a\u{0085}b"), "a\u{0085}b");
    }

    #[test]
    fn adjacent_runs_stay_separate_across_clean_text() {
        assert_eq!(owned("a\x00b\x00c"), "a b c");
    }

    #[test]
    fn multibyte_text_around_controls_is_preserved() {
        assert_eq!(owned("caf\u{e9}\x00\u{1f600}"), "caf\u{e9} \u{1f600}");
    }
}
