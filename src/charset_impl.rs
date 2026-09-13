//! Batch codepoint-set validation, the pure-Rust core of
//! `tors.first_invalid_charset`: the identifier-style rules a caller like
//! TaskQ spells with anchored regexes (`\A[A-Za-z_][A-Za-z0-9_]*\Z` for
//! schema identifiers, `\A[A-Za-z0-9_][A-Za-z0-9_.-]*\Z` for queue names,
//! `\A[A-Za-z0-9_\-:.]+\Z` for keyed-ref names), expressed as plain
//! caller-supplied codepoint sets and checked for a whole batch in one
//! pass. The consumer's own measurements drive the shape: each regex call
//! costs 84-950 ns and a whole enqueue's validation cluster sits under a
//! single `py.detach` round trip, so per-item calls through this crate
//! would be slower than the regexes they replace — the only winnable shape
//! is batch (one detach, one pass over all items), which is why the API is
//! batch-only and why the core takes the whole slice at once.
//!
//! The sets are data, not patterns: plain strings of permitted codepoints,
//! membership per codepoint (never per byte). No ranges, no escapes, no
//! classes — an `^[a-z_][a-z0-9_]*$`-style rule is spelled by listing the
//! codepoints, and the Unicode-category classes (`\w`) that would need
//! property tables stay out (the lexical-data charter boundary,
//! docs/design.md). Duplicates in a spelling are idempotent and order is
//! irrelevant: a set, however spelled.
//!
//! Representation, sized for the expected case (a few dozen ASCII
//! codepoints, the shape every real identifier/queue/tag rule has): a
//! one-bit-per-codepoint `u128` over the ASCII range plus a sorted,
//! deduplicated `Vec<char>` tail for the non-ASCII codepoints (usually
//! empty, so the common build allocates nothing). Membership is one
//! shift-and for ASCII, a binary search over the handful of non-ASCII
//! entries otherwise. The core is a trivial one-pass membership walk — no
//! index arithmetic beyond a guarded shift, no `unsafe`, no state — which
//! is why it ships no cargo-fuzz target (the issue's "only if the core
//! grows past trivial" bar): the hypothesis differentials over arbitrary
//! Unicode on the Python side (tests/test_first_invalid_charset.py) cover
//! this input space, and the fuzz ledger's adversarial-byte class does not
//! apply to a function with no byte decoding to get wrong.

/// A set of permitted codepoints in the shape the expected rules produce:
/// a one-bit-per-codepoint mask over ASCII (the whole of every realistic
/// identifier/queue/tag charset) and a sorted, deduplicated vector for the
/// non-ASCII tail. Both structures are idempotent under duplicates in the
/// caller's spelling (a bit set twice is set; the vector dedups), so a set
/// however spelled builds to the same membership.
struct CharSet {
    /// Bit `b` set iff codepoint `b` (an ASCII codepoint, `b < 128`) is in
    /// the set: 16 bytes covering the whole ASCII range.
    ascii: u128,
    /// The set's non-ASCII codepoints, sorted ascending and deduplicated,
    /// queried by binary search. Expected empty (the rules that motivate
    /// this validator are ASCII); when not, it holds the caller's astral
    /// and 2/3-byte codepoints, at most a few dozen of them.
    non_ascii: Vec<char>,
}

impl CharSet {
    fn build(set: &str) -> CharSet {
        let mut ascii = 0u128;
        let mut non_ascii = Vec::new();
        for c in set.chars() {
            let cp = c as u32;
            if cp < 128 {
                ascii |= 1 << cp;
            } else {
                non_ascii.push(c);
            }
        }
        non_ascii.sort_unstable();
        non_ascii.dedup();
        CharSet { ascii, non_ascii }
    }

    /// Whether `c` is in the set: one shift-and for an ASCII codepoint,
    /// a binary search over the (handful-sized) non-ASCII tail otherwise.
    #[inline]
    fn contains(&self, c: char) -> bool {
        let cp = c as u32;
        if cp < 128 {
            (self.ascii >> cp) & 1 != 0
        } else {
            self.non_ascii.binary_search(&c).is_ok()
        }
    }
}

/// The index into `items` of the first item not built entirely from the
/// caller's two sets, `-1` when all pass, short-circuiting at the first
/// offender.
///
/// The positional rule: `first` (when `Some`) is the set of codepoints
/// allowed at position 0, `rest` the set allowed at every position after
/// it — and at position 0 too when `first` is `None` (the uniform
/// spelling, one set everywhere). An empty item is an offender (there is
/// no codepoint at position 0 to check); `Some("")` allows nothing at
/// position 0, so every item offends; `rest == ""` allows nothing after
/// position 0, so under the uniform spelling every item offends and with
/// a non-empty `first` only single-codepoint items drawn from `first`
/// pass. Membership is per codepoint over the whole item, and the answer
/// is an item index, never a position within an item.
pub fn first_invalid_charset(items: &[&str], first: Option<&str>, rest: &str) -> isize {
    let first_set = first.map(CharSet::build);
    let rest_set = CharSet::build(rest);
    for (idx, item) in items.iter().enumerate() {
        let mut chars = item.chars();
        // The empty item: an offender wherever it sits, whatever the sets
        // allow — there is no codepoint at position 0 to check.
        let Some(head) = chars.next() else {
            return idx as isize;
        };
        // Position 0 is governed by `first` when given, by `rest`
        // otherwise (the uniform spelling).
        let head_ok = match &first_set {
            Some(set) => set.contains(head),
            None => rest_set.contains(head),
        };
        if !head_ok || !chars.all(|c| rest_set.contains(c)) {
            return idx as isize;
        }
    }
    -1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(items: &[&str], first: Option<&str>, rest: &str) -> isize {
        first_invalid_charset(items, first, rest)
    }

    #[test]
    fn an_empty_batch_is_vacuously_valid() {
        assert_eq!(check(&[], None, "a"), -1);
        assert_eq!(check(&[], Some(""), ""), -1);
    }

    #[test]
    fn all_valid_batches_answer_minus_one() {
        let first = "abcdefghijklmnopqrstuvwxyz_";
        let rest = "abcdefghijklmnopqrstuvwxyz0123456789_";
        assert_eq!(
            check(&["taskq", "worker_id", "job_42"], Some(first), rest),
            -1
        );
    }

    #[test]
    fn the_first_offending_items_index_is_answered() {
        assert_eq!(check(&["ok", "9bad", "worse"], None, "okwrste"), 1);
        // A second offender past the first is never reported.
        assert_eq!(check(&["ok", "9bad", "worse!!"], None, "okwrste"), 1);
        // The offender at the very end is found.
        assert_eq!(check(&["ok", "ok", "9bad"], None, "ok"), 2);
    }

    #[test]
    fn the_answer_is_an_item_index_not_a_char_position() {
        // "ab" offends at its second codepoint; the answer is 0, the item.
        assert_eq!(check(&["ab"], Some("a"), "a"), 0);
    }

    #[test]
    fn an_empty_item_is_an_offender_wherever_it_sits() {
        assert_eq!(check(&[""], None, "a"), 0);
        assert_eq!(check(&["a", "", "a"], None, "a"), 1);
        // ... regardless of what first allows.
        assert_eq!(check(&[""], Some(""), "a"), 0);
    }

    #[test]
    fn first_none_governs_position_zero_with_rest() {
        // The uniform spelling: rest decides position 0 too.
        assert_eq!(check(&["ab"], None, "ab"), -1);
        assert_eq!(check(&["ba"], None, "ab"), -1);
        assert_eq!(check(&["ca"], None, "ab"), 0);
    }

    #[test]
    fn an_empty_first_set_allows_nothing_at_position_zero() {
        assert_eq!(check(&["a"], Some(""), "a"), 0);
        assert_eq!(check(&["a", "b"], Some(""), "a"), 0);
    }

    #[test]
    fn an_empty_rest_set_allows_nothing_after_position_zero() {
        // Uniform: every item offends.
        assert_eq!(check(&["a"], None, ""), 0);
        // With a real first set, only single-codepoint items from it pass:
        // the positional rule applied literally.
        assert_eq!(check(&["a"], Some("a"), ""), -1);
        assert_eq!(check(&["a", "aa"], Some("a"), ""), 1);
    }

    #[test]
    fn first_only_codepoints_are_legal_only_at_position_zero() {
        assert_eq!(check(&["Aaa"], Some("AB"), "ab"), -1);
        assert_eq!(check(&["aAa"], Some("AB"), "ab"), 0);
        assert_eq!(check(&["A"], Some("AB"), "ab"), -1);
    }

    #[test]
    fn duplicate_codepoints_and_spelling_order_are_irrelevant() {
        let items = ["ab_1", "xa", "1a"];
        assert_eq!(check(&items, Some("ab_"), "ab_1"), 1);
        assert_eq!(check(&items, Some("aabb__"), "aabb__11"), 1);
        assert_eq!(check(&items, Some("_ba"), "1_ba"), 1);
        assert_eq!(check(&items, Some("a_b"), "ba_1"), 1);
    }

    #[test]
    fn membership_is_per_codepoint_across_every_utf8_width() {
        // 2-byte é, 3-byte CJK, 4-byte astral: membership per codepoint,
        // never per byte.
        assert_eq!(check(&["é"], None, "é"), -1);
        assert_eq!(check(&["eé"], None, "é"), 0);
        assert_eq!(check(&["東京"], None, "東京"), -1);
        assert_eq!(check(&["東b京"], None, "東京"), 0);
        assert_eq!(check(&["\u{1f980}\u{1f980}"], None, "\u{1f980}"), -1);
        assert_eq!(check(&["\u{1d54f}x"], None, "\u{1d54f}"), 0);
        assert_eq!(check(&["é東\u{1f980}"], None, "é東\u{1d54f}"), 0);
    }

    #[test]
    fn an_all_ascii_set_builds_no_non_ascii_tail() {
        // The expected case: the ASCII rule's tail stays empty (no
        // allocation), and every ASCII codepoint of the spelling is a
        // member.
        let set = CharSet::build("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_");
        assert!(set.non_ascii.is_empty());
        assert!(set.contains('a'));
        assert!(set.contains('Z'));
        assert!(set.contains('0'));
        assert!(set.contains('_'));
        assert!(!set.contains(' '));
        assert!(!set.contains('é'));
    }
}
