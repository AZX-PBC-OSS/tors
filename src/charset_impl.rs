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
//! irrelevant: a set, however spelled. Membership is per scalar value with
//! no normalization: precomposed é (U+00E9) and decomposed e + U+0301 are
//! different inputs with different verdicts — callers who need NFC/NFD to
//! agree normalize before validating, which still does not fold
//! confusables (visually similar but distinct codepoints stay distinct, so
//! allow-list exactly the codepoints you mean).
//!
//! One scan answers both published spellings. The walk stops at the first
//! offending codepoint of the first offending item, and at that stop point
//! it already holds the whole offender detail — which item, which CODEPOINT
//! position within it (never a byte offset: the family's data model is
//! codepoints), which codepoint — so the core returns that detail
//! (`FirstInvalid`) and each spelling projects it: `first_invalid_charset`
//! the item index (`-1` when clean), `first_invalid_offender` the
//! `(item, position, codepoint)` tuple the rejection-UX callers need (the
//! consumer's own messages name the losing character and position). The
//! detail is free at the stop point by construction; the one thing the
//! shared walk gains over the index-only spelling is the position counter
//! on the rest walk (one integer add per codepoint, dead in the int
//! projection — the criterion group's band is the honest check).
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
    /// Build a set from its spelling: O(set) over the ASCII codepoints
    /// plus O(set log set) over the non-ASCII tail (sort + dedup, inside
    /// the caller's detach) — negligible for the few-dozen-codepoint
    /// ASCII rules this validator exists for, a real sort for a
    /// 10k-codepoint non-ASCII spelling.
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

/// The one scan's full answer: where the first offending item sits and,
/// within it, where the rule broke — the detail a rejection message needs
/// (the consumer's per-character messages name the losing character and
/// its position), of which `first_invalid_charset`'s `-1`/index answer is
/// the projection.
pub enum FirstInvalid {
    /// Every item passed.
    Clean,
    /// The first offending item: its index into `items`, the CODEPOINT
    /// position within it where the rule broke (0, the head position, when
    /// the first codepoint itself is outside the position-0 set — and for
    /// the empty item, which has no codepoint there to check), and the
    /// codepoint at that position — `None` exactly for the empty item, an
    /// offender with no codepoint at position 0 to name (the tuple
    /// spelling's char field is empty exactly when the item is).
    Offender {
        /// The offending item's index into `items`.
        item: usize,
        /// The first offending position, a codepoint index within the
        /// item (never a byte offset).
        position: usize,
        /// The codepoint at `position`, a 1-char string on the Python
        /// side; `None` for the empty item.
        codepoint: Option<char>,
    },
}

/// The one core scan both published spellings project: the positional rule
/// over every item, stopping at the first offending codepoint of the first
/// offending item and returning the full detail of that stop
/// ([`FirstInvalid`]).
///
/// The positional rule: `first` (when `Some`) is the set of codepoints
/// allowed at position 0, `rest` the set allowed at every position after
/// it — and at position 0 too when `first` is `None` (the uniform
/// spelling, one set everywhere). An empty item is an offender (there is
/// no codepoint at position 0 to check); `Some("")` allows nothing at
/// position 0, so every item offends; `rest == ""` allows nothing after
/// position 0, so under the uniform spelling every item offends and with
/// a non-empty `first` only single-codepoint items drawn from `first`
/// pass. Membership is per codepoint over the whole item.
///
/// The detail is what the walk holds at its stop point — the codepoint
/// the membership test just rejected, and its position, counted in
/// codepoints by the rest walk's counter (the one integer add per
/// codepoint the shared walk carries; the int projection's dead field).
///
/// `#[inline]` so the enum never crosses a call boundary: each projection
/// (`first_invalid_charset`'s isize, the binding's tuple) collapses the
/// scan into itself and the fields it does not keep die at compile time —
/// without it the multi-word stop state is returned through memory and
/// the int spelling pays a fixed marshalling cost for detail it drops
/// (measured: ~+10% on the criterion group's 1/10-item cells until the
/// inline, the band back at baseline after).
#[inline]
pub fn scan_first_invalid(items: &[&str], first: Option<&str>, rest: &str) -> FirstInvalid {
    let first_set = first.map(CharSet::build);
    let rest_set = CharSet::build(rest);
    for (idx, item) in items.iter().enumerate() {
        let mut chars = item.chars();
        // The empty item: an offender wherever it sits, whatever the sets
        // allow — there is no codepoint at position 0 to check, so there
        // is none to name either.
        let Some(head) = chars.next() else {
            return FirstInvalid::Offender {
                item: idx,
                position: 0,
                codepoint: None,
            };
        };
        // Position 0 is governed by `first` when given, by `rest`
        // otherwise (the uniform spelling).
        let head_ok = match &first_set {
            Some(set) => set.contains(head),
            None => rest_set.contains(head),
        };
        if !head_ok {
            return FirstInvalid::Offender {
                item: idx,
                position: 0,
                codepoint: Some(head),
            };
        }
        // The rest walk stops at the first codepoint outside `rest`, and
        // the stop point IS the offender detail: `enumerate` counts
        // codepoints (offset 0 within the rest, so the item position is
        // offset + 1), `find` hands back the rejecting codepoint itself.
        if let Some((offset, c)) = chars.enumerate().find(|(_, c)| !rest_set.contains(*c)) {
            return FirstInvalid::Offender {
                item: idx,
                position: offset + 1,
                codepoint: Some(c),
            };
        }
    }
    FirstInvalid::Clean
}

/// The index into `items` of the first item not built entirely from the
/// caller's two sets, `-1` when all pass, short-circuiting at the first
/// offender: the int projection of [`scan_first_invalid`], the same scan
/// the offender-detail spelling projects (so the two spellings cannot
/// disagree — the Python-side consistency invariant pins it).
pub fn first_invalid_charset(items: &[&str], first: Option<&str>, rest: &str) -> isize {
    match scan_first_invalid(items, first, rest) {
        FirstInvalid::Clean => -1,
        FirstInvalid::Offender { item, .. } => item as isize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(items: &[&str], first: Option<&str>, rest: &str) -> isize {
        first_invalid_charset(items, first, rest)
    }

    fn detail(items: &[&str], first: Option<&str>, rest: &str) -> FirstInvalid {
        scan_first_invalid(items, first, rest)
    }

    #[test]
    fn an_empty_batch_is_vacuously_valid() {
        assert_eq!(check(&[], None, "a"), -1);
        assert_eq!(check(&[], Some(""), ""), -1);
        assert!(matches!(detail(&[], None, "a"), FirstInvalid::Clean));
    }

    #[test]
    fn all_valid_batches_answer_minus_one() {
        let first = "abcdefghijklmnopqrstuvwxyz_";
        let rest = "abcdefghijklmnopqrstuvwxyz0123456789_";
        assert_eq!(
            check(&["taskq", "worker_id", "job_42"], Some(first), rest),
            -1
        );
        assert!(matches!(
            detail(&["taskq", "worker_id", "job_42"], Some(first), rest),
            FirstInvalid::Clean
        ));
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

    // --- The offender detail (the scan's full answer) ---------------------

    fn offender_of(
        items: &[&str],
        first: Option<&str>,
        rest: &str,
    ) -> Option<(usize, usize, Option<char>)> {
        match detail(items, first, rest) {
            FirstInvalid::Clean => None,
            FirstInvalid::Offender {
                item,
                position,
                codepoint,
            } => Some((item, position, codepoint)),
        }
    }

    #[test]
    fn the_detail_names_the_head_offenders_position_zero_and_codepoint() {
        // The uniform spelling: "b" at position 0 is outside "a".
        assert_eq!(offender_of(&["b"], None, "a"), Some((0, 0, Some('b'))));
        // ... and under a first spelling the same position, the first set
        // doing the rejecting.
        assert_eq!(
            offender_of(&["b"], Some("a"), "ab"),
            Some((0, 0, Some('b')))
        );
        // An offender past the first item keeps its own index.
        assert_eq!(
            offender_of(&["a", "b", "a"], None, "a"),
            Some((1, 0, Some('b')))
        );
    }

    #[test]
    fn the_detail_names_the_rest_offenders_codepoint_position() {
        // "ab" under first="a"/rest="a": the rule breaks at position 1.
        assert_eq!(
            offender_of(&["ab"], Some("a"), "a"),
            Some((0, 1, Some('b')))
        );
        // rest="" with a first set: only the head passes, position 1 next.
        assert_eq!(offender_of(&["aa"], Some("a"), ""), Some((0, 1, Some('a'))));
        // A first-only codepoint at position 1 is the offender there (the
        // head must pass first: "a" is in the first set).
        assert_eq!(
            offender_of(&["aAa"], Some("aAB"), "ab"),
            Some((0, 1, Some('A')))
        );
    }

    #[test]
    fn the_empty_items_offender_carries_no_codepoint() {
        // The spelling decision: an empty item has no offending character;
        // the tuple's char field is empty exactly when the item is.
        assert_eq!(offender_of(&[""], None, "a"), Some((0, 0, None)));
        assert_eq!(offender_of(&["a", "", "a"], None, "a"), Some((1, 0, None)));
        assert_eq!(offender_of(&[""], Some(""), "a"), Some((0, 0, None)));
    }

    #[test]
    fn the_detail_position_is_a_codepoint_index_not_a_byte_offset() {
        // 4-byte head, ASCII offender at codepoint 1 (byte offset 4).
        assert_eq!(
            offender_of(&["\u{1f980}x"], None, "\u{1f980}"),
            Some((0, 1, Some('x')))
        );
        // 2-byte + 3-byte members, astral offender at codepoint 2 (byte
        // offset 5), named as its own 1-char codepoint.
        assert_eq!(
            offender_of(&["é東\u{1f980}"], None, "é東\u{1d54f}"),
            Some((0, 2, Some('\u{1f980}')))
        );
    }

    #[test]
    fn the_int_projection_is_the_details_item_index() {
        // The two spellings are projections of one scan: the int answer is
        // the detail's item field, -1 exactly when the detail is Clean.
        for (items, first, rest) in [
            (
                &["taskq", "job_42"][..],
                Some("abcdefghijklmnopqrstuvwxyz_"),
                "abcdefghijklmnopqrstuvwxyz0123456789_",
            ),
            (&["ok", "9bad", "worse"][..], None, "okwrste"),
            (&["ab"][..], Some("a"), "a"),
            (&["a", "", "a"][..], None, "a"),
            (&["\u{1f980}x"][..], None, "\u{1f980}"),
        ] {
            let scanned = offender_of(items, first, rest);
            let projected = check(items, first, rest);
            match scanned {
                None => assert_eq!(projected, -1),
                Some((item, _, _)) => assert_eq!(projected, item as isize),
            }
        }
    }
}
