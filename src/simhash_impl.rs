//! Simhash: the pure-Rust core of tors's fuzzy dedupe gate: a 64-bit
//! locality-sensitive fingerprint of a text's word content whose Hamming
//! distance grows slowly with edit distance, so near-duplicates cluster
//! within a few bits while unrelated texts sit far apart (Charikar's
//! simhash primitive in the web-scale near-dup-detection shape Manku,
//! Jain, and Das Sarma built at Google, WWW 2007; the standard token unit
//! for text is the word, which is why the tokenizer is the crate's UAX #29
//! word segmentation and not, say, shingles over bytes).
//!
//! # Relationship to the exact gates
//!
//! `finalize`'s SHA-256 tail and `merkle_root` answer "is this text
//! byte-identical", gates that split a text from an original differing by
//! a single comma. simhash answers the question those gates cannot: "is
//! this text NEARLY the same." The compatibility row is pinned by the
//! tests below: identical input always yields the identical fingerprint
//! (`simhash64(a) == simhash64(b)` whenever `a == b`), so a pipeline can
//! run both gates off one store: exact dupes at distance 0, near-dupes at
//! small distance, everything else far.
//!
//! # The token hash is FNV-1a, inline: not std's `DefaultHasher`
//!
//! A dedupe fingerprint must be stable ACROSS PROCESSES: `DefaultHasher`
//! is seeded per process (`RandomState`), so a fingerprint it produced
//! would change between runs and between machines, silently breaking
//! every cross-run/cross-machine dedupe built on it. FNV-1a is ~10 lines,
//! deterministic forever, and adequate for a VOTING hash: it is asked
//! only to spread tokens reasonably uniformly across the 64 bit
//! positions, not to be cryptographic; a collision between two distinct
//! tokens merely makes them vote alike, which blurs one vote among 64
//! counters without threatening determinism. No adversarial resistance is
//! claimed or needed: the fingerprint compares texts a caller already
//! holds, not adversaries crafting collisions.
//!
//! # Bag of words: ordering does not matter
//!
//! The vote is over the MULTISET of tokens, not their sequence: word order
//! never affects the fingerprint (the API's most surprising guarantee:
//! "the quick brown fox" and "fox brown quick the" are the SAME
//! fingerprint, pinned by the permutation test below). A repeated token
//! votes once per occurrence, so frequency is preserved; a permutation
//! preserves the multiset exactly, which is why the pin is an equality,
//! not a bound.
//!
//! # Tokenization
//!
//! Tokens are the segments of the SAME UAX #29 word segmentation walk
//! `segmentation_impl::word_bounds` drives (`split_word_bounds`, its
//! sibling spelling over the same tables; the count/list invariant is
//! pinned in `segmentation_impl`), with ONE skip: a segment that is
//! entirely whitespace is not a token. The WSegSpace rule joins a
//! whitespace run into ONE segment, a boundary-rule artifact that makes
//! a pure delimiter run look like a word, and a fingerprint votes over
//! lexical content, so whitespace-only text has NO tokens and
//! fingerprints 0 (pinned). Punctuation-only segments ARE tokens:
//! deterministic, preserved by small edits, and excluding them would be a
//! lexicon judgment the segmentation tables do not make.
//!
//! # Reading the distance
//!
//! 0 bits means an identical token MULTISET, a coarser equality than the
//! exact gates' byte equality (same words in any order and any
//! whitespace). The near-duplicate band, measured by the battery below
//! and pinned at both scales: a 95-word base's every single-word
//! swap/drop/insert position moved its fingerprint at most 4 bits (the
//! document-scale band the fuzzy gate is for; the literature's
//! conventional "0-2 bits ≈ near-duplicate" guidance is the same
//! statement with a tighter battery), while the IDENTICAL edits over
//! 8-12-word sentences moved up to 14 bits (vote margins scale like
//! sqrt(token count), so short texts have thin margins and every edit
//! flips more bits). Unrelated sentence pairs sat at least 23 bits apart
//! (measured min, pinned). HONEST CAVEAT: thresholds are corpus-dependent.
//! The distance a "same document, small edit" pair sits at scales with
//! document length, and the unrelated floor depends on vocabulary overlap,
//! so there is NO universal cutoff; calibrate per deployment against
//! known near-dup and known-far pairs. Hamming distance itself is a
//! one-liner at the call site, `(a ^ b).bit_count()` in Python, which
//! is why this core returns the raw u64 (the pyo3 layer marshals it as a
//! Python int) and ships no redundant distance function.

use unicode_segmentation::UnicodeSegmentation;

/// The fingerprint width's arithmetic: FNV-1a at that width plus the bit
/// access the vote needs. Implemented for `u64` (the default spelling) and
/// `u128` (the wide spelling, twice the bit positions, for corpora whose
/// near-dup bands overlap at 64 bits); a future width is one impl away.
trait FnvBits: Copy + Default {
    const BITS: usize;
    const OFFSET_BASIS: Self;
    const PRIME: Self;
    fn xor_byte(self, byte: u8) -> Self;
    fn wrapping_mul_prime(self) -> Self;
    fn bit(self, index: usize) -> bool;
    fn with_bit(self, index: usize) -> Self;
    // Test-only: the `hamming_bits` helper's width-generic Hamming
    // distance. Not used by the production vote (`fingerprint`), which is
    // why these are gated rather than left to trip the dead-code lint on a
    // non-test build.
    #[cfg(test)]
    fn xor(self, other: Self) -> Self;
    #[cfg(test)]
    fn count_ones(self) -> u32;
}

impl FnvBits for u64 {
    #[cfg(test)]
    fn xor(self, other: u64) -> u64 {
        self ^ other
    }
    #[cfg(test)]
    fn count_ones(self) -> u32 {
        u64::count_ones(self)
    }
    const BITS: usize = 64;
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    fn xor_byte(self, byte: u8) -> u64 {
        self ^ u64::from(byte)
    }
    fn wrapping_mul_prime(self) -> u64 {
        self.wrapping_mul(Self::PRIME)
    }
    fn bit(self, index: usize) -> bool {
        (self >> index) & 1 == 1
    }
    fn with_bit(self, index: usize) -> u64 {
        self | (1 << index)
    }
}

impl FnvBits for u128 {
    #[cfg(test)]
    fn xor(self, other: u128) -> u128 {
        self ^ other
    }
    #[cfg(test)]
    fn count_ones(self) -> u32 {
        u128::count_ones(self)
    }
    const BITS: usize = 128;
    const OFFSET_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    fn xor_byte(self, byte: u8) -> u128 {
        self ^ u128::from(byte)
    }
    fn wrapping_mul_prime(self) -> u128 {
        self.wrapping_mul(Self::PRIME)
    }
    fn bit(self, index: usize) -> bool {
        (self >> index) & 1 == 1
    }
    fn with_bit(self, index: usize) -> u128 {
        self | (1 << index)
    }
}

/// FNV-1a over `token`'s UTF-8 bytes at width `B`, the deterministic
/// voting hash (see the module doc for why not `DefaultHasher`). The
/// offset basis and prime are the fixed FNV-1a constants at that width;
/// the multiply wraps at the width's power of two, exactly as the
/// reference algorithm specifies.
fn fnv1a<B: FnvBits>(token: &str) -> B {
    let mut hash = B::OFFSET_BASIS;
    for &byte in token.as_bytes() {
        hash = hash.xor_byte(byte).wrapping_mul_prime();
    }
    hash
}

/// The tokens the fingerprints vote over: the UAX #29 word segments of
/// `text` (`word_bounds`' walk, its `split_word_bounds` sibling), skipping
/// whitespace-only segments (the module doc's one tokenizer rule).
fn word_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split_word_bounds()
        .filter(|segment| !segment.chars().all(char::is_whitespace))
}

/// The width-generic vote every spelling shares: for each of the `B::BITS`
/// positions, +1 per token whose hash has that bit set, −1 per token whose
/// hash does not; the fingerprint bit is 1 iff the sum is positive (ties,
/// including the zero-token case, pin to 0).
fn fingerprint<B: FnvBits>(text: &str) -> B {
    // A stack buffer at the widest implemented width, sliced to B::BITS:
    // an associated const cannot size an array directly, and this keeps
    // the vote pass allocation-free. A future width beyond 128 bumps the
    // cap alongside its impl.
    let mut buffer = [0i64; 128];
    let votes = &mut buffer[..B::BITS];
    for token in word_tokens(text) {
        let hash = fnv1a::<B>(token);
        for (bit, vote) in votes.iter_mut().enumerate() {
            *vote += if hash.bit(bit) { 1 } else { -1 };
        }
    }
    let mut fingerprint = B::default();
    for (bit, &vote) in votes.iter().enumerate() {
        if vote > 0 {
            fingerprint = fingerprint.with_bit(bit);
        }
    }
    fingerprint
}

/// The 64-bit simhash fingerprint of `text`: FNV-1a each UAX #29 word
/// token, then vote. For each of the 64 bit positions, +1 per token
/// whose hash has that bit set, −1 per token whose hash does not; the
/// fingerprint bit is 1 iff the sum is positive (ties, including the
/// zero-token case, pin to 0). Empty text or a text with no word tokens
/// (e.g. whitespace-only) fingerprints 0. Deterministic across processes,
/// versions, and machines (FNV-1a; see the module doc); word order does
/// not matter (a bag-of-words vote; pinned by the tests below).
pub fn simhash64(text: &str) -> u64 {
    fingerprint::<u64>(text)
}

/// The 128-bit spelling of [`simhash64`]: the same tokens, the same vote,
/// FNV-1a at 128 bits, twice the bit positions. What that buys, measured
/// by the same battery and pinned: the unrelated floor widens from 23 bits
/// at 64 bits to 40 at 128, while the near-dup bands grow only sublinearly
/// (document scale: 3 bits worst vs the 64-bit 4; sentence scale: 20 vs
/// 14). The near-dup band does NOT scale with the width the way the
/// unrelated floor does. The win is the SEPARATION between the near-dup
/// band and the unrelated floor, the property a corpus whose 64-bit bands
/// overlap needs. The calibration caveat carries over unchanged: thresholds are
/// corpus-dependent. Same contract
/// otherwise: deterministic across processes (FNV-1a), order-invariant
/// (a bag-of-words vote), empty/whitespace-only → 0, `(a ^ b).bit_count()`
/// at the call site.
pub fn simhash128(text: &str) -> u128 {
    fingerprint::<u128>(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hamming(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    /// The naive spec spelling the incremental core is differentially
    /// pinned against: collect the tokens, hash them, and vote BIT-MAJOR
    /// (one pass over the token list per bit) instead of the core's
    /// token-major one-pass form. Same spec, opposite loop nesting; it
    /// guards the incremental implementation, not the spec. The generic
    /// spelling covers both widths.
    fn naive_simhash<B: FnvBits>(text: &str) -> B {
        let tokens: Vec<B> = word_tokens(text).map(fnv1a::<B>).collect();
        let mut fingerprint = B::default();
        for bit in 0..B::BITS {
            let mut sum = 0i64;
            for &hash in &tokens {
                sum += if hash.bit(bit) { 1 } else { -1 };
            }
            if sum > 0 {
                fingerprint = fingerprint.with_bit(bit);
            }
        }
        fingerprint
    }

    #[test]
    fn same_text_fingerprints_identically_and_word_order_does_not_matter() {
        // Determinism: the same text (a fresh, content-equal copy included,
        // to check the cross-run/cross-object stability the FNV choice
        // buys) yields the same fingerprint.
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "Hello, world! One. Two.",
            "caf\u{e9} \u{6771}\u{4eac}\u{3002} \u{5927}\u{962a}\u{3002}",
            "a\r\nb c\r d\ne",
            "",
        ];
        for text in texts {
            assert_eq!(simhash64(text), simhash64(text), "drift on {text:?}");
            let fresh_copy = text.to_owned();
            assert_eq!(
                simhash64(text),
                simhash64(&fresh_copy),
                "object identity leaked into the fingerprint for {text:?}"
            );
        }
        // The bag-of-words property, pinned as EQUALITY (a permutation
        // preserves the token multiset exactly, so the vote is identical):
        // reversed word order, and a permutation that moves punctuation
        // tokens with their words, both fingerprint the same.
        assert_eq!(
            simhash64("one two three four five six"),
            simhash64("six five four three two one")
        );
        assert_eq!(simhash64("Hello, world!"), simhash64("world! Hello,"));
        // Frequency is part of the bag: a fourth "the" is a DIFFERENT
        // multiset, and on this row the changed margins move the
        // fingerprint, the same thin-margin scale-dependence the
        // near-dup battery measures (a duplicate's vote margin does NOT
        // always absorb one occurrence; this deterministic row pins that
        // it does not here). Near-equal is not equal.
        assert_ne!(
            simhash64("the cat sat on the mat with the hat"),
            simhash64("the cat sat on the mat with the hat the")
        );
    }

    #[test]
    fn different_texts_get_different_fingerprints() {
        // Spot rows: content-word differences move the fingerprint. These
        // are deterministic literals (FNV), so inequality here is exactly
        // reproducible, not probabilistic.
        let fox = simhash64("the quick brown fox jumps over the lazy dog");
        assert_ne!(fox, simhash64("the slow brown fox jumps over the lazy dog"));
        assert_ne!(fox, simhash64("pack my box with five dozen liquor jugs"));
        assert_ne!(simhash64("hello world"), simhash64("goodbye world"));
        assert_ne!(simhash64("hello"), simhash64("hello there"));
    }

    /// The generated single-word-edit battery over one base text: every
    /// swap position (word i -> "silently"), every drop position, every
    /// insert position ("gently" before word i), returning the WORST
    /// Hamming distance seen. The bound this measures is scale-dependent
    /// (thin vote margins at low token counts: see the near-dup test), so
    /// the battery runs at both sentence and document scale. The generic
    /// spelling measures either width; the 128-bit anchors are measured
    /// with the same battery and pinned in their own test.
    fn worst_single_edit_distance<B: FnvBits>(base: &str) -> u32 {
        let words: Vec<&str> = base.split_whitespace().collect();
        let base_fp = fingerprint::<B>(base);
        let mut worst = 0u32;
        for i in 0..words.len() {
            let swap = {
                let mut w = words.clone();
                w[i] = "silently";
                w.join(" ")
            };
            let drop = {
                let mut w = words.clone();
                w.remove(i);
                w.join(" ")
            };
            let insert = {
                let mut w = words.clone();
                w.insert(i, "gently");
                w.join(" ")
            };
            for variant in [&swap, &drop, &insert] {
                worst = worst.max(hamming_bits(base_fp, fingerprint::<B>(variant)));
            }
        }
        worst
    }

    /// Width-generic Hamming distance, the test-side spelling of the
    /// `(a ^ b).bit_count()` one-liner the API hands to Python.
    fn hamming_bits<B: FnvBits>(a: B, b: B) -> u32 {
        a.xor(b).count_ones()
    }

    #[test]
    fn near_duplicate_edits_stay_within_the_measured_hamming_bounds() {
        // DOCUMENT scale, the shape the fuzzy gate is for: a 95-word
        // base, every single-word swap/drop/insert position. Measured max
        // 4 bits, pinned at 4 (a deterministic battery, so the pin is
        // exactly reproducible); the module doc's near-duplicate band is
        // THIS scale's statement.
        let document = "the old lighthouse keeper walked down the stone \
                        steps every morning before the sun rose over the \
                        harbor and he checked the lamp and the wicks and \
                        the glass for cracks that the winter storms might \
                        have left behind because a light that fails on a \
                        dark coast is a shipwreck waiting and he had kept \
                        this light for forty years through two wars and \
                        one great flood and he knew the sea the way a \
                        farmer knows his fields every rock and every \
                        current and every wind that bends the pines above \
                        the cliff";
        assert_eq!(worst_single_edit_distance::<u64>(document), 4);
        // SENTENCE scale, the other face of the same property:
        // the IDENTICAL edits over 8-12-word bases moved up to 14 bits.
        // Each bit's vote sum has margins on the order of sqrt(token
        // count), so short texts have thin margins and the same two-vote
        // change flips more bits. This is why the near-dup band is stated
        // at document scale and why the calibration caveat is not
        // boilerplate: at sentence scale the near-dup max (14, pinned
        // here) sits only 9 bits under the unrelated floor (23, pinned by
        // the far-apart test), separable, but a margin no deployer
        // should lift to document scale unchanged.
        let sentences = [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "summer rain falls quietly on the empty garden wall",
            "we hold these truths to be self evident that all people are equal",
        ];
        let worst = sentences
            .iter()
            .map(|base| worst_single_edit_distance::<u64>(base))
            .max()
            .expect("the sentence battery is non-empty");
        assert_eq!(worst, 14);
    }

    #[test]
    fn near_duplicate_edits_stay_within_the_measured_hamming_bounds_u128() {
        // The 128-bit counterpart of the u64 battery above, over the same
        // document and sentence bases: measured max 3 bits at document
        // scale and 20 bits at sentence scale, both pinned here, the exact
        // figures the module doc and the `simhash128` binding doc cite.
        let document = "the old lighthouse keeper walked down the stone \
                        steps every morning before the sun rose over the \
                        harbor and he checked the lamp and the wicks and \
                        the glass for cracks that the winter storms might \
                        have left behind because a light that fails on a \
                        dark coast is a shipwreck waiting and he had kept \
                        this light for forty years through two wars and \
                        one great flood and he knew the sea the way a \
                        farmer knows his fields every rock and every \
                        current and every wind that bends the pines above \
                        the cliff";
        assert_eq!(worst_single_edit_distance::<u128>(document), 3);
        let sentences = [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "summer rain falls quietly on the empty garden wall",
            "we hold these truths to be self evident that all people are equal",
        ];
        let worst = sentences
            .iter()
            .map(|base| worst_single_edit_distance::<u128>(base))
            .max()
            .expect("the sentence battery is non-empty");
        assert_eq!(worst, 20);
    }

    #[test]
    fn unrelated_texts_sit_far_apart_u128() {
        // The 128-bit counterpart of `unrelated_texts_sit_far_apart`, same
        // six vocabulary-disjoint sentences, all 15 pairs. Measured min 40
        // bits, pinned here: the width does widen the unrelated floor
        // (40 vs the 64-bit 23), just not by a clean doubling.
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "summer rain falls quietly on the empty garden wall",
            "we hold these truths to be self evident that all people are equal",
            "lorem ipsum dolor sit amet consectetur adipiscing elit sed do",
            "compiler backends schedule instructions over directed acyclic graphs",
        ];
        let mut min_seen = 128u32;
        for (i, a) in texts.iter().enumerate() {
            for b in &texts[i + 1..] {
                min_seen =
                    min_seen.min(hamming_bits(fingerprint::<u128>(a), fingerprint::<u128>(b)));
            }
        }
        assert_eq!(min_seen, 40);
    }

    #[test]
    fn unrelated_texts_sit_far_apart() {
        // The far side of the same calibration: vocabulary-disjoint
        // sentences, all 15 pairs. Measured min 23 bits, pinned at 23;
        // the >= 16 floor is the interpretation band, comfortably above
        // the document-scale near-dup max of 4, with the sentence-scale
        // tightness (14 vs 23) named in the near-dup test.
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "pack my box with five dozen liquor jugs",
            "summer rain falls quietly on the empty garden wall",
            "we hold these truths to be self evident that all people are equal",
            "lorem ipsum dolor sit amet consectetur adipiscing elit sed do",
            "compiler backends schedule instructions over directed acyclic graphs",
        ];
        let mut min_seen = 64u32;
        for (i, a) in texts.iter().enumerate() {
            for b in &texts[i + 1..] {
                min_seen = min_seen.min(hamming(simhash64(a), simhash64(b)));
            }
        }
        assert!(min_seen >= 16, "unrelated floor too low: {min_seen}");
        assert_eq!(min_seen, 23);
    }

    #[test]
    fn degenerate_inputs_pin_zero_and_the_vote_of_one_shape() {
        // Empty text: no tokens, all votes 0, no bit is > 0, so 0.
        assert_eq!(simhash64(""), 0);
        // Whitespace-only: the WSegSpace artifact segment is skipped, so
        // there are no tokens, hence 0. The mixed-run row (tabs, newlines,
        // the WB3-joined CRLF, and NBSP, all `char::is_whitespace`)
        // exercises the skip across segment shapes, not just the
        // single-space run.
        assert_eq!(simhash64("   "), 0);
        assert_eq!(simhash64("\t\n \r\n\u{a0}"), 0);
        // Single word: the vote of one. Every set bit of the token hash
        // votes +1 (bit 1), every clear bit votes -1 (bit 0), so the
        // fingerprint IS the token's FNV-1a hash, bit for bit. Derived
        // from the vote-of-one shape, then pinned as the literal.
        assert_eq!(simhash64("hello"), fnv1a::<u64>("hello"));
        assert_eq!(simhash64("hello"), 0xa430d84680aabd0b);
        assert_eq!(simhash128("hello"), fnv1a::<u128>("hello"));
        assert_eq!(simhash128("   "), 0);
    }

    #[test]
    fn agrees_with_the_naive_reference_over_an_exhaustive_battery() {
        // Every string over the alphabet {a, b, space} up to length 5
        // (363 strings; the space exercises the whitespace skip and the
        // WSegSpace joining at every position), plus the crate's tricky
        // non-ASCII rows (CRLF, ZWJ emoji, regional indicators, Hangul
        // jamo, the SARA AM spacing mark, CJK): the incremental core and
        // the naive bit-major reference must agree everywhere.
        let mut battery: Vec<String> = Vec::new();
        let alphabet = ["a", "b", " "];
        let mut frontier: Vec<String> = vec![String::new()];
        for _ in 0..5 {
            let mut next = Vec::new();
            for prefix in &frontier {
                for c in alphabet {
                    let mut s = prefix.clone();
                    s.push_str(c);
                    next.push(s.clone());
                    battery.push(s);
                }
            }
            frontier = next;
        }
        battery.extend([
            "a\r\nb".to_string(),
            "\u{1f469}\u{200d}\u{1f52c} \u{30c6}\u{30b9}\u{30c8}".to_string(),
            "\u{1f1fa}\u{1f1f8}\u{1f1fa}".to_string(),
            "\u{1100}\u{1161}\u{11a8} \u{e0}\u{30d}".to_string(),
            "\u{0e33}\u{0eb3} caf\u{e9}".to_string(),
            "Hello, world! One. Two.".to_string(),
            "   \t  ".to_string(),
        ]);
        for text in &battery {
            assert_eq!(
                simhash64(text),
                naive_simhash::<u64>(text),
                "core/reference disagreement for {text:?}"
            );
            assert_eq!(
                simhash128(text),
                naive_simhash::<u128>(text),
                "core/reference disagreement for {text:?}"
            );
        }
    }

    #[test]
    fn exact_gate_compatibility_identical_inputs_fingerprint_identically() {
        // The finalize/merkle compatibility row: the exact gate's
        // precondition (a == b, byte-identical content) implies equal
        // simhash fingerprints AND equal merkle roots: both gates agree
        // on identical input, which is what lets one store serve both.
        // Near-dup pairs (roots differ, fingerprints close) are the fuzzy
        // gate's own territory and are pinned by the battery tests above.
        let texts = [
            "",
            "hello",
            "Hello, world!",
            "the quick brown fox jumps over the lazy dog",
            "caf\u{e9} \u{6771}\u{4eac}\u{3002}",
        ];
        for a in texts {
            for b in texts {
                if a == b {
                    assert_eq!(simhash64(a), simhash64(b));
                    assert_eq!(
                        crate::merkle_impl::merkle_root(&[a.as_bytes()]),
                        crate::merkle_impl::merkle_root(&[b.as_bytes()])
                    );
                }
            }
        }
    }
}
