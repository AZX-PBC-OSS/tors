use pyo3::prelude::*;

use crate::phonetic_impl;

/// `tors.soundex(text: str) -> str`: the classic Soundex phonetic code (a
/// letter followed by three digits, e.g. `"Robert"` and `"Rupert"` both
/// encode to `"R163"`), via `rphonetic`'s port of Apache Commons Codec's
/// implementation. A 1918-patent-era, ENGLISH/Latin-script-oriented
/// heuristic, not a general Unicode phonetic algorithm; input is
/// pre-filtered to ASCII letters before encoding (accents/non-Latin
/// characters are dropped, not encoded), which also works around a real,
/// verified panic in the underlying crate on ordinary accented names.
/// See `src/phonetic_impl.rs`'s module docs. Groups phonetically-similar
/// names/words for name-matching and dedup pipelines, typically alongside
/// `levenshtein`/`jaro_winkler` distance scoring rather than instead of
/// it. Empty input (or input with no ASCII letters) → `""`.
///
/// GIL model: the whole encode under `py.detach`; a single `str` return
/// (no marshalling class).
#[pyfunction]
pub fn soundex(py: Python<'_>, text: &str) -> String {
    py.detach(|| phonetic_impl::soundex(text))
}

/// `tors.metaphone(text: str) -> str`: the Double Metaphone PRIMARY code
/// (Lawrence Philips' 2000 successor to classic Metaphone, e.g.
/// `"jumped"` → `"JMPT"`), via `rphonetic`. The same
/// ENGLISH/Latin-script-oriented scope, ASCII-letters-only pre-filter, and
/// upstream-panic-avoidance note as `soundex` applies. Double Metaphone
/// can also produce an ALTERNATE code for words with two plausible
/// pronunciations; only the primary code is exposed here (a compatible,
/// additive future addition, not exposed because no consumer has asked
/// for it yet). Empty input (or input with no ASCII letters) → `""`.
///
/// GIL model: the whole encode under `py.detach`; a single `str` return
/// (no marshalling class).
#[pyfunction]
pub fn metaphone(py: Python<'_>, text: &str) -> String {
    py.detach(|| phonetic_impl::metaphone(text))
}

/// `tors.double_metaphone(text: str) -> tuple[str, str]`: the Double
/// Metaphone PRIMARY and ALTERNATE codes as `(primary, alternate)` (the
/// full dual-key form of what `tors.metaphone` returns, e.g.
/// `"jumped"` → `("JMPT", "AMPT")`), via `rphonetic`. The alternate key
/// is the algorithm's whole point: for names readable two ways
/// (Germanic/Slavic vs Anglicized) it carries the second pronunciation,
/// so a name-matching pipeline scores a match when EITHER key of two
/// names agrees; for words with one pronunciation the two elements are
/// equal. The same ENGLISH/Latin-script-oriented scope,
/// ASCII-letters-only pre-filter, and upstream-panic-avoidance note as
/// `soundex` applies (see `src/phonetic_impl.rs`). Empty input (or
/// input with no ASCII letters) → `("", "")`.
///
/// GIL model: the whole encode under `py.detach`; a 2-tuple of `str`
/// returns (no marshalling class).
#[pyfunction]
pub fn double_metaphone(py: Python<'_>, text: &str) -> (String, String) {
    py.detach(|| phonetic_impl::double_metaphone(text))
}

/// `tors.nysiis(text: str) -> str`: the NYSIIS code (New York State
/// Identification and Intelligence System, 1970; strict variant, codes
/// capped at 6 characters, e.g. `"Washington"` → `"WASANG"`), via
/// `rphonetic`'s port of Apache Commons Codec. A Soundex successor with
/// better first-letter and vowel handling for name matching. The same
/// ENGLISH/Latin-script-oriented scope, ASCII-letters-only pre-filter,
/// and upstream-panic-avoidance note as `soundex` applies (see
/// `src/phonetic_impl.rs`; the filter additionally keeps NYSIIS keys
/// pure ASCII, since the crate's own clean step would let accented
/// letters through into the code). Empty input (or input with no ASCII
/// letters) → `""`.
///
/// GIL model: the whole encode under `py.detach`; a single `str` return
/// (no marshalling class).
#[pyfunction]
pub fn nysiis(py: Python<'_>, text: &str) -> String {
    py.detach(|| phonetic_impl::nysiis(text))
}

/// `tors.daitch_mokotoff(text: str) -> list[str]`: the Daitch-Mokotoff
/// Soundex codes (1985; 6-digit codes via `rphonetic`'s port of Apache
/// Commons Codec with branching enabled, e.g. `"Peters"` →
/// `["734000", "739400"]`). The standard code of Jewish-genealogy
/// surname matching, designed for the Central/Eastern European
/// surnames classic Soundex conflates. Returns a LIST because the rule
/// table branches on ambiguous transliterations (one name can encode
/// to several codes; a single code per name is commons-codec's
/// non-branching convenience, not the algorithm): match two names when
/// their code lists intersect. Each code is padded to 6 digits, so
/// input with no encodable letters yields `["000000"]` (not `""`).
/// The same ENGLISH/Latin-script-oriented scope and ASCII-letters-only
/// pre-filter as `soundex` applies (see `src/phonetic_impl.rs`).
///
/// GIL model: the whole encode under `py.detach`; a `list` of `str`
/// returns (no marshalling class).
#[pyfunction]
pub fn daitch_mokotoff(py: Python<'_>, text: &str) -> Vec<String> {
    py.detach(|| phonetic_impl::daitch_mokotoff(text))
}

/// `tors.refined_soundex(text: str) -> str`: a Soundex variant with a
/// finer-grained letter-to-digit mapping than classic Soundex (more
/// consonant classes distinguished, at the cost of a longer, uncapped
/// code rather than Soundex's fixed 1-letter-plus-3-digit shape), via
/// `rphonetic`'s port of Apache Commons Codec. A genuinely distinct
/// mapping table, not a formatting variant of `tors.soundex`. The same
/// ENGLISH/Latin-script-oriented scope, ASCII-letters-only pre-filter,
/// and upstream-panic-avoidance note as `soundex` applies (see
/// `src/phonetic_impl.rs`). Empty input (or input with no ASCII
/// letters) → `""`.
///
/// GIL model: the whole encode under `py.detach`; a single `str` return
/// (no marshalling class).
#[pyfunction]
pub fn refined_soundex(py: Python<'_>, text: &str) -> String {
    py.detach(|| phonetic_impl::refined_soundex(text))
}
