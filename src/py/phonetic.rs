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
