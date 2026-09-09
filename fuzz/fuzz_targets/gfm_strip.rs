//! `tors::gfm_strip_impl::strip` — the markdown→plain-text stripper
//! behind `to_text`'s normalizer pass (documents_impl routes and
//! converts first, then strips) — never panics on arbitrary strings.
//! `from_utf8_lossy` maps raw fuzzer bytes onto every possible String,
//! and `unescape_entities: false` is the spelling three of the four
//! engines' text lanes run under (pdf_oxide, html-to-markdown-rs,
//! office_oxide; anydoc's is the `true` lane — see strip's own docs).

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let s = String::from_utf8_lossy(data);
    let _ = tors::gfm_strip_impl::strip(&s, false);
});
