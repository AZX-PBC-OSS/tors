//! `tors::documents_impl::to_markdown_with` — the extraction core the
//! tors-documents payload wheel wraps, with the whole engine tree
//! (pdf_oxide, anydoc, office_oxide, html-to-markdown-rs) behind the
//! crate's `documents` feature — never panics on arbitrary bytes. Err is
//! the surface's job on hostile input (unknown format, input-ceiling
//! refusals, encrypted-PDF refusals); only a panic or a crash is a bug.
//! No name hint and no explicit format: resolution rests on the content
//! markers alone, the widest engine surface one target can reach in a
//! single call.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::documents_impl::{Backend, ConvertOptions, to_markdown_with};

fuzz_target!(|data: &[u8]| {
    // Err is fine (see the header) — nothing further to check; the output
    // is a String, valid UTF-8 by construction.
    let _ = to_markdown_with(
        data.to_vec(),
        None,
        None,
        Backend::Auto,
        None,
        ConvertOptions::default(),
    );
});
