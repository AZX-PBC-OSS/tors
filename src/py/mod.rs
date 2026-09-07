//! The pyo3 binding layer, one submodule per feature area: the
//! `#[pyfunction]` wrappers that add the argument borrow, GIL release, and
//! return marshalling around each `*_impl` module's pure-Rust core. Split
//! out of `lib.rs` (which held every binding directly and had grown past
//! 1900 lines) so each feature's binding code sits next to its own
//! concerns rather than in one crate-wide file; `lib.rs` keeps only the
//! genuinely cross-feature helpers (`detached_transform`, `EagerIter`,
//! `validate_deadline_ms`, `parse_boundary`) and the `#[pymodule]`
//! registration.

pub mod bm25;
pub mod chunk;
pub mod codec;
pub mod diff;
pub mod encoding;
pub mod fence;
pub mod forms;
pub mod fuzzy;
pub mod grounded;
pub mod html;
pub mod lemma_dict;
pub mod merkle;
pub mod normalize;
pub mod phonetic;
pub mod pipeline;
pub mod search;
pub mod segmentation;
pub mod simhash;
pub mod tfidf;
pub mod truncate;
pub mod url;
