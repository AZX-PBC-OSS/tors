//! The pyo3 binding layer, one submodule per feature area: the
//! `#[pyfunction]` wrappers that add the argument borrow, GIL release, and
//! return marshalling around each `*_impl` module's pure-Rust core. Split
//! out of `lib.rs` (which held every binding directly and had grown past
//! 1900 lines) so each feature's binding code sits next to its own
//! concerns rather than in one crate-wide file; `lib.rs` keeps only the
//! cross-feature helpers (`detached_transform`, `EagerIter`,
//! `validate_deadline_ms`, `parse_boundary`) and the `#[pymodule]`
//! registration. This layer's own cross-feature helpers live here:
//! `_borrow` (the shared list/dict argument walks and validators, private
//! to the py layer) and the `eager_iter_class!` macro (the shared
//! `__iter__`/`__next__`/`__length_hint__` trio over `EagerIter`).

mod _borrow;

pub mod bm25;
pub mod chunk;
pub mod codec;
pub mod compiled_patterns;
pub mod diff;
pub mod encoding;
pub mod fence;
pub mod forms;
pub mod fuzzy;
pub mod grounded;
pub mod html;
pub mod json_repair;
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

/// Emits one eager-iterator `#[pyclass]`: the Python-visible class name,
/// its per-class doc comment (the payload's `#[doc]` attribute lands on
/// the struct through pyo3's doc-attribute path, so the class `__doc__`
/// is exactly a hand-written doc comment's), and the item type, wrapped
/// over the shared `EagerIter` core: `__iter__` returning the iterator
/// itself, `__next__` delegating to the buffer cursor, and
/// `__length_hint__` reporting the remaining count. The trio is
/// identical for every `*_iter` surface (the class name is per-struct in
/// pyo3; the logic is the one core), so one declarative spelling keeps
/// the classes from drifting. The trailing semicolon is optional,
/// statement and block invocation styles alike.
macro_rules! eager_iter_class {
    ($(#[$doc:meta])* $name:ident, $item:ty $(;)?) => {
        $(#[$doc])*
        #[pyclass]
        pub struct $name(crate::EagerIter<$item>);

        #[pymethods]
        impl $name {
            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            fn __next__(&mut self) -> Option<$item> {
                self.0.next()
            }

            fn __length_hint__(&self) -> usize {
                self.0.remaining()
            }
        }
    };
}
pub(crate) use eager_iter_class;
