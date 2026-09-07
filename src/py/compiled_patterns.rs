//! `tors.CompiledPatterns`: a pattern list compiled once, the
//! `re.compile()` answer to the per-call automaton build every free
//! search spelling pays.
//!
//! The cost being amortized: each of `find_patterns` /
//! `find_patterns_iter` / `count_matches` / `replace_many` /
//! `replace_many_masked` builds its aho-corasick automaton (and the
//! duplicate-id remap or first-value map) from the pattern list on EVERY
//! call, linear in the total pattern bytes. For a one-off call that is
//! the right shape; for a pipeline that runs the SAME fixed vocabulary
//! (a redaction list, a terminology rewrite table) over many texts or
//! many times, the build is a fixed cost the work keeps re-paying for an
//! automaton it already had, and at a document scale with a large
//! vocabulary it can dominate the scan itself. `CompiledPatterns` builds
//! the automaton and the remaps ONCE (the same build, the same engine
//! configuration, `search_impl::CompiledPatterns`), holds them behind one
//! `Arc`, and every call afterwards is the free function's scan classes
//! MINUS the build: the scans are the free functions' own
//! (`search_impl::scan_matches` / the count scan /
//! `scan_replace` / `scan_replace_masked`), driven over the held
//! automaton, so parity is by construction and is pinned by
//! tests/test_compiled_patterns.py re-running the free functions' own
//! batteries through a compiled fixture.
//!
//! The `CompiledLemmaDict` discipline, carried over whole: the compiled
//! object is immutable once built (every field is only ever read), so
//! sharing one across many calls and threads is sound with no
//! synchronization beyond the `Arc` refcount itself; and it is scoped
//! narrowly and explicitly opt-in, no silent identity-keyed cache inside
//! the pyo3 layer, because a caller mutating a list in place between
//! calls (same `id()`, different content) would then silently search a
//! stale vocabulary with no signal anything is wrong. The caller controls
//! exactly when a new one gets created, so there is no staleness
//! question to get wrong.
//!
//! The one contract the compiled replace spellings add, validated at
//! CALL time (values change per call; the automaton is the compiled
//! part): the `replacements` dict must key EXACTLY the compiled pattern
//! set, every pattern paired with a value and no other keys. A key the
//! automaton cannot match could never be honored (the free function
//! would have built it in), and a pattern with no value could never be
//! spliced, so both directions are refused with a `ValueError` naming
//! them; the exact set makes `cp.replace_many(text, m) ==
//! tors.replace_many(text, m)` an unconditional theorem. The empty
//! compiled list accepts only the empty dict (and then answers the input
//! itself, the identity lane).

use std::borrow::Cow;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::{Py, PyAny};

use crate::EagerIter;
use crate::py::_borrow::{EmptyPolicy, borrow_dict_pairs, borrow_str_list};
use crate::py::eager_iter_class;
use crate::search_impl;

eager_iter_class! {
    /// `CompiledPatterns.find_iter(text)`: the streaming spelling of the
    /// compiled find (`find_patterns_iter`'s design over a compiled
    /// fixture's matches: the whole scan under ONE detach at
    /// construction, one 3-tuple of ints per `__next__`,
    /// `__length_hint__` the remaining count), yielding the SAME
    /// `(start, end, pattern_index)` triples as `find`, in the same order
    /// (pinned to sequence-parity by the compiled gate battery).
    CompiledFindIter, (usize, usize, usize)
}

/// A pattern list compiled once. Immutable after construction (the
/// automaton and the remaps are only ever read, never mutated through
/// this handle), so sharing one `Arc` across many calls and threads is
/// sound with no synchronization beyond the refcount itself.
#[pyclass(name = "CompiledPatterns", frozen)]
pub struct CompiledPatterns {
    /// The compiled core: the automaton, the owned pattern strings, the
    /// find side's canonical-id remap, and the replace side's validation
    /// set (`search_impl::CompiledPatterns`), `Arc`-shared so every call
    /// is a refcount bump, not a rebuild.
    core: Arc<search_impl::CompiledPatterns>,
}

#[pymethods]
impl CompiledPatterns {
    /// `tors.CompiledPatterns(patterns)`: compiles `patterns` (exactly a
    /// `list` of `str`, the `find_patterns` argument contract: a tuple or
    /// a non-`str` entry raises `TypeError`, an empty pattern STRING
    /// raises `ValueError("empty pattern")`, lone surrogates raise
    /// `UnicodeEncodeError` at the str-in boundary) ONCE: the automaton
    /// build, the remaps, and the owned pattern copies, under one
    /// `py.detach`. An empty list is legal and compiles to a
    /// zero-pattern fixture whose scans find nothing (the free
    /// spellings' own empty-list answer). A build failure on engine
    /// limits maps to a `ValueError` carrying the engine's message, the
    /// free functions' own mapping.
    #[new]
    fn new(py: Python<'_>, patterns: &Bound<'_, PyList>) -> PyResult<Self> {
        // The shared pattern-list walk (`_borrow.rs`'s soundness story:
        // handles alive across the detach by construction), patterns
        // refused empty, then ONE detached build.
        let core = borrow_str_list(patterns, EmptyPolicy::Refuse, |_items, borrowed| {
            py.detach(|| search_impl::CompiledPatterns::build(borrowed))
                .map_err(|err| PyValueError::new_err(err.to_string()))
        })?;
        Ok(CompiledPatterns {
            core: Arc::new(core),
        })
    }

    /// The list's length, mirroring `len(patterns)` on the source list
    /// (duplicates included, the automaton's own id-space count): a cheap
    /// sanity check that needs no scan.
    fn __len__(&self) -> usize {
        self.core.pattern_count()
    }

    fn __repr__(&self) -> String {
        format!("CompiledPatterns({} patterns)", self.core.pattern_count())
    }

    /// `cp.find(text)`: `find_patterns(patterns, text)`'s exact answer
    /// over the held automaton: the same leftmost-longest,
    /// non-overlapping matches, the same PYTHON `str` INDEX offsets (the
    /// ASCII fast path and the byte→char conversion pass alike), the same
    /// first-index duplicate reporting. Argument contract:
    /// `find_patterns`' text side exactly (exactly `str`; lone surrogates
    /// `UnicodeEncodeError`).
    ///
    /// GIL model: the free function's classes minus the build: one
    /// `Arc`-clone of the compiled core (a refcount bump), the whole scan
    /// under one `py.detach`, then the O(matches) return marshalling, one
    /// 3-tuple of ints per match.
    fn find(&self, py: Python<'_>, text: &str) -> PyResult<Vec<(usize, usize, usize)>> {
        let core = self.core.clone();
        let matches = py.detach(|| core.find(text));
        Ok(matches
            .into_iter()
            .map(|m| (m.start, m.end, m.pattern))
            .collect())
    }

    /// `cp.count(text)`: `count_matches(patterns, text)`'s exact answer:
    /// the same scan, the matches counted, O(1) memory, a single int out
    /// (no marshalling class at all). Argument contract: the text side of
    /// `find`'s.
    ///
    /// GIL model: one `Arc`-clone, the whole scan under one `py.detach`,
    /// a single int return.
    fn count(&self, py: Python<'_>, text: &str) -> PyResult<usize> {
        let core = self.core.clone();
        Ok(py.detach(|| core.count(text)))
    }

    /// `cp.find_iter(text)`: `find_patterns_iter(patterns, text)`'s exact
    /// sequence over the held automaton: the whole scan fills an internal
    /// buffer under ONE detach at construction, then one 3-tuple of ints
    /// per `__next__` (µs-scale GIL holds), `__length_hint__` the
    /// remaining count, the streaming answer to `find`'s O(matches)
    /// list-marshalling caveat.
    fn find_iter(
        &self,
        py: Python<'_>,
        text: Bound<'_, PyString>,
    ) -> PyResult<Py<CompiledFindIter>> {
        let s = text.to_str()?;
        let core = self.core.clone();
        let items: Vec<(usize, usize, usize)> = py.detach(|| {
            core.find(s)
                .into_iter()
                .map(|m| (m.start, m.end, m.pattern))
                .collect()
        });
        Py::new(py, CompiledFindIter(EagerIter::new(py, text, items)))
    }

    /// `cp.replace_many(text, replacements)`: `tors.replace_many(text,
    /// replacements)`'s exact answer over the held automaton, with the
    /// SAME leftmost-longest, non-overlapping, never-rescanned splice and
    /// the SAME identity contract (`cp.replace_many(s, m) is s` exactly
    /// when `== s`), plus the one compiled-side contract: `replacements`
    /// (exactly a `dict` of `str -> str`, the free function's argument
    /// contract) must key EXACTLY the compiled pattern set, validated at
    /// CALL time (values change per call; the automaton is the compiled
    /// part): an unknown key or a pattern left without a value raises
    /// `ValueError` naming them, the exact set makes the answer equal to
    /// the free function's by construction.
    ///
    /// GIL model: the free function's classes minus the build: one
    /// GIL-held dict walk (the standard str-in borrow class over keys and
    /// values), then validation + scan + splice under one `py.detach`,
    /// then either the identity return (zero marshalling) or the
    /// O(output) string marshalling.
    fn replace_many(
        &self,
        py: Python<'_>,
        text: Bound<'_, PyString>,
        replacements: Bound<'_, PyDict>,
    ) -> PyResult<Py<PyAny>> {
        // The str-in boundary first (the crate's contract), then the
        // shared dict walk (`_borrow.rs`'s soundness story: handles alive
        // across the detach by construction), empty keys refused.
        // Validation is plain Rust (no Python API), so it runs inside the
        // detach with the scan; its failure crosses as the message string,
        // the same currency a build failure uses, and becomes the
        // ValueError after the GIL is reacquired.
        let s = text.to_str()?;
        let core = self.core.clone();
        let out = borrow_dict_pairs(&replacements, |pairs| {
            py.detach(|| match core.replace_values(pairs) {
                Ok(values) => Ok(core.replace_many(s, &values)),
                Err(mismatch) => Err(mismatch.message()),
            })
            .map_err(PyValueError::new_err)
        })?;
        match out {
            Cow::Borrowed(_) => Ok(text.into_any().unbind()),
            Cow::Owned(out) => Ok(out.into_pyobject(py)?.into_any().unbind()),
        }
    }

    /// `cp.replace_many_masked(text, replacements, mask="*")`: the
    /// LENGTH-PRESERVING spelling, `tors.replace_many_masked`'s exact
    /// answer over the held automaton: the same scan, each matched span
    /// replaced by the value truncated to the span's CHARACTER count or
    /// padded with `mask`, so every pre-computed offset stays valid.
    /// `mask` must be exactly one character (`ValueError` otherwise, the
    /// free function's own refusal); the replacements contract and the
    /// identity contract are `cp.replace_many`'s exactly.
    ///
    /// GIL model: `cp.replace_many`'s classes exactly, with the mask
    /// validated under the GIL before the walk.
    #[pyo3(signature = (text, replacements, mask = "*"))]
    fn replace_many_masked(
        &self,
        py: Python<'_>,
        text: Bound<'_, PyString>,
        replacements: Bound<'_, PyDict>,
        mask: &str,
    ) -> PyResult<Py<PyAny>> {
        // The str-in boundary first (the crate's contract), then the mask
        // check (the free function's own order: the mask is validated
        // before any dict handling).
        let s = text.to_str()?;
        let mut mask_char = mask.chars();
        let (Some(mask_char), None) = (mask_char.next(), mask_char.next()) else {
            return Err(PyValueError::new_err(
                "mask must be exactly one character (the length arithmetic requires it)",
            ));
        };
        let core = self.core.clone();
        let out = borrow_dict_pairs(&replacements, |pairs| {
            py.detach(|| match core.replace_values(pairs) {
                Ok(values) => Ok(core.replace_many_masked(s, &values, mask_char)),
                Err(mismatch) => Err(mismatch.message()),
            })
            .map_err(PyValueError::new_err)
        })?;
        match out {
            Cow::Borrowed(_) => Ok(text.into_any().unbind()),
            Cow::Owned(out) => Ok(out.into_pyobject(py)?.into_any().unbind()),
        }
    }
}
