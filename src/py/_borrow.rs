//! The pyo3 layer's shared argument walks and argument validators, the
//! code every list/dict-taking binding repeated per file until it had one
//! home: the GIL-held collect-handles-then-borrow walks (`&str` lists,
//! `&[u8]` lists, `&str` dict pairs), the `[0.0, 1.0]` closed-interval
//! check, and the `TimeoutError` construction. The leading underscore
//! marks the module private to the py layer: nothing here is a binding,
//! so nothing here belongs in the crate's public surface or in `lib.rs`'s
//! cross-feature helper set.
//!
//! The walks take a `run` closure rather than returning the borrows:
//! pyo3's `&str`/`&[u8]` extraction hands back a borrow tied to the
//! HANDLE it was extracted from (pyo3 0.29's
//! `impl<'a> FromPyObject<'a, '_> for &'a str`), so a walk that returned
//! `(handles, borrows-of-handles)` would be returning a value that
//! references data it also owns, the self-referential-return shape safe
//! Rust cannot express. Running `run` inside the walk's scope solves
//! that and pays a safety dividend: the handles are alive across
//! everything `run` does, any `py.detach` included, BY CONSTRUCTION
//! rather than by call-site discipline.

use pyo3::exceptions::{PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

/// The empty-entry contract of a `&str` list walk: a pattern list refuses
/// an empty entry (`ValueError("empty pattern")`: an empty pattern would
/// match at every position and has no leftmost-longest meaning, the
/// `find_patterns` contract), a candidate/corpus/texts list allows one
/// (an empty candidate scores `0.0` unless the query is empty too,
/// difflib's own behavior; an empty document is simply a document).
pub(crate) enum EmptyPolicy {
    /// An empty entry raises `ValueError("empty pattern")`.
    Refuse,
    /// An empty entry is legal and walks through like any other.
    Allow,
}

/// The GIL-held `&str` list walk shared by `find_patterns`' pattern list,
/// `get_close_matches`' candidate list, `tf_idf`/`bm25_rank`'s corpus, and
/// `apply_pipeline`'s texts: collect the item handles, borrow each
/// entry's UTF-8 (the standard str-in class), optionally refuse an empty
/// entry, then run `run` over the handles and the borrows while both are
/// in scope.
///
/// Soundness, the one place this story lives (each call site keeps a
/// one-line pointer here): the borrows `run` receives point into the str
/// objects' immutable UTF-8 buffers (for a non-ASCII object, the UTF-8
/// copy pyo3 caches on the object at first extraction), valid for as
/// long as the objects are referenced. The caller's list argument holds
/// the objects and this walk's handle vector re-pins that for the
/// compiler, so the borrows are readable everywhere inside `run`,
/// including across a `py.detach` it performs. This is the same
/// soundness argument as every str-in borrow in this crate.
///
/// `run` also receives the handles themselves (the `[Bound]` slice)
/// because one caller needs them after its detach: `get_close_matches`
/// returns the ORIGINAL candidate objects, selected by index, so it must
/// reach the handles on the marshalling side of the scan.
pub(crate) fn borrow_str_list<R>(
    list: &Bound<'_, PyList>,
    empty: EmptyPolicy,
    run: impl FnOnce(&[Bound<'_, PyAny>], &[&str]) -> PyResult<R>,
) -> PyResult<R> {
    let items: Vec<_> = list.iter().collect();
    let mut borrowed: Vec<&str> = Vec::with_capacity(items.len());
    for item in &items {
        let s = item.extract::<&str>()?;
        if matches!(empty, EmptyPolicy::Refuse) && s.is_empty() {
            return Err(PyValueError::new_err("empty pattern"));
        }
        borrowed.push(s);
    }
    run(&items, &borrowed)
}

/// The `&[u8]` list twin of [`borrow_str_list`], `merkle_root`'s and
/// `merkle_diff`'s chunk lists: the same collect-handles-then-borrow walk
/// with the same borrows-alive-inside-`run` contract, over zero-copy
/// `&[u8]` borrows of the bytes objects' immutable buffers (pyo3's
/// `&[u8]` extraction has no cached-copy case at all, every input
/// borrows directly). No empty-entry refusal exists on this twin: an
/// empty chunk is a legal leaf (it hashes like any other length, and
/// `merkle_diff` compares it positionally like any other), and no
/// caller of this walk needs the handles, so `run` takes the borrows
/// alone.
pub(crate) fn borrow_bytes_list<R>(
    list: &Bound<'_, PyList>,
    run: impl FnOnce(&[&[u8]]) -> PyResult<R>,
) -> PyResult<R> {
    let items: Vec<_> = list.iter().collect();
    let mut borrowed: Vec<&[u8]> = Vec::with_capacity(items.len());
    for item in &items {
        borrowed.push(item.extract::<&[u8]>()?);
    }
    run(&borrowed)
}

/// The `&str` dict-pair walk shared by `replace_many` and
/// `replace_many_masked`: the [`borrow_str_list`] shape over a dict's
/// (key, value) pairs, keys refused empty (`ValueError("empty pattern")`,
/// the same find_patterns contract an empty key would break: it would
/// match at every position), values extracted AFTER the key's empty
/// check so a non-`str` value under an empty key reports the empty-key
/// `ValueError` first, exactly the per-entry check order the inline walks
/// had. Same soundness story as [`borrow_str_list`]: the caller's dict
/// argument holds the objects, the walk's handle pairs re-pin them for
/// the compiler, and the borrows are readable everywhere inside `run`.
/// No caller needs the handles, so `run` takes the pairs alone.
pub(crate) fn borrow_dict_pairs<R>(
    dict: &Bound<'_, PyDict>,
    run: impl FnOnce(&[(&str, &str)]) -> PyResult<R>,
) -> PyResult<R> {
    let items: Vec<_> = dict.iter().collect();
    let mut pairs: Vec<(&str, &str)> = Vec::with_capacity(items.len());
    for (key, value) in &items {
        let old = key.extract::<&str>()?;
        if old.is_empty() {
            return Err(PyValueError::new_err("empty pattern"));
        }
        let new = value.extract::<&str>()?;
        pairs.push((old, new));
    }
    run(&pairs)
}

/// The closed `[0.0, 1.0]` interval check shared by `is_grounded`'s
/// `threshold` and `get_close_matches`' `cutoff`. The two call sites'
/// refusal messages differ on purpose, and `echo_value` selects the
/// shape so each site's exact message is preserved: difflib's own cutoff
/// message echoes the value (`"cutoff must be in [0.0, 1.0]: 1.5"`, the
/// stdlib parity contract), while tors's own threshold message does not.
/// Both spell the parameter name into the message so a caller knows
/// which argument to fix.
pub(crate) fn validate_unit_interval(name: &str, value: f64, echo_value: bool) -> PyResult<()> {
    if !(0.0..=1.0).contains(&value) {
        let mut message = format!("{name} must be in [0.0, 1.0]");
        if echo_value {
            message.push_str(&format!(": {value}"));
        }
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

/// The `TimeoutError` construction shared by every `deadline_ms`-bearing
/// binding (`diff_opcodes` and its line spelling, the three fuzzy
/// metrics, `similarity_ratio`, `get_close_matches`, `is_grounded`): the
/// deadline core's own message carried on the builtins `TimeoutError`
/// type the tests pin (`type(excinfo.value) is TimeoutError`). The
/// exception is always constructed AFTER the GIL is reacquired (nothing
/// raises from inside a detached region); the deadline cores are three
/// deliberately distinct nominal types with no shared trait, so the
/// message string, not the error value, is the shared currency, and this
/// one construction spelling replaces the two (`PyTimeoutError::new_err`
/// and `PyErr::new::<PyTimeoutError, _>`, the same exception either way)
/// that could otherwise drift apart.
pub(crate) fn timeout_err(message: String) -> PyErr {
    PyTimeoutError::new_err(message)
}
