//! The pyo3 layer's shared argument walks and argument validators, the
//! code every list/dict-taking binding repeated per file until it had one
//! home: the GIL-held collect-handles-then-borrow walks (`&str` lists,
//! `&[u8]` lists, `&str` dict pairs), the single-argument str-in
//! conversion, the `[0.0, 1.0]` closed-interval check, the chunkers'
//! count/overlap triple, and the `TimeoutError` construction. The leading
//! underscore
//! marks the module private to the py layer: nothing here is a binding,
//! so nothing here belongs in the crate's public surface or in `lib.rs`'s
//! cross-feature helper set.
//!
//! The walks take a `run` closure rather than returning the borrows:
//! pyo3's `&str`/`&[u8]` extraction hands back a borrow tied to the
//! handle it was extracted from (pyo3 0.29's
//! `impl<'a> FromPyObject<'a, '_> for &'a str`), so a walk that returned
//! `(handles, borrows-of-handles)` would be returning a value that
//! references data it also owns, the self-referential-return shape safe
//! Rust cannot express. Running `run` inside the walk's scope solves
//! that and pays a safety dividend: the handles are alive across
//! everything `run` does, any `py.detach` included, by construction
//! rather than by call-site discipline.

use pyo3::exceptions::{PyTimeoutError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyInt, PyList, PySequence, PyString};

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
/// returns the original candidate objects, selected by index, so it must
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
/// match at every position), values extracted after the key's empty
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

/// The bounded list-parameter walk's cap: past this many yielded items
/// the walk aborts with a catchable `ValueError` (the
/// `borrow_str_sequence` discipline, `src/py/charset.rs`
/// MAX_BATCH_ITEMS — never trust a reported size, walk under a cap).
/// One order of magnitude looser than the honest population any current
/// caller's parameter carries (`scrub_pii`'s closed rule/family sets,
/// `chunk_hierarchical`'s separator hierarchy), so a legitimate call is
/// never near it: anything past 100_000 is not a miscounted batch, it
/// is the bomb itself. The message is deliberately generic (no cap
/// value): the bound is a DoS backstop, not a contract to advertise to
/// sequence authors.
pub(crate) const MAX_LIST_ITEMS: usize = 100_000;

/// The bounded manual list walk for `list`-taking bindings whose
/// parameters pyo3 extracts as `Vec<...>` (currently `chunk_hierarchical`'s
/// `separators=`, whose elements are `Option<String>`). It replaces the
/// `Vec<...>` parameter spelling precisely because that spelling let pyo3
/// size the Vec from the argument's `__len__` before iterating it — a
/// `Sequence` whose `__len__` lies (2**62) blew up `Vec::with_capacity`
/// as a `PanicException` (capacity overflow), which `except Exception`
/// cannot catch: the one uncatchable crash class on the pyo3 boundary
/// (the `pages=` range bomb's class, the same fix shape). The manual walk
/// never reads `__len__`, so the cap is the only bound it needs.
///
/// The per-item extraction is the caller's `push` closure (the module's
/// run-closure convention, the same reason [`borrow_str_list`] takes
/// `run`: the element types differ per call site, and pyo3 0.29's
/// `FromPyObject` lifetimes do not generalize over them here), called
/// once per yielded item BEFORE the cap check so extraction errors
/// propagate in the same per-item order pyo3's own iteration raised
/// them. Every refusal is byte-identical to the `Vec<...>` extraction
/// the walk replaced, pinned at each call site's test file: a bare
/// `str` is refused up front (pyo3's own `Vec` special case — `str`
/// satisfies the Sequence protocol and would silently validate its own
/// characters one by one); a non-`Sequence` object, a non-`str` item,
/// and a `__getitem__` that raises all surface the same `TypeError`/
/// propagated error pyo3's iteration raised; a `__len__` that lies LOW
/// changes nothing (the walk iterates, it never reserves) and yields
/// every item. The one behavior change is the bomb's: an unbounded (or
/// lying-huge) sequence now dies as a catchable `ValueError` at
/// [`MAX_LIST_ITEMS`] instead of an uncatchable `PanicException` inside
/// `Vec::with_capacity`.
///
/// DEDUP AT MERGE: `fix/bounded-dos-111-115` lands a private twin of
/// this walk as `bounded_str_list` in `src/py/pii.rs` (for `scrub_pii`/
/// `scrub_pii_report`'s `rules=`/`families=`, element type `String`) —
/// the two branches were built against the same issue (#112 class)
/// without seeing each other. When that branch merges, its copy moves
/// into this module (this function IS the shared home; its cap const is
/// this file's [`MAX_LIST_ITEMS`], its message bytes are identical) and
/// `pii.rs`'s call sites switch to it, so exactly one walk and one cap
/// survive.
pub(crate) fn bounded_str_list(
    function: &str,
    param: &str,
    items: &Bound<'_, PyAny>,
    mut push: impl FnMut(&Bound<'_, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    if items.is_instance_of::<PyString>() {
        // The refusal pyo3's own `Vec<...>` extraction made (its
        // message, kept verbatim): a bare str is the char-split footgun.
        return Err(PyTypeError::new_err("Can't extract `str` to `Vec`"));
    }
    let seq = items.cast::<PySequence>()?;
    let mut count = 0usize;
    for handle in seq.try_iter()? {
        let handle = handle?;
        push(&handle)?;
        count += 1;
        if count > MAX_LIST_ITEMS {
            return Err(PyValueError::new_err(format!(
                "{function}() {param} sequence yielded too many items: refusing an unbounded batch"
            )));
        }
    }
    Ok(())
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

/// The `from_py_with` extractor that gives an argument both halves of the
/// str-in contract at once: the `&str` conversion pyo3's wrapper would
/// perform for a `&str`-typed parameter (the standard str-in class: a
/// non-`str` object fails the `PyString` downcast, a lone-surrogate `str`
/// fails the UTF-8 borrow, byte-identical errors either way, because this
/// is that code path) and the handle a `Bound<'_, PyString>` parameter
/// carries, in one argument slot. The chunking `_iter` twins use it on
/// their `text` so their wrappers run exactly the conversion the list
/// spellings' `text: &str` parameters get, in the same argument-0 slot:
/// the streaming spelling must hold the handle (the `EagerIter` keep-alive)
/// but must not let its weaker no-conversion extraction reorder any other
/// argument's error ahead of the text's, the error-precedence contract
/// tests/test_chunk_text.py::TestErrorPrecedence pins, family-wide. The
/// returned handle's UTF-8 is already validated (and cached on the object)
/// by the time the body sees it, so the body's own `to_str` re-borrow
/// cannot fail.
pub(crate) fn convert_str_arg<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyString>> {
    obj.extract::<&str>()?;
    Ok(obj.cast::<PyString>()?.clone())
}

/// The `count >= 1` / `overlap >= 0` / `overlap < count` validation
/// triple shared by every chunking binding's window arguments:
/// `chunk_text`/`chunk_text_iter`/`chunk_hierarchical`'s `max_chars` and
/// the four segment-count list/`_iter` pairs'
/// `words_per_chunk`/`sentences_per_chunk`/`paragraphs_per_chunk`/
/// `lines_per_chunk`, with the call site's own parameter name passed in
/// as `count_name` so every message spells it (a caller knows which
/// argument to fix). The three checks are the no-empty-chunk floor, the
/// no-negative-overlap floor, and the forward-progress ceiling
/// (`overlap >= count` would give every chunk after the first a stride
/// of `count - overlap <= 0`, an infinite loop by construction), in the
/// order the eleven inline blocks this replaces all had, count first,
/// and because every caller reaches it only after all of its arguments
/// are converted (the `_iter` twins' wrappers included, via
/// [`convert_str_arg`]), that order is what both spellings of a pair
/// observe, the family-wide error-precedence contract
/// tests/test_chunk_text.py::TestErrorPrecedence pins.
pub(crate) fn validate_count_overlap(count_name: &str, count: i64, overlap: i64) -> PyResult<()> {
    if count < 1 {
        return Err(PyValueError::new_err(format!(
            "{count_name} must be >= 1, got {count}"
        )));
    }
    if overlap < 0 {
        return Err(PyValueError::new_err(format!(
            "overlap must be >= 0, got {overlap}"
        )));
    }
    if overlap >= count {
        return Err(PyValueError::new_err(format!(
            "overlap must be < {count_name}, got overlap={overlap}, {count_name}={count}"
        )));
    }
    Ok(())
}

/// The bounded manual walk behind every `rules=`-style `Option<Vec<String>>`
/// list parameter (`scrub_log_text`'s `rules=` today; `scrub_pii`'s
/// `rules=`/`families=` carry the same walk locally in `src/py/pii.rs` on
/// the `fix/bounded-dos-111-115` branch — the merge-time dedup obligation
/// is to fold that copy into this one, this file being the py layer's
/// shared-walk home): precisely because the `Option<Vec<String>>` spelling
/// let pyo3 size the Vec from the argument's `__len__` before iterating it
/// — a `Sequence` whose `__len__` lies (2**62) blew up `Vec::with_capacity`
/// as a `PanicException` (capacity overflow), which `except Exception`
/// cannot catch: the one uncatchable crash class on the pyo3 boundary
/// (the `pages=` range bomb's class, the same fix shape: never trust a
/// reported size, walk under a cap). The manual walk never reads
/// `__len__`, so the cap is the only bound it needs: past this many
/// yielded items the walk aborts with a catchable `ValueError` — the
/// `borrow_str_sequence` discipline (`src/py/charset.rs`
/// MAX_BATCH_ITEMS, the content_hash walk cap's spirit), one order of
/// magnitude tighter because the honest population here is tiny (the
/// rules are closed sets of names): a legitimate call carries a handful
/// of strings, and anything past 100_000 is not a miscounted batch, it is
/// the bomb itself. The message is deliberately generic (no cap value):
/// the bound is a DoS backstop, not a contract to advertise to sequence
/// authors.
/// The `TimeoutError` construction shared by every `deadline_ms`-bearing
/// binding (`diff_opcodes` and its line spelling, the three fuzzy
/// metrics, `similarity_ratio`, `get_close_matches`, `is_grounded`): the
/// deadline core's own message carried on the builtins `TimeoutError`
/// type the tests pin (`type(excinfo.value) is TimeoutError`). The
/// exception is always constructed after the GIL is reacquired (nothing
/// raises from inside a detached region); the deadline cores are three
/// distinct nominal types with no shared trait, so the
/// message string, not the error value, is the shared currency, and this
/// one construction spelling replaces the two (`PyTimeoutError::new_err`
/// and `PyErr::new::<PyTimeoutError, _>`, the same exception either way)
/// that could otherwise drift apart.
pub(crate) fn timeout_err(message: String) -> PyErr {
    PyTimeoutError::new_err(message)
}

/// The shared `__index__`-protocol extraction behind every int-like
/// parameter (`minhash_signature`'s `num_perm`/`shingle_size`/`seed`,
/// `shingle_jaccard`/`shingle_dice`'s `width`, `simhash_distance`'s
/// fingerprint arguments): `bool` is rejected up front (it would
/// otherwise launder to 0/1 through the index), then the `__index__`
/// SLOT is dispatched — `getattr` plus call, never the instance's own
/// `__and__`, so masking cannot alter the value and `__index__`-only
/// int-likes (numpy integers) reduce identically. `__index__` itself IS
/// caller code: it runs with its own side effects, exactly once, and its
/// own failure propagates unchanged rather than masking as a parameter
/// error. Only a MISSING `__index__` (str, float, None, bytes) and an
/// `__index__` result that is not an exact int (including `bool`, the
/// same caller bug one dispatch removed) are `TypeError`. (Moved here
/// from `py/minhash.rs`, which had the only copy, when
/// `shingle_jaccard`/`simhash_distance` became the second and third
/// consumers — the same one-home discipline as the walks above.)
pub(crate) fn extract_index<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyInt>> {
    if obj.cast::<PyBool>().is_ok() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an int, not bool"
        )));
    }
    let index = obj
        .getattr("__index__")
        .map_err(|_| PyTypeError::new_err(format!("{name} must be an int")))?;
    let index = index.call0()?;
    if index.cast::<PyBool>().is_ok() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an int, not bool"
        )));
    }
    if index.cast::<PyInt>().is_err() {
        return Err(PyTypeError::new_err(format!("{name} must be an int")));
    }
    Ok(index.cast_into::<PyInt>().expect("checked exact int above"))
}
