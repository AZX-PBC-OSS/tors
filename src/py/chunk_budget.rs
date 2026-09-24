//! The token-budget chunking bindings: `tors.chunk_to_budget` (the
//! caller's token counter, a Python callable) and `tors.chunk_to_offsets`
//! (pre-computed token spans, fully GIL-free packing).
//!
//! GIL model, stated honestly because the doctrine requires it: the two
//! spellings sit at opposite ends of the family's GIL spectrum.
//!
//! `chunk_to_budget`'s `token_counter` is a Python callable and can only
//! run while the calling thread holds the GIL. The packing core
//! ([`crate::chunk_budget_impl::chunk_to_budget`]) runs under ONE
//! `py.detach`, and the counter is invoked from inside it via
//! `Python::attach` — so the GIL is released for every byte of native
//! work between measurements (segmentation, offset arithmetic, slicing)
//! and held only while the counter itself runs, plus the O(chunk) argument
//! `PyString` construction per call. This is NOT a GIL-free function: a
//! slow counter dominates the call and holds the GIL for its duration,
//! exactly as it would in pure Python. That is the honest statement the
//! doctrine demands; `tors.aio.chunk_to_budget` (the to_thread hop)
//! helps when the loop can actually take the GIL between callbacks,
//! which holds when each callback exceeds `sys.getswitchinterval()`
//! (5ms by default) or the native windows between them are substantial
//! (a callback that straddles the interval fires `gil_drop_request`,
//! CPython's fair handoff). The measured caveat, stated because the
//! doctrine forbids false GIL claims: a GIL-held callback SHORTER than
//! the switch interval on a small text (microsecond detach windows) can
//! starve the loop for the whole call — the worker drops and re-acquires
//! the GIL faster than the woken loop thread can take it.
//! tests/test_gil_release.py pins the schedulable band for
//! super-interval callbacks and states the caveat where it cannot.
//!
//! `chunk_to_offsets` takes the token spans PRE-COMPUTED (the caller's
//! tokenizer has already run; HuggingFace `Encoding.offsets` is exactly
//! this shape) and never calls back: the O(tokens) argument walk runs
//! under the GIL (the standard extraction class), then the whole pack —
//! segmentation, greedy budget cuts, overlap walk-backs — is one
//! end-to-end `py.detach`, and the return marshalling is the family's
//! usual O(chunks) 2-tuples. The GIL-free choice for hot paths.

use std::sync::Mutex;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt, PyString};

use crate::chunk_budget_impl::{self, BudgetError};
use crate::truncate_impl::char_count;

/// The `overlap=` argument's two accepted spellings, validated: an int
/// token count in `[0, max_tokens)` or a float ratio in `[0, 1)`
/// (resolved as `floor(ratio * max_tokens)` tokens). Anything else — a
/// negative count, an overlap at least as large as the budget (no
/// forward progress), a ratio of 1.0 or more, NaN, or a non-numeric
/// type — is refused before any packing runs.
fn resolve_overlap(overlap: &Bound<'_, PyAny>, max_tokens: i64) -> PyResult<u64> {
    if overlap.is_instance_of::<PyInt>() {
        // bool is an int subclass in Python: True == 1 token, a legal
        // (if odd) spelling, accepted the way isinstance(.*, int) does.
        let v: i64 = overlap.extract()?;
        if v < 0 {
            return Err(PyValueError::new_err(format!(
                "overlap must be >= 0, got {v}"
            )));
        }
        if v >= max_tokens {
            return Err(PyValueError::new_err(format!(
                "overlap must be < max_tokens (no forward progress otherwise), \
                 got overlap={v}, max_tokens={max_tokens}"
            )));
        }
        Ok(v as u64)
    } else if overlap.is_instance_of::<PyFloat>() {
        let f: f64 = overlap.extract()?;
        // The range check rejects NaN too (`contains` is false for NaN),
        // the same double-comparison the int arm's `v < 0` applies.
        if !(0.0..1.0).contains(&f) {
            return Err(PyValueError::new_err(format!(
                "overlap ratio must be in [0, 1), got {f}"
            )));
        }
        // floor(ratio * budget): a ratio's overlap is always strictly
        // below max_tokens by construction, so no further check.
        Ok((f * max_tokens as f64).floor() as u64)
    } else {
        Err(PyTypeError::new_err(
            "overlap must be an int (a token count) or a float (a ratio of max_tokens)",
        ))
    }
}

fn max_tokens_valid(max_tokens: i64) -> PyResult<u64> {
    if max_tokens < 1 {
        return Err(PyValueError::new_err(format!(
            "max_tokens must be >= 1, got {max_tokens}"
        )));
    }
    Ok(max_tokens as u64)
}

/// One counter return, validated: an int (a `bool` rides Python's int
/// subclassing) whose value fits a 64-bit integer and is not negative.
/// Zero is returned as-is here: rejecting it is the packing core's
/// sentence-level contract, not the raw measurement's (word-count
/// tokenizers legitimately return 0 for the whitespace runs between
/// words).
fn validate_counter_return(result: &Bound<'_, PyAny>) -> Result<u64, BudgetError> {
    if !result.is_instance_of::<PyInt>() {
        let name = result
            .get_type()
            .name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "<untyped>".to_string());
        return Err(BudgetError::WrongType(format!(
            "token_counter must return an int, not {name}"
        )));
    }
    let value: i64 = result.extract().map_err(|_| {
        BudgetError::invalid(
            "token_counter returned an unreasonably large value (it must fit a 64-bit integer)"
                .to_string(),
        )
    })?;
    if value < 0 {
        return Err(BudgetError::invalid(format!(
            "token_counter returned a negative count ({value})"
        )));
    }
    Ok(value as u64)
}

/// `tors.chunk_to_budget(text, token_counter, *, max_tokens, overlap=0)`:
/// token-budget chunking measured by the caller's own token counter.
/// Splits `text` at UAX #29 sentence boundaries (a sentence whose own
/// measured count exceeds `max_tokens` is re-cut at UAX #29 word
/// boundaries; a single word still wider than the whole budget goes out
/// whole), greedily packs consecutive segments into chunks whose measured
/// token count fits `max_tokens`, and returns `(start, end)` pairs in
/// Python str index (codepoint) units: `text[start:end]` is the chunk.
/// Empty text returns `[]`; text that fits the budget whole returns one
/// chunk.
///
/// `token_counter` is called with one candidate chunk's text at a time —
/// NOT once per boundary: a chunk's whole candidate span is measured per
/// packing decision (O(segments) calls total, each counting at most one
/// chunk's worth of text), so counters that merge tokens across spaces
/// or boundaries are measured exactly as the emitted chunk will be. The
/// counter must return an int >= 1 for every sentence (0 or a negative
/// count raises `ValueError` — a sentence measuring no tokens makes the
/// budget contract meaningless, whitespace-only text under a word-count
/// tokenizer included), and an int at all
/// (anything else raises `TypeError`). A counter that raises propagates
/// its exception unchanged.
///
/// `overlap` repeats trailing context into the next chunk: an int token
/// count in `[0, max_tokens)` or a float ratio in `[0, 1)`
/// (`floor(ratio * max_tokens)` tokens). The next chunk starts at the
/// trailing boundary whose span back to the closed chunk's end measures
/// at least the requested overlap — genuine shared content between
/// consecutive chunks, the RAG-retrieval shape. The overlap is declined
/// for a transition when it cannot buy new context (a chunk shorter than
/// the requested overlap, or a re-cut that would land a span strictly
/// inside its predecessor): that one transition degrades to zero overlap
/// rather than stall, loop, or emit the same text twice. Chunks are
/// non-empty, strictly increasing in both start and end, cover to the
/// end of the text, and each fits the budget per the same counter (the
/// one oversized exception above); with `overlap=0` they are a
/// contiguous lossless covering partition.
///
/// `max_tokens < 1`, a negative or over-budget int `overlap`, a
/// out-of-range float ratio, or a `token_counter` that returns 0/a
/// negative/an unreasonably large value for a sentence raise
/// `ValueError`; a non-callable `token_counter` raises `TypeError`.
/// An int beyond the i64 range the binding extracts
/// (`max_tokens=10**30`) raises pyo3's own `OverflowError` at
/// extraction instead — the `truncate_to_bounds`-identical pattern for
/// every i64-typed size argument; text beyond `u32::MAX` bytes (the
/// codepoint→byte offset grid's width, see
/// [`crate::chunk_budget_impl::grid_overflow`]) raises `ValueError`
/// rather than silently truncating offsets.
///
/// GIL model: NOT GIL-free, and not documented as one — the counter is
/// Python. The packing core runs under one `py.detach` and re-attaches
/// the GIL per counter call, so the GIL is held only while the counter
/// runs (plus O(chunk) argument construction per call) and released for
/// all native work between measurements. The loop is schedulable
/// between the callbacks when each callback exceeds
/// `sys.getswitchinterval()` (5ms default) or the native windows
/// between them are substantial; a sub-switch-interval callback on a
/// small text can starve the loop for the whole call (the drop and
/// re-acquire outruns the woken loop thread — `gil_drop_request`'s fair
/// handoff fires only for callbacks that straddle the interval).
/// `tors.aio.chunk_to_budget` hops to a thread, which interleaves the
/// per-callback GIL handoffs with the event loop within that boundary.
/// For fully GIL-free packing see [`chunk_to_offsets`].
// The `overlap=` default is spelled `0` to Python (the text_signature
// override below; the drift guard reads it) while the runtime spelling
// is `Option` + None: pyo3's default machinery needs the default
// expression to construct the parameter's Rust type, and a Python-object
// type has no GIL-free literal — so the runtime default is `None`
// (resolved as zero tokens) and the introspected default is pinned to
// the honest `0` by tests/test_pyi_drift.py against this string.
#[pyfunction(signature = (text, token_counter, *, max_tokens, overlap = None))]
#[pyo3(text_signature = "(text, token_counter, *, max_tokens, overlap=0)")]
pub fn chunk_to_budget<'py>(
    py: Python<'py>,
    text: &str,
    token_counter: Bound<'py, PyAny>,
    max_tokens: i64,
    overlap: Option<Bound<'py, PyAny>>,
) -> PyResult<Vec<(usize, usize)>> {
    let max_tokens = max_tokens_valid(max_tokens)?;
    let overlap_tokens = match &overlap {
        None => 0,
        Some(any) => resolve_overlap(any, max_tokens as i64)?,
    };
    // The codepoint→byte grid the packing resolves spans through stores
    // its byte offsets as `u32` (see chunk_budget_impl::pack): text
    // beyond `u32::MAX` bytes would silently truncate them. Refused
    // loudly instead, before any work runs.
    if chunk_budget_impl::grid_overflow(text.len() as u64) {
        return Err(PyValueError::new_err(format!(
            "text is {} bytes, beyond the {}-byte offset grid \
             token-budget chunking resolves spans through; \
             split the text and pack the pieces",
            text.len(),
            u32::MAX
        )));
    }
    if !token_counter.is_callable() {
        return Err(PyTypeError::new_err(
            "token_counter must be callable (it is called with one candidate chunk's text at a time)",
        ));
    }
    let counter: Py<PyAny> = token_counter.unbind();
    // The counter's own exception, captured inside the detached pass and
    // re-raised after the GIL is reacquired (nothing raises from inside
    // py.detach). A Mutex, not a RefCell: the capture rides the
    // detach-closure's Send bound.
    let raised: Mutex<Option<PyErr>> = Mutex::new(None);
    let result = py.detach(|| {
        chunk_budget_impl::chunk_to_budget(text, max_tokens, overlap_tokens, |span| {
            Python::attach(|py| {
                let candidate = PyString::new(py, span);
                let measured = match counter.bind(py).call1((candidate,)) {
                    Ok(measured) => measured,
                    Err(err) => {
                        *raised.lock().unwrap() = Some(err);
                        return Err(BudgetError::Raised);
                    }
                };
                validate_counter_return(&measured)
            })
        })
    });
    match result {
        Ok(chunks) => Ok(chunks),
        Err(BudgetError::Raised) => Err(raised.into_inner().unwrap().unwrap_or_else(|| {
            PyValueError::new_err("token_counter raised (the original exception was lost)")
        })),
        Err(err) => Err(match err {
            BudgetError::Invalid(message) => PyValueError::new_err(message),
            BudgetError::WrongType(message) => PyTypeError::new_err(message),
            BudgetError::Raised => {
                PyValueError::new_err("token_counter raised (the original exception was lost)")
            }
        }),
    }
}

/// `tors.chunk_to_offsets(text, token_offsets, *, max_tokens, overlap=0)`:
/// [`chunk_to_budget`]'s GIL-free twin — the same packing over
/// PRE-COMPUTED token spans. `token_offsets` is a sequence of
/// `(start, end)` pairs in Python str index (codepoint) units, one per
/// token, sorted and non-overlapping (HuggingFace tokenizers'
/// `Encoding.offsets` is exactly this shape; gaps are allowed —
/// untokenized text such as inter-token whitespace measures 0 tokens).
/// A span's token count is the number of token pairs fully contained in
/// it, so the packing is additive and exact, with no callback anywhere:
/// the argument walk runs under the GIL (the standard extraction class,
/// O(tokens)), then the whole pack runs under one `py.detach` end to
/// end.
///
/// Same contract, same validation, same return shape as
/// [`chunk_to_budget`] for every argument the two share (`max_tokens`,
/// `overlap`, empty text, the budget/coverage invariants); no counter
/// validation exists because there is no counter. `token_offsets`
/// entries must be `(start, end)` int pairs with
/// `0 <= start < end <= len(text)`, sorted and non-overlapping —
/// anything else raises `ValueError` before any packing runs.
// The same Option + text_signature-override shape as chunk_to_budget's
// `overlap` (see that function's comment).
#[pyfunction(signature = (text, token_offsets, *, max_tokens, overlap = None))]
#[pyo3(text_signature = "(text, token_offsets, *, max_tokens, overlap=0)")]
pub fn chunk_to_offsets(
    py: Python<'_>,
    text: &str,
    token_offsets: Bound<'_, PyAny>,
    max_tokens: i64,
    overlap: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<(usize, usize)>> {
    let max_tokens = max_tokens_valid(max_tokens)?;
    let overlap_tokens = match &overlap {
        None => 0,
        Some(any) => resolve_overlap(any, max_tokens as i64)?,
    };
    // The same u32 offset-grid bound as the callback spelling above
    // (same grid, same silent-truncation hazard): refused loudly, and
    // cheap enough to check before the O(tokens) walk pays for it.
    if chunk_budget_impl::grid_overflow(text.len() as u64) {
        return Err(PyValueError::new_err(format!(
            "text is {} bytes, beyond the {}-byte offset grid \
             token-budget chunking resolves spans through; \
             split the text and pack the pieces",
            text.len(),
            u32::MAX
        )));
    }
    let total = char_count(text);
    // The bounded manual walk (the #112 discipline: never size a Vec
    // from a lying `__len__`, never loop an unbounded iterator):
    // entries are pushed one at a time, so memory tracks the real
    // sequence, and a runaway iterable aborts at the cap instead of
    // hanging.
    const MAX_TOKEN_SPANS: usize = 1 << 26;
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let iterator = token_offsets.try_iter().map_err(|_| {
        PyTypeError::new_err("token_offsets must be a sequence of (start, end) int pairs")
    })?;
    for (index, item) in iterator.enumerate() {
        if index >= MAX_TOKEN_SPANS {
            return Err(PyValueError::new_err(format!(
                "token_offsets has more than {MAX_TOKEN_SPANS} entries"
            )));
        }
        let item = item.map_err(|_| {
            PyValueError::new_err("token_offsets must be a sequence of (start, end) int pairs")
        })?;
        let (start, end) = item.extract::<(i64, i64)>().map_err(|_| {
            PyValueError::new_err(
                "token_offsets must be a sequence of (start, end) int pairs, one per token",
            )
        })?;
        if start < 0 || end <= start || end > total as i64 {
            return Err(PyValueError::new_err(format!(
                "token_offsets entry ({start}, {end}) is out of bounds: \
                 each must satisfy 0 <= start < end <= {total}"
            )));
        }
        if let Some(&(prev_start, prev_end)) = spans.last()
            && start < prev_end as i64
        {
            return Err(PyValueError::new_err(format!(
                "token_offsets must be sorted and non-overlapping: \
                 ({prev_start}, {prev_end}) is followed by ({start}, {end})"
            )));
        }
        spans.push((start as usize, end as usize));
    }
    py.detach(|| chunk_budget_impl::chunk_to_offsets(text, &spans, max_tokens, overlap_tokens))
        .map_err(|err| match err {
            // The offsets measure is infallible; the core's error type
            // is shared with the callback spelling. This arm is
            // structural.
            BudgetError::Raised | BudgetError::Invalid(_) | BudgetError::WrongType(_) => {
                PyValueError::new_err(
                    "internal error: precomputed-offset packing cannot fail measurement",
                )
            }
        })
}
