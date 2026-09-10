use std::borrow::Cow;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::html_impl;

/// The running interpreter's integer string conversion limit for decimal
/// numeric refs, read once per `html_unescape` call under the GIL. A Python
/// call per ref would be a real cost on entity-dense text; per call it is a
/// sys-module dict lookup plus one C call (µs-scale). The limit and the
/// getter exist everywhere from 3.11 on and on 3.10.7+ (the CVE-2020-10735
/// backport); only 3.10.0–3.10.6 predate the attribute, and
/// `sys.set_int_max_str_digits(0)` disables the limit. The absent attribute
/// and the disabled limit both spell `None` (no check), matching the running
/// stdlib's `html.unescape` exactly. Any other failure importing sys or
/// calling the getter propagates.
fn current_int_max_str_digits(py: Python<'_>) -> PyResult<Option<usize>> {
    let sys = PyModule::import(py, "sys")?;
    let getter = match sys.getattr("get_int_max_str_digits") {
        Ok(getter) => getter,
        // 3.10: the attribute itself does not exist, so no limit, by design.
        Err(_) => return Ok(None),
    };
    let limit = getter.call0()?.extract::<isize>()?;
    Ok(if limit > 0 {
        Some(limit as usize)
    } else {
        None // 0 (or somehow negative): the limit is disabled
    })
}

/// `tors.html_unescape`: `html.unescape(text)` over the full HTML5 named
/// entity table (all 2231 entries of `html.entities.html5`, with- and
/// without-semicolon spellings) plus CPython's exact numeric-reference
/// classification, in one GIL-released pass. The algorithm and the crate
/// decision are documented in `src/html_impl.rs` (the parity gate is
/// tests/test_html_unescape.py: full-table iteration, the tricky battery,
/// hypothesis over entity-bearing text).
///
/// CPython's integer string conversion limit applies (3.11+, and 3.10.7+
/// via the CVE-2020-10735 backport; only 3.10.0–3.10.6 lack it): a decimal
/// numeric ref whose digit run exceeds `sys.get_int_max_str_digits()`
/// (default 4300) raises the stdlib's own
/// `ValueError("Exceeds the limit (…) for integer string conversion: …")`.
/// The limit is read from the running interpreter once per call, so a
/// caller's `sys.set_int_max_str_digits()` change is honored on the next
/// call; hex refs are exempt (power-of-two base). The raised message comes
/// from the interpreter itself (the error path replays `int()` on the
/// offending digit run), because CPython's wording of the limit error is
/// version-dependent and tors must match the running stdlib, not one
/// branch of it.
///
/// GIL model: str-in. The argument is taken as `Bound<PyString>` so the
/// identity path can return the input object itself, exactly as CPython's
/// `if '&' not in s: return s` does (returning a marshalled copy instead
/// would turn a ~µs no-op into an O(n) copy of a 12 MiB string, measured
/// 1.17ms vs 0.08ms). The limit read is one µs-scale sys call under the GIL
/// before the detach. The identity contract covers more than the no-`&`
/// bail: ampersand-bearing text where nothing decodes also returns the
/// original object. Entity-bearing input is ASCII (zero-copy UTF-8 borrow),
/// the scan (memchr-hopped between `&`s) runs under `py.detach`, and the
/// residue is the marshalling of the decoded output string (O(output)),
/// plus, on the over-limit error path, the `ValueError` construction after
/// the GIL is reacquired (the scan returns a plain Rust error from the
/// detached region; nothing raises inside it, and the exception raised is
/// the interpreter's own, via the `int()` replay above).
#[pyfunction]
pub fn html_unescape(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    let s = text.to_str()?;
    let limit = current_int_max_str_digits(py)?;
    let outcome = py.detach(|| html_impl::unescape_checked(s, limit));
    match outcome {
        Ok(Cow::Borrowed(_)) => Ok(text.into_any().unbind()),
        Ok(Cow::Owned(out)) => Ok(out.into_pyobject(py)?.into_any().unbind()),
        Err(err) => {
            // Replay the running interpreter's own `int()` on the offending
            // digit run and raise the very ValueError it returns. CPython's
            // wording of the limit error is version-dependent (3.12+ and late
            // 3.11.x interpolate "the limit (4300 digits)"; 3.10.7–3.11.x
            // "the limit (4300)"), so formatting a message here would diverge
            // from the running stdlib on exactly the legs the parity gate
            // runs on. The replay is exact on every interpreter and costs
            // only the error path (CPython's own `html.unescape` pays the
            // same `int()` on the same digit run). If the replay cannot
            // produce the ValueError (the limit was disabled concurrently
            // and `int()` unexpectedly succeeds, or something unrelated
            // raises), the recorded fallback shape is raised instead.
            let int_callable = PyModule::import(py, "builtins")?.getattr("int")?;
            match int_callable.call1((err.digit_run.as_str(),)) {
                Ok(_) => Err(PyValueError::new_err(err.message())),
                Err(exc) if exc.is_instance_of::<PyValueError>(py) => Err(exc),
                Err(_) => Err(PyValueError::new_err(err.message())),
            }
        }
    }
}
