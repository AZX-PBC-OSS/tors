//! The pyo3 bindings for the random-generation family (`random_impl`'s
//! core): the argument validation and seed reduction under the GIL, then
//! ONE `py.detach` per call around the whole entropy draw +
//! sampling/formatting pass, then the O(output) string marshalling — the
//! family GIL model.
//!
//! GIL model: these are the crate's cheapest native passes (a syscall plus
//! sampling or SIMD formatting; microseconds at real token/key sizes), but
//! the discipline is the same one every surface keeps: the GIL-held residue of a call is the
//! argument validation (and the seed's rare big-int mask, a Python-level `&`
//! paid only when `|seed| >= 2^63`), plus the return marshalling. An OS
//! entropy failure (effectively never post-boot; see `RandomError::Os`)
//! raises `RuntimeError` carrying the OS error string, constructed after the
//! GIL is reacquired — nothing raises from inside the detached region.
//!
//! No `aio` twins: these are fast CPU/syscall calls, not the
//! detached-transform input class `tors.aio` exists for (a thread hop costs
//! more than the call at every realistic token/key size); see docs/async.md.

use pyo3::exceptions::{
    PyAttributeError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyInt};

use crate::random_impl::{self, RandomError};

/// Reduce the `seed=` argument to the u64 the core keys its ChaCha20 stream
/// with. Any int-LIKE is legal and documented: an `int` instance (bool and
/// IntEnum ride along as the ints they are, `True` is 1) or any object
/// implementing `__index__` — the house convention for int-ish params
/// (pyo3's `i64` extraction accepts exactly these, `length` already rides
/// it, and the chunkers' size params and documents' `max_bytes=` keep the
/// same rule; the strict int-instance-only gate this replaced was the
/// family's one outlier). The gate is one normalization through
/// `__index__` — identity on int instances, so both sides of the
/// convention take the same path — and the value is reduced mod 2^64
/// (two's complement for negatives — exactly what `seed & 0xFFFFFFFFFFFFFFFF`
/// computes in Python). `|seed| < 2^63` takes the zero-conversion i64 path;
/// anything wider (up from 2^63, down from -2^63, or arbitrarily huge) goes
/// through Python's own `&` mask, one small-int C call under the GIL.
/// Anything that is not `None` and not int-like raises `TypeError` naming
/// the parameter and the accepted forms; an `__index__` that itself raises
/// surfaces its own error (pyo3's `length` extraction does the same — the
/// gate called the protocol and the protocol answered).
fn seed_to_u64(seed: Option<Bound<'_, PyAny>>) -> PyResult<Option<u64>> {
    let Some(seed) = seed else {
        return Ok(None);
    };
    let not_int_like = || {
        PyTypeError::new_err(format!(
            "seed must be int-like (__index__) or None, not {}",
            seed.get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_else(|_| "an unknown type".to_string())
        ))
    };
    // The int-like gate: `__index__` is identity on int instances and the
    // whole protocol for everything else; a non-int-like answers the
    // lookup with AttributeError and becomes the TypeError above.
    let value = seed.call_method0("__index__").map_err(|err| {
        if err.is_instance_of::<PyAttributeError>(seed.py()) {
            not_int_like()
        } else {
            err
        }
    })?;
    // `__index__`'s return must BE an int (Python's own PyNumber_Index
    // contract); a pathological implementation handing back something
    // else is the same refusal as not being an int-like at all.
    let value = value.cast::<PyInt>().map_err(|_| not_int_like())?;
    match value.extract::<i64>() {
        // The widening `as u64` IS the documented mod-2^64 reduction for
        // negatives (two's complement).
        Ok(n) => Ok(Some(n as u64)),
        // Out of i64 range: mask through Python's own int `&`, whose
        // semantics for negatives are the same two's-complement reduction.
        Err(_) => {
            let masked = value.call_method1("__and__", (0xFFFF_FFFF_FFFF_FFFFu64,))?;
            Ok(Some(masked.extract::<u64>()?))
        }
    }
}

/// Map the core's error shapes onto the Python exception taxonomy: the OS
/// entropy failure and the uuid7 clock failure become `RuntimeError`
/// carrying the reason (raise-time only, after the GIL is reacquired), the
/// empty alphabet becomes the family's ValueError naming the parameter, and
/// an impossible length's failed reservation becomes `MemoryError` —
/// catchable, the `'x' * n` / `secrets.token_hex(n)` convention, where the
/// `with_capacity` reserve it replaced aborted the whole process on the
/// same request.
fn into_pyerr(err: RandomError) -> PyErr {
    match err {
        RandomError::EmptyAlphabet => PyValueError::new_err(err.message()),
        RandomError::Memory(_) => PyMemoryError::new_err(err.message()),
        RandomError::Os(_) | RandomError::Clock(_) => PyRuntimeError::new_err(err.message()),
    }
}

/// `tors.random_string(length, alphabet, *, seed=None)`: `length` characters
/// sampled uniformly (Lemire's unbiased method — no modulo bias at any
/// alphabet size) from any non-empty `alphabet` str, as one GIL-released
/// pass.
///
/// Entropy contract: the default (no `seed`) draws fresh bytes from the
/// operating system's CSPRNG on every call — no process or thread RNG state,
/// so it is fork-safe, matching `secrets`' own per-call semantics — safe for
/// keys, tokens, and secrets. `seed=` switches to a deterministic ChaCha20
/// stream: the output becomes a pure function of (seed, arguments), fully
/// predictable from the seed — a reproducible-test/fixture tool, NEVER safe
/// for secrets, keys, or tokens (any adversary who learns the seed can
/// reproduce the stream); the unseeded spelling is the secrets-safe one.
///
/// The alphabet is any non-empty `str` (empty raises `ValueError`; non-`str`
/// raises `TypeError`): multibyte characters are sampled as characters, so
/// the output is always exactly `length` characters over the alphabet's own
/// characters. Duplicate characters are weighted, not deduplicated — each
/// position is an independent draw over the alphabet's character positions,
/// so `"aaab"` yields `a` with probability 3/4: dedupe the alphabet first
/// for uniform-over-distinct-characters. Sampling is over Unicode scalar
/// values, not grapheme clusters: a combining mark in the alphabet samples
/// independently of its base character. The alphabet is materialized fresh
/// on every call (no cross-call cache, by the fork-safe no-state
/// discipline); bulk callers reusing one huge alphabet should prefer the
/// stdlib. `length=0` returns `""`; negative raises `ValueError`; the only
/// size ceiling is the argument itself (`Py_ssize_t`, 2^63 - 1 on 64-bit —
/// anything wider raises `OverflowError` at extraction) and memory beyond
/// it: the memory bound is a catchable `MemoryError` — `try_reserve`
/// refuses an impossible length before any allocation is attempted,
/// `'x' * n`'s own shape — never a process abort.
///
/// GIL model: argument validation under the GIL, the whole fill + sampling +
/// string build under one `py.detach`, then the O(output) marshalling.
#[pyfunction(signature = (length, alphabet, *, seed = None))]
pub fn random_string(
    py: Python<'_>,
    length: isize,
    alphabet: &str,
    seed: Option<Bound<'_, PyAny>>,
) -> PyResult<String> {
    if length < 0 {
        return Err(PyValueError::new_err(format!(
            "length must be >= 0, got {length}"
        )));
    }
    let seed = seed_to_u64(seed)?;
    py.detach(|| random_impl::random_string(length as usize, alphabet, seed))
        .map_err(into_pyerr)
}

/// `tors.random_hex(length, *, seed=None)`: `length` lowercase hex
/// characters — exactly `random_string(length, "0123456789abcdef")` (one
/// engine, delegated) — the length-first hex spelling: the caller asks for
/// the id they want ("a 32-char hex key"), not for the bytes behind it.
/// Odd lengths are legal (a 31-char hex id is a real shape); even lengths
/// are what digest-shaped keys want (every 2 characters are exactly one
/// byte). `secrets.token_hex(n)` and `random_hex(2 * n)` are the same
/// uniform distribution over 2n-char hex strings — different draws.
///
/// Entropy contract: the default (no `seed`) draws fresh bytes from the
/// operating system's CSPRNG on every call — no process or thread RNG state,
/// so it is fork-safe, matching `secrets`' own per-call semantics — safe for
/// keys, tokens, and secrets. `seed=` switches to a deterministic ChaCha20
/// stream: the output becomes a pure function of (seed, arguments), fully
/// predictable from the seed — a reproducible-test/fixture tool, NEVER safe
/// for secrets, keys, or tokens (any adversary who learns the seed can
/// reproduce the stream); the unseeded spelling is the secrets-safe one.
///
/// `length=0` returns `""` (`secrets.token_hex(0)`'s own shape); negative
/// raises `ValueError`; the only size ceiling is the argument itself
/// (`Py_ssize_t`, 2^63 - 1 on 64-bit) and memory beyond it: the memory
/// bound is a catchable `MemoryError` (see `random_string`), never a
/// process abort.
///
/// GIL model: validation under the GIL, the sampling + string build under
/// one `py.detach`, then the O(output) marshalling.
#[pyfunction(signature = (length, *, seed = None))]
pub fn random_hex(
    py: Python<'_>,
    length: isize,
    seed: Option<Bound<'_, PyAny>>,
) -> PyResult<String> {
    if length < 0 {
        return Err(PyValueError::new_err(format!(
            "length must be >= 0, got {length}"
        )));
    }
    let seed = seed_to_u64(seed)?;
    py.detach(|| random_impl::random_hex(length as usize, seed))
        .map_err(into_pyerr)
}

/// `tors.random_b62(length, *, seed=None)`: `length` characters over
/// `[0-9A-Za-z]` — exactly `random_string(length, BASE62_CHARS)` (one
/// engine, delegated), the URL-safe human-transcribable id spelling.
///
/// Entropy contract: the default (no `seed`) draws fresh bytes from the
/// operating system's CSPRNG on every call — no process or thread RNG state,
/// so it is fork-safe, matching `secrets`' own per-call semantics — safe for
/// keys, tokens, and secrets. `seed=` switches to a deterministic ChaCha20
/// stream: the output becomes a pure function of (seed, arguments), fully
/// predictable from the seed — a reproducible-test/fixture tool, NEVER safe
/// for secrets, keys, or tokens (any adversary who learns the seed can
/// reproduce the stream); the unseeded spelling is the secrets-safe one.
///
/// `length=0` returns `""`; negative raises `ValueError`; the only size
/// ceiling is the argument itself (`Py_ssize_t`, 2^63 - 1 on 64-bit) and
/// memory beyond it (the memory bound is a catchable `MemoryError`,
/// never an abort).
///
/// GIL model: `random_string`'s exactly.
#[pyfunction(signature = (length, *, seed = None))]
pub fn random_b62(
    py: Python<'_>,
    length: isize,
    seed: Option<Bound<'_, PyAny>>,
) -> PyResult<String> {
    if length < 0 {
        return Err(PyValueError::new_err(format!(
            "length must be >= 0, got {length}"
        )));
    }
    let seed = seed_to_u64(seed)?;
    py.detach(|| random_impl::random_b62(length as usize, seed))
        .map_err(into_pyerr)
}

/// `tors.random_b64url(length, *, seed=None)`: `length` characters uniform
/// over the 64-character RFC 4648 §5 urlsafe alphabet (`A-Za-z0-9-_`; `+`
/// and `/` never), every position unconstrained — exactly
/// `random_string(length, B64URL_CHARS)` (one engine, delegated). The
/// length-first opaque-token spelling: "I want a 43-char urlsafe token"
/// (the JWT-signature shape) is the whole call.
///
/// The boundary, stated honestly: this is NOT "a valid base64 encoding of
/// N random bytes" — an encoding's final character is constrained (at 43
/// characters, an encoding of 32 bytes can only ever show 16 distinct
/// final characters, the final char carrying the final byte's low 4 bits
/// shifted into place; the 31-byte encoding is the 4-distinct case, its
/// final char carrying just the low 2 bits; this spelling shows all 64),
/// and lengths that are not valid base64 output lengths (41, 45, ...) are
/// legal here. There is no `padded=` parameter: padding is an encoding
/// concept, not a token concept, and `=` never appears. Callers who want
/// encodable random material should take `random_hex` of even length
/// (byte-exact via hex).
///
/// Entropy contract: the default (no `seed`) draws fresh bytes from the
/// operating system's CSPRNG on every call — no process or thread RNG state,
/// so it is fork-safe, matching `secrets`' own per-call semantics — safe for
/// keys, tokens, and secrets. `seed=` switches to a deterministic ChaCha20
/// stream: the output becomes a pure function of (seed, arguments), fully
/// predictable from the seed — a reproducible-test/fixture tool, NEVER safe
/// for secrets, keys, or tokens (any adversary who learns the seed can
/// reproduce the stream); the unseeded spelling is the secrets-safe one.
///
/// `length=0` returns `""`; negative raises `ValueError`; the only size
/// ceiling is the argument itself (`Py_ssize_t`, 2^63 - 1 on 64-bit) and
/// memory beyond it (the memory bound is a catchable `MemoryError`,
/// never an abort).
///
/// GIL model: `random_string`'s exactly.
#[pyfunction(signature = (length, *, seed = None))]
pub fn random_b64url(
    py: Python<'_>,
    length: isize,
    seed: Option<Bound<'_, PyAny>>,
) -> PyResult<String> {
    if length < 0 {
        return Err(PyValueError::new_err(format!(
            "length must be >= 0, got {length}"
        )));
    }
    let seed = seed_to_u64(seed)?;
    py.detach(|| random_impl::random_b64url(length as usize, seed))
        .map_err(into_pyerr)
}

/// `tors.uuid4(*, seed=None)`: an RFC 4122 version-4 UUID string (36 chars,
/// lowercase, hyphens at 8/13/18/23) from one 16-byte entropy fill — the
/// stdlib `uuid.uuid4()` spelling, GIL-released, with the family's optional
/// deterministic mode.
///
/// Entropy contract: the default (no `seed`) draws fresh bytes from the
/// operating system's CSPRNG on every call — no process or thread RNG state,
/// so it is fork-safe, matching `secrets`' own per-call semantics — safe for
/// keys, tokens, and secrets. `seed=` switches to a deterministic ChaCha20
/// stream: the output becomes a pure function of (seed, arguments), fully
/// predictable from the seed — a reproducible-test/fixture tool, NEVER safe
/// for secrets, keys, or tokens (any adversary who learns the seed can
/// reproduce the stream); the unseeded spelling is the secrets-safe one.
///
/// Uniqueness is probabilistic (122 random bits): a collision needs ~2^61
/// draws (the birthday bound), the same guarantee `uuid.uuid4()` carries.
///
/// GIL model: `random_hex`'s exactly.
#[pyfunction(signature = (*, seed = None))]
pub fn uuid4(py: Python<'_>, seed: Option<Bound<'_, PyAny>>) -> PyResult<String> {
    let seed = seed_to_u64(seed)?;
    py.detach(|| random_impl::uuid4(seed)).map_err(into_pyerr)
}

/// `tors.uuid7()`: an RFC 9562 version-7 UUID string (36 chars, lowercase,
/// hyphens at 8/13/18/23): 48-bit Unix-epoch milliseconds + version 7 +
/// variant + 74 random bits (12-bit rand_a + 62-bit rand_b) from one OS
/// draw — the sortable-id spelling (`uuid_utils.uuid7()`'s format class).
///
/// No `seed=` parameter, by design: the millisecond timestamp is external
/// state, so a seeded uuid7 would still vary with the clock — the
/// deterministic tool is `uuid4(seed=...)`.
///
/// The uniqueness boundary, stated honestly: probabilistically unique (74
/// random bits per millisecond, distinct timestamps across milliseconds —
/// birthday bound ~2^37 same-millisecond draws), NOT counter-monotonic.
/// Two calls within one millisecond are ordered by their random bits, not
/// by call order, and a backwards clock step flows straight into the
/// timestamp; `uuid_utils`' strict-monotonic counter is a different product
/// promise. The caller-visible contract is the timestamp itself: the 48-bit
/// field is the canonical string's first two dash-free groups, so
/// `int(u[:8] + u[9:13], 16)` is the call's Unix-epoch milliseconds (the
/// round-trip shape).
///
/// GIL model: the timestamp read and the draw + formatting all run under
/// one `py.detach` (nothing Python-visible happens before it); a clock
/// before the Unix epoch raises `RuntimeError` after the GIL is reacquired.
#[pyfunction]
pub fn uuid7(py: Python<'_>) -> PyResult<String> {
    py.detach(random_impl::uuid7).map_err(into_pyerr)
}

/// `tors.uuid4_bytes(*, seed=None)`: the RFC 4122 version-4 UUID's 16 raw
/// bytes — version and variant nibbles set, NO canonical formatting — from
/// one 16-byte entropy fill: the buffer `uuid4` formats. The bytes-out
/// spelling for consumers who re-wrap the canonical str back into bytes
/// anyway (`UUID(bytes=...)` construction, `.hex()` slicing): one native
/// draw and the field layout, no format-then-reparse roundtrip.
///
/// Entropy contract: the default (no `seed`) draws fresh bytes from the
/// operating system's CSPRNG on every call — no process or thread RNG state,
/// so it is fork-safe, matching `secrets`' own per-call semantics — safe for
/// keys, tokens, and secrets. `seed=` switches to a deterministic ChaCha20
/// stream: the output becomes a pure function of (seed, arguments), fully
/// predictable from the seed — a reproducible-test/fixture tool, NEVER safe
/// for secrets, keys, or tokens (any adversary who learns the seed can
/// reproduce the stream); the unseeded spelling is the secrets-safe one.
///
/// Uniqueness is probabilistic (122 random bits), `uuid4`'s own guarantee.
/// Fixed size: 16 bytes, always — there is no length argument, so the
/// token spellings' memory-bound class does not exist here.
///
/// GIL model: `uuid4`'s exactly (the 16-byte `bytes` marshalling after the
/// GIL is reacquired).
#[pyfunction(signature = (*, seed = None))]
pub fn uuid4_bytes(py: Python<'_>, seed: Option<Bound<'_, PyAny>>) -> PyResult<Py<PyBytes>> {
    let seed = seed_to_u64(seed)?;
    let bytes = py
        .detach(|| random_impl::uuid4_bytes(seed))
        .map_err(into_pyerr)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// `tors.uuid7_bytes()`: the RFC 9562 version-7 UUID's 16 raw bytes —
/// 48-bit Unix-epoch milliseconds + version 7 + variant + 74 random bits
/// (12-bit rand_a + 62-bit rand_b) from one OS draw, NO canonical
/// formatting: the buffer `uuid7` formats. The bytes-out spelling for the
/// same re-wrap consumers (`UUID(bytes=...)`, `.hex()[:12]` timestamp
/// slicing).
///
/// No `seed=` parameter, by design: the millisecond timestamp is external
/// state, so a seeded uuid7 would still vary with the clock — the
/// deterministic tool is `uuid4(seed=...)`.
///
/// The uniqueness boundary, stated honestly: probabilistically unique (74
/// random bits per millisecond, distinct timestamps across milliseconds),
/// NOT counter-monotonic — `uuid7`'s own boundary, on the buffer spelling.
/// The caller-visible contract is the timestamp itself: the first 6 bytes
/// big-endian are the 48-bit millisecond field
/// (`int.from_bytes(b[:6], "big")`), whose hex spelling is `.hex()[:12]` —
/// the two consumer slice shapes. Fixed size: 16 bytes, always.
///
/// GIL model: `uuid7`'s exactly (the 16-byte `bytes` marshalling after the
/// GIL is reacquired).
#[pyfunction]
pub fn uuid7_bytes(py: Python<'_>) -> PyResult<Py<PyBytes>> {
    let bytes = py.detach(random_impl::uuid7_bytes).map_err(into_pyerr)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}
