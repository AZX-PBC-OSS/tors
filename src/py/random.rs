//! The pyo3 bindings for the random-generation family (`random_impl`'s
//! core): the argument validation and seed reduction under the GIL, then
//! ONE `py.detach` per call around the whole entropy fill + formatting pass,
//! then the O(output) string marshalling — the family GIL model.
//!
//! GIL model: these are the crate's cheapest native passes (a syscall plus
//! SIMD formatting; microseconds at real token/key sizes), but the discipline
//! is the same one every surface keeps: the GIL-held residue of a call is the
//! argument validation (and the seed's rare big-int mask, a Python-level `&`
//! paid only when `|seed| >= 2^63`), plus the return marshalling. An OS
//! entropy failure (effectively never post-boot; see `RandomError::Os`)
//! raises `RuntimeError` carrying the OS error string, constructed after the
//! GIL is reacquired — nothing raises from inside the detached region.
//!
//! No `aio` twins: these are fast CPU/syscall calls, not the
//! detached-transform input class `tors.aio` exists for (a thread hop costs
//! more than the call at every realistic token/key size); see docs/async.md.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyInt};

use crate::random_impl::{self, RandomError};

/// Reduce the `seed=` argument to the u64 the core keys its ChaCha20 stream
/// with. Any Python int is legal and documented: the value is reduced mod
/// 2^64 (two's complement for negatives — exactly what `seed &
/// 0xFFFFFFFFFFFFFFFF` computes in Python). `|seed| < 2^63` takes the
/// zero-conversion i64 path; anything wider (up from 2^63, down from -2^63,
/// or arbitrarily huge) goes through Python's own `&` mask, one small-int
/// C call under the GIL. `bool` rides along as the int it is (`True` is 1).
/// Anything that is not `None` and not an `int` raises `TypeError` naming
/// the parameter and the accepted forms.
fn seed_to_u64(seed: Option<Bound<'_, PyAny>>) -> PyResult<Option<u64>> {
    let Some(seed) = seed else {
        return Ok(None);
    };
    let value = seed.cast::<PyInt>().map_err(|_| {
        PyTypeError::new_err(format!(
            "seed must be an int or None, not {}",
            seed.get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_else(|_| "an unknown type".to_string())
        ))
    })?;
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
/// carrying the reason (raise-time only, after the GIL is reacquired), and
/// the empty alphabet becomes the family's ValueError naming the parameter.
fn into_pyerr(err: RandomError) -> PyErr {
    match err {
        RandomError::EmptyAlphabet => PyValueError::new_err(err.message()),
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
/// characters. `length=0` returns `""`; negative raises `ValueError`; there
/// is no size cap (memory is the only bound).
///
/// GIL model: argument validation under the GIL, the whole fill + sampling +
/// string build under one `py.detach`, then the O(output) marshalling.
#[pyfunction(signature = (length, alphabet, *, seed = None))]
pub fn random_string(
    py: Python<'_>,
    length: i64,
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

/// `tors.random_hex(n_bytes, *, seed=None)`: `secrets.token_hex(n_bytes)`
/// parity — `2 * n_bytes` lowercase hex characters from one n-byte entropy
/// fill, as one GIL-released pass. The hex-key spelling of the family.
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
/// `n_bytes=0` returns `""` (`secrets.token_hex(0)`'s own shape); negative
/// raises `ValueError`; no size cap (memory is the only bound).
///
/// GIL model: validation under the GIL, the fill + hex formatting under one
/// `py.detach`, then the O(output) marshalling.
#[pyfunction(signature = (n_bytes, *, seed = None))]
pub fn random_hex(
    py: Python<'_>,
    n_bytes: i64,
    seed: Option<Bound<'_, PyAny>>,
) -> PyResult<String> {
    if n_bytes < 0 {
        return Err(PyValueError::new_err(format!(
            "n_bytes must be >= 0, got {n_bytes}"
        )));
    }
    let seed = seed_to_u64(seed)?;
    py.detach(|| random_impl::random_hex(n_bytes as usize, seed))
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
/// `length=0` returns `""`; negative raises `ValueError`; no size cap.
///
/// GIL model: `random_string`'s exactly.
#[pyfunction(signature = (length, *, seed = None))]
pub fn random_b62(py: Python<'_>, length: i64, seed: Option<Bound<'_, PyAny>>) -> PyResult<String> {
    if length < 0 {
        return Err(PyValueError::new_err(format!(
            "length must be >= 0, got {length}"
        )));
    }
    let seed = seed_to_u64(seed)?;
    py.detach(|| random_impl::random_b62(length as usize, seed))
        .map_err(into_pyerr)
}

/// `tors.random_b64url(n_bytes, *, padded=False, seed=None)`: RFC 4648 §5
/// urlsafe base64 (`A-Za-z0-9-_`; `+`/`/` never) of one n-byte entropy
/// fill — `secrets.token_urlsafe(n_bytes)` parity at the default
/// `padded=False` (length `ceil(4n/3)`, no `=` tail); `padded=True` adds
/// the `=` tail per the RFC (`(3 - n mod 3) mod 3` of them, total
/// `4 * ceil(n/3)`). The urlsafe-token spelling of the family.
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
/// `n_bytes=0` returns `""` either way; negative raises `ValueError`; no
/// size cap.
///
/// GIL model: `random_hex`'s exactly.
#[pyfunction(signature = (n_bytes, *, padded = false, seed = None))]
pub fn random_b64url(
    py: Python<'_>,
    n_bytes: i64,
    padded: bool,
    seed: Option<Bound<'_, PyAny>>,
) -> PyResult<String> {
    if n_bytes < 0 {
        return Err(PyValueError::new_err(format!(
            "n_bytes must be >= 0, got {n_bytes}"
        )));
    }
    let seed = seed_to_u64(seed)?;
    py.detach(|| random_impl::random_b64url(n_bytes as usize, padded, seed))
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
