use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

use crate::scan_impl;

/// The empty-needle refusal shared by both spellings: an empty needle would
/// match at every position and has no parity meaning (there is no occurrence
/// to stand before), the same rationale as `find_patterns`' empty pattern.
/// Raised under the GIL, ahead of the detach — the call's only error path.
fn refuse_empty_needle(needle: &[u8]) -> PyResult<()> {
    if needle.is_empty() {
        return Err(PyValueError::new_err("empty needle"));
    }
    Ok(())
}

/// `tors.contains_unescaped(haystack, needle)`: is there an occurrence of
/// `needle` in `haystack` that is not itself escaped — the boolean spelling
/// of the escape-parity scan. An occurrence at byte offset `i` counts only
/// when the maximal run of `b"\\"` immediately before `i` has even length
/// (an empty run is even, so an occurrence at offset 0 counts); an odd run
/// means the run's backslash pairs escape each other and the leftover one
/// escapes the occurrence's first byte, so the occurrence is literal text.
/// See `src/scan_impl.rs` for the full contract, the JSON `\u0000`-vs-
/// `\\u0000` motivation (a real NUL is fatal in a PostgreSQL jsonb column,
/// the literal text is fine, and the two are byte-ambiguous), and the
/// parity walk's cost story; the differential oracle and the golden
/// battery are tests/test_unescaped_scan.py.
///
/// No JSON knowledge lives in the function: parity is the mechanism, and
/// "the needle is an escape sequence" is the caller's reading of it — any
/// backslash-escaped grammar can drive the same scan.
///
/// Argument contract: both arguments exactly `bytes` (`bytearray` /
/// `memoryview` / `str` raise `TypeError`, the bytes-in surface's
/// zero-copy immutable-borrow contract — and a `str` haystack would be a
/// different coordinate system entirely); an empty needle raises
/// `ValueError("empty needle")` before any scanning runs.
///
/// GIL model: `utf8_is_valid`'s extreme point exactly. The two `&[u8]`
/// extractions are zero-copy `PyBytes` borrows, the whole scan (the memmem
/// occurrence loop and the parity walk) runs under one `py.detach`, and
/// the `bool` return has no marshalling class at all, so the argument
/// borrows alone are the call's GIL-held residue — ceiling-only heartbeat
/// budget, pinned by tests/test_gil_release.py. The empty-needle
/// `ValueError` above is the only error path, and it fires under the GIL
/// before the detach: nothing raises from inside the detached region.
#[pyfunction]
pub fn contains_unescaped(py: Python<'_>, haystack: &[u8], needle: &[u8]) -> PyResult<bool> {
    refuse_empty_needle(needle)?;
    Ok(py.detach(|| scan_impl::find_unescaped(haystack, needle).is_some()))
}

/// `tors.find_unescaped(haystack, needle)`: the byte offset of the first
/// unescaped (even-run) occurrence of `needle` in `haystack`, or `-1` when
/// none exists — `bytes.find`'s own sentinel, kept over `Optional[int]`
/// deliberately: it is the spelling stdlib callers already branch on, and
/// it keeps the return a bare `int`. Everything else is
/// `contains_unescaped`'s contract exactly: the parity rule, the bytes-only
/// argument boundary, the `ValueError("empty needle")` refusal, the
/// no-JSON-knowledge scope, and the `utf8_is_valid` GIL class (one detach
/// around the whole scan; a single-int return, no marshalling class).
///
/// **The offsets are byte offsets, not `str` indices**: the return indexes
/// the `bytes` argument it was handed, so
/// `haystack[i:i + len(needle)] == needle` for every answer that is not
/// `-1` — over multibyte UTF-8 content the byte offset and the decoded
/// text's character offset are different numbers (the byte-offset battery
/// in tests/test_unescaped_scan.py pins the divergence numerically), the
/// same class of confusion `find_patterns` solved with its byte→char
/// mapping, deliberately not solved here because the input is bytes and
/// the contract is byte-space end to end.
///
/// Rejected (odd-run) hits advance the scan one byte past the hit, not
/// past the whole match, so self-overlapping needles stay correct: the
/// two-byte needle `b"00"` in a haystack of one backslash then `000`
/// rejects the hit at 1 and finds the overlapping live hit at 2.
#[pyfunction]
pub fn find_unescaped(py: Python<'_>, haystack: &[u8], needle: &[u8]) -> PyResult<isize> {
    refuse_empty_needle(needle)?;
    Ok(py
        .detach(|| scan_impl::find_unescaped(haystack, needle))
        .map_or(-1, |hit| {
            isize::try_from(hit).expect("haystack offset fits in isize")
        }))
}

/// `tors.utf8_byte_len(s)`: the UTF-8 byte length of `s` — the answer
/// `len(s.encode("utf-8"))` computes by allocating and copying the whole
/// `bytes` object first, taken here without the copy. The count a caller
/// wants when a size cap sits in front of a store: an
/// idempotency-key and scope byte caps on every enqueue, and the
/// terminal's re-encode of a serialized result of up to 64 KiB
/// (`MAX_RESULT_BYTES`) on every success — a genuine double pass, the
/// byte count having existed inside the serializer's output and been
/// discarded by the `.decode()` that produced the `str`.
///
/// Fresh vs repeat, stated up front (the cold-lane economics the lane
/// table pins: cold-encode 4.9 ms | first-call 4.7 ms | warm-encode
/// 196 µs | cached-tors 0.13 µs at 12 MiB): the terminal's fresh `str`
/// per success never repeats on the same object, so its one-shot lane
/// is encode-parity by construction — the win there is only the absence
/// of a Python-visible `bytes` object, not wall time. The wall-time win
/// is repeat counts on the same object (O(1) cached borrow vs a fresh
/// alloc+memcpy per `encode`) and every ASCII count (zero-copy alias,
/// flat ~0.1 µs). Memory: zero-alloc — one field read off the borrowed
/// `&str` (the chunked count is the utf16 twin's; this twin allocates
/// nothing on any lane).
///
/// **Companion, not standalone**: this ships in the scan family's binding
/// module as the pinned companion of `contains_unescaped`/
/// `find_unescaped` (#50) — same module, same harness patterns — and
/// honest sizing says it would not stand alone: a short-string encode is
/// a few hundred nanoseconds, so the win is large inputs and hot paths,
/// where the copy is the cost — the 64 KiB terminal case is ~0.9 µs of
/// pure alloc+memcpy per success, the lane table's ASCII 64 KiB
/// expression cell).
///
/// Cost model (the deliberate deviation from the issue's sketch: no
/// hand-rolled UCS1/UCS2/UCS4 arithmetic — the module docs in
/// `src/scan_impl.rs` carry the full rationale — the standard str-in
/// borrow instead, and the core is the borrowed `&str`'s `len()`, one
/// field read):
///
/// * ASCII (serialized JSON with `ensure_ascii=True`): compact ASCII data
///   is its own UTF-8, so the borrow is a zero-copy alias and the call is
///   O(1), no allocation at all.
/// * Non-ASCII, first call on the object (a cold UTF-8 cache): CPython
///   materializes and CACHES the UTF-8 view on the `str` object (an
///   internal cache, not a Python-visible `bytes`; filled by this borrow
///   and by any earlier str-in tors call on the same object, read by
///   `encode` — which never fills it), so the first call is O(n) —
///   encode-parity in cost class, with no Python-visible object to
///   allocate and collect. The sharing with `encode` is one-directional:
///   a prior `len(s.encode())` does not warm this lane (measured
///   ~3.8-4.8 ms for the first call after an encode at 12 MiB on fresh
///   objects, the cold class exactly — in every CPython 3.10-3.14
///   `unicode_encode_utf8` reads the cache and only
///   `PyUnicode_AsUTF8AndSize`, the str-in borrow, writes it).
/// * Non-ASCII, repeat calls on the same object: O(1) — strictly better
///   than the expression, which re-copies on every call.
///
/// Error parity: a `str` holding lone surrogates cannot be UTF-8-encoded,
/// and the borrow raises CPython's own `UnicodeEncodeError` (pyo3
/// propagates it from the `&str` extraction, before any tors code runs) —
/// the same exception `encode` raises, attributes included; there is no
/// tors-side error path at all. Pinned attribute-for-attribute in
/// tests/test_utf8_byte_len.py.
///
/// GIL model: `grapheme_count`'s marshalling class (a single `int`
/// return, nothing else held past the borrow), with one honest
/// difference: the call's only O(n) work IS the borrow — the first
/// non-ASCII call's materialization runs under the GIL (the standard
/// str-in first-call class every str-argument tors function pays; there
/// is no way to borrow the view without it), and the core after the
/// borrow is an O(1) field read. There is deliberately NO `py.detach`:
/// #108 measured that a detach bracketing no work is pure cost and is
/// what starves a co-resident event loop — the GIL-held materialization
/// followed by a nanosecond detach/re-attach bumps `switch_number` from
/// this thread on every call, inside the waiting thread's `take_gil`
/// window, so a loop thread waiting for the GIL never escalates to a
/// drop request and loses every re-acquire race (reproduced 2/2 as a
/// ZERO-tick 2 s window on a 1 ms heartbeat; the table lives in the
/// issue). The whole call is GIL-held either way, so the detach bought
/// nothing and defeated the switch request; sibling lanes keep their
/// detach because theirs brackets the real O(n) scan (`utf16_byte_len`)
/// or a zero-copy bytes borrow + scan (`utf8_is_valid`, `decode_utf8`),
/// neither of which is this shape. The heartbeat cell in
/// tests/test_gil_release.py pins the band: the 12 MiB non-ASCII first
/// call's materialization sits under the 10 ms ping floor, so the cell
/// is ceiling-only like every sub-floor member. No aio twin: an
/// O(1)-to-O(n)-borrow call needs no thread hop.
#[pyfunction]
pub fn utf8_byte_len(_py: Python<'_>, s: &str) -> usize {
    // No `py.detach` here — see the GIL model above (#108): the whole
    // call is the GIL-held borrow plus an O(1) field read, and a detach
    // around that is the starvation mechanism, not a release.
    scan_impl::utf8_byte_len(s)
}

/// `tors.utf16_byte_len(s)`: the UTF-16 byte length of `s` — 2 bytes per
/// BMP codepoint, 4 per astral codepoint (the surrogate pair) — the
/// answer `len(s.encode("utf-16-le"))` computes by allocating and
/// copying the whole 2n `bytes` object first. The interop twin of
/// `utf8_byte_len` (the maintainer's "convert a UTF8/UTF16 python
/// len() into bytes for the API" pair, same binding module): UTF-16 is
/// the code-unit world of JavaScript, Java, Windows, and .NET —
/// `String.prototype.length` counts UTF-16 units, and an astral emoji
/// is length 2 there — so column caps (`NVARCHAR`), wire caps, and
/// interop size checks in that world are UTF-16 bytes, and the Python
/// spelling of the count allocates the entire copy to take it.
///
/// Fresh vs repeat, stated up front (the cold-lane economics): on a
/// FRESH non-ASCII object the first call pays the borrow's UTF-8-cache
/// materialization (encode-parity, GIL-held) BEFORE the scan, while the
/// utf-16 expression never touches UTF-8 — so a one-shot count of a
/// fresh non-ASCII string is the one lane the expression wins, and
/// `utf16_byte_len` is NOT recommended there; the win is repeat counts
/// on the same object (4-5x, no 2n allocation per call) and every ASCII
/// count (no cold case at all). The wall cells in
/// tests/test_performance.py pin both lanes; the fresh-object cell is
/// the end-to-end one-shot bench.
///
/// > **Breaking differences from `len(s.encode("utf-16-le"))`** (the
/// > replaced expression): (1) the error's `.encoding` is `"utf-8"`
/// > (the str-in borrow materializing the UTF-8 view is the step that
/// > fails) where the expression's says `"utf-16-le"`; (2) on a
/// > multi-surrogate run the borrow's `(start, end)` names the whole
/// > run where the utf-16-le error reports only the first unit;
/// > (3) `errors="surrogatepass"` (one 2-byte unit per lone surrogate)
/// > is unsupported — no str-argument tors function offers a surrogate
/// > mode, since every one needs the UTF-8 view first. Refusal parity
/// > otherwise: the strict codec refuses the same strings tors refuses.
///
/// Overflow: `2 * (codepoints + astral)` is checked arithmetic, and
/// unreachable for real inputs on every width — a `&str` is at most
/// `isize::MAX` bytes and the count sum never exceeds one per byte, so
/// the doubled answer is at most `2 * isize::MAX`, which fits `usize`
/// on 32-bit and 64-bit alike. The `OverflowError` is the loud refusal
/// if that invariant ever breaks, pinned at synthetic boundary counts
/// crate-side (the injectable combine unit).
///
/// The implementation is the utf8 twin's borrow plus derived
/// arithmetic, no FFI: the standard str-in borrow hands the core a
/// Rust `&str`, and the core derives the answer from its UTF-8 bytes —
/// `2 * (#codepoints + #astral)`, `#codepoints` the lead-byte count,
/// `#astral` the count of 4-byte leads (`matches!(b, 0xF0..=0xF4)` —
/// `0xF5..=0xFF` never occur in a `&str`, so the range fails safe) —
/// one pass, no allocation, no per-codepoint decoding. The derivation,
/// its proof against a `chars()`-based naive count, and the exhaustive
/// boundary sweep are `src/scan_impl.rs`'s; the stdlib-oracle parity
/// (`len(s.encode("utf-16-le"))` over generated text, every reference
/// corpus, and the same sweep) is tests/test_utf8_byte_len.py's, the
/// byte-len family file. The corners the identity buys: no astral
/// codepoints means exactly `2 * len(s)` for ALL BMP text (CJK and
/// combining marks included, where the UTF-8 byte count diverges), and
/// pure ASCII means `2 *` the UTF-8 byte count.
///
/// Cost model, the utf8 twin's exactly (same borrow, same CPython
/// UTF-8 view cache):
///
/// * ASCII: the borrow is a zero-copy alias and the scan is one
///   detached pass — the answer is exactly `2 * len(s)` in UTF-8 bytes.
/// * Non-ASCII, first call on the object (a cold UTF-8 cache): the
///   borrow materializes and caches the view under the GIL
///   (encode-parity cost; `encode` reads that cache and never fills
///   it), then the scan runs detached.
/// * Non-ASCII, repeat calls on the same object: an O(1) zero-copy
///   borrow plus the detached scan — against the expression's full
///   2n alloc+encode on every call.
///
/// **The surrogate lane is REFUSAL PARITY with the replaced expression,
/// measured and pinned** (a first-draft claim that the stdlib's
/// utf-16-le encode accepts lone surrogates was wrong — the strict
/// codec refuses them, and the battery caught it): the exact expression
/// this function replaces, `len(s.encode("utf-16-le"))`, raises
/// `UnicodeEncodeError` on the same strings this function refuses —
/// same reason ("surrogates not allowed"), the utf-8 codec's own
/// policy. The refusal happens at the str-in borrow every tors function
/// performs (the crate-wide str-in contract: the borrow must
/// materialize the object's UTF-8 view before any arithmetic runs), so
/// the error is the borrow's own and its `.encoding` is `"utf-8"` — the
/// flavor of the step that actually fails — where the expression's
/// error says `"utf-16-le"` (and, on a multi-surrogate run, reports
/// only the first unit; the borrow's error, identical to the utf-8
/// encode's, names the whole run). The one acceptance path the stdlib
/// does offer — `errors="surrogatepass"`, one 2-byte unit per lone
/// surrogate — is a mode tors deliberately does not: no str-argument
/// tors function can see past the UTF-8 view. Pinned
/// attribute-for-attribute in tests/test_utf8_byte_len.py.
///
/// GIL model: the borrow is the call's GIL-held residue (the
/// cold-cache first call's materialization — there is no way to fill
/// an object's cache without the GIL; the `finalize` first-call class),
/// and the `py.detach` around the core carries REAL work here — the
/// O(n) byte-class scan, memchr-class, far under the 10 ms heartbeat
/// floor at 12 MiB (the wall and heartbeat cells carry the numbers) —
/// where the utf8 twin's detach is nominal around one field read; the
/// heartbeat cell in tests/test_gil_release.py is ceiling-only like
/// the twin's. A single `int` return, no marshalling class. No aio
/// twin: the GIL-held residue is the borrow alone, and the scan
/// detaches.
#[pyfunction]
pub fn utf16_byte_len(py: Python<'_>, s: &str) -> PyResult<usize> {
    // Fallible core, no unwinding: the core returns Option — the None
    // leg is unreachable for real inputs on every width (a &str is at
    // most isize::MAX bytes and the count sum never exceeds one per
    // byte, so the doubled answer is at most 2 * isize::MAX, which fits
    // usize on 32-bit and 64-bit alike) — mapped here to the
    // Python-side OverflowError contract so an invariant break refuses
    // loudly. No catch_unwind, no
    // expect on this path — overflow travels as a value, so there is
    // no panic to mask and no unwind/GIL-restore assumption to make.
    py.detach(|| scan_impl::utf16_byte_len(s)).ok_or_else(|| {
        PyOverflowError::new_err(
            "utf16_byte_len overflow: input too large for usize on this target (32-bit)",
        )
    })
}
