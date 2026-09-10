use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::merkle_impl;
use crate::py::_borrow::borrow_bytes_list;

/// `tors.merkle_root(chunks: list[bytes]) -> str`: the Merkle root over
/// `chunks` as lowercase hex, one GIL-released native pass. Domain-separated
/// SHA-256 (RFC 6962 style: leaves hash `0x00 || chunk`, internal nodes hash
/// `0x01 || left || right`). See `src/merkle_impl.rs` for why this does not
/// use the wrapped crate's built-in undifferentiated hasher. An empty list
/// raises `ValueError` ("root of no chunks" has no non-arbitrary value); a
/// non-`list` argument or a non-`bytes` entry raises `TypeError`.
///
/// GIL model: the list walk (zero-copy `&[u8]` borrows) happens under the
/// GIL; the whole tree build runs under one `py.detach`.
#[pyfunction]
pub fn merkle_root(py: Python<'_>, chunks: Bound<'_, PyList>) -> PyResult<String> {
    // The shared walk (`_borrow.rs`'s soundness story: handles alive
    // across the detach by construction).
    borrow_bytes_list(&chunks, |borrowed| {
        py.detach(|| merkle_impl::merkle_root(borrowed))
            .ok_or_else(|| PyValueError::new_err("root of no chunks"))
    })
}

/// `tors.merkle_diff(chunks_a: list[bytes], chunks_b: list[bytes]) ->
/// list[int]`: indices where `chunks_a[i] != chunks_b[i]`, comparing chunk
/// digests rather than raw contents (each chunk is hashed once regardless of
/// size; the comparison itself is a fixed 32-byte cost per index). Every
/// index at or beyond the shorter list's length is reported. Two empty lists
/// diff to `[]`. Argument contract matches `merkle_root`'s.
///
/// GIL model: both list walks happen under the GIL; the whole hash-and-scan
/// runs under one `py.detach`.
#[pyfunction]
pub fn merkle_diff(
    py: Python<'_>,
    chunks_a: Bound<'_, PyList>,
    chunks_b: Bound<'_, PyList>,
) -> PyResult<Vec<usize>> {
    // The shared walk twice (`_borrow.rs`'s soundness story: handles
    // alive across the detach by construction), nested because both
    // borrows must be in scope for the one detached diff.
    borrow_bytes_list(&chunks_a, |a| {
        borrow_bytes_list(&chunks_b, |b| {
            Ok(py.detach(|| merkle_impl::merkle_diff(a, b)))
        })
    })
}
