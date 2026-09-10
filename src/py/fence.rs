use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::fence_impl;

/// `tors.extract_code_blocks(text, lang=None)`: every fenced code block in
/// `text` (CommonMark §4.5's fenced-code-block grammar, hand-rolled in
/// `src/fence_impl.rs` rather than pulled from a general Markdown parser,
/// since LLM chat output is rarely deeply-nested Markdown, so the extra
/// correctness a full parser buys is mostly wasted weight here), as
/// `(language, code, start, end)` tuples in document order: `language` is
/// the info string's first whitespace-delimited word or `None`; `code` is
/// the content between the fences with the fence's own indentation (0-3
/// leading spaces) stripped from every content line; `start`/`end` are
/// Python str index (codepoint) offsets of the block's raw span, from the
/// opening fence line's first character through the end of the closing
/// fence line (or end of input, for an unterminated fence: a model that
/// forgot to close its fence still yields a block). `lang=` filters to
/// blocks whose language matches exactly (case-sensitive).
///
/// Deliberately narrower than full CommonMark: no indented-code-block
/// recognition (fenced blocks only, the shape LLM output actually uses),
/// and no tab-expansion (a line's leading whitespace counts literal ASCII
/// spaces only; a tab-indented fence is not recognized as one).
///
/// GIL model: the whole scan runs under `py.detach` (the standard str-in
/// argument borrow first); the return marshalling is O(blocks) tuples, a
/// handful per call at the single-LLM-response scale this targets, so no
/// dedicated bench was added (unlike the multi-MB corpora `diff`/`search`
/// benchmark).
#[pyfunction(signature = (text, lang = None))]
pub fn extract_code_blocks(
    py: Python<'_>,
    text: &str,
    lang: Option<&str>,
) -> Vec<(Option<String>, String, usize, usize)> {
    py.detach(|| fence_impl::extract_code_blocks(text, lang))
        .into_iter()
        .map(|b| (b.language, b.code, b.start, b.end))
        .collect()
}

/// `tors.strip_code_fences(text)`: if `text`, trimmed of leading/trailing
/// whitespace, is exactly one fenced code block, return its dedented code
/// content; otherwise return `text` unchanged. The "whole response wrapped
/// in one fence" cleanup is safe to call unconditionally on arbitrary model
/// output, since anything that is not exactly the single-block case is a
/// no-op (not even whitespace-trimmed).
///
/// Identity-return contract (the crate's `Cow` identity convention): `tors.strip_code_fences(s)
/// is s` exactly when the input is not the single-fenced-block case.
///
/// GIL model: `detached_transform`'s shape, meaning the str-in argument borrow,
/// the scan under `py.detach`, and either the original object back
/// (zero marshalling) or the O(output) unwrapped-code string.
#[pyfunction]
pub fn strip_code_fences(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, fence_impl::strip_code_fences)
}

/// `tors.dedent(text)`: `textwrap.dedent(text)` in one GIL-released pass:
/// the longest common leading-whitespace-run string shared by every
/// non-whitespace-only line is stripped from each line, and whitespace-only
/// lines normalize to empty, matching CPython's exact algorithm (`src/fence_impl.rs`
/// ports `Lib/textwrap.py`'s two-regex reduction; the parity gate is
/// tests/test_fence.py's hypothesis differential against the running
/// stdlib). Tabs and spaces are distinct characters for the common-prefix
/// computation (not tab-expanded): `"  x"` and `"\tx"` share no margin,
/// matching `textwrap.dedent`'s documented behavior exactly.
///
/// Identity-return contract: `tors.dedent(s) is s` exactly when
/// `tors.dedent(s) == s` (no whitespace-only line to normalize and no
/// common margin to strip).
///
/// GIL model: `detached_transform`'s shape, the same as `strip_code_fences`.
#[pyfunction]
pub fn dedent(py: Python<'_>, text: Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
    detached_transform(py, text, fence_impl::dedent)
}
