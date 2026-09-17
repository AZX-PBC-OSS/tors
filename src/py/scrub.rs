use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::py::_borrow::bounded_str_list;
use crate::scrub_impl::{self, RuleSet};

/// The `rules=` parameter's accepted spellings, the same
/// closed-set-of-strings convention as `errors=`/`boundary=` (the
/// `parse_errors_mode` shape). Anything else is a `ValueError` naming the
/// accepted set; the names dedupe and apply in canonical order
/// (`pg_detail_lines` → `uri_userinfo` → the conninfo pass, whose
/// `uri_query_creds`/`libpq_conninfo_creds` names select the two anchor
/// grammars of one pass — the chain's own order), so caller order is
/// irrelevant by construction.
fn parse_rules(rules: Option<Vec<String>>) -> PyResult<RuleSet> {
    let Some(names) = rules else {
        return Ok(RuleSet::ALL);
    };
    let mut set = RuleSet::EMPTY;
    for name in &names {
        match name.as_str() {
            "pg_detail_lines" => set |= RuleSet::PG_DETAIL_LINES,
            "uri_userinfo" => set |= RuleSet::URI_USERINFO,
            "uri_query_creds" => set |= RuleSet::URI_QUERY_CREDS,
            "libpq_conninfo_creds" => set |= RuleSet::LIBPQ_CONNINFO_CREDS,
            other => {
                return Err(PyValueError::new_err(format!(
                    "rules must be one of ('pg_detail_lines', 'uri_userinfo', \
                     'uri_query_creds', 'libpq_conninfo_creds'), not {other:?}"
                )));
            }
        }
    }
    Ok(set)
}

/// `tors.scrub_log_text`: the TaskQ exception-text scrub chain as four
/// linear scans + splice under one `py.detach` — drop PostgreSQL DETAIL
/// lines (real-newline, ExceptionGroup gutters included, and
/// repr()-flattened, fail-closed), mask `scheme://user:password@host`
/// userinfo passwords, and mask password-family connection parameters in
/// both spellings (URI query string and libpq keyword/value conninfo,
/// case-insensitively) — byte-identical to the consumer's four compiled
/// regexes (pinned by `tests/test_scrub_log_text_parity.py`).
///
/// `rules=None` (the default) runs the full chain in canonical order;
/// `rules=[]` is the identity; duplicates dedupe and caller order is
/// irrelevant (rule interaction is why order is a contract, not a choice).
///
/// Identity-return contract: `tors.scrub_log_text(s, rules) is s` exactly
/// when no rule fires — including the `***` fixed points, where a rule
/// fires and splices to an equal value: those return a fresh, equal
/// string.
///
/// GIL model: `detached_transform`'s shape. The argument borrow and the
/// O(rules) name walk (each entry the standard str-in borrow class) run
/// under the GIL; the whole multi-rule pass runs under one `py.detach`;
/// the residue is the single output string's marshalling, or nothing at
/// all on the identity lane.
#[pyfunction(signature = (text, rules = None))]
pub fn scrub_log_text(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    rules: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // The list param extracts through the bounded manual walk
    // ([`bounded_str_list`]): pyo3's `Option<Vec<String>>` sizing from a
    // lying `__len__` was the uncatchable capacity-overflow class (#112's
    // residue — `scrub_log_text("x", rules=LyingLenSequence())`). The
    // walk's per-item `push` closure carries the `String` element here
    // (the same element type `src/py/pii.rs`'s call sites fold in — see
    // that file's dedup note).
    let rules = match rules {
        None => None,
        Some(any) => {
            let mut out: Vec<String> = Vec::new();
            bounded_str_list("scrub_log_text", "rules", &any, |handle| {
                out.push(handle.extract::<&str>()?.to_owned());
                Ok(())
            })?;
            Some(out)
        }
    };
    let rules = parse_rules(rules)?;
    detached_transform(py, text, move |s| scrub_impl::scrub_log_text(s, rules))
}
