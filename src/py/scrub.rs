use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::scrub_impl::{self, RuleSet};

/// The `rules=` parameter's three accepted spellings, the same
/// closed-set-of-strings convention as `errors=`/`boundary=` (the
/// `parse_errors_mode` shape). Anything else is a `ValueError` naming the
/// accepted set; the names dedupe and apply in canonical order
/// (`pg_detail_lines` → `uri_userinfo` → `uri_query_creds`, the chain's
/// own order), so caller order is irrelevant by construction.
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
            other => {
                return Err(PyValueError::new_err(format!(
                    "rules must be one of ('pg_detail_lines', 'uri_userinfo', \
                     'uri_query_creds'), not {other:?}"
                )))
            }
        }
    }
    Ok(set)
}

/// `tors.scrub_log_text`: the TaskQ exception-text scrub chain as one
/// GIL-released pass — drop PostgreSQL DETAIL lines (real-newline and
/// repr()-flattened), mask `scheme://user:password@host` userinfo
/// passwords, and mask password-family query parameters — byte-identical
/// to the consumer's four compiled regexes (pinned by
/// `tests/test_scrub_log_text_parity.py`).
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
    rules: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    let rules = parse_rules(rules)?;
    detached_transform(py, text, move |s| scrub_impl::scrub_log_text(s, rules))
}
