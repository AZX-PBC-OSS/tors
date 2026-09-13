use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::pii_impl::{self, PiiRules};

/// The `rules=` parameter's two accepted spellings, the same
/// closed-set-of-strings convention as `errors=`/`boundary=` (anything
/// else is a `ValueError` naming the accepted set, the
/// `unicodedata.normalize` form-argument convention). `None` is both
/// rules; duplicates dedupe and caller order is irrelevant, so the
/// parse folds the names straight into the rule flags.
pub(crate) fn parse_pii_rules(rules: Option<Vec<String>>) -> PyResult<PiiRules> {
    let Some(names) = rules else {
        return Ok(PiiRules::BOTH);
    };
    let mut parsed = PiiRules {
        email: false,
        phone: false,
    };
    for name in &names {
        match name.as_str() {
            "contact_email" => parsed.email = true,
            "contact_phone" => parsed.phone = true,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "rules must be one of ('contact_email', 'contact_phone'), not {name:?}"
                )));
            }
        }
    }
    Ok(parsed)
}

/// `tors.scrub_pii`: replace contact material (email addresses,
/// `+`-led phone numbers) inside free text with correlation tokens —
/// `@domain~digest` for an email, `prefix~digest` for a phone number —
/// in one GIL-released pass, the scrub an error excerpt or rejection
/// message needs before it reaches telemetry, the one store a data
/// purge cannot reach. A port of a private consumer's telemetry-safety
/// module, pinned byte-identical to it at `salt=""`
/// (tests/test_scrub_pii.py, tests/test_scrub_pii_parity.py; see
/// `src/pii_impl.rs` for the full contract and its two documented
/// corners: the re-fire corner — re-scrub once to a fixed point — and
/// the salt trade-off).
///
/// `rules=None` applies both rules in the canonical order (email
/// substitution first, then phone over its result — an email's local
/// part may itself contain a `+`-led digit run); `[]` is the identity
/// (the original object); each name restricts to that rule.
/// `salt=None` is tors's documented default constant (a fixed,
/// non-secret domain-separation tag: a known salt still leaves
/// candidate-list confirmation possible — the tokens are redaction, not
/// pseudonymization crypto; deployments that care pass their own salt,
/// and a consumer migrating from an unsalted scrubber passes `salt=""`
/// to keep its token values byte-identical).
///
/// Identity-return contract: `tors.scrub_pii(s, ...) is s` exactly when
/// no active rule matches. Lone surrogates are refused at the argument
/// boundary with `UnicodeEncodeError` ("surrogates not allowed"), the
/// same boundary every str-in function here documents — the ported
/// chain diverges there by design (its classes never match a
/// surrogate, so it returns such text unchanged), and the divergence is
/// pinned in both directions.
///
/// GIL model: `detached_transform`'s shape, the same as
/// `strip_controls` — the text borrow and the rule/salt validation
/// under the GIL, then the whole multi-rule pass (both scans, both
/// splices, every digest) under one `py.detach`, then either the
/// identity return (zero allocation, zero marshalling) or the O(output)
/// string marshalling.
#[pyfunction(signature = (text, rules = None, *, salt = None))]
pub fn scrub_pii(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    rules: Option<Vec<String>>,
    salt: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let rules = parse_pii_rules(rules)?;
    let salt = salt.unwrap_or(pii_impl::DEFAULT_SALT);
    detached_transform(py, text, |s| pii_impl::scrub_pii(s, rules, salt))
}
