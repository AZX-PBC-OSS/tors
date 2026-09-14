use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::pii_impl::{self, PiiRules};

/// The `rules=` parameter's three accepted spellings, the same
/// closed-set-of-strings convention as `errors=`/`boundary=` (anything
/// else is a `ValueError` naming the accepted set, the
/// `unicodedata.normalize` form-argument convention). `None` is every
/// rule in the canonical order; duplicates dedupe and caller order is
/// irrelevant, so the parse folds the names straight into the rule flags.
pub(crate) fn parse_pii_rules(rules: Option<Vec<String>>) -> PyResult<PiiRules> {
    let Some(names) = rules else {
        return Ok(PiiRules::BOTH);
    };
    let mut parsed = PiiRules {
        email: false,
        phone: false,
        keys: false,
    };
    for name in &names {
        match name.as_str() {
            "contact_email" => parsed.email = true,
            "contact_phone" => parsed.phone = true,
            "api_keys" => parsed.keys = true,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "rules must be one of ('contact_email', 'contact_phone', 'api_keys'), not {name:?}"
                )));
            }
        }
    }
    Ok(parsed)
}

/// `tors.scrub_pii`: replace contact material (email addresses,
/// `+`-led phone numbers) and credential material (the evidence-backed
/// api-key families) inside free text with correlation tokens —
/// `@domain~digest` for an email, `prefix~digest` for a phone number,
/// `<family prefix>~digest` for a key — in one GIL-released pass, the
/// scrub an error excerpt or rejection message needs before it reaches
/// telemetry, the one store a data purge cannot reach. The contact rules
/// are a port of a private consumer's telemetry-safety module, pinned
/// byte-identical to it at `salt=""` (tests/test_scrub_pii.py,
/// tests/test_scrub_pii_parity.py; see `src/pii_impl.rs` for the full
/// contract and its documented corners: the re-fire corner — re-scrub
/// once to a fixed point — and the salt trade-off).
///
/// `rules=None` applies every rule in the canonical order — the keys
/// substitution FIRST (a key's tail can spell a dash-separated domestic
/// phone run and a whole key an email local part, so the credential must
/// be eaten before the contact passes scan), then email, then phone over
/// its result (an email's local part may itself contain a `+`-led digit
/// run); `[]` is the identity (the original object); each name restricts
/// to that rule. `salt=None` resolves PER RULE: the contact rules keep
/// tors's documented default constant and the keys rule its own
/// `"tors/scrub_keys/v1"` tag (a fixed, non-secret domain-separation
/// tag per rule: a key digest can never alias a contact digest; a KNOWN
/// salt still leaves candidate-list confirmation possible — the tokens
/// are redaction, not pseudonymization crypto; deployments that care
/// pass their own salt, and a consumer migrating from an unsalted
/// scrubber passes `salt=""` to keep its token values byte-identical for
/// every rule).
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
/// under the GIL, then the whole multi-rule pass (all three scans,
/// every splice, every digest — the keys pass rides the same single
/// detach, no new GIL class) under one `py.detach`, then either the
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
    // salt=None resolves per rule — the contact tag and the keys tag —
    // while an explicit string ("" included) salts every rule alike.
    let (contact_salt, keys_salt) = match salt {
        None => (pii_impl::DEFAULT_SALT, pii_impl::KEYS_DEFAULT_SALT),
        Some(s) => (s, s),
    };
    detached_transform(py, text, |s| {
        pii_impl::scrub_pii(s, rules, contact_salt, keys_salt)
    })
}
