use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::pii_impl::{self, KeyFamily, PiiRules, SpanKind};

/// The `rules=` parameter's three accepted spellings, the same
/// closed-set-of-strings convention as `errors=`/`boundary=` (anything
/// else is a `ValueError` naming the accepted set, the
/// `unicodedata.normalize` form-argument convention). `None` is every
/// rule in the canonical order; duplicates dedupe and caller order is
/// irrelevant, so the parse folds the names straight into the rule flags
/// (the family mask is resolved separately: `parse_key_families`).
pub(crate) fn parse_pii_rules(rules: Option<Vec<String>>) -> PyResult<PiiRules> {
    let Some(names) = rules else {
        return Ok(PiiRules::BOTH);
    };
    let mut parsed = PiiRules {
        email: false,
        phone: false,
        keys: false,
        key_families: pii_impl::KEY_FAMILY_MASK_ALL,
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

/// The `families=` parameter's closed set: the fourteen
/// `tors.KEY_FAMILIES` names, spelled from the single
/// `KEY_FAMILY_NAMES` source so the unknown-name error can never drift
/// from the tuple. `None` is every family this version knows (the set
/// grows on new families — callers needing stability list names
/// explicitly); a list selects exactly those families (duplicates
/// dedupe, caller order irrelevant — the `rules=` discipline); `[]`
/// selects nothing and is refused (a silent no-op would be a
/// misconfiguration trap). The validation is unconditional — it runs at
/// the argument boundary even when `api_keys` is not in the active
/// rules, where a VALID selection is then simply ignored (an irrelevant
/// knob never errors late).
pub(crate) fn parse_key_families(families: Option<Vec<String>>) -> PyResult<u16> {
    let Some(names) = families else {
        return Ok(pii_impl::KEY_FAMILY_MASK_ALL);
    };
    if names.is_empty() {
        return Err(PyValueError::new_err(
            "families selects no key families; use None for all or list names",
        ));
    }
    let mut mask = 0u16;
    for name in &names {
        match KeyFamily::ALL.iter().find(|f| f.name() == name.as_str()) {
            Some(f) => mask |= f.bit(),
            None => {
                let accepted = pii_impl::KEY_FAMILY_NAMES
                    .iter()
                    .map(|n| format!("'{n}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(PyValueError::new_err(format!(
                    "families must be one of ({accepted}), not {name:?}"
                )));
            }
        }
    }
    Ok(mask)
}

/// `tors.scrub_pii`: replace contact material (email addresses, phone
/// numbers) and credential material (the evidence-backed
/// api-key families) inside free text with correlation tokens —
/// `@domain~digest` for an email, `prefix~digest` for a phone number (a
/// `+`-led match keeps its dialling prefix, the first three code
/// points; a domestic match keeps the digest alone (its head digits
/// are the area code), `<family prefix>~digest` for a key, in one
/// GIL-released pass, the
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
/// to that rule. `families=None` (the default) scrubs every key family
/// this version knows; a list selects exactly those families (the
/// `tors.KEY_FAMILIES` names — unknown names and `[]` are `ValueError`s,
/// and the selection is ignored when `api_keys` is not active).
/// `salt=None` resolves PER RULE: the contact rules keep
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
/// `strip_controls` — the text borrow and the rule/salt/family
/// validation under the GIL, then the whole multi-rule pass (all three
/// scans, every splice, every digest — the keys pass rides the same
/// single detach, no new GIL class) under one `py.detach`, then either
/// the identity return (zero allocation, zero marshalling) or the O(output)
/// string marshalling.
#[pyfunction(signature = (text, rules = None, *, salt = None, families = None))]
pub fn scrub_pii(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    rules: Option<Vec<String>>,
    salt: Option<&str>,
    families: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    let mut rules = parse_pii_rules(rules)?;
    rules.key_families = parse_key_families(families)?;
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

/// `tors.scrub_pii_report`: the same scrub as `tors.scrub_pii` for the
/// same arguments (`report["text"] == scrub_pii(...)`, byte-exact —
/// asserted by the fuzz target over arbitrary input), plus the
/// accounting: `redacted` (per-rule counts plus per-family counts —
/// lowercase family names, absent types omitted), `skipped` (families
/// NOT in the active selection that would have matched anyway —
/// detection ran, redaction did not; the "we preserved a JWT, log it
/// separately" signal; always `{}` when `families=None`), and `spans`
/// (ordered by start, codepoint indices into the INPUT text, each
/// `{"type": ..., "start": ..., "end": ...}` with `type` a rule name or
/// `api_keys:<family>`). Empty input is the empty accounting:
/// `{"text": "", "redacted": {}, "skipped": {}, "spans": []}`.
///
/// Span coordinates are input coordinates: the keys pass runs on the
/// input directly, and the email/phone passes' spans are mapped back
/// through each substitution's offset map (see `src/pii_impl.rs`'
/// `map_back`: a token's verbatim head maps to its source bytes, its
/// digest half collapses to the replaced span's end — so a match that
/// began inside a prior token's digest is recorded from that token's
/// input end, and a match that ran into a token's verbatim head
/// overlaps the producing span; spans are ordered by start and MAY
/// overlap exactly there — the input bytes fed two tokens, and both
/// spans are recorded. Reconstruction from input + spans + the digest
/// construction reproduces `"text"`, pinned in
/// tests/test_scrub_pii.py).
///
/// GIL model: the same single detach as `scrub_pii` — the text borrow
/// and the rule/salt/family validation under the GIL, the whole
/// recorded pipeline (every scan, every splice, every digest, the
/// offset-map folds, the byte-to-codepoint span rendering) under one
/// `py.detach`, then the O(output + spans) dict marshalling.
#[pyfunction(signature = (text, rules = None, *, salt = None, families = None))]
pub fn scrub_pii_report(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    rules: Option<Vec<String>>,
    salt: Option<&str>,
    families: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    let mut rules = parse_pii_rules(rules)?;
    rules.key_families = parse_key_families(families)?;
    let (contact_salt, keys_salt) = match salt {
        None => (pii_impl::DEFAULT_SALT, pii_impl::KEYS_DEFAULT_SALT),
        Some(s) => (s, s),
    };
    let s = text.to_str()?;
    // The whole recorded pipeline under one detach: the scrub, the
    // counts, and the spans with byte offsets rendered to codepoint
    // indices (one linear walk — spans are few, inputs can be MiBs).
    let (out_text, email_count, phone_count, key_counts, skipped_counts, char_spans) =
        py.detach(|| {
            let rep = pii_impl::scrub_pii_report(s, rules, contact_salt, keys_salt);
            let char_spans = render_char_spans(s, &rep.spans);
            (
                rep.text,
                rep.email_count,
                rep.phone_count,
                rep.key_counts,
                rep.skipped_counts,
                char_spans,
            )
        });
    let redacted = PyDict::new(py);
    if email_count > 0 {
        redacted.set_item("contact_email", email_count)?;
    }
    if phone_count > 0 {
        redacted.set_item("contact_phone", phone_count)?;
    }
    if key_counts.iter().sum::<usize>() > 0 {
        redacted.set_item("api_keys", key_counts.iter().sum::<usize>())?;
    }
    for (i, &c) in key_counts.iter().enumerate() {
        if c > 0 {
            redacted.set_item(KeyFamily::ALL[i].name(), c)?;
        }
    }
    let skipped = PyDict::new(py);
    for (i, &c) in skipped_counts.iter().enumerate() {
        if c > 0 {
            skipped.set_item(KeyFamily::ALL[i].name(), c)?;
        }
    }
    let span_dicts: Vec<Bound<'_, PyDict>> = char_spans
        .iter()
        .map(|(kind, start, end)| {
            let span = PyDict::new(py);
            span.set_item("type", kind)?;
            span.set_item("start", start)?;
            span.set_item("end", end)?;
            Ok(span)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let spans = PyList::new(py, span_dicts)?;
    let report = PyDict::new(py);
    report.set_item("text", out_text)?;
    report.set_item("redacted", redacted)?;
    report.set_item("skipped", skipped)?;
    report.set_item("spans", spans)?;
    Ok(report.into_any().unbind())
}

/// Render the report's byte-offset spans as `(type, char_start,
/// char_end)`: one linear walk over the input's `char_indices`,
/// consuming the sorted unique byte offsets — O(input + spans), never
/// O(input × spans). Every span boundary the scanners emit sits on a
/// char boundary (matches start and end at ASCII-adjacent positions);
/// a mid-char offset would map to the following char, defensively.
fn render_char_spans(text: &str, spans: &[pii_impl::ReportSpan]) -> Vec<(String, usize, usize)> {
    let mut offsets: Vec<usize> = spans.iter().flat_map(|s| [s.start, s.end]).collect();
    offsets.sort_unstable();
    offsets.dedup();
    let mut char_at: Vec<usize> = Vec::with_capacity(offsets.len());
    let mut chars = text.char_indices();
    let mut ahead = chars.next();
    let mut char_idx = 0usize;
    for &b in &offsets {
        while let Some((cb, _)) = ahead {
            if cb >= b {
                break;
            }
            char_idx += 1;
            ahead = chars.next();
        }
        char_at.push(char_idx);
    }
    let index_of = |b: usize| {
        char_at[offsets
            .binary_search(&b)
            .expect("every span offset was collected")]
    };
    spans
        .iter()
        .map(|s| {
            let kind = match s.kind {
                SpanKind::Email => "contact_email".to_string(),
                SpanKind::Phone => "contact_phone".to_string(),
                SpanKind::Key(f) => format!("api_keys:{}", f.name()),
            };
            (kind, index_of(s.start), index_of(s.end))
        })
        .collect()
}
