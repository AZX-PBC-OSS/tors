use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::{Py, PyAny};

use crate::detached_transform;
use crate::py::_borrow::bounded_str_list;
use crate::secret_impl::{self, SecretKind, SecretRules};

/// The `rules=` parameter's accepted spellings, the same
/// closed-set-of-strings convention as `scrub_log_text`'s (anything
/// else is a `ValueError` naming the accepted set). `None` is every
/// grammar; duplicates dedupe and caller order is irrelevant, so the
/// parse folds the names straight into the rule bits.
fn parse_secret_rules(rules: Option<Vec<String>>) -> PyResult<SecretRules> {
    let Some(names) = rules else {
        return Ok(SecretRules::ALL);
    };
    let mut set = SecretRules::EMPTY;
    for name in &names {
        match name.as_str() {
            "aws_access_key" => set |= SecretRules::AWS_ACCESS_KEY,
            "slack_token" => set |= SecretRules::SLACK_TOKEN,
            "stripe_key" => set |= SecretRules::STRIPE_KEY,
            "github_token" => set |= SecretRules::GITHUB_TOKEN,
            "pem_key" => set |= SecretRules::PEM_KEY,
            other => {
                return Err(PyValueError::new_err(format!(
                    "rules must be one of ('aws_access_key', 'slack_token', \
                     'stripe_key', 'github_token', 'pem_key'), not {other:?}"
                )));
            }
        }
    }
    Ok(set)
}

/// `tors.scrub_secrets`: replace secret-token material — AWS access
/// keys, Slack tokens, Stripe keys, GitHub tokens, PEM private-key
/// blocks — inside free text with correlation tokens, in one
/// GIL-released pass. The token shape is the keys rule's own:
/// `<head>~<digest>` where the head is the match's non-secret prefix
/// verbatim (`AKIA`, `xoxb`, `sk_live_`, `ghp_`) or a constant when the
/// grammar has none (`github` for the legacy 40-hex class, `PEM` for
/// the block span), and `<digest>` is the first 12 hex chars of
/// `sha256(salt + match)`. `salt=None` resolves to tors's documented
/// default `"tors/scrub_secrets/v1"` tag (a third domain-separation
/// tag: a secret digest never aliases a `scrub_pii` digest at the
/// default settings); a KNOWN salt still leaves candidate-list
/// confirmation possible — the tokens are redaction, not
/// pseudonymization crypto.
///
/// The five grammars are cited vendor shapes (AWS's access-key-ID
/// example spelling, Slack's documented prefixes and section
/// structure, Stripe's six live/test prefixes, GitHub's token-format
/// post's five prefixes plus the pre-2021 40-hex class, RFC 7468 PEM
/// framing), each a hand-rolled single-pass scanner with anchored
/// boundaries; see `src/secret_impl.rs` for the full pins and the
/// per-grammar false-positive posture (prefix + length classes are
/// intentionally recall-biased; the legacy 40-hex class matches every
/// clean 40-char hex run, SHA-1s and git commit ids included).
///
/// `rules=None` runs every grammar; `rules=[]` is the identity; each
/// name selects that grammar. Identity-return contract:
/// `tors.scrub_secrets(s, rules) is s` exactly when no grammar matched
/// (tokens are fixed points, so scrubbing twice is the identity, a
/// stronger contract than `scrub_pii`'s single re-fire corner).
///
/// GIL model: `detached_transform`'s shape — the text borrow and the
/// `rules=`/`salt=` validation under the GIL, the whole scan (every
/// grammar walk, every splice, every digest) under one `py.detach`,
/// then either the identity return or the O(output) string
/// marshalling.
#[pyfunction(signature = (text, rules = None, *, salt = None))]
pub fn scrub_secrets(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    rules: Option<Bound<'_, PyAny>>,
    salt: Option<&str>,
) -> PyResult<Py<PyAny>> {
    // The list param extracts through the bounded manual walk
    // ([`bounded_str_list`]): pyo3's `Option<Vec<String>>` sizing from a
    // lying `__len__` was the uncatchable capacity-overflow class (the
    // same boundary every scrub here shares).
    let rules = match rules {
        None => None,
        Some(any) => {
            let mut out: Vec<String> = Vec::new();
            bounded_str_list("scrub_secrets", "rules", &any, |handle| {
                out.push(handle.extract::<&str>()?.to_owned());
                Ok(())
            })?;
            Some(out)
        }
    };
    let rules = parse_secret_rules(rules)?;
    let salt = salt.unwrap_or(secret_impl::SECRETS_DEFAULT_SALT);
    detached_transform(py, text, move |s| {
        secret_impl::scrub_secrets(s, rules, salt)
    })
}

/// `tors.scrub_secrets_report`: the same scrub as `tors.scrub_secrets`
/// for the same arguments (`report["text"] == scrub_secrets(...)`,
/// byte-exact), plus the accounting, `scrub_pii_report`'s shape with
/// all three keys present every time: `redacted` (per-kind counts,
/// lowercase names, absent kinds omitted: `aws_access_key`,
/// `slack_token`, `stripe_live`, `stripe_test`, `github_token`,
/// `github_legacy_token`, `pem_private_key` — the `stripe_key` and
/// `github_token` rules report their environments/classes separately,
/// so the report says WHICH Stripe credential to rotate) and `spans`
/// (ordered by start, codepoint indices into the INPUT text, each
/// `{"type": ..., "start": ..., "end": ...}`). The scan is one pass, so
/// the spans are input coordinates with no offset mapping. Empty input
/// is the empty accounting: `{"text": "", "redacted": {}, "spans": []}`.
///
/// GIL model: the same single detach as `scrub_secrets` — the whole
/// recorded pass (scan, splice, digests, the byte-to-codepoint span
/// rendering) under one `py.detach`, then the O(output + spans) dict
/// marshalling.
#[pyfunction(signature = (text, rules = None, *, salt = None))]
pub fn scrub_secrets_report(
    py: Python<'_>,
    text: Bound<'_, PyString>,
    rules: Option<Bound<'_, PyAny>>,
    salt: Option<&str>,
) -> PyResult<Py<PyAny>> {
    // The same bounded extraction as `scrub_secrets` (same walk, same
    // cap, the same refusal bytes): the report spelling is the scrub
    // plus accounting, and its boundary must be the scrub's boundary.
    let rules = match rules {
        None => None,
        Some(any) => {
            let mut out: Vec<String> = Vec::new();
            bounded_str_list("scrub_secrets_report", "rules", &any, |handle| {
                out.push(handle.extract::<&str>()?.to_owned());
                Ok(())
            })?;
            Some(out)
        }
    };
    let rules = parse_secret_rules(rules)?;
    let salt = salt.unwrap_or(secret_impl::SECRETS_DEFAULT_SALT);
    let s = text.to_str()?;
    let (out_text, kind_counts, char_spans) = py.detach(|| {
        let rep = secret_impl::scrub_secrets_report(s, rules, salt);
        let char_spans = render_char_spans(s, &rep.spans);
        (rep.text, rep.kind_counts, char_spans)
    });
    let redacted = PyDict::new(py);
    for (i, &c) in kind_counts.iter().enumerate() {
        if c > 0 {
            redacted.set_item(SecretKind::ALL[i].name(), c)?;
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
    report.set_item("spans", spans)?;
    Ok(report.into_any().unbind())
}

/// Render the report's byte-offset spans as `(type, char_start,
/// char_end)`: one linear walk over the input's `char_indices`,
/// consuming the sorted unique byte offsets (the same renderer the
/// `scrub_pii_report` binding runs). Every span boundary the scanner
/// emits sits on a char boundary (matches start and end at
/// ASCII-adjacent positions).
fn render_char_spans(text: &str, spans: &[secret_impl::SecretSpan]) -> Vec<(String, usize, usize)> {
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
            (
                s.kind.name().to_string(),
                index_of(s.start),
                index_of(s.end),
            )
        })
        .collect()
}
