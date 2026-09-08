//! The parenthesized-value parser: a port of json_repair's
//! `parser_parenthesized.py` (upstream:
//! https://github.com/mangiucugna/json_repair by Stefano Baccianella, MIT,
//! pinned at commit 251d141786d0f6ec04d90188a338e2470, version 0.63.4) —
//! the two quote/bracket/backslash classifier state machines that decide
//! whether a `(` starts an explicit Python tuple literal or a standalone
//! top-level value, and the parse entry that reuses the array loop with
//! `)` as the closer (the scalar-unwrap rule for single grouped values).
//! Every branch is ported 1:1 in upstream's order of checks.
//!
//! # Porting notes (the decisions this file's dynamics forced)
//!
//! - The classifier counters are `isize` where upstream decrements
//!   unconditionally (`top_level_parenthesized_can_start_value`'s `)` arm
//!   can drive `nested_parentheses` negative when the `)` sits inside
//!   brackets — plain Python ints upstream); the guarded decrements in
//!   `parenthesized_is_explicit_tuple` never go negative but use the same
//!   type for symmetry.
//! - `str.isdigit()` uses `char::is_ascii_digit` (documented divergence
//!   §9.3: Python's isdigit accepts non-ASCII Nd digits); `str.isspace`
//!   uses `crate::normalize_impl::is_py_whitespace`; `.lower()` over the
//!   inner-text tail is per-char `to_lowercase`.
//! - The classifiers are read-only scans (`&self`); no context pushes live
//!   here (the ARRAY context comes from the `parse_array` the parse entry
//!   calls, and the depth guard from `parse_json`'s `(` branch).
//!
//! # Test provenance
//!
//! The `#[cfg(test)]` battery ports the VALUE-level assertions of
//! upstream's `tests/test_parse_array.py` that exercise
//! `parser_parenthesized.py` (the Python tuple literals, the
//! boolean/null-valued tuples, and the grouped-scalar unwrap cases, driven
//! through `Parser::parse` — what `repair_json(..., skip_json_loads=True,
//! return_objects=True)` runs), plus the two direct classifier batteries.
//! The assertions that intentionally live ONLY in the pytest corpus
//! (`tests/test_json_repair.py`, agent C's): the serialized-STRING forms of
//! these inputs and every case whose behavior belongs to
//! `string.rs`/`parser.rs`/`object.rs`/`array.rs`'s own batteries.

use super::parser::Parser;
use super::{STRING_DELIMITERS, Value};
use crate::normalize_impl::is_py_whitespace;

impl Parser {
    /// parser_parenthesized.py's `parenthesized_is_explicit_tuple`: does the
    /// `(` at the cursor start an explicit Python tuple literal? Empty
    /// parentheses count as a tuple; a single grouped value like `(1)` does
    /// not; content at depth zero means "not the tuple's own separator".
    pub(crate) fn parenthesized_is_explicit_tuple(&self) -> bool {
        let mut i = self.index + 1;
        let n = self.s.len();
        let mut nested_parentheses: isize = 0;
        let mut square_brackets: isize = 0;
        let mut braces: isize = 0;
        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;
        let mut saw_top_level_content = false;

        while i < n {
            let ch = self.s[i];

            if ch == '\\' {
                backslashes += 1;
                i += 1;
                continue;
            }

            if let Some(quote) = in_quote {
                if ch == quote && backslashes.is_multiple_of(2) {
                    in_quote = None;
                }
                backslashes = 0;
                i += 1;
                continue;
            }

            if STRING_DELIMITERS.contains(&ch) && backslashes.is_multiple_of(2) {
                in_quote = Some(ch);
                // A quote at depth zero is content: '()' and '("")' differ.
                saw_top_level_content |=
                    nested_parentheses == 0 && square_brackets == 0 && braces == 0;
                backslashes = 0;
                i += 1;
                continue;
            }

            backslashes = 0;

            if !is_py_whitespace(ch)
                && ch != ','
                && ch != ')'
                && nested_parentheses == 0
                && square_brackets == 0
                && braces == 0
            {
                saw_top_level_content = true;
            }

            if ch == '(' {
                nested_parentheses += 1;
            } else if ch == ')' {
                if nested_parentheses == 0 && square_brackets == 0 && braces == 0 {
                    // The group closes: a tuple iff no content preceded it.
                    return !saw_top_level_content;
                }
                if nested_parentheses > 0 {
                    nested_parentheses -= 1;
                }
            } else if ch == '[' {
                square_brackets += 1;
            } else if ch == ']' && square_brackets > 0 {
                square_brackets -= 1;
            } else if ch == '{' {
                braces += 1;
            } else if ch == '}' && braces > 0 {
                braces -= 1;
            } else if ch == ',' && nested_parentheses == 0 && square_brackets == 0 && braces == 0 {
                // A top-level comma IS the tuple separator.
                return true;
            }

            i += 1;
        }

        // Never closed: a tuple iff nothing was inside.
        !saw_top_level_content
    }

    /// parser_parenthesized.py's `top_level_parenthesized_can_start_value`:
    /// does a top-level `(` look like a standalone value rather than inline
    /// prose? Keeps tuple support for direct inputs while avoiding
    /// regressions on surrounding text like `foo (clarification): {...}`.
    pub(crate) fn top_level_parenthesized_can_start_value(&self) -> bool {
        // Only whitespace between the '(' and the start of its line.
        let mut i: isize = self.index as isize - 1;
        while i >= 0 {
            let ch = self.s[i as usize];
            if matches!(ch, '\n' | '\r') {
                break;
            }
            if !is_py_whitespace(ch) {
                return false;
            }
            i -= 1;
        }

        let idx = self.scroll_whitespaces(1);
        let first_inner_char = self.get(idx as isize);
        let Some(first_inner_char) = first_inner_char else {
            return false;
        };

        // inner_text: the lowercased tail from index + idx.
        let inner_start = self.index + idx;
        let inner_text: String = self.s[inner_start..]
            .iter()
            .copied()
            .flat_map(char::to_lowercase)
            .collect();
        let starts_like_value = matches!(first_inner_char, ')' | '{' | '[' | '(')
            || STRING_DELIMITERS.contains(&first_inner_char)
            // Python isdigit: ASCII digits only here (documented divergence
            // §9.3).
            || first_inner_char.is_ascii_digit()
            || matches!(first_inner_char, '-' | '.');
        let starts_like_literal = {
            let prefix4: String = inner_text.chars().take(4).collect();
            let prefix5: String = inner_text.chars().take(5).collect();
            prefix4 == "true" || prefix4 == "null" || prefix4 == "none" || prefix5 == "false"
        };
        if !starts_like_value && !starts_like_literal {
            return false;
        }

        let mut i = self.index + 1;
        let n = self.s.len();
        let mut nested_parentheses: isize = 0;
        let mut square_brackets: isize = 0;
        let mut braces: isize = 0;
        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;

        while i < n {
            let ch = self.s[i];

            if ch == '\\' {
                backslashes += 1;
                i += 1;
                continue;
            }

            if let Some(quote) = in_quote {
                if ch == quote && backslashes.is_multiple_of(2) {
                    in_quote = None;
                }
                backslashes = 0;
                i += 1;
                continue;
            }

            if STRING_DELIMITERS.contains(&ch) && backslashes.is_multiple_of(2) {
                in_quote = Some(ch);
                backslashes = 0;
                i += 1;
                continue;
            }

            backslashes = 0;

            if ch == '(' {
                nested_parentheses += 1;
            } else if ch == ')' {
                if nested_parentheses == 0 && square_brackets == 0 && braces == 0 {
                    // The group closes at the top level: a standalone value
                    // iff only whitespace (or a line end) follows it.
                    i += 1;
                    while i < n {
                        let trailer = self.s[i];
                        if matches!(trailer, '\n' | '\r') {
                            return true;
                        }
                        if !is_py_whitespace(trailer) {
                            return false;
                        }
                        i += 1;
                    }
                    return true;
                }
                // Upstream decrements unconditionally here — a ')' inside
                // brackets can drive the counter negative (plain ints
                // upstream, isize here).
                nested_parentheses -= 1;
            } else if ch == '[' {
                square_brackets += 1;
            } else if ch == ']' && square_brackets > 0 {
                square_brackets -= 1;
            } else if ch == '{' {
                braces += 1;
            } else if ch == '}' && braces > 0 {
                braces -= 1;
            }

            i += 1;
        }

        true
    }

    /// json_parser.py's `parse_parenthesized` (the parser_parenthesized.py
    /// entry): skip the `(`, parse the group with the array loop under `)`
    /// as the closer, keep the array for explicit tuples (and multi-item
    /// groups), and unwrap a single grouped value to the value itself.
    pub(crate) fn parse_parenthesized(
        &mut self,
        schema: Option<&Value>,
        path: &str,
    ) -> Result<Value, String> {
        let explicit_tuple = self.parenthesized_is_explicit_tuple();
        self.index += 1;
        let values = self.parse_array(schema, path, ')')?;
        if explicit_tuple {
            return Ok(values);
        }
        match values {
            // `values[0]`: the len-1 guard makes next() total (the
            // fallback is inert).
            Value::Array(items) if items.len() == 1 => Ok(items
                .into_iter()
                .next()
                .unwrap_or(Value::Str(String::new()))),
            values => Ok(values),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(raw: &str) -> Value {
        Parser::new(raw, false, None)
            .parse()
            .expect("plain parse should not fail")
    }

    fn s(text: &str) -> Value {
        Value::Str(text.to_string())
    }

    fn obj(entries: &[(&str, Value)]) -> Value {
        Value::Object(
            entries
                .iter()
                .map(|(key, value)| (key.to_string(), value.clone()))
                .collect(),
        )
    }

    fn a(items: Vec<Value>) -> Value {
        Value::Array(items)
    }

    #[test]
    fn parse_array_python_tuple_literals() {
        // test_parse_array.py::test_parse_array_python_tuple_literals
        assert_eq!(
            parse_ok(r#"("a", "b", "c")"#),
            a(vec![s("a"), s("b"), s("c")])
        );
        assert_eq!(
            parse_ok("((1, 2), (3, 4))"),
            a(vec![
                a(vec![Value::Int(1), Value::Int(2)]),
                a(vec![Value::Int(3), Value::Int(4)]),
            ])
        );
        assert_eq!(
            parse_ok(r#"{"coords": (1, 2), "ok": true}"#),
            obj(&[
                ("coords", a(vec![Value::Int(1), Value::Int(2)])),
                ("ok", Value::Bool(true)),
            ])
        );
        assert_eq!(parse_ok(r#"{"empty": ()}"#), obj(&[("empty", a(vec![]))]));
    }

    #[test]
    fn parse_array_python_tuple_literals_accept_boolean_and_null_values() {
        // test_parse_array.py::test_parse_array_python_tuple_literals_accept_boolean_and_null_values
        assert_eq!(
            parse_ok("(true, false, null)"),
            a(vec![Value::Bool(true), Value::Bool(false), Value::Null])
        );
        assert_eq!(
            parse_ok("(True, False, None)"),
            a(vec![Value::Bool(true), Value::Bool(false), Value::Null])
        );
        assert_eq!(
            parse_ok(r#"{"coords": (True, None)}"#),
            obj(&[("coords", a(vec![Value::Bool(true), Value::Null]))])
        );
    }

    #[test]
    fn parse_array_parenthesized_scalar_keeps_scalar_shape() {
        // test_parse_array.py::test_parse_array_parenthesized_scalar_keeps_scalar_shape
        assert_eq!(parse_ok("(1)"), Value::Int(1));
        assert_eq!(parse_ok(r#"("x")"#), s("x"));
        assert_eq!(
            parse_ok(r#"{"scalar_group": (1)}"#),
            obj(&[("scalar_group", Value::Int(1))])
        );
        assert_eq!(
            parse_ok(r#"{"string_group": ("x")}"#),
            obj(&[("string_group", s("x"))])
        );
    }

    #[test]
    fn parenthesized_tuple_classifier_handles_nested_delimiters_and_missing_close() {
        // test_parse_array.py::test_parenthesized_tuple_classifier_handles_nested_delimiters_and_missing_close
        let parser = Parser::new(r#"({"text": "a\b", "items": [1]})"#, false, None);
        assert!(!parser.parenthesized_is_explicit_tuple());

        let parser = Parser::new("(1", false, None);
        assert!(!parser.parenthesized_is_explicit_tuple());
    }

    #[test]
    fn top_level_parenthesized_value_gate_rejects_prose_and_accepts_jsonish_values() {
        // test_parse_array.py::test_top_level_parenthesized_value_gate_rejects_prose_and_accepts_standalone_jsonish_values
        let parser = Parser::new("(", false, None);
        assert!(!parser.top_level_parenthesized_can_start_value());

        let parser = Parser::new("(note)\n{\"key\": 1}", false, None);
        assert!(!parser.top_level_parenthesized_can_start_value());

        let parser = Parser::new("(1", false, None);
        assert!(parser.top_level_parenthesized_can_start_value());

        let parser = Parser::new(
            r#"(["a\b"], {"k": 1})
"#,
            false,
            None,
        );
        assert!(parser.top_level_parenthesized_can_start_value());
    }
}
