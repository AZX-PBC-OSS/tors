//! The pyo3 bindings for the JSON repair surface: `tors.repair_json`,
//! `tors.repair_json_loads`, and `tors.repair_json_diagnostics` over
//! `crate::json_repair`'s pure-Rust core (the json_repair port; see that
//! module tree's docs for the behavior contract and the documented
//! divergences).
//!
//! GIL model: one detached native pass for the whole repair (strict fast
//! path, repair parser, schema repairer, and the jsonschema-crate
//! validation all run under `py.detach`); the GIL-held residue is the
//! schema argument walk (O(schema) handles, the standard str-in borrow
//! class over each dict/list entry) and the return marshalling: O(output)
//! string marshalling for `repair_json`, O(result) object-tree
//! construction for the loads/diagnostics spellings (the `word_bounds`
//! list-marshalling class), plus O(diagnostics) dict construction for the
//! diagnostics flavor.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::json_repair::DEADLINE_TAG;
use crate::json_repair::{self, Diagnostic, NumericLocale, RepairConfig, Value};
use crate::py::_borrow::timeout_err;
use crate::validate_deadline_ms;

/// Map a `repair()` error string to the right Python exception: a
/// `DEADLINE_TAG`-prefixed payload is a deadline abort -> `TimeoutError`
/// carrying the called spelling's own name, in the same wording as
/// `diff_opcodes`' TimeoutError; anything else is the normal `ValueError`.
fn map_repair_err(message: String, name: &str) -> PyErr {
    match message.strip_prefix(DEADLINE_TAG) {
        Some(rest) => timeout_err(format!("{name} deadline exceeded:{rest}")),
        None => PyValueError::new_err(message),
    }
}

/// The schema-side depth cap: Python-side nesting in a schema dict can be
/// arbitrary (it is caller data, not parser output), so the walk is capped
/// at the same threshold the schema layer enforces, with upstream's
/// normalized message.
const MAX_SCHEMA_WALK_DEPTH: usize = 200;

/// A Python object -> [`Value`] walk for the `schema=` argument (and only
/// it: the JSON under repair arrives as `str`). Accepts exactly the JSON
/// types a schema is made of (`dict` (insertion order preserved), `list`
/// and `tuple` (json.dumps' own tuple tolerance), `str`, `int`, `float`,
/// `bool`, `None`), with `bool` checked before `int` (Python `bool` is an
/// `int` subclass; the JSON meaning differs). `int` beyond i64 becomes the
/// unbounded-int spelling via its decimal text. A non-JSON type at the top
/// level gets the catalog's schema-rejection message; nested non-JSON
/// types name the JSON types instead. A `str` holding lone surrogates
/// fails the UTF-8 borrow (the standard str-in class: such strings cannot
/// reach any tors function).
fn py_to_value(_py: Python<'_>, obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<Value> {
    if depth > MAX_SCHEMA_WALK_DEPTH {
        return Err(PyValueError::new_err(
            "Input schema nesting exceeds the supported schema recursion depth.",
        ));
    }
    if let Ok(boolean) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(boolean.is_true()));
    }
    if let Ok(text) = obj.cast::<PyString>() {
        return Ok(Value::Str(text.to_str()?.to_owned()));
    }
    if let Ok(number) = obj.cast::<PyInt>() {
        // The common case is i64; a Python int beyond that range keeps its
        // exact value through the decimal text (the unbounded-int spelling).
        if let Ok(small) = number.extract::<i64>() {
            return Ok(Value::Int(small));
        }
        let text = number.str()?.to_str()?.to_owned();
        return Ok(Value::BigInt(normalize_int_text(&text)));
    }
    if let Ok(number) = obj.cast::<PyFloat>() {
        return Ok(Value::Float(number.value()));
    }
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut entries = Vec::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let key = key
                .cast::<PyString>()
                .map_err(|_| PyValueError::new_err("Object keys must be strings."))?;
            entries.push((
                key.to_str()?.to_owned(),
                py_to_value(_py, &value, depth + 1)?,
            ));
        }
        return Ok(Value::Object(entries));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_value(_py, &item, depth + 1)?);
        }
        return Ok(Value::Array(items));
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_to_value(_py, &item, depth + 1)?);
        }
        return Ok(Value::Array(items));
    }
    if depth == 0 {
        Err(PyValueError::new_err(
            "schema must be a JSON Schema dict, boolean schema, or pydantic v2 model.",
        ))
    } else {
        Err(PyValueError::new_err(
            "schema must contain only JSON values (dict, list, str, int, \
             float, bool, None).",
        ))
    }
}

/// Normalize a Python int's `str()` text into the BigInt spelling: strip
/// whitespace, drop a leading `+`, and collapse `-0` to `0` (leading zeros
/// cannot occur in a Python int's repr).
fn normalize_int_text(text: &str) -> String {
    let text = text.trim();
    let (sign, digits) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text.strip_prefix('+').unwrap_or(text)),
    };
    if digits.bytes().all(|b| b == b'0') {
        return "0".to_string();
    }
    format!("{sign}{digits}")
}

/// The int class for values past i64: pyo3 has no arbitrary-precision
/// constructor, so the decimal text goes through the int builtin itself
/// (`int(text, 10)`: exact, no float detour).
fn bigint_to_py(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    let builtins = PyModule::import(py, "builtins")?;
    let constructor = builtins.getattr("int")?;
    Ok(constructor.call1((text, 10))?.unbind())
}

/// [`Value`] -> Python object tree (the loads/diagnostics return): `dict`
/// in insertion order, `list`, `str`, `bool`, `None`, `float`, `int`:
/// with the unbounded-int spelling reconstructed exactly. The repair core
/// never lets the MISSING_VALUE sentinel escape (the schema layer
/// normalizes it), so a defensive `""` render (its normalized form)
/// stands in for the unreachable case rather than a panic.
fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(flag) => Ok(PyBool::new(py, *flag).to_owned().into_any().unbind()),
        Value::Int(number) => Ok(number.into_pyobject(py)?.into_any().unbind()),
        Value::BigInt(text) => bigint_to_py(py, text),
        Value::Float(number) => Ok(number.into_pyobject(py)?.into_any().unbind()),
        Value::Str(text) => Ok(text.into_pyobject(py)?.into_any().unbind()),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, value_to_py(py, item)?)?;
            }
            Ok(dict.into_any().unbind())
        }
        // Unreachable through repair() (the schema layer normalizes it);
        // its normalized form is the empty string.
        Value::Missing => Ok("".into_pyobject(py)?.into_any().unbind()),
    }
}

/// One diagnostics entry -> `dict`, with all six keys present every time
/// (`from`/`to`/`suggestion` are `None` when the action did not move a
/// value or offer a hint): a stable shape consumers can index blindly.
fn diagnostic_to_py(py: Python<'_>, diagnostic: &Diagnostic) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("action", diagnostic.action.clone())?;
    dict.set_item("path", diagnostic.path.clone())?;
    dict.set_item("detail", diagnostic.detail.clone())?;
    match &diagnostic.from {
        Some(value) => dict.set_item("from", value_to_py(py, value)?)?,
        None => dict.set_item("from", py.None())?,
    }
    match &diagnostic.to {
        Some(value) => dict.set_item("to", value_to_py(py, value)?)?,
        None => dict.set_item("to", py.None())?,
    }
    match &diagnostic.suggestion {
        Some(hint) => dict.set_item("suggestion", hint.clone())?,
        None => dict.set_item("suggestion", py.None())?,
    }
    Ok(dict.into_any().unbind())
}

/// Resolve the `schema=` argument to a plain JSON `Value`: pydantic v2
/// models (anything exposing `model_json_schema()`) are converted through
/// their generated JSON Schema dict first, so the caller's round-trip
/// (model -> schema -> LLM -> repair -> `Model.model_validate`) needs no
/// manual `.model_json_schema()` step. Plain dicts/bools pass straight
/// through to the JSON walk.
fn resolve_schema_arg<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if obj.hasattr("model_json_schema")? {
        // Upstream's `schema_from_input` requires pydantic v2 for models.
        // A duck-typed non-pydantic object with the same attribute fails
        // the import or the version gate and lands in the catalog error,
        // never a ModuleNotFoundError.
        let pydantic = PyModule::import(py, "pydantic").map_err(|_| {
            PyValueError::new_err(
                "schema must be a JSON Schema dict, boolean schema, or pydantic v2 model.",
            )
        })?;
        let version = pydantic
            .getattr("VERSION")
            .ok()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default();
        let major = version
            .split('.')
            .next()
            .and_then(|n| n.parse::<u32>().ok())
            .unwrap_or(0);
        if major < 2 {
            return Err(PyValueError::new_err(
                "pydantic v2 is required for schema models.",
            ));
        }
        let schema = obj.getattr("model_json_schema")?.call0()?;
        inject_model_defaults(py, obj, &schema)?;
        return Ok(schema);
    }
    Ok(obj.clone())
}

/// Upstream's `schema_from_input` pydantic branch: fields with defaults
/// (or default factories) get their `"default"` injected into the
/// generated property schemas, so the repairer's missing-value fills and
/// optional-default insertions see them. Operates on the fresh dict
/// `model_json_schema()` returned (never caller-owned data).
fn inject_model_defaults(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let Ok(fields) = model.getattr("model_fields") else {
        return Ok(());
    };
    let Ok(properties) = schema
        .getattr("get")
        .and_then(|get| get.call1(("properties",)))
    else {
        return Ok(());
    };
    let Ok(fields_dict) = fields.cast::<PyDict>() else {
        return Ok(());
    };
    let Ok(properties_dict) = properties.cast::<PyDict>() else {
        return Ok(());
    };
    for (name, field) in fields_dict.iter() {
        // Required fields carry no default to inject.
        let required = field
            .getattr("is_required")
            .and_then(|check| check.call0())
            .and_then(|flag| flag.extract::<bool>())
            .unwrap_or(true);
        if required {
            continue;
        }
        let Ok(property) = properties_dict.get_item(&name) else {
            continue;
        };
        let Some(property) = property else { continue };
        let Ok(property_dict) = property.cast::<PyDict>() else {
            continue;
        };
        if property_dict.contains("default")? {
            continue;
        }
        // Upstream's exact order: the factory first (a factory field's
        // `default` is the PydanticUndefined sentinel, never a real value),
        // then the plain default. A factory needing arguments raises here:
        // surfacing the misconfiguration instead of silently skipping.
        if let Ok(factory) = field.getattr("default_factory")
            && !factory.is_none()
        {
            let made = factory.call0()?;
            property_dict.set_item("default", made)?;
        } else if let Ok(default) = field.getattr("default") {
            // Enum members travel as the member; JSON wants the value.
            let default = enum_member_value(py, default)?;
            property_dict.set_item("default", default)?;
        }
    }
    Ok(())
}

/// `enum.Enum` members as their `.value` (the JSON spelling pydantic's own
/// schema emission uses elsewhere); every other object passes through.
fn enum_member_value<'py>(py: Python<'py>, obj: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let enum_mod = PyModule::import(py, "enum")?;
    let enum_type = enum_mod.getattr("Enum")?;
    let is_member = enum_type
        .getattr("__instancecheck__")
        .and_then(|check| check.call1((&obj,)))
        .and_then(|flag| flag.extract::<bool>())
        .unwrap_or(false);
    if is_member {
        return obj.getattr("value");
    }
    Ok(obj)
}

/// Resolve the `locale=` argument: a BCP 47 tag string ("en-US",
/// "de-DE", case-insensitive, `-` or `_`) through the CLDR table, or a
/// dict `{"decimal": ",", "grouping": "."}` of single-character
/// separators for conventions the table does not carry. `None` selects
/// Auto: assume en-US for the separator-ambiguous shapes, but only
/// schema-checked and disclosed (never a silent 1000x guess); shapes no
/// locale could resolve refuse with the retry-able hint.
fn resolve_locale_arg(obj: &Bound<'_, PyAny>) -> PyResult<NumericLocale> {
    if let Ok(tag) = obj.extract::<String>() {
        return match crate::json_schema_impl::locale_spec_from_tag(&tag) {
            Some(spec) => Ok(NumericLocale::Known(spec)),
            None => Err(PyValueError::new_err(
                "locale must be a supported BCP 47 tag like 'en-US' or 'de-DE' \
                 (Lakh-style grouping, e.g. en-IN, is not supported)",
            )),
        };
    }
    let dict = obj.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(
            "locale must be a BCP 47 tag string or a \
                 {'decimal': ..., 'grouping': ...} dict of single-character separators",
        )
    })?;
    let single_char = |key: &str| -> PyResult<char> {
        let value = dict
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("locale dict needs a '{key}' key")))?;
        let text = value.extract::<String>().map_err(|_| {
            PyValueError::new_err(format!("locale '{key}' must be a one-character string"))
        })?;
        let mut chars = text.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) => Ok(c),
            _ => Err(PyValueError::new_err(format!(
                "locale '{key}' must be exactly one character, not {text:?}"
            ))),
        }
    };
    let decimal = single_char("decimal")?;
    let grouping = single_char("grouping")?;
    Ok(NumericLocale::Known(
        crate::json_schema_impl::LocaleSpec::custom(decimal, grouping),
    ))
}

/// The shared `schema=` argument walk + config assembly for all three
/// spellings: the GIL-held schema walk happens here, then the whole repair
/// runs detached with the assembled config.
#[allow(clippy::too_many_arguments)]
fn build_config(
    py: Python<'_>,
    skip_json_loads: bool,
    strict: bool,
    salvage: bool,
    diagnostics: bool,
    schema: Option<&Bound<'_, PyAny>>,
    locale: Option<&Bound<'_, PyAny>>,
    deadline_ms: Option<f64>,
) -> PyResult<RepairConfig> {
    let schema_value = match schema {
        None => None,
        Some(obj) if obj.is_none() => None,
        Some(obj) => {
            let resolved = resolve_schema_arg(py, obj)?;
            // The catalog gate: only dicts, booleans (and models, already
            // resolved above) cross into the repair. A JSON scalar like 42
            // would otherwise convert cleanly and fail later with the wrong
            // message.
            if resolved.cast::<PyDict>().is_err() && resolved.cast::<PyBool>().is_err() {
                return Err(PyValueError::new_err(
                    "schema must be a JSON Schema dict, boolean schema, or pydantic v2 model.",
                ));
            }
            Some(py_to_value(py, &resolved, 0)?)
        }
    };
    let locale = match locale {
        None => NumericLocale::Auto,
        Some(obj) if obj.is_none() => NumericLocale::Auto,
        Some(obj) => resolve_locale_arg(obj)?,
    };
    Ok(RepairConfig {
        skip_json_loads,
        strict,
        salvage,
        schema: schema_value,
        diagnostics,
        locale,
        deadline_ms,
    })
}

/// `tors.repair_json`: the repaired JSON as a string: a drop-in spelling
/// for the "give me valid JSON back" pipeline, re-serialized to
/// `json.dumps`' canonical form exactly like upstream (so valid-but-
/// noncanonical input normalizes; `ensure_ascii=False` keeps non-ASCII
/// verbatim). The nothing-recoverable sentinel renders as the bare empty
/// string, upstream's own convention.
#[pyfunction(signature = (s, *, skip_json_loads = false, ensure_ascii = true, strict = false, schema = None, salvage = false, locale = None, deadline_ms = None))]
#[allow(clippy::too_many_arguments)]
pub fn repair_json(
    py: Python<'_>,
    s: &str,
    skip_json_loads: bool,
    ensure_ascii: bool,
    strict: bool,
    schema: Option<Bound<'_, PyAny>>,
    salvage: bool,
    locale: Option<Bound<'_, PyAny>>,
    deadline_ms: Option<f64>,
) -> PyResult<String> {
    validate_deadline_ms(deadline_ms)?;
    let cfg = build_config(
        py,
        skip_json_loads,
        strict,
        salvage,
        false,
        schema.as_ref(),
        locale.as_ref(),
        deadline_ms,
    )?;
    py.detach(|| match json_repair::repair(s, &cfg) {
        // The sentinel renders as the bare empty string, not '""'.
        Ok((Value::Str(empty), _)) if empty.is_empty() => Ok(String::new()),
        Ok((value, _)) => Ok(json_repair::dumps(&value, ensure_ascii)),
        Err(message) => Err(message),
    })
    .map_err(|e| map_repair_err(e, "repair_json"))
}

/// `tors.repair_json_loads`: the repaired JSON as decoded objects: the
/// `json.loads` drop-in. Returns `""` (the empty string, not `None`) when
/// nothing is recoverable, exactly like upstream. The GIL-held residue is
/// the O(result) object-tree construction (the `word_bounds`
/// list-marshalling class).
#[pyfunction(signature = (s, *, skip_json_loads = false, strict = false, schema = None, salvage = false, locale = None, deadline_ms = None))]
#[allow(clippy::too_many_arguments)]
pub fn repair_json_loads(
    py: Python<'_>,
    s: &str,
    skip_json_loads: bool,
    strict: bool,
    schema: Option<Bound<'_, PyAny>>,
    salvage: bool,
    locale: Option<Bound<'_, PyAny>>,
    deadline_ms: Option<f64>,
) -> PyResult<Py<PyAny>> {
    validate_deadline_ms(deadline_ms)?;
    let cfg = build_config(
        py,
        skip_json_loads,
        strict,
        salvage,
        false,
        schema.as_ref(),
        locale.as_ref(),
        deadline_ms,
    )?;
    let value = py
        .detach(|| json_repair::repair(s, &cfg))
        .map_err(|e| map_repair_err(e, "repair_json_loads"))?;
    value_to_py(py, &value.0)
}

/// `tors.repair_json_diagnostics`: the loads-mode value plus the structured
/// action log: a list of `{action, path, detail, from, to, suggestion}`
/// dicts (the last three `None` when unused) covering every schema-layer
/// repair, coercion, fill, drop, remap, and tors-native suggestion. The v1
/// scope note: schema-free calls return an empty list (parser-level
/// narration is a follow-up); see the docs for the action vocabulary.
#[pyfunction(signature = (s, *, skip_json_loads = false, strict = false, schema = None, salvage = false, locale = None, deadline_ms = None))]
#[allow(clippy::too_many_arguments)]
pub fn repair_json_diagnostics(
    py: Python<'_>,
    s: &str,
    skip_json_loads: bool,
    strict: bool,
    schema: Option<Bound<'_, PyAny>>,
    salvage: bool,
    locale: Option<Bound<'_, PyAny>>,
    deadline_ms: Option<f64>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    validate_deadline_ms(deadline_ms)?;
    let cfg = build_config(
        py,
        skip_json_loads,
        strict,
        salvage,
        true,
        schema.as_ref(),
        locale.as_ref(),
        deadline_ms,
    )?;
    let (value, diagnostics) = py
        .detach(|| json_repair::repair(s, &cfg))
        .map_err(|e| map_repair_err(e, "repair_json_diagnostics"))?;
    let value = value_to_py(py, &value)?;
    let list = PyList::empty(py);
    for diagnostic in &diagnostics {
        list.append(diagnostic_to_py(py, diagnostic)?)?;
    }
    Ok((value, list.into_any().unbind()))
}
