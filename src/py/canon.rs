//! `tors.content_hash`: the pyo3 binding whose GIL-held walk materializes
//! an owned [`Canon`] tree, and whose one `py.detach` emits the canonical
//! form into SHA-256.
//!
//! # The walk (GIL-held, iterative, the standard arg-walk class)
//!
//! The walk visits every node of the object tree under the GIL,
//! materializing an owned [`Canon`] tree: one `String` copy per str (the
//! standard str-in borrow, `to_str`, plus the copy an owned tree needs),
//! one i64 read per fast-path int, one Python `repr` call per float or
//! big int. That is the call's GIL-held residue: O(tree), the same class
//! `find_patterns`' pattern-list walk and `repair_json`'s `schema=` walk
//! pay, scaled to the whole object; `tests/test_gil_release.py`'s
//! content_hash cell pins its band at 12 MiB. The alternative -- walking
//! under the detach -- is impossible by construction: every step is a
//! CPython API call, and the emitter plus the SHA-256 (the other half of
//! the work) already run detached, so the release covers the byte-heavy
//! half of the call.
//!
//! The walk is ITERATIVE (an explicit frame stack), not recursive: depth
//! costs heap, never the call stack, so any tree the interpreter can hold
//! walks clean (json.dumps itself `RecursionError`s on deep trees at a
//! version-dependent depth -- a documented divergence lane: tors accepts
//! deeper input than the stdlib spelling, deterministically).
//!
//! # Key handling: classify, sort, THEN coerce (json.dumps's own order)
//!
//! json.dumps with `sort_keys=True` sorts the dict's `(key, value)` items
//! BEFORE stringifying any key: all-int keys come out in numeric order
//! (`2` before `10`), and a mixed-type key set raises the sort's own
//! `TypeError` before any key or value is encoded. The walk reproduces
//! that order structurally: classification touches no Python API that can
//! fail and materializes nothing; the sort runs next (so its comparison
//! `TypeError` is the FIRST error a mixed-key dict can raise, exactly as
//! in json.dumps); coercion happens last, per key, in sorted order.
//!
//! Three sort paths:
//!
//! - **All-exact-str keys** (the overwhelmingly common shape): each key's
//!   UTF-8 is borrowed and the pairs sorted by bytes -- identical to
//!   Python's codepoint-order `str` comparison, because UTF-8 byte order
//!   IS codepoint order for the valid UTF-8 a `to_str` borrow can produce.
//! - **All exact int/bool keys inside i64**: a numeric i64 sort (bool as
//!   its 0/1 int value; a `True`/`1` pair cannot coexist as dict keys, so
//!   no tie is possible).
//! - **Everything else** (any float key, any big-int key, mixed
//!   int/float, `None` alongside others, and ANY str/int-SUBCLASS key):
//!   the sort is DELEGATED to CPython -- a `list` of the dict's own
//!   `(key, value)` pairs, sorted with `list.sort()`, the very items
//!   `json.dumps` itself sorts the same way. The permutation is read back
//!   by object identity, which is unambiguous (a dict cannot hold two
//!   entries with the same key AND value object). This is byte-exact
//!   parity by construction -- the same tuples, the same timsort -- which
//!   is what carries the corners no reimplementation would dare: NaN
//!   keys (an inconsistent comparator, where the output order is
//!   timsort's behavior, not a mathematical property; two distinct NaN
//!   objects legally coexist as dict keys), exact int/float cross-type
//!   comparison at 2**63-scale magnitudes, arbitrary-precision int keys,
//!   and subclass keys whose overridden rich comparison (`__lt__`,
//!   `__eq__`-lying equals that fall the tiebreak to the values)
//!   `json.dumps` honors and a numeric/byte sort would silently ignore --
//!   which is exactly why the fast paths above are gated on EXACT
//!   instances. The mixed-type comparison `TypeError` that delegation
//!   surfaces is json.dumps's own message, byte-identical, for free.
//!
//! # The spellings (Python's own, never reimplemented)
//!
//! Finite floats and big ints are materialized via Python's own `repr`
//! (`float.__repr__` / `int.__repr__`, the BASE type's -- json.dumps uses
//! `PyLong_Type.tp_repr`, so an int or float SUBCLASS dumps as its numeric
//! spelling, not the subclass's `__repr__`; verified against the running
//! interpreter, and matched by branching on `is_exact_instance_of`).
//! Non-finite floats use json's `allow_nan` literals (`NaN`, `Infinity`,
//! `-Infinity`), decided by the raw `f64` storage value (subclass-tolerant,
//! the same read the C encoder makes). Big ints beyond i64 go through
//! Python's `int`->`str`, so the interpreter's
//! `sys.set_int_max_str_digits` limit raises identically on both sides.
//! The i64 fast path (a plain storage read plus fixed-buffer decimal
//! digits at emission, no Python call, no per-int allocation) is the
//! measured choice: 4.8ns per int against 29.3ns for the repr call over a
//! million ints on the dev box (~6x, release build), and exact byte-parity
//! holds across the whole i64 range, so the fallback exists only for the
//! magnitudes beyond it.
//!
//! # The surrogate divergence (the crate-wide lane)
//!
//! A str holding lone surrogates (value or key) raises `UnicodeEncodeError`
//! ("surrogates not allowed") from the `to_str` borrow -- the same
//! boundary every str-in surface in this crate documents. json.dumps
//! ACCEPTS lone surrogates (it emits `\udXXX` escapes), a documented,
//! pinned divergence (`tests/test_content_hash.py`::
//! `TestSurrogateDivergence`).

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::canon_impl::Canon;

/// A dict key's classification: which coercion bucket it falls into.
/// Infallible to compute (a storage read or a type check, nothing that
/// can raise), and deliberately materializing nothing -- the sort must
/// run before any key is spelled, json.dumps's own order.
enum KeyKind {
    /// A `str` (exact or subclass): content borrowed and copied post-sort.
    Str,
    /// `True`/`False` (checked before `PyLong`: bool IS an int).
    Bool(bool),
    /// An int inside the i64 fast path's range.
    SmallInt(i64),
    /// An int beyond i64 (an `IntEnum` value or a plain big int): spelled
    /// via the BASE `int.__repr__` post-sort.
    BigInt,
    /// A float: the raw `f64` (for the finite check and the delegated
    /// sort); the spelling is materialized post-sort.
    Float(f64),
    /// `None`.
    Null,
    /// None of the coercible buckets: rejected at coercion time (after
    /// the sort), json.dumps's own error order for doubly-bad dicts.
    Unknown,
}

/// A classified dict key with its handle (the handle is what the delegated
/// sort compares and what the repr calls spell).
struct KeyEntry<'py> {
    handle: Bound<'py, PyAny>,
    kind: KeyKind,
}

/// The base-type `__repr__` handles, looked up once per call and reused
/// for every big int and float subclass (per-call `getattr` would be a
/// per-object cost on float-dense trees).
struct Reprs<'py> {
    long_repr: Bound<'py, PyAny>,
    float_repr: Bound<'py, PyAny>,
}

fn classify_key(key: &Bound<'_, PyAny>) -> KeyKind {
    if key.is_none() {
        KeyKind::Null
    } else if let Ok(b) = key.cast::<PyBool>() {
        KeyKind::Bool(b.is_true())
    } else if let Ok(long) = key.cast::<PyInt>() {
        // A storage read; only out-of-i64-range can fail, and that is
        // the BigInt bucket, not an error.
        match long.extract::<i64>() {
            Ok(v) => KeyKind::SmallInt(v),
            Err(_) => KeyKind::BigInt,
        }
    } else if let Ok(f) = key.cast::<PyFloat>() {
        KeyKind::Float(f.extract::<f64>().unwrap_or(f64::NAN))
    } else if key.cast::<PyString>().is_ok() {
        KeyKind::Str
    } else {
        KeyKind::Unknown
    }
}

/// The offending type's name, for the two house-worded TypeErrors.
fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .and_then(|name| name.to_str().map(str::to_owned))
        .unwrap_or_else(|_| "unknown".to_owned())
}

fn value_type_error(obj: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "content_hash() values must be str, int, float, bool, None, list, tuple, or dict, not {}",
        type_name(obj)
    ))
}

fn key_type_error(obj: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "content_hash() keys must be str, int, float, bool, or None, not {}",
        type_name(obj)
    ))
}

/// A big int's (or int subclass's) decimal spelling via the BASE
/// `int.__repr__`: json.dumps uses `PyLong_Type.tp_repr`, so the subclass
/// hook is bypassed exactly like the C encoder bypasses it.
fn spell_long(obj: &Bound<'_, PyAny>, reprs: &Reprs<'_>) -> PyResult<String> {
    let spelled = if obj.is_exact_instance_of::<PyInt>() {
        obj.repr()?
    } else {
        reprs.long_repr.call1((obj,))?.cast::<PyString>()?.clone()
    };
    Ok(spelled.to_str()?.to_owned())
}

/// A finite float's spelling via Python's own float repr (the base
/// `float.__repr__` for subclasses, mirroring the C encoder's observed
/// behavior: a `__repr__`-overriding float subclass still dumps as its
/// numeric spelling). Non-finite floats never reach here (the walk emits
/// json's `allow_nan` literals from the raw storage value).
fn spell_float(obj: &Bound<'_, PyAny>, reprs: &Reprs<'_>) -> PyResult<String> {
    let spelled = if obj.is_exact_instance_of::<PyFloat>() {
        obj.repr()?
    } else {
        reprs.float_repr.call1((obj,))?.cast::<PyString>()?.clone()
    };
    Ok(spelled.to_str()?.to_owned())
}

/// A leaf's [`Canon`], or `None` for the containers the frame machine
/// handles. The classification order mirrors json.dumps's C dispatch:
/// the `None`/`True`/`False` singletons first, then the scalar types
/// (bool before int, because bool IS an int), then str, then the
/// containers.
fn walk_leaf(obj: &Bound<'_, PyAny>, reprs: &Reprs<'_>) -> PyResult<Option<Canon>> {
    if obj.is_none() {
        return Ok(Some(Canon::Null));
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Some(Canon::Bool(b.is_true())));
    }
    if let Ok(long) = obj.cast::<PyInt>() {
        return Ok(Some(match long.extract::<i64>() {
            Ok(v) => Canon::Int(v),
            Err(_) => Canon::BigInt(spell_long(obj, reprs)?),
        }));
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        let v = f.extract::<f64>().unwrap_or(f64::NAN);
        let spelling = if v.is_finite() {
            spell_float(obj, reprs)?
        } else if v.is_nan() {
            "NaN".to_owned()
        } else if v > 0.0 {
            "Infinity".to_owned()
        } else {
            "-Infinity".to_owned()
        };
        return Ok(Some(Canon::Float(spelling)));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        // The standard str-in borrow: a lone-surrogate str fails here
        // with UnicodeEncodeError, the crate-wide divergence lane.
        return Ok(Some(Canon::Str(s.to_str()?.to_owned())));
    }
    if obj.cast::<PyDict>().is_ok() || obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok()
    {
        return Ok(None); // a container: the frame machine's job
    }
    Err(value_type_error(obj))
}

/// A dict's (coerced key, value handle) pairs, sorted per the module docs:
/// classify everything, sort (fast paths for all-str and all-small-int
/// keys, CPython's own timsort for everything else), then coerce.
fn dict_pairs<'py>(
    py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    reprs: &Reprs<'py>,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let mut entries: Vec<KeyEntry<'_>> = Vec::with_capacity(dict.len());
    let mut values: Vec<Bound<'_, PyAny>> = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let kind = classify_key(&key);
        entries.push(KeyEntry { handle: key, kind });
        values.push(value);
    }
    let n = entries.len();

    // The all-exact-str fast path: borrow and copy each key ONCE (the only
    // place a surrogate key can raise, after the sort-shape decision),
    // sort the pairs by UTF-8 bytes (== Python's codepoint-order str
    // comparison), done. A lone-surrogate key raises here for an all-str
    // dict -- the documented divergence -- while a MIXED-type dict never
    // reaches this path and raises the sort's comparison error first,
    // matching json.dumps's error order. EXACT str instances only: a str
    // SUBCLASS key may override rich comparison, which json.dumps's own
    // sort honors -- such keys take the delegated path below.
    if entries.iter().all(|e| matches!(e.kind, KeyKind::Str))
        && entries
            .iter()
            .all(|e| e.handle.is_exact_instance_of::<PyString>())
    {
        let mut pairs: Vec<(String, Bound<'_, PyAny>)> = Vec::with_capacity(n);
        for (entry, value) in entries.into_iter().zip(values) {
            let key = entry
                .handle
                .cast::<PyString>()
                .expect("classified Str")
                .to_str()?;
            pairs.push((key.to_owned(), value));
        }
        pairs.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        return Ok(pairs);
    }

    // The order the pairs will be coerced in. n <= 1 needs no comparison
    // (any single key is trivially sorted); the all-exact-int/bool fast
    // path is a numeric i64 sort (same exact-instance gate as the str
    // path, for the same overridden-comparison reason); everything else
    // delegates to CPython's timsort over the dict's own (key, value)
    // pairs -- json.dumps's own items sort, byte-exact by construction,
    // which also raises json.dumps's own comparison TypeError for mixed
    // unsortable key types.
    let order: Vec<usize> = if n <= 1 {
        (0..n).collect()
    } else if entries
        .iter()
        .all(|e| matches!(e.kind, KeyKind::Bool(_) | KeyKind::SmallInt(_)))
        && entries
            .iter()
            .all(|e| e.handle.is_exact_instance_of::<PyInt>())
    {
        let mut keyed: Vec<(i64, usize)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let v = match e.kind {
                    KeyKind::Bool(b) => b as i64,
                    KeyKind::SmallInt(v) => v,
                    _ => unreachable!("the all-int/bool guard"),
                };
                (v, i)
            })
            .collect();
        keyed.sort_by_key(|&(v, _)| v);
        keyed.into_iter().map(|(_, i)| i).collect()
    } else {
        // Identity index: (key ptr, value ptr) -> entry index. A dict
        // cannot hold two entries with the same key AND value object
        // (inserting an equal key updates; two coexisting keys are
        // pairwise !=, and NaN's k != k lets the same KEY object coexist
        // only under different values), so each identity pair maps to
        // exactly one entry.
        let mut index_of: HashMap<(usize, usize), usize> = HashMap::with_capacity(n);
        for (i, (entry, value)) in entries.iter().zip(&values).enumerate() {
            index_of.insert((entry.handle.as_ptr() as usize, value.as_ptr() as usize), i);
        }
        let tuples: Vec<Bound<'py, PyTuple>> = entries
            .iter()
            .zip(&values)
            .map(|(entry, value)| (entry.handle.clone(), value.clone()).into_pyobject(py))
            .collect::<PyResult<_>>()?;
        let list = PyList::new(py, tuples)?;
        list.call_method0("sort")?;
        let mut order = Vec::with_capacity(n);
        for item in list.iter() {
            let pair = item
                .cast::<PyTuple>()
                .expect("the list holds (key, value) tuples");
            let key_ptr = pair.get_item(0)?.as_ptr() as usize;
            let value_ptr = pair.get_item(1)?.as_ptr() as usize;
            let idx = index_of
                .remove(&(key_ptr, value_ptr))
                .expect("the sort returned a pair we did not build");
            order.push(idx);
        }
        order
    };

    // Coercion, in sorted order: each bucket's json.dumps string form.
    let mut pairs = Vec::with_capacity(n);
    for i in order {
        let entry = &entries[i];
        let key = match &entry.kind {
            KeyKind::Str => entry
                .handle
                .cast::<PyString>()
                .expect("classified Str")
                .to_str()?
                .to_owned(),
            KeyKind::Bool(true) => "true".to_owned(),
            KeyKind::Bool(false) => "false".to_owned(),
            KeyKind::SmallInt(v) => v.to_string(),
            KeyKind::BigInt => spell_long(&entry.handle, reprs)?,
            KeyKind::Float(v) => {
                if v.is_finite() {
                    spell_float(&entry.handle, reprs)?
                } else if v.is_nan() {
                    "NaN".to_owned()
                } else if *v > 0.0 {
                    "Infinity".to_owned()
                } else {
                    "-Infinity".to_owned()
                }
            }
            KeyKind::Null => "null".to_owned(),
            KeyKind::Unknown => return Err(key_type_error(&entry.handle)),
        };
        pairs.push((key, values[i].clone()));
    }
    Ok(pairs)
}

/// One open container on the walk's explicit stack. `container` is held
/// for its lifetime (never read after construction): every OPEN container
/// stays referenced for the whole walk, which is also what makes the
/// circular-marker set sound -- a marker's pointer is an alive object, so
/// no freed-and-recycled address can ever false-positive as a cycle.
enum Frame<'py> {
    Seq {
        container: Bound<'py, PyAny>,
        built: Vec<Canon>,
        pending: std::vec::IntoIter<Bound<'py, PyAny>>,
    },
    Map {
        container: Bound<'py, PyAny>,
        built: Vec<(String, Canon)>,
        open_key: Option<String>,
        pending: std::vec::IntoIter<(String, Bound<'py, PyAny>)>,
    },
}

impl Frame<'_> {
    fn container_ptr(&self) -> usize {
        match self {
            Frame::Seq { container, .. } | Frame::Map { container, .. } => {
                container.as_ptr() as usize
            }
        }
    }
}

/// The whole GIL-held walk: `root` (and every object reachable from it)
/// into one owned [`Canon`] tree, iteratively, with json.dumps's circular
/// reference semantics (markers entered at container entry, exited at
/// completion, so a shared sibling is fine and only a true cycle raises).
pub(crate) fn walk(py: Python<'_>, root: Bound<'_, PyAny>) -> PyResult<Canon> {
    let reprs = Reprs {
        long_repr: py.get_type::<PyInt>().getattr("__repr__")?,
        float_repr: py.get_type::<PyFloat>().getattr("__repr__")?,
    };
    let mut stack: Vec<Frame<'_>> = Vec::new();
    let mut markers: HashSet<usize> = HashSet::new();
    let mut finished: Option<Canon> = None;
    let mut to_walk: Option<Bound<'_, PyAny>> = Some(root);

    loop {
        // 1. Deliver finished subtrees upward, cascading through frames
        //    that just completed (their last child arrived).
        while let Some(done) = finished.take() {
            let Some(frame) = stack.last_mut() else {
                return Ok(done); // the root itself completed
            };
            let exhausted = match frame {
                Frame::Seq { built, pending, .. } => {
                    built.push(done);
                    pending.as_slice().is_empty()
                }
                Frame::Map {
                    built,
                    open_key,
                    pending,
                    ..
                } => {
                    let key = open_key
                        .take()
                        .expect("a map value completed with no open key");
                    built.push((key, done));
                    pending.as_slice().is_empty()
                }
            };
            if !exhausted {
                break;
            }
            let frame = stack.pop().expect("just checked non-empty");
            markers.remove(&frame.container_ptr());
            finished = Some(match frame {
                Frame::Seq { built, .. } => Canon::Seq(built),
                Frame::Map { built, .. } => Canon::Map(built),
            });
        }

        // 2. Pick the next object to walk: the unwalked root (the first
        //    iteration), or the top frame's next child. A frame with no
        //    children left here is one that was pushed empty (exhausted
        //    frames are popped in step 1): finalize it and loop.
        let obj = match to_walk.take() {
            Some(obj) => Some(obj),
            None => match stack.last_mut() {
                None => unreachable!("no work, no frames: step 1 returned"),
                Some(Frame::Seq { pending, .. }) => pending.next(),
                Some(Frame::Map {
                    pending, open_key, ..
                }) => match pending.next() {
                    Some((key, value)) => {
                        *open_key = Some(key);
                        Some(value)
                    }
                    None => None,
                },
            },
        };
        let obj = match obj {
            Some(obj) => obj,
            None => {
                let frame = stack
                    .pop()
                    .expect("step 2 with an empty stack is unreachable");
                markers.remove(&frame.container_ptr());
                finished = Some(match frame {
                    Frame::Seq { built, .. } => Canon::Seq(built),
                    Frame::Map { built, .. } => Canon::Map(built),
                });
                continue;
            }
        };

        // 3. Walk it: a leaf completes immediately; a container pushes a
        //    frame (after the circular check), and its children become
        //    the frame's pending work.
        if let Some(leaf) = walk_leaf(&obj, &reprs)? {
            finished = Some(leaf);
            continue;
        }
        let ptr = obj.as_ptr() as usize;
        if markers.contains(&ptr) {
            return Err(PyValueError::new_err(
                "circular reference detected in the content_hash() argument",
            ));
        }
        markers.insert(ptr);
        if let Ok(dict) = obj.cast::<PyDict>() {
            let pairs = dict_pairs(py, dict, &reprs)?;
            stack.push(Frame::Map {
                container: obj,
                built: Vec::with_capacity(pairs.len()),
                open_key: None,
                pending: pairs.into_iter(),
            });
        } else if let Ok(list) = obj.cast::<PyList>() {
            let children: Vec<Bound<'_, PyAny>> = list.iter().collect();
            stack.push(Frame::Seq {
                container: obj,
                built: Vec::with_capacity(children.len()),
                pending: children.into_iter(),
            });
        } else if let Ok(tuple) = obj.cast::<PyTuple>() {
            let children: Vec<Bound<'_, PyAny>> = tuple.iter().collect();
            stack.push(Frame::Seq {
                container: obj,
                built: Vec::with_capacity(children.len()),
                pending: children.into_iter(),
            });
        } else {
            // walk_leaf's guard makes this unreachable; spelled out so a
            // future container type added to walk_leaf fails loudly here.
            unreachable!("walk_leaf returned None for a non-container");
        }
    }
}

/// `tors.content_hash(obj: str | int | float | bool | None | list | tuple
/// | dict) -> str`: the lowercase-hex SHA-256 of the object's canonical
/// form, where the canonical form is EXACTLY
/// `json.dumps(obj, sort_keys=True, separators=(",", ":"))` with default
/// `ensure_ascii` and `allow_nan` -- byte-identical with the stdlib
/// expression the doc site spells out, pinned differentially against it
/// (tests/test_content_hash.py) and literal-pinned at the byte level
/// (src/canon_impl.rs).
///
/// Accepted leaves: `str`, `int` (arbitrary precision), `float`, `bool`,
/// `None`; containers: `list`, `tuple` (serializes as a list -- equal-value
/// list/tuple hash identically), `dict` (keys sorted before
/// stringification, `str`/`int`/`float`/`bool`/`None` keys coerced to
/// their json string form). Anything else raises `TypeError` naming the
/// type; circular references raise `ValueError`; a str holding lone
/// surrogates raises `UnicodeEncodeError` where json.dumps accepts it
/// (the crate-wide str-borrow divergence, documented).
///
/// GIL model: the object walk and the leaf spellings run under the GIL
/// (the standard arg-walk class, O(tree): one borrow+copy per str, one
/// storage read per int, one repr call per float -- the module docs
/// above); the canonical-form emission and the SHA-256 run under one
/// `py.detach`. Deterministic: any dict key order yields the same hash.
#[pyfunction]
pub fn content_hash(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<String> {
    let tree = walk(py, obj)?;
    Ok(py.detach(|| crate::canon_impl::digest_hex(&tree)))
}
