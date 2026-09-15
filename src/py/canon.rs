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
//! SUBCLASS HOOKS RUN TO COMPLETION UNDER THE GIL: a dict subclass's
//! `.items()` and a list/tuple subclass's `__iter__` are interpreter calls
//! the walk must drive to completion before the child walk begins. A hook
//! yielding an unbounded stream therefore holds the GIL until the
//! per-container materialization cap (`MAX_PROTOCOL_ITEMS`) aborts it with
//! `ValueError` -- bounded, never silent, never infinite. Treat
//! `content_hash` as trusted-input-only for subclass instances with
//! attacker-controlled hooks, the same posture `json.dumps` itself has
//! (it materializes unboundedly and would spin forever).
//!
//! UNTRUSTED-INPUT CEILING: exact containers nest without consulting the
//! interpreter's recursion budget, so the walk enforces its own total cap
//! (`MAX_TOTAL_DEPTH` open frames, exact+protocol, plus `MAX_WALK_NODES`
//! visited objects). Past either cap the walk raises `RecursionError`
//! (depth) or `ValueError` (node count) instead of growing the owned tree
//! without bound. The documented deep-nesting superset lives INSIDE the
//! ceiling (100k exact levels hash; 200k raises); size the caps for the
//! caller's threat model before hashing adversarial input.
//!
//! The walk is ITERATIVE (an explicit frame stack), not recursive: depth
//! costs heap, never the call stack, so any tree the interpreter can hold
//! walks clean for EXACT containers up to the ceiling above (json.dumps
//! itself `RecursionError`s on deep trees at a version-dependent depth --
//! a documented divergence lane: tors accepts deeper input than the stdlib
//! spelling, deterministically, within its cap). The subclass lane is
//! capped instead, at json's own failure boundary for it (the runaway
//! guard, below).
//!
//! # Subclass containers: json.dumps's own protocol iteration
//!
//! json.dumps does NOT walk a container subclass's concrete storage: a
//! non-exact dict is iterated through `PyMapping_Items` -- the
//! OVERRIDABLE `.items()`, its result materialized into a snapshot list
//! and sorted with CPython's own timsort -- and a non-exact list/tuple
//! through `PyObject_GetIter` (its `__iter__`), materialized
//! `PySequence_Fast`-style before any child is encoded
//! (Modules/_json.c's `encoder_listencode_dict` /
//! `encoder_listencode_list`, stable 3.10 through 3.14). A subclass
//! that hides, fakes, reorders, or empties its content through those
//! hooks therefore changes json's hash, and the walk must follow or the
//! two silently disagree. It does, by delegation -- the same
//! exact-instance gate idiom the key sorts already use. EXACT instances
//! keep the concrete-storage fast path (json's own exact-dict
//! `sort_keys` branch materializes the same items, so the paths are
//! byte-equivalent there, and the measured concrete walk stays).
//!
//! The dict lane spells json's own algorithm: `.items()` called through
//! the interpreter, the result materialized (snapshot semantics: a
//! mutation after it is invisible to the walk, one during it is
//! captured), that list sorted with `list.sort()` over the pairs AS
//! YIELDED -- before any pair is validated or any key spelled, so an
//! unsortable mix raises the sort's own TypeError first -- and then, per
//! pair, json's own checks in json's own order: each item must be a
//! 2-sized tuple (subclass-tolerant, read from concrete storage, an
//! overriding `__getitem__` ignored) or the shared
//! `ValueError: items must return 2-tuples` fires; a subclass dict with
//! EMPTY concrete storage emits `{}` without ever calling `.items()`
//! (`PyDict_GET_SIZE == 0` short-circuits the C encoder first). The
//! validation and the key coercion are LAZY, pair by pair in sorted
//! order at frame-pull time, because that is json's encode order: a bad
//! value at pair i raises before pair i+1 is validated.
//!
//! # The runaway guard (protocol frames vs. the recursion limit)
//!
//! json's encoder recurses in C at every container
//! (`Py_EnterRecursiveCall`), so protocol-mediated nesting -- hooks that
//! yield ever-fresh subclasses, which no circular marker can catch --
//! dies by `RecursionError` when the interpreter's recursion budget
//! runs out. tors's walk is iterative, so the same input would descend
//! forever instead. The guard: PROTOCOL frames (subclass containers
//! only) are counted against `sys.getrecursionlimit()`, read under the
//! GIL once at walk start; at the cap the walk raises `RecursionError`
//! -- json's own failure class. EXACT containers are uncapped by the
//! INTERPRETER budget but capped by the untrusted-input ceiling above
//! (`MAX_TOTAL_DEPTH` total frames, `MAX_WALK_NODES` visited objects),
//! preserving the documented deep-nesting superset inside the ceiling
//! (tors green at 100k exact levels where json.dumps raises; 200k raises
//! tors's own `RecursionError`). The rule: exact nesting is the superset
//! lane up to the ceiling; protocol nesting is json parity, its failure
//! boundary included. On interpreters where json's budget IS the Python
//! recursion limit (3.10/3.11) the boundary matches (depth <= limit hashes,
//! depth == limit+1 raises on both sides -- pinned limit-1/limit/limit+1);
//! on
//! 3.12+ json's C-stack budget is looser than the recursion limit, so
//! tors's cap is deliberately conservative there -- the same error
//! class, a tighter boundary, never a silent loop.
//!
//! MIXED NESTING DIVERGES BY CONSTRUCTION: json counts EVERY container
//! against one C budget, tors counts only protocol frames against the
//! interpreter budget (exact frames count only against the much larger
//! `MAX_TOTAL_DEPTH`). An interleaving of exact and protocol nesting can
//! therefore raise under json while tors succeeds (or vice versa at the
//! ceiling). Pinned as a documented divergence, not parity:
//! `tests/test_content_hash.py`'s interleaved exact+protocol differential
//! pins the shape.
//!
//! PROTOCOL BUDGET HONESTY (>1000 UNSUPPORTED): the protocol-frame cap is
//! `sys.getrecursionlimit()` (~1000), matching json's failure boundary on
//! 3.10/3.11 where the C budget IS the recursion limit. On 3.12+ json's
//! C-stack budget is looser (a legit 2k-deep subclass chain hashes under
//! `json.dumps` at ~100k frames on 3.14 while tors raises `RecursionError`
//! at ~1000) -- a deliberate conservative divergence, same error class,
//! tighter boundary, never a silent hash. Protocol nesting past ~1000 is
//! therefore UNSUPPORTED by contract (documented in `docs/api.md` and the
//! `.pyi` stub): exact nesting is the deep lane, protocol nesting is the
//! parity lane up to the interpreter limit.
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

use pyo3::exceptions::{PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::canon_impl::Canon;

/// Untrusted-input ceiling: total OPEN frames (exact + protocol) the
/// iterative walk will hold. 100k-deep exact trees (the pinned superset
/// lane) pass with headroom; 200k-deep adversarial nesting raises
/// `RecursionError` instead of materializing 200k `Canon` nodes plus the
/// marker/work stacks GIL-held. Tune down for stricter postures.
///
/// THREAT-FITTED ENVELOPE: the caps above are test-fitted to the pinned
/// superset lane, not to an adversarial wall/RSS budget -- a 149k-deep
/// exact tree still materializes tens of MiB of `Canon` nodes plus the
/// frame/marker stacks GIL-held before the ceiling fires. Treat
/// `content_hash` as trusted-input-only for depth/breadth beyond a modest
/// envelope (depth on the order of 10-20k frames, visited objects on the
/// order of 200-500k nodes); size the caps for the caller's threat model
/// before hashing adversarial input.
pub(crate) const MAX_TOTAL_DEPTH: usize = 150_000;
/// Untrusted-input ceiling: total visited objects (leaves + containers).
/// The 12 MiB records corpus walks ~1M nodes; the cap sits at 2x that, so
/// legitimate corpora pass while breadth-DoS (a multi-million-element
/// list) raises `ValueError` instead of growing the owned tree unbounded.
/// Same envelope caveat as `MAX_TOTAL_DEPTH`: untrusted input belongs
/// under ~200-500k nodes; beyond that the GIL-held owned tree is tens of
/// MiB and the caller must treat the input as trusted.
pub(crate) const MAX_WALK_NODES: usize = 2_000_000;
/// Per-container protocol materialization cap: a subclass `__iter__` /
/// `.items()` result is pulled to completion under the GIL, so an
/// unbounded hook (an infinite iterator) would spin forever. Past this
/// many pulled items the walk aborts with `ValueError`. Legitimate
/// containers (10k-key dicts, 1 MiB corpora) sit orders of magnitude
/// below it. The message is deliberately generic (no cap value): the
/// bound is a DoS backstop, not a contract to advertise to hook authors.
pub(crate) const MAX_PROTOCOL_ITEMS: usize = 1_000_000;
/// Per-dict delegated-sort bound (HIGH-3): the exotic-key lane delegates
/// to CPython's `list.sort()` over live `(key, value)` tuples -- O(n log n)
/// Python comparisons plus one `HashMap` entry and one Python tuple per
/// key, all GIL-held. A 1M-exotic-key dict is ~100 MiB GIL-held before a
/// single byte is emitted. Past this many keys the walk refuses with
/// `ValueError` instead of paying that residue. The str and int/bool fast
/// paths are unaffected (no interpreter sort); 10k-key dicts sit an order
/// of magnitude below the bound.
pub(crate) const MAX_DELEGATED_SORT_KEYS: usize = 100_000;
/// Total delegated-sort work bound (HIGH-2): breadth-of-exotics -- e.g.
/// 500k 2-key exotic dicts, each taking the delegated lane -- is 500k
/// interpreter sorts plus 500k identity `HashMap`s, all under the node
/// caps. Past this many total delegated pairs walked the call aborts with
/// `ValueError`. Counts exact-lane delegated dicts AND protocol-lane
/// `.items()` sorts; fast-path dicts cost nothing against it.
pub(crate) const MAX_TOTAL_DELEGATED_PAIRS: usize = 500_000;

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

/// Classifies a dict key into its coercion bucket. Infallible except for
/// the float storage read: a successful `PyFloat` cast always extracts an
/// f64, so a failure propagates as a real error (`?`) rather than
/// silently coercing into a NaN key.
fn classify_key(key: &Bound<'_, PyAny>) -> PyResult<KeyKind> {
    Ok(if key.is_none() {
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
        KeyKind::Float(f.extract::<f64>()?)
    } else if key.cast::<PyString>().is_ok() {
        KeyKind::Str
    } else {
        KeyKind::Unknown
    })
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

/// One classified key's json.dumps string form: the coercion half of the
/// sort-then-coerce order (the exact lane calls it post-sort at push;
/// the protocol lane at frame-pull, pair by pair, json's own encode
/// order). Str is borrowed here (the crate-wide lone-surrogate
/// boundary), bools/ints/floats spell per the module docs, `None` is
/// "null", and the non-coercible bucket raises the house-worded key
/// TypeError.
fn spell_key(kind: &KeyKind, handle: &Bound<'_, PyAny>, reprs: &Reprs<'_>) -> PyResult<String> {
    Ok(match kind {
        KeyKind::Str => handle
            .cast::<PyString>()
            .expect("classified Str")
            .to_str()?
            .to_owned(),
        KeyKind::Bool(true) => "true".to_owned(),
        KeyKind::Bool(false) => "false".to_owned(),
        KeyKind::SmallInt(v) => v.to_string(),
        KeyKind::BigInt => spell_long(handle, reprs)?,
        KeyKind::Float(v) => {
            if v.is_finite() {
                spell_float(handle, reprs)?
            } else if v.is_nan() {
                "NaN".to_owned()
            } else if *v > 0.0 {
                "Infinity".to_owned()
            } else {
                "-Infinity".to_owned()
            }
        }
        KeyKind::Null => "null".to_owned(),
        KeyKind::Unknown => return Err(key_type_error(handle)),
    })
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
        // Same `?` discipline as `classify_key`: never silently coerce an
        // extraction failure into NaN.
        let v = f.extract::<f64>()?;
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
    delegated_total: &mut usize,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let mut entries: Vec<KeyEntry<'_>> = Vec::with_capacity(dict.len());
    let mut values: Vec<Bound<'_, PyAny>> = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let kind = classify_key(&key)?;
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
    } else if entries.iter().all(|e| match &e.kind {
        // The exact gate is per bucket: bool IS its 0/1 int value (and
        // bool cannot be subclassed, so a Bool-classified key is exact
        // by construction -- the explicit gate mirrors the str/int
        // lanes' idiom), while a SmallInt-classified key may be an int
        // SUBCLASS with overridden rich comparison, which the numeric
        // sort would silently ignore, so it must stay exact to take
        // this path. True/1 and False/0 cannot coexist as dict keys,
        // so no numeric tie is possible and the i64 sort is the tuples'
        // own total order. The original whole-set gate
        // (is_exact_instance_of::<PyInt> over every entry) is FALSE for
        // True/False -- bool's type is bool, not int -- so every
        // bool-bearing key set silently took the delegated path and
        // this fast path never ran: the docstring described it, the
        // code did not deliver it.
        KeyKind::Bool(_) => e.handle.is_exact_instance_of::<PyBool>(),
        KeyKind::SmallInt(_) => e.handle.is_exact_instance_of::<PyInt>(),
        _ => false,
    }) {
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
        // The delegated lane: bounded BEFORE any materialization, so a
        // 1M-exotic-key dict refuses instead of building ~100 MiB of
        // tuples + HashMap GIL-held. Per-dict cap first (the single-sort
        // DoS), then the running total (the breadth-of-exotics DoS:
        // hundreds of thousands of tiny delegated dicts). Both bounds
        // are generic `ValueError`s (no cap values leaked). The list is
        // pre-sized by construction: exactly `n` appends into one Python
        // list CPython sorts in place (no intermediate Rust Vec<PyTuple>
        // -> PyList copy); the delegated path already pays one
        // interpreter sort, it must not also pay a double collect.
        // Identity index: (key ptr, value ptr) -> entry index. A dict
        // cannot hold two entries with the same key AND value object
        // (inserting an equal key updates; two coexisting keys are
        // pairwise !=, and NaN's k != k lets the same KEY object coexist
        // only under different values), so each identity pair maps to
        // exactly one entry.
        if n > MAX_DELEGATED_SORT_KEYS {
            return Err(PyValueError::new_err(
                "content_hash() dict has too many keys requiring interpreter sort: refusing an unbounded delegated sort",
            ));
        }
        *delegated_total = delegated_total.saturating_add(n);
        if *delegated_total > MAX_TOTAL_DELEGATED_PAIRS {
            return Err(PyValueError::new_err(
                "content_hash() walked too many interpreter-sorted pairs: refusing unbounded delegated-sort work",
            ));
        }
        let mut index_of: HashMap<(usize, usize), usize> = HashMap::with_capacity(n);
        for (i, (entry, value)) in entries.iter().zip(&values).enumerate() {
            let id = (entry.handle.as_ptr() as usize, value.as_ptr() as usize);
            debug_assert!(
                !index_of.contains_key(&id),
                "duplicate (key, value) identity pair: the dict holds two entries sharing both objects"
            );
            index_of.insert(id, i);
        }
        // The delegated sort is the known slow lane vs. the
        // byte/numeric fast paths above: one `list.sort()` over live
        // objects, documented here so a caller sorting exotic key zoos at
        // scale can read the cost.
        let list = PyList::empty(py);
        for (entry, value) in entries.iter().zip(&values) {
            list.append((entry.handle.clone(), value.clone()).into_pyobject(py)?)?;
        }
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
        pairs.push((
            spell_key(&entry.kind, &entry.handle, reprs)?,
            values[i].clone(),
        ));
    }
    Ok(pairs)
}

/// The circular-marker entry (json's markers are enter/exit over object
/// identity: a shared sibling is fine, only a true cycle raises). The
/// call order is json's own per lane -- the protocol LIST lane enters
/// after the iterator materialization (`encoder_listencode_list`
/// materializes via `PySequence_Fast` first), both dict lanes before
/// anything else runs -- and the exact lanes enter first too, where the
/// order is unobservable (a concrete collect cannot raise).
fn enter_marker(markers: &mut HashSet<usize>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let ptr = obj.as_ptr() as usize;
    if markers.contains(&ptr) {
        return Err(PyValueError::new_err(
            "circular reference detected in the content_hash() argument",
        ));
    }
    markers.insert(ptr);
    Ok(())
}

/// A non-exact dict's items, spelled exactly as json.dumps spells them
/// (Modules/_json.c's `encoder_listencode_dict`, the
/// `sort_keys || !PyDict_CheckExact` branch -- and content_hash is
/// always sort_keys): the OVERRIDABLE `.items()` called through the
/// interpreter, the result materialized into a snapshot
/// (`PyMapping_Items`'s own semantics: every yielded element pulled,
/// the first error propagating, later mutations invisible), and that
/// list sorted with CPython's own timsort over the pairs AS YIELDED --
/// before any pair is validated or any key spelled, so an unsortable
/// mix raises the sort's own TypeError first. The 2-tuple validation
/// and the key coercion happen lazily at frame-pull time, pair by pair
/// in sorted order -- json's own encode order, where a bad value at
/// pair i raises before pair i+1 is validated.
fn sorted_items_protocol<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    delegated_total: &mut usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let items = obj.call_method0("items")?;
    let iter = items.try_iter()?;
    // Single materialization, bounded: each yielded element is appended
    // straight into the Python list CPython sorts in place (no Rust Vec
    // -> PyList copy). The pull runs to completion under the GIL, so the
    // cap is what bounds a malicious unbounded `.items()` -- past it the
    // walk aborts with a generic ValueError (no cap value leaked) instead
    // of spinning forever.
    let list = PyList::empty(py);
    let mut pulled: usize = 0;
    for item in iter {
        let item = item?;
        pulled += 1;
        if pulled > MAX_PROTOCOL_ITEMS {
            return Err(PyValueError::new_err(
                "content_hash() subclass hook yielded too many items: refusing an unbounded hook result",
            ));
        }
        list.append(item)?;
    }
    // The protocol sort is delegated work too: count it against the same
    // total-delegated bound as the exact exotic lane (HIGH-2), and refuse
    // a single huge `.items()` result before paying the interpreter sort.
    if pulled > MAX_DELEGATED_SORT_KEYS {
        return Err(PyValueError::new_err(
            "content_hash() subclass hook yielded too many items to sort: refusing an unbounded delegated sort",
        ));
    }
    *delegated_total = delegated_total.saturating_add(pulled);
    if *delegated_total > MAX_TOTAL_DELEGATED_PAIRS {
        return Err(PyValueError::new_err(
            "content_hash() walked too many interpreter-sorted pairs: refusing unbounded delegated-sort work",
        ));
    }
    list.call_method0("sort")?;
    Ok(list.iter().collect())
}

/// One open container on the walk's explicit stack. `container` is held
/// for its lifetime (never read after construction): every OPEN container
/// stays referenced for the whole walk, which is also what makes the
/// circular-marker set sound -- a marker's pointer is an alive object, so
/// no freed-and-recycled address can ever false-positive as a cycle.
/// `protocol` marks the subclass lane (json's own protocol iteration,
/// and the only lane the runaway guard counts).
enum Frame<'py> {
    Seq {
        container: Bound<'py, PyAny>,
        built: Vec<Canon>,
        pending: std::vec::IntoIter<Bound<'py, PyAny>>,
        protocol: bool,
    },
    Map {
        container: Bound<'py, PyAny>,
        built: Vec<(String, Canon)>,
        open_key: Option<String>,
        pending: MapPending<'py>,
        protocol: bool,
    },
}

/// A map frame's remaining work: the exact lane's pre-coerced (spelled
/// key, value) pairs, or the protocol lane's sorted raw items -- each
/// validated and key-coerced at pull time, json's own encode order, so
/// pair i's value error fires before pair i+1 is validated.
enum MapPending<'py> {
    Coerced(std::vec::IntoIter<(String, Bound<'py, PyAny>)>),
    Items(std::vec::IntoIter<Bound<'py, PyAny>>),
}

impl MapPending<'_> {
    fn len(&self) -> usize {
        match self {
            MapPending::Coerced(pairs) => pairs.as_slice().len(),
            MapPending::Items(items) => items.as_slice().len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Frame<'_> {
    fn container_ptr(&self) -> usize {
        match self {
            Frame::Seq { container, .. } | Frame::Map { container, .. } => {
                container.as_ptr() as usize
            }
        }
    }

    fn is_protocol(&self) -> bool {
        matches!(
            self,
            Frame::Seq { protocol: true, .. } | Frame::Map { protocol: true, .. }
        )
    }
}

/// The whole GIL-held walk: `root` (and every object reachable from it)
/// into one owned [`Canon`] tree, iteratively, with json.dumps's circular
/// reference semantics (markers entered at container entry, exited at
/// completion, so a shared sibling is fine and only a true cycle raises)
/// and json.dumps's own subclass iteration (the module docs' protocol
/// lane, with its runaway guard).
pub(crate) fn walk(py: Python<'_>, root: Bound<'_, PyAny>) -> PyResult<Canon> {
    let reprs = Reprs {
        long_repr: py.get_type::<PyInt>().getattr("__repr__")?,
        float_repr: py.get_type::<PyFloat>().getattr("__repr__")?,
    };
    // The protocol lane's runaway cap: json's C encoder enters a
    // recursive call at every container, so protocol-mediated nesting
    // dies by RecursionError at the interpreter's recursion budget; the
    // iterative walk counts its PROTOCOL frames against the same limit
    // (read here, under the GIL, once per call): depth <= limit hashes,
    // depth == limit+1 raises -- the same `>` boundary C enforces, same
    // error class, never a silent loop.
    // EXACT containers bypass the interpreter budget but count against
    // MAX_TOTAL_DEPTH below: the deep-nesting superset lives inside the
    // untrusted-input ceiling, not outside all bounds.
    let recursion_cap: usize = PyModule::import(py, "sys")?
        .call_method0("getrecursionlimit")?
        .extract()?;
    let mut protocol_depth: usize = 0;
    let mut nodes: usize = 0;
    let mut delegated_total: usize = 0;
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
                    pending.is_empty()
                }
            };
            if !exhausted {
                break;
            }
            let frame = stack.pop().expect("just checked non-empty");
            markers.remove(&frame.container_ptr());
            if frame.is_protocol() {
                protocol_depth -= 1;
            }
            finished = Some(match frame {
                Frame::Seq { built, .. } => Canon::Seq(built),
                Frame::Map { built, .. } => Canon::Map(built),
            });
        }

        // 2. Pick the next object to walk: the unwalked root (the first
        //    iteration), or the top frame's next child. A frame with no
        //    children left here is one that was pushed empty (exhausted
        //    frames are popped in step 1): finalize it and loop. The
        //    protocol lane's map pull validates the pair and coerces its
        //    key HERE -- lazily, in json's own encode order, so pair i's
        //    value error fires before pair i+1 is validated.
        let pulled: PyResult<Option<Bound<'_, PyAny>>> = match to_walk.take() {
            Some(obj) => Ok(Some(obj)),
            None => match stack.last_mut() {
                None => unreachable!("no work, no frames: step 1 returned"),
                Some(Frame::Seq { pending, .. }) => Ok(pending.next()),
                Some(Frame::Map {
                    pending, open_key, ..
                }) => match pending {
                    MapPending::Coerced(pairs) => Ok(match pairs.next() {
                        Some((key, value)) => {
                            *open_key = Some(key);
                            Some(value)
                        }
                        None => None,
                    }),
                    MapPending::Items(items) => match items.next() {
                        None => Ok(None),
                        Some(item) => {
                            // json's own per-pair gate, at json's own
                            // point in the order: the item must be a
                            // 2-sized tuple read from concrete storage
                            // (an overriding __getitem__ ignored -- see
                            // below), or the shared ValueError fires.
                            let pair = match item.cast::<PyTuple>() {
                                Ok(pair) if pair.len() == 2 => pair,
                                _ => {
                                    return Err(PyValueError::new_err(
                                        "items must return 2-tuples",
                                    ));
                                }
                            };
                            // `Bound<PyTuple>::get_item` reads the tuple's
                            // concrete storage slot (`PyTuple_GET_ITEM`
                            // semantics, pyo3 0.29 -- a tuple-subclass
                            // override of `__getitem__` is NOT consulted,
                            // matching the C encoder's direct slot read.
                            // If pyo3 changes this accessor's semantics,
                            // this lane must move to explicit FFI):
                            // a tuple-subclass override of
                            // `__getitem__` is NOT consulted, matching the
                            // C encoder's direct slot read. Pinned by the
                            // `GetItemLiar` differential below.
                            debug_assert_eq!(pair.len(), 2, "pair length checked above");
                            let key = pair.get_item(0)?;
                            let value = pair.get_item(1)?;
                            let kind = classify_key(&key)?;
                            *open_key = Some(spell_key(&kind, &key, &reprs)?);
                            Ok(Some(value))
                        }
                    },
                },
            },
        };
        let obj = match pulled? {
            Some(obj) => obj,
            None => {
                let frame = stack
                    .pop()
                    .expect("step 2 with an empty stack is unreachable");
                markers.remove(&frame.container_ptr());
                if frame.is_protocol() {
                    protocol_depth -= 1;
                }
                finished = Some(match frame {
                    Frame::Seq { built, .. } => Canon::Seq(built),
                    Frame::Map { built, .. } => Canon::Map(built),
                });
                continue;
            }
        };

        // 3. Walk it: a leaf completes immediately. An EXACT container
        //    walks its concrete storage (the measured fast path; json's
        //    own exact-dict sort_keys branch materializes the same
        //    items, so the two are byte-equivalent there). A SUBCLASS
        //    container delegates to the interpreter's own protocol --
        //    json.dumps's subclass iteration, never the concrete storage
        //    -- under the runaway guard and the untrusted-input ceiling.
        nodes += 1;
        if nodes > MAX_WALK_NODES {
            return Err(PyValueError::new_err(
                "content_hash() argument visits too many objects: refusing an unbounded tree",
            ));
        }
        if let Some(leaf) = walk_leaf(&obj, &reprs)? {
            finished = Some(leaf);
            continue;
        }
        let protocol = !(obj.is_exact_instance_of::<PyDict>()
            || obj.is_exact_instance_of::<PyList>()
            || obj.is_exact_instance_of::<PyTuple>());
        if protocol && protocol_depth >= recursion_cap {
            return Err(PyRecursionError::new_err(
                "maximum recursion depth exceeded while walking a subclass container",
            ));
        }
        // Total-depth ceiling (exact + protocol): the exact lane's DoS
        // bound. `stack.len()` is the count of currently open frames;
        // pushing past the ceiling raises RecursionError instead of
        // materializing an unbounded owned tree GIL-held.
        if stack.len() >= MAX_TOTAL_DEPTH {
            return Err(PyRecursionError::new_err(
                "content_hash() nesting exceeds the untrusted-input ceiling",
            ));
        }
        if obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok() {
            // The list/tuple lanes. json's `encoder_listencode_list`
            // materializes FIRST (PySequence_Fast -> PyObject_GetIter on
            // a subclass: its __iter__, every yield pulled) and checks
            // the circular markers after; the exact lane checks first
            // (its collect cannot raise, so the order is unobservable
            // there).
            let children = if protocol {
                // Bounded pull: an infinite `__iter__` aborts here with a
                // generic ValueError (no cap value leaked) instead of
                // spinning GIL-held forever.
                let mut children: Vec<Bound<'_, PyAny>> = Vec::new();
                for child in obj.try_iter()? {
                    children.push(child?);
                    if children.len() > MAX_PROTOCOL_ITEMS {
                        return Err(PyValueError::new_err(
                            "content_hash() subclass hook yielded too many items: refusing an unbounded hook result",
                        ));
                    }
                }
                enter_marker(&mut markers, &obj)?;
                children
            } else {
                enter_marker(&mut markers, &obj)?;
                if let Ok(list) = obj.cast::<PyList>() {
                    list.iter().collect()
                } else {
                    obj.cast::<PyTuple>()
                        .expect("walk_leaf guarded the tuple arm")
                        .iter()
                        .collect()
                }
            };
            if protocol {
                protocol_depth += 1;
            }
            stack.push(Frame::Seq {
                container: obj,
                built: Vec::with_capacity(children.len()),
                pending: children.into_iter(),
                protocol,
            });
        } else if let Ok(dict) = obj.cast::<PyDict>() {
            // The dict lanes. json's `encoder_listencode_dict`: the {}
            // gate reads the CONCRETE storage size first (a subclass
            // dict with empty storage emits {} without ever calling
            // .items()), the circular markers come next, and only then
            // does the overridable .items() run (materialized, sorted,
            // validated lazily at pull time).
            let pending = if protocol {
                if dict.len() == 0 {
                    MapPending::Items(Vec::new().into_iter())
                } else {
                    enter_marker(&mut markers, &obj)?;
                    MapPending::Items(
                        sorted_items_protocol(py, &obj, &mut delegated_total)?.into_iter(),
                    )
                }
            } else {
                enter_marker(&mut markers, &obj)?;
                MapPending::Coerced(dict_pairs(py, dict, &reprs, &mut delegated_total)?.into_iter())
            };
            if protocol {
                protocol_depth += 1;
            }
            let capacity = pending.len();
            stack.push(Frame::Map {
                container: obj,
                built: Vec::with_capacity(capacity),
                open_key: None,
                pending,
                protocol,
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
/// expression the doc site spells out for surrogate-free input where json
/// succeeds, pinned differentially against it
/// (tests/test_content_hash.py) and literal-pinned at the byte level
/// (src/canon_impl.rs).
///
/// Accepted leaves: `str` (lone-surrogate strings excluded -- they raise
/// `UnicodeEncodeError` where json succeeds, the documented divergence),
/// `int` (arbitrary precision), `float`, `bool`,
/// `None`; containers: `list`, `tuple` (serializes as a list -- equal-value
/// list/tuple hash identically), `dict` (keys sorted before
/// stringification, `str`/`int`/`float`/`bool`/`None` keys coerced to
/// their json string form). Container SUBCLASSES are iterated exactly as
/// json.dumps iterates them -- a dict subclass through its (overridable)
/// `.items()`, a list/tuple subclass through its `__iter__`, each hook run
/// to completion under the GIL and bounded by `MAX_PROTOCOL_ITEMS` (generic
/// `ValueError`, no bound leaked) -- never
/// their concrete storage, so hiding/faking/reordering subclasses hash
/// identically on both sides; non-pair `items()` yields raise json's own
/// `ValueError: items must return 2-tuples`, and subclass chains nested
/// past `sys.getrecursionlimit()` raise `RecursionError` on both sides
/// (EXACT containers nest deeper than json up to the `MAX_TOTAL_DEPTH`
/// untrusted-input ceiling: the documented deep-nesting superset, bounded).
/// Protocol nesting past ~1000 is UNSUPPORTED by contract even where the
/// stdlib 3.12+ succeeds (tighter boundary, same error class). Exotic-key
/// dicts (any float/big-int/mixed/NaN/subclass key) delegate their sort to
/// CPython and are bounded by `MAX_DELEGATED_SORT_KEYS` per dict and
/// `MAX_TOTAL_DELEGATED_PAIRS` total (generic `ValueError`). Treat
/// `content_hash` as trusted-input-only for subclass hooks, for depth
/// beyond ~10-20k frames, and for breadth beyond ~200-500k visited objects
/// -- the same posture `json.dumps` itself has, which materializes
/// unboundedly.
/// Anything else raises `TypeError` naming the
/// type; circular references raise `ValueError`; a str holding lone
/// surrogates raises `UnicodeEncodeError` where json.dumps accepts it
/// (the crate-wide str-borrow divergence, documented).
///
/// GIL model: the object walk and the leaf spellings run under the GIL
/// (the standard arg-walk class, O(tree): one borrow+copy per str, one
/// storage read per int, one repr call per float -- the module docs
/// above); the canonical-form emission, the SHA-256, AND the owned tree's
/// teardown all run under one `py.detach` (the tree is moved into the
/// detached closure, so no deep-tree `Drop` tail holds the GIL after the
/// digest). Deterministic: any dict key order yields the same hash.
#[pyfunction]
pub fn content_hash(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<String> {
    let tree = walk(py, obj)?;
    Ok(py.detach(move || {
        let digest = crate::canon_impl::digest_hex(&tree);
        drop(tree);
        digest
    }))
}
