//! `tors::json_repair::StreamingRepairer` (the streaming repairer; see
//! src/json_repair/streaming.rs for the contract) never panics on
//! arbitrary chunk sequences, and its closed renders stay loadable. The
//! input is an arbitrary string replayed as a chunk sequence (several
//! deterministic split sets per input, boundaries mid-escape and
//! mid-surrogate-pair included); the asserts:
//!
//! - panic-freedom through push -> push -> end (the primary; Err is
//!   fine: the depth cap's normalized ValueError, after which the
//!   machine's state is unspecified and that splitting just stops);
//! - on every error-free path: every snapshot (and the end) is the
//!   empty sentinel or strict-loadable JSON, and snapshot == end on the
//!   same prefix;
//! - chunking invariance: any split set ends at the one-push answer;
//! - on the ENGINE-PARITY classes, the streamed end() equals the
//!   whole-text engine's repair byte for byte. Two classes, decided by
//!   one conservative scan of the input:
//!   - the strict-loadable class: the whole input is one strict JSON
//!     value (within the depth cap, where the engine itself succeeds);
//!   - the heal class (the review round's addition): the input is a
//!     CLEAN CUT of a strict document — the scan closes the cut up
//!     (the open string's quote, then per open frame the filler for a
//!     dangling key or a missing value and the frame's closer) and the
//!     closed text must be strict-loadable, with none of the documented
//!     divergence triggers in the input: comments or non-`"` string
//!     delimiters anywhere; a mismatched closer; content after the root
//!     value closes; a second top-level value; a missing-key colon past
//!     an object's first member; the missing-colon shape (a value where
//!     a colon was due); a structural char inside an open string (the
//!     re-sync family); a closed string whose raw content ends in a
//!     backslash run (the doubled-escape family); a closed string
//!     ending on a newline-run with content after it (the mid-document
//!     escape-tail divergence).
//!
//! Panics, non-loadable renders, snapshot/end disagreement, chunking
//! drift, or a parity break on either class are bugs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tors::json_repair::{RepairConfig, StreamingRepairer, Value, dumps, loads_strict, repair};

/// Where the scan stands inside the innermost container.
#[derive(Clone, Copy, PartialEq)]
enum Expect {
    Key,
    Colon,
    Value,
    AfterValue,
}

/// One open container in the scan's stack (`paren` for the tuple/
/// grouping container: only `)` closes it).
struct ScanFrame {
    obj: bool,
    paren: bool,
    members: usize,
    expect: Expect,
}

/// The string's escape sub-state.
enum StrEsc {
    None,
    Backslash,
    U(usize),
}

/// The conservative scan: does the input sit in an engine-parity class?
/// One pass; every documented divergence trigger it can see flips
/// `divergent` (the input leaves the parity classes; coverage loss only,
/// never a false assert).
struct Scan {
    stack: Vec<ScanFrame>,
    in_string: bool,
    esc: StrEsc,
    /// The raw trailing backslash run inside the open string.
    raw_bs_run: usize,
    /// The open string's raw content contains a backslash anywhere: the
    /// doubled-quote/escape re-pair family's territory (out at the cut).
    saw_backslash: bool,
    /// The open string's content (modulo trailing spaces) ends with a
    /// `\n` escape.
    content_ends_nl: bool,
    /// The LAST closed string ended on a newline-run; a following token
    /// makes the input the mid-document escape-tail divergence.
    nl_tail_string: bool,
    /// The value-run in flight: its length and its last nine raw chars
    /// (the longest special-float spelling, `-Infinity`, is nine) — a
    /// run spelling exactly `NaN`/`Infinity`/`-Infinity` renders the
    /// strict float mid-stream but the engine's repair lane quotes it,
    /// so any COMPLETED one (a token after it) leaves the parity class.
    run_len: usize,
    run_tail: [u8; 9],
    run_tail_len: usize,
    run_is_float_spelling: bool,
    top_done: bool,
    divergent: bool,
}

impl Scan {
    fn new() -> Scan {
        Scan {
            stack: Vec::new(),
            in_string: false,
            esc: StrEsc::None,
            raw_bs_run: 0,
            saw_backslash: false,
            content_ends_nl: false,
            nl_tail_string: false,
            run_len: 0,
            run_tail: [0; 9],
            run_tail_len: 0,
            run_is_float_spelling: false,
            top_done: false,
            divergent: false,
        }
    }

    /// A value-run char outside a string: keep the bounded window.
    fn run_char(&mut self, c: char) {
        if !c.is_ascii() {
            // a non-ASCII word char can never be a float spelling
            self.run_is_float_spelling = false;
            return;
        }
        self.run_len += 1;
        if self.run_tail_len == 9 {
            self.run_tail.copy_within(1.., 0);
            self.run_tail[8] = c as u8;
        } else {
            self.run_tail[self.run_tail_len] = c as u8;
            self.run_tail_len += 1;
        }
        let tail = &self.run_tail[..self.run_tail_len];
        self.run_is_float_spelling = matches!(self.run_len, 3 | 8 | 9)
            && matches!(tail, b"NaN" | b"Infinity" | b"-Infinity");
    }

    /// A value-run's end: a completed float-spelling run plus a
    /// following token is the strict-vs-repair-lane split (out).
    fn end_run(&mut self, terminated: bool) {
        if self.run_is_float_spelling && terminated {
            self.divergent = true;
        }
        self.run_len = 0;
        self.run_tail_len = 0;
        self.run_is_float_spelling = false;
    }

    fn walk(&mut self, s: &str) {
        let chars: Vec<char> = s.chars().collect();
        let n = chars.len();
        let mut i = 0usize;
        while i < n {
            let c = chars[i];
            // The global exclusions: comments and the non-`"` delimiters
            // never enter a parity class.
            if c == '#'
                || c == '/'
                || matches!(c, '\'' | '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}')
            {
                self.divergent = true;
                return;
            }
            if self.in_string {
                match std::mem::replace(&mut self.esc, StrEsc::None) {
                    StrEsc::Backslash => match c {
                        'u' => {
                            self.esc = StrEsc::U(0);
                            self.saw_backslash = true;
                        }
                        // an escaped structural char: conservative out
                        // (the engine's re-sync/re-pair machinery may
                        // fire)
                        '{' | '}' | '[' | ']' | ',' | ':' => {
                            self.divergent = true;
                            return;
                        }
                        // the newline escape: the content now ends with
                        // it (trailing spaces survive it)
                        'n' => {
                            self.raw_bs_run = 0;
                            self.content_ends_nl = true;
                        }
                        // an escaped backslash IS another raw backslash:
                        // the run grows (the doubled-escape family's
                        // trigger)
                        '\\' => {
                            self.raw_bs_run += 1;
                            self.content_ends_nl = false;
                        }
                        _ => {
                            self.raw_bs_run = 0;
                            self.content_ends_nl = false;
                        }
                    },
                    StrEsc::U(digits) => {
                        if c.is_ascii_hexdigit() {
                            if digits < 3 {
                                self.esc = StrEsc::U(digits + 1);
                            }
                            // the fourth digit completes the escape
                            i += 1;
                            continue;
                        }
                        // fewer than four digits then a non-hex char:
                        // the escape is literal text, the char plain
                        self.string_content(c);
                    }
                    StrEsc::None => {
                        if c == '\\' {
                            self.esc = StrEsc::Backslash;
                            self.raw_bs_run += 1;
                            self.saw_backslash = true;
                            i += 1;
                            continue;
                        }
                        self.string_content(c);
                    }
                }
                i += 1;
                continue;
            }
            // Not in a string: whitespace skips, then the token arms.
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            if self.top_done || self.nl_tail_string {
                // trailing content after the root value closed (the
                // multi-value divergence), or a token after a closed
                // newline-tailed string (the mid-document escape-tail
                // divergence): out, both
                self.divergent = true;
                return;
            }
            match c {
                '{' | '[' => {
                    if self
                        .stack
                        .last()
                        .is_some_and(|f| f.expect == Expect::Colon || f.expect == Expect::Key)
                    {
                        // a container where a colon was due, or where a
                        // KEY was due (the machine ignores it, the
                        // engine's whole-text walk re-decides): out
                        self.divergent = true;
                        return;
                    }
                    self.stack.push(ScanFrame {
                        obj: c == '{',
                        paren: false,
                        members: 0,
                        expect: if c == '{' { Expect::Key } else { Expect::Value },
                    });
                    i += 1;
                }
                '}' | ']' | ')' => {
                    let closer_obj = c == '}';
                    let closer_paren = c == ')';
                    let Some(f) = self.stack.last() else {
                        self.divergent = true;
                        return;
                    };
                    if f.paren != closer_paren || (!f.paren && f.obj != closer_obj) {
                        // a mismatched closer: the engine's close-up
                        // machinery, whole-text-only (out)
                        self.divergent = true;
                        return;
                    }
                    self.stack.pop();
                    if self.stack.is_empty() {
                        self.top_done = true;
                    } else if let Some(p) = self.stack.last_mut() {
                        p.members += 1;
                        p.expect = Expect::AfterValue;
                    }
                    i += 1;
                }
                ':' => match self.stack.last().map(|f| &f.expect) {
                    Some(Expect::Colon) => {
                        self.stack.last_mut().unwrap().expect = Expect::Value;
                        i += 1;
                    }
                    Some(Expect::Key) => {
                        // the missing-key colon: the pair drops (fixed
                        // parity) before an object's first member; the
                        // compound shape is the engine's key-reparse (out)
                        if self.stack.last().unwrap().members > 0 {
                            self.divergent = true;
                            return;
                        }
                        i += 1;
                    }
                    _ => {
                        self.divergent = true;
                        return;
                    }
                },
                ',' => match self.stack.last().map(|f| &f.expect) {
                    Some(Expect::AfterValue) => {
                        let f = self.stack.last_mut().unwrap();
                        f.expect = if f.obj { Expect::Key } else { Expect::Value };
                        i += 1;
                    }
                    Some(Expect::Value) => {
                        // a double comma: the engine heals, the stream
                        // too (pinned); stay in ExpectValue
                        i += 1;
                    }
                    _ => {
                        self.divergent = true;
                        return;
                    }
                },
                '"' => {
                    // a string opens: a key where a key was due, else a
                    // value; a value-start at Colon state is the
                    // missing-colon shape (out)
                    match self.stack.last().map(|f| &f.expect) {
                        Some(Expect::Key) => {
                            self.stack.last_mut().unwrap().members += 1;
                        }
                        Some(Expect::Value) => {
                            self.stack.last_mut().unwrap().expect = Expect::AfterValue;
                        }
                        _ => {
                            self.divergent = true;
                            return;
                        }
                    }
                    self.in_string = true;
                    self.esc = StrEsc::None;
                    self.raw_bs_run = 0;
                    self.saw_backslash = false;
                    self.content_ends_nl = false;
                    i += 1;
                }
                c if c.is_ascii_digit() || c == '-' || c == '.' || c.is_alphabetic() => {
                    // a number/word run: a value-start at Colon state is
                    // the missing-colon shape (out)
                    let Some(f) = self.stack.last_mut() else {
                        // a top-level scalar: the whole input one value
                        self.top_done = true;
                        continue;
                    };
                    match f.expect {
                        Expect::Key => {
                            // a bare-word key: the run runs to its
                            // terminator (the scan does not model it)
                            f.members += 1;
                            f.expect = Expect::Colon;
                        }
                        Expect::Value => {
                            f.expect = Expect::AfterValue;
                        }
                        Expect::AfterValue => {
                            // the missing-comma shape (pinned parity in
                            // both container kinds): the next member/item
                            f.members += 1;
                            f.expect = if f.obj { Expect::Colon } else { Expect::Value };
                        }
                        Expect::Colon => {
                            self.divergent = true;
                            return;
                        }
                    }
                    // the run's chars, to its terminator: `:` after an
                    // OBJECT-KEY run; `,`/`}`/`]`/`:`/whitespace at a
                    // value; the terminator decides whether the run
                    // COMPLETED (the mid-document float-spelling split
                    // needs a token AFTER the run)
                    let obj_key_run = self.stack.last().unwrap().expect == Expect::Colon;
                    let mut terminated = false;
                    self.run_char(c);
                    i += 1;
                    while i < n {
                        let d = chars[i];
                        if d.is_whitespace() {
                            terminated = true;
                            break;
                        }
                        if obj_key_run && d == ':' {
                            terminated = true;
                            break;
                        }
                        if !obj_key_run && matches!(d, ',' | '}' | ']' | ':') {
                            terminated = true;
                            break;
                        }
                        self.run_char(d);
                        i += 1;
                    }
                    self.end_run(terminated);
                }
                '(' => {
                    if self.stack.last().is_some_and(|f| f.expect == Expect::Colon) {
                        self.divergent = true;
                        return;
                    }
                    match self.stack.last_mut() {
                        Some(f) => {
                            if f.expect == Expect::Key {
                                f.members += 1;
                            }
                            f.expect = Expect::AfterValue;
                        }
                        None => self.top_done = true,
                    }
                    self.stack.push(ScanFrame {
                        obj: false,
                        paren: true,
                        members: 0,
                        expect: Expect::Value,
                    });
                    i += 1;
                }
                _ => {
                    // any other char: garbage the engine's repair lane
                    // re-decides — out
                    self.divergent = true;
                    return;
                }
            }
        }
    }

    /// A plain (escape-free) char inside the open string: the structural
    /// content is the re-sync family (out); the string closes at its
    /// own delimiter.
    fn string_content(&mut self, c: char) {
        if c == '"' {
            if self.raw_bs_run >= 2 {
                // the doubled-escape family: the engine re-pairs the run
                self.divergent = true;
                return;
            }
            self.nl_tail_string = self.content_ends_nl;
            self.raw_bs_run = 0;
            self.content_ends_nl = false;
            self.in_string = false;
            match self.stack.last_mut() {
                Some(f) => {
                    f.expect = match f.expect {
                        Expect::Key | Expect::Colon => Expect::Colon,
                        _ => Expect::AfterValue,
                    };
                }
                None => self.top_done = true,
            }
            return;
        }
        if matches!(c, '{' | '}' | '[' | ']' | ',' | ':') {
            // a structural char inside an open string: the engine's
            // re-sync family (out)
            self.divergent = true;
            return;
        }
        if c < ' ' || c == '\u{7f}' || c == '\u{85}' || c == '\u{2028}' || c == '\u{2029}' {
            // a raw control/line-break char inside a string: the
            // engine's string lane re-decides (the LLM multi-line
            // handling) — out
            self.divergent = true;
            return;
        }
        if c != ' ' {
            self.content_ends_nl = false;
        }
    }

    /// The close-up: the open string's quote, then per open frame the
    /// filler (a dangling key's `: null`, a missing value's `null`) and
    /// the frame's closer. `None` when the cut sits in a documented
    /// divergence: an object's FIRST member dangling keyless (the
    /// engine's whole-text fallback heals `{"a"`/`{""` to an array,
    /// the stream drops the member).
    fn close_up(&self, s: &str) -> Option<String> {
        if self
            .stack
            .iter()
            .any(|f| f.obj && f.expect == Expect::Colon && f.members <= 1)
        {
            return None;
        }
        if self.in_string && self.saw_backslash {
            // the doubled-quote/escape re-pair family's territory
            return None;
        }
        let mut out = String::from(s);
        if self.in_string {
            out.push('"');
        }
        for f in self.stack.iter().rev() {
            match f.expect {
                Expect::Colon => out.push_str(": null"),
                Expect::Value => out.push_str("null"),
                Expect::Key | Expect::AfterValue => {}
            }
            out.push(if f.obj { '}' } else { ']' });
        }
        Some(out)
    }
}

fuzz_target!(|s: &str| {
    // Several deterministic split sets per input: one-push (no splits),
    // three-char chunks (escape/pair straddling), single-char pushes,
    // and a strided scatter. Every assertion below runs per split set.
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let split_sets: Vec<Vec<usize>> = vec![
        Vec::new(),
        (0..n).step_by(3).collect(),
        (0..n).collect(),
        (0..n).filter(|i| i % 7 == 3).collect(),
    ];

    // The whole-text spelling's answer: the empty-string sentinel
    // renders bare (the repaired value Str("") is the nothing-
    // recoverable spelling), exactly what repair_json returns.
    let engine_out = match repair(s, &RepairConfig::default()) {
        Ok((Value::Str(empty), _)) if empty.is_empty() => Some(String::new()),
        Ok((v, _)) => Some(dumps(&v, true)),
        Err(_) => None,
    };
    // The one-push answer (None when the depth cap fired: the state is
    // unspecified after an error, so error-free splittings compare only
    // against a None-free reference).
    let one_out = (|| {
        let mut once = StreamingRepairer::new(true);
        once.push(s).ok()?;
        once.end().ok()
    })();

    for splits in &split_sets {
        let mut r = StreamingRepairer::new(true);
        let mut prev = 0usize;
        let mut errored = false;
        for &p in splits {
            let p = p.min(n);
            if p < prev {
                continue;
            }
            let chunk: String = chars[prev..p].iter().collect();
            prev = p;
            if r.push(&chunk).is_err() {
                errored = true;
                break;
            }
            // Snapshot is loadable-or-sentinel at every point, and equals
            // end on the same prefix (the machine's own render, pure).
            let snap = r.snapshot();
            assert!(
                snap.is_empty() || loads_strict(&snap).is_ok(),
                "snapshot is not strict-loadable: {snap}"
            );
            assert_eq!(r.end().expect("end never fails"), snap, "snapshot != end");
        }
        if errored {
            continue;
        }
        let tail: String = chars[prev..].iter().collect();
        if r.push(&tail).is_err() {
            continue;
        }
        let final_text = r.end().expect("end never fails");
        assert!(
            final_text.is_empty() || loads_strict(&final_text).is_ok(),
            "end() is not strict-loadable: {final_text}"
        );
        if let Some(want) = &one_out {
            assert_eq!(&final_text, want, "end() drifted across chunkings");
        }
    }

    // The ENGINE-PARITY classes: the scan decides (one pass), the
    // streamed end() must equal the whole-text engine's answer under
    // every split set.
    let mut scan = Scan::new();
    scan.walk(s);
    let parity = if loads_strict(s).is_ok() {
        Some(())
    } else if !scan.divergent {
        scan.close_up(s)
            .filter(|closed| loads_strict(closed).is_ok())
            .map(|_| ())
    } else {
        None
    };
    if parity.is_some()
        && let Some(want) = engine_out
    {
        for splits in &split_sets {
            let mut r = StreamingRepairer::new(true);
            let mut prev = 0usize;
            let mut ok = true;
            for &p in splits {
                let p = p.min(n);
                if p < prev {
                    continue;
                }
                let chunk: String = chars[prev..p].iter().collect();
                prev = p;
                if r.push(&chunk).is_err() {
                    // past the 200-container cap: the engine's own
                    // normalized guard fired on both sides; skip
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let tail: String = chars[prev..].iter().collect();
            if r.push(&tail).is_err() {
                continue;
            }
            assert_eq!(
                r.end().expect("end never fails"),
                want,
                "streamed != engine on split set {splits:?} of {s:?}"
            );
        }
    }
});
