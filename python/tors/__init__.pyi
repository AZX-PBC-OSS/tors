from collections.abc import Iterator, Sequence
from typing import Any, Literal, SupportsIndex, TypedDict

# The recursive JSON value: what `content_hash` accepts — the JSON
# grammar's set plus tuple, with dict keys restricted to
# str/int/float/bool/None (anything else is a TypeError). Non-str keys
# hash as their JSON spelling ("1", "1.0", "true", "null"), so dicts
# that compare equal under Python's key aliasing ({True: x} == {1: x} ==
# {1.0: x}) hash differently; canonicalize numeric-key dicts before
# using content_hash as a cache or dedup key. Recursive aliases need
# forward references on every level below the top.
JSONValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JSONValue"]
    | tuple["JSONValue", ...]
    | dict[str | int | float | bool | None, "JSONValue"]
)

# The scrub report's one redaction span, ordered by start, codepoint
# indices into the INPUT text.
class Span(TypedDict):
    type: str
    start: int
    end: int

# `scrub_pii_report`'s shape, all four keys present every time:
# `text` is the scrubbed output (== scrub_pii(...) byte-exact),
# `redacted` the per-rule + per-family counts (lowercase family names,
# absent types omitted), `skipped` the detected-but-preserved families
# (always {} when families=None), `spans` the redaction spans.
class ScrubPiiReport(TypedDict):
    text: str
    redacted: dict[str, int]
    skipped: dict[str, int]
    spans: list[Span]

# One `highlight` snippet: CHARACTER offsets (`start`/`end`, Python
# codepoint indices) into the ORIGINAL text argument — `text[start:end]`
# is exactly `text` — and the span's ROUGE-W-shaped F1 against the query
# (`score`, in [0.0, 1.0]; the monotone max-on-match variant — see the
# note on `highlight` below).
class GroundingSnippet(TypedDict):
    text: str
    start: int
    end: int
    score: float

# `highlight`'s shape, both keys present every time: `snippets` ordered by
# position, non-overlapping (empty when there is no overlap at all), and
# `score` the best snippet's score (0.0 when `snippets` is empty).
class GroundingResult(TypedDict):
    snippets: list[GroundingSnippet]
    score: float

# `ground_sentences`' shape: one GroundingSnippet-shaped entry per UAX #29
# sentence of the text, in position order (its start/end are the sentence's
# bounds — the exact tuples `tors.sentence_bounds(text)` returns — so
# text[start:end] is exactly `text`, token-free sentences included), and
# `score` the best sentence's score — the aggregate is the MAX, not the
# mean (0.0 when `sentences` is empty).
class SentenceGrounding(TypedDict):
    sentences: list[GroundingSnippet]
    score: float
# One `repair_json_diagnostics` entry, all six keys present every time
# (`from`/`to`/`suggestion` are None when the action did not move a value
# or offer a hint — a stable shape consumers can index blindly). The
# action vocabulary is the docstring's closed list (coerce/... /
# unwrap_root_array) but the runtime spell is a plain str: the stub does
# not over-claim a Literal the Rust does not enforce. Functional spelling:
# `from` is a keyword.
RepairAction = TypedDict(
    "RepairAction",
    {
        "action": str,
        "path": str,
        "detail": str,
        "from": "JSONValue | None",
        "to": "JSONValue | None",
        "suggestion": str | None,
    },
)

__version__: str
"""The installed distribution's version, read from its metadata
(``importlib.metadata.version("tors")``) at import; ``"unknown"`` where
the distribution metadata is absent (a source tree imported off-path) —
never a second literal to drift against the wheel."""

# The Snowball languages `rust-stemmers` ships: see tokenize_impl.rs's
# STEMMER_LANGUAGES (this is that same list, spelled as a type). Shared by
# tf_idf's and bm25_rank's stemmer= parameter rather than duplicated.
StemmerLanguage = Literal[
    "arabic",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hungarian",
    "italian",
    "norwegian",
    "portuguese",
    "romanian",
    "russian",
    "spanish",
    "swedish",
    "tamil",
    "turkish",
]

def normalize(text: str) -> str: ...
def finalize(text: str) -> tuple[str, str]: ...

# Replace every maximal run of C0 controls (U+0000-U+001F, tabs and
# newlines included) and DEL (U+007F) with a single ASCII space:
# byte-identical to re.compile(r"[\x00-\x1f\x7f]+").sub(" ", text). C1
# controls (U+0080-U+009F) pass through untouched, deliberately (the
# adopted call-site regexes do not cover them either; covering them would
# silently change adopted behavior). No edge strip: a control run at
# either end becomes an edge space for the caller to strip.
# tors.strip_controls(s) is s exactly when s holds no C0/DEL character.
#
# GIL note: detached_transform's shape, the same as normalize.
def strip_controls(text: str) -> str: ...

# Replace contact material (email addresses, phone numbers) and
# credential material (the evidence-backed api-key families) inside
# free text with correlation tokens: `@domain~<12 hex>` for an email,
# `prefix~<12 hex>` for a phone number: a `+`-led match keeps its
# dialling prefix, the first three code points ("+47" compact, "+1 "
# for the international spelling of a NANP number); a domestic match
# keeps the digest alone, no digits of the number (its head digits are
# the area code), and
# `<family prefix>~<12 hex>` for an API key (the prefix verbatim --
# sk-, sk-ant-, github_pat_, AIza, Bearer -- the non-secret half that
# tells the operator WHICH credential to rotate). The contact rules are
# a port of a private consumer's telemetry-safety module, pinned
# byte-identical to it at salt="" (tests/reference.py's quoted-pin
# oracle is the transcription); the api_keys rule is the credential
# extension past that contract (the JWT family is marker-scoped: a
# bare eyJ triple is never touched, the non-secret-cursor hazard).
# rules=None applies every rule in the canonical order (the keys pass
# FIRST -- a key's tail can spell a domestic phone run and a whole key
# an email local part -- then email, then phone over its result); []
# is the identity (the original object); an unknown name is a
# ValueError naming the accepted set. salt=None resolves PER RULE: the
# contact rules keep "tors/scrub_pii/v1" and the keys rule its own
# "tors/scrub_keys/v1" tag (a fixed, non-secret domain-separation tag
# per rule -- a key digest can never alias a contact digest; a KNOWN
# salt still leaves candidate-list confirmation possible -- the
# tokens are redaction, not pseudonymization crypto; deployments that
# care pass their own salt, and a consumer migrating from an unsalted
# scrubber passes salt="" to keep its token values byte-identical for
# every rule). The phone rule's digit class is Unicode Nd (every
# decimal digit script). Two grammars feed it: the `+`-led
# international one and the domestic NANP shapes (an un-plussed run of
# exactly ten digits, or eleven with an ASCII leading `1`, carrying a
# separator): the domestic match's token keeps no digits, the digest
# alone. Bare digit runs never match (the `+` anchoring is deliberate:
# order numbers and byte counts must survive). families=None scrubs every key
# family this version knows
# (the set grows on new families -- callers needing stability list
# names explicitly); a list selects exactly those families (order
# irrelevant, duplicates deduped); unknown names and [] are
# ValueErrors; the selection is ignored when api_keys is not active.
# tors.scrub_pii(s, ...) is s exactly when no active rule matches.
#
# GIL note: detached_transform's shape, the same as
# normalize/strip_controls -- the keys pass rides the same single
# detach.
def scrub_pii(
    text: str,
    rules: Sequence[Literal["contact_email", "contact_phone", "api_keys"]] | None = None,
    *,
    salt: str | None = None,
    families: Sequence[str] | None = None,
) -> str: ...

# The canonical key-family tuple, in the KeyFamily discriminant order: the
# base for "all but X" comprehensions
# (families=[f for f in tors.KEY_FAMILIES if f != "jwt"]) and the set
# the families= unknown-name error names. The set grows on new
# families (semver-visible); list names explicitly for stability.
KEY_FAMILIES: tuple[str, ...]

# The report twin of scrub_pii: the same scrub for the same arguments
# (report["text"] == scrub_pii(...) byte-exact) plus the accounting --
# per-rule counts plus per-family counts (lowercase family names,
# absent types omitted) under "redacted", the detected-but-preserved
# families under "skipped" (the "preserved a JWT, log it separately"
# signal; always {} when families=None), and the redaction spans under
# "spans" (ordered by start, codepoint indices into the INPUT text,
# {"type": <rule> | "api_keys:<family>", "start": int, "end": int}).
# Empty input is the empty accounting. Same single-detach GIL model.
def scrub_pii_report(
    text: str,
    rules: Sequence[Literal["contact_email", "contact_phone", "api_keys"]] | None = None,
    *,
    salt: str | None = None,
    families: Sequence[str] | None = None,
) -> ScrubPiiReport: ...


# Named-rule log and exception-text scrubbing, byte-identical to the
# grammar definition it ships with (the four compiled regexes are quoted
# in tests/reference.py and differentially enforced by
# tests/test_scrub_log_text_parity.py). rules=None
# runs the full chain in canonical order (pg_detail_lines -> uri_userinfo
# -> the conninfo pass, whose uri_query_creds / libpq_conninfo_creds names
# select the two anchor grammars of ONE pass); [] is the identity;
# duplicates dedupe and caller order is irrelevant. An unknown name raises
# ValueError naming the accepted set. SECURITY POLICY (issue #107,
# inverting 0.7.0): the repr-flattened DETAIL run is FAIL-CLOSED — a
# delimiter miss (an unterminated repr, or one with text after the quote)
# scrubs through end of line, never left alone.
# tors.scrub_log_text(s, rules) is s exactly when no rule fires (the ***
# fixed points fire and return a fresh, equal string).
#
# GIL note: detached_transform's shape (the rules= name walk under the
# GIL, the whole multi-rule pass under one detach).
def scrub_log_text(
    text: str,
    rules: Sequence[
        Literal["pg_detail_lines", "uri_userinfo", "uri_query_creds", "libpq_conninfo_creds"]
    ]
    | None = None,
) -> str: ...
def nfc(text: str) -> str: ...
def nfd(text: str) -> str: ...
def nfkc(text: str) -> str: ...
def nfkd(text: str) -> str: ...

# Raises ValueError on CPython 3.11+ when a decimal numeric reference's digit
# run exceeds sys.get_int_max_str_digits() (default 4300): the stdlib's own
# int-parse limit, with its exact message ("Exceeds the limit (4300 digits)
# for integer string conversion: value has 4301 digits; use
# sys.set_int_max_str_digits() to increase the limit"); the limit is read
# from the running interpreter once per call, so sys.set_int_max_str_digits()
# changes are honored on the next call. HEX references are exempt (base 16 is
# a power of two; the limit applies only to non-power-of-two bases). CPython
# 3.10 has no limit at all, and tors matches it there: nothing raises.
def html_unescape(text: str) -> str: ...
def grapheme_count(text: str) -> int: ...
def word_bounds(text: str) -> list[tuple[int, int]]: ...
# Single-thread only: holds a mutable native cursor (the class is not
# frozen) and must not be drained from two threads — a concurrent
# __next__ raises "RuntimeError: Already borrowed" (clean, no corruption);
# the Compiled* classes are the shareable ones (frozen + Arc).
def word_bounds_iter(text: str) -> Iterator[tuple[int, int]]: ...

# GIL note: same shape as the list/iterator word_bounds pair: the whole
# segmentation runs with the GIL released (one detached pass; the iterator
# fills its buffer under that one detach at construction and each __next__
# holds the GIL for a single 2-tuple). Sentence counts are ~1/10th word
# counts on prose, so the list API's marshalling is proportionally cheaper
# here. UAX #29 rule-based segmentation only: no dictionary segmentation
# for spaceless scripts (Thai/Khmer/Burmese/Japanese); see docs/api.md.
def sentence_bounds(text: str) -> list[tuple[int, int]]: ...
def sentence_bounds_iter(text: str) -> Iterator[tuple[int, int]]: ...
def decode_utf8(raw: bytes, *, errors: Literal["strict", "replace"] = "strict") -> str: ...
def finalize_utf8(
    raw: bytes, *, errors: Literal["strict", "replace"] = "strict"
) -> tuple[str, str]: ...
def b64_encode_bytes(raw: bytes) -> str: ...
def b64_decode(s: str, *, validate: bool = True) -> bytes: ...
def utf8_is_valid(raw: bytes) -> bool: ...

# BOM-sniffing native order, or an explicit little/big order that never
# sniffs or strips a BOM: matching Python's "utf-16" vs "utf-16-le"/
# "utf-16-be" codec names exactly. No encode_utf16/finalize_utf16: encoding
# a trusted str carries none of the untrusted-bytes-parsing risk that
# motivates the decode side, and str.encode("utf-16-le") already covers it.
def decode_utf16(
    raw: bytes,
    *,
    errors: Literal["strict", "replace"] = "strict",
    byteorder: Literal["native", "little", "big"] = "native",
) -> str: ...
def utf16_is_valid(
    raw: bytes, *, byteorder: Literal["native", "little", "big"] = "native"
) -> bool: ...

# The RFC 8259 validity gate: True exactly when orjson.loads(data) would
# succeed -- the acceptance set is orjson 3.x's, not the stdlib's (three
# classes differ from json.loads: float-overflow literals reject -- 1e400
# and friends, where the stdlib hands back inf; NaN/Infinity/-Infinity
# reject; a leading UTF-8 BOM rejects), because the gate stands in front
# of a parse-and-discard consumer whose next step IS orjson.loads. Booleans
# only: invalid input answers False (the 1024-container depth cap, orjson's,
# included) -- nothing raises for invalid input; a wrong-TYPE argument
# (not bytes, not str) raises TypeError like the bytes-in surface.
#
# GIL note: utf8_is_valid's class for a bytes argument (zero-copy borrow,
# one detached scan, a bool return -- no marshalling class at all); a str
# argument pays the standard str-in borrow first under the GIL (zero-copy
# for ASCII or an already-cached UTF-8 view, a one-time O(input)
# materialization on the first non-ASCII call, cached on the object), then
# the scan detaches. No aio twin (see docs/async.md).
def json_is_valid(data: bytes | str) -> bool: ...

# A heuristic guess, not a validator: the intended pipeline is utf8_is_valid
# first, and detect_encoding only on bytes that already failed that check.
# Always returns some codec name (WHATWG Encoding Standard labels, e.g.
# "windows-1252" / "Shift_JIS" / "UTF-8": case-insensitively valid for
# bytes.decode()), never raises for well-formedness reasons. tld is an
# ASCII top-level domain without the leading dot ("jp", not ".jp").
def detect_encoding(raw: bytes, *, tld: str | None = None) -> str: ...

# GIL note (the word_bounds list-shape precedent): the diff itself runs with
# the GIL released, but building the returned list constructs one 5-tuple per
# opcode under the GIL: O(number of opcodes), measured ~0.1-0.15 µs/op (a
# ~10-15 ms hold above the ping floor for the 103,421-opcode 12 MiB
# shuffled-pair cell in tests/test_gil_release.py; see docs/performance.md
# for the measured band).
#
# deadline_ms bounds the whole call (default None = the unbounded diff,
# unchanged). The Myers search's work on hard inputs (few anchorable unique
# records, e.g. a character-level permutation) grows superlinearly with size
# (measured on the dev box, ambient load 2.6-3.7: 50k chars 0.32 s, 200k
# 3.67 s, 400k 13.81 s, 1M 183.6 s, roughly ~n^2; see docs/performance.md
# for the ladder). On expiry the
# incomplete result is discarded and TimeoutError is raised naming the
# elapsed cost and the deadline; the exception is constructed after the GIL
# is reacquired, so nothing raises from inside the GIL-released region. A
# non-positive or non-finite value raises ValueError before any work runs.
def diff_opcodes(
    a: str, b: str, *, deadline_ms: float | None = None
) -> list[tuple[str, int, int, int, int]]: ...

# GIL note: identical to diff_opcodes' classes: the whole line split +
# Myers search runs with the GIL released (the deadline_ms budget checked
# inside the detached region; TimeoutError constructed after the GIL is
# reacquired), and the return marshalling constructs one 5-tuple per line
# opcode with interned tag strings (op[0] is "equal" holds, exactly as it
# does for difflib's own tuples). Indices address lines in '\n'-only
# tokenization shape (a.split("\n")-with-terminators-reattached, each line
# keeping its \n, the last may lack one) — not str.splitlines(keepends=True),
# which additionally breaks on \r: a bare \r is not a break here, so a
# \r-only file is one line, not several; a line diff's op count sits far
# below its char-level twin's. See docs/api.md for the full contract.
def diff_opcodes_lines(
    a: str, b: str, *, deadline_ms: float | None = None
) -> list[tuple[str, int, int, int, int]]: ...

# GIL note: the find_patterns argument shape over a dict: one GIL-held walk
# borrowing each key and value, then automaton build + scan + splice with
# the GIL released, then either the identity return (no key matched, or the
# net effect is the identity: tors.replace_many(s, m) is s exactly when
# tors.replace_many(s, m) == s) or the O(output) string marshalling. No
# list-shape class: the return is one string. Leftmost-longest semantics
# (not re.sub's leftmost-first), non-overlapping, replacements never
# re-scanned; dict order cannot matter.
def replace_many(text: str, replacements: dict[str, str]) -> str: ...

# GIL note (the word_bounds list-shape precedent, again): the automaton build,
# the scan, and the byte→char offset conversion all run with the GIL released,
# but building the returned list constructs one 3-tuple of ints per match under
# the GIL: O(number of matches). See docs/performance.md and
# tests/test_gil_release.py for the measured sparse/dense bands.
def find_patterns(patterns: list[str], text: str) -> list[tuple[int, int, int]]: ...

# GIL note: the streaming spelling of find_patterns: the whole search
# (pattern-list walk, automaton build, scan, byte→char conversion) fills an
# internal buffer under one GIL-released pass at construction, and each
# __next__ holds the GIL only for a single 3-tuple of ints. The streaming
# answer to the list shape's O(matches) tuple-marshalling caveat (~13 ms
# held per 100k matches in the list shape). Same sequence as the list API,
# pinned.
def find_patterns_iter(patterns: list[str], text: str) -> Iterator[tuple[int, int, int]]: ...

# GIL note: the count spelling of find_patterns: the same
# leftmost-longest, non-overlapping search answering just the number, with
# no match vector (the list spelling materializes ~250 MiB of matches on a
# 100 MiB dense corpus just to answer "how many") and no byte→char pass
# (counting is offset-free). count_matches(p, t) == len(find_patterns(p, t)),
# pinned. A single int return: no marshalling class at all.
def count_matches(patterns: list[str], text: str) -> int: ...

# The escape-parity byte scan: an occurrence of needle at offset i counts
# only when the maximal run of b"\\" immediately before i has EVEN length
# (0 is even, so an occurrence at offset 0 counts); an odd run means the
# run's backslash pairs escape each other and the leftover one escapes the
# occurrence's first byte, so the occurrence is literal text. The
# motivating case is JSON's \u0000: a real NUL codepoint and the literal
# six-character text render byte-ambiguously (the literal contains the
# escape at +1, behind one backslash), and PostgreSQL jsonb rejects only
# the real one (SQLSTATE 22P05) — parity settles it without re-parsing.
#
# BYTES IN, BYTE OFFSETS OUT — read this twice: find_unescaped's return
# indexes the haystack's BYTES, never str codepoints. Over multibyte UTF-8
# content the byte offset and the decoded text's character offset are
# different numbers (find_patterns needed a byte→char mapping for exactly
# this confusion; this API has none because the input is bytes and the
# contract is byte-space end to end: haystack[i:i + len(needle)] is the
# needle). find_unescaped returns -1 when no live occurrence exists
# (bytes.find's own sentinel, kept over Optional[int] deliberately).
# Rejected (odd-run) hits advance the scan one byte past the hit, not
# past the whole match, so self-overlapping needles stay correct. An empty
# needle raises ValueError("empty needle") (it would match at every
# position, the find_patterns empty-pattern rationale). Exactly bytes on
# both arguments (bytearray/memoryview/str -> TypeError): the bytes-in
# surface's zero-copy immutable-borrow contract. No JSON knowledge lives
# in the functions: parity is the mechanism, "needle is an escape
# sequence" is the caller's reading of it.
#
# GIL note: utf8_is_valid's class exactly — two zero-copy PyBytes borrows,
# the whole scan under one py.detach, bool/int returns so no marshalling
# class exists, and no error path past the empty-needle ValueError (which
# fires under the GIL, before the detach). Ceiling-only heartbeat budget;
# no aio twin (a memchr-class scan at document scale is a
# sub-heartbeat-floor call).
def contains_unescaped(haystack: bytes, needle: bytes) -> bool: ...
def find_unescaped(haystack: bytes, needle: bytes) -> int: ...

# The scan surface's pinned companion (#52): the UTF-8 byte length of a
# str — len(s.encode("utf-8")) with the copy taken out. The count a
# caller wants when a size cap sits in front of a store (an enqueue
# path's idempotency-key/scope byte caps per enqueue, the terminal's re-encode
# of a serialized result of up to 64 KiB per success — a double pass:
# the byte count existed inside the serializer's output and was
# discarded by the .decode()). Companion, not standalone: it ships in
# the scan family's binding module with the same harness patterns, and
# honest sizing says the win is large inputs and hot paths only.
#
# Cache semantics (the deliberate implementation: the standard str-in
# borrow, not hand-rolled UCS arithmetic — a Rust &str IS its UTF-8
# bytes, so the core is one field read): ASCII is a zero-copy alias, so
# the call is O(1) with no allocation; a non-ASCII input's FIRST call —
# exactly the cold-cache case — materializes and caches the UTF-8 view
# on the str object (a CPython-internal cache, not a Python-visible
# bytes, filled by this borrow and by any earlier str-in tors call on
# the same object, read but never filled by encode: a prior
# len(s.encode()) does not warm it) — encode-parity cost, no
# Python-visible object; repeat calls on the same object are O(1),
# strictly better than len(s.encode()), which re-copies every call.
#
# Error parity: a str holding lone surrogates raises UnicodeEncodeError —
# CPython's own error from the borrow (the same exception encode raises,
# attributes included); no tors-side error path exists.
#
# GIL note: a single int return (no marshalling class); the call's only
# O(n) work is the borrow itself — the first non-ASCII call's
# materialization is GIL-held (the standard str-in first-call class),
# under the 10ms ping floor at 12 MiB; the detach around the O(1) core
# is nominal. No aio twin (an O(1)-to-borrow call needs no thread hop).
def utf8_byte_len(s: str) -> int: ...

# The interop twin (#52, the "len() to bytes" pair's other half): the
# UTF-16 byte length of a str — 2 bytes per BMP codepoint, 4 per astral
# codepoint (the surrogate pair) — len(s.encode("utf-16-le")) with the
# 2n copy taken out. The world that caps in these units: UTF-16 is the
# code-unit world of JavaScript, Java, Windows, and .NET (an astral
# emoji is length 2 in JS), so column caps (NVARCHAR), wire caps, and
# interop size checks there are UTF-16 bytes.
#
# Implementation: the utf8 twin's standard str-in borrow (NOT
# hand-rolled UCS arithmetic, NOT FFI) plus derived arithmetic over the
# UTF-8 view — 2 * (#codepoints + #astral), both counts byte classes
# (lead bytes; 4-byte leads 0xF0..=0xF4) — one pass, no allocation; the
# corners:
# no astral codepoints -> exactly 2 * len(s) for ALL BMP text (where
# the UTF-8 byte count diverges on CJK and combining marks), pure
# ASCII -> 2 * the UTF-8 byte count, and every answer is even.
#
# Cache semantics: the utf8 twin's exactly (same borrow, same
# CPython-internal UTF-8 view cache): ASCII is a zero-copy alias; a
# non-ASCII input's FIRST call — exactly the cold-cache case —
# materializes and caches the view (encode-parity cost, GIL-held; a
# prior encode does not warm it: encode reads the cache and never fills
# it); repeat calls borrow zero-copy and pay only the detached
# byte-class scan.
#
# Surrogates: REFUSAL PARITY with the replaced expression, measured —
# the strict encode("utf-16-le") raises UnicodeEncodeError on lone
# surrogates exactly like encode("utf-8") ("surrogates not allowed"),
# so utf16_byte_len refuses the same strings the expression itself
# refuses. The tors error is the str-in borrow's own (the crate-wide
# contract, every str-argument tors function's lane): .encoding "utf-8"
# (the flavor of the step that fails, materializing the UTF-8 view),
# where the expression's error says "utf-16-le". The stdlib's
# errors="surrogatepass" mode WOULD encode them (one unit each) — a
# mode tors deliberately does not offer.
#
# ⚠ Breaking differences from len(s.encode("utf-16-le")): (1) the
# error's .encoding is "utf-8", not "utf-16-le" (encoding-label switch
# for callers matching on it); (2) on a multi-surrogate run the
# borrow's (start, end) names the whole run (utf-8 whole-run span) where
# the utf-16-le error reports only the first unit (first-unit span);
# (3) errors="surrogatepass" is unsupported. See docs/api.md.
#
# Overflow: 2 * (codepoints + astral) is checked, and unreachable for
# real inputs on every width (a &str is at most isize::MAX bytes and
# the count sum never exceeds one per byte, so the doubled answer is at
# most 2 * isize::MAX, which fits usize on 32-bit and 64-bit alike):
# OverflowError is the loud refusal if that invariant ever breaks.
#
# GIL note: a single int return (no marshalling class); the GIL-held
# residue is the borrow (the cold-cache first call's materialization,
# under the 10ms ping floor at 12 MiB), and the detach carries the
# real O(n) scan (memchr-class, sub-floor). No aio twin (the residue
# is the borrow alone; the scan detaches).
def utf16_byte_len(s: str) -> int: ...

# GIL note (the CompiledLemmaDict discipline, over the search surface): the
# pattern list compiled once (one detached build at construction), then
# every call is the free function's scan classes minus the per-call
# automaton build: one Arc refcount bump, the scan under one detach, the
# same marshalling as the free spelling. Immutable after construction, so
# sharing one across many calls and threads is sound with no
# synchronization beyond the refcount (the *_iter objects are the
# opposite: single-thread, see word_bounds_iter). The replace spellings validate the
# replacements dict at call time (values change per call; the automaton is
# the compiled part): it must key exactly the compiled pattern set, every
# pattern paired with a value and no others (ValueError otherwise, naming
# the unknown keys and the missing count), which is what makes
# cp.replace_many(text, m) == tors.replace_many(text, m) hold by
# construction. The empty pattern list compiles (its scans find nothing)
# and then accepts only the empty dict. Argument contracts otherwise match
# the free functions' exactly (list-of-str patterns at construction, the
# empty-pattern-string ValueError, lone-surrogate UnicodeEncodeError, the
# one-character mask rule on the masked spelling).
class CompiledPatterns:
    """A pattern list compiled once: the `re.compile()` answer to the free
    search spellings' per-call automaton build. Build once, reuse across
    many calls (a fixed vocabulary over a document pipeline) instead of
    rebuilding the same automaton every call. Immutable once built."""

    def __init__(self, patterns: list[str]) -> None: ...
    def __len__(self) -> int: ...
    def find(self, text: str) -> list[tuple[int, int, int]]: ...
    def find_iter(self, text: str) -> Iterator[tuple[int, int, int]]: ...
    def count(self, text: str) -> int: ...
    def replace_many(self, text: str, replacements: dict[str, str]) -> str: ...
    def replace_many_masked(
        self, text: str, replacements: dict[str, str], mask: str = "*"
    ) -> str: ...

# GIL note: the grapheme_count precedent for the word segmenter: a single
# int return (no marshalling class), the whole scan GIL-released, and O(1)
# memory where len(word_bounds(text)) materializes the full tuple list.
# word_count(t) == len(word_bounds(t)), pinned.
def word_count(text: str) -> int: ...

# GIL note: word_count's shape over sentences: single int, O(1) memory,
# sentence_count(t) == len(sentence_bounds(t)), pinned.
def sentence_count(text: str) -> int: ...

# CommonMark §4.5 fenced code blocks, hand-rolled (not a Markdown parser
# dependency): (language, code, start, end) tuples in document order.
# start/end are Python str index (codepoint) offsets of the block's raw
# span; language is the info string's first word or None; code has the
# fence's own indentation (0-3 leading spaces) stripped from each content
# line. lang= filters to an exact (case-sensitive) language match. An
# unterminated fence still yields a block, running to end of input.
# Narrower than full CommonMark: no indented code blocks, no
# tab-expansion of indentation.
#
# GIL note: the whole scan runs with the GIL released; the return
# marshalling is O(blocks) tuples.
def extract_code_blocks(
    text: str, lang: str | None = None
) -> list[tuple[str | None, str, int, int]]: ...

# If text, trimmed of leading/trailing whitespace, is exactly one fenced
# code block (including the unterminated-fence case), return its dedented
# code content; otherwise return text unchanged (not even
# whitespace-trimmed). tors.strip_code_fences(s) is s exactly when s is not
# the single-fenced-block case.
#
# GIL note: the detached_transform shape shared by normalize/nfc/etc: str-in
# argument borrow, the scan with the GIL released, then either the original
# object back (zero marshalling) or the O(output) unwrapped string.
def strip_code_fences(text: str) -> str: ...

# textwrap.dedent(text), byte-for-byte: the longest common leading
# whitespace-run string (tabs and spaces are distinct characters: "  x"
# and "\tx" share no margin) is stripped from every line, and
# whitespace-only lines normalize to empty. tors.dedent(s) is s exactly
# when tors.dedent(s) == s.
#
# GIL note: detached_transform's shape, the same as strip_code_fences.
def dedent(text: str) -> str: ...

# Repair malformed JSON (the LLM-output shape: missing commas/quotes/
# brackets, truncated values, stray prose, comments, Python-isms) and
# return the repaired document as a string, re-serialized to json.dumps'
# canonical form exactly like the json_repair port this implements
# (upstream: mangiucugna/json_repair, MIT): valid-but-noncanonical input
# normalizes; the nothing-recoverable sentinel renders as the bare empty
# string. Whole-input single-fence payloads unwrap first via the CommonMark
# fence grammar (~~~, longer closers, indented fences included).
#
# GIL note: the whole repair (strict fast path, repair parser, schema
# alignment, validator) runs with the GIL released; the residue is the
# schema-argument walk plus the O(output) string marshalling.
#
# deadline_ms (default None = unbounded) bounds the whole repair the way
# diff_opcodes' deadline_ms does: a positive-finite-or-None budget validated
# up front, TimeoutError on expiry ("<spelling> deadline exceeded: elapsed
# Xms > deadline_ms Yms"). The clock starts at the top of the call: the
# fence pre-pass and the json.loads fast-path attempt burn the budget too
# (a fast path that completes past the budget still returns its answer).
# It is a DoS backstop for the quadratic parser shapes shared with upstream
# json_repair (splice rescans and the backslash-run string scan); when a
# schema is passed it also bounds the schema alignment layer (key-remap
# ladder, union/type-union retries, coercion, fill, validation, same soft
# bound, including the salvage unwrap's nested repair which inherits the
# caller's budget).
# A
# bounded abort, not a speed-up; a completing parse is byte-identical
# whether or not a deadline is set. The bound is soft (the tight loops
# sample the clock 1-in-256, re-tightened after every O(n) splice/scan)
# and bounds CPU time, not native stack growth (runaway continuation
# recursions are depth-guarded separately). Unset costs nothing on the
# valid-JSON fast path and one predicted branch per parser dispatch turn
# (~+6% worst-case on a multi-MB skip_json_loads parse); set adds ≤2%.
# Applies to all three spellings.
def repair_json(
    s: str,
    *,
    skip_json_loads: bool = False,
    ensure_ascii: bool = True,
    strict: bool = False,
    # a pydantic v2 model (class or instance) is also accepted
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
    deadline_ms: float | None = None,
) -> str: ...

# The loads-mode spelling: the repaired document as decoded objects, the
# json.loads drop-in (the "" sentinel, not None, when nothing is
# recoverable). No ensure_ascii (no serialization happens).
#
# GIL note: the repair itself runs detached; the GIL-held residue is the
# O(result) object-tree construction (the word_bounds list-marshalling
# class) plus the schema-argument walk.
def repair_json_loads(
    s: str,
    *,
    skip_json_loads: bool = False,
    strict: bool = False,
    # a pydantic v2 model (class or instance) is also accepted
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
    deadline_ms: float | None = None,
) -> dict[str, Any] | list[Any] | str | int | float | bool | None: ...

# The diagnostics spelling: the loads-mode value plus the structured action
# log: a list of {action, path, detail, from, to, suggestion} dicts (the
# last three None when unused) over the closed action vocabulary
# (coerce/fill/insert_default/remap_key/suggest/drop_property/drop_item/
# unwrap_string/wrap_array/fill_required/format_date/skip_fragment/
# map_array_to_object/unwrap_root_array). Schema-free calls return an empty
# list in v1 (parser-level narration is a follow-up).
#
# GIL note: the loads shape plus O(diagnostics) dict construction.
def repair_json_diagnostics(
    s: str,
    *,
    skip_json_loads: bool = False,
    strict: bool = False,
    # a pydantic v2 model (class or instance) is also accepted
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
    deadline_ms: float | None = None,
) -> tuple[
    dict[str, Any] | list[Any] | str | int | float | bool | None,
    list[RepairAction],
]: ...

# Truncate to at most max_chars codepoints, cutting at the last word (or,
# boundary="sentence", sentence) boundary at or before max_chars: composing
# the crate's own word_bounds/sentence_bounds segmentation, not a new
# algorithm. Every cut point is also grapheme-cluster-safe: a word/sentence
# boundary that would split a cluster (e.g. Thai SARA AM, combining accents,
# ZWJ emoji sequences) is never used, so a combining mark is never separated
# from its base character. If no boundary fits at or before max_chars (a
# single word/sentence longer than the budget, or max_chars == 0), the
# fallback is a hard cut at the largest grapheme boundary <= max_chars
# (still cluster-safe, so it can land short of max_chars when the budget
# would otherwise split a cluster); the result never exceeds max_chars
# codepoints either way. The cut point is then trimmed of trailing
# whitespace, whole whitespace clusters only: the trim never ends the
# result mid-cluster (a Prepend plus a no-break space is one cluster, and
# its non-whitespace half keeps it whole). max_chars < 0 and an
# unrecognized boundary both raise
# ValueError. tors.truncate_to_bounds(s, n) is s exactly when s already has
# <= n codepoints.
#
# GIL note: detached_transform's shape: the segmentation scan and cut run
# with the GIL released, then either the original object back (zero
# marshalling) or the O(output) truncated string.
def truncate_to_bounds(
    text: str, max_chars: int, boundary: Literal["word", "sentence"] = "word"
) -> str: ...

# The DB-column truncation shape: hard cut to at most max_chars codepoints
# plus a U+2026 ellipsis marker, never mid-grapheme-cluster (combining
# accents, ZWJ sequences, flag pairs snap back past the whole cluster). No
# word/sentence awareness (unlike truncate_to_bounds): a storage bound is
# positional, not semantic. At most max_chars - 1 codepoints are kept plus
# the one-codepoint marker, so the result never exceeds max_chars (it falls
# short when cluster backoff requires it); no trailing-whitespace trim.
# tors.truncate_ellipsis(s, n) is s exactly when s already has <= n
# codepoints. max_chars == 0 yields "" (no room for even the marker);
# max_chars < 0 raises ValueError.
#
# GIL note: detached_transform's shape, the same as truncate_to_bounds.
def truncate_ellipsis(text: str, max_chars: int) -> str: ...

# Is claim grounded in source: fuzzy=False (default) is source.contains(claim)
# exactly; fuzzy=True is a windowed difflib-ratio scan of source against
# threshold (a lexical check: no NLI/semantic model; see the crate's
# grounded_impl module docs for exactly what the score measures and its
# DoS-bounded windowing over long sources). fuzzy=True is a superset of
# fuzzy=False: a verbatim substring is grounded before windowing and before
# deadline_ms applies (so threshold=1.0 fuzzy subsumes exact containment, and
# a verbatim claim never times out). Every windowed score uses the
# claim-length denominator 2*L -- a truncated tail window's missing chars
# are mismatches, never a discounted denominator, so the verdict never
# depends on where the evidence sits relative to the source's end (issue
# #40); a source shorter than the claim is the one exception, scored as one
# direct whole-source difflib 2*M/(m+n) ratio. Near matches are
# alignment-independent
# in the guarantee band: a region with aligned ratio r is detected at any
# offset whenever r >= max(0.75, threshold + 1/32) (a bounded refinement
# pass over the best coarse windows; one typo in a 9+ char claim clears the
# 0.85 default wherever it sits), best-effort below r = 0.75, and evictable
# from the 64 refinement candidates by adversarial decoys (the regime
# deadline_ms exists for). An empty claim is vacuously
# grounded in anything on both paths. threshold must be in [0.0, 1.0].
# deadline_ms is only accepted (and only meaningful) when fuzzy=True; it
# bounds the whole fuzzy scan the same way diff_opcodes' deadline_ms does:
# TimeoutError on expiry, a positive-finite-or-None precondition validated
# before any work runs.
#
# GIL note: fuzzy=False is one detached str.contains call; fuzzy=True runs
# the whole windowed scan with the GIL released, the TimeoutError (if any)
# constructed after the GIL is reacquired: the diff_opcodes shape.
def is_grounded(
    claim: str,
    source: str,
    *,
    fuzzy: bool = False,
    threshold: float = 0.85,
    deadline_ms: float | None = None,
) -> bool: ...

# The recall twin of is_grounded (the precision side): what fraction of the
# SOURCE's tokens does `text` actually utilize — the model-free
# operationalization of TRACe's uTilization metric (Friel, Belyi & Sanyal
# 2024, RAGBench §3.2: utilization = the length of the utilized context
# spans over the context's length). Measured as ROUGE-W recall (Lin 2004's
# Equation 15 R factor) over the grounding family's own UAX #29
# tokenization (case-fold + NFC, CJK per character) — one tokenization and
# one shaping shared with `highlight`/`ground_sentences`, so the precision
# and recall surfaces never disagree about what a token is; a text quoting
# a CONTIGUOUS passage of the source outscores one scattering the same
# tokens through filler (the weighted-LCS shaping). Qualification: the WLCS
# fill is a ROUGE-W-shaped monotone max-on-match recurrence (Lin 2004,
# Eq. 15, with the forced-diagonal branch replaced by max to preserve
# candidate monotonicity) — a greedy-run-weighted alignment score, not the
# literal weighted-LCS optimum. NOT bit-compatible with the official ROUGE
# package or rouge-score; the deviation is deliberate (monotonicity) and
# verified in tests. A lexical overlap
# signal, not a semantic one: it measures token coverage, not whether the
# information was genuinely used. Identical text and source are 1.0 (up to
# f64 rounding of the DP's accumulation, within 1e-9); disjoint, token-free,
# or empty operands are exactly 0.0 (TRACe's ratio is 0/0 there; 0.0 is the
# conservative reading). Cost: the classic
# weighted-LCS DP, O(|S|*|T|) time with O(min(|S|, |T|)) memory (two rows,
# never an n*m matrix); at most the first 16384 tokens of each operand are
# scanned, the denominator being the source tokens actually scanned.
#
# GIL note: the two argument borrows under the GIL, the whole
# tokenize/intern/score pass under one py.detach; the residue is a single
# float.
def grounding_coverage(source: str, text: str) -> float: ...

# Snippet-provenance grounding: WHERE the query's evidence sits in a chunk.
# A ROUGE-W-shaped weighted-LCS F1 (recall-oriented length-weighted LCS,
# Lin 2004) over UAX #29-tokenized anchor runs, expanded to sentence bounds
# when they fit the budget, returned as non-overlapping snippets in position
# order with
# CHARACTER offsets into the original `text` (`text[start:end]` is exactly
# the snippet's `text`, round-tripping through CJK/accents/emoji), plus the
# best snippet's score. Qualification: the WLCS fill is a ROUGE-W-shaped
# monotone max-on-match recurrence (Lin 2004, Eq. 15, with the
# forced-diagonal branch replaced by max to preserve candidate
# monotonicity) — a greedy-run-weighted alignment score, not the literal
# weighted-LCS optimum. NOT bit-compatible with the official ROUGE package
# or rouge-score; the deviation is deliberate (monotonicity) and verified
# in tests. Empty/token-free query or text, and
# `max_snippets=0`, return the empty result (never an error); `max_chars`
# bounds every snippet's length at token boundaries (a snippet always holds
# at least one token, even when the budget is smaller than that token; 0 is
# a ValueError). Pathological chunks are bounded: at most the first 16384
# text tokens and 128 query terms are scanned.
#
# GIL note: the argument borrows under the GIL, the whole tokenize/score/
# select pass under one py.detach (the is_grounded shape): the GIL-held
# residue is only the O(snippets) dict marshalling.
def highlight(
    query: str,
    text: str,
    *,
    max_snippets: int = 3,
    max_chars: int = 400,
) -> GroundingResult: ...

# Sentence-level grounding batch: EVERY UAX #29 sentence of `text`, scored
# against `query` with the same ROUGE-W F1 the snippet surface ranks spans
# with (the same qualified monotone max-on-match variant — see `highlight`'s
# note; not bit-compatible with the official ROUGE package), in position
# order — the bridge primitive a downstream NLI verifier
# (MiniCheck/SummaC style) consumes. Note the argument order:
# `ground_sentences(text, query)` — the OPPOSITE of `highlight(query,
# text)`. The citation unit is the sentence (the
# ALCE baselines' unit: Gao et al. 2023); the score is a RANKING signal, not
# an answerability verdict (Joren et al. 2024, "Sufficient Context": a
# lexical overlap cannot judge sufficiency — run a model over the
# top-scored sentences for that). The aggregate `score` is the best
# sentence's F1 (the MAX, not the mean: the retrieval signal the bridge
# needs, `highlight`'s own aggregate shape, and stable under irrelevant
# additions — one evidence sentence in a long document must not read as
# ungrounded because the document is long). An empty/token-free query
# scores every sentence 0.0 (the segmentation is the answer's shape; the
# query only drives scores); an empty text returns the empty result —
# degenerate input is a valid answer, never an error. `max_chars` bounds
# each sentence's SCORED window (a longer sentence is scored over its
# leading token-boundary window, at least one token; its reported span
# still covers the whole sentence — the exact window boundary is an
# implementation detail, deliberately not exposed); `None` scores whole
# sentences, 0 is a
# ValueError. Like every integer parameter in the library, `max_chars`
# takes a plain int (a bool is its 0/1 int value — the family-wide
# convention). Pathological chunks are bounded: at most the first 16384
# text tokens and 128 query terms are scanned (sentences past the cap score
# 0.0, their offsets and text still exact).
#
# GIL note: the argument borrows under the GIL, the whole segment/tokenize/
# score pass (linear in the text at a bounded query width) under one
# py.detach (the highlight shape): the GIL-held residue is only the
# O(sentences) dict marshalling.
def ground_sentences(
    text: str,
    query: str,
    *,
    max_chars: int | None = None,
) -> SentenceGrounding: ...

# GIL note: urllib.parse.quote/unquote are pure Python: a GIL-held
# whole-text pass for the most-used encoding operation in web/ingestion
# pipelines. These are byte-exact stdlib parity (pinned differentially per
# CI leg), one detached native pass, with the identity lane
# f(s) is s when nothing encodes/decodes. The stdlib's encoding/errors
# parameters are out of scope (UTF-8 only).
def quote(text: str, safe: str = "/") -> str: ...
def quote_plus(text: str, safe: str = "") -> str: ...
def unquote(text: str) -> str: ...
def unquote_plus(text: str) -> str: ...

# GIL note: difflib's ratio/get_close_matches shape over the same Myers
# engine as diff_opcodes: two str-in borrows, the search GIL-released, a
# single float out. get_close_matches returns the original candidate
# objects (references, zero marshalling). Validity-first parity (difflib's
# M is anchoring-dependent): exact agreement on the forced-alignment
# classes, the divergence rows pinned. deadline_ms bounds the whole call
# (TimeoutError on expiry; an enormous-but-finite budget saturates to
# unbounded).
def similarity_ratio(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...
def get_close_matches(
    word: str,
    possibilities: list[str],
    n: int = 3,
    cutoff: float = 0.6,
    *,
    deadline_ms: float | None = None,
) -> list[str]: ...

# GIL note: the edit-distance/similarity metrics CPython has no stdlib
# spelling of: O(n*m) DP passes GIL-released with a per-row/phase
# deadline_ms check (TimeoutError on expiry; the budget is the DoS guard:
# two 1 MiB strings are minutes of DP). Character-level; pinned to strsim
# 0.11 crate-side as the differential oracle; symmetric; single int/float
# returns, no marshalling class.
def levenshtein(a: str, b: str, *, deadline_ms: float | None = None) -> int: ...
def jaro(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...
def jaro_winkler(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...

# GIL note: replace_many's classes exactly (one GIL-held dict walk, the
# scan+splice GIL-released, one string out), with the length guarantee: for
# each matched span of L characters, its replacement value is truncated to
# its first L characters if the value has >= L characters, or emitted in
# full and padded with copies of `mask` out to L characters if it has fewer
# (mask is never consulted on the truncation branch), so the output's
# character count and every non-matching span's offsets equal the input's
# (redaction that keeps pre-computed offsets valid). Pass "" as a value to
# force pure masking. mask must be exactly one character (one codepoint;
# ValueError otherwise). Identity contract: is s exactly when == s.
def replace_many_masked(text: str, replacements: dict[str, str], mask: str = "*") -> str: ...

# The object content hash: lowercase-hex SHA-256 of the canonical form,
# EXACTLY json.dumps(obj, sort_keys=True, separators=(",", ":")) with
# default ensure_ascii and allow_nan -- byte-identical with the stdlib
# expression, pinned differentially against it. Leaves: str, int
# (arbitrary precision), float (Python's own repr spelling; NaN/Infinity/
# -Infinity literals), bool, None; containers: list, tuple (serializes as
# a list), dict (keys sorted BEFORE stringification; str/int/float/bool/
# None keys coerced to their json string form). Anything else raises
# TypeError naming the type; circular references raise ValueError; a str
# holding lone surrogates raises UnicodeEncodeError where json.dumps
# accepts it (the crate-wide str-borrow divergence, documented in
# docs/api.md). Deterministic: any dict key order yields the same hash,
# except a dict holding two distinct NaN keys — their output order is
# timsort's over an inconsistent comparator, insertion-order-dependent,
# matching the json oracle (see docs/api.md).
#
# GIL note: the object walk and leaf spellings run under the GIL (the
# standard arg-walk class, O(tree): one borrow+copy per str, one storage
# read per int, one repr call per float); the canonical-form emission and
# the SHA-256 run under one py.detach. No tors.aio twin: a fast one-shot
# call (see docs/async.md's family list).
#
# Bounds (generic ValueError, no bound values leaked): subclass hooks run to
# completion under the GIL and abort past the per-container bound; dict
# subclasses whose keys are all exact str/int/bool sort on the native fast
# paths (no interpreter sort, nothing against these bounds), while
# exotic-key subclass dicts (any float/big-int/mixed/NaN/subclass key)
# delegate to CPython's own
# list.sort and abort past 100k keys in one dict or 500k delegated pairs per
# call; exact+protocol nesting aborts past the untrusted-input ceiling
# (100k exact levels hash, 200k raises RecursionError). Protocol nesting past
# ~1000 (sys.getrecursionlimit()) raises RecursionError even where stdlib
# 3.12+ succeeds (documented conservative divergence). Treat content_hash as
# trusted-input-only for subclass hooks, for depth beyond ~10-20k frames,
# and for breadth beyond ~200-500k visited objects.
def content_hash(
    obj: JSONValue,
) -> str: ...

# Domain-separated SHA-256 (RFC 6962-style: leaves hash 0x00‖chunk, internal
# nodes hash 0x01‖left‖right): not the crate's undifferentiated default,
# which is forgeable (CVE-2012-2459-class leaf/internal-node confusion).
# An EMPTY chunk list refuses (`ValueError` "root of no chunks"): the
# RFC 6962 empty-tree hash exists but is a statement about an ABSENT
# structure — a caller hashing nothing almost certainly means an upstream
# bug, so the error is the contract (the empty case is pinned in the
# battery).
def merkle_root(chunks: list[bytes]) -> str: ...
def merkle_diff(chunks_a: list[bytes], chunks_b: list[bytes]) -> list[int]: ...

# One-shot hashing, the request-signing/content-check primitives. Each
# algorithm computes its digest once and offers two output spellings:
# lowercase hex (the _hex names) and the raw digest bytes (the _digest
# names — the call sites that base64-encode a signature, chain a digest
# back in as a key, or slice a stable int off it). The whole digest (hex
# formatting included, on the hex spellings) runs under one
# py.detach. str input is its UTF-8 bytes (tors.sha256_hex(s) ==
# hashlib.sha256(s.encode("utf-8")).hexdigest(); hashlib itself refuses
# str — the convenience is deliberate). bytes input is exactly bytes
# (bytearray/memoryview raise TypeError, the bytes-in family's
# immutable-buffer doctrine — wrap first, bytes(buf), then hash); a lone
# surrogate raises UnicodeEncodeError
# at the argument boundary (the crate-wide str-in contract). Stateless
# one-shot only: no hash object, no streaming surface (tors is stateless
# by charter; for incremental feeding, hashlib's object API is the right
# tool and is not duplicated).
#
# SECURITY: md5 and sha1 — either spelling, _hex or _digest — are
# checksum/legacy-interop only (Content-MD5, S3 ETags, cache-busting,
# quick compares) — broken for security since the 2000s (md5 collisions
# since 2004, sha1's first practical collision 2017). Never use either
# for signatures, certificates, or passwords; the sha256/sha512/hmac
# names are the security side.
def md5_hex(data: str | bytes) -> str: ...
def sha1_hex(data: str | bytes) -> str: ...
def sha256_hex(data: str | bytes) -> str: ...
def sha512_hex(data: str | bytes) -> str: ...
def md5_digest(data: str | bytes) -> bytes: ...
def sha1_digest(data: str | bytes) -> bytes: ...
def sha256_digest(data: str | bytes) -> bytes: ...
def sha512_digest(data: str | bytes) -> bytes: ...

# HMAC-SHA-256, the request-signing primitive (webhook signatures, AWS
# SigV4-style HMAC chains, API auth): byte-identical to
# hmac.new(key, data, hashlib.sha256).hexdigest(), in the same two
# output spellings (hmac_sha256_digest returns the raw 32 bytes — the
# shape the base64-encoding webhook schemes want). Each argument carries
# the hashing family's str|bytes contract independently (a str key is
# its UTF-8 bytes, the spelling a webhook secret arrives in); any key
# length is legal, empty included (parity with stdlib hmac). The key is
# held in memory for the call and is not zeroized on return — the same
# posture as the stdlib hmac/hashlib spelling. A non-ASCII str argument
# pays the one-time O(input) UTF-8 materialization independently per
# argument; the measured HMAC wall cells use bytes key+data, equivalently
# the ASCII zero-copy lane. GIL model:
# both borrows under the GIL, the whole keyed digest (key derivation
# included) plus hex formatting under one detach.
def hmac_sha256_hex(key: str | bytes, data: str | bytes) -> str: ...
def hmac_sha256_digest(key: str | bytes, data: str | bytes) -> bytes: ...


# The UUIDv7 helper trio (RFC 9562 layout): the keyset-pagination /
# time-bucketed-query primitives over time-ordered IDs. uuid7_timestamp_ms
# returns the 48-bit big-endian unix-millisecond field (the leading six
# bytes; datetime.fromtimestamp(ms / 1000, UTC) is the ID's creation
# instant), ValueError naming the found version when the version nibble is
# not 7. uuid_version returns the version nibble (byte 6's high half,
# 0-15) for any UUID of any variant: the field itself, no variant check
# (the variant is byte 8's top two bits, a different field, out of scope).
# uuid_parse is canonical text -> the 16 raw bytes, strict: exactly 36
# characters, hyphens at 8/13/18/23, lowercase hex elsewhere, ValueError
# naming the problem and the accepted form otherwise (positions 0-based).
# The stdlib uuid.UUID also accepts braces, urn:uuid:, hyphen-less hex,
# and uppercase; tors deliberately does not (the validation-primitive
# contract, the same closed-set strictness as errors=/boundary=), pinned
# as deliberate divergences in tests/test_uuid.py.
#
# GIL note: the bytes spelling borrows the argument zero-copy and the bit
# extraction runs under py.detach; the str spelling validates and
# transcodes under the GIL (36 bytes, smaller than the call's own
# marshalling residue -- a detached parse would be overhead for its own
# sake) with the extraction detached after it, so the int-out pair keeps
# the crate's GIL-free-core contract uniform. uuid_parse is the trio's
# zero-detach member: its whole work is that 36-byte parse (no int-out
# tail exists to detach) and it runs GIL-held by design, ~70ns a call
# (re-measured after the uuid-crate adoption). Exactly
# bytes or str for the int-out pair (bytearray/memoryview: TypeError, the
# bytes-in surface's exactly-bytes contract); exactly str for uuid_parse.
def uuid7_timestamp_ms(value: bytes | str) -> int: ...
def uuid_version(value: bytes | str) -> int: ...
def uuid_parse(value: str) -> bytes: ...


# FastCDC 2020 content-defined chunking: (start, end) byte spans (not
# codepoints: a byte-level primitive, unlike word_bounds/sentence_bounds),
# partitioning data exactly. Empty input -> []; input shorter than min_size
# -> one chunk covering the whole input. Deterministic; a small edit only
# perturbs the chunks nearest it, not every boundary after it (the natural
# upstream of merkle_root/merkle_diff's list[bytes] for byte-level dedup).
# min_size/avg_size/max_size must satisfy fastcdc's own bounds (each even;
# min_size <= avg_size <= max_size) or ValueError is raised before any
# chunking runs. Defaults are the crate's own documented example values.
def chunk_cdc(
    data: bytes, *, min_size: int = 4096, avg_size: int = 16384, max_size: int = 65534
) -> list[tuple[int, int]]: ...

# GIL note: the context-window/RAG packing primitive: boundary-aware cuts
# under a character budget (truncate_to_bounds' own cut rule applied
# repeatedly; the whole-text bounds computed once, GIL-released). Offsets
# are Python str indices. overlap=0 (default): the original lossless-
# partition contract, unchanged; chunks are non-empty, contiguous,
# covering, each <= max_chars codepoints, joining them reproduces the
# input exactly; a grapheme-safe hard cut applies when a single
# word/sentence exceeds the budget. Every cut is grapheme-cluster-safe, so
# max_chars can be exceeded only in the pathological case of a single
# cluster (e.g. an oversized ZWJ emoji chain) wider than the remaining
# budget. overlap > 0: each chunk after the first starts `overlap`
# codepoints before the previous chunk's end, snapped to the nearest
# boundary (never mid-word/mid-sentence): the RAG-retrieval shape, at
# the cost of the lossless-join guarantee. overlap must be < max_chars
# (ValueError otherwise); a chunk shorter than the requested overlap
# silently degrades to zero overlap for that one transition rather than
# stall, and so does an overlap whose re-cut would land the next chunk
# strictly inside its predecessor (the same text twice): the transition
# falls back to the zero-overlap cut, so chunk ends always strictly
# advance. The return marshalling is O(chunks) 2-tuples of ints (the
# word_bounds list-shape class).
def chunk_text(
    text: str,
    max_chars: int,
    *,
    overlap: int = 0,
    boundary: Literal["word", "sentence"] = "word",
) -> list[tuple[int, int]]: ...

# The streaming twin of chunk_text: same sequence, same argument contract,
# the word_bounds_iter shape (whole scan under one detach at construction,
# one 2-tuple per __next__): avoids materializing a list for documents
# that chunk into the hundreds of thousands of pieces. Error precedence
# matches the list spelling exactly: a lone-surrogate str raises
# UnicodeEncodeError (the text conversion) before any other argument's
# conversion or validation error, list and iter alike.
def chunk_text_iter(
    text: str,
    max_chars: int,
    *,
    overlap: int = 0,
    boundary: Literal["word", "sentence"] = "word",
) -> Iterator[tuple[int, int]]: ...

# GIL note: word-count-windowed chunking: the semantic-chunking RAG
# shape, measured in real word tokens (word_bounds' segments filtered to
# non-whitespace-only ones: word_bounds itself gives an inter-word space
# run its own segment, so grouping raw segments would silently mean
# "words_per_chunk roughly halved" on ordinary prose) rather than a
# character budget. Each chunk spans `words_per_chunk` consecutive word
# tokens, its span the first token's start through the last's end (not
# through trailing whitespace after it, so unlike chunk_text this is not
# a covering partition: chunks are not necessarily contiguous); `overlap`
# words repeat at the start of the next chunk. The final chunk may hold
# fewer than words_per_chunk tokens when the total doesn't divide evenly.
# Empty text, or text with no word tokens at all, -> []. overlap must be
# < words_per_chunk (ValueError otherwise): the stride
# words_per_chunk - overlap is always >= 1 once validated, so forward
# progress needs no runtime fallback the way chunk_text's
# character-granularity overlap does. GIL-released whole pass; O(chunks)
# 2-tuple marshalling.
def chunk_by_words(
    text: str, words_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...

# The streaming twin of chunk_by_words: same shape as chunk_text_iter.
# Error precedence matches the list spelling exactly: a lone-surrogate str
# raises UnicodeEncodeError (the text conversion) before any other
# argument's conversion or validation error, list and iter alike.
def chunk_by_words_iter(
    text: str, words_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...

# chunk_by_words' sentence-count twin (sentence_bounds' UAX #29
# segmenter). Same contract, same argument validation, same empty-input
# answer.
def chunk_by_sentences(
    text: str, sentences_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...

# The streaming twin of chunk_by_sentences: same shape as chunk_text_iter.
# Error precedence matches the list spelling exactly: a lone-surrogate str
# raises UnicodeEncodeError (the text conversion) before any other
# argument's conversion or validation error, list and iter alike.
def chunk_by_sentences_iter(
    text: str, sentences_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...

# chunk_by_words/chunk_by_sentences' paragraph-count twin. A paragraph
# boundary is a run of 2+ consecutive newline characters (\r\n counts as
# one unit, matching normalize's own CR/CRLF folding): the "2+ newlines
# survive as the paragraph gap" convention normalize's own pipeline
# already uses (it collapses 3+ down to exactly 2, never below). This is
# a heuristic, not a Unicode Standard segmentation (there is no UAX for
# paragraphs): a single \n is ordinary content, not a break. Unlike the
# word/line twins, paragraphs have no content filter: a whitespace-only
# paragraph is emitted as a chunk (only fully-empty spans are dropped),
# so an overlapping pair of chunks can share blank content. Same
# contract, same argument validation, same empty-input answer as its
# siblings.
def chunk_by_paragraphs(
    text: str, paragraphs_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...

# The streaming twin of chunk_by_paragraphs: same shape as chunk_text_iter,
# and the same rationale as every other _iter spelling: the list shape's
# GIL-held marshalling cost is measured for segment-count-heavy outputs
# (word_bounds on 12 MiB of prose, 3.67M segments, holds the GIL for
# 328-344 ms, box-pace-dependent; ~0.72 of the call's wall is the
# constant, just marshalling the list), and a paragraph-heavy corpus (a
# multi-MiB article dump or report batch) is in that piece-count class,
# chunking into hundreds of thousands of pieces. Error precedence matches
# the list spelling exactly: a lone-surrogate str raises UnicodeEncodeError
# (the text conversion) before any other argument's conversion or
# validation error, list and iter alike.
def chunk_by_paragraphs_iter(
    text: str, paragraphs_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...

# chunk_by_words/chunk_by_sentences/chunk_by_paragraphs' line-count twin.
# A line break is a \n, a lone \r, or a \r\n pair counted as one unit
# (the same CR/CRLF folding convention; str.splitlines' exotic separators
# \v, \f, NEL, LS, PS are not breaks here). A line counts as a line
# only when it carries at least one non-whitespace codepoint, the same
# real-token discipline chunk_by_words applies to word segments: blank
# lines neither count toward lines_per_chunk nor split a chunk's interior
# (they ride along inside a chunk's span exactly as inter-word whitespace
# rides along in chunk_by_words), so lines_per_chunk=200 means 200
# content lines. "Non-whitespace" is definitional: the Unicode
# White_Space property (char::is_whitespace), under which an NBSP-only
# line is blank and U+001C-U+001F (FS/GS/RS/US) count as line content,
# diverging from Python's str.isspace() (which treats those four as
# whitespace) and from str.splitlines (which even breaks on them; tors
# does not). Spans run first included line's start through last
# included line's end, not through the trailing break (not a covering
# partition); a trailing break at end of text yields no trailing empty
# line. Final chunk may hold fewer lines; empty or no-content-line text
# -> []. overlap repeats whole lines at the next chunk's start and must
# be < lines_per_chunk (ValueError otherwise; the stride
# lines_per_chunk - overlap is always >= 1 once validated). GIL-released
# whole pass; O(chunks) 2-tuple marshalling.
def chunk_by_lines(
    text: str, lines_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...

# The streaming twin of chunk_by_lines: same shape as chunk_text_iter,
# and the same rationale as every other _iter spelling: the list shape's
# GIL-held marshalling cost is measured for segment-count-heavy outputs
# (word_bounds on 12 MiB of prose, 3.67M segments, holds the GIL for
# 328-344 ms, box-pace-dependent; ~0.72 of the call's wall is the
# constant, just marshalling the list), and a line-oriented corpus (a
# multi-MiB log or transcript) is in that piece-count class, chunking into
# hundreds of thousands of pieces. Error precedence matches the
# list spelling exactly: a lone-surrogate str raises UnicodeEncodeError
# (the text conversion) before any other argument's conversion or
# validation error, list and iter alike.
def chunk_by_lines_iter(
    text: str, lines_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...

# Priority-ordered fallback chunking (LangChain's RecursiveCharacterTextSplitter
# pattern): cut at the coarsest level that fits max_chars, falling back to
# finer levels only when a coarser one has no in-budget cut. separators=None
# uses tors's own accurate hierarchy (heading -> paragraph -> sentence ->
# word -> a grapheme-safe raw cut, always the final unconditional fallback).
# The heading level (#63) bounds chunks at ATX markdown heading lines: a
# section's content never merges across a heading of higher rank, each cut
# lands before the heading (the heading rides with the section that follows
# it, the pre-heading newline run dropped), and a whole-document budget over
# a heading-bearing document comes back as its sections, not one giant chunk.
# The level is ATX-only (verified against the documents engines' emitters,
# which all emit ATX: setext underlines are excluded), tracks fenced code
# blocks (a '# comment' inside a fence is code, not a heading), and is
# gated: text with no '#' byte never realizes it, so heading-free input
# chunks exactly as the pre-#63 hierarchy did. Precedence: budget > heading
# > paragraph > sentence > word (a section wider than max_chars still
# splits at finer levels).
# separators=[...] is a caller-supplied sequence of literal strings (not
# regex), any Sequence (list or tuple) of literals and None entries; str,
# dict, set, and generators raise TypeError at extraction, coarsest first,
# e.g. ["\n## ", "\n\n", ". ", " "] for markdown-header-aware
# chunking: replaces the default hierarchy, but the raw cut is still always
# appended. A None entry in an otherwise-literal sequence splices the default
# hierarchy's accurate levels in at that position: ["\n", None] is
# line -> heading -> paragraph -> sentence -> word -> raw cut, the
# line-oriented-text shape (a chat thread, one message per line, never split
# mid-line) whose oversized-line fallback is the real UAX #29 segmenter
# rather than the ". "/" " literal guesses an all-literal list pins it to;
# [None] is identical to separators=None. The reserved literal "heading" is
# the heading level itself (the explicit opt-in for custom hierarchies), not
# a split on the word. Not a lossless partition (unlike
# chunk_text): the separator itself is dropped between chunks, the same
# chunk_by_paragraphs convention. Since #103 that includes the final
# window: a window that opens on a separator match is skipped even at the
# final-chunk exit, so a trailing separator run survives only as a suffix
# of a content-bearing chunk and an all-separator document chunks to zero
# chunks. overlap snaps to the nearest grapheme boundary (not necessarily
# a semantic one, a documented simplification of chunk_text_overlapping's
# single-level snap); overlap_boundary="word" additionally snaps that
# candidate back to the nearest UAX #29 word boundary (the word-bounds
# level, realized lazily and shared with the windows; falls back to the
# grapheme candidate when the word level has no boundary in the
# snap-back range, and is a no-op at overlap=0); the decline-the-snap
# lookahead runs after the word snap, unchanged. Unknown overlap_boundary
# values raise ValueError. max_chars < 1 or overlap < 0
# raise ValueError; overlap >= max_chars raises ValueError. Empty text
# returns []; an empty separators sequence is legal and skips straight to the
# raw-cut fallback.
def chunk_hierarchical(
    text: str,
    max_chars: int,
    separators: Sequence[str | None] | None = None,
    *,
    overlap: int = 0,
    overlap_boundary: Literal["grapheme", "word"] = "grapheme",
) -> list[tuple[int, int]]: ...

# GIL note: the whole tokenize (UAX #29 words) + FNV-1a hash + 64-bit vote
# pass runs GIL-released; a single int return (no marshalling class).
# Deterministic across processes (FNV-1a, not the per-process-seeded
# DefaultHasher). Near-duplicate detection: Hamming distance
# (a ^ b).bit_count(). Measured anchors: single-word edits at document
# scale move it <= 4 bits, at sentence scale <= 14; unrelated sentences
# sit >= 23 bits apart. Thresholds are corpus-dependent: calibrate per
# deployment. Order-invariant (a bag-of-words vote).
def simhash64(text: str) -> int: ...

# GIL note: simhash64's classes exactly: the whole tokenize+hash+vote
# pass GIL-released, a single int return. The 128-bit spelling: twice the
# bit positions; unrelated distances roughly double while the near-dup
# bands grow sublinearly (measured: document 3 bits vs 64-bit 4, sentence
# 20 vs 14), so the separation between near-dup and unrelated widens,
# the shape for corpora whose 64-bit bands overlap. Same contract
# otherwise (deterministic, order-invariant, empty → 0, (a ^ b).bit_count()
# for the distance; thresholds corpus-dependent).
def simhash128(text: str) -> int: ...

# The recall-side near-dup complement to the simhash family (simhash is
# the precision side; minhash recalls similar shingle sets at corpus
# scale, the quantity an LSH-banding table -- caller state, tors stays
# stateless -- buckets on). num_perm min-hashes over shingle_size-token
# word shingles (the tf_idf/bm25 UAX #29 token stream, lowercased, each
# window hashed under the injective length-prefixed framing); each element is min over shingles of
# (a_i * x + b_i) mod (2^61 - 1), x the shingle's XXH64 (frozen-spec,
# deterministic across processes/machines/versions), (a_i, b_i) derived
# from seed by a pinned SplitMix64 stream. The signature is
# deterministic given the segmentation tables: identical across
# processes, machines, and platforms within one tors version, but a
# release that bumps unicode-segmentation can change signatures (re-
# fingerprinting every affected document) -- re-baseline persisted
# signatures/LSH tables on upgrade. The agreement fraction of two
# signatures estimates their shingle-set Jaccard similarity with standard
# error sqrt(J(1-J)/num_perm) (~0.044 at 128). Empty text / whitespace
# only / fewer tokens than shingle_size: every element 2**64 - 1 (the
# empty-set sentinel, outside the affine range). num_perm in [1, 1024]
# and shingle_size >= 1, else ValueError (an in-range value out of
# bounds); an int outside the i64 range the binding extracts raises
# OverflowError instead (pyo3, the truncate_to_bounds-identical
# pattern); seed is any int reduced mod
# 2**64 (two's complement for negatives), and all three ride __index__
# (numpy integers work; bool is rejected with TypeError in every
# position, including as an __index__ result; a raising __index__
# propagates). GIL: borrow + validation
# under the GIL, the whole pass under one detach, then the
# num_perm-element int list. No aio twin: a fast one-shot call.
def minhash_signature(
    text: str,
    *,
    num_perm: int = 128,
    shingle_size: int = 3,
    seed: int = 0,
) -> list[int]: ...

# Stateless: no vocabulary/vectorizer object persists between calls.
# Tokenization: UAX #29 word segments, non-whitespace only, lowercased
# (Unicode-correct str.lower, not ASCII-only). TF is the raw term count
# per document (not length-normalized). IDF is the scikit-learn-style
# smoothed formula ln((1 + N) / (1 + df)) + 1 (N = corpus size, df = document
# frequency): not the textbook ln(N/df), which gives a
# term in every document an IDF of exactly 0; smoothing keeps that case
# strictly positive while still favoring rarer terms. score(t, d) =
# tf(t, d) * idf(t). Output is sparse: one (term, score) list per
# document, alphabetically sorted, holding only that document's own
# terms, never a vocabulary-size-by-corpus-size dense structure. Empty
# corpus -> []; an empty-string document -> [] at its position (the
# output always has exactly len(corpus) entries). strip_accents=True
# NFD-decomposes each token and drops combining marks ("café" -> "cafe")
# before scoring, unconditionally (not the short-circuit-on-already-
# decomposed-input bug scikit-learn's own strip_accents_unicode has).
# stemmer names a Snowball algorithm applied after lowercasing/accent-
# folding; an unrecognized name raises ValueError naming every valid
# choice. lemma_dict is a caller-supplied word -> lemma map applied last
# (after stemming, if any): tors does not bundle a lemma dictionary (that
# needs a per-language dataset or POS model, out of scope); it only
# applies one you supply, the same shape replace_many takes a
# caller-supplied replacement map instead of a bundled one. All three
# default off, reproducing the original lowercase-only tokenization
# exactly. No stop-word removal or n-grams.
#
# lemma_dict accepts a raw dict[str, str] (materialized fresh every call)
# or a CompiledLemmaDict (built once, reused across calls at O(1) cost per
# call thereafter): see CompiledLemmaDict below for why this exists: a
# realistic 20,000-entry lemma table costs roughly 1.5ms to materialize,
# and a raw dict pays that on every call.
class CompiledLemmaDict:
    """A pre-built `lemma_dict` mapping: the `re.compile()` answer to
    tf_idf/bm25_rank/apply_pipeline's `lemma_dict` cost: build once,
    reuse across many calls instead of re-materializing the same Python
    dict into a Rust map on every call. Immutable once constructed."""

    def __init__(self, mapping: dict[str, str]) -> None: ...
    def __len__(self) -> int: ...

def tf_idf(
    corpus: list[str],
    *,
    strip_accents: bool = False,
    stemmer: StemmerLanguage | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
) -> list[list[tuple[str, float]]]: ...

# A reranking primitive, not a search index: recomputes corpus statistics
# from scratch every call, the right shape for scoring a small,
# already-retrieved candidate set (tens to a few hundred documents) against
# one query: the wrong shape for a large corpus queried repeatedly (reach
# for a real search engine, e.g. tantivy, for that; tors does not build
# persistent index objects). Okapi bm25, the always-non-negative "+1" IDF
# variant (ln((N - df + 0.5) / (df + 0.5) + 1), not the classic form, which
# goes negative for a term in over half the corpus). k1 (>= 0, default 1.5)
# tunes term-frequency saturation; b (in [0, 1], default 0.75) tunes length
# normalization: both are Lucene/Elasticsearch's own defaults. Returns
# (index, score) for every document (no top-k cutoff: slice/sort
# yourself), sorted by score descending, ties broken by ascending original
# index. Tokenization: the same "real word token, lowercased" convention
# tf_idf uses. Empty corpus -> []; empty query -> every document scores
# 0.0 (not an error). No claim about retrieval/relevance quality for any
# particular corpus or query: a correct implementation of a well-known
# ranking formula, not a model-quality promise.
#
# strip_accents/stemmer/lemma_dict: tf_idf's exact same opt-in knobs,
# applied identically to query and every corpus document (required for the
# scores to mean anything, not just a style choice), all default off.
def bm25_rank(
    query: str,
    corpus: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    strip_accents: bool = False,
    stemmer: StemmerLanguage | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
) -> list[tuple[int, float]]: ...

# A stateless, general-purpose batch text preprocessor: every requested
# step fused into one GIL-released pass over the whole texts list. Pure
# function composition, not a re.compile()-style compiled-
# pipeline object (a stateful handle was considered and explicitly ruled
# out: every call re-describes and re-applies its steps fresh). Order:
# nfd -> lowercase -> strip_accents -> (stemmer / lemma_dict) ->
# collapse_whitespace, each skipped when its flag is off/None. nfd/
# lowercase/strip_accents are codepoint-level (run over the whole text);
# stemmer/lemma_dict are word-level (walk UAX #29 word-boundary segments,
# transforming only real-word segments, preserving every other segment
# (punctuation, whitespace) verbatim, so the output stays readable prose).
# collapse_whitespace runs last, reducing every run of Python-whitespace-
# equivalent codepoints to one ASCII space (not tors.normalize's full
# pipeline: no CRLF folding, no blank-line collapsing, no strip).
#
# All six steps default off: apply_pipeline(texts) with nothing else is a
# true identity: the original texts list object comes back unchanged, not
# just content-equal output, the same zero-allocation contract normalize/
# quote/replace_many already give for their own no-op case. Empty texts ->
# []. A non-list argument or non-str element raises TypeError; an invalid
# stemmer name raises ValueError naming every valid choice; a non-dict
# lemma_dict, or one with a non-str key/value, raises TypeError.
#
# Relationship to tf_idf/bm25_rank: those two already fuse the same
# strip_accents/stemmer/lemma_dict knobs into their own tokenization:
# calling apply_pipeline first and then tf_idf/bm25_rank on the result
# tokenizes twice for no benefit. Reach for their own knobs when they're
# the only consumer; reach for apply_pipeline to preprocess text feeding
# anything else (chunk_text, find_patterns, your own logic).
def apply_pipeline(
    texts: list[str],
    *,
    nfd: bool = False,
    lowercase: bool = False,
    strip_accents: bool = False,
    stemmer: StemmerLanguage | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
    collapse_whitespace: bool = False,
) -> list[str]: ...

# Classic phonetic-code algorithms (rphonetic, an Apache Commons Codec
# port): English/Latin-script-oriented heuristics, not general Unicode
# phonetics. Input is pre-filtered to ASCII letters before encoding
# (accents/non-Latin characters dropped, not encoded): this also works
# around a real, verified panic in rphonetic 4.0.0 on ordinary accented
# names ("José", "café", "Björk" all crash the raw crate; see
# src/phonetic_impl.rs). Empty input, or input with no ASCII letters,
# -> "". Deterministic; typically paired with levenshtein/jaro_winkler
# for name-matching/dedup pipelines (phonetic grouping first or
# alongside edit-distance scoring, not instead of it).
def soundex(text: str) -> str: ...
def metaphone(text: str) -> str: ...

# The full dual-key form of metaphone: Double Metaphone's alternate code
# carries the second plausible (typically non-Anglicized) pronunciation,
# so a name match on either key counts; two equal elements when there is
# only one pronunciation.
def double_metaphone(text: str) -> tuple[str, str]: ...

# NYSIIS (1970), strict commons-codec variant (codes capped at 6
# characters): a Soundex successor for name matching.
def nysiis(text: str) -> str: ...

# Daitch-Mokotoff Soundex (1985), the Jewish-genealogy standard for
# Central/Eastern European surnames. A list because the rule table
# branches on ambiguous transliterations: one name can encode to
# several 6-digit codes, and two names match if their lists intersect.
# Unlike soundex/metaphone, no-letters input is ["000000"] (each code
# padded to 6 digits), not "".
def daitch_mokotoff(text: str) -> list[str]: ...

# A Soundex variant with a finer-grained letter-to-digit mapping than
# classic soundex (more consonant classes distinguished, an uncapped
# code rather than soundex's fixed letter-plus-3-digit shape): a
# distinct algorithm, not a formatting variant. Same
# ASCII-letters-only pre-filter and upstream-panic-avoidance note as
# soundex.
def refined_soundex(text: str) -> str: ...


# The random-generation family. SECURITY CONTRACT, the same paragraph on
# every seeded surface: the default (no seed) draws fresh bytes from the
# operating system's CSPRNG on every call — no process or thread RNG state,
# so it is fork-safe, matching secrets' own per-call semantics — safe for
# keys, tokens, and secrets. seed= switches to a deterministic ChaCha20
# stream: the output becomes a pure function of (seed, arguments), fully
# predictable from the seed — a reproducible-test/fixture tool, NEVER safe
# for secrets, keys, or tokens (any adversary who learns the seed can
# reproduce the stream); the unseeded spelling is the secrets-safe one.
# The seed is any int-like — an int instance (bools, IntEnums) or any
# __index__ object, the same convention length accepts — reduced mod 2**64
# (two's complement for negatives).
# Length-first, uniformly: the four token spellings take the OUTPUT length
# ("I want a base62 id X characters long" is the whole call), and all four
# are one char-sampling engine — random_hex/random_b62/random_b64url are
# exactly random_string over their fixed alphabets (Lemire, no modulo
# bias). All six are one GIL-released pass (draw + sampling/formatting
# under py.detach).
# The seed contract shared by every seeded spelling below: any int-like —
# an int instance (bools, IntEnums) or any __index__ object — reduced mod
# 2**64, or None for the unseeded OS-entropy spelling. A non-int-like seed
# raises TypeError naming the int-like (__index__) convention.
_SeedLike = int | SupportsIndex
def random_string(
    length: int, alphabet: str, *, seed: _SeedLike | None = None
) -> str: ...

# length lowercase hex characters ("0123456789abcdef"), uniform per
# character: exactly random_string(length, HEX_CHARS). Odd lengths are
# legal (a 31-char hex id is a real shape); even lengths are what
# digest-shaped keys want (every 2 chars exactly one byte).
# secrets.token_hex(n) is the same uniform distribution as random_hex(2*n)
# — different draws.
def random_hex(length: int, *, seed: _SeedLike | None = None) -> str: ...

# Exactly random_string(length, BASE62_CHARS): the [0-9A-Za-z] id spelling.
def random_b62(length: int, *, seed: _SeedLike | None = None) -> str: ...

# length characters uniform over the 64-char RFC 4648 §5 urlsafe alphabet
# (A-Za-z0-9-_, never + or /), every position unconstrained: the
# opaque-TOKEN contract, NOT a base64 encoding of N random bytes (an
# encoding's final char is constrained; '=' never appears; there is no
# padded= parameter — padding is an encoding concept, not a token
# concept). Callers wanting encodable random material: random_hex of even
# length (byte-exact via hex).
def random_b64url(length: int, *, seed: _SeedLike | None = None) -> str: ...

# An RFC 4122 v4 UUID string (36 chars, lowercase, hyphens at 8/13/18/23):
# 122 random bits, the uuid.uuid4() spelling. Deterministic under seed=
# (predictable — the security contract above).
def uuid4(*, seed: _SeedLike | None = None) -> str: ...

# An RFC 9562 v7 UUID string: 48-bit Unix-millisecond timestamp + 74 random
# bits. No seed parameter: the timestamp is external state (a seeded uuid7
# would still vary with the clock; the deterministic tool is uuid4(seed=...)).
# Probabilistically unique, NOT counter-monotonic: same-millisecond calls
# order by their random bits and a backwards clock step flows into the
# timestamp (uuid_utils' strict monotonicity is a different product
# promise). The caller-visible contract is the timestamp: the canonical
# string's first two dash-free groups, int(u[:8] + u[9:13], 16), are the
# call's Unix epoch milliseconds.
def uuid7() -> str: ...

# The uuid4 buffer spelling: the same 16 raw bytes uuid4 formats (version
# and variant nibbles set, NO canonical hyphenation) — for consumers who
# re-wrap the str back into bytes anyway (UUID(bytes=...), .hex() slicing):
# one native draw and the field layout, no format-then-reparse roundtrip.
# Same seed contract as uuid4 (any int-like, reduced mod 2**64) and the
# same security paragraph above (seed= is predictable, never for secrets).
# Fixed 16 bytes: no length argument, so the token spellings' memory-bound
# class does not exist here.
def uuid4_bytes(*, seed: _SeedLike | None = None) -> bytes: ...

# The uuid7 buffer spelling: the same 16 raw bytes uuid7 formats —
# 48-bit Unix-millisecond timestamp + 74 random bits, NO canonical
# hyphenation. The consumer slice shapes: the first 6 bytes big-endian are
# the timestamp (int.from_bytes(b[:6], "big")), and .hex()[:12] is its hex
# spelling. No seed parameter (uuid7's own rationale: the timestamp is
# external state); probabilistically unique, NOT counter-monotonic.
def uuid7_bytes() -> bytes: ...


# GIL note: one GIL-held walk of the items sequence (the standard str-in
# borrow class, O(items) handles, over any Sequence), then the set builds
# and the whole batch scan under one GIL-released pass, then a single int
# return: no marshalling class at all (the count_matches shape). The
# answer is the INDEX of the first item not built entirely from the two
# sets, -1 when all pass (empty batch answers -1 even when every item
# would offend); the scan short-circuits at the first offender,
# but the argument walk validates the whole sequence up front (a bad
# entry anywhere raises at the boundary, past a first offender or not).
# The walk is bounded (content_hash's protocol-walk cap, the same DoS
# backstop): a sequence that yields past the ceiling aborts with a
# generic ValueError instead of holding the GIL for an unbounded
# __iter__.
# Batch-only by design: per-item validation is under a detach round trip,
# so per-item calls would be slower than the regexes this replaces; the
# batch form — one detach, one pass — is the only shape that wins.
# Per-scalar engine with no normalization (normalize with tors.normalize
# first when NFC/NFD must agree, which still does not fold confusables:
# allow-list exactly the codepoints you mean); huge set spellings belong
# in module constants (define once, reuse the same string: the str-in
# borrow stays warm — the per-call set build itself is unchanged, its
# 10k-spelling sort unmeasured beyond the ASCII band). Both-bad
# precedence is extraction order: first beats rest, set-argument
# errors beat the items walk.
def first_invalid_charset(
    items: Sequence[str], *, first: str | None = None, rest: str
) -> int: ...

# The offender-detail spelling of the same scan: (item_index,
# char_position, offending_char) for the first offending item's FIRST
# offending position — the detail a rejection message needs (the
# consumer's per-character messages name the losing character and
# position) — None when every item passes. char_position is a CODEPOINT
# index within the item (the family's data model), never a UTF-8 byte
# offset, and may land inside a grapheme cluster (flag-partial (0, 1,
# "🇷") under rest="🇫"): do not slice at that position, build messages
# from (item, char); offending_char is that codepoint as a 1-char str. The empty
# item reports (i, 0, ""): no offending character to name, the char
# field empty exactly when the item is. Same engine, same walk, same
# one-detach batch pass and the same argument contract (byte-identical
# refusals) as the int spelling; the int answer is the tuple's item
# index, -1 exactly when the tuple is None. A == -1 test does not
# transfer from the int spelling (a tuple never equals -1): spell the
# check is None / is not None. See docs/api.md's
# "Building rejection messages".
def first_invalid_offender(
    items: Sequence[str], *, first: str | None = None, rest: str
) -> tuple[int, int, str] | None: ...

# Pinned common alphabets for first_invalid_charset: module constants, not
# functions (no signature to diff). The stub carries their type only —
# never their content: the live module is the single spelling of a
# 62-character alphabet, and the byte-exact contract pins live in
# tests/test_first_invalid_charset.py. See docs/api.md's "Common
# alphabets" for what ships, what deliberately does not (padded base64,
# UUID, digits), and why.
CHARSET_B62: str
CHARSET_B64URL: str
CHARSET_HEX_LOWER: str
CHARSET_HEX_MIXED: str
CHARSET_HEX_UPPER: str
