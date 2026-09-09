# API reference

Every function releases the GIL for its whole native pass (`py.detach`); the
GIL-held residue of a call is only pyo3's argument borrow and the return
marshalling. Signatures below are the typed surface of
`python/tors/__init__.pyi`, pinned to the live functions by
`tests/test_pyi_drift.py`.

## `tors.normalize`

```python
def normalize(text: str) -> str: ...
```

Applies, in order:

1. Unicode NFC normalization.
2. `\r\n` and `\r` folded to `\n`.
3. Trailing whitespace (spaces/tabs) before a newline is dropped.
4. Runs of 3 or more consecutive newlines collapse to exactly 2.
5. A final strip, using Python's exact whitespace set (including the `0x1c`-`0x1f`
   separator control characters, which `str.strip()` treats as whitespace but Rust's
   `char::is_whitespace()` does not).

This is a common preprocessing pipeline for text extracted from PDFs, OCR, and other messy
sources.

### GIL behavior

The entire transform runs inside `py.detach` (PyO3's GIL-release call): the GIL is free
for the rest of your program for the whole call, not just part of it. This is the reason
`tors` exists: Python's `re` module and `str` methods never release the GIL regardless of
input size, so the equivalent pure-Python pipeline holds the GIL for its whole duration no
matter what thread it runs on.

### Identity return

When the complete pipeline is a no-op on the input, i.e. already NFC (Unicode
quick-check Yes), no CR, no spaces/tabs before a newline, no 3+ newline runs, and no
whitespace at either end, `normalize` returns the ORIGINAL input object:

```python
tors.normalize(s) is s  # True whenever the pipeline changes nothing
```

Zero allocation, zero copy, zero marshalling (the same idiom CPython's
`unicodedata.normalize` fast path uses). An output-equals-input comparison after any
full pass extends the guarantee beyond provably-clean input: `tors.normalize(s) is s`
holds exactly whenever `tors.normalize(s) == s`. The same contract applies to
`tors.nfc` / `nfd` / `nfkc` / `nfkd`, `tors.html_unescape`, and the string element of
`tors.finalize` / `tors.finalize_utf8`.

### Example

```python
import tors

tors.normalize("line one  \n\n\n\nline two\r\n")
# "line one\n\nline two"
```

**Async**: `await tors.aio.normalize(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.finalize`

```python
def finalize(text: str) -> tuple[str, str]: ...
```

`(normalize(text), sha256)` in the same single GIL-released pass, where `sha256` is the
lowercase-hex SHA-256 of the normalized text's UTF-8 bytes: byte-identical to
`hashlib.sha256(normalized.encode("utf-8")).hexdigest()`. It exists for pipelines that
normalize then content-hash (dedupe gates), collapsing two whole-text passes into one.
On the identity path the digest is computed straight from the borrowed input buffer and
the string element is the ORIGINAL object (`tors.finalize(s)[0] is s` when the pipeline
changes nothing). A `str` holding lone surrogates is refused at the argument boundary
with `UnicodeEncodeError` ("surrogates not allowed"): pyo3's `&str` extraction
behavior, the same boundary every str-in function here documents.

```python
text, digest = tors.finalize("line one  \n\n\n\nline two\r\n")
# ("line one\n\nline two", "e986ba08...f7b942a")
```

**Async**: `await tors.aio.finalize(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.strip_controls`

```python
def strip_controls(text: str) -> str: ...
```

Replace every maximal run of C0 controls (`U+0000`–`U+001F`, tabs and newlines
included) and DEL (`U+007F`) with a single ASCII space, one GIL-released native
pass: byte-identical to `re.compile(r"[\x00-\x1f\x7f]+").sub(" ", text)`. The
scrub model-authored display text needs before it is stored or served (a chatty
or injected model cannot plant terminal control sequences in a row the UI
renders).

Two scope cuts, both deliberate. C1 controls (`U+0080`–`U+009F`) pass through
untouched: the adopted call-site regexes do not cover them either, so covering
them here would silently change adopted behavior — C1 scrubbing is a follow-up
with its own contract, not a silent extension of this one. And `\t`, `\n`,
`\r` ARE scrubbed (they are C0): do not reach for this on multi-line prose you
want to keep line-shaped. No edge strip either: a control run at either end
becomes an edge space for the caller to `.strip()`.

`tors.strip_controls(s) is s` exactly when `s` holds no C0/DEL character.

```python
tors.strip_controls("score: 4\x00\x01great\x7f")
# "score: 4 great "
```

**Async**: `await tors.aio.strip_controls(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.nfc` / `tors.nfd` / `tors.nfkc` / `tors.nfkd`

```python
def nfc(text: str) -> str: ...
def nfd(text: str) -> str: ...
def nfkc(text: str) -> str: ...
def nfkd(text: str) -> str: ...
```

`unicodedata.normalize(form, text)` for each form, in one GIL-released pass each: the
standalone normalization forms without normalize's folding/collapsing/strip
stages. Compatibility decompositions fire under the K-forms and only under them
(`"\ufb01"` stays `"\ufb01"` for nfc/nfd, becomes `"fi"` for nfkc/nfkd). Same
identity-return contract as `normalize` (`tors.nfc(s) is s` whenever `tors.nfc(s) ==
s`). tors ships its own Unicode tables (Unicode 16.0.0, the same UCD CPython 3.14
ships), so the same input produces the same output on every supported Python: on older
interpreters the only divergences from that interpreter's own `unicodedata` are on
codepoints it leaves unassigned; no ASSIGNED codepoint diverges in any form (pinned
exhaustively per interpreter by `tests/test_parity.py`).

## `tors.html_unescape`

```python
def html_unescape(text: str) -> str: ...
```

`html.unescape(text)` over the FULL HTML5 named-entity table (all 2231 entries of
`html.entities.html5`, with- and without-semicolon spellings) plus CPython's exact
numeric-reference classification (the Windows-1252 remap, the 126
invalid-codepoints-to-empty quirk, out-of-range to U+FFFD, the greedy-decimal
`&#10FFFF;` → `"\nFFFF;"` behavior, the longest-matching-prefix fallback with verbatim
remainder). On text the unescape leaves unchanged it returns the input object itself
(CPython's own no-`&` fast-path idiom).

**Raises `ValueError` on CPython 3.11+** when a DECIMAL numeric reference's digit run
exceeds `sys.get_int_max_str_digits()` (default 4300): the stdlib's own integer string
conversion limit, with its exact message (`"Exceeds the limit (4300 digits) for integer
string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase
the limit"`). Leading zeros count toward the run; the raise precedes every
classification; the limit is read from the running interpreter once per call, so
`sys.set_int_max_str_digits()` changes are honored on the next call. HEX references are
exempt (base 16 is a power of two: the limit applies only to non-power-of-two bases).
CPython 3.10 has no limit at all, and tors matches it there: nothing raises.

```python
tors.html_unescape("Tom &amp; Jerry &#233;")
# "Tom & Jerry é"
```

## `tors.grapheme_count`

```python
def grapheme_count(text: str) -> int: ...
```

The number of UAX #29 EXTENDED grapheme clusters (ZWJ emoji sequences are one cluster,
combining-mark chains join their base, CRLF is one cluster, regional-indicator pairs
join). The stdlib has no segmenter at all: the gap this exists to fill. The return is
a single `int`: no list marshalling class at all, the cleanest GIL cell in the suite.
Pinned by a hand-derived UAX #29 table plus structural properties
(`tests/test_segmentation.py`).

```python
tors.grapheme_count("a\r\n\U0001f469\u200d\U0001f52c")  # 3
```

## `tors.word_bounds`

```python
def word_bounds(text: str) -> list[tuple[int, int]]: ...
```

The UAX #29 word-boundary segments as `(start, end)` pairs in Python `str` index
(codepoint) units: `text[start:end]` IS the segment; bounds cover `[0, len(text))` and
joining the slices reproduces the input. OFFSETS, never string lists (marshalling
thousands of small `PyString`s under the GIL would eat the win). One measured caveat,
pinned openly: the return marshalling constructs one 2-tuple of ints per segment under
the GIL, O(number-of-segments), a measured 428–567 ms hold at 12 MiB of prose
(3.67 M segments). Fine at document scale; for whole-file sizes use the iterator
below.

```python
tors.word_bounds("Hello, world!")
# [(0, 5), (5, 6), (6, 7), (7, 12), (12, 13)]
```

## `tors.word_bounds_iter`

```python
def word_bounds_iter(text: str) -> Iterator[tuple[int, int]]: ...
```

The streaming spelling of `word_bounds`: the same `(start, end)` sequence (pinned to
sequence-parity with the list API over every tricky row and hypothesis text), yielded
lazily. The segmentation runs under one GIL-released pass when the iterator is
constructed, and each `__next__` holds the GIL only to construct one tuple (µs-scale):
worst heartbeat gap 15.4 ms at 12 MiB, against the list shape's structurally
unattainable 428–567 ms band; and the full drain is also ~2.1× FASTER in wall time
than the list API (347 ms vs 724 ms at 12 MiB, measured). `__length_hint__` reports the
remaining bound count and tracks partial consumption. The list API stays the right
shape for small inputs and one-shot batch work.

## `tors.word_count`

```python
def word_count(text: str) -> int: ...
```

The `grapheme_count` precedent for the word segmenter: a single `int` return (no
marshalling class at all), the whole scan GIL-released, `O(1)` memory where
`len(word_bounds(text))` materializes the full tuple list. `word_count(t) ==
len(word_bounds(t))`, pinned: the answer for a caller that only needs the count, not
the bounds themselves.

```python
tors.word_count("Hello, world!")
# 5
```

## `tors.decode_utf8`

```python
def decode_utf8(raw: bytes, *, errors: Literal["strict", "replace"] = "strict") -> str: ...
```

`raw.decode("utf-8", errors=...)` byte-exact over arbitrary bytes, including every
ill-formed class. Strict raises the very `UnicodeDecodeError` CPython's own decoder
raises (same type, `start`/`end`/`reason`, `object`, and message), and replace emits
CPython's exact U+FFFD placements (maximal-subpart substitution). An `errors` value
outside `{"strict", "replace"}` raises `ValueError` (the `unicodedata.normalize`
closed-set convention). The argument must be exactly `bytes` (`bytearray`/`memoryview`
raise `TypeError`): the pass reads a zero-copy borrow of the immutable buffer with the
GIL released, and a writable buffer would be a data race, not a semantic difference.

**Async**: `await tors.aio.decode_utf8(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.finalize_utf8`

```python
def finalize_utf8(raw: bytes, *, errors: Literal["strict", "replace"] = "strict") -> tuple[str, str]: ...
```

Decode + normalize + hash in one GIL-released call:
`(normalize(raw.decode("utf-8", errors=...)), sha256-of-the-result)`: the shape an
extraction pipeline wants for its text reads. Strict (default) raises the stdlib's own
`UnicodeDecodeError` on invalid bytes; `errors="replace"` flows the U+FFFD
substitutions through the pipeline. Same bytes-in GIL model as `decode_utf8`: no
argument-materialization class at all.

**Async**: `await tors.aio.finalize_utf8(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.b64_encode_bytes`

```python
def b64_encode_bytes(raw: bytes) -> str: ...
```

`base64.b64encode(raw).decode("ascii")` (RFC 4648 standard alphabet, padded), for
content-addressing paths that today hold the GIL for the whole encode. The argument
contract matches the bytes-in surface (exactly `bytes`).

**Async**: `await tors.aio.b64_encode_bytes(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.b64_decode`

```python
def b64_decode(s: str, *, validate: bool = True) -> bytes: ...
```

`base64.b64decode(s, validate=...)` over ASCII strings: same decoded bytes, same
raised exception (the REAL `binascii.Error`, so `except binascii.Error` and
`except ValueError` both keep working), same messages, including all five strict-mode
messages and the lenient mode's discard-non-alphabet rules (`'Zm9v=Zg=='` → `b'foof'`).
The core is a line-for-line port of the post-gh-145264 `binascii_a2b_base64_impl`
(CPython's 3.13/3.14 maintenance branches): in lenient mode, excess padding is ignored
and data after a completed pad sequence is DECODED, not silently dropped
(`'Zg==Zg=='` → `b'f\x06`'`). CPython before that fix truncated the trailing data
there, a parser differential it fixed as a security issue, and tors ships the fixed
machine on every interpreter it supports rather than matching each stdlib it happens
to run under (pre-fix stdlibs, and 3.10's distinct regex validator, are documented
divergences). `validate=True` is the default (decode-side callers want invalid input
to fail loudly); a non-ASCII `str` (including one holding lone surrogates) raises the
stdlib's own plain `ValueError`; non-`str` input raises `TypeError`.

**Async**: `await tors.aio.b64_decode(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.utf8_is_valid`

```python
def utf8_is_valid(raw: bytes) -> bool: ...
```

Answers "is this well-formed UTF-8?" without materializing a `str` or paying exception
flow: SIMD validation, GIL-released, `bool` out. `True` exactly when
`raw.decode("utf-8")` would succeed. The stdlib has no boolean primitive for the
question (its only spelling is decode-and-catch); the argument contract matches the
bytes-in surface.

## `tors.decode_utf16`

```python
def decode_utf16(
    raw: bytes,
    *,
    errors: Literal["strict", "replace"] = "strict",
    byteorder: Literal["native", "little", "big"] = "native",
) -> str: ...
```

`raw.decode("utf-16", errors=...)`, or the `"utf-16-le"`/`"utf-16-be"` spellings via
`byteorder=`: byte-exact over arbitrary bytes, one GIL-released pass. `byteorder="native"`
(the default) sniffs a leading BOM (`FF FE` little, `FE FF` big) and strips it from the
output, falling back to the host's own endianness when no BOM is present, matching the
plain `"utf-16"` codec name exactly. `byteorder="little"`/`"big"` never sniff or strip a
BOM: a leading BOM-like byte pair decodes as the literal U+FEFF character, matching
`"utf-16-le"`/`"utf-16-be"`.

Strict mode raises CPython's own `UnicodeDecodeError`, including `.encoding` (always the
RESOLVED label, `"utf-16-le"` or `"utf-16-be"`, never the bare `"utf-16"` name, even under
`byteorder="native"`). UTF-16 has FOUR distinct reasons, all verified against a running
interpreter:

- `"truncated data"`: a lone trailing byte, one short of a full code unit, with no
  pending high surrogate. Span: that one byte.
- `"unexpected end of data"`: a high surrogate with fewer than two bytes following it.
  Span: the surrogate through the end of the input (the partial tail has nothing useful
  left to do with it, so it folds into the one error).
- `"illegal UTF-16 surrogate"`: a high surrogate followed by a full code unit that is
  not a valid low surrogate. Span: the high surrogate's two bytes alone; the following
  unit is reprocessed from scratch.
- `"illegal encoding"`: a low surrogate reached other than as the second half of a
  valid pair. Span: its own two bytes alone.

`errors="replace"` emits exactly one U+FFFD per error unit and continues decoding:
CPython's own granularity, not one U+FFFD per byte.

No `encode_utf16` or `finalize_utf16`: encoding a trusted `str` carries none of the
untrusted-bytes-parsing risk that motivates the decode side, and `str.encode("utf-16-le")`
already covers it; a fused decode+normalize+hash twin has no demonstrated caller need,
unlike `finalize_utf8`'s.

```python
tors.decode_utf16(b"\xff\xfeh\x00i\x00")                        # "hi"
tors.decode_utf16(b"h\x00i", errors="replace")                  # "h�"
```

**Async**: `await tors.aio.decode_utf16(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.utf16_is_valid`

```python
def utf16_is_valid(raw: bytes, *, byteorder: Literal["native", "little", "big"] = "native") -> bool: ...
```

`tors.utf8_is_valid`'s shape for the UTF-16 side: `True` exactly when
`decode_utf16(raw, byteorder=byteorder)` (strict) would succeed, with no `str`
materialized on either path.

```python
tors.utf16_is_valid(b"h\x00i\x00")                              # True
tors.utf16_is_valid(b"h\x00i")                                  # False
```

## `tors.detect_encoding`

```python
def detect_encoding(raw: bytes, *, tld: str | None = None) -> str: ...
```

Best-guess character encoding for non-UTF8 legacy/OCR byte content, via `chardetng`
(the detector Firefox ships), GIL-released. Unlike `utf8_is_valid`/`decode_utf8`
there is no ground truth here: this is a heuristic guesser, not a validator, and it
always returns SOME codec name rather than raising for well-formedness reasons. The
intended pipeline: call `utf8_is_valid` first, and only reach for `detect_encoding`
on the bytes that already failed that check, then decode with the returned name.

The returned name is a Python `bytes.decode()`-usable codec name: mostly the WHATWG
Encoding Standard label `chardetng` reports (`"windows-1252"`, `"Shift_JIS"`,
`"UTF-8"`), translated to Python's own spelling for the two labels in this
detector's candidate set that Python's codec registry doesn't recognize under
their WHATWG names (`"windows-874"` → `"cp874"`; the logical-ordered Hebrew label
`"ISO-8859-8-I"` → `"ISO-8859-8"`, since the byte mapping is identical and a
decode doesn't consult the bidi-ordering hint the `-I` suffix carries). `tld` is
an optional top-level-domain hint that
disambiguates language-family-ambiguous input: accepted in whatever natural
spelling a caller has (`".jp"`, `"JP"`, or `"jp"`; a full domain or non-ASCII input
degrades to "no hint" rather than raising, since `chardetng`'s own `tld` parameter
PANICS on anything but a bare lowercase ASCII label with no period: a caller
shouldn't crash over a hint spelled the normal way).

```python
tors.detect_encoding("Café".encode("windows-1252"))
# "windows-1252"
```

## `tors.diff_opcodes`

```python
def diff_opcodes(
    a: str, b: str, *, deadline_ms: float | None = None
) -> list[tuple[str, int, int, int, int]]: ...
```

**Async**: `await tors.aio.diff_opcodes(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

`difflib.SequenceMatcher(None, a, b).get_opcodes()`'s SHAPE at native speed:
`(tag, i1, i2, j1, j2)` tuples with `tag` in `{"equal", "replace", "delete",
"insert"}`, ranges monotone/contiguous/covering both sides, adjacent delete+insert
merged into `replace` exactly as difflib presents it, indices in Python `str`
(codepoint) units. CHARACTER-level, like difflib on `str` operands: that is what
makes difflib the parity oracle; `tors.diff_opcodes_lines` (below) is the
line-level spelling. Exact agreement with difflib is pinned on the classes whose
canonical opcode list is FORCED: verified by difflib's own answer carrying the
canonical single-op shape (pure insert/delete, single-run replace, all-equal,
empty operands); structural validity (the opcodes reconstruct both sides) is
pinned by hypothesis over arbitrary pairs; and the boundary cases where the two
algorithms pick different valid alignments (repeated-flank contexts, where
difflib's longest-match recursion emits a non-minimal insert+delete split and
Myers + run-maximization emits the minimal contiguous change) are documented
and tested, never silent. The four tag strings are interned once per call, so
`op[0] is "equal"` holds exactly as it does for difflib's own tuples.

`deadline_ms` bounds the superlinear worst case: on hard inputs (few anchorable unique
records: a character-level permutation is the measured shape) the Myers search's work
grows roughly ~n² with size (measured: 50k chars 0.32 s, 200k 3.67 s, 400k 13.81 s, 1M
183.6 s). On expiry the incomplete result is discarded and `TimeoutError` is raised
naming the elapsed cost and the deadline; `None` (the default) is the unbounded diff,
unchanged; a non-positive or non-finite value raises `ValueError` before any work runs
(an enormous-but-finite value is legal: it saturates to "unbounded" rather than
erroring).

```python
tors.diff_opcodes("qabxcd", "abYcd")
# [("delete", 0, 1, 0, 0), ("equal", 1, 3, 0, 2),
#  ("replace", 3, 4, 2, 3), ("equal", 4, 6, 3, 5)]
```

## `tors.diff_opcodes_lines`

```python
def diff_opcodes_lines(
    a: str, b: str, *, deadline_ms: float | None = None
) -> list[tuple[str, int, int, int, int]]: ...
```

**Async**: `await tors.aio.diff_opcodes_lines(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

The LINE-level spelling of `diff_opcodes`: the same opcode shape, engine, validity
contract, boundary-class divergences, and `deadline_ms` machinery; but the operands
are tokenized as lines (each line keeps its `\n`, the last may lack one) and the
indices address LINES, so `a_lines[i1:i2]` slicing reconstructs. Measured on the
12 MiB near-identical pair: 5 line-opcodes and 1.9–2.6 ms walls where the char-level
spelling emits 235 opcodes over 76–88 ms: one call instead of a Python split
round-trip plus the diff.

**The tokenization is `'\n'`-only, not full `str.splitlines()`.** `a_lines`/`b_lines`
must be built as `a.split("\n")`-with-terminators-reattached (equivalently,
`re.split(r"(?<=\n)", a)` with a trailing empty string dropped if `a` ends
in `'\n'`) for the reconstruction contract above to hold: NOT
Python's `str.splitlines(keepends=True)`, which additionally breaks on `\r`,
`\r\n` collapsed to one line, `\v`, `\f`, `\x1c`–`\x1e`, `\x85`, U+2028 LINE SEPARATOR, and U+2029 PARAGRAPH
SEPARATOR. Text using any of those as its only line terminator (classic Mac
`\r`-only line endings are the realistic case) tokenizes as ONE line under
`diff_opcodes_lines` where `str.splitlines()` would see several, a real,
measured divergence. `'\n'` is the terminator every
practical document/version-diff pipeline actually uses; the narrower contract is a
documented choice, not an oversight.

```python
tors.diff_opcodes_lines("l1\nl2\nl3\n", "l1\nX\nl3\nl4\n")
# [("equal", 0, 1, 0, 1), ("replace", 1, 2, 1, 2), ("equal", 2, 3, 2, 3),
#  ("insert", 3, 3, 3, 4)]
```

## `tors.replace_many`

```python
def replace_many(text: str, replacements: dict[str, str]) -> str: ...
```

Simultaneous multi-pattern replace in one GIL-released native pass, a primitive
CPython does not have: every occurrence of every key is replaced by its value with
`find_patterns`'s exact search semantics.

- **leftmost-longest**: among the keys matching at a position, the longest wins
  regardless of dict order (NOT `re.sub` alternation's leftmost-first priority);
- **non-overlapping**: the scan resumes at each match's end;
- **never re-scanned**: a value that itself contains a key does not cascade:
  `replace_many("a", {"a": "ba"})` is `"ba"`, not `"baba"`.

The alternatives are all worse: chained `str.replace` calls are N whole-text
GIL-held passes, and `re.sub` alternation is leftmost-first AND rescans its own
output. The canonical consumers are redaction and normalization maps. Identity
contract: `replace_many(s, m) is s` exactly when `replace_many(s, m) == s` (no
match, net-identity, or the empty dict all return the ORIGINAL object). Argument
contract: `replacements` must be exactly a `dict[str, str]` (a list/tuple of pairs
raises `TypeError`; dict order cannot matter: pinned); an empty key raises
`ValueError("empty pattern")`; lone surrogates raise `UnicodeEncodeError` at the
standard str-in boundary. No list-shape marshalling class: the return is one
string; the GIL-held residue is the argument walk plus the O(output) marshalling,
measured at the ping floor even on a 1.28M-replacement dense map.

```python
tors.replace_many("the cat sat in the catalogue", {"cat": "dog", "catalogue": "library"})
# "the dog sat in the library"
```

## `tors.replace_many_masked`

```python
def replace_many_masked(text: str, replacements: dict[str, str], mask: str = "*") -> str: ...
```

The length-preserving redaction spelling of `replace_many`: same one-automaton
leftmost-longest, non-overlapping, never-rescanned scan, but each matched span's
character length is never allowed to change. For a matched span of L characters
and its key's replacement value of C characters, exactly one branch fires:

- **C >= L (value at least as long) means TRUNCATION**: the value's first L characters
  are emitted; `mask` is never consulted.
- **C < L (value shorter) means PADDING**: the whole value is emitted, followed by
  `L - C` copies of `mask`.

The replacement VALUE is never ignored: it is truncated or padded to the
MATCHED PATTERN's own character length, not masked outright. To redact a span
completely, use `""` as the value: the empty value always takes the padding
branch and becomes pure `mask * L`.

```python
tors.replace_many_masked("the cat sat", {"cat": "[REDACTED]"}, "*")
# "the [RE sat"   -- "[REDACTED]" (10 chars) truncates to "[RE" (3 chars); mask unused
tors.replace_many_masked("the cat sat", {"cat": "X"}, "*")
# "the X** sat"   -- "X" (1 char) pads to "X**" (3 chars)
tors.replace_many_masked("the cat sat", {"cat": ""}, "*")
# "the *** sat"   -- empty value pads to pure mask
```

Because every matched span is replaced by exactly as many characters as it
matched, and every non-matching span is copied through untouched, the output's
CHARACTER count and every non-matching span's offsets equal the input's:
offsets computed before redaction (`find_patterns` spans, word/sentence bounds,
diff opcodes) stay valid against the redacted text. BYTE length may still change
(a multibyte value or mask replaces the matched key's bytes); only the
character-unit guarantee is made, which is the unit every offset this crate
reports uses.

`mask` must be exactly one character (`str` of length 1, i.e. one Python
codepoint): a multi-character or empty `mask` raises `ValueError`, including a
visually-single grapheme built from more than one codepoint (e.g. NFD `"e" +
COMBINING ACUTE ACCENT"`, two codepoints, is rejected). The rest of the argument
contract is `replace_many`'s exactly: `replacements` must be a `dict[str, str]`,
an empty key raises `ValueError("empty pattern")`, lone surrogates raise
`UnicodeEncodeError`. Identity contract: `replace_many_masked(s, m, c) is s`
exactly when `replace_many_masked(s, m, c) == s` (the empty dict, no match, or a
net-identity mask/truncation/padding all return the ORIGINAL object).

## `tors.sentence_bounds` / `tors.sentence_bounds_iter`

```python
def sentence_bounds(text: str) -> list[tuple[int, int]]: ...
def sentence_bounds_iter(text: str) -> Iterator[tuple[int, int]]: ...
```

UAX #29 sentence segmentation (rules SB1–SB999) via the same `unicode-segmentation`
tables as the word segmenter: the other segmenter the stdlib lacks. Offsets are
Python `str` indices (`text[start:end]` IS the sentence); per the standard
(SB10/SB11), trailing spaces after a terminator belong to the PRECEDING sentence.
Pinned by a hand-derived rule-cited table (`"3.4"` SB6, `"U.S.A"` SB7, `"etc. and
so on"` SB8, the ideographic `。`, the degenerates) plus structural properties and
list/iter sequence parity. Measured at 12 MiB of prose: 170,037 sentences (~1/22nd
the word count), so the list API's marshalling meets the shared GIL budgets. The
limitations note shared with `word_bounds`: rule-based UAX #29 only, no dictionary
segmentation for spaceless scripts (Thai/Khmer/Burmese/Japanese); ICU4X is the
heavyweight alternative if you need those.

```python
tors.sentence_bounds("One. Two. U.S. stocks fell.")
# [(0, 5), (5, 10), (10, 27)]
```

## `tors.sentence_count`

```python
def sentence_count(text: str) -> int: ...
```

`word_count`'s shape over sentences: a single `int`, `O(1)` memory, the whole scan
GIL-released. `sentence_count(t) == len(sentence_bounds(t))`, pinned.

```python
tors.sentence_count("One. Two. U.S. stocks fell.")
# 3
```

## `tors.find_patterns`

```python
def find_patterns(patterns: list[str], text: str) -> list[tuple[int, int, int]]: ...
```

Leftmost-longest, non-overlapping multi-pattern substring search in one GIL-released
native pass: every occurrence of every pattern, reported as `(start, end,
pattern_index)` with `end` exclusive, offsets in Python `str` (codepoint) units:
`text[start:end] == patterns[pattern_index]` for every reported match.

- **leftmost**: a match is reported at the earliest position any pattern matches;
- **longest**: among the patterns matching at that position, the longest wins
  regardless of list order (NOT regex-alternation leftmost-first priority);
- **non-overlapping**: the scan resumes at each match's end; results come back in
  strictly increasing start order;
- **duplicates report the first index** they occupy in the list.

No stdlib primitive has these semantics (`re` alternation is leftmost-first), so the
contract is proven against a brute-force pure-Python leftmost-longest reference plus
structural validity over hypothesis-generated multi-byte alphabets. Argument contract:
exactly `list[str]` (a tuple raises `TypeError`); an empty pattern string raises
`ValueError("empty pattern")`; an empty list returns `[]` immediately; lone surrogates
raise `UnicodeEncodeError` at the argument boundary.

```python
tors.find_patterns(["cat", "catalogue"], "the cat sat in the catalogue")
# [(4, 7, 0), (19, 28, 1)]
```

## `tors.find_patterns_iter`

```python
def find_patterns_iter(patterns: list[str], text: str) -> Iterator[tuple[int, int, int]]: ...
```

The streaming spelling of `find_patterns`: the whole search (pattern-list walk,
automaton build, scan, byte→char offset conversion) fills an internal buffer under
ONE GIL-released pass at construction, and each `__next__` holds the GIL only for a
single 3-tuple of ints: the streaming answer to the list shape's O(matches)
tuple-marshalling caveat (~13 ms held per 100k matches in the list shape). Same
match sequence as the list API, pinned to sequence parity.

```python
list(tors.find_patterns_iter(["cat", "catalogue"], "the cat sat in the catalogue"))
# [(4, 7, 0), (19, 28, 1)]
```

## `tors.count_matches`

```python
def count_matches(patterns: list[str], text: str) -> int: ...
```

The count spelling of `find_patterns`: the same leftmost-longest, non-overlapping
search answering just the number, with NO match vector materialized (the list
spelling would materialize ~250 MiB of matches on a 100 MiB dense corpus just to
answer "how many") and no byte→char offset pass (counting is offset-free).
`count_matches(p, t) == len(find_patterns(p, t))`, pinned. A single `int` return: no
marshalling class at all.

```python
tors.count_matches(["cat", "catalogue"], "the cat sat in the catalogue")
# 2
```

## `tors.CompiledPatterns`

```python
class CompiledPatterns:
    def __init__(self, patterns: list[str]) -> None: ...
    def __len__(self) -> int: ...
    def find(self, text: str) -> list[tuple[int, int, int]]: ...
    def find_iter(self, text: str) -> Iterator[tuple[int, int, int]]: ...
    def count(self, text: str) -> int: ...
    def replace_many(self, text: str, replacements: dict[str, str]) -> str: ...
    def replace_many_masked(
        self, text: str, replacements: dict[str, str], mask: str = "*"
    ) -> str: ...
```

The `re.compile()` answer to `find_patterns`/`replace_many`'s per-call automaton build
(the same `tors.CompiledLemmaDict` pattern, applied to pattern search instead of
lemmatization — see `tors.apply_pipeline`'s docs above): building the Aho-Corasick
automaton is the expensive part of every one of these calls, and a fixed vocabulary
scanned over many documents (a redaction pipeline, a tagger) otherwise rebuilds it on
every single call for no reason. `CompiledPatterns(patterns)` builds it once under one
GIL-released pass; every method after that is the free function's exact scan MINUS the
automaton build, sharing the compiled automaton by one `Arc` clone per call — sound to
reuse across many calls and threads with no synchronization beyond that refcount.

Each method mirrors its free-function twin exactly: `cp.find(text) ==
tors.find_patterns(patterns, text)`, `cp.count(text) == tors.count_matches(patterns,
text)`, and so on, for every method above. The two `replace_many*` methods validate
their `replacements` dict at CALL time (values can change per call; only the pattern
set is fixed at construction): it must key EXACTLY the compiled pattern set — every
compiled pattern paired with a value, no extra keys — or `ValueError` names the unknown
and missing keys. `len(cp)` is the number of compiled patterns. Construction takes the
same argument contract as `find_patterns`' pattern list (`list[str]`, non-empty
entries); the empty pattern list compiles successfully (every scan finds nothing) and
then accepts only the empty replacements dict. Immutable once built: there is no way to
add or remove a pattern from an existing `CompiledPatterns`.

```python
cp = tors.CompiledPatterns(["cat", "catalogue"])   # pay the automaton build once
for doc in corpus:
    cp.find(doc)                                    # O(1) automaton reuse per call
cp.replace_many("the cat sat", {"cat": "dog", "catalogue": "library"})
# 'the dog sat'
```

## `tors.extract_code_blocks`

```python
def extract_code_blocks(text: str, lang: str | None = None) -> list[tuple[str | None, str, int, int]]: ...
```

Extracts every fenced code block in `text` per CommonMark §4.5's fenced-code-block
grammar, one GIL-released native pass: `(language, code, start, end)` per block, `end`
exclusive, offsets in Python `str` (codepoint) units. The grammar is hand-rolled
directly in Rust rather than delegated to a general Markdown parser: LLM chat output
is rarely deeply-nested Markdown, so the extra correctness a full CommonMark engine
buys is mostly wasted weight for this. Rules: 0–3 leading spaces, then 3+ backticks or
tildes of the same character open a fence; an optional info string's first word is the
reported `language`; content lines are dedented by the fence's own indentation; a
closing fence needs the same character, at least as long as the opener, with nothing
but whitespace after it; absent one, the block simply runs to end of input (an
unterminated fence still yields a block). `lang=` filters to an exact, case-sensitive
language match. The scope is narrower than full CommonMark: no indented-code-block
recognition, no tab-expansion of indentation; both are documented non-goals.

```python
tors.extract_code_blocks("hi\n```py\nprint(1)\n```\n")
# [("py", "print(1)\n", 3, 22)]
```

## `tors.strip_code_fences`

```python
def strip_code_fences(text: str) -> str: ...
```

Unwraps the single most common case on its own: a whole response wrapped in one
fence. If `text`, trimmed of leading/trailing whitespace, is *exactly* one fenced
code block, its dedented content comes back; anything else (prose around a fence,
multiple blocks, no fence at all) comes back completely unchanged (not even
whitespace-trimmed), so it is safe to call unconditionally on arbitrary model output.
`tors.strip_code_fences(s) is s` exactly when `s` is not that single-block case (the
same identity-return idiom as `normalize`/`replace_many`).

```python
tors.strip_code_fences("```py\nprint(1)\n```")
# "print(1)\n"
```

## `tors.dedent`

```python
def dedent(text: str) -> str: ...
```

`textwrap.dedent(text)`, byte-exact against CPython 3.14's rewritten algorithm
(`Lib/textwrap.py`, gh-131792): the longest common leading-whitespace-run *string*
shared by every non-whitespace-only line is stripped from each line, and
whitespace-only lines normalize to empty. Tabs and spaces are distinct characters for
the common-prefix computation, exactly as the stdlib documents (`"  x"` and `"\tx"`
share no margin); never tab-expanded. `tors.dedent(s) is s` exactly when
`tors.dedent(s) == s`. The natural companion to
`extract_code_blocks`/`strip_code_fences` for re-indenting pulled-out code, and a
GIL-released primitive in its own right for any pipeline calling `textwrap.dedent` on
large text today.

**Version note**: `textwrap.dedent` was rewritten in CPython 3.14, and the rewrite
changed observable behavior, not just performance: a line consisting solely of some
*other* Unicode whitespace character (`\v`, `\f`, a non-breaking space, ...) is now
recognized as blank and normalized, where the pre-3.14 implementation only recognized
`[ \t]` as blank-line whitespace. `tors.dedent` ships the 3.14 behavior unconditionally
on every Python version it supports (3.10+), the same cross-version parity convention
`b64_decode` uses (see its docs above): on an older interpreter, `tors.dedent` and that
interpreter's own `textwrap.dedent` can diverge on inputs containing a non-space/tab
whitespace-only line. Ordinary space/tab indentation, the overwhelming common case, is
unaffected either way.

```python
tors.dedent("  a\n  b\n")
# "a\nb\n"
```

## `tors.repair_json`

```python
def repair_json(
    s: str,
    *,
    skip_json_loads: bool = False,
    ensure_ascii: bool = True,
    strict: bool = False,
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
    deadline_ms: float | None = None,
) -> str: ...
```

Repairs malformed JSON from LLMs, APIs, logs, and user input, returning the
repaired document as a JSON string: a Rust port of the Python library
json_repair (Stefano Baccianella, MIT,
[github.com/mangiucugna/json_repair](https://github.com/mangiucugna/json_repair)),
with parity pinned to json-repair==0.63.4 — the upstream behavior corpus
is ported into `tests/test_json_repair_{core,schema,parity}.py`, and a
differential suite runs tors and json_repair over the same inputs, so the
pin is proven, not asserted. The repair parser handles the failure modes model output actually
exhibits: missing and trailing commas, unquoted keys and values,
single-quoted and curly-quoted strings, truncated containers, comments,
stray prose around the payload, Python-isms (`None`/`True`/`False`, tuple
literals), doubled quotes, and broken escapes — in the default mode it
repairs instead of raising, whatever the input looks like.

**The fast path.** Unless `skip_json_loads`, a strict `json.loads`-parity
parse of the (fence-unwrapped, below) input runs first, and valid JSON
short-circuits the repair parser. The result is re-serialized through a
`json.dumps`-parity serializer either way, so valid-but-noncanonical input
normalizes — `{ "a" : 1 }` comes back as `{"a": 1}` — exactly like
upstream, which also re-dumps. Under a `schema` the probe does not
short-circuit: valid JSON is validated and repaired if noncompliant before
any fallback to the schema-guided parser, so a schema is enforced on clean
input too.

**The empty-string sentinel.** When nothing recoverable is found, the
repaired value is the empty string and `repair_json` returns the empty
string — not `"\"\""` — upstream's convention for never returning a bare
pair of quotes. The price is an ambiguity shared with upstream: a
legitimately repaired top-level empty-string value renders identically.
Check `result == ""` for "nothing recoverable". **Under `schema=` the sentinel
does not escape as a return**: the empty-string value is itself validated against
the schema, so a non-string-typed schema answers the same `ValueError` every other
nonconformant value does (an object schema: `"" is not of type "object"`) —
"nothing recoverable" becomes a raise, not a sentinel — while a string-typed
schema accepts it, an empty string being a valid string. Pinned by
`tests/test_json_repair_native.py`.

**The fence pre-pass.** Before anything else, the whole input is tested
against CommonMark's fenced-code-block grammar — the same single-block
unwrapping `tors.strip_code_fences` documents. If the trimmed input is
EXACTLY one fenced block, its content is unwrapped and repaired: tilde
fences, closers longer than their openers, and indented fences are handled
by the grammar rather than by the repair parser's garbage-skip reaching the
same answer by accident, and fenced-but-valid JSON takes the fast path
instead of the repair parser. The pre-pass also recovers fenced top-level
scalars (a fence wrapping just `"hi"` yields `"hi"`) where upstream returns
`""` — its one behavior change beyond parity, listed with the divergences
below. A response with prose around a fence, or multiple blocks, is not the
single-block case; compose with `tors.extract_code_blocks` for those:

```python
tors.repair_json("{ 'name': 'Ada', 'role': 'admin', }")
# '{"name": "Ada", "role": "admin"}'

tors.repair_json('```json\n{"ok": true}\n```')
# '{"ok": true}'

# a multi-block response: pull the json blocks, repair each
blocks = tors.extract_code_blocks(response, lang="json")
repaired = [tors.repair_json(code) for _, code, _, _ in blocks]
```

**`skip_json_loads`** skips the upfront strict parse only, forcing the
repair parser even on valid JSON — upstream's knob for callers who already
know the input is broken. It does NOT skip the parser-internal suffix
probe: after a prose prefix, once a top-level container starts, the parser
still tries a targeted strict decode of the value from that point — that
probe is part of the repair parser proper and stays on, exactly as
upstream behaves.

**`ensure_ascii`** is the one `json.dumps` serialization knob carried over:
`True` (the default) escapes every non-ASCII codepoint as `\uXXXX` (astral
characters as surrogate pairs), `False` emits them verbatim. Upstream's
other pass-through kwargs (`indent`, `sort_keys`, ...) are not ported.

**`strict`** flips the documented leniencies into `ValueError`s at the
first structural ambiguity instead of repairing past them: duplicate keys,
empty keys, a missing `:` after a key, an empty parsed value, an object
that parses empty but still carries characters, multiple top-level
elements, doubled quotes. The mode for input that SHOULD be valid and whose
first real defect you want named rather than patched. `strict=True`
together with `schema` raises
`ValueError("schema and strict cannot be used together.")`.

**`schema`** switches on schema-guided repair: the parsed value is aligned
to the schema — scalar coercions (`"4"` → 4, `"yes"` → `true`), fills for
missing values, defaults inserted for absent properties, extra properties
dropped where `additionalProperties` forbids them, union branches tried
until one validates — then validated in full, with failures raising
`ValueError` at the offending path (`"Expected string at $.name."`).
Accepts a JSON Schema dict, a boolean schema, or a pydantic v2 model (class
or instance): the model's `model_json_schema()` output is used directly,
with field `default`s and `default_factory`s injected into it (factory
first) and Enum-member defaults carried as their `.value` — so the
model → schema → LLM → repair → `Model.model_validate` agent loop needs no
manual schema step. Mutually exclusive with `strict` (above). See
`tors.repair_json_diagnostics` for the action-by-action log of everything
schema mode does.

**`salvage`** is the best-effort spelling of schema mode (upstream's
`schema_repair_mode="salvage"`, spelled as a bool): top-level fragments
that fail the schema are skipped until one validates, invalid array items
and extra properties are dropped rather than raised, and missing `required`
properties are filled from their subschema's `default`/`const`/`enum[0]`.
It requires a schema: `salvage=True` without one raises
`ValueError("salvage=True requires schema.")`.

**`deadline_ms`** (default `None` = unbounded) bounds the whole repair the
way `diff_opcodes`' `deadline_ms` does: a positive-finite-or-`None` budget
validated up front, `TimeoutError` on expiry. The clock starts at the top of
the call — the fence pre-pass and the `json.loads` fast-path attempt burn
the budget too, and a fast path that *completes* past the budget still
returns its answer (the deadline stops further work; it does not nullify
done work). It is a DoS backstop for the pathological O(n²) parser shapes
`tors` shares with upstream `json_repair` — duplicate-key-in-array splices,
empty-object splices, and a backslash-run string scan — a bounded *abort*,
not a speed-up: a completing parse is
byte-identical whether or not a deadline is set, and a benign large document
does not trip a generous budget (the deadline discriminates pathological
*shape*, not *size*). It applies to all three spellings and is checked with
the GIL released, so `TimeoutError` is raised after reacquiring it — the
same shape as `diff_opcodes`, including the message:
`"<spelling> deadline exceeded: elapsed 101.2ms > deadline_ms 100.0ms"`.

Two honest limits. The bound is *soft*: the tight loops sample the clock
1-in-256, but every O(n) unit — a buffer splice, a long scan, a wide span
build — forces the very next check to read it, so at most one such unit
runs past an expired budget (measured worst overshoot ~8% at n=1M). And
it bounds CPU *time*, not native stack growth: a runaway continuation
recursion can still overflow the stack before the budget expires — that
class is depth-guarded separately (`MAX_NESTING`), not time-bounded.
Cost when unset: nothing on the valid-JSON fast path, and one predicted
branch per dispatch turn in the repair parser — measured ~+6% worst-case
on a multi-MB `skip_json_loads=True` parse, ~+3% on a corrupt-document
repair, within noise on the pathological shapes. Cost when set: ≤2% on
top of that (the checks are sampled).

Argument contract: a non-`str` `s` raises `TypeError` (pyo3 extraction); a
`schema` that is not a dict, bool, model, or `None` raises
`ValueError("schema must be a JSON Schema dict, boolean schema, or pydantic
v2 model.")`; a bad `locale` (below) raises a `ValueError` naming it; a
lone surrogate in `s` raises `UnicodeEncodeError` at the argument boundary,
the same str-in convention every function here documents.

**GIL.** One detached native pass covers the fence pre-pass, the repair
parse, the schema alignment, and the validator compile. The GIL-held
residue is the caller-supplied `schema` dict walk (converted into the
internal value tree before the pass starts) and the return marshalling
after it: one string for this spelling; for `tors.repair_json_loads` and
`tors.repair_json_diagnostics`, the construction of the Python object tree
and the diagnostics list, O(result) — the same disclosed marshalling class
the list-returning search functions document (see `tors.find_patterns_iter`'s
O(matches) note and the README's Performance section).

**Divergences from upstream json_repair** — the complete list; the
differential suite pins everything else to json-repair==0.63.4:

- **Not ported**: `stream_stable`, the text repair log (superseded by
  `tors.repair_json_diagnostics`), the file and CLI flavors
  (`json_fd`/`load`/`from_file` and the CLI module), `json.dumps`
  pass-through kwargs beyond `ensure_ascii`, and the `schema_repair_mode`
  string (the `salvage=` bool instead).
- **Lone surrogates**: a `\uXXXX` escape that decodes to an unpaired
  surrogate becomes U+FFFD — a Rust string cannot hold a lone surrogate,
  and pyo3 could not return one anyway. Input text containing literal lone
  surrogates never reaches the parser: it raises `UnicodeEncodeError` at
  the argument boundary, the same str-in convention every function here
  documents.
- **Non-ASCII digits**: Unicode Nd digits beyond 0-9 (Arabic-Indic and
  friends) do not enter the number path — upstream's `str.isdigit` is
  Unicode-wide, tors's check is ASCII-only. Pure runs (like `"١٢٣"`) fail
  to repair on both sides; a non-ASCII digit LEADING ASCII digits
  (`"²5"`) makes upstream abandon the value where tors skips the mark and
  parses the digits — the one mixed-run shape where the classes differ.
- **Fenced top-level scalars** are recovered where upstream returns `""`
  (the fence pre-pass above).
- **The validation boundary**: schema validation runs on the Rust
  `jsonschema` crate, so failure-message wording is that crate's, not
  Python `jsonschema`'s; integers beyond u64 validate lossily as f64;
  non-finite numbers under a schema raise `ValueError` where Python
  tolerates `NaN`; union branches are validated wrapped so `#/...` refs
  keep root scope (a pathological subschema-local `$defs` shadowing root
  `$defs` diverges). `format` is unasserted on BOTH sides — upstream passes
  no `format_checker`, and tors matches it.
- **Deep nesting**: `ValueError("Input nesting exceeds the supported parser
  recursion depth.")` at 200 nested containers, where upstream raises an
  uncaught `RecursionError` at roughly its own recursion limit — the same
  failure normalized into the error catalog at a lower, pinned threshold.
- **On by default, tors-native**: key-typo remap, enum "Did you mean ..."
  suffixes, date/uuid normalization, numeric extraction tiers, and the
  diagnostics output are extensions upstream does not have — see
  `tors.repair_json_diagnostics`. One consequence: on the strict fast path
  tors normalizes already-valid values (date formats, fold-matching key
  spellings, directly-declared `properties` and `allOf` members) where
  upstream's valid-JSON shortcut returns them untouched — matching what
  upstream's own repair lane does with `skip_json_loads=True`, so tors is
  self-consistent across its two lanes for everything except
  `oneOf`/`anyOf`-wrapped guidance, where the fast path deliberately does
  not guess which branch applies (upstream's fast path has the same
  reach).

## `tors.repair_json_loads`

```python
def repair_json_loads(
    s: str,
    *,
    skip_json_loads: bool = False,
    strict: bool = False,
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
) -> dict[str, Any] | list[Any] | str | int | float | bool | None: ...
```

The `json.loads` drop-in spelling of `tors.repair_json`: the same repair
pipeline and the same knobs (`skip_json_loads`, `strict`, `schema`,
`salvage` — see `tors.repair_json`'s docs above), with the decoded object
returned directly instead of a re-serialized string, so a repaired document
goes straight into use with no `json.loads` round trip. There is no
`ensure_ascii` here because nothing is serialized. The empty-string
sentinel carries over as a real `""` return — "nothing recoverable",
ambiguous with a legitimately repaired top-level empty-string value, the
same collapse upstream's `loads` has. Under `schema=` the sentinel is
schema-validated instead: a non-string-typed schema turns "nothing
recoverable" into the same `ValueError` every other nonconformant value
raises — see `tors.repair_json`'s sentinel paragraph above.

```python
tors.repair_json_loads("{'users': [{'name': 'Ada',}]}")
# {'users': [{'name': 'Ada'}]}
```

## `tors.repair_json_diagnostics`

```python
def repair_json_diagnostics(
    s: str,
    *,
    skip_json_loads: bool = False,
    strict: bool = False,
    schema: dict[str, Any] | bool | type[Any] | None = None,
    salvage: bool = False,
    locale: str | dict[str, str] | None = None,
) -> tuple[dict[str, Any] | list[Any] | str | int | float | bool | None, list[dict[str, Any]]]: ...
```

The `(value, diagnostics)` spelling: `tors.repair_json_loads`'s exact
result paired with one record per repair action taken. Upstream narrates
these actions to its `logging` facility; tors does not port that text log —
this function is the replacement, the same narration as data. Each record
carries `action` (from the closed vocabulary below), `path` (json_repair's
path spelling: `"$"`, `"$.key"`, `"$.items[3]"`), `detail` (one human
sentence), and, where the action has them, `from`/`to` (the value before
and after) and `suggestion`. The vocabulary:

| action | when it fires |
|---|---|
| `coerce` | a scalar is converted to the schema's type: `"4"` → 4, `"4.0"` → 4, `4` → `"4"`, `"1.5"` → 1.5, `"yes"`/`"on"`/`"1"` → `true` and `"no"`/`"off"`/`"0"` → `false`, a number to its truthiness |
| `fill` | a missing value (an object value slot that runs straight into `,` or `}`, e.g. `{"key":}`) is filled from the schema: `const`, else `enum[0]`, else `default`, else the type's empty value (`""`, `0`, `false`, `[]`, `{}`, `null`) |
| `insert_default` | a property absent from the object whose subschema declares `default` (and is not `required`) gets that default |
| `remap_key` | tors-native: a key matching no property is remapped to the closest property name — thresholds below |
| `suggest` | tors-native, report-only: a near-miss that was NOT auto-corrected — a key, an enum member, an ambiguous date, or a disclosed numeric-format assumption, per the four features below |
| `drop_property` | an extra property not covered by the schema is dropped (`additionalProperties` does not allow it) |
| `drop_item` | an array item is dropped: invalid under its item schema while salvaging, or beyond tuple-form `items` not covered by `additionalItems` |
| `unwrap_string` | a string value holding a JSON document is parsed and unwrapped to the object/array the schema expects (under salvage, repaired first if merely malformed) |
| `wrap_array` | a non-array value is wrapped in a single-element array to match an array schema |
| `fill_required` | salvage: a `required` property missing from the object is filled from its subschema's `default`/`const`/`enum[0]` |
| `format_date` | tors-native: a date/date-time string is normalized to the RFC 3339 form — below |
| `skip_fragment` | salvage: a top-level fragment that does not match the schema is skipped while hunting for one that does |
| `map_array_to_object` | salvage: a list with exactly the schema's property count is mapped onto those property names, in order |
| `unwrap_root_array` | salvage: a single-item root array `[{...}]` is unwrapped to `{...}` |

**Scope of the log (v1).** Parser-level repair narration — the syntax-layer
fixes `repair_json` performs without a schema — is not recorded yet, so a
schema-free call returns an empty list; the log covers schema-layer actions
and the tors-native suggestions below.

**Key-normalization ladder (deterministic tier).** A key that differs
from a property only by case or separator style (`first-name`,
`First Name`, `FIRST_NAME` → `first_name`) remaps with confidence 1.0 —
the match is exact after folding, not a guess — so it fires even on
permissive schemas and on the valid-JSON fast path (the un-remapped shape
strands real data on a dead key while the property takes its default).
One guard keeps it safe: the rename is kept only when the value can live
under the target property (a speculative repair through it succeeds), so
an incompatible value keeps its original, already-valid key instead of
turning valid input into a coercion failure. This tier also reaches
`allOf` members (pydantic's inheritance shape); `oneOf`/`anyOf` stay
unreached on the fast path.

**Key-typo remap.** When an object key matches no `properties` entry and no
`patternProperties` pattern, tors scores it against every property name
with `tors.jaro_winkler` and remaps it (`from` the old key, `to` the new)
only when the evidence is strong AND the alternative is loss or failure:
the best score is >= 0.75, it is unique (the second-best more than 0.05
lower), the target property is absent from the object, and either
`additionalProperties` is false (the key would otherwise be dropped) or the
target is in `required` (validation would otherwise fail). An exact
case-insensitive match scores a perfect 1.0. Below the remap bar, a best
score >= 0.60 still emits a report-only `suggest` diagnostic and falls
through to upstream semantics: no remap, no drop, the key keeps its
spelling.

**Enum suggestions.** When a value fails an `enum` check (never `const`),
the `ValueError` gains " Did you mean '...'?" naming the closest STRING
enum member under the same jaro-winkler >= 0.60 threshold, plus a
`suggest` diagnostic. Enum values are never auto-remapped: a near-miss is
reported, not guessed.

**Date, time, and uuid normalization.** For string values whose subschema
declares `format: "date"`, `"date-time"`, `"time"`, or `"uuid"` (directly
or through an `allOf` member), accepted forms normalize: ISO dates
(`YYYY-MM-DD`, two-digit padded) and slash dates (`YYYY/MM/DD`,
padding-tolerant); date-times `<date>[T ]HH:MM[:SS[.frac]]` on either date
spelling; month-name forms (`March 15, 2024`, `15 March 2024`,
`15 Mar, 2024`, case-insensitive). Dates come out `YYYY-MM-DD`;
date-times come out `YYYY-MM-DDTHH:MM:SS[.frac]` — seconds always
emitted, the fractional part trimmed to its shortest exact form — and an
input that carried an offset (Z or ±HH:MM/±HHMM) normalizes to its UTC
INSTANT with a `Z` rendering (`2024-03-15T14:30:00+0530` →
`2024-03-15T09:00:00Z`); a missing offset stays missing. `format: "time"`
gains seconds (`14:30` → `14:30:00`); `format: "uuid"` canonicalizes to
lowercase when the shape is a UUID (non-UUID strings pass through for
validation to judge). Numeric `X/Y/YYYY` (or `YYYY/X/Y`) forms coerce ONLY
when a component over 12 disambiguates month from day; when both
candidates are 12 or under (`03/04/2024`) the date is genuinely ambiguous
and gets a `suggest` diagnostic instead of a guess. Invalid calendar dates
(month lengths, leap years) are left for validation. `format` is not
otherwise enforced — upstream passes no `format_checker`, and tors matches
it; this normalization is the one place `format` is consulted at all.

**Numeric coercion ladder and `locale`.** String values under an
`integer`/`number` property climb a four-tier ladder: (1) the whole
trimmed string parses; (2) the string minus unambiguous noise (underscores,
fullwidth and Arabic-Indic script digits, and — with a known locale — that
locale's own separators) parses; (3) exactly one number token in the prose
extracts (`"USD 50"` → 50, `"$1,234.56"` → 1234.56, `-"50"` → -50), with
percent suffixes read by the declared type (`"50%"` → 0.5 on `number`
fields, the fraction; → 50 on `integer` fields, the percent count); (4)
Auto mode's separator-ambiguity resolution, below. A dropped decimal
marker never extracts (`.5` on an integer field refuses, never 5), and
Python's unbounded integer semantics hold at any magnitude —
`"12345678901234567890123"` coerces exactly, never a saturating cast.

`locale=` tells tors which separator convention the model uses: a BCP 47
tag string (`"de-DE"`, case-insensitive, `-` or `_`; region variants like
`de-CH` carry their CLDR separators; Lakh-style grouping locales such as
`en-IN` are refused) or a dict `{"decimal": ..., "grouping": ...}` of
one-character separators for conventions the table does not carry. With a
known locale every form is deterministic: `"1,234"` reads 1.234 in German
and 1234 in English, by data rather than by guess. The default (`locale=None`,
Auto) assumes en-US for the separator-ambiguous shapes — but never
silently: both readings are extracted and filtered by the declared type
and the property schema, a single surviving reading is the deterministic
answer (silent), and when both survive the en-US reading wins WITH a
`suggest` diagnostic naming the discarded reading's `locale=` override
(`"1,234"` on a number field → 1234 plus `locale='de-DE'`). A single
separated number whose readings all fail the declared type refuses with
the retry-able hint (`pass locale='en-US' or 'de-DE' ...`); prose without
a single number gets the plain upstream refusal.

```python
value, diags = tors.repair_json_diagnostics(
    '{"count": "4"}',
    schema={"type": "object", "properties": {"count": {"type": "integer"}}},
)
# ({'count': 4}, [{'action': 'coerce', 'path': '$.count', ...}])
```

## `tors.truncate_to_bounds`

```python
def truncate_to_bounds(text: str, max_chars: int, boundary: Literal["word", "sentence"] = "word") -> str: ...
```

Truncates `text` to at most `max_chars` codepoints, one GIL-released native pass,
cutting at the last word (or, `boundary="sentence"`, sentence) boundary at or before
`max_chars` instead of mid-word/mid-sentence: composing `word_bounds`/
`sentence_bounds`, the crate's own UAX #29 segmentation already shipped, rather than a
new algorithm. The context-window/token-budget-fitting primitive: `text[:max_chars]`
risks cutting a word or a grapheme in half, which this avoids.

Every cut point is also grapheme-cluster-safe: a word/sentence boundary that would
split a cluster (Thai SARA AM, combining accents, ZWJ emoji sequences) is never used,
so a combining mark is never separated from its base character. If no boundary fits at
or before `max_chars` (a single word/sentence longer than the budget, or `max_chars ==
0`), the fallback is a hard cut at the largest grapheme boundary `<= max_chars`: a
documented fallback, never a silent surprise, and still cluster-safe, so it
can land short of `max_chars` when the budget would otherwise split a cluster. The one
invariant that never breaks either way: the result never exceeds `max_chars`
codepoints. The cut point is then trimmed of trailing whitespace (`str.rstrip`-
equivalent): cutting right after a word/sentence boundary can otherwise leave a
dangling separator space, since `word_bounds` segments the inter-word space on its own
and `sentence_bounds` carries a sentence-terminal's trailing space on the PRECEDING
sentence (UAX #29 SB9-SB11).

`tors.truncate_to_bounds(s, n) is s` exactly when `s` already has `<= n` codepoints
(the `Cow` identity convention: zero allocation, zero marshalling). `max_chars < 0`
and an unrecognized `boundary` both raise `ValueError` before any work runs.

```python
tors.truncate_to_bounds("cats are cute", 9)
# "cats are"
tors.truncate_to_bounds("One. Two. Three.", 10, boundary="sentence")
# "One. Two."
```

**Async**: `await tors.aio.truncate_to_bounds(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.truncate_ellipsis`

```python
def truncate_ellipsis(text: str, max_chars: int) -> str: ...
```

The DB-column truncation shape: hard cut to at most `max_chars` codepoints plus
a U+2026 `…` marker, one GIL-released native pass, never mid-grapheme-cluster.
Unlike `truncate_to_bounds` there is no word/sentence awareness — a storage
bound is positional, not semantic, and the marker tells the reader the value
continues.

At most `max_chars - 1` codepoints are kept plus the one-codepoint marker, so
the result never exceeds `max_chars` (it falls short when cluster backoff
requires it: a cut landing inside a combining sequence, ZWJ emoji chain, or
regional-indicator flag pair snaps back past the whole cluster rather than
splitting it). On plain text the stored length is exactly the bound. No
trailing-whitespace trim: the cut is positional.

`tors.truncate_ellipsis(s, n) is s` exactly when `s` already has `<= n`
codepoints. `max_chars == 0` yields `""` (no room for even the marker — the
naive `value[:0] + "…"` spelling answers `"…"` here, exceeding a zero bound);
`max_chars < 0` raises `ValueError` before any work runs.

```python
tors.truncate_ellipsis("hello world", 6)
# "hello…"
```

**Async**: `await tors.aio.truncate_ellipsis(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

## `tors.is_grounded`

```python
def is_grounded(
    claim: str,
    source: str,
    *,
    fuzzy: bool = False,
    threshold: float = 0.85,
    deadline_ms: float | None = None,
) -> bool: ...
```

Checks whether `claim` is grounded in `source`: a LEXICAL check, not a semantic/NLI
one; be precise about that boundary, this is not a hallucination-detection model.

`fuzzy=False` (the default) is exact substring containment: the `memchr` crate's
SIMD-skipped two-way search (`memmem`, already a dependency), a byte-level find that is
UTF-8-boundary-safe by construction. `fuzzy=True` compares `claim` against
overlapping same-length windows of `source` (stride `claim`'s length / 2) using the
only diffing engine already in the crate (the `similar` Myers engine backing
`diff_opcodes`), and reports whether the BEST window's difflib-style ratio (`2 *
matched_chars / (len(claim) + len(window))`) reaches `threshold`. `fuzzy=True` is a
superset of `fuzzy=False`: an exact-containment floor runs first, so a claim present
verbatim in `source` is grounded before any windowing (independent of window alignment,
and before `deadline_ms` applies — a verbatim substring never times out). The windowed
ratio is consulted only when there is no exact match.

The floor guarantees the verbatim case unconditionally; near matches get a bounded
guarantee band instead of raw window luck: a same-length source region whose aligned
ratio is `r` is detected at ANY offset whenever `r >= max(0.75, threshold + 1/32)` —
a bounded refinement pass re-scans the best coarse windows at a fine stride, a
constant budget on top of the linear scan. One substitution in a 9+ character claim
clears the `0.85` default wherever it sits. Below `r = 0.75` detection is
best-effort (the recall floor of the DoS windowing), and a genuine region can be
evicted from the 64 refinement candidates by adversarial decoy text scoring higher —
the regime `deadline_ms` exists for (both limits are pinned in `tests/test_grounded.py`).

Windowing, rather than one whole-string diff of `claim` against all of `source`, is
DoS discipline: the realistic RAG-grounding shape is a short claim against a
long retrieved passage, so bounding each diff's operands to roughly `claim`'s length
keeps the total work close to linear in `source`'s length instead of the O(source ×
claim) a single unwindowed diff would cost — and windows slide through one reusable
O(claim)-sized buffer, so a 12 MiB passage costs kilobytes rather than a
whole-source char vector, and an early exit stops consuming input mid-source.
`deadline_ms` (only accepted, and only
meaningful, when `fuzzy=True`) bounds the WHOLE scan on top of that, the same
discretionary escape hatch `diff_opcodes`'s `deadline_ms` already has — checked after every window
diff, coarse and refinement alike: `TimeoutError`
on expiry naming the elapsed cost and the deadline, a positive-finite-or-`None`
precondition validated before any work runs. Even a single very large window's own
Myers search is itself deadline-bounded (`similar`'s `capture_diff_slices_deadline`),
so one pathological window cannot blow through the budget uninterrupted between checks.

An empty `claim` is vacuously grounded in anything on both paths. `threshold` must be
in `[0.0, 1.0]`.

```python
tors.is_grounded("cat", "the cat sat")
# True
tors.is_grounded("the cat sat", "Lorem ipsum. The cats sit on mats today.",
                  fuzzy=True, threshold=0.6)
# True
```

## `tors.similarity_ratio` / `tors.get_close_matches`

```python
def similarity_ratio(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...

def get_close_matches(
    word: str,
    possibilities: list[str],
    n: int = 3,
    cutoff: float = 0.6,
    *,
    deadline_ms: float | None = None,
) -> list[str]: ...
```

`difflib.SequenceMatcher(None, a, b).ratio()` and `difflib.get_close_matches()`'s
shapes at native speed, over the same Myers engine `diff_opcodes` uses.
`similarity_ratio` is `2.0 * M / T` (`T = len(a) + len(b)`, both in CHARACTER units)
with `M` the matched-character total over the Myers equal-ops, difflib's own formula
but over a DIFFERENT alignment: difflib's `M` comes from its longest-match recursion,
which anchors a match and splits the surrounding change around it, so on
repeated-pattern inputs its `M` can be smaller than the maximal one. The pinned
divergence rows (`tests/test_similarity.py`): `"ppp"` vs `"pwpp"`, difflib `4/7`
(its anchored `"pp"` splits the insert, `M = 2`) vs tors `6/7` (`M = 3 = LCS`);
and `"qpqpq"` vs `"qpwqpq"`, difflib `6/11` (the anchored ROTATED equal block
`"qpq"`, a non-minimal insert+delete split, `M = 3`) vs tors `10/11` (`M = 5 =
LCS`). tors's `M` is always maximal: `M == LCS(a,
b)` exactly, the minimal-edit-script consequence of the Myers engine, so the two
ratios agree exactly wherever the alignment is FORCED (identical operands, empty
pairs, disjoint alphabets, pure insert/delete with differing flanks) and are BOTH
valid but may diverge on repeated-flank contexts. difflib's anchored `M` is also
direction-dependent (`similarity_ratio` is symmetric; difflib's `ratio()` is not, in
general). `("", "")` is `1.0`, the convention both engines share.

`get_close_matches` keeps every candidate scoring `similarity_ratio(candidate, word)
>= cutoff` and returns the top `n` sorted by score descending, then by the candidate
STRING descending: `heapq.nlargest`'s tuple order, the stdlib's own tie-break (`ac`
sorts after `ca`, so `get_close_matches("ab", ["ac", "ca"], 2, 0.5)` returns `["ca",
"ac"]`). Returned elements are the ORIGINAL candidate objects, not copies. `n <= 0`
and a `cutoff` outside `[0.0, 1.0]` raise `ValueError` with difflib's exact
message, the offending value interpolated (`"n must be > 0: 0"`, `"cutoff must
be in [0.0, 1.0]: -0.5"`, measured against the running stdlib on 3.10–3.15;
`n` is taken signed at the pyo3 boundary precisely so every `n <= 0` case lands
in `except ValueError` identically to difflib's own). An empty `possibilities`
list returns `[]`.

`deadline_ms` bounds the whole call: every candidate's diff for `get_close_matches`,
one pair's diff for `similarity_ratio`: under one shared clock; `TimeoutError` on
expiry names the elapsed cost and the deadline, `None` (the default) is unbounded, and
an enormous-but-finite budget saturates to unbounded rather than erroring.

```python
tors.similarity_ratio("kitten", "sitting")
# 0.6153846153846154
tors.get_close_matches("appel", ["ape", "apple", "peach", "puppy"])
# ['apple', 'ape']
```

## `tors.levenshtein` / `tors.jaro` / `tors.jaro_winkler`

```python
def levenshtein(a: str, b: str, *, deadline_ms: float | None = None) -> int: ...
def jaro(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...
def jaro_winkler(a: str, b: str, *, deadline_ms: float | None = None) -> float: ...
```

The classic edit-distance/similarity metrics CPython has no stdlib spelling of
(`difflib`'s ratio is not a metric: see above; every real spelling is third-party).
All three operate on CHARACTER sequences (Rust `char`s, i.e. Unicode scalar values:
the same unit a Python `str` index addresses), so an emoji or a combining-mark
sequence costs what its codepoint count says, not its UTF-8 byte count.

`levenshtein(a, b)` is the unit-cost edit distance (insert/delete/substitute each cost
1) as an `int`: `levenshtein("kitten", "sitting") == 3`. Symmetric; `("", "")` is
`0`; one empty operand is the other's character count. Implemented
as a two-row DP (O(len(b)) space, not a full O(n·m) matrix), so memory stays LINEAR:
not quadratic: in operand size even on adversarial multi-megabyte inputs; identical
operands short-circuit to `0` via a single equality scan, without entering the DP at
all.

`jaro(a, b)` is the Jaro similarity in `[0.0, 1.0]` (higher is more similar): a
flag-based matching window of `max(|a|, |b|)//2 - 1`, transpositions counted over the
matched subsequences, `(m/|a| + m/|b| + (m - t)/m) / 3`. `("", "")` is `1.0`; one
empty operand is `0.0`. `jaro_winkler(a, b)` is `jaro(a, b)` plus a common-prefix
boost: `jaro + l * 0.1 * (1 - jaro)`: applied only when the Jaro score is `> 0.7`,
with `l` the common-prefix length capped at 4 characters; the `0.1` scale, the `4`-char
cap, and the `> 0.7` threshold are the standard convention from the original papers
(also `strsim`'s exact spelling, used as the differential oracle for all three
functions). Known literature vectors: `jaro("MARTHA", "MARHTA") == 17/18 ≈ 0.944`,
`jaro_winkler("MARTHA", "MARHTA") ≈ 0.961`.

All three are O(n·m) worst case; `deadline_ms` bounds the DP/matching pass with a
check once per row (Levenshtein) or per phase/1024 outer steps (Jaro), the same
DoS-discipline shape as `diff_opcodes`'s: `TimeoutError` on expiry naming the elapsed
cost and the deadline, `None` (the default) unbounded, an enormous-but-finite budget
saturating to unbounded.

```python
tors.levenshtein("kitten", "sitting")
# 3
tors.jaro_winkler("MARTHA", "MARHTA")
# 0.9611111111111111
```

## `tors.quote` / `tors.quote_plus` / `tors.unquote` / `tors.unquote_plus`

```python
def quote(text: str, safe: str = "/") -> str: ...
def quote_plus(text: str, safe: str = "") -> str: ...
def unquote(text: str) -> str: ...
def unquote_plus(text: str) -> str: ...
```

`urllib.parse.quote` / `quote_plus` / `unquote` / `unquote_plus`, byte-exact, for `str`
input (the stdlib's `bytes`-in/`encoding=`/`errors=` overloads are out of scope: the
encode side is always strict UTF-8, the decode side always `errors="replace"`). The
stdlib's own spellings are pure Python: a GIL-held regex/loop over the whole string;
these are one `py.detach`ed native pass each.

`quote(text, safe="/")` never percent-encodes ASCII letters, digits, or `_.-~` (the RFC
3986 unreserved set, the stdlib's `_ALWAYS_SAFE`), plus the ASCII members of `safe`;
every other byte of `text`'s UTF-8 encoding becomes `%XX` with UPPERCASE hex. `safe` is
BYTE-level and ASCII-only, exactly like the stdlib's own `safe.encode("ascii", "ignore")`
normalization: a non-ASCII `safe` member is silently dropped (`tors.quote("é", "é") ==
"%C3%A9"`, not `"é"`), and `%` in `safe` is honored like any other byte (stays literal).
`quote_plus(text, safe="")` is NOT "quote, then replace `%20` with `+`": the stdlib
quotes with `" "` APPENDED to `safe` (so a space never encodes at all) and then replaces
every `" "` with `"+"`; the observable difference is a literal `+` in `text`, which
escapes to `%2B` unless the caller puts `+` in `safe` (`tors.quote_plus("a+b") ==
"a%2Bb"`).

`unquote(text)` decodes `%XX` (either hex case) as UTF-8 with `errors="replace"`
(an invalid sequence becomes one or more U+FFFD, CPython's maximal-subpart rule); a `%`
not followed by two hex digits (`%zz`, a trailing `%`, `%e` at end of input) stays
VERBATIM. It also reproduces the stdlib's `_asciire` fragmentation exactly: each maximal
ASCII run is unquoted and UTF-8-decoded INDEPENDENTLY, with non-ASCII segments passed
through verbatim: so a multi-byte escape interrupted by a non-ASCII character is not an
escape at all (`tors.unquote("%Cé3") == "%Cé3"`), and an escape split across an
ASCII/non-ASCII boundary decodes as two independently-replaced fragments
(`tors.unquote("%C3é%A9") == "�é�"`, NOT `"éé"`). `unquote_plus(text)` replaces
every `+` with a space BEFORE unquoting: order is semantics, not an implementation
detail: so an escaped `%2B` survives as a literal `+` while a raw `+` becomes a space
(`tors.unquote_plus("%2B") == "+"`, `tors.unquote_plus("+") == " "`).

On text the call leaves byte-for-byte unchanged, all four return the ORIGINAL input
object: `quote`/`quote_plus` when no byte needs encoding, `unquote` when `text` has no
`%`, `unquote_plus` when `text` has neither `+` nor `%`: CPython's own fast-path idiom,
zero allocation, zero copy, zero marshalling. `unquote`'s borrow is narrower than a plain
equality check would suggest: an input whose every `%` is invalid hex (`"%zz"`) decodes
to an EQUAL but NEW string in the stdlib, not the original object, and `tors.unquote`
matches that residue exactly rather than widening the identity lane past parity.

**Known deviation: lone (unpaired) surrogates.** pyo3's `str` argument extraction
(`PyUnicode_AsUTF8AndSize`) requires the WHOLE input to be valid UTF-8 up front, which a
lone surrogate codepoint never is. `quote`/`quote_plus` are unaffected in practice: the
stdlib's own encode-based implementation raises the identical `UnicodeEncodeError` for
such input, so the two still agree: but `unquote`/`unquote_plus` diverge: CPython's
`unquote` only UTF-8-encodes the ASCII runs it is about to percent-decode and passes
every non-ASCII character (surrogates included) through untouched, so it never raises
for a lone surrogate anywhere in `text`. `tors.unquote`/`tors.unquote_plus` raise
`UnicodeEncodeError` for ANY `text` containing a lone surrogate, even one nowhere near a
`%` escape, because the whole-string extraction fails before the Rust core ever runs.
This is a real, narrow parity gap (native Rust `&str`/`String` cannot represent an
unpaired surrogate at all, so silently downgrading to `errors="replace"`-style
substitution at the boundary would corrupt the character rather than reproduce it):
callers who round-trip `surrogateescape`-decoded text (e.g. from `os.fsdecode`) through
`unquote` should be aware of it.

```python
tors.quote("café/data", "/")           # "caf%C3%A9/data"
tors.quote_plus("a b+c")               # "a+b%2Bc"
tors.unquote("caf%C3%A9%20data")       # "café data"
tors.unquote_plus("a+b%2Bc")           # "a b+c"
```

## `tors.chunk_text`

```python
def chunk_text(
    text: str,
    max_chars: int,
    *,
    overlap: int = 0,
    boundary: Literal["word", "sentence"] = "word",
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_text(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

The context-window/RAG packing primitive: boundary-aware chunking of `text` into
`(start, end)` pairs in Python `str` index (codepoint) units, each chunk at most
`max_chars` codepoints, cut at word or sentence boundaries wherever the budget
allows: `truncate_to_bounds`'s own cut rule (the largest boundary end within the
budget, a grapheme-safe hard cut when a single word/sentence exceeds it) applied
repeatedly, with the whole-text bounds computed once, GIL-released. Every cut is also
grapheme-cluster-safe, so `max_chars` can be exceeded only in the pathological case of
a single grapheme cluster (e.g. an oversized ZWJ emoji chain) wider than the remaining
budget: a covering chunker cannot drop content, so that one chunk goes out past
`max_chars` rather than split the cluster; ordinary text never hits this.

**`overlap=0`** (the default): the ORIGINAL lossless-partition contract, unchanged;
chunks are non-empty, contiguous, strictly increasing, cover `[0, len(text))`, and
joining the slices reproduces the input exactly.

**`overlap > 0`**: each chunk after the first starts `overlap` codepoints before the
previous chunk's end, SNAPPED to the nearest `boundary`, never mid-word or
mid-sentence, so a fact split across a cut is still whole in at least one chunk (the
RAG-retrieval shape). This trades the lossless-join guarantee for genuine shared
content between consecutive chunks; every chunk's own `<= max_chars` and
boundary-safety invariants still hold regardless. `overlap` must be `< max_chars`
(`ValueError` otherwise: an overlap at least as large as the budget means no forward
progress is possible). A chunk shorter than the requested `overlap` silently
degrades to zero overlap for just that one transition rather than stall or violate
the budget, a documented degradation under the one invariant that never breaks:
forward progress (the chunk count can never exceed the codepoint count).

`max_chars < 1` or `overlap < 0` raise `ValueError`; an unrecognized `boundary` raises
`ValueError` (the `truncate_to_bounds` spelling). `chunk_cdc`'s byte-level sibling:
`chunk_text` is the semantic/embedding-pipeline chunker (word/sentence-aware, sized
for a context window), `chunk_cdc` is the storage/sync chunker (content-defined byte
boundaries, sized for dedup). Both compose naturally with `merkle_root`/`merkle_diff`
for integrity-checked chunks on top of either strategy.

The return marshalling is `O(chunks)` 2-tuples of ints: the `word_bounds` list-shape
class, at chunk-count scale rather than segment-count scale.

```python
tors.chunk_text("cats are cute and cats are fun", 12)
# [(0, 8), (8, 17), (17, 26), (26, 30)]
tors.chunk_text("cats are cute and cats are fun", 12, overlap=3)
# [(0, 8), (5, 17), (14, 26), (23, 30)]
```

## `tors.chunk_text_iter`

```python
def chunk_text_iter(
    text: str, max_chars: int, *, overlap: int = 0, boundary: Literal["word", "sentence"] = "word"
) -> Iterator[tuple[int, int]]: ...
```

`chunk_text`'s streaming twin, the `word_bounds`/`word_bounds_iter` shape: the whole
scan runs once under `py.detach` at iterator construction, and each `__next__` holds
the GIL only to build one 2-tuple, rather than marshalling the whole result into a
`list` under one GIL hold. Same sequence, same argument contract as `chunk_text`:
worth reaching for over the list API once a document chunks into the hundreds of
thousands of pieces, where the marshalling cost dominates.

```python
list(tors.chunk_text_iter("cats are cute and cats are fun", 12))
# [(0, 8), (8, 17), (17, 26), (26, 30)]
```

## `tors.chunk_by_words`

```python
def chunk_by_words(text: str, words_per_chunk: int, *, overlap: int = 0) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_by_words(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

The unit-count twin of `chunk_text`: instead of a character budget, each chunk spans
exactly `words_per_chunk` consecutive WORD TOKENS, not `word_bounds`'
raw segment count. `word_bounds` follows UAX #29 exactly, which gives an inter-word
space run its own segment (`"one two"` is three segments: `"one"`, `" "`, `"two"`, the
same convention `word_count` already carries): grouping *raw* segments here would
silently mean "`words_per_chunk` roughly halved" for ordinary space-separated prose,
the opposite of what a caller reaching for `words_per_chunk=100` (a "~100 word chunk"
for an embedding budget) wants. So this filters to segments carrying at least one
non-whitespace codepoint FIRST, and only then windows over what remains: a "word" is
a real token, and whitespace between two tokens *inside* one chunk still rides along
naturally (each chunk's span is a contiguous slice of the original text between two
real absolute offsets). `(start, end)` codepoint offsets span the first included word
token's start through the last included token's end: NOT through any trailing
whitespace after it, so unlike `chunk_text`'s covering-partition contract,
non-overlapping chunks here are not necessarily contiguous. The final chunk may hold
fewer than `words_per_chunk` tokens when the total doesn't divide evenly. `overlap`
WORDS repeat at the start of the next chunk. Empty text, or text with no word tokens
at all, returns `[]`.

`words_per_chunk < 1` or `overlap < 0` raise `ValueError`; `overlap >= words_per_chunk`
raises `ValueError`: the chunk stride is `words_per_chunk - overlap` tokens, and
unlike `chunk_text`'s character-granularity overlap this stride is always `>= 1` by
construction once validated, so forward progress needs no runtime fallback.

```python
tors.chunk_by_words("one two three four five six seven", 3)
# [(0, 13), (14, 27), (28, 33)]
tors.chunk_by_words("one two three four five six seven", 3, overlap=1)
# [(0, 13), (8, 23), (19, 33)]
```

## `tors.chunk_by_words_iter`

```python
def chunk_by_words_iter(text: str, words_per_chunk: int, *, overlap: int = 0) -> Iterator[tuple[int, int]]: ...
```

`chunk_by_words`' streaming twin, the same `chunk_text_iter` shape: one detached
whole-text pass at construction, one 2-tuple per `__next__`, identical sequence to
the list API.

```python
list(tors.chunk_by_words_iter("one two three four five six seven", 3))
# [(0, 13), (14, 27), (28, 33)]
```

## `tors.chunk_by_sentences`

```python
def chunk_by_sentences(text: str, sentences_per_chunk: int, *, overlap: int = 0) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_by_sentences(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

`chunk_by_words`' sentence-count twin (`sentence_bounds`'s UAX #29 segmenter): each
chunk spans `sentences_per_chunk` consecutive sentence segments, `overlap` SENTENCES
repeated. Same argument contract, same empty-input answer, same
forward-progress-by-construction guarantee as `chunk_by_words`.

```python
tors.chunk_by_sentences("One. Two. Three. Four. Five.", 2)
# [(0, 10), (10, 23), (23, 28)]
```

## `tors.chunk_by_sentences_iter`

```python
def chunk_by_sentences_iter(text: str, sentences_per_chunk: int, *, overlap: int = 0) -> Iterator[tuple[int, int]]: ...
```

`chunk_by_sentences`' streaming twin, the same `chunk_text_iter` shape.

```python
list(tors.chunk_by_sentences_iter("One. Two. Three. Four. Five.", 2))
# [(0, 10), (10, 23), (23, 28)]
```

## `tors.chunk_by_paragraphs`

```python
def chunk_by_paragraphs(text: str, paragraphs_per_chunk: int, *, overlap: int = 0) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_by_paragraphs(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

`chunk_by_words`/`chunk_by_sentences`'s paragraph-count twin: each chunk spans
`paragraphs_per_chunk` consecutive paragraphs, `overlap` PARAGRAPHS repeated. A
paragraph boundary here is a run of 2+ consecutive newlines (`\r\n` counts as one
unit, matching `tors.normalize`'s own CR/CRLF folding): the same "2+ newlines
survive as the paragraph gap" convention `normalize`'s own pipeline already uses (it
collapses 3+ down to exactly 2, never below). **This is a heuristic, not a Unicode
Standard segmentation** (there is no UAX for paragraphs, unlike UAX #29 for
words/sentences): a single `\n` is ordinary content, not a break. A
leading or trailing blank-line run is trimmed rather than emitted as an empty
paragraph; text with no qualifying run at all is one paragraph. Same argument
contract, same empty-input answer, same forward-progress-by-construction guarantee
as `chunk_by_words`/`chunk_by_sentences`.

No retrieval or LLM-quality claim is made for any chunking strategy in this family:
tors guarantees the mechanical contract (correct boundaries, genuine overlap, the
right knobs), not an outcome it doesn't control.

```python
tors.chunk_by_paragraphs("First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.", 2)
# [(0, 45), (47, 68)]
```

## `tors.chunk_hierarchical`

```python
def chunk_hierarchical(text: str, max_chars: int, separators: list[str] | None = None, *, overlap: int = 0) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_hierarchical(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

Priority-ordered fallback chunking: the `chunk_text`/`chunk_by_*` family's
fourth shape, and the pattern LangChain's `RecursiveCharacterTextSplitter`
popularized: a list of separator LEVELS, coarsest first, tried in order for
each chunk, falling back to the next level only when the coarser one has no
in-budget cut over the current window.

`separators=None` (the default) uses tors's OWN accurate hierarchy:
paragraph → sentence → word → a grapheme-safe raw cut, always the final,
unconditional fallback (this never fails to produce a chunk); it reuses the
same UAX #29 segmenters `chunk_by_sentences`/`chunk_by_words` do, rather
than LangChain's own naive literal guesses (`"\n\n"`, `". "`, `" "`).
`separators=[...]` is a caller-supplied list of LITERAL strings (**not
regex**, a documented scope line: literals are LangChain's own default
too, cover the motivating markdown-header case completely, and avoid
reopening the regex-semantics question `re` support was already declined
over), coarsest first, e.g. `["\n## ", "\n\n", ". ", " "]` for
markdown-header-aware chunking. A custom list REPLACES the default
hierarchy for the levels it specifies, but the grapheme-safe raw cut is
still always appended as the final fallback regardless; unlike LangChain,
no trailing `""` sentinel is required (one is accepted and ignored if
supplied).

**Unlike `chunk_text`, this is NOT a lossless covering partition**: at
every level except the raw cut, the separator itself is DROPPED between
chunks: the chunk ends where the separator starts, the next chunk begins
where it ends, the same convention `chunk_by_paragraphs` already applies
to blank-line runs. A caller splitting ON a marker wants it gone, not
duplicated.

`overlap` snaps the next chunk's start backward to the nearest GRAPHEME
boundary at or before the target, not necessarily a semantic
paragraph/sentence/word boundary the way `chunk_text_overlapping`'s
single-hierarchy overlap snap is (a documented simplification of the
general multi-level case). A target at or before the chunk's own start
silently degrades to zero overlap for just that one transition, the same
snap-collapse `chunk_text` already applies.

`max_chars < 1` or `overlap < 0` raise `ValueError`; `overlap >= max_chars`
raises `ValueError`. Empty `text` returns `[]`. An empty `separators` list
is legal and skips straight to the raw-cut fallback for every chunk. Every
level's cut candidates are additionally grapheme-cluster-safe (the same
Thai SARA AM / combining-mark fix applied crate-wide), including custom
literal separators.

No retrieval or LLM-quality claim is made for any chunking strategy in
this family: tors guarantees the mechanical contract (correct boundaries,
genuine overlap, the right knobs), not an outcome it doesn't control. This
is additive: `chunk_text`/`chunk_by_words`/`chunk_by_sentences`/
`chunk_by_paragraphs` remain the right choice for the common case;
`chunk_hierarchical` is for custom, format-aware, or multi-granularity
needs those simpler functions can't express.

```python
md = "# Title\nSome intro text here.\n## Section\nMore content in this section."
tors.chunk_hierarchical(md, 60, ["\n## ", "\n\n", ". ", " "])
# [(0, 29), (33, 70)]  ->  "# Title\nSome intro text here." / "Section\nMore content in this section."
```

## `tors.chunk_cdc`

```python
def chunk_cdc(
    data: bytes, *, min_size: int = 4096, avg_size: int = 16384, max_size: int = 65534
) -> list[tuple[int, int]]: ...
```

**Async**: `await tors.aio.chunk_cdc(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

FastCDC 2020 content-defined chunking, one GIL-released native pass: `(start, end)`
**byte** spans (not codepoints: unlike every other segmentation function here, this
operates on raw bytes, not text) partitioning `data` exactly: `end` exclusive, the last
span's `end == len(data)`, no gaps or overlaps.

Content-defined chunking's whole point over fixed-size splitting: cut points are chosen
by local content (a rolling hash over a sliding window), not a fixed stride, so a small
edit near the start of `data` only perturbs the 1-2 chunks nearest the edit; every
chunk further away reappears unchanged, just shifted by the edit's byte delta. Fixed-size
chunking has no such property: an insertion reshuffles every boundary after it. This is
the natural upstream of `tors.merkle_root`/`tors.merkle_diff`'s `list[bytes]` argument
for a byte-level dedup/incremental-sync pipeline (chunk, then hash each chunk into the
tree): same framing `tors.finalize`'s normalize-then-hash dedupe gate already
established at the whole-document level, extended to sub-document granularity.

Empty input returns `[]`. Input shorter than `min_size` returns exactly one chunk
covering the whole input (the underlying algorithm's own documented special case).
Deterministic: the same bytes at the same parameters always cut at the same offsets.

`min_size`/`avg_size`/`max_size` must satisfy the wrapped `fastcdc` crate's own
documented bounds: each even; `min_size` in `[64, 1_048_576]`, `avg_size` in `[256,
4_194_304]`, `max_size` in `[1024, 16_777_216]`, and `min_size <= avg_size <=
max_size`, checked and raised as `ValueError` before any chunking runs. The crate itself
only `debug_assert!`s these bounds, a no-op in a release build, so an out-of-range call
would otherwise silently misbehave rather than error: tors validates them itself at the
argument boundary instead, the same discipline `is_grounded`'s `threshold` and
`truncate_to_bounds`' `max_chars` already apply. Defaults are the crate's own documented
example values, not independently chosen.

```python
tors.chunk_cdc(b"hello world " * 10_000)
# [(0, 65534), (65534, 120000)]
```

## `tors.merkle_root`

```python
def merkle_root(chunks: list[bytes]) -> str: ...
```

A domain-separated SHA-256 Merkle root over `chunks`, as lowercase hex, one
GIL-released native pass wrapping the `rs_merkle` crate. Leaves hash as
`SHA-256(0x00 || chunk)`; internal (two-child) nodes hash as `SHA-256(0x01 ||
left || right)`: the RFC 6962 / Certificate Transparency convention.

**Why this is tors's own hasher, not the wrapped crate's built-in one.**
`rs_merkle`'s default `Sha256Algorithm` has NO domain separation: its
`hash()` is plain undifferentiated `SHA-256(data)`, and its default
`concat_and_hash` feeds `SHA-256(left || right)` through that same
function; verified directly against the crate's vendored source
(`hasher.rs`'s default `concat_and_hash`, `algorithms/sha256.rs`'s `hash`,
rs_merkle 1.5.0), not just its docs. With no domain byte anywhere, a leaf
hash and an internal-node hash live in the same output space: exactly the
ambiguity behind **CVE-2012-2459**, the Bitcoin Merkle-tree bug class where a
forged proof can present an internal node's hash as though it were some
leaf's digest. `merkle_root`/`merkle_diff` don't expose proof generation
yet, but the hash SCHEME is part of the root's output contract from v1
regardless: roots are meant to be computed once and compared/stored across
calls, and changing the scheme later would silently change every
previously-computed root. It needs to be correct now, not patched in when
proof generation is added.

An unpaired left node at any layer is **promoted** unchanged to the next
layer (rs_merkle's own default `concat_and_hash`, its `None => *left` arm),
never duplicated against itself. Duplication is Bitcoin's original
convention and the actual mechanism CVE-2012-2459 exploited: two
differently-shaped chunk lists (one a duplicate-padded version of a shorter
one) could otherwise produce the SAME root; promotion is the standard
mitigation and matches RFC 6962. A single chunk's root is just its own leaf
hash: no internal node is built for it.

Deterministic and order-sensitive: the same chunks in the same order always
produce the same root; reordering changes it (this is a Merkle tree, not an
order-insensitive digest/set hash). An empty `chunks` list raises
`ValueError("root of no chunks")`: no non-arbitrary root value exists for
it, and returning some fixed sentinel hash risks being mistaken for a real
chunk's digest by a caller comparing roots. A non-`list` argument or a
non-`bytes` list entry raises `TypeError`.

```python
tors.merkle_root([b"a", b"b", b"c"])
# "36642e73...e6c021ec1"  (hex-encoded SHA-256, RFC 6962-style domain-separated)
```

## `tors.merkle_diff`

```python
def merkle_diff(chunks_a: list[bytes], chunks_b: list[bytes]) -> list[int]: ...
```

Indices where `chunks_a[i] != chunks_b[i]`, one GIL-released native pass.
Comparison is over chunk DIGESTS, the same `0x00`-prefixed leaf hash
`merkle_root` uses at a fixed 32-byte cost per index, rather than raw chunk
contents. Every index at or beyond the shorter list's length is reported:
there is no counterpart chunk to compare against there, so a length
mismatch is, in full, "differs at every trailing index," not a partial
answer. Two empty lists diff to `[]`.

This does **not** walk a tree. With both chunk lists held
locally as random-access arrays, hashing each chunk once already answers
"does chunk `i` differ" in O(1) per index afterward: a flat scan over the
digest arrays does exactly the work a tree walk would, without needing one.
A Merkle tree's "skip identical subtrees without transferring them" payoff
matters when the comparison itself is expensive to perform per index (e.g.
over a network); not here, where the O(n) hashing pass already is the
entire cost. `merkle_root` still builds a real tree: that's its actual
job; `merkle_diff` just doesn't need one to answer this particular
question. Argument contract matches `merkle_root`'s: a non-`list` argument
or a non-`bytes` list entry raises `TypeError`.

```python
tors.merkle_diff([b"a", b"b", b"c"], [b"a", b"X", b"c"])
# [1]
tors.merkle_diff([b"a", b"b"], [b"a", b"b", b"c", b"d"])
# [2, 3]  — every trailing index beyond the shorter list's length
```

## `tors.simhash64`

```python
def simhash64(text: str) -> int: ...
```

A 64-bit SimHash fingerprint of `text`, one GIL-released native pass: the
fuzzy near-duplicate gate that sits alongside `tors.finalize`'s exact
SHA-256 gate. Charikar's weighted-bit-voting construction (the near-web-scale
near-duplicate-detection shape Manku, Jain, and Das Sarma built at Google,
WWW 2007): `text` is tokenized into words via the SAME UAX #29 word
segmentation `tors.word_bounds` drives (`split_word_bounds`, its sibling
spelling over the same tables), skipping any segment that is entirely
whitespace; each token's UTF-8 bytes are hashed with a deterministic
FNV-1a-64; then, for each of the 64 bit positions, every token votes +1 if
its hash has that bit set and −1 if it does not, and the output bit is 1
wherever the vote sums positive (ties, including the zero-token case,
resolve to 0).

**Why FNV-1a and not `std`'s `DefaultHasher`.** A dedupe fingerprint must be
stable ACROSS PROCESSES and machines: `DefaultHasher` is seeded per process
(`RandomState`), so a fingerprint it produced would silently change between
runs, breaking any cross-run/cross-machine dedupe built on it. FNV-1a is
deterministic forever and adequate for a VOTING hash: it only needs to
spread tokens reasonably uniformly across the 64 bit positions, not resist
adversarial collisions; a collision between two distinct tokens merely
blurs one vote among 64 counters.

**What this answers that `finalize` cannot.** `finalize`'s SHA-256 tail and
`merkle_root` answer "is this text byte-identical": a single changed comma
already fails that gate. `simhash64` answers "is this text NEARLY the
same": Hamming distance `(a ^ b).bit_count()` (a one-liner at the call
site, which is why this returns the raw `int` rather than shipping a
redundant distance function) grows slowly with edit distance, so
near-duplicates cluster within a handful of differing bits while unrelated
texts sit far apart. The realistic pipeline runs both gates off one store:
exact dupes at Hamming distance 0 via `finalize`'s hash, near-dupes at
small Hamming distance via `simhash64`, everything else far apart.

**Bag of words: order does not matter.** The vote is over the MULTISET of
tokens, not their sequence: `tors.simhash64("the quick brown fox")` and
`tors.simhash64("fox brown quick the")` fingerprint IDENTICALLY. A repeated
token votes once per occurrence, so frequency is still part of the bag:
appending one more `"the"` to a sentence that already has several is a
different multiset and (usually) a different fingerprint.

**Tokenization is UAX #29, not a whitespace split**: the same segmentation
`word_bounds` uses, not an ad-hoc `str.split()`. This matters for
scriptio-continua text (CJK, Thai, and similar scripts with no spaces
between words): the segmenter still finds word boundaries inside a
space-free run rather than treating it as one giant token. A whitespace-only
`text` has no word tokens (the WSegSpace rule joins a whitespace run into
one segment, which the tokenizer then skips as not-a-word) and fingerprints
to `0`, same as empty text.

**Limitations: read before deploying a threshold.** SimHash is
**not cryptographic and not collision-resistant**: FNV-1a is a fast,
uniformly-spreading voting hash, not a security primitive, and two
unrelated documents can coincidentally land close together, especially on
short text. There is **no universal near-duplicate cutoff**: the distance
a "same document, small edit" pair sits at scales with document length
(short texts have thin per-bit vote margins, on the order of the square
root of the token count, so the same edit flips more bits), and the
unrelated floor depends on vocabulary overlap. The measured anchors
(pinned in `src/simhash_impl.rs`'s test suite and mirrored in
`tests/test_simhash.py`): over a 95-word document, every single-word
swap/drop/insert moves the fingerprint at most 4 bits; over 8-12-word
sentences, the identical class of edit moves up to 14 bits; unrelated
sentence pairs measured as close as 23 bits apart in the same battery.
Calibrate any "probable near-duplicate" threshold per deployment against
known near-dup and known-far pairs: do not import a textbook rule of
thumb unchanged.

Deterministic across processes, versions, and machines (FNV-1a has no
per-process seed). `O(n)` in the length of `text` with `O(1)` extra memory
beyond the 64 vote counters: no intermediate token list is materialized.

```python
tors.simhash64("the quick brown fox jumps over the lazy dog")
# 14607312263354641902
tors.simhash64("the quick brown fox jumps over the lazy dog!")  # one edit
# 14601682762745638348 — Hamming distance 6, not ~32 (unrelated-text scale)
tors.simhash64("")
# 0
```

## `tors.simhash128`

```python
def simhash128(text: str) -> int: ...
```

The 128-bit spelling of `tors.simhash64`: identical tokenization, identical
vote (each token's FNV-1a hash votes ±1 per bit, positive sum sets the bit),
run at 128 bits instead of 64: genuinely twice the bit positions, not a
64-bit fingerprint zero-extended into a wider int (the FNV-1a offset basis
and prime are the real 128-bit constants, and every one of the 128 bits gets
its own independent vote). Use it for corpora where a 64-bit fingerprint's
near-duplicate band and unrelated floor sit too close together to separate
reliably.

Measured by the same battery as `simhash64` and pinned in
`src/simhash_impl.rs`'s test suite: the unrelated floor widens from 23 bits
at 64 bits to 40 at 128, while the near-duplicate bands grow only
sublinearly (95-word document scale: 3 bits worst case versus the 64-bit
4; 8-12-word sentence scale: 20 versus 14). The wider width buys more
separation between the near-duplicate band and the unrelated floor, which is
what a corpus whose 64-bit bands overlap needs. The same calibration caveat
applies: there is no universal cutoff, and thresholds must be measured
against known near-dup and known-far pairs for the corpus at hand.

Same contract otherwise: deterministic across processes and machines
(FNV-1a has no per-process seed), order-independent (a bag-of-words vote),
empty or whitespace-only text fingerprints to `0`, and `(a ^ b).bit_count()`
at the call site gives the Hamming distance.

```python
tors.simhash128("the quick brown fox jumps over the lazy dog")
# 317101931942163558849153286541522090150
tors.simhash128("")
# 0
```

## `tors.tf_idf`

```python
def tf_idf(
    corpus: list[str],
    *,
    strip_accents: bool = False,
    stemmer: str | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
) -> list[list[tuple[str, float]]]: ...
```

**Async**: `await tors.aio.tf_idf(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

Stateless TF-IDF over `corpus`, one GIL-released native pass: no
vocabulary/vectorizer object persists between calls; every call scores
fresh over exactly the documents given. Fills a real gap: Python's stdlib
has no TF-IDF at all, and `scikit-learn`'s `TfidfVectorizer` pulls in
numpy/scipy for a lightweight pipeline that just wants keyword weighting or
document similarity. Every TF-IDF crate on crates.io is stale or effectively
abandoned, and the math is a few dozen lines with no ML machinery: hand-rolled
directly, the same surgical tradition as `html_unescape`/`extract_code_blocks`,
not a dependency pull.

**Tokenization**: UAX #29 word segments (`tors.word_bounds`'s own tables),
restricted to segments carrying at least one non-whitespace codepoint: a
"term" is a real token, not a raw `word_bounds` segment (which gives
inter-word whitespace its own segment). Terms are lowercased with Rust's
Unicode-correct `str::to_lowercase` (not an ASCII-only fold) before
counting: `"Cat"` and `"cat"` are the same term.

**TF** (per document `d`, term `t`): the RAW count of `t` in `d`, not
length-normalized. A caller wanting `tf / len(d)` divides the returned raw
count themselves; the raw count is the more broadly reusable number (a
length-normalized score would silently discard the total-count information
some callers want directly).

**IDF** (term `t`, corpus size `N`, document frequency `df(t)` = number of
documents containing `t` at least once): the SMOOTHED formula
`ln((1 + N) / (1 + df(t))) + 1`: scikit-learn's own `smooth_idf=True`
default (as if one extra document existed containing every term exactly
once), not the textbook `ln(N / df(t))`. The textbook formula
gives a term appearing in EVERY document an IDF of exactly `ln(1) = 0`, so
its score is `0` regardless of how often it occurs: an unhelpfully sharp
cliff for exactly the "practically universal term" case a caller most wants
a small-but-nonzero weight for. The smoothed form stays strictly positive
there (`ln((1+N)/(1+N)) + 1 = 1` exactly, for the universal-term case
`df(t) = N`) while still monotonically favoring rarer terms.

**score**(t, d) = `tf(t, d) * idf(t)`.

**Output is SPARSE**: one `(term, score)` list per input document,
alphabetically sorted, holding only that document's own terms, never a
vocabulary-size-by-corpus-size dense structure (wasteful for anything but a
tiny shared vocabulary). An empty corpus returns `[]`. An empty-string
document returns `[]` at its position: the output always has exactly
`len(corpus)` entries, position-matched to the input. A non-`list` argument
or a non-`str` entry raises `TypeError`; a lone surrogate raises
`UnicodeEncodeError` at the argument boundary, the same str-in convention
every other function here documents.

**`strip_accents=True`** NFD-decomposes each token (`tors.nfd`'s own
algorithm) and drops every combining-mark codepoint before scoring:
`"café"` and `"cafe"` become the same term. This is unconditional: it does
NOT replicate a real bug in scikit-learn's own `strip_accents_unicode`
(gh-15087), which short-circuits and silently skips stripping when a token
arrives already NFD-decomposed (e.g. `"e"` + a combining accent rather than
precomposed `"é"`), tors always strips regardless of whether decomposition
itself was a no-op. NFD, not NFKD: NFKD's extra compatibility decomposition
would also touch ligatures/width variants (`"ﬁ"` → `"fi"`), which
accent-folding shouldn't. Default `False`: accents are preserved unless
asked to fold them.

A token made ENTIRELY of combining marks (a bare accent with no base
letter, e.g. from already-decomposed input) strips down to the empty
string; such tokens are dropped, never counted as a `""` term. This is
the same structural exclusion `scikit-learn`'s default `token_pattern`
(`r"(?u)\b\w\w+\b"`, which can never match zero characters) achieves for
`TfidfVectorizer`, applied here after folding rather than via a regex over
the raw text.

**`stemmer`** names a Snowball algorithm (`"english"`, `"french"`,
`"german"`, ... 18 languages via the `rust-stemmers` crate: an
unrecognized name raises `ValueError` naming every valid choice), applied
after lowercasing/accent-folding: `"running"`/`"runs"`/`"runner"` all stem
toward `"run"`. Default `None`: no stemming. Full lemmatization is
explicitly out of scope: it needs a per-language dictionary or a
POS-tagging model, not an algorithm, which breaks tors's no-external-model
posture (the same boundary that kept schema-aware JSON/YAML coercion out of
this crate). Stemming is cruder: it can't distinguish "better" the
comparative from "better" the verb, but it's correct, deterministic, and
dependency-light.

**`lemma_dict`** is a caller-supplied `word -> lemma` map, applied LAST
(after any stemming): the fully-folded token is looked up, and its mapped
value REPLACES it if present, else the folded token is kept as-is. tors
does not bundle a lemma dictionary: full lemmatization needs a
per-language dataset or a POS model, the same "no external model"
boundary that kept stemming's own decision above. `lemma_dict` is the
mechanism, not the data: the same shape `replace_many` already takes a
caller-supplied replacement map instead of a bundled one. Combining
`stemmer` and `lemma_dict` together is unusual but well-defined, not an
error: the dict is consulted on the ALREADY-stemmed form. A non-`dict`
argument, or one with a non-`str` key/value, raises `TypeError`. Default
`None`: no substitution.

**Loading a lemma dictionary.** `lemma_dict` takes a plain `dict[str, str]`,
so any source you can turn into one works. Three common ones:

- **spaCy's lemma lookup tables** ship as JSON, already `word -> lemma`:
  `lemma_dict = json.load(open("spacy-lookups-data/spacy_lookups_data/data/en_lemma_lookup.json"))`.
- **The Lemmatization Lists project** (`michmech/lemmatization-lists`) ships
  TSV in the opposite direction, `lemma<TAB>word`: invert each row:
  `lemma_dict = {word: lemma for lemma, word in (line.split("\t") for line in open("lemmatization-en.txt"))}`.
- **NLTK's WordNet exception lists** (`nltk_data/corpora/wordnet/*.exc`, one
  file per part of speech: `adj.exc`, `adv.exc`, `noun.exc`, `verb.exc`)
  are already `word lemma` pairs, one per line, space-separated:
  `lemma_dict = {k: v for path in exc_paths for line in open(path) for k, v in [line.split()[:2]]}`.
  Each `.exc` file carries no part-of-speech tag of its own, so merging all
  four into one `lemma_dict` means a word ambiguous across parts of speech
  (a verb and a noun spelled the same, lemmatized differently by each)
  collapses to whichever file's entry was merged in last; if that matters
  for your corpus, keep one `lemma_dict` per part of speech instead and pass
  each into its own `tf_idf`/`bm25_rank`/`apply_pipeline` call.

**`lemma_dict`'s real cost, measured, and its fix.** A raw
`dict[str, str]` marshals the WHOLE Python `dict` into a Rust
`HashMap<String, String>` fresh on every call. For a small map (a handful
of entries) that cost is negligible. For a realistically-sized lemma table
(spaCy's own English lookup data is tens of thousands of entries), it is
not. Measured on a 12-core dev box, a 20,000-entry `lemma_dict` costs
roughly 1.4ms of marshalling PER CALL, independent of how much text that
call processes. Called once over a large batch, that cost amortizes away.
Called repeatedly (once per short text, the shape a naive loop reaches
for), that fixed cost dominates and can make `apply_pipeline`/
`tf_idf`/`bm25_rank` measurably SLOWER than an equivalent idiomatic Python
loop (`tools/bench_lemma_dict.py` measured roughly 10-300x slower at small
per-call batch sizes with a 20,000-entry map).

`tors.CompiledLemmaDict` is the fix: build the `HashMap` once, reuse it
across every call. `lemma_dict` accepts either a raw `dict[str, str]` (the
per-call cost above) or a `CompiledLemmaDict` (an `Arc::clone` per call
after the one-time build: measured at roughly 57x faster than the raw-
`dict` path for a 20,000-entry map called repeatedly). Building a
`CompiledLemmaDict` still costs the same materialization time; the whole
point is paying it once instead of on every call:

```python
compiled = tors.CompiledLemmaDict(lemma_dict)   # pay the cost once
for batch in many_batches:
    tors.apply_pipeline(batch, lowercase=True, lemma_dict=compiled)  # O(1) per call after
```

`CompiledLemmaDict` is immutable once built and NOT a general
caching mechanism (no identity-keyed cache lives inside tors, silently
reusing a stale mapping if a caller mutated their `dict` between calls:
that footgun is exactly what an explicit, caller-controlled handle avoids).
It does not reopen the case for a stateful pipeline object: it is scoped
to this one parameter, on this one measured cost, the same narrow shape
`re.compile()` has in the stdlib.

**What this is for, and what it is NOT.** A lightweight keyword-weighting
and document-similarity primitive for pipelines that don't want an ML
dependency: surfacing a document's most distinctive terms, or comparing
documents by their score vectors. It is NOT a full NLP pipeline stage: no
lemmatization, no stop-word removal, no n-grams; stemming and accent-folding
are opt-in, everything else stays case-folding only. And, matching this
crate's stated posture throughout: this is a correctness/capability
primitive, not a retrieval- or model-quality promise; whether TF-IDF
weighting helps a particular downstream task is the caller's question to
answer, not a claim made here.

```python
tors.tf_idf(["the cat sat on the mat", "the dog sat on the log", "birds fly in the sky"])
# [[('cat', 1.6931471805599454), ('mat', 1.6931471805599454), ('on', 1.2876820724517808),
#   ('sat', 1.2876820724517808), ('the', 2.0)],
#  [('dog', 1.6931471805599454), ('log', 1.6931471805599454), ('on', 1.2876820724517808),
#   ('sat', 1.2876820724517808), ('the', 2.0)],
#  [('birds', 1.6931471805599454), ('fly', 1.6931471805599454),
#   ('in', 1.6931471805599454), ('sky', 1.6931471805599454),
#   ('the', 1.0)]]
```

## `tors.bm25_rank`

```python
def bm25_rank(
    query: str,
    corpus: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    strip_accents: bool = False,
    stemmer: str | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
) -> list[tuple[int, float]]: ...
```

**Async**: `await tors.aio.bm25_rank(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

Okapi BM25 score for every document in `corpus` against `query`, one
GIL-released native pass: `(index, score)` pairs for EVERY document: no
top-k cutoff baked in, slice/sort the result yourself; sorted by score
descending, ties broken by ascending original index.

**A RERANKING primitive, not a search index.** `bm25_rank` recomputes
corpus statistics from scratch on every call. That is the right shape for
the common RAG pattern of reranking a SMALL, already-retrieved candidate
set (tens to a few hundred documents: a vector-search step's top-k, say)
against one query: no state to manage, composes with the rest of tors's
flat, stateless primitives, cheap enough at that scale to recompute per
call. It is NOT a search engine: a corpus with thousands of
documents queried repeatedly wants a real inverted index built once and
queried many times: recomputing corpus statistics from scratch on every
single query wastes that work every time. For that, reach for a real
search engine (`tantivy` is the mature, dominant choice in Rust); tors
does not build persistent index objects, the same scope line that kept a
Merkle inclusion-proof API out of this crate.

**The formula**: for query `Q` (tokenized to a set of DISTINCT terms: a
repeated query word contributes its IDF once, the standard Robertson/
Spärck-Jones convention) and document `D`:

```text
score(D, Q) = sum over t in Q of IDF(t) * f(t,D) * (k1 + 1)
                                  -----------------------------------
                                  f(t,D) + k1 * (1 - b + b * |D| / avgdl)

IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
```

`N` = corpus size, `n(t)` = number of documents containing `t`, `f(t,D)` =
`t`'s occurrence count in `D`, `|D|` = `D`'s token count, `avgdl` = the
corpus's mean document length. `IDF` is the always-non-negative "+1"
(Lucene-since-2011) variant, not the classic
`ln((N-n(t)+0.5)/(n(t)+0.5))` form, which goes NEGATIVE for a term
appearing in more than half the corpus: a surprising, unwanted answer for
a reranking primitive with no stopword list to filter such terms out
first.

`k1` (>= 0, default 1.5) tunes term-frequency saturation: how much repeat
occurrences of a term keep adding to the score; `b` (in `[0, 1]`, default
0.75) tunes length normalization: how much a longer-than-average document
is penalized. Both are Lucene/Elasticsearch's own defaults. Out-of-range
values raise `ValueError`.

**Tokenization**: the same "real word token, lowercased" convention
`tf_idf` uses (UAX #29 word segments, non-whitespace only). `strip_accents`/
`stemmer`/`lemma_dict` are `tf_idf`'s exact same opt-in knobs (see its docs
for the accent-folding/stemming/lemma-substitution details), applied
IDENTICALLY to `query` and every `corpus` document, since scoring a query
normalized differently from its corpus produces meaningless scores, not
just imprecise ones. All default off, reproducing the original
lowercase-only tokenization exactly.

An empty `corpus` returns `[]`. An empty (or all-whitespace, or
no-real-tokens) `query` scores every document `0.0`: there are no query
terms to accumulate a score over, the correct, unsurprising answer, not an
error. A non-`list` `corpus` or a non-`str` entry raises `TypeError`; a
lone surrogate raises `UnicodeEncodeError` at the argument boundary. A
non-`dict` `lemma_dict`, or one with a non-`str` key/value, raises
`TypeError`.

**No `deadline_ms`**: every deadline-bearing primitive in this crate
protects against adversarial-input superlinear blowup (`levenshtein`/
`jaro`'s O(n·m) DP tables, `similarity_ratio`'s windowed Myers scans).
`bm25_rank` has no such shape: cost is linear in total corpus token count
plus `corpus_size * distinct_query_terms`, both driven directly and
proportionally by the SIZES of the caller's own arguments, not by
adversarial structure within them. A caller already controls the one lever
that bounds the cost (how large a `corpus` they pass).

`tors` makes no claim about retrieval or relevance QUALITY for any
particular corpus or query: BM25 is a well-specified ranking FORMULA,
correctly implemented here, not a model-quality promise.

```python
tors.bm25_rank(
    "quick fox",
    ["the quick brown fox jumps over the lazy dog",
     "a lazy cat sleeps all day",
     "the fox and the dog are friends"],
)
# [(0, 1.3162195220480066), (2, 0.4798180901812613), (1, 0.0)]

tors.bm25_rank("cafe", ["café société", "totally unrelated text"], strip_accents=True)
# [(0, 0.7617001984175222), (1, 0.0)]
```

## `tors.apply_pipeline`

```python
def apply_pipeline(
    texts: list[str],
    *,
    nfd: bool = False,
    lowercase: bool = False,
    strip_accents: bool = False,
    stemmer: str | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
    collapse_whitespace: bool = False,
) -> list[str]: ...
```

**Async**: `await tors.aio.apply_pipeline(...)` runs this under `asyncio.to_thread` so the event loop stays responsive across the call — see the README's [Async use](https://github.com/AZX-PBC-OSS/tors#async-use) section.

A stateless, general-purpose batch text preprocessor: every requested step
fused into ONE GIL-released native pass over the WHOLE `texts` list.

**No pipeline object for the pipeline itself: pure function composition.**
A `re.compile()`-style compiled-pipeline handle for the WHOLE pipeline
(build once, `.apply()` many times) was considered and ruled out: tors
does not build persistent Rust-side state, the same scope line that kept
a Merkle inclusion-proof API and a real search index out of this crate.
Every call re-describes and re-applies its steps fresh. `lemma_dict` is
the one narrow, measured exception: materializing a large caller-supplied
dict into a Rust `HashMap` is linear and, for a realistic multi-thousand-
entry lemma table, costs enough per call to matter at small batch sizes
(see `tors.CompiledLemmaDict` above): that one measured cost is why
`lemma_dict` alone accepts a pre-built handle. It does not reopen the case
for a general pipeline object: nothing else in `apply_pipeline` gets one.

**Order of operations**: `nfd` → `lowercase` → `strip_accents` →
(`stemmer` / `lemma_dict`) → `collapse_whitespace`, each step skipped
entirely when its flag is off/`None`. `nfd`/`lowercase`/`strip_accents`
are CODEPOINT-level transforms: they don't care about word boundaries,
so they run over the whole text directly (reusing `tors.nfd`'s and
`tf_idf`'s own accent-folding algorithm verbatim). `stemmer`/`lemma_dict`
are WORD-level: the already-transformed text is walked segment-by-segment
(the same UAX #29 `split_word_bounds` walk `tf_idf`/`bm25_rank` tokenize
with), transforming only real-word segments and preserving every other
segment (punctuation, whitespace) VERBATIM, so the output stays
readable prose, not a token list. `collapse_whitespace` runs last,
reducing every run of Python-whitespace-equivalent codepoints to exactly
one ASCII space, NOT `tors.normalize`'s full pipeline (no
CRLF folding, no blank-line-run collapsing, no leading/trailing strip),
just whitespace-run collapsing.

**Identity contract**: all six steps default off, and
`apply_pipeline(texts)` with nothing else is a true identity: the
original `texts` list OBJECT comes back unchanged, not just
content-equal output, matching the zero-allocation contract
`normalize`/`quote`/`replace_many` already give for their own no-op case.
Argument validation (every element genuinely a `str`) still runs even on
this fast path: a non-`str` element always raises `TypeError`, never
silently passes through untouched. Empty `texts` → `[]`.

**`stemmer`/`lemma_dict`** are `tf_idf`'s exact same opt-in knobs (see its
docs for the accent-folding/stemming/lemma-substitution details, the
Snowball language list, and why lemmatization stays out of scope as a
bundled dataset). One note specific to this function: `rust-stemmers`'
`Stemmer::stem` expects already-lowercased input; passing `stemmer`
without `lowercase=True` is not an error, but the stem quality degrades:
this is not silently corrected, matching tors's "the caller composes"
posture throughout.

**Relationship to `tf_idf`/`bm25_rank`**: those two already fuse the SAME
`strip_accents`/`stemmer`/`lemma_dict` knobs directly into their own
tokenization. Calling `apply_pipeline` first and then `tf_idf`/`bm25_rank`
on the result tokenizes TWICE for no benefit: reach for their own knobs
when they're the only consumer; reach for `apply_pipeline` to preprocess
text feeding anything else (`chunk_text`, `find_patterns`, your own
logic).

A non-`list` argument or non-`str` element raises `TypeError`; an
unrecognized `stemmer` name raises `ValueError` naming every valid
choice; a non-`dict` `lemma_dict`, or one with a non-`str` key/value,
raises `TypeError`.

```python
tors.apply_pipeline(["  Café  RUNNERS   are   RUNNING!  "],
    nfd=True, lowercase=True, strip_accents=True, stemmer="english",
    collapse_whitespace=True)
# [' cafe runner are run! ']

tors.apply_pipeline(["This is better, right?"], lowercase=True,
    lemma_dict={"better": "good"})
# ['this is good, right?']
```

## `tors.soundex` / `tors.metaphone`

```python
def soundex(text: str) -> str: ...
def metaphone(text: str) -> str: ...
```

Two classic phonetic-code algorithms, via `rphonetic` (an Apache Commons
Codec port): `soundex` (a 1918-patent-era letter-plus-three-digits code,
e.g. `"Robert"`/`"Rupert"` both encode to `"R163"`) and `metaphone` (the
Double Metaphone PRIMARY code, Lawrence Philips' 2000 successor to
classic Metaphone, e.g. `"jumped"` → `"JMPT"`). Both are
ENGLISH/Latin-script-oriented heuristics, not general Unicode phonetics:
they group words that SOUND alike, typically alongside
`levenshtein`/`jaro_winkler` distance scoring rather than instead of it,
for name-matching and dedup pipelines.

**Input is pre-filtered to ASCII letters before encoding: a real
correctness fix, not a style choice.** `rphonetic` 4.0.0's
`Soundex::encode` and `DoubleMetaphone::encode` both PANIC on ordinary
accented input, confirmed directly against the raw crate: `Soundex`'s own
"clean" step filters by the FULL-UNICODE `char::is_alphabetic` (too
broad: Cyrillic, CJK, Greek, and accented Latin like `'é'` all pass it),
then unconditionally indexes a 26-element ASCII mapping table with
`ch as usize - 65`, out of bounds for anything outside plain `A`-`Z`;
`DoubleMetaphone` separately panics on multi-byte characters via a
byte-index slice that assumes one byte per character. Both crash on
exactly the realistic input a name-matching consumer would pass:
`Soundex::default().encode("José")` and
`DoubleMetaphone::default().encode("Björk")` both panic on the raw crate.
tors never lets a Rust panic reach Python, so `text` is filtered to ASCII
letters (`char::is_ascii_alphabetic`) before either algorithm sees it:
accents and non-Latin characters are DROPPED, not encoded, a documented
degradation consistent with these algorithms' documented English-only
scope even where the upstream crate doesn't crash. Empty input, or input
with no ASCII letters at all, → `""`.

`py.detach` around each call; a single `str` return (no marshalling
class).

```python
tors.soundex("Robert"), tors.soundex("Rupert")
# ('R163', 'R163')
tors.metaphone("jumped")
# 'JMPT'
tors.soundex("José"), tors.metaphone("café")
# ('J200', 'KF')  — accents dropped, not crashed on
```

## `tors.double_metaphone`

```python
def double_metaphone(text: str) -> tuple[str, str]: ...
```

The full dual-key form of `tors.metaphone`: `(primary, alternate)`. The
alternate code is the algorithm's whole point — for names readable two
ways (Germanic/Slavic vs. Anglicized) it carries the second
pronunciation, so a name-matching pipeline scores a match when EITHER
key of two names agrees; for words with one plausible pronunciation the
two elements are equal. Same ENGLISH/Latin-script scope, ASCII-letters
pre-filter, and upstream-panic-avoidance note as `soundex`/`metaphone`
above. Empty input (or input with no ASCII letters) → `("", "")`.

```python
tors.double_metaphone("jumped")
# ('JMPT', 'AMPT')
```

## `tors.nysiis`

```python
def nysiis(text: str) -> str: ...
```

The NYSIIS code (New York State Identification and Intelligence System,
1970), via `rphonetic`'s strict commons-codec variant (codes capped at 6
characters). A Soundex successor with better first-letter and vowel
handling. Same scope/pre-filter/panic-avoidance note as `soundex` — the
filter additionally keeps NYSIIS keys pure ASCII, since the crate's own
clean step would otherwise let accented letters through into the code
itself. Empty input (or input with no ASCII letters) → `""`.

```python
tors.nysiis("Washington")
# 'WASANG'
```

## `tors.daitch_mokotoff`

```python
def daitch_mokotoff(text: str) -> list[str]: ...
```

The Daitch-Mokotoff Soundex codes (1985), via `rphonetic`'s port of
Apache Commons Codec with branching enabled: 6-digit codes designed for
Central/Eastern European surnames, the standard of Jewish-genealogy
surname matching, distinguishing sounds (guttural vs. sibilant) classic
Soundex conflates. Returns a LIST, not a single string: the rule table
branches on ambiguous transliterations, so one name can legitimately
encode to several codes — two names match if ANY of their code lists
intersect. Each code is padded to 6 digits, so input with no encodable
letters yields `["000000"]`, not `""`. Same
scope/pre-filter/upstream-panic-avoidance note as `soundex`.

```python
tors.daitch_mokotoff("Peters")
# ['734000', '739400']
```

## `tors.refined_soundex`

```python
def refined_soundex(text: str) -> str: ...
```

A Soundex variant with a finer-grained letter-to-digit mapping table
than classic Soundex — more consonant classes distinguished, at the
cost of a longer, uncapped code rather than Soundex's fixed
letter-plus-three-digit shape. A genuinely distinct mapping, not a
formatting variant of `tors.soundex` (confirmed: `"Robert"` encodes
differently under each). Shares Soundex's exact upstream panic bug
(confirmed directly against the raw crate: `RefinedSoundex::default()
.encode("José")` panics with an out-of-bounds table index), so it
carries the same ASCII-letters pre-filter. Empty input (or input with
no ASCII letters) → `""`.

```python
tors.refined_soundex("Robert"), tors.refined_soundex("Rupert")
# ('R901096', 'R901096')
```
