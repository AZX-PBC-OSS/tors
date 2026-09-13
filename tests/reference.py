"""The pure-Python reference pipeline (the differential oracle) and the deterministic
corpora shared by the correctness, parity, GIL-release-proof, and wall-time tests.

``reference_normalize`` reimplements tors's pipeline directly against the *running
interpreter's* ``unicodedata`` and ``re`` (the pipeline's original pure-Python
spelling, unchunked), so the oracle is fully self-contained while still pinning tors
against each interpreter's own Unicode tables. ``reference_finalize`` appends the
SHA-256 tail a normalize-then-hash dedupe gate computes. The corpus builders are
deterministic (no rng), so every gap/wall number quoted in the test docstrings is
reproducible byte-for-byte.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Sequence

from hypothesis import strategies as st

# Named constants built from codepoints (pure-ASCII source, never a literal typed
# accented/space character): a combining-mark sequence and its precomposed form are
# visually identical in an editor but byte-distinct, and cases built from these constants
# depend on that distinction.
_E_ACUTE_PRECOMPOSED = chr(0x00E9)
_COMBINING_ACUTE = chr(0x0301)
_NBSP = chr(0x00A0)
_LINE_SEP = chr(0x2028)
_PARA_SEP = chr(0x2029)
_IDEOGRAPHIC_SPACE = chr(0x3000)

_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def reference_normalize(text: str) -> str:
    """Pure-Python reimplementation of the pipeline's original spelling, unchunked. The
    self-contained correctness oracle: it depends only on the running interpreter."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _TRAILING_WS.sub("\n", normalized)
    normalized = _BLANK_RUN.sub("\n\n", normalized)
    return normalized.strip()


def reference_finalize(text: str) -> tuple[str, str]:
    """The ``finalize`` oracle: ``(normalized, lowercase-hex sha256 of the normalized
    text's UTF-8 bytes)``, exactly ``hashlib.sha256(...).hexdigest()``, the expression a
    normalize-then-hash dedupe gate computes, and what ``tors.finalize`` must stay
    byte-identical to."""
    normalized = reference_normalize(text)
    return normalized, hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@st.composite
def pathological_text(draw: st.DrawFn) -> str:
    """Strings built from whitespace-heavy, line-ending-heavy, and NFC-sensitive pieces,
    biased toward exactly the content classes that stress this pipeline, rather than
    uniform random unicode (covered separately by plain ``st.text`` in the tests that
    also use this strategy)."""
    pieces = st.one_of(
        st.just(""),
        st.text(alphabet="ab" + _E_ACUTE_PRECOMPOSED, min_size=0, max_size=10),
        st.text(alphabet="e" + _COMBINING_ACUTE, min_size=0, max_size=6),
        st.text(alphabet=" \t", min_size=1, max_size=30),
        st.text(alphabet="\n", min_size=1, max_size=10),
        st.text(alphabet="\r", min_size=1, max_size=10),
        st.sampled_from(["\r\n", "\r\n\r\n", " \n", "\t\n", " \t\n", "\n\n\n", "\r\r\r"]),
        st.sampled_from([_NBSP, _IDEOGRAPHIC_SPACE, _LINE_SEP, _PARA_SEP]),
    )
    parts = draw(st.lists(pieces, min_size=0, max_size=25))
    return "".join(parts)


# --- Deterministic corpora ---------------------------------------------------------
#
# The prose recipe is the deterministic prose-corpus idiom the original specification
# pinned: a repeated sentence + "\n\n", so every measurement in this suite is comparable
# across cells for the same shape of text. Sizes are UTF-8 bytes, unit-quantized: a
# 12 MiB target lands within one unit of 12 MiB.

_PROSE_SENTENCE = (
    "The quarterly oil sample interval for field outages was adjusted after the bushing "
    "torque specifications changed. Maintenance windows now close within fourteen days. "
)

# Same sentence with three decomposed accents (base letter + U+0301, built by
# concatenation so the source stays ASCII), the composition-heavy corpus planned for
# the consumer's loop-safety re-measure at swap-in time (spec m3).
_DECOMPOSED_SENTENCE = (
    "The quarte"
    + _COMBINING_ACUTE
    + "rly oil sa"
    + _COMBINING_ACUTE
    + "mple interval for field outa"
    + _COMBINING_ACUTE
    + "ges was adjusted after the bushing "
    "torque specifications changed. Maintenance windows now close within fourteen days. "
)

# The decomposed sentence plus a compatibility ligature (U+FB01) and a
# fullwidth digit (U+FF10) per unit, the corpus for the K-forms' full-pass
# cells. Why it exists (a finding): the plain decomposed corpus carries
# no compatibility mappings, so under NFKC/NFKD it quick-checks Yes and the
# identity-return fast path hands it back unchanged, so the D-forms' transform
# band needs input that still pays the pass.
_LIGATURE_FI = chr(0xFB01)
_FULLWIDTH_ZERO = chr(0xFF10)

_COMPAT_SENTENCE = (
    "The quarte"
    + _COMBINING_ACUTE
    + "rly oil sa"
    + _COMBINING_ACUTE
    + "mple interval "
    + _LIGATURE_FI
    + "eld outa"
    + _COMBINING_ACUTE
    + "ges was adjusted "
    + _FULLWIDTH_ZERO
    + " days after the bushing "
    "torque specifications changed. Maintenance windows now close within fourteen days. "
)

# Entity-bearing prose for tors.html_unescape's cells: the prose sentence shape
# with nine html5 entity refs per ~131 chars (~7% density, dense but the
# shape real escaped text produces). Pure ASCII on the wire (entities are);
# the decoded output is non-ASCII (e, copyright, nbsp, therefore), which is
# what makes the output-marshalling band of the GIL cells the interesting one.
_ENTITY_SENTENCE = (
    "The quarterly &amp; field &lt;outage&gt; interval &quot;adjusted&quot; after "
    "&#233; the bushing &copy; changed &nbsp; for torque &there4; specs. "
)


def _repeat_to(target_bytes: int, unit: str) -> str:
    return unit * max(1, target_bytes // len(unit.encode("utf-8")))


def prose(target_bytes: int) -> str:
    """Plain prose paragraphs ("\n\n"-separated): the corpus shape the original
    specification's audit used."""
    return _repeat_to(target_bytes, _PROSE_SENTENCE * 4 + "\n\n")


def decomposed(target_bytes: int) -> str:
    """Prose with decomposed accents: every paragraph pays the NFC composition pass."""
    return _repeat_to(target_bytes, _DECOMPOSED_SENTENCE * 4 + "\n\n")


def compat(target_bytes: int) -> str:
    """Decomposed prose with compatibility mappings (ligature, fullwidth
    digit): the corpus that still pays NFKC/NFKD's full pass now that the
    quick-check fast path identity-returns plain decomposed text."""
    return _repeat_to(target_bytes, _COMPAT_SENTENCE * 4 + "\n\n")


def crlf(target_bytes: int) -> str:
    """Prose with CRLF endings and trailing space/tab runs: the fold/drop-heavy variant."""
    return _repeat_to(target_bytes, _PROSE_SENTENCE * 4 + " \t\r\n\r\n")


def entities(target_bytes: int) -> str:
    """Prose dense with html5 entity refs (nine per sentence): the
    html_unescape corpus: ASCII in, non-ASCII decoded out."""
    return _repeat_to(target_bytes, _ENTITY_SENTENCE * 4 + "\n\n")


# Exception-shaped text for tors.scrub_log_text's cells: a rendered asyncpg-style
# failure whose DETAIL line quotes a caller-supplied key value (the row-value
# leak the pg_detail_lines rule exists for; HINT kept — it is structural), a DSN
# carrying both a userinfo password and a password-family query parameter (both
# mask rules fire on it), and a repr()-flattened twin whose DETAIL run rides
# literal \n separators (the escaped segmenter's shape). Every rule fires once
# per unit, so a scrub cell over this corpus measures the scan+splice, never the
# identity fast path. Pure ASCII: the GIL cell's argument borrow is then the
# zero-copy class (the replace_many dense cell's shape); non-ASCII scrub shapes
# are the differential battery's job (tests/test_scrub_log_text_parity.py), not
# a corpus concern. benches/text.rs mirrors this constant (SCRUB_SENTENCE) and
# tests/test_bench_corpus_parity.py pins the two byte-identical, so the bench
# numbers and the Python-side cell numbers cross-reference on the same bytes.
_SCRUB_SENTENCE = (
    "Traceback (most recent call last):\n"
    "  File 'worker/run.py', line 88, in run\n"
    "JobError: duplicate key value violates unique constraint 'jobs_idempotency_key'\n"
    "DETAIL:  Key (idempotency_key)=(customer-4417-a3f2) already exists.\n"
    "HINT: The SQL statement is unchanged.\n"
    "connect dsn=postgresql://worker:S3cr3t-x9@db.internal:5432/prod?password=fallback\n"
    "JobError('duplicate key\\nDETAIL:  Key (idempotency_key)=(customer-4417-a3f2) "
    "already exists.')\n"
)


def scrub_corpus(target_bytes: int) -> str:
    """Rendered-exception text (real-newline DETAIL line, DSN with both
    credential shapes, repr()-flattened DETAIL run): the scrub_log_text
    corpus, every rule firing once per unit."""
    return _repeat_to(target_bytes, _SCRUB_SENTENCE * 4 + "\n")


# The bytes-in corpora for the bytes surface (decode_utf8 / finalize_utf8 /
# b64_encode_bytes): the named str corpora rendered to UTF-8. benches/bytes.rs
# builds the same bytes in Rust and tests/test_bench_corpus_parity.py pins that
# identity, so this rendering is the single source for both sides.
_CORPUS_BUILDERS: dict[str, Callable[[int], str]] = {
    "prose": prose,
    "decomposed": decomposed,
    "compat": compat,
    "crlf": crlf,
    "entities": entities,
    "scrub": scrub_corpus,
}


def corpus_utf8(kind: str, target_bytes: int) -> bytes:
    """The ``kind`` corpus (``prose`` / ``decomposed`` / ``compat`` / ``crlf``
    / ``entities`` / ``scrub``) as UTF-8 bytes, the byte-compatible
    counterpart of the str corpora, for the bytes-in API."""
    return _CORPUS_BUILDERS[kind](target_bytes).encode("utf-8")


def corpus_b64(kind: str, target_bytes: int) -> str:
    """The ``kind`` corpus base64-encoded (standard alphabet, padded, ASCII), the
    input corpus for ``tors.b64_decode`` cells: decoding it yields the
    ``corpus_utf8`` bytes back, so the ``target_bytes`` semantics (the decoded
    size) stay comparable with every other corpus kind."""
    return base64.b64encode(corpus_utf8(kind, target_bytes)).decode("ascii")


# --- diff-opcode pair corpora ------------------------------------------------------
#
# (a, b) string pairs for ``tors.diff_opcodes``'s wall and GIL cells, built from
# the prose recipe so every size stays comparable with the rest of the suite's
# corpora. ``benches/diff.rs`` builds the same pairs in Rust (from the shared
# ``prose`` recipe in ``benches/common/mod.rs``), and
# tests/test_bench_corpus_parity.py pins the constants the two sides share, so
# the bench numbers and the Python-side cell numbers cross-reference on the
# same bytes.

# The word swap applied to the "replaced" lines of the near-identical pair (and
# to its one inserted line): a small, realistic in-line edit (four word
# substitutions inside a 524-char paragraph line) rather than a wholesale
# different line.
_DIFF_WORD_SWAP = ("quarterly", "monthly")

# Edit positions as non-dyadic line fractions, spelled as (numerator,
# denominator) pairs so the integer math mirrors the Rust side exactly:
# n // 5, n // 3, (2 * n) // 3, (7 * n) // 9 for the four replaced lines,
# n // 7 for the deleted line, (4 * n) // 9 for the inserted line. Dyadic
# fractions (1/4, 1/2, 3/4 ...) collide with the Myers divide-and-conquer's
# bisection midpoints at every recursion depth and fragment the op stream:
# measured, the same six edits at n/4, n/2, 3n/4 produce 18,917 ops at 12 MiB
# where these positions produce 235. The op count is the algorithm's business;
# the corpus's job is to not manufacture pathology.
_DIFF_REPLACE_FRACTIONS = ((1, 5), (1, 3), (2, 3), (7, 9))
_DIFF_DELETE_FRACTION = (1, 7)
_DIFF_INSERT_FRACTION = (4, 9)

# The shuffled pair's deterministic u64 LCG (Knuth-style constants), inline
# arithmetic, not the ``random`` module, so the corpus is reproducible
# byte-for-byte on every machine and matches the Rust twin exactly (Python's
# big ints masked to 64 bits behave identically to u64 wrapping ops).
_DIFF_LCG_SEED = 0x9E3779B97F4A7C15
_DIFF_LCG_MUL = 6364136223846793005
_DIFF_LCG_INC = 1442695040888963407
_U64_MASK = (1 << 64) - 1

# Zero-padded decimal width of the line numbers that make the shuffled pair's
# lines distinct (the prose recipe's paragraph lines are byte-identical, so an
# un-numbered line shuffle would be a no-op).
_DIFF_LINE_NUMBER_WIDTH = 6


def diff_pair_near_identical(target_bytes: int) -> tuple[str, str]:
    """``(a, b)``: the prose corpus vs itself with six scattered line-level
    edits (four word-swapped replacement lines, one deleted line, one inserted
    word-swapped line) at the non-dyadic fractions above. The line-level edit
    shape a re-diff of an edited document pays for; at 256 KiB this is the size
    where ``difflib`` takes seconds (measured: 3.1 s) and tors single-digit
    milliseconds."""
    a = prose(target_bytes)
    lines = a.split("\n")
    n = len(lines)
    old_word, new_word = _DIFF_WORD_SWAP
    replace_at = {num * n // den for num, den in _DIFF_REPLACE_FRACTIONS}
    delete_at = _DIFF_DELETE_FRACTION[0] * n // _DIFF_DELETE_FRACTION[1]
    insert_at = _DIFF_INSERT_FRACTION[0] * n // _DIFF_INSERT_FRACTION[1]
    out: list[str] = []
    for idx, line in enumerate(lines):
        if idx in replace_at:
            out.append(line.replace(old_word, new_word))
        elif idx == delete_at:
            continue
        else:
            out.append(line)
        if idx == insert_at:
            out.append(_PROSE_SENTENCE.replace(old_word, new_word))
    return a, "\n".join(out)


def diff_pair_shuffled(target_bytes: int) -> tuple[str, str]:
    """``(a, b)``: the prose corpus with every paragraph line numbered (making
    the lines distinct; see ``_DIFF_LINE_NUMBER_WIDTH``), vs the same numbered
    lines permuted by the deterministic LCG Fisher-Yates shuffle. The
    many-opcode shape: the character-level diff of two same-content
    different-order corpora, which is what makes the O(ops) return-marshalling
    band measurable (103,421 opcodes at 12 MiB, measured). Note: this pair is
    easy for the Myers engine: every line is unique (numbered), so the
    preflight anchors them and the diff stays fast at every size."""
    a = prose(target_bytes)
    lines = [
        "" if not line else f"{idx:0{_DIFF_LINE_NUMBER_WIDTH}d} {line}"
        for idx, line in enumerate(a.split("\n"))
    ]
    first = "\n".join(lines)
    state = _DIFF_LCG_SEED
    for i in range(len(lines) - 1, 0, -1):
        state = (state * _DIFF_LCG_MUL + _DIFF_LCG_INC) & _U64_MASK
        j = state % (i + 1)
        lines[i], lines[j] = lines[j], lines[i]
    return first, "\n".join(lines)


def diff_pair_char_shuffled(target_bytes: int) -> tuple[str, str]:
    """``(a, b)``: the prose corpus vs the same characters permuted by the
    deterministic LCG Fisher-Yates, the hard Myers shape. Where the
    line-shuffled pair's unique numbered lines give the preflight cheap
    anchors, a character permutation of prose has almost none (prose is
    dominated by a small alphabet), so the bounded search's work grows
    superlinearly with size; measured on the dev box, ambient load 2.7:
    50k chars 0.33 s, 100k 1.06 s, and the ladder in the api.md diff
    section (the cost curve ``deadline_ms`` exists to bound). The
    ``diff_opcodes`` deadline test's slow pair is built from this: small
    enough to build in milliseconds, hard enough that its unbounded diff
    costs ~1-3 s while ``deadline_ms=50`` returns in tens of milliseconds."""
    a = prose(target_bytes)
    chars = list(a)
    state = _DIFF_LCG_SEED
    for i in range(len(chars) - 1, 0, -1):
        state = (state * _DIFF_LCG_MUL + _DIFF_LCG_INC) & _U64_MASK
        j = state % (i + 1)
        chars[i], chars[j] = chars[j], chars[i]
    return a, "".join(chars)


# --- search pattern sets ------------------------------------------------------------
#
# The shared pattern sets for ``tors.find_patterns``'s wall/GIL cells.
# ``benches/search.rs`` mirrors both tuples, and
# ``tests/test_bench_corpus_parity.py`` cross-checks the mirrors, so the bench
# numbers and the Python-side cell numbers cross-reference on the same
# patterns.

# The dense set: seventeen words drawn from the prose sentence, every one
# matching once per sentence (and none a substring of another), so over the
# prose corpus this is the matches-heavy shape (68 matches per 666-byte unit,
# ~1.28M matches at 12 MiB) that makes the O(matches) return marshalling
# measurable.
SEARCH_DENSE_PATTERNS: tuple[str, ...] = (
    "quarterly",
    "oil",
    "sample",
    "interval",
    "field",
    "outages",
    "adjusted",
    "bushing",
    "torque",
    "specifications",
    "changed",
    "Maintenance",
    "windows",
    "close",
    "within",
    "fourteen",
    "days",
)

# The sparse set: plausible terminology-scan words absent from the prose
# sentence. Over the plain prose corpus nothing matches (the pure-scan shape
# the bench measures); over the diff near-identical pair's edited side,
# ``"monthly"`` (the swap word) appears exactly once at 12 MiB (measured:
# the inserted line carries it; the builder's four replace positions all
# land on the corpus's empty separator lines at that size, so they are
# no-ops there; the diff builder's own fraction arithmetic, not a
# search-side choice). The few-matches shape the GIL cell measures.
SEARCH_SPARSE_PATTERNS: tuple[str, ...] = ("monthly", "weekly", "annually")

# The escape-parity scan's needle and corpora (the issue #50 surface):
# ``tors.contains_unescaped`` / ``tors.find_unescaped`` answer, over raw bytes,
# whether a six-byte escape text is "live" (preceded by an even run of
# backslashes) or literal text. ``benches/search.rs`` mirrors both constants
# (the needle and the false-positive injection unit), and
# ``tests/test_bench_corpus_parity.py`` pins the mirrors, so the bench numbers
# and the Python-side cell numbers cross-reference on the same bytes.
UNESCAPED_NEEDLE: bytes = b"\\u0000"

# orjson's rendering of the literal six-character TEXT "\u0000": the
# backslash itself escaped, so the text spans SEVEN bytes (two backslashes
# then ``u0000``) and the needle occurs once, at +1, behind a single
# backslash — an odd run, a rejected hit. Every occurrence in the
# false-positive corpus is therefore a false positive: the exact shape a
# confirm-by-re-parse walk exists to clear, and the hit-dense workload the
# bench and the GIL/wall cells drive (a corpus of real NUL escapes would
# answer at the first hit and measure nothing).
_ESCAPE_LITERAL_TEXT = "\\\\u0000"


def unescaped_false_positive(target_bytes: int) -> bytes:
    """Prose with one literal escape text (the seven-byte two-backslash
    rendering) injected per sentence: pure ASCII, deterministic,
    unit-quantized like every ``reference`` corpus, and every needle
    occurrence sits behind an odd backslash run, so the scan rejects every
    hit and runs to the end — the worst case for both wall time and GIL
    release (no early exit), at a realistic orjson false-positive density
    (one literal per 166-byte prose sentence: 173.5 bytes of corpus per
    injection once the seven-byte literal and the paragraph break are
    amortized)."""
    return _repeat_to(target_bytes, (_PROSE_SENTENCE + _ESCAPE_LITERAL_TEXT) * 4 + "\n\n").encode(
        "utf-8"
    )


# --- content_hash object corpus -------------------------------------------------------
#
# The JSON-shaped object tree for ``tors.content_hash``'s cells: the 1 MiB
# parity case in ``tests/test_content_hash.py``, the 12 MiB GIL cell in
# ``tests/test_gil_release.py``, and the wall cells in
# ``tests/test_performance.py``. A document dict of records, each a pure
# function of its index (the no-rng determinism every corpus here follows):
# ``{"id": int, "name": str, "note": str, "score": float, "tags": list[str],
# "active": bool}``, the str-heavy shape a serialized API response or a model
# batch row actually has. The ``note`` carries the shared prose sentence twice
# so the payload bytes cross-reference the same recipe every other corpus
# family measures.
#
# Sizing: ``target_bytes`` counts the CANONICAL form's bytes
# (``json.dumps(record, sort_keys=True, separators=(",", ":"))``), quantized
# to whole records via the canonical size of record 0, computed at call time
# from the builder itself (the same dynamic-unit idiom ``_repeat_to`` uses;
# the ``id`` and ``name`` fields stay fixed-width, and the ``score`` repr is
# fixed-width per residue class, so record 0's size is representative to
# within a few bytes across any build).


def _content_record(i: int) -> dict[str, object]:
    """Record ``i``: every field a pure function of ``i``, no rng."""
    return {
        "id": i,
        "name": f"record-{i:06d}",
        "note": _PROSE_SENTENCE * 2,
        "score": (i % 40) * 0.125,
        "tags": ["alpha", "beta"] if i % 2 else [],
        "active": i % 3 == 0,
    }


def content_object(target_bytes: int) -> dict[str, object]:
    """The ``tors.content_hash`` corpus at ``target_bytes`` of canonical form:
    ``{"schema": 2, "count": n, "records": [...]}`` with ``n`` whole records
    (``max(1, target // per-record canonical bytes)``, so a 12 MiB target
    lands within one record of 12 MiB of canonical output)."""
    unit = len(json.dumps(_content_record(0), sort_keys=True, separators=(",", ":"))) + 1
    n = max(1, target_bytes // unit)
    return {"schema": 2, "count": n, "records": [_content_record(i) for i in range(n)]}


# --- shared differential oracles -------------------------------------------------------
#
# The brute-force references and structural checkers that more than one gate
# needs, in one place so the gates cannot drift apart: the leftmost-longest
# search and replace oracles (pure-Python, character space, independent of
# every implementation detail on the tors side (bytes, automata, offset
# mapping), and the difflib-shape opcode validity checker (works over any
# token sequence: a ``str`` for the char-level diff, a ``list[str]`` of
# lines for the line-level one).


def _lcs_len(a: str, b: str) -> int:
    """Longest-common-subsequence length by the classic O(len(a)·len(b))
    rolling-row DP: an algorithm sharing no machinery with ``similar``'s
    Myers engine, so agreement between the two is evidence about the
    contract (M is maximal), not a shared bug."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0]
        extend = cur.extend  # the DP's whole body: one row per char of a
        for j, bj in enumerate(b, start=1):
            extend((prev[j - 1] + 1,) if ch == bj else (max(prev[j], cur[j - 1]),))
        prev = cur
    return prev[-1]


def reference_is_grounded_fuzzy(claim: str, source: str, threshold: float) -> bool:
    """The pure-Python model of ``tors.is_grounded(fuzzy=True)``'s documented
    contract: the exact-containment floor, then the best region score over
    the stride-``len(claim)//2`` windows of ``source`` (the truncated tail
    window included), then (when the coarse best falls short) the bounded
    refinement pass: the top 64 windows scoring >= 0.5 (by
    ``(score, start)``, the streaming keep-K-largest set; the truncated tail
    competes like any other), each re-scanned at fine stride
    ``max(1, L // 16)`` across ``[w - L//2, min(w + L//2, n - L)]``, the grid
    always extended to the range's upper end. Every coarse window is scored:
    the implementation's early break at ``best >= threshold`` is
    verdict-equivalent, since ``best`` only ever rises and the comparison is
    the same at the end; likewise the refinement's best-first order and
    early break, so this model scans each candidate's full fine range and
    still lands on the identical verdict.

    Every window in the windowed regime (``len(source) > len(claim)``) is
    scored as the claim-length region it truncates: ``2*M / (2*L)`` with
    ``M`` the window's LCS length against the claim, so a full window scores
    ``2*M/(L + L)`` (the difflib shape over equal-length operands) and the
    truncated tail window scores the same ``M/L`` -- its missing chars are
    mismatches, never a discounted ``L + len(window)`` denominator, which
    would inflate a source that just ends partway through the evidence and
    make the verdict depend on where the evidence sits (issue #40's
    position-dependence: the same 76%-of-claim tail scored 0.861 flush with
    the end and 0.756 mid-source). ``len(source) <= len(claim)`` keeps the
    one direct whole-source difflib ratio ``2*M/(m + n)``: a different,
    pinned convention (the unwindowed difflib-parity lane in
    ``test_grounded.py``), the boundary between the two sitting exactly at
    ``len(source) == len(claim)`` where both formulas agree.
    ``M`` here is the LCS length from the independent DP above, not
    difflib's anchored matching blocks: tors's ``M`` is the maximal one
    (``M == LCS`` exactly, the minimal-edit-script consequence of the
    Myers engine), so this oracle is exact on the repeated-character
    inputs where difflib's own recursion can pick a smaller-but-valid ``M``
    (the pinned ``"010"``/``"120"`` divergence class in the grounded tests)."""
    if not claim:
        return True
    if claim in source:
        return True  # the exact-containment floor
    m, n = len(claim), len(source)
    if n <= m:
        # No windowing possible: one direct comparison, the same convention.
        return 2 * _lcs_len(claim, source) / (m + n) >= threshold
    stride = max(m // 2, 1)
    best = 0.0
    band: list[tuple[float, int]] = []  # (score, char start), full + tail
    start = 0
    while True:
        end = min(start + m, n)
        # 2*M/(2*m): identical for full windows (end - start == m) and the
        # truncated tail (missing chars are mismatches).
        score = _lcs_len(claim, source[start:end]) / m
        best = max(best, score)
        if score >= 0.5:
            band.append((score, start))
        if end == n:
            break
        start += stride
    if best >= threshold:
        return True
    # The candidate set: top 64 by (score, start): the same set the
    # implementation's streaming keep-K-largest maintains.
    candidates = sorted(band, key=lambda c: (c[0], c[1]), reverse=True)[:64]
    fine = max(m // 16, 1)
    for w in [c[1] for c in candidates]:
        if best >= threshold:
            break
        lo = max(0, w - m // 2)
        hi = min(w + m // 2, n - m)
        if lo > hi:
            continue
        starts = list(range(lo, hi + 1, fine))
        if starts[-1] != hi:
            starts.append(hi)  # the grid alone could leave a fine-1 gap at hi
        for fstart in starts:
            best = max(best, _lcs_len(claim, source[fstart : fstart + m]) / m)
    return best >= threshold


def reference_find_patterns(patterns: list[str], text: str) -> list[tuple[int, int, int]]:
    """The leftmost-longest oracle: a brute-force, non-overlapping reference,
    O(len(text) · len(patterns)) over character positions, pure Python ``str``
    operations (``startswith`` at an index is char-space by definition), so it is
    independent of every implementation detail on the tors side (bytes,
    automata, offset mapping). Among patterns matching at a position, the
    strictly-longer one replaces the current best, so an equal-length tie
    (only possible for byte-identical duplicates) keeps the first index."""
    matches: list[tuple[int, int, int]] = []
    pos = 0
    while pos < len(text):
        best_len = 0
        best_idx = -1
        for idx, pattern in enumerate(patterns):
            if len(pattern) > best_len and text.startswith(pattern, pos):
                best_len = len(pattern)
                best_idx = idx
        if best_idx >= 0:
            matches.append((pos, pos + best_len, best_idx))
            pos += best_len
        else:
            pos += 1
    return matches


def reference_replace_many(text: str, replacements: dict[str, str]) -> str:
    """The simultaneous-replace oracle: brute-force leftmost-longest,
    non-overlapping replacement over character positions: at each position
    the longest key that matches wins (dict order cannot matter: two distinct
    keys of the same length can never both match at one position), the scan
    resumes at the consumed span, and replacement output is never re-scanned
    (the value is emitted and the scan moves on; the no-cascade rule).
    Keys are assumed non-empty (tors refuses empty keys at the boundary)."""
    out: list[str] = []
    pos = 0
    while pos < len(text):
        best_len = 0
        best_new: str | None = None
        for old, new in replacements.items():
            if len(old) > best_len and text.startswith(old, pos):
                best_len = len(old)
                best_new = new
        if best_new is not None:
            out.append(best_new)
            pos += best_len
        else:
            out.append(text[pos])
            pos += 1
    return "".join(out)


def reference_first_invalid_charset(items: Sequence[str], first: str | None, rest: str) -> int:
    """The membership-loop oracle for ``tors.first_invalid_charset``: the
    positional rule spelled directly in pure Python. ``first`` (when given)
    is the set of codepoints allowed at position 0, ``rest`` the set allowed
    at every position after it (and at position 0 too when ``first`` is
    ``None``, the uniform spelling); an empty item is an offender; the
    answer is the first offending item's index, ``-1`` when all pass. The
    membership test is per codepoint (``ch in set`` over a one-codepoint
    needle is exact membership, never substring semantics), and the whole
    oracle shares no machinery with the tors side (bitmaps, sorted vectors,
    bytes-vs-codepoints), so agreement is evidence about the contract."""
    for idx, item in enumerate(items):
        if not item:
            return idx
        allowed_first = rest if first is None else first
        if item[0] not in allowed_first:
            return idx
        if any(ch not in rest for ch in item[1:]):
            return idx
    return -1


def reference_first_invalid_offender(
    items: Sequence[str], first: str | None, rest: str
) -> tuple[int, int, str] | None:
    """The offender-detail oracle for ``tors.first_invalid_offender``: the
    membership loop above re-spelled to return the detail a rejection
    message needs — ``(item_index, char_position, offending_char)`` for the
    first offending item's first offending position, ``None`` when every
    item passes. ``char_position`` counts CODEPOINTS within the item (the
    family's data model: ``item[position]`` is the offending codepoint, a
    1-char ``str``, never a byte offset), and the empty item — an offender
    with no codepoint at position 0 to name — reports ``(idx, 0, "")``: the
    char field is empty exactly when the item is. Shares no machinery with
    the tors side, so agreement is evidence about the contract."""
    for idx, item in enumerate(items):
        if not item:
            return (idx, 0, "")
        allowed_first = rest if first is None else first
        if item[0] not in allowed_first:
            return (idx, 0, item[0])
        for position, ch in enumerate(item[1:], start=1):
            if ch not in rest:
                return (idx, position, ch)
    return None


# --- the scrub_log_text oracle (the TaskQ exception-text chain) --------------------
#
# ``tors.scrub_log_text`` is a named-rule port of TaskQ's exception-text scrub
# chain (src/taskq/obs/_redact_exc.py, the consumer it exists for), pinned
# byte-identical to it: the four compiled regexes below are QUOTED VERBATIM
# from that module, and the canonical rule order (pg_detail_lines' two
# segmenters first, then uri_userinfo, then uri_query_creds) is _scrub_text's
# own application order with the redaction flag on. The differential harness
# (tests/test_scrub_log_text_parity.py) runs tors against this chain and,
# when the TaskQ checkout is present, re-syncs these patterns against the live
# module source — a TaskQ change to any of them is a visible re-sync request,
# not a silent tors behavior change.
_PG_DETAIL_RE = re.compile(r"^[ \t]*DETAIL:.*$", re.MULTILINE)
_PG_DETAIL_ESCAPED_RE = re.compile(
    r"(?:\\r)?\\n[ \t]*DETAIL:.*?(?=(?:\\r)?\\n|['\"]\)?\s*$)",
    re.MULTILINE,
)
_URI_CRED_RE = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@]*):([^\s@]+)@")
_URI_PARAM_CRED_RE = re.compile(r"([?&](?:password|passphrase|passwd|pwd)=)([^\s&@]+)")

#: The accepted rule names, in canonical application order.
SCRUB_RULES: tuple[str, ...] = ("pg_detail_lines", "uri_userinfo", "uri_query_creds")

#: Each rule's passes, in order: the DETAIL rule is one name over two
#: segmenters (real-newline lines, then repr()-flattened escaped runs); the
#: two URI rules are one pass each.
_SCRUB_RULE_PASSES: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "pg_detail_lines": ((_PG_DETAIL_RE, ""), (_PG_DETAIL_ESCAPED_RE, "")),
    "uri_userinfo": ((_URI_CRED_RE, r"\1:***@"),),
    "uri_query_creds": ((_URI_PARAM_CRED_RE, r"\1***"),),
}


def reference_scrub_log_text(text: str, rules: Sequence[str] | None = None) -> str:
    """The scrub oracle: the TaskQ chain applied per rule selection. ``rules
    is None`` runs the full chain in canonical order; a list/tuple selects a
    sub-chain (deduped, canonical order — the same contract tors spells);
    ``[]`` is the identity."""
    selected = frozenset(rules) if rules is not None else None
    names = SCRUB_RULES if selected is None else [n for n in SCRUB_RULES if n in selected]
    for name in names:
        for pattern, repl in _SCRUB_RULE_PASSES[name]:
            text = pattern.sub(repl, text)
    return text


_BACKSLASH = 0x5C  # b"\\"[0], the parity byte the whole escape question turns on


def reference_find_unescaped(haystack: bytes, needle: bytes) -> int:
    """The escape-parity oracle (the ``reference_find_patterns`` shape): a
    brute-force backward parity walk, pure-Python ``bytes`` operations only,
    independent of every implementation detail on the tors side (memmem
    engine, resume arithmetic). An occurrence of ``needle``
    at offset ``i`` counts only when the maximal run of backslashes
    immediately before ``i`` has even length (0 is even: an occurrence at
    offset 0 is live); a rejected hit advances the scan one byte past the
    hit, not past the whole match, so self-overlapping needles stay correct;
    no live occurrence answers ``-1`` (``bytes.find``'s sentinel).

    This is also the manual parity loop the wall cells race: the hand-rolled
    expression a consumer writes today (find the needle in the raw bytes,
    count the backslash run before each hit), the exact algorithm TaskQ
    verified against a re-parse walk before lifting it here."""
    pos = 0
    while True:
        hit = haystack.find(needle, pos)
        if hit == -1:
            return -1
        run = 0
        j = hit - 1
        while j >= 0 and haystack[j] == _BACKSLASH:
            run += 1
            j -= 1
        if run % 2 == 0:
            return hit
        pos = hit + 1


_OPCODE_TAGS = frozenset({"equal", "replace", "delete", "insert"})


def assert_opcodes_are_valid(
    a: str | list[str],
    b: str | list[str],
    ops: list[tuple[str, int, int, int, int]],
) -> None:
    """The difflib-shape structural contract for an opcode list over any token
    sequence: ``a``/``b`` are the operands as sequences (a ``str`` for the
    character-level diff, a ``list[str]`` of lines for the line-level one;
    slicing and concatenation work identically for both): ranges monotone,
    contiguous and covering both sides, the tag set, difflib's alternation
    shape (never two non-equal ops adjacent), per-tag nonemptiness, equal ops
    carrying equal-length equal-content ranges, and full reconstruction of
    both operands from the ops."""
    rebuilt_a: list[str] = []
    rebuilt_b: list[str] = []
    prev_i = 0
    prev_j = 0
    prev_equal: bool | None = None
    for op in ops:
        tag, i1, i2, j1, j2 = op
        assert tag in _OPCODE_TAGS, f"tag {tag!r} outside the difflib set"
        assert len(op) == 5, f"opcode is not a 5-tuple: {op!r}"
        assert (i1, j1) == (prev_i, prev_j), f"ranges not contiguous: {op!r} after {prev_i, prev_j}"
        is_equal = tag == "equal"
        assert prev_equal is None or prev_equal != is_equal, (
            f"tags do not alternate equal/non-equal around {op!r} (difflib's shape)"
        )
        if is_equal:
            assert i1 < i2 and j1 < j2, f"empty equal op: {op!r}"
            assert i2 - i1 == j2 - j1, f"equal op with differing side lengths: {op!r}"
            assert a[i1:i2] == b[j1:j2], f"equal op with unequal content: {op!r}"
            rebuilt_a.append(a[i1:i2])
            rebuilt_b.append(b[j1:j2])
        elif tag == "delete":
            assert i1 < i2, f"empty delete op: {op!r}"
            rebuilt_a.append(a[i1:i2])
        elif tag == "insert":
            assert j1 < j2, f"empty insert op: {op!r}"
            rebuilt_b.append(b[j1:j2])
        else:
            assert i1 < i2 and j1 < j2, f"empty replace op: {op!r}"
            rebuilt_a.append(a[i1:i2])
            rebuilt_b.append(b[j1:j2])
        prev_i, prev_j, prev_equal = i2, j2, is_equal
    assert prev_i == len(a) and prev_j == len(b), "ranges do not cover both sides"
    assert _joined(rebuilt_a, a), "opcodes do not reconstruct a"
    assert _joined(rebuilt_b, b), "opcodes do not reconstruct b"


def _joined(parts: list[str], whole: str | list[str]) -> bool:
    """Reconstruction check that works for both token shapes: char-level
    (parts join to the whole string) and line-level (parts join to the whole
    lines list; a list operand's slices are themselves lists, so the parts
    are flattened one level first)."""
    if isinstance(whole, str):
        return "".join(parts) == whole
    flattened = [token for part in parts for token in (part if isinstance(part, list) else [part])]
    return flattened == list(whole)


# --- MinHash oracle --------------------------------------------------------------------
#
# The pure-Python MinHash oracle for tests/test_minhash.py: the same token
# stream, the same shingle construction, the same shingle hash, and the same
# permutation arithmetic as ``tors.minhash_signature``, each transcribed from
# the pinned contract in ``src/minhash_impl.rs``'s module docs, so agreement
# between the two is evidence about that contract, not a shared bug.
#
# One deliberate non-self-containment, named here: the token stream comes
# from ``tors.word_bounds`` (the UAX #29 segmentation) rather than a pure
# Python reimplementation. UAX #29 word segmentation has no stdlib spelling
# and hand-deriving it would duplicate the unicode-segmentation tables; the
# segmentation surface carries its own independent contract gate
# (tests/test_segmentation.py), so the oracle stands on it and pins
# everything DOWNSTREAM: the whitespace-segment skip, the lowercase fold,
# the shingle join, the XXH64 shingle hash, and the whole
# SplitMix64-to-affine permutation pipeline.

# The Mersenne prime the affine permutations live over: h_i(x) = (a_i * x +
# b_i) mod p, p = 2^61 - 1, the standard MinHash field.
_MINHASH_MERSENNE = (1 << 61) - 1
# The u64 MAX sentinel: every element of the empty-shingle-set signature
# (empty text, or fewer tokens than shingle_size).
_MINHASH_EMPTY = (1 << 64) - 1
# The shingle join separator: U+001F ASCII unit separator. The UAX #29 walk
# never yields a token containing it (a C0 control is its own word segment),
# so the join is injective over token sequences.
_MINHASH_SEP = "\u001f"

# SplitMix64 (Steele/Marsaglia's fixed arithmetic, the standard
# fixture-grade 64-bit generator): the state advance and mixer constants.
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX64_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_M2 = 0x94D049BB133111EB

# The Unicode White_Space property's exact 25 codepoints: Rust's
# ``char::is_whitespace`` (the skip the Rust tokenizer applies), NOT
# ``str.isspace``/``str.strip`` -- Python's own space predicate also counts
# U+001C..U+001F, which Rust keeps as real word tokens.
_WHITESPACE = frozenset(
    chr(cp)
    for cp in (
        list(range(0x0009, 0x000E))  # TAB LF VT FF CR
        + [0x0020, 0x0085, 0x00A0, 0x1680]
        + list(range(0x2000, 0x200B))
        + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
    )
)


def reference_minhash_tokens(text: str) -> list[str]:
    """The token stream ``minhash_signature`` shingles: ``tors.word_bounds``
    segments, segments made entirely of White_Space codepoints skipped
    (Rust's ``char::is_whitespace`` set, the module comment's parity note),
    each lowercased with Python's full Unicode ``str.lower`` (the same full
    case mapping, SpecialCasing included, Rust's ``str::to_lowercase``
    implements)."""
    import tors

    tokens: list[str] = []
    for start, end in tors.word_bounds(text):
        segment = text[start:end]
        if not all(ch in _WHITESPACE for ch in segment):
            tokens.append(segment.lower())
    return tokens


def _splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 step: ``(state', output)`` -- advance the state by the
    golden-ratio gamma mod 2^64, then mix a copy of the new state (xorshift
    twice through the two mixer constants, final xorshift-right)."""
    state = (state + _SPLITMIX64_GAMMA) & _U64_MASK
    z = state
    z = ((z ^ (z >> 30)) * _SPLITMIX64_M1) & _U64_MASK
    z = ((z ^ (z >> 27)) * _SPLITMIX64_M2) & _U64_MASK
    z = z ^ (z >> 31)
    return state, z


def reference_minhash_coefficients(num_perm: int, seed: int) -> list[tuple[int, int]]:
    """The ``(a_i, b_i)`` pairs, the pinned derivation: a SplitMix64 stream
    seeded with ``seed`` reduced mod 2^64 (two's complement for negatives),
    two draws per permutation -- ``a_i`` first, in ``[1, 2^61 - 2]`` (a zero
    multiplier would collapse the permutation to a constant, so it is
    excluded), then ``b_i`` in ``[0, 2^61 - 2]``. Fixture-grade
    determinism, not crypto: the same arithmetic is pinned in
    ``src/minhash_impl.rs`` and golden-pinned by the test battery."""
    state = seed & _U64_MASK
    pairs: list[tuple[int, int]] = []
    for _ in range(num_perm):
        state, za = _splitmix64(state)
        state, zb = _splitmix64(state)
        a = za % (_MINHASH_MERSENNE - 1) + 1
        b = zb % _MINHASH_MERSENNE
        pairs.append((a, b))
    return pairs


def reference_minhash_signature(
    text: str, *, num_perm: int = 128, shingle_size: int = 3, seed: int = 0
) -> list[int]:
    """The MinHash oracle: ``reference_minhash_tokens``' stream cut into
    consecutive ``shingle_size``-token shingles joined with U+001F, each
    shingle hashed with XXH64 (seed 0; the pinned ``xxhash`` package
    wrapping the C reference implementation of the same frozen spec
    twox-hash implements on the Rust side), then ``signature[i] =
    min over shingles of (a_i * x + b_i) mod (2^61 - 1)`` over the
    ``reference_minhash_coefficients`` pairs. Fewer tokens than
    ``shingle_size`` (empty text included) is the empty-shingle-set
    convention: every element the u64 MAX sentinel."""
    import xxhash

    tokens = reference_minhash_tokens(text)
    if len(tokens) < shingle_size:
        return [_MINHASH_EMPTY] * num_perm
    coefficients = reference_minhash_coefficients(num_perm, seed)
    signature = [_MINHASH_EMPTY] * num_perm
    for i in range(len(tokens) - shingle_size + 1):
        shingle = _MINHASH_SEP.join(tokens[i : i + shingle_size])
        x = xxhash.xxh64_intdigest(shingle.encode("utf-8"))
        for j, (a, b) in enumerate(coefficients):
            h = (a * x + b) % _MINHASH_MERSENNE
            if h < signature[j]:
                signature[j] = h
    return signature
