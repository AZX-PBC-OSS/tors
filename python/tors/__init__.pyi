from collections.abc import Iterator
from typing import Literal

# The Snowball languages `rust-stemmers` ships: see tokenize_impl.rs's
# STEMMER_LANGUAGES (this is that same list, spelled as a type). Shared by
# tf_idf's and bm25_rank's stemmer= parameter rather than duplicated.
_StemmerLanguage = Literal[
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


def nfc(text: str) -> str: ...


def nfd(text: str) -> str: ...


def nfkc(text: str) -> str: ...


def nfkd(text: str) -> str: ...


# Raises ValueError on CPython 3.11+ when a DECIMAL numeric reference's digit
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


def word_bounds_iter(text: str) -> Iterator[tuple[int, int]]: ...


# GIL note: same shape as the list/iterator word_bounds pair: the whole
# segmentation runs with the GIL released (one detached pass; the iterator
# fills its buffer under that one detach at construction and each __next__
# holds the GIL for a single 2-tuple). Sentence counts are ~1/10th word
# counts on prose, so the list API's marshalling is proportionally cheaper
# here. UAX #29 rule-based segmentation only: no dictionary segmentation
# for spaceless scripts (Thai/Khmer/Burmese/Japanese); see the README.
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


# A heuristic guess, not a validator: the intended pipeline is utf8_is_valid
# first, and detect_encoding only on bytes that already failed that check.
# Always returns SOME codec name (WHATWG Encoding Standard labels, e.g.
# "windows-1252" / "Shift_JIS" / "UTF-8": case-insensitively valid for
# bytes.decode()), never raises for well-formedness reasons. tld is an
# ASCII top-level domain WITHOUT the leading dot ("jp", not ".jp").
def detect_encoding(raw: bytes, *, tld: str | None = None) -> str: ...


# GIL note (the word_bounds list-shape precedent): the diff itself runs with
# the GIL released, but building the returned list constructs one 5-tuple per
# opcode under the GIL: O(number of opcodes), measured ~0.1-0.15 µs/op (a
# ~10-15 ms hold above the ping floor for the 103,421-opcode 12 MiB
# shuffled-pair cell in tests/test_gil_release.py; see the README's diff
# section for the measured band).
#
# deadline_ms bounds the WHOLE call (default None = the unbounded diff,
# unchanged). The Myers search's work on hard inputs (few anchorable unique
# records, e.g. a character-level permutation) grows superlinearly with size
# (measured on the dev box, ambient load 2.6-3.7: 50k chars 0.32 s, 200k
# 3.67 s, 400k 13.81 s, 1M 183.6 s, roughly ~n^2; see the README's diff
# section for the ladder). On expiry the
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
# reacquired), and the return marshalling constructs one 5-tuple per LINE
# opcode with INTERNED tag strings (op[0] is "equal" holds, exactly as it
# does for difflib's own tuples). Indices address LINES in
# splitlines(keepends=True) shape (each line keeps its \n, the last may
# lack one); a line diff's op count sits far below its char-level twin's.
def diff_opcodes_lines(
    a: str, b: str, *, deadline_ms: float | None = None
) -> list[tuple[str, int, int, int, int]]: ...


# GIL note: the find_patterns argument shape over a dict: one GIL-held walk
# borrowing each key and value, then automaton build + scan + splice with
# the GIL released, then either the identity return (no key matched, or the
# net effect is the identity: tors.replace_many(s, m) is s exactly when
# tors.replace_many(s, m) == s) or the O(output) string marshalling. No
# list-shape class: the return is one string. Leftmost-longest semantics
# (NOT re.sub's leftmost-first), non-overlapping, replacements never
# re-scanned; dict order cannot matter.
def replace_many(text: str, replacements: dict[str, str]) -> str: ...


# GIL note (the word_bounds list-shape precedent, again): the automaton build,
# the scan, and the byte→char offset conversion all run with the GIL released,
# but building the returned list constructs one 3-tuple of ints per match under
# the GIL: O(number of matches). See the README's Performance section and
# tests/test_gil_release.py for the measured sparse/dense bands.
def find_patterns(patterns: list[str], text: str) -> list[tuple[int, int, int]]: ...


# GIL note: the streaming spelling of find_patterns: the whole search
# (pattern-list walk, automaton build, scan, byte→char conversion) fills an
# internal buffer under ONE GIL-released pass at construction, and each
# __next__ holds the GIL only for a single 3-tuple of ints. The streaming
# answer to the list shape's O(matches) tuple-marshalling caveat (~13 ms
# held per 100k matches in the list shape). Same sequence as the list API,
# pinned.
def find_patterns_iter(
    patterns: list[str], text: str
) -> Iterator[tuple[int, int, int]]: ...


# GIL note: the count spelling of find_patterns: the same
# leftmost-longest, non-overlapping search answering just the number, with
# NO match vector (the list spelling materializes ~250 MiB of matches on a
# 100 MiB dense corpus just to answer "how many") and no byte→char pass
# (counting is offset-free). count_matches(p, t) == len(find_patterns(p, t)),
# pinned. A single int return: no marshalling class at all.
def count_matches(patterns: list[str], text: str) -> int: ...


# GIL note (the CompiledLemmaDict discipline, over the search surface): the
# pattern list compiled ONCE (one detached build at construction), then
# every call is the free function's scan classes MINUS the per-call
# automaton build: one Arc refcount bump, the scan under one detach, the
# same marshalling as the free spelling. Immutable after construction, so
# sharing one across many calls and threads is sound with no
# synchronization beyond the refcount. The replace spellings validate the
# replacements dict at CALL time (values change per call; the automaton is
# the compiled part): it must key EXACTLY the compiled pattern set, every
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


# If text, trimmed of leading/trailing whitespace, is EXACTLY one fenced
# code block (including the unterminated-fence case), return its dedented
# code content; otherwise return text UNCHANGED (not even
# whitespace-trimmed). tors.strip_code_fences(s) is s exactly when s is not
# the single-fenced-block case.
#
# GIL note: the detached_transform shape shared by normalize/nfc/etc: str-in
# argument borrow, the scan with the GIL released, then either the ORIGINAL
# object back (zero marshalling) or the O(output) unwrapped string.
def strip_code_fences(text: str) -> str: ...


# textwrap.dedent(text), byte-for-byte: the longest common leading
# whitespace-run STRING (tabs and spaces are distinct characters: "  x"
# and "\tx" share no margin) is stripped from every line, and
# whitespace-only lines normalize to empty. tors.dedent(s) is s exactly
# when tors.dedent(s) == s.
#
# GIL note: detached_transform's shape, the same as strip_code_fences.
def dedent(text: str) -> str: ...


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
# would otherwise split a cluster); the result NEVER exceeds max_chars
# codepoints either way. The cut point is then trimmed of trailing
# whitespace. max_chars < 0 and an unrecognized boundary both raise
# ValueError. tors.truncate_to_bounds(s, n) is s exactly when s already has
# <= n codepoints.
#
# GIL note: detached_transform's shape: the segmentation scan and cut run
# with the GIL released, then either the ORIGINAL object back (zero
# marshalling) or the O(output) truncated string.
def truncate_to_bounds(
    text: str, max_chars: int, boundary: Literal["word", "sentence"] = "word"
) -> str: ...


# Is claim grounded in source: fuzzy=False (default) is source.contains(claim)
# exactly; fuzzy=True is a windowed difflib-ratio scan of source against
# threshold (a LEXICAL check: no NLI/semantic model; see the crate's
# grounded_impl module docs for exactly what the score measures and its
# DoS-bounded windowing over long sources). An empty claim is vacuously
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
# single float out. get_close_matches returns the ORIGINAL candidate
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
# each matched span of L characters, its replacement VALUE is truncated to
# its first L characters if the value has >= L characters, or emitted in
# full and padded with copies of `mask` out to L characters if it has fewer
# (mask is never consulted on the truncation branch), so the output's
# CHARACTER count and every non-matching span's offsets equal the input's
# (redaction that keeps pre-computed offsets valid). Pass "" as a value to
# force pure masking. mask must be exactly one character (one codepoint;
# ValueError otherwise). Identity contract: is s exactly when == s.
def replace_many_masked(
    text: str, replacements: dict[str, str], mask: str = "*"
) -> str: ...


# Domain-separated SHA-256 (RFC 6962-style: leaves hash 0x00‖chunk, internal
# nodes hash 0x01‖left‖right): not the crate's undifferentiated default,
# which is forgeable (CVE-2012-2459-class leaf/internal-node confusion).
def merkle_root(chunks: list[bytes]) -> str: ...
def merkle_diff(chunks_a: list[bytes], chunks_b: list[bytes]) -> list[int]: ...


# FastCDC 2020 content-defined chunking: (start, end) BYTE spans (not
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
# are Python str indices. overlap=0 (default): the ORIGINAL lossless-
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
# stall. The return marshalling is O(chunks) 2-tuples of ints (the
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
# that chunk into the hundreds of thousands of pieces.
def chunk_text_iter(
    text: str,
    max_chars: int,
    *,
    overlap: int = 0,
    boundary: Literal["word", "sentence"] = "word",
) -> Iterator[tuple[int, int]]: ...


# GIL note: word-count-windowed chunking: the semantic-chunking RAG
# shape, measured in real WORD TOKENS (word_bounds' segments filtered to
# non-whitespace-only ones: word_bounds itself gives an inter-word space
# run its own segment, so grouping RAW segments would silently mean
# "words_per_chunk roughly halved" on ordinary prose) rather than a
# character budget. Each chunk spans `words_per_chunk` consecutive word
# tokens, its span the first token's start through the last's end (NOT
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
def chunk_by_sentences_iter(
    text: str, sentences_per_chunk: int, *, overlap: int = 0
) -> Iterator[tuple[int, int]]: ...


# chunk_by_words/chunk_by_sentences' paragraph-count twin. A paragraph
# boundary is a run of 2+ consecutive newline characters (\r\n counts as
# one unit, matching normalize's own CR/CRLF folding): the "2+ newlines
# survive as the paragraph gap" convention normalize's own pipeline
# already uses (it collapses 3+ down to exactly 2, never below). This is
# a HEURISTIC, not a Unicode Standard segmentation (there is no UAX for
# paragraphs): a single \n is ordinary content, not a break. Same
# contract, same argument validation, same empty-input answer as its
# siblings.
def chunk_by_paragraphs(
    text: str, paragraphs_per_chunk: int, *, overlap: int = 0
) -> list[tuple[int, int]]: ...


# Priority-ordered fallback chunking (LangChain's RecursiveCharacterTextSplitter
# pattern): cut at the COARSEST level that fits max_chars, falling back to
# finer levels only when a coarser one has no in-budget cut. separators=None
# uses tors's own accurate hierarchy (paragraph -> sentence -> word -> a
# grapheme-safe raw cut, always the final unconditional fallback).
# separators=[...] is a caller-supplied list of LITERAL strings (not regex),
# coarsest first, e.g. ["\n## ", "\n\n", ". ", " "] for markdown-header-aware
# chunking: replaces the default hierarchy, but the raw cut is still always
# appended. NOT a lossless partition (unlike chunk_text): the separator
# itself is dropped between chunks, the same chunk_by_paragraphs convention.
# overlap snaps to the nearest GRAPHEME boundary (not necessarily a semantic
# one, a documented simplification of chunk_text_overlapping's single-level
# snap). max_chars < 1 or overlap < 0 raise ValueError; overlap >= max_chars
# raises ValueError. Empty text returns []; an empty separators list is legal
# and skips straight to the raw-cut fallback.
def chunk_hierarchical(
    text: str,
    max_chars: int,
    separators: list[str] | None = None,
    *,
    overlap: int = 0,
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


# Stateless: no vocabulary/vectorizer object persists between calls.
# Tokenization: UAX #29 word segments, non-whitespace only, lowercased
# (Unicode-correct str.lower, not ASCII-only). TF is the RAW term count
# per document (not length-normalized). IDF is the scikit-learn-style
# SMOOTHED formula ln(N / (1 + df)) + 1 (N = corpus size, df = document
# frequency): not the textbook ln(N/df), which gives a
# term in every document an IDF of exactly 0; smoothing keeps that case
# strictly positive while still favoring rarer terms. score(t, d) =
# tf(t, d) * idf(t). Output is SPARSE: one (term, score) list per
# document, alphabetically sorted, holding only that document's own
# terms, never a vocabulary-size-by-corpus-size dense structure. Empty
# corpus -> []; an empty-string document -> [] at its position (the
# output always has exactly len(corpus) entries). strip_accents=True
# NFD-decomposes each token and drops combining marks ("café" -> "cafe")
# before scoring, unconditionally (not the short-circuit-on-already-
# decomposed-input bug scikit-learn's own strip_accents_unicode has).
# stemmer names a Snowball algorithm applied after lowercasing/accent-
# folding; an unrecognized name raises ValueError naming every valid
# choice. lemma_dict is a caller-supplied word -> lemma map applied LAST
# (after stemming, if any): tors does not bundle a lemma dictionary (that
# needs a per-language dataset or POS model, out of scope); it only
# APPLIES one you supply, the same shape replace_many takes a
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
    stemmer: _StemmerLanguage | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
) -> list[list[tuple[str, float]]]: ...


# A RERANKING primitive, not a search index: recomputes corpus statistics
# from scratch every call, the right shape for scoring a small,
# already-retrieved candidate set (tens to a few hundred documents) against
# one query: the wrong shape for a large corpus queried repeatedly (reach
# for a real search engine, e.g. tantivy, for that; tors does not build
# persistent index objects). Okapi BM25, the always-non-negative "+1" IDF
# variant (ln((N - df + 0.5) / (df + 0.5) + 1), not the classic form, which
# goes negative for a term in over half the corpus). k1 (>= 0, default 1.5)
# tunes term-frequency saturation; b (in [0, 1], default 0.75) tunes length
# normalization: both are Lucene/Elasticsearch's own defaults. Returns
# (index, score) for EVERY document (no top-k cutoff: slice/sort
# yourself), sorted by score descending, ties broken by ascending original
# index. Tokenization: the same "real word token, lowercased" convention
# tf_idf uses. Empty corpus -> []; empty query -> every document scores
# 0.0 (not an error). No claim about retrieval/relevance QUALITY for any
# particular corpus or query: a correct implementation of a well-known
# ranking formula, not a model-quality promise.
#
# strip_accents/stemmer/lemma_dict: tf_idf's exact same opt-in knobs,
# applied IDENTICALLY to query and every corpus document (required for the
# scores to mean anything, not just a style choice), all default off.
def bm25_rank(
    query: str,
    corpus: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    strip_accents: bool = False,
    stemmer: _StemmerLanguage | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
) -> list[tuple[int, float]]: ...


# A stateless, GENERAL-PURPOSE batch text preprocessor: every requested
# step fused into ONE GIL-released pass over the WHOLE texts list. Pure
# function composition, NOT a re.compile()-style compiled-
# pipeline object (a stateful handle was considered and explicitly ruled
# out: every call re-describes and re-applies its steps fresh). Order:
# nfd -> lowercase -> strip_accents -> (stemmer / lemma_dict) ->
# collapse_whitespace, each skipped when its flag is off/None. nfd/
# lowercase/strip_accents are codepoint-level (run over the whole text);
# stemmer/lemma_dict are word-level (walk UAX #29 word-boundary segments,
# transforming only real-word segments, preserving every other segment
# (punctuation, whitespace) verbatim, so the output stays readable prose).
# collapse_whitespace runs last, reducing every run of Python-whitespace-
# equivalent codepoints to one ASCII space (NOT tors.normalize's full
# pipeline: no CRLF folding, no blank-line collapsing, no strip).
#
# All six steps default off: apply_pipeline(texts) with nothing else is a
# true IDENTITY: the original texts list OBJECT comes back unchanged, not
# just content-equal output, the same zero-allocation contract normalize/
# quote/replace_many already give for their own no-op case. Empty texts ->
# []. A non-list argument or non-str element raises TypeError; an invalid
# stemmer name raises ValueError naming every valid choice; a non-dict
# lemma_dict, or one with a non-str key/value, raises TypeError.
#
# Relationship to tf_idf/bm25_rank: those two already fuse the SAME
# strip_accents/stemmer/lemma_dict knobs into their own tokenization:
# calling apply_pipeline first and then tf_idf/bm25_rank on the result
# tokenizes TWICE for no benefit. Reach for their own knobs when they're
# the only consumer; reach for apply_pipeline to preprocess text feeding
# anything else (chunk_text, find_patterns, your own logic).
def apply_pipeline(
    texts: list[str],
    *,
    nfd: bool = False,
    lowercase: bool = False,
    strip_accents: bool = False,
    stemmer: _StemmerLanguage | None = None,
    lemma_dict: dict[str, str] | CompiledLemmaDict | None = None,
    collapse_whitespace: bool = False,
) -> list[str]: ...


# Classic phonetic-code algorithms (rphonetic, an Apache Commons Codec
# port): ENGLISH/Latin-script-oriented heuristics, not general Unicode
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
# so a name match on EITHER key counts; two equal elements when there is
# only one pronunciation.
def double_metaphone(text: str) -> tuple[str, str]: ...
# NYSIIS (1970), strict commons-codec variant (codes capped at 6
# characters): a Soundex successor for name matching.
def nysiis(text: str) -> str: ...
# Daitch-Mokotoff Soundex (1985), the Jewish-genealogy standard for
# Central/Eastern European surnames. A LIST because the rule table
# branches on ambiguous transliterations: one name can encode to
# several 6-digit codes, and two names match if their lists intersect.
# Unlike soundex/metaphone, no-letters input is ["000000"] (each code
# padded to 6 digits), not "".
def daitch_mokotoff(text: str) -> list[str]: ...


# A Soundex variant with a finer-grained letter-to-digit mapping than
# classic soundex (more consonant classes distinguished, an uncapped
# code rather than soundex's fixed letter-plus-3-digit shape) — a
# genuinely distinct algorithm, not a formatting variant. Same
# ASCII-letters-only pre-filter and upstream-panic-avoidance note as
# soundex.
def refined_soundex(text: str) -> str: ...

