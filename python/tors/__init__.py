from __future__ import annotations

from types import ModuleType

from tors._tors import (
    CompiledLemmaDict,
    CompiledPatterns,
    apply_pipeline,
    b64_decode,
    b64_encode_bytes,
    bm25_rank,
    chunk_by_lines,
    chunk_by_lines_iter,
    chunk_by_paragraphs,
    chunk_by_paragraphs_iter,
    chunk_by_sentences,
    chunk_by_sentences_iter,
    chunk_by_words,
    chunk_by_words_iter,
    chunk_cdc,
    chunk_hierarchical,
    chunk_text,
    chunk_text_iter,
    contains_unescaped,
    count_matches,
    daitch_mokotoff,
    decode_utf8,
    decode_utf16,
    dedent,
    detect_encoding,
    diff_opcodes,
    diff_opcodes_lines,
    double_metaphone,
    extract_code_blocks,
    finalize,
    finalize_utf8,
    find_patterns,
    find_patterns_iter,
    find_unescaped,
    first_invalid_charset,
    first_invalid_offender,
    get_close_matches,
    grapheme_count,
    html_unescape,
    is_grounded,
    jaro,
    jaro_winkler,
    levenshtein,
    merkle_diff,
    merkle_root,
    metaphone,
    nfc,
    nfd,
    nfkc,
    nfkd,
    normalize,
    nysiis,
    quote,
    quote_plus,
    refined_soundex,
    repair_json,
    repair_json_diagnostics,
    repair_json_loads,
    replace_many,
    replace_many_masked,
    scrub_log_text,
    sentence_bounds,
    sentence_bounds_iter,
    sentence_count,
    simhash64,
    simhash128,
    similarity_ratio,
    soundex,
    strip_code_fences,
    strip_controls,
    tf_idf,
    truncate_ellipsis,
    truncate_to_bounds,
    unquote,
    unquote_plus,
    utf8_byte_len,
    utf8_is_valid,
    utf16_byte_len,
    utf16_is_valid,
    word_bounds,
    word_bounds_iter,
    word_count,
)

__all__ = [
    "apply_pipeline",
    "b64_decode",
    "b64_encode_bytes",
    "bm25_rank",
    "CHARSET_B62",
    "CHARSET_B64URL",
    "CHARSET_HEX_LOWER",
    "CHARSET_HEX_MIXED",
    "CHARSET_HEX_UPPER",
    "chunk_by_lines",
    "chunk_by_lines_iter",
    "chunk_by_paragraphs",
    "chunk_by_paragraphs_iter",
    "chunk_by_sentences",
    "chunk_by_sentences_iter",
    "chunk_by_words",
    "chunk_by_words_iter",
    "chunk_cdc",
    "chunk_hierarchical",
    "chunk_text",
    "chunk_text_iter",
    "CompiledLemmaDict",
    "CompiledPatterns",
    "contains_unescaped",
    "count_matches",
    "daitch_mokotoff",
    "decode_utf16",
    "decode_utf8",
    "dedent",
    "detect_encoding",
    "diff_opcodes",
    "diff_opcodes_lines",
    "double_metaphone",
    "extract_code_blocks",
    "finalize",
    "finalize_utf8",
    "find_patterns",
    "find_patterns_iter",
    "find_unescaped",
    "first_invalid_charset",
    "first_invalid_offender",
    "get_close_matches",
    "grapheme_count",
    "html_unescape",
    "is_grounded",
    "jaro",
    "jaro_winkler",
    "levenshtein",
    "merkle_diff",
    "merkle_root",
    "metaphone",
    "nfkc",
    "nfkd",
    "nfc",
    "nfd",
    "normalize",
    "nysiis",
    "quote",
    "quote_plus",
    "refined_soundex",
    "repair_json",
    "repair_json_diagnostics",
    "repair_json_loads",
    "replace_many",
    "replace_many_masked",
    "scrub_log_text",
    "sentence_bounds",
    "sentence_bounds_iter",
    "sentence_count",
    "simhash64",
    "simhash128",
    "similarity_ratio",
    "soundex",
    "strip_code_fences",
    "strip_controls",
    "tf_idf",
    "truncate_ellipsis",
    "truncate_to_bounds",
    "unquote",
    "unquote_plus",
    "utf16_byte_len",
    "utf16_is_valid",
    "utf8_byte_len",
    "utf8_is_valid",
    "word_bounds",
    "word_bounds_iter",
    "word_count",
]

# Pinned common alphabets for first_invalid_charset — published module data,
# not validator functions: the generic engine stays the single engine (named
# wrappers would delegate to the same core for zero performance gain, pure
# API surface), and the constants kill the real friction, spelling a
# 62-character alphabet correctly at every call site. See docs/api.md's
# "Common alphabets" for the scope decision these encode, including what is
# deliberately absent (padded base64, UUID, digits) and why.

# The base62 id alphabet: digits, then upper, then lower.
CHARSET_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# The RFC 4648 §5 url-safe alphabet, unpadded: JWT segments, url-safe tokens.
CHARSET_B64URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
# Lowercase hex digests.
CHARSET_HEX_LOWER = "0123456789abcdef"
# Uppercase hex digests.
CHARSET_HEX_UPPER = "0123456789ABCDEF"
# The 22-char union of the two hex spellings: case-insensitive hex digests.
CHARSET_HEX_MIXED = "0123456789abcdefABCDEF"


def __getattr__(name: str) -> ModuleType:
    """The lazy ``documents`` door (PEP 562): ``tors.documents`` on an
    imported base package imports the shim (and through it the
    tors-documents payload wheel) on first touch, so a plain ``import
    tors`` still loads no engine (the split-wheel doctrine, pinned by the
    laziness gate). A missing payload wheel answers with the shim's own
    ImportError install hint; every other name is the standard module
    AttributeError."""
    if name == "documents":
        import tors.documents as documents

        return documents
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
