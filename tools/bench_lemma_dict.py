"""Honest measurement: tors's native `lemma_dict` substitution (inside
`apply_pipeline`) against an idiomatic pure-Python equivalent, applied to
the SAME token list so the comparison isolates lookup-application cost,
not tokenization differences.

The claim under test: tors's `lemma_dict` is faster because it avoids the
GIL-hold and per-token Python bytecode-dispatch/call overhead of an
equivalent Python loop, NOT because of a smarter lookup algorithm --- a
dict lookup is O(1) on both sides. This script measures the real ratio
rather than asserting it. Run with `uv run python tools/bench_lemma_dict.py`.
"""

from __future__ import annotations

import time

import tors

WORDS = (
    "the quarterly oil sample interval for field outages was adjusted "
    "after the bushing torque specifications changed windows now close "
    "within fourteen days"
).split()

# A lemma dict sized like a real one (spaCy's English lookup table is
# tens of thousands of entries) --- most lookups miss, a few hit, matching
# realistic coverage rather than a dict tuned to make either side look good.
LEMMA_DICT = {f"inflected_form_{i}": f"lemma_{i}" for i in range(20_000)}
LEMMA_DICT.update(
    {
        "outages": "outage",
        "adjusted": "adjust",
        "changed": "change",
        "windows": "window",
    }
)


def pure_python_apply(tokens: list[str], lemma_dict: dict[str, str]) -> list[str]:
    return [lemma_dict.get(tok, tok) for tok in tokens]


def run(doc_count: int, repeat: int) -> tuple[float, float]:
    texts = [" ".join(WORDS) for _ in range(doc_count)]

    start = time.perf_counter()
    for _ in range(repeat):
        tors.apply_pipeline(texts, lemma_dict=LEMMA_DICT)
    tors_seconds = time.perf_counter() - start

    tokenized = [t.split() for t in texts]
    start = time.perf_counter()
    for _ in range(repeat):
        for tokens in tokenized:
            pure_python_apply(tokens, LEMMA_DICT)
    python_seconds = time.perf_counter() - start

    return tors_seconds, python_seconds


if __name__ == "__main__":
    for doc_count in (10, 100, 500):
        repeat = max(1, 2000 // doc_count)
        tors_seconds, python_seconds = run(doc_count, repeat)
        ratio = python_seconds / tors_seconds
        print(
            f"doc_count={doc_count:>4} repeat={repeat:>4}  "
            f"tors={tors_seconds * 1000:8.2f}ms  "
            f"python={python_seconds * 1000:8.2f}ms  "
            f"ratio={ratio:5.2f}x"
        )
