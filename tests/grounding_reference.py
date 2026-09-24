"""Pure-Python reference implementations of the ROUGE-W family, the
differential oracles for `tors.highlight`, `tors.ground_sentences`, and
`tors.grounding_coverage`.

Three deliberately diverging spellings of the weighted-LCS score pin what the
Rust core does and what it explicitly does not do:

- ``wlcs_lin``: Lin 2004's published WLCS fill verbatim (the paper's own
  dynamic program: a match cell ALWAYS extends the diagonal run). The core
  does NOT match this spelling; the divergence is the documented monotone
  variant (see docs/api.md).
- ``wlcs_max_on_match``: the spelling the core implements (a match cell
  takes ``max(diagonal extension, local best)``), which restores candidate
  monotonicity.
- ``wlcs_bruteforce``: the literal max over all monotone matchings. The
  two-row DP does not compute this optimum (it cannot represent Pareto
  (value, trailing-run) states); the gap is documented and pinned.
"""

from __future__ import annotations

from hypothesis import strategies as st

import tors


def _f(k: float) -> float:
    return k ** 1.2


def _finv(x: float) -> float:
    return x ** (1 / 1.2)


# ---------------------------------------------------------------------------
# Pure-Python reference implementations (independent of the Rust code)
# ---------------------------------------------------------------------------


def _f(k: float) -> float:
    """Lin 2004's shaping function, f(k) = k^1.2 (the paper's polynomial)."""
    return k**1.2 if k > 0 else 0.0


def _finv(x: float) -> float:
    """The shaping function's inverse (Lin's Equation 13/14 normalization)."""
    return x ** (1.0 / 1.2)


def wlcs_lin(q: list[str], c: list[str]) -> float:
    """Lin 2004's published WLCS fill, verbatim: a match cell ALWAYS
    extends the diagonal run (forced diagonal); w resets on non-match."""

    class _Runner:
        def __init__(self) -> None:
            self.value = 0.0

    runner = _Runner()
    n, m = len(q), len(c)
    s = [[0.0] * (m + 1) for _ in range(n + 1)]
    g = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if q[i - 1] == c[j - 1]:
                k = g[i - 1][j - 1]
                s[i][j] = s[i - 1][j - 1] + _f(k + 1) - _f(k)
                g[i][j] = k + 1
            else:
                if s[i - 1][j] > s[i][j - 1]:
                    s[i][j] = s[i - 1][j]
                else:
                    s[i][j] = s[i][j - 1]
                g[i][j] = 0
    runner.value = s[n][m]
    return runner.value


def wlcs_max_on_match(q: list[str], c: list[str]) -> float:
    """The recurrence the core documents (independent full-matrix
    re-derivation): a match cell takes the diagonal extension only when it
    beats both skips, resetting the run when a skip wins."""

    class _Runner:
        def __init__(self) -> None:
            self.value = 0.0

    runner = _Runner()
    n, m = len(q), len(c)
    s = [[0.0] * (m + 1) for _ in range(n + 1)]
    g = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            up, left = s[i - 1][j], s[i][j - 1]
            if q[i - 1] == c[j - 1]:
                run = g[i - 1][j - 1] + 1
                ext = s[i - 1][j - 1] + _f(run) - _f(run - 1)
                if up > ext or left > ext:
                    s[i][j] = max(up, left)
                    g[i][j] = 0
                else:
                    s[i][j] = ext
                    g[i][j] = run
            else:
                s[i][j] = max(up, left)
                g[i][j] = 0
    runner.value = s[n][m]
    return runner.value


def wlcs_bruteforce(q: list[str], c: list[str]) -> float:
    """The literal maximum over all monotone matchings: sum f(run) over
    maximal consecutive runs.  Exponential; tiny inputs only."""

    class _Runner:
        def __init__(self) -> None:
            self.best = 0.0

    runner = _Runner()
    n, m = len(q), len(c)

    def rec(i: int, j: int, pairs: list[tuple[int, int]]) -> None:
        value = 0.0
        run = 0
        prev: tuple[int, int] | None = None
        for a, b in pairs:
            if prev is not None and a == prev[0] + 1 and b == prev[1] + 1:
                run += 1
            else:
                run = 1
            value += _f(run) - _f(run - 1)
            prev = (a, b)
        runner.best = max(runner.best, value)
        if i >= n or j >= m:
            return
        rec(i + 1, j, pairs)
        rec(i, j + 1, pairs)
        if q[i] == c[j]:
            rec(i + 1, j + 1, pairs + [(i, j)])

    rec(0, 0, [])
    return runner.best


def rouge_w_f1_ref(q: list[str], c: list[str], wlcs=wlcs_max_on_match) -> float:
    """Equation 15's F1 (beta = 1) over a WLCS fill, Lin's Equation 13/14
    normalization through f^-1, defensively clamped to [0, 1]."""
    if not q or not c:
        return 0.0
    w = wlcs(q, c)
    if w <= 0.0:
        return 0.0
    r = _finv(w / _f(len(q)))
    p = _finv(w / _f(len(c)))
    return min(1.0, max(0.0, 2.0 * r * p / (r + p)))


def coverage_ref(source: list[str], text: list[str]) -> float:
    """The coverage core's documented score: Equation 15's R factor
    (f^-1(WLCS / f(|source|))) over the max-on-match fill, clamped."""
    if not source or not text:
        return 0.0
    w = wlcs_max_on_match(source, text)
    if w <= 0.0:
        return 0.0
    return min(1.0, max(0.0, _finv(w / _f(len(source)))))


# ---------------------------------------------------------------------------
# Tokenization-locked lanes: single-sentence ASCII/letter texts where the
# tokenizer's output is exactly the space-separated words, so the reference
# and the core score the same token streams.
# ---------------------------------------------------------------------------

_WORDS = ["ax", "by", "cz", "dd", "ee", "ff"]


def _score_sentence(text: str, query: str) -> float:
    """The core's per-sentence F1 for a one-sentence text."""
    res = tors.ground_sentences(text, query)
    assert len(res["sentences"]) == 1, f"{text!r}: {len(res['sentences'])} sentences"
    return res["sentences"][0]["score"]


def _cov(source: str, text: str) -> float:
    return tors.grounding_coverage(source, text)


@st.composite
def _word_seqs(draw, max_size=12):
    n = draw(st.integers(min_value=1, max_value=max_size))
    return draw(st.lists(st.sampled_from(_WORDS), min_size=n, max_size=n))


