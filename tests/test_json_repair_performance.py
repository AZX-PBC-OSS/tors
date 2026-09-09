"""The tors-vs-upstream json_repair wall lane: prove the native port is
faster than the pure-Python oracle it pins, on the same bytes, with the
outputs asserted equal in the same cell.

Every cell is a measurement (min-of-samples wall) plus a differential
correctness assertion: tors.repair_json_loads(raw) ==
json_repair.loads(raw) on the exact benchmark payload, so a perf
comparison can never silently diverge from the parity contract. The
oracle is json_repair pinned to 0.63.4 (the same pin the parity suite
differential-tests against).

Corpus builders mirror benches/json_repair.rs (the Criterion cells) so
the Python lane and the Rust lane measure the same shapes: a
records-array document with escaped prose values, and the fixed
five-defect rotation (dropped brace, unquoted key, single quotes,
dropped comma, truncation) applied to every other record.

The continuation cells sit BELOW both engines' recursion caps (tors
raises past 200 fragments, the oracle past ~165 on the array-merge
chain): the point is a fair wall comparison on inputs both parse. Past
the caps the engines raise (tors ValueError at MAX_NESTING, the oracle
its own ValueError near the interpreter recursion limit) — that
both-raise shape is the parity suite's job, not this lane's.
"""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

import pytest

pytestmark = pytest.mark.timing

pytest.importorskip("json_repair")  # pin: json-repair==0.63.4

import json_repair  # noqa: E402
import tors  # noqa: E402

_MIB = 1024 * 1024
_SAMPLES = 5
# tors must beat the oracle by at least 2x (measured ratios 0.02-0.09,
# so the gate has an order of magnitude of headroom on every cell).
_MARGIN = 0.5

# The corpus's fixed note value, mirroring benches/json_repair.rs.
_NOTE_JSON = 'He said \\"replace the gasket\\" and left \\\\ the spec on the shelf.'


def _escape_json_string(s: str) -> str:
    out: list[str] = []
    for c in s:
        out.append(
            {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(c, c)
        )
    return "".join(out)


def _prose(n: int) -> str:
    sentence = "The quick brown fox jumps over the lazy dog while the crane watches. "
    return (sentence * (n // len(sentence) + 1))[:n]


def _object_json(index: int, defect: int | None) -> str:
    body = _escape_json_string(_prose(256))
    quote = "'" if defect == 2 else '"'
    clean = f'{{"id": {index}, "body": {quote}{body}{quote}, "note": {quote}{_NOTE_JSON}{quote}}}'
    if defect is None:
        return clean
    if defect == 0:  # dropped closing brace
        return clean[:-1]
    if defect == 1:  # unquoted key
        return clean.replace('"id":', "id:", 1)
    if defect == 2:  # single-quoted values
        return clean
    if defect == 3:  # dropped comma
        return clean.replace(', "body":', ' "body":', 1)
    cut = len(clean) * 7 // 8  # mid-object truncation
    return clean[:cut]


def valid_json(target_bytes: int) -> str:
    out = ['{"records": [']
    index = 0
    while sum(len(part) for part in out) < target_bytes:
        if index:
            out.append(",")
        out.append(_object_json(index, None))
        index += 1
    out.append("]}")
    return "".join(out)


def malformed_llm_output(target_bytes: int) -> str:
    out = ['{"records": [']
    index = 0
    while sum(len(part) for part in out) < target_bytes:
        if index:
            out.append(",")
        defect = [0, 1, 2, 3, 4][(index // 2) % 5] if index % 2 == 0 else None
        out.append(_object_json(index, defect))
        index += 1
    out.append("]}")
    return "".join(out)


def _min_wall_ms(op: Callable[[str], object], arg: str) -> float:
    op(arg)
    best = float("inf")
    for _ in range(_SAMPLES):
        started = monotonic()
        op(arg)
        best = min(best, monotonic() - started)
    return best * 1000.0


def _assert_cell_beats_the_oracle(name: str, raw: str) -> None:
    """One measurement cell: outputs equal, then tors' best wall under the
    margin times the oracle's best wall on the same bytes."""
    tors_value = tors.repair_json_loads(raw, skip_json_loads=True)
    oracle_value = json_repair.loads(raw, skip_json_loads=True)
    assert tors_value == oracle_value, (
        f"{name}: the perf cell's payload diverged from the oracle — the "
        f"comparison is no longer apples-to-apples "
        f"(tors={tors_value!r:.200} oracle={oracle_value!r:.200})"
    )
    tors_ms = _min_wall_ms(
        lambda p: tors.repair_json(p, skip_json_loads=True), raw
    )
    oracle_ms = _min_wall_ms(
        lambda p: json_repair.repair_json(p, skip_json_loads=True), raw
    )
    assert tors_ms < _MARGIN * oracle_ms, (
        f"{name}: tors {tors_ms:.2f}ms vs oracle {oracle_ms:.2f}ms "
        f"(ratio {tors_ms / oracle_ms:.3f}): the native port lost more than "
        "the margin to the pure-Python oracle"
    )


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("valid_fast_1mib", valid_json(1 * _MIB)),
        ("malformed_1mib", malformed_llm_output(1 * _MIB)),
        # The continuation chains, below BOTH caps so both engines return:
        # the array-merge chain (tors caps at 198 fragments, the oracle
        # near ~165) measured at 150.
        ("merge_chain_150", '{"a":[0],' + '["b":[0],' * 150 + "1]"),
        # The comma-merge chain (tors caps at 199 fragments, the oracle at
        # ~330) measured at 150.
        ("comma_chain_150", '{"a":1}' + ', "k":1}' * 150),
        # Same-level sequential merges never accrue depth on either engine;
        # 2_000 past any cap to keep the balanced-continuation path honest.
        ("seq_merges_2k", '{"a":[1]' + ", [2]" * 2_000 + "}"),
        # String-colon objects nested through parse_json's guarded '['
        # branch (the sibling shape that never needed a fix) at 150.
        ("strcolon_150", "[" + '"b": [' * 150),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_repair_beats_the_oracle_on_the_same_bytes(name: str, raw: str) -> None:
    _assert_cell_beats_the_oracle(name, raw)


def _seq_merge_payload(count: int) -> str:
    return '{"a":[1]' + ", [2]" * count + "}"


def test_escaped_delimiter_run_wall_scales_linearly_not_quadratically() -> None:
    """The escape normalizer's pop-then-push repairs carry a one-level
    undo record (see StringParseState's field docs), so quadrupling the
    escaped-delimiter count quadruples the work: the 32k-fragment wall
    stays within a small factor of 4x the 8k-fragment wall. The
    whole-accumulator rescan this gate pins out was O(n^2) — 16x per
    quadrupling, ~1.2s at 16k fragments. Machine-speed-immune by
    construction: a ratio of two walls on the same box."""
    frag = r'{\"k\": 1}'

    def payload(count: int) -> str:
        return "{" + frag * count + "}"

    small = _min_wall_ms(
        lambda p: tors.repair_json(p, skip_json_loads=True), payload(8_000)
    )
    large = _min_wall_ms(
        lambda p: tors.repair_json(p, skip_json_loads=True), payload(32_000)
    )
    # Linear scaling: 4x the fragments = 4x the wall; 1.5x slack for cache
    # effects. Quadratic would need 16x and fails loudly.
    assert large < 6.0 * small, (
        f"escaped-delimiter runs scale super-linearly: 32k {large:.1f}ms vs "
        f"8k {small:.1f}ms (ratio {large / small:.1f}x; linear would be "
        "~4x) — the whole-accumulator rescan is back"
    )


def test_sequential_merge_wall_scales_linearly_not_quadratically() -> None:
    """The array-merge continuation maintains its row-width summary
    incrementally across a same-level merge run (the summary lives in
    parse_object_key's key-scan loop and folds only what each merge
    appends), so quadrupling the merge count quadruples the work: the
    100k-merge wall stays within a small factor of 4x the 25k-merge wall.
    The rescan-the-whole-previous-array-per-merge shape this gate pins out
    was O(M^2) — 16x per quadrupling, ~5s at 200k merges from 1.6 MB of
    input. Machine-speed-immune by construction: the assertion is a ratio
    of two walls on the same box, never an absolute time."""
    small = _min_wall_ms(
        lambda p: tors.repair_json(p, skip_json_loads=True), _seq_merge_payload(25_000)
    )
    large = _min_wall_ms(
        lambda p: tors.repair_json(p, skip_json_loads=True),
        _seq_merge_payload(100_000),
    )
    # Linear scaling: 4x the merges = 4x the wall; 1.5x slack for cache
    # effects. Quadratic would need 16x and fails loudly.
    assert large < 6.0 * small, (
        f"sequential merges scale super-linearly: 100k merges {large:.1f}ms "
        f"vs 25k merges {small:.1f}ms (ratio {large / small:.1f}x; linear "
        "would be ~4x) — the per-merge row-width rescan is back"
    )
